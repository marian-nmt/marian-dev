#include "marian.h"
#include "translator/swappable.h"
#include "common/logging.h"
#include "data/corpus.h"
#include "data/text_input.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#include "common/io.h"
#include "common/timer.h"
#include <vector>
#include "tensors/gpu/swap.h"

namespace marian {

void GPUEngine::SwapPointers(std::vector<MemoryPiece::PtrType> &with) {
  auto write_it = graph_->params()->begin();
  auto read_it = with.begin();
  for (; read_it != with.end(); ++write_it, ++read_it) {
    std::swap(*(*write_it)->val()->memory(), **read_it);
  }
}

GPUEngine::GPUEngine(Ptr<Options> options, size_t deviceIdx) 
  : options_(options), graph_(New<ExpressionGraph>()), myDeviceId_(Config::getDevices(options)[deviceIdx]), allocator_(myDeviceId_, 0, 128 * 1048576) {
  ABORT_IF(myDeviceId_.type == DeviceType::cpu, "Swappable slot only works for GPU devices.");
  options_->set("inference", true);
  options_->set("shuffle", "none");

  // Create graph
  auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
  graph_->setDefaultElementType(typeFromString(prec[0]));
  graph_->setDevice(myDeviceId_);
  graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));

  scorers_ = createScorers(options_);
  for (auto scorer : scorers_) {
    scorer->init(graph_);
    // TODO lexical shortlists are not supported yet.
  }
  graph_->forward();
  // TODO: reach into graph_->params() private members and free the parameter memory.
}

GPUEngine::~GPUEngine() {}

GPULoadedModel::GPULoadedModel(Ptr<GPUEngine> gpu) : engine_(gpu) {
  for (auto &param : *engine_->graph_->params()) {
    parameters_.push_back(engine_->allocator_.alloc(param->val()->memory()->size()));
  }
}

GPULoadedModel::~GPULoadedModel() {
  for (MemoryPiece::PtrType &p : parameters_) {
    engine_->allocator_.free(p);
  }
}

void GPULoadedModel::OverwriteFrom(const GPULoadedModel &from) {
  srcVocabs_ = from.srcVocabs_;
  trgVocab_ = from.trgVocab_;

  ABORT_IF(engine_ != from.engine_, "TODO: copy across GPUs.");

  for (size_t i = 0; i < parameters_.size(); ++i) {
    swapper::copyGpuToGpu(reinterpret_cast<char*>(parameters_[i]->data()), reinterpret_cast<const char*>(from.parameters_[i]->data()), parameters_[i]->size(), engine_->myDeviceId_);
  }
}

void GPULoadedModel::OverwriteFrom(const CPULoadedModel &from) {
  srcVocabs_ = from.SrcVocabs();
  trgVocab_ = from.TrgVocab();
  for (size_t i = 0; i < parameters_.size(); ++i) {
    swapper::copyCpuToGpu(reinterpret_cast<char*>(parameters_[i]->data()), from.Parameters()[i].data(), from.Parameters()[i].size(), engine_->myDeviceId_);
  }
}

std::string MultilineInputHack(const std::vector<std::string> &input) {
  if (input.size() == 1) {
    return input[0];
  } else {
    std::string ret;
    std::size_t size = 0;
    for (auto&& line : input) {
      size += line.size() + 1;
    }
    ret.reserve(size);
    for (auto&& line : input) {
      ret.append(line);
      ret.append("\n");
    }
    return ret;
  }
}

Histories GPULoadedModel::Translate(const std::vector<std::string> &input) {
  ABORT_IF(!trgVocab_, "GPULoadedModel needs to be overwritten by a CPU model first.");
  engine_->SwapPointers(parameters_);

  auto corpus = New<data::TextInput>(std::vector<std::string>(1, MultilineInputHack(input)), srcVocabs_, engine_->options_); // @TODO dirty hack
  data::BatchGenerator<data::TextInput> batchGenerator(corpus, engine_->options_, nullptr, false); // @TODO if the asynchronous batch preparation = true, but we supply less text than the mini-batch size we crash

  BeamSearch search(engine_->options_, engine_->scorers_, trgVocab_);
  Histories ret;
  ret.reserve(input.size());
  for (auto&& batch : batchGenerator) {
    auto result = search.search(engine_->graph_, batch);
    ret.insert(ret.end(), result.begin(), result.end());
  }
  std::sort(ret.begin(), ret.end(),[](marian::Ptr<marian::History> a, marian::Ptr<marian::History> b){return a->getLineNum() < b->getLineNum();});
  engine_->SwapPointers(parameters_);
  return ret;
}

CPULoadedModel::CPULoadedModel(Ptr<Options> options, const std::string &parameters, const std::vector<std::string> &sourceVocabPaths, const std::string &targetVocabPath)
  : parameters_(io::loadItems(parameters)) {
  // Load parameters.
  // Find the special element and remove it:
  size_t special_idx = 0;
  for (size_t i = 0; i < parameters_.size(); i++) {
    if (parameters_[i].name == "special:model.yml") {
      special_idx = i;
      break;
    }
  }
  parameters_.erase(parameters_.begin() + special_idx);
  // Prepare the name so that it matches the named map
  for (auto&& item : parameters_) {
    item.name = "F0::" + item.name;
  }
  // Sort by name to match params order.
  std::sort(parameters_.begin(), parameters_.end(), [](const io::Item &a, const io::Item &b){return a.name < b.name;});

  // Load source vocabs.
  const std::vector<int> &maxVocabs = options->get<std::vector<int>>("dim-vocabs");
  for(size_t i = 0; i < sourceVocabPaths.size(); ++i) {
    Ptr<Vocab> vocab = New<Vocab>(options, i);
    vocab->load(sourceVocabPaths[i], maxVocabs[i]);
    srcVocabs_.emplace_back(vocab);
  }

  // Load target vocab.
  trgVocab_ = New<Vocab>(options, sourceVocabPaths.size());
  trgVocab_->load(targetVocabPath);
}

} // namespace marian
