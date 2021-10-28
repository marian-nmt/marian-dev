#include "translator/swappable.h"
#include <vector>
#include "common/io.h"
#include "common/logging.h"
#include "common/timer.h"
#include "data/corpus.h"
#include "data/text_input.h"
#include "marian.h"
#include "models/amun.h"
#include "models/nematus.h"
#include "tensors/gpu/swap.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

namespace marian {

namespace {
  DeviceId LookupGPU(const Ptr<Options> options, size_t deviceIdx) {
    auto devices = Config::getDevices(options);
    ABORT_IF(deviceIdx >= devices.size(), "GPU device index higher than configured.");
    return devices[deviceIdx];
  }
} // namespace

// For debugging memory
void get(std::vector<uint8_t> &out, MemoryPiece::PtrType mem, Ptr<Backend> backend) {
  out.resize(mem->size());
#ifdef CUDA_FOUND
  gpu::copy(backend, mem->data<uint8_t>(), mem->data<uint8_t>() + mem->size(), out.data());
#endif
}

GPUEngineTrain::GPUEngineTrain(Ptr<Options> options, size_t deviceIdx)
  : options_(options), myDeviceId_(LookupGPU(options, deviceIdx)) {
  ABORT_IF(myDeviceId_.type == DeviceType::cpu, "Swappable slot only works for GPU devices.");
  options_->set("inference", false);
  options_->set("shuffle", "none");

  // There is no need to initialize the graph or builder here because that's done before
  // each Train() invokation
}

void GPUEngineTrain::RecreateGraphAndBuilder() {
  graph_ = New<ExpressionGraph>();
  auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
  graph_->setDefaultElementType(typeFromString(prec[0]));
  graph_->setDevice(myDeviceId_);
  graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));

  builder_ = models::createCriterionFunctionFromOptions(options_, models::usage::training);
}

GPUEngineTrain::~GPUEngineTrain() {}

SwappableModelTrainer::SwappableModelTrainer(Ptr<GPUEngineTrain> gpu) : engine_(gpu) {
}

SwappableModelTrainer::~SwappableModelTrainer() {
}

void SwappableModelTrainer::SetModel(Ptr<CPULoadedModel> from) {
  srcVocabs_ = from->SrcVocabs();
  trgVocab_  = from->TrgVocab();
  cpuModel_ = from;
}

std::vector<MemoryPiece::PtrType> SwappableModelTrainer::Parameters() const {
  return engine_->graph_->params()->toMemoryPieces();
}

void SwappableModelTrainer::Train(const std::vector<std::string> &input) {
  ABORT_IF(!trgVocab_, "GPULoadedModelTrain needs to be overwritten by a CPU model first.");

  auto state     = New<TrainingState>(engine_->options_->get<float>("learn-rate"));
  auto scheduler = New<Scheduler>(engine_->options_, state);
  auto optimizer = Optimizer(engine_->options_);
  scheduler->registerTrainingObserver(scheduler);
  scheduler->registerTrainingObserver(optimizer);

  std::vector<Ptr<Vocab>> allVocabs;
  allVocabs.reserve(srcVocabs_.size() + 1);
  allVocabs.insert(allVocabs.end(), srcVocabs_.begin(), srcVocabs_.end());
  allVocabs.emplace_back(trgVocab_);
  auto corpus = New<data::TextInput>(input, allVocabs, engine_->options_);
  data::BatchGenerator<data::TextInput> batchGenerator(corpus, engine_->options_, nullptr, false);

  // We reset the training graph to the original model parameters to prepare
  // for adapting it to the new inputs
  engine_->RecreateGraphAndBuilder();
  engine_->graph_->load(cpuModel_->Parameters(), true, true);

  scheduler->started();
  while(scheduler->keepGoing()) {
    batchGenerator.prepare();

    // LOG(info, "## NEW BATCHES");
    for(auto&& batch : batchGenerator) {
      if(!scheduler->keepGoing())
        break;

      // LOG(info, "### NEW BATCH");
      // Make an update step on the copy of the model
      auto lossNode = engine_->builder_->build(engine_->graph_, batch);
      engine_->graph_->forward();
      StaticLoss loss = *lossNode;
      engine_->graph_->backward();

      // Notify optimizer and scheduler
      optimizer->update(engine_->graph_, 1);
      scheduler->update(loss, batch);
    }
    if(scheduler->keepGoing())
      scheduler->increaseEpoch();
  }
  scheduler->finished();
}




  // ##### ^ above is stuff for runtime domain adaptation





void GPUEngineTranslate::SwapPointers(std::vector<MemoryPiece::PtrType> &with) {
  auto write_it = graph_->params()->begin();
  auto read_it = with.begin();
  for (; read_it != with.end(); ++write_it, ++read_it) {
    std::swap(*(*write_it)->val()->memory(), **read_it);
  }
}

GPUEngineTranslate::GPUEngineTranslate(Ptr<Options> options, size_t deviceIdx) 
  : options_(options), graph_(New<ExpressionGraph>(true)), myDeviceId_(LookupGPU(options, deviceIdx)), allocator_(myDeviceId_, 0, 128 * 1048576) {
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

GPUEngineTranslate::~GPUEngineTranslate() {}

GPULoadedModel::GPULoadedModel(Ptr<GPUEngineTranslate> gpu) : engine_(gpu) {
  for (auto &param : *engine_->graph_->params()) {
    names_.push_back(param->name());
    parameters_.push_back(engine_->allocator_.alloc(param->val()->memory()->size()));
  }
}

GPULoadedModel::~GPULoadedModel() {
  for (MemoryPiece::PtrType &p : parameters_) {
    engine_->allocator_.free(p);
  }
}

void GPULoadedModel::PointToParams(const SwappableModelTrainer &from) {
  ABORT_IF(engine_->myDeviceId_ != from.engine_->myDeviceId_, "TODO: copy across GPUs.");
  srcVocabs_ = from.srcVocabs_;
  trgVocab_  = from.trgVocab_;
  parameters_ = from.Parameters();
}

void GPULoadedModel::Load(const CPULoadedModel &from) {
  srcVocabs_ = from.SrcVocabs();
  trgVocab_ = from.TrgVocab();
  auto fromParams = from.Parameters();

  auto printParamsAndExit = [&]() {
    std::ostringstream paramNames;
    for(size_t i = 0; i < parameters_.size(); ++i) {
      paramNames << "  TO (" << names_[i] << ") size: " << parameters_[i]->size() << "\n";
    }
    for(size_t i = 0; i < fromParams.size(); ++i) {
      paramNames << "  FROM (" << fromParams[i].name << ") size: " << fromParams[i].size() << "\n";
    }
    LOG(error,
        "Attempting to load parameters with mismatched names or sizes:\n{}",
        paramNames.str());
    ABORT("Attempting to load parameters with mismatched names or sizes.");
  };

  // Sanity check
  if (parameters_.size() != fromParams.size())
    printParamsAndExit();

  for(size_t i = 0; i < parameters_.size(); ++i) {
    // Sanity check
    // Not sure if that's ok, but we don't check for size equality because for
    // some reason the target memory location sometimes can be bigger
    if (names_[i] != fromParams[i].name || parameters_[i]->size() < fromParams[i].size())
      printParamsAndExit();

    swapper::copyCpuToGpu(reinterpret_cast<char *>(parameters_[i]->data()),
                          fromParams[i].data(),
                          fromParams[i].size(),
                          engine_->myDeviceId_);
  }
}

Histories GPULoadedModel::Translate(const Ptr<data::CorpusBatch> batch) {
  ABORT_IF(!trgVocab_, "GPULoadedModel needs to be overwritten by a CPU model first.");
  // std::vector<uint8_t> outvec;
  // get(outvec, parameters_[0], engine_->graph_->getBackend());
  engine_->SwapPointers(parameters_);

  BeamSearch search(engine_->options_, engine_->scorers_, trgVocab_);
  Histories ret;
  ret.reserve(batch->size()); // TODO: input.size() was here previously, this is likely wrong

  auto result = search.search(engine_->graph_, batch);
  ret.insert(ret.end(), result.begin(), result.end());

  std::sort(ret.begin(), ret.end(),[](marian::Ptr<marian::History> a, marian::Ptr<marian::History> b){return a->getLineNum() < b->getLineNum();});

  engine_->SwapPointers(parameters_);
  return ret;
}

CPULoadedModel::CPULoadedModel(Ptr<Options> options, const std::string &parameters, const std::vector<std::string> &sourceVocabPaths, const std::string &targetVocabPath)
  : parameters_(io::loadItems(parameters)) {
  //Remap the parameter names if the model uses an older naming convention
  if (options->get<std::string>("type") == "amun") {
    bool tied = options->get<bool>("tied-embeddings-src") || options->get<bool>("tied-embeddings-all");
    Amun::remapIoItems(parameters_, tied);
  } else if (options->get<std::string>("type") == "nematus") {
    Nematus::remapIoItems(parameters_, options);
  }

  // Find the special element and remove it:
  auto pred = [](const io::Item &item) { return item.name == "special:model.yml"; };
  auto special_it = std::find_if(parameters_.begin(), parameters_.end(), pred);
  if (special_it != parameters_.end()) {
    parameters_.erase(special_it);
  }

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
