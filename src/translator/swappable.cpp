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

namespace {
  DeviceId LookupGPU(const Ptr<Options> options, size_t deviceIdx) {
    auto devices = Config::getDevices(options);
    ABORT_IF(deviceIdx >= devices.size(), "GPU device index higher than configured.");
    return devices[deviceIdx];
  }
} // namespace

void get(std::vector<uint8_t> &v, MemoryPiece::PtrType mem, Ptr<Backend> backend) {
  v.resize(mem->size());
  gpu::copy(backend, mem->data<uint8_t>(), mem->data<uint8_t>() + mem->size(), v.data());
}

void GPUEngineTrain::SwapPointers(
    std::vector<MemoryPiece::PtrType> &with /*, std::vector<std::string> &with_names*/) {
  auto write_it = graph_->params()->begin();
  auto read_it = with.begin();
  // auto read_it_names  = with_names.begin();
  bool first = true;
  std::vector<uint8_t> outvec;
  for(; read_it != with.end(); ++write_it, ++read_it /*, ++read_it_names*/ ) {
    if (first){
      get(outvec, (*write_it)->val()->memory(), graph_->getBackend());
      get(outvec, *read_it, graph_->getBackend());
    }
    std::swap(*(*write_it)->val()->memory(), **read_it);
    // *graph_->params()->get(*read_it_names)->val()->memory() = std::move(**read_it);
    // assign(*graph_->params()->get(*read_it_names)->val()->memory(), **read_it);
    if(first) {
      get(outvec, (*write_it)->val()->memory(), graph_->getBackend());
      get(outvec, *read_it, graph_->getBackend());
      first = false;
    }
  }
  // graph_->params()->init(graph_->getBackend(), graph_->getDeviceId());
}

void GPUEngineTrain::Initialize(Ptr<data::Batch> batch) {
  if (!initialized_) {
    builder_->build(graph_, batch);
    graph_->forward();
    initialized_ = true;
  }
}

GPUEngineTrain::GPUEngineTrain(Ptr<Options> options, size_t deviceIdx) 
  : options_(options), graph_(New<ExpressionGraph>()), myDeviceId_(LookupGPU(options, deviceIdx)), allocator_(myDeviceId_, 0, 128 * 1048576) {
  ABORT_IF(myDeviceId_.type == DeviceType::cpu, "Swappable slot only works for GPU devices.");
  options_->set("inference", false);
  options_->set("shuffle", "none");

  // Create graph
  auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
  graph_->setDefaultElementType(typeFromString(prec[0]));
  graph_->setDevice(myDeviceId_);
  graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));

  builder_ = models::createCriterionFunctionFromOptions(options_, models::usage::training);
  // scorers_ = createScorers(options_);
  // for (auto scorer : scorers_) {
  //   scorer->init(graph_);
  //   // TODO lexical shortlists are not supported yet.
  // }
  // graph_->forward();
  // // TODO: reach into graph_->params() private members and free the parameter memory.
}

void GPUEngineTrain::recreateGraphAndBuilder() {
  // Create graph
  graph_ = New<ExpressionGraph>();
  auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
  graph_->setDefaultElementType(typeFromString(prec[0]));
  graph_->setDevice(myDeviceId_);
  graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));

  builder_ = models::createCriterionFunctionFromOptions(options_, models::usage::training);
}

GPUEngineTrain::~GPUEngineTrain() {}

GPULoadedModelTrain::GPULoadedModelTrain(Ptr<GPUEngineTrain> gpu) : engine_(gpu) {
  // NOTE: engine_ must contain an initialized graph already at this point
  // for (auto &param : *engine_->graph_->params()) {
  //   parameters_.push_back(engine_->allocator_.alloc(param->val()->memory()->size()));
  // }
}

// void GPULoadedModelTrain::AllocateParamsLike(const CPULoadedModel &from) {
//   for (auto &param : from.Parameters()) {
//     parameters_.push_back(engine_->allocator_.alloc(param.size()));
//   }
// }

GPULoadedModelTrain::~GPULoadedModelTrain() {
  // for (MemoryPiece::PtrType &p : parameters_) {
  //   engine_->allocator_.free(p);
  // }
}

// void GPULoadedModelTrain::Load(const GPULoadedModelTrain &from) {
//   srcVocabs_ = from.srcVocabs_;
//   trgVocab_ = from.trgVocab_;

//   ABORT_IF(engine_ != from.engine_, "TODO: copy across GPUs.");

//   for (size_t i = 0; i < parameters_.size(); ++i) {
//     swapper::copyGpuToGpu(reinterpret_cast<char*>(parameters_[i]->data()), reinterpret_cast<const char*>(from.parameters_[i]->data()), parameters_[i]->size(), engine_->myDeviceId_);
//   }
// }

void GPULoadedModelTrain::Load(Ptr<CPULoadedModel> from) {
  srcVocabs_ = from->SrcVocabs();
  trgVocab_  = from->TrgVocab();
  cpuModel_ = from;
}

// void GPULoadedModelTrain::Load(const CPULoadedModel &from) {
//   srcVocabs_ = from.SrcVocabs();
//   trgVocab_ = from.TrgVocab();
//   for (size_t i = 0; i < parameters_.size(); ++i) {
//     swapper::copyCpuToGpu(reinterpret_cast<char*>(parameters_[i]->data()), from.Parameters()[i].data(), from.Parameters()[i].size(), engine_->myDeviceId_);
//   }
// }

void GPULoadedModelTrain::Train(const std::vector<std::string> &input) {
  ABORT_IF(!trgVocab_, "GPULoadedModelTrain needs to be overwritten by a CPU model first.");
  // engine_->SwapPointers(parameters_);
  std::vector<uint8_t> outvec;
  // get(outvec, parameters_[0], engine_->graph_->getBackend());

  auto state     = New<TrainingState>(engine_->options_->get<float>("learn-rate"));
  auto scheduler = New<Scheduler>(engine_->options_, state);
  auto optimizer = Optimizer(engine_->options_);
  scheduler->registerTrainingObserver(scheduler);
  scheduler->registerTrainingObserver(optimizer);

  // LOG(info, "GAAAH: vocabs is {}", srcVocabs_);
  for (auto vocab: srcVocabs_) {
    LOG(info, "GAAAH: single vocab is {}", vocab);
  }

  std::vector<Ptr<Vocab>> allVocabs;
  allVocabs.reserve(srcVocabs_.size() + 1);
  allVocabs.insert(allVocabs.end(), srcVocabs_.begin(), srcVocabs_.end());
  allVocabs.emplace_back(trgVocab_);
  auto corpus = New<data::TextInput>(input, allVocabs, engine_->options_);  // @TODO dirty hack
  // auto corpus = New<data::TextInput>(input, srcVocabs_, engine_->options_); // @TODO dirty hack
  data::BatchGenerator<data::TextInput> batchGenerator(corpus, engine_->options_, nullptr, false); // @TODO if the asynchronous batch preparation = true, but we supply less text than the mini-batch size we crash

  bool first = true;
  scheduler->started();
  while(scheduler->keepGoing()) {
    batchGenerator.prepare();

    LOG(info, "## NEW BATCHES");
    for(auto&& batch : batchGenerator) {
      if(!scheduler->keepGoing())
        break;

      LOG(info, "### NEW BATCH");
      if(first) {
        // This is a bit awkward but for some reason
        // ICriterionFunction::build, which Initialize invokes underneath,
        // expects a batch. So, afaik, this is the first time where i can
        // invoke build and, as a result i can call SwapPointers only
        // afterwards. TODO: verify last claim.

        // Create graph
        engine_->recreateGraphAndBuilder();
        engine_->graph_->load(cpuModel_->Parameters(), true, true);
        engine_->Initialize(batch);
        std::vector<uint8_t> outvec;
        // get(outvec, parameters_[0], engine_->graph_->getBackend());
        // engine_->SwapPointers(parameters_);
        // get(outvec, parameters_[0], engine_->graph_->getBackend());
        first = false;
      }

      // Make an update step on the copy of the model
      auto lossNode = engine_->builder_->build(engine_->graph_, batch);
      // LOG(info, "Before: {}", engine_->graph_->params()->vals()->debug());
      engine_->graph_->forward();
      StaticLoss loss = *lossNode;
      engine_->graph_->backward();

      // auto out = engine_->graph_->params()->toMemoryPieces();

      // Notify optimizer and scheduler
      optimizer->update(engine_->graph_, 1);
      scheduler->update(loss, batch);
      // LOG(info, "After: {}", engine_->graph_->params()->vals()->debug());
    }
    if(scheduler->keepGoing())
      scheduler->increaseEpoch();
  }
  scheduler->finished();

  if(!first) {
    std::vector<uint8_t> outvec;
    // get(outvec, parameters_[0], engine_->graph_->getBackend());
    // engine_->SwapPointers(parameters_);
    // get(outvec, parameters_[0], engine_->graph_->getBackend());
    // does nothing, need a place for a breakpoint
    first = false;
  }
}




  // ##### ^ above is stuff for runtime domain adaptation





void GPUEngine::SwapPointers(std::vector<MemoryPiece::PtrType> &with) {
  auto write_it = graph_->params()->begin();
  auto read_it = with.begin();
  for (; read_it != with.end(); ++write_it, ++read_it) {
    std::swap(*(*write_it)->val()->memory(), **read_it);
  }
}

GPUEngine::GPUEngine(Ptr<Options> options, size_t deviceIdx) 
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

void GPULoadedModel::Load(const GPULoadedModel &from) {
  srcVocabs_ = from.srcVocabs_;
  trgVocab_ = from.trgVocab_;

  ABORT_IF(engine_ != from.engine_, "TODO: copy across GPUs.");

  for (size_t i = 0; i < parameters_.size(); ++i) {
    swapper::copyGpuToGpu(reinterpret_cast<char*>(parameters_[i]->data()), reinterpret_cast<const char*>(from.parameters_[i]->data()), parameters_[i]->size(), engine_->myDeviceId_);
  }
}

// void GPULoadedModel::Load(const GPULoadedModelTrain &from) {
//   srcVocabs_ = from.srcVocabs_;
//   trgVocab_  = from.trgVocab_;

//   ABORT_IF(engine_->myDeviceId_ != from.engine_->myDeviceId_, "TODO: copy across GPUs.");

//   for(size_t i = 0; i < parameters_.size(); ++i) {
//     swapper::copyGpuToGpu(reinterpret_cast<char *>(parameters_[i]->data()),
//                           reinterpret_cast<const char *>(from.parameters_[i]->data()),
//                           parameters_[i]->size(),
//                           engine_->myDeviceId_);
//   }
// }

void GPULoadedModel::PointToParams(const GPULoadedModelTrain &from) {
  ABORT_IF(engine_->myDeviceId_ != from.engine_->myDeviceId_, "TODO: copy across GPUs.");
  srcVocabs_ = from.srcVocabs_;
  trgVocab_  = from.trgVocab_;
  // TODO: this might be wrong and could be droped in favor of using SwapPointers
  parameters_ = from.engine_->graph_->params()->toMemoryPieces();
}

void GPULoadedModel::Load(const CPULoadedModel &from) {
  srcVocabs_ = from.SrcVocabs();
  trgVocab_ = from.TrgVocab();
  for (size_t i = 0; i < parameters_.size(); ++i) {
    swapper::copyCpuToGpu(reinterpret_cast<char*>(parameters_[i]->data()), from.Parameters()[i].data(), from.Parameters()[i].size(), engine_->myDeviceId_);
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

Histories GPULoadedModel::Translate(const Ptr<data::CorpusBatch> batch) {
  ABORT_IF(!trgVocab_, "GPULoadedModel needs to be overwritten by a CPU model first.");
  std::vector<uint8_t> outvec;
  get(outvec, parameters_[0], engine_->graph_->getBackend());
  engine_->SwapPointers(parameters_);
  // LOG(info, "Before translation: {}", engine_->graph_->params()->vals()->debug());

  BeamSearch search(engine_->options_, engine_->scorers_, trgVocab_);
  Histories ret;
  ret.reserve(batch->size()); // TODO: input.size() was here previously, this is likely wrong

  auto result = search.search(engine_->graph_, batch);
  ret.insert(ret.end(), result.begin(), result.end());

  std::sort(ret.begin(), ret.end(),[](marian::Ptr<marian::History> a, marian::Ptr<marian::History> b){return a->getLineNum() < b->getLineNum();});

  // LOG(info, "After translation: {}", engine_->graph_->params()->vals()->debug());
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
