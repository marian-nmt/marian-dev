#include "training/graph_group.h"

namespace marian {

GraphGroup::GraphGroup(Ptr<Options> options, const std::vector<DeviceId> devices)
  : options_(options),
    devices_(devices) {
  if(options_->hasAndNotEmpty("cost-scaling")) {
    auto vcs = options_->get<std::vector<std::string>>("cost-scaling");
    costScale_ = true;
    float costExponent = std::stof(vcs[0]);
    costScaleFactor_ = std::pow(2.0f, costExponent);
    
    if(vcs.size() > 1) costScaleFreq_ = std::stoul(vcs[1]);
    if(vcs.size() > 2) costScaleMultiplier_ = std::stof(vcs[2]);
    if(vcs.size() > 3) costScaleNanTolerance_ = std::stof(vcs[3]);
    if(vcs.size() > 4) costScaleNanRange_ = std::stoul(vcs[4]);
    if(vcs.size() > 5) costScaleFactorMinimum_ = std::stof(vcs[5]);
    

    LOG_ONCE(info,
             "Training with cost scaling - factor: 2^{} = {}, frequency: {}, multiplier: {}, tolerance: {}, range: {}",
             costExponent,
             costScaleFactor_,
             costScaleFreq_,
             costScaleMultiplier_,
             costScaleNanTolerance_,
             costScaleNanRange_);
  }

  if(options_->hasAndNotEmpty("check-gradient-norm")) {
    auto vgc = options_->get<std::vector<std::string>>("check-gradient-norm");
    checkGradientNorm_ = true;
    if(vgc.size() > 0) checkGradientNormWindow_ = std::stoul(vgc[0]);
    if(vgc.size() > 1) checkGradientNormFactor_ = std::stof(vgc[1]);

    LOG_ONCE(info,
             "Checking gradient norm with window {} and factor {:.2f}",
             checkGradientNormWindow_,
             checkGradientNormFactor_);
  }

  if(options_->get<bool>("check-gradient-nan")) {
    checkGradientNan_ = true;
    LOG_ONCE(info, "Checking gradient for NaN");
  }
}

GraphGroup::GraphGroup(Ptr<Options> options)
  : GraphGroup(options, Config::getDevices(options)) {}

void GraphGroup::initGraphs() {
  for(auto graph : graphs_) {
    // @TODO: validate precisions in config
    auto precisions = options_->get<std::vector<std::string>>("precision");
    Type parameterType = typeFromString(precisions[0]);
    // Type saveType = typeFromString(precisions[2]); // @TODO: respect this

    graph->setDefaultElementType(parameterType);
    graph->setCheckpointing(options_->get<bool>("gradient-checkpointing"));

    if(options_->get<bool>("check-nan")) // @TODO: add to other places
      graph->setThrowNaN(true);

    graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    graph->getBackend()->setClip(options_->get<float>("clip-gemm"));
  }
}

// increase cost-scaling factor if no NaN has been detected for a
// given number of iterations. Usually we increase by 2 which adds
// one more bit for precision.
void GraphGroup::increaseCostScaleFactor() {
  if(!costScale_)
    return;

  noNanSeen_++;

  size_t total = nanSeen_ + noNanSeen_;
  float nanPercent = noNanSeen_ == (float)nanSeen_ / (float)total; // total is at least 1 because of noNanSeen_++

  if(noNanSeen_ % costScaleFreq_ == 0) {
    costScaleFactor_ *= costScaleMultiplier_;
    LOG(info,
        "NaN/Inf percentage {:.2f} after {} gradient updates. Increasing cost-scaling factor to {}",
        nanPercent,
        total,
        costScaleFactor_);

    // Resetting counts after cost-scale change
    noNanSeen_ = 0;
    nanSeen_ = 0;
  }
}

// call when a NaN was seen to decrease cost-scaling factor
void GraphGroup::decreaseCostScaleFactor() {
  if(!costScale_)
    return;

  nanSeen_++;
  
  size_t total = nanSeen_ + noNanSeen_;
  float nanPercent = (float)nanSeen_ / (float)total; // total is at least 1 because of nanSeen_++
  if(total >= costScaleNanRange_ && nanPercent > costScaleNanTolerance_) {
    if(costScaleFactor_ > costScaleFactorMinimum_) {
      costScaleFactor_ /= costScaleMultiplier_;
      LOG(warn,
          "NaN/Inf percentage {:.2f} in {} gradient updates, reducing cost-scaling factor to {}",
          nanPercent,
          total,
          costScaleFactor_);
    } else {
      // @TODO: think if should this rather abort?
      LOG(warn,
          "NaN/Inf percentage {:.2f} in {} gradient updates, but cost-scaling factor {} is already at minimum",
          nanPercent,
          total,
          costScaleFactor_);
    }

    // Resetting counts after cost-scale change
    noNanSeen_ = 0;
    nanSeen_ = 0;
  }
}

void GraphGroup::load(const OptimizerBase::ScatterStateFunc& scatterFn) {
  /*
  if not no-reload (=> i.e. do reload):
    restore scheduler
    if checkpoint is available or not no-reload-checkpoint:
      reload from checkpoint
    else if model is available:
      reload from model, but warn that no checkpoint was used and the model could be smoothed
  else if pretrained-model path given:
    initialize matching weights from pretrained model
  else:
    (implicitly) don't do anything => initialize randomly later
  */

  if(!options_->get<bool>("no-reload")) {
    std::string name = options_->get<std::string>("model");

    if(filesystem::exists(name)) {
      if(scheduler_)
        scheduler_->load(name);

      std::string nameGraph = name;
      size_t i = 0;
      for(auto graph : graphs_)
        models_[i++]->load(graph, nameGraph); // we just load it N times from disk (it'll be in disk cache after the first)

      restoreFromCheckpoint(scatterFn);

    } else if(options_->hasAndNotEmpty("pretrained-model")) {
      std::string nameInit = options_->get<std::string>("pretrained-model");
      LOG(info,
          "[training] Initializing model weights with pre-trained model {}",
          nameInit);

      size_t i = 0;
      for(auto graph : graphs_)
        models_[i++]->load(graph, nameInit, false);
    }
  }
}

void GraphGroup::restoreFromCheckpoint(const OptimizerBase::ScatterStateFunc& scatterFn) {
  /*
  if model checkpoint is available:
    - load model from checkpoint, not from model.npz
    - abort if checkpoint model and graph size do not match, probably due to different model or precision
  */

  std::string name = options_->get<std::string>("model");
  std::string checkpointName = name + ".optimizer.npz"; // @TODO: change to .checkpoint.npz, would break backwards compat

  if(!filesystem::exists(checkpointName)) {
    LOG(warn, "No checkpoint found, parameters reloaded from last inference model");
    return;
  }

  auto items = io::loadItems(checkpointName);

  // @TODO: probably we want to have the list of DeviceIds as an attribute
  std::vector<Ptr<Backend>> backends;
  for(auto graph : graphs_)
    backends.push_back(graph->getBackend());
  optimizerShards_[0]->load(items, optimizerShards_, backends, scatterFn);

  // restore the graph parameters from the checkpoint master copy.
  auto found = std::find_if(items.begin(), items.end(),
    [](const io::Item& item) { return item.name == "master_parameters"; });

  if(found == items.end()) {
    LOG(warn, "No master parameters found in checkpoint, parameters reloaded from last inference model");
    return;
  }

  auto& masterParameters = *found;
  for(auto graph : graphs_) {
    graph->forward(); // allocate graph parameter memory and initialize parameters from inference model. This needs to
    // run a full forward pass over the paramters to allocato the parameters values in order (by parameter name).
    // Just doing graph->params()->allocateForward() is not sufficient.
    ABORT_IF(graph->params()->vals()->shape() != masterParameters.shape,
             "Graph parameter sizes and master copy parameter sizes in checkpoint do not match");

    // Convert type of io::Item to match graph parameter type.
    if(masterParameters.type != graph->params()->vals()->type())
      masterParameters.convert(graph->params()->vals()->type());

    graph->params()->vals()->set(masterParameters); // @TODO: make this work for fp16
    graph->clear();
  }

  LOG(info, "[training] Master parameters and optimizers restored from training checkpoint {} and {}", name, checkpointName);
}

void GraphGroup::save(bool isFinal,
                      const std::function<void()>& distributeParamtersFn,
                      const OptimizerBase::GatherStateFunc& gatherOptimizerStateFn,
                      bool isMainProcess) {
  barrier(); // (for better grouping of log messages)
  if(isMainProcess) { // only save from one MPI process
    // bring the smoothed model in
    // Note that it is sharded. For multi-node, it is sharded over multiple machines, so this is a network access.
    // Also note that the swap must run on all MPI processes concurrently, although only one actually validates.

    swapWithSmoothed(graphs_, optimizerShards_, distributeParamtersFn);

    // do final validation
    if(isFinal && scheduler_)
      scheduler_->validate(graphs_, isFinal);

    barrier();// (for better grouping of log messages)
    // save main model file
    saveModel(isFinal);  // if not overwrite then save a copy with number of updates in the model pathname

    swapWithOriginal(graphs_, optimizerShards_, distributeParamtersFn);

    saveCheckpoint(gatherOptimizerStateFn);
  }
  barrier(); // (for better grouping of log messages)
}

void GraphGroup::saveModel(bool isFinal) {
  std::string name = options_->get<std::string>("model");

  if(options_->get<bool>("overwrite")) {
    models_[0]->save(graphs_[0], name, /*saveTranslatorConfig=*/true);
    // save scheduler-related state
    if(scheduler_)
      scheduler_->save(name);
  } else {
    if(!isFinal) { // save a model with iteration number
      std::string numberOfBatches
          = scheduler_ ? std::to_string(scheduler_->numberOfBatches())
                        : "unknown";
      std::string nameOverwrite = name;
      nameOverwrite.replace(name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
      models_[0]->save(graphs_[0], nameOverwrite);
    }

    models_[0]->save(graphs_[0], name, /*saveTranslatorConfig=*/true);

    // save scheduler-related state
    if(scheduler_)
      scheduler_->save(name);
  }
}

void GraphGroup::saveCheckpoint(const OptimizerBase::GatherStateFunc& gatherFn) {
  std::string name = options_->get<std::string>("model");
  std::string checkpointName = name + ".optimizer.npz"; // @TODO: change to .checkpoint.npz, would break backwards compat

  std::vector<io::Item> items;
  optimizerShards_[0]->save(items,
                            optimizerShards_,
                            gatherFn);

  auto found = std::find_if(items.begin(), items.end(),
    [](const io::Item& item) { return item.name == "master_parameters"; });

  if(found == items.end()) {
    // if the optimizer does not provide a master parameters copy (the default when training with full precision)
    // then dump the parameters of graphs_[0] into the checkpoint. This should be called when the original parameters
    // are in the graph, not the smoothed version. Here we are getting called after a double swap, so that should be
    // the case.
    io::Item masterParameters;
    graphs_[0]->params()->vals()->get(masterParameters, "master_parameters");
    items.push_back(masterParameters);
  }

  LOG(info, "[training] Saving training checkpoint to {} and {}", name, checkpointName);
  io::saveItems(checkpointName, items);
}

void GraphGroup::swapWithSmoothed(const std::vector<Ptr<ExpressionGraph>>& graphs,
                                  const std::vector<Ptr<OptimizerBase>>& opts,
                                  const std::function<void()>& distribute) {
  ABORT_IF(graphs.size() != opts.size(), "Number of graphs and optimizers has to be equal ({} != {})", graphs.size() != opts.size());
  for(size_t i = 0; i < graphs.size(); ++i)
    opts[i]->swapWithSmoothed(graphs[i], i, graphs.size(), /*swapAvg=*/true);
  distribute();
}

void GraphGroup::swapWithOriginal(const std::vector<Ptr<ExpressionGraph>>& graphs,
                                  const std::vector<Ptr<OptimizerBase>>& opts,
                                  const std::function<void()>& distribute) {
  ABORT_IF(graphs.size() != opts.size(), "Number of graphs and optimizers has to be equal ({} != {})", graphs.size() != opts.size());
  for(size_t i = 0; i < graphs.size(); ++i)
    opts[i]->swapWithSmoothed(graphs[i], i, graphs.size(), /*swapAvg=*/false);
  distribute();
}

void GraphGroup::validate() {
  ABORT_IF(finalized_, "Training has already finished.");
}

void GraphGroup::finalize() {
  finalized_ = true;
}

/**
 * Determine maximal batch size that can fit into the given workspace
 * so that reallocation does not happen. Rather adjust the batch size
 * based on the stastistics collected here. Activated with
 * `--mini-batch-fit`.
 * In a multi-GPU scenario, the first GPU is used to determine the size.
 * The actual allowed size is then determined by multiplying it with the
 * number of devices, which is passed in as the 'multiplier'.
 */
// @TODO: Can this be made const? It seems wrong to have a stateful method that still returns a result.
Ptr<data::BatchStats> GraphGroup::collectStats(Ptr<ExpressionGraph> graph,
                                               Ptr<models::ICriterionFunction> model,
                                               const std::vector<Ptr<Vocab>>& vocabs,
                                               double multiplier) {
  // this runs with fake values, we do not care for overflow/underflow
  bool throwNan = graph->getThrowNaN();

  //graph->setFake(true);
  graph->setThrowNaN(false);

  auto stats = New<data::BatchStats>();

  size_t numFiles = options_->get<bool>("tsv", false)
                        ? options_->get<size_t>("tsv-fields")
                        : options_->get<std::vector<std::string>>("train-sets").size();

  // Initialize first batch to step size
  size_t first = options_->get<size_t>("mini-batch-fit-step");

  // Increase batch size and sentence length by this step size
  size_t step = options_->get<size_t>("mini-batch-fit-step");

  size_t maxLength = options_->get<size_t>("max-length");
  maxLength = (size_t)(std::ceil(maxLength / (float)step) * step);

  // this should be only one class label per line on input, hence restricting length to 1
  std::vector<size_t> localMaxes(numFiles, maxLength);
  auto inputTypes = options_->get<std::vector<std::string>>("input-types", {});
  for(int i = 0; i < inputTypes.size(); ++i)
    if(inputTypes[i] == "class")
      localMaxes[i] = 1;

  size_t maxBatch = 512;
  bool fits = true;
  while(fits) {
    std::vector<size_t> lengths(numFiles, first);

    for(int j = 0; j < lengths.size(); ++j) // apply length restrictions
      lengths[j] = std::min(lengths[j], localMaxes[j]);

    auto batch = data::CorpusBatch::fakeBatch(lengths, vocabs, maxBatch, options_);
    auto loss = model->build(graph, batch);
    fits = graph->fits();
    if(fits)
      maxBatch *= 2;
  }

  // Do a binary search for maxmimum batch size that fits into given workspace memory
  // for a tested sentence length.
  for(size_t i = step; i <= maxLength; i += step) {
    size_t start = 1;
    size_t end = maxBatch;

    std::vector<size_t> lengths(numFiles, i);
    for(int j = 0; j < lengths.size(); ++j)  // apply length restrictions
      lengths[j] = std::min(lengths[j], localMaxes[j]);
    fits = true;

    do {
      size_t current = (start + end) / 2;
      auto batch = data::CorpusBatch::fakeBatch(lengths, vocabs, current, options_);
      auto loss = model->build(graph, batch);
      fits = graph->fits();

      LOG(debug, "[batching] length: {} - size: {} - fits: {}", lengths[0], current, fits);

      if(fits) {
        stats->add(batch, multiplier);
        start = current + 1;
      } else {
        end = current - 1;
      }
    } while(end >= start);

    maxBatch = start;
  }

  // set back to original value for aborting on NaN or Inf
  graph->setThrowNaN(throwNan);
  // graph->setFake(false);

  return stats;
}

void GraphGroup::setTypicalTrgBatchWords(size_t typicalTrgBatchWords) { // needed for dynamic MB scaling
  typicalTrgBatchWords_ = typicalTrgBatchWords;
}

}
