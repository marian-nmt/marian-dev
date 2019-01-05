#pragma once

#include "common/definitions.h"
#include "common/options.h"
#include "data/batch_generator.h"
#include "graph/expression_graph.h"
#include "models/model_base.h"
#include "optimizers/optimizers.h"
#include "training/scheduler.h"
#include "training/communicator.h"

namespace marian {

/**
 *  Base class for managing the training process across one, multiple gpus,
 *  or even multiple machines with multiple gpus.
 */
class GraphGroup {
protected:
  Ptr<Options> options_;
  Ptr<OptimizerBase> opt_;
  Ptr<Scheduler> scheduler_; // scheduler that keeps track of how much has been processed
  bool finalized_{false};    // 'true' if training has completed (further updates are no longer allowed)
  size_t typicalTrgBatchWords_{ 0 }; // for dynamic batch sizing: typical batch size in words

  bool costScale_{false};
  float costScaleFactor_{1.f}; // @TODO, add current costScaleFactor_ to trainingState for serialization
  size_t costScaleFreq_{2000};
  float costScaleMultiplier_{2.f};
  size_t noNanSeen_{0}; // @TODO, add current noNanSeen_ to trainingState for serialization

public:
  GraphGroup(Ptr<Options> options) : options_(options) {
    if(options_->has("cost-scaling")) {
      auto vcs = options_->get<std::vector<std::string>>("cost-scaling");
      costScale_ = true;
      costScaleFactor_ = std::stof(vcs[0]);
      costScaleFreq_ = std::stoul(vcs[1]);
      costScaleMultiplier_ = std::stof(vcs[2]);

      LOG_ONCE(info,
               "Training with cost scaling - factor: {}, frequency: {}, multiplier: {}",
               costScaleFactor_,
               costScaleFreq_,
               costScaleMultiplier_);
    }
  }

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch> batch) = 0;

  // increase cost-scaling factor if no NaN has been detected for a
  // given number of iterations. Usually we increase by 2 which adds
  // one more bit for precision.
  void increaseCostScaleFactor() {
    if(!costScale_)
      return;

    noNanSeen_++;
    if(noNanSeen_ % costScaleFreq_ == 0) {
      costScaleFactor_ *= costScaleMultiplier_;
      LOG(info,
          "No NaN/Inf seen for {} updates. Increasing cost-scaling factor to {}",
          noNanSeen_,
          costScaleFactor_);
    }
  }

  // call when a NaN was seen to decrease cost-scaling factor
  void decreaseCostScaleFactor() {
    if(!costScale_)
      return;

    costScaleFactor_ /= costScaleMultiplier_;
    LOG(warn,
        "Seen NaN/Inf in gradient, skipping update, reducing cost-scaling factor to {}",
        costScaleFactor_);
    noNanSeen_ = 0;
  }

  virtual void load() = 0;

  virtual void save(bool isFinal = false) = 0;

  void validate() {
    ABORT_IF(finalized_, "Training has already finished.");
  }

  virtual void finalize() {
    finalized_ = true;
  }

  virtual void setScheduler(Ptr<Scheduler> scheduler) = 0;

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
  virtual Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph,
                                             Ptr<models::ModelBase> model,
                                             double multiplier = 1.) {
    // this runs with fake values, we do not care for overflow/underflow
    bool throwNan = graph->getThrowNan();
    graph->setThrowNan(false);

    auto stats = New<data::BatchStats>();

    size_t numFiles
        = options_->get<std::vector<std::string>>("train-sets").size();

    // Initialize first batch to step size
    size_t first = options_->get<size_t>("mini-batch-fit-step");

    // Increase batch size and sentence length by this step size
    size_t step = options_->get<size_t>("mini-batch-fit-step");

    size_t maxLength = options_->get<size_t>("max-length");
    maxLength = (size_t)(std::ceil(maxLength / (float)step) * step);

    size_t maxBatch = 512;
    bool fits = true;
    while(fits) {
      std::vector<size_t> lengths(numFiles, first);
      auto batch = data::CorpusBatch::fakeBatch(lengths, maxBatch, options_);
      auto cost = model->build(graph, batch);
      fits = graph->fits();
      if(fits)
        maxBatch *= 2;
    }

    for(size_t i = step; i <= maxLength; i += step) {
      size_t start = 1;
      size_t end = maxBatch;

      std::vector<size_t> lengths(numFiles, i);
      fits = true;

      do {
        size_t current = (start + end) / 2;
        auto batch = data::CorpusBatch::fakeBatch(lengths, current, options_);
        auto cost = model->build(graph, batch);
        fits = graph->fits();

        if(fits) {
          stats->add(batch, multiplier);
          start = current + 1;
        } else {
          end = current - 1;
        }
      } while(end - start > step);

      maxBatch = start;
    }

    // set back to original value for aborting on NaN or Inf
    graph->setThrowNan(throwNan);
    return stats;
  }

  void setTypicalTrgBatchWords(size_t typicalTrgBatchWords) { // needed for dynamic MB scaling
    typicalTrgBatchWords_ = typicalTrgBatchWords;
  }
};

/**
 *  Base class for multi-node versions of GraphGroups.
 */
class MultiNodeGraphGroupBase : public GraphGroup {
  using Base = GraphGroup;

protected:
  Ptr<IMPIWrapper> mpi_; // all MPI-like communication goes through this

  /** Devices (GPUs) on this node. */
  std::vector<size_t> devices_; // [num local GPUs]

  /** Graph builders for clients (which run forward and backward passes). */
  std::vector<Ptr<models::ModelBase>> clientBuilders_;

  /** Graphs of clients. One entry per GPU on this node. */
  std::vector<Ptr<ExpressionGraph>> clientGraphs_; // [num local GPUs]

public:
  MultiNodeGraphGroupBase(Ptr<Options> options)
    : Base(options) {

    // Setup MPI
    setupMPI();

    // Set up devices for this node
    std::vector<size_t> devices; // set of GPU device ids for this MPI process
    for (auto& d : Config::getDevices(options_))
      devices.push_back(d.no);
    loadDeviceConfig(devices); // set up numberClientsOfNodes_[] and devices_[]

    // Create builders and graphs for clients; that is, for each GPU we use on this node.
    for (size_t i = 0; i < devices_.size(); i++) {
      clientGraphs_.push_back(New<ExpressionGraph>());

      if(options_->get<bool>("fp16"))
          clientGraphs_[i]->setParameterType(Type::float16);

      clientGraphs_[i]->setDevice({ devices_[i], DeviceType::gpu });
      clientGraphs_[i]->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      clientBuilders_.push_back(models::from_options(options_, models::usage::training));
    }
  }

  /**
   * Setup MPI world size and rank of this node.
   */
  void setupMPI() {
    mpi_ = initMPI(/*multiThreaded=*/!options_->get<bool>("sync-sgd"));
  }

  /**
   * Load the GPU configuration of this node (i.e. which GPUs to use) and the
   * number of GPUs on the other nodes.
   */
  // deviceConfig has this format
  //  - for each node
  //     - number of GPUs on that node
  //     - GPU ids for that node
  // e.g. 0:0 1 1: 2 3 -> (2, (0, 1)) (2, (2,3))
  void loadDeviceConfig(std::vector<size_t> deviceConfig) {
    // parse device config array
    size_t index = 0; // cursor for next()
    auto next = [&]() { // helper function to get the next item
      ABORT_IF(index == deviceConfig.size(), "mal-formed device config array??");
      return deviceConfig[index++];
    };
    std::vector<std::vector<size_t>> allDevices(mpi_->numMPIProcesses());
    for (auto& devices : allDevices) {
      devices.resize(next());
      for (auto& device : devices)
        device = next();
    }
    ABORT_IF(index != deviceConfig.size(), "mal-formed device config array??");

    // validate
    ABORT_IF(allDevices.front().size() == 0, "no devices specified??");
    for (auto& devices : allDevices) {
      ABORT_IF(devices.size() != allDevices.front().size(), "all MPI nodes must use the same number of devices");
    }

    // get our own config
    devices_ = allDevices[mpi_->myMPIRank()];

    // log
    LOG(info, "[mpi rank {}] device configuration", mpi_->myMPIRank());
    for (auto& device : devices_)
      LOG(info, "[mpi rank {}]  - {}", mpi_->myMPIRank(), device);
  }

  virtual void finalize() override {
    if (mpi_)
      finalizeMPI(std::move(mpi_));
    Base::finalize();
  }
};

class TempExpressionGraph : public ExpressionGraph {
private:
  Ptr<Allocator> allocator_;
  MemoryPiece::PtrType memory_;

  // this is private, should only be used within copyParams()
  void setDevice(DeviceId deviceId = {0, DeviceType::gpu},
                 Ptr<Device> device = nullptr) override {
    ExpressionGraph::setDevice(deviceId, device);
  }

public:
  TempExpressionGraph(Ptr<Allocator> allocator)
    : ExpressionGraph(/*inference=*/true, /*optimize=*/false), 
      allocator_(allocator) {
  }

  void copyParams(Ptr<ExpressionGraph> graph) override {
    if(memory_)
      allocator_->free(memory_);

    Type   graphType = graph->params()->vals()->type();
    size_t graphSize = graph->params()->vals()->size();
    
    auto tempDevice = New<cpu::WrappedDevice>(allocator_->getDeviceId()); // @TODO: move out of namespace cpu
    memory_ = allocator_->alloc(graphSize, graphType);
    tempDevice->set(memory_->data(), memory_->size());
    this->setDevice(allocator_->getDeviceId(), tempDevice);
    this->setParameterType(graphType);
  
    ExpressionGraph::copyParams(graph);
  }

  ~TempExpressionGraph() {
    if(memory_)
      allocator_->free(memory_);
  }
};

static Ptr<ExpressionGraph> graphFromOptimizer(Ptr<ExpressionGraph> graph, const std::vector<Ptr<OptimizerBase>>& /*opts*/, bool /*getAverage*/ = true) {
  // @TODO: implement function that creates a temporary graph from input graph and shared optimizers
  auto tempGraph = New<TempExpressionGraph>(graph->allocator());
  
  tempGraph->copyParams(graph);
  tempGraph->reuseWorkspace(graph);
  
  return tempGraph;
}

}  // namespace marian
