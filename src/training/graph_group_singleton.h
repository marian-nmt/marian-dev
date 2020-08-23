#pragma once

#include "training/graph_group.h"
#include "common/filesystem.h"

#include <future>

namespace marian {

/**
 * Single GPU training
 */
class SingletonGraph : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) override;

private:
  void execute(Ptr<data::Batch> batch);
  void barrier() const override {} // dummy barrier

public:
  SingletonGraph(Ptr<Options> options, Ptr<IMPIWrapper> mpi)
      : GraphGroup(options) {

    LOG(warn, "This class only serves demonstration purposes. It should currently not be called from actual Marian code.");

    ABORT_IF(mpi->numMPIProcesses() != 1, "SingletonGraph does not support multiple MPI processes");
    ABORT_IF(devices_.size() != 1, "Only one device ID should be provided for singleton training");
    
    GraphGroup::initGraphs();

    optimizerShards_.push_back(Optimizer(options_));
    models_.push_back(models::createCriterionFunctionFromOptions(options_, models::usage::training));
  }

  void update(Ptr<data::Batch> batch) override {
    validate();
    execute(batch);
  }

  void load() override {
    auto scatterFn = [&](const io::Item& optimizerState, const OptimizerBase::ScatterStateSetFunc& setFn) {
      setFn(/*localDeviceIndex=*/0, optimizerState.bytes.data(), optimizerState.bytes.data() + optimizerState.size());
    };

    // This function loads the main parameters in the graphs.
    GraphGroup::load(scatterFn);
  }

  void save(bool isFinal = false) override {
    auto distParams = [](){}; // do nothing, only one process and GPU

    auto gatherOpt  = [&](const OptimizerBase::GatherStateGetFunc& getFn) {
      return getFn(/*localDeviceIndex=*/0); // dummy
    };

    GraphGroup::save(isFinal, distParams, gatherOpt, /*isMainProcess=*/true);
  }

  Ptr<data::BatchStats> collectStats(const std::vector<Ptr<Vocab>>& vocabs) {
    return GraphGroup::collectStats(graphs_[0], models_[0], vocabs);
  }

  virtual void finalize() override { finalized_ = true; }
};
}  // namespace marian
