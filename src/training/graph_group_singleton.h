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
  Ptr<models::ModelBase> builder_;
  Ptr<ExpressionGraph> graph_;

  void execute(Ptr<data::Batch> batch);

public:
  SingletonGraph(Ptr<Options> options, Ptr<IMPIWrapper> mpi)
      : GraphGroup(config) {
    ABORT_IF(mpi->numMPIProcesses() != 1, "SingletonGraph does not support multiple MPI processes");
    // Get device ID
    auto devices = Config::getDevices(options_);
    ABORT_IF(devices.size() != 1, "Only one device ID should be provided for singleton training");
    auto deviceId = devices[0];
    // Initialize graph
    graph_ = New<ExpressionGraph>();
    graph_->setDevice(deviceId);

    auto precisions = options_->get<std::vector<std::string>>("precision");
    graph_->setParameterType(typeFromString(precisions[0]));

    if(options_->get<bool>("check-nan"))
      graph_->setThrowNan(true);

    graph_->getBackend()->setClip(options_->get<float>("clip-gemm"));
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));

    opt_ = Optimizer(options_, graph_->allocator());
    builder_ = models::from_options(options_, models::usage::training);
  }

  void update(Ptr<data::Batch> batch) override {
    validate();
    execute(batch);
  }

  void load() override {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);

        builder_->load(graph_, name);

        opt_->load(name + ".optimizer.npz", {opt_}, {graph_->getBackend()},
          /*scatterStateFn=*/[&](const io::Item& data, const OptimizerBase::ScatterStateSetFunc& setFn) {
            setFn(/*localDeviceIndex=*/0, data.bytes.data(), data.bytes.data() + data.size());
          });
      } else if(options_->hasAndNotEmpty("pretrained-model")) {
        std::string init = options_->get<std::string>("pretrained-model");
        LOG(info,
            "Initialize model weights with the pre-trained model {}",
            init);
        builder_->load(graph_, init, false);
      }
    }
  }

  void save(bool final = false) override {
    auto saveGraph = graphFromOptimizer(graph_, {opt_});
    if(final && scheduler_)
      scheduler_->validate({saveGraph}, true);

    save(saveGraph, final);
  }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    std::string name = options_->get<std::string>("model");

    if(options_->get<bool>("overwrite")) {
      builder_->save(graph, name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      if(!final) {
        std::string numberOfBatches
            = scheduler_ ? std::to_string(scheduler_->numberOfBatches())
                         : "unknown";
        std::string nameOverwrite = name;
        nameOverwrite.replace(
            name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        builder_->save(graph, nameOverwrite);
      }

      builder_->save(graph, name, true);
      if(scheduler_)
        scheduler_->save(name);
    }

    opt_->save(name + ".optimizer.npz", {opt_},
      /*gatherStateFn=*/[&](const OptimizerBase::GatherStateGetFunc& getFn) {
        return getFn(/*localDeviceIndex=*/0);
      });
  }

  Ptr<data::BatchStats> collectStats(const std::vector<Ptr<Vocab>>& vocabs) {
    return GraphGroup::collectStats(graph_, builder_, vocabs);
  }

  virtual void finalize() override { finalized_ = true; }
};
}  // namespace marian
