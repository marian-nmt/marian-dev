#pragma once

#include "common/definitions.h"
#include "data/batch_generator.h"
#include "models/model_base.h"
#include "optimizers/optimizers.h"
#include "training/scheduler.h"

namespace marian {

class GraphGroup {
protected:
  Ptr<Config> options_;
  Ptr<OptimizerBase> opt_;
  Ptr<Scheduler> scheduler_;
  std::vector<size_t> batch_size_perthread_;
  std::unordered_map<std::thread::id, size_t> threadIDs_;

  bool scaleLearningRate_;
  float avgBatchWords_;

public:
  GraphGroup(Ptr<Config> options)
      : options_(options),
        opt_(Optimizer(options)),
        scaleLearningRate_(options->get<bool>("batch-flexible-lr")),
        avgBatchWords_(options->get<float>("batch-normal-words")),
          batch_size_perthread_(options->get<std::vector<int> >("devices").size()) {}

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch>) = 0;

  virtual void load() = 0;

  virtual void save(bool = false) = 0;

  virtual void setScheduler(Ptr<Scheduler> scheduler) = 0;

  virtual Ptr<data::BatchStats> collectStats() = 0;
};
}
