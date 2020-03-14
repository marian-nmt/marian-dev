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
  
  std::vector<DeviceId> devices_;                   // [deviceIndex]
  
  // common for all graph groups, individual graph groups decide how to fill them
  std::vector<Ptr<ExpressionGraph>> graphs_;            // [deviceIndex]
  std::vector<Ptr<models::ICriterionFunction>> models_; // [deviceIndex]
  std::vector<Ptr<OptimizerBase>> optimizerShards_;     // [deviceIndex]

  Ptr<Scheduler> scheduler_; // scheduler that keeps track of how much has been processed
  
  bool finalized_{false};    // 'true' if training has completed (further updates are no longer allowed)
  size_t typicalTrgBatchWords_{ 0 }; // for dynamic batch sizing: typical batch size in words

  bool costScale_{false};
  float costScaleFactor_{1.f}; // @TODO, add current costScaleFactor_ to trainingState for serialization
  size_t costScaleFreq_{2000};
  float costScaleMultiplier_{2.f};
  float costScaleNanTolerance_{0.f};
  size_t costScaleNanRange_{1};
  float costScaleFactorMinimum_{1.f}; // @TODO make this configureable
  size_t noNanSeen_{0}; // @TODO, add current noNanSeen_ to trainingState for serialization
  size_t nanSeen_{0};

  bool checkGradientNorm_{false};
  size_t checkGradientNormWindow_{100};
  float checkGradientNormFactor_{4.f};

  bool checkGradientNan_{false};

public:
  GraphGroup(Ptr<Options> options, const std::vector<DeviceId> devices);
  GraphGroup(Ptr<Options> options);

  void initGraphs();

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch> batch) = 0;

  // increase cost-scaling factor if no NaN has been detected for a
  // given number of iterations. Usually we increase by 2 which adds
  // one more bit for precision.
  void increaseCostScaleFactor();

  // call when a NaN was seen to decrease cost-scaling factor
  void decreaseCostScaleFactor();

  virtual void load() = 0;

  void load(const OptimizerBase::ScatterStateFunc& scatterFn);

  void restoreFromCheckpoint(const OptimizerBase::ScatterStateFunc& scatterFn);

  virtual void save(bool isFinal = false) = 0;

  virtual void barrier() const = 0; // Used by multi-device training

  void save(bool isFinal,
            const std::function<void()>& distributeParamtersFn,
            const OptimizerBase::GatherStateFunc& gatherOptimizerStateFn,
            bool isMainProcess);

  void saveModel(bool isFinal = false);

  void saveCheckpoint(const OptimizerBase::GatherStateFunc& gatherFn);

  void swapWithSmoothed(const std::vector<Ptr<ExpressionGraph>>& graphs,
                        const std::vector<Ptr<OptimizerBase>>& opts,
                        const std::function<void()>& distribute = [](){});

  void swapWithOriginal(const std::vector<Ptr<ExpressionGraph>>& graphs,
                        const std::vector<Ptr<OptimizerBase>>& opts,
                        const std::function<void()>& distribute = [](){});

  void validate();

  virtual void finalize();

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
                                             Ptr<models::ICriterionFunction> model,
                                             const std::vector<Ptr<Vocab>>& vocabs,
                                             double multiplier = 1.);

  void setTypicalTrgBatchWords(size_t typicalTrgBatchWords);
};

}  // namespace marian