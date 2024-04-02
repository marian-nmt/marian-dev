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

#ifdef _MSC_VER // MS Visual studio insists that this funtion is not being referenced although is being referenced by name as an argument
#pragma warning(push)
#pragma warning(disable: 4505) //Unreferenced local function has been removed
#endif
// to accumulate gradients norms, first undo sqrt, sum, re-apply sqrt.
// if one value is nonfinite propagate Nan into the reduction.
static inline void accNanOrNorm(float& lhs, float rhs) {
  if(isFinite(lhs) && isFinite(rhs)) {
    lhs = sqrtf(lhs * lhs + rhs * rhs);
  } else
    lhs = std::numeric_limits<float>::quiet_NaN();
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

/**
 *  Base class for managing the training process across one, multiple gpus,
 *  or even multiple machines with multiple gpus.
 */
class GraphGroup {
protected:
  Ptr<Options> options_;

  Ptr<ICommunicator> comm_; // [not null] communicator, e.g. NCCLCommunicator
  Ptr<IMPIWrapper> mpi_;    // [not null] all MPI-like communication goes through this (this is a dummy implementation if no MPI run)

  std::vector<DeviceId> devices_;                   // [deviceIndex]
  ShardingMode shardingMode_{ShardingMode::global}; // If local and multi-node training, shard only on local devices and do full sync (faster). If global shard across entire set of GPUs (more RAM).

  // common for all graph groups, individual graph groups decide how to fill them
  std::vector<Ptr<ExpressionGraph>> graphs_;            // [deviceIndex]
  std::vector<Ptr<models::ICriterionFunction>> models_; // [deviceIndex]
  std::vector<Ptr<OptimizerBase>> optimizerShards_;     // [deviceIndex]

  Ptr<io::ModelWeights> modelWeights_; // handle for model weights, we keep this around to make sure weights are not deallocated while we are still using them
  Ptr<Scheduler> scheduler_; // scheduler that keeps track of how much has been processed

  bool finalized_{false};    // 'true' if training has completed (further updates are no longer allowed)
  float typicalTrgBatchWords_{0}; // for dynamic batch sizing: typical batch size in words
  float averageEffectiveBatchSize_{0}; // record average effective batch size
  bool mbRoundUp_{true}; // round up batches for more efficient training but can make batch size less stable, disable with --mini-batch-round-up=false

  bool costScaling_{false};
  float costScalingFactor_{1.f}; // @TODO, add current costScalingFactor_ to trainingState for serialization
  size_t costScalingFreq_{2000};
  float costScalingMultiplier_{2.f};
  float costScalingFactorMinimum_{1.f};

  size_t noNanSeen_{0}; // @TODO, add current noNanSeen_ to trainingState for serialization
  size_t nanSeen_{0};

  bool checkGradientNan_{false};

  bool normalizeGradient_{false};
  bool normalizeGradientByAverageRatio_{true};

  bool dynamicGradientScaling_{false};
  float dynamicGradientScalingFactor_{2.f};
  bool dynamicGradientScalingUseLogs_{false};
  size_t dynamicGradientScalingFadeout_{0ul};

  // determines the number of input streams (i.e. input files or fields in the TSV input) that need
  // to be included in the batch, i.e. without alignments and weights
  size_t numberOfInputFiles();

public:
  GraphGroup(Ptr<Options> options, Ptr<IMPIWrapper> mpi);
  GraphGroup(Ptr<Options> options);

  void initGraphsAndOpts();
  void syncParametersAndShards();

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch> batch) = 0;

  // increase cost-scaling factor if no NaN has been detected for a
  // given number of iterations. Usually we increase by 2 which adds
  // one more bit for precision.
  void increaseCostScaleFactor();

  // call when a NaN was seen to decrease cost-scaling factor
  void decreaseCostScaleFactor();

  virtual void load();
  virtual void save(bool isFinal = false);

private:
  void load(const OptimizerBase::ScatterStateFunc& scatterFn);

  bool loadOptimizerState(const std::string& modelFileName,
                          const OptimizerBase::ScatterStateFunc& scatterFn);

  void save(bool isFinal,
            const OptimizerBase::GatherStateFunc& gatherOptimizerStateFn);

  void saveCheckPoint(const std::string& modelFileName,
                      bool isFinal,
                      bool doSaveOptimizerState,
                      const OptimizerBase::GatherStateFunc& gatherOptimizerStateFn);

  void saveOptimizerState(const std::string& modelFileName,
                          const OptimizerBase::GatherStateFunc& gatherFn);

public:
  // This function swaps out the current optimizer parameters with the smoothed version (provided smoothing is enabled).
  // Usually we will call this twice, to swap in and to swap out.
  void swapWithSmoothed();

  // This function replaces the current optimizer parameters with the smoothed version (provided smoothing is enabled).
  // This is different from swapping (swapping twice restores original state) as the original parameters get overwritten.
  void replaceWithSmoothed();

  bool isMainProcess() const { return mpi_->isMainProcess(); } // (we need this test a few times)
  void barrier() const { mpi_->barrier(); } // (we need this several times)

  void validate();

  virtual void finalize();

  virtual void setScheduler(Ptr<Scheduler> scheduler) = 0;

  float checkNanOrNorm(size_t i, size_t begin, size_t end);
  float executeAndCollectNorm(const std::function<float(size_t, size_t, size_t)>& task);

  float computeNormalizationFactor(float gNorm, size_t effectiveBatchSize);

  /**
   * Determine maximal batch size that can fit into the given workspace
   * so that reallocation does not happen. Rather adjust the batch size
   * based on the statistics collected here. Activated with
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

  virtual Ptr<data::BatchStats> collectStats(const std::vector<Ptr<Vocab>>& vocabs) = 0;

  // used to estimate the number of words in a batch and figure out statistics for batch growing etc.
  void setTypicalTrgBatchWords(size_t typicalTrgBatchWords);
  float getTypicalTrgBatchWords();
  void updateAverageTrgBatchWords(size_t trgBatchWords);

  // similar to above but counts the number of labels including delayed updates. This is used for gradient normalization.
  float getAverageEffectiveBatchSize();
  void updateAverageEffectiveBatchSize(size_t effectiveBatchSize);
};

}  // namespace marian
