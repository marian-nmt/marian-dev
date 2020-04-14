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

// With -Ofast enabled gcc will fail to identify NaN or Inf. Safeguard here.
static bool isFinite(float x) {
  ABORT_IF(std::isfinite(0.f / 0.f), "NaN detection unreliable. Disable -Ofast compiler option.");
  return std::isfinite(x);
}

static void accNanOrNorm(float& lhs, float rhs) {
  if(isFinite(lhs) && isFinite(rhs)) {
    lhs = sqrtf(lhs * lhs + rhs * rhs); // to accumulate gradients norms, first undo sqrt, sum, re-apply sqrt.
  } else
    lhs = std::numeric_limits<float>::quiet_NaN();
}

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
  double typicalTrgBatchWords_{ 0 }; // for dynamic batch sizing: typical batch size in words

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

  float checkNanOrNorm(size_t i, size_t begin, size_t end) {
    auto curGrad = graphs_[i]->params()->grads()->subtensor(begin, end-begin);
    
    if(checkGradientNan_ || costScale_) {
      bool hasNan = false, hasInf = false;
      IsNaN(curGrad, graphs_[i]->allocator(), hasNan, hasInf); // @TODO: make safe with different compiler options
      if(hasNan || hasInf) {
        LOG(debug, "Found Nan ({}) or Inf ({})", hasNan, hasInf);
        return std::numeric_limits<float>::quiet_NaN();
      }
    }
    
    if(checkGradientNorm_) {
      auto gNorm = L2Norm(curGrad, graphs_[i]->allocator());
      if(isFinite(gNorm) && gNorm > 0.0)
        return gNorm;
      else 
        return std::numeric_limits<float>::quiet_NaN();
    }

    return 0.f;
  };

  float normalize(float gNorm, size_t updateTrgWords) {
    float normalizer = 1.f;

    if(costScale_)
      normalizer *= costScaleFactor_;

    if(options_->get<bool>("normalize-gradient"))
      normalizer *= updateTrgWords;

    if(!isFinite(gNorm))
      return normalizer;
    
    if(checkGradientNorm_) {
      // make invariant to changes in costScaleFactor_, luckily norm(c * g) = c * norm(g)
      if(costScale_)
        gNorm = gNorm / costScaleFactor_;
      
      // Normalize gradient norm w.r.t. number of labels in batch for statistics, 
      // there should be no gradient normalization before this point, @TODO: check this
      gNorm = gNorm / updateTrgWords;
    
      static size_t t = 0; // @TODO: replace by scheduler batches seen
      float alpha = 2.f / (checkGradientNormWindow_ + 1);
      float factor = checkGradientNormFactor_;
      
      auto logGNorm = std::log(gNorm); 
      static auto logGNormAvg = logGNorm;
      static decltype(logGNorm) logGNormVar = 0.0;
      
      auto delta = logGNorm - logGNormAvg;
      auto logGNormStd = std::sqrt(logGNormVar);

      // delta of log gradient norm vs log gradient norm average is larger than N standard deviations
      // hence rescale gradient using norm
      if(t >= checkGradientNormWindow_ && delta > factor * logGNormStd) {
        LOG(debug, "{:.4f} - {:.4f} -> logGNorm delta {:.4f} > {:.4f} * std {:.4f}", gNorm, std::exp(logGNormAvg), delta, factor, logGNormStd);
        normalizer *= std::exp(delta); // @TODO: normalize to avg + 1 sigma instead of to avg (exp(delta - logGNormStd)?)
        delta = logGNormStd;
      }

      t++;
      logGNormAvg = logGNormAvg + alpha * delta;
      logGNormVar = (1.0 - alpha) * (logGNormVar + alpha * delta * delta);
    }
  
    return normalizer;
  };

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
  double getTypicalTrgBatchWords();
  void updateAverageTrgBatchWords(size_t trgBatchWords);
};

}  // namespace marian