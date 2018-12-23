#pragma once

#include "common/options.h"
#include "graph/expression_graph.h"
#include "optimizers/clippers.h"
#include "optimizers/exponential_smoothing.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "training/training_state.h"

#include <algorithm>
#include <map>
#include <memory>

namespace marian {

/**
 * Base class for optimizers.
 */
class OptimizerBase
  : public TrainingObserver,
    public ExponentialSmoothing
{
public:
  OptimizerBase(Ptr<Options> options)
  : ExponentialSmoothing(options),
    options_(options),
    eta_(options_->get<float>("learn-rate")),
    refMBWordsParam_(options_->get<size_t>("mini-batch-words-ref")) {

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

    auto precisions = options_->get<std::vector<std::string>>("precision");
    optimizerType_ = typeFromString(precisions[1]);

    float clipNorm = options->get<float>("clip-norm");
    if(clipNorm > 0)
      clipper_ = Clipper<Norm>(clipNorm);

    // automatic learning-rate adjustment
    // If users provide, in addition to the hyper-parameters, a reference minibatch size,
    // that these hyper-parameters were originally tuned for, then the learning-rate gets
    // adjusted accordingly. Note: Requires user to also use ce-sum criterion.
    if (refMBWordsParam_ != 0)
      LOG(info, "Note: Learning rate gets automatically adjusted as if minibatch size was {}", refMBWordsParam_);
  }

  static constexpr size_t mbSizeNotProvided = SIZE_MAX;

  void update(Ptr<ExpressionGraph> graph, size_t mbSize = mbSizeNotProvided) {
    Tensor p = graph->params()->vals();
    Tensor g = graph->params()->grads();

    update(p, g, mbSize);
  }

  void update(Tensor params, Tensor grads, size_t mbSize = mbSizeNotProvided);

  virtual void init(TrainingState& state) override {
    eta_ = state.eta;
  }
  virtual void actAfterLoaded(TrainingState& state) override {
    eta_ = state.eta;
  }
  virtual void actAfterEpoch(TrainingState& state) override {
    eta_ = state.eta;
    if(state.reset)
      resetStats();
  }
  virtual void actAfterBatches(TrainingState& state) override {
    eta_ = state.eta;
    batchesSeen_ = state.batches;

    if(state.reset)
      resetStats();
  }
  virtual void actAfterStalled(TrainingState& state) override {
    eta_ = state.eta;
    if(state.reset)
      resetStats();
  }

  virtual void setParams(const std::vector<float>& params) = 0;

  virtual void setAllocator(Ptr<Allocator> allocator) { allocator_ = allocator; }

  typedef std::function<void(size_t /*localDeviceIndex*/,
                             std::vector<float>::const_iterator /*begin*/,
                             std::vector<float>::const_iterator /*end*/)> ScatterStateSetFunc;
  typedef std::function<io::Item(size_t /*localDeviceIndex*/)> GatherStateGetFunc;

  typedef std::function<void(const std::vector<float>& /*data*/, const ScatterStateSetFunc& /*setFn*/)> ScatterStateFunc;
  typedef std::function<io::Item(const GatherStateGetFunc& /*getFn*/)> GatherStateFunc;

  virtual void load(const std::string& /*name*/,
                    const std::vector<Ptr<OptimizerBase>>& /*opts*/,
                    const std::vector<Ptr<Backend>>& /*backends*/,
                    const ScatterStateFunc& /*scatterFn*/) {}
  virtual void save(const std::string& /*name*/,
                    const std::vector<Ptr<OptimizerBase>>& /*opts*/,
                    const GatherStateFunc& /*gatherFn*/,
                    bool /*isMainProcess*/ = true) {}

  float getCostScaleFactor() { return costScaleFactor_; }

protected:
  virtual void updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) = 0;
  virtual void resetStats() = 0;

  virtual void save(std::vector<io::Item>& items,
                    const std::vector<Ptr<OptimizerBase>>& opts,
                    const GatherStateFunc& gatherFn);

  Ptr<Options> options_;

  // Learning rate
  float eta_;
  // Reference MB size. This enables automatic adjustment of optimizer hyper-parameters to MB size.
  size_t refMBWordsParam_{0}; // 0 means no adjustment
  // Cost scaling factor
  bool costScale_{false};
  float costScaleFactor_{1.f};
  size_t costScaleFreq_{2000};
  float costScaleMultiplier_{2.f};
  // Seen updates so far
  size_t batchesSeen_{0};

  Type optimizerType_{Type::float32};
  bool castOptimizerType_{false};

    // Clip gradient norm
  Ptr<ClipperBase> clipper_;

  Ptr<Allocator> allocator_;
  Ptr<TensorAllocator> optAlloc_;

  Tensor avg_;

  Tensor pm_;
  Tensor gd_;
};

/**
 * @brief Stochastic gradient descent optimizer.
 */
class Sgd : public OptimizerBase {
public:
  Sgd(Ptr<Options> options) : OptimizerBase(options) {}

  virtual void setParams(const std::vector<float>& /*params*/) override {}
private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) override;

  virtual void resetStats() override {}
};

/**
 * @brief Adagrad optimizer
 *
 * http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
 */
class Adagrad : public OptimizerBase {
public:
  Adagrad(Ptr<Options> options) : OptimizerBase(options) {}

  void load(const std::string& name,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const std::vector<Ptr<Backend>>& backends,
            const ScatterStateFunc& scatterFn) override;
  void save(const std::string& name,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const GatherStateFunc& gatherFn,
            bool /*isMainProcess*/ = true) override;

  void setParams(const std::vector<float>& params) override {
    if(params.size() > 0)
      eps_ = params[0];
  }

private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) override;
  void resetStats() override;

  void save(std::vector<io::Item>& items,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const GatherStateFunc& gatherFn) override;

  float eps_ = 1e-8f;
  Ptr<TensorAllocator> alloc_;
  Tensor gt_;
};

/**
 * @brief Adam optimizer
 *
 * https://arxiv.org/pdf/1412.6980v8.pdf
 *
 * with Frank's modifications for automatic hyper-parameter adjustment.
 */
class Adam : public OptimizerBase {
public:
  Adam(Ptr<Options> options) : OptimizerBase(options) {}

  void load(const std::string& name,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const std::vector<Ptr<Backend>>& backends,
            const ScatterStateFunc& scatterFn) override;
  void save(const std::string& name,
            const std::vector<Ptr<OptimizerBase>>& opts,
            const GatherStateFunc& gatherFn,
            bool isMainProcess = true) override;

private:
  void updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) override;
  void resetStats() override;

  void save(std::vector<io::Item>& items,
          const std::vector<Ptr<OptimizerBase>>& opts,
          const GatherStateFunc& gatherFn) override;

  // Adam parameters:
  // [beta1, beta2, eps, w, refMBWords]
  virtual void setParams(const std::vector<float>& params) override {
    if(params.size() > 0)
      beta1_ = params[0];
    if(params.size() > 1)
      beta2_ = params[1];
    if(params.size() > 2)
      eps_ = params[2];

    // weighted decay for AdamW, to be explored, disabled by default
    if(params.size() > 3)
      w_ = params[3]; // default (disabled): 0
  }

  // hyper-parameters
  float beta1_ = 0.9f;
  float beta2_ = 0.999f;
  float eps_ = 1e-8f;
  float w_ = 0.0f;

  // CPU-side running accumulators
  double denom1_ = 0;
  double denom2_ = 0;

  // GPU-side running accumulators
  Ptr<TensorAllocator> alloc_;
  Tensor mt_;
  Tensor vt_;
};

Ptr<OptimizerBase> Optimizer(Ptr<Options> options, Ptr<Allocator> allocator = nullptr);
}  // namespace marian
