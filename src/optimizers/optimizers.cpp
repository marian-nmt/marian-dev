#include "optimizers.h"

#include "common/io.h"
#include "tensors/tensor_operators.h"
#include <array>

namespace marian {

void OptimizerBase::update(Tensor params, Tensor grads, size_t mbSize) {
  size_t refMBWords = refMBWordsParam_;
  if (refMBWords == 0) { // optimizer not configured to use hyper-parameter auto-adjustment
    refMBWords = mbSize = 1; // neutral settings that keep the standard behavior
  } else { // optimizer is configured to auto-adjust hyper-parameters
    ABORT_IF(mbSize == mbSizeNotProvided, "Using rational optimizer auto-adjustment with trainer that does not provide MB size");
    // note: this behavior is only meaningful if using the ce-sum criterion
  }

  // if true model for forward/backward uses a different type than the optimizer
  castOptimizerType_ = params->type() != optimizerType_;
  int elements = (int)params->size();

  LOG_ONCE(info, "Parameter type {}, optimization type {}",
           params->type(), optimizerType_);

  int numAllocateShards = 0;
  if(mvAvg_) numAllocateShards += 1; // one shard for exp smoothing
  if(castOptimizerType_) numAllocateShards += 2; // two shards for conversion

  // allocate storage for shards
  if(numAllocateShards > 0 && !optAlloc_) {
    LOG_ONCE(info, "Allocating memory for general optimizer shards");
    optAlloc_ = New<TensorAllocator>(params->getBackend());
    optAlloc_->reserveExact(numAllocateShards * elements * sizeOf(optimizerType_));
  }


  if(mvAvg_ && !avg_)
    // allocate exp smooth shard tensor
    optAlloc_->allocate(avg_, {1, elements}, optimizerType_);

  if(castOptimizerType_) {
    if(!pm_) {
      // create parameter master copy and temporary gradient shard
      optAlloc_->allocate(pm_, {1, elements}, optimizerType_);
      optAlloc_->allocate(gd_, {1, elements}, optimizerType_);

      // keep parameter master copy around and initialize once, converting types
      CopyCast(pm_, params);
    }
  } else {
    // no conversion, just assign at each update
    pm_ = params;
  }

  bool hasNanOrInf = false;
  if(costScale_) {
    bool hasNan = false, hasInf = false;
    // perform NaN/Inf check on original gradients
    IsNan(grads, allocator_, hasNan, hasInf); // what about padded space? Make sure it's set to 0!
    hasNanOrInf = hasNan || hasInf;
  }

  if(!hasNanOrInf) {
    noNanSeen_++;

    using namespace functional;
    if(castOptimizerType_)
      CopyCast(gd_, grads);
    else
      gd_ = grads;

    // reverse cost scaling when used
    if(costScaleFactor_ != 1.f)
      Element(_1 = _1 / costScaleFactor_, gd_);

    // clip gradients when used
    if(clipper_)
      clipper_->clip(gd_);

    // perform update on master copy with cast gradients
    // if a type cast has been performed. Otherwise the
    // original tensors are used.
    updateImpl(pm_, gd_, mbSize, refMBWords);

    if(mvAvg_)
      updateAvgParams(avg_, pm_, batchesSeen_, mbSize);

    // undo paramter type cast if required
    if(castOptimizerType_)
      CopyCast(params, pm_);

    if(costScale_ && noNanSeen_ > 0 && noNanSeen_ % costScaleFreq_ == 0) {
      costScaleFactor_ *= costScaleMultiplier_;
      LOG(info, "No NaN/Inf seen for {} updates. Increasing cost-scaling factor to {}", noNanSeen_, costScaleFactor_);
    }
  } else if(costScale_) {
    costScaleFactor_ /= costScaleMultiplier_;
    LOG(warn, "Seen NaN/Inf in gradient, skipping update, reducing cost-scaling factor to {}", costScaleFactor_);
    noNanSeen_ = 0;
  } else {
    // actually we should not be NaN checking without scaling, abort.
    ABORT("Seen NaN/Inf, but not cost-scaling. Don't know what to do.");
    noNanSeen_ = 0;
  }

  params->getBackend()->synchronize();
}

void OptimizerBase::save(std::vector<io::Item>& items,
                         const std::vector<Ptr<OptimizerBase>>& opts,
                         const GatherStateFunc& gatherFn) {
  if(castOptimizerType_) {
    // fetch and concatenate state vectors for high precision copy
    io::Item pm = gatherFn([&](size_t localDeviceIndex) {
        auto opt = opts[localDeviceIndex];
        return opt->pm_->toItem("master_parameters");
      });
    items.emplace_back(std::move(pm));
  }
  if(mvAvg_) {
    // fetch and concatenate state vectors for smoothed parameters
    io::Item avg = gatherFn([&](size_t localDeviceIndex) {
        auto opt = opts[localDeviceIndex];
        return opt->avg_->toItem("exp_smoothing");
      });
    items.emplace_back(std::move(avg));
  }
  std::vector<float> vCostScale{costScaleFactor_};
  items.emplace_back(std::move(io::fromVector(vCostScale, "cost_scale")));
}

void Sgd::updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) {
  actualMBSize, refMBWords; // (no correction for base update needed beyond using ce-sum)
  using namespace functional;
  Element(_1 -= eta_ * _2, params, grads);
}

// Adagrad update rule
void Adagrad::updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) {
  ABORT_IF(actualMBSize != refMBWords, "Adagrad does not support rational hyper-parameter adjustment");

  // allocate optimizer-specific parameters
  if(!alloc_) {
    LOG_ONCE(info, "Allocating memory for Adagrad-specific shards");
    alloc_ = New<TensorAllocator>(params->getBackend());
  }

  if(!gt_) {
    int elements = (int)params->size();
    alloc_->reserveExact(params->memory()->size());
    alloc_->allocate(gt_, {1, elements}, params->type());
    gt_->set(0.f);
  }

  using namespace functional;
  Element(_1 += (_2 * _2), gt_, grads);

  // make sure eps_ does not drop below minimum value, add some reserve by multiplying with 2
  eps_ = std::max(NumericLimits<double>(params->type()).min * 2.f, (double)eps_);
  Element(_1 -= (eta_ / (sqrt(_2) + eps_)) * _3, params, gt_, grads);
}

void Adagrad::load(const std::string& name,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const std::vector<Ptr<Backend>>& backends,
                   const ScatterStateFunc& scatterFn) {
  ABORT_IF(opts.size() != backends.size(), "opts and backends of different sizes??");

  if(!filesystem::exists(name))
    return;

  LOG(info, "Loading Adagrad parameters from {}", name);

  std::vector<float> vGt;
  auto items = io::loadItems(name);
  for(auto item : items) {
    // extract data into vectors
    if(item.name == "adagrad_gt") {
      // get the size of gt_
      auto totalSize = item.shape.elements();
      vGt.resize(totalSize);
      std::copy((float*)item.data(), ((float*)item.data()) + totalSize, vGt.begin());
    }
  }

  if(vGt.empty()) {
    LOG(warn, "[warn] Adagrad parameters not found in .npz file");
    return;
  }

  scatterFn(vGt,
    [&](size_t localDeviceIndex, std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) {
    auto opt = std::dynamic_pointer_cast<Adagrad>(opts[localDeviceIndex]);
    if(!opt->gt_) {
      if(!opt->alloc_)
        opt->alloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
      auto size = end-begin;
      opt->alloc_->reserveExact(sizeof(float) * size);
      opt->alloc_->allocate(opt->gt_, {1, (int)size});
    }
    opt->gt_->set(std::vector<float>(begin, end));
  });
}

void Adagrad::save(std::vector<io::Item>& items,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const GatherStateFunc& gatherFn) {
  // fetch and concatenate state vectors from distributed shards into a CPU-side vector
  io::Item gt = gatherFn([&](size_t localDeviceIndex) {
      auto opt = std::dynamic_pointer_cast<Adagrad>(opts[localDeviceIndex]);
      return opt->gt_->toItem("adagrad_gt");
    });
  items.emplace_back(std::move(gt));
}

void Adagrad::save(const std::string& name,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const GatherStateFunc& gatherFn,
                   bool isMainProcess /*= true*/) {
  // if not main MPI process then we have done our duty
  if (!isMainProcess)
    return;

  LOG(info, "Saving Adagrad parameters to {}", name);

  std::vector<io::Item> items;
  OptimizerBase::save(items, opts, gatherFn); // collect parameters from base
  save(items, opts, gatherFn); // collect parameters for this optimizer class
  io::saveItems(name, items); // save all to file
}

void Adagrad::resetStats() {
  if(gt_)
    gt_->set(0.f);
}

// Adam
void Adam::updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) {
  // lazy allocation
  if(!alloc_) {
    LOG_ONCE(info, "Allocating memory for Adam-specific shards");
    alloc_ = New<TensorAllocator>(params->getBackend());
  }

  if(!mt_) {
    int elements = (int)params->size();
    alloc_->reserveExact(2 * elements * sizeOf(params->type()));
    alloc_->allocate(mt_, {1, elements}, params->type());
    mt_->set(0.f);

    alloc_->allocate(vt_, {1, elements}, params->type());
    vt_->set(0.f);
  }

  double T    = (double)actualMBSize;
  double Tref = (double)refMBWords;

  // adjust for minibatch-size changes if Adam parameters are given a reference size (else do nothing)
  double eta   = eta_ * (T/Tref);
  double beta1 = beta1_;
  double beta2 = beta2_;
  double decay = w_    ;

  // denominators. At steady state: =1. This recursion does the same as the Adam beta correction term.
  denom1_ = (beta1 * denom1_) + (1 - beta1); // momentum smoothing
  denom2_ = (beta2 * denom2_) + (1 - beta2); // RMS normalization

  //LOG_ONCE(info, "[adam] First update: Tref = {}, T = {}, eta = {} -> {}, beta = {}, {}", Tref, T, eta_, eta, beta1, beta2);

  // numerators. Divide by T to convert ce-sum gradient to avg gradient.
  using namespace functional;
  Element(_1 = ((float)beta1 * _1) + float((1 - beta1) / T    ) *  _2,       mt_, grads); // momentum smoothing. At steady state: =smoothed avg gradient
  Element(_1 = ((float)beta2 * _1) + float((1 - beta2) / T / T) * (_2 * _2), vt_, grads); // RMS normalization.  At steady state: =mean square of the avg gradients

  // make sure eps_ does not drop below minimum value, this is important
  // when training with mixed precision. Otherwise we divide by 0.
  // We multiply the minimum by 2 in order to step away from the abyss.
  eps_ = std::max(NumericLimits<float>(params->type()).min * 2.f, eps_);

  // apply Adam normalization
  float etaf = (float)eta, denom1f = (float)denom1_, denom2f = (float)denom2_, decayf = (float)decay; // (get casts out of Element expression for readability)
  Element(_1 -= etaf                               // learning-rate: x_t = x_{t-1} - \eta * (...)
                * ((  (     _2 / denom1f)          // momentum-smoothed per-sample gradient: m_{t-1}
                    / (sqrt(_3 / denom2f) + eps_)) // normalize by RMS: \sqrt(v_{t-1})
                   + decayf * _1),                 // weight-decay: w * x_{t-1}
          params,  // =_1
          mt_,     // =_2
          vt_      // =_3
          );
}

void Adam::load(const std::string& name,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const std::vector<Ptr<Backend>>& backends,
                const ScatterStateFunc& scatterFn) {
  ABORT_IF(opts.size() != backends.size(), "opts and backends of different sizes??");

  if(!filesystem::exists(name))
    return;

  LOG(info, "Loading Adam parameters from {}", name);

  std::vector<float> vMt;
  std::vector<float> vVt;
  std::array<double, 2> vDenoms;

  auto items = io::loadItems(name);
  for(auto item : items) {
    // get the size of mt_ and vt_, they are the same
    auto totalSize = item.shape.elements();

    // extract data into vectors
    if(item.name == "adam_mt") {
      vMt.resize(totalSize);
      std::copy((float*)item.data(), ((float*)item.data()) + totalSize, vMt.begin());
    }
    else if(item.name == "adam_vt") {
      vVt.resize(totalSize);
      std::copy((float*)item.data(), ((float*)item.data()) + totalSize, vVt.begin());
    }
    else if(item.name == "adam_denoms") {
      ABORT_IF(totalSize != 2, "adam_denoms should have 2 entries");
      std::copy((double*)item.data(), ((double*)item.data()) + totalSize, vDenoms.begin());
      // Back compat note: Old files lacked "adam_denoms". For those, vDenoms will remain 0, which reproduces the old behavior.
    }
  }
  if(vMt.empty() || vVt.empty()) {
    LOG(warn, "[warn] Adam parameters not found in .npz file");
    return;
  }
  ABORT_IF(vMt.size() != vVt.size(), "mt and vt have different sizes??");

  //LOG(info, "loading Adam params");
  scatterFn(vMt,
    [&](size_t localDeviceIndex, std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
    if(!opt->mt_ || !opt->vt_) { // lazily allocate
      if(!opt->alloc_)
        opt->alloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
      auto size = end-begin;
      opt->alloc_->reserveExact(2 * sizeof(float) * size);
      opt->alloc_->allocate(opt->mt_, {1, (int)size});
      opt->alloc_->allocate(opt->vt_, {1, (int)size});
    }
    opt->mt_->set(std::vector<float>(begin, end)); // set the value
  });

  scatterFn(vVt,
    [&](size_t id, std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[id]);
    opt->vt_->set(std::vector<float>(begin, end));
  });

  denom1_ = vDenoms[0];
  denom2_ = vDenoms[1];
  //LOG(info, "done loading Adam params");
}

void Adam::save(std::vector<io::Item>& items,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const GatherStateFunc& gatherFn) {
  // @TODO: switch to bytes

  // fetch and concatenate state vectors from distributed shards into a CPU-side vector
  io::Item mt = gatherFn([&](size_t localDeviceIndex) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
    return opt->mt_->toItem("adam_mt");
  });
  items.emplace_back(std::move(mt));

  io::Item vt = gatherFn([&](size_t localDeviceIndex) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
    return opt->vt_->toItem("adam_vt");
  });
  items.emplace_back(std::move(vt));

  std::vector<double> vDenoms{denom1_, denom2_};
  items.emplace_back(std::move(io::fromVector(vDenoms, "adam_denoms")));
}

void Adam::save(const std::string& name,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const GatherStateFunc& gatherFn,
                bool isMainProcess /*= true*/) {
  // if not main MPI process then we have done our duty
  if (!isMainProcess)
    return;

  LOG(info, "Saving Adam parameters to {}", name);

  std::vector<io::Item> items;
  OptimizerBase::save(items, opts, gatherFn); // collect parameters from base
  save(items, opts, gatherFn); // collect parameters for this optimizer class
  io::saveItems(name, items); // save all to file
}

void Adam::resetStats() {
  if(mt_)
    mt_->set(0.f);

  if(vt_)
    vt_->set(0.f);

  denom1_ = 0; // @BUGBUG: or 1 or refMBWords if so specified. Fix once we have proper parameterization for that.
  denom2_ = 0;
}

Ptr<OptimizerBase> Optimizer(Ptr<Options> options, Ptr<Allocator> allocator) {
  auto optType = options->get<std::string>("optimizer");
  auto params = options->has("optimizer-params")
                     ? options->get<std::vector<float>>("optimizer-params")
                     : std::vector<float>({});
  Ptr<OptimizerBase> opt;
  if(optType == "sgd") {
    opt = New<Sgd>(options);
  } else if(optType == "adagrad") {
    opt = New<Adagrad>(options);
  } else if(optType == "adam") {
    opt = New<Adam>(options);
  } else {
    ABORT("Unknown optimizer type: {}", opt);
  }

  opt->setParams(params);
  opt->setAllocator(allocator);
  return opt;
}
}  // namespace marian
