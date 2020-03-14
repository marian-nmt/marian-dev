#include "optimizers.h"

#include "common/io.h"
#include "tensors/tensor_operators.h"
#include <array>

namespace marian {

float OptimizerBase::update(Tensor params, Tensor grads, size_t mbSize, float costScaleFactor) {
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
  if(numAllocateShards > 0 && !baseAlloc_) {
    LOG_ONCE(info, "Allocating memory for general optimizer shards");
    baseAlloc_ = New<TensorAllocator>(params->getBackend());
    //baseAlloc_->throwAtReallocation(true);
    baseAlloc_->reserveExact(std::vector<size_t>(numAllocateShards, elements * sizeOf(optimizerType_)));
  }

  if(mvAvg_ && !avg_) {
    // allocate exp smooth shard tensor
    baseAlloc_->allocate(avg_, {1, elements}, optimizerType_);
    // initialize from parameters, this will be overwritten by checkpoint data if a checkpoint is found or by the first update.
    // If we resume training with no checkpoint this initialization will survive and be the basis for further averaging, which is 
    // what we want in that slightly pathological circumstance. 
    CopyCast(avg_, params);
  }

  if(castOptimizerType_) {
    if(!pm_) {
      // create parameter master copy and temporary gradient shard
      baseAlloc_->allocate(pm_, {1, elements}, optimizerType_);
      baseAlloc_->allocate(gd_, {1, elements}, optimizerType_);

      // keep parameter master copy around and initialize once, converting types
      CopyCast(pm_, params);
    }
  } else {
    // no conversion, just assign at each update
    pm_ = params;
  }

  if(!alloc_) {
    size_t size = pm_->memory()->size();
    alloc_ = New<Allocator>(pm_->getBackend()->getDeviceId(), size, size);
  }

  if(castOptimizerType_)
#if 0
    CopyCastStochastic(gd_, grads, alloc_);
#else
    CopyCast(gd_, grads);
#endif
  else
    gd_ = grads;

  // reverse cost scaling when used
  if(costScaleFactor != 1.f)
    Element(functional::_1 = functional::_1 / costScaleFactor, gd_);

  // clip gradients when used
  if(!clipper_) {
#if 0
    float clipNorm = options_->get<float>("clip-norm", 0.f);
    if(clipNorm > 0)
       clipper_ = New<NormClipper>(clipNorm);
    else
#endif
    clipper_ = New<ReportNormClipper>(0); // don't clip, just report
    
    auto clipAlloc = New<Allocator>(pm_->getBackend()->getDeviceId(), pow(2, 16) * 4, 1024);
    clipper_->setAllocator(clipAlloc);
  }

  float gNorm = clipper_->clip(gd_); // clip and rescale, report norm from before clipping

  // perform update on master copy with cast gradients
  // if a type cast has been performed. Otherwise the
  // original tensors are used.
  updateImpl(pm_, gd_, mbSize, refMBWords);

  // if exponential smoothing is used update the average
  if(mvAvg_)
    updateAvgParams(avg_, pm_, batchesSeen_, mbSize);

  // undo paramter type cast if required
  if(castOptimizerType_)
#if 0
    CopyCastStochastic(params, pm_, alloc_);
#else
    CopyCast(params, pm_);
#endif

  params->getBackend()->synchronize();

  return gNorm;
}

void OptimizerBase::swapWithSmoothed(Ptr<ExpressionGraph> graph, size_t i, size_t n, bool swapAvg) {
  if(!mvAvg_) // no smoothing, don't do anything
    return;

  // since we are here that means we are smoothing parameters, so let's get to work

  // Get the shard size. This needs to be divisible by n, right?
  size_t size = std::ceil(graph->params()->vals()->size() / (float)n);

  ABORT_IF(size != avg_->size(), "Graph shard size has to match smoothed parameter size ({} != {})", size, avg_->size());

  // Get the offset
  size_t offset = i * size;

  // Get the parameter subtensor of the graph that is being updated by this shard.
  // should this be handed in from the outside?
  auto subtensor = graph->params()->vals()->subtensor(offset, size);

  if(castOptimizerType_) {
    // If true then optimizer type is different from the graph type,
    // hence a parameter master copy exists and we do not need to swap.
    // We can just overwrite the current parameters with the smoothed
    // version avg_ and then restore from the master copy pm_.
    // We also cast from optimizer parameter type to graph parameter type
    CopyCast(subtensor, swapAvg ? avg_ : pm_);
  } else {
    // Types are equal hence there is no parameter master copy. This means
    // we need to do a proper swap between the graph params and the smoothed
    // version. We will then swap again with the next call restoring original
    // parameters. This assumes that two swaps are going to happen eventually.
    subtensor->swap(avg_);
  }
}

void OptimizerBase::load(std::vector<io::Item>& items,
                    const std::vector<Ptr<OptimizerBase>>& opts,
                    const std::vector<Ptr<Backend>>& backends,
                    const ScatterStateFunc& scatterFn) {
  ABORT_IF(opts.size() != backends.size(), "opts and backends of different sizes??");

  size_t numShards = 0;
  if(mvAvg_) numShards += 1;
  if(castOptimizerType_) numShards += 2;

  if(castOptimizerType_) {
    io::Item iParams;
    for(auto item : items)
      if(item.name == "master_parameters")
        iParams = std::move(item);

    if(iParams.bytes.empty()) {
      LOG(warn, "[warn] Parameters not found in .npz file");
    } else {
      ABORT_IF(optimizerType_ != iParams.type,
               "Current ({}) and previous ({}) optimization type do not match",
               optimizerType_,
               iParams.type);

      scatterFn(iParams,
        [&](size_t localDeviceIndex, const char* begin, const char* end) {
          auto opt = opts[localDeviceIndex];
          if(!opt->pm_) { // lazily allocate
            size_t size = end - begin;  // this is size in bytes now
            if(!opt->baseAlloc_) {
              LOG_ONCE(info, "Allocating memory for general optimizer shards");
              opt->baseAlloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
              opt->baseAlloc_->reserveExact(std::vector<size_t>(numShards, size));
            }
            int elements = (int)size / (int)sizeOf(iParams.type);
            opt->baseAlloc_->allocate(opt->pm_, {1, elements}, iParams.type);
            opt->baseAlloc_->allocate(opt->gd_, {1, elements}, iParams.type);
          }
          opt->pm_->set(begin, end, iParams.type); // set the value
        });
    }
  }

  if(mvAvg_) {
    io::Item iAvg;
    for(auto item : items)
      if(item.name == "exp_smoothing")
        iAvg = std::move(item);

    if(iAvg.bytes.empty()) {
      LOG(warn, "[warn] Average not found in .npz file");
    } else {
      ABORT_IF(optimizerType_ != iAvg.type,
          "Current ({}) and previous ({}) optimization type do not match",
          optimizerType_,
          iAvg.type);

      scatterFn(iAvg,
        [&](size_t localDeviceIndex, const char* begin, const char* end) {
          auto opt = opts[localDeviceIndex];
          if(!opt->avg_) { // lazily allocate
            size_t size = end - begin;  // this is size in bytes now
            if(!opt->baseAlloc_) {
              LOG_ONCE(info, "Allocating memory for general optimizer shards");
              opt->baseAlloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
              opt->baseAlloc_->reserveExact(numShards * size);
            }
            int elements = (int)size / (int)sizeOf(iAvg.type);
            opt->baseAlloc_->allocate(opt->avg_, {1, elements}, iAvg.type);
          }
          opt->avg_->set(begin, end, iAvg.type); // set the value
        });
    }
  }
}

void OptimizerBase::save(std::vector<io::Item>& items,
                         const std::vector<Ptr<OptimizerBase>>& opts,
                         const GatherStateFunc& gatherFn) {
  if(castOptimizerType_) {
    // fetch and concatenate state vectors for high precision copy
    io::Item pm = gatherFn([&](size_t localDeviceIndex) {
        auto opt = opts[localDeviceIndex];
        io::Item item;
        opt->pm_->get(item, "master_parameters");
        return item;
      });
    items.emplace_back(std::move(pm));
  }
  if(mvAvg_) {
    // fetch and concatenate state vectors for smoothed parameters
    io::Item avg = gatherFn([&](size_t localDeviceIndex) {
        auto opt = opts[localDeviceIndex];
        io::Item item;
        opt->avg_->get(item, "exp_smoothing");
        return item;
      });
    items.emplace_back(std::move(avg));
  }
}

void Sgd::updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) {
  actualMBSize, refMBWords; // (no correction for base update needed beyond using ce-sum)
  using namespace functional;
  Element(_1 -= eta_ * _2, params, grads);
}

void Sgd::load(std::vector<io::Item>& items,
               const std::vector<Ptr<OptimizerBase>>& opts,
               const std::vector<Ptr<Backend>>& backends,
               const ScatterStateFunc& scatterFn) {
  OptimizerBase::load(items, opts, backends, scatterFn);
}

void Sgd::save(std::vector<io::Item>& items,
               const std::vector<Ptr<OptimizerBase>>& opts,
               const GatherStateFunc& gatherFn) {
  OptimizerBase::save(items, opts, gatherFn); // collect parameters from base
}


// Adagrad update rule
void Adagrad::updateImpl(Tensor params, Tensor grads, size_t actualMBSize, size_t refMBWords) {
  ABORT_IF(actualMBSize != refMBWords, "Adagrad does not support rational hyper-parameter adjustment");

  // allocate optimizer-specific parameters
  if(!alloc_) {
    LOG_ONCE(info, "Allocating memory for Adagrad-specific shards");
    alloc_ = New<TensorAllocator>(params->getBackend());
    //alloc_->throwAtReallocation(true);
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

void Adagrad::load(std::vector<io::Item>& items,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const std::vector<Ptr<Backend>>& backends,
                   const ScatterStateFunc& scatterFn) {
  ABORT_IF(opts.size() != backends.size(), "opts and backends of different sizes??");


  OptimizerBase::load(items, opts, backends, scatterFn);

  LOG(info, "Loading Adagrad parameters");

  io::Item iGt;
  for(auto item : items)
    // extract data into vectors
    if(item.name == "adagrad_gt")
      iGt = std::move(item);

  if(iGt.bytes.empty()) {
    LOG(warn, "[warn] Adagrad parameters not found in checkpoint");
    return;
  }

  ABORT_IF(optimizerType_ != iGt.type,
          "Current ({}) and previous ({}) optimization type do not match",
          optimizerType_,
          iGt.type);

  scatterFn(iGt,
    [&](size_t localDeviceIndex, const char* begin, const char* end) {
      auto opt = std::dynamic_pointer_cast<Adagrad>(opts[localDeviceIndex]);
      if(!opt->gt_) {
        if(!opt->alloc_)
          opt->alloc_ = New<TensorAllocator>(backends[localDeviceIndex]);

        size_t size = end - begin; // this is size in bytes now
        int elements = (int)size / (int)sizeOf(iGt.type);
        opt->alloc_->reserveExact({size});
        opt->alloc_->allocate(opt->gt_, {1, elements}, iGt.type);
      }

      opt->gt_->set(begin, end, iGt.type);
    });
}

void Adagrad::save(std::vector<io::Item>& items,
                   const std::vector<Ptr<OptimizerBase>>& opts,
                   const GatherStateFunc& gatherFn) {

  OptimizerBase::save(items, opts, gatherFn); // collect parameters from base

  LOG(info, "Saving Adagrad parameters");
  // fetch and concatenate state vectors from distributed shards into a CPU-side vector
  io::Item gt = gatherFn([&](size_t localDeviceIndex) {
      auto opt = std::dynamic_pointer_cast<Adagrad>(opts[localDeviceIndex]);
      io::Item item;
      opt->gt_->get(item, "adagrad_gt");
      return item;
    });
  items.emplace_back(std::move(gt));
}

void Adagrad::resetStats() {
  if(gt_)
    gt_->set(0.f);
}

// Adam
void Adam::updateImpl(Tensor params, Tensor grads, size_t /*actualMBSize*/, size_t /*refMBWords*/) {
  // lazy allocation
  if(!alloc_) {
    LOG_ONCE(info, "Allocating memory for Adam-specific shards");
    alloc_ = New<TensorAllocator>(params->getBackend());
    //alloc_->throwAtReallocation(true);
  }

  if(!mt_) {
    int elements = (int)params->size();
    size_t shard = (size_t)elements * sizeOf(params->type());
    alloc_->reserveExact({shard, shard});

    alloc_->allocate(mt_, {1, elements}, params->type());
    mt_->set(0.f);

    alloc_->allocate(vt_, {1, elements}, params->type());
    vt_->set(0.f);
  }

  double T    = 1; //(double)actualMBSize;
  double Tref = 1; //(double)refMBWords;

  // adjust for minibatch-size changes if Adam parameters are given a reference size (else do nothing)
  double eta   = eta_ * (T/Tref);
  double beta1 = beta1_;
  double beta2 = beta2_;
  double decay = w_    ;

  // denominators. At steady state: =1. This recursion does the same as the Adam beta correction term.
  denom1_ = (beta1 * denom1_) + (1 - beta1); // momentum smoothing
  denom2_ = (beta2 * denom2_) + (1 - beta2); // RMS normalization

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

void Adam::load(std::vector<io::Item>& items,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const std::vector<Ptr<Backend>>& backends,
                const ScatterStateFunc& scatterFn) {
  ABORT_IF(opts.size() != backends.size(), "opts and backends of different sizes??");

  OptimizerBase::load(items, opts, backends, scatterFn);

  LOG(info, "Loading Adam parameters");

  io::Item iMt;
  io::Item iVt;
  std::array<double, 2> vDenoms;

  for(auto item : items) {
    // extract data into vectors
    if(item.name == "adam_mt") {
      iMt = std::move(item);
    } else if(item.name == "adam_vt") {
      iVt = std::move(item);
    } else if(item.name == "adam_denoms") {
      ABORT_IF(item.size() != 2 * sizeof(double), "adam_denoms should have 2 entries");
      std::copy((double*)item.data(), ((double*)item.data()) + 2, vDenoms.begin());
      // Back compat note: Old files lacked "adam_denoms". For those, vDenoms will remain 0, which reproduces the old behavior.
    }
  }

  if(iMt.bytes.empty() || iVt.bytes.empty()) {
    LOG(warn, "[warn] Adam parameters not found in .npz file");
    return;
  }

  ABORT_IF(optimizerType_ != iMt.type,
          "Current ({}) and previous ({}) optimization type do not match",
          optimizerType_,
          iMt.type);

  ABORT_IF(iMt.size() != iVt.size(), "mt and vt have different sizes??");

  //LOG(info, "loading Adam params");
  scatterFn(iMt,
    [&](size_t localDeviceIndex, const char* begin, const char* end) {
      auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
      if(!opt->mt_ || !opt->vt_) { // lazily allocate
        if(!opt->alloc_)
          opt->alloc_ = New<TensorAllocator>(backends[localDeviceIndex]);
        size_t size = end - begin;  // this is size in bytes now
        int elements = (int)size / (int)sizeOf(iMt.type);
        opt->alloc_->reserveExact(2 * size);
        opt->alloc_->allocate(opt->mt_, {1, elements}, iMt.type);
        opt->alloc_->allocate(opt->vt_, {1, elements}, iMt.type);
      }
      opt->mt_->set(begin, end, iMt.type); // set the value
    });

  scatterFn(iVt,
    [&](size_t id, const char* begin, const char* end) {
      auto opt = std::dynamic_pointer_cast<Adam>(opts[id]);
      opt->vt_->set(begin, end, iVt.type);
    });

  denom1_ = vDenoms[0];
  denom2_ = vDenoms[1];
  //LOG(info, "done loading Adam params");
}

void Adam::save(std::vector<io::Item>& items,
                const std::vector<Ptr<OptimizerBase>>& opts,
                const GatherStateFunc& gatherFn) {


  OptimizerBase::save(items, opts, gatherFn); // collect parameters from base

  LOG(info, "Saving Adam parameters");

  // fetch and concatenate state vectors from distributed shards into a CPU-side vector
  io::Item mt = gatherFn([&](size_t localDeviceIndex) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
    io::Item item;
    opt->mt_->get(item, "adam_mt");
    return item;
  });
  items.emplace_back(std::move(mt));

  io::Item vt = gatherFn([&](size_t localDeviceIndex) {
    auto opt = std::dynamic_pointer_cast<Adam>(opts[localDeviceIndex]);
    io::Item item;
    opt->vt_->get(item, "adam_vt");
    return item;
  });
  items.emplace_back(std::move(vt));

  std::vector<double> vDenoms{denom1_, denom2_};
  items.emplace_back(std::move(io::fromVector(vDenoms, "adam_denoms")));
}

void Adam::resetStats() {
  if(mt_)
    mt_->set(0.f);

  if(vt_)
    vt_->set(0.f);

  denom1_ = 0; // @BUGBUG: or 1 or refMBWords if so specified. Fix once we have proper parameterization for that.
  denom2_ = 0;
}

Ptr<OptimizerBase> Optimizer(Ptr<Options> options) {
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
  return opt;
}
}  // namespace marian
