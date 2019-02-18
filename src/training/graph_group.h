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
  float nanTolerance_{0.f};
  size_t noNanSeen_{0}; // @TODO, add current noNanSeen_ to trainingState for serialization
  size_t nanSeen_{0};

public:
  GraphGroup(Ptr<Options> options) : options_(options) {
    if(options_->hasAndNotEmpty("cost-scaling")) {
      auto vcs = options_->get<std::vector<std::string>>("cost-scaling");
      costScale_ = true;
      float costExponent = std::stof(vcs[0]);
      costScaleFactor_ = std::pow(2.0f, costExponent);
      costScaleFreq_ = std::stoul(vcs[1]);
      costScaleMultiplier_ = std::stof(vcs[2]);
      nanTolerance_ = std::stof(vcs[3]);

      LOG_ONCE(info,
               "Training with cost scaling - factor: 2^{} = {}, frequency: {}, multiplier: {}, tolerance: {}",
               costExponent,
               costScaleFactor_,
               costScaleFreq_,
               costScaleMultiplier_,
               nanTolerance_);
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

    float nanPercent = noNanSeen_ == 0 ? 1.f : (float)nanSeen_ / (float)noNanSeen_;

    if(noNanSeen_ % costScaleFreq_ == 0) {
      costScaleFactor_ *= costScaleMultiplier_;
      LOG(info,
          "NaN/Inf percentage {:.2f} after {} updates. Increasing cost-scaling factor to {}",
          nanPercent,
          noNanSeen_,
          costScaleFactor_);
    }
  }

  // call when a NaN was seen to decrease cost-scaling factor
  void decreaseCostScaleFactor() {
    if(!costScale_)
      return;

    nanSeen_++;
    float nanPercent = noNanSeen_ == 0 ? 1.f : (float)nanSeen_ / (float)noNanSeen_;
    if(nanPercent > nanTolerance_) {
      costScaleFactor_ /= costScaleMultiplier_;
      LOG(warn,
          "NaN/Inf percentage {:.2f} in gradients, skipping update, reducing cost-scaling factor to {}",
          nanPercent,
          costScaleFactor_);

      noNanSeen_ = 0;
      nanSeen_ = 0;
    }
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
                                             const std::vector<Ptr<Vocab>>& vocabs,
                                             double multiplier = 1.) {
    // this runs with fake values, we do not care for overflow/underflow
    bool throwNan = graph->getThrowNan();
    graph->setThrowNan(false);

    auto stats = New<data::BatchStats>();

    size_t numFiles = options_->get<std::vector<std::string>>("train-sets").size();

    // Initialize first batch to step size
    size_t first = options_->get<size_t>("mini-batch-fit-step");

    // Increase batch size and sentence length by this step size
    size_t step = options_->get<size_t>("mini-batch-fit-step");

    size_t maxLength = options_->get<size_t>("max-length");
    maxLength = (size_t)(std::ceil(maxLength / (float)step) * step);

    // this should be only one class label per line on input, hence restricting length to 1
    std::vector<size_t> localMaxes(numFiles, maxLength);
    auto inputTypes = options_->get<std::vector<std::string>>("input-types", {});
    for(int i = 0; i < inputTypes.size(); ++i)
      if(inputTypes[i] == "class")
        localMaxes[i] = 1;
 
    size_t maxBatch = 512;
    bool fits = true;
    while(fits) {
      std::vector<size_t> lengths(numFiles, first);

      for(int j = 0; j < lengths.size(); ++j) // apply length restrictions
        lengths[j] = std::min(lengths[j], localMaxes[j]);

      auto batch = data::CorpusBatch::fakeBatch(lengths, vocabs, maxBatch, options_);
      auto loss = model->build(graph, batch);
      fits = graph->fits();
      if(fits)
        maxBatch *= 2;
    }

    // Do a binary search for maxmimum batch size that fits into given workspace memory 
    // for a tested sentence length. 
    for(size_t i = step; i <= maxLength; i += step) {
      size_t start = 1;
      size_t end = maxBatch;

      std::vector<size_t> lengths(numFiles, i);
      for(int j = 0; j < lengths.size(); ++j)  // apply length restrictions
        lengths[j] = std::min(lengths[j], localMaxes[j]);
      fits = true;

      do {
        size_t current = (start + end) / 2;
        auto batch = data::CorpusBatch::fakeBatch(lengths, vocabs, current, options_);
        auto loss = model->build(graph, batch);
        fits = graph->fits();

        LOG(debug, "[batching] length: {} - size: {} - fits: {}", lengths[0], current, fits);

        if(fits) {
          stats->add(batch, multiplier);
          start = current + 1;
        } else {
          end = current - 1;
        }
      } while(end > start);

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

static void swapWithSmoothed(const std::vector<Ptr<ExpressionGraph>>& graphs, 
                             const std::vector<Ptr<OptimizerBase>>& opts, 
                             const std::function<void()> distribute = [](){}) {
  ABORT_IF(graphs.size() != opts.size(), "Number of graphs and optimizers has to be equal ({} != {})", graphs.size() != opts.size());
  for(size_t i = 0; i < graphs.size(); ++i)
    opts[i]->swapWithSmoothed(graphs[i], i, graphs.size(), /*swapAvg=*/true);
  distribute();
}

static void swapWithOriginal(const std::vector<Ptr<ExpressionGraph>>& graphs, 
                             const std::vector<Ptr<OptimizerBase>>& opts,
                             const std::function<void()> distribute = [](){}) {
  ABORT_IF(graphs.size() != opts.size(), "Number of graphs and optimizers has to be equal ({} != {})", graphs.size() != opts.size());
  for(size_t i = 0; i < graphs.size(); ++i)
    opts[i]->swapWithSmoothed(graphs[i], i, graphs.size(), /*swapAvg=*/false);
  distribute();
}

}  // namespace marian
