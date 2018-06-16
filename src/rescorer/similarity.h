#pragma once

#include "marian.h"

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/corpus_nbest.h"
#include "models/model_task.h"
#include "rescorer/score_collector.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

using namespace data;

class Similarity : public ModelTask {
private:
  Ptr<Config> options_;
  Ptr<CorpusBase> corpus_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<Ptr<models::ModelBase>> models_;

public:
  Similarity(Ptr<Config> options)
      : options_(options),
        corpus_(
            options_->get<bool>("n-best")
                ? std::static_pointer_cast<CorpusBase>(
                      New<CorpusNBest>(options_))
                : std::static_pointer_cast<CorpusBase>(New<Corpus>(options_))) {
    corpus_->prepare();

    auto devices = options_->getDevices();

    for(auto device : devices) {
      auto graph = New<ExpressionGraph>(true, options_->get<bool>("optimize"));
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);
    }

    auto modelFile = options_->get<std::string>("model");

    Ptr<Options> temp = New<Options>();
    temp->merge(options);
    temp->set("inference", true);
    temp->set("type", "sim-s2s");

    models_.resize(graphs_.size());
    ThreadPool pool(graphs_.size(), graphs_.size());
    for(int i = 0; i < graphs_.size(); ++i) {
      pool.enqueue(
          [=](int j) {
            models_[j] = models::from_options(temp, models::usage::raw);
            models_[j]->load(graphs_[j], modelFile);
          },
          i);
    }
  }

  void run() {
    LOG(info, "Scoring");

    auto batchGenerator = New<BatchGenerator<CorpusBase>>(corpus_, options_);
    batchGenerator->prepare(false);

    Ptr<ScoreCollector> output = New<ScoreCollector>();

    size_t batchId = 0;
    std::mutex smutex;

    {
      ThreadPool pool(graphs_.size(), graphs_.size());

      while(*batchGenerator) {
        auto batch = batchGenerator->next();

        auto task = [=, &smutex](int id) {

          thread_local Ptr<ExpressionGraph> graph;
          thread_local Ptr<models::ModelBase> builder;

          if(!graph) {
            graph = graphs_[id % graphs_.size()];
            builder = models_[id % graphs_.size()];
          }

          auto costNode = builder->build(graph, batch);
          graph->forward();

          std::vector<float> scores;
          costNode->val()->get(scores);

          std::unique_lock<std::mutex> lock(smutex);
          for(size_t i = 0; i < batch->size(); ++i) {
            output->Write(batch->getSentenceIds()[i], scores[i]);
          }
        };

        pool.enqueue(task, batchId % graphs_.size());
        batchId++;
      }
    }

  }
};
}
