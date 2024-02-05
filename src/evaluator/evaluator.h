#pragma once

#include "marian.h"

#include "common/config.h"
#include "common/options.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/corpus_nbest.h"
#include "models/costs.h"
#include "models/model_task.h"
#include "embedder/vector_collector.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

using namespace data;

/*
 * The tool is used to calculate metric score for various neural metrics.
 * @TODO: add the string-based matrics that we have already implemented like bleu and chrf.
 */
class Evaluator {
private:
  Ptr<models::IModel> model_;

public:
  Evaluator(Ptr<Options> options)
    : model_(createModelFromOptions(options, models::usage::evaluating)) {}

  void load(Ptr<ExpressionGraph> graph, Ptr<io::ModelWeights> modelFile) {
    model_->load(graph, modelFile);
  }

  Expr build(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
    auto evaluator = std::dynamic_pointer_cast<EncoderPooler>(model_);
    ABORT_IF(!evaluator, "Could not cast to EncoderPooler");
    return evaluator->apply(graph, batch, /*clearGraph=*/true)[0];
  }
};

/*
 * Actual Evaluate task. @TODO: this should be simplified in the future.
 */
template <class Model>
class Evaluate : public ModelTask {
private:
  Ptr<Options> options_;

  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<Ptr<Model>> models_;
  Ptr<io::ModelWeights> modelWeights_;

public:
  Evaluate(Ptr<Options> options) : options_(options) {
    options_ = options_->with("inference", true,
                              "shuffle", "none");

    /* Number of embeddings parameter is determined at runtime based on the given vocabulary file.
      In addtiion, this parameter has to be set before initializing the model object.
      Corpus initializer is the one that sets the number of embeddings into options_ object.
      However, we do not need to use corpus object here, so we just create a dummy corpus object.
    */
    Ptr<CorpusBase> corpus = New<Corpus>(options_);

    auto devices = Config::getDevices(options_);

    auto modelPath = options_->get<std::string>("model");
    LOG(info, "Loading model from {}", modelPath);

    modelWeights_ = New<io::ModelWeights>(modelPath);

    graphs_.resize(devices.size());
    models_.resize(devices.size());

    ThreadPool pool(devices.size(), devices.size());
    for(size_t i = 0; i < devices.size(); ++i) {
      pool.enqueue(
          [=](size_t j) {
            auto graph     = New<ExpressionGraph>(true);
            auto precison  = options_->get<std::vector<std::string>>("precision", {"float32"});
            graph->setDefaultElementType(typeFromString(precison[0])); // only use first type, used for parameter type in graph
            graph->setDevice(devices[j]);
            graph->reserveWorkspaceMB(options_->get<int>("workspace"));

            auto model = New<Model>(options_);
            model->load(graph, modelWeights_);

            models_[j] = model;
            graphs_[j] = graph;
          },
          i);
    }
  }

  void run() override {
    LOG(info, "Evaluating");
    timer::Timer timer;

    Ptr<CorpusBase> corpus = New<Corpus>(options_);
    corpus->prepare();
    auto batchGenerator = New<BatchGenerator<CorpusBase>>(corpus, options_);
    batchGenerator->prepare();

    Ptr<VectorCollector> output = VectorCollector::Create(options_);
    run(batchGenerator, output);
    LOG(info, "Total time: {:.5f}s wall", timer.elapsed());
  }

  template <typename T>
  void run(Ptr<BatchGenerator<T>> batchGenerator,  Ptr<VectorCollector> collector) {

    size_t batchId = 0;
    {
      ThreadPool pool(graphs_.size(), graphs_.size());

      for(auto batch : *batchGenerator) {
        auto task = [=](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local Ptr<Model> builder;

          if(!graph) {
            graph = graphs_[id % graphs_.size()];
            builder = models_[id % graphs_.size()];
          }

          auto scores = builder->build(graph, batch);
          graph->forward();

          // handle copying from fp32 or fp16 scores correctly.
          std::vector<float> sentVectors;
          if(scores->value_type() == Type::float32) {
            scores->val()->get(sentVectors);
          } else if (scores->value_type() == Type::float16) {
            std::vector<float16> sentVectors16;
            scores->val()->get(sentVectors16);
            sentVectors.reserve(sentVectors16.size());
            for(auto& v: sentVectors16)
              sentVectors.push_back(v);
          } else {
            ABORT("Unknown value type {}", scores->value_type());
          }

          // collect embedding vector per sentence.
          // if we compute similarities this is only one similarity per sentence pair.
          for(size_t i = 0; i < batch->size(); ++i) {
              auto numScores = scores->shape()[-1];
              auto beg = i * numScores;
              auto end = (i + 1) * numScores;
              std::vector<float> sentVector(sentVectors.begin() + beg, sentVectors.begin() + end);
              collector->Write((long)batch->getSentenceIds()[i], sentVector);
          }
        };

        pool.enqueue(task, batchId++);
      }
    }
  }

  std::string getModelConfig() {
    ABORT_IF(!modelWeights_, "Model weights are not loaded");
    YAML::Emitter outYaml;
    cli::OutputYaml(modelWeights_->getYamlFromModel(), outYaml);
    return outYaml.c_str();
  }

};

}  // namespace marian
