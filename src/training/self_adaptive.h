#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/text_input.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

using namespace data;

class TrainSetReader {
  std::vector<UPtr<InputFileStream>> files_;

public:
  TrainSetReader(std::vector<std::string> paths) {
    for(auto& path : paths)
      files_.emplace_back(new InputFileStream(path));
  }

  std::vector<std::string> getSamples() {
    // extracted lines for source and target corpora
    std::vector<std::string> samples;
    // counters of number of lines extracted for source and target
    std::vector<size_t> counts;

    for(auto const& file : files_) {
      size_t currCount = 0;
      std::string lines;
      std::string line;
      while(std::getline((std::istream&)*file, line)) {
        if(line.empty())
          break;

        if(currCount)
          lines += "\n";
        lines += line;
        currCount += 1;
      }

      samples.emplace_back(lines);
      counts.push_back(currCount);

      // check if the same number of lines is extracted for source and target
      size_t prevCount = counts[0];
      for(size_t i = 1; i < counts.size(); ++i) {
        ABORT_IF(prevCount != counts[i],
                 "An empty source or target sentence has been encountered!");
        prevCount = counts[i];
      }
    }

    return samples;
  }
};

class TrainMultiDomain : public ModelTask {
public:
  TrainMultiDomain(Ptr<Config> options) : options_(options) {
    options_->set("max-length", 1000);

    // TODO: get rid of options_ or toptions_
    tOptions_ = New<Options>();
    tOptions_->merge(options_);

    auto deviceId = options_->getDevices()[0];

    // Initialize model for training
    graph_ = New<ExpressionGraph>();
    graph_->setDevice(deviceId);
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    builder_ = models::from_config(options_, models::usage::training);

    optimizer_ = Optimizer(options_);

    // Initialize model for translation
    Ptr<Options> opts = New<Options>();
    opts->merge(options_);
    opts->set("inference", true);
    builderTrans_ = models::from_options(opts, models::usage::translation);

    // Initialize a scorer for translation
    auto model = options_->get<std::string>("model");
    Ptr<Scorer> scorer = New<ScorerWrapper>(builderTrans_, "", 1.0f, model);
    scorers_.push_back(scorer);

    // Read vocabularies
    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");
    for(size_t i = 0; i < vocabPaths.size(); ++i) {
      Ptr<Vocab> vocab = New<Vocab>();
      vocab->load(vocabPaths[i], maxVocabs[i]);
      vocabs_.emplace_back(vocab);
    }

    // Load model
    builder_->load(graph_, model);
  }

  void run() {
    auto opts = New<Config>(*options_);
    opts->set<size_t>("mini-batch", 1);
    opts->set<size_t>("maxi-batch", 1);

    // Initialize input data
    auto srcPaths = options_->get<std::vector<std::string>>("input");
    std::vector<Ptr<Vocab>> srcVocabs(vocabs_.begin(), vocabs_.end() - 1);
    auto testset = New<Corpus>(srcPaths, srcVocabs, opts);

    // Prepare batches
    auto testBatches = New<BatchGenerator<Corpus>>(testset, opts);
    testBatches->prepare(false);
    size_t id = 0;

    // Initialize train data
    auto trainPaths = options_->get<std::vector<std::string>>("train-sets");
    auto trainSet = New<TrainSetReader>(trainPaths);

    // Initialize output printing
    auto collector = New<OutputCollector>();
    if(options_->get<bool>("quiet-translation"))
      collector->setPrintingStrategy(New<QuietPrinting>());
    auto printer = New<OutputPrinter>(options_, vocabs_.back());

    LOG(info, "Running...");

    while(*testBatches) {
      auto testBatch = testBatches->next();
      auto trainSents = trainSet->getSamples();

      if(!trainSents.empty()) {
        train(trainSents);
        translate(testBatch, collector, printer);
      } else {
        translate(testBatch, collector, printer, true);
      }
    }
  }

private:
  Ptr<Config> options_;
  Ptr<Options> tOptions_;

  Ptr<models::ModelBase> builder_;      // Training model
  Ptr<models::ModelBase> builderTrans_; // Translation model
  Ptr<ExpressionGraph> graph_;          // A graph with original parameters
  Ptr<ExpressionGraph> graphTemp_;      // A graph on which training is performed

  std::vector<Ptr<Vocab>> vocabs_;
  std::vector<Ptr<Scorer>> scorers_;
  Ptr<OptimizerBase> optimizer_;

  void train(std::vector<std::string> trainSents) {
    auto state = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, state);
    scheduler->registerTrainingObserver(scheduler);
    scheduler->registerTrainingObserver(optimizer_);

    auto trainSet = New<data::TextInput>(trainSents, vocabs_, options_);
    auto trainBatches
        = New<BatchGenerator<data::TextInput>>(trainSet, options_);

    bool first = true;

    scheduler->started();
    while(scheduler->keepGoing()) {
      trainBatches->prepare(false);

      while(*trainBatches && scheduler->keepGoing()) {
        auto batch = trainBatches->next();

        // Copy params from the original model
        if(first) {
          builder_->build(graph_, batch);
          graph_->forward();

          graphTemp_ = New<ExpressionGraph>();
          graphTemp_->setDevice(graph_->getDeviceId());
          graphTemp_->reuseWorkspace(graph_);

          graphTemp_->copyParams(graph_);
          first = false;
        }

        // Make an update step on the copy of the model
        auto costNode = builder_->build(graphTemp_, batch);
        graphTemp_->forward();
        float cost = costNode->scalar();
        graphTemp_->backward();

        // Notify optimizer and scheduler
        optimizer_->update(graphTemp_);
        scheduler->update(cost, batch);
      }
      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();
  }

  void translate(Ptr<data::CorpusBatch> batch,
                 Ptr<OutputCollector> collector,
                 Ptr<OutputPrinter> printer,
                 bool originalModel = false) {
    if(originalModel) {
      graph_->setInference(true);
      graph_->clear();
    } else {
      graphTemp_->setInference(true);
      graphTemp_->clear();
    }

    {

      auto search = New<BeamSearch>(tOptions_, scorers_,
          vocabs_.back()->GetEosId(), vocabs_.back()->GetUnkId());
      auto histories = search->search(originalModel ? graph_ : graphTemp_, batch);

      for(auto history : histories) {
        std::stringstream best1;
        std::stringstream bestn;
        printer->print(history, best1, bestn);
        collector->Write(history->GetLineNum(),
                         best1.str(),
                         bestn.str(),
                         options_->get<bool>("n-best"));
      }
    }

    if(originalModel)
      graph_->setInference(false);
    else
      graphTemp_->setInference(false);
  }
};
}
