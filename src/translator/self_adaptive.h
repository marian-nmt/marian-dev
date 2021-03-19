#pragma once

#include "common/config.h"
#include "common/file_stream.h"
#include "data/batch_generator.h"
#include "data/text_input.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

using namespace data;

class TrainSetReader {
  std::vector<UPtr<io::InputFileStream>> files_;

public:
  TrainSetReader(std::vector<std::string> paths) {
    for(auto& path : paths)
      files_.emplace_back(new io::InputFileStream(path));
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
      while(io::getline(*file, line)) {
        if(line.empty())
          break;

        if(currCount)
          lines += "\n";
        lines += line;
        currCount += 1;
      }

      if(!lines.empty())
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

class TrainSelfAdaptive : public ModelTask, public ModelServiceTask {
public:
  TrainSelfAdaptive(Ptr<Options> options) : options_(options) {

    // @TODO: should probably better re-enable the shuffling related options
    // in config for marian-adaptive
    options_->set("shuffle", "none");
    // Set up translator options
    optionsTrans_ = New<Options>(options_->clone());
    optionsTrans_->set<size_t>("mini-batch", 1);
    optionsTrans_->set<size_t>("maxi-batch", 1);
    optionsTrans_->set<size_t>("max-length", 1000);
    optionsTrans_->set("shuffle", "none");

    auto deviceId = Config::getDevices(options_)[0];

    // Initialize model for training
    graph_ = New<ExpressionGraph>();
    graph_->setDevice(deviceId);
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    builder_ = models::createCriterionFunctionFromOptions(options_, models::usage::training);

    optimizer_ = Optimizer(options_);

    // Initialize model for translation
    Ptr<Options> opts = New<Options>();
    opts->merge(options_);
    opts->set("inference", true);
    builderTrans_ = models::createModelFromOptions(opts, models::usage::translation);

    // Initialize a scorer for translation
    auto model = options_->get<std::string>("model");
    Ptr<Scorer> scorer = New<ScorerWrapper>(builderTrans_, "", 1.0f, model);
    scorers_.push_back(scorer);

    // Read vocabularies
    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");
    for(size_t i = 0; i < vocabPaths.size(); ++i) {
      Ptr<Vocab> vocab = New<Vocab>(options_, i);
      vocab->load(vocabPaths[i], maxVocabs[i]);
      vocabs_.emplace_back(vocab);
    }

    // Load model
    builder_->load(graph_, model);
  }

  std::string run(const std::string& json) override {
    //LOG(warn, "REMOVEME Received Json:\n{}", json);

    // Check if input is in JSON
    YAML::Node yaml = YAML::Load(json);
    if(!yaml["input"]) {
      LOG(warn, "No 'input' node found in the request");
      return "";
    }

    // Get input sentences
    auto input = yaml["input"].as<std::string>();
    std::vector<Ptr<Vocab>> srcVocabs(vocabs_.begin(), vocabs_.end() - 1);
    auto testSet = New<TextInput>(std::vector<std::string>({input}), srcVocabs, optionsTrans_);

    // Prepare batches
    auto testBatches = New<BatchGenerator<TextInput>>(testSet, optionsTrans_);
    testBatches->prepare();

    // Initialize output printing
    auto collector = New<StringCollector>();
    auto printer = New<OutputPrinter>(options_, vocabs_.back());

    // Get training sentences
    std::vector<std::vector<std::string>> contexts;
    if(yaml["context"])
      contexts = yaml["context"].as<std::vector<std::vector<std::string>>>();

    LOG(info, "Running...");

    size_t id = 0;
    for(auto testBatch : *testBatches) {
      if(contexts.size() > id && !contexts[id].empty()) {
        train(contexts[id]);
        translate(testBatch, collector, printer, graphAdapt_);
      } else {
        LOG(info, "No context provided for sentence {}", id);
        translate(testBatch, collector, printer, graph_);
      }

      // iterating by 1 is quite safe because the mini-batch size for
      // translation is always 1
      ++id;
    }

    auto translations = collector->collect(options_->get<bool>("n-best"));
    YAML::Emitter output;
    output << YAML::DoubleQuoted << YAML::Flow << utils::join(translations, "\\n");
    return "{\"output\":" + std::string(output.c_str()) + "}";
  }

  void run() override {
    // Initialize input data
    auto srcPaths = options_->get<std::vector<std::string>>("input");
    std::vector<Ptr<Vocab>> srcVocabs(vocabs_.begin(), vocabs_.end() - 1);
    auto testSet = New<Corpus>(srcPaths, srcVocabs, optionsTrans_);

    // Prepare batches
    auto testBatches = New<BatchGenerator<Corpus>>(testSet, optionsTrans_);
    testBatches->prepare();

    // Initialize output printing
    auto collector = New<OutputCollector>(options_->get<std::string>("output"));
    if(options_->get<bool>("quiet-translation"))
      collector->setPrintingStrategy(New<QuietPrinting>());
    auto printer = New<OutputPrinter>(options_, vocabs_.back());

    // Initialize train data
    auto trainPaths = options_->get<std::vector<std::string>>("train-sets");
    auto trainSets = New<TrainSetReader>(trainPaths);

    LOG(info, "Running...");

    for(auto testBatch : *testBatches) {
      auto trainSet = trainSets->getSamples();

      if(!trainSet.empty()) {
        LOG(info, "# NEW TEST BATCH");
        train(trainSet);
        translate(testBatch, collector, printer, graphAdapt_);
      } else {
        LOG(info, "# EMPTY TEST BATCH");
        translate(testBatch, collector, printer, graph_);
      }
    }
  }

private:
  Ptr<Options> options_;       // Options for training
  Ptr<Options> optionsTrans_;  // Options for translator

  Ptr<models::ICriterionFunction> builder_;      // Training model
  Ptr<models::IModel> builderTrans_; // Translation model
  Ptr<ExpressionGraph> graph_;          // A graph with original parameters
  Ptr<ExpressionGraph> graphAdapt_;     // A graph on which training is performed

  std::vector<Ptr<Vocab>> vocabs_;
  std::vector<Ptr<Scorer>> scorers_;
  Ptr<OptimizerBase> optimizer_;

  void train(std::vector<std::string> trainSents) {
    auto state = New<TrainingState>(options_->get<float>("learn-rate"));
    auto scheduler = New<Scheduler>(options_, state);
    scheduler->registerTrainingObserver(scheduler);
    scheduler->registerTrainingObserver(optimizer_);

    auto trainSet = New<TextInput>(trainSents, vocabs_, options_);
    auto trainBatches = New<BatchGenerator<TextInput>>(trainSet, options_);

    bool first = true;

    scheduler->started();
    while(scheduler->keepGoing()) {
      trainBatches->prepare();

      LOG(info, "## NEW BATCHES");
      for(auto batch : *trainBatches) {
        if(!scheduler->keepGoing())
          break;

        LOG(info, "### NEW BATCH");
        // Copy params from the original model
        if(first) {
          builder_->build(graph_, batch);
          // TODO: Why do we need to do a froward pass here?
          graph_->forward();

          graphAdapt_ = New<ExpressionGraph>();
          graphAdapt_->setDevice(graph_->getDeviceId());
          graphAdapt_->reuseWorkspace(graph_);

          // TODO: why aren't we using a builder before this?
          // it's probably because the order doesn't matter and the
          // builder is used below
          graphAdapt_->copyParams(graph_);
          first = false;
        }

        // Make an update step on the copy of the model
        auto lossNode = builder_->build(graphAdapt_, batch);
        graphAdapt_->forward();
        StaticLoss loss = *lossNode;
        graphAdapt_->backward();

        // Notify optimizer and scheduler
        optimizer_->update(graphAdapt_, 1);
        scheduler->update(loss, batch);
      }
      if(scheduler->keepGoing())
        scheduler->increaseEpoch();
    }
    scheduler->finished();
  }

  void translate(Ptr<data::CorpusBatch> batch,
                 Ptr<CollectorBase> collector,
                 Ptr<OutputPrinter> printer,
                 Ptr<ExpressionGraph> graph) {
    graph->setInference(true);
    graph->clear();

    {
      auto search = New<BeamSearch>(options_,
                                    scorers_,
                                    vocabs_.back());
      auto histories = search->search(graph, batch);

      for(auto history : histories) {
        std::stringstream best1;
        std::stringstream bestn;
        printer->print(history, best1, bestn);
        collector->Write(history->getLineNum(),
                         best1.str(),
                         bestn.str(),
                         options_->get<bool>("n-best"));
      }
    }

    graph->setInference(false);
  }
};
}
