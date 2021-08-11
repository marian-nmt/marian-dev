#pragma once

#include "common/config.h"
#include "common/file_stream.h"
#include "data/batch_generator.h"
#include "data/text_input.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"
#include "translator/swappable.h"

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

    auto modelFilename = options_->get<std::string>("model");
    optionsTrans_->set<std::vector<std::string>>("models", {modelFilename});

    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<std::string> srcVocabPaths(vocabPaths.begin(), vocabPaths.end() - 1);
    // TODO: or use optionsTrans_ here? cpuModel_ is used by both, trainin and translation, code
    // so i don't yet know what's the correct approach
    cpuModel_ = New<CPULoadedModel>(options_, modelFilename, srcVocabPaths, vocabPaths.back());
    translateEngine_ = New<GPUEngine>(optionsTrans_, 0);
    translateSlot_ = New<GPULoadedModel>(translateEngine_);
    trainEngine_ = New<GPUEngineTrain>(options_, 0);
    trainSlot_   = New<GPULoadedModelTrain>(trainEngine_);
    // trainSlot_->AllocateParamsLike(*cpuModel_);
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
    auto testSet = New<TextInput>(std::vector<std::string>({input}), cpuModel_->SrcVocabs(), optionsTrans_);

    // Prepare batches
    auto testBatches = New<BatchGenerator<TextInput>>(testSet, optionsTrans_);
    testBatches->prepare();

    // Initialize output printing
    auto collector = New<StringCollector>();
    auto printer = New<OutputPrinter>(optionsTrans_, cpuModel_->TrgVocab());

    // Get training sentences
    std::vector<std::vector<std::string>> contexts;
    if(yaml["context"])
      contexts = yaml["context"].as<std::vector<std::vector<std::string>>>();

    LOG(info, "Running...");

    size_t id = 0;
    for(auto testBatch : *testBatches) {
      if(contexts.size() > id && !contexts[id].empty()) {
        trainSlot_->SetModel(cpuModel_);
        trainSlot_->Train(contexts[id]);
        translateSlot_->PointToParams(*trainSlot_);
        translate(testBatch, collector, printer);
        needsSwitching_ = true;
      } else {
        LOG(info, "No context provided for sentence {}", id);
        if(needsSwitching_) {
          translateSlot_->Load(*cpuModel_);
          needsSwitching_ = false;
        }
        translate(testBatch, collector, printer);
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
    auto testSet = New<Corpus>(srcPaths, cpuModel_->SrcVocabs(), optionsTrans_);

    // Prepare batches
    auto testBatches = New<BatchGenerator<Corpus>>(testSet, optionsTrans_);
    testBatches->prepare();

    // Initialize output printing
    auto collector = New<OutputCollector>(options_->get<std::string>("output"));
    if(options_->get<bool>("quiet-translation"))
      collector->setPrintingStrategy(New<QuietPrinting>());
    auto printer = New<OutputPrinter>(options_, cpuModel_->TrgVocab());

    // Initialize train data
    auto trainPaths = options_->get<std::vector<std::string>>("train-sets");
    auto trainSets = New<TrainSetReader>(trainPaths);

    LOG(info, "Running...");

    for(auto testBatch : *testBatches) {
      auto trainSet = trainSets->getSamples();

      if(!trainSet.empty()) {
        LOG(info, "# NEW TEST BATCH");
        trainSlot_->SetModel(cpuModel_);
        trainSlot_->Train(trainSet);
        // translateSlot_->Load(*trainSlot_);
        translateSlot_->PointToParams(*trainSlot_);
        translate(testBatch, collector, printer);
        needsSwitching_ = true;
      } else {
        LOG(info, "# EMPTY TEST BATCH");
        if (needsSwitching_) {
          translateSlot_->Load(*cpuModel_);
          needsSwitching_ = false;
        }
        translate(testBatch, collector, printer);
      }
    }
  }

private:
  Ptr<Options> options_;       // Options for training
  Ptr<Options> optionsTrans_;  // Options for translator
  Ptr<CPULoadedModel> cpuModel_;
  Ptr<GPULoadedModelTrain> trainSlot_;
  Ptr<GPULoadedModel> translateSlot_;
  Ptr<GPUEngineTrain> trainEngine_;
  Ptr<GPUEngine> translateEngine_;
  bool needsSwitching_ = true;

  void translate(Ptr<data::CorpusBatch> batch,
                 Ptr<CollectorBase> collector,
                 Ptr<OutputPrinter> printer) {
    auto histories = translateSlot_->Translate(batch);

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
};
}
