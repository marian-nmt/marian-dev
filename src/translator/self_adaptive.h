#pragma once

#include "common/config.h"
#include "common/file_stream.h"
#include "data/batch_generator.h"
#include "data/iterator_facade.h"
#include "data/text_input.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"
#include "translator/swappable.h"

namespace marian {

using namespace data;

class AdaptiveContextReader;

/**
 * @brief An iterator for easier access of the context sentences produced by
 * `AdaptiveContextReader::getSamples()`
 */
class AdaptiveContextIterator
  : public IteratorFacade<AdaptiveContextIterator, std::vector<std::string>> {
private:
  AdaptiveContextReader* trainSetReader_;
  std::vector<std::string> currentSamples_;
public:
  // TODO: should we use a smart pointer here instead? The TrainSetReader::begin() method
  // would make it difficult
  AdaptiveContextIterator(AdaptiveContextReader* trainSetReader);

  bool equal(const AdaptiveContextIterator& other) const override {
    return other.trainSetReader_ == trainSetReader_;
  }

  const std::vector<std::string>& dereference() const override { return currentSamples_; }

  void increment() override;
};

/**
 * @brief Reads the context sentences, that are used for on-the-fly training in
 * the self-adaptive translation mode, from files.
 */
class AdaptiveContextReader {
  std::vector<UPtr<io::InputFileStream>> files_;
  /// Indicates whether the input files have been exhausted.
  bool eof_ = false;

public:
  /**
   * @brief Initializes a new reader by supplying paths to the files with
   * context sentences
   *
   * @param paths paths to the input files. The input files contain
   * newline-separated parallel sentence pairs (as usual for MT). Sentences are
   * grouped by the translatable sentences (which are provided in a different
   * file). Each group is delimited by a single empty line. The sentence group
   * can be empty (no context is provided for the respective translatable
   * sentence) in which case it is also represented by a single empty line.
   */
  AdaptiveContextReader(std::vector<std::string> paths) {
    for(auto& path : paths)
      files_.emplace_back(new io::InputFileStream(path));
  }

  /**
   * @brief Returns an iterator over the sets of context sentences produced by
   * `getSamples()`
   *
   * @return the beginning of the iterator.
   */
  AdaptiveContextIterator begin() {
    return AdaptiveContextIterator(this);
  }

  AdaptiveContextIterator end() {
    return AdaptiveContextIterator(nullptr);
  }

  bool eof() {
    return eof_;
  }

  /**
   * @brief Reads the next set of samples -- the contaxt sentences -- for
   * on-the-fly training in the self-adaptive translation mode.
   *
   * @details The input files contain newline-separated parallel sentence pairs
   * (as usual for MT). Sentences are grouped by the translatable sentences
   * (which are provided in a different file). Each group is delimited by a
   * single empty line. The sentence group can be empty (no context is provided
   * for the respective translatable sentence) in which case it is also
   * represented by a single empty line.
   *
   * @return a vector representing a single group of context sentences. Each
   * element in the vector contains newline seperated input lines comming from a
   * single file, e.g., [0] could contain 3 newline separated sentences in
   * English and [1] would contain their 3 respective translations in Latvian.
   */
  std::vector<std::string> getSamples() {
    // extracted lines for source and target corpora
    std::vector<std::string> samples;
    // counters of number of lines extracted for source and target
    std::vector<size_t> counts;

    // Early exit if input files are exhausted
    if (eof_) return samples;

    for(auto const& file : files_) {
      size_t currCount = 0;
      std::string lines;
      std::string line;
      bool fileEnded = true;
      while(io::getline(*file, line)) {
        if(line.empty()) {
          fileEnded = false;
          break;
        }

        if(currCount)
          lines += "\n";
        lines += line;
        currCount += 1;
      }
      eof_ = fileEnded;

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

AdaptiveContextIterator::AdaptiveContextIterator(AdaptiveContextReader* trainSetReader) : trainSetReader_(trainSetReader) {
  if(trainSetReader) {
    currentSamples_ = trainSetReader_->getSamples();
  }
}

void AdaptiveContextIterator::increment() {
  // If the previous increment has exhausted the file, we must indicate that the we've reached
  // the iterator's end
  if(trainSetReader_->eof() && trainSetReader_ != nullptr) {
    trainSetReader_ = nullptr;
    return;
  }
  // If we're at the end of the iterator and increment has been called yet another time, there's
  // a bug in the calling code
  ABORT_IF(trainSetReader_ == nullptr, "Incrementing past the end of the iterator isn't allowed");

  currentSamples_ = trainSetReader_->getSamples();
}

class TrainSelfAdaptive : public ModelTask, public ModelServiceTask {
public:
  TrainSelfAdaptive(Ptr<Options> options) : options_(options) {

    // @TODO: should probably better re-enable the shuffling related options
    // in config for marian-adaptive
    options_->set("shuffle", "none");
    // Set up translator options
    optionsTrans_ = New<Options>(options_->clone());
    // We will only ever translate a single sentence at a time because dynamic
    // adaptation happens per sentence
    optionsTrans_->set<size_t>("mini-batch", 1);
    optionsTrans_->set<size_t>("maxi-batch", 1);
    // TODO: should probably un-hardcode this? The issue is, though, that the users
    // might want separate options for training and translation
    optionsTrans_->set<size_t>("max-length", 1000);
    optionsTrans_->set("shuffle", "none");

    auto modelFilename = options_->get<std::string>("model");
    // Training has a single "model", translation can have multiple "models" in the general case.
    // Adaptive options also take a single "model" so we have to adapt translation options manually.
    optionsTrans_->set<std::vector<std::string>>("models", {modelFilename});

    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<std::string> srcVocabPaths(vocabPaths.begin(), vocabPaths.end() - 1);
    cpuModel_ = New<CPULoadedModel>(options_, modelFilename, srcVocabPaths, vocabPaths.back());
    translateEngine_ = New<GPUEngine>(optionsTrans_, 0);
    translateSlot_ = New<GPULoadedModel>(translateEngine_);
    trainEngine_ = New<GPUEngineTrain>(options_, 0);
    trainSlot_   = New<GPULoadedModelTrain>(trainEngine_);
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

    // Get training sentences
    std::vector<std::vector<std::string>> contexts;
    if(yaml["context"])
      contexts = yaml["context"].as<std::vector<std::vector<std::string>>>();

    LOG(info, "Running...");

    translate(testBatches, contexts.begin(), contexts.end(), collector);

    auto translations = collector->collect(options_->get<bool>("n-best"));
    YAML::Emitter output;
    output << YAML::DoubleQuoted << YAML::Flow << utils::join(translations, "\\n");
    return "{\"output\":" + std::string(output.c_str()) + "}";
  }

  template <class Iterator, class DataSet>
  void translate(
      Ptr<marian::data::BatchGenerator<DataSet>>
          testBatches,
      Iterator trainBegin,
      Iterator trainEnd,
      Ptr<marian::CollectorBase> collector) {
    auto printer = New<OutputPrinter>(options_, cpuModel_->TrgVocab());

    for(auto testBatch : *testBatches) {
      ABORT_IF(trainBegin == trainEnd, "Context batches ran out before test batches");

      auto trainSet = *trainBegin;
      ++trainBegin;

      if(!trainSet.empty()) {
        LOG(info, "# NEW TEST BATCH");
        trainSlot_->SetModel(cpuModel_);
        trainSlot_->Train(trainSet);
        translateSlot_->PointToParams(*trainSlot_);
        translate(testBatch, collector, printer);
        needsSwitching_ = true;
      } else {
        LOG(info, "# EMPTY TEST BATCH");
        if(needsSwitching_) {
          translateSlot_->Load(*cpuModel_);
          needsSwitching_ = false;
        }
        translate(testBatch, collector, printer);
      }
    }
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

    // Initialize train data
    auto trainPaths = options_->get<std::vector<std::string>>("train-sets");
    auto trainSets = New<AdaptiveContextReader>(trainPaths);

    LOG(info, "Running...");

    translate(testBatches, trainSets->begin(), trainSets->end(), collector);
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
