#pragma once

#include "common/config.h"
#include "common/file_stream.h"
#include "data/batch_generator.h"
#include "data/text_input.h"
#include "models/model_task.h"
#include "training/scheduler.h"
#include "training/validator.h"
#include "translator/swappable.h"
#include "data/adaptive_context.h"

namespace marian {

using namespace data;

/**
 * Implementation of the self-adaptive translation mode.
 * Self-adaptive translation means optionally using a set of context sentences
 * (e.g., provided by a translation memory), that are similar to the
 * translatable sentence, to train the model for a few iterations to fine-tune
 * it before translating the given sentence.
 */
class TrainSelfAdaptive : public ModelTask, public ModelServiceTask {
public:
  TrainSelfAdaptive(Ptr<Options> options) : options_(options) {
    options_->set("shuffle", "none");
    // Validation options are disabled for self-adaptive marian because
    // typically training would happen for only a few iterations and it seems to
    // not make much sense to run validation metrics on the validation dataset
    // then (especially if you care about translation performance). However, we
    // have to manually set the early-stopping option as disabled because the
    // scheduler crashes if it's not present.
    options_->set<size_t>("early-stopping", 0);
    // Set up translator options
    optionsTrans_ = New<Options>(options_->clone());
    // We will only ever translate a single sentence at a time because dynamic
    // adaptation happens per sentence
    optionsTrans_->set<size_t>("mini-batch", 1);
    optionsTrans_->set<size_t>("maxi-batch", 1);
    auto maxTranslationInput = options_->get<size_t>("max-length-translate");
    optionsTrans_->set<size_t>("max-length", maxTranslationInput);
    auto translationWorkspace = options_->get<size_t>("workspace-translate");
    optionsTrans_->set<size_t>("workspace", translationWorkspace);
    optionsTrans_->set("shuffle", "none");

    auto modelFilename = options_->get<std::string>("model");
    // Training has a single "model", translation can have multiple "models" in
    // the general case. Adaptive options also take only a single "model" so we
    // have to adapt translation options manually.
    optionsTrans_->set<std::vector<std::string>>("models", {modelFilename});

    // We mask the alignment option for training so that the alignment loss
    // nodes (self-attention heads) don't get added to the graph (for
    // transformers). Adding the alignment loss nodes and not supplying guided
    // alignments during training results in a crash with "There are more (n)
    // than one top most nodes for the backward pass". In self-adaptive
    // translation we don't support training the alignments because they are
    // likely to remain good enough after the few self-adaptive updates.
    //
    // TODO: regarding the above, make the alignment heads non-trainable; afaik,
    // they are treated like regular attantion heads currently which might
    // decrease alignment precision.
    options_->set("alignment", "");

    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<std::string> srcVocabPaths(vocabPaths.begin(), vocabPaths.end() - 1);
    cpuModel_ = New<CPULoadedModel>(options_, modelFilename, srcVocabPaths, vocabPaths.back());
    auto translateEngine = New<GPUEngineTranslate>(optionsTrans_, 0);
    translateSlot_ = New<GPULoadedModel>(translateEngine);
    auto trainEngine = New<GPUEngineTrain>(options_, 0);
    trainSlot_   = New<SwappableModelTrainer>(trainEngine);
  }

  /**
   * Implementation for self-adaptive translation where data come from a
   * web request.
   *
   * @param json Input data in JSON. An "input" array of strings is expected to
   * contain translatable sentences, each of which has a corresponding set of
   * context sentences as a sub-array in the "context" array.
   *
   * @return JSON-encoded translations
   */
  std::string run(const std::string& json) override {
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

    adaptAndTranslate(testBatches, contexts.begin(), contexts.end(), collector);

    auto translations = collector->collect(options_->get<bool>("n-best"));
    YAML::Emitter output;
    output << YAML::DoubleQuoted << YAML::Flow << utils::join(translations, "\\n");
    return "{\"output\":" + std::string(output.c_str()) + "}";
  }

  /**
   * Implementation for self-adaptive translation where inputs and
   * outputs are specified in CLI options.
   */
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

    adaptAndTranslate(testBatches, trainSets->begin(), trainSets->end(), collector);
  }

private:
  Ptr<Options> options_;                  // Options for training
  Ptr<Options> optionsTrans_;             // Options for translator
  Ptr<CPULoadedModel> cpuModel_;          // Holds model parameters and vocabularies
  Ptr<SwappableModelTrainer> trainSlot_;  // Performs model training
  Ptr<GPULoadedModel> translateSlot_;     // Performs translation with the model
  bool needsSwitching_ = true;            // Tracks whether translate slot's model needs to be reset

  template <class Iterator, class DataSet>
  void adaptAndTranslate(
      Ptr<marian::data::BatchGenerator<DataSet>> testBatches,
      Iterator trainBegin,
      Iterator trainEnd,
      Ptr<marian::CollectorBase> collector) {
    auto printer = New<OutputPrinter>(optionsTrans_, cpuModel_->TrgVocab());

    for(auto testBatch : *testBatches) {
      ABORT_IF(trainBegin == trainEnd, "Context batches ran out before test batches");

      auto trainSet = *trainBegin;
      ++trainBegin;

      if(!trainSet.empty()) {
        LOG(info, "Got {} context sentences", trainSet.size());
        trainSlot_->SetModel(cpuModel_);
        trainSlot_->Train(trainSet);
        translateSlot_->PointToParams(*trainSlot_);
        translate(testBatch, collector, printer);
        needsSwitching_ = true;
      } else {
        LOG(info, "No context");
        if(needsSwitching_) {
          translateSlot_->Load(*cpuModel_);
          needsSwitching_ = false;
        }
        translate(testBatch, collector, printer);
      }
    }
  }

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
