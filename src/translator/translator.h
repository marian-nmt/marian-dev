#pragma once

#include <string>

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
#include "data/text_input.h"

#include "common/scheduling_parameter.h"
#include "common/timer.h"

#include "3rd_party/threadpool.h"

#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"

#include "models/model_task.h"
#include "translator/scorers.h"

namespace marian {

template <class Search>
class Translate : public ModelTask {
private:
  Ptr<Options> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  Ptr<data::Corpus> corpus_;
  Ptr<Vocab> trgVocab_;
  Ptr<const data::ShortlistGenerator> shortlistGenerator_;

  size_t numDevices_;
  std::vector<Ptr<io::ModelWeights>> modelWeights_;

public:
  Translate(Ptr<Options> options)
    : options_(options->clone()) {
    // This is currently safe as the translator is either created stand-alone or
    // or config is created anew from Options in the validator

    options_->set("inference", true,
                  "shuffle", "none");

    corpus_ = New<data::Corpus>(options_, /*translate=*/true);

    auto vocabs = options_->get<std::vector<std::string>>("vocabs");
    trgVocab_ = New<Vocab>(options_, vocabs.size() - 1);
    trgVocab_->load(vocabs.back());
    auto srcVocab = corpus_->getVocabs()[0];

    std::vector<int> lshOpts = options_->get<std::vector<int>>("output-approx-knn", {});
    ABORT_IF(lshOpts.size() != 0 && lshOpts.size() != 2, "--output-approx-knn takes 2 parameters");

    if (lshOpts.size() == 2 || options_->hasAndNotEmpty("shortlist")) {
      shortlistGenerator_ = data::createShortlistGenerator(options_, srcVocab, trgVocab_, lshOpts, 0, 1, vocabs.front() == vocabs.back());
    }

    auto devices = Config::getDevices(options_);
    numDevices_ = devices.size();

    ThreadPool threadPool(numDevices_, numDevices_);
    scorers_.resize(numDevices_);
    graphs_.resize(numDevices_);

    auto modelPaths = options->get<std::vector<std::string>>("models");

    // We now opportunistically mmap the model files anyways, but to keep backward compatibility
    // with the old --model-mmap option, we now croak if mmap is explicitly requested during decoding
    // but not possible in the actual graph, e.g. if --model-mmap is specified but the model file is
    // a npz-file or we decode on the GPU (will croak in different places).
    bool mmap     = options_->get<bool>("model-mmap", false);
    auto mmapMode = mmap ? io::MmapMode::RequiredMmap : io::MmapMode::OpportunisticMmap;

    for(auto modelPath : modelPaths) {
      LOG(info, "Loading model from {}", modelPath);
      modelWeights_.push_back(New<io::ModelWeights>(modelPath, mmapMode));
    }

    size_t id = 0;
    for(auto device : devices) {
      auto task = [&](DeviceId device, size_t id) {
        auto graph = New<ExpressionGraph>(true);
        auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
        graph->setDefaultElementType(typeFromString(prec[0]));
        graph->setDevice(device);
        if (device.type == DeviceType::cpu) {
          graph->getBackend()->setOptimized(options_->get<bool>("optimize"));
          graph->getBackend()->setGemmType(options_->get<std::string>("gemm-type"));
          graph->getBackend()->setQuantizeRange(options_->get<float>("quantize-range"));
        }
        graph->reserveWorkspaceMB(options_->get<int>("workspace"));
        graphs_[id] = graph;

        std::vector<Ptr<Scorer>> scorers = createScorers(options_, modelWeights_);

        for(auto scorer : scorers) {
          scorer->init(graph);
          if(shortlistGenerator_)
            scorer->setShortlistGenerator(shortlistGenerator_);
        }

        scorers_[id] = scorers;
        graph->forward();
      };

      threadPool.enqueue(task, device, id++);
    }

    if(options_->hasAndNotEmpty("output-sampling")) {
      if(options_->get<size_t>("beam-size") > 1)
        LOG(warn,
            "[warning] Enabling output sampling and beam search together (--output-sampling [...] && --beam-size > 1) results in so-called stochastic beam-search. "
            "Are you sure this is desired? For normal sampling, use --beam-size 1.");
    }
  }

  void run() override {
    data::BatchGenerator<data::Corpus> bg(corpus_, options_);

    ThreadPool threadPool(numDevices_, numDevices_);

    size_t batchId = 0;
    auto collector = New<OutputCollector>(options_->get<std::string>("output"));
    auto printer = New<OutputPrinter>(options_, trgVocab_);
    if(options_->get<bool>("quiet-translation"))
      collector->setPrintingStrategy(New<QuietPrinting>());

    // mutex for syncing counter and timer updates
    std::mutex syncCounts;

    // timer and counters for total elapsed time and statistics
    std::unique_ptr<timer::Timer> totTimer(new timer::Timer());
    size_t totBatches      = 0;
    size_t totLines        = 0;
    size_t totSourceTokens = 0;

    // timer and counters for elapsed time and statistics between updates
    std::unique_ptr<timer::Timer> curTimer(new timer::Timer());
    size_t curBatches      = 0;
    size_t curLines        = 0;
    size_t curSourceTokens = 0;

    // determine if we want to display timer statistics, by default off
    auto statFreq = SchedulingParameter::parse(options_->get<std::string>("stat-freq", "0u"));
    // abort early to avoid potentially costly batching and translation before error message
    ABORT_IF(statFreq.unit != SchedulingUnit::updates, "Units other than 'u' are not supported for --stat-freq value {}", statFreq);

    bool doNbest = options_->get<bool>("n-best");

    bg.prepare();
    for(auto batch : bg) {
      auto task = [=, &syncCounts,
                      &totBatches, &totLines, &totSourceTokens, &totTimer,
                      &curBatches, &curLines, &curSourceTokens, &curTimer](size_t id) {
        thread_local Ptr<ExpressionGraph> graph;
        thread_local std::vector<Ptr<Scorer>> scorers;

        if(!graph) {
          graph = graphs_[id % numDevices_];
          scorers = scorers_[id % numDevices_];
        }

        auto search = New<Search>(options_, scorers, trgVocab_);
        auto histories = search->search(graph, batch);

        for(auto history : histories) {
          std::stringstream best1;
          std::stringstream bestn;
          printer->print(history, best1, bestn);
          collector->Write((long)history->getLineNum(),
                           best1.str(),
                           bestn.str(),
                           doNbest);
        }

        // if we asked for speed information display this
        if(statFreq.n > 0) {
          std::lock_guard<std::mutex> lock(syncCounts);
          totBatches++;
          totLines        += batch->size();
          totSourceTokens += batch->front()->batchWords();

          curBatches++;
          curLines        += batch->size();
          curSourceTokens += batch->front()->batchWords();

          if(totBatches % statFreq.n == 0) {
            double totTime = totTimer->elapsed();
            double curTime = curTimer->elapsed();

            LOG(info,
                "Processed {} batches, {} lines, {} source tokens in {:.2f}s - Speed (since last): {:.2f} batches/s - {:.2f} lines/s - {:.2f} tokens/s",
                totBatches, totLines, totSourceTokens, totTime, curBatches / curTime, curLines / curTime, curSourceTokens / curTime);

            // reset stats between updates
            curBatches = curLines = curSourceTokens = 0;
            curTimer.reset(new timer::Timer());
          }
        }
      };

      threadPool.enqueue(task, batchId++);
    }

    // make sure threads are joined before other local variables get de-allocated
    threadPool.join_all();

    // display final speed numbers over total translation if intermediate displays were requested
    if(statFreq.n > 0) {
      double totTime = totTimer->elapsed();
      LOG(info,
          "Processed {} batches, {} lines, {} source tokens in {:.2f}s - Speed (total): {:.2f} batches/s - {:.2f} lines/s - {:.2f} tokens/s",
          totBatches, totLines, totSourceTokens, totTime, totBatches / totTime, totLines / totTime, totSourceTokens / totTime);
    }
  }
};

template <class Search>
class TranslateService : public ModelServiceTask {
private:
  Ptr<Options> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  std::vector<Ptr<Vocab>> srcVocabs_;
  Ptr<Vocab> trgVocab_;
  Ptr<const data::ShortlistGenerator> shortlistGenerator_;

  std::vector<Ptr<io::ModelWeights>> modelWeights_;

  size_t numDevices_;
  std::vector<std::vector<io::Item>> model_items_; // non-mmap

public:
  virtual ~TranslateService() {}

  TranslateService(const std::string& cliString)
    : TranslateService(parseOptions(cliString, cli::mode::translation, /*validate=*/true)) {}

  TranslateService(Ptr<Options> options)
    : options_(options->clone()) {
    // initialize vocabs
    options_->set("inference", true);
    options_->set("shuffle", "none");

    auto vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");

    for(size_t i = 0; i < vocabPaths.size() - 1; ++i) {
      Ptr<Vocab> vocab = New<Vocab>(options_, i);
      vocab->load(vocabPaths[i], maxVocabs[i]);
      srcVocabs_.emplace_back(vocab);
    }

    trgVocab_ = New<Vocab>(options_, vocabPaths.size() - 1);
    trgVocab_->load(vocabPaths.back());
    auto srcVocab = srcVocabs_.front();

    std::vector<int> lshOpts = options_->get<std::vector<int>>("output-approx-knn", {});
    ABORT_IF(lshOpts.size() != 0 && lshOpts.size() != 2, "--output-approx-knn takes 2 parameters");

    // load lexical shortlist
    if (lshOpts.size() == 2 || options_->hasAndNotEmpty("shortlist")) {
        shortlistGenerator_ = data::createShortlistGenerator(options_, srcVocab, trgVocab_, lshOpts, 0, 1, vocabPaths.front() == vocabPaths.back());
    }

    // get device IDs
    auto devices = Config::getDevices(options_);
    numDevices_ = devices.size();

    ThreadPool threadPool(numDevices_, numDevices_);
    scorers_.resize(numDevices_);
    graphs_.resize(numDevices_);

    bool mmap     = options_->get<bool>("model-mmap", false);
    auto mmapMode = mmap ? io::MmapMode::RequiredMmap : io::MmapMode::OpportunisticMmap;

    // preload models
    auto modelPaths = options->get<std::vector<std::string>>("models");
    for(auto modelPath : modelPaths)
      modelWeights_.push_back(New<io::ModelWeights>(modelPath, mmapMode));

    // initialize scorers
    size_t id = 0;
    for(auto device : devices) {
      auto task = [&](DeviceId device, size_t id) {
        auto graph = New<ExpressionGraph>(true);

        auto precison = options_->get<std::vector<std::string>>("precision", {"float32"});
        graph->setDefaultElementType(typeFromString(precison[0])); // only use first type, used for parameter type in graph
        graph->setDevice(device);
        if (device.type == DeviceType::cpu) {
          graph->getBackend()->setOptimized(options_->get<bool>("optimize"));
          graph->getBackend()->setGemmType(options_->get<std::string>("gemm-type"));
          graph->getBackend()->setQuantizeRange(options_->get<float>("quantize-range"));
        }
        graph->reserveWorkspaceMB(options_->get<int>("workspace"));
        graphs_[id] = graph;

        auto scorers = createScorers(options_, modelWeights_);
        for(auto scorer : scorers) {
          scorer->init(graph);
          if(shortlistGenerator_)
            scorer->setShortlistGenerator(shortlistGenerator_);
        }

        scorers_[id] = scorers;
        graph->forward();
      };

      threadPool.enqueue(task, device, id++);
    }
  }

  std::vector<std::string> run(const std::vector<std::string>& inputs, const std::string& yamlOverridesStr="") override {
      auto input = utils::join(inputs, "\n");
      auto translations = run(input, yamlOverridesStr);
      return utils::split(translations, "\n", /*keepEmpty=*/true);
  }

  std::string run(const std::string& input, const std::string& yamlOverridesStr="") override {
    YAML::Node configOverrides = YAML::Load(yamlOverridesStr);

    auto currentOptions = New<Options>(options_->clone());
    if (!configOverrides.IsNull()) {
      LOG(info,  "Overriding options:\n {}", configOverrides);
      currentOptions->merge(configOverrides, /*overwrite=*/true);
    }

    // split tab-separated input into fields if necessary
    auto inputs = currentOptions->get<bool>("tsv", false)
                      ? convertTsvToLists(input, currentOptions->get<size_t>("tsv-fields", 1))
                      : std::vector<std::string>({input});
    auto corpus_ = New<data::TextInput>(inputs, srcVocabs_, currentOptions);
    data::BatchGenerator<data::TextInput> batchGenerator(corpus_, currentOptions, nullptr, /*runAsync=*/false);

    auto collector = New<StringCollector>(currentOptions->get<bool>("quiet-translation", false));
    auto printer = New<OutputPrinter>(currentOptions, trgVocab_);
    size_t batchId = 0;

    batchGenerator.prepare();

    {
      ThreadPool threadPool_(numDevices_, numDevices_);

      for(auto batch : batchGenerator) {
        auto task = [=](size_t id) {
          thread_local Ptr<ExpressionGraph> graph;
          thread_local std::vector<Ptr<Scorer>> scorers;

          if(!graph) {
            graph = graphs_[id % numDevices_];
            scorers = scorers_[id % numDevices_];
          }

          auto search = New<Search>(currentOptions, scorers, trgVocab_);
          auto histories = search->search(graph, batch);

          for(auto history : histories) {
            std::stringstream best1;
            std::stringstream bestn;
            printer->print(history, best1, bestn);
            collector->add((long)history->getLineNum(), best1.str(), bestn.str());
          }
        };

        threadPool_.enqueue(task, batchId);
        batchId++;
      }
    }

    auto translations = collector->collect(currentOptions->get<bool>("n-best"));
    return utils::join(translations, "\n");
  }

private:
  // Converts a multi-line input with tab-separated source(s) and target sentences into separate lists
  // of sentences from source(s) and target sides, e.g.
  // "src1 \t trg1 \n src2 \t trg2" -> ["src1 \n src2", "trg1 \n trg2"]
  std::vector<std::string> convertTsvToLists(const std::string& inputText, size_t numFields) {
    std::vector<std::string> outputFields(numFields);

    std::string line;
    std::vector<std::string> lineFields(numFields);
    std::istringstream inputStream(inputText);
    bool first = true;
    while(std::getline(inputStream, line)) {
      utils::splitTsv(line, lineFields, numFields);
      for(size_t i = 0; i < numFields; ++i) {
        if(!first)
          outputFields[i] += "\n";  // join sentences with a new line sign
        outputFields[i] += lineFields[i];
      }
      if(first)
        first = false;
    }

    return outputFields;
  }
};
}  // namespace marian
