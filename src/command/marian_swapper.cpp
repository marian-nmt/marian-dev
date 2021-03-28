#include "marian.h"
#include "common/logging.h"
#include "data/corpus.h"
#include "data/text_input.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#include "common/io.h"
#include "common/timer.h"
#include <vector>
#include "tensors/gpu/swap.h"
namespace marian {

/* A model loaded on the CPU and possibly on a GPU */
class Model {
  private:
    std::vector<io::Item> parameters_;
    std::vector<Ptr<Vocab>> srcVocabs_;
    Ptr<Vocab> trgVocab_;

  public:
    Model(Ptr<Options> options, const std::string &parameters, const std::vector<std::string> &sourceVocabPaths, const std::string &targetVocabPath)
        : parameters_(io::loadItems(parameters)) {
      // Load parameters.
      // Find the special element and remove it:
      size_t special_idx = 0;
      for (size_t i = 0; i < parameters_.size(); i++) {
        if (parameters_[i].name == "special:model.yml") {
          special_idx = i;
          break;
        }
      }
      parameters_.erase(parameters_.begin() + special_idx);
      // Prepare the name so that it matches the named map
      for (auto&& item : parameters_) {
        item.name = "F0::" + item.name;
      }

      // Load source vocabs.
      const std::vector<int> &maxVocabs = options->get<std::vector<int>>("dim-vocabs");
      for(size_t i = 0; i < sourceVocabPaths.size(); ++i) {
        Ptr<Vocab> vocab = New<Vocab>(options, i);
        vocab->load(sourceVocabPaths[i], maxVocabs[i]);
        srcVocabs_.emplace_back(vocab);
      }

      // Load target vocab.
      trgVocab_ = New<Vocab>(options, sourceVocabPaths.size());
      trgVocab_->load(targetVocabPath);
    }

    const std::vector<io::Item> &Parameters() const { return parameters_; }

    const std::vector<Ptr<Vocab>> &SrcVocabs() const { return srcVocabs_; }

    Ptr<Vocab> TrgVocab() const { return trgVocab_; }
};

/* Reserved space on a GPU with which to translate */
class GPUSlot {
	private:
    Ptr<Options> options_;
    Ptr<ExpressionGraph> graph_;
    std::vector<Ptr<Scorer> > scorers_;

    // Last model used for translation.  Used to skip loading.
    const Model *loadedModel_;

    void Load(const std::vector<io::Item> &parameters) {
      timer::Timer timer;
      auto namedMap = graph_->getParamsNamedMap();
      for (auto&& item : parameters) {
        auto to = reinterpret_cast<char *>(namedMap[item.name]->val()->memory()->data());
        swapper::copyCpuToGpu(to, &item.bytes[0], item.bytes.size());
      }
      LOG(info, "Load took: {:.8f}s wall", timer.elapsed());
    }

  public:
    explicit GPUSlot(Ptr<Options> options) : options_(options), loadedModel_(nullptr) {
      options_->set("inference", true);
      options_->set("shuffle", "none");
      // get device IDs
      auto devices = Config::getDevices(options_);
      auto numDevices = devices.size();
      std::cerr << "Num devices: " << numDevices << std::endl;
        
      // Create graph
      graph_ = New<ExpressionGraph>();
      auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
      graph_->setDefaultElementType(typeFromString(prec[0]));
      graph_->setDevice(devices[0]);
      graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      // TODO: multiple scorers.
      Ptr<Scorer> scorer = createScorers(options_)[0];
      scorer->init(graph_);
      scorers_.push_back(scorer);
      graph_->forward();
    }

    void Translate(const Model &model, const std::vector<std::string> &input) {
      if (loadedModel_ != &model) {
        Load(model.Parameters());
        loadedModel_ = &model;
      }
      auto corpus = New<data::TextInput>(input, model.SrcVocabs(), options_);
      data::BatchGenerator<data::TextInput> batchGenerator(corpus, options_, nullptr, false);

      auto search = New<BeamSearch>(options_, scorers_, model.TrgVocab());
      auto printer = New<OutputPrinter>(options_, model.TrgVocab());
      for (auto&& batch : batchGenerator) {
        auto histories = search->search(graph_, batch);
        for(auto history : histories) {
          std::stringstream best1;
          std::stringstream bestn;
          printer->print(history, best1, bestn);
          LOG(info, "Translation {}", best1.str());
        }
      }
    }
};

} // namespace marian

/* Demo program */
int main(int argc, char** argv) {
  using namespace marian;
  Ptr<Options> options = parseOptions(argc, argv, cli::mode::translation);
  GPUSlot slot(options);
  Model pten(options,
      "/home/ubuntu/consistent-big-models/padded/pten.npz",
      {"/home/ubuntu/consistent-big-models/padded/pten.vocab"},
      "/home/ubuntu/consistent-big-models/padded/pten.vocab");

  Model enit(options,
      "/home/ubuntu/consistent-big-models/padded/enit.npz",
      {"/home/ubuntu/consistent-big-models/padded/enit.vocab"},
      "/home/ubuntu/consistent-big-models/padded/enit.vocab");

  const Model *model = &pten;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == " TRANSLATE PTEN") {
      model = &pten;
    } else if (line == " TRANSLATE ENIT") {
      model = &enit;
    } else {
      slot.Translate(*model, {line});
    }
  }

  return 0;
}
