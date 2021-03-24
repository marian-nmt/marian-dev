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
class SwapperTranslator {
    private:
        Ptr<Options> opts_;
        Ptr<ExpressionGraph> graph_;
        Ptr<Scorer> scorer_;

        std::vector<Ptr<Vocab>> srcVocabs_;
        Ptr<Vocab> trgVocab_;

        // Models to store model;
        bool primary_ = true;
        std::vector<io::Item> primaryModel_;
        std::vector<io::Item> secondaryModel_;

    std::vector<io::Item> prepareItem(std::string path){
        std::vector<io::Item> ret = io::loadItems(path);
        // Find the special element and remove it:
        size_t special_idx = 0;
        for (size_t i = 0; i < ret.size(); i++) {
            if (ret[i].name == "special:model.yml") {
                special_idx = i;
                break;
            }
        }
        ret.erase(ret.begin() + special_idx);
        // Prepare the name so that it matches the named map
        for (auto&& item : ret) {
            item.name = "F0::" + item.name;
        }
        return ret;
    }

    public:
    SwapperTranslator(Ptr<Options> opt) : opts_(opt),
                                          primaryModel_(prepareItem(opt->get<std::vector<std::string>>("models")[0])),
                                          secondaryModel_(prepareItem(opt->get<std::string>("swap-model"))) {
        opts_->set("inference", true);
        opts_->set("shuffle", "none");

        // Get vocabs
        auto vocabPaths = opts_->get<std::vector<std::string>>("vocabs");
        std::vector<int> maxVocabs = opts_->get<std::vector<int>>("dim-vocabs");

        for(size_t i = 0; i < vocabPaths.size() - 1; ++i) {
            Ptr<Vocab> vocab = New<Vocab>(opts_, i);
            vocab->load(vocabPaths[i], maxVocabs[i]);
            srcVocabs_.emplace_back(vocab);
        }

        trgVocab_ = New<Vocab>(opts_, vocabPaths.size() - 1);
        trgVocab_->load(vocabPaths.back());

        // get device IDs
        auto devices = Config::getDevices(opts_);
        auto numDevices = devices.size();
        std::cerr << "Num devices: " << numDevices << std::endl;
        
        // Create graph
        graph_ = New<ExpressionGraph>();
        auto prec = opts_->get<std::vector<std::string>>("precision", {"float32"});
        graph_->setDefaultElementType(typeFromString(prec[0]));
        graph_->setDevice(devices[0]);
        graph_->reserveWorkspaceMB(opts_->get<size_t>("workspace"));
        scorer_ = createScorers(opts_)[0];
        scorer_->init(graph_);
        graph_->forward();
    }

    void translateTxt(std::string txt) {
        std::vector<std::string> instr(1, txt);
        auto corpus_ = New<data::TextInput>(instr, srcVocabs_, opts_);
        data::BatchGenerator<data::TextInput> batchGenerator(corpus_, opts_, nullptr, false);

        static const std::vector<Ptr<Scorer> > scorers(1, scorer_);
        auto search = New<BeamSearch>(opts_, scorers, trgVocab_);
        auto printer = New<OutputPrinter>(opts_, trgVocab_);
        static int i = 0;
        for (auto&& batch : batchGenerator) {
            auto histories = search->search(graph_, batch);
            for(auto history : histories) {
                std::stringstream best1;
                std::stringstream bestn;
                printer->print(history, best1, bestn);
                LOG(info, "Translation {} : {}", i, best1.str());
                i++;
            }
        }
    }

    void swapActual(std::vector<io::Item>& from) {
        auto namedMap = graph_->getParamsNamedMap();
        for (auto&& item : from) {
            auto to = reinterpret_cast<char *>(namedMap[item.name]->val()->memory()->data());
            swapper::copyCpuToGpu(to, &item.bytes[0], item.bytes.size());
        }
    }

    void swap() {
        timer::Timer timer;
        if (primary_) {
            swapActual(secondaryModel_);
            primary_ = false;
        } else {
            swapActual(primaryModel_);
            primary_ = true;
        }
        LOG(info, "Swap took: {:.8f}s wall", timer.elapsed());
    }
};
} // namespace marian

int main(int argc, char** argv) {
  using namespace marian;
  auto options = parseOptions(argc, argv, cli::mode::translation);
  SwapperTranslator swapper(options);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "quit") {
        break;
    } else if (line == "swap") {
        swapper.swap();
    } else {
        swapper.translateTxt(line);
    }
  }

  return 0;
}
