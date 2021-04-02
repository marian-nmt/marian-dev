#include "translator/history.h"
#include "translator/output_printer.h"
#include "translator/swappable.h"

#include <iostream>
#include <string>
#include <unordered_map>

namespace marian {
void LoadBig(Ptr<Options> options, std::unordered_map<std::string, CPULoadedModel> &to) {
  to.emplace("pten", CPULoadedModel(options,
      "/home/ubuntu/consistent-big-models/padded/pten.npz",
      {"/home/ubuntu/consistent-big-models/padded/pten.vocab"},
      "/home/ubuntu/consistent-big-models/padded/pten.vocab"));

  to.emplace("enit", CPULoadedModel(options,
      "/home/ubuntu/consistent-big-models/padded/enit.npz",
      {"/home/ubuntu/consistent-big-models/padded/enit.vocab"},
      "/home/ubuntu/consistent-big-models/padded/enit.vocab"));
}

void LoadTiny(Ptr<Options> options, std::unordered_map<std::string, CPULoadedModel> &to) {
  std::vector<std::string> models = {"csen", "encs", "enet", "eten", "esen", "enes"};
  for (const std::string m : models) {
    std::string base = "/home/ubuntu/consistent-bergamot-students/padded/";
    base += m + ".";
    to.emplace(m, CPULoadedModel(options, base + "npz", {base + "spm"}, base + "spm"));
  }
}

} // namespace

/* Demo program: run with options for any of the models */
int main(int argc, char** argv) {
  using namespace marian;
  Ptr<Options> options = parseOptions(argc, argv, cli::mode::translation);

  Ptr<GPUEngine> engine = New<GPUEngine>(options, 0);
  GPULoadedModel slot(engine);

  std::unordered_map<std::string, CPULoadedModel> models;
//  LoadBig(options, models);
  LoadTiny(options, models);

  // begin with a space to avoid conflict with a real sentence.
  const std::string kSwitchPrefix(" CHANGE ");

  bool alignments = !options->get<std::string>("alignment").empty();

  bool loaded = false;
  std::string line;
  while (std::getline(std::cin, line)) {
    // Switch out which model is used.
    if (line.substr(0, kSwitchPrefix.size()) == kSwitchPrefix) {
      std::string key = line.substr(kSwitchPrefix.size());
      auto found = models.find(key);
      if (found == models.end()) {
        std::cerr << "Model for " << key << " not loaded." << std::endl;
        return 1;
      }
      slot.Load(found->second);
      loaded = true;
      continue;
    }
    if (!loaded) {
      std::cerr << "Select a model first." << std::endl;
      continue;
    }

    // Actually translating with a model.
    marian::Histories histories = slot.Translate({line});
    // In practice there is one history because we provided one line.
    for(auto history : histories) {
      Result result(history->top());
      Words words = std::get<0>(result);
      std::cout << slot.TrgVocab()->decode(words) << std::endl;

      /* Print alignments */
      if (alignments) {
        Hypothesis &hypo = *std::get<1>(result);
        // [t][s] -> P(s|t)
        marian::data::SoftAlignment alignment(hypo.tracebackAlignment());
        // An easier call for this is:
        // std:cout << data::SoftAlignToString(alignment);
        // The below is just there to show how access them programatically.
        // NB you can convert to hard with data::ConvertSoftAlignToHardAlign(alignment, threshold)
        for (auto target : alignment) {
          for (float source : target) {
            std::cout << source << ' ';
          }
          std::cout << '\n';
        }
      }
    }
  }

  return 0;
}
