#include "translator/printer.h"
#include "translator/hypothesis.h"
#include "common/config.h"


namespace marian {

// std::vector<size_t> GetAlignment(const Ptr<Hypothesis>& hypothesis) {
  // std::vector<SoftAlignment> aligns;
  // auto last = hypothesis->GetPrevHyp();
  // while(last->GetPrevHyp().get() != nullptr) {
    // aligns.push_back(*(last->GetAlignment(0)));
    // last = last->GetPrevHyp();
  // }

  // std::vector<size_t> alignment;
  // for(auto it = aligns.rbegin(); it != aligns.rend(); ++it) {
    // size_t maxArg = 0;
    // for(size_t i = 0; i < it->size(); ++i) {
      // if((*it)[maxArg] < (*it)[i]) {
        // maxArg = i;
      // }
    // }
    // alignment.push_back(maxArg);
  // }

  // return alignment;
// }

std::string GetAlignmentString(const std::vector<size_t>& alignment) {
  std::stringstream alignString;
  alignString << " |||";
  for(size_t wordIdx = 0; wordIdx < alignment.size(); ++wordIdx) {
    alignString << " " << wordIdx << "-" << alignment[wordIdx];
  }
  return alignString.str();
}

void Printer(Ptr<Config> options,
             Ptr<Vocab> vocab,
             Ptr<History> history,
             std::ostream& best1,
             std::ostream& bestn) {
  if(options->has("n-best") && options->get<bool>("n-best")) {
    const auto& nbl = history->NBest(options->get<size_t>("beam-size"));

    for(size_t i = 0; i < nbl.size(); ++i) {
      const auto& result = nbl[i];
      const auto& words = std::get<0>(result);
      const auto& hypo = std::get<1>(result);

      float realCost = std::get<2>(result);

      std::string translation = Join((*vocab)(words));

      bestn << history->GetLineNum() << " ||| " << translation << " |||";

      if(hypo->GetCostBreakdown().empty()) {
        bestn << " F0=" << hypo->GetCost();
      } else {
        for(size_t j = 0; j < hypo->GetCostBreakdown().size(); ++j) {
          bestn << " F" << j << "= " << hypo->GetCostBreakdown()[j];
        }
      }

      bestn << " ||| " << realCost;

      if(i < nbl.size() - 1)
        bestn << std::endl;
      else
        bestn << std::flush;
    }
  }

  auto bestTranslation = history->Top();
  std::string translation = Join((*vocab)(std::get<0>(bestTranslation)));
  best1 << translation << std::flush;
}
}
