#include "history.h"

namespace marian {

History::History(size_t lineNo, float alpha)
  : lineNo_(lineNo),
    alpha_(alpha) {}

float History::LengthPenalty(size_t length) {
  return std::pow((float)length, alpha_);
}

void History::Add(const Beam& beam, bool last) {
  if(beam.back()->GetPrevHyp() != nullptr) {
    for(size_t j = 0; j < beam.size(); ++j)
    if(beam[j]->GetWord() == 0 || last) {
      float cost = beam[j]->GetCost() / LengthPenalty(history_.size());
      topHyps_.push({history_.size(), j, cost});
      //std::cerr << "Add " << history_.size() << " " << j << " " << cost << std::endl;
    }
  }
  history_.push_back(beam);
}

size_t History::size() const {
  return history_.size();
}

NBestList History::NBest(size_t n) const {
  NBestList nbest;
  auto topHypsCopy = topHyps_;
  while(nbest.size() < n && !topHypsCopy.empty()) {
    auto bestHypCoord = topHypsCopy.top();
    topHypsCopy.pop();

    size_t start = bestHypCoord.i;
    size_t j = bestHypCoord.j;
    //float c = bestHypCoord.cost;
    //std::cerr << "h: " << start << " " << j << " " << c << std::endl;

    Words targetWords;
    Ptr<Hypothesis> bestHyp = history_[start][j];
    while(bestHyp->GetPrevHyp() != nullptr) {
      targetWords.push_back(bestHyp->GetWord());
      //std::cerr << bestHyp->GetWord() << " " << bestHyp << std::endl;
      bestHyp = bestHyp->GetPrevHyp();
    }

    std::reverse(targetWords.begin(), targetWords.end());
    nbest.emplace_back(targetWords,
                       history_[bestHypCoord.i][bestHypCoord.j],
                       bestHypCoord.cost);
  }
  return nbest;
}

Result History::Top() const { return NBest(1)[0]; }

size_t History::GetLineNum() const { return lineNo_; }
}
