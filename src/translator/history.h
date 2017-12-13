#pragma once

#include <queue>

#include "hypothesis.h"

namespace marian {

class History {
private:
  struct HypothesisCoord {
    bool operator<(const HypothesisCoord& hc) const { return cost < hc.cost; }

    size_t i;
    size_t j;
    float cost;
  };

public:
  History(size_t lineNo, float alpha = 1.f);

  float LengthPenalty(size_t length);

  void Add(const Beam& beam, bool last = false);

  size_t size() const;

  NBestList NBest(size_t n) const;

  Result Top() const;

  size_t GetLineNum() const;

private:
  std::vector<Beam> history_;
  std::priority_queue<HypothesisCoord> topHyps_;
  size_t lineNo_;
  float alpha_;
};

typedef std::vector<Ptr<History>> Histories;
}
