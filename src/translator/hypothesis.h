#pragma once
#include <memory>

#include "common/definitions.h"
#include "data/alignment.h"
#include "3rd_party/trieannosaurus/trieMe.h"

namespace marian {

class Hypothesis {
public:
  Hypothesis(std::vector<trieannosaurus::Node>* currTrieNode) : prevHyp_(nullptr), 
  prevIndex_(0), word_(0), pathScore_(0.0), currTrieNode_(currTrieNode), length_(0) {}

  Hypothesis(const Ptr<Hypothesis> prevHyp,
             Word word,
             IndexType prevIndex,
             float pathScore)
      : prevHyp_(prevHyp), prevIndex_(prevIndex), word_(word), pathScore_(pathScore), 
      currTrieNode_(prevHyp_->currTrieNode_), length_(prevHyp_->GetLength() + 1) {}

  const Ptr<Hypothesis> GetPrevHyp() const { return prevHyp_; }

  Word GetWord() const { return word_; }

  /*@TODO this one has the side effect of updating the trie node*/
  bool hasTrieContinuatuions() {
    //Assume matching vocabulary IDs. Will break otherwise.
    uint16_t id = (uint16_t)word_;
    currTrieNode_ = trieannosaurus::trieMeARiver::find(id, currTrieNode_);
    if (currTrieNode_) {
      return true;
    } else {
      return false;
    }
  }

  std::vector<trieannosaurus::Node>* GetTrieNode() { return currTrieNode_; }

  IndexType GetPrevStateIndex() const { return prevIndex_; }

  float GetPathScore() const { return pathScore_; }
  
  size_t GetLength() const { return length_; }

  std::vector<float>& GetScoreBreakdown() { return scoreBreakdown_; }
  std::vector<float>& GetAlignment() { return alignment_; }

  void SetAlignment(const std::vector<float>& align) { alignment_ = align; };

  // helpers to trace back paths referenced from this hypothesis
  Words TracebackWords()
  {
      Words targetWords;
      for (auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
          targetWords.push_back(hyp->GetWord());
          // std::cerr << hyp->GetWord() << " " << hyp << std::endl;
      }
      std::reverse(targetWords.begin(), targetWords.end());
      return targetWords;
  }

  std::vector<float> TracebackScores() {
    std::vector<float> scores;
    // traverse hypotheses backward
    for(auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
      // calculate a word score from the cumulative path score for all but the first word
      if(hyp->GetPrevHyp()) {
        scores.push_back(hyp->pathScore_ - hyp->GetPrevHyp().get()->pathScore_);
      } else {
        scores.push_back(hyp->pathScore_);
      }
    }
    std::reverse(scores.begin(), scores.end());
    return scores;
  }

  // get soft alignments for each target word starting from the hyp one
  typedef data::SoftAlignment SoftAlignment;
  SoftAlignment TracebackAlignment()
  {
      SoftAlignment align;
      for (auto hyp = this; hyp->GetPrevHyp(); hyp = hyp->GetPrevHyp().get()) {
          align.push_back(hyp->GetAlignment());
      }
      std::reverse(align.begin(), align.end());
      return align;
  }

private:
  const Ptr<Hypothesis> prevHyp_;
  const IndexType prevIndex_;
  const Word word_;
  const float pathScore_;
  std::vector<trieannosaurus::Node>* currTrieNode_;

  std::vector<float> scoreBreakdown_;
  std::vector<float> alignment_;
  const size_t length_;
};

typedef std::vector<Ptr<Hypothesis>> Beam;                // Beam = vector of hypotheses
typedef std::vector<Beam> Beams;                          // Beams = vector of vector of hypotheses
typedef std::tuple<Words, Ptr<Hypothesis>, float> Result; // (word ids for hyp, hyp, normalized sentence score for hyp)
typedef std::vector<Result> NBestList;                    // sorted vector of (word ids, hyp, sent score) tuples
}  // namespace marian
