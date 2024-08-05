#pragma once

#include "marian.h"
#include "translator/history.h"
#include "translator/scorers.h"
#include "translator/nth_element.h"

namespace marian {

class BeamSearch {
private:
  Ptr<Options> options_;
  std::vector<Ptr<Scorer>> scorers_;
  size_t beamSize_;
  Ptr<const Vocab> trgVocab_;

  const float INVALID_PATH_SCORE;
  const bool PURGE_BATCH = true; // @TODO: diagnostic, to-be-removed once confirmed there are no issues.

  static float chooseInvalidPathScore(Ptr<Options> options) {
    // We are now using float32 for accumulation along path scores, so we can just use float32 for the invalid scores
    // Division by 2 to stay away from -inf. Here lowest / 2.f is bascially a magic number that marks finished beams.
    return NumericLimits<float>(Type::float32).lowest / 2.f;
  }

public:
  BeamSearch(Ptr<Options> options, const std::vector<Ptr<Scorer>>& scorers, const Ptr<const Vocab> trgVocab)
      : options_(options), scorers_(scorers), beamSize_(options_->get<size_t>("beam-size")), trgVocab_(trgVocab),
        INVALID_PATH_SCORE{chooseInvalidPathScore(options)}
  {}

  // combine new expandedPathScores and previous beams into new set of beams
  Beams toHyps(const std::vector<unsigned int>& nBestKeys, // [currentDimBatch, beamSize] flattened -> ((batchIdx, beamHypIdx) flattened, word idx) flattened
               const std::vector<float>& nBestPathScores,  // [currentDimBatch, beamSize] flattened
               const size_t nBestBeamSize, // for interpretation of nBestKeys
               const size_t vocabSize,     // ditto.
               const Beams& beams,
               const std::vector<Ptr<ScorerState /*const*/>>& states,
               Ptr<data::CorpusBatch /*const*/> batch, // for alignments only
               Ptr<class FactoredVocab/*const*/> factoredVocab, size_t factorGroup,
               const std::vector<bool>& dropBatchEntries, // [origDimBatch] - empty source batch entries are marked with true, should be cleared after first use.
               const std::vector<IndexType>& batchIdxMap) const;

  std::vector<float> getAlignmentsForHypothesis( // -> P(s|t) for current t and given beam and batch dim
      const std::vector<float> alignAll, // [beam depth, max src length, batch size, 1], flattened vector of all attention probablities
      Ptr<data::CorpusBatch> batch,
      int beamHypIdx,
      int currentBatchIdx,
      int origBatchIdx,
      int currentDimBatch) const;

  // remove all beam entries that have reached EOS
  Beams purgeBeams(const Beams& beams, /*in/out=*/std::vector<IndexType>& batchIdxMap);

  // main decoding function
  Histories search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch);
};

}  // namespace marian
