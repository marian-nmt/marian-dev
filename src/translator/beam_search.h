#pragma once
#include <algorithm>

#include "marian.h"
#include "translator/history.h"
#include "translator/scorers.h"

#include "translator/helpers.h"
#include "translator/nth_element.h"

namespace marian {

class BeamSearch {
private:
  Ptr<Config> options_;
  std::vector<Ptr<Scorer>> scorers_;
  size_t beamSize_;

public:
  template <class... Args>
  BeamSearch(Ptr<Config> options,
             const std::vector<Ptr<Scorer>>& scorers,
             Args... args)
      : options_(options),
        scorers_(scorers),
        beamSize_(options_->has("beam-size")
                      ? options_->get<size_t>("beam-size")
                      : 3) {}

  Beams toHyps(const std::vector<uint> keys,
               const std::vector<float> costs,
               size_t vocabSize,
               const Beams& beams,
               std::vector<Ptr<ScorerState>>& states,
               size_t beamSize,
               bool first,
               Ptr<data::CorpusBatch> batch,
               const std::vector<size_t>& batchIndeces) {
    Beams newBeams(beams.size());

    std::vector<float> alignments;
    if(options_->get<bool>("alignment", false))
      alignments = scorers_[0]->getAlignment();

    for(int i = 0; i < keys.size(); ++i) {
      // Keys contains indices to vocab items in the entire beam.
      // Values can be between 0 and beamSize * vocabSize.
      int embIdx = keys[i] % vocabSize;
      int beamIdx = i / beamSize;

      // Retrieve short list for final softmax (based on words aligned
      // to source sentences). If short list has been set, map the indices
      // in the sub-selected vocabulary matrix back to their original positions.
      auto shortlist = scorers_[0]->getShortlist();
      if(shortlist)
        embIdx = shortlist->reverseMap(embIdx);

      if(newBeams[beamIdx].size() < beams[beamIdx].size()) {
        auto& beam = beams[beamIdx];
        auto& newBeam = newBeams[beamIdx];

        int hypIdx = keys[i] / vocabSize;
        float cost = costs[i];

        int hypIdxTrans
            = (hypIdx / beamSize) + (hypIdx % beamSize) * beams.size();
        if(first)
          hypIdxTrans = hypIdx;

        int beamHypIdx = hypIdx % beamSize;
        if(beamHypIdx >= beam.size())
          beamHypIdx = beamHypIdx % beam.size();

        if(first)
          beamHypIdx = 0;

        auto hyp = New<Hypothesis>(beam[beamHypIdx], embIdx, hypIdxTrans, cost);

        // Set cost breakdown for n-best lists
        if(options_->get<bool>("n-best")) {
          std::vector<float> breakDown(states.size(), 0);
          beam[beamHypIdx]->GetCostBreakdown().resize(states.size(), 0);
          for(int j = 0; j < states.size(); ++j) {
            int key = embIdx + hypIdxTrans * vocabSize;
            breakDown[j] = states[j]->breakDown(key)
                           + beam[beamHypIdx]->GetCostBreakdown()[j];
          }
          hyp->GetCostBreakdown() = breakDown;
        }

        // Set alignments
        if(!alignments.empty()) {
          // special handling if beam subdivided
          int batchIdx = beamIdx;
          if (batchIndeces.size() > 0 ) {
            batchIdx = batchIndeces[ beamIdx ];
          }
          auto align = getHardAlignmentsForHypothesis(
              alignments, batch, beamSize, beamHypIdx, batchIdx);
          hyp->SetAlignment(align);
        }

        newBeam.push_back(hyp);
      }
    }
    return newBeams;
  }

  std::vector<float> getHardAlignmentsForHypothesis(
      const std::vector<float> alignments,
      Ptr<data::CorpusBatch> batch,
      int beamSize,
      int beamHypIdx,
      int beamIdx) {
    // Let's B be the beam size, N be the number of batched sentences,
    // and L the number of words in the longest sentence in the batch.
    // The alignment vector:
    //
    // if(first)
    //   * has length of N x L if it's the first beam
    //   * stores elements in the following order:
    //     beam1 = [word1-batch1, word1-batch2, ..., word2-batch1, ...]
    // else
    //   * has length of N x L x B
    //   * stores elements in the following order:
    //     beams = [beam1, beam2, ..., beam_n]
    //
    // The mask vector is always of length N x L and has 1/0s stored like
    // in a single beam, i.e.:
    //   * [word1-batch1, word1-batch2, ..., word2-batch1, ...]
    //
    size_t batchSize = batch->size();
    size_t batchWidth = batch->width() * batchSize;
    std::vector<float> align;

    for(size_t w = 0; w < batchWidth / batchSize; ++w) {
      size_t a = ((batchWidth * beamHypIdx) + beamIdx) + (batchSize * w);
      size_t m = a % batchWidth;
      if(batch->front()->mask()[m] != 0)
        align.emplace_back(alignments[a]);
    }

    return align;
  }

  Beams pruneBeam(const Beams& beams) {
    Beams newBeams;
    for(auto beam : beams) {
      Beam newBeam;
      for(auto hyp : beam) {
        if(hyp->GetWord() > 0) {
          newBeam.push_back(hyp);
        }
      }
      newBeams.push_back(newBeam);
    }
    return newBeams;
  }

  Histories search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
    int dimBatch = batch->size();
    // initialize data structure for the search graph
    Histories histories;
    for(int i = 0; i < dimBatch; ++i) {
      size_t sentId = batch->getSentenceIds()[i];
      auto history = New<History>(sentId,
                                  options_->get<float>("normalize"),
                                  options_->get<float>("word-penalty"));
      histories.push_back(history);
    }

    size_t localBeamSize = beamSize_;

    // XML TODO: not dimBatch*2, but maximum number of subbeams
    // @TODO: unify this
    Ptr<NthElement> nth;
#ifdef CUDA_FOUND
    if(graph->getDevice().type == DeviceType::gpu)
      nth = New<NthElementGPU>(localBeamSize, dimBatch*2, graph->getDevice());
    else
#endif
      nth = New<NthElementCPU>(localBeamSize, dimBatch*2);

    // create a new beam (one for each input sentence = dimBatch)
    Beams beams(dimBatch);
    for(auto& beam : beams)
      beam.resize(localBeamSize, New<Hypothesis>());

    bool first = true;
    bool final = false;

    // add each beam to its history (the complete search graph)
    for(int i = 0; i < dimBatch; ++i)
      histories[i]->Add(beams[i]);

    // initialize computation graph
    for(auto scorer : scorers_) {
      scorer->clear(graph);
    }

    // create and add start states
    std::vector<Ptr<ScorerState>> states;
    for(auto scorer : scorers_) {
      states.push_back(scorer->startState(graph, batch));
    }

    // loop over output word predictions
    do {
      //**********************************************************************
      // create constant containing previous costs for current beam
      std::vector<size_t> hypIndices;
      std::vector<size_t> embIndices;
      std::vector<size_t> batchIndeces;;
      Expr prevCosts;

      if(first) {
        // initial hypothesis, no cost, no subbeams
        prevCosts = graph->constant({1, 1, 1, 1}, inits::from_value(0));
      } else {
        std::vector<float> beamCosts;

        // XML TODO break up beam into subbeams, based on xml state
        if (options_->get<bool>("xml-input")) {
          std::cerr << "splitting up beams\n";
          Beams subbeams(0);
          for(int j = 0; j < beams.size(); ++j) {
            auto& beam = beams[j];
            Beams singleSubbeams(2);
            for(int i = 0; i < beam.size(); ++i) {
              auto hyp = beam[i];
              size_t word = hyp->GetWord();
              // BELOW IS A DUMMY DIVISION - TODO: XML state
              singleSubbeams[ word % 2 ].push_back( hyp );
            }
            // merge into consolidated list
            for(int jj=0; jj<2; jj++) {
              auto& subbeam = singleSubbeams[jj];
              subbeams.push_back( subbeam );
              batchIndeces.push_back( j );
            }
          }
          beams = subbeams;
        }
        // XML TODO -------------------/

        std::cerr << "beam sizes ...";
        for(int j = 0; j < beams.size(); ++j) {
          auto& beam = beams[j];
          std::cerr << " " << beam.size();
        }
        std::cerr << "\n";

        for(int i = 0; i < localBeamSize; ++i) {
          for(int j = 0; j < beams.size(); ++j) {
            auto& beam = beams[j];
            if(i < beam.size()) {
              auto hyp = beam[i];
              hypIndices.push_back(hyp->GetPrevStateIndex());
              embIndices.push_back(hyp->GetWord());
              beamCosts.push_back(hyp->GetCost());
            } else {
              // WHY ALL THESE EMPTY ENTRIES???
              hypIndices.push_back(0);
              embIndices.push_back(0);
              beamCosts.push_back(-9999);
            }
          }
        }

        prevCosts = graph->constant({(int)localBeamSize, 1, (int)beams.size(), 1},
                                    inits::from_vector(beamCosts));
      }

      //**********************************************************************
      // prepare costs for beam search
      auto totalCosts = prevCosts;

      int beamCount = beams.size();
      for(int i = 0; i < scorers_.size(); ++i) {
        states[i] = scorers_[i]->step(graph,
                                      states[i],
                                      hypIndices,
                                      embIndices,
                                      beamCount,
                                      localBeamSize);

        if(scorers_[i]->getWeight() != 1.f)
          totalCosts
              = totalCosts + scorers_[i]->getWeight() * states[i]->getProbs();
        else
          totalCosts = totalCosts + states[i]->getProbs();
      }

      // make beams continuous
      if(beamCount > 1 && localBeamSize > 1)
        totalCosts = transpose(totalCosts, {2, 1, 0, 3});

      // forward step in computation graph - predict next word distribution
      if(first)
        graph->forward();
      else
        graph->forwardNext();

      //**********************************************************************
      // suppress specific symbols if not at right positions
      if(options_->has("allow-unk") && !options_->get<bool>("allow-unk"))
        suppressUnk(totalCosts);
      for(auto state : states)
        state->blacklist(totalCosts, batch);

      //**********************************************************************
      // perform beam search and pruning
      std::vector<unsigned> outKeys;
      std::vector<float> outCosts;

      // create maximum number of hypotheses for each beam
      std::vector<size_t> beamSizes(beamCount, localBeamSize);
      nth->getNBestList(beamSizes, totalCosts->val(), outCosts, outKeys, first);

      int dimTrgVoc = totalCosts->shape()[-1];
      beams = toHyps(outKeys,
                     outCosts,
                     dimTrgVoc,
                     beams,
                     states,
                     localBeamSize,
                     first,
                     batch,
                     batchIndeces);

      // XML TODO create additional hyps
      // XML TODO - that enter constraints
      // XML TODO - that continue constraints
      // XML TODO ---------------------------/

      // XML TODO merge subbeams
      if (options_->get<bool>("xml-input") && beamCount > 1) {
        Beams combinedBeams(batch->size()); 
        for(int j = 0; j < beams.size(); j++) {
          auto& beam = beams[j];
          for(int i = 0; i < beam.size(); ++i) {
            auto hyp = beam[i];
            combinedBeams[ batchIndeces[j] ].push_back( hyp );
          }
        }
        beams = combinedBeams;
      }
      // XML TODO ---------------------------/
      std::cerr << "merged beams ...";
      for(int j = 0; j < beams.size(); ++j) {
        auto& beam = beams[j];
        std::cerr << " " << beam.size();
      }
      std::cerr << "\n";

      // remove hypothesis that hit end of sentence (</s>)
      auto prunedBeams = pruneBeam(beams);

      for(int i = 0; i < (int)beams.size(); ++i) {
        if(!beams[i].empty()) {
          final = final
                  || histories[i]->size()
                         >= options_->get<float>("max-length-factor")
                                * batch->front()->batchWidth();
          histories[i]->Add(beams[i], prunedBeams[i].empty() || final);
        }
      }
      beams = prunedBeams;

      // reduce maximum beam size
      if(!first) {
        size_t maxBeam = 0;
        for(auto& beam : beams)
          if(beam.size() > maxBeam)
            maxBeam = beam.size();
        localBeamSize = maxBeam;
      }
      first = false;
      std::cerr << "pruned beams ...";
      for(int j = 0; j < beams.size(); ++j) {
        auto& beam = beams[j];
        std::cerr << " " << beam.size();
      }
      std::cerr << "\n";

    } while(localBeamSize != 0 && !final);

    return histories;
  }
};
}
