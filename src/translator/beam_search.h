#pragma once
#include <algorithm>

#include "marian.h"
#include "translator/history.h"
#include "translator/scorers.h"

#include "translator/helpers.h"
#include "translator/nth_element.h"

#include "data/xml.h"

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
               Ptr<data::CorpusBatch> batch) {
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
        //std::cerr << "mapping state " << hypIdx << " (" << (hypIdx / beamSize) << "," << (hypIdx % beamSize) << ") -> " << hypIdxTrans << "\n";

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
          auto align = getHardAlignmentsForHypothesis(
              alignments, batch, beamSize, beamHypIdx, beamIdx);
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

    // XML TODO - this is just for debugging, remove it afterwards
    std::vector<int> maxVocabs = options_->get<std::vector<int>>("dim-vocabs");
    std::vector<std::string> vocabPaths = options_->get<std::vector<std::string>>("vocabs");
    auto targetVocab = New<Vocab>();
    size_t id = vocabPaths.size()-1;
    int vocSize = targetVocab->load(vocabPaths[id], maxVocabs[id]);
    // XML TODO ----/
    for(int i = 0; i < dimBatch; ++i) {
      size_t sentId = batch->getSentenceIds()[i];
      auto history = New<History>(sentId,
                                  options_->get<float>("normalize"),
                                  options_->get<float>("word-penalty"));
      histories.push_back(history);
    }

    size_t localBeamSize = beamSize_;

    // @TODO: unify this
    Ptr<NthElement> nth;
#ifdef CUDA_FOUND
    if(graph->getDevice().type == DeviceType::gpu)
      nth = New<NthElementGPU>(localBeamSize, dimBatch, graph->getDevice());
    else
#endif
      nth = New<NthElementCPU>(localBeamSize, dimBatch);

    // create a new beam (one for each input sentence = dimBatch)
    const Ptr<data::XmlOptionsList> xmlOptionsList = batch->getXmlOptionsList();
    Beams beams(dimBatch);
    for(int i = 0; i < dimBatch; ++i) {
      auto& beam = beams[i];
      //beam.resize(localBeamSize, New<Hypothesis>( xmlOptions[i] ));
      beam.resize(localBeamSize, New<Hypothesis>());
    }

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
      Expr prevCosts;

      if(first) {
        // initial hypothesis, no cost, no subbeams
        prevCosts = graph->constant({1, 1, 1, 1}, inits::from_value(0));
      } else {
        std::vector<float> beamCosts;

        std::cerr << "starting beam sizes";
        for(int j = 0; j < beams.size(); ++j) {
          std::cerr << " " << beams[j].size();
        }
        std::cerr << "\n";

        for(int i = 0; i < localBeamSize; ++i) {
          for(int j = 0; j < beams.size(); ++j) {
            auto& beam = beams[j];
            if(i < beam.size()) {
              auto hyp = beam[i];
              //std::cerr << "i=" << i << " j=" << j << " pushback: " << hyp->GetPrevStateIndex() << "," << hyp->GetWord() << "," << hyp->GetCost() << "\n";
              hypIndices.push_back(hyp->GetPrevStateIndex());
              embIndices.push_back(hyp->GetWord());
              beamCosts.push_back( hyp->GetCost() );
            } else {
              hypIndices.push_back(0);
              embIndices.push_back(0);
              beamCosts.push_back(-9999);
            }
          }
        }

        prevCosts = graph->constant({(int)localBeamSize, 1, dimBatch, 1},
                                    inits::from_vector(beamCosts));
      }

      //**********************************************************************
      // prepare costs for beam search
      auto totalCosts = prevCosts;

      //if (states[0]->getProbs()) {
      //  std::cerr << "PROBS (A): ";
      //  std::cerr << states[0]->getProbs()->val()->debug();
      //  std::cerr << "\n";
      //}
      //if (0 && ! first) {
      //  std::cerr << "totalCosts (A): ";
      //  std::cerr << totalCosts->val()->debug();
      //  std::cerr << "\n";
      //}

      for(int i = 0; i < scorers_.size(); ++i) {
        states[i] = scorers_[i]->step(
            graph, states[i], hypIndices, embIndices, dimBatch, localBeamSize);

        if(scorers_[i]->getWeight() != 1.f)
          totalCosts
              = totalCosts + scorers_[i]->getWeight() * states[i]->getProbs();
        else
          totalCosts = totalCosts + states[i]->getProbs();
      }

      // make beams continuous
      if(dimBatch > 1 && localBeamSize > 1)
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

      //if (states[0]->getProbs()) {
      //  std::cerr << "PROBS (B): ";
      //  std::cerr << states[0]->getProbs()->val()->debug();
      //  std::cerr << "\n";
      //}
      //std::cerr << "totalCosts (B): ";
      //std::cerr << totalCosts->val()->debug();
      //std::cerr << "\n";

      // create maximum number of hypotheses for each beam
      std::vector<size_t> beamSizes(dimBatch, localBeamSize);

      int dimTrgVoc = totalCosts->shape()[-1];
      if (first) {
        nth->clearHypMask();
        nth->getNBestList(beamSizes, totalCosts->val(), outCosts, outKeys, first);
      }
      else {
        std::vector< std::vector<unsigned> > collectedKeys;
        std::vector< std::vector<float> > collectedCosts;
        
        // TODO: divide up subbeams by XML coverage
        // TODO: do not assigned started XML hyp to any beam
        for(int subbeam = 0; subbeam < 2; subbeam++) {
          std::vector<char> hypMask;
          std::vector<int> subbeamSize(beams.size(),0);
          for(int j = 0; j < beams.size(); j++) {
            auto& beam = beams[j];
            for(int i = 0; i < beam.size(); ++i) {
              auto hyp = beam[i];
              hypMask.push_back( i%2==subbeam ? 1 : 0 );
              if (i%2==subbeam) subbeamSize[j]++;
            }
            // do not expand filler hyps
            for(int i = beam.size(); i < localBeamSize; ++i) {
              hypMask.push_back( 0 );
            }
          }
          std::vector<unsigned> subKeys;
          std::vector<float> subCosts;

          // find n-best predictions
          nth->setHypMask(hypMask, dimTrgVoc);
          nth->getNBestList(beamSizes, totalCosts->val(), subCosts, subKeys, first);
          // merge them into the global list
          collectedKeys.push_back( subKeys );
          collectedCosts.push_back( subCosts );

          std::cerr << "SUBBEAM " << subbeam << "\n";
          for(size_t i=0; i<subCosts.size(); i++) {
            int embIdx = subKeys[i] % dimTrgVoc;
            int beamNo = i / localBeamSize;
            int hypInBeam = i % localBeamSize;
            int hypIdx = (subKeys[i] / dimTrgVoc) % localBeamSize;
            auto& beam = beams[beamNo];
            if (beam.size() == 0) continue;
            if (subbeamSize[beamNo] == 0) continue;
            if (subCosts[i] < -9999) continue; // junk hypothesis extension
            std::cerr << "beam " << beamNo << " hyp " << hypIdx << ">" << hypInBeam << "\tcost " << subCosts[i] << "\t " << (*targetVocab)[embIdx] << " ...";
            auto hyp = beam[hypIdx];
            std::cerr << "[" << hyp->GetPrevStateIndex() << "] ";
            while (hyp->GetWord() != 0) {
              std::cerr << " " << (*targetVocab)[hyp->GetWord()];
              hyp = hyp->GetPrevHyp();
            }
            std::cerr << std::endl;
          }
        }

        // create additional keys from XML constraints
        for(int j = 0; j < beams.size(); j++) {
          auto& beam = beams[j];
          // loop over all prior hypotheses
          for(int i = 0; i < beam.size(); ++i) {
            auto hyp = beam[i];
            auto& xmlCoveredList = hyp->GetXmlOptionCovered();
            // check on status of each XML constraints
            for(int k=0; k < xmlCoveredList.size(); k++) {   
              data::XmlOptionCovered &xmlCovered = xmlCoveredList[k];
              // already handled, move on
              if (xmlCovered.GetCovered()) {
                continue;
              }
              // check what word needs to be generated
              size_t wordPos = 0;
              if (xmlCovered.GetStarted()) {
                wordPos = xmlCovered.GetPosition();
              }
              const Words &output = xmlCovered.GetOption()->GetOutput();
              std::cerr << "start a hypothesis with word " << output[0] << "\n";
              // find out the score
              // merge into appropriate collectedCosts[ sub ], collectedKeys
            }
          }
        }

        // merge beams

        for(int j = 0; j < beams.size(); j++) {
          std::vector<int> index;
          index.push_back( j*localBeamSize );
          index.push_back( j*localBeamSize );
          for(int i=0; i<localBeamSize; i++) {
            float nextCost0 = collectedCosts[0][ index[0] ];
            float nextCost1 = collectedCosts[1][ index[1] ];
            if (nextCost0 > nextCost1) {
              std::cerr << "merge beam " << j << " from subbeam 0: " << collectedKeys[0][ index[0] ] << "," << nextCost0 << "\n";
              outCosts.push_back( nextCost0 );
              outKeys.push_back( collectedKeys[0][ index[0] ] );
              index[0]++;
            }
            else {
              std::cerr << "merge beam " << j << " from subbeam 1: " << collectedKeys[1][ index[1] ] << "," << nextCost1 << "\n";
              outCosts.push_back( nextCost1 );
              outKeys.push_back( collectedKeys[1][ index[1] ] );
              index[1]++;
            }
          }
        }
      }
      
      std::cerr << "outCosts.size() = " << outCosts.size() << "\n";

      for(size_t i=0; i<outCosts.size(); i++) {
        int embIdx = outKeys[i] % dimTrgVoc;
        int beamNo = i / localBeamSize;
        int hypInBeam = i % localBeamSize;
        int hypIdx = (outKeys[i] / dimTrgVoc) % localBeamSize;
        auto& beam = beams[beamNo];
        if (beam.size() == 0) continue;
        if (outCosts[i] < -9999) continue; // junk hypothesis extension
        auto hyp = beam[hypIdx];
        std::cerr << "beam " << beamNo << " hyp " << hypIdx << ">" << hypInBeam << "\tcost " << outCosts[i] << "\t " << (*targetVocab)[embIdx] << " ...";
        std::cerr << "[" << hyp->GetPrevStateIndex() << "] ";
        while (hyp->GetWord() != 0) {
           std::cerr << " " << (*targetVocab)[hyp->GetWord()];
           hyp = hyp->GetPrevHyp();
        }
        std::cerr << std::endl;
      }

      beams = toHyps(
          outKeys, outCosts, dimTrgVoc, beams, states, localBeamSize, first, batch);

      for(int j = 0; j < beams.size(); j++) {
        auto& beam = beams[j];
        for(int i = 0; i < beam.size(); ++i) {
          auto hyp = beam[i];
          std::cerr << "beam " << j << " hyp " << i << "\tcost " << hyp->GetCost() << "\t";
          std::cerr << "[" << hyp->GetPrevStateIndex() << "] ";
          while (hyp->GetWord() != 0) {
            std::cerr << " " << (*targetVocab)[hyp->GetWord()];
            hyp = hyp->GetPrevHyp();
          }
          std::cerr << std::endl;
        }
      }

      // XML TODO create additional hyps
      // XML TODO - that enter constraints
      // XML TODO - that continue constraints
      // XML TODO ---------------------------/

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
      std::cerr << "remaining beam sizes";
      for(int j = 0; j < beams.size(); ++j) {
        std::cerr << " " << beams[j].size();
      }
      std::cerr << "\n";


      // reduce maximum beam size
      if(!first) {
        size_t maxBeam = 0;
        for(auto& beam : beams)
          if(beam.size() > maxBeam)
            maxBeam = beam.size();
        localBeamSize = maxBeam;
      }
      first = false;

      for(int j = 0; j < beams.size(); j++) {
        auto& beam = beams[j];
        for(int i = 0; i < beam.size(); ++i) {
          auto hyp = beam[i];
          std::cerr << "beam " << j << " hyp " << i << "\tcost " << hyp->GetCost() << "\t";
          std::cerr << "[" << hyp->GetPrevStateIndex() << "] ";
          while (hyp->GetWord() != 0) {
            std::cerr << " " << (*targetVocab)[hyp->GetWord()];
            hyp = hyp->GetPrevHyp();
          }
          std::cerr << std::endl;
        }
      }

    } while(localBeamSize != 0 && !final);

    return histories;
  }
};
}
