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

  void mergeIntoSortedKeysCosts(std::vector<uint> &keys,
                                std::vector<float> &costs,
                                std::vector<data::XmlOptionCoveredList *> &xmls,
                                uint key, 
                                float cost,
                                data::XmlOptionCoveredList *xml) {
    auto k=keys.begin();
    auto c=costs.begin();
    auto x=xmls.begin();
    while(c != costs.end() && cost < *c) {
      k++; c++; x++;
    }
    keys.insert(k, key);
    costs.insert(c, cost);
    xmls.insert(x, xml);
  }

  Beams toHyps(const std::vector<uint> keys,
               const std::vector<float> costs,
               const std::vector<data::XmlOptionCoveredList *> &xmls,
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

      // Condition: do not add filler items in the keys/costs matrix
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

        if (options_->get<bool>("xml-input")) {
          hyp->SetXml(xmls[i]);
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
      size_t validCount = 0;
      for(auto hyp : beam) {
        if(hyp->GetWord() > 0) {
          newBeam.push_back(hyp);
          validCount++;
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
    const data::XmlOptionsList *xmlOptionsList = 
      options_->get<bool>("xml-input") ? batch->getXmlOptionsList() : NULL;
    if (options_->get<bool>("xml-input")) {
      std::cerr << "pulling xmlOptionsList " << batch->getXmlOptionsList() << "\n";
      std::cerr << "xmlOptions " << xmlOptionsList->at(0) << "\n";
    }
    Beams beams(dimBatch);
    for(int i = 0; i < dimBatch; ++i) {
      auto& beam = beams[i];
      if (options_->get<bool>("xml-input"))
        beam.resize(localBeamSize, New<Hypothesis>( xmlOptionsList->at(i) ));
      else
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
    std::cerr << "still alive and kicking 4\n";

    // loop over output word predictions
    do {
      //**********************************************************************
      // create constant containing previous costs for current beam
      std::vector<size_t> hypIndices;
      std::vector<size_t> embIndices;
      Expr prevCosts;

      for(int j = 0; j < beams.size(); ++j) {
        auto& beam = beams[j];
        for(int i = 0; i < localBeamSize; ++i) {
          if(i < beam.size()) {
            auto hyp = beam[i];
            std::cerr << "beam=" << j << " i=" << i;
            if (options_->get<bool>("xml-input")) {
              std::cerr << " xml status=" << hyp->GetXmlStatus() << "/" << hyp->GetXmlOptionCovered()->size();
            }
            std::cerr << "\n";
          }
        }
      }
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
      int dimTrgVoc = totalCosts->shape()[-1];
      std::cerr << "starting beam sizes";
      for(int j = 0; j < beams.size(); ++j) {
        std::cerr << " " << beams[j].size();
      }
      std::cerr << "\n";

      // create maximum number of hypotheses for each beam
      std::vector<unsigned> outKeys;
      std::vector<float> outCosts;
      std::vector<data::XmlOptionCoveredList *> outXmls;
      if (options_->get<bool>("xml-input")) {
        xmlSearch(nth, beams, localBeamSize, totalCosts, outCosts, outKeys, outXmls, first, targetVocab, batch);
      }
      else {
        std::vector<size_t> beamSizes(dimBatch, localBeamSize);
        nth->getNBestList(beamSizes, totalCosts->val(), outCosts, outKeys, first);
      }

      std::cerr << "outCosts.size() = " << outCosts.size() << ", localBeamSize = " << localBeamSize << "\n";
      for(size_t i=0; i<outCosts.size(); i++) {
        int beamNo = i / localBeamSize;
        int hypInBeam = i % localBeamSize;
        int embIdx = outKeys[i] % dimTrgVoc;
        int hypIdx = (outKeys[i] / dimTrgVoc) % localBeamSize;
        auto& beam = beams[beamNo];
        if (hypInBeam >= beam.size()) { // do not report on filler hyps
          continue;
        }
        if (hypIdx >= beam.size()) {
          std::cerr << "ERR " << "beam " << beamNo << " hyp " << hypIdx << ">" << hypInBeam << "\tcost " << outCosts[i] << "\t " << (*targetVocab)[embIdx] << " ... OUT OF RANGE " << beam.size() << "\n";
        }
        if (beam.size() == 0) continue;
        if (outCosts[i] < -9999) continue; // junk hypothesis extension
        if (hypIdx >= beam.size()) continue;
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
          outKeys, outCosts, outXmls, dimTrgVoc, beams, states, localBeamSize, first, batch);

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

      // remove hypothesis that hit end of sentence (</s>)
      std::cerr << "pruning the beam\n";
      auto prunedBeams = pruneBeam(beams);
      for(int i = 0; i < (int)beams.size(); ++i) {
        if(!beams[i].empty()) {
          final = final
                  || histories[i]->size()
                         >= options_->get<float>("max-length-factor")
                                * batch->front()->batchWidth();
          std::cerr << "histories[i]->Add(beams[i]\n";
          histories[i]->Add(beams[i], prunedBeams[i].empty() || final);
          std::cerr << "histories[i]->Add(beams[i] OK\n";
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
      std::cerr << "DONE WITH LOOP, localBeamSize now " << localBeamSize << "\n";

    } while(localBeamSize != 0 && !final);

    return histories;
  }

  void xmlSearch(Ptr<NthElement> nth,
                 Beams &beams,
                 size_t localBeamSize, 
                 Expr &totalCosts, 
                 std::vector<float> &outCosts, 
                 std::vector<unsigned> &outKeys, 
                 std::vector<data::XmlOptionCoveredList *> &outXmls,
                 bool first,
                 Ptr<Vocab> targetVocab,
                 Ptr<data::CorpusBatch> batch) {

    int dimBatch = beams.size();
    int dimTrgVoc = totalCosts->shape()[-1];

    std::vector< std::vector< std::vector<unsigned> > > collectedKeys;
    std::vector< std::vector< std::vector<float> > > collectedCosts;
    std::vector< std::vector< std::vector<data::XmlOptionCoveredList*> > > collectedXmls;
    // get maximum number of xml options per sentence
    // (-> plus 1 = number of subbeams)
    std::vector< size_t > xmlCount;
    size_t maxXmlCount = 0;
    for(int j = 0; j < beams.size(); j++) {
      auto& beam = beams[j];
      if (beam.size() == 0) {
        xmlCount.push_back( 0 );
      }
      else {
        auto hyp = beam[0];
        size_t thisXmlCount = hyp->GetXmlOptionCovered()->size();
        xmlCount.push_back( thisXmlCount );
        std::cerr << "beam " << j << ": " << thisXmlCount << "\n";
        if (thisXmlCount > maxXmlCount) {
          maxXmlCount = thisXmlCount;
        }
      }
    }

    // create (empty) lists for keys, costs, and xml states
    size_t subbeamCount = maxXmlCount+1;
    std::cerr << "subbeamCount = " << subbeamCount << "\n";
    for(int j = 0; j < beams.size(); j++) {
      std::vector< std::vector<unsigned> > keysVector;
      std::vector< std::vector<float> > costsVector;
      std::vector< std::vector<data::XmlOptionCoveredList*> > xmlVector;
      collectedKeys.push_back( keysVector );
      collectedCosts.push_back( costsVector );
      collectedXmls.push_back( xmlVector );
      for(int subbeam = 0; subbeam < subbeamCount; subbeam++) {
        std::vector<unsigned> keys;
        std::vector<float> costs;
        std::vector<data::XmlOptionCoveredList*> xml;
        collectedKeys[j].push_back(keys);
        collectedCosts[j].push_back(costs);
        collectedXmls[j].push_back(xml);
      }
    }


    std::cerr << "ADDING ADDITIONAL HYPOTHESES\n";
    for(int j = 0; j < beams.size(); j++) {
      auto& beam = beams[j];
      // loop over all prior hypotheses
      for(int i = 0; i < beam.size(); ++i) {
        if (first && i > 0) {
          break; // not sure why, but first has additional fake hyps
        }
        auto hyp = beam[i];
        auto xmlCoveredList = hyp->GetXmlOptionCovered();
        // check on status of each XML constraints
        for(int k=0; k < xmlCoveredList->size(); k++) {   
          std::cerr << "checking xml option " << k << "\n";
          const data::XmlOptionCovered &xmlCovered = xmlCoveredList->at(k);
          // already handled, move on
          if (xmlCovered.GetCovered()) {
            std::cerr << "already covered\n";
            continue;
          }
          // check what word needs to be generated
          size_t wordPos = 0;
          if (xmlCovered.GetStarted()) {
            wordPos = xmlCovered.GetPosition();
            std::cerr << "already started\n";
          }
          std::cerr << "proceeding at wordPos " << wordPos << "\n";
          const data::XmlOption *xmlOption = xmlCovered.GetOption();
          const Words &output = xmlOption->GetOutput();
          std::cerr << "xmlCovered = " << xmlOption->GetStart() << "-" << xmlOption->GetEnd() << ", output length " << output.size();
          for(size_t o=0;o<output.size();o++) {
            std::cerr << " " << (*targetVocab)[output[o]];
          }
          std::cerr << "\n";
          // find out the score
          //std::cerr << "it's in here somewhere... ";
          //std::cerr << totalCosts->val()->debug();
          //std::cerr << "\n";
          uint key = ((j * localBeamSize) + i) * dimTrgVoc + output[wordPos];
          if (first) { // only one hyp per beam
            key = j * dimTrgVoc + output[wordPos];
          }
          float cost = totalCosts->val()->get(key);
          std::cerr << "beam j=" << j << " hyp i=" << i << ", word " << (*targetVocab)[output[wordPos]] << ":" << output[wordPos] << ", total cost is in " << key << ": " << cost << "\n";
          // subbeam to be placed into 
          int subbeam = hyp->GetXmlStatus();
          // update xmlCoveredList
          std::cerr << "build new XmlOptionCoveredList\n";
          auto newXmlCoveredList = new data::XmlOptionCoveredList;
          for(int kk=0; kk < xmlCoveredList->size(); kk++) {
            data::XmlOptionCovered newXmlCovered = xmlCoveredList->at(kk);//copy
            std::cerr << "option " << kk << "\n";
            if (kk == k) {
              // the one we are currently processing
              if (newXmlCovered.GetStarted()) {
                // already started
                std::cerr << "newXmlCovered.Proceed();\n";
                newXmlCovered.Proceed();
              }
              else {
                std::cerr << "newXmlCovered.Start();\n";
                newXmlCovered.Start();
                subbeam++; // resulting hyp will be in next subbeam
              }
              auto alignments = scorers_[0]->getAlignment();
              float weight = options_->get<float>("xml-alignment-weight");
              std::cerr << "ALIGN: alignments.size() = " << alignments.size() << "\n";
              if (!alignments.empty() && weight != 0.0) {
                auto align = getHardAlignmentsForHypothesis(alignments, batch, localBeamSize, i, j);
                std::cerr << "ALIGN: align.size() = " << align.size() << "\n";
                cost -= weight * newXmlCovered.GetAlignmentCost();
                newXmlCovered.AddAlignmentCost( align );
                cost += weight * newXmlCovered.GetAlignmentCost();
              }
            }
            else {
              // other xml options
              if (newXmlCovered.GetStarted()) {
                std::cerr << "newXmlCovered.Abandon();\n";
                float weight = options_->get<float>("xml-alignment-weight");
                cost -= weight * newXmlCovered.GetAlignmentCost();
                newXmlCovered.Abandon();
                subbeam--; // resulting hyp will be one subbeam lower
              }
              else {
                std::cerr << "just copy\n";
              }
            }
            newXmlCoveredList->push_back(newXmlCovered);
          }
          //subbeam = 0;
          std::cerr << "merge onto subbeam " << subbeam << "\n";
          mergeIntoSortedKeysCosts(collectedKeys[j][subbeam], 
                                   collectedCosts[j][subbeam],
                                   collectedXmls[j][subbeam],
                                   key, cost, newXmlCoveredList);
          totalCosts->val()->set(key,-3.40282e+38f);
        }
      }
    }

        
    // divide up subbeams by XML coverage
    std::cerr << "REGULAR BEAM EXPANSION\n";
    for(int subbeam = 0; subbeam < subbeamCount; subbeam++) {
      std::vector<char> hypMask;
      std::vector<int> subbeamSize(beams.size(),0);
      for(int j = 0; j < beams.size(); j++) {
        auto& beam = beams[j];
        std::cerr << "beam " << j << ", subbeam " << subbeam << ": ";
        for(int i = 0; i < beam.size(); ++i) {
          auto hyp = beam[i];
          if (hyp->GetXmlStatus() == subbeam) {
            hypMask.push_back( 1 );
            subbeamSize[j]++;
            std::cerr << "1";
          }
          else {
            hypMask.push_back( 0 );
            std::cerr << "0";
          }
          //hypMask.push_back( i%2==subbeam ? 1 : 0 );
          //if (i%2==subbeam) subbeamSize[j]++;
        }
        // do not expand filler hyps
        for(int i = beam.size(); i < localBeamSize; ++i) {
          hypMask.push_back( 0 );
            std::cerr << "-";
        }
        std::cerr << "\n";
      }
      std::vector<unsigned> subKeys;
      std::vector<float> subCosts;

      // find n-best predictions
      std::cerr << "nth->setHypMask\n";
      nth->setHypMask(hypMask, dimTrgVoc);
      std::vector<size_t> beamSizes(dimBatch, localBeamSize);
      nth->getNBestList(beamSizes, totalCosts->val(), subCosts, subKeys, first);
      // merge them into the subbeam list
      for(size_t i=0; i<subCosts.size(); i++) {
        if (subCosts[i] > -9999) {
          // update xmlCoveredList
          int embIdx = subKeys[i] % dimTrgVoc;
          int hypIdx = (subKeys[i] / dimTrgVoc) % localBeamSize;
          int beamNo = i / localBeamSize;
          auto& beam = beams[beamNo];
          auto hyp = beam[hypIdx];
          auto xmlCoveredList = hyp->GetXmlOptionCovered();
          int newSubbeam = subbeam;
          std::cerr << "collectedKeys.size() = " << collectedKeys.size() << "\n";
          std::cerr << "collectedKeys[beamNo].size() = " << collectedKeys[beamNo].size() << "\n";
          std::cerr << "collectedKeys[beamNo][subbeam].size() = " << collectedKeys[beamNo][subbeam].size() << "\n";
          std::cerr << "collectedCosts[" << beamNo << "][" << subbeam << "].size() = " << collectedCosts[beamNo][subbeam].size() << "\n";
          std::cerr << "build new XmlOptionCoveredList for beam " << beamNo << " hyp " << hypIdx << "\tcost " << subCosts[i] << "\t " << (*targetVocab)[embIdx] << ":" << embIdx << " ...";
          auto newXmlCoveredList = new data::XmlOptionCoveredList;
          for(int k=0; k < xmlCoveredList->size(); k++) {
            data::XmlOptionCovered newXmlCovered = xmlCoveredList->at(k);//copy
            std::cerr << "option " << k << "\n";
            if (newXmlCovered.GetStarted()) {
              const data::XmlOption* xmlOption = newXmlCovered.GetOption();
              const Words& output = xmlOption->GetOutput();
              size_t wordPos = newXmlCovered.GetPosition();
              std::cerr << "next word at position " << wordPos << " is " << output[wordPos] << ", while we predict " << embIdx << "\n";
              if (output[wordPos] == embIdx) {
                std::cerr << "newXmlCovered.Proceed();\n";
                newXmlCovered.Proceed();
                auto alignments = scorers_[0]->getAlignment();
                float weight = options_->get<float>("xml-alignment-weight");
                if (!alignments.empty() && weight != 0.0) {
                  auto align = getHardAlignmentsForHypothesis(alignments, batch, localBeamSize, hypIdx, beamNo);
                  std::cerr << "ALIGN2: align.size() = " << align.size() << "\n";
                  subCosts[i] -= weight * newXmlCovered.GetAlignmentCost();
                  newXmlCovered.AddAlignmentCost( align );
                  subCosts[i] += weight * newXmlCovered.GetAlignmentCost();
                }
              }
              else {
                std::cerr << "newXmlCovered.Abandon();\n";
                float weight = options_->get<float>("xml-alignment-weight");
                subCosts[i] -= weight * newXmlCovered.GetAlignmentCost();
                newXmlCovered.Abandon();
                newSubbeam--;
              }
            }
            else {
              std::cerr << "just copy\n";
            }
            newXmlCoveredList->push_back(newXmlCovered);
          }
          mergeIntoSortedKeysCosts(collectedKeys[beamNo][newSubbeam], 
                                   collectedCosts[beamNo][newSubbeam],
                                   collectedXmls[beamNo][newSubbeam],
                                   subKeys[i], 
                                   subCosts[i],
                                   newXmlCoveredList);
        }
      }

      std::cerr << "SUBBEAM " << subbeam << " COST/KEY\n";
      for(size_t i=0; i<subCosts.size(); i++) {
        int embIdx = subKeys[i] % dimTrgVoc;
        int beamNo = i / localBeamSize;
        int hypInBeam = i % localBeamSize;
        int hypIdx = (subKeys[i] / dimTrgVoc) % localBeamSize;
        auto& beam = beams[beamNo];
        if (beam.size() == 0) continue;
        if (subbeamSize[beamNo] == 0) continue;
        if (subCosts[i] < -9999) continue; // junk hypothesis extension
        std::cerr << "beam " << beamNo << " hyp " << hypIdx << ">" << hypInBeam << "\tcost " << subCosts[i] << "\t " << (*targetVocab)[embIdx] << ":" << embIdx << " ...";
        auto hyp = beam[hypIdx];
        std::cerr << "[" << hyp->GetPrevStateIndex() << "] ";
        while (hyp->GetWord() != 0) {
          std::cerr << " " << (*targetVocab)[hyp->GetWord()];
          hyp = hyp->GetPrevHyp();
        }
        std::cerr << std::endl;
      }
    }

    // merge subbeams
    for(int j = 0; j < beams.size(); j++) {
      std::vector<int> allotted;
      size_t thisSubbeamCount = xmlCount[j]+1;
      // by default each subbeam gets same number of hypothesis
      int totalAllotted = 0;
      std::cerr << "allotted:";
      for(int subbeam=0;subbeam<thisSubbeamCount;subbeam++) {
        int thisAllotted=(subbeam+1)*beams[j].size()/thisSubbeamCount-totalAllotted;
        allotted.push_back( thisAllotted );
        totalAllotted += thisAllotted;
        std::cerr << " " << thisAllotted << "/" << collectedCosts[j][subbeam].size();
      }
      std::cerr << "\n";

      // if there are not enough hypotheses in the subbeam,
      // redistribute its alottment to neighboring subbeams
      for(int subbeam=0;subbeam<thisSubbeamCount;subbeam++) {
        int toBeRedistributed = 
              allotted[subbeam] - collectedCosts[j][subbeam].size();
        std::cerr << " toBeRedistributed=" << toBeRedistributed;
        for(int n=1; n<thisSubbeamCount && toBeRedistributed>=0; n++) {
          for(int sign = 1; sign >= -1 && toBeRedistributed>=0; sign -= 2) {
            int neighbor = subbeam + n*sign;
            std::cerr << " neighbor=" << neighbor;
            if (neighbor >= 0 &&
                neighbor < thisSubbeamCount) {
              int space=collectedCosts[j][neighbor].size()-allotted[neighbor];
              std::cerr << " space=" << space;
              if (space > 0) {
                int redistribute = toBeRedistributed;
                if (redistribute > space) {
                  redistribute = space;
                }
                std::cerr << " redistribute=" << redistribute;
                allotted[neighbor] += redistribute;
                allotted[subbeam]  -= redistribute;
                toBeRedistributed  -= redistribute;
              }
            }
          }
        }
        std::cerr << " " << allotted[subbeam] << "/" << collectedCosts[j][subbeam].size();
      }
      std::cerr << "redistributed:";
      for(int subbeam=0;subbeam<thisSubbeamCount;subbeam++) {
        std::cerr << " " << allotted[subbeam] << "/" << collectedCosts[j][subbeam].size();
      }
      std::cerr << "\n";

      // merge the hypotheses (sorted by probability)
      std::vector<int> index(thisSubbeamCount,0);
      size_t hypCount = beams[j].size();
      for(int i=0; i<hypCount; i++) {
        float bestCost = -9e9;
        int bestSubbeam = -1;
        int bestIndex = -1;
        // find the best hypothesis across all subbeams
        for(int subbeam=0;subbeam<thisSubbeamCount;subbeam++) {
          if (index[subbeam] < allotted[subbeam] &&
              collectedCosts[j][subbeam][ index[subbeam] ] > bestCost) {
            bestCost = collectedCosts[j][subbeam][ index[subbeam] ];
            bestIndex = index[subbeam];
            bestSubbeam = subbeam;
            // if sentence complete, but not highest subbeam
            // make space for one additional hypothesis
            //int word = collectedKeys[j][bestSubbeam][bestIndex] % dimTrgVoc;
            //if (word == 0 && subbeam < thisSubbeamCount-1) {
            //  for(int s=thisSubbeamCount-1;s>=0;s--) {
            //    if(allotted[s] < collectedCosts[j][s].size()) {
            //      allotted[s]++;
            //      hypCount++;
            //      break;
            //    }
            //  }
            //}
          }
        }
        if (bestIndex >= 0) {
          int word = collectedKeys[j][bestSubbeam][bestIndex] % dimTrgVoc;
          std::cerr << "merge beam " << j << " from subbeam " << bestSubbeam << ", hyp " << bestIndex << ": " << (*targetVocab)[word] << ":" << word << "," << bestCost << "\n";
          outCosts.push_back( bestCost );
          outKeys.push_back( collectedKeys[j][bestSubbeam][bestIndex] );
          outXmls.push_back( collectedXmls[j][bestSubbeam][bestIndex] );
          index[bestSubbeam]++;
        }
        else { // not sure if this ever happens
          outCosts.push_back( -9e9 );
          outKeys.push_back( 0 );
          outXmls.push_back( NULL );
        }
      }
      // add filler keys/costs values if needed
      for(int i=beams[j].size(); i<localBeamSize; i++) {
        outCosts.push_back( -9e9 );
        outKeys.push_back( 0 );
        outXmls.push_back( NULL );
      }
    }
    std::cerr << "outCosts.size() = " << outCosts.size() << "\n";
  }
};
}
