// TODO: NthElement is not available in the newest version of BeamSearch
#include "translator/beam_search.h"

namespace marian {
void BeamSearch::xmlSearch(GetNBestListFn getNBestList,
                           Beams &beams,
                           size_t localBeamSize,
                           Expr &totalCosts,
                           std::vector<float> &outCosts,
                           std::vector<unsigned> &outKeys,
                           std::vector<Ptr<data::XmlOptionCoveredList> > &outXmls,
                           bool first,
                           Ptr<Vocab> targetVocab,
                           Ptr<data::CorpusBatch> batch) {
  int dimBatch = beams.size();
  int dimTrgVoc = totalCosts->shape()[-1];

  std::vector<std::vector<std::vector<unsigned> > > collectedKeys;
  std::vector<std::vector<std::vector<float> > > collectedCosts;
  std::vector<std::vector<std::vector<Ptr<data::XmlOptionCoveredList> > > > collectedXmls;
  // get maximum number of xml options per sentence
  // (-> plus 1 = number of subbeams)
  std::vector<size_t> xmlCount;
  size_t maxXmlCount = 0;
  for(int j = 0; j < beams.size(); j++) {
    auto &beam = beams[j];
    if(beam.size() == 0) {
      xmlCount.push_back(0);
    } else {
      auto hyp = beam[0];
      size_t thisXmlCount = hyp->GetXmlOptionCovered()->size();
      xmlCount.push_back(thisXmlCount);
      std::cerr << "beam " << j << ": " << thisXmlCount << "\n";
      if(thisXmlCount > maxXmlCount) {
        maxXmlCount = thisXmlCount;
      }
    }
  }

  // create (empty) lists for keys, costs, and xml states
  size_t subbeamCount = maxXmlCount + 1;
  std::cerr << "subbeamCount = " << subbeamCount << "\n";
  for(int j = 0; j < beams.size(); j++) {
    std::vector<std::vector<unsigned> > keysVector;
    std::vector<std::vector<float> > costsVector;
    std::vector<std::vector<Ptr<data::XmlOptionCoveredList> > > xmlVector;
    collectedKeys.push_back(keysVector);
    collectedCosts.push_back(costsVector);
    collectedXmls.push_back(xmlVector);
    for(int subbeam = 0; subbeam < subbeamCount; subbeam++) {
      std::vector<unsigned> keys;
      std::vector<float> costs;
      std::vector<Ptr<data::XmlOptionCoveredList> > xml;
      collectedKeys[j].push_back(keys);
      collectedCosts[j].push_back(costs);
      collectedXmls[j].push_back(xml);
    }
  }

  std::cerr << "ADDING ADDITIONAL HYPOTHESES\n";
  for(int j = 0; j < beams.size(); j++) {
    auto &beam = beams[j];
    // loop over all prior hypotheses
    for(int i = 0; i < beam.size(); ++i) {
      if(first && i > 0) {
        break;  // not sure why, but first has additional fake hyps
      }
      auto hyp = beam[i];
      auto xmlCoveredList = hyp->GetXmlOptionCovered();
      // check on status of each XML constraints
      for(int k = 0; k < xmlCoveredList->size(); k++) {
        std::cerr << "checking xml option " << k << "\n";
        const data::XmlOptionCovered &xmlCovered = xmlCoveredList->at(k);
        // already handled, move on
        if(xmlCovered.GetCovered()) {
          std::cerr << "already covered\n";
          continue;
        }
        // check what word needs to be generated
        size_t wordPos = 0;
        if(xmlCovered.GetStarted()) {
          wordPos = xmlCovered.GetPosition();
          std::cerr << "already started\n";
        }
        std::cerr << "proceeding at wordPos " << wordPos << "\n";
        const Ptr<data::XmlOption> xmlOption = xmlCovered.GetOption();
        const Words &output = xmlOption->GetOutput();
        std::cerr << "xmlCovered = " << xmlOption->GetStart() << "-" << xmlOption->GetEnd()
                  << ", output length " << output.size();
        for(size_t o = 0; o < output.size(); o++) {
          std::cerr << " " << (*targetVocab)[output[o]];
        }
        std::cerr << "\n";
        // find out the score
        // std::cerr << "it's in here somewhere... ";
        // std::cerr << totalCosts->val()->debug();
        // std::cerr << "\n";
        uint key = ((j * localBeamSize) + i) * dimTrgVoc + output[wordPos];
        if(first) {  // only one hyp per beam
          key = j * dimTrgVoc + output[wordPos];
        }
        float cost = totalCosts->val()->get(key);
        std::cerr << "beam j=" << j << " hyp i=" << i << ", word "
                  << (*targetVocab)[output[wordPos]] << ":" << output[wordPos]
                  << ", total cost is in " << key << ": " << cost << "\n";
        // subbeam to be placed into
        int subbeam = hyp->GetXmlStatus();
        // update xmlCoveredList
        std::cerr << "build new XmlOptionCoveredList\n";
        auto newXmlCoveredList = New<data::XmlOptionCoveredList>();
        for(int kk = 0; kk < xmlCoveredList->size(); kk++) {
          data::XmlOptionCovered newXmlCovered = xmlCoveredList->at(kk);  // copy
          std::cerr << "option " << kk << "\n";
          if(kk == k) {
            // the one we are currently processing
            if(newXmlCovered.GetStarted()) {
              // already started
              std::cerr << "newXmlCovered.Proceed();\n";
              newXmlCovered.Proceed();
            } else {
              std::cerr << "newXmlCovered.Start();\n";
              newXmlCovered.Start();
              subbeam++;  // resulting hyp will be in next subbeam
            }
            auto alignments = scorers_[0]->getAlignment();
            float weight = options_->get<float>("xml-alignment-weight");
            std::cerr << "ALIGN: alignments.size() = " << alignments.size() << "\n";
            if(!alignments.empty() && weight != 0.0) {
              // TODO: make sure that the new getAlignments() returns the same as old
              // getHardAlignments()
              auto align = getAlignmentsForHypothesis(alignments, batch, i, j);
              std::cerr << "ALIGN: align.size() = " << align.size() << "\n";
              cost -= weight * newXmlCovered.GetAlignmentCost();
              newXmlCovered.AddAlignmentCost(align);
              cost += weight * newXmlCovered.GetAlignmentCost();
            }
          } else {
            // other xml options
            if(newXmlCovered.GetStarted()) {
              std::cerr << "newXmlCovered.Abandon();\n";
              float weight = options_->get<float>("xml-alignment-weight");
              cost -= weight * newXmlCovered.GetAlignmentCost();
              newXmlCovered.Abandon();
              subbeam--;  // resulting hyp will be one subbeam lower
            } else {
              std::cerr << "just copy\n";
            }
          }
          newXmlCoveredList->push_back(newXmlCovered);
        }
        // subbeam = 0;
        std::cerr << "merge onto subbeam " << subbeam << "\n";
        mergeIntoSortedKeysCosts(collectedKeys[j][subbeam],
                                 collectedCosts[j][subbeam],
                                 collectedXmls[j][subbeam],
                                 key,
                                 cost,
                                 newXmlCoveredList);
        totalCosts->val()->set(key, -3.40282e+38f);
      }
    }
  }

  // divide up subbeams by XML coverage
  std::cerr << "REGULAR BEAM EXPANSION\n";
  for(int subbeam = 0; subbeam < subbeamCount; subbeam++) {
    std::vector<char> hypMask;
    std::vector<int> subbeamSize(beams.size(), 0);
    for(int j = 0; j < beams.size(); j++) {
      auto &beam = beams[j];
      std::cerr << "beam " << j << ", subbeam " << subbeam << ": ";
      for(int i = 0; i < beam.size(); ++i) {
        auto hyp = beam[i];
        if(hyp->GetXmlStatus() == subbeam) {
          hypMask.push_back(1);
          subbeamSize[j]++;
          std::cerr << "1";
        } else {
          hypMask.push_back(0);
          std::cerr << "0";
        }
        // hypMask.push_back( i%2==subbeam ? 1 : 0 );
        // if (i%2==subbeam) subbeamSize[j]++;
      }
      // do not expand filler hyps
      for(int i = beam.size(); i < localBeamSize; ++i) {
        hypMask.push_back(0);
        std::cerr << "-";
      }
      std::cerr << "\n";
    }
    std::vector<unsigned> subKeys;
    std::vector<float> subCosts;

    // find n-best predictions
    std::cerr << "nth->setHypMask\n";
    std::vector<size_t> beamSizes(dimBatch, localBeamSize);

    // TODO: check if it is OK
    //nth->setHypMask(hypMask, dimTrgVoc);
    //nth->getNBestList(beamSizes, totalCosts->val(), subCosts, subKeys, first);

    getNBestList(beamSizes, totalCosts->val(), subCosts, subKeys, first, hypMask, dimTrgVoc);

    // merge them into the subbeam list
    for(size_t i = 0; i < subCosts.size(); i++) {
      if(subCosts[i] > -9999) {
        // update xmlCoveredList
        int embIdx = subKeys[i] % dimTrgVoc;
        int hypIdx = (subKeys[i] / dimTrgVoc) % localBeamSize;
        int beamNo = i / localBeamSize;
        auto &beam = beams[beamNo];
        auto hyp = beam[hypIdx];
        auto xmlCoveredList = hyp->GetXmlOptionCovered();
        int newSubbeam = subbeam;
        std::cerr << "collectedKeys.size() = " << collectedKeys.size() << "\n";
        std::cerr << "collectedKeys[beamNo].size() = " << collectedKeys[beamNo].size() << "\n";
        std::cerr << "collectedKeys[beamNo][subbeam].size() = "
                  << collectedKeys[beamNo][subbeam].size() << "\n";
        std::cerr << "collectedCosts[" << beamNo << "][" << subbeam
                  << "].size() = " << collectedCosts[beamNo][subbeam].size() << "\n";
        std::cerr << "build new XmlOptionCoveredList for beam " << beamNo << " hyp " << hypIdx
                  << "\tcost " << subCosts[i] << "\t " << (*targetVocab)[embIdx] << ":" << embIdx
                  << " ...";
        auto newXmlCoveredList = New<data::XmlOptionCoveredList>();
        for(int k = 0; k < xmlCoveredList->size(); k++) {
          data::XmlOptionCovered newXmlCovered = xmlCoveredList->at(k);  // copy
          std::cerr << "option " << k << "\n";
          if(newXmlCovered.GetStarted()) {
            const Ptr<data::XmlOption> xmlOption = newXmlCovered.GetOption();
            const Words &output = xmlOption->GetOutput();
            size_t wordPos = newXmlCovered.GetPosition();
            std::cerr << "next word at position " << wordPos << " is " << output[wordPos]
                      << ", while we predict " << embIdx << "\n";
            if(output[wordPos] == embIdx) {
              std::cerr << "newXmlCovered.Proceed();\n";
              newXmlCovered.Proceed();
              auto alignments = scorers_[0]->getAlignment();
              float weight = options_->get<float>("xml-alignment-weight");
              if(!alignments.empty() && weight != 0.0) {
                // TODO: make sure that the new getAlignments() returns the same as old
                // getHardAlignments()
                auto align = getAlignmentsForHypothesis(alignments, batch, hypIdx, beamNo);
                std::cerr << "ALIGN2: align.size() = " << align.size() << "\n";
                subCosts[i] -= weight * newXmlCovered.GetAlignmentCost();
                newXmlCovered.AddAlignmentCost(align);
                subCosts[i] += weight * newXmlCovered.GetAlignmentCost();
              }
            } else {
              std::cerr << "newXmlCovered.Abandon();\n";
              float weight = options_->get<float>("xml-alignment-weight");
              subCosts[i] -= weight * newXmlCovered.GetAlignmentCost();
              newXmlCovered.Abandon();
              newSubbeam--;
            }
          } else {
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
    for(size_t i = 0; i < subCosts.size(); i++) {
      int embIdx = subKeys[i] % dimTrgVoc;
      int beamNo = i / localBeamSize;
      int hypInBeam = i % localBeamSize;
      int hypIdx = (subKeys[i] / dimTrgVoc) % localBeamSize;
      auto &beam = beams[beamNo];
      if(beam.size() == 0)
        continue;
      if(subbeamSize[beamNo] == 0)
        continue;
      if(subCosts[i] < -9999)
        continue;  // junk hypothesis extension
      std::cerr << "beam " << beamNo << " hyp " << hypIdx << ">" << hypInBeam << "\tcost "
                << subCosts[i] << "\t " << (*targetVocab)[embIdx] << ":" << embIdx << " ...";
      auto hyp = beam[hypIdx];
      std::cerr << "[" << hyp->GetPrevStateIndex() << "] ";
      while(hyp->GetWord() != 0) {
        std::cerr << " " << (*targetVocab)[hyp->GetWord()];
        hyp = hyp->GetPrevHyp();
      }
      std::cerr << std::endl;
    }
  }

  // merge subbeams
  for(int j = 0; j < beams.size(); j++) {
    std::vector<int> allotted;
    size_t thisSubbeamCount = xmlCount[j] + 1;
    // by default each subbeam gets same number of hypothesis
    int totalAllotted = 0;
    std::cerr << "allotted:";
    for(int subbeam = 0; subbeam < thisSubbeamCount; subbeam++) {
      int thisAllotted = (subbeam + 1) * beams[j].size() / thisSubbeamCount - totalAllotted;
      allotted.push_back(thisAllotted);
      totalAllotted += thisAllotted;
      std::cerr << " " << thisAllotted << "/" << collectedCosts[j][subbeam].size();
    }
    std::cerr << "\n";

    // if there are not enough hypotheses in the subbeam,
    // redistribute its alottment to neighboring subbeams
    for(int subbeam = 0; subbeam < thisSubbeamCount; subbeam++) {
      int toBeRedistributed = allotted[subbeam] - collectedCosts[j][subbeam].size();
      std::cerr << " toBeRedistributed=" << toBeRedistributed;
      for(int n = 1; n < thisSubbeamCount && toBeRedistributed >= 0; n++) {
        for(int sign = 1; sign >= -1 && toBeRedistributed >= 0; sign -= 2) {
          int neighbor = subbeam + n * sign;
          std::cerr << " neighbor=" << neighbor;
          if(neighbor >= 0 && neighbor < thisSubbeamCount) {
            int space = collectedCosts[j][neighbor].size() - allotted[neighbor];
            std::cerr << " space=" << space;
            if(space > 0) {
              int redistribute = toBeRedistributed;
              if(redistribute > space) {
                redistribute = space;
              }
              std::cerr << " redistribute=" << redistribute;
              allotted[neighbor] += redistribute;
              allotted[subbeam] -= redistribute;
              toBeRedistributed -= redistribute;
            }
          }
        }
      }
      std::cerr << " " << allotted[subbeam] << "/" << collectedCosts[j][subbeam].size();
    }
    std::cerr << "redistributed:";
    for(int subbeam = 0; subbeam < thisSubbeamCount; subbeam++) {
      std::cerr << " " << allotted[subbeam] << "/" << collectedCosts[j][subbeam].size();
    }
    std::cerr << "\n";

    // merge the hypotheses (sorted by probability)
    std::vector<int> index(thisSubbeamCount, 0);
    size_t hypCount = beams[j].size();
    for(int i = 0; i < hypCount; i++) {
      float bestCost = -9e9;
      int bestSubbeam = -1;
      int bestIndex = -1;
      // find the best hypothesis across all subbeams
      for(int subbeam = 0; subbeam < thisSubbeamCount; subbeam++) {
        if(index[subbeam] < allotted[subbeam]
           && collectedCosts[j][subbeam][index[subbeam]] > bestCost) {
          bestCost = collectedCosts[j][subbeam][index[subbeam]];
          bestIndex = index[subbeam];
          bestSubbeam = subbeam;
          // if sentence complete, but not highest subbeam
          // make space for one additional hypothesis
          // int word = collectedKeys[j][bestSubbeam][bestIndex] % dimTrgVoc;
          // if (word == 0 && subbeam < thisSubbeamCount-1) {
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
      if(bestIndex >= 0) {
        int word = collectedKeys[j][bestSubbeam][bestIndex] % dimTrgVoc;
        std::cerr << "merge beam " << j << " from subbeam " << bestSubbeam << ", hyp " << bestIndex
                  << ": " << (*targetVocab)[word] << ":" << word << "," << bestCost << "\n";
        outCosts.push_back(bestCost);
        outKeys.push_back(collectedKeys[j][bestSubbeam][bestIndex]);
        outXmls.push_back(collectedXmls[j][bestSubbeam][bestIndex]);
        index[bestSubbeam]++;
      } else {  // not sure if this ever happens
        outCosts.push_back(-9e9);
        outKeys.push_back(0);
        outXmls.push_back(NULL);
      }
    }
    // add filler keys/costs values if needed
    for(int i = beams[j].size(); i < localBeamSize; i++) {
      outCosts.push_back(-9e9);
      outKeys.push_back(0);
      outXmls.push_back(NULL);
    }
  }
  std::cerr << "outCosts.size() = " << outCosts.size() << "\n";
  }
}  // namespace marian
