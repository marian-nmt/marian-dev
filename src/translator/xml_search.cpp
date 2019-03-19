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
      if(thisXmlCount > maxXmlCount) {
        maxXmlCount = thisXmlCount;
      }
    }
  }

  // create (empty) lists for keys, costs, and xml states
  size_t subbeamCount = maxXmlCount + 1;
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
        const data::XmlOptionCovered &xmlCovered = xmlCoveredList->at(k);
        // already handled, move on
        if(xmlCovered.getCovered()) {
          continue;
        }
        // check what word needs to be generated
        size_t wordPos = 0;
        if(xmlCovered.getStarted()) {
          wordPos = xmlCovered.getPosition();
        }
        const Ptr<data::XmlOption> xmlOption = xmlCovered.getOption();
        const Words &output = xmlOption->getOutput();
        // find out the score
        uint key = ((j * localBeamSize) + i) * dimTrgVoc + output[wordPos];
        if(first) {  // only one hyp per beam
          key = j * dimTrgVoc + output[wordPos];
        }
        float cost = totalCosts->val()->get(key);
        // subbeam to be placed into
        int subbeam = hyp->GetXmlStatus();
        // update xmlCoveredList
        auto newXmlCoveredList = New<data::XmlOptionCoveredList>();
        for(int kk = 0; kk < xmlCoveredList->size(); kk++) {
          data::XmlOptionCovered newXmlCovered = xmlCoveredList->at(kk);  // copy
          if(kk == k) {
            // the one we are currently processing
            if(newXmlCovered.getStarted()) {
              // already started
              newXmlCovered.proceed();
            } else {
              newXmlCovered.start();
              subbeam++;  // resulting hyp will be in next subbeam
            }
            auto alignments = scorers_[0]->getAlignment();
            float weight = options_->get<float>("xml-alignment-weight");
            if(!alignments.empty() && weight != 0.0) {
              // TODO: make sure that the new getAlignments() returns the same as old
              // getHardAlignments()
              auto align = getAlignmentsForHypothesis(alignments, batch, i, j);
              cost -= weight * newXmlCovered.getAlignmentCost();
              newXmlCovered.addAlignmentCost(align);
              cost += weight * newXmlCovered.getAlignmentCost();
            }
          } else {
            // other xml options
            if(newXmlCovered.getStarted()) {
              float weight = options_->get<float>("xml-alignment-weight");
              cost -= weight * newXmlCovered.getAlignmentCost();
              newXmlCovered.abandon();
              subbeam--;  // resulting hyp will be one subbeam lower
            }
          }
          newXmlCoveredList->push_back(newXmlCovered);
        }
        // subbeam = 0;
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
  for(int subbeam = 0; subbeam < subbeamCount; subbeam++) {
    std::vector<char> hypMask;
    std::vector<int> subbeamSize(beams.size(), 0);
    for(int j = 0; j < beams.size(); j++) {
      auto &beam = beams[j];
      for(int i = 0; i < beam.size(); ++i) {
        auto hyp = beam[i];
        if(hyp->GetXmlStatus() == subbeam) {
          hypMask.push_back(1);
          subbeamSize[j]++;
        } else {
          hypMask.push_back(0);
        }
        // hypMask.push_back( i%2==subbeam ? 1 : 0 );
        // if (i%2==subbeam) subbeamSize[j]++;
      }
      // do not expand filler hyps
      for(int i = beam.size(); i < localBeamSize; ++i) {
        hypMask.push_back(0);
      }
    }
    std::vector<unsigned> subKeys;
    std::vector<float> subCosts;

    // find n-best predictions
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
        auto newXmlCoveredList = New<data::XmlOptionCoveredList>();
        for(int k = 0; k < xmlCoveredList->size(); k++) {
          data::XmlOptionCovered newXmlCovered = xmlCoveredList->at(k);  // copy
          if(newXmlCovered.getStarted()) {
            const Ptr<data::XmlOption> xmlOption = newXmlCovered.getOption();
            const Words &output = xmlOption->getOutput();
            size_t wordPos = newXmlCovered.getPosition();
            if(output[wordPos] == embIdx) {
              newXmlCovered.proceed();
              auto alignments = scorers_[0]->getAlignment();
              float weight = options_->get<float>("xml-alignment-weight");
              if(!alignments.empty() && weight != 0.0) {
                // TODO: make sure that the new getAlignments() returns the same as old
                // getHardAlignments()
                auto align = getAlignmentsForHypothesis(alignments, batch, hypIdx, beamNo);
                subCosts[i] -= weight * newXmlCovered.getAlignmentCost();
                newXmlCovered.addAlignmentCost(align);
                subCosts[i] += weight * newXmlCovered.getAlignmentCost();
              }
            } else {
              float weight = options_->get<float>("xml-alignment-weight");
              subCosts[i] -= weight * newXmlCovered.getAlignmentCost();
              newXmlCovered.abandon();
              newSubbeam--;
            }
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
  }

  // merge subbeams
  for(int j = 0; j < beams.size(); j++) {
    std::vector<int> allotted;
    size_t thisSubbeamCount = xmlCount[j] + 1;
    // by default each subbeam gets same number of hypothesis
    int totalAllotted = 0;
    for(int subbeam = 0; subbeam < thisSubbeamCount; subbeam++) {
      int thisAllotted = (subbeam + 1) * beams[j].size() / thisSubbeamCount - totalAllotted;
      allotted.push_back(thisAllotted);
      totalAllotted += thisAllotted;
    }

    // if there are not enough hypotheses in the subbeam,
    // redistribute its alottment to neighboring subbeams
    for(int subbeam = 0; subbeam < thisSubbeamCount; subbeam++) {
      int toBeRedistributed = allotted[subbeam] - collectedCosts[j][subbeam].size();
      for(int n = 1; n < thisSubbeamCount && toBeRedistributed >= 0; n++) {
        for(int sign = 1; sign >= -1 && toBeRedistributed >= 0; sign -= 2) {
          int neighbor = subbeam + n * sign;
          if(neighbor >= 0 && neighbor < thisSubbeamCount) {
            int space = collectedCosts[j][neighbor].size() - allotted[neighbor];
            if(space > 0) {
              int redistribute = toBeRedistributed;
              if(redistribute > space) {
                redistribute = space;
              }
              allotted[neighbor] += redistribute;
              allotted[subbeam] -= redistribute;
              toBeRedistributed -= redistribute;
            }
          }
        }
      }
    }

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
        }
      }
      if(bestIndex >= 0) {
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
  }
}  // namespace marian
