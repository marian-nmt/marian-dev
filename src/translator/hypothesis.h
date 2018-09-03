#pragma once
#include <memory>

#include "common/definitions.h"
#include "data/xml.h"

namespace marian {

class Hypothesis {
public:
  Hypothesis() : prevHyp_(nullptr), prevIndex_(0), word_(0), cost_(0.0), xmlOptionCovered_(NULL) {}

  Hypothesis(const Ptr<Hypothesis> prevHyp,
             size_t word,
             size_t prevIndex,
             float cost)
      : prevHyp_(prevHyp), prevIndex_(prevIndex), word_(word), cost_(cost) {
    xmlOptionCovered_ = prevHyp->GetXmlOptionCovered();
  }

  Hypothesis( const data::XmlOptions *xmlOptions )
      : prevHyp_(nullptr), prevIndex_(0), word_(0), cost_(0.0) {
    // create XmlOptionCovered objects
    xmlOptionCovered_ = new data::XmlOptionCoveredList;
    std::cerr << "Hypothesis xmlOptions " << xmlOptions << "\n";
    for(size_t i=0; i<xmlOptions->size(); i++) {
      std::cerr << "Hypothesis xmlOption " << (*xmlOptions)[i] << "\n";
      data::XmlOptionCovered covered((*xmlOptions)[i]);
      xmlOptionCovered_->push_back( covered );
    }
  }

  const Ptr<Hypothesis> GetPrevHyp() const { return prevHyp_; }

  size_t GetWord() const { return word_; }

  size_t GetPrevStateIndex() const { return prevIndex_; }

  float GetCost() const { return cost_; }

  std::vector<float>& GetCostBreakdown() { return costBreakdown_; }
  std::vector<float>& GetAlignment() { return alignment_; }
  std::vector<data::XmlOptionCovered>* GetXmlOptionCovered() { return xmlOptionCovered_; }

  // how many Xml constraints already satisfied or started
  size_t GetXmlStatus() {
    if (xmlOptionCovered_ == NULL) {
      return 0;
    }
    size_t status=0;
    for(data::XmlOptionCovered &covered : *xmlOptionCovered_) {
      if (covered.GetCovered() || covered.GetStarted()) {
        status++;
      }
    }
    return status;
  }

  void SetAlignment(const std::vector<float>& align) { alignment_ = align; };

  void SetXml( data::XmlOptionCoveredList *xmlOptionCovered ) {
    xmlOptionCovered_ = xmlOptionCovered;
  }
private:
  const Ptr<Hypothesis> prevHyp_;
  const size_t prevIndex_;
  const size_t word_;
  const float cost_;
  data::XmlOptionCoveredList *xmlOptionCovered_;

  std::vector<float> costBreakdown_;
  std::vector<float> alignment_;
};

typedef std::vector<Ptr<Hypothesis>> Beam;
typedef std::vector<Beam> Beams;
typedef std::vector<size_t> Words;
typedef std::tuple<Words, Ptr<Hypothesis>, float> Result;
typedef std::vector<Result> NBestList;
}
