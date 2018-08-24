#pragma once

#include <vector>

#include "common/definitions.h"
#include "data/xml.h"

namespace marian {
namespace data {

class Batch {
public:
  virtual size_t size() const = 0;
  virtual size_t words(int which = 0) const { return 0; };
  virtual size_t width() const { return 0; };

  virtual size_t sizeTrg() const { return 0; };
  virtual size_t wordsTrg() const { return 0; };
  virtual size_t widthTrg() const { return 0; };

  virtual void debug(){};

  virtual std::vector<Ptr<Batch>> split(size_t n) = 0;

  const std::vector<size_t>& getSentenceIds() const { return sentenceIds_; }
  void setSentenceIds(const std::vector<size_t>& ids) { sentenceIds_ = ids; }

  const XmlOptionsList* getXmlOptionsList() const { return xmlOptionsList_; }
  void setXmlOptionsList(XmlOptionsList *xopsl) { 
    xmlOptionsList_ = xopsl; 
    std::cerr << "setXmlOptionsList " << xopsl << "\n"; 
    std::cerr << "xopsl->size() = " << xopsl->size() << ", " << (*xopsl)[0] << "\n"; 
    const XmlOptions *xops = (*xopsl)[0];
    std::cerr << "xops->size() = " << xops->size() << ", " << (*xops)[0] << "\n"; 
  }

  virtual void setGuidedAlignment(const std::vector<float>&) = 0;
  virtual void setDataWeights(const std::vector<float>&) = 0;

protected:
  std::vector<size_t> sentenceIds_;
  XmlOptionsList *xmlOptionsList_;
};
}
}
