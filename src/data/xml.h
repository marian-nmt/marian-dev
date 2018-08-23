#pragma once

#include "data/vocab.h"

namespace marian {
namespace data {

/**
 * @brief data structure to support specification of translation
 * constraints for decoding
 */

class XmlOption {
  private:
    size_t start_;
    size_t end_;

  public:
    XmlOption(size_t start, size_t end, Words output)
      : start_(start),
        end_(end),
       output_(output) {
    }

    size_t GetStart() const {
      return start_;
    }

    size_t GetEnd() const {
      return end_;
    }

    const Words& GetOutput() const {
      return output_;
    }
    Words output_;
};

class XmlOptionCovered {
  private:
    const XmlOption *option_;
    size_t position_;
    bool started_;
    bool covered_;

  public:
    XmlOptionCovered(const XmlOption option)
      : started_(false),
        covered_(false),
        option_(&option) {
    }

    bool GetStarted() const {
      return started_;
    }

    bool GetCovered() const {
      return covered_;
    }

    bool GetPosition() const {
      return position_;
    }

    const XmlOption* GetOption() const {
      return option_;
    }

    void Start() {
      started_ = true;
      position_ = 1;
      if (option_->GetOutput().size() == 1) {
        covered_ = true;
      }
      // std::cerr << "option" << option_->GetOutput().size();
    }
    void Proceed() {
      position_++;
      if (option_->GetOutput().size() == position_) {
        covered_ = true;
      }
    }
};

typedef std::vector< Ptr<XmlOption>  > XmlOptions;
typedef std::vector< const XmlOptions* > XmlOptionsList;

// end data / marian
}
}
