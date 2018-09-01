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
    XmlOptionCovered(const XmlOption *option)
      : started_(false),
        covered_(false),
        option_(option) {
      const Words &output = option->GetOutput();
      std::cerr << "created XmlOptionCovered from option " << option << ": " << option->GetStart() << "-" << option->GetEnd() << ", output length " << output.size() << "\n";
    } 

    XmlOptionCovered(const XmlOptionCovered &covered)
      : started_(covered.GetStarted()),
        covered_(covered.GetCovered()),
        option_(covered.GetOption()),
        position_(covered.GetPosition()) {
      const XmlOption *option = covered.GetOption();
      const Words &output = option->GetOutput();
      std::cerr << "created XmlOptionCovered from covered " << option->GetStart() << "-" << option->GetEnd() << ", output length " << output.size() << "\n";
    }

    bool GetStarted() const {
      return started_;
    }

    bool GetCovered() const {
      return covered_;
    }

    size_t GetPosition() const {
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
        started_ = false;
      }
    }
    void Abandon() {
      started_ = false;
    }
};

typedef std::vector< XmlOption* > XmlOptions;
typedef std::vector< const XmlOptions* > XmlOptionsList;
typedef std::vector< XmlOptionCovered > XmlOptionCoveredList;

// end data / marian
}
}
