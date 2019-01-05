#pragma once

#include "data/vocab.h"

namespace marian {
namespace data {

class SentenceTuple;

namespace xml {

// Helper function parsing a line with XML tags, stripping them out and adding XML Options to the
// sentence tuple.
void processXml(const std::string& line,
                std::string& stripped_line,
                const Ptr<Vocab> target_vocab,
                SentenceTuple& tup);

std::vector<std::string> tokenizeXml(const std::string& line);
std::string TrimXml(const std::string& str);
bool isXmlTag(const std::string& tag);
std::string parseXmlTagAttribute(const std::string& tag, const std::string& attributeName);

/**
 * @brief Data structure to support specification of translation constraints for decoding
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
    bool started_;
    bool covered_;
    float alignmentCost_;
    const Ptr<XmlOption> option_;
    size_t position_;

  public:
    XmlOptionCovered(const Ptr<XmlOption> option)
      : started_(false),
        covered_(false),
        alignmentCost_(0.0),
        option_(option) {
      const Words &output = option->GetOutput();
      std::cerr << "created XmlOptionCovered from option " << option << ": " << option->GetStart() << "-" << option->GetEnd() << ", output length " << output.size() << "\n";
    }

    XmlOptionCovered(const XmlOptionCovered &covered)
      : started_(covered.GetStarted()),
        covered_(covered.GetCovered()),
        alignmentCost_(covered.GetAlignmentCost()),
        option_(covered.GetOption()),
        position_(covered.GetPosition()) {
      const Ptr<XmlOption> option = covered.GetOption();
      const Words &output = option->GetOutput();
      std::cerr << "created XmlOptionCovered from covered " << option->GetStart() << "-" << option->GetEnd() << ", output length " << output.size() << "\n";
    }

    bool GetStarted() const {
      return started_;
    }

    bool GetCovered() const {
      return covered_;
    }

    float GetAlignmentCost() const {
      return alignmentCost_;
    }

    size_t GetPosition() const {
      return position_;
    }

    const Ptr<XmlOption> GetOption() const {
      return option_;
    }

    void Start() {
      position_ = 1;
      if (option_->GetOutput().size() == 1) {
        // single word, already done
        covered_ = true;
        started_ = false;
      }
      else {
        started_ = true;
      }
      alignmentCost_ = 0.0;
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
      alignmentCost_ = 0.0;
    }

    void AddAlignmentCost(const std::vector<float> &alignments) {
      float sum = 0;
      std::cerr << "alignment cost for span " << option_->GetStart() << "-" << option_->GetEnd() << ":";
      for(size_t i=option_->GetStart(); i<option_->GetEnd(); i++) {
        std::cerr << " " << alignments[i];
        sum += alignments[i];
      }
      if (sum < 0.001) sum = 0.001; // floor
      alignmentCost_ += std::log(sum);
      std::cerr << " --log--> " << std::log(sum) << ", alignmentCost_ " << alignmentCost_ << "\n";
    }
};

}  // namespace xml

using namespace xml;

typedef std::vector<Ptr<XmlOption> >  XmlOptions;
typedef std::vector<Ptr<XmlOptions> > XmlOptionsList;
typedef std::vector<XmlOptionCovered> XmlOptionCoveredList;

}  // namespace data
}  // namespace marian
