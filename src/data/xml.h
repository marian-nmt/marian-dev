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
  Words output_;

public:
  XmlOption(size_t start, size_t end, Words output) : start_(start), end_(end), output_(output) {}

  size_t getStart() const { return start_; }
  size_t getEnd() const { return end_; }

  const Words& getOutput() const { return output_; }
};

class XmlOptionCovered {
private:
  bool started_;
  bool covered_;
  float alignmentCost_;
  size_t position_;

  const Ptr<XmlOption> option_;

public:
  XmlOptionCovered(const Ptr<XmlOption> option)
      : started_(false), covered_(false), alignmentCost_(0.0f), position_(0), option_(option) {
    // XML TODO: debugs
    const Words& output = option->getOutput();
    std::cerr << "created XmlOptionCovered from option " << option << ": " << option->getStart()
              << "-" << option->getEnd() << ", output length " << output.size() << "\n";
  }

  XmlOptionCovered(const XmlOptionCovered& covered)
      : started_(covered.getStarted()),
        covered_(covered.getCovered()),
        alignmentCost_(covered.getAlignmentCost()),
        position_(covered.getPosition()),
        option_(covered.getOption()) {
    // XML TODO: debugs
    const Ptr<XmlOption> option = covered.getOption();
    const Words& output = option->getOutput();
    std::cerr << "created XmlOptionCovered from covered " << option->getStart() << "-"
              << option->getEnd() << ", output length " << output.size() << "\n";
  }

  bool getStarted() const { return started_; }
  bool getCovered() const { return covered_; }
  float getAlignmentCost() const { return alignmentCost_; }
  size_t getPosition() const { return position_; }

  const Ptr<XmlOption> getOption() const { return option_; }

  void start() {
    position_ = 1;
    if(option_->getOutput().size() == 1) {
      // single word, already done
      covered_ = true;
      started_ = false;
    } else {
      started_ = true;
    }
    alignmentCost_ = 0.0f;
  }

  void proceed() {
    position_++;
    if(option_->getOutput().size() == position_) {
      covered_ = true;
      started_ = false;
    }
  }

  void abandon() {
    started_ = false;
    alignmentCost_ = 0.0f;
  }

  void addAlignmentCost(const std::vector<float>& alignments) {
    // XML TODO: debugs
    std::cerr << "alignment cost for span " << option_->getStart() << "-" << option_->getEnd()
              << ":";
    float sum = 0.0f;
    for(size_t i = option_->getStart(); i < option_->getEnd(); i++) {
      std::cerr << " " << alignments[i];
      sum += alignments[i];
    }
    if(sum < 0.001)
      sum = 0.001;  // floor
    alignmentCost_ += std::log(sum);
    std::cerr << " --log--> " << std::log(sum) << ", alignmentCost_ " << alignmentCost_ << "\n";
  }
};

}  // namespace xml

using namespace xml;

typedef std::vector<Ptr<XmlOption> > XmlOptions;
typedef std::vector<Ptr<XmlOptions> > XmlOptionsList;
typedef std::vector<XmlOptionCovered> XmlOptionCoveredList;

}  // namespace data
}  // namespace marian
