#include "data/adaptive_context.h"

namespace marian {
namespace data {

AdaptiveContextIterator::AdaptiveContextIterator(AdaptiveContextReader* trainSetReader)
    : trainSetReader_(trainSetReader) {
  if(trainSetReader) {
    currentSamples_ = trainSetReader_->getSamples();
  }
}

bool AdaptiveContextIterator::equal(const AdaptiveContextIterator& other) const {
  return other.trainSetReader_ == trainSetReader_;
}

const std::vector<std::string>& AdaptiveContextIterator::dereference() const {
  return currentSamples_;
}

void AdaptiveContextIterator::increment() {
  // If the previous increment has exhausted the file, we must indicate that the we've reached
  // the iterator's end
  if(trainSetReader_->eof() && trainSetReader_ != nullptr) {
    trainSetReader_ = nullptr;
    return;
  }
  // If we're at the end of the iterator and increment has been called yet another time, there's
  // a bug in the calling code
  ABORT_IF(trainSetReader_ == nullptr, "Incrementing past the end of the iterator isn't allowed");

  currentSamples_ = trainSetReader_->getSamples();
}


AdaptiveContextReader::AdaptiveContextReader(std::vector<std::string> paths) {
  for(auto& path : paths)
    files_.emplace_back(new io::InputFileStream(path));
}

AdaptiveContextIterator AdaptiveContextReader::begin() {
  return AdaptiveContextIterator(this);
}

AdaptiveContextIterator AdaptiveContextReader::end() {
  return AdaptiveContextIterator(nullptr);
}

bool AdaptiveContextReader::eof() {
  return eof_;
}

std::vector<std::string> AdaptiveContextReader::getSamples() {
  // extracted lines for source and target corpora
  std::vector<std::string> samples;
  // counters of number of lines extracted for source and target
  std::vector<size_t> counts;

  // Early exit if input files are exhausted
  if (eof_) return samples;

  for(auto const& file : files_) {
    size_t currCount = 0;
    std::string lines;
    std::string line;
    bool fileEnded = true;
    while(io::getline(*file, line)) {
      if(line.empty()) {
        fileEnded = false;
        break;
      }

      if(currCount)
        lines += "\n";
      lines += line;
      currCount += 1;
    }
    eof_ = fileEnded;

    if(!lines.empty())
      samples.emplace_back(lines);
    counts.push_back(currCount);

    // check if the same number of lines is extracted for source and target
    size_t prevCount = counts[0];
    for(size_t i = 1; i < counts.size(); ++i) {
      ABORT_IF(prevCount != counts[i],
                "An empty source or target sentence has been encountered!");
      prevCount = counts[i];
    }
  }

  return samples;
}
}  // namespace data
}  // namespace marian
