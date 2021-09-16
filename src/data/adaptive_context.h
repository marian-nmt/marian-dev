#pragma once

#include "common/file_stream.h"
#include "data/iterator_facade.h"

namespace marian {

class AdaptiveContextReader;

/**
 * @brief An iterator for easier access of the context sentences produced by
 * `AdaptiveContextReader::getSamples()`
 */
class AdaptiveContextIterator
  : public IteratorFacade<AdaptiveContextIterator, std::vector<std::string>> {
private:
  AdaptiveContextReader* trainSetReader_;
  std::vector<std::string> currentSamples_;
public:
  // TODO: should we use a smart pointer here instead? The TrainSetReader::begin() method
  // would make it difficult
  AdaptiveContextIterator(AdaptiveContextReader* trainSetReader);

  bool equal(const AdaptiveContextIterator& other) const override {
    return other.trainSetReader_ == trainSetReader_;
  }

  const std::vector<std::string>& dereference() const override { return currentSamples_; }

  void increment() override;
};

/**
 * @brief Reads the context sentences, that are used for on-the-fly training in
 * the self-adaptive translation mode, from files.
 */
class AdaptiveContextReader {
  std::vector<UPtr<io::InputFileStream>> files_;
  /// Indicates whether the input files have been exhausted.
  bool eof_ = false;

public:
  /**
   * @brief Initializes a new reader by supplying paths to the files with
   * context sentences
   *
   * @param paths paths to the input files. The input files contain
   * newline-separated parallel sentence pairs (as usual for MT). Sentences are
   * grouped by the translatable sentences (which are provided in a different
   * file). Each group is delimited by a single empty line. The sentence group
   * can be empty (no context is provided for the respective translatable
   * sentence) in which case it is also represented by a single empty line.
   */
  AdaptiveContextReader(std::vector<std::string> paths) {
    for(auto& path : paths)
      files_.emplace_back(new io::InputFileStream(path));
  }

  /**
   * @brief Returns an iterator over the sets of context sentences produced by
   * `getSamples()`
   *
   * @return the beginning of the iterator.
   */
  AdaptiveContextIterator begin() {
    return AdaptiveContextIterator(this);
  }

  AdaptiveContextIterator end() {
    return AdaptiveContextIterator(nullptr);
  }

  bool eof() {
    return eof_;
  }

  /**
   * @brief Reads the next set of samples -- the contaxt sentences -- for
   * on-the-fly training in the self-adaptive translation mode.
   *
   * @details The input files contain newline-separated parallel sentence pairs
   * (as usual for MT). Sentences are grouped by the translatable sentences
   * (which are provided in a different file). Each group is delimited by a
   * single empty line. The sentence group can be empty (no context is provided
   * for the respective translatable sentence) in which case it is also
   * represented by a single empty line.
   *
   * @return a vector representing a single group of context sentences. Each
   * element in the vector contains newline seperated input lines comming from a
   * single file, e.g., [0] could contain 3 newline separated sentences in
   * English and [1] would contain their 3 respective translations in Latvian.
   */
  std::vector<std::string> getSamples() {
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
};

AdaptiveContextIterator::AdaptiveContextIterator(AdaptiveContextReader* trainSetReader) : trainSetReader_(trainSetReader) {
  if(trainSetReader) {
    currentSamples_ = trainSetReader_->getSamples();
  }
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
}  // namespace marian
