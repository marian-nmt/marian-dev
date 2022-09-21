#pragma once

#include "common/file_stream.h"
#include "data/iterator_facade.h"

namespace marian {
namespace data {


class AdaptiveContextReader;


/**
 * An iterator for easier access of the context sentences produced by
 * `AdaptiveContextReader::getSamples()`
 */
class AdaptiveContextIterator
  : public IteratorFacade<AdaptiveContextIterator, std::vector<std::string>> {

  AdaptiveContextReader* trainSetReader_;
  std::vector<std::string> currentSamples_;

public:
  // TODO: should we use a smart pointer here instead? The TrainSetReader::begin() method
  // would make it difficult
  AdaptiveContextIterator(AdaptiveContextReader* trainSetReader);

  bool equal(const AdaptiveContextIterator& other) const override;

  const std::vector<std::string>& dereference() const override;

  void increment() override;
};


/**
 * Reads the context sentences, that are used for on-the-fly training in
 * the self-adaptive translation mode, from files.
 */
class AdaptiveContextReader {

  std::vector<UPtr<io::InputFileStream>> files_;
  /// Indicates whether the input files have been exhausted.
  bool eof_ = false;

public:
  /**
   * Initializes a new reader by supplying paths to the files with
   * context sentences
   *
   * @param paths paths to the input files. The input files contain
   * newline-separated parallel sentence pairs (as usual for MT). Sentences are
   * grouped by the translatable sentences (which are provided in a different
   * file). Each group is delimited by a single empty line. The sentence group
   * can be empty (no context is provided for the respective translatable
   * sentence) in which case it is also represented by a single empty line.
   */
  AdaptiveContextReader(std::vector<std::string> paths);

  /**
   * Returns an iterator over the sets of context sentences produced by
   * `getSamples()`
   *
   * @return the beginning of the iterator.
   */
  AdaptiveContextIterator begin();

  AdaptiveContextIterator end();

  bool eof();

  /**
   * Reads the next set of samples -- the contaxt sentences -- for
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
   * element in the vector contains newline separated input lines comming from a
   * single file, e.g., [0] could contain 3 newline separated sentences in
   * English and [1] would contain their 3 respective translations in Latvian.
   */
  std::vector<std::string> getSamples();
};


}  // namespace data
}  // namespace marian
