#include "data/text_input.h"
#include "common/utils.h"

namespace marian {
namespace data {

TextIterator::TextIterator() : pos_(-1), tup_(0) {}
TextIterator::TextIterator(TextInput& corpus) : corpus_(&corpus), pos_(0), tup_(corpus_->next()) {}

void TextIterator::increment() {
  tup_ = corpus_->next();
  pos_++;
}

bool TextIterator::equal(TextIterator const& other) const {
  bool ended = ( this->tup_.valid() && other.tup_.valid() && this->tup_.empty() && other.tup_.empty());
  return ended || this->pos_ == other.pos_  || (!this->tup_.valid() && !other.tup_.valid());
}

const SentenceTuple& TextIterator::dereference() const {
  return tup_;
}

TextInput::TextInput(std::vector<std::string> inputs,
                     std::vector<Ptr<Vocab>> vocabs,
                     Ptr<Options> options)
    : DatasetBase(inputs, options),
      vocabs_(vocabs),
      maxLength_(options_->get<size_t>("max-length")),
      maxLengthCrop_(options_->get<bool>("max-length-crop")) {
  // Note: inputs are automatically stored in the inherited variable named paths_, but these are
  // texts not paths!
  for(const auto& text : paths_)
    files_.emplace_back(new std::istringstream(text));
}

// TextInput is mainly used for inference in the server mode, not for training, so skipping too long
// or ill-formed inputs is not necessary here
SentenceTuple TextInput::next() {
  // get index of the current sentence
  size_t curId = pos_++;
  // read next row, i.e. vector<string> from files
  // if any file is empty, we are done
  std::vector<std::string> row;
  for(size_t i = 0; i < files_.size(); ++i) {
    std::string line;
    if(io::getline(*files_[i], line)) {
      row.push_back(line);
    } else {
      return SentenceTupleImpl(); // return an empty tuple if above test does not pass();
    }
  }
  return encode(row, curId);
}

}  // namespace data
}  // namespace marian
