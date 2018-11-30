#include "data/text_input.h"

namespace marian {
namespace data {

TextIterator::TextIterator() : pos_(-1), tup_(0) {}

TextIterator::TextIterator(TextInput& corpus)
    : corpus_(&corpus), pos_(0), tup_(corpus_->next()) {}

void TextIterator::increment() {
  tup_ = corpus_->next();
  pos_++;
}

bool TextIterator::equal(TextIterator const& other) const {
  return this->pos_ == other.pos_ || (this->tup_.empty() && other.tup_.empty());
}

const SentenceTuple& TextIterator::dereference() const {
  return tup_;
}

TextInput::TextInput(std::vector<std::string> paths,
                     std::vector<Ptr<Vocab>> vocabs,
                     Ptr<Config> options)
    : DatasetBase(paths), vocabs_(vocabs), options_(options) {
  for(auto path : paths_)
    files_.emplace_back(new std::istringstream(path));
}

TextInput::TextInput(std::vector<std::string> paths,
                     std::vector<Ptr<Vocab>> vocabs,
                     Ptr<Vocab> target_vocab,
                     Ptr<Config> options)
    : DatasetBase(paths), vocabs_(vocabs), target_vocab_(target_vocab), options_(options) {
  for(auto path : paths_)
    files_.emplace_back(new std::istringstream(path));
}

SentenceTuple TextInput::next() {
  bool cont = true;
  while(cont) {
    // get index of the current sentence
    size_t curId = pos_++;

    // fill up the sentence tuple with sentences from all input files
    SentenceTuple tup(curId);
    for(size_t i = 0; i < files_.size(); ++i) {
      std::string line;
      if(std::getline(*files_[i], line)) {
        if(options_->get<bool>("xml-input")) {
          std::cerr << "process xml for " << line << std::endl;
          std::cerr << "vocabs_.size() = " << vocabs_.size() << std::endl;
          std::string stripped_line;
          processXml(line, stripped_line, target_vocab_, tup);
          Words words = (*vocabs_[i])(stripped_line);
          if(words.empty())
            words.push_back(0);
          tup.push_back(words);
        } else {
          Words words = (*vocabs_[i])(line);
          if(words.empty())
            words.push_back(0);
          tup.push_back(words);
        }
      }
    }

    // continue only if each input file has provided an example
    cont = tup.size() == files_.size();
    if(cont)
      return tup;
  }
  return SentenceTuple(0);
}
}
}
