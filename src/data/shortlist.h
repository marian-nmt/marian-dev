#pragma once

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/corpus_base.h"
#include "data/types.h"
#include "3rd_party/mio/mio.hpp"

#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>

namespace marian {
namespace data {
// Magic signature for binary shortlist:
// ASCII and Unicode text files never start with the following 64 bits
const uint64_t BINARY_SHORTLIST_MS = 0xF11A48D5013417F5;

static bool isBinaryShortlist(const std::string& fileName){
  uint64_t magic;
  io::InputFileStream in(fileName);
  in.read((char*)(&magic), sizeof(magic));
  return in && (magic == BINARY_SHORTLIST_MS);
}

class Shortlist {
private:
  std::vector<WordIndex> indices_;    // // [packed shortlist index] -> word index, used to select columns from output embeddings

public:
  Shortlist(const std::vector<WordIndex>& indices)
    : indices_(indices) {}

  const std::vector<WordIndex>& indices() const { return indices_; }
  WordIndex reverseMap(int idx) { return indices_[idx]; }

  int tryForwardMap(WordIndex wIdx) {
    auto first = std::lower_bound(indices_.begin(), indices_.end(), wIdx);
    if(first != indices_.end() && *first == wIdx)         // check if element not less than wIdx has been found and if equal to wIdx
      return (int)std::distance(indices_.begin(), first); // return coordinate if found
    else
      return -1;                                          // return -1 if not found
  }

};

class ShortlistGenerator {
public:
  virtual ~ShortlistGenerator() {}

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) const = 0;

  // Writes text version of (possibly) pruned short list to file
  // with given prefix and implementation-specific suffixes.
  virtual void dump(const std::string& /*prefix*/) const {
    ABORT("Not implemented");
  }
};


// Intended for use during training in the future, currently disabled
#if 0
class SampledShortlistGenerator : public ShortlistGenerator {
private:
  Ptr<Options> options_;
  size_t maxVocab_{50000};

  size_t total_{10000};
  size_t firstNum_{1000};

  size_t srcIdx_;
  size_t trgIdx_;
  bool shared_{false};

  // static thread_local std::random_device rd_;
  static thread_local std::unique_ptr<std::mt19937> gen_;

public:
  SampledShortlistGenerator(Ptr<Options> options,
                            size_t srcIdx = 0,
                            size_t trgIdx = 1,
                            bool shared = false)
      : options_(options),
        srcIdx_(srcIdx),
        trgIdx_(trgIdx),
        shared_(shared)
        { }

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) const override {
    auto srcBatch = (*batch)[srcIdx_];
    auto trgBatch = (*batch)[trgIdx_];

    // add firstNum most frequent words
    std::unordered_set<WordIndex> indexSet;
    for(WordIndex i = 0; i < firstNum_ && i < maxVocab_; ++i)
      indexSet.insert(i);

    // add all words from ground truth
    for(auto i : trgBatch->data())
      indexSet.insert(i.toWordIndex());

    // add all words from source
    if(shared_)
      for(auto i : srcBatch->data())
        indexSet.insert(i.toWordIndex());

    std::uniform_int_distribution<> dis((int)firstNum_, (int)maxVocab_);
    if (gen_ == NULL)
      gen_.reset(new std::mt19937(std::random_device{}()));
    while(indexSet.size() < total_ && indexSet.size() < maxVocab_)
      indexSet.insert(dis(*gen_));

    // turn into vector and sort (selected indices)
    std::vector<WordIndex> idx(indexSet.begin(), indexSet.end());
    std::sort(idx.begin(), idx.end());

    // assign new shifted position
    std::unordered_map<WordIndex, WordIndex> pos;
    std::vector<WordIndex> reverseMap;

    for(WordIndex i = 0; i < idx.size(); ++i) {
      pos[idx[i]] = i;
      reverseMap.push_back(idx[i]);
    }

    Words mapped;
    for(auto i : trgBatch->data()) {
      // mapped postions for cross-entropy
      mapped.push_back(Word::fromWordIndex(pos[i.toWordIndex()]));
    }

    return New<Shortlist>(idx, mapped, reverseMap);
  }
};
#endif

class LexicalShortlistGenerator : public ShortlistGenerator {
private:
  Ptr<Options> options_;
  Ptr<const Vocab> srcVocab_;
  Ptr<const Vocab> trgVocab_;

  size_t srcIdx_;
  bool shared_{false};

  size_t firstNum_{100};
  size_t bestNum_{100};

  std::vector<std::unordered_map<WordIndex, float>> data_; // [WordIndex src] -> [WordIndex tgt] -> P_trans(tgt|src) --@TODO: rename data_ accordingly

  void load(const std::string& fname) {
    io::InputFileStream in(fname);

    std::string src, trg;
    float prob;
    while(in >> trg >> src >> prob) {
      // @TODO: change this to something safer other than NULL
      if(src == "NULL" || trg == "NULL")
        continue;

      auto sId = (*srcVocab_)[src].toWordIndex();
      auto tId = (*trgVocab_)[trg].toWordIndex();

      if(data_.size() <= sId)
        data_.resize(sId + 1);
      data_[sId][tId] = prob;
    }
  }

  void prune(float threshold = 0.f) {
    size_t i = 0;
    for(auto& probs : data_) {
      std::vector<std::pair<float, WordIndex>> sorter;
      for(auto& it : probs)
        sorter.emplace_back(it.second, it.first);

      std::sort(
          sorter.begin(), sorter.end(), std::greater<std::pair<float, WordIndex>>()); // sort by prob

      probs.clear();
      for(auto& it : sorter) {
        if(probs.size() < bestNum_ && it.first > threshold)
          probs[it.second] = it.first;
        else
          break;
      }

      ++i;
    }
  }

public:
  LexicalShortlistGenerator(Ptr<Options> options,
                            Ptr<const Vocab> srcVocab,
                            Ptr<const Vocab> trgVocab,
                            size_t srcIdx = 0,
                            size_t /*trgIdx*/ = 1,
                            bool shared = false)
      : options_(options),
        srcVocab_(srcVocab),
        trgVocab_(trgVocab),
        srcIdx_(srcIdx),
        shared_(shared) {
    std::vector<std::string> vals = options_->get<std::vector<std::string>>("shortlist");

    ABORT_IF(vals.empty(), "No path to filter path given");
    std::string fname = vals[0];

    firstNum_ = vals.size() > 1 ? std::stoi(vals[1]) : 100;
    bestNum_ = vals.size() > 2 ? std::stoi(vals[2]) : 100;
    float threshold = vals.size() > 3 ? std::stof(vals[3]) : 0;
    std::string dumpPath = vals.size() > 4 ? vals[4] : "";
    LOG(info,
        "[data] Loading lexical shortlist as {} {} {} {}",
        fname,
        firstNum_,
        bestNum_,
        threshold);

    // @TODO: Load and prune in one go.
    load(fname);
    prune(threshold);

    if(!dumpPath.empty())
      dump(dumpPath);
  }

  virtual void dump(const std::string& prefix) const override {
    // Dump top most frequent words from target vocabulary
    LOG(info, "[data] Saving shortlist dump to {}", prefix + ".{top,dic}");
    io::OutputFileStream outTop(prefix + ".top");
    for(WordIndex i = 0; i < firstNum_ && i < trgVocab_->size(); ++i)
      outTop << (*trgVocab_)[Word::fromWordIndex(i)] << std::endl;

    // Dump translation pairs from dictionary
    io::OutputFileStream outDic(prefix + ".dic");
    for(WordIndex srcId = 0; srcId < data_.size(); srcId++) {
      for(auto& it : data_[srcId]) {
        auto trgId = it.first;
        outDic << (*srcVocab_)[Word::fromWordIndex(srcId)] << "\t" << (*trgVocab_)[Word::fromWordIndex(trgId)] << std::endl;
      }
    }
  }

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) const override {
    auto srcBatch = (*batch)[srcIdx_];

    // add firstNum most frequent words
    std::unordered_set<WordIndex> indexSet;
    for(WordIndex i = 0; i < firstNum_ && i < trgVocab_->size(); ++i)
      indexSet.insert(i);

    // add all words from ground truth
    // for(auto i : trgBatch->data())
    //  indexSet.insert(i.toWordIndex());

    // collect unique words form source
    std::unordered_set<WordIndex> srcSet;
    for(auto i : srcBatch->data())
      srcSet.insert(i.toWordIndex());

    // add aligned target words
    for(auto i : srcSet) {
      if(shared_)
        indexSet.insert(i);
      for(auto& it : data_[i])
        indexSet.insert(it.first);
    }
    // Ensure that the generated vocabulary items from a shortlist are a multiple-of-eight
    // This is necessary until intgemm supports non-multiple-of-eight matrices.
    // TODO better solution here? This could potentially be slow.
    WordIndex i = static_cast<WordIndex>(firstNum_);
    while (indexSet.size() % 8 != 0) {
      indexSet.insert(i);
      i++;
    }

    // turn into vector and sort (selected indices)
    std::vector<WordIndex> indices(indexSet.begin(), indexSet.end());
    std::sort(indices.begin(), indices.end());

    return New<Shortlist>(indices);
  }
};

class BinaryShortlistGenerator : public ShortlistGenerator {
private:
  Ptr<Options> options_;
  Ptr<const Vocab> srcVocab_;
  Ptr<const Vocab> trgVocab_;

  size_t srcIdx_;
  bool shared_{false};

  uint64_t firstNum_{100};  // baked into binary header
  uint64_t bestNum_{100};   // baked into binary header

  // shortlist is stored in a skip list
  // [&shortLists_[wordToOffset_[word]], &shortLists_[wordToOffset_[word + 1]])
  // is a sorted array of word indices in the shortlist for word
  mio::mmap_source mmapMem_;
  uint64_t wordToOffsetSize_;
  uint64_t shortListsSize_;
  const uint64_t *wordToOffset_;
  const WordIndex *shortLists_;

void contentCheck() {
  bool failFlag = 0;
  // The offset table has to be within the size of shortlists.
  for(int i = 0; i < wordToOffsetSize_; i++)
    failFlag |= wordToOffset_[i] > shortListsSize_;
  // The vocabulary indices have to be within the vocabulary size.
  size_t vSize = trgVocab_->size();
  for(int j = 0; j < shortListsSize_; j++)
    failFlag |= shortLists_[j] > vSize;
  ABORT_IF(failFlag, "Error: shortlists content is wrong");
}

public:
  // load shortlist from buffer
  void load(const void* ptr, size_t blobSize, bool check=true){
    char *bytePtr = (char*)ptr;

    // Read preamble: magicSignature + bodySize + checksum
    uint64_t *preamblePtr = (uint64_t *)bytePtr;

    // Magic signature check
    uint64_t magicSignature = *(preamblePtr++);
    ABORT_IF(magicSignature != BINARY_SHORTLIST_MS, "This binary shortlist format is not supported");

    uint64_t bodySize = *(preamblePtr++);
    uint64_t checksum = *(preamblePtr++);

    // Bounds check
    size_t blobSizeExpected = sizeof(magicSignature) + bodySize + sizeof(checksum);
    ABORT_IF(blobSize < blobSizeExpected,
             "Bounds check failed: actual blob size {} < expected {}", blobSize, blobSizeExpected);

    // checksum check
    uint64_t *bodyHeaderPtr = preamblePtr;
    uint64_t checksumActual
        = (uint64_t)util::hashMem<uint64_t>(bodyHeaderPtr, bodySize / sizeof(uint64_t));
    ABORT_IF(checksumActual != checksum, "checksum check failed: this binary shortlist is corrupted");

    // Read firstNum_ and bestNum_
    firstNum_ = *(bodyHeaderPtr++);
    bestNum_ = *(bodyHeaderPtr++);
    LOG(info, "[data] The first no. is {} and best no. is {}", firstNum_, bestNum_);

    // Read the lengths of vectors
    wordToOffsetSize_ = *(bodyHeaderPtr++);
    shortListsSize_ = *(bodyHeaderPtr++);

    // Read the contents of the vectors
    wordToOffset_ = bodyHeaderPtr;
    const uint64_t *wordToOffsetImageBound = wordToOffset_ + wordToOffsetSize_;
    shortLists_ = (uint32_t *)wordToOffsetImageBound;

    // Shortlists content check (x2 speed-down)
    if(check)
      contentCheck();
  }

  // load shortlist from file
  void load(const std::string& filename, bool check=true) {
    std::error_code error;
    mmapMem_.map(filename, error);
    ABORT_IF(error, "Error mapping file: {}", error.message());
    load(mmapMem_.data(), mmapMem_.mapped_length(), check);
  }

public:
  BinaryShortlistGenerator(Ptr<Options> options,
                           Ptr<const Vocab> srcVocab,
                           Ptr<const Vocab> trgVocab,
                           size_t srcIdx = 0,
                           size_t /*trgIdx*/ = 1,
                           bool shared = false)
      : options_(options),
        srcVocab_(srcVocab),
        trgVocab_(trgVocab),
        srcIdx_(srcIdx),
        shared_(shared) {

    std::vector<std::string> vals = options_->get<std::vector<std::string>>("shortlist");
    ABORT_IF(vals.empty(), "No path to shortlist file given");
    std::string fname = vals[0];
    bool check = vals.size() > 1 ? std::stoi(vals[1]) : 1;
    std::string dumpPath = vals.size() > 2 ? vals[2] : "";

    LOG(info, "[data] Loading binary shortlist as {} {}", fname, check);
    load(fname, check);

    if(!dumpPath.empty())
      dump(dumpPath);
  }

  ~BinaryShortlistGenerator(){
    mmapMem_.unmap();
  }

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) const override {
    auto srcBatch = (*batch)[srcIdx_];
    size_t srcVocabSize = srcVocab_->size();
    size_t trgVocabSize = trgVocab_->size();

    // Since V=trgVocab_->size() is not large, anchor the time and space complexity to O(V).
    // Attempt to squeeze the truth tables into CPU cache
    std::vector<bool> srcTruthTable(srcVocabSize, 0);  // holds selected source words
    std::vector<bool> trgTruthTable(trgVocabSize, 0);  // holds selected target words

    // add firstNum most frequent words
    for(WordIndex i = 0; i < firstNum_ && i < trgVocabSize; ++i)
      trgTruthTable[i] = 1;

    // collect unique words from source
    // add aligned target words: mark trgTruthTable[word] to 1
    for(auto word : srcBatch->data()) {
      WordIndex srcIndex = word.toWordIndex();
      if(shared_)
        trgTruthTable[srcIndex] = 1;
      // If srcIndex has not been encountered, add the corresponding target words
      if (!srcTruthTable[srcIndex]) {
        for (uint64_t j = wordToOffset_[srcIndex]; j < wordToOffset_[srcIndex+1]; j++)
          trgTruthTable[shortLists_[j]] = 1;
        srcTruthTable[srcIndex] = 1;
      }
    }

    // Due to the 'multiple-of-eight' issue, the following O(N) patch is inserted
    size_t trgTruthTableOnes = 0;   // counter for no. of selected target words
    for (size_t i = 0; i < trgVocabSize; i++) {
      if(trgTruthTable[i])
        trgTruthTableOnes++;
    }

    // Ensure that the generated vocabulary items from a shortlist are a multiple-of-eight
    // This is necessary until intgemm supports non-multiple-of-eight matrices.
    for (size_t i = firstNum_; i < trgVocabSize && trgTruthTableOnes%8!=0; i++){
      if (!trgTruthTable[i]){
        trgTruthTable[i] = 1;
        trgTruthTableOnes++;
      }
    }

    // turn selected indices into vector and sort (Bucket sort: O(V))
    std::vector<WordIndex> indices;
    for (WordIndex i = 0; i < trgVocab_->size(); i++) {
      if(trgTruthTable[i])
        indices.push_back(i);
    }

    return New<Shortlist>(indices);
  }

  virtual void dump(const std::string& prefix) const override {
      // Dump top most frequent words from target vocabulary
      LOG(info, "[data] Saving shortlist dump to {}", prefix + ".{top,dic}");
      io::OutputFileStream outTop(prefix + ".top");
      for(WordIndex i = 0; i < firstNum_ && i < trgVocab_->size(); ++i)
        outTop << (*trgVocab_)[Word::fromWordIndex(i)] << std::endl;

      // Dump translation pairs from dictionary
      io::OutputFileStream outDic(prefix + ".dic");
      for(int i =1; i < wordToOffsetSize_; i++){
        for (int slowIndex= wordToOffset_[i-1]; slowIndex< wordToOffset_[i]; slowIndex++) {
          WordIndex srcId = i-1;
          WordIndex trgId = shortLists_[slowIndex];
          outDic << (*srcVocab_)[Word::fromWordIndex(srcId)]
                 << "\t" << (*trgVocab_)[Word::fromWordIndex(trgId)] << std::endl;
        }
      }
  }

};

class FakeShortlistGenerator : public ShortlistGenerator {
private:
  std::vector<WordIndex> indices_;

public:
  FakeShortlistGenerator(const std::unordered_set<WordIndex>& indexSet)
      : indices_(indexSet.begin(), indexSet.end()) {
    std::sort(indices_.begin(), indices_.end());
  }

  Ptr<Shortlist> generate(Ptr<data::CorpusBatch> /*batch*/) const override {
    return New<Shortlist>(indices_);
  }
};

}  // namespace data
}  // namespace marian
