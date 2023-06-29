#include "embedder/vector_collector.h"

#include "common/logging.h"
#include "common/utils.h"

#include <iostream>
#include <iomanip>

namespace marian {

// This class manages multi-threaded writing of embedded vectors to stdout or an output file.
// It will either output string versions of float vectors or binary equal length versions depending
// on its binary_ flag.
VectorCollector::VectorCollector(bool binary, size_t width)
  : nextId_(0),
    binary_(binary),
    width_{width} {}

VectorCollector::VectorCollector(std::string outFile, bool binary, size_t width)
  : nextId_(0),
    outStrm_(new std::ostream(std::cout.rdbuf())),
    binary_(binary),
    width_(width) {
  if (outFile != "stdout")
    outStrm_.reset(new io::OutputFileStream(outFile));
}

void VectorCollector::Write(long id, const std::vector<float>& vec) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(id == nextId_) {
    WriteVector(vec);

    ++nextId_;

    typename Outputs::const_iterator iter, iterNext;
    iter = outputs_.begin();
    while(iter != outputs_.end()) {
      long currId = iter->first;

      if(currId == nextId_) {
        // 1st element in the map is the next
        WriteVector(iter->second);
        
        ++nextId_;

        // delete current record, move iter on 1
        iterNext = iter;
        ++iterNext;
        outputs_.erase(iter);
        iter = iterNext;
      } else {
        // not the next. stop iterating
        assert(nextId_ < currId);
        break;
      }
    }

  } else {
    // save for later
    outputs_[id] = vec;
  }
}

void VectorCollector::WriteVector(const std::vector<float>& vec) {
  if(binary_) {
    outStrm_->write((char*)vec.data(), vec.size() * sizeof(float));
  } else {
    *outStrm_ << std::fixed << std::setprecision(width_);
    for(auto v : vec)
      *outStrm_ << v << " ";
    *outStrm_ << std::endl;
  }
}

void AveragingVectorCollector::WriteVector(const std::vector<float>& vec) {
  if(!onlyLast_)
    VectorCollector::WriteVector(vec);
  
  if(sum_.size() < vec.size())
    sum_.resize(vec.size());
  for(size_t i = 0; i < vec.size(); ++i)
    sum_[i] += vec[i];
  count_++;
}

void AveragingVectorCollector::WriteAverage() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto avg = sum_;
  for(auto& val : avg)
    val /= (float)count_;
  VectorCollector::WriteVector(avg);
}

Ptr<VectorCollector> VectorCollector::Create(Ptr<Options> options) {
  std::string average = options->get<std::string>("average", "skip");
  std::string output  = options->get<std::string>("output");
  size_t width        = options->get<size_t>("width", DEFAULT_WIDTH);

  Ptr<VectorCollector> collector;
  if(average == "skip")
    collector = New<VectorCollector>(output, /*binary=*/false, width);
  else if(average == "append")
    collector = New<AveragingVectorCollector>(output, /*binary=*/false, width, /*onlyLast=*/false);
  else if(average == "only")
    collector = New<AveragingVectorCollector>(output, /*binary=*/false, width, /*onlyLast=*/true);
  else
    ABORT("Unknown configuration for VectorCollector");
  
  return collector;
}

}  // namespace marian
