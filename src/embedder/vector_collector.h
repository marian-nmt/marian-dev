#pragma once

#include "common/options.h"
#include "common/definitions.h"
#include "common/file_stream.h"

#include <map>
#include <mutex>

namespace marian {

// This class manages multi-threaded writing of embedded vectors to stdout or an output file.
// It will either output string versions of float vectors or binary equal length versions depending
// on its binary flag. If binary=false, width can be used to set the number of decimal places.
class VectorCollector {
public:
  static const size_t DEFAULT_WIDTH;

  VectorCollector(bool binary=false, size_t width=DEFAULT_WIDTH);
  VectorCollector(std::string outFile, bool binary=false, size_t width=DEFAULT_WIDTH);
  virtual ~VectorCollector() {}
  
  virtual void Write(long id, const std::vector<float>& vec);

  static Ptr<VectorCollector> Create(Ptr<Options> options);

protected:
  long nextId_{0};
  UPtr<std::ostream> outStrm_;
  bool binary_; // output binary floating point vectors if set
  size_t width_{DEFAULT_WIDTH};

  std::mutex mutex_;

  typedef std::map<long, std::vector<float>> Outputs;
  Outputs outputs_;

  virtual void WriteVector(const std::vector<float>& vec);
};

// Add a running summation of vector elements and outputs the average vector on destruction.
// Can also be configured to omit line-by-line results.
class AveragingVectorCollector : public VectorCollector {
private:
  std::vector<float> sum_;
  size_t count_{0};
  bool onlyLast_{false};

protected:
  virtual void WriteVector(const std::vector<float>& vec) override;

public:
  AveragingVectorCollector(bool binary=false, size_t width=DEFAULT_WIDTH, bool onlyLast=false) 
  : VectorCollector(binary, width), onlyLast_(onlyLast) {}
  
  AveragingVectorCollector(std::string outFile, bool binary=false, size_t width=DEFAULT_WIDTH, bool onlyLast=false) 
  : VectorCollector(outFile, binary, width), onlyLast_(onlyLast) {}
  
  virtual ~AveragingVectorCollector() {
    WriteAverage();
  }

  virtual void WriteAverage();
};


// collects vectors and hold them in memory
class BufferedVectorCollector : public VectorCollector {

private:
  std::vector<std::vector<float>> buffer;

protected:
  virtual void WriteVector(const std::vector<float>& vec) override;

public:
  BufferedVectorCollector(bool binary=false, size_t width=DEFAULT_WIDTH) 
  : VectorCollector(binary, width) {}
  
  BufferedVectorCollector(std::string outFile, bool binary=false, size_t width=DEFAULT_WIDTH)
  : VectorCollector(outFile, binary, width) {}

  auto getBuffer() -> decltype(buffer) {
    return buffer;
  }

  virtual ~BufferedVectorCollector() {}

};


}  // namespace marian
