#pragma once

#include <string>

namespace marian {

struct ModelTask {
  virtual ~ModelTask() {}
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual ~ModelServiceTask() {}
  virtual std::string run(const std::string&) = 0;
  virtual std::vector<std::string> run(const std::vector<std::string>&) = 0;
};
}  // namespace marian
