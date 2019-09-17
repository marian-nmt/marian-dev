#pragma once

#include <string>
#include <vector>

#include "common/logging.h"

namespace marian {

struct ModelTask {
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual std::string run(const std::string&) = 0;
  virtual std::string run(const std::vector<std::string>&) {
    ABORT("Not implemented");
    return "";
  }
};
}  // namespace marian
