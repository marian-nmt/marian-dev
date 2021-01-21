#pragma once

#include <string>
#include <functional>
namespace marian {

struct ModelTask {
  virtual ~ModelTask() {}
  virtual void run() = 0;
};

struct ModelCallbackTask {
  virtual ~ModelCallbackTask() {}
  virtual void run(std::function<void(const int, const std::string&)>) = 0;
};

struct ModelServiceTask {
  virtual ~ModelServiceTask() {}
  virtual std::string run(const std::string&, std::function<void(const int, const std::string&)>) = 0;
};
}  // namespace marian
