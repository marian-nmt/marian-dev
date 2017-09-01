#pragma once

#include "models/s2s.h"
#include "models/amun.h"
#include "models/lm.h"
#include "models/hardatt.h"
#include "models/multi_s2s.h"

namespace marian {

struct ModelTask {
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual void init() = 0;
  virtual std::vector<std::string> run(const std::vector<std::string>&) = 0;
};

template <template <class> class TaskName, template <class> class Wrapper>
Ptr<ModelTask>
WrapModelType(Ptr<Config> options) {
  auto type = options->get<std::string>("type");
  if (type == "s2s")
    return New<TaskName<Wrapper<S2S>>>(options);
  else if (type == "amun")
    return New<TaskName<Wrapper<Amun>>>(options);
  else if (type == "hard-att")
    return New<TaskName<Wrapper<HardAtt>>>(options);
  else if (type == "hard-soft-att")
    return New<TaskName<Wrapper<HardSoftAtt>>>(options);
  else if (type == "multi-s2s")
    return New<TaskName<Wrapper<MultiS2S>>>(options);
  else if (type == "multi-hard-att")
    return New<TaskName<Wrapper<MultiHardSoftAtt>>>(options);
  else if (type == "lm")
    return New<TaskName<Wrapper<LM>>>(options);
  else
    UTIL_THROW2("Unknown model type: " << type);
}

}
