#include "../models/model_factory.h"
#include "../models/model_task.h"
#include "marian.h"

namespace marian {

class ReproTask : public marian::ModelServiceTask {
private:
  Ptr<ExpressionGraph> graph_;
  Ptr<models::ICriterionFunction> builder_;  // Training model

public:
  ReproTask() {
    graph_ = New<ExpressionGraph>();
    graph_->setDevice({0, DeviceType::cpu});
    graph_->reserveWorkspaceMB(128);
    // builder_ = models::createCriterionFunctionFromOptions(options_, models::usage::training);
  }
  std::string run(const std::string& json) override {
    return "";
  }
};

int main(int argc, char **argv) {
  return 0;
}
}
