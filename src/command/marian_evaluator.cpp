#include "marian.h"

#include "models/model_task.h"
#include "evaluator/evaluator.h"
#include "common/timer.h"

int main(int argc, char** argv) {
  using namespace marian;

  // @TODO: add mode evaluating
  auto options = parseOptions(argc, argv, cli::mode::evaluating);
  New<Evaluate<Evaluator>>(options)->run();
  
  return 0;
}
