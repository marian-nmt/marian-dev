#include "marian.h"

#include "common/timer.h"
#include "common/utils.h"
#include "training/training.h"
#include "translator/self_adaptive.h"

using namespace marian;

int main(int argc, char **argv) {
  auto options = parseOptions(argc, argv, cli::mode::selfadaptive);
  auto task = New<TrainSelfAdaptive>(options);

  timer::Timer timer;
  task->run();
  LOG(info, "Total time: {:.5f}s", timer.elapsed());

  return 0;
}
