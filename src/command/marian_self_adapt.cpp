#include "marian.h"

#include "common/file_stream.h"
#include "common/utils.h"
#include "training/self_adaptive.h"
#include "training/training.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, cli::mode::selfadaptive);
  auto task = New<TrainMultiDomain>(options);

  boost::timer::cpu_timer timer;
  task->run();
  LOG(info, "Total time: {}", timer.format());

  return 0;
}
