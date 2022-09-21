#include "translator/self_adaptive.h"
#include "translator/server_common.h"

int main(int argc, char **argv) {
  using namespace marian;

  auto options = parseOptions(argc, argv, cli::mode::selfadaptiveServer);
  auto task = New<TrainSelfAdaptive>(options);

  return runServer(task, options);
}
