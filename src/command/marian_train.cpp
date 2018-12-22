#include "marian.h"

#include "training/graph_group_async.h"
#include "training/graph_group_singleton.h"
#include "training/graph_group_sync.h"
#include "training/training.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto options = parseOptions(argc, argv);

  // --sync-sgd always selects SyncGraphGroup
  //
  // If given, then this implementation is used for all combinations of (single, multiple) MPI
  // processes x (single, multiple) GPUs per MPI process.  This variant is presently up-to-date and
  // best supported.
  if(options->get<bool>("sync-sgd")) {
    LOG(info, "Using synchronous training");
    New<Train<SyncGraphGroup>>(options)->run();
  }
  else {
    auto devices = Config::getDevices(options);
    if(devices.size() == 1) {
      LOG(info, "Using single-device training");
      New<Train<SingletonGraph>>(options)->run();
    } else {
      LOG(info, "Using asynchronous training");
      New<Train<AsyncGraphGroup>>(options)->run();
    }
  }

  return 0;
}
