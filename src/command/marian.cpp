#include "marian.h"

#include "training/graph_group_async.h"
#include "training/graph_group_multinode.h"
#include "training/graph_group_singleton.h"
#include "training/graph_group_sync.h"
#include "training/training.h"

#ifdef CUDA_FOUND
#include "training/graph_group_async_drop.h"
#include <cuda.h>
#include <cuda_runtime.h>
//#include "tensors/gpu/common_helpers.h"

//@TODO mb wrap inside CUDA_CHECK
void enablePeerAccess(size_t deviceA, size_t deviceB) {
  //Attempt to enable peer access
  int result;
  cudaDeviceCanAccessPeer(&result, deviceA, deviceB);
  if (result) {
    cudaSetDevice(deviceA);
    cudaDeviceEnablePeerAccess (deviceB, 0);
    LOG(info, "[GPU] PeerMemoryAccess enabled between devices {} and {}", deviceA, deviceB);
  } else {
    LOG(warn, "[GPU[ PeerMemoryAccess unavailable between devices {} and {}", deviceA, deviceB);
  }
  cudaDeviceCanAccessPeer(&result, deviceB, deviceA);
}

#endif

bool configureMPI(int, char**);


int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv);
  auto devices = options->getDevices();

#ifdef CUDA_FOUND
  if (options->get<bool>("peer-access")) {
    for (auto deviceA : devices) {
      if (deviceA.type == DeviceType::gpu) {
        for (auto deviceB : devices) {
          if (deviceA != deviceB) {
            enablePeerAccess(deviceA.no, deviceB.no);
          }
        }
      } else {
        break;
      }
    }
  }
#endif


  if(options->get<bool>("multi-node")) {
    ABORT_IF(!configureMPI(argc, argv), "MPI not found.");

    LOG(warn, "[experimental] Running multi-node training");
    New<Train<MultiNodeGraphGroup>>(options)->run();
  } else {
    if(devices.size() == 1) {
      New<Train<SingletonGraph>>(options)->run();
    } else {
      if(options->get<bool>("sync-sgd"))
        New<Train<SyncGraphGroup>>(options)->run();
#ifdef CUDA_FOUND
      else if(options->get<float>("grad-dropping-rate") > 0.0)
        New<Train<AsyncGraphGroupDrop>>(options)->run();
#endif
      else
        New<Train<AsyncGraphGroup>>(options)->run();
    }
  }

  return 0;
}

bool configureMPI(int argc, char** argv) {
  bool enable = false;
#if MPI_FOUND
  int provided_thread_mode = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_mode);
  // Enable if occasional truncation errors
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  ABORT_IF(
      provided_thread_mode < MPI_THREAD_MULTIPLE,
      "Your version of MPI does not support multi-threaded communication.");

  enable = true;
#endif
  return enable;
}
