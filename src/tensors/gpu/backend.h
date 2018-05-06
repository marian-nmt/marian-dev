#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <mutex>

#include "common/config.h"
#include "tensors/backend.h"
#include "tensors/gpu/common_helpers.h" //CUDA_CHECK
#define MAX_DEVICES 8 //@TODO mb change


#define CURAND_CALL(x)                                \
  do {                                                \
    if((x) != CURAND_STATUS_SUCCESS) {                \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      exit(1);                                        \
    }                                                 \
  } while(0)

namespace marian {
namespace gpu {

class Backend : public marian::Backend {
public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    setHandles();
    tryCreateStreams();
  }

  void setDevice() { cudaSetDevice(deviceId_.no); }

  void synchronize() { cudaStreamSynchronize(0); }

  void synchronizeAllStreams() {
    for (int i = 0; i < MAX_DEVICES; i++) {
      for (int j = 0; j < MAX_DEVICES; j++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i][j]));
      }
    }
  }

  cublasHandle_t getCublasHandle() { return cublasHandle_; }

  curandGenerator_t getCurandGenerator() { return curandGenerator_; }

  void * getStream(int this_id, int other_id) {
      return (void *)&streams[this_id][other_id];
  }

  static cudaStream_t streams[MAX_DEVICES][MAX_DEVICES];
  static std::mutex createStreamMutex;
  static bool streamsCreated;

private:
  void tryCreateStreams() {
    //We need to lock in order to prevent multiple
    //gpus from creating streams.
    std::lock_guard<std::mutex> guard(createStreamMutex);
    if (!streamsCreated) {
      streamsCreated = true;
      for (int i = 0; i < MAX_DEVICES; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < MAX_DEVICES; j++) {
          CUDA_CHECK(cudaStreamCreate(&streams[i][j]));
        }
      }
      std::cout << "Streams created successfully." << std::endl;
    }
  }

  cublasHandle_t cublasHandle_;
  curandGenerator_t curandGenerator_;

  void setHandles() {
    cublasHandle_ = create_handle();
    curandGenerator_ = createCurandGenerator();
  }

  curandGenerator_t createCurandGenerator() {
    cudaSetDevice(deviceId_.no);
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed_));

    // cudaStream_t stream = 0;
    // CURAND_CALL(curandSetStream(generator, stream));
    // CURAND_CALL(curandDestroyGenerator(generator));
    return generator;
  }

  cublasHandle_t create_handle() {
    cudaSetDevice(deviceId_.no);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    return cublasHandle;
  }
};
}
}
