#include <iostream>
#include <cmath>
#include <vector>

#include <unistd.h>
#include <cuda.h>
#include <nccl.h>

#define CUDA_CHECK(expr) do {                                                       \
  cudaError_t rc = (expr);                                                         \
  if(rc != cudaSuccess) {                                                           \
    std::cerr << "CUDA error " << rc << " " << cudaGetErrorString(rc) << std::endl; \
    abort(); \
  } \
 } while(0)

#define NCCL_CHECK(expr) do {                                                       \
  ncclResult_t rc = (expr);                                                         \
  if(rc != ncclSuccess) {                                                           \
    std::cerr << "NCCL error " << rc << " " << ncclGetErrorString(rc) << std::endl; \
    abort(); \
  } \
 } while(0)

int main(int argc, char** argv) {
    int devices = 8;
    int devs[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    size_t bytes = std::pow(2, 32); // 4GB

    size_t total = bytes / sizeof(float);
    size_t shard = total / bytes; shard;

    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    
    std::vector<float*> grads(devices);
    std::vector<float*> params(devices);

    cudaStream_t streams[devices];
    ncclComm_t comms[devices]; 

    std::cerr << "test 1" << std::endl;

    for(int i = 0; i < devices; ++i) {
        CUDA_CHECK(cudaSetDevice(devs[i]));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&params[i], bytes));
        CUDA_CHECK(cudaMalloc(&grads[i], bytes));
    }

    NCCL_CHECK(ncclCommInitAll(comms, devices, devs));

    std::cerr << "test 2" << std::endl;

    for(int j = 0; j < 10; ++j) {
        std::cerr << "Attempt: " << j << std::endl;

        for(int i = 0; i < devices; ++i) {
            CUDA_CHECK(cudaSetDevice(devs[i]));
            CUDA_CHECK(cudaStreamSynchronize(0));
        }

        std::cerr << "test 3" << std::endl;

        NCCL_CHECK(ncclGroupStart());
        for(int i = 0; i < devices; ++i) {
            const float*  sendbuf = grads[i];
            float* recvbuf        = grads[i] + i * shard;
            NCCL_CHECK(ncclReduceScatter(sendbuf, recvbuf, shard, ncclFloat32, ncclSum, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());

        std::cerr << "test 4" << std::endl;
        
        for(int i = 0; i < devices; ++i) {
            CUDA_CHECK(cudaSetDevice(devs[i]));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }

        std::cerr << "test 5" << std::endl;

        for(int i = 0; i < devices; ++i) {
            CUDA_CHECK(cudaSetDevice(devs[i]));
            CUDA_CHECK(cudaStreamSynchronize(0));
            CUDA_CHECK(cudaMemcpy(params[i], grads[i], bytes, cudaMemcpyDeviceToDevice));
        } // do stuff on stream 0
        

        for(int i = 0; i < devices; ++i) {
            CUDA_CHECK(cudaSetDevice(devs[i]));
            CUDA_CHECK(cudaStreamSynchronize(0));
        }

        std::cerr << "test 6" << std::endl;

        NCCL_CHECK(ncclGroupStart());
        for(int i = 0; i < devices; ++i) {
            const float*  sendbuf = params[i] + i * shard;
            float* recvbuf        = params[i];
            NCCL_CHECK(ncclAllGather(sendbuf, recvbuf, shard, ncclFloat32, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());

        std::cerr << "test 7" << std::endl;
        
        for(int i = 0; i < devices; ++i) {
            CUDA_CHECK(cudaSetDevice(devs[i]));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }

        std::cerr << "test 8" << std::endl;

    }

    for(int i = 0; i < devices; ++i) {
        CUDA_CHECK(cudaSetDevice(devs[i]));
        CUDA_CHECK(cudaFree(params[i]));
        CUDA_CHECK(cudaFree(grads[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        NCCL_CHECK(ncclCommDestroy(comms[i]));
    }

    std::cerr << "test 9" << std::endl;
}
