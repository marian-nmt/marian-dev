#pragma once

// ROCm/HIP compatibility shim for Marian.
//
// This is the ONLY file (plus the forwarding headers under hip_compat/) that
// knows about HIP. It is force-included on every HIP translation unit via
// CMAKE_HIP_FLAGS (-include .../cuda_to_hip.h) so the aliases and steering
// macros below are visible before any other include, regardless of per-file
// include order. On an NVIDIA build this file is never compiled (it is only
// pulled in under USE_HIP), so the CUDA path is unchanged.

#if defined(__HIPCC__) || defined(USE_HIP)

// Host-side standard headers some .cu files rely on transitively through the
// CUDA toolkit headers; pull them in before <hip/hip_runtime.h> so a host
// memcpy/malloc in a .cu compiled as HIP still resolves (gpuRIR lesson).
#include <cstring>
#include <cstdlib>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Marian's FP16 and toolkit-include steering keys on CUDA_VERSION (and, in
// types.h/operators.h, on __CUDACC__/__HIP_DEVICE_COMPILE__). Provide a
// CUDA_VERSION >= 11000 sentinel so the modern library code paths -- the
// cublasLt affine epilogue (prod.cpp) and the cuSPARSE generic API
// (prod_sparse_cu11.h) -- are selected on HIP. We deliberately do NOT define
// __CUDACC__: that would make rocThrust treat hipcc as nvcc and select its
// (unimplemented) CUDA/CUB backend. types.h/operators.h handle __HIPCC__
// directly instead.
#ifndef CUDA_VERSION
#define CUDA_VERSION 11080
#endif

// HIP has no atomicAdd_block; a plain atomicAdd on a __shared__ address is
// workgroup-scoped on AMD, so map it. Used only by the LSH/hash path.
#ifndef atomicAdd_block
#define atomicAdd_block atomicAdd
#endif

// ---- CUDA runtime -> HIP runtime symbol aliases ----------------------------
// Only the symbols Marian actually uses. Types first, then enums/values, then
// functions.

using cudaError_t   = hipError_t;
using cudaStream_t  = hipStream_t;
using cudaEvent_t   = hipEvent_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaDataType  = hipDataType;
using cudaDataType_t = hipDataType;

#define cudaSuccess hipSuccess

#define cudaMemcpyHostToHost     hipMemcpyHostToHost
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault        hipMemcpyDefault

#define cudaHostAllocDefault     hipHostMallocDefault

#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

#define cudaMalloc            hipMalloc
#define cudaFree              hipFree
#define cudaHostAlloc(p, s, f) hipHostMalloc((p), (s), (f))
#define cudaFreeHost          hipHostFree
#define cudaMemcpy            hipMemcpy
#define cudaMemcpyAsync       hipMemcpyAsync
#define cudaMemset           hipMemset
#define cudaMemsetAsync      hipMemsetAsync
#define cudaSetDevice         hipSetDevice
#define cudaGetDevice         hipGetDevice
#define cudaGetDeviceCount    hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute  hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamCreate      hipStreamCreate
#define cudaStreamDestroy     hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaGetLastError      hipGetLastError
#define cudaGetErrorString    hipGetErrorString

#define CUDA_R_16F hipDataType::HIP_R_16F
#define CUDA_R_32F hipDataType::HIP_R_32F

#endif // __HIPCC__ || USE_HIP
