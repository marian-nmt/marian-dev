#pragma once
// MOAT HIP shim: <curand.h> -> hipRAND. Aliases the cuRAND host-generator
// symbols Marian's rand.cpp uses.
#include <hiprand/hiprand.h>

using curandGenerator_t = hiprandGenerator_t;
using curandStatus_t    = hiprandStatus_t;

#define CURAND_STATUS_SUCCESS     HIPRAND_STATUS_SUCCESS
#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT

#define curandCreateGenerator             hiprandCreateGenerator
#define curandCreateGeneratorHost         hiprandCreateGeneratorHost
#define curandDestroyGenerator            hiprandDestroyGenerator
#define curandSetPseudoRandomGeneratorSeed hiprandSetPseudoRandomGeneratorSeed
#define curandGenerateUniform             hiprandGenerateUniform
#define curandGenerateNormal              hiprandGenerateNormal
