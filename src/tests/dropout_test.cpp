#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "marian.h"
#include "functional/functional.h"
#include "functional/array.h"
#include "functional/tensor.h"
#include "functional/tmp.h"
#include "tensors/cpu/backend.h"
#include "tensors/cpu/element.h"

#include "tensors/cpu/sharp/int_gemm.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <cassert>
#include <cstddef>
#include <boost/timer/timer.hpp>

using namespace marian;
using namespace cpu;

int main(int argc, char** argv) {
  using namespace functional;

  auto backend = New<cpu::Backend>(CPU0, 1111);
  auto alloc = New<TensorAllocator>(backend);
  alloc->reserveExact(100000000);

  marian::Tensor params, grads;
  alloc->allocate(params, {256 * 512 * 512}, Type::float32);
  alloc->allocate(grads,  {256 * 512 * 512}, Type::float32);
  
  auto adam = New<Adam>(0.0003);
  adam->update(params, grads);

  {
    boost::timer::auto_cpu_timer timer;
    for(int i = 0; i < 100; i++) {
      adam->update(params, grads);
    }
  }

  //std::cerr << in2->debug() << std::endl;
  //std::cerr << out->debug() << std::endl;

  return 0;
}
