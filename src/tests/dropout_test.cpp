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

// main call to function executing element-wise operation
template <class Functor, class... Tensors>
void Element2(const Functor& functor, marian::Tensor out, Tensors... tensors) {

  switch(out->type()) {
    case Type::float32: element<FloatM128>(functor, out, tensors...); break;
    case Type::uint32:  element<uint32_t>(functor, out, tensors...); break;
    case Type::uint16:  element<uint16_t>(functor, out, tensors...); break;
    default: ABORT("Unsupported type for element-wise operation"); break;
  }
}

int main(int argc, char** argv) {
  using namespace functional;

  auto backend = New<cpu::Backend>(CPU0, 1111);
  auto alloc = New<TensorAllocator>(backend);
  alloc->reserveExact(100000000);

  marian::Tensor out, in1, in2;
  alloc->allocate(out, {512, 512}, Type::float32);
  //alloc->allocate(in1, {512, 512}, Type::float32);
  alloc->allocate(in2, {1, 512}, Type::float32);

  // std::vector<float> v1(in1->size());
  // std::iota(v1.begin(), v1.end(), 1.f);

  std::vector<float> v2(in2->size()); 
  std::iota(v2.begin(), v2.end(), 2.f);

  //in1->set(v1);
  in2->set(v2);

  //std::cerr << in1->debug(4, 10) << std::endl;
  //std::cerr << in2->debug(4, 10) << std::endl;

  {
    boost::timer::auto_cpu_timer timer;
    auto f = _1 = _1 + _2;
    for(int i = 0; i < 100000; i++) {
      element<float>(f, out, in2);
    }
  }

  {
    boost::timer::auto_cpu_timer timer;
    auto f = _1 = _1 + _2;
    for(int i = 0; i < 100000; i++) {
      element<FloatM128>(f, out, in2);
    }
  }

  {
    boost::timer::auto_cpu_timer timer;
    auto f = _1 = _1 + _2;
    for(int i = 0; i < 100000; i++) {
      cpu::int16::AddBias(out, in2);
    }
  }

  //std::cerr << out->debug(4, 10) << std::endl;

  return 0;
}
