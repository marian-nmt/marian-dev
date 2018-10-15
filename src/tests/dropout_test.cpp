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

  marian::Tensor out, in2;
  alloc->allocate(out, {6, 4}, Type::float32);
  alloc->allocate(in2, {1, 4}, Type::float32);


  std::vector<float> vo(out->size());
  std::vector<float> vi(in2->size());
  std::iota(vo.begin(), vo.end(), 1.f);
  std::iota(vi.begin(), vi.end(), 1.f);

  out->set(vo);
  in2->set(vi);

  // {
  //   boost::timer::auto_cpu_timer timer;
  //   auto f = _1 = _1 + _2;
  //   for(int i = 0; i < 100000; i++) {
  //     element<float>(f, out, in2);
  //  ./ }
  // }

  {
    boost::timer::auto_cpu_timer timer;
    auto f = _1 = exp(_2) * 2.f;
    std::cerr << f.to_string() << std::endl;
    for(int i = 0; i < 1; i++) {
      element<float32x4>(f, out, in2);
    }
  }

  std::cerr << in2->debug() << std::endl;
  std::cerr << out->debug() << std::endl;

  return 0;
}
