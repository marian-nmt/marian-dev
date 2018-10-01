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

using namespace marian;
using namespace cpu;

struct float4 {
  float f[4];
};

float4 operator+(float4 x, float4 y) {
  float4 out;
  for(int i = 0; i < 4; ++i)
    out.f[i] = x.f[i] + y.f[i];
  return out;
}

float4 operator*(float4 x, float4 y) {
  float4 out;
  for(int i = 0; i < 4; ++i)
    out.f[i] = x.f[i] * y.f[i];
  return out;
}

// main call to function executing element-wise operation
template <class Functor, class... Tensors>
void Element2(const Functor& functor, marian::Tensor out, Tensors... tensors) {

  switch(out->type()) {
    case Type::float32: element<float4>(functor, out, tensors...); break;
    case Type::uint32:  element<uint32_t>(functor, out, tensors...); break;
    case Type::uint16:  element<uint16_t>(functor, out, tensors...); break;
    default: ABORT("Unsupported type for element-wise operation"); break;
  }
}

int main(int argc, char** argv) {
  using namespace functional;

  auto backend = New<cpu::Backend>(CPU0, 1111);
  auto alloc = New<TensorAllocator>(backend);
  alloc->reserveExact(10000);

  marian::Tensor out, in1, in2;
  alloc->allocate(out, {4}, Type::float32);
  alloc->allocate(in1, {4}, Type::float32);
  alloc->allocate(in2, {4}, Type::float32);

  in1->set<float>(2);
  in2->set<float>(3);

  std::cerr << in1->debug() << std::endl;
  std::cerr << in2->debug() << std::endl;

  auto f = _1 = _2 * _3;
  Element2(f, out, in1, in2);

  std::cerr << out->debug() << std::endl;

  return 0;
}
