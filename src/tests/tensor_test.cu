#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "marian.h"
#include "functional/functional.h"

using namespace marian;

template <typename T>
__global__ void gLogSoftmax(gpu::Tensor<T> out,
                            gpu::Tensor<T> in) {

  /****************************************************************************/

  int rows = out.shape().elements() / out.shape().back();
  int cols = out.shape().back();

  using namespace functional;
  namespace facc = functional::accumulator;

  auto lambda = [&](int row_index) {
    auto inRow = in.row(row_index);
    auto outRow = out.row(row_index);

    ref<1> x;
    auto r_max = max_row(x);
    auto r_sum = sum_row(exp(x - r_max));
    auto logsoftmax = x - log(r_sum) - r_max;

    gpu::transform_row(outRow, inRow, reduce(logsoftmax, inRow));
  };

  gpu::foreach_row(rows, lambda);
}

const int MAX_THREADS = 512;
const int MAX_BLOCKS = 65535;

int main(int argc, char** argv) {

using namespace marian;
using namespace marian::functional;


  size_t m = 1024 / 128;
  size_t k = 128;

  int blocks = std::min(MAX_BLOCKS, (int)m);
  int threads = std::min(MAX_THREADS, (int)k);
  int shared = sizeof(float) * threads * 2;

  gpu::Tensor<float> gOut;
  gpu::Tensor<float> gIn;
  gLogSoftmax<<<blocks, threads, shared>>>(gOut, gIn);


  return 0;
}
