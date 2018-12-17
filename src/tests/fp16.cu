#include <iostream>
#include <cuda_fp16.h>

__global__ void testHalf(float x, float y) {
  __half hx = __float2half(x);
  __half hy = __float2half(y);
  __half hz = __hsub(hx, hy);
  printf("%f + %f = %f\n", __half2float(hx), __half2float(hy), __half2float(hz));
}

int main(int argc, char** argv) {

  testHalf<<<1, 1>>>(3.14, 2.71);
  cudaDeviceSynchronize();

  return 0;
}