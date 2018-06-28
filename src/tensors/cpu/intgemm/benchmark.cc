#include "aligned.h"
#include "avx512_gemm.h"
#include "avx2_gemm.h"
#include "ssse3_gemm.h"
#include "sse2_gemm.h"
#include "intgemm.h"
#include "stop_watch.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <iomanip>

namespace intgemm {

struct RandomMatrices {
  RandomMatrices(int A_rows_in, int width_in, int B_cols_in) :
    A_rows(A_rows_in), width(width_in), B_cols(B_cols_in),
    A(A_rows * width), B(width * B_cols) {
    for (int i = 0; i < A_rows * width; i++) {
        A[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
    
    for (int i = 0; i < B_cols * width; i++) {
        B[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
  }

  const int A_rows, width, B_cols;
  AlignedVector<float> A, B;
};

template <class Backend> void Run(const RandomMatrices &m, std::vector<uint64_t> &stats) {
  typedef typename Backend::Integer Integer;
  float quant_mult = 127.0 / 2;
  float unquant_mult = 1.0 / (quant_mult * quant_mult);
  AlignedVector<Integer> A_prepared(m.A_rows * m.width);
  Backend::PrepareA(m.A.get(), A_prepared.get(), quant_mult, m.A_rows, m.width);
  AlignedVector<Integer> B_prepared(m.width * m.B_cols);
  Backend::PrepareB(m.B.get(), B_prepared.get(), quant_mult, m.width, m.B_cols);
  AlignedVector<float> output(m.A_rows * m.B_cols);
  // Burn in
  Backend::Multiply(A_prepared.get(), B_prepared.get(), output.get(), unquant_mult, m.A_rows, m.width, m.B_cols);
  {
    StopWatch w(stats);
    Backend::Multiply(A_prepared.get(), B_prepared.get(), output.get(), unquant_mult, m.A_rows, m.width, m.B_cols);
  }
}

template <class Backend> void RunAll(RandomMatrices *matrices, RandomMatrices *matrices_end, std::vector<std::vector<uint64_t> > &stats) {
  if (Backend::kUses > kCPU) return;
  std::size_t size = matrices_end - matrices;
  if (stats.size() < size)
    stats.resize(size);
  for (std::size_t i = 0; i < size; ++i) {
    Run<Backend>(matrices[i], stats[i]);
  }
}

struct BackendStats {
  std::vector<std::vector<uint64_t> > ssse3_8bit;
  std::vector<std::vector<uint64_t> > avx2_8bit;
  std::vector<std::vector<uint64_t> > avx512_8bit;
  std::vector<std::vector<uint64_t> > sse2_16bit;
  std::vector<std::vector<uint64_t> > avx2_16bit;
  std::vector<std::vector<uint64_t> > avx512_16bit;
};

const float kOutlierThreshold = 0.75;
void Summarize(std::vector<uint64_t> &stats) {
  // Throw out outliers.
  std::vector<uint64_t>::iterator keep = stats.begin() + stats.size() * kOutlierThreshold;
  std::nth_element(stats.begin(), keep, stats.end());
  double avg = 0.0;
  for (std::vector<uint64_t>::const_iterator i = stats.begin(); i != keep; ++i) {
    avg += *i;
  }
  avg /= (keep - stats.begin());
  double s = 0.0;
  for (std::vector<uint64_t>::const_iterator i = stats.begin(); i != keep; ++i) {
    double off = (double)*i - avg;
    s += off * off;
  }
  s = sqrt(s / (keep - stats.begin() - 1));
  std::cout << std::setw(8) << *std::min_element(stats.begin(), stats.end()) << '\t' << std::setw(8) << avg << '\t' << std::setw(8) << s;
/*  std::cout << '\n';
  for (std::vector<uint64_t>::const_iterator i = stats.begin(); i != stats.end(); ++i) {
    std::cout << *i << ' ';
  }*/
}

template <class Backend> void Print(std::vector<std::vector<uint64_t> > &stats, int index) {
  if (stats.empty()) return;
  std::cout << Backend::kName << '\t';
  Summarize(stats[index]);
  std::cout << '\n';
}

} // namespace intgemm

// Program takes no input
int main(int argc, char ** argv) {
  std::cerr << "Remember to run this on a specific core:\ntaskset --cpu-list 0 " << argv[0] << std::endl;
  std::srand(45678);
  using namespace intgemm;
  RandomMatrices matrices[] = {
    {1, 64, 8},
    {8, 256, 256},
    {8, 2048, 256},
    {8, 256, 2048},
    {320, 256, 256},
    {472, 256, 256},
    {248, 256, 256},
    {200, 256, 256},
    // Additional stuff
    {256, 256, 256},
    {512, 512, 512},
    {1024, 1024, 1024},
/*    {4096, 4096, 4096},
    {4096, 4096, 2048},
    {4096, 4096, 1024},
    {4096, 4096, 512},
    {4096, 4096, 256},*/
    {4096, 4096, 128}
  };
  RandomMatrices *matrices_end = (RandomMatrices*)matrices + sizeof(matrices) / sizeof(RandomMatrices);
  // Only do full sampling for <1024 rows.
  RandomMatrices *full_sample;
  for (full_sample = matrices_end - 1; full_sample >= matrices && full_sample->A_rows >= 1024; --full_sample) {}
  ++full_sample;

  BackendStats stats;
  const int kSamples = 100;
  // Run samples far apart to reduce temporary noise.
  for (int samples = 0; samples < kSamples; ++samples) {
    std::cerr << "Sample " << samples << " / " << kSamples << std::endl;
    RandomMatrices *end = (samples < 4) ? matrices_end : full_sample;
    RunAll<SSSE3_8bit>(matrices, end, stats.ssse3_8bit);
    RunAll<SSE2_16bit>(matrices, end, stats.sse2_16bit);
    RunAll<AVX2_8bit>(matrices, end, stats.avx2_8bit);
    RunAll<AVX2_16bit>(matrices, end, stats.avx2_16bit);
#ifndef INTGEMM_NO_AVX512
    RunAll<AVX512_8bit>(matrices, end, stats.avx512_8bit);
    RunAll<AVX512_16bit>(matrices, end, stats.avx512_16bit);
#endif
  }

  if (stats.sse2_16bit.empty()) {
    std::cerr << "No CPU support." << std::endl;
    return 1;
  }
  for (std::size_t i = 0; i < sizeof(matrices) / sizeof(RandomMatrices); ++i) {
    std::cout << "Multiply\t" << matrices[i].A_rows << '\t' << matrices[i].width << '\t' << matrices[i].B_cols << '\t' << "Samples=" << (kOutlierThreshold * stats.sse2_16bit[i].size()) << '\n';
    Print<SSSE3_8bit>(stats.ssse3_8bit, i);
    Print<AVX2_8bit>(stats.avx2_8bit, i);
#ifndef INTGEMM_NO_AVX512
    Print<AVX512_8bit>(stats.avx512_8bit, i);
#endif
    Print<SSE2_16bit>(stats.sse2_16bit, i);
    Print<AVX2_16bit>(stats.avx2_16bit, i);
#ifndef INTGEMM_NO_AVX512
    Print<AVX512_16bit>(stats.avx512_16bit, i);
#endif
  }
  return 0;
}


