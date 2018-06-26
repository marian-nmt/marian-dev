#pragma once
#include <stdint.h>
#include <cstdlib>
#include <vector>
#include <iostream>

namespace intgemm {

uint64_t rdtsc_begin(uint32_t &processor) {
  uint32_t lo, hi;
  __asm__ __volatile__ (
      "cpuid\n\t"
      "rdtscp\n\t"
      "mov %%eax, %0\n\t"
      "mov %%edx, %1\n\t"
      "mov %%ecx, %2\n\t"
      : "=r" (lo), "=r" (hi), "=r" (processor)
      : /* no input */
      : "rax", "rbx", "rcx", "rdx");
  return static_cast<uint64_t>(hi) << 32 | lo;
}

uint64_t rdtsc_end(uint32_t &processor) {
  uint32_t lo, hi;
  __asm__ __volatile__ (
      "rdtscp\n\t"
      "mov %%eax, %0\n\t"
      "mov %%edx, %1\n\t"
      "mov %%ecx, %2\n\t"
      "cpuid\n\t"
      : "=r" (lo), "=r" (hi), "=r" (processor)
      : /* no input */
      : "rax", "rbx", "rcx", "rdx");
  return static_cast<uint64_t>(hi) << 32 | lo;
}

class StopWatch {
  public:
    StopWatch(std::vector<uint64_t> &stats)
      : stats_(stats), start_(rdtsc_begin(processor_)) {}

    ~StopWatch() {
      uint32_t proc;
      uint64_t stop = rdtsc_end(proc);
      if (proc != processor_) {
        std::cerr << "Detected core change from " << processor_ << " to " << proc << std::endl;
        abort();
      }
      stats_.push_back(stop - start_);
    }

  private:
    std::vector<uint64_t> &stats_;
    uint32_t processor_;
    uint64_t start_;
};

} // namespace intgemm
