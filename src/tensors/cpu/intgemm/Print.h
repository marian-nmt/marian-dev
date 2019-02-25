#pragma once

#include <stdint.h>
#include <iostream>

namespace intgemm {

template <class T> T PrintWrap(const T val) {
  return val;
}
inline int16_t PrintWrap(const int8_t val) {
  return val;
}

template <class T, class Reg> void Print(const Reg reg) {
  const T *val = reinterpret_cast<const T*>(&reg);
  for (std::size_t i = 0; i < sizeof(Reg) / sizeof(T); ++i) {
    std::cout << ' ' << PrintWrap(val[i]);
  }
}

} // namespace intgemm
