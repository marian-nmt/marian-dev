#pragma once

#include <functional>

namespace marian {
namespace util {

template <class T> using hash = std::hash<T>;

// This combinator is based on boost::hash_combine, but uses
// std::hash as the hash implementation. Used as a drop-in
// replacement for boost::hash_combine.
template <class T, class HashType = std::size_t>
inline void hash_combine(HashType& seed, T const& v) {
  hash<T> hasher;
  seed ^= static_cast<HashType>(hasher(v)) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// Hash a whole chunk of memory, mostly used for diagnostics
template <class T, class HashType = std::size_t>
inline HashType hashMem(const T* beg, size_t len, HashType seed = 0) {
  for(auto it = beg; it < beg + len; ++it)
    hash_combine(seed, *it);
  return seed;
}

/**
 * Base case for template recursion below (no arguments are hashed to 0)
 */
template <class HashType = std::size_t>
inline HashType hashArgs() {
  return 0;
}

/**
 * Hash an arbitrary number of arguments of arbitrary type via template recursion
 */
template <class T, class ...Args, class HashType = std::size_t>
inline HashType hashArgs(T arg, Args... args) {
  // Hash arguments without first arg
  HashType seed = hashArgs(args...);
  // Hash first arg and combine which above hash
  hash_combine(seed, arg);
  return seed;
}

}
}
