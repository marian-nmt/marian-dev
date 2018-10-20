#pragma once

#include "phf/phf.h"
#include "static_any/any.hpp"

template <typename T>
class PerfectHash {
private:
  phf phf_;

public:
  PerfectHash(T[] keys, size_t num) {
    int error = PHF::init(&phf, keys, num);
    ABORT_IF(error != 0, "PHF error {}", error);
  }

  ~PerfectHash() {
    PHF::destroy(&phf);
  }

  uint32_t operator[](const T& key) const {
    return PHF::hash(&phf, key);
  }

  size_t size() const {
    return phf_.m;
  }
};

// template <size_t S>
// class Node {
// private:
//   struct ElementType {
//     static_any<S> value;
//     size_t fingerprint;

//     template <typename T>
//     const T get() const { return value.get<T>(); }
//   };

//   PerfectHash<uint32_t>* ph_;
//   ElementType* array_;
//   size_t elements_;

//   template <typename T>
//   inline const ElementType& phLookup<T>(size_t keyId) const {
//     return (*array_)[(*ph_)[i]];
//   }

// public:
//   bool has(size_t keyId) const {
//     return phLookup(keyId).fingerprint == keyId;
//   }

//   bool has(const char* key) const {
//     return has(crc32(key));
//   }

//   template <typename T>
//   inline const T get<T>(const char* key) const {
//     return get<T>(crc32(key));
//   }

//   template <typename t>
//   inline const T get<T>(size_t keyId) const {
//     const auto& value = phLookup(keyId);
//     ABORT_IF(value.fingerprint != keyId, "Unseen key {}" , keyId);
//     return value.get<T>();
//   }

//   // const ConstMap& operator[](size_t keyId) {
//   // }
// }

// auto node = ConstMap(YAML::node);
// node["devices"][4][1]["test"].get<size_t>();
