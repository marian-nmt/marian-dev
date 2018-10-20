#pragma once

#include "phf/phf.h"
#include "3rd_party/any_type.h"

constexpr uint32_t val_32_const = 0x811c9dc5;
constexpr uint32_t prime_32_const = 0x1000193;
constexpr uint64_t val_64_const = 0xcbf29ce484222325;
constexpr uint64_t prime_64_const = 0x100000001b3;

inline constexpr uint32_t
hash_32_fnv1a_const(const char* const str,
                    const uint32_t value = val_32_const) noexcept {
  return (str[0] == '\0') ?
      value :
      hash_32_fnv1a_const(&str[1], (value ^ uint32_t(str[0])) * prime_32_const);
}

inline constexpr uint64_t
hash_64_fnv1a_const(const char* const str,
                    const uint64_t value = val_64_const) noexcept {
  return (str[0] == '\0') ?
      value :
      hash_64_fnv1a_const(&str[1], (value ^ uint64_t(str[0])) * prime_64_const);
}

inline constexpr uint64_t crc(const char* const str) noexcept {
  return hash_64_fnv1a_const(str);
}

/*****************************************************************************/

class PerfectHash {
private:
  phf phf_;

  PerfectHash(const uint64_t keys[], size_t num) {
    int error = PHF::init<uint64_t, true>(&phf_, keys, num,
      /* bucket size */ 4,
      /* loading factor */ 90,
      /* seed */ 123456);
    ABORT_IF(error != 0, "PHF error {}", error);
  }

public:

  PerfectHash(const std::vector<uint64_t>& v)
   : PerfectHash(v.data(), v.size()) { }

  ~PerfectHash() {
    PHF::destroy(&phf_);
  }

  uint32_t operator[](const uint64_t& key) const {
    return PHF::hash<uint64_t>(const_cast<phf*>(&phf_), key);
  }

  uint32_t operator[](const char* const keyStr) const {
    return (*this)[crc(keyStr)];
  }

  size_t size() const {
    return phf_.m;
  }
};

/*****************************************************************************/

class FastOpt {
private:
  struct ElementType {
    any_type value;
    uint64_t fingerprint{0};

    template <typename T>
    const T& as() const { return value.as<T>(); }
  };

  ElementType value_;

  PerfectHash* ph_{0};
  FastOpt** array_;
  size_t elements_{0};

  inline const FastOpt* phLookup(size_t keyId) const {
    return array_[(*ph_)[keyId]];
  }

public:

  template <class V>
  FastOpt(const std::map<uint64_t, V>& m) {
    std::vector<uint64_t> keys;
    for(const auto& it : m) {
      keys.push_back(it.first);
    }
    ph_ = new PerfectHash(keys);

    array_ = new FastOpt*[ph_->size()];
    elements_ = keys.size();

    for(const auto& it : m) {
      size_t pos = (*ph_)[it.first];
      auto node = new FastOpt(it.second);
      node->value_.fingerprint = it.first;
      array_[pos] = node;
    }
  }

  template <class V>
  FastOpt(const std::map<std::string, V>& m) {
    std::vector<uint64_t> keys;
    for(const auto& it : m) {
      keys.push_back(crc(it.first.c_str()));
    }
    ph_ = new PerfectHash(keys);

    array_ = new FastOpt*[ph_->size()];
    elements_ = keys.size();

    for(const auto& it : m) {
      uint64_t key = crc(it.first.c_str());
      size_t pos = (*ph_)[key];
      auto node = new FastOpt(it.second);
      node->value_.fingerprint = key;
      array_[pos] = node;
    }
  }

  template <class V>
  FastOpt(const V& v) {
    value_.value = v;
    elements_ = 0;
  }

  ~FastOpt() {
    for(int i = 0; i < elements_; ++i)
      delete array_[i];

    if(elements_ > 0)
      delete[] array_;
  }

  bool has(size_t keyId) const {
    return phLookup(keyId)->value_.fingerprint == keyId;
  }

  bool has(const char* key) const {
    return has(crc(key));
  }

  template <typename T>
  inline const T& get(size_t keyId) const {
    const auto& node = phLookup(keyId);
    ABORT_IF(node->value_.fingerprint != keyId, "Unseen key {}" , keyId);
    return node->as<T>();
  }

  template <typename T>
  inline const T& get(const char* const key) const {
    return get<T>(crc(key));
  }

  template <typename T>
  inline const T& as() const {
    ABORT_IF(elements_ != 0, "Not a leaf node");
    return value_.as<T>();
  }

  const FastOpt& operator[](size_t keyId) const {
    const auto& node = phLookup(keyId);
    ABORT_IF(node->value_.fingerprint != keyId, "Unseen key {}" , keyId);
    return *node;
  }

  const FastOpt& operator[](const char* const key) const {
    return operator[](crc(key));
  }

  const FastOpt& operator[](int key) const {
    return operator[]((size_t)key);
  }
};

// auto node = ConstMap(YAML::node);
// node["devices"][4][1]["test"].get<size_t>();
