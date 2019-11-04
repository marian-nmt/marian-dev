#pragma once

#include "common/intrusive_ptr.h"
#include "3rd_party/any_type.h"
#include "3rd_party/phf/phf.h"
#include "3rd_party/yaml-cpp/yaml.h"

namespace marian {

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
  ENABLE_INTRUSIVE_PTR(PerfectHash)

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

class FastOpt;

namespace helper {
  template <typename T> 
  struct As {
    static T apply(const FastOpt&);
  };

  template <typename T> 
  struct As<std::vector<T>> {
    static std::vector<T> apply(const FastOpt&);
  };
}

class FastOpt {
private:
  ENABLE_INTRUSIVE_PTR(FastOpt)

  template <typename T>
  friend struct helper::As;

public:
  enum struct NodeType {
    Null, Bool, Int64, Float64, String, Sequence, Map
  };

  struct RuntimeError : public std::runtime_error {
    RuntimeError() 
      : std::runtime_error("FastOpt RuntimeError") 
    {}
  };

private:
  struct ElementType {
    ENABLE_INTRUSIVE_PTR(ElementType)

    any_type value{0};

    template <typename T>
    const T& as() const { return value.as<T>(); }
  };

  IntrusivePtr<ElementType> value_;
  IntrusivePtr<PerfectHash> ph_;
  std::vector<IntrusivePtr<FastOpt>> array_;
  NodeType type_{NodeType::Null};

  uint64_t fingerprint_{0};
  size_t elements_{0};

  inline const IntrusivePtr<FastOpt> arrayLookup(size_t keyId) const {
    if(keyId < array_.size())
      return array_[keyId];
    else
      return nullptr;
  }

  inline const IntrusivePtr<FastOpt> phLookup(size_t keyId) const {
    if(ph_)
      return array_[(*ph_)[keyId]];
    else
      return nullptr;
  }

  void makeNull() {
    value_.reset(nullptr);
    elements_ = 0;
    type_ = NodeType::Null;
  }

  void makeScalar(const YAML::Node& v) {
    elements_ = 0;
    value_.reset(new ElementType());
    
    try {
      value_->value = v.as<bool>();
      type_ = NodeType::Bool;
    } catch(const YAML::BadConversion& /*e*/) {
      try {
        value_->value = v.as<int64_t>();
        type_ = NodeType::Int64;
      } catch(const YAML::BadConversion& /*e*/) {
        try {
          value_->value = v.as<double>();
          type_ = NodeType::Float64;
        } catch(const YAML::BadConversion& /*e*/) {
          try { 
            value_->value = v.as<std::string>();
            type_ = NodeType::String;
          } catch (const YAML::BadConversion& /*e*/) {
            throw FastOpt::RuntimeError();
          }
        }
      }
    }
  }

  void makeSequence(const std::vector<YAML::Node>& v) {
    array_.resize(v.size());
    elements_ = v.size();

    for(size_t pos = 0; pos < v.size(); ++pos) {
      array_[pos].reset(new FastOpt(v[pos]));
      array_[pos]->fingerprint_ = pos;
    }

    type_ = NodeType::Sequence;
  }

  void makeMap(const std::map<uint64_t, YAML::Node>& m) {
    std::vector<uint64_t> keys;
    for(const auto& it : m)
      keys.push_back(it.first);

    ph_.reset(new PerfectHash(keys));

    array_.resize(ph_->size());
    elements_ = keys.size();

    for(const auto& it : m) {
      uint64_t key = it.first;
      size_t pos = (*ph_)[key];
      array_[pos].reset(new FastOpt(it.second));
      array_[pos]->fingerprint_ = key;
    }

    type_ = NodeType::Map;
  }

  void makeMap(const std::map<std::string, YAML::Node>& m) {
    std::map<uint64_t, YAML::Node> mi;
    for(const auto& it : m) {
      auto key = it.first.c_str();
      mi[crc(key)] = it.second;
    }

    makeMap(mi);
  }

  FastOpt(const FastOpt&) = delete;
  FastOpt() = delete;

public:
  FastOpt(const YAML::Node& node) {
    switch(node.Type()) {
      case YAML::NodeType::Scalar:
        makeScalar(node);
        break;
      case YAML::NodeType::Sequence: {
        std::vector<YAML::Node> nodesVec;
        for(auto&& n : node)
          nodesVec.push_back(n);
        makeSequence(nodesVec);
      } break;
      case YAML::NodeType::Map: {
        std::map<std::string, YAML::Node> nodesMap;
        for(auto& n : node) {
          auto key = n.first.as<std::string>();
          nodesMap[key] = n.second;
        }
        makeMap(nodesMap);
      } break;
      case YAML::NodeType::Undefined:
      case YAML::NodeType::Null:
        makeNull();
    }
  }

  bool isSequence() const {
    return type_ == NodeType::Sequence;
  }

  bool isMap() const {
    return type_ == NodeType::Map;
  }

  bool isScalar() const {
    return type_ == NodeType::Bool 
      || type_ == NodeType::Float64 
      || type_ == NodeType::Int64 
      || type_ == NodeType::String;
  }

  bool isNull() const {
    return type_ == NodeType::Null;
  }  

  bool isInt() const {
    return type_ == NodeType::Int64;
  }  
  
  bool isBool() const {
    return type_ == NodeType::Bool;
  }

  bool isFloat() const {
    return type_ == NodeType::Float64;
  }

  bool isString() const {
    return type_ == NodeType::String;
  }

  size_t size() const {
    return elements_;
  }

  void swap(FastOpt& other) {
    value_.swap(other.value_);
    ph_.swap(other.ph_);
    array_.swap(other.array_);
    std::swap(type_, other.type_);
    std::swap(elements_, other.elements_);
    // leave fingerprint alone as it needed by parent node. @TODO: move to parent in separate array. 
  }

  bool has(size_t keyId) const {
    const auto ptr = phLookup(keyId);
    return ptr ? ptr->fingerprint_ == keyId : false;
  }

  bool has(const char* const key) const {
    return has(crc(key));
  }

  bool has(const std::string& key) const {
    return has(key.c_str());
  }

  template <typename T>
  inline T as() const {
    return helper::As<T>::apply(*this);
  }

  const FastOpt& operator[](size_t keyId) const {
    if(isSequence()) {
      return *arrayLookup(keyId);
    } else if(isMap()) {
      const auto ptr = phLookup(keyId);
      ABORT_IF(!ptr || ptr->fingerprint_ != keyId, 
               "Unseen key {}" , 
               keyId);
      return *ptr; 
    } else {
      ABORT("Not a sequence or map node");
    }
  }

  const FastOpt& operator[](int key) const {
    return operator[]((size_t)key);
  }

  const FastOpt& operator[](const char* const key) const {
    return operator[](crc(key));
  }

  const FastOpt& operator[](const std::string& key) const {
    return operator[](key.c_str());
  }
};

}