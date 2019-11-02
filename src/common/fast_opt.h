#pragma once

#include "common/intrusive_ptr.h"
#include "3rd_party/any_type.h"
#include "3rd_party/phf/phf.h"
#include "3rd_party/yaml-cpp/yaml.h"

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

class FastOpt {
private:
  ENABLE_INTRUSIVE_PTR(FastOpt)

public:
  enum struct NodeType {
    Null, Int64, Float64, String, List, Map
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

  uint64_t fingerprint{0};
  size_t elements_{0};

  inline const IntrusivePtr<FastOpt>& arrayLookup(size_t keyId) const {
    return array_[keyId];
  }

  inline const IntrusivePtr<FastOpt>& phLookup(size_t keyId) const {
    const auto& node = array_[(*ph_)[keyId]];
    return node;
  }

  void makeNull() {
    // std::cerr << "Creating null " << std::endl;
    value_.reset(nullptr);
    elements_ = 0;
    type_ = NodeType::Null;
  }

  template <class V>
  void makeScalar(const V& v) {
    // std::cerr << "Creating scalar " << v << std::endl;

    elements_ = 0;
    value_.reset(new ElementType());

    try {
      value_->value = v.as<int64_t>();
      type_ = NodeType::Int64;
    } catch(...) {
      try {
        value_->value = v.as<double>();
        type_ = NodeType::Float64;
      } catch(...) {
        try { 
          value_->value = v.as<std::string>();
          type_ = NodeType::String;
        } catch (...) {
          ABORT("Could not convert scalar {}", v);
        }
      }
    }
  }

  template <class V>
  void makeList(const std::vector<V>& v) {
    // std::cerr << "Creating list (" << v.size() << ")" << std::endl;

    array_.resize(v.size());
    elements_ = v.size();

    for(size_t pos = 0; pos < v.size(); ++pos) {
      array_[pos].reset(new FastOpt(v[pos]));
      array_[pos]->fingerprint = pos;
    }

    type_ = NodeType::List;
  }

  template <class V>
  void makeMap(const std::map<uint64_t, V>& m) {
    // std::cerr << "Creating map (" << m.size() << ")" << std::endl;

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
      array_[pos]->fingerprint = key;
    }

    type_ = NodeType::Map;
  }

  template <class V>
  void makeMap(const std::map<std::string, V>& m) {
    std::map<uint64_t, V> mi;
    for(const auto& it : m) {
      auto key = it.first.c_str();
      // std::cerr << "k: " << key << std::endl;
      mi[crc(key)] = it.second;
    }

    makeMap(mi);
  }

  template <class T, class V>
  static T convert(T, V value) {
    return (T)value;
  }

  template <class T>
  static T convert(T, const std::string& value) {
    ABORT("Not implemented");
    return T(0);
  }

  template <class V>
  static std::string convert(std::string, V value) {
    return std::to_string(value);
  }

  static std::string convert(std::string, const std::string& value) {
    return value;
  }

  template <typename T>
  struct As {
    static inline T apply(const FastOpt* node) {
      ABORT_IF(node->type_ == NodeType::Null, "Null node has no value");
      ABORT_IF(node->elements_ != 0, "Not a leaf node");

      if(node->type_ == NodeType::Int64)
        return convert(T(), node->value_->as<int64_t>());
      else if(node->type_ == NodeType::Float64)
        return convert(T(), node->value_->as<double>());
      else
        return convert(T(), node->value_->as<std::string>());      
    }
  };

  template <typename T>
  struct As<std::vector<T>> {
    static inline std::vector<T> apply(const FastOpt* node) {
      ABORT_IF(node->type_ != NodeType::List, "Not a list node");
      std::vector<T> seq;
      for(const auto& elem : node->array_)
        seq.push_back(elem->as<T>());
      return seq;
    }
  };

  FastOpt(const FastOpt&) = delete;

  template <class V>
  FastOpt(const std::map<uint64_t, V>& m) {
    makeMap(m);
  }

  template <class V>
  FastOpt(const std::map<std::string, V>& m) {
    makeMap(m);
  }

  template <class V>
  FastOpt(const std::vector<V>& v) {
    makeList(v);
  }

  template <class V>
  FastOpt(const V& v) {
    makeScalar(v);
  }

public:
  FastOpt() {}

  FastOpt(const YAML::Node& node) {
    reset(node);
  }

  void reset(const YAML::Node& node) {
    value_.reset();
    ph_.reset();
    {
      std::vector<IntrusivePtr<FastOpt>> temp;
      array_.swap(temp);
    }
    type_ = NodeType::Null;
    fingerprint = 0;
    elements_ = 0;

    switch(node.Type()) {
      case YAML::NodeType::Scalar:
        makeScalar(node);
        break;
      case YAML::NodeType::Sequence: {
        std::vector<YAML::Node> nodesVec;
        for(auto&& n : node)
          nodesVec.push_back(n);
        makeList(nodesVec);
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

  bool has(size_t keyId) const {
    return phLookup(keyId)->fingerprint == keyId;
  }

  bool has(const char* key) const {
    return has(crc(key));
  }

  bool has(const std::string& key) const {
    return has(crc(key.c_str()));
  }

  template <typename T>
  inline T as() const {
    return As<T>::apply(this);
  }

  // @TODO: missing specialization for as<std::vector<T>>()

  const FastOpt& operator[](size_t keyId) const {
    switch(type()) {
      case NodeType::List : return *arrayLookup(keyId);
      case NodeType::Map  :
        ABORT_IF(phLookup(keyId)->fingerprint != keyId, "Unseen key {}" , keyId);
        return *phLookup(keyId);
      default:
        ABORT("Not a map or list node");
    }
  }

  const FastOpt& operator[](const char* const key) const {
    return operator[](crc(key));
  }

  const FastOpt& operator[](const std::string& key) const {
    return operator[](crc(key.c_str()));
  }

  const FastOpt& operator[](int key) const {
    return operator[]((size_t)key);
  }

  const NodeType& type() const {
    return type_;
  }

  size_t size() const {
    return elements_;
  }
};
