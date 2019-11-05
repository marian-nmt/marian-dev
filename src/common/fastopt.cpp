#include "common/fastopt.h"

namespace marian {

namespace helper {

template <typename To, typename From>
struct Convert { 
  static inline To apply(const From& from) { 
    return (To)from; 
  } 
};

template <typename To>
struct Convert<To, std::string> { 
  static inline To apply(const std::string& from) { 
    ABORT("Not implemented");
  }
};

template <typename From>
struct Convert<std::string, From> { 
  static inline std::string apply(const From& from) { 
    return std::to_string(from);
  }
};

template <>
struct Convert<std::string, std::string> { 
  static inline std::string apply(const std::string& from) { 
    return from;
  }
};

template <typename T>
T As<T>::apply(const FastOpt& node) {
  ABORT_IF(!node.isScalar(), "Node is not a scalar node");

  if(node.isBool())
    return Convert<T, bool>::apply(node.value_->as<bool>());
  else if(node.isInt())
    return Convert<T, int64_t>::apply(node.value_->as<int64_t>());
  else if(node.isFloat())
    return Convert<T, double>::apply(node.value_->as<double>());
  else if(node.isString())
    return Convert<T, std::string>::apply(node.value_->as<std::string>());      
  else {
    ABORT("Casting of value failed");
  }
}

template <typename T>
std::vector<T> As<std::vector<T>>::apply(const FastOpt& node) {
  ABORT_IF(!node.isSequence(), "Node is not a sequence node");

  std::vector<T> seq;
  for(const auto& elem : node.array_)
    seq.push_back(elem->as<T>());
  return seq;
}

template struct As<bool>;
template struct As<int>;
template struct As<unsigned long>;
template struct As<float>;
template struct As<double>;
template struct As<std::string>;

template struct As<std::vector<bool>>;
template struct As<std::vector<int>>;
template struct As<std::vector<unsigned long>>;
template struct As<std::vector<float>>;
template struct As<std::vector<double>>;
template struct As<std::vector<std::string>>;

}
}