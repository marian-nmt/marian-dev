#include "options.h"

namespace marian {

// name space for helper template specializations
namespace options_helpers {

// Generic template-based implementation
template <class T> 
T Get<T>::apply(const Options* opt, const char* const key) {
#if FASTOPT
  opt->lazyRebuild();
  ABORT_IF(!opt->has(key), "Required option '{}' has not been set", key);
  return opt->fastOptions_[key].as<T>();
#else
  ABORT_IF(!opt->has(key), "Required option '{}' has not been set", key);
  return opt->options_[key].as<T>();
#endif
}

// Generic template-based implementation
template <class T> 
T Get<T>::apply(const Options* opt, const char* const key, const T& defaultValue) {
#if FASTOPT
  opt->lazyRebuild();
  if(opt->has(key))
    return opt->fastOptions_[key].as<T>();
#else
  if(opt->has(key))
    return opt->options_[key].as<T>();
#endif
  else
    return defaultValue;
}

// specializations for simple types
template struct Get<bool>;
template struct Get<int>;
template struct Get<unsigned long>;
template struct Get<unsigned long long>;
template struct Get<float>;
template struct Get<double>;
template struct Get<std::string>;

// specialization for vector of simple types
template struct Get<std::vector<bool>>;
template struct Get<std::vector<int>>;
template struct Get<std::vector<unsigned long long>>;
template struct Get<std::vector<unsigned long>>;
template struct Get<std::vector<float>>;
template struct Get<std::vector<double>>;
template struct Get<std::vector<std::string>>;

// specializations for std::vector<YAML::Node>
template <>
std::vector<YAML::Node> Get<std::vector<YAML::Node>>::apply(const Options* opt, const char* const key) {
  ABORT_IF(!opt->has(key), "Required option '{}' has not been set", key);
  auto vec = opt->options_[key].as<std::vector<YAML::Node>>();
  for(auto& node : vec)  {
    if(node.IsScalar())
      node = YAML::Load(node.as<std::string>());
  }
  return vec;
}

template <>
std::vector<YAML::Node> Get<std::vector<YAML::Node>>::apply(const Options* opt, const char* const key, const std::vector<YAML::Node>& defaultValue) {
  if(opt->has(key))
    return apply(opt, key);
  return defaultValue;
}

template struct Get<std::vector<YAML::Node>>;

// specializations for YAML::Node
template <>
YAML::Node Get<YAML::Node>::apply(const Options* opt, const char* const key) {
  ABORT_IF(!opt->has(key), "Required option '{}' has not been set", key);
  YAML::Node node = opt->options_[key];
  if(node.IsScalar())
    node = YAML::Load(node.as<std::string>());
  return node;
}

template <>
YAML::Node Get<YAML::Node>::apply(const Options* opt, const char* const key, const YAML::Node& defaultValue) {
  if(opt->has(key))
    return apply(opt, key);
  return defaultValue;
}

template struct Get<YAML::Node>;
}

Options::Options()
#if FASTOPT
  : fastOptions_(options_)
#endif
{}

Options::Options(const Options& other)
#if FASTOPT
  : options_(YAML::Clone(other.options_)),
    fastOptions_(options_) {}
#else
  : options_(YAML::Clone(other.options_)) {}
#endif

Ptr<Options> Options::clone() const {
  return New<Options>(*this); // fastOptions_ get set in constructor above
}

YAML::Node Options::cloneToYamlNode() const {
  return YAML::Clone(options_); // Do not give access to internal YAML object
}

void Options::parse(const std::string& yaml) {
  auto node = YAML::Load(yaml);
  for(auto it : node)
    options_[it.first.as<std::string>()] = YAML::Clone(it.second);
#if FASTOPT
  setLazyRebuild();
#endif
}

void Options::merge(const YAML::Node& node, bool overwrite) {
  for(auto it : node)
    if(overwrite || !options_[it.first.as<std::string>()])
      options_[it.first.as<std::string>()] = YAML::Clone(it.second);
#if FASTOPT
  setLazyRebuild();
#endif
}

void Options::merge(Ptr<Options> options) {
  merge(options->options_);
}

std::string Options::asYamlString() {
  std::stringstream ss;
  ss << options_;
  return ss.str();
}

bool Options::hasAndNotEmpty(const char* const key) const {
#if FASTOPT
  lazyRebuild();
  if(!fastOptions_.has(key)) {
    return false;
  } else {
    auto& node = fastOptions_[key];
    if(node.isSequence())
      return node.size() != 0;
    else if(node.isScalar()) // numerical values count as non-empty
      return !node.as<std::string>().empty();
    else
      ABORT("Wrong node type");
  }
#else
  if(!options_[key]) {
    return false;
  } else {
    auto& node = options_[key];
    if(node.IsSequence())
      return node.size() != 0;
    else if(node.IsScalar()) // numerical values count as non-empty
      return !node.as<std::string>().empty();
    else
      ABORT("Wrong node type");
  }
#endif
}

bool Options::hasAndNotEmpty(const std::string& key) const {
  return hasAndNotEmpty(key.c_str());
}

bool Options::has(const char* const key) const {
#if FASTOPT
  lazyRebuild();
  return fastOptions_.has(key);
#else
  return options_[key];
#endif
}

bool Options::has(const std::string& key) const {
  return has(key.c_str());
}

}
