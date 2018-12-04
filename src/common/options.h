#pragma once

#include "common/definitions.h"
#include "common/fast_opt.h"

#include "3rd_party/yaml-cpp/yaml.h"

#include <sstream>
#include <string>

#define YAML_REGISTER_TYPE(registered, type)                \
  namespace YAML {                                          \
  template <>                                               \
  struct convert<registered> {                              \
    static Node encode(const registered& rhs) {             \
      type value = static_cast<type>(rhs);                  \
      return Node(value);                                   \
    }                                                       \
    static bool decode(const Node& node, registered& rhs) { \
      type value = node.as<type>();                         \
      rhs = static_cast<registered>(value);                 \
      return true;                                          \
    }                                                       \
  };                                                        \
  }

namespace marian {

/**
 * Container for options stored as key-value pairs. Keys are unique strings.
 */
class Options {
protected:
  YAML::Node options_;
  std::unique_ptr<FastOpt> fastOptions_;

public:
  Options() {}
  Options(const Options& other)
  : options_(YAML::Clone(other.options_)),
    fastOptions_(new FastOpt(options_)) {}

  /**
   * @brief Return a copy of the object that can be safely modified.
   */
  Options clone() const { return Options(*this); }

  YAML::Node& getYaml() {
    ABORT_IF(fastOptions_, "YAML should not be modified");
    return options_;
  } // this should be removed

  const YAML::Node& getYaml() const {
    return options_;
  }

  void parse(const std::string& yaml) {
    auto node = YAML::Load(yaml);
    for(auto it : node)
      options_[it.first.as<std::string>()] = YAML::Clone(it.second);
    fastOptions_.reset(new FastOpt(options_));
  }

  /**
   * @brief Splice options from a YAML node
   *
   * By default, only options with keys that do not already exist in options_ are extracted from
   * node. These options are cloned if overwirte is true.
   *
   * @param node a YAML node to transfer the options from
   * @param overwrite overwrite all options
   */
  void merge(YAML::Node& node, bool overwrite = false) {
    for(auto it : node)
      if(overwrite || !options_[it.first.as<std::string>()])
        options_[it.first.as<std::string>()] = YAML::Clone(it.second);
    fastOptions_.reset(new FastOpt(options_));
  }

  void merge(const YAML::Node& node, bool overwrite = false) { merge(node, overwrite); }
  void merge(Ptr<Options> options) { merge(options->getYaml()); }

  std::string str() const {
    std::stringstream ss;
    ss << options_;
    return ss.str();
  }

  template <typename T>
  void set(const std::string& key, T value) {
    options_[key] = value;
    fastOptions_.reset(new FastOpt(options_));
  }

  template <typename T>
  T get(const std::string& key) const {
    ABORT_IF(!fastOptions_->has(key.c_str()), "Required option '{}' has not been set", key);
    return (*fastOptions_)[key.c_str()].as<T>();
  }

  template <typename T>
  T get(const std::string& key, T defaultValue) const {
    if(fastOptions_->has(key.c_str()))
      return (*fastOptions_)[key.c_str()].as<T>();
    else
      return defaultValue;
  }

  bool has(const std::string& key) const { return fastOptions_->has(key.c_str()); }
};

}  // namespace marian
