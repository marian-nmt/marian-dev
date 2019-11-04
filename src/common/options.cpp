#include "options.h"

namespace marian {
  Options::Options() {}

  Options::Options(const Options& other)
    : options_(YAML::Clone(other.options_)) {}

  Options Options::clone() const {
    return Options(*this);
  }

  YAML::Node& Options::getYaml() {
    ABORT_IF(fixed_, "Options fixed and cannot be modified unless cloned");
    return options_;
  }

  const YAML::Node& Options::getYaml() const {
    return options_;
  }

  void Options::parse(const std::string& yaml) {
    ABORT_IF(fixed_, "Options fixed and cannot be modified unless cloned");
    auto node = YAML::Load(yaml);
    for(auto it : node)
      options_[it.first.as<std::string>()] = YAML::Clone(it.second);
  }

  void Options::merge(const YAML::Node& node, bool overwrite) {
    ABORT_IF(fixed_, "Options fixed and cannot be modified unless cloned");
    for(auto it : node)
      if(overwrite || !options_[it.first.as<std::string>()])
        options_[it.first.as<std::string>()] = YAML::Clone(it.second);
  }

  void Options::merge(Ptr<Options> options) {
    ABORT_IF(fixed_, "Options fixed and cannot be modified unless cloned");
    merge(options->getYaml());
  }

  std::string Options::str() {
    std::stringstream ss;
    ss << options_;
    return ss.str();
  }

  bool Options::hasAndNotEmpty(const std::string& key) const {
    if(!has(key)) {
      return false;
    }
    if(fixed_ ? fastOptions_[key].type() == FastOpt::NodeType::List : options_[key].IsSequence()) {
      return fixed_ ? fastOptions_[key].size() != 0 : options_[key].size() != 0;
    }
    try {
      return fixed_ ? !fastOptions_[key].as<std::string>().empty() : !options_[key].as<std::string>().empty();
    } catch(const YAML::BadConversion& /* e */) {
      ABORT("Option '{}' is neither a sequence nor text");
    }
    return false;
  }

  bool Options::has(const std::string& key) const {
    return fixed_ ? fastOptions_.has(key) : options_[key];
  }
}
