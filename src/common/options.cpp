#include "options.h"

namespace marian {
  Options::Options() 
  : fastOptions_(options_)
  {
    setLazyRebuild();
  }

  Options::Options(const Options& other)
    : options_(YAML::Clone(other.options_)),
      fastOptions_(options_)
  {}

  Options Options::clone() const {
    return Options(*this); // fastOptions_ get set in constructor above
  }

  // @TODO: use this everywhere instead of above
  YAML::Node Options::getYamlClone() const {
    return YAML::Clone(options_);
  }

  void Options::parse(const std::string& yaml) {
    auto node = YAML::Load(yaml);
    for(auto it : node)
      options_[it.first.as<std::string>()] = YAML::Clone(it.second);
    setLazyRebuild();
  }

  void Options::merge(const YAML::Node& node, bool overwrite) {
    for(auto it : node)
      if(overwrite || !options_[it.first.as<std::string>()])
        options_[it.first.as<std::string>()] = YAML::Clone(it.second);
    setLazyRebuild();
  }

  void Options::merge(Ptr<Options> options) {
    merge(options->options_);
  }

  std::string Options::str() {
    std::stringstream ss;
    ss << options_;
    return ss.str();
  }

  bool Options::hasAndNotEmpty(const std::string& key) const {
    checkLazyRebuild();
    if(!fastOptions_.has(key)) {
      return false;
    } else {
      auto& node = fastOptions_[key];
      if(node.isString())
        return !node.as<std::string>().empty();
      else if(node.isSequence())
        return node.size() != 0;
      else
        return false;
    }
  }

  bool Options::has(const std::string& key) const {
    checkLazyRebuild();
    return fastOptions_.has(key);
  }
}
