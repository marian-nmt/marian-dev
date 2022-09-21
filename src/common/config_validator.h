#pragma once

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/config_parser.h"

namespace marian {

class ConfigValidator {
private:
  const YAML::Node& config_;

  bool has(const std::string& key) const;
  template <typename T>
  T get(const std::string& key) const {
    return config_[key].as<T>();
  }
  // Return value for given option key cast to given type. Return the supplied
  // default value if option is not set.
  template <typename T>
  T get(const std::string& key, T defaultValue) const {
    if(has(key))
      return config_[key].as<T>();
    else
      return defaultValue;
  }

  // The option --dump-config is used, so alleviate some constraints, e.g. we don't want to require
  // --train-sets or --vocabs
  bool dumpConfigOnly_{false};

  void validateOptionsTranslation() const;
  void validateOptionsVocabularies() const;
  void validateOptionsParallelData() const;
  void validateOptionsScoring() const;
  void validateOptionsTraining() const;

  void validateModelExtension(cli::mode mode) const;
  void validateDevices(cli::mode mode) const;

public:
  ConfigValidator(const YAML::Node& config);
  virtual ~ConfigValidator();

  // Validate options according to the given mode. Abort on first validation error
  void validateOptions(cli::mode mode) const;
};

}  // namespace marian
