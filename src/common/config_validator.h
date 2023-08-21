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

  // When --dump-config is used, alleviate some constraints, for example, do not
  // require --train-sets or --vocabs
  bool dumpConfigOnly_{false};

  void validateOptionsTranslation() const;
  void validateOptionsParallelData() const;
  void validateOptionsScoring() const;
  void validateOptionsTraining() const;

  void validateModelExtension(cli::mode mode) const;
  void validateDevices(cli::mode mode) const;

public:
  ConfigValidator(const YAML::Node& config);
  ConfigValidator(const YAML::Node& config, bool dumpConfigOnly);
  virtual ~ConfigValidator();

  // Validate options according to the given mode. Abort on first validation error
  void validateOptions(cli::mode mode) const;
};

}  // namespace marian
