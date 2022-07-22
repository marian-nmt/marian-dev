#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "marian.h"
#include "translator/translator.h"
#include "translator/beam_search.h"

namespace py = pybind11;

class TranslateServicePyWrapper {
private:
  marian::Ptr<marian::TranslateService<marian::BeamSearch>> pImpl_;

  std::string convertKwargsToYamlString(const py::kwargs& kwargs) {
    std::stringstream ss;
    if (kwargs) {
      for (auto& [key, value]: kwargs) {
        // Depythonize the keys
        std::string yamlKey = marian::utils::findReplace(key.cast<std::string>(), "_", "-");
        ss << yamlKey << ": " << value << std::endl;
      }
    }
    return ss.str();
  }

public:
  TranslateServicePyWrapper(const std::string& cliString) : pImpl_(marian::New<marian::TranslateService<marian::BeamSearch>>(cliString)) {}

  std::vector<std::string> run(const std::vector<std::string>& inputs, const py::kwargs& kwargs) {
    return this->pImpl_->run(inputs, convertKwargsToYamlString(kwargs));
  }

  std::string run(const std::string& input, const py::kwargs& kwargs) {
    return this->pImpl_->run(input, convertKwargsToYamlString(kwargs));
  }
};