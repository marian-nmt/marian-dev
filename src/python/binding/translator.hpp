#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "marian.h"

#include "common/logging.h"
#include "common/timer.h"
#include "evaluator/evaluator.h"
#include "models/model_task.h"
#include "translator/beam_search.h"
#include "translator/translator.h"


namespace py = pybind11;
using namespace marian;

namespace pymarian {

  class TranslateServicePyWrapper {
  private:
    Ptr<TranslateService<BeamSearch>> pImpl_;

    /**
     * @brief Convert a pybind11::kwargs object to a YAML string
     * 
     * @param kwargs - the kwargs object from pybind11
     * @return std::string - the YAML string
     */
    std::string convertKwargsToYamlString(const py::kwargs& kwargs) {
      std::stringstream ss;
      if (kwargs) {
        for (auto& [key, value] : kwargs) {
          // Depythonize the keys
          std::string yamlKey = utils::findReplace(key.cast<std::string>(), "_", "-");
          ss << yamlKey << ": " << value << std::endl;
        }
      }
      return ss.str();
    }

  public:
    TranslateServicePyWrapper(const std::string& cliString)
    : pImpl_(New<TranslateService<BeamSearch>>(cliString)) {}

    /**
     * @brief Translate a vector of strings
     * 
     * @param inputs - the vector of strings to translate
     * @param kwargs - the kwargs object from pybind11
     * @return std::vector<std::string> - the vector of translated strings
     */
    std::vector<std::string> run(const std::vector<std::string>& inputs, const py::kwargs& kwargs) {
      return this->pImpl_->run(inputs, convertKwargsToYamlString(kwargs));
    }

    /**
     * @brief Translate a single string
     * 
     * @param input - the string to translate
     * @param kwargs - the kwargs object from pybind11
     * @return std::string - the translated string
     */
    std::string run(const std::string& input, const py::kwargs& kwargs) {
      return this->pImpl_->run(input, convertKwargsToYamlString(kwargs));
    }
  };

}

