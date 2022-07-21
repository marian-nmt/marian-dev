#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "marian.h"
#include "translator/translator.h"
#include "translator/beam_search.h"

namespace py = pybind11;

class TranslateServicePyWrapper {
private:
  marian::Ptr<marian::TranslateService<marian::BeamSearch>> pImpl_;

public:
  TranslateServicePyWrapper(const std::string& cliString) : pImpl_(marian::New<marian::TranslateService<marian::BeamSearch>>(cliString)) {}

  std::vector<std::string> run(const std::vector<std::string>& inputs) {
    return this->pImpl_->run(inputs);
  }

  std::string run(const std::string& input) {
    return this->pImpl_->run(input);
  }

  std::string run(const std::string& input, const py::kwargs& kwargs) {
    if (kwargs) {
      std::cout << "kwargs: " << kwargs << std::endl;
    }

    // ignoring it for now
    return this->pImpl_->run(input);
  }

  // translate(string, keywords) {

  //   pImpl_->translate(std::string, string)
  // }

};