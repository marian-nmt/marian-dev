#include <pybind11/pybind11.h>
#include "translator/translator.h"
#include "translator/beam_search.h"

namespace py = pybind11;

namespace marian {
  class BeamTranslateService : public TranslateService<BeamSearch> {};
};

PYBIND11_MODULE(pymarian, m) {
  // Classes
  py::class_<marian::BeamTranslateService>(m, "BeamTranslator")
      .def(py::init<std::string>())
      .def("translate", &marian::BeamTranslateService::run);
}

