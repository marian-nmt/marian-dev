#include "pybind11/pybind11.h"

#include "marian.h"
#include "translator/translator.h"
#include "translator/beam_search.h"

namespace py = pybind11;

PYBIND11_MODULE(pymarian, m) {
  // Classes
  py::class_<marian::TranslateService<marian::BeamSearch>>(m, "Translator")
      .def(py::init<std::string>())
      .def("translate", &marian::TranslateService<marian::BeamSearch>::run);
}

