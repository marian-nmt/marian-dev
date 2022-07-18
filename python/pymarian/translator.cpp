#include <pybind11/pybind11.h>
#include "translator/translator.h"

namespace py = pybind11;

PYBIND11_MODULE(_torchtext, m) {
  // Classes
  py::class_<marian::TranslateService>(m, "Translator")
      .def(py::init<std::string>())
      .def("translate", &TranslateService::run);
}

