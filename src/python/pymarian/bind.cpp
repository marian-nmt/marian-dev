#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "py_translator.h"

namespace py = pybind11;

PYBIND11_MODULE(pymarian, m) {
  // Classes
  py::class_<TranslateServicePyWrapper>(m, "Translator")
      .def(py::init<std::string>())
      .def("translate", py::overload_cast<const std::vector<std::string>&>(&TranslateServicePyWrapper::run))
      .def("translate", py::overload_cast<const std::string&>(&TranslateServicePyWrapper::run))
      .def("translate", py::overload_cast<const std::string&, const py::kwargs&>(&TranslateServicePyWrapper::run));
}
