#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "translator.hpp"
#include "evaluator.hpp"

#define PYBIND11_DETAILED_ERROR_MESSAGES

namespace py = pybind11;
using namespace pymarian;

PYBIND11_MODULE(_pymarian, m) {
    //py::module m = base.def_submodule("capi", "Marian CAPI");   //pymarian.capi

    // Classes
    py::class_<TranslateServicePyWrapper>(m, "Translator")
        .def(py::init<std::string>())
        .def("translate", py::overload_cast<const std::string&, const py::kwargs&>(&TranslateServicePyWrapper::run))
        .def("translate", py::overload_cast<const std::vector<std::string>&, const py::kwargs&>(&TranslateServicePyWrapper::run));

    py::class_<EvaluatorPyWrapper>(m, "Evaluator")
        .def(py::init<std::string>())
        //.def("run", py::overload_cast<const StrVector&>(&EvaluatorPyWrapper::run))
        .def("run", py::overload_cast<const StrVectors&>(&EvaluatorPyWrapper::run));
}

