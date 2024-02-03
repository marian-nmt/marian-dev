#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
// if your IDE/vscode complains about missing paths 
// pybind11 can be found by "python -m pybind11 --includes"; you may need to add both pybind11 and Python.h
#include "embedder.hpp"
#include "evaluator.hpp"
#include "trainer.hpp"
#include "translator.hpp"


#define PYBIND11_DETAILED_ERROR_MESSAGES

namespace py = pybind11;
using namespace pymarian;


PYBIND11_MODULE(_pymarian, m) {
    m.doc() = "Marian C++ API bindings via pybind11";

    /** TODOS 
     *  1. API to check if gpu available: cuda_is_available() -> bool
     *  2. API to check number of gpus:: cuda_device_count() -> int
    */

    py::class_<TranslateServicePyWrapper>(m, "Translator")
        .def(py::init<std::string>())
        .def("translate", py::overload_cast<const std::string&, const py::kwargs&>(&TranslateServicePyWrapper::run))
        .def("translate", py::overload_cast<const std::vector<std::string>&, const py::kwargs&>(&TranslateServicePyWrapper::run))
        ;

    py::class_<EvaluatorPyWrapper>(m, "Evaluator")
        .def(py::init<std::string>())
        .def("evaluate", py::overload_cast<const StrVectors&>(&EvaluatorPyWrapper::run))
        ;

    py::class_<PyTrainer>(m, "Trainer")
        .def(py::init<std::string>())
        .def("train", py::overload_cast<>(&PyTrainer::train))
        ;

      py::class_<PyEmbedder>(m, "Embedder")
        .def(py::init<std::string>())
        .def("embed", py::overload_cast<>(&PyEmbedder::embed))
        ;

}

