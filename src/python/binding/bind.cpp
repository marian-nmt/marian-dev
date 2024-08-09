#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
// if your IDE/vscode complains about missing paths
// pybind11 can be found by "python -m pybind11 --includes"; you may need to add both pybind11 and Python.h
#include "embedder.hpp"
#include "evaluator.hpp"
#include "trainer.hpp"
#include "translator.hpp"
#include "command/marian_main.cpp"

namespace py = pybind11;
using namespace pymarian;

/**
 * @brief Wrapper function to call Marian main entry point from Python
 *
 * Calls Marian main entry point from Python.
 * It converts args from a vector of strings (Python-ic API) to char* (C API)
 *  before passsing on to the main function.
 * @param args vector of strings
 * @return int return code
 */
int main_wrap(std::vector<std::string> args) {
    // Convert vector of strings to vector of char*
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("pymarian"));
    for (auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    return main(argv.size() - 1, argv.data());
}

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
        .def("get_model_config", py::overload_cast<>(&EvaluatorPyWrapper::getModelConfig))
        ;

    py::class_<PyTrainer>(m, "Trainer")
        .def(py::init<std::string>())
        .def("train", py::overload_cast<>(&PyTrainer::train))
        ;

      py::class_<PyEmbedder>(m, "Embedder")
        .def(py::init<std::string>())
        .def("embed", py::overload_cast<>(&PyEmbedder::embed))
        ;

    m.def("main", &main_wrap, "Marian main entry point");

}

