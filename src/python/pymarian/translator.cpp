#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "marian.h"
#include "translator/translator.h"
#include "translator/beam_search.h"

namespace py = pybind11;

PYBIND11_MODULE(pymarian, m) {
  // Classes
  py::class_<marian::TranslateService<marian::BeamSearch>>(m, "Translator")
      .def(py::init<std::string>())
<<<<<<< HEAD
      .def("translate", py::overload_cast<const std::vector<std::string>&>(&marian::TranslateService<marian::BeamSearch>::run))
      .def("translate", py::overload_cast<const std::string&>(&marian::TranslateService<marian::BeamSearch>::run));
=======
      .def("translate", py::overload_cast<const std::string&>(&marian::TranslateService<marian::BeamSearch>::run))
      .def("translate", py::overload_cast<const std::vector<std::string>&>(&marian::TranslateService<marian::BeamSearch>::run));
>>>>>>> 703643ae238607a0001653aa4299a8fe35cb21fa
}

