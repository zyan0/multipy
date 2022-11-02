#include <multipy/runtime/deploy.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/lazy/core/debug_util.h>

namespace py = pybind11;

using namespace torch::deploy;

PYBIND11_MODULE(multipy_pybind, m) {
  m.doc() = "pybind11 example plugin";  // optional module docstring

  py::class_<InterpreterManager>(m, "InterpreterManager")
      .def(py::init<size_t>())
      .def("acquire_one", &InterpreterManager::acquireOne)
      .def("all_instances", &InterpreterManager::allInstances);

  py::class_<Interpreter>(m, "Interpreter")
      .def("acquire_session", &Interpreter::acquireSession);

  py::class_<InterpreterSession>(m, "InterpreterSession")
      .def("global", &InterpreterSession::global);

  py::class_<Obj>(m, "Obj")
      .def("__call__",
           [](Obj& self, py::args args, const py::kwargs& kwargs) -> Obj {
             std::vector<at::IValue> iargs;
             std::unordered_map<std::string, at::IValue> ikwargs;

             for (auto& arg : args) {
               iargs.emplace_back(torch::jit::toTypeInferredIValue(arg));
             }
             for (auto& arg : kwargs) {
               ikwargs.emplace(arg.first.cast<std::string>(),
                               torch::jit::toTypeInferredIValue(arg.second));
             }

             return self.callKwargs(iargs, ikwargs);
           })
      .def("deref", [](Obj& self) -> py::object {
        return ::torch::jit::toPyObject(self.toIValue());
      });
}
