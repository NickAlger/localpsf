

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>

class Profile : public dolfin::Expression
{
public:

  std::shared_ptr<dolfin::Function> u;

  Profile(std::shared_ptr<dolfin::Function> u_) : dolfin::Expression(2){
    u = u_;
  }

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
  {
    u->eval(values, x);
    const double val = values[0];
    values[0] = 0.0;
    values[1] = val;
  }

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Profile, std::shared_ptr<Profile>, dolfin::Expression>
    (m, "Profile")
    .def(py::init<std::shared_ptr<dolfin::Function> >())
    .def_readwrite("u", &Profile::u);
}

