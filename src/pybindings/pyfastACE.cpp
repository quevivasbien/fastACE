#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <Eigen/Dense>
#include "base.h"

namespace py = pybind11;

namespace pybindHelpers {
    std::shared_ptr<Person> create_person(
        Economy* economy,
        std::vector<double> inventory,
        double money
    ) {
        Eigen::Map<Eigen::ArrayXd> inventory_(inventory.data(), inventory.size());
        return Person::init(economy, inventory_, money);
    }

    const std::vector<double> get_inventory(std::shared_ptr<Agent> agent) {
        return eigenToVector(agent->get_inventory());
    }
}


PYBIND11_MODULE(pyfastACE, m) {
    py::class_<Economy>(m, "Economy")
        .def(py::init<std::vector<std::string>>())
        .def("time_step", &Economy::time_step)
        .def_property("time", &Economy::get_time, nullptr)
        .def("get_name_for_good_id", &Economy::get_name_for_good_id)
        .def_property("persons", &Economy::get_persons, nullptr)
        .def_property("firms", &Economy::get_firms, nullptr)
        .def_property("goods", &Economy::get_goods, nullptr)
        .def_property("numGoods", &Economy::get_numGoods, nullptr)
        // get market methods currently missing
        .def("print_summary", &Economy::print_summary);
    py::class_<Person, std::shared_ptr<Person>>(m, "Person")
        .def(py::init(&pybindHelpers::create_person))
        .def("time_step", &Person::time_step)
        .def_property("economy", &Person::get_economy, nullptr)
        .def_property("money", &Person::get_money, nullptr)
        // .def_property("inventory", [](py::object self){ return pybindHelpers::get_inventory(py::object); }, nullptr)
        .def("print_summary", &Person::print_summary)
        .def_property("laborSupplied", &Person::get_laborSupplied, nullptr);
}
