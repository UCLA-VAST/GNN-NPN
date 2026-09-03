#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "aig.hpp"
#include "apply.hpp"
#include "cut.hpp"
#include "npn.hpp"
#include "rewrite.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mockturtle_wrapper, m) {
  m.doc() = "Mockturtle wrapper for Python.";
  m.attr("__version__") = "dev";
  m.def(
    "aig_to_tt", &aig_to_tt,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Convert an AIG file to a truth table."
  );
  m.def(
    "apply_to_tt", &apply_to_tt,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Apply NPN transformation to a truth table."
  );
  m.def(
    "exact_npn", &exact_npn,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Exact NPN Canonization."
  );
  m.def(
    "exact_n", &exact_n,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Exact N Canonization."
  );
  m.def(
    "exact_p", &exact_p,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Exact P Canonization."
  );
  m.def(
    "flip_swap_npn", &flip_swap_npn,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Flip-swap Heuristic NPN Canonization."
  );
  m.def(
    "sifting_npn", &sifting_npn,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Sifting Heuristic NPN Canonization."
  );
  m.def(
    "cut_aig", &cut_aig,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Extract reconvergence-driven cuts from an AIG."
  );
  m.def(
    "rewrite_aig", &rewrite_aig,
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    "Rewrite the AIG."
  );
  m.def(
    "generate_random_aig",
    []( uint32_t num_pis, double mean_nodes, double std_nodes, uint64_t seed ) {
      return generate_random_aig( num_pis, mean_nodes, std_nodes, seed );
    },
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    py::arg("num_pis"),
    py::arg("mean_nodes") = 0.0, py::arg("std_nodes") = 0.0, py::arg("seed") = 0,
    "Generate a random AIG with exactly num_pis inputs whose final AND-gate "
    "count (excluding PIs and xdata NOT nodes) follows N(mean_nodes, std_nodes) "
    "via rejection sampling on the proposal. "
    "If mean_nodes <= 0, all candidates are accepted (no shaping). "
    "Returns (x_data, edge_index, tt_hex)."
  );
  m.def(
    "generate_random_aig_rewriting",
    []( uint32_t num_pis, double mean_nodes, double std_nodes, uint64_t seed ) {
      return generate_random_aig_rewriting( num_pis, mean_nodes, std_nodes, seed );
    },
    pybind11::call_guard<pybind11::gil_scoped_release>(),
    py::arg("num_pis"),
    py::arg("mean_nodes") = 0.0, py::arg("std_nodes") = 0.0, py::arg("seed") = 0,
    "Like generate_random_aig but rewrites the AIG before returning. The "
    "AND-gate count acceptance test runs on the pre-rewrite AIG, so the "
    "sampled distribution is identical to generate_random_aig. "
    "Returns (x_data, edge_index, tt_hex) of the rewritten AIG."
  );
}
