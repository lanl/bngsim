// bngsim/src/_bngsim_core.cpp — pybind11 bindings for bngsim
//
// Wraps the C++ engine as a Python extension module `_bngsim_core`.
// The high-level Python API (Model, Simulator, Result) lives in python/bngsim/
// and delegates to these bindings.
//
// Exposes NumPy-friendly bindings, using zero-copy transfers where possible.
// GIL is released during simulation for parallel-friendly execution.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <bngsim/bngsim.hpp>
#include <bngsim/function_columns.hpp>
#include <bngsim/cc_jit.hpp>
#include <bngsim/mir_jit.hpp>
#include <bngsim/steady_state.hpp>
#include <bngsim/wallclock.hpp>

#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;

// ─── Helper: convert flat row-major vector to 2D NumPy array ─────────────────
//
// Zero-copy when possible: we move the data into a new vector owned by the
// capsule attached to the NumPy array. The original Result object can be freed.
static py::array_t<double> vec_to_ndarray_2d(const std::vector<double> &data, int n_rows,
                                             int n_cols) {
    if (data.empty() || n_rows == 0 || n_cols == 0) {
        return py::array_t<double>({n_rows, n_cols});
    }
    // Copy data into a heap-allocated vector that the capsule will own
    auto *vec = new std::vector<double>(data);
    auto capsule = py::capsule(vec, [](void *p) { delete static_cast<std::vector<double> *>(p); });
    return py::array_t<double>({n_rows, n_cols},         // shape
                               {n_cols * sizeof(double), // strides (row-major)
                                sizeof(double)},
                               vec->data(), // data pointer
                               capsule      // prevent deallocation
    );
}

// .gdat/.scan function-column convention: see bngsim/function_columns.hpp.
// bngsim's function enumerator collects all functions (including the internal
// _rateLawN entries), so the default Result expression columns are filtered at
// this boundary — downstream consumers use them as in-memory column keys. Names
// are always bare (no "()"); the _rateLawN columns are recoverable via the
// raw_expression_* accessors and selectable at write time with print_rate_laws.
using bngsim::is_auto_rate_law;
using bngsim::user_function_indices;

// ─── Helper: convert 1D vector to NumPy array ────────────────────────────────
static py::array_t<double> vec_to_ndarray_1d(const std::vector<double> &data) {
    if (data.empty()) {
        return py::array_t<double>(0);
    }
    auto *vec = new std::vector<double>(data);
    auto capsule = py::capsule(vec, [](void *p) { delete static_cast<std::vector<double> *>(p); });
    return py::array_t<double>({static_cast<py::ssize_t>(vec->size())}, {sizeof(double)},
                               vec->data(), capsule);
}

// ─── Helper: reshape a stored output-sensitivity buffer to a 3D NumPy array ───
// GH #196. The buffer is row-major [time][row][depth]; the row count is
// recovered from the buffer size (data.size() / (n_times * depth)) since the
// Result class does not track it separately. An empty/unpopulated buffer yields
// an empty (0,0,0) array, matching the species-sensitivity accessors.
static py::array_t<double> sens_block_to_ndarray_3d(const std::vector<double> &data, int n_times,
                                                    int depth) {
    if (data.empty() || n_times == 0 || depth == 0) {
        return py::array_t<double>({0, 0, 0});
    }
    int n_rows = static_cast<int>(data.size() / (static_cast<size_t>(n_times) * depth));
    auto *vec = new std::vector<double>(data);
    auto capsule = py::capsule(vec, [](void *p) { delete static_cast<std::vector<double> *>(p); });
    return py::array_t<double>(
        {n_times, n_rows, depth},
        {static_cast<py::ssize_t>(static_cast<size_t>(n_rows) * depth * sizeof(double)),
         static_cast<py::ssize_t>(static_cast<size_t>(depth) * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        vec->data(), capsule);
}

// Like sens_block_to_ndarray_3d but projects the middle (row) axis to ``keep``.
// The expression output-sensitivity blocks (GH #198) are recorded with one row
// per *raw* function (including the auto-generated _rateLawN intermediates); this
// drops those rows so the returned block aligns with the filtered
// ``expression_names`` a selector indexes into — exactly mirroring how
// ``expression_data`` filters the value columns.
static py::array_t<double> sens_block_to_ndarray_3d_rows(const std::vector<double> &data,
                                                         int n_times, int depth,
                                                         const std::vector<size_t> &keep) {
    if (data.empty() || n_times == 0 || depth == 0 || keep.empty()) {
        return py::array_t<double>({0, 0, 0});
    }
    const int n_rows_raw = static_cast<int>(data.size() / (static_cast<size_t>(n_times) * depth));
    const int n_rows = static_cast<int>(keep.size());
    auto *vec = new std::vector<double>();
    vec->reserve(static_cast<size_t>(n_times) * n_rows * depth);
    for (int t = 0; t < n_times; ++t) {
        const size_t tbase = static_cast<size_t>(t) * n_rows_raw * depth;
        for (size_t r : keep) {
            const size_t rbase = tbase + r * static_cast<size_t>(depth);
            for (int d = 0; d < depth; ++d) {
                vec->push_back(data[rbase + static_cast<size_t>(d)]);
            }
        }
    }
    auto capsule = py::capsule(vec, [](void *p) { delete static_cast<std::vector<double> *>(p); });
    return py::array_t<double>(
        {n_times, n_rows, depth},
        {static_cast<py::ssize_t>(static_cast<size_t>(n_rows) * depth * sizeof(double)),
         static_cast<py::ssize_t>(static_cast<size_t>(depth) * sizeof(double)),
         static_cast<py::ssize_t>(sizeof(double))},
        vec->data(), capsule);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Module definition
// ═══════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_bngsim_core, m) {
    m.doc() = "bngsim C++ engine bindings (low-level)";

    // Note: Exception types are defined in Python (_exceptions.py), not here.
    // The Python wrapper layer catches C++ std::runtime_error / std::invalid_argument
    // and re-raises as bngsim.ModelError / ParameterError / SimulationError.
    //
    // Exception: bngsim::TimeoutError. The Python-side `SimulationTimeout`
    // class carries `timeout` and `elapsed` attributes which the wrapper
    // layer can't reconstruct from a generic RuntimeError message — those
    // would round-trip lossily. We translate it directly so the attributes
    // survive. `bngsim._exceptions` is imported lazily inside the lambda so
    // module construction order remains free of cycles.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const bngsim::TimeoutError &e) {
            py::object cls = py::module_::import("bngsim._exceptions").attr("SimulationTimeout");
            py::object exc = cls(py::str(e.what()), py::arg("timeout") = e.timeout_seconds(),
                                 py::arg("elapsed") = e.elapsed_seconds());
            PyErr_SetObject(cls.ptr(), exc.ptr());
        }
    });

    // ─── SolverStats ─────────────────────────────────────────────────────────
    py::class_<bngsim::SolverStats>(m, "SolverStats")
        .def(py::init<>())
        .def_readwrite("n_steps", &bngsim::SolverStats::n_steps)
        .def_readwrite("n_rhs_evals", &bngsim::SolverStats::n_rhs_evals)
        .def_readwrite("n_jac_evals", &bngsim::SolverStats::n_jac_evals)
        .def_readwrite("n_err_test_fails", &bngsim::SolverStats::n_err_test_fails)
        .def_readwrite("n_nonlin_iters", &bngsim::SolverStats::n_nonlin_iters)
        .def_readwrite("n_nonlin_conv_fails", &bngsim::SolverStats::n_nonlin_conv_fails)
        .def_readwrite("linear_solver", &bngsim::SolverStats::linear_solver)
        .def_readwrite("n_dense_blas_factorizations",
                       &bngsim::SolverStats::n_dense_blas_factorizations)
        .def_readwrite("steady_state_reached", &bngsim::SolverStats::steady_state_reached)
        .def_readwrite("steady_state_residual", &bngsim::SolverStats::steady_state_residual)
        .def("__repr__",
             [](const bngsim::SolverStats &s) {
                 std::ostringstream os;
                 os << "SolverStats(n_steps=" << s.n_steps << ", n_rhs_evals=" << s.n_rhs_evals
                    << ", n_jac_evals=" << s.n_jac_evals
                    << ", n_err_test_fails=" << s.n_err_test_fails
                    << ", n_nonlin_iters=" << s.n_nonlin_iters
                    << ", n_nonlin_conv_fails=" << s.n_nonlin_conv_fails << ")";
                 return os.str();
             })
        .def("to_dict", [](const bngsim::SolverStats &s) {
            py::dict d;
            d["n_steps"] = s.n_steps;
            d["n_rhs_evals"] = s.n_rhs_evals;
            d["n_jac_evals"] = s.n_jac_evals;
            d["n_err_test_fails"] = s.n_err_test_fails;
            d["n_nonlin_iters"] = s.n_nonlin_iters;
            d["n_nonlin_conv_fails"] = s.n_nonlin_conv_fails;
            d["linear_solver"] = s.linear_solver;
            d["steady_state_reached"] = s.steady_state_reached;
            d["steady_state_residual"] = s.steady_state_residual;
            return d;
        });

    // ─── SsaDiagnostics (GH #110) ──────────────────────────────────────────────
    py::class_<bngsim::SsaDiagnostics>(m, "SsaDiagnostics")
        .def(py::init<>())
        .def_readwrite("n_negative_crossings", &bngsim::SsaDiagnostics::n_negative_crossings)
        .def_readwrite("first_negative_species", &bngsim::SsaDiagnostics::first_negative_species)
        .def_readwrite("n_reverse_fires", &bngsim::SsaDiagnostics::n_reverse_fires)
        .def_readwrite("first_reverse_reaction", &bngsim::SsaDiagnostics::first_reverse_reaction)
        .def_readwrite("propensity_backend", &bngsim::SsaDiagnostics::propensity_backend)
        .def("__repr__",
             [](const bngsim::SsaDiagnostics &d) {
                 std::ostringstream os;
                 os << "SsaDiagnostics(n_negative_crossings=" << d.n_negative_crossings
                    << ", first_negative_species='" << d.first_negative_species
                    << "', n_reverse_fires=" << d.n_reverse_fires << ", first_reverse_reaction='"
                    << d.first_reverse_reaction << "')";
                 return os.str();
             })
        .def("to_dict", [](const bngsim::SsaDiagnostics &d) {
            py::dict out;
            out["n_negative_crossings"] = d.n_negative_crossings;
            out["first_negative_species"] = d.first_negative_species;
            out["n_reverse_fires"] = d.n_reverse_fires;
            out["first_reverse_reaction"] = d.first_reverse_reaction;
            out["propensity_backend"] = d.propensity_backend;
            return out;
        });

    // ─── SolverOptions ───────────────────────────────────────────────────────
    py::class_<bngsim::SolverOptions>(m, "SolverOptions")
        .def(py::init<>())
        .def_readwrite("rtol", &bngsim::SolverOptions::rtol)
        .def_readwrite("atol", &bngsim::SolverOptions::atol)
        .def_readwrite("max_steps", &bngsim::SolverOptions::max_steps)
        .def_readwrite("max_step_size", &bngsim::SolverOptions::max_step_size)
        .def_readwrite("jacobian", &bngsim::SolverOptions::jacobian)
        .def_readwrite("force_dense_linear_solver",
                       &bngsim::SolverOptions::force_dense_linear_solver)
        .def_readwrite("codegen_so_path", &bngsim::SolverOptions::codegen_so_path)
        .def_readwrite("codegen_c_source", &bngsim::SolverOptions::codegen_c_source)
        .def_readwrite("timeout_seconds", &bngsim::SolverOptions::timeout_seconds)
        .def_readwrite("steady_state", &bngsim::SolverOptions::steady_state)
        .def_readwrite("steady_state_tol", &bngsim::SolverOptions::steady_state_tol)
        .def_readwrite("carry_sensitivities", &bngsim::SolverOptions::carry_sensitivities)
        .def_readwrite("event_seed", &bngsim::SolverOptions::event_seed)
        // Forward sensitivity options
        .def(
            "set_sensitivity_params",
            [](bngsim::SolverOptions &self, std::vector<std::string> params) {
                self.sensitivity.param_names = std::move(params);
            },
            py::arg("params"), "Set parameter names for forward sensitivity analysis")
        .def(
            "set_sensitivity_ic",
            [](bngsim::SolverOptions &self, std::vector<std::string> species) {
                self.sensitivity.ic_species_names = std::move(species);
            },
            py::arg("species"),
            "Set species names for forward initial-condition sensitivity. "
            "Each named species's IC contributes a sensitivity column "
            "dY(t)/dY_k(0); the variational ODE has zero source term and "
            "is integrated by the codegen sens RHS, so codegen must be "
            "enabled for IC-sens workflows.")
        .def(
            "set_sensitivity_method",
            [](bngsim::SolverOptions &self, const std::string &method) {
                self.sensitivity.method = method;
            },
            py::arg("method"),
            "Set the CVODES corrector strategy for the coupled state + "
            "sensitivity ODE system. Both modes integrate state and all "
            "sensitivity equations as one extended ODE in a single "
            "CVODES pass; they differ only in how each step's nonlinear "
            "solve is structured.\n\n"
            "  'staggered' (default, CV_STAGGERED): state first, then "
            "sensitivities as a separate solve. Two smaller nonlinear "
            "solves per step. Often more robust for stiff or large "
            "systems; CVODES' default.\n"
            "  'simultaneous' (CV_SIMULTANEOUS): state and all "
            "sensitivities solved together as one coupled nonlinear "
            "system at every step. Often a touch faster per step on "
            "small / well-conditioned problems. AMICI's default.")
        .def(
            "set_sensitivity_error_control",
            [](bngsim::SolverOptions &self, bool val) { self.sensitivity.error_control = val; },
            py::arg("error_control"), "Include sensitivities in error test (default: true)")
        // JAX AD Jacobian callback.
        // The Python callable takes (t, y_array, param_array) and returns
        // a flat column-major Jacobian array. The C++ wrapper bridges this
        // to the std::function<void(double, const double*, double*, int)>
        // expected by CvodeSimulator, handling GIL acquisition.
        .def(
            "set_jax_jac_fn",
            [](bngsim::SolverOptions &self, py::object py_fn) {
                if (py_fn.is_none()) {
                    self.jax_jac_fn = nullptr;
                    return;
                }
                // Capture the Python callable. The C++ callback will acquire the
                // GIL before calling into Python (CVODE releases GIL during run).
                self.jax_jac_fn = [py_fn](double t, const double *y_ptr, double *jac_ptr, int ns) {
                    // Acquire GIL to call Python
                    py::gil_scoped_acquire acquire;

                    // Create numpy views (zero-copy) of the C arrays
                    auto y_arr =
                        py::array_t<double>({static_cast<py::ssize_t>(ns)}, {sizeof(double)}, y_ptr,
                                            py::cast(0) // no owner — data lives in CVODE
                        );

                    // Call Python: fn(t, y_array) -> flat column-major jac array
                    py::object result = py_fn(t, y_arr);

                    // Copy result into CVODE's Jacobian buffer
                    auto jac_arr = result.cast<py::array_t<double>>();
                    auto buf = jac_arr.request();
                    const double *src = static_cast<const double *>(buf.ptr);
                    std::memcpy(jac_ptr, src, ns * ns * sizeof(double));
                };
            },
            py::arg("fn"),
            "Set JAX AD Jacobian callback. fn(t, y_array) -> flat_jac_array (col-major)");

    // ─── TimeSpec ────────────────────────────────────────────────────────────
    py::class_<bngsim::TimeSpec>(m, "TimeSpec")
        .def(py::init<>())
        .def_readwrite("t_start", &bngsim::TimeSpec::t_start)
        .def_readwrite("t_end", &bngsim::TimeSpec::t_end)
        .def_readwrite("n_points", &bngsim::TimeSpec::n_points)
        .def_readwrite("sample_times", &bngsim::TimeSpec::sample_times);

    // ─── Result ──────────────────────────────────────────────────────────────
    //
    // Wraps bngsim::Result. Data is converted to NumPy arrays on access.
    // The arrays own copies of the data (via capsule), so the C++ Result
    // can be freed after the Python Result is constructed.
    py::class_<bngsim::Result>(m, "ResultCore")
        .def(py::init<>())
        .def_property_readonly("n_times", &bngsim::Result::n_times)
        // n_species/species_{data,names} project to the reported species subset
        // (GH #71). reported_species_indices() is empty for every all-reported
        // model — the trivial path returns the full set, byte-identical. Only an
        // SBML model with an event-mutated parameter/compartment promoted to a
        // species carries a non-empty projection; those columns are full
        // integrator state but not floating-species trajectory output.
        .def_property_readonly("n_species",
                               [](const bngsim::Result &r) {
                                   const auto &keep = r.reported_species_indices();
                                   return keep.empty() ? r.n_species()
                                                       : static_cast<int>(keep.size());
                               })
        // observable_{names,data,n_observables} drop the loader's internal
        // network-rewrite scaffolding observables (`__bngsim_net_rewrite_obs_*`,
        // issue #61). These back the rewritten legacy Sat/Hill functional rate
        // laws inside the model but are not user observables and must not leak
        // into the .gdat surface or the public API. Mirrors the _rateLawN
        // function filtering above.
        .def_property_readonly(
            "n_observables",
            [](const bngsim::Result &r) {
                return static_cast<int>(
                    bngsim::public_observable_indices(r.observable_names()).size());
            })
        .def_property_readonly("time",
                               [](const bngsim::Result &r) { return vec_to_ndarray_1d(r.time()); })
        .def_property_readonly("species_data",
                               [](const bngsim::Result &r) {
                                   const auto &keep = r.reported_species_indices();
                                   const int n_t = r.n_times();
                                   const int n_raw = r.n_species();
                                   if (keep.empty()) {
                                       return vec_to_ndarray_2d(r.species_data(), n_t, n_raw);
                                   }
                                   const auto &raw = r.species_data();
                                   std::vector<double> filtered;
                                   filtered.reserve(static_cast<size_t>(n_t) * keep.size());
                                   for (int t = 0; t < n_t; ++t) {
                                       const size_t row = static_cast<size_t>(t) * n_raw;
                                       for (size_t c : keep) {
                                           filtered.push_back(raw[row + c]);
                                       }
                                   }
                                   return vec_to_ndarray_2d(filtered, n_t,
                                                            static_cast<int>(keep.size()));
                               })
        .def_property_readonly(
            "observable_data",
            [](const bngsim::Result &r) {
                const auto keep = bngsim::public_observable_indices(r.observable_names());
                const int n_t = r.n_times();
                const int n_raw = r.n_observables();
                if (keep.empty() || n_t == 0) {
                    return vec_to_ndarray_2d({}, n_t, static_cast<int>(keep.size()));
                }
                const auto &raw = r.observable_data();
                std::vector<double> filtered;
                filtered.reserve(static_cast<size_t>(n_t) * keep.size());
                for (int t = 0; t < n_t; ++t) {
                    const size_t row = static_cast<size_t>(t) * n_raw;
                    for (size_t c : keep) {
                        filtered.push_back(raw[row + c]);
                    }
                }
                return vec_to_ndarray_2d(filtered, n_t, static_cast<int>(keep.size()));
            })
        .def_property_readonly("species_names",
                               [](const bngsim::Result &r) -> std::vector<std::string> {
                                   const auto &keep = r.reported_species_indices();
                                   const auto &raw = r.species_names();
                                   if (keep.empty()) {
                                       return raw;
                                   }
                                   std::vector<std::string> out;
                                   out.reserve(keep.size());
                                   for (size_t i : keep) {
                                       out.push_back(raw[i]);
                                   }
                                   return out;
                               })
        .def_property_readonly("observable_names",
                               [](const bngsim::Result &r) {
                                   const auto &raw = r.observable_names();
                                   std::vector<std::string> out;
                                   for (size_t i : bngsim::public_observable_indices(raw)) {
                                       out.push_back(raw[i]);
                                   }
                                   return out;
                               })
        // expression_{data,names,n_expressions} drop the auto-generated
        // _rateLawN rate-law functions that BNG2.pl synthesises for
        // run_network/NFsim but filters out of its .gdat/.scan output. Names
        // stay bare here: downstream consumers use them as in-memory column
        // keys (constraint/experimental-data matching expects the bare name,
        // matching how a BNG2.pl .gdat is read back with the "()" stripped).
        // The BNG2.pl "()" header convention is a file-write concern applied
        // at serialization, not here. Unfiltered values: raw_expression_*.
        .def_property_readonly(
            "expression_data",
            [](const bngsim::Result &r) {
                const auto keep = user_function_indices(r.expression_names());
                const int n_t = r.n_times();
                const int n_raw = r.n_expressions();
                if (keep.empty() || n_t == 0) {
                    return vec_to_ndarray_2d({}, n_t, static_cast<int>(keep.size()));
                }
                const auto &raw = r.expression_data();
                std::vector<double> filtered;
                filtered.reserve(static_cast<size_t>(n_t) * keep.size());
                for (int t = 0; t < n_t; ++t) {
                    const size_t row = static_cast<size_t>(t) * n_raw;
                    for (size_t c : keep) {
                        filtered.push_back(raw[row + c]);
                    }
                }
                return vec_to_ndarray_2d(filtered, n_t, static_cast<int>(keep.size()));
            })
        .def_property_readonly("expression_names",
                               [](const bngsim::Result &r) {
                                   const auto &raw = r.expression_names();
                                   std::vector<std::string> out;
                                   for (size_t i : user_function_indices(raw)) {
                                       out.push_back(raw[i]);
                                   }
                                   return out;
                               })
        .def_property_readonly("n_expressions",
                               [](const bngsim::Result &r) {
                                   return static_cast<int>(
                                       user_function_indices(r.expression_names()).size());
                               })
        // .gdat/.scan function-column header labels. Since bngsim emits bare
        // headers (no "()", issue #58), these are identical to expression_names;
        // the property is retained for consumers that assemble file headers
        // (e.g. a consumer-built .scan) and want an intent-revealing name.
        .def_property_readonly("gdat_expression_names",
                               [](const bngsim::Result &r) {
                                   const auto &raw = r.expression_names();
                                   std::vector<std::string> out;
                                   for (size_t i : user_function_indices(raw)) {
                                       out.push_back(raw[i]);
                                   }
                                   return out;
                               })
        // Unfiltered function columns (includes internal _rateLawN, bare names).
        // For debugging internal rate-law values; not part of BNG2.pl output.
        .def_property_readonly("raw_expression_data",
                               [](const bngsim::Result &r) {
                                   return vec_to_ndarray_2d(r.expression_data(), r.n_times(),
                                                            r.n_expressions());
                               })
        .def_property_readonly("raw_expression_names", &bngsim::Result::expression_names)
        .def_property_readonly("raw_n_expressions", &bngsim::Result::n_expressions)
        .def_property_readonly(
            "solver_stats",
            [](const bngsim::Result &r) -> const bngsim::SolverStats & { return r.solver_stats(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "ssa_diagnostics",
            [](const bngsim::Result &r) -> const bngsim::SsaDiagnostics & {
                return r.ssa_diagnostics();
            },
            py::return_value_policy::reference_internal)
        .def("to_gdat", &bngsim::Result::to_gdat, py::arg("path"),
             py::arg("print_functions") = false, py::arg("print_rate_laws") = false,
             "Export observables in BNG .gdat format. Function headers are bare "
             "(no '()') and identical across methods. With print_functions=True, "
             "append user-named function columns; with print_rate_laws=True, also "
             "append the auto-generated _rateLawN columns (still bare).")
        .def("to_cdat", &bngsim::Result::to_cdat, py::arg("path"),
             "Export species in BNG .cdat format")
        // Sensitivity data (CVODES forward sensitivities)
        .def_property_readonly("n_sens_params", &bngsim::Result::n_sens_params)
        .def_property_readonly("sens_param_names", &bngsim::Result::sens_param_names)
        .def_property_readonly(
            "sensitivity_data",
            [](const bngsim::Result &r) {
                // Return as 3D array: (n_times, n_species, n_sens_params)
                int nt = r.n_times();
                int ns = r.n_species();
                int np = r.n_sens_params();
                if (np == 0 || r.sensitivity_data().empty()) {
                    return py::array_t<double>({0, 0, 0});
                }
                auto *vec = new std::vector<double>(r.sensitivity_data());
                auto capsule =
                    py::capsule(vec, [](void *p) { delete static_cast<std::vector<double> *>(p); });
                return py::array_t<double>({nt, ns, np},
                                           {static_cast<py::ssize_t>(ns * np * sizeof(double)),
                                            static_cast<py::ssize_t>(np * sizeof(double)),
                                            static_cast<py::ssize_t>(sizeof(double))},
                                           vec->data(), capsule);
            })
        // IC sensitivity data
        .def_property_readonly("n_sens_ic_species", &bngsim::Result::n_sens_ic_species)
        .def_property_readonly("sens_ic_species_names", &bngsim::Result::sens_ic_species_names)
        .def_property_readonly("sensitivity_ic_data", [](const bngsim::Result &r) {
            int nt = r.n_times();
            int ns = r.n_species();
            int nic = r.n_sens_ic_species();
            if (nic == 0 || r.sensitivity_ic_data().empty()) {
                return py::array_t<double>({0, 0, 0});
            }
            auto *vec = new std::vector<double>(r.sensitivity_ic_data());
            auto capsule =
                py::capsule(vec, [](void *p) { delete static_cast<std::vector<double> *>(p); });
            return py::array_t<double>({nt, ns, nic},
                                       {static_cast<py::ssize_t>(ns * nic * sizeof(double)),
                                        static_cast<py::ssize_t>(nic * sizeof(double)),
                                        static_cast<py::ssize_t>(sizeof(double))},
                                       vec->data(), capsule);
        })
        // GH #196 — observable/expression output sensitivities (storage only).
        // Shape (n_times, n_rows, depth), or empty (0,0,0) when not computed.
        // The parameter blocks share the species parameter axis (n_sens_params);
        // the IC blocks share the species IC axis (n_sens_ic_species). The row
        // count is recovered from the buffer size. The species-only
        // ``sensitivity_data`` / ``sensitivity_ic_data`` properties above are
        // intentionally left untouched.
        .def_property_readonly("observable_sensitivity_data",
                               [](const bngsim::Result &r) {
                                   return sens_block_to_ndarray_3d(r.observable_sensitivity_data(),
                                                                   r.n_times(), r.n_sens_params());
                               })
        .def_property_readonly("expression_sensitivity_data",
                               [](const bngsim::Result &r) {
                                   return sens_block_to_ndarray_3d_rows(
                                       r.expression_sensitivity_data(), r.n_times(),
                                       r.n_sens_params(),
                                       user_function_indices(r.expression_names()));
                               })
        .def_property_readonly("observable_sensitivity_ic_data",
                               [](const bngsim::Result &r) {
                                   return sens_block_to_ndarray_3d(
                                       r.observable_sensitivity_ic_data(), r.n_times(),
                                       r.n_sens_ic_species());
                               })
        .def_property_readonly("expression_sensitivity_ic_data", [](const bngsim::Result &r) {
            return sens_block_to_ndarray_3d_rows(r.expression_sensitivity_ic_data(), r.n_times(),
                                                 r.n_sens_ic_species(),
                                                 user_function_indices(r.expression_names()));
        });

    // ─── NetworkModel ────────────────────────────────────────────────────────
    py::class_<bngsim::NetworkModel>(m, "NetworkModel")
        // Factory: from_net (static method)
        .def_static(
            "from_net",
            [](const std::string &path) {
                try {
                    auto model = bngsim::NetworkModel::from_net(path);
                    for (const auto &warning : model.load_warnings()) {
                        if (PyErr_WarnEx(PyExc_UserWarning, warning.c_str(), 1) < 0) {
                            throw py::error_already_set();
                        }
                    }
                    return model;
                } catch (const py::error_already_set &) {
                    throw;
                } catch (const std::exception &e) {
                    throw py::value_error(std::string("Failed to load .net file: ") + e.what());
                }
            },
            py::arg("path"), "Load a model from a BNG .net file")

        // Clone (deep copy for parallel workers)
        .def("clone", &bngsim::NetworkModel::clone, "Deep copy the model (for parallel workers)")

        // GH #106: True iff the model references rateOf(species) and therefore
        // runs the derivative-probe RHS path.
        .def_property_readonly("uses_rateof", &bngsim::NetworkModel::uses_rateof,
                               "Whether the model uses the SBML rateOf csymbol (GH #106).")

        // T1: RHS observable/function-eval gate instrumentation. For a pure
        // mass-action model the RHS skips update_observables + evaluate_functions
        // (dead work); these expose that gate so tests/benchmarks can prove it.
        .def_property_readonly(
            "rhs_evaluates_observables", &bngsim::NetworkModel::rhs_evaluates_observables,
            "Whether compute_derivs_core refreshes observables/functions on the RHS "
            "(true iff the model has functions; false for pure mass-action).")
        .def_property_readonly("rhs_eval_count", &bngsim::NetworkModel::rhs_eval_count,
                               "Number of RHS (compute_derivs_core) evaluations so far.")
        .def_property_readonly(
            "rhs_observable_eval_count", &bngsim::NetworkModel::rhs_observable_eval_count,
            "Number of RHS evaluations that ran update_observables + evaluate_functions.")
        .def("reset_rhs_counters", &bngsim::NetworkModel::reset_rhs_counters,
             "Reset the RHS instrumentation counters to zero.")

        // T7: per-reported-species volume factor (V_c), in reported-species
        // order. Narrow accessor for the trajectory-output layer's
        // amounts-from-concentrations conversion — avoids materializing the
        // whole model as a Python dict via codegen_data() just to read this.
        .def("reported_volume_factors", &bngsim::NetworkModel::reported_volume_factors,
             "Per-reported-species volume_factor (V_c), in reported-species order.")

        // Parameter access
        .def(
            "set_param",
            [](bngsim::NetworkModel &self, const std::string &name, double value) {
                try {
                    self.set_param(name, value);
                } catch (const std::exception &e) {
                    throw py::key_error(std::string("Parameter not found: ") + name);
                }
            },
            py::arg("name"), py::arg("value"), "Set a parameter value by name")

        .def(
            "get_param",
            [](const bngsim::NetworkModel &self, const std::string &name) {
                try {
                    return self.get_param(name);
                } catch (const std::exception &e) {
                    throw py::key_error(std::string("Parameter not found: ") + name);
                }
            },
            py::arg("name"), "Get a parameter value by name")

        .def(
            "set_params",
            [](bngsim::NetworkModel &self, py::dict params) {
                for (auto item : params) {
                    std::string name = py::str(item.first);
                    double value = item.second.cast<double>();
                    try {
                        self.set_param(name, value);
                    } catch (const std::exception &e) {
                        throw py::key_error(std::string("Parameter not found: ") + name);
                    }
                }
            },
            py::arg("params"), "Set multiple parameters from a dict")

        .def_property_readonly("param_names", &bngsim::NetworkModel::param_names,
                               "List of all parameter names")

        .def_property_readonly(
            "param_is_expression",
            [](const bngsim::NetworkModel &self) {
                const auto &params = self.parameters();
                std::vector<bool> out;
                out.reserve(params.size());
                for (const auto &p : params)
                    out.push_back(p.is_expression);
                return out;
            },
            "Per-parameter ``is_expression`` flag (True for derived ConstantExpression "
            "parameters such as BNG2.pl-emitted ``_rateLaw{N}``).")

        // State management
        .def("reset", &bngsim::NetworkModel::reset, "Reset species to initial concentrations")

        .def("save_concentrations", &bngsim::NetworkModel::save_concentrations,
             "Snapshot current concentrations as new initial state")

        .def(
            "set_concentration",
            [](bngsim::NetworkModel &self, const std::string &name, double value) {
                try {
                    self.set_concentration(name, value);
                } catch (const std::exception &e) {
                    throw py::key_error(std::string("Species not found: ") + name);
                }
            },
            py::arg("name"), py::arg("value"), "Set a single species concentration by name")

        .def(
            "get_concentration",
            [](const bngsim::NetworkModel &self, const std::string &name) {
                try {
                    return self.get_concentration(name);
                } catch (const std::exception &e) {
                    throw py::key_error(std::string("Species not found: ") + name);
                }
            },
            py::arg("name"), "Get a single species concentration by name")

        // ── Bulk state-vector access (GH #102) ───────────────────────────────
        // The per-step state-exchange primitive for driving bngsim as a
        // reaction kernel from an external orchestrator: one Python call
        // marshals the whole live concentration vector, so per-step exchange
        // cost stays negligible next to the ODE solve even at ~100K species.
        .def(
            "get_state",
            [](const bngsim::NetworkModel &self) {
                const auto ns = static_cast<py::ssize_t>(self.n_species());
                py::array_t<double> arr(ns);
                if (ns > 0)
                    self.get_state_into(static_cast<double *>(arr.request().ptr));
                return arr;
            },
            "Bulk-copy all species concentrations into a new float64 ndarray, "
            "ordered like species_names(). O(n_species), one Python call (GH #102).")
        .def(
            "set_state",
            [](bngsim::NetworkModel &self,
               py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
                if (arr.ndim() != 1)
                    throw py::value_error("set_state expects a 1-D array");
                if (arr.shape(0) != self.n_species())
                    throw py::value_error("set_state: expected " +
                                          std::to_string(self.n_species()) + " values, got " +
                                          std::to_string(arr.shape(0)));
                if (self.n_species() > 0)
                    self.set_state_from(arr.data());
            },
            py::arg("state"),
            "Bulk-assign all species concentrations from a 1-D float64 ndarray, "
            "ordered like species_names(). O(n_species), one Python call (GH #102).")

        // ── Pre-equilibration / carry-over sensitivity state (GH #210) ───────
        .def_property_readonly(
            "ic_state_dirty", &bngsim::NetworkModel::ic_state_dirty,
            "True iff the current species state is carried-over dynamics from a "
            "previous run() (not a fresh initial condition). Forward sensitivities "
            "requested on a dirty state require carry_sensitivities=True (else "
            "they raise). Cleared by reset()/save_concentrations() (GH #210).")
        .def_property_readonly(
            "has_pending_sensitivity_seed",
            [](const bngsim::NetworkModel &self) { return !self.pending_sens_seed().empty(); },
            "True iff a forward-sensitivity carry-over seed (dx/dθ) from a prior "
            "phase is pending, ready to seed a carry_sensitivities=True run (GH #210).")
        .def(
            "pending_sensitivity_seed",
            [](const bngsim::NetworkModel &self) {
                const auto &seed = self.pending_sens_seed();
                const auto np = static_cast<py::ssize_t>(
                    self.pending_sens_seed_param_names().size());
                const auto ns = static_cast<py::ssize_t>(self.n_species());
                if (seed.empty() || np == 0) {
                    return py::array_t<double>(std::vector<py::ssize_t>{0, 0});
                }
                py::array_t<double> arr(std::vector<py::ssize_t>{ns, np});
                double *dst = static_cast<double *>(arr.request().ptr);
                for (size_t i = 0; i < seed.size(); ++i)
                    dst[i] = seed[i];
                return arr;
            },
            "The pending carry-over forward-sensitivity seed dx/dθ as an "
            "(n_species, n_params) ndarray, or shape (0, 0) when none is pending. "
            "Columns are pending_sensitivity_seed_param_names() (GH #210).")
        .def_property_readonly(
            "pending_sensitivity_seed_param_names",
            &bngsim::NetworkModel::pending_sens_seed_param_names,
            "Parameter names labeling the columns of pending_sensitivity_seed() (GH #210).")

        // ── Functional analytical Jacobian (GH #76) ──────────────────────────
        // Python (bngsim._jacobian) reads this context, differentiates each
        // Functional rate law with sympy, and writes the result back via
        // set_functional_jacobian. The engine never calls Python.
        .def(
            "functional_jacobian_context",
            [](const bngsim::NetworkModel &self) {
                auto ctx = self.functional_jacobian_context();
                py::list rxns;
                for (const auto &r : ctx.functional_reactions) {
                    py::dict d;
                    d["rxn_idx"] = r.rxn_idx;
                    d["rate_expr"] = r.rate_expr;
                    d["apply_species_factor"] = r.apply_species_factor;
                    d["reactant_idx0"] = py::cast(r.reactant_idx0);
                    d["product_idx0"] = py::cast(r.product_idx0);
                    d["stat_factor"] = r.stat_factor;
                    d["per_species_volume_scaling"] = r.per_species_volume_scaling;
                    rxns.append(std::move(d));
                }
                py::dict fmap;
                for (const auto &kv : ctx.function_map)
                    fmap[py::str(kv.first)] = kv.second;
                py::list obs;
                for (const auto &ob : ctx.observables) {
                    py::list entries;
                    for (const auto &e : ob.second)
                        entries.append(py::make_tuple(e.first, e.second));
                    obs.append(py::make_tuple(ob.first, std::move(entries)));
                }
                py::list smeta;
                for (const auto &m : ctx.species_meta)
                    smeta.append(py::make_tuple(m.first, m.second));
                py::dict out;
                out["functional_reactions"] = std::move(rxns);
                out["function_map"] = std::move(fmap);
                out["observables"] = std::move(obs);
                out["species_meta"] = std::move(smeta);
                out["constant_names"] = py::cast(ctx.constant_names);
                return out;
            },
            "Read-only context for the Python analytical-Jacobian differentiator (GH #76).")

        .def(
            "set_functional_jacobian",
            [](bngsim::NetworkModel &self, const py::list &terms) {
                std::vector<bngsim::FunctionalJacobianInput> inputs;
                for (const auto &item : terms) {
                    auto tup = item.cast<py::tuple>();
                    bngsim::FunctionalJacobianInput in;
                    in.rxn_idx = tup[0].cast<int>();
                    in.per_observable = tup[1].cast<bool>();
                    for (const auto &dt : tup[2].cast<py::list>()) {
                        auto p = dt.cast<py::tuple>();
                        in.deriv_terms.emplace_back(p[0].cast<int>(), p[1].cast<std::string>());
                    }
                    inputs.push_back(std::move(in));
                }
                return self.set_functional_jacobian(inputs);
            },
            py::arg("terms"),
            "Compile and attach symbolically-derived Functional Jacobian terms "
            "(GH #76). Returns True if the analytical Jacobian was populated.")

        .def_property_readonly("analytical_jacobian_complete",
                               &bngsim::NetworkModel::analytical_jacobian_complete,
                               "True iff the analytical Jacobian covers every reaction "
                               "(Elementary + attached Functional terms). GH #76.")

        // Test hooks (GH #76): RHS and the assembled dense analytical Jacobian
        // at an arbitrary (t, conc), for the entrywise finite-difference check.
        .def(
            "_eval_rhs",
            [](bngsim::NetworkModel &self, double t, const std::vector<double> &conc) {
                int ns = self.n_species();
                std::vector<double> y = conc, dydt(ns, 0.0);
                self.compute_derivs(t, y.data(), dydt.data());
                return dydt;
            },
            py::arg("t"), py::arg("conc"), "Evaluate dy/dt at (t, conc). Test/diagnostic hook.")
        .def(
            "_dense_analytical_jacobian",
            [](bngsim::NetworkModel &self, double t, const std::vector<double> &conc) {
                int ns = self.n_species();
                std::vector<double> y = conc,
                                    jac(static_cast<size_t>(ns) * static_cast<size_t>(ns), 0.0);
                self.fill_dense_analytical_jacobian(t, y.data(), jac.data());
                return jac; // flat column-major: jac[j*ns + i] = ∂f_i/∂x_j
            },
            py::arg("t"), py::arg("conc"),
            "Assemble the dense analytical Jacobian at (t, conc), flat "
            "column-major. Test/diagnostic hook for the FD cross-check. GH #76.")

        // Dense analytical-Jacobian scatter plan for the codegen .so (GH #76
        // Task 4). Serializes exactly what NetworkModel::fill_dense_analytical_
        // jacobian iterates for the Elementary + MM blocks — rows pre-resolved
        // through the CSC sparsity so the Python emitter never sees CSC indices.
        // The Functional contributions are NOT serialized here: the codegen
        // emitter reconstructs them from functional_jacobian_context() the same
        // way attach_functional_jacobian does (the derivative *math* lives in the
        // Python symbolic core, not in the compiled ExprTk eval_ids). `available`
        // mirrors analytical_jacobian_complete() so the emitter only emits when
        // the interpreted analytical Jacobian is itself complete; nnz/density/
        // has_klu let the emitter skip models that take the sparse KLU path
        // (dense codegen Jacobian is the only path wired so far).
        .def(
            "codegen_jacobian_plan",
            [](const bngsim::NetworkModel &self) {
                py::dict out;
                const bool avail = self.analytical_jacobian_complete();
                const int ns = self.n_species();
                const auto &sp = self.jacobian_sparsity();
                out["available"] = avail;
                out["n_species"] = ns;
                out["nnz"] = sp.nnz;
                out["density"] = sp.density;
#ifdef BNGSIM_HAS_KLU
                out["has_klu"] = true;
#else
                out["has_klu"] = false;
#endif
                py::list elem, mm, fixed_rows;
                if (avail) {
                    const auto &ajd = self.analytical_jacobian();
                    // Elementary mass-action closed-form scatter (mirrors the
                    // first block of fill_dense_analytical_jacobian).
                    for (const auto &rt : ajd.reactions) {
                        if (rt.rate_param_idx0 < 0 || rt.reactants.empty())
                            continue;
                        py::dict rd;
                        rd["rate_param_idx0"] = rt.rate_param_idx0;
                        rd["stat_factor"] = rt.stat_factor;
                        rd["amount_factor"] = rt.amount_factor;
                        py::list reactants;
                        for (const auto &pr : rt.reactants) {
                            py::dict prd;
                            prd["species_idx"] = pr.species_idx;
                            prd["multiplicity"] = pr.multiplicity;
                            py::list others;
                            for (const auto &o : pr.others)
                                others.append(py::make_tuple(o.species_idx, o.multiplicity));
                            prd["others"] = others;
                            py::list affected;
                            py::list affected_csc;
                            for (const auto &[csc, stoich] : pr.affected) {
                                affected.append(
                                    py::make_tuple(static_cast<int>(sp.row_indices[csc]), stoich));
                                // GH #162: raw CSC data index for the sparse emitter
                                // (jac_data[csc] += …) — the dense path keeps the
                                // pre-resolved row above.
                                affected_csc.append(
                                    py::make_tuple(static_cast<int64_t>(csc), stoich));
                            }
                            prd["affected"] = affected;
                            prd["affected_csc"] = affected_csc;
                            reactants.append(std::move(prd));
                        }
                        rd["reactants"] = std::move(reactants);
                        elem.append(std::move(rd));
                    }
                    // Michaelis–Menten (tQSSA) closed-form scatter (second block).
                    for (const auto &mt : ajd.mm_reactions) {
                        py::dict md;
                        md["e_idx"] = mt.e_idx;
                        md["s_idx"] = mt.s_idx;
                        md["kcat_param_idx0"] = mt.kcat_param_idx0;
                        md["km_param_idx0"] = mt.km_param_idx0;
                        md["stat_factor"] = mt.stat_factor;
                        py::list ea, sa, ea_csc, sa_csc;
                        for (const auto &[csc, coeff] : mt.e_affected) {
                            ea.append(py::make_tuple(static_cast<int>(sp.row_indices[csc]), coeff));
                            ea_csc.append(py::make_tuple(static_cast<int64_t>(csc), coeff));
                        }
                        for (const auto &[csc, coeff] : mt.s_affected) {
                            sa.append(py::make_tuple(static_cast<int>(sp.row_indices[csc]), coeff));
                            sa_csc.append(py::make_tuple(static_cast<int64_t>(csc), coeff));
                        }
                        md["e_affected"] = std::move(ea);
                        md["s_affected"] = std::move(sa);
                        md["e_affected_csc"] = std::move(ea_csc);
                        md["s_affected_csc"] = std::move(sa_csc);
                        mm.append(std::move(md));
                    }
                    // Fixed (boundary-condition) species rows (final block).
                    const auto &species = self.species();
                    for (int i = 0; i < static_cast<int>(species.size()); ++i)
                        if (species[i].fixed)
                            fixed_rows.append(i);
                    // GH #162: CSC structure for the sparse codegen emitter. The
                    // Functional scatter and fixed-row zeroing map (col, row) →
                    // data index — the dense path's pre-resolved rows discard the
                    // index, and Elementary/MM carry theirs via affected_csc above.
                    // Exposed as int64 arrays (copied) so the emitter never
                    // materializes nnz Python ints for the columns it skips.
                    out["col_ptrs"] = py::array_t<int64_t>(
                        static_cast<py::ssize_t>(sp.col_ptrs.size()), sp.col_ptrs.data());
                    out["row_indices"] = py::array_t<int64_t>(
                        static_cast<py::ssize_t>(sp.row_indices.size()), sp.row_indices.data());
                }
                out["elementary"] = std::move(elem);
                out["mm"] = std::move(mm);
                out["fixed_rows"] = std::move(fixed_rows);
                return out;
            },
            "Dense analytical-Jacobian scatter plan (Elementary + MM, rows "
            "resolved) for the codegen .so. GH #76 Task 4.")

        .def(
            "_eval_functions",
            [](bngsim::NetworkModel &self, double t, const std::vector<double> &conc) {
                self.update_observables(conc.data());
                self.evaluate_functions(t);
                py::dict out;
                auto names = self.function_names();
                auto vals = self.function_values();
                for (size_t i = 0; i < names.size(); ++i)
                    out[py::str(names[i])] = vals[i];
                return out;
            },
            py::arg("t"), py::arg("conc"),
            "Evaluate functions at (t, conc) → {name: value}. Test/diagnostic hook.")

        // Dimensions
        .def_property_readonly("n_species", &bngsim::NetworkModel::n_species)
        .def_property_readonly("n_reactions", &bngsim::NetworkModel::n_reactions)
        .def_property_readonly("n_observables", &bngsim::NetworkModel::n_observables)
        .def_property_readonly("n_parameters", &bngsim::NetworkModel::n_parameters)
        .def_property_readonly("n_functions", &bngsim::NetworkModel::n_functions)
        .def_property_readonly("n_events", &bngsim::NetworkModel::n_events)
        .def("event_sensitivity_unsupported_reason",
             &bngsim::NetworkModel::event_sensitivity_unsupported_reason,
             pybind11::arg("sens_param_names"),
             "Return a reason string if any event blocks Phase-1 forward "
             "sensitivity for the given sensitivity-parameter names, else None "
             "(GH #212).")
        .def_property_readonly("n_discontinuity_triggers",
                               &bngsim::NetworkModel::n_discontinuity_triggers)
        .def_property_readonly("load_warnings", &bngsim::NetworkModel::load_warnings)

        // Named lists
        .def_property_readonly("species_names", &bngsim::NetworkModel::species_names)
        .def_property_readonly("observable_names", &bngsim::NetworkModel::observable_names)

        // Table functions
        .def(
            "add_table_function_file",
            [](bngsim::NetworkModel &self, const std::string &name, const std::string &filepath,
               const std::string &index_name, const std::string &method) {
                try {
                    self.add_table_function(name, filepath, index_name, method);
                } catch (const std::exception &e) {
                    throw py::value_error(std::string("Failed to add table function: ") + e.what());
                }
            },
            py::arg("name"), py::arg("filepath"), py::arg("index_name") = "time",
            py::arg("method") = "linear", "Add a table function from a .tfun file")

        .def(
            "add_table_function_arrays",
            [](bngsim::NetworkModel &self, const std::string &name, const std::vector<double> &xs,
               const std::vector<double> &ys, const std::string &index_name,
               const std::string &method) {
                try {
                    self.add_table_function(name, xs, ys, index_name, method);
                } catch (const std::exception &e) {
                    throw py::value_error(std::string("Failed to add table function: ") + e.what());
                }
            },
            py::arg("name"), py::arg("xs"), py::arg("ys"), py::arg("index_name") = "time",
            py::arg("method") = "linear", "Add a table function from arrays")

        .def_property_readonly("n_table_functions", &bngsim::NetworkModel::n_table_functions)
        .def_property_readonly("table_function_names", &bngsim::NetworkModel::table_function_names)

        // Conservation laws
        .def_property_readonly(
            "conservation_laws",
            [](const bngsim::NetworkModel &m) {
                const auto &cl = m.conservation_laws();
                py::dict d;
                d["n_laws"] = cl.n_laws;
                d["n_species"] = cl.n_species;
                d["dependent"] = cl.dependent;
                d["independent"] = cl.independent;
                d["constants"] = cl.constants;
                // Convert coefficients to list of lists
                py::list coeffs;
                for (const auto &row : cl.coefficients) {
                    coeffs.append(py::cast(row));
                }
                d["coefficients"] = coeffs;
                return d;
            },
            "Conservation laws detected from stoichiometry matrix")

        // Model introspection for code generation
        .def(
            "codegen_data",
            [](const bngsim::NetworkModel &m) {
                py::dict d;

                // Parameters: [{name, value, is_const, expression}, ...]
                // is_const / expression let the model-based sensitivity codegen
                // chain-rule through derived rate constants like the .net path
                // does (closes #15).
                const auto &params = m.parameters();
                py::list param_list;
                for (const auto &p : params) {
                    py::dict pd;
                    pd["name"] = p.name;
                    pd["value"] = p.value;
                    pd["is_const"] = !p.is_expression;
                    pd["expression"] = p.expression;
                    param_list.append(pd);
                }
                d["parameters"] = param_list;

                // Species: [(name, fixed, volume_factor, amount_valued), ...]
                const auto &species = m.species();
                py::list sp_list;
                for (const auto &s : species) {
                    py::dict sd;
                    sd["name"] = s.name;
                    sd["fixed"] = s.fixed;
                    sd["volume_factor"] = s.volume_factor;
                    sd["amount_valued"] = s.amount_valued;
                    sd["reported"] = s.reported;                         // GH #71
                    sd["ode_live_volume_idx0"] = s.ode_live_volume_idx0; // GH #144 case 4
                    sd["report_rateof_amount"] = s.report_rateof_amount; // GH #231 sub-cluster 3
                    sp_list.append(sd);
                }
                d["species"] = sp_list;

                // Observables: [(name, [(species_idx_0based, factor), ...]), ...]
                const auto &observables = m.observables();
                py::list obs_list;
                for (const auto &o : observables) {
                    py::dict od;
                    od["name"] = o.name;
                    py::list entries;
                    for (const auto &e : o.entries) {
                        // Convert 1-based species index to 0-based
                        entries.append(py::make_tuple(e.species_index - 1, e.factor));
                    }
                    od["entries"] = entries;
                    obs_list.append(od);
                }
                d["observables"] = obs_list;

                // Functions: [(name, expression), ...]
                const auto &functions = m.functions();
                py::list func_list;
                for (const auto &f : functions) {
                    py::dict fd;
                    fd["name"] = f.name;
                    fd["expression"] = f.expression;
                    func_list.append(fd);
                }
                d["functions"] = func_list;

                // Reactions: [(type, function_name, stat_factor, reactants_0based,
                // products_0based), ...]
                const auto &reactions = m.reactions();
                py::list rxn_list;
                for (const auto &r : reactions) {
                    py::dict rd;
                    // Rate law type as string
                    switch (r.rate_law_type) {
                    case bngsim::RateLawType::Elementary:
                        rd["type"] = "elementary";
                        break;
                    case bngsim::RateLawType::Functional:
                        rd["type"] = "functional";
                        break;
                    case bngsim::RateLawType::MichaelisMenten:
                        rd["type"] = "mm";
                        break;
                    }
                    rd["function_name"] = r.function_name;
                    rd["stat_factor"] = r.stat_factor;
                    rd["apply_species_factor"] = r.apply_species_factor;
                    rd["ssa_volume_factor"] = r.ssa_volume_factor;
                    rd["per_species_volume_scaling"] = r.per_species_volume_scaling;
                    rd["is_rate_rule_ode"] = r.is_rate_rule_ode;
                    rd["ssa_live_volume_idx0"] = r.ssa_live_volume_idx0;
                    rd["ssa_live_volume_exp"] = r.ssa_live_volume_exp;
                    // GH #144 case 4: cross-compartment SSA live-volume terms.
                    py::list live_terms;
                    for (const auto &lt : r.ssa_live_volume_terms) {
                        py::dict td;
                        td["live_idx0"] = lt.live_idx0;
                        td["v_static"] = lt.v_static;
                        td["exp"] = lt.exp;
                        live_terms.append(td);
                    }
                    rd["ssa_live_volume_terms"] = live_terms;
                    rd["ode_only"] = r.ode_only;
                    // Convert 1-based indices to 0-based, filter nulls (0 in 1-based)
                    py::list reactants;
                    for (int idx : r.reactant_indices) {
                        if (idx > 0)
                            reactants.append(idx - 1);
                    }
                    rd["reactants"] = reactants;
                    py::list products;
                    for (int idx : r.product_indices) {
                        if (idx > 0)
                            products.append(idx - 1);
                    }
                    rd["products"] = products;
                    // Rate law parameter indices (0-based) for Elementary/MM
                    py::list rate_params;
                    for (int idx : r.rate_law_param_indices) {
                        rate_params.append(idx - 1); // 1-based to 0-based
                    }
                    rd["rate_param_indices"] = rate_params;
                    rxn_list.append(rd);
                }
                d["reactions"] = rxn_list;

                // Table functions: [{name, index_kind, index_param_idx,
                // index_obs_idx}, ...] in dispatch-id order. Lets the
                // model-based codegen path emit
                //   data->tfun_eval(tf_id, idx_c, data->tfun_ctx)
                // for any BNGL function whose name appears here. Empty
                // for SBML/Antimony models without tfuns.
                py::list tf_list;
                for (const auto &spec : m.table_function_specs()) {
                    py::dict td;
                    td["name"] = spec.name;
                    td["index_kind"] = spec.index_kind;
                    td["index_param_idx"] = spec.index_param_idx;
                    td["index_obs_idx"] = spec.index_obs_idx;
                    tf_list.append(td);
                }
                d["table_functions"] = tf_list;

                return d;
            },
            "Return model data for code generation")

        .def("__repr__", [](const bngsim::NetworkModel &m) {
            std::ostringstream os;
            os << "NetworkModel(species=" << m.n_species() << ", reactions=" << m.n_reactions()
               << ", observables=" << m.n_observables() << ", parameters=" << m.n_parameters()
               << ")";
            return os.str();
        });

    // ─── CvodeSimulator ──────────────────────────────────────────────────────
    //
    // GIL is released during sim.run() for parallel-friendly execution.
    py::class_<bngsim::CvodeSimulator>(m, "CvodeSimulator")
        .def(py::init<bngsim::NetworkModel &>(), py::arg("model"),
             py::keep_alive<1, 2>(), // simulator keeps model alive
             "Create an ODE (CVODE) simulator for the given model")
        .def(
            "run",
            [](bngsim::CvodeSimulator &self, const bngsim::TimeSpec &times,
               const bngsim::SolverOptions &opts) {
                // Release GIL during simulation
                bngsim::Result result;
                {
                    py::gil_scoped_release release;
                    result = self.run(times, opts);
                }
                return result;
            },
            py::arg("times"), py::arg("opts") = bngsim::SolverOptions{},
            "Run ODE simulation (releases GIL)")
        .def("set_tolerances", &bngsim::CvodeSimulator::set_tolerances, py::arg("rtol"),
             py::arg("atol"))
        .def("set_max_steps", &bngsim::CvodeSimulator::set_max_steps, py::arg("max_steps"));

    // ─── SsaSimulator ────────────────────────────────────────────────────────
    py::class_<bngsim::SsaSimulator>(m, "SsaSimulator")
        .def(py::init<bngsim::NetworkModel &>(), py::arg("model"), py::keep_alive<1, 2>(),
             "Create a stochastic simulator (SSA/PSA) for the given model")
        .def(
            "run",
            [](bngsim::SsaSimulator &self, const bngsim::TimeSpec &times, uint64_t seed,
               double timeout_seconds) {
                // Release GIL during simulation
                bngsim::Result result;
                {
                    py::gil_scoped_release release;
                    result = self.run(times, seed, timeout_seconds);
                }
                return result;
            },
            py::arg("times"), py::arg("seed") = 42, py::arg("timeout_seconds") = 0.0,
            "Run exact SSA simulation with deterministic seed (releases GIL). "
            "timeout_seconds > 0 enables a wall-clock budget; on overrun, "
            "raises bngsim.SimulationTimeout.")
        .def(
            "run_psa",
            [](bngsim::SsaSimulator &self, const bngsim::TimeSpec &times, uint64_t seed,
               double poplevel, double timeout_seconds) {
                // Release GIL during simulation
                bngsim::Result result;
                {
                    py::gil_scoped_release release;
                    result = self.run_psa(times, seed, poplevel, timeout_seconds);
                }
                return result;
            },
            py::arg("times"), py::arg("seed") = 42, py::arg("poplevel") = 100.0,
            py::arg("timeout_seconds") = 0.0,
            "Run PSA (Partial Scaling Algorithm) simulation.\n"
            "Lin, Feng, Hlavacek, J. Chem. Phys. 150, 244101 (2019).\n"
            "poplevel = N_c (critical population size, must be > 1). Releases GIL. "
            "timeout_seconds > 0 enables a wall-clock budget; on overrun, "
            "raises bngsim.SimulationTimeout.")
        .def("set_propensity_library", &bngsim::SsaSimulator::set_propensity_library,
             py::arg("so_path"),
             "GH #190: supply a cc-compiled value-specialized propensity .so "
             "(symbol bngsim_ssa_propensities). When set and the model is "
             "recompute-all eligible (pure mass-action exact SSA, no events, small "
             "nr), the run takes the RR-style recompute-all + flat-scan loop by "
             "default. No-op for ineligible models; '' clears it.");

    // ─── NfsimSimulator (conditional on BNGSIM_HAS_NFSIM) ────────────────────
    //
    // Wraps the C++ NfsimSimulator for in-process rule-based stochastic
    // simulation from BNG XML files. GIL is released during simulation.
#ifdef BNGSIM_HAS_NFSIM
    py::class_<bngsim::NfsimSimulator>(m, "NfsimSimulator")
        .def(py::init<const std::string &>(), py::arg("xml_path"),
             "Create an NFsim simulator from a BNG XML file path")
        .def(
            "run",
            [](bngsim::NfsimSimulator &self, const bngsim::TimeSpec &times, uint64_t seed,
               double timeout_seconds) {
                // Release GIL during simulation
                bngsim::Result result;
                {
                    py::gil_scoped_release release;
                    result = self.run(times, seed, timeout_seconds);
                }
                return result;
            },
            py::arg("times"), py::arg("seed") = 42, py::arg("timeout_seconds") = 0.0,
            "Run NFsim simulation with deterministic seed (releases GIL). "
            "timeout_seconds > 0 enables a wall-clock budget checked between "
            "output points; on overrun, raises bngsim.SimulationTimeout.")
        .def("set_param", &bngsim::NfsimSimulator::set_param, py::arg("name"), py::arg("value"),
             "Set a parameter override (applied before each run)")
        .def("clear_param_overrides", &bngsim::NfsimSimulator::clear_param_overrides,
             "Clear all parameter overrides")
        .def("set_molecule_limit", &bngsim::NfsimSimulator::set_molecule_limit, py::arg("limit"),
             "Set global molecule limit (default: INT_MAX, no artificial limit)")
        .def("set_connectivity", &bngsim::NfsimSimulator::set_connectivity, py::arg("enabled"),
             "Enable or disable NFsim connectivity inference at XML initialization")
        .def("set_nfsim_v1143_compat", &bngsim::NfsimSimulator::set_nfsim_v1143_compat,
             py::arg("enabled"), "Enable or disable NFsim v1.14.3 selector compatibility mode")
        .def("set_block_same_complex_binding",
             &bngsim::NfsimSimulator::set_block_same_complex_binding, py::arg("enabled"),
             "Block same-complex binding (NFsim CLI -bscb). Default: false.")
        .def("set_traversal_limit", &bngsim::NfsimSimulator::set_traversal_limit, py::arg("limit"),
             "Set the universal traversal limit (NFsim CLI -utl N). Negative = auto.")
        .def_property_readonly("xml_path", &bngsim::NfsimSimulator::xml_path,
                               "Path to the XML file")
        // ── Session API (stateful, multi-action workflows) ──────────────
        .def(
            "initialize",
            [](bngsim::NfsimSimulator &self, uint64_t seed) {
                py::gil_scoped_release release;
                self.initialize(seed);
            },
            py::arg("seed") = 42, "Initialize a simulation session (parse XML, seed RNG, prepare)")
        .def(
            "step_to",
            [](bngsim::NfsimSimulator &self, double time) {
                py::gil_scoped_release release;
                self.step_to(time);
            },
            py::arg("time"), "Advance simulation to target time (no output, state preserved)")
        .def(
            "simulate",
            [](bngsim::NfsimSimulator &self, double t_start, double t_end, int n_points,
               double timeout_seconds, bool relative_time,
               const std::vector<double> &sample_times) {
                bngsim::Result result;
                {
                    py::gil_scoped_release release;
                    result = self.simulate(t_start, t_end, n_points, timeout_seconds, relative_time,
                                           sample_times);
                }
                return result;
            },
            py::arg("t_start"), py::arg("t_end"), py::arg("n_points"),
            py::arg("timeout_seconds") = 0.0, py::arg("relative_time") = false,
            py::arg("sample_times") = std::vector<double>{},
            "Simulate from t_start to t_end, capturing n_points snapshots. "
            "timeout_seconds > 0 enables a wall-clock budget checked between "
            "output points; on overrun, raises bngsim.SimulationTimeout and the "
            "session must be destroyed before reuse. "
            "relative_time=true offsets the returned time axis to start at 0 "
            "(matching BNG2.pl simulate_nf convention) without changing the "
            "internal NFsim clock. "
            "sample_times (non-empty) overrides the uniform grid: the result is "
            "labelled with exactly those sorted absolute times and the live "
            "clock advances by their span from the current session time.")
        .def("add_molecules", &bngsim::NfsimSimulator::add_molecules, py::arg("molecule_type_name"),
             py::arg("count"), "Add count molecules of named type in default (unbound) state")
        .def("get_molecule_count", &bngsim::NfsimSimulator::get_molecule_count,
             py::arg("molecule_type_name"), "Get current molecule count for named type")
        .def("get_species_count", &bngsim::NfsimSimulator::get_species_count, py::arg("pattern"),
             "Get current count for an exact single-molecule BNGL species pattern")
        .def("add_species", &bngsim::NfsimSimulator::add_species, py::arg("pattern"),
             py::arg("count"), "Add count exact single-molecule BNGL species instances")
        .def("remove_species", &bngsim::NfsimSimulator::remove_species, py::arg("pattern"),
             py::arg("count"), "Remove count exact single-molecule BNGL species instances")
        .def("set_species_count", &bngsim::NfsimSimulator::set_species_count, py::arg("pattern"),
             py::arg("count"), "Set count for an exact single-molecule BNGL species pattern")
        .def("get_observable_names", &bngsim::NfsimSimulator::get_observable_names,
             "Get observable names (session must be active)")
        .def("get_observable_values", &bngsim::NfsimSimulator::get_observable_values,
             "Get current observable values (session must be active)")
        .def("get_parameter", &bngsim::NfsimSimulator::get_parameter, py::arg("name"),
             "Get a parameter value by name (session must be active)")
        .def(
            "evaluate_expression",
            [](const bngsim::NfsimSimulator &self, const std::string &expr,
               const std::unordered_map<std::string, double> &extra) {
                return self.evaluate_expression(expr, extra);
            },
            py::arg("expr"), py::arg("extra") = std::unordered_map<std::string, double>{},
            "Evaluate a BNG expression with current overrides (and optional extra bindings)")
        .def("has_session", &bngsim::NfsimSimulator::has_session,
             "Whether a session is currently active")
        .def("save_species", &bngsim::NfsimSimulator::save_species, py::arg("path"),
             "Write the live System's molecular species to a BNG-format .species file")
        .def("save_concentrations", &bngsim::NfsimSimulator::save_concentrations,
             "Snapshot the live System's molecular state for later in-process restore "
             "(BNG saveConcentrations()).")
        .def("restore_concentrations", &bngsim::NfsimSimulator::restore_concentrations,
             "Restore the molecular state captured by the most recent "
             "save_concentrations() (BNG resetConcentrations()). Raises if none saved.")
        .def("has_saved_concentrations", &bngsim::NfsimSimulator::has_saved_concentrations,
             "Whether a save_concentrations() snapshot is available to restore")
        .def("destroy_session", &bngsim::NfsimSimulator::destroy_session,
             "Destroy the current session, freeing the NFsim System");

    m.attr("HAS_NFSIM") = true;
#else
    m.attr("HAS_NFSIM") = false;
#endif

    // ─── RuleMonkeySimulator (conditional on BNGSIM_HAS_RULEMONKEY) ─────────
    //
    // Wraps the C++ RuleMonkeySimulator for in-process exact network-free
    // simulation from BNG XML files. GIL is released during simulation.
#ifdef BNGSIM_HAS_RULEMONKEY
    py::class_<bngsim::RuleMonkeySimulator>(m, "RuleMonkeySimulator")
        .def(py::init<const std::string &>(), py::arg("xml_path"),
             "Create a RuleMonkey simulator from a BNG XML file path")
        .def(
            "run",
            [](bngsim::RuleMonkeySimulator &self, const bngsim::TimeSpec &times, uint64_t seed,
               double timeout_seconds) {
                bngsim::Result result;
                {
                    py::gil_scoped_release release;
                    result = self.run(times, seed, timeout_seconds);
                }
                return result;
            },
            py::arg("times"), py::arg("seed") = 42, py::arg("timeout_seconds") = 0.0,
            "Run RuleMonkey simulation with deterministic seed (releases GIL). "
            "timeout_seconds > 0 enables a wall-clock budget polled by upstream "
            "every ~1024 SSA events; on overrun, raises bngsim.SimulationTimeout.")
        .def("set_param", &bngsim::RuleMonkeySimulator::set_param, py::arg("name"),
             py::arg("value"), "Set a parameter override")
        .def("clear_param_overrides", &bngsim::RuleMonkeySimulator::clear_param_overrides,
             "Clear all parameter overrides")
        .def("set_molecule_limit", &bngsim::RuleMonkeySimulator::set_molecule_limit,
             py::arg("limit"), "Set global molecule limit")
        .def("set_block_same_complex_binding",
             &bngsim::RuleMonkeySimulator::set_block_same_complex_binding, py::arg("enabled"),
             "Block same-complex binding. Default: true.")
        .def_property_readonly("xml_path", &bngsim::RuleMonkeySimulator::xml_path,
                               "Path to the XML file")
        // ── Session API (stateful, multi-action workflows) ──────────────
        .def(
            "initialize",
            [](bngsim::RuleMonkeySimulator &self, uint64_t seed) {
                py::gil_scoped_release release;
                self.initialize(seed);
            },
            py::arg("seed") = 42, "Initialize a simulation session")
        .def(
            "step_to",
            [](bngsim::RuleMonkeySimulator &self, double time, double timeout_seconds) {
                py::gil_scoped_release release;
                self.step_to(time, timeout_seconds);
            },
            py::arg("time"), py::arg("timeout_seconds") = 0.0,
            "Advance simulation to target time without returning output. "
            "timeout_seconds > 0 enables a wall-clock budget; on overrun, "
            "raises bngsim.SimulationTimeout.")
        .def(
            "simulate",
            [](bngsim::RuleMonkeySimulator &self, double t_start, double t_end, int n_points,
               double timeout_seconds, const std::vector<double> &sample_times) {
                bngsim::Result result;
                {
                    py::gil_scoped_release release;
                    result = self.simulate(t_start, t_end, n_points, timeout_seconds, sample_times);
                }
                return result;
            },
            py::arg("t_start"), py::arg("t_end"), py::arg("n_points"),
            py::arg("timeout_seconds") = 0.0, py::arg("sample_times") = std::vector<double>{},
            "Simulate from t_start to t_end, capturing n_points snapshots. "
            "timeout_seconds > 0 enables a wall-clock budget polled by upstream "
            "every ~1024 SSA events; on overrun, raises bngsim.SimulationTimeout "
            "and the session must be destroyed before reuse. "
            "sample_times (non-empty) overrides the uniform grid: the result is "
            "labelled with exactly those sorted absolute times and the live "
            "session advances by their span from the current session time.")
        .def("add_molecules", &bngsim::RuleMonkeySimulator::add_molecules,
             py::arg("molecule_type_name"), py::arg("count"),
             "Add count molecules of named type in default (unbound) state")
        .def("get_molecule_count", &bngsim::RuleMonkeySimulator::get_molecule_count,
             py::arg("molecule_type_name"), "Get current molecule count for named type")
        .def("get_observable_names", &bngsim::RuleMonkeySimulator::get_observable_names,
             "Get observable names")
        .def("get_observable_values", &bngsim::RuleMonkeySimulator::get_observable_values,
             "Get current observable values")
        .def("get_parameter", &bngsim::RuleMonkeySimulator::get_parameter, py::arg("name"),
             "Get a parameter value by name")
        .def("has_session", &bngsim::RuleMonkeySimulator::has_session,
             "Whether a session is currently active")
        .def("current_time", &bngsim::RuleMonkeySimulator::current_time,
             "Get the active session current time")
        .def("destroy_session", &bngsim::RuleMonkeySimulator::destroy_session,
             "Destroy the current session")
        .def("save_state", &bngsim::RuleMonkeySimulator::save_state, py::arg("path"),
             "Save the active session state")
        .def("load_state", &bngsim::RuleMonkeySimulator::load_state, py::arg("path"),
             "Load a session state")
        // ── Exact-species queries & mutations + expression eval ─────────
        .def("get_species_count", &bngsim::RuleMonkeySimulator::get_species_count,
             py::arg("pattern"),
             "Get current count for an exact, fully-specified BNGL species pattern")
        .def("add_species", &bngsim::RuleMonkeySimulator::add_species, py::arg("pattern"),
             py::arg("count"), "Add count exact BNGL species instances")
        .def("remove_species", &bngsim::RuleMonkeySimulator::remove_species, py::arg("pattern"),
             py::arg("count"), "Remove count exact BNGL species instances")
        .def("set_species_count", &bngsim::RuleMonkeySimulator::set_species_count,
             py::arg("pattern"), py::arg("count"),
             "Set count for an exact BNGL species pattern (adds/removes the diff)")
        .def("save_species", &bngsim::RuleMonkeySimulator::save_species, py::arg("path"),
             "Write the live pool to a BNG-format .species file (readNFspecies-compatible)")
        .def(
            "evaluate_expression",
            [](bngsim::RuleMonkeySimulator &self, const std::string &expr,
               const std::unordered_map<std::string, double> &extra) {
                return self.evaluate_expression(expr, extra);
            },
            py::arg("expr"), py::arg("extra") = std::unordered_map<std::string, double>{},
            "Evaluate a BNG expression against the active session (optional extra bindings)");

    m.attr("HAS_RULEMONKEY") = true;
#else
    m.attr("HAS_RULEMONKEY") = false;
#endif

    // ─── KLU sparse linear solver availability (GH #209) ──────────────────────
    // True when this extension was built with SuiteSparse/KLU (BNGSIM_HAS_KLU),
    // so the ODE backend can route large/sparse models to the KLU direct solver
    // instead of the dense O(N^3) fallback. Surfaced to Python so downstream
    // tools (PyBNF, PyBioNetGen) and bngsim.capabilities() can detect a
    // dense-only install programmatically rather than discovering it on a
    // multi-hour genome-scale run. Mirrors codegen_jacobian_plan()["has_klu"].
#ifdef BNGSIM_HAS_KLU
    m.attr("HAS_KLU") = true;
#else
    m.attr("HAS_KLU") = false;
#endif

    // True when this extension embeds the vendored MIR micro-JIT codegen backend
    // (BNGSIM_ENABLE_MIR=ON, GH #78). When true, setting BNGSIM_CODEGEN_JIT=mir
    // routes the code-generated ODE RHS through an in-process c2mir + MIR_gen JIT
    // instead of shelling out to `cc` + dlopen — a compiler-free codegen path.
    // Surfaced so tests and downstream tools can gate the JIT path (and its
    // cc-equivalence check) on whether the backend was actually compiled in.
#ifdef BNGSIM_HAS_MIR
    m.attr("HAS_MIR") = true;
#else
    m.attr("HAS_MIR") = false;
#endif

    // ─── ModelBuilder (programmatic model construction) ──────────────────────
    py::class_<bngsim::ModelBuilder>(m, "ModelBuilder")
        .def(py::init<>())
        .def("add_parameter", &bngsim::ModelBuilder::add_parameter, py::arg("name"),
             py::arg("value"), py::arg("expression") = "", py::arg("is_expression") = false,
             "Add a parameter. Returns 0-based index.")
        .def("add_species", &bngsim::ModelBuilder::add_species, py::arg("name"),
             py::arg("init_conc"), py::arg("fixed") = false, py::arg("volume_factor") = 1.0,
             py::arg("amount_valued") = false, py::arg("reported") = true,
             "Add a species. Returns 0-based index. volume_factor is the storage→amount "
             "conversion factor used by the SSA fire step (Δstorage = ±1/volume_factor "
             "per fire); SBML loaders pass the species's compartment volume V_c. Default "
             "1.0 leaves .net and V=1 SBML behavior unchanged. amount_valued=True makes "
             "the species participate in reaction species factors by its amount (stored × "
             "volume_factor) rather than the stored concentration (GH #75, "
             "hasOnlySubstanceUnits=true semantics); default False is byte-identical. "
             "reported=False keeps the species as full integrator state but omits it from "
             "the trajectory output columns (GH #71 — event-mutated parameters/compartments "
             "promoted to species); default True is byte-identical.")
        .def("add_species_param_ref", &bngsim::ModelBuilder::add_species_param_ref,
             py::arg("species_idx0"), py::arg("param_name"),
             "Record that species[species_idx0]'s initial concentration is "
             "controlled by the parameter named param_name. Used by CVODES "
             "forward sensitivity analysis to seed dY_i(0)/dp_k = 1 when "
             "p_k is requested via sensitivity_params.")
        .def(
            "add_observable",
            [](bngsim::ModelBuilder &self, const std::string &name,
               const std::vector<std::pair<int, double>> &entries) {
                return self.add_observable(name, entries);
            },
            py::arg("name"), py::arg("entries"),
            "Add an observable. entries = [(sp_idx_0based, factor), ...]. Returns 0-based index.")
        .def("add_function", &bngsim::ModelBuilder::add_function, py::arg("name"),
             py::arg("expression"), "Add a function (named expression). Returns 0-based index.")
        .def(
            "add_reaction",
            [](bngsim::ModelBuilder &self, const std::vector<int> &reactants,
               const std::vector<int> &products, const std::string &type_str,
               const std::string &rate_law, double stat_factor, bool apply_species_factor,
               double ssa_volume_factor, bool per_species_volume_scaling, bool is_rate_rule_ode,
               int ssa_live_volume_idx0, double ssa_live_volume_exp, bool ode_only) {
                bngsim::RateLawType type;
                if (type_str == "elementary" || type_str == "Elementary")
                    type = bngsim::RateLawType::Elementary;
                else if (type_str == "functional" || type_str == "Functional")
                    type = bngsim::RateLawType::Functional;
                else if (type_str == "mm" || type_str == "MichaelisMenten")
                    type = bngsim::RateLawType::MichaelisMenten;
                else
                    throw py::value_error("Unknown rate law type: " + type_str +
                                          ". Use 'elementary', 'functional', or 'mm'.");
                return self.add_reaction(reactants, products, type, rate_law, stat_factor,
                                         apply_species_factor, ssa_volume_factor,
                                         per_species_volume_scaling, is_rate_rule_ode,
                                         ssa_live_volume_idx0, ssa_live_volume_exp, ode_only);
            },
            py::arg("reactants"), py::arg("products"), py::arg("type"), py::arg("rate_law"),
            py::arg("stat_factor") = 1.0, py::arg("apply_species_factor") = true,
            py::arg("ssa_volume_factor") = 1.0, py::arg("per_species_volume_scaling") = false,
            py::arg("is_rate_rule_ode") = false, py::arg("ssa_live_volume_idx0") = -1,
            py::arg("ssa_live_volume_exp") = 0.0, py::arg("ode_only") = false,
            "Add a reaction. type = 'elementary', 'functional', or 'mm'. "
            "apply_species_factor=False marks an SBML-style functional rate where the "
            "kinetic-law expression already includes the reactant population factor. "
            "ssa_volume_factor is an SSA-only multiplier converting an ODE-units rate "
            "(storage/time) to amount/time propensity; SBML loaders pass V_c (the "
            "reaction's compartment volume). Default 1.0 leaves .net and V=1 SBML "
            "behavior unchanged. per_species_volume_scaling=True is the cross-compartment "
            "ODE accumulator: compute_derivs divides the rate by each affected species's "
            "volume_factor instead of using a scalar rate. Default false preserves .net "
            "and uniform-V_s SBML behavior. is_rate_rule_ode=True (GH #81) marks a "
            "Functional `[] -> [X]` reaction compiled from an SBML rate rule; the SSA/PSA "
            "loop integrates X deterministically instead of firing it as a stochastic "
            "channel (ODE path unaffected). ssa_live_volume_idx0/ssa_live_volume_exp "
            "(GH #81) apply the SSA live-volume correction for a variable-volume "
            "compartment: the propensity is multiplied by (ssa_volume_factor/conc[idx])^exp "
            "where idx is the promoted compartment species and exp = n_h-1. Default "
            "idx0=-1 leaves .net / static-V byte-identical. Returns 0-based index.")
        .def("set_reaction_live_volume", &bngsim::ModelBuilder::set_reaction_live_volume,
             py::arg("rxn_idx0"), py::arg("ssa_live_volume_idx0"), py::arg("ssa_live_volume_exp"),
             "GH #81: set the SSA live-volume correction on an already-added reaction "
             "(rxn_idx0 = 0-based index from add_reaction). Used by the SBML loader to "
             "tag a variable-volume reaction once its event-resized compartment has been "
             "promoted to a species (which happens after reaction emission). No-op if "
             "rxn_idx0 is out of range.")
        .def("set_species_ode_live_volume", &bngsim::ModelBuilder::set_species_ode_live_volume,
             py::arg("species_idx0"), py::arg("live_idx0"),
             "GH #144 (case 4): set the cross-compartment ODE live-volume divide on a "
             "species. compute_derivs divides this species's per-species accumulation by "
             "conc[live_idx0] (the promoted compartment species = V_live) instead of its "
             "static volume_factor. No-op if species_idx0 is out of range.")
        .def("set_species_rateof_amount", &bngsim::ModelBuilder::set_species_rateof_amount,
             py::arg("species_idx0"),
             "GH #231 (rateOf): mark a hasOnlySubstanceUnits=true species so its rateOf "
             "csymbol reports the amount-rate (volume_factor * stored-rate) instead of the "
             "stored d(conc)/dt. Correct for constant- and variable-volume compartments alike "
             "(the integrator stores amount/V_static). No-op if species_idx0 is out of range.")
        .def("add_reaction_live_volume_term", &bngsim::ModelBuilder::add_reaction_live_volume_term,
             py::arg("rxn_idx0"), py::arg("live_idx0"), py::arg("v_static"), py::arg("exp"),
             "GH #144 (case 4): append a cross-compartment SSA live-volume term to a "
             "reaction. compute_rxn_rate multiplies the SSA propensity by "
             "(v_static / conc[live_idx0])^exp. One term per variable-volume compartment "
             "the reaction touches. No-op if rxn_idx0 is out of range.")
        .def(
            "add_event",
            [](bngsim::ModelBuilder &self, const std::string &id, const std::string &trigger_expr,
               const std::vector<std::pair<int, std::string>> &assignments, double delay,
               int priority, bool persistent, bool initial_value, bool use_values_from_trigger_time,
               const std::string &delay_expr, const std::string &priority_expr,
               const std::vector<bool> &assignment_ode_only) {
                self.add_event(id, trigger_expr, assignments, delay, priority, persistent,
                               initial_value, use_values_from_trigger_time, delay_expr,
                               priority_expr, assignment_ode_only);
            },
            py::arg("id"), py::arg("trigger_expr"), py::arg("assignments"), py::arg("delay") = 0.0,
            py::arg("priority") = 0, py::arg("persistent") = true, py::arg("initial_value") = true,
            py::arg("use_values_from_trigger_time") = true, py::arg("delay_expr") = "",
            py::arg("priority_expr") = "", py::arg("assignment_ode_only") = std::vector<bool>{},
            "Add a discrete event. assignments = [(sp_idx_0based, value_expr_str), ...]. "
            "delay_expr/priority_expr (when non-empty) are evaluated at trigger time and "
            "override the constant delay/priority. assignment_ode_only (GH #81) is a "
            "parallel bool list; true entries apply under ODE only and are skipped under "
            "SSA (the compartment-resize concentration rescale that must not perturb counts).")
        .def("set_compute_conservation_laws", &bngsim::ModelBuilder::set_compute_conservation_laws,
             py::arg("enabled"),
             "Enable/disable conservation-law detection in build() (GH #102). The "
             "detector is dense O(n_species^3) Gaussian elimination consumed only by the "
             "steady-state solver; disable it to keep setup O(reactions) for very large "
             "ODE-only networks (~100K species). Default True preserves existing behavior.")
        .def("add_discontinuity_trigger", &bngsim::ModelBuilder::add_discontinuity_trigger,
             py::arg("condition_expr"),
             "Register a time-dependent inequality condition (e.g. 'time()<=0.125') as a "
             "discontinuity trigger (GH #72). It is compiled and registered as a CVODE root so "
             "the ODE integrator stops at the time threshold and cannot step over a narrow "
             "forcing pulse encoded by a piecewise assignment rule. No state assignment; "
             "duplicate strings are ignored.")
        .def("enable_rateof", &bngsim::ModelBuilder::enable_rateof,
             "Enable SBML rateOf csymbol support (GH #106). Call when the model references "
             "rateOf(species) in event triggers or rate-rule / assignment-rule functions "
             "(emitted as 'rate_of__<species>' tokens). build() then sizes the live "
             "derivative buffer and registers the accessor variables those expressions read.")
        .def("build", &bngsim::ModelBuilder::build,
             "Finalize and build the NetworkModel. Builder is consumed.");

    // ─── Module-level functions ──────────────────────────────────────────────

    m.def(
        "reserved_names",
        []() {
            auto names = bngsim::reserved_names();
            py::dict d;
            d["constants"] = names.constants;
            d["functions"] = names.functions;
            return d;
        },
        "Return dict of reserved constant and function names");

    // --- SteadyStateOptions ---
    py::class_<bngsim::SteadyStateOptions>(m, "SteadyStateOptions")
        .def(py::init<>())
        .def_readwrite("tol", &bngsim::SteadyStateOptions::tol)
        .def_readwrite("max_time", &bngsim::SteadyStateOptions::max_time)
        .def_readwrite("rtol", &bngsim::SteadyStateOptions::rtol)
        .def_readwrite("atol", &bngsim::SteadyStateOptions::atol)
        .def_readwrite("max_steps", &bngsim::SteadyStateOptions::max_steps)
        .def_readwrite("method", &bngsim::SteadyStateOptions::method)
        .def_readwrite("jacobian", &bngsim::SteadyStateOptions::jacobian)
        .def_readwrite("codegen_so_path", &bngsim::SteadyStateOptions::codegen_so_path)
        .def_readwrite("sensitivity_params", &bngsim::SteadyStateOptions::sensitivity_params);

    // --- SteadyStateResult ---
    py::class_<bngsim::SteadyStateResult>(m, "SteadyStateResultCore")
        .def(py::init<>())
        .def_readonly("concentrations", &bngsim::SteadyStateResult::concentrations)
        .def_readonly("species_names", &bngsim::SteadyStateResult::species_names)
        .def_readonly("residual", &bngsim::SteadyStateResult::residual)
        .def_readonly("method_used", &bngsim::SteadyStateResult::method_used)
        .def_readonly("converged", &bngsim::SteadyStateResult::converged)
        .def_readonly("n_steps", &bngsim::SteadyStateResult::n_steps)
        .def_readonly("n_rhs_evals", &bngsim::SteadyStateResult::n_rhs_evals)
        .def_readonly("n_sens_params", &bngsim::SteadyStateResult::n_sens_params)
        .def_readonly("sens_param_names", &bngsim::SteadyStateResult::sens_param_names)
        .def_property_readonly("sensitivity_data", [](const bngsim::SteadyStateResult &r) {
            int ns = static_cast<int>(r.concentrations.size());
            int np = r.n_sens_params;
            if (np == 0 || r.sensitivity.empty()) {
                return py::array_t<double>(py::array::ShapeContainer{0, 0});
            }
            auto *vec = new std::vector<double>(r.sensitivity);
            auto capsule =
                py::capsule(vec, [](void *p) { delete static_cast<std::vector<double> *>(p); });
            return py::array_t<double>({ns, np},
                                       {static_cast<py::ssize_t>(np * sizeof(double)),
                                        static_cast<py::ssize_t>(sizeof(double))},
                                       vec->data(), capsule);
        });

    // --- find_steady_state ---
    m.def(
        "find_steady_state",
        [](bngsim::NetworkModel &model, const bngsim::SteadyStateOptions &opts) {
            bngsim::SteadyStateResult result;
            {
                py::gil_scoped_release release;
                result = bngsim::find_steady_state(model, opts);
            }
            return result;
        },
        py::arg("model"), py::arg("opts") = bngsim::SteadyStateOptions{},
        "Find steady state of the ODE system (releases GIL)");

    // Version info — set via the BNGSIM_VERSION_STR compile define so
    // pyproject.toml stays the single source of truth (issue #31).
#ifndef BNGSIM_VERSION_STR
#error "BNGSIM_VERSION_STR not defined — CMake should set it from pyproject.toml"
#endif
    m.attr("__version__") = BNGSIM_VERSION_STR;

    // GH #84: whether this build links the optimized BLAS dense factor backend
    // (Accelerate / system LAPACK). When false, the dense path always uses the
    // built-in dense LU and the density-aware gate is a no-op.
#ifdef BNGSIM_HAS_LAPACK_DENSE
    m.attr("HAS_LAPACK_DENSE") = true;
#else
    m.attr("HAS_LAPACK_DENSE") = false;
#endif

    // Build-provenance stamp (issue #125): the source commit this binary was
    // compiled from, set via the BNGSIM_BUILD_COMMIT_STR compile define. Unlike
    // __version__ (which only moves on a pyproject bump), this moves with every
    // C++ edit, so the Python-side guard (bngsim/_build_provenance.py) and the
    // investigation oracles can print exactly which source produced the loaded
    // binary. Defaults to "unknown" if CMake could not resolve git.
#ifndef BNGSIM_BUILD_COMMIT_STR
#define BNGSIM_BUILD_COMMIT_STR "unknown"
#endif
    m.attr("__build_commit__") = BNGSIM_BUILD_COMMIT_STR;

    // ── GH #190: structure-specialized SSA propensity-vector codegen ────────────
    // Emit the model's structure-specialized propensity source (rate constants
    // read from a runtime params[] arg, only the structural factor baked).
    m.def(
        "emit_ssa_propensity_source_structure",
        [](bngsim::NetworkModel &model) { return model.emit_ssa_propensity_source_structure(); },
        py::arg("model"),
        "Emit STRUCTURE-specialized C for the full SSA propensity vector — rate "
        "constants read from a runtime params[] argument, signature "
        "bngsim_ssa_propensities(const double* x, const double* p, double* a); "
        "the source/.so cache key depends only on model structure. Returns "
        "(source, n_unsupported).");

    // Benchmark the structure-specialized propensity kernel (cc + MIR) against the
    // per-reaction compute_propensity() loop. All timing is native (C++ loops), so
    // it isolates the kernel from Python overhead. n_iters full-vector fills each.
    m.def(
        "bench_ssa_propensity_jit",
        [](bngsim::NetworkModel &model, int n_iters) {
            namespace chr = std::chrono;
            const int ns = model.n_species();
            const int nr = model.n_reactions();
            const int last = nr > 0 ? nr - 1 : 0;
            std::vector<double> x(static_cast<size_t>(ns), 0.0);
            model.get_state_into(x.data());
            std::vector<double> pvals;
            for (const auto &pp : model.parameters())
                pvals.push_back(pp.value);
            std::vector<double> a_ref(static_cast<size_t>(nr), 0.0);
            std::vector<double> a_k(static_cast<size_t>(nr), 0.0);

            // Reference: the current per-reaction native propensity loop.
            volatile double sink = 0.0;
            auto t0 = chr::steady_clock::now();
            for (int it = 0; it < n_iters; ++it) {
                for (int r = 0; r < nr; ++r)
                    a_ref[r] = model.compute_propensity(r, x.data());
                sink += a_ref[last];
            }
            auto t1 = chr::steady_clock::now();
            const double per_rxn_ns =
                chr::duration<double, std::nano>(t1 - t0).count() / std::max(1, n_iters);

            auto emitted = model.emit_ssa_propensity_source_structure();
            const std::string &src = emitted.first;
            const int n_unsupported = emitted.second;
            using PropFn = void (*)(const double *, const double *, double *);

            auto bench_backend = [&](auto make_fn) -> std::pair<double, double> {
                try {
                    PropFn fn = make_fn();
                    auto j0 = chr::steady_clock::now();
                    for (int it = 0; it < n_iters; ++it) {
                        fn(x.data(), pvals.data(), a_k.data());
                        sink += a_k[last];
                    }
                    auto j1 = chr::steady_clock::now();
                    double ns_per =
                        chr::duration<double, std::nano>(j1 - j0).count() / std::max(1, n_iters);
                    double mrel = 0.0;
                    for (int r = 0; r < nr; ++r) {
                        const double d = std::fabs(a_k[r] - a_ref[r]);
                        const double mg = std::max(std::fabs(a_k[r]), std::fabs(a_ref[r]));
                        if (mg > 1e-300)
                            mrel = std::max(mrel, d / mg);
                    }
                    return {ns_per, mrel};
                } catch (const std::exception &) {
                    return {-1.0, -1.0};
                }
            };

            // MIR in-process kernel (if built); cc -O3 kernel (the shipped path).
            std::unique_ptr<bngsim::MirJit> mir;
            auto [mir_ns, mir_max_rel] = bench_backend([&]() {
                mir = std::make_unique<bngsim::MirJit>(src);
                return mir->symbol<PropFn>("bngsim_ssa_propensities");
            });
            std::unique_ptr<bngsim::CcJit> cc;
            double cc_compile_ms = -1.0;
            auto [cc_ns, cc_max_rel] = bench_backend([&]() {
                std::remove(bngsim::CcJit::cache_path(src).c_str());
                auto k0 = chr::steady_clock::now();
                cc = std::make_unique<bngsim::CcJit>(src);
                auto k1 = chr::steady_clock::now();
                cc_compile_ms = chr::duration<double, std::milli>(k1 - k0).count();
                return cc->symbol<PropFn>("bngsim_ssa_propensities");
            });
            (void) sink;

            py::dict out;
            out["n_species"] = ns;
            out["n_reactions"] = nr;
            out["n_unsupported"] = n_unsupported;
            out["per_rxn_ns"] = per_rxn_ns;
            out["mir_ns"] = mir_ns;
            out["mir_max_rel"] = mir_max_rel;
            out["cc_ns"] = cc_ns;
            out["cc_compile_ms"] = cc_compile_ms;
            out["cc_max_rel"] = cc_max_rel;
            out["cc_speedup"] = cc_ns > 0.0 ? per_rxn_ns / cc_ns : 0.0;
            out["source_bytes"] = static_cast<int>(src.size());
            return out;
        },
        py::arg("model"), py::arg("n_iters") = 200000,
        "GH #190: benchmark the structure-specialized SSA propensity kernel "
        "(cc -O3 and MIR) vs the per-reaction compute_propensity loop.");
}
