// bngsim/include/bngsim/result.hpp — Simulation result container
//
// Stores simulation output in memory as arrays of species, observables,
// expressions, and optional sensitivities.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace bngsim {

// Which direct linear solver the ODE integrator actually used for a run
// (SolverStats::linear_solver). Reported for benchmark observability — in
// particular to confirm the GH #84 density-aware gate routed a model to the
// BLAS dense factor vs the built-in dense LU vs KLU.
enum LinearSolverKind {
    LINEAR_SOLVER_DENSE = 0,        // SUNDIALS built-in unblocked dense LU
    LINEAR_SOLVER_KLU = 1,          // KLU sparse direct (low-density Jacobians)
    LINEAR_SOLVER_LAPACK_DENSE = 2, // GH #84: BLAS dgetrf factor + built-in back-solve
};

struct SolverStats {
    int n_steps = 0;                         // total internal solver steps
    int n_rhs_evals = 0;                     // RHS function evaluations
    int n_jac_evals = 0;                     // Jacobian evaluations
    int n_err_test_fails = 0;                // error test failures
    int n_nonlin_iters = 0;                  // nonlinear solver iterations
    int n_nonlin_conv_fails = 0;             // nonlinear (Newton) convergence failures —
                                             // the most direct robustness signal for the
                                             // Jacobian: an inexact Jacobian makes Newton
                                             // give up more often, forcing step cuts
    int linear_solver = LINEAR_SOLVER_DENSE; // LinearSolverKind actually used

    // GH #132 adaptive gate: of this run's dense factorizations, how many took the
    // BLAS dgetrf path. Only meaningful when linear_solver == LAPACK_DENSE — that
    // solver runs the built-in dense GETRF for the first K factorizations (K=5 by
    // default) and only switches to dgetrf once a run refactors past K at N>=256,
    // so a short LAPACK-dense run reports 0 here (it stayed byte-identical to the
    // built-in dense LU). 0 on every other solver/backend.
    int n_dense_blas_factorizations = 0;

    // Steady-state early-termination flags (set when SolverOptions::steady_state
    // is true). steady_state_reached is true iff ||f||/n fell below the
    // tolerance before t_end; steady_state_residual records the final
    // ||f(t,y)||_2 / n_species value at the last recorded row.
    bool steady_state_reached = false;
    double steady_state_residual = 0.0;
};

// GH #110 — SSA boundary diagnostics. The exact SSA evaluates rate laws
// literally: it neither floors species at zero nor suppresses a reaction whose
// rate law evaluates negative. Instead a negative-rate reaction fires in
// REVERSE (products → reactants) with propensity |v|, so the SSA's expected
// drift equals the ODE RHS (ν·v) at every state — the two engines agree at the
// level of means. Both boundary events are otherwise invisible, so we surface
// them: the Python layer turns nonzero counts into one filterable warning each.
// These are modeler-facing signals, not errors — non-negativity and reaction
// directionality are the rate law's responsibility (e.g. piecewise(X<=0,0,k)).
// Default-constructed (all zero / empty) on every non-SSA result.
struct SsaDiagnostics {
    // A species count crossed from >= 0 to < 0 after a fire (no floor applied).
    long n_negative_crossings = 0;
    std::string first_negative_species; // name of the first species to cross; empty ⇒ none

    // A reaction's rate law evaluated negative and was fired in reverse.
    long n_reverse_fires = 0;
    std::string first_reverse_reaction; // label of the first reaction reversed; empty ⇒ none

    // GH #190 — how propensities were evaluated this run, for accurate reporting
    // (parity matrix / instrumentation). One of: "cc" (cc-compiled .so +
    // recompute-all), "mir" (in-process MIR JIT + recompute-all), or "interpreted"
    // (per-reaction compute_propensity + Fenwick — native arithmetic for
    // mass-action, ExprTk for Functional rate laws). Default "interpreted".
    std::string propensity_backend = "interpreted";
};

class Result {
  public:
    Result();
    ~Result();

    // Allocate storage for n_times output points
    void allocate(int n_times, int n_species, int n_observables);

    // Record a single time point (called by simulators)
    void record(int time_index, double t,
                const double *species_conc,     // n_species values
                const double *observable_vals); // n_observables values

    // Record expression (function) values at a time point.
    // Must be called after set_expression_names() and allocate().
    void record_expressions(int time_index,
                            const double *expr_vals); // n_expressions values

    // ─── Accessors ───────────────────────────────────────────────────────────
    int n_times() const;
    int n_species() const;
    int n_observables() const;

    // Raw data access (row-major: [time_index * n_cols + col_index])
    const std::vector<double> &time() const;
    const std::vector<double> &species_data() const;    // n_times × n_species
    const std::vector<double> &observable_data() const; // n_times × n_observables
    const std::vector<double> &expression_data() const; // n_times × n_expressions
    int n_expressions() const;

    // Named access helpers
    void set_species_names(const std::vector<std::string> &names);
    void set_observable_names(const std::vector<std::string> &names);
    void set_expression_names(const std::vector<std::string> &names);
    const std::vector<std::string> &species_names() const;
    const std::vector<std::string> &observable_names() const;
    const std::vector<std::string> &expression_names() const;

    // Trajectory-output projection (GH #71). 0-based indices, in species order,
    // of the species that should appear as trajectory columns. Empty (the
    // default) means "all species reported" — no projection, byte-identical.
    // Set by the simulators from NetworkModel::reported_species_indices() only
    // when at least one species is unreported (an event-mutated parameter/
    // compartment promoted to a species). The Python bindings and to_cdat
    // project species_data/species_names to this subset; the stored species_
    // matrix and species_names_ stay complete so sensitivities and internal
    // indexing are unaffected.
    void set_reported_species_indices(std::vector<std::size_t> indices);
    const std::vector<std::size_t> &reported_species_indices() const;

    // Solver diagnostics
    SolverStats &solver_stats();
    const SolverStats &solver_stats() const;

    // SSA boundary diagnostics (GH #110). Populated only by SsaSimulator;
    // default-constructed (all zero) on every other backend.
    SsaDiagnostics &ssa_diagnostics();
    const SsaDiagnostics &ssa_diagnostics() const;

    // ─── Sensitivity data (CVODES forward sensitivities) ────────────────────
    //
    // dY_i/dp_j at each output time point. Only populated when CVODES
    // sensitivity analysis is enabled (opts.sensitivity.param_names non-empty).
    //
    // Layout: sensitivities_[time_idx * (ns * np) + species_idx * np + param_idx]
    // where ns = n_species, np = n_sens_params.
    void allocate_sensitivities(int n_times, int n_species, int n_params);
    void record_sensitivities(int time_index, const double *const *sens_data, int n_species,
                              int n_params);
    void set_sens_param_names(const std::vector<std::string> &names);
    const std::vector<double> &sensitivity_data() const;
    int n_sens_params() const;
    const std::vector<std::string> &sens_param_names() const;

    // ─── Initial-condition sensitivity data (CVODES forward IC sensitivities) ──
    //
    // dY_i(t)/dY_k(0) at each output time, where k iterates over the species
    // listed in opts.sensitivity.ic_species_names. Stored separately from
    // parameter sensitivities to keep the two axes semantically distinct.
    //
    // Layout: sensitivities_ic_[time_idx * (ns * nic) + species_idx * nic + ic_idx]
    void allocate_sensitivities_ic(int n_times, int n_species, int n_ic);
    void record_sensitivities_ic(int time_index, const double *const *sens_data, int n_species,
                                 int n_ic);
    void set_sens_ic_species_names(const std::vector<std::string> &names);
    const std::vector<double> &sensitivity_ic_data() const;
    int n_sens_ic_species() const;
    const std::vector<std::string> &sens_ic_species_names() const;

    // ─── Observable / expression output sensitivities (GH #196) ──────────────
    //
    // Chain-rule sensitivities of observable and expression (function) outputs.
    // Storage only — populated by a later stage (the sensitivity-stitching
    // issue); empty until then, and every accessor returns an empty buffer.
    //
    // Two axes are reused, not duplicated: the parameter axis is the same as the
    // species parameter sensitivities (sens_param_names()), and the IC axis is
    // the same as the species IC sensitivities (sens_ic_species_names()) — an
    // observable/expression is differentiated wrt the same parameters / initial
    // conditions as a species. The row axis is the observable (resp. expression)
    // list, in the simulator's recording order.
    //
    // Layout mirrors the species blocks:
    //   observable_sensitivities_   [t*(no*np)  + o*np  + p]  d obs_o  / dp_p
    //   expression_sensitivities_   [t*(ne*np)  + e*np  + p]  d expr_e / dp_p
    //   observable_sensitivities_ic_[t*(no*nic) + o*nic + k]  d obs_o  / dY_k(0)
    //   expression_sensitivities_ic_[t*(ne*nic) + e*nic + k]  d expr_e / dY_k(0)
    // The row count (no / ne) is whatever the recording stage allocated; it is
    // recovered from the buffer size, so no separate count field is stored.
    void allocate_observable_sensitivities(int n_times, int n_observables, int n_params);
    void record_observable_sensitivities(int time_index, const double *const *sens_data,
                                         int n_observables, int n_params);
    const std::vector<double> &observable_sensitivity_data() const;

    void allocate_expression_sensitivities(int n_times, int n_expressions, int n_params);
    void record_expression_sensitivities(int time_index, const double *const *sens_data,
                                         int n_expressions, int n_params);
    const std::vector<double> &expression_sensitivity_data() const;

    void allocate_observable_sensitivities_ic(int n_times, int n_observables, int n_ic);
    void record_observable_sensitivities_ic(int time_index, const double *const *sens_data,
                                            int n_observables, int n_ic);
    const std::vector<double> &observable_sensitivity_ic_data() const;

    void allocate_expression_sensitivities_ic(int n_times, int n_expressions, int n_ic);
    void record_expression_sensitivities_ic(int time_index, const double *const *sens_data,
                                            int n_expressions, int n_ic);
    const std::vector<double> &expression_sensitivity_ic_data() const;

    // Shrink the recorded buffers to the first ``new_n_times`` rows. Used by
    // CvodeSimulator after a ``steady_state`` early-stop to drop the pre-
    // allocated rows that were never integrated. Must satisfy
    // ``0 <= new_n_times <= n_times()``; calling with new_n_times == n_times()
    // is a no-op.
    void truncate(int new_n_times);

    // ─── Export (convenience, not core) ──────────────────────────────────────
    // BNG .gdat format. Function headers are always bare (no "()") and
    // identical across every simulation method (issue #58). With
    // print_functions=true, user-named function columns are appended after the
    // observables; with print_rate_laws=true, the auto-generated _rateLawN
    // columns are also appended (still bare). Both default false keeps the
    // observables-only output.
    void to_gdat(const std::string &path, bool print_functions = false,
                 bool print_rate_laws = false) const;
    void to_cdat(const std::string &path) const; // BNG .cdat format

  private:
    int n_times_ = 0;
    int n_species_ = 0;
    int n_observables_ = 0;

    std::vector<double> time_;
    std::vector<double> species_;     // row-major [time][species]
    std::vector<double> observables_; // row-major [time][observable]

    int n_expressions_ = 0;
    std::vector<double> expressions_; // row-major [time][expression]

    // Sensitivity data
    int n_sens_params_ = 0;
    std::vector<double> sensitivities_; // row-major [time][species][param]
    std::vector<std::string> sens_param_names_;

    // IC sensitivity data
    int n_sens_ic_species_ = 0;
    std::vector<double> sensitivities_ic_; // row-major [time][species][ic_species]
    std::vector<std::string> sens_ic_species_names_;

    // GH #196 — observable/expression output sensitivities (storage only; empty
    // until a later stage populates them). Parameter axis = sens_param_names_;
    // IC axis = sens_ic_species_names_. Row counts recovered from buffer size.
    std::vector<double> observable_sensitivities_;    // [time][observable][param]
    std::vector<double> expression_sensitivities_;    // [time][expression][param]
    std::vector<double> observable_sensitivities_ic_; // [time][observable][ic_species]
    std::vector<double> expression_sensitivities_ic_; // [time][expression][ic_species]

    std::vector<std::string> species_names_;
    std::vector<std::string> observable_names_;
    std::vector<std::string> expression_names_;

    // GH #71 — empty ⇒ all species reported (no projection).
    std::vector<std::size_t> reported_species_indices_;

    SolverStats stats_;
    SsaDiagnostics ssa_diag_;
};

} // namespace bngsim
