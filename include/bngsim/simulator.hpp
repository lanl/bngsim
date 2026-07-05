// bngsim/include/bngsim/simulator.hpp — Simulator interfaces (ODE + SSA)
//
// CvodeSimulator wraps SUNDIALS v7 CVODE.
// SsaSimulator implements Gillespie's direct method.

#pragma once

#include "bngsim/model.hpp"
#include "bngsim/result.hpp"
#include "bngsim/types.hpp"

#include <cstdint>
#include <memory>

namespace bngsim {

// ─── CVODE ODE Simulator ─────────────────────────────────────────────────────
//
// Uses BDF method with Newton iteration (correct for stiff biochemical systems).
// Auto-selects dense (N<50) or sparse (KLU, N≥50) direct solver based on
// model size. Jacobian sparsity pattern is computed at load time from the
// stoichiometry matrix (CSC format for SUNDIALS/KLU).
// SUNDIALS v7 SUNContext for per-instance re-entrancy.
class CvodeSimulator {
  public:
    explicit CvodeSimulator(NetworkModel &model);
    ~CvodeSimulator();

    // Non-copyable, non-movable (owns SUNDIALS resources)
    CvodeSimulator(const CvodeSimulator &) = delete;
    CvodeSimulator &operator=(const CvodeSimulator &) = delete;

    // Run simulation, return results
    Result run(const TimeSpec &times, const SolverOptions &opts = {});

    // Solver configuration
    void set_tolerances(double rtol, double atol);
    void set_max_steps(int max_steps);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ─── SSA Simulator ───────────────────────────────────────────────────────────
//
// Gillespie's direct method with dependency graph + Fenwick tree.
// Dependency graph: O(k) propensity updates per step (k ≈ 5–20).
// Fenwick tree: O(log N) reaction selection (binary indexed tree).
// Per-instance RNG (std::mt19937_64). Deterministic seeding.
//
// When poplevel > 0, uses the Partial Scaling Algorithm (PSA) of
// Lin, Feng, Hlavacek, J. Chem. Phys. 150, 244101 (2019):
//   - N_min^r = min population among REACTANT species only (Eq. 14)
//   - Per-reaction scaling factor λ_r = 1/max(1, ⌊N_min^r / N_c⌋)
//   - Propensities scaled down by λ_r
//   - Stoichiometric coefficients scaled up by 1/λ_r
//   - N_c = poplevel (critical population size)
class SsaSimulator {
  public:
    explicit SsaSimulator(NetworkModel &model);
    ~SsaSimulator();

    // Non-copyable
    SsaSimulator(const SsaSimulator &) = delete;
    SsaSimulator &operator=(const SsaSimulator &) = delete;

    // Run exact SSA simulation with deterministic seed.
    // `timeout_seconds <= 0` (default) disables the wall-clock budget; a
    // positive value causes the loop to throw `bngsim::TimeoutError` once
    // elapsed time exceeds the limit.
    Result run(const TimeSpec &times, uint64_t seed = 42, double timeout_seconds = 0.0);

    // Run PSA (Partial Scaling Algorithm) simulation.
    // poplevel = N_c, the critical population size (must be > 1).
    // Lin, Feng, Hlavacek, J. Chem. Phys. 150, 244101 (2019).
    Result run_psa(const TimeSpec &times, uint64_t seed, double poplevel,
                   double timeout_seconds = 0.0);

    // GH #190 — supply the path to a cc-compiled value-specialized propensity
    // .so (symbol bngsim_ssa_propensities), produced by the Python codegen
    // layer. When set and the model is recompute-all eligible (pure mass-action
    // exact SSA, no events, small reaction count), the run takes the RR-style
    // recompute-all + flat-scan loop by default — no MIR required. A no-op for
    // ineligible models (PSA, events, functional/rate-rule rates, large nr),
    // which keep the incremental Fenwick path. Empty string clears it.
    void set_propensity_library(const std::string &so_path);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Internal: shared SSA/PSA simulation loop
    Result run_internal(const TimeSpec &times, uint64_t seed, double poplevel,
                        double timeout_seconds);
};

} // namespace bngsim
