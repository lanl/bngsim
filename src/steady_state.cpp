// bngsim/src/steady_state.cpp -- steady-state solver
//
// Default (newton): two-tier integrate-first solver (GH #27). Tier 1 is a CVODE
//   burst that carries the state into the physical root's basin; tier 2 is a
//   KINSOL Newton polish, accepted only once it is seed-stable (agrees across
//   two successively tighter bursts). Falls back to integration otherwise. See
//   solve_by_newton_two_tier for the rationale (why Newton-first returned wrong
//   / NaN roots).
// integration: CVODE integration with early termination on the BNG2.pl
//   parity criterion ||f(y)||_2 / n_species < tol (run_network -c).
// Steady-state sensitivity: dY_ss/dp = -J^{-1} * df/dp
//
// Convergence criterion: every integrate-to-steady-state path here uses the
// SAME rule as Simulator.run(steady_state=True) / BNG2.pl run_network -c:
// the L2 norm of the derivative vector divided by n_species, ||f||_2 / n.

#include "bngsim/steady_state.hpp"
#include "bngsim/model.hpp"
#include "bngsim/types.hpp"

#include <cvodes/cvodes.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_logger.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>

#include "bngsim/lapack_dense_linsol.hpp"
#include "bngsim/platform_compat.hpp"
#include "bngsim/sundials_guards.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace bngsim {

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct SteadyStateUserData {
    NetworkModel *model;
};

// Compute the BNG2.pl parity steady-state residual ||f(y)||_2 / n_species.
// This is the SAME quantity Simulator.run(steady_state=True) checks at each
// output point (Network3 network.cpp run_network -c). It is the single
// convergence criterion used by every integrate-to-steady-state path and by
// the post-solve verification of the Newton path, so there is one rule.
static double compute_residual(NetworkModel &model, const double *y, int ns) {
    std::vector<double> f(ns, 0.0);
    model.update_observables(y);
    model.evaluate_functions(0.0); // steady state: time irrelevant
    model.compute_derivs(0.0, y, f.data());
    double sumsq = 0.0;
    for (int i = 0; i < ns; ++i) {
        sumsq += f[i] * f[i];
    }
    return (ns > 0) ? std::sqrt(sumsq) / static_cast<double>(ns) : 0.0;
}

// A steady state must be finite and (up to a small scale-relative slack)
// non-negative. Newton can walk a species negative, where Hill/power rate laws
// return NaN — compute_residual then returns NaN, and `NaN >= tol` is false, so
// the old convergence check accepted it (GH #27 Bug 1). This predicate rejects
// any non-finite or clearly-negative concentration. The negativity floor is
// relative to the largest concentration so a root that lands a near-zero species
// at -1e-9 (roundoff around a true zero, e.g. simple decay's A*≈0) still passes.
static bool ss_state_is_physical(const std::vector<double> &y) {
    double maxabs = 0.0;
    for (double v : y) {
        if (!std::isfinite(v))
            return false;
        maxabs = std::max(maxabs, std::abs(v));
    }
    const double neg_floor = -1e-7 * std::max(maxabs, 1e-300);
    for (double v : y) {
        if (v < neg_floor)
            return false;
    }
    return true;
}

// Do two candidate steady states agree to AGREE_RTOL? Used by the two-tier
// solver (GH #27 Bug 2) to accept a KINSOL root only once it is *seed-stable*:
// two Newton solves from successively tighter integration bursts landing on the
// same state. The floor `1e-6*max|b|` keeps near-zero species (relative diff
// explodes as a component → 0) from dominating; this is the scale-robust analog
// of the benchmark's XCHECK metric.
static bool ss_states_agree(const std::vector<double> &a, const std::vector<double> &b,
                            double agree_rtol) {
    if (a.size() != b.size())
        return false;
    double maxabs = 0.0;
    for (double v : a)
        maxabs = std::max(maxabs, std::abs(v));
    for (double v : b)
        maxabs = std::max(maxabs, std::abs(v));
    const double floor = 1e-6 * std::max(maxabs, 1e-300);
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double denom = floor + std::abs(b[i]);
        worst = std::max(worst, std::abs(a[i] - b[i]) / denom);
    }
    return worst < agree_rtol;
}

// Build the dense direct linear solver for an n×n steady-state system,
// applying the GH #84 gate (bngsim/lapack_dense_linsol.hpp). The steady-state
// paths have no KLU option — they always factor densely — so the density floor
// (not force_dense) is the right guard: a structurally-sparse SS Jacobian stays
// on the built-in dense LU, whose zero-skipping beats a full BLAS dgetrf, while
// large AND dense SS systems take the optimized factor. force_dense is false.
static SUNLinearSolver ss_make_dense_linsol(N_Vector v, SUNMatrix A, SUNContext ctx,
                                            NetworkModel &model, int n) {
    const bool use_lapack =
        should_use_lapack_dense(n, model.jacobian_sparsity().density, /*force_dense=*/false);
    return make_dense_linear_solver(v, A, ctx, use_lapack);
}

// ---------------------------------------------------------------------------
// Tier 1: CVODE integration with early termination
// ---------------------------------------------------------------------------

static int cvode_ss_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *ud) {
    auto *data = static_cast<SteadyStateUserData *>(ud);
    const double *yp = N_VGetArrayPointer(y);
    double *yp_dot = N_VGetArrayPointer(ydot);
    data->model->compute_derivs(static_cast<double>(t), yp, yp_dot);
    return 0;
}

static SteadyStateResult solve_by_integration(NetworkModel &model, const SteadyStateOptions &opts) {

    const int ns = model.n_species();
    SteadyStateResult result;
    result.method_used = "integration";
    result.species_names = model.species_names();
    result.concentrations.resize(ns);

    // RAII guards
    SunContextGuard ctx;
    if (!ctx) {
        throw std::runtime_error("SUNContext_Create failed (steady_state)");
    }

    NVectorGuard y(N_VNew_Serial(ns, ctx));
    double *y_data = y.data();
    for (int i = 0; i < ns; ++i) {
        y_data[i] = model.species()[i].concentration;
    }

    CvodeMemGuard cvode_mem(CVodeCreate(CV_BDF, ctx));
    SteadyStateUserData ud{&model};

    int flag = CVodeInit(cvode_mem, cvode_ss_rhs, 0.0, y);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeInit failed (steady_state)");
    }

    CVodeSStolerances(cvode_mem, opts.rtol, opts.atol);
    CVodeSetUserData(cvode_mem, &ud);
    CVodeSetMaxNumSteps(cvode_mem, opts.max_steps);

    SUNMatrixGuard A_guard(SUNDenseMatrix(ns, ns, ctx));
    SUNLinSolGuard LS_guard(ss_make_dense_linsol(y, A_guard, ctx, model, ns));
    CVodeSetLinearSolver(cvode_mem, LS_guard, A_guard);

    // Analytical Jacobian if available and not "fd"
    if (opts.jacobian != "fd") {
        // We reuse the existing dense analytical Jacobian callback from
        // cvode_simulator.cpp. But since we can't call that static function
        // directly, we compute derivs manually and let CVODE do FD.
        // For this implementation, we rely on CVODE's internal FD Jacobian.
        // The analytical Jacobian is used by KINSOL (Tier 2) instead.
    }

    // March forward one internal CVODE step at a time, checking the BNG2.pl
    // parity criterion ||f(y)||_2 / n_species < tol after each step. This is
    // the SAME rule Simulator.run(steady_state=True) applies (run_network -c);
    // the old geometric time-horizon (t = 10, 100, 1000, ...) has been
    // removed so there is one convergence rule everywhere. We cap integration
    // at max_time via CVodeSetStopTime so a non-equilibrating system returns
    // unconverged rather than running forever.
    result.converged = false;
    CVodeSetStopTime(cvode_mem, opts.max_time);

    sunrealtype t_ret = 0.0;
    while (t_ret < opts.max_time) {
        flag = CVode(cvode_mem, opts.max_time, y, &t_ret, CV_ONE_STEP);
        if (flag < 0) {
            // Integration failed -- return unconverged.
            break;
        }

        // compute_residual re-evaluates observables/functions at y internally.
        double resid = compute_residual(model, y_data, ns);
        if (resid < opts.tol) {
            result.converged = true;
            result.residual = resid;
            break;
        }

        if (flag == CV_TSTOP_RETURN) {
            // Reached max_time without converging.
            break;
        }
    }

    // Collect stats
    long int nst = 0, nfe = 0;
    CVodeGetNumSteps(cvode_mem, &nst);
    CVodeGetNumRhsEvals(cvode_mem, &nfe);
    result.n_steps = static_cast<int>(nst);
    result.n_rhs_evals = static_cast<int>(nfe);

    // Copy final concentrations
    for (int i = 0; i < ns; ++i) {
        result.concentrations[i] = y_data[i];
    }

    // If not converged, compute final residual
    if (!result.converged) {
        result.residual = compute_residual(model, y_data, ns);
    }

    // RAII guards handle cleanup automatically

    return result;
}

// ---------------------------------------------------------------------------
// Tier 2: KINSOL Newton solver (with reduced-space for conservation laws)
// ---------------------------------------------------------------------------

// User data for reduced-space KINSOL
struct ReducedKinsolData {
    NetworkModel *model;
    const ConservationLaws *cl;
};

// Reconstruct full y from independent species y_ind using conservation laws
static void reconstruct_full(const double *y_ind, double *y_full, int ns,
                             const ConservationLaws &cl, const std::vector<Species> &species) {

    // First, copy independent species into full vector
    for (size_t k = 0; k < cl.independent.size(); ++k) {
        y_full[cl.independent[k]] = y_ind[k];
    }
    // Then, reconstruct dependent species from conservation constraints
    // Σ L[k,i] * y_full[i] = constants[k]
    // L[k,dep] * y_full[dep] = constants[k] - Σ_{i≠dep} L[k,i] * y_full[i]
    for (int k = 0; k < cl.n_laws; ++k) {
        int dep = cl.dependent[k];
        double coeff_dep = cl.coefficients[k][dep];
        if (std::abs(coeff_dep) < 1e-15)
            continue; // degenerate
        double rhs = cl.constants[k];
        for (int i = 0; i < ns; ++i) {
            if (i != dep) {
                rhs -= cl.coefficients[k][i] * y_full[i];
            }
        }
        y_full[dep] = rhs / coeff_dep;
    }
}

// Reduced-space KINSOL RHS: evaluate f(y) for independent species only
static int kinsol_reduced_rhs(N_Vector y_ind, N_Vector fval, void *ud) {
    auto *data = static_cast<ReducedKinsolData *>(ud);
    NetworkModel *model = data->model;
    const auto &cl = *data->cl;
    const int ns = model->n_species();
    const int n_ind = static_cast<int>(cl.independent.size());

    const double *y_ind_data = N_VGetArrayPointer(y_ind);
    double *f_ind = N_VGetArrayPointer(fval);

    // Reconstruct full state vector
    std::vector<double> y_full(ns, 0.0);
    // Initialize with current concentrations (for fixed species)
    const auto &species = model->species();
    for (int i = 0; i < ns; ++i)
        y_full[i] = species[i].concentration;
    reconstruct_full(y_ind_data, y_full.data(), ns, cl, species);

    // Compute full f(y)
    std::vector<double> f_full(ns, 0.0);
    model->update_observables(y_full.data());
    model->evaluate_functions(0.0);
    model->compute_derivs(0.0, y_full.data(), f_full.data());

    // Extract independent species residuals
    for (int k = 0; k < n_ind; ++k) {
        f_ind[k] = f_full[cl.independent[k]];
    }
    return 0;
}

// Full-space KINSOL RHS (for models without conservation laws)
static int kinsol_rhs(N_Vector y, N_Vector fval, void *ud) {
    auto *data = static_cast<SteadyStateUserData *>(ud);
    const double *yp = N_VGetArrayPointer(y);
    double *fp = N_VGetArrayPointer(fval);
    int ns = data->model->n_species();

    data->model->update_observables(yp);
    data->model->evaluate_functions(0.0);
    data->model->compute_derivs(0.0, yp, fp);

    // Zero out fixed species residuals
    const auto &species = data->model->species();
    for (const auto &s : species) {
        if (s.fixed) {
            fp[s.index - 1] = 0.0;
        }
    }
    return 0;
}

static SteadyStateResult solve_by_newton(NetworkModel &model, const SteadyStateOptions &opts) {

    const int ns = model.n_species();
    const auto &cl = model.conservation_laws();
    SteadyStateResult result;
    result.method_used = "newton";
    result.species_names = model.species_names();
    result.concentrations.resize(ns);

    // ── Recompute conservation constants from CURRENT concentrations ──
    // (important: PSet may have changed ICs since model load time)
    ConservationLaws cl_copy = cl;
    if (!cl_copy.empty()) {
        for (int k = 0; k < cl_copy.n_laws; ++k) {
            double c = 0.0;
            for (int i = 0; i < ns; ++i) {
                c += cl_copy.coefficients[k][i] * model.species()[i].concentration;
            }
            cl_copy.constants[k] = c;
        }
    }

    const bool use_reduced = !cl_copy.empty();
    const int n_ind = use_reduced ? static_cast<int>(cl_copy.independent.size()) : ns;

    // RAII guards
    SunContextGuard ctx;
    if (!ctx) {
        throw std::runtime_error("SUNContext_Create failed (kinsol)");
    }

    // Route this KINSOL context's error/warning log to the null sink. In the
    // two-tier solver a KINSOL failure (non-convergence, or the unrecoverable
    // dense linear-solver setup failure on a structurally singular reduced
    // Jacobian — GH #27 Bug 3, e.g. Barua 2013's 404×404 rank-deficient system)
    // is EXPECTED and handled by falling back to integration, so its stderr
    // spam is noise. Hard misuse still surfaces via the flag checks below.
    {
        SUNLogger logger = nullptr;
        if (SUNContext_GetLogger(ctx, &logger) == SUN_SUCCESS && logger != nullptr) {
            SUNLogger_SetErrorFilename(logger, bngsim::null_device);
            SUNLogger_SetWarningFilename(logger, bngsim::null_device);
        }
    }

    NVectorGuard y(N_VNew_Serial(n_ind, ctx));
    double *y_data = y.data();

    if (use_reduced) {
        // Extract independent species concentrations
        for (int k = 0; k < n_ind; ++k) {
            y_data[k] = model.species()[cl_copy.independent[k]].concentration;
        }
    } else {
        for (int i = 0; i < ns; ++i) {
            y_data[i] = model.species()[i].concentration;
        }
    }

    NVectorGuard scale(N_VNew_Serial(n_ind, ctx));
    N_VConst(1.0, scale);

    KinsolMemGuard kin_mem(KINCreate(ctx));
    if (!kin_mem) {
        throw std::runtime_error("KINCreate failed");
    }

    int flag;
    ReducedKinsolData rd{&model, &cl_copy};
    SteadyStateUserData ud{&model};

    if (use_reduced) {
        flag = KINInit(kin_mem, kinsol_reduced_rhs, y);
        KINSetUserData(kin_mem, &rd);
    } else {
        flag = KINInit(kin_mem, kinsol_rhs, y);
        KINSetUserData(kin_mem, &ud);
    }

    if (flag != KIN_SUCCESS) {
        throw std::runtime_error("KINInit failed");
    }

    KINSetFuncNormTol(kin_mem, opts.tol);
    KINSetScaledStepTol(kin_mem, 1e-15); // allow very small steps
    KINSetNumMaxIters(kin_mem, 200);
    // Set max Newton step large enough for the problem scale
    double max_newton_step = 0.0;
    for (int i = 0; i < n_ind; ++i) {
        double v = std::abs(y_data[i]);
        if (v > max_newton_step)
            max_newton_step = v;
    }
    max_newton_step = std::max(max_newton_step * 100.0, 1e6);
    KINSetMaxNewtonStep(kin_mem, max_newton_step);

    SUNMatrixGuard A_guard(SUNDenseMatrix(n_ind, n_ind, ctx));
    SUNLinSolGuard LS_guard(ss_make_dense_linsol(y, A_guard, ctx, model, n_ind));
    KINSetLinearSolver(kin_mem, LS_guard, A_guard);

    // Solve — use KIN_NONE (pure Newton) for reduced systems since the
    // reduced Jacobian is non-singular by construction. For full-space
    // systems, also use KIN_NONE (the auto fallback to integration
    // handles convergence failure gracefully).
    flag = KINSol(kin_mem, y, KIN_NONE, scale, scale);

    // Check result
    long int nfe = 0, nni = 0;
    KINGetNumFuncEvals(kin_mem, &nfe);
    KINGetNumNonlinSolvIters(kin_mem, &nni);
    result.n_steps = static_cast<int>(nni);
    result.n_rhs_evals = static_cast<int>(nfe);

    if (flag >= 0) {
        result.converged = true;
    }

    // Reconstruct full concentrations
    if (use_reduced) {
        std::vector<double> y_full(ns, 0.0);
        for (int i = 0; i < ns; ++i)
            y_full[i] = model.species()[i].concentration;
        reconstruct_full(y_data, y_full.data(), ns, cl_copy, model.species());
        for (int i = 0; i < ns; ++i)
            result.concentrations[i] = y_full[i];
    } else {
        for (int i = 0; i < ns; ++i)
            result.concentrations[i] = y_data[i];
    }

    // Compute actual residual using full state and verify convergence.
    //
    // GH #27 Bug 1: the old guard was `if (residual >= tol) converged = false`.
    // When Newton walks a species negative, Hill/power rate laws yield NaN, so
    // compute_residual returns NaN — and `NaN >= tol` is false, so a NaN result
    // was reported converged (returning conc=[nan, …]). Use the positive test
    // `!(residual < tol)` (true for NaN) and additionally reject any non-finite
    // or clearly-negative concentration, so an unphysical Newton root never
    // passes as a converged steady state.
    result.residual = compute_residual(model, result.concentrations.data(), ns);
    if (!(result.residual < opts.tol) || !ss_state_is_physical(result.concentrations)) {
        result.converged = false;
    }

    // RAII guards handle cleanup automatically

    return result;
}

// ---------------------------------------------------------------------------
// Two-tier steady-state solver (GH #27): integrate FIRST, then Newton
// ---------------------------------------------------------------------------
//
// The manuscript's two-tier method is tier 1 = CVODE integration with early
// termination, tier 2 = KINSOL Newton polish. The previous default ran them in
// the OPPOSITE order — Newton seeded at the raw initial condition, integration
// only as a non-convergence fallback — so on any model whose f(y)=0 has several
// roots (e.g. kinetic proofreading) Newton would *converge* to a spurious root
// the dynamics never reach and the fallback never fired (GH #27 Bug 2).
//
// Here integration carries the state into the physical root's basin and Newton
// polishes from there. The open question the issue flags is how tight the burst
// must be — it is model-dependent. We make it ADAPTIVE without ever trusting a
// single unvalidated Newton solve: a KINSOL root is accepted only when it is
// *seed-stable*, i.e. two Newton solves from successively tighter integration
// bursts land on the same state (ss_states_agree). A unique-root model confirms
// on the first pair of bursts (fast — the speedup the issue wants to see); a
// multi-root model's Newton roots keep drifting with the seed until the burst
// has essentially converged, so it simply integrates to the physical root. Any
// non-finite / non-converged / structurally-singular (Bug 3) Newton attempt is
// discarded and integration continues. The result is correct on every model,
// with the root-finding speedup surfacing exactly where it is trustworthy.
//
// Burst tolerances form a scale-free ladder: the residual ||f||_2/n at the IC,
// reduced by a decade for the first burst and by two decades per rung after,
// floored at opts.tol. Each rung CONTINUES from the previous burst's end state
// (integration conserves mass, so the reduced Newton's conservation constants,
// recomputed from that state, are unchanged) — so the total integration work is
// a single march to the tightest rung, not a restart per rung.

// Newton polish seeded from `seed`; returns the KINSOL result. The model's live
// species are set to `seed` first (solve_by_integration / solve_by_newton read
// the IC from there). KINSOL construction can throw (KINInit/KINCreate) — the
// caller treats a throw as a failed, non-accepted attempt.
static SteadyStateResult ss_newton_from(NetworkModel &model, const SteadyStateOptions &opts,
                                        const std::vector<double> &seed) {
    model.set_state_from(seed.data());
    return solve_by_newton(model, opts);
}

static SteadyStateResult solve_by_newton_two_tier(NetworkModel &model,
                                                  const SteadyStateOptions &opts) {
    const int ns = model.n_species();

    // Snapshot the initial condition so the model is restored on return (the
    // public contract: steady_state() without sensitivity leaves ICs intact).
    std::vector<double> ic(ns);
    model.get_state_into(ic.data());

    auto restore = [&]() { model.set_state_from(ic.data()); };

    // Accept a KINSOL result only if it converged to a finite, physical root.
    auto accept = [&](const SteadyStateResult &r) {
        return r.converged && r.method_used == "newton" && (r.residual < opts.tol) &&
               ss_state_is_physical(r.concentrations);
    };

    const double r0 = compute_residual(model, ic.data(), ns);

    // IC already at steady state: a single Newton polish (which converges
    // immediately) reports the canonical "newton" without any integration.
    if (r0 < opts.tol) {
        SteadyStateResult r;
        try {
            r = ss_newton_from(model, opts, ic);
        } catch (...) {
            r.converged = false;
        }
        if (accept(r)) {
            restore();
            return r;
        }
    }

    constexpr double AGREE_RTOL = 1e-4; // seed-stability tolerance (see header)
    constexpr int MAX_RUNGS = 14;
    // Cap on burst-seeded Newton attempts. Two agreeing attempts accept a
    // unique-root model on rung 1; a multi-root model's attempts keep drifting,
    // so after this many we stop probing and just integrate (KINSOL adds only
    // cost there, e.g. Barua 2013 whose reduced Jacobian is singular at every
    // seed — Bug 3). Correctness is unaffected: integration is always the answer.
    constexpr int MAX_NEWTON_ATTEMPTS = 6;

    std::vector<double> prev_newton;
    bool have_prev = false;
    int newton_attempts = 0;

    std::vector<double> seed = ic;
    double bt = std::max(r0 * 0.1, opts.tol);

    for (int rung = 0; rung < MAX_RUNGS; ++rung) {
        // Tier 1: continue integrating from the previous rung's end state to bt.
        SteadyStateOptions bopts = opts;
        bopts.tol = bt;
        model.set_state_from(seed.data());
        SteadyStateResult burst = solve_by_integration(model, bopts);
        seed = burst.concentrations;

        if (burst.residual < opts.tol) {
            // Integration itself reached the parity tolerance — done.
            restore();
            return burst;
        }
        if (!burst.converged) {
            // Could not reach even this (looser) burst tolerance within max_time
            // (a slow/oscillatory system, or a residual floor above tol like
            // Barua 2013). Integration is the best available answer.
            restore();
            return burst;
        }

        // Tier 2: Newton polish from the burst state. Accept only when a second
        // attempt lands on the same root (seed-stable); the first attempt just
        // seeds the comparison. A raw-IC Newton is never trusted alone — that is
        // exactly the Bug 2 hazard.
        if (newton_attempts < MAX_NEWTON_ATTEMPTS) {
            ++newton_attempts;
            SteadyStateResult nr;
            try {
                nr = ss_newton_from(model, opts, seed);
            } catch (...) {
                nr.converged = false;
            }
            if (accept(nr)) {
                if (have_prev && ss_states_agree(nr.concentrations, prev_newton, AGREE_RTOL)) {
                    restore();
                    return nr; // seed-stable root — accept
                }
                prev_newton = nr.concentrations;
                have_prev = true;
            }
        }

        if (bt <= opts.tol)
            break;
        bt = std::max(bt * 0.01, opts.tol);
    }

    // Ladder exhausted without a seed-stable Newton: integrate to the full
    // tolerance from the tightest burst state (a correct, if slower, answer).
    model.set_state_from(seed.data());
    SteadyStateResult fin = solve_by_integration(model, opts);
    restore();
    return fin;
}

// ---------------------------------------------------------------------------
// Steady-state sensitivity: dY_ss/dp = -J^{-1} * df/dp
// ---------------------------------------------------------------------------

static void compute_ss_sensitivity(NetworkModel &model, SteadyStateResult &result,
                                   const std::vector<std::string> &param_names) {

    const int ns = model.n_species();
    const int np = static_cast<int>(param_names.size());
    if (ns == 0 || np == 0)
        return;

    const double *y_ss = result.concentrations.data();
    const auto &params = model.parameters();

    // Map param names to indices
    std::vector<int> pidx(np);
    for (int p = 0; p < np; ++p) {
        bool found = false;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i].name == param_names[p]) {
                pidx[p] = static_cast<int>(i);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Steady-state sensitivity: parameter '" + param_names[p] +
                                     "' not found");
        }
    }

    // Step 1: Compute dense Jacobian J at y_ss
    // We use finite differences (works for all rate law types)
    std::vector<double> J(ns * ns, 0.0); // column-major for LAPACK
    std::vector<double> f0(ns), f1(ns), y_pert(ns);
    const double eps = 1.4901161193847656e-8; // sqrt(machine eps)

    // Evaluate f(y_ss)
    model.update_observables(y_ss);
    model.evaluate_functions(0.0);
    model.compute_derivs(0.0, y_ss, f0.data());

    // J[:,j] = (f(y+h*e_j) - f(y)) / h
    for (int j = 0; j < ns; ++j) {
        std::memcpy(y_pert.data(), y_ss, ns * sizeof(double));
        double h = eps * std::max(std::abs(y_ss[j]), 1.0);
        y_pert[j] += h;
        model.update_observables(y_pert.data());
        model.evaluate_functions(0.0);
        model.compute_derivs(0.0, y_pert.data(), f1.data());
        for (int i = 0; i < ns; ++i) {
            J[j * ns + i] = (f1[i] - f0[i]) / h; // column-major
        }
    }

    // Step 2: Compute df/dp for each sensitivity parameter
    // df/dp[:,p] = (f(y_ss; p+h) - f(y_ss; p)) / h
    std::vector<double> dfdp(ns * np, 0.0); // column-major: dfdp[p*ns+i]

    // Restore observables at y_ss with original params
    model.update_observables(y_ss);
    model.evaluate_functions(0.0);
    model.compute_derivs(0.0, y_ss, f0.data());

    for (int p = 0; p < np; ++p) {
        int pi = pidx[p];
        double pval = params[pi].value;
        double h = eps * std::max(std::abs(pval), 1.0);

        // Perturb parameter
        const_cast<std::vector<Parameter> &>(params)[pi].value = pval + h;
        model.update_observables(y_ss);
        model.evaluate_functions(0.0);
        model.compute_derivs(0.0, y_ss, f1.data());

        // Restore parameter
        const_cast<std::vector<Parameter> &>(params)[pi].value = pval;

        for (int i = 0; i < ns; ++i) {
            dfdp[p * ns + i] = (f1[i] - f0[i]) / h;
        }
    }

    // Step 3: Solve J * sens[:,p] = -dfdp[:,p] for each parameter.
    //
    // If conservation laws are present, J is singular (rank-deficient).
    // Use reduced-space solve: extract independent rows/cols from J and df/dp,
    // solve the reduced system, then reconstruct dependent species from
    // conservation constraints (their sensitivity follows from the chain rule).

    const auto &cl = model.conservation_laws();

    result.sensitivity.resize(ns * np, 0.0);
    result.sens_param_names = param_names;
    result.n_sens_params = np;

    if (!cl.empty()) {
        // Reduced-space sensitivity solve
        const int n_ind = static_cast<int>(cl.independent.size());

        // Build reduced Jacobian J_red (n_ind × n_ind) from independent species
        // J_red[i][j] = dfi/dyj where i,j ∈ independent, but including
        // the chain rule through dependent species:
        // dfi/dyj_ind = J[ind_i][ind_j] + Σ_k J[ind_i][dep_k] * (ddep_k/dyj_ind)
        // where ddep_k/dyj_ind comes from differentiating the conservation law.

        // For simplicity and robustness, use FD on the reduced residual directly
        std::vector<double> f_ind0(n_ind), f_ind1(n_ind), y_ind(n_ind);
        std::vector<double> y_full(ns), f_full(ns);

        // Get current y_full = y_ss
        std::memcpy(y_full.data(), y_ss, ns * sizeof(double));
        for (int k = 0; k < n_ind; ++k)
            y_ind[k] = y_ss[cl.independent[k]];

        // Evaluate f_ind at y_ss
        model.update_observables(y_ss);
        model.evaluate_functions(0.0);
        model.compute_derivs(0.0, y_ss, f_full.data());
        for (int k = 0; k < n_ind; ++k)
            f_ind0[k] = f_full[cl.independent[k]];

        // Build reduced J_red via FD
        std::vector<double> J_red(n_ind * n_ind, 0.0);
        for (int j = 0; j < n_ind; ++j) {
            std::vector<double> y_pert_full(y_full);
            double h = eps * std::max(std::abs(y_ind[j]), 1.0);
            y_pert_full[cl.independent[j]] += h;

            // Recompute dependent species from perturbed independents
            // (This accounts for the chain rule through conservation laws)
            ConservationLaws cl_tmp = cl;
            for (int kk = 0; kk < cl_tmp.n_laws; ++kk) {
                double c = 0.0;
                for (int ii = 0; ii < ns; ++ii)
                    c += cl_tmp.coefficients[kk][ii] * y_ss[ii];
                cl_tmp.constants[kk] = c;
            }
            // Reconstruct dependent species
            for (int kk = 0; kk < cl_tmp.n_laws; ++kk) {
                int dep = cl_tmp.dependent[kk];
                double coeff_dep = cl_tmp.coefficients[kk][dep];
                if (std::abs(coeff_dep) < 1e-15)
                    continue;
                double rhs = cl_tmp.constants[kk];
                for (int ii = 0; ii < ns; ++ii)
                    if (ii != dep)
                        rhs -= cl_tmp.coefficients[kk][ii] * y_pert_full[ii];
                y_pert_full[dep] = rhs / coeff_dep;
            }

            model.update_observables(y_pert_full.data());
            model.evaluate_functions(0.0);
            model.compute_derivs(0.0, y_pert_full.data(), f_full.data());
            for (int i = 0; i < n_ind; ++i)
                J_red[j * n_ind + i] = (f_full[cl.independent[i]] - f_ind0[i]) / h;
        }

        // Build reduced df/dp
        std::vector<double> dfdp_red(n_ind * np, 0.0);
        for (int p = 0; p < np; ++p)
            for (int i = 0; i < n_ind; ++i)
                dfdp_red[p * n_ind + i] = dfdp[p * ns + cl.independent[i]];

        // Solve J_red * sens_ind = -dfdp_red using SUNDIALS with RAII guards
        SunContextGuard ctx;
        SUNMatrixGuard A_guard(SUNDenseMatrix(n_ind, n_ind, ctx));
        NVectorGuard bv(N_VNew_Serial(n_ind, ctx));
        NVectorGuard xv(N_VNew_Serial(n_ind, ctx));
        SUNLinSolGuard LS_guard(ss_make_dense_linsol(xv, A_guard, ctx, model, n_ind));

        for (int p = 0; p < np; ++p) {
            sunrealtype *A_data = SUNDenseMatrix_Data(A_guard);
            std::memcpy(A_data, J_red.data(), n_ind * n_ind * sizeof(double));
            SUNLinSolSetup(LS_guard, A_guard);
            double *b_data = N_VGetArrayPointer(bv);
            for (int i = 0; i < n_ind; ++i)
                b_data[i] = -dfdp_red[p * n_ind + i];
            SUNLinSolSolve(LS_guard, A_guard, xv, bv, 0.0);
            const double *x_data = N_VGetArrayPointer(xv);

            // Fill independent species sensitivity
            for (int i = 0; i < n_ind; ++i)
                result.sensitivity[cl.independent[i] * np + p] = x_data[i];

            // Reconstruct dependent species sensitivity from conservation:
            // Σ L[k,i] * dy_i/dp = 0 → dy_dep/dp = -(1/L[k,dep]) * Σ_{i≠dep} L[k,i] * dy_i/dp
            for (int k = 0; k < cl.n_laws; ++k) {
                int dep = cl.dependent[k];
                double cd = cl.coefficients[k][dep];
                if (std::abs(cd) < 1e-15)
                    continue;
                double s = 0.0;
                for (int i = 0; i < ns; ++i)
                    if (i != dep)
                        s += cl.coefficients[k][i] * result.sensitivity[i * np + p];
                result.sensitivity[dep * np + p] = -s / cd;
            }
        }

        // RAII guards handle cleanup
    } else {
        // Full-space solve (no conservation laws, still using RAII guards)
        SunContextGuard ctx;
        SUNMatrixGuard A_guard(SUNDenseMatrix(ns, ns, ctx));
        sunrealtype *A_data = SUNDenseMatrix_Data(A_guard);
        std::memcpy(A_data, J.data(), ns * ns * sizeof(double));

        NVectorGuard b(N_VNew_Serial(ns, ctx));
        NVectorGuard x(N_VNew_Serial(ns, ctx));
        SUNLinSolGuard LS_guard(ss_make_dense_linsol(x, A_guard, ctx, model, ns));
        SUNLinSolSetup(LS_guard, A_guard);

        for (int p = 0; p < np; ++p) {
            double *b_data = N_VGetArrayPointer(b);
            for (int i = 0; i < ns; ++i)
                b_data[i] = -dfdp[p * ns + i];
            std::memcpy(A_data, J.data(), ns * ns * sizeof(double));
            SUNLinSolSetup(LS_guard, A_guard);
            SUNLinSolSolve(LS_guard, A_guard, x, b, 0.0);
            const double *x_data = N_VGetArrayPointer(x);
            for (int i = 0; i < ns; ++i)
                result.sensitivity[i * np + p] = x_data[i];
        }

        // RAII guards handle cleanup
    }
}

// ---------------------------------------------------------------------------
// Steady-state OUTPUT sensitivities (GH #12)
// ---------------------------------------------------------------------------
//
// Project the species sensitivity dY_ss/dp (from compute_ss_sensitivity) onto
// the model's observables and global functions, so a gradient consumer can read
// d(observable)/dp and d(function)/dp directly instead of re-deriving the output
// Jacobian:
//
//   d(obs_j)/dp  = Σ_i (∂obs_j/∂x_i)·dY_ss_i/dp                (exact; linear groups)
//   d(func_m)/dp = Σ_i (∂func_m/∂x_i)·dY_ss_i/dp + ∂func_m/∂p  (finite differences)
//
// Observables are Σ factor·x, so ∂obs/∂x is exactly the group factor and the
// observable projection is exact. The function projection reuses the same
// finite-difference primitive as compute_ss_sensitivity: the state-chain Jacobian
// ∂func/∂x from per-species perturbations, plus the function's explicit parameter
// dependence ∂func/∂p (e.g. `k3/(K4+G)` w.r.t. k3) from per-parameter
// perturbations at the fixed steady state. BOTH terms are needed for the total
// derivative and match the CVODES codegen output-sensitivity chain rule.
//
// Precondition: result.sensitivity is populated and the model species are set to
// the steady state. The model is left evaluated at the steady state with the
// original parameter values on return.
static void compute_ss_output_sensitivity(NetworkModel &model, SteadyStateResult &result,
                                          const std::vector<std::string> &param_names) {
    const int ns = model.n_species();
    const int np = static_cast<int>(param_names.size());
    const int n_obs = model.n_observables();
    const int n_func = model.n_functions();
    if (np == 0 || result.sensitivity.empty()) {
        return;
    }

    result.observable_names = model.observable_names();
    result.function_names = model.function_names();

    const double *y_ss = result.concentrations.data();

    // ── Observables: exact linear projection through the group factors ────────
    // obs_j = Σ_{(i,f) ∈ group_j} f·x_i  ⇒  d(obs_j)/dp = Σ f·dY_ss_i/dp.
    if (n_obs > 0) {
        result.observable_sensitivity.assign(static_cast<size_t>(n_obs) * np, 0.0);
        const auto &observables = model.observables();
        for (int j = 0; j < n_obs; ++j) {
            double *out = result.observable_sensitivity.data() + static_cast<size_t>(j) * np;
            for (const auto &entry : observables[j].entries) {
                const int i = entry.species_index - 1; // group entries are 1-based
                if (i < 0 || i >= ns) {
                    continue;
                }
                const double *dxi = result.sensitivity.data() + static_cast<size_t>(i) * np;
                for (int p = 0; p < np; ++p) {
                    out[p] += entry.factor * dxi[p];
                }
            }
        }
    }

    // ── Functions: finite-difference total derivative ─────────────────────────
    if (n_func > 0) {
        result.function_sensitivity.assign(static_cast<size_t>(n_func) * np, 0.0);
        const double eps = 1.4901161193847656e-8; // sqrt(machine eps)

        // Base function values at the steady state with the original parameters.
        // function_value_cache() returns a reference reused by every subsequent
        // evaluate_functions() call, so snapshot it into f0.
        model.update_observables(y_ss);
        model.evaluate_functions(0.0);
        const std::vector<double> f0(model.function_value_cache());
        std::vector<double> f1;
        std::vector<double> y_pert(ns);

        // State-chain term: ∂func_m/∂x_i via one-sided FD (perturb one species,
        // re-evaluate observables + functions), folded into
        // Σ_i (∂func_m/∂x_i)·dY_ss_i/dp as each species column is produced.
        for (int i = 0; i < ns; ++i) {
            std::memcpy(y_pert.data(), y_ss, ns * sizeof(double));
            const double h = eps * std::max(std::abs(y_ss[i]), 1.0);
            y_pert[i] += h;
            model.update_observables(y_pert.data());
            model.evaluate_functions(0.0);
            f1 = model.function_value_cache();
            const double *dxi = result.sensitivity.data() + static_cast<size_t>(i) * np;
            for (int m = 0; m < n_func; ++m) {
                const double dfm_dxi = (f1[m] - f0[m]) / h;
                double *out = result.function_sensitivity.data() + static_cast<size_t>(m) * np;
                for (int p = 0; p < np; ++p) {
                    out[p] += dfm_dxi * dxi[p];
                }
            }
        }

        // Explicit-parameter term: ∂func_m/∂p at the fixed steady state (perturb
        // one parameter, keep the state fixed). Observables are functions of
        // species only, so update_observables(y_ss) restores the same totals; the
        // function evaluator picks up the live parameter value.
        const auto &params = model.parameters();
        for (int p = 0; p < np; ++p) {
            // param_names[p] was validated to exist by compute_ss_sensitivity.
            int pi = -1;
            for (size_t k = 0; k < params.size(); ++k) {
                if (params[k].name == param_names[p]) {
                    pi = static_cast<int>(k);
                    break;
                }
            }
            if (pi < 0) {
                continue;
            }
            const double pval = params[pi].value;
            const double h = eps * std::max(std::abs(pval), 1.0);
            const_cast<std::vector<Parameter> &>(params)[pi].value = pval + h;
            model.update_observables(y_ss);
            model.evaluate_functions(0.0);
            f1 = model.function_value_cache();
            const_cast<std::vector<Parameter> &>(params)[pi].value = pval;
            for (int m = 0; m < n_func; ++m) {
                result.function_sensitivity[static_cast<size_t>(m) * np + p] += (f1[m] - f0[m]) / h;
            }
        }

        // Leave the model evaluated at the steady state with original parameters.
        model.update_observables(y_ss);
        model.evaluate_functions(0.0);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

SteadyStateResult find_steady_state(NetworkModel &model, const SteadyStateOptions &opts) {

    const int ns = model.n_species();
    if (ns == 0) {
        throw std::runtime_error("Cannot find steady state: model has no species");
    }

    // Normalize and validate method. "kinsol" is an input alias for "newton";
    // "auto" was removed (newton already means try-Newton-then-parity-fallback).
    std::string method = opts.method;
    if (method == "kinsol") {
        method = "newton";
    }
    if (method != "integration" && method != "newton") {
        throw std::runtime_error("Invalid steady-state method '" + opts.method +
                                 "'. "
                                 "Must be \"newton\", \"integration\", or \"kinsol\" "
                                 "(alias for \"newton\").");
    }

    SteadyStateResult result;

    if (method == "integration") {
        result = solve_by_integration(model, opts);
    } else {
        // "newton" (default): two-tier integrate-first solver (GH #27). A short
        // CVODE burst carries the state into the physical root's basin, then
        // KINSOL polishes; the polish is accepted only once it is seed-stable
        // (agrees across two successively tighter bursts), otherwise integration
        // continues. This is correct on multi-root and NaN-prone models where
        // the old Newton-first ordering returned spurious / non-finite roots,
        // while still surfacing the root-finding speedup on unique-root models.
        result = solve_by_newton_two_tier(model, opts);
    }

    // Compute sensitivity if requested and converged
    if (result.converged && !opts.sensitivity_params.empty()) {
        // Update model state to steady-state values for sensitivity
        auto &species = const_cast<std::vector<Species> &>(model.species());
        for (int i = 0; i < ns; ++i) {
            species[i].concentration = result.concentrations[i];
        }
        compute_ss_sensitivity(model, result, opts.sensitivity_params);
        // GH #12 — project dY_ss/dp onto observables/functions for direct
        // d(output)/dp access (mirrors Result.output_sensitivities).
        compute_ss_output_sensitivity(model, result, opts.sensitivity_params);
    }

    return result;
}

} // namespace bngsim
