// bngsim/tests/test_bngsim.cpp — Unit tests for bngsim Phase A
//
// Simple test driver (no external test framework for Phase A).
// Each test function returns 0 on success, 1 on failure.

#include <bngsim/bngsim.hpp>
#include <bngsim/bngsim_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// Test data directory (set by CMake)
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "data"
#endif

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "  FAIL: " << msg << " [" << __FILE__ << ":"       \
                      << __LINE__ << "]" << std::endl;                      \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define CHECK_CLOSE(a, b, tol, msg)                                         \
    do {                                                                    \
        double _a = (a), _b = (b), _tol = (tol);                           \
        if (std::abs(_a - _b) > _tol) {                                     \
            std::cerr << "  FAIL: " << msg << " — expected " << _b          \
                      << ", got " << _a << " (tol=" << _tol << ")"          \
                      << " [" << __FILE__ << ":" << __LINE__ << "]"         \
                      << std::endl;                                         \
            return 1;                                                       \
        }                                                                   \
    } while (0)

#define RUN_TEST(func)                                                      \
    do {                                                                    \
        ++tests_run;                                                        \
        std::cout << "  " << #func << "... " << std::flush;                 \
        int _rc = func();                                                   \
        if (_rc == 0) {                                                     \
            ++tests_passed;                                                 \
            std::cout << "OK" << std::endl;                                 \
        } else {                                                            \
            std::cout << "FAILED" << std::endl;                             \
        }                                                                   \
    } while (0)

static std::string data_path(const std::string& filename) {
    return std::string(TEST_DATA_DIR) + "/" + filename;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: Load a .net file and check model dimensions
// ═══════════════════════════════════════════════════════════════════════════════

int test_load_simple_decay() {
    auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));

    CHECK(model.n_species() == 2, "Expected 2 species");
    CHECK(model.n_reactions() == 1, "Expected 1 reaction");
    CHECK(model.n_observables() == 2, "Expected 2 observables");
    CHECK(model.n_parameters() == 1, "Expected 1 parameter");

    CHECK_CLOSE(model.get_param("k1"), 0.1, 1e-15, "k1 value");
    CHECK(model.species()[0].name == "A()", "Species 0 name");
    CHECK(model.species()[1].name == "B()", "Species 1 name");
    CHECK_CLOSE(model.species()[0].concentration, 100.0, 1e-15, "A(0)");
    CHECK_CLOSE(model.species()[1].concentration, 0.0, 1e-15, "B(0)");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: ODE simulation of first-order decay (analytical solution)
// ═══════════════════════════════════════════════════════════════════════════════

int test_ode_simple_decay() {
    auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
    bngsim::CvodeSimulator sim(model);

    bngsim::TimeSpec times;
    times.t_start = 0.0;
    times.t_end = 50.0;
    times.n_points = 51;

    auto result = sim.run(times);

    CHECK(result.n_times() == 51, "Expected 51 time points");
    CHECK(result.n_species() == 2, "Expected 2 species in result");
    CHECK(result.n_observables() == 2, "Expected 2 observables in result");

    // Check against analytical solution: A(t) = 100*exp(-0.1*t)
    const auto& time = result.time();
    const auto& species = result.species_data();

    for (int i = 0; i < result.n_times(); ++i) {
        double t = time[i];
        double A_exact = 100.0 * std::exp(-0.1 * t);
        double B_exact = 100.0 - A_exact;
        double A_sim = species[i * 2 + 0];
        double B_sim = species[i * 2 + 1];

        CHECK_CLOSE(A_sim, A_exact, 1e-4,
                    "A(" + std::to_string(t) + ")");
        CHECK_CLOSE(B_sim, B_exact, 1e-4,
                    "B(" + std::to_string(t) + ")");
    }

    // Check solver stats are populated
    CHECK(result.solver_stats().n_steps > 0, "Solver took some steps");
    CHECK(result.solver_stats().n_rhs_evals > 0, "Some RHS evals");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: Parameter update and re-simulation
// ═══════════════════════════════════════════════════════════════════════════════

int test_param_update() {
    auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));

    // Change k1 from 0.1 to 1.0
    model.set_param("k1", 1.0);
    CHECK_CLOSE(model.get_param("k1"), 1.0, 1e-15, "k1 after set");

    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 5.0, 51};
    auto result = sim.run(times);

    // A(5) = 100*exp(-1.0*5) ≈ 0.674
    double A_end = result.species_data()[(result.n_times() - 1) * 2 + 0];
    CHECK_CLOSE(A_end, 100.0 * std::exp(-5.0), 1e-3, "A(5) with k1=1.0");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: Model reset restores initial conditions
// ═══════════════════════════════════════════════════════════════════════════════

int test_reset() {
    auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));

    // Simulate to change species concentrations
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 10.0, 11};
    sim.run(times);

    // Reset and check initial conditions restored
    model.reset();
    CHECK_CLOSE(model.species()[0].concentration, 100.0, 1e-15, "A after reset");
    CHECK_CLOSE(model.species()[1].concentration, 0.0, 1e-15, "B after reset");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: Reversible binding reaches equilibrium
// ═══════════════════════════════════════════════════════════════════════════════

int test_ode_reversible() {
    auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
    bngsim::CvodeSimulator sim(model);

    bngsim::TimeSpec times{0.0, 1000.0, 101};
    auto result = sim.run(times);

    // At equilibrium: kf*A*B = kr*C, and A + C = 100, B + C = 50
    // So C_eq satisfies: kf*(100-C)*(50-C) = kr*C
    // 0.001*(100-C)*(50-C) = 0.1*C
    // This is a quadratic — at equilibrium C should be around 3.1
    int last = result.n_times() - 1;
    double C_eq = result.species_data()[last * 3 + 2];

    CHECK(C_eq > 2.0, "C_eq should be > 2");
    CHECK(C_eq < 50.0, "C_eq should be < 50");

    // Conservation: A + C = 100
    double A_eq = result.species_data()[last * 3 + 0];
    CHECK_CLOSE(A_eq + C_eq, 100.0, 1e-3, "Conservation A + C = 100");

    // Conservation: B + C = 50
    double B_eq = result.species_data()[last * 3 + 1];
    CHECK_CLOSE(B_eq + C_eq, 50.0, 1e-3, "Conservation B + C = 50");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: steady_state early-stop truncates the Result (issue #47)
// ═══════════════════════════════════════════════════════════════════════════════

int test_ode_steady_state_early_stop() {
    // Baseline: full run, no early stop.
    auto full_model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
    bngsim::CvodeSimulator full_sim(full_model);
    bngsim::TimeSpec times{0.0, 200.0, 21};
    auto full = full_sim.run(times);
    CHECK(full.n_times() == 21, "Full run keeps all 21 rows");
    CHECK(!full.solver_stats().steady_state_reached, "Full run did not flag steady state");

    // Early-stop run with a loose tolerance: ||f||/n crosses 1e-4 at t=120,
    // so the Result is truncated to the first 13 rows (t = 0..120).
    auto ss_model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
    bngsim::CvodeSimulator ss_sim(ss_model);
    bngsim::SolverOptions opts;
    opts.steady_state = true;
    opts.steady_state_tol = 1e-4;
    auto ss = ss_sim.run(times, opts);

    CHECK(ss.solver_stats().steady_state_reached, "Early-stop flagged steady state");
    CHECK(ss.n_times() < 21, "Early-stop truncated the Result");
    CHECK_CLOSE(ss.time().back(), 120.0, 1e-9, "Early-stop final time is t=120");
    CHECK(static_cast<int>(ss.time().size()) == ss.n_times(), "time() length matches n_times");
    CHECK(static_cast<int>(ss.species_data().size()) == ss.n_times() * ss.n_species(),
          "species_data length matches truncated n_times");

    // Truncated rows must equal the same rows of the full run (early-stop only
    // drops trailing rows; it does not perturb the kept ones).
    for (int i = 0; i < ss.n_times(); ++i) {
        CHECK_CLOSE(ss.time()[i], full.time()[i], 1e-12, "time row matches full run");
        for (int s = 0; s < ss.n_species(); ++s) {
            CHECK_CLOSE(ss.species_data()[i * ss.n_species() + s],
                        full.species_data()[i * full.n_species() + s], 1e-12,
                        "species row matches full run");
        }
    }

    // Tolerance too tight to reach within the window → no truncation.
    auto no_ss_model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
    bngsim::CvodeSimulator no_ss_sim(no_ss_model);
    bngsim::SolverOptions tight;
    tight.steady_state = true;
    tight.steady_state_tol = 1e-14;
    auto no_ss = no_ss_sim.run(times, tight);
    CHECK(!no_ss.solver_stats().steady_state_reached, "Tight tol does not reach steady state");
    CHECK(no_ss.n_times() == 21, "Tight tol keeps all rows");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: find_steady_state method routing — default newton, kinsol alias,
// integration parity criterion, invalid method (ss_method plan)
// ═══════════════════════════════════════════════════════════════════════════════

int test_find_steady_state_methods() {
    // Default method is "newton" (two-tier integrate-first: CVODE burst into
    // the physical basin, then a seed-stable KINSOL polish; GH #27). On a
    // reversible A<->B model the polish is accepted and reports "newton".
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        bngsim::SteadyStateOptions opts; // method defaults to "newton"
        CHECK(opts.method == "newton", "Default steady-state method is newton");
        auto ss = bngsim::find_steady_state(model, opts);
        CHECK(ss.converged, "Default newton converges on reversible model");
        CHECK(ss.method_used == "newton", "Default reports method_used=newton");
        CHECK(ss.residual < opts.tol, "Newton residual below tol (||f||_2/n)");
    }

    // "kinsol" is an input alias for "newton"; the canonical name is echoed.
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        bngsim::SteadyStateOptions opts;
        opts.method = "kinsol";
        auto ss = bngsim::find_steady_state(model, opts);
        CHECK(ss.converged, "kinsol alias converges");
        CHECK(ss.method_used == "newton", "kinsol alias echoes canonical newton");
    }

    // "integration" uses the BNG2.pl parity early-stop (||f||_2/n < tol) and
    // agrees with Newton on the equilibrium concentrations.
    {
        auto m_int = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        bngsim::SteadyStateOptions int_opts;
        int_opts.method = "integration";
        int_opts.tol = 1e-8;
        auto ss_int = bngsim::find_steady_state(m_int, int_opts);
        CHECK(ss_int.converged, "integration converges on reversible model");
        CHECK(ss_int.method_used == "integration", "integration reports method_used=integration");
        CHECK(ss_int.residual < int_opts.tol, "integration residual below tol (||f||_2/n)");

        auto m_new = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        bngsim::SteadyStateOptions new_opts;
        new_opts.method = "newton";
        auto ss_new = bngsim::find_steady_state(m_new, new_opts);
        for (size_t i = 0; i < ss_int.concentrations.size(); ++i) {
            CHECK_CLOSE(ss_int.concentrations[i], ss_new.concentrations[i], 1e-4,
                        "integration and newton agree on equilibrium");
        }
    }

    // Removed "auto" (and any other unknown method) is rejected.
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        bngsim::SteadyStateOptions opts;
        opts.method = "auto";
        bool threw = false;
        try {
            bngsim::find_steady_state(model, opts);
        } catch (const std::exception &) {
            threw = true;
        }
        CHECK(threw, "method=auto is rejected (removed)");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: SSA simulation produces reasonable results
// ═══════════════════════════════════════════════════════════════════════════════

int test_ssa_simple_decay() {
    auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
    bngsim::SsaSimulator sim(model);

    bngsim::TimeSpec times{0.0, 50.0, 51};
    auto result = sim.run(times, 42);

    CHECK(result.n_times() == 51, "Expected 51 time points");

    // SSA with 100 molecules: A(50) ≈ 100*exp(-5) ≈ 0.67
    // Large stochastic fluctuations expected — just check it decreased
    double A_init = result.species_data()[0 * 2 + 0];
    double A_end = result.species_data()[50 * 2 + 0];

    CHECK_CLOSE(A_init, 100.0, 1e-10, "A(0) should be 100");
    CHECK(A_end < A_init, "A should decrease over time");
    CHECK(A_end < 10.0, "A(50) should be small (< 10) with k1=0.1");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: SSA deterministic seeding produces reproducible results
// ═══════════════════════════════════════════════════════════════════════════════

int test_ssa_reproducibility() {
    auto model1 = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
    auto model2 = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));

    bngsim::SsaSimulator sim1(model1);
    bngsim::SsaSimulator sim2(model2);

    bngsim::TimeSpec times{0.0, 10.0, 11};
    auto result1 = sim1.run(times, 12345);
    auto result2 = sim2.run(times, 12345);

    // Same seed → identical results
    for (int i = 0; i < result1.n_times(); ++i) {
        for (int j = 0; j < result1.n_species(); ++j) {
            int idx = i * result1.n_species() + j;
            CHECK_CLOSE(result1.species_data()[idx],
                        result2.species_data()[idx],
                        1e-15, "Reproducibility at t=" + std::to_string(i));
        }
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression: SSA rounds fractional initial molecule populations
// ═══════════════════════════════════════════════════════════════════════════════

int test_ssa_fractional_initial_population_rounds() {
    auto model = bngsim::NetworkModel::from_net(data_path("fractional_ssa.net"));
    bngsim::SsaSimulator sim(model);

    bngsim::TimeSpec times{0.0, 5.0, 6};
    auto result = sim.run(times, 1);

    CHECK_CLOSE(result.species_data()[0], 6.0, 1e-12, "SSA should record rounded A(0)");

    for (int i = 0; i < result.n_times(); ++i) {
        double a = result.species_data()[i];
        CHECK_CLOSE(a, std::round(a), 1e-12, "SSA trajectory should stay integer-valued");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: Observable group totals are computed correctly
// ═══════════════════════════════════════════════════════════════════════════════

int test_observables() {
    auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
    bngsim::CvodeSimulator sim(model);

    bngsim::TimeSpec times{0.0, 10.0, 11};
    auto result = sim.run(times);

    // A_tot = species[0], B_tot = species[1]
    // Observables should match species for this simple model
    for (int i = 0; i < result.n_times(); ++i) {
        double A = result.species_data()[i * 2 + 0];
        double A_obs = result.observable_data()[i * 2 + 0];
        CHECK_CLOSE(A_obs, A, 1e-10,
                    "Observable A_tot at t=" + std::to_string(result.time()[i]));
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: C API basic workflow
// ═══════════════════════════════════════════════════════════════════════════════

int test_c_api() {
    BNGModel* model = bng_model_create(data_path("simple_decay.net").c_str());
    CHECK(model != nullptr, "bng_model_create should succeed");

    CHECK(bng_n_species(model) == 2, "2 species");
    CHECK(bng_n_reactions(model) == 1, "1 reaction");

    double k1;
    CHECK(bng_get_param(model, "k1", &k1) == 0, "get_param should succeed");
    CHECK_CLOSE(k1, 0.1, 1e-15, "k1 value via C API");

    CHECK(bng_set_param(model, "k1", 0.5) == 0, "set_param should succeed");

    BNGResult* result = bng_simulate_ode(model, 0.0, 10.0, 11, 1e-8, 1e-8);
    CHECK(result != nullptr, "bng_simulate_ode should succeed");
    CHECK(bng_result_n_times(result) == 11, "11 time points");

    const double* t = bng_result_time(result);
    CHECK(t != nullptr, "time pointer should be valid");
    CHECK_CLOSE(t[0], 0.0, 1e-15, "t[0]");
    CHECK_CLOSE(t[10], 10.0, 1e-10, "t[10]");

    bng_result_free(result);
    bng_reset(model);
    bng_model_free(model);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: Reserved names API
// ═══════════════════════════════════════════════════════════════════════════════

int test_reserved_names() {
    auto names = bngsim::reserved_names();
    CHECK(names.constants.size() == 7, "7 reserved constants");
    CHECK(names.functions.size() > 30, ">30 reserved functions");
    CHECK(names.constants[0] == "_pi", "First constant is _pi");
    CHECK(names.functions[0] == "time", "First function is time");

    // C API
    int n_const = 0;
    const char** consts = bng_reserved_constants(&n_const);
    CHECK(n_const == 7, "C API: 7 constants");
    CHECK(consts != nullptr, "C API: non-null");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: gdat/cdat file export
// ═══════════════════════════════════════════════════════════════════════════════

int test_file_export() {
    auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 10.0, 11};
    auto result = sim.run(times);

    // Export to temporary files (will be in build dir)
    result.to_gdat("test_output.gdat");
    result.to_cdat("test_output.cdat");

    // Check files exist and have content
    FILE* f = fopen("test_output.gdat", "r");
    CHECK(f != nullptr, "gdat file should exist");
    fclose(f);

    f = fopen("test_output.cdat", "r");
    CHECK(f != nullptr, "cdat file should exist");
    fclose(f);

    // Clean up
    std::remove("test_output.gdat");
    std::remove("test_output.cdat");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: to_gdat function-column flags (print_functions / print_rate_laws)
// ═══════════════════════════════════════════════════════════════════════════════

int test_gdat_function_columns() {
    // func_composition has user functions (fRate0, fRate_cond) AND synthetic
    // rate laws (_rateLaw1, _rateLaw2). Issue #58: headers are always bare
    // (no "()"); _rateLawN omitted unless print_rate_laws.
    auto model = bngsim::NetworkModel::from_net(data_path("func_composition.net"));
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 1.0, 3};
    auto result = sim.run(times);

    auto header_of = [](const std::string &path) {
        std::ifstream in(path);
        std::string line;
        std::getline(in, line);
        return line;
    };

    // (1) default: observables only, no function columns.
    result.to_gdat("tg_default.gdat");
    std::string h = header_of("tg_default.gdat");
    CHECK(h.find("fRate0") == std::string::npos, "default omits user functions");
    CHECK(h.find("_rateLaw") == std::string::npos, "default omits rate laws");

    // (2) print_functions: user functions, bare, no rate laws.
    result.to_gdat("tg_funcs.gdat", true);
    h = header_of("tg_funcs.gdat");
    CHECK(h.find("fRate0") != std::string::npos, "print_functions includes fRate0");
    CHECK(h.find("fRate_cond") != std::string::npos, "print_functions includes fRate_cond");
    CHECK(h.find("()") == std::string::npos, "headers are bare (no parens)");
    CHECK(h.find("_rateLaw") == std::string::npos, "print_functions alone omits rate laws");

    // (3) print_rate_laws only: rate laws, bare, no user functions.
    result.to_gdat("tg_rl.gdat", false, true);
    h = header_of("tg_rl.gdat");
    CHECK(h.find("_rateLaw1") != std::string::npos, "print_rate_laws includes _rateLaw1");
    CHECK(h.find("_rateLaw2") != std::string::npos, "print_rate_laws includes _rateLaw2");
    CHECK(h.find("()") == std::string::npos, "rate-law headers are bare");
    CHECK(h.find("fRate0") == std::string::npos, "print_rate_laws alone omits user functions");

    // (4) both: declared order fRate0, _rateLaw1, _rateLaw2, fRate_cond.
    result.to_gdat("tg_both.gdat", true, true);
    h = header_of("tg_both.gdat");
    const size_t p0 = h.find("fRate0");
    const size_t p1 = h.find("_rateLaw1");
    const size_t p2 = h.find("_rateLaw2");
    const size_t p3 = h.find("fRate_cond");
    CHECK(p0 != std::string::npos && p1 != std::string::npos && p2 != std::string::npos &&
              p3 != std::string::npos,
          "all four function columns present");
    CHECK(p0 < p1 && p1 < p2 && p2 < p3, "columns appear in declared order");
    CHECK(h.find("()") == std::string::npos, "combined headers are bare");

    for (const char *f : {"tg_default.gdat", "tg_funcs.gdat", "tg_rl.gdat", "tg_both.gdat"}) {
        std::remove(f);
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression: Fix #1 — time() in functional rate laws tracks simulation time
// ═══════════════════════════════════════════════════════════════════════════════

int test_fix1_time_in_functions() {
    auto model = bngsim::NetworkModel::from_net(data_path("time_dependent_func.net"));
    bngsim::CvodeSimulator sim(model);

    // Rate = (time()+1) * A, A is constant (appears in both reactants and products).
    // So dB/dt = (t+1)*1, B(t) = t^2/2 + t
    bngsim::TimeSpec times{0.0, 10.0, 11};
    auto result = sim.run(times);

    // At t=10: B = 10^2/2 + 10 = 60
    double B_10 = result.species_data()[(result.n_times() - 1) * 2 + 1];
    CHECK_CLOSE(B_10, 60.0, 0.1, "B(10) should be ~60 with time-dependent rate");

    // At t=0: B = 0
    double B_0 = result.species_data()[0 * 2 + 1];
    CHECK_CLOSE(B_0, 0.0, 1e-10, "B(0) should be 0");

    // At t=5: B = 25/2 + 5 = 17.5
    double B_5 = result.species_data()[5 * 2 + 1];
    CHECK_CLOSE(B_5, 17.5, 0.1, "B(5) should be ~17.5");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression: Fix #2 — SSA propensity for homodimerization (repeated reactants)
// ═══════════════════════════════════════════════════════════════════════════════

int test_fix2_ssa_homodimer() {
    auto model = bngsim::NetworkModel::from_net(data_path("homodimer_ssa.net"));
    bngsim::SsaSimulator sim(model);

    // A(0)=1, reaction 1,1->2 with 0.5*k1
    // Propensity = 0.5 * k1 * 1 * (1-1) = 0 (falling factorial)
    // SSA should never fire — A stays at 1
    bngsim::TimeSpec times{0.0, 100.0, 11};
    auto result = sim.run(times, 42);

    // A should remain 1 throughout
    for (int i = 0; i < result.n_times(); ++i) {
        double A = result.species_data()[i * 2 + 0];
        CHECK_CLOSE(A, 1.0, 1e-10,
                    "A should stay at 1 (homodimer can't fire with 1 molecule)");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression: Fix #3 — SSA honors fixed ($) species
// ═══════════════════════════════════════════════════════════════════════════════

int test_fix3_ssa_fixed_species() {
    auto model = bngsim::NetworkModel::from_net(data_path("fixed_species.net"));
    bngsim::SsaSimulator sim(model);

    bngsim::TimeSpec times{0.0, 50.0, 51};
    auto result = sim.run(times, 42);

    // $A should stay at 100 throughout
    for (int i = 0; i < result.n_times(); ++i) {
        double A = result.species_data()[i * 2 + 0];
        CHECK_CLOSE(A, 100.0, 1e-10,
                    "Fixed species $A should stay constant in SSA");
    }

    // B should have grown (reactions still fire, producing B)
    double B_end = result.species_data()[(result.n_times() - 1) * 2 + 1];
    CHECK(B_end > 0.0, "B should have grown via $A -> B");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression: Issue #41 — .net loader honors `$` clamp marker after
// `@<compartment>::` prefix (cBNGL form emitted by BNG2.pl).
// ═══════════════════════════════════════════════════════════════════════════════

int test_issue41_compartmental_fixed_species() {
    auto model = bngsim::NetworkModel::from_net(data_path("sink_compart.net"));

    const auto& sp = model.species();
    CHECK(sp.size() == 2, "Expected 2 species");

    // The `$` marker is stripped from the stored name; the
    // `@compartment::` prefix is preserved.
    CHECK(sp[0].name == "@CP::X()", "Species 0 keeps compartment prefix");
    CHECK(sp[0].fixed == false, "Species 0 (X) is not fixed");
    CHECK(sp[1].name == "@CP::Sink()", "Species 1 has `$` stripped, prefix kept");
    CHECK(sp[1].fixed == true, "Species 1 (Sink) is recognized as fixed");

    // Behavioural: under ODE, Sink must stay at its IC (0) even though
    // X -> Sink fires; without the loader fix this drifts upward.
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 10.0, 11};
    auto result = sim.run(times);

    for (int i = 0; i < result.n_times(); ++i) {
        double sink_i = result.species_data()[i * 2 + 1];
        CHECK_CLOSE(sink_i, 0.0, 1e-10,
                    "Clamped @CP::Sink() must stay at IC under ODE");
    }
    double x_end = result.species_data()[(result.n_times() - 1) * 2 + 0];
    CHECK(x_end < 1.0, "Free X should drain under X -> Sink");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression: Fix #4 — C API null safety
// ═══════════════════════════════════════════════════════════════════════════════

int test_fix4_c_api_null_safety() {
    // These should return errors, not crash
    CHECK(bng_model_create(nullptr) == nullptr, "null path returns null");
    CHECK(bng_get_param(nullptr, "k1", nullptr) == -1, "null model returns -1");
    CHECK(bng_set_param(nullptr, "k1", 1.0) == -1, "null model returns -1");
    CHECK(bng_simulate_ode(nullptr, 0, 10, 11, 1e-8, 1e-8) == nullptr,
          "null model returns null");
    CHECK(bng_simulate_ssa(nullptr, 0, 10, 11, 42) == nullptr,
          "null model returns null");

    // Create a valid model, then test null param name and output pointer
    BNGModel* model = bng_model_create(data_path("simple_decay.net").c_str());
    CHECK(model != nullptr, "valid model creates OK");

    CHECK(bng_set_param(model, nullptr, 1.0) == -1, "null name returns -1");
    CHECK(bng_get_param(model, nullptr, nullptr) == -1, "null name returns -1");
    double val;
    CHECK(bng_get_param(model, "k1", nullptr) == -1, "null output returns -1");

    bng_model_free(model);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression: Fix #5 — Species from expression-valued parameters
// ═══════════════════════════════════════════════════════════════════════════════

int test_fix5_expr_param_species() {
    auto model = bngsim::NetworkModel::from_net(data_path("expr_param_species.net"));

    // p = 2+3 = 5. Species A should start at 5, not 2.
    CHECK_CLOSE(model.get_param("p"), 5.0, 1e-10, "p should be 5 (=2+3)");
    CHECK_CLOSE(model.species()[0].concentration, 5.0, 1e-10,
                "A(0) should be 5 from expression param p=2+3");
    CHECK_CLOSE(model.species()[0].initial_conc, 5.0, 1e-10,
                "A initial_conc should be 5");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Regression: Fix #6 — set_param on ConstantExpression detaches expression
// ═══════════════════════════════════════════════════════════════════════════════

int test_fix6_set_param_const_expr() {
    // Model has: d__FREE=70 (Constant), d=d__FREE (ConstantExpression)
    // Bug: set_param("d", 27) was overwritten by re-evaluation of "d = d__FREE"
    // Fix: set_param detaches expression, so d stays at 27
    auto model = bngsim::NetworkModel::from_net(data_path("const_expr_setparam.net"));

    // Verify initial state
    CHECK_CLOSE(model.get_param("d__FREE"), 70.0, 1e-10, "d__FREE initial");
    CHECK_CLOSE(model.get_param("d"), 70.0, 1e-10, "d initial (= d__FREE)");

    // Set d directly to 27 (like BNG's setParameter("d", 27))
    model.set_param("d", 27.0);
    CHECK_CLOSE(model.get_param("d"), 27.0, 1e-10,
                "d should be 27 after set_param (not reverted to d__FREE)");
    CHECK_CLOSE(model.get_param("d__FREE"), 70.0, 1e-10,
                "d__FREE should still be 70");

    // Changing d__FREE should NOT affect d anymore (expression detached)
    model.set_param("d__FREE", 99.0);
    CHECK_CLOSE(model.get_param("d__FREE"), 99.0, 1e-10, "d__FREE updated");
    CHECK_CLOSE(model.get_param("d"), 27.0, 1e-10,
                "d should still be 27 (detached from d__FREE)");

    // Clone should preserve the detached state
    auto clone = model.clone();
    CHECK_CLOSE(clone.get_param("d"), 27.0, 1e-10,
                "cloned d should be 27 (detached)");
    CHECK_CLOSE(clone.get_param("d__FREE"), 99.0, 1e-10,
                "cloned d__FREE should be 99");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SSA propensity: A+A+B → C (mixed repeated + distinct)
// ═══════════════════════════════════════════════════════════════════════════════

int test_ssa_aab_propensity() {
    // A(0)=1, B(0)=10: propensity = 0.5*k*1*(1-1)*10 = 0 → stuck
    auto model = bngsim::NetworkModel::from_net(data_path("ssa_aab.net"));
    bngsim::SsaSimulator sim(model);

    bngsim::TimeSpec times{0.0, 100.0, 11};
    auto result = sim.run(times, 42);

    // A should stay at 1 (can't pick 2 A's from 1)
    for (int i = 0; i < result.n_times(); ++i) {
        double A = result.species_data()[i * 3 + 0];
        CHECK_CLOSE(A, 1.0, 1e-10, "A+A+B: A should stay 1 when nA=1");
    }
    // C should stay at 0
    double C_end = result.species_data()[(result.n_times() - 1) * 3 + 2];
    CHECK_CLOSE(C_end, 0.0, 1e-10, "A+A+B: C should stay 0 when nA=1");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SSA propensity: A+A+A → C (homotrimer)
// ═══════════════════════════════════════════════════════════════════════════════

int test_ssa_aaa_propensity() {
    // A(0)=2: propensity = (1/6)*k*2*1*0 = 0 → stuck (need 3 molecules)
    auto model = bngsim::NetworkModel::from_net(data_path("ssa_aaa.net"));
    bngsim::SsaSimulator sim(model);

    bngsim::TimeSpec times{0.0, 100.0, 11};
    auto result = sim.run(times, 42);

    // A should stay at 2 (falling factorial 2*1*0 = 0)
    for (int i = 0; i < result.n_times(); ++i) {
        double A = result.species_data()[i * 2 + 0];
        CHECK_CLOSE(A, 2.0, 1e-10, "A+A+A: A should stay 2 when nA=2");
    }
    // C should stay at 0
    double C_end = result.species_data()[(result.n_times() - 1) * 2 + 1];
    CHECK_CLOSE(C_end, 0.0, 1e-10, "A+A+A: C should stay 0 when nA=2");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SSA propensity: A+B+C → D (all distinct, one zero blocks reaction)
// ═══════════════════════════════════════════════════════════════════════════════

int test_ssa_abc_propensity() {
    // A(0)=5, B(0)=3, C(0)=0: propensity = k*5*3*0 = 0 → stuck
    auto model = bngsim::NetworkModel::from_net(data_path("ssa_abc.net"));
    bngsim::SsaSimulator sim(model);

    bngsim::TimeSpec times{0.0, 100.0, 11};
    auto result = sim.run(times, 42);

    // All species should be unchanged (C=0 blocks the reaction)
    for (int i = 0; i < result.n_times(); ++i) {
        double A = result.species_data()[i * 4 + 0];
        double B = result.species_data()[i * 4 + 1];
        double C = result.species_data()[i * 4 + 2];
        double D = result.species_data()[i * 4 + 3];
        CHECK_CLOSE(A, 5.0, 1e-10, "A+B+C: A should stay 5");
        CHECK_CLOSE(B, 3.0, 1e-10, "A+B+C: B should stay 3");
        CHECK_CLOSE(C, 0.0, 1e-10, "A+B+C: C should stay 0");
        CHECK_CLOSE(D, 0.0, 1e-10, "A+B+C: D should stay 0");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: Sat rate law loads via loader-level rewrite (deprecated → Functional)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The loader (src/net_file_loader.cpp:rewrite_legacy_sat_hill_rate_laws)
// auto-rewrites the deprecated `Sat k K` token into an equivalent Functional
// rate law and records a deprecation warning on the model. The Python
// equivalent lives in test_new_features.py::TestSatHillNetRewrite.

int test_sat_loads_with_rewrite_warning() {
    auto model = bngsim::NetworkModel::from_net(data_path("sat_rewrite.net"));

    CHECK(model.n_reactions() == 1, "expected 1 reaction post-rewrite");

    const auto warnings = model.load_warnings();
    CHECK(!warnings.empty(), "rewrite should emit at least one load warning");
    const std::string joined = [&]() {
        std::string out;
        for (const auto &w : warnings) {
            out += w;
            out += '\n';
        }
        return out;
    }();
    CHECK(joined.find("Legacy/deprecated") != std::string::npos,
          "warning should label the rewrite as legacy/deprecated: " + joined);
    CHECK(joined.find("Sat") != std::string::npos, "warning should mention 'Sat': " + joined);
    CHECK(joined.find("Functional") != std::string::npos,
          "warning should point at the Functional replacement: " + joined);
    CHECK(joined.find("k/(K+S)") != std::string::npos,
          "warning should show the k/(K+S) formula: " + joined);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: Hill rate law loads via loader-level rewrite (deprecated → Functional)
// ═══════════════════════════════════════════════════════════════════════════════

int test_hill_loads_with_rewrite_warning() {
    auto model = bngsim::NetworkModel::from_net(data_path("hill_rewrite.net"));

    CHECK(model.n_reactions() == 1, "expected 1 reaction post-rewrite");

    const auto warnings = model.load_warnings();
    CHECK(!warnings.empty(), "rewrite should emit at least one load warning");
    const std::string joined = [&]() {
        std::string out;
        for (const auto &w : warnings) {
            out += w;
            out += '\n';
        }
        return out;
    }();
    CHECK(joined.find("Legacy/deprecated") != std::string::npos,
          "warning should label the rewrite as legacy/deprecated: " + joined);
    CHECK(joined.find("Hill") != std::string::npos, "warning should mention 'Hill': " + joined);
    CHECK(joined.find("Functional") != std::string::npos,
          "warning should point at the Functional replacement: " + joined);
    CHECK(joined.find("Vmax*S") != std::string::npos,
          "warning should show the Vmax*S formula: " + joined);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: MM tQSSA simulation — E + S → E + P
// ═══════════════════════════════════════════════════════════════════════════════

int test_mm_tqssa() {
    auto model = bngsim::NetworkModel::from_net(data_path("mm_tqssa.net"));

    CHECK(model.n_species() == 3, "Expected 3 species");
    CHECK(model.n_reactions() == 1, "Expected 1 reaction");

    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 200.0, 201};
    auto result = sim.run(times);

    // Conservation: E should stay constant (catalyst)
    for (int i = 0; i < result.n_times(); ++i) {
        double E = result.species_data()[i * 3 + 0];
        CHECK_CLOSE(E, 10.0, 1e-6, "E (enzyme) should be conserved");
    }

    // Conservation: S + P = S_0 = 100
    for (int i = 0; i < result.n_times(); ++i) {
        double S = result.species_data()[i * 3 + 1];
        double P = result.species_data()[i * 3 + 2];
        CHECK_CLOSE(S + P, 100.0, 1e-4, "S + P should be conserved");
    }

    // Substrate should decrease over time
    int last = result.n_times() - 1;
    double S_end = result.species_data()[last * 3 + 1];
    double P_end = result.species_data()[last * 3 + 2];
    CHECK(S_end < 10.0, "S should decrease substantially by t=200");
    CHECK(P_end > 90.0, "P should increase substantially by t=200");

    // tQSSA specific: verify initial rate.
    // At t=0: E=10, S=100, Km=50, kcat=1
    // delta = S - Km - E = 100 - 50 - 10 = 40
    // sFree = 0.5*(40 + sqrt(40^2 + 4*50*100)) = 0.5*(40 + sqrt(1600+20000))
    //       = 0.5*(40 + sqrt(21600)) = 0.5*(40 + 146.969) = 93.485
    // rate  = 1 * 93.485 * 10 / (50 + 93.485) = 934.85 / 143.485 = 6.515
    //
    // Compare with sQSSA: rate = kcat * E * S / (Km + S) = 1*10*100/150 = 6.667
    // The tQSSA rate is slightly lower because it accounts for enzyme-bound substrate.
    //
    // Check P at small t ≈ rate * dt
    double P_1 = result.species_data()[1 * 3 + 2];
    double dt = result.time()[1] - result.time()[0];
    double approx_rate = P_1 / dt;
    CHECK(approx_rate > 6.0, "Initial MM rate should be > 6.0");
    CHECK(approx_rate < 7.0, "Initial MM rate should be < 7.0");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: TFUN time-indexed — load .net with tfun(), simulate, verify interpolation
// ═══════════════════════════════════════════════════════════════════════════════

int test_tfun_time_indexed() {
    auto model = bngsim::NetworkModel::from_net(data_path("tfun_time_indexed.net"));

    CHECK(model.n_table_functions() == 1, "Should have 1 table function");
    CHECK(model.table_function_names()[0] == "cumNcases", "TFUN name");
    CHECK(model.n_functions() == 1, "Should have 1 function");

    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 7.0, 71};
    auto result = sim.run(times);

    // cumNcases(0) = 0, cumNcases(1) = 0, cumNcases(2) = 1, ...
    // At t=0.5: cumNcases = lerp(0, 0, 0.5) = 0
    // At t=1.5: cumNcases = lerp(0, 1, 0.5) = 0.5
    // dB/dt = cumNcases(t), so B = integral of cumNcases over [0,t]

    // B should be 0 at t=0
    double B_0 = result.species_data()[0 * 2 + 1];
    CHECK_CLOSE(B_0, 0.0, 1e-10, "B(0) should be 0");

    // B should be > 0 after t=2 (that's when cumNcases starts rising)
    // At t=7: cumNcases has been rising, B should be substantial
    int last = result.n_times() - 1;
    double B_end = result.species_data()[last * 2 + 1];
    CHECK(B_end > 10.0, "B(7) should be > 10 (integral of rising cumNcases)");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: TFUN parameter-indexed — response indexed by drug_conc
// ═══════════════════════════════════════════════════════════════════════════════

int test_tfun_param_indexed() {
    auto model = bngsim::NetworkModel::from_net(data_path("tfun_param_indexed.net"));

    CHECK(model.n_table_functions() == 1, "Should have 1 table function");
    CHECK(model.table_function_names()[0] == "response", "TFUN name");

    // drug_conc = 1.0 → response = 50.0 (from table: 1.0 → 50.0)
    // The reaction rate is response(drug_conc) = 50.0
    // So dA/dt = -50*A, dB/dt = 50*A → fast exponential decay
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 1.0, 11};
    auto result = sim.run(times);

    // A should decrease substantially (rate = 50 * A)
    double A_end = result.species_data()[(result.n_times() - 1) * 2 + 0];
    CHECK(A_end < 1.0, "A should be nearly 0 at t=1 with rate=50");

    // Change drug_conc to 0.0 → response should be 0.0
    model.set_param("drug_conc", 0.0);
    model.reset();
    auto result2 = sim.run(times);
    double A_end2 = result2.species_data()[(result2.n_times() - 1) * 2 + 0];
    CHECK_CLOSE(A_end2, 100.0, 0.1, "A should stay ~100 when drug_conc=0 (response=0)");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: TFUN from in-memory arrays (add_table_function API)
// ═══════════════════════════════════════════════════════════════════════════════

int test_tfun_from_arrays() {
    auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));

    // Add a table function programmatically, indexed by time
    std::vector<double> xs = {0.0, 5.0, 10.0, 50.0};
    std::vector<double> ys = {1.0, 1.0,  2.0,  2.0};
    model.add_table_function("scale_factor", xs, ys, "time");

    CHECK(model.n_table_functions() == 1, "Should have 1 table function");

    // Evaluate at various times via the internal tfun_ prefixed name
    model.set_current_time(0.0);
    auto& eval = model.evaluator();
    int expr_id = eval.compile("tfun_scale_factor()");

    model.set_current_time(0.0);
    CHECK_CLOSE(eval.evaluate(expr_id), 1.0, 1e-10, "scale_factor(0) = 1.0");

    model.set_current_time(7.5);
    CHECK_CLOSE(eval.evaluate(expr_id), 1.5, 1e-10, "scale_factor(7.5) = 1.5 (interpolated)");

    model.set_current_time(50.0);
    CHECK_CLOSE(eval.evaluate(expr_id), 2.0, 1e-10, "scale_factor(50) = 2.0");

    // Beyond endpoints: constant extrapolation
    model.set_current_time(100.0);
    CHECK_CLOSE(eval.evaluate(expr_id), 2.0, 1e-10, "scale_factor(100) = 2.0 (extrapolated)");

    model.set_current_time(-5.0);
    CHECK_CLOSE(eval.evaluate(expr_id), 1.0, 1e-10, "scale_factor(-5) = 1.0 (extrapolated)");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test: TFUN survives clone (deep copy)
// ═══════════════════════════════════════════════════════════════════════════════

int test_tfun_clone() {
    auto model = bngsim::NetworkModel::from_net(data_path("tfun_time_indexed.net"));
    CHECK(model.n_table_functions() == 1, "Original has 1 TFUN");

    auto clone = model.clone();
    CHECK(clone.n_table_functions() == 1, "Clone has 1 TFUN");
    CHECK(clone.table_function_names()[0] == "cumNcases", "Clone TFUN name");

    // Simulate the clone — should work independently
    bngsim::CvodeSimulator sim(clone);
    bngsim::TimeSpec times{0.0, 5.0, 51};
    auto result = sim.run(times);

    CHECK(result.n_times() == 51, "Clone simulation produced 51 points");
    double B_end = result.species_data()[(result.n_times() - 1) * 2 + 1];
    CHECK(B_end > 0.0, "Clone: B should be > 0 after simulation");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session 12: Function composition, && operator, empty groups
// ═══════════════════════════════════════════════════════════════════════════════

int test_func_composition() {
    // Tests three fixes at once:
    //   1. Function composition: -1*fRate0() where fRate0 is another function
    //   2. C-style logical operators: && → and in if() conditions
    //   3. Model loads and runs ODE without errors

    auto model = bngsim::NetworkModel::from_net(data_path("func_composition.net"));

    CHECK(model.n_species() == 3, "Expected 3 species");
    CHECK(model.n_reactions() == 3, "Expected 3 reactions");
    CHECK(model.n_functions() == 4, "Expected 4 functions (including composed)");

    // Run ODE simulation
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 1.0, 11};
    auto result = sim.run(times);

    CHECK(result.n_times() == 11, "Expected 11 time points");

    // fRate0 = k1 * A_tot = 0.1 * 100 = 10 at t=0
    // _rateLaw1 = -1 * fRate0 = -10 (produces A, so source rate = |rate| = 10)
    // Reaction 1: A→B with rate fRate0 (rate = 10 at t=0)
    // Reaction 2: 0→A with rate _rateLaw1 (= -(-10) = +10... but it's Functional)
    // So at t=0 there is nonzero flow — model runs without NaN.

    // Verify no NaN/Inf in output
    for (int i = 0; i < result.n_times() * result.n_species(); ++i) {
        CHECK(!std::isnan(result.species_data()[i]), "No NaN in species data");
        CHECK(!std::isinf(result.species_data()[i]), "No Inf in species data");
    }

    // Verify fRate_cond works (uses && operator):
    // fRate_cond = if((A_tot>=0 && A_tot<=200), k2*C_tot, 0)
    // At t=0: A_tot=100 ∈ [0,200], so fRate_cond = 0.05*50 = 2.5
    // This feeds reaction 3: 0→C, which means C should grow
    double C_end = result.species_data()[(result.n_times() - 1) * 3 + 2];
    CHECK(C_end > 50.0, "C should grow (fRate_cond with && operator is working)");

    // Clone also works with composed functions
    auto clone = model.clone();
    CHECK(clone.n_functions() == 4, "Clone has 4 functions");
    bngsim::CvodeSimulator sim2(clone);
    auto result2 = sim2.run(times);
    CHECK(result2.n_times() == 11, "Clone simulation works");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session 16: ExprTk case-sensitivity regression test
// ═══════════════════════════════════════════════════════════════════════════════

int test_case_sensitivity() {
    // Tests that ExprTk treats k3 and K3 as distinct parameters.
    // Before the fix (exprtk_disable_caseinsensitivity), ExprTk lowered
    // all identifiers, so k3=0.025 and K3=15.0 collapsed into one variable.
    // The Kholodenko_2000 MAPK model triggered this: 600× error in rates.
    // See bngsim/dev/investigations/KHOLODENKO_BUG_REPORT.md.

    auto model = bngsim::NetworkModel::from_net(data_path("case_sensitivity.net"));

    // Verify parameters loaded with correct distinct values
    CHECK_CLOSE(model.get_param("k3"), 0.025, 1e-15, "k3 should be 0.025");
    CHECK_CLOSE(model.get_param("K3"), 15.0, 1e-15, "K3 should be 15.0");
    CHECK_CLOSE(model.get_param("k4"), 0.025, 1e-15, "k4 should be 0.025");
    CHECK_CLOSE(model.get_param("K4"), 15.0, 1e-15, "K4 should be 15.0");

    // Run ODE — the key test is that k3 and K3 are distinct values.
    // With case-insensitive bug: K3=k3=0.025, denominator ≈ 0.025+S2 (tiny)
    //   → rate is ~100× higher → P grows enormously fast
    // With fix: K3=15.0, denominator ≈ 15+50 = 65 → rate is moderate
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 10.0, 11};
    auto result = sim.run(times);

    // At t=10, S1 should still have substantial concentration left
    // because the rate (k3*S1)/(K3+S2) with K3=15 is moderate.
    // With the bug (K3=0.025), the denominator would be tiny,
    // making the rate enormous → S1 would be depleted almost instantly.
    double S1_end = result.species_data()[(result.n_times() - 1) * 3 + 0];
    double S2_end = result.species_data()[(result.n_times() - 1) * 3 + 1];

    // Correct behavior: S1 and S2 decrease moderately
    CHECK(S1_end > 1.0, "S1(10) should be > 1 (moderate rate with K3=15)");
    CHECK(S2_end > 1.0, "S2(10) should be > 1 (moderate rate with K4=15)");

    // Bug behavior would have S1 ≈ 0, S2 ≈ 0 by t=10 (rate ≈ 100× too fast)
    // This is the key discrimination: case-sensitive → moderate rates

    // Verify no NaN/Inf
    for (int i = 0; i < result.n_times() * result.n_species(); ++i) {
        CHECK(!std::isnan(result.species_data()[i]), "No NaN in species data");
        CHECK(!std::isinf(result.species_data()[i]), "No Inf in species data");
    }

    // Clone preserves case-sensitive params
    auto clone = model.clone();
    CHECK_CLOSE(clone.get_param("k3"), 0.025, 1e-15, "Clone: k3 should be 0.025");
    CHECK_CLOSE(clone.get_param("K3"), 15.0, 1e-15, "Clone: K3 should be 15.0");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session 18: Graph coloring for sparse Jacobian (Curtis-Powell-Reid)
// ═══════════════════════════════════════════════════════════════════════════════

int test_graph_coloring() {
    // Tests graph coloring on a model large enough to trigger sparse Jacobian.
    // Uses the reversible binding model (3 species) to test basic coloring,
    // and verifies that coloring data is populated and valid.

    // 1. Coloring is lazy: absent until asked for, then valid.
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));

        // build() does not color (GH #29) — only the sparse colored-FD callback
        // consumes it, so nothing is computed until ensure_jacobian_coloring().
        CHECK(!model.jacobian_sparsity().has_coloring(),
              "build() leaves the pattern uncolored");

        const auto& sp = model.ensure_jacobian_coloring();
        CHECK(sp.n == 3, "Reversible model has 3 species");
        CHECK(sp.nnz > 0, "Sparsity pattern should have nonzeros");

        // Any pattern with a structural nonzero colors, at any density: a fully
        // dense one just degenerates to one column per color.
        CHECK(sp.has_coloring(), "Pattern with nonzeros is colored on demand");
        CHECK(sp.n_colors > 0, "n_colors should be positive");
        CHECK(sp.n_colors <= sp.n, "n_colors <= n_species");
        CHECK(static_cast<int>(sp.colors.size()) == sp.n,
              "colors vector size == n_species");
        CHECK(static_cast<int>(sp.color_groups.size()) == sp.n_colors,
              "color_groups size == n_colors");

        // Verify all columns are assigned valid colors
        for (int j = 0; j < sp.n; ++j) {
            CHECK(sp.colors[j] >= 0 && sp.colors[j] < sp.n_colors,
                  "Color in valid range for column " + std::to_string(j));
        }

        // Verify coloring validity: no two same-color columns share a row
        for (int c = 0; c < sp.n_colors; ++c) {
            const auto& group = sp.color_groups[c];
            // Collect all rows covered by this color group
            std::vector<bool> row_used(sp.n, false);
            for (int j : group) {
                for (int64_t k = sp.col_ptrs[j]; k < sp.col_ptrs[j + 1]; ++k) {
                    int row = static_cast<int>(sp.row_indices[k]);
                    CHECK(!row_used[row],
                          "Color " + std::to_string(c) + ": row " +
                          std::to_string(row) + " used by multiple columns");
                    row_used[row] = true;
                }
            }
        }

        // Idempotent: a second call returns the same materialized coloring.
        CHECK(model.ensure_jacobian_coloring().n_colors == sp.n_colors,
              "ensure_jacobian_coloring is idempotent");

        // Simulate with ODE — should produce same results as before
        // (graph coloring doesn't affect small models using dense solver,
        //  but the code path is exercised if KLU is available and N >= 50).
        bngsim::CvodeSimulator sim(model);
        bngsim::TimeSpec times{0.0, 1000.0, 11};
        auto result = sim.run(times);

        int last = result.n_times() - 1;
        double A_eq = result.species_data()[last * 3 + 0];
        double C_eq = result.species_data()[last * 3 + 2];
        CHECK_CLOSE(A_eq + C_eq, 100.0, 1e-3, "Conservation A + C = 100 (with coloring)");
    }

    // 2. Clone shares the coloring — including one materialized only by the clone
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        auto clone = model.clone();

        // Neither has colored yet; the clone triggers the shared compute, and
        // the original sees it without asking (same SharedModelData).
        const auto& sp_clone = clone.ensure_jacobian_coloring();
        const auto& sp_orig = model.jacobian_sparsity();

        CHECK(sp_clone.n == sp_orig.n, "Clone n matches");
        CHECK(sp_clone.nnz == sp_orig.nnz, "Clone nnz matches");
        CHECK(sp_orig.has_coloring(), "Coloring materialized by the clone is shared");
        CHECK(sp_clone.n_colors == sp_orig.n_colors, "Clone n_colors matches");
        CHECK(sp_clone.colors == sp_orig.colors, "Clone colors vector matches");

        // Simulate clone — independent from original
        bngsim::CvodeSimulator sim(clone);
        bngsim::TimeSpec times{0.0, 100.0, 11};
        auto result = sim.run(times);
        CHECK(result.n_times() == 11, "Clone ODE simulation succeeded");
    }

    // 3. Verify the func_composition model (has Functional rate laws → denser Jacobian)
    {
        auto model = bngsim::NetworkModel::from_net(data_path("func_composition.net"));
        const auto& sp = model.ensure_jacobian_coloring();
        CHECK(sp.n == 3, "func_composition has 3 species");

        // Functional rate laws conservatively mark every species as a
        // dependency, which densifies the pattern. Coloring no longer has a
        // density ceiling (GH #29), so these are colored like any other — the
        // denser the pattern the closer n_colors gets to n, until a fully dense
        // one degenerates to one column per color (plain FD). That degenerate
        // case is exactly what force_sparse_linear_solver needs to fill the CSC
        // matrix on small functional models; python/tests/
        // test_force_sparse_linear_solver.py drives it end-to-end.
        CHECK(sp.has_coloring(), "Functional-rate-law pattern is colored on demand");
        CHECK(sp.n_colors <= sp.n, "n_colors <= n_species however dense");

        // ... and the model should still load and simulate correctly
        bngsim::CvodeSimulator sim(model);
        bngsim::TimeSpec times{0.0, 1.0, 11};
        auto result = sim.run(times);

        for (int i = 0; i < result.n_times() * result.n_species(); ++i) {
            CHECK(!std::isnan(result.species_data()[i]), "No NaN with func coloring");
            CHECK(!std::isinf(result.species_data()[i]), "No Inf with func coloring");
        }
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session 19: Analytical Jacobian for Elementary mass-action models
// ═══════════════════════════════════════════════════════════════════════════════

int test_analytical_jacobian() {
    // Tests that analytical Jacobian data is pre-computed for all-Elementary
    // models and not for models with Functional rate laws.

    // 1. All-Elementary model: analytical Jacobian should be available
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        const auto& ajd = model.analytical_jacobian();

        CHECK(ajd.available, "Analytical Jacobian should be available for all-Elementary model");
        CHECK(ajd.reactions.size() == static_cast<size_t>(model.n_reactions()),
              "One ReactionTerms per reaction");

        // Verify reaction terms are populated
        // Reaction 0: A + B → C (kf=0.001)
        //   reactants: A (idx 0, mult 1), B (idx 1, mult 1)
        //   affected: A(-1), B(-1), C(+1) → 3 affected species
        const auto& rxn0 = ajd.reactions[0];
        CHECK(rxn0.reactants.size() == 2, "Rxn 0 has 2 unique reactant species");
        CHECK(rxn0.rate_param_idx0 >= 0, "Rxn 0 has valid rate param");

        // ODE simulation should produce identical results to before
        // (the analytical Jacobian is exact — no approximation error)
        bngsim::CvodeSimulator sim(model);
        bngsim::TimeSpec times{0.0, 1000.0, 101};
        auto result = sim.run(times);

        int last = result.n_times() - 1;
        double A_eq = result.species_data()[last * 3 + 0];
        double C_eq = result.species_data()[last * 3 + 2];
        CHECK_CLOSE(A_eq + C_eq, 100.0, 1e-3, "Conservation A + C = 100 (analytical Jac)");

        double B_eq = result.species_data()[last * 3 + 1];
        CHECK_CLOSE(B_eq + C_eq, 50.0, 1e-3, "Conservation B + C = 50 (analytical Jac)");

        // No NaN/Inf
        for (int i = 0; i < result.n_times() * result.n_species(); ++i) {
            CHECK(!std::isnan(result.species_data()[i]), "No NaN with analytical Jac");
            CHECK(!std::isinf(result.species_data()[i]), "No Inf with analytical Jac");
        }
    }

    // 2. Simple decay: unimolecular reaction (no "others")
    {
        auto model = bngsim::NetworkModel::from_net(data_path("simple_decay.net"));
        const auto& ajd = model.analytical_jacobian();

        CHECK(ajd.available, "Analytical Jac available for simple_decay");
        CHECK(ajd.reactions.size() == 1, "1 reaction");

        // A → B with rate k1*A
        // ∂v/∂A = k1, J[A][A] = -k1, J[B][A] = +k1
        const auto& rxn = ajd.reactions[0];
        CHECK(rxn.reactants.size() == 1, "1 unique reactant (A)");
        CHECK(rxn.reactants[0].multiplicity == 1, "A appears once");
        CHECK(rxn.reactants[0].others.empty(), "No other reactants (unimolecular)");

        // Simulation should match analytical: A(t) = 100*exp(-0.1*t)
        bngsim::CvodeSimulator sim(model);
        bngsim::TimeSpec times{0.0, 50.0, 51};
        auto result = sim.run(times);

        for (int i = 0; i < result.n_times(); ++i) {
            double t = result.time()[i];
            double A_exact = 100.0 * std::exp(-0.1 * t);
            double A_sim = result.species_data()[i * 2 + 0];
            CHECK_CLOSE(A_sim, A_exact, 1e-4,
                        "A(" + std::to_string(t) + ") analytical Jac");
        }
    }

    // 3. Homodimer model: A + A → B (stat_factor = 0.5, multiplicity = 2)
    {
        auto model = bngsim::NetworkModel::from_net(data_path("homodimer_ssa.net"));
        const auto& ajd = model.analytical_jacobian();

        CHECK(ajd.available, "Analytical Jac available for homodimer");
        // A+A→B: ∂v/∂A = 0.5*k*2*A = k*A (multiplicity 2, sf 0.5)
        const auto& rxn = ajd.reactions[0];
        CHECK(rxn.reactants.size() == 1, "1 unique reactant (A appears twice)");
        CHECK(rxn.reactants[0].multiplicity == 2, "A multiplicity is 2");
        CHECK(rxn.reactants[0].others.empty(), "No other species");
    }

    // 4. Functional model: analytical structure is built (GH #76), but the
    //    Jacobian is not COMPLETE until the symbolic Functional derivative terms
    //    are attached (opt-in, via the Python attach_functional_jacobian driver),
    //    which does not happen in this pure-C++ path.
    {
        auto model = bngsim::NetworkModel::from_net(data_path("func_composition.net"));
        const auto& ajd = model.analytical_jacobian();

        CHECK(ajd.available,
              "Analytical Jac structure available for Functional model (GH #76)");
        CHECK(ajd.n_functional > 0, "Functional reactions are counted (GH #76)");
        CHECK(!model.analytical_jacobian_complete(),
              "Analytical Jac NOT complete until Functional terms are attached");
    }

    // 5. MM model: closed-form analytical Jacobian (GH #76 task 3). MM reactions
    //    are now analytically covered (kcat·E·sFree/(Km+sFree), tQSSA), so the
    //    model's analytical Jacobian is available + complete, and
    //    fill_dense_analytical_jacobian must match central FD of the engine RHS.
    {
        auto model = bngsim::NetworkModel::from_net(data_path("mm_tqssa.net"));
        const auto& ajd = model.analytical_jacobian();

        CHECK(ajd.available, "Analytical Jac available for MM model (GH #76 task 3)");
        CHECK(ajd.mm_reactions.size() == 1, "One MM closed-form term built");
        CHECK(model.analytical_jacobian_complete(), "MM analytical Jacobian complete");

        // fill_dense vs central FD at several states spanning E≷S and the
        // sFree clamp.
        int ns = model.n_species();
        std::vector<std::vector<double>> states = {
            {10.0, 100.0, 0.0},   // initial (E<S)
            {80.0, 30.0, 12.0},   // E>S
            {5.0, 5.0, 50.0},     // E≈S
            {0.3, 0.05, 1.0},     // tiny substrate (near clamp)
        };
        std::vector<double> Ja(static_cast<size_t>(ns) * ns), fp(ns), fm(ns);
        double maxrel = 0.0;
        for (const auto& y : states) {
            model.fill_dense_analytical_jacobian(0.0, y.data(), Ja.data());
            for (int j = 0; j < ns; ++j) {
                double h = 1e-6 * std::max(std::abs(y[j]), 1.0);
                auto yp = y, ym = y;
                yp[j] += h;
                ym[j] -= h;
                model.compute_derivs(0.0, yp.data(), fp.data());
                model.compute_derivs(0.0, ym.data(), fm.data());
                for (int i = 0; i < ns; ++i) {
                    double fd = (fp[i] - fm[i]) / (2.0 * h);
                    double an = Ja[static_cast<size_t>(j) * ns + i];
                    double denom = std::max(std::max(std::abs(fd), std::abs(an)), 1e-6);
                    maxrel = std::max(maxrel, std::abs(fd - an) / denom);
                }
            }
        }
        CHECK(maxrel < 1e-5,
              "MM analytical Jac matches FD across states (max_rel=" + std::to_string(maxrel) + ")");
    }

    // 6. Clone preserves analytical Jac
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        auto clone = model.clone();
        const auto& ajd_orig = model.analytical_jacobian();
        const auto& ajd_clone = clone.analytical_jacobian();

        CHECK(ajd_clone.available == ajd_orig.available, "Clone preserves availability");
        CHECK(ajd_clone.reactions.size() == ajd_orig.reactions.size(),
              "Clone preserves reaction count");
    }

    // 7. Per-observable .net path (GH #76 task 2): a Functional rate with a
    //    species factor, rate = fr()*A*B with fr() = k1*Atot and Atot = A + 3*C.
    //    The product rule + observable-group chain rule must reproduce the engine
    //    RHS derivative exactly (validated here against central FD), and a wrong
    //    derivative must be rejected by the self-check.
    {
        auto model = bngsim::NetworkModel::from_net(data_path("per_observable_jac.net"));
        int atot = -1;
        auto onames = model.observable_names();
        for (int i = 0; i < static_cast<int>(onames.size()); ++i)
            if (onames[i] == "Atot")
                atot = i;
        CHECK(atot >= 0, "Atot observable found");

        bngsim::FunctionalJacobianInput in;
        in.rxn_idx = 0; // A+B -> C (functional, per-observable)
        in.per_observable = true;
        in.deriv_terms = {{atot, "k1"}}; // ∂fr/∂Atot = k1
        bool ok = model.set_functional_jacobian({in});
        CHECK(ok, "Per-observable Jacobian attaches + passes self-check");
        CHECK(model.analytical_jacobian_complete(),
              "Analytical Jac complete after per-observable attach");

        // fill_dense_analytical_jacobian vs central FD of the engine RHS at a
        // probe state away from the initial point.
        int ns = model.n_species();
        std::vector<double> y = {37.0, 21.0, 14.0};
        std::vector<double> Ja(static_cast<size_t>(ns) * ns);
        model.fill_dense_analytical_jacobian(0.0, y.data(), Ja.data());
        std::vector<double> fp(ns), fm(ns);
        double maxrel = 0.0;
        for (int j = 0; j < ns; ++j) {
            double h = 1e-6 * std::max(std::abs(y[j]), 1.0);
            auto yp = y, ym = y;
            yp[j] += h;
            ym[j] -= h;
            model.compute_derivs(0.0, yp.data(), fp.data());
            model.compute_derivs(0.0, ym.data(), fm.data());
            for (int i = 0; i < ns; ++i) {
                double fd = (fp[i] - fm[i]) / (2.0 * h);
                double an = Ja[static_cast<size_t>(j) * ns + i];
                double denom = std::max(std::max(std::abs(fd), std::abs(an)), 1e-6);
                maxrel = std::max(maxrel, std::abs(fd - an) / denom);
            }
        }
        CHECK(maxrel < 1e-5,
              "Per-observable analytical Jac matches FD (max_rel=" + std::to_string(maxrel) + ")");

        // A 2x-wrong ∂fr/∂Atot must be caught by the self-check (→ FD fallback).
        auto model2 = bngsim::NetworkModel::from_net(data_path("per_observable_jac.net"));
        bngsim::FunctionalJacobianInput bad;
        bad.rxn_idx = 0;
        bad.per_observable = true;
        bad.deriv_terms = {{atot, "2.0*k1"}};
        CHECK(!model2.set_functional_jacobian({bad}),
              "Corrupted per-observable derivative rejected by self-check");

        // Clone recompiles the per-observable derivative ids and stays complete.
        auto clone = model.clone();
        CHECK(clone.analytical_jacobian_complete(),
              "Clone preserves per-observable analytical Jac completeness");
        std::vector<double> Jc(static_cast<size_t>(ns) * ns);
        clone.fill_dense_analytical_jacobian(0.0, y.data(), Jc.data());
        double clone_diff = 0.0;
        for (size_t i = 0; i < Ja.size(); ++i)
            clone_diff = std::max(clone_diff, std::abs(Ja[i] - Jc[i]));
        CHECK(clone_diff < 1e-12, "Clone reproduces the per-observable Jacobian");
    }

    return 0;
}

// ─── test_jacobian_strategy — Jacobian strategy flag (Session 20) ────────────
//
// Tests the jacobian="auto"|"analytical"|"fd" option on SolverOptions.

int test_jacobian_strategy() {
    bngsim::TimeSpec times;
    times.t_start = 0.0;
    times.t_end = 10.0;
    times.n_points = 11;

    // 1. Default is "auto"
    {
        bngsim::SolverOptions opts;
        CHECK(opts.jacobian == "auto", "Default jacobian is 'auto'");
    }

    // 2. jacobian="fd" forces CVODE DQ (baseline) — should work on any model
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        bngsim::CvodeSimulator sim(model);

        bngsim::SolverOptions opts;
        opts.jacobian = "fd";
        auto result = sim.run(times, opts);
        CHECK(result.solver_stats().n_steps > 0, "DQ Jacobian ran successfully");
    }

    // 3. jacobian="analytical" works on all-Elementary model
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        const auto& ajd = model.analytical_jacobian();
        CHECK(ajd.available, "Analytical Jac available for Elementary model");

        bngsim::CvodeSimulator sim(model);

        bngsim::SolverOptions opts;
        opts.jacobian = "analytical";
        auto result = sim.run(times, opts);
        CHECK(result.solver_stats().n_steps > 0, "Analytical Jacobian ran successfully");
    }

    // 4. jacobian="analytical" and jacobian="fd" give same trajectory
    {
        auto model_aj = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        auto model_dq = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));

        bngsim::SolverOptions opts_aj;
        opts_aj.jacobian = "analytical";
        bngsim::SolverOptions opts_dq;
        opts_dq.jacobian = "fd";

        bngsim::CvodeSimulator sim_aj(model_aj);
        auto result_aj = sim_aj.run(times, opts_aj);

        bngsim::CvodeSimulator sim_dq(model_dq);
        auto result_dq = sim_dq.run(times, opts_dq);

        // Compare final species values — should match to ~1e-6
        const auto& sp_aj = result_aj.species_data();
        const auto& sp_dq = result_dq.species_data();
        int ns = result_aj.n_species();
        int nt = result_aj.n_times();
        double max_diff = 0.0;
        for (int i = 0; i < nt * ns; ++i) {
            double diff = std::abs(sp_aj[i] - sp_dq[i]);
            double scale = std::max(1.0, std::abs(sp_dq[i]));
            max_diff = std::max(max_diff, diff / scale);
        }
        CHECK(max_diff < 1e-6, "Analytical vs DQ trajectories match (max_rel_diff=" +
              std::to_string(max_diff) + ")");
    }

    // 5. jacobian="invalid" throws
    {
        auto model = bngsim::NetworkModel::from_net(data_path("two_species_reversible.net"));
        bngsim::CvodeSimulator sim(model);

        bngsim::SolverOptions opts;
        opts.jacobian = "invalid_value";
        bool threw = false;
        try {
            sim.run(times, opts);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "Invalid jacobian option throws runtime_error");
    }

    // 6. jacobian="analytical" on a Functional model WITHOUT attached terms throws
    {
        // func_composition.net has Functional rate laws; the symbolic Functional
        // Jacobian terms are not attached in this pure-C++ path, so the analytical
        // Jacobian is not COMPLETE and an explicit jacobian="analytical" request
        // must fail fast (GH #76).
        auto model = bngsim::NetworkModel::from_net(data_path("func_composition.net"));
        CHECK(!model.analytical_jacobian_complete(),
              "Analytical Jac NOT complete for Functional model without attached terms");

        bngsim::CvodeSimulator sim(model);

        bngsim::SolverOptions opts;
        opts.jacobian = "analytical";
        bool threw = false;
        try {
            sim.run(times, opts);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "jacobian='analytical' on Functional model throws");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session 66: Event support — bolus dose via ModelBuilder + CVODE rootfinding
// ═══════════════════════════════════════════════════════════════════════════════

#include <bngsim/model_builder.hpp>

int test_event_bolus_dose() {
    // Model: S decays exponentially, dS/dt = -k*S, S(0) = 100, k = 0.1
    // Event: at time() >= 10, set S = S + 50  (bolus dose)
    //
    // Expected behavior:
    //   t < 10:  S(t) = 100*exp(-0.1*t)
    //   t = 10:  S(10⁻) = 100*exp(-1) ≈ 36.788
    //            S(10⁺) = 36.788 + 50 = 86.788 (event fires)
    //   t > 10:  S(t) = 86.788*exp(-0.1*(t-10))

    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    int s_idx = b.add_species("S", 100.0);

    // dS/dt = -k*S as elementary rate law: S → ∅ with rate k
    b.add_reaction({s_idx}, {}, bngsim::RateLawType::Elementary, "k");

    // Event: trigger = "time() >= 10", assignment = S := S + 50
    // Assignment expression "S + 50" reads the species concentration
    // (registered as ExprTk variable during event compilation step 8b)
    b.add_event("dose", "time() >= 10", {{s_idx, "S + 50"}});

    auto model = b.build();

    CHECK(model.n_events() == 1, "Model should have 1 event");
    CHECK(model.n_species() == 1, "Model should have 1 species");

    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 20.0, 201};
    auto result = sim.run(times);

    CHECK(result.n_times() == 201, "Expected 201 time points");

    // Before event (t=5): S ≈ 100*exp(-0.5) ≈ 60.65
    // Find index closest to t=5
    double S_5 = result.species_data()[50 * 1 + 0];  // t=5 is at index 50
    double S_5_expected = 100.0 * std::exp(-0.1 * 5.0);
    CHECK_CLOSE(S_5, S_5_expected, 0.5,
                "S(5) before event should be ~60.65");

    // After event (t=10.1): should be ~86.788 * exp(-0.01) ≈ 85.9
    // More precisely, check that S increased around t=10
    // Find S just before and just after t=10
    double S_before_event = -1, S_after_event = -1;
    for (int i = 0; i < result.n_times() - 1; ++i) {
        double t = result.time()[i];
        double t_next = result.time()[i + 1];
        if (t < 10.0 && t_next >= 10.0) {
            S_before_event = result.species_data()[i * 1 + 0];
            S_after_event = result.species_data()[(i + 1) * 1 + 0];
            break;
        }
    }

    CHECK(S_before_event > 0, "Found S before event");
    CHECK(S_after_event > 0, "Found S after event");

    // S should jump UP after the event (bolus adds 50)
    // Before: ~36.8, After: should be near 86.8 (then decays slightly to output point)
    CHECK(S_after_event > S_before_event,
          "S should increase after event (bolus dose)");
    CHECK(S_after_event > 70.0,
          "S after event should be > 70 (bolus adds 50 to ~37)");

    // At t=20: S ≈ 86.788 * exp(-0.1*10) ≈ 86.788 * 0.3679 ≈ 31.93
    double S_20 = result.species_data()[(result.n_times() - 1) * 1 + 0];
    double S_post_event = 100.0 * std::exp(-1.0) + 50.0;  // value right after event
    double S_20_expected = S_post_event * std::exp(-0.1 * 10.0);
    CHECK_CLOSE(S_20, S_20_expected, 1.0,
                "S(20) should be ~31.9 (decayed from bolus)");

    // Key test: S(20) should be MUCH higher than without event
    // Without event: S(20) = 100*exp(-2) ≈ 13.5
    double S_20_no_event = 100.0 * std::exp(-2.0);
    CHECK(S_20 > S_20_no_event * 1.5,
          "S(20) with event should be much higher than without event");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session 65: Model validation on construction (P9)
// ═══════════════════════════════════════════════════════════════════════════════

int test_validate_duplicate_species() {
    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    b.add_species("A", 10.0);
    b.add_species("A", 20.0);  // duplicate!
    b.add_reaction({0}, {1}, bngsim::RateLawType::Elementary, "k");
    bool threw = false;
    try { b.build(); } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        threw = msg.find("duplicate species") != std::string::npos;
    }
    CHECK(threw, "Duplicate species name should throw with 'duplicate species'");
    return 0;
}

int test_validate_duplicate_params() {
    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    b.add_parameter("k", 0.2);  // duplicate!
    b.add_species("A", 10.0);
    b.add_species("B", 0.0);
    b.add_reaction({0}, {1}, bngsim::RateLawType::Elementary, "k");
    bool threw = false;
    try { b.build(); } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        threw = msg.find("duplicate parameter") != std::string::npos;
    }
    CHECK(threw, "Duplicate parameter name should throw with 'duplicate parameter'");
    return 0;
}

int test_validate_bad_reactant_index() {
    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    b.add_species("A", 10.0);
    // reactant index 5 is out of range (only 1 species, 0-based → 1-based = 6)
    b.add_reaction({5}, {0}, bngsim::RateLawType::Elementary, "k");
    bool threw = false;
    try { b.build(); } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        threw = msg.find("reactant species index") != std::string::npos;
    }
    CHECK(threw, "Out-of-range reactant index should throw");
    return 0;
}

int test_validate_bad_product_index() {
    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    b.add_species("A", 10.0);
    // product index 3 is out of range (only 1 species)
    b.add_reaction({0}, {3}, bngsim::RateLawType::Elementary, "k");
    bool threw = false;
    try { b.build(); } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        threw = msg.find("product species index") != std::string::npos;
    }
    CHECK(threw, "Out-of-range product index should throw");
    return 0;
}

int test_validate_bad_observable_index() {
    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    b.add_species("A", 10.0);
    b.add_species("B", 0.0);
    // Observable references species index 5 (0-based → 1-based = 6), out of range
    b.add_observable("bad_obs", {{5, 1.0}});
    b.add_reaction({0}, {1}, bngsim::RateLawType::Elementary, "k");
    bool threw = false;
    try { b.build(); } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        threw = msg.find("observable") != std::string::npos &&
                msg.find("out of range") != std::string::npos;
    }
    CHECK(threw, "Out-of-range observable species index should throw");
    return 0;
}

int test_validate_unknown_elementary_param() {
    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    b.add_species("A", 10.0);
    b.add_species("B", 0.0);
    // References nonexistent parameter "k_missing"
    b.add_reaction({0}, {1}, bngsim::RateLawType::Elementary, "k_missing");
    bool threw = false;
    try { b.build(); } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        threw = msg.find("Elementary") != std::string::npos &&
                msg.find("unknown parameter") != std::string::npos;
    }
    CHECK(threw, "Unknown Elementary rate param should throw");
    return 0;
}

int test_validate_unknown_functional_func() {
    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    b.add_species("A", 10.0);
    b.add_species("B", 0.0);
    // References nonexistent function "ghost_func"
    b.add_reaction({0}, {1}, bngsim::RateLawType::Functional, "ghost_func");
    bool threw = false;
    try { b.build(); } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        threw = msg.find("Functional") != std::string::npos &&
                msg.find("unknown function") != std::string::npos;
    }
    CHECK(threw, "Unknown Functional rate function should throw");
    return 0;
}

int test_validate_valid_model_ok() {
    // A well-formed model should build successfully
    bngsim::ModelBuilder b;
    b.add_parameter("k", 0.1);
    int a = b.add_species("A", 100.0);
    int bb = b.add_species("B", 0.0);
    b.add_observable("A_tot", {{a, 1.0}});
    b.add_reaction({a}, {bb}, bngsim::RateLawType::Elementary, "k");
    auto model = b.build();
    CHECK(model.n_species() == 2, "Valid model has 2 species");
    CHECK(model.n_reactions() == 1, "Valid model has 1 reaction");
    CHECK(model.n_observables() == 1, "Valid model has 1 observable");

    // Simulate to confirm it works end-to-end
    bngsim::CvodeSimulator sim(model);
    bngsim::TimeSpec times{0.0, 10.0, 11};
    auto result = sim.run(times);
    CHECK(result.n_times() == 11, "Valid model simulates OK");
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Issue #42: mratio() must use the modified-Lentz CF so that large-|z|
// Kummer ratios stop overflowing to nan. The previous direct power-series
// summation produced inf/inf=nan for the BNG2.pl-supported parameter range
// hit by test_Mratio_1.bngl (a=-1000, b=9001, z=-10000).
// ═══════════════════════════════════════════════════════════════════════════════

int test_mratio_lentz_overflow_regime() {
    // Helper: evaluate mratio(a,b,z) through ExprTk to exercise the same
    // code path that .net rate laws / parameter expressions use.
    auto eval_mratio = [](double a, double b, double z) {
        bngsim::ExprTkEvaluator ev;
        ev.define_constant("a", a);
        ev.define_constant("b", b);
        ev.define_constant("z", z);
        int id = ev.compile("mratio(a, b, z)");
        return ev.evaluate(id);
    };

    // 1. Issue #42 reference: BNG2.pl's modified-Lentz CF gives
    //    mratio(-1000, 9001, -10000) = 0.46128328365229 (matches
    //    test_Mratio_1_ode.gdat to every printed digit). The naive
    //    series overflowed to nan.
    {
        const double r = eval_mratio(-1000.0, 9001.0, -10000.0);
        CHECK(std::isfinite(r), "issue #42: mratio(-1000,9001,-10000) finite");
        CHECK_CLOSE(r, 0.46128328365229, 1e-12,
                    "issue #42: mratio(-1000,9001,-10000)");
    }

    // 2. Small-|z| regression: the previous naive series handled this fine;
    //    Lentz must also. With a a negative integer the series for M
    //    terminates, so the ratio is exact rational.
    //    M(-2,5,-0.5) = 1 + 1/5 + 1/120  = 145/120 = 1.20833333...
    //    M(-1,6,-0.5) = 1 + 1/12         = 13/12  = 1.08333333...
    //    ratio       = (13/12)/(145/120) = 26/29   = 0.89655172413...
    {
        const double r = eval_mratio(-2.0, 5.0, -0.5);
        CHECK_CLOSE(r, 26.0 / 29.0, 1e-14, "mratio(-2,5,-0.5) = 26/29");
    }

    // 3. z = 0: M(*,*,0) = 1 identically, so the ratio is exactly 1.
    {
        const double r = eval_mratio(-7.0, 3.0, 0.0);
        CHECK_CLOSE(r, 1.0, 1e-14, "mratio(a,b,0) = 1");
    }

    // 4. test_Mratio_1 downstream parameters (BNGL: U1_U0, C_mean, C_sdev).
    //    Reproduces test_Mratio_1_ode.gdat from BNG2.pl bit-for-bit per #42.
    {
        const double a = -1000.0, b = 9001.0, z = -10000.0;
        const double mr0 = eval_mratio(a, b, z);
        const double mr1 = eval_mratio(a + 1.0, b + 1.0, z);

        const double U1_U0 = (-1.0 / b) * mr0;
        const double U2_U1 = (-1.0 / (b + 1.0)) * mr1;
        const double C_mean = -a * (1.0 - z * U1_U0);
        const double C_sdev = std::sqrt(a * (a + 1.0) * z * z * U2_U1 * U1_U0
                                        - a * z * U1_U0 * (1.0 + a * z * U1_U0));

        CHECK_CLOSE(U1_U0, -5.124800396093e-05, 1e-15, "test_Mratio_1 U1_U0");
        CHECK_CLOSE(C_mean, 487.51996039074, 1e-9, "test_Mratio_1 C_mean");
        CHECK_CLOSE(C_sdev, 15.60309027589, 1e-9, "test_Mratio_1 C_sdev");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// #64 parity: a declared model symbol that collides with an ExprTk reserved
// name (e.g. `frac`) and is *also* used in call form is ambiguous in a single
// flat namespace. The NFsim mu::Parser shim raises a clear error for this; the
// ODE/RuleMonkey ExprTkEvaluator must do the same (same message), instead of
// rewriting to r_<name> and letting ExprTk emit a cryptic "not a function".
// ═══════════════════════════════════════════════════════════════════════════════

int test_reserved_call_form_ambiguity() {
    // (1) Operator form: a declared parameter `frac` shadows the built-in.
    {
        bngsim::ExprTkEvaluator ev;
        double x = 10.0;
        ev.define_constant("frac", 0.3);
        ev.define_variable("x", &x);
        int id = ev.compile("frac * x");
        CHECK_CLOSE(ev.evaluate(id), 3.0, 1e-12, "declared `frac` operator form");
    }

    // (2) Genuine built-in still resolves when no symbol shadows it:
    //     ExprTk frac(2.5) = fractional part = 0.5.
    {
        bngsim::ExprTkEvaluator ev;
        double x = 2.5;
        ev.define_variable("x", &x);
        int id = ev.compile("frac(x)");
        CHECK_CLOSE(ev.evaluate(id), 0.5, 1e-12, "built-in frac(2.5) unshadowed");
    }

    // (3) Ambiguous declare-and-call → clear, deterministic error (parity with
    //     the NFsim mu::Parser shim, same message).
    {
        bngsim::ExprTkEvaluator ev;
        ev.define_constant("frac", 0.3);
        bool threw = false;
        std::string msg;
        try {
            ev.compile("frac(2)");
        } catch (const std::exception &e) {
            threw = true;
            msg = e.what();
        }
        CHECK(threw, "declared `frac` used as `frac(...)` raises");
        CHECK(msg.find("declared model symbol") != std::string::npos &&
                  msg.find("frac") != std::string::npos,
              "ambiguous-call message is the clear #64 form");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// #90: a user parameter named literally `u_h` aliased the registration key that
// bngsim's built-in Planck constant `_h` occupies after the unconditional
// "_X" → "u_X" remap, so define_variable("u_h") failed with a duplicate-name
// error. The fix reserves the `u_X` constant keys so such a parameter takes the
// transparent `r_<name>` mangling path. The user variable and the built-in must
// then coexist: `u_h` resolves to the user value, `_h` to Planck's constant.
// ═══════════════════════════════════════════════════════════════════════════════

int test_u_constant_key_collision() {
    bngsim::ExprTkEvaluator ev;
    double u_h = 0.5;
    ev.define_variable("u_h", &u_h); // must not throw (pre-fix: duplicate name)

    // (1) `u_h` resolves to the user variable, not Planck's constant.
    {
        int id = ev.compile("u_h * 2");
        CHECK_CLOSE(ev.evaluate(id), 1.0, 1e-12, "user `u_h` resolves to its own value");
    }

    // (2) The built-in `_h` (Planck) still resolves via its `u_h` key — the
    //     two share the source `u_h` namespace but the user var is mangled away.
    {
        int id = ev.compile("_h");
        CHECK_CLOSE(ev.evaluate(id), 6.62607015e-34, 1e-44, "built-in `_h` unshadowed by user `u_h`");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Issue #42 follow-up: surface a one-time warning when any custom function
// registered with the expression evaluator returns nan/inf. The symptom in
// #42 was silent nan propagation; without this diagnostic the next mratio-
// shaped bug would slip past code review again.
// ═══════════════════════════════════════════════════════════════════════════════

int test_nonfinite_return_warning() {
    // Redirect std::cerr to a stringstream so the warning is observable
    // from the test driver.
    std::stringstream captured;
    std::streambuf *old_cerr = std::cerr.rdbuf(captured.rdbuf());
    struct CerrRestorer {
        std::streambuf *buf;
        ~CerrRestorer() {
            if (buf) std::cerr.rdbuf(buf);
        }
    } restorer{old_cerr};

    // User-registered 1-arg function returning nan for x<0, +inf for x>=2,
    // finite otherwise. Drives four call sites:
    //   bad(-1.0)  → nan      (first time: warns)
    //   bad(-1.0)  → nan      (dedup'd: silent — same args)
    //   bad(2.0)   → +inf     (new args: warns)
    //   bad(0.5)   → 0.5      (finite: never warns)
    // Total expected warning lines: 2.
    bngsim::ExprTkEvaluator ev;
    double x = 0.0;
    ev.define_variable("x", &x);
    ev.define_function("bad", bngsim::ExpressionEvaluator::Func1([](double v) -> double {
        if (v < 0.0) return std::nan("");
        if (v >= 2.0) return std::numeric_limits<double>::infinity();
        return v;
    }));
    int id = ev.compile("bad(x)");

    x = -1.0;
    double r1 = ev.evaluate(id);
    CHECK(std::isnan(r1), "bad(-1) returns nan");

    x = -1.0;
    double r2 = ev.evaluate(id);
    CHECK(std::isnan(r2), "bad(-1) again returns nan (dedup test)");

    x = 2.0;
    double r3 = ev.evaluate(id);
    CHECK(std::isinf(r3), "bad(2) returns +inf");

    x = 0.5;
    double r4 = ev.evaluate(id);
    CHECK_CLOSE(r4, 0.5, 1e-15, "bad(0.5) returns 0.5");

    const std::string msg = captured.str();

    // Restore cerr now so subsequent CHECK failures land on the terminal.
    std::cerr.rdbuf(old_cerr);
    restorer.buf = nullptr;

    std::size_t line_count = 0;
    for (char c : msg) {
        if (c == '\n') ++line_count;
    }
    CHECK(line_count == 2,
          "expected 2 warning lines, got " + std::to_string(line_count) + ": " + msg);

    CHECK(msg.find("'bad(-1)'") != std::string::npos,
          "warning should identify the nan call: " + msg);
    CHECK(msg.find("'bad(2)'") != std::string::npos,
          "warning should identify the inf call: " + msg);
    CHECK(msg.find("nan") != std::string::npos, "warning should mention nan");
    CHECK(msg.find("inf") != std::string::npos, "warning should mention inf");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "bngsim Phase A tests" << std::endl;
    std::cout << "====================" << std::endl;

    RUN_TEST(test_load_simple_decay);
    RUN_TEST(test_ode_simple_decay);
    RUN_TEST(test_param_update);
    RUN_TEST(test_reset);
    RUN_TEST(test_ode_reversible);
    RUN_TEST(test_ode_steady_state_early_stop);
    RUN_TEST(test_find_steady_state_methods);
    RUN_TEST(test_ssa_simple_decay);
    RUN_TEST(test_ssa_reproducibility);
    RUN_TEST(test_ssa_fractional_initial_population_rounds);
    RUN_TEST(test_observables);
    RUN_TEST(test_c_api);
    RUN_TEST(test_reserved_names);
    RUN_TEST(test_file_export);
    RUN_TEST(test_gdat_function_columns);

    // Regression tests for code review fixes
    RUN_TEST(test_fix1_time_in_functions);
    RUN_TEST(test_fix2_ssa_homodimer);
    RUN_TEST(test_fix3_ssa_fixed_species);
    RUN_TEST(test_issue41_compartmental_fixed_species);
    RUN_TEST(test_fix4_c_api_null_safety);
    RUN_TEST(test_fix5_expr_param_species);

    // Regression: set_param on ConstantExpression
    RUN_TEST(test_fix6_set_param_const_expr);

    // Higher-order SSA propensity tests
    RUN_TEST(test_ssa_aab_propensity);
    RUN_TEST(test_ssa_aaa_propensity);
    RUN_TEST(test_ssa_abc_propensity);

    // Rate law changes: deprecated Sat/Hill auto-rewrite to Functional, MM tQSSA
    RUN_TEST(test_sat_loads_with_rewrite_warning);
    RUN_TEST(test_hill_loads_with_rewrite_warning);
    RUN_TEST(test_mm_tqssa);

    // Table functions (ADR-001 §3.12)
    RUN_TEST(test_tfun_time_indexed);
    RUN_TEST(test_tfun_param_indexed);
    RUN_TEST(test_tfun_from_arrays);
    RUN_TEST(test_tfun_clone);

    // Function composition + && operator + empty groups (Session 12 fixes)
    RUN_TEST(test_func_composition);

    // ExprTk case-sensitivity fix (Session 16, Kholodenko_2000 bug)
    RUN_TEST(test_case_sensitivity);

    // Graph coloring for sparse Jacobian (Session 18)
    RUN_TEST(test_graph_coloring);

    // Analytical Jacobian for mass-action models (Session 19)
    RUN_TEST(test_analytical_jacobian);

    // Jacobian strategy flag (Session 20)
    RUN_TEST(test_jacobian_strategy);

    // Event support (Session 66, P0)
    RUN_TEST(test_event_bolus_dose);

    // Issue #42: mratio() Lentz CF (no nan on large |z|)
    RUN_TEST(test_mratio_lentz_overflow_regime);
    RUN_TEST(test_reserved_call_form_ambiguity);
    RUN_TEST(test_u_constant_key_collision);

    // Issue #42 follow-up: nan/inf return diagnostic for custom functions
    RUN_TEST(test_nonfinite_return_warning);

    // Model validation on construction (Session 65, P9)
    RUN_TEST(test_validate_duplicate_species);
    RUN_TEST(test_validate_duplicate_params);
    RUN_TEST(test_validate_bad_reactant_index);
    RUN_TEST(test_validate_bad_product_index);
    RUN_TEST(test_validate_bad_observable_index);
    RUN_TEST(test_validate_unknown_elementary_param);
    RUN_TEST(test_validate_unknown_functional_func);
    RUN_TEST(test_validate_valid_model_ok);

    std::cout << std::endl;
    std::cout << tests_passed << "/" << tests_run << " tests passed." << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
