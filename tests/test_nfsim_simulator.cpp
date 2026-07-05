/**
 * test_nfsim_simulator.cpp — Phase C.5 Step 2: NfsimSimulator wrapper tests
 *
 * Tests the bngsim::NfsimSimulator C++ wrapper class which provides a clean
 * interface to NFsim via the bngsim Result type.
 *
 * Uses simple_system.xml which has:
 *   - 4 reaction rules (bind, unbind, catalyze, dephos)
 *   - 2 molecule types: X(y,p~0~1) and Y(x)
 *   - Initial: 5000 X(p~0,y), 500 Y(x)
 *   - 6 observables: X_free, X_p_total, Xp_free, XY, Ytotal, Xtotal
 */

#include <bngsim/nfsim_simulator.hpp>
#include <bngsim/result.hpp>
#include <bngsim/types.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace bngsim;
using namespace std;

// ─── Helper: find XML path ───────────────────────────────────────────────────

static string find_xml_path() {
    // Try compile-time path first
    string path = string(NFSIM_TEST_DATA_DIR) + "/simple_system.xml";
    return path;
}

// ─── Test 1: Basic run and Result shape ──────────────────────────────────────

static bool test_basic_run() {
    cout << "  Test 1: Basic run and Result shape ... ";

    NfsimSimulator sim(find_xml_path());
    TimeSpec times{0.0, 1.0, 11};  // 11 points: t=0, 0.1, ..., 1.0
    auto result = sim.run(times, /*seed=*/42);

    // Check dimensions
    if (result.n_times() != 11) {
        cout << "FAILED (n_times=" << result.n_times() << ", expected 11)" << endl;
        return false;
    }
    if (result.n_observables() != 6) {
        cout << "FAILED (n_observables=" << result.n_observables()
             << ", expected 6)" << endl;
        return false;
    }
    // NFsim doesn't produce species-level data
    if (result.n_species() != 0) {
        cout << "FAILED (n_species=" << result.n_species()
             << ", expected 0)" << endl;
        return false;
    }

    // Check time vector
    const auto& t = result.time();
    if (t.size() != 11) {
        cout << "FAILED (time vector size=" << t.size() << ")" << endl;
        return false;
    }
    if (std::abs(t[0] - 0.0) > 1e-10) {
        cout << "FAILED (t[0]=" << t[0] << ")" << endl;
        return false;
    }
    if (std::abs(t[10] - 1.0) > 1e-10) {
        cout << "FAILED (t[10]=" << t[10] << ")" << endl;
        return false;
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 2: Observable names ────────────────────────────────────────────────

static bool test_observable_names() {
    cout << "  Test 2: Observable names ... ";

    NfsimSimulator sim(find_xml_path());
    auto result = sim.run({0.0, 0.1, 2}, 42);

    const auto& names = result.observable_names();
    if (names.size() != 6) {
        cout << "FAILED (size=" << names.size() << ")" << endl;
        return false;
    }

    // Expected names from simple_system.xml
    vector<string> expected = {
        "X_free", "X_p_total", "Xp_free", "XY", "Ytotal", "Xtotal"};

    for (size_t i = 0; i < expected.size(); ++i) {
        if (names[i] != expected[i]) {
            cout << "FAILED (names[" << i << "]='" << names[i]
                 << "', expected '" << expected[i] << "')" << endl;
            return false;
        }
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 3: Observable values are non-negative ──────────────────────────────

static bool test_observables_nonnegative() {
    cout << "  Test 3: Observable values are non-negative ... ";

    NfsimSimulator sim(find_xml_path());
    auto result = sim.run({0.0, 10.0, 101}, 12345);

    const auto& data = result.observable_data();
    int n_obs = result.n_observables();

    for (int t = 0; t < result.n_times(); ++t) {
        for (int j = 0; j < n_obs; ++j) {
            double val = data[t * n_obs + j];
            if (val < 0.0) {
                cout << "FAILED (negative value " << val << " at t_idx="
                     << t << ", obs=" << j << ")" << endl;
                return false;
            }
        }
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 4: Conservation law — Xtotal and Ytotal are constant ───────────────

static bool test_conservation() {
    cout << "  Test 4: Conservation law (Xtotal=5000, Ytotal=500) ... ";

    NfsimSimulator sim(find_xml_path());
    auto result = sim.run({0.0, 10.0, 101}, 42);

    const auto& names = result.observable_names();
    const auto& data = result.observable_data();
    int n_obs = result.n_observables();

    // Find Xtotal and Ytotal indices
    int idx_xtotal = -1, idx_ytotal = -1;
    for (int j = 0; j < n_obs; ++j) {
        if (names[j] == "Xtotal") idx_xtotal = j;
        if (names[j] == "Ytotal") idx_ytotal = j;
    }

    if (idx_xtotal < 0 || idx_ytotal < 0) {
        cout << "FAILED (couldn't find Xtotal/Ytotal)" << endl;
        return false;
    }

    // Check conservation at all time points
    for (int t = 0; t < result.n_times(); ++t) {
        double xtotal = data[t * n_obs + idx_xtotal];
        double ytotal = data[t * n_obs + idx_ytotal];

        if (std::abs(xtotal - 5000.0) > 0.5) {
            cout << "FAILED (Xtotal=" << xtotal << " at t_idx=" << t
                 << ", expected 5000)" << endl;
            return false;
        }
        if (std::abs(ytotal - 500.0) > 0.5) {
            cout << "FAILED (Ytotal=" << ytotal << " at t_idx=" << t
                 << ", expected 500)" << endl;
            return false;
        }
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 5: Deterministic seeding — same seed → same result ─────────────────

static bool test_deterministic_seeding() {
    cout << "  Test 5: Deterministic seeding (same seed → same result) ... ";

    NfsimSimulator sim(find_xml_path());
    TimeSpec times{0.0, 1.0, 11};

    auto result1 = sim.run(times, 99999);
    auto result2 = sim.run(times, 99999);

    const auto& d1 = result1.observable_data();
    const auto& d2 = result2.observable_data();

    if (d1.size() != d2.size()) {
        cout << "FAILED (different data sizes)" << endl;
        return false;
    }

    for (size_t i = 0; i < d1.size(); ++i) {
        if (d1[i] != d2[i]) {
            cout << "FAILED (d1[" << i << "]=" << d1[i]
                 << " != d2[" << i << "]=" << d2[i] << ")" << endl;
            return false;
        }
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 6: Different seeds → different results ─────────────────────────────

static bool test_different_seeds() {
    cout << "  Test 6: Different seeds produce different results ... ";

    NfsimSimulator sim(find_xml_path());
    TimeSpec times{0.0, 10.0, 101};

    auto result1 = sim.run(times, 11111);
    auto result2 = sim.run(times, 22222);

    const auto& d1 = result1.observable_data();
    const auto& d2 = result2.observable_data();

    // At least some values should differ (over 101 time points,
    // with a stochastic system, this is essentially guaranteed)
    bool any_differ = false;
    for (size_t i = 0; i < d1.size(); ++i) {
        if (d1[i] != d2[i]) {
            any_differ = true;
            break;
        }
    }

    if (!any_differ) {
        cout << "FAILED (all values identical with different seeds)" << endl;
        return false;
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 7: Parameter override via set_param ────────────────────────────────

static bool test_param_override() {
    cout << "  Test 7: Parameter override via set_param ... ";

    // Run with default kon=10
    NfsimSimulator sim1(find_xml_path());
    auto result_default = sim1.run({0.0, 1.0, 11}, 42);

    // Run with kon=0 (no binding → XY should stay 0)
    NfsimSimulator sim2(find_xml_path());
    sim2.set_param("kon", 0.0);
    auto result_no_bind = sim2.run({0.0, 1.0, 11}, 42);

    const auto& names = result_no_bind.observable_names();
    const auto& data = result_no_bind.observable_data();
    int n_obs = result_no_bind.n_observables();

    // Find XY index
    int idx_xy = -1;
    for (int j = 0; j < n_obs; ++j) {
        if (names[j] == "XY") idx_xy = j;
    }

    if (idx_xy < 0) {
        cout << "FAILED (couldn't find XY observable)" << endl;
        return false;
    }

    // With kon=0, XY should remain 0 at all times
    for (int t = 0; t < result_no_bind.n_times(); ++t) {
        double xy = data[t * n_obs + idx_xy];
        if (xy > 0.5) {
            cout << "FAILED (XY=" << xy << " at t_idx=" << t
                 << " with kon=0, expected 0)" << endl;
            return false;
        }
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 8: Multiple sequential runs (clean state each time) ────────────────

static bool test_sequential_runs() {
    cout << "  Test 8: Multiple sequential runs (clean state) ... ";

    NfsimSimulator sim(find_xml_path());
    TimeSpec times{0.0, 1.0, 11};

    // Run 3 times — each should start from the same initial state
    for (int run = 0; run < 3; ++run) {
        auto result = sim.run(times, 42 + run);

        // Check initial Xtotal = 5000 (proves we re-parsed from clean state)
        const auto& names = result.observable_names();
        const auto& data = result.observable_data();
        int n_obs = result.n_observables();

        int idx_xtotal = -1;
        for (int j = 0; j < n_obs; ++j) {
            if (names[j] == "Xtotal") idx_xtotal = j;
        }

        if (idx_xtotal < 0) {
            cout << "FAILED (couldn't find Xtotal on run " << run << ")" << endl;
            return false;
        }

        double xtotal_t0 = data[0 * n_obs + idx_xtotal];
        if (std::abs(xtotal_t0 - 5000.0) > 0.5) {
            cout << "FAILED (Xtotal at t=0 is " << xtotal_t0
                 << " on run " << run << ", expected 5000)" << endl;
            return false;
        }
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 8b: save/restore concentrations round-trip (issue #52) ─────────────
//
// Regression for internal#52: resetConcentrations() used to SIGSEGV inside
// SystemSnapshot::restore() because destroyAllMolecules() freed objects still
// owned by the MoleculeList pool and deleted the Complex objects the recycled
// pool molecules referenced. This drives the session save/restore path on a
// model that forms bonded complexes (XY) and asserts no crash plus an exact
// round-trip of the saved observable state.
static bool test_save_restore_concentrations() {
    cout << "  Test 8b: save/restore concentrations round-trip ... ";

    NfsimSimulator sim(find_xml_path());
    sim.initialize(/*seed=*/42);

    if (sim.has_saved_concentrations()) {
        cout << "FAILED (has_saved_concentrations true before any save)" << endl;
        return false;
    }

    // Equilibrate so bonds (XY) and component states (Xp) populate.
    sim.simulate(0.0, 2.0, 2);
    auto names = sim.get_observable_names();
    auto saved = sim.get_observable_values();

    sim.save_concentrations();
    if (!sim.has_saved_concentrations()) {
        cout << "FAILED (has_saved_concentrations false after save)" << endl;
        return false;
    }

    // Mutate further, then rewind to the saved state.
    sim.simulate(2.0, 8.0, 2);
    sim.restore_concentrations();  // used to crash here

    auto restored = sim.get_observable_values();
    if (restored.size() != saved.size()) {
        cout << "FAILED (observable count changed across restore)" << endl;
        return false;
    }
    for (size_t i = 0; i < saved.size(); ++i) {
        if (std::abs(restored[i] - saved[i]) > 1e-9) {
            cout << "FAILED (" << names[i] << " " << restored[i]
                 << " != saved " << saved[i] << ")" << endl;
            return false;
        }
    }

    // Session must remain usable and tear down cleanly (no double free).
    sim.simulate(8.0, 10.0, 2);
    sim.destroy_session();

    cout << "OK" << endl;
    return true;
}

// ─── Test 9: Solver stats populated ──────────────────────────────────────────

static bool test_solver_stats() {
    cout << "  Test 9: Solver stats populated ... ";

    NfsimSimulator sim(find_xml_path());
    auto result = sim.run({0.0, 10.0, 101}, 42);

    int n_steps = result.solver_stats().n_steps;
    if (n_steps <= 0) {
        cout << "FAILED (n_steps=" << n_steps << ", expected > 0)" << endl;
        return false;
    }

    cout << "OK (n_steps=" << n_steps << ")" << endl;
    return true;
}

// ─── Test 10: System dynamics — X_p_total increases over time ────────────────

static bool test_dynamics() {
    cout << "  Test 10: Dynamics — X_p_total increases over time ... ";

    NfsimSimulator sim(find_xml_path());
    auto result = sim.run({0.0, 10.0, 101}, 42);

    const auto& names = result.observable_names();
    const auto& data = result.observable_data();
    int n_obs = result.n_observables();

    int idx_xp = -1;
    for (int j = 0; j < n_obs; ++j) {
        if (names[j] == "X_p_total") idx_xp = j;
    }

    if (idx_xp < 0) {
        cout << "FAILED (couldn't find X_p_total)" << endl;
        return false;
    }

    // At t=0, X_p_total should be 0 (all X start unphosphorylated)
    double xp_t0 = data[0 * n_obs + idx_xp];
    if (xp_t0 > 0.5) {
        cout << "FAILED (X_p_total at t=0 is " << xp_t0 << ", expected 0)" << endl;
        return false;
    }

    // At t=10, X_p_total should be > 0 (some phosphorylation has occurred)
    double xp_final = data[100 * n_obs + idx_xp];
    if (xp_final < 1.0) {
        cout << "FAILED (X_p_total at t=10 is " << xp_final
             << ", expected > 0)" << endl;
        return false;
    }

    cout << "OK (X_p_total at t=10: " << xp_final << ")" << endl;
    return true;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    cout << "\n=== NfsimSimulator Wrapper Test (Phase C.5 Step 2) ===" << endl;
    cout << "Using XML: " << find_xml_path() << endl;
    cout << endl;

    int passed = 0, failed = 0;

    auto run = [&](bool (*test)()) {
        if (test())
            passed++;
        else
            failed++;
    };

    run(test_basic_run);
    run(test_observable_names);
    run(test_observables_nonnegative);
    run(test_conservation);
    run(test_deterministic_seeding);
    run(test_different_seeds);
    run(test_param_override);
    run(test_sequential_runs);
    run(test_save_restore_concentrations);
    run(test_solver_stats);
    run(test_dynamics);

    cout << "\n=== Results: " << passed << " passed, " << failed
         << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}
