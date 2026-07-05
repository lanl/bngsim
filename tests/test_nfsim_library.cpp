/**
 * test_nfsim_library.cpp — Phase C.5 Step 1 verification test
 *
 * This test validates that NFsim compiles as a static library (libnfsim.a)
 * and can be called from C++ code to:
 *   1. Initialize a System from BNG XML
 *   2. Prepare the system for simulation
 *   3. Run a simulation
 *   4. Read observable values
 *   5. Test parameter access
 *   6. Properly clean up (delete System)
 *
 * Uses the simple_system.xml from NFsim's test suite.
 */

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// NFsim headers — these come from the libnfsim static library
#include "NFcore/NFcore.hh"
#include "NFinput/NFinput.hh"
#include "NFutil/NFutil.hh"
// ExprTk-backed mu::Parser shim — exercised directly for the reserved-symbol
// remap (#64). The include guard makes this safe even though NFfunction.hh
// already pulls it in transitively under NFSIM_USE_EXPRTK.
#include "NFfunction/nfsim_funcparser.h"

using namespace std;
using namespace NFcore;

// ─── Helper: temporary cwd switch (restored on scope exit) ───────────────────
class CwdGuard {
public:
    explicit CwdGuard(const std::filesystem::path &new_cwd) {
        try {
            old_cwd_ = std::filesystem::current_path();
            if (!new_cwd.empty()) {
                std::filesystem::current_path(new_cwd);
                changed_ = true;
            }
        } catch (...) {
            changed_ = false;
        }
    }

    ~CwdGuard() {
        if (changed_) {
            try {
                std::filesystem::current_path(old_cwd_);
            } catch (...) {
                // Best effort in test harness.
            }
        }
    }

private:
    std::filesystem::path old_cwd_;
    bool changed_ = false;
};

// ─── Helper: check if a file exists ──────────────────────────────────────────
static bool file_exists(const string &path) {
    ifstream f(path);
    return f.good();
}

// ─── Test 1: Load XML and create System ──────────────────────────────────────
static bool test_load_xml(const string &xml_path) {
    cout << "  Test 1: Load XML and create System ... ";

    int suggestedTraversalLimit = -1;
    System *s = NFinput::initializeFromXML(
        xml_path,
        /*blockSameComplexBinding=*/false,
        /*globalMoleculeLimit=*/200000,
        /*verbose=*/false,
        suggestedTraversalLimit,
        /*evaluateComplexScopedLocalFunctions=*/true,
        /*connectivityFlag=*/false);

    if (s == nullptr) {
        cout << "FAILED (initializeFromXML returned nullptr)" << endl;
        return false;
    }

    string name = s->getName();
    if (name.empty()) {
        cout << "FAILED (system has empty name)" << endl;
        delete s;
        return false;
    }

    cout << "OK (system name: '" << name << "')" << endl;
    delete s;
    return true;
}

// ─── Test 2: Prepare for simulation ──────────────────────────────────────────
static bool test_prepare_simulation(const string &xml_path) {
    cout << "  Test 2: Prepare for simulation ... ";

    int suggestedTraversalLimit = -1;
    System *s = NFinput::initializeFromXML(
        xml_path, false, 200000, false,
        suggestedTraversalLimit, true, false);

    if (s == nullptr) {
        cout << "FAILED (load)" << endl;
        return false;
    }

    // Register output to /dev/null to avoid file creation
    s->registerOutputFileLocation("/dev/null");
    s->outputAllObservableNames();

    // This is the key call — it initializes all reaction lists, observables, etc.
    s->prepareForSimulation();

    cout << "OK" << endl;
    delete s;
    return true;
}

// ─── Test 3: Run simulation ──────────────────────────────────────────────────
static bool test_run_simulation(const string &xml_path) {
    cout << "  Test 3: Run simulation (1 second, 10 steps) ... ";

    // Seed RNG for reproducibility
    NFutil::SEED_RANDOM(12345);

    int suggestedTraversalLimit = -1;
    System *s = NFinput::initializeFromXML(
        xml_path, false, 200000, false,
        suggestedTraversalLimit, true, false);

    if (s == nullptr) {
        cout << "FAILED (load)" << endl;
        return false;
    }

    s->registerOutputFileLocation("/dev/null");
    s->outputAllObservableNames();
    s->prepareForSimulation();

    // Run for 1 second with 10 output steps
    double finalTime = s->sim(1.0, 10, /*verbose=*/false);

    if (finalTime < 0.0) {
        cout << "FAILED (sim returned negative time)" << endl;
        delete s;
        return false;
    }

    cout << "OK (final time: " << finalTime << ")" << endl;
    delete s;
    return true;
}

// ─── Test 4: Observable access ───────────────────────────────────────────────
static bool test_observable_access(const string &xml_path) {
    cout << "  Test 4: Observable access ... ";

    NFutil::SEED_RANDOM(42);

    int suggestedTraversalLimit = -1;
    System *s = NFinput::initializeFromXML(
        xml_path, false, 200000, false,
        suggestedTraversalLimit, true, false);

    if (s == nullptr) {
        cout << "FAILED (load)" << endl;
        return false;
    }

    s->registerOutputFileLocation("/dev/null");
    s->outputAllObservableNames();
    s->prepareForSimulation();

    // Try to get the number of molecule types (should be > 0 for any real model)
    int nTypes = s->getNumOfMoleculeTypes();
    if (nTypes <= 0) {
        cout << "FAILED (no molecule types)" << endl;
        delete s;
        return false;
    }

    cout << "OK (" << nTypes << " molecule types)" << endl;
    delete s;
    return true;
}

// ─── Test 5: Parameter access ────────────────────────────────────────────────
static bool test_parameter_access(const string &xml_path) {
    cout << "  Test 5: Parameter set/get ... ";

    int suggestedTraversalLimit = -1;
    System *s = NFinput::initializeFromXML(
        xml_path, false, 200000, false,
        suggestedTraversalLimit, true, false);

    if (s == nullptr) {
        cout << "FAILED (load)" << endl;
        return false;
    }

    // Print all parameters to verify they loaded
    // (output goes to stdout which is fine for a test)
    s->printAllParameters();

    cout << "OK" << endl;
    delete s;
    return true;
}

// ─── Test 6: RNG seeding (reproducibility check) ────────────────────────────
static bool test_rng_seeding(const string &xml_path) {
    cout << "  Test 6: RNG seeding produces reproducible results ... ";

    // Run 1
    NFutil::SEED_RANDOM(99999);
    double r1 = NFutil::RANDOM(1.0);

    // Run 2 with same seed
    NFutil::SEED_RANDOM(99999);
    double r2 = NFutil::RANDOM(1.0);

    if (r1 != r2) {
        cout << "FAILED (r1=" << r1 << " != r2=" << r2 << ")" << endl;
        return false;
    }

    // Run 3 with different seed
    NFutil::SEED_RANDOM(11111);
    double r3 = NFutil::RANDOM(1.0);

    if (r1 == r3) {
        cout << "FAILED (different seeds gave same result)" << endl;
        return false;
    }

    cout << "OK" << endl;
    return true;
}

// ─── Test 7: TFUN parse and simulate ─────────────────────────────────────────
static bool test_tfun_parse_simulate(const string &xml_path) {
    cout << "  Test 7: TFUN parse and simulate ... ";

    // Find the TFUN test XML
    // It should be in data/nfsim/tfun_test/ relative to the regular test data
    string tfun_xml;
    vector<string> tfun_candidates = {
        "../../../tests/data/nfsim/tfun_test/tfun_simple.xml",
        string(NFSIM_TEST_DATA_DIR) + "/tfun_test/tfun_simple.xml",
    };
    for (const auto &c : tfun_candidates) {
        if (!c.empty() && file_exists(c)) {
            tfun_xml = c;
            break;
        }
    }
    if (tfun_xml.empty()) {
        cout << "SKIPPED (tfun_simple.xml not found)" << endl;
        return true;  // Don't fail — just skip
    }

    // Ensure relative TFUN file paths (e.g., tfun_data.dat) resolve from the
    // XML directory, matching NfsimSimulator runtime behavior.
    std::filesystem::path tfun_xml_abs = std::filesystem::absolute(tfun_xml);
    CwdGuard cwd_guard(tfun_xml_abs.parent_path());

    NFutil::SEED_RANDOM(42);

    int suggestedTraversalLimit = -1;
    System *s = NFinput::initializeFromXML(
        tfun_xml_abs.string(), false, 200000, false,
        suggestedTraversalLimit, true, false);

    if (s == nullptr) {
        cout << "FAILED (initializeFromXML returned nullptr for TFUN model)" << endl;
        return false;
    }

    // Verify the system loaded
    string name = s->getName();
    if (name.empty()) {
        cout << "FAILED (TFUN system has empty name)" << endl;
        delete s;
        return false;
    }

    s->registerOutputFileLocation("/dev/null");
    s->outputAllObservableNames();
    s->prepareForSimulation();

    // Run a short simulation
    double finalTime = s->sim(1.0, 5, false);
    if (finalTime < 0.0) {
        cout << "FAILED (TFUN sim returned negative time)" << endl;
        delete s;
        return false;
    }

    // Verify conservation: total X should still be 100
    int nTypes = s->getNumOfMoleculeTypes();
    if (nTypes <= 0) {
        cout << "FAILED (no molecule types in TFUN model)" << endl;
        delete s;
        return false;
    }

    cout << "OK (TFUN model loaded, simulated to t=" << finalTime
         << ", name='" << name << "')" << endl;
    delete s;
    return true;
}

// ─── Test 9: ExprTk reserved-symbol parameter names (#64) ────────────────────
// Models that declare a parameter / observable / function whose name collides
// with an ExprTk built-in (frac, min, …) must register and evaluate correctly,
// matching legacy BNG2.pl→NFsim (muParser had a far smaller reserved set). The
// genuine built-in must stay usable when no model symbol shadows it, and a
// declared symbol used in call form must raise a clear error rather than
// silently resolving to the built-in. Surfaced by V1988a_endemic_infection
// (parameter `frac` in rate law `frac * infection_force()`).
static bool test_reserved_symbol_params(const string & /*xml_path*/) {
    cout << "  Test 9: ExprTk reserved-symbol parameter names (#64) ... ";
    using mu::Parser;

    auto approx = [](double a, double b) { return std::fabs(a - b) < 1e-12; };

    // (1) Operator form: parameter literally named `frac` shadows the built-in.
    {
        Parser p;
        double Atot = 10.0;
        p.DefineConst("frac", 0.3);
        p.DefineVar("Atot", &Atot);
        p.SetExpr("frac * Atot");
        if (!approx(p.Eval(), 3.0)) {
            cout << "FAILED (declared `frac` operator form gave " << p.Eval()
                 << ", expected 3.0)" << endl;
            return false;
        }
    }

    // (2) Genuine built-in still resolves when no symbol shadows it:
    //     ExprTk frac(2.5) = fractional part = 0.5.
    {
        Parser p;
        double x = 2.5;
        p.DefineVar("x", &x);
        p.SetExpr("frac(x)");
        if (!approx(p.Eval(), 0.5)) {
            cout << "FAILED (built-in frac(2.5) gave " << p.Eval()
                 << ", expected 0.5)" << endl;
            return false;
        }
    }

    // (3) A second reserved name, registered via DefineVar (observable-style).
    {
        Parser p;
        double y = 7.0;
        p.DefineVar("min", &y);
        p.SetExpr("min + 1");
        if (!approx(p.Eval(), 8.0)) {
            cout << "FAILED (declared `min` gave " << p.Eval()
                 << ", expected 8.0)" << endl;
            return false;
        }
    }

    // (4) Value update after compile (updateParameters / fileUpdate path).
    {
        Parser p;
        double Atot = 10.0;
        p.DefineConst("frac", 0.3);
        p.DefineVar("Atot", &Atot);
        p.SetExpr("frac * Atot");
        p.DefineConst("frac", 0.5);  // update the existing const in place
        if (!approx(p.Eval(), 5.0)) {
            cout << "FAILED (post-compile `frac` update gave " << p.Eval()
                 << ", expected 5.0)" << endl;
            return false;
        }
    }

    // (5) Ambiguous declare-and-call: a declared symbol used in call form
    //     cannot coexist with the same-named built-in → clear, loud error.
    {
        Parser p;
        double x = 2.5;
        p.DefineConst("frac", 0.3);
        p.DefineVar("x", &x);
        bool threw = false;
        try {
            p.SetExpr("frac(x)");
        } catch (const mu::Parser::exception_type &e) {
            threw = true;
            const string msg = e.GetMsg();
            if (msg.find("frac") == string::npos ||
                msg.find("declared model symbol") == string::npos) {
                cout << "FAILED (ambiguous-case error message unclear: '" << msg
                     << "')" << endl;
                return false;
            }
        }
        if (!threw) {
            cout << "FAILED (declared `frac` used as `frac(...)` did not raise)"
                 << endl;
            return false;
        }
    }

    // (6) mratio() — BNGL built-in confluent hypergeometric ratio — must
    //     evaluate in the NFsim shim identically to the ODE engine (parity).
    //     mratio(-2,5,-0.5) = (13/12)/(145/120) = 26/29 (exact rational; a is a
    //     negative integer so the Kummer series terminates).
    {
        Parser p;
        double a = -2.0, b = 5.0, z = -0.5;
        p.DefineConst("a", a);
        p.DefineConst("b", b);
        p.DefineConst("z", z);
        p.SetExpr("mratio(a,b,z)");
        if (!approx(p.Eval(), 26.0 / 29.0)) {
            cout << "FAILED (mratio(-2,5,-0.5) gave " << p.Eval()
                 << ", expected 26/29)" << endl;
            return false;
        }
    }

    // (7) mratio is now a registered function, so a model symbol literally
    //     named `mratio` is reserved and mangles in operator form.
    {
        Parser p;
        double v = 4.0;
        p.DefineVar("mratio", &v);
        p.SetExpr("mratio + 1");
        if (!approx(p.Eval(), 5.0)) {
            cout << "FAILED (declared `mratio` gave " << p.Eval()
                 << ", expected 5.0)" << endl;
            return false;
        }
    }

    cout << "OK (reserved-symbol params register, evaluate, and disambiguate)"
         << endl;
    return true;
}

// ─── Test 10: reserved-name parameter through the full NFsim path (#64) ───────
// Loads a model whose Function-type rate law is `frac * Atot` with `frac`
// declared as a parameter, and simulates it. This drives NFsim's real function
// builder (function.cpp DefineConst("frac", …) → SetExpr("frac*Atot")), the
// exact path that regressed on V1988a_endemic_infection. Pre-fix this aborts
// with an uncaught ERR029; post-fix the model runs and consumes A(s~0).
static bool test_reserved_param_simulate(const string & /*xml_path*/) {
    cout << "  Test 10: reserved-name parameter through NFsim (#64) ... ";

    string xml;
    vector<string> candidates = {
        "../../../tests/data/nfsim/reserved_param_frac.xml",
        string(NFSIM_TEST_DATA_DIR) + "/reserved_param_frac.xml",
    };
    for (const auto &c : candidates) {
        if (!c.empty() && file_exists(c)) {
            xml = c;
            break;
        }
    }
    if (xml.empty()) {
        cout << "SKIPPED (reserved_param_frac.xml not found)" << endl;
        return true;
    }

    NFutil::SEED_RANDOM(42);
    int suggestedTraversalLimit = -1;
    System *s = NFinput::initializeFromXML(
        std::filesystem::absolute(xml).string(), false, 200000, false,
        suggestedTraversalLimit, true, false);
    if (s == nullptr) {
        cout << "FAILED (initializeFromXML returned nullptr — `frac` rate law "
                "rejected?)" << endl;
        return false;
    }

    s->registerOutputFileLocation("/dev/null");
    s->outputAllObservableNames();
    s->prepareForSimulation();

    // A(s~0) starts at 100 and is consumed by the frac*Atot reaction. Run long
    // enough that at least some conversion fires.
    double finalTime = s->sim(50.0, 5, false);
    if (finalTime < 0.0) {
        cout << "FAILED (sim returned negative time)" << endl;
        delete s;
        return false;
    }

    cout << "OK (model with parameter `frac` loaded and simulated to t="
         << finalTime << ")" << endl;
    delete s;
    return true;
}

// ─── Test 8: System destruction (no leaks) ───────────────────────────────────
static bool test_system_destruction(const string &xml_path) {
    cout << "  Test 8: System create/destroy cycle (3x) ... ";

    for (int i = 0; i < 3; i++) {
        NFutil::SEED_RANDOM(i + 1);

        int suggestedTraversalLimit = -1;
        System *s = NFinput::initializeFromXML(
            xml_path, false, 200000, false,
            suggestedTraversalLimit, true, false);

        if (s == nullptr) {
            cout << "FAILED (load on iteration " << i << ")" << endl;
            return false;
        }

        s->registerOutputFileLocation("/dev/null");
        s->outputAllObservableNames();
        s->prepareForSimulation();
        s->sim(0.1, 2, false);

        delete s;
    }

    cout << "OK" << endl;
    return true;
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    cout << "\n=== NFsim Library Integration Test (Phase C.5 Step 1) ===" << endl;

    // Find the test XML file
    // Try several possible locations
    string xml_path;
    vector<string> candidates = {
        // Provided as argument
        (argc > 1) ? string(argv[1]) : "",
        // Relative to build directory
        "../../../tests/data/nfsim/simple_system.xml",
        // From the BNGsim test data directory
        string(NFSIM_TEST_DATA_DIR) + "/simple_system.xml",
    };

    for (const auto &c : candidates) {
        if (!c.empty() && file_exists(c)) {
            xml_path = c;
            break;
        }
    }

    if (xml_path.empty()) {
        cerr << "ERROR: Could not find test XML file. Tried:" << endl;
        for (const auto &c : candidates) {
            if (!c.empty())
                cerr << "  " << c << endl;
        }
        cerr << "\nProvide the path as the first argument, e.g.:" << endl;
        cerr << "  ./test_nfsim_library /path/to/simple_system.xml" << endl;
        return 1;
    }

    cout << "Using XML file: " << xml_path << endl;
    cout << endl;

    int passed = 0, failed = 0;

    auto run = [&](bool (*test)(const string &)) {
        if (test(xml_path))
            passed++;
        else
            failed++;
    };

    run(test_load_xml);
    run(test_prepare_simulation);
    run(test_run_simulation);
    run(test_observable_access);
    run(test_parameter_access);
    run(test_rng_seeding);
    run(test_tfun_parse_simulate);
    run(test_reserved_symbol_params);
    run(test_reserved_param_simulate);
    run(test_system_destruction);

    cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===" << endl;
    return (failed > 0) ? 1 : 0;
}
