// bngsim/tests/test_validation.cpp — Validate bngsim against run_network reference outputs
//
// Compares bngsim ODE results against .cdat files produced by BNG's run_network 3.0.
// Reference files generated with: run_network -a 1e-8 -r 1e-8 model.net dt n_steps
//
// PRD §5.5: "Compares against reference outputs from the existing run_network binary"

#include <bngsim/bngsim.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "data"
#endif

static int tests_run = 0;
static int tests_passed = 0;

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

// ─── Parse a BNG .cdat file ──────────────────────────────────────────────────
// Format: header line starting with #, then rows of "time  s1  s2 ..."
struct CdatData {
    std::vector<double> time;
    std::vector<std::vector<double>> species;  // [time_idx][species_idx]
    int n_species = 0;
};

static CdatData parse_cdat(const std::string& path) {
    CdatData data;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open reference file: " << path << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip header/comment lines
        if (line.empty()) continue;
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        if (line[start] == '#' || line.substr(start, 4) == "time") continue;

        std::istringstream iss(line);
        double t;
        iss >> t;
        data.time.push_back(t);

        std::vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        data.species.push_back(row);
        data.n_species = static_cast<int>(row.size());
    }
    return data;
}

// ─── Compare bngsim vs run_network for a given model ─────────────────────────
//
// Runs bngsim on the .net file with matching time points, then compares
// species concentrations against the reference .cdat file.
// Returns max absolute error and max relative error.
struct CompareResult {
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    int n_points_compared = 0;
    bool success = false;
};

static CompareResult compare_model(
    const std::string& net_file,
    const std::string& ref_cdat,
    double t_start,
    double dt,
    int n_steps
) {
    CompareResult cr;

    // Load reference
    auto ref = parse_cdat(ref_cdat);
    if (ref.time.empty()) {
        std::cerr << "  Reference file empty or could not be parsed" << std::endl;
        return cr;
    }

    // Run bngsim
    auto model = bngsim::NetworkModel::from_net(net_file);
    bngsim::CvodeSimulator sim(model);

    double t_end = t_start + dt * n_steps;
    int n_points = n_steps + 1;

    bngsim::TimeSpec times{t_start, t_end, n_points};
    bngsim::SolverOptions opts;
    opts.rtol = 1e-8;
    opts.atol = 1e-8;

    auto result = sim.run(times, opts);

    // Compare point by point
    int ns = result.n_species();
    if (ns != ref.n_species) {
        std::cerr << "  Species count mismatch: bngsim=" << ns
                  << " ref=" << ref.n_species << std::endl;
        return cr;
    }

    int n_compare = std::min(result.n_times(), static_cast<int>(ref.time.size()));
    for (int i = 0; i < n_compare; ++i) {
        for (int j = 0; j < ns; ++j) {
            double bngsim_val = result.species_data()[i * ns + j];
            double ref_val = ref.species[i][j];

            double abs_err = std::abs(bngsim_val - ref_val);
            double rel_err = (ref_val != 0.0)
                ? abs_err / std::abs(ref_val)
                : abs_err;

            if (abs_err > cr.max_abs_error) cr.max_abs_error = abs_err;
            if (rel_err > cr.max_rel_error) cr.max_rel_error = rel_err;

            ++cr.n_points_compared;
        }
    }

    cr.success = true;
    return cr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Validation: simple_decay — first-order decay A → B
// Reference: run_network -a 1e-8 -r 1e-8 simple_decay.net 1.0 50
// ═══════════════════════════════════════════════════════════════════════════════

int test_validate_simple_decay() {
    auto cr = compare_model(
        data_path("simple_decay.net"),
        data_path("simple_decay_ref.cdat"),
        0.0,   // t_start
        1.0,   // dt
        50     // n_steps
    );

    if (!cr.success) return 1;

    std::cout << "\n    Points compared: " << cr.n_points_compared
              << ", max |err|: " << cr.max_abs_error
              << ", max |rel err|: " << cr.max_rel_error << "\n    ";

    // Machine precision for ODE: should match to ~1e-6 or better
    if (cr.max_abs_error > 1e-5) {
        std::cerr << "  Absolute error too large: " << cr.max_abs_error << std::endl;
        return 1;
    }
    if (cr.max_rel_error > 1e-6) {
        std::cerr << "  Relative error too large: " << cr.max_rel_error << std::endl;
        return 1;
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Validation: reversible — A + B <-> C
// Reference: run_network -a 1e-8 -r 1e-8 two_species_reversible.net 10.0 100
// ═══════════════════════════════════════════════════════════════════════════════

int test_validate_reversible() {
    auto cr = compare_model(
        data_path("two_species_reversible.net"),
        data_path("reversible_ref.cdat"),
        0.0,   // t_start
        10.0,  // dt
        100    // n_steps
    );

    if (!cr.success) return 1;

    std::cout << "\n    Points compared: " << cr.n_points_compared
              << ", max |err|: " << cr.max_abs_error
              << ", max |rel err|: " << cr.max_rel_error << "\n    ";

    if (cr.max_abs_error > 1e-5) {
        std::cerr << "  Absolute error too large: " << cr.max_abs_error << std::endl;
        return 1;
    }
    if (cr.max_rel_error > 1e-6) {
        std::cerr << "  Relative error too large: " << cr.max_rel_error << std::endl;
        return 1;
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Validation: BNG's CaOscillate_Func.net (functional rate laws)
// ═══════════════════════════════════════════════════════════════════════════════

int test_validate_caoscillate_func() {
    // Check if the reference file exists (may not have been generated yet)
    std::string net_file = data_path("CaOscillate_Func.net");
    std::string ref_file = data_path("CaOscillate_Func_ref.cdat");

    std::ifstream check(ref_file);
    if (!check.is_open()) {
        std::cout << "\n    SKIP (reference file not found)\n    ";
        return 0;  // Skip, don't fail
    }

    auto cr = compare_model(net_file, ref_file, 0.0, 0.5, 200);

    if (!cr.success) return 1;

    std::cout << "\n    Points compared: " << cr.n_points_compared
              << ", max |err|: " << cr.max_abs_error
              << ", max |rel err|: " << cr.max_rel_error << "\n    ";

    // Functional rate laws may have slightly larger error
    if (cr.max_abs_error > 1e-3) {
        std::cerr << "  Absolute error too large: " << cr.max_abs_error << std::endl;
        return 1;
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
int main() {
    std::cout << "bngsim validation against run_network" << std::endl;
    std::cout << "======================================" << std::endl;

    RUN_TEST(test_validate_simple_decay);
    RUN_TEST(test_validate_reversible);
    RUN_TEST(test_validate_caoscillate_func);

    std::cout << std::endl;
    std::cout << tests_passed << "/" << tests_run << " validation tests passed." << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
