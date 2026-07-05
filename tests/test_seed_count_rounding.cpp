// bngsim/tests/test_seed_count_rounding.cpp — unit tests for the cold-start
// stochastic seed-count rounding policy (GH #51).
//
// Exercises the header-only rewriter that bngsim applies to BNG XML before
// handing it to the network-free engines: round-half-up of fractional
// <Species concentration> values, the integer fast-path (no temp file), and the
// fail-safe (unresolvable tokens left verbatim). The end-to-end behavior through
// real NFsim/RuleMonkey is covered by python/tests/test_seed_count_rounding.py;
// this pins the rewriter contract directly.

#include "seed_count_rounding.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                              \
            std::cout << "    CHECK failed: " #cond " (line " << __LINE__ << ")\n"; \
            return 1;                                                               \
        }                                                                           \
    } while (0)

#define RUN_TEST(func)                                       \
    do {                                                     \
        ++tests_run;                                         \
        std::cout << "  " << #func << "... " << std::flush;  \
        int _rc = func();                                    \
        if (_rc == 0) {                                      \
            ++tests_passed;                                  \
            std::cout << "OK" << std::endl;                  \
        } else {                                             \
            std::cout << "FAILED" << std::endl;              \
        }                                                    \
    } while (0)

namespace seedround = bngsim::seedround;

static int test_round_half_up_count_matrix() {
    // The issue's representative values and exact halves (ties away from zero).
    CHECK(seedround::round_half_up_count(0.389) == 0);
    CHECK(seedround::round_half_up_count(0.10001) == 0);
    CHECK(seedround::round_half_up_count(5.7) == 6);
    CHECK(seedround::round_half_up_count(155.6747) == 156);
    CHECK(seedround::round_half_up_count(466.98) == 467);
    CHECK(seedround::round_half_up_count(0.5) == 1);
    CHECK(seedround::round_half_up_count(1.5) == 2);
    CHECK(seedround::round_half_up_count(2.5) == 3);
    CHECK(seedround::round_half_up_count(3.5) == 4);
    // Idempotent on integers.
    CHECK(seedround::round_half_up_count(0.0) == 0);
    CHECK(seedround::round_half_up_count(5.0) == 5);
    CHECK(seedround::round_half_up_count(467.0) == 467);
    // Symmetric (never arises for populations, but defined).
    CHECK(seedround::round_half_up_count(-0.5) == -1);
    CHECK(seedround::round_half_up_count(-2.5) == -3);
    return 0;
}

static std::filesystem::path write_temp(const std::string &contents) {
    static int counter = 0;
    auto p = std::filesystem::temp_directory_path() /
             ("bngsim_seedround_test_in_" + std::to_string(::getpid()) + "_" +
              std::to_string(counter++) + ".xml");
    std::ofstream out(p);
    out << contents;
    out.close();
    return p;
}

static std::string read_file(const std::filesystem::path &p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static const char *XML_HEAD =
    "<?xml version=\"1.0\"?>\n<sbml><model id=\"m\">\n  <ListOfParameters>\n";
static const char *XML_MID = "  </ListOfParameters>\n  <ListOfSpecies>\n";
static const char *XML_TAIL = "  </ListOfSpecies>\n</model></sbml>\n";

static int test_literal_fractional_rounds() {
    std::string xml = std::string(XML_HEAD) + XML_MID +
                      "    <Species id=\"S1\" concentration=\"5.7\" name=\"X()\"/>\n" + XML_TAIL;
    auto in = write_temp(xml);
    auto out = seedround::write_seed_rounded_xml(in.string(), {});
    CHECK(out.has_value());
    std::string body = read_file(*out);
    CHECK(body.find("concentration=\"6\"") != std::string::npos);
    CHECK(body.find("5.7") == std::string::npos);
    std::filesystem::remove(in);
    std::filesystem::remove(*out);
    return 0;
}

static int test_param_id_concentration_rounds() {
    // concentration references a parameter whose value= is fractional.
    std::string xml = std::string(XML_HEAD) +
                      "    <Parameter id=\"n_init\" value=\"1.9\"/>\n" + XML_MID +
                      "    <Species id=\"S1\" concentration=\"n_init\" name=\"X()\"/>\n" + XML_TAIL;
    auto in = write_temp(xml);
    auto out = seedround::write_seed_rounded_xml(in.string(), {});
    CHECK(out.has_value());
    std::string body = read_file(*out);
    CHECK(body.find("concentration=\"2\"") != std::string::npos);
    std::filesystem::remove(in);
    std::filesystem::remove(*out);
    return 0;
}

static int test_overlay_wins_over_xml_value() {
    // Overlay (e.g. a setParameter override) supersedes the XML value= attr.
    std::string xml = std::string(XML_HEAD) +
                      "    <Parameter id=\"n_init\" value=\"1.9\"/>\n" + XML_MID +
                      "    <Species id=\"S1\" concentration=\"n_init\" name=\"X()\"/>\n" + XML_TAIL;
    auto in = write_temp(xml);
    std::unordered_map<std::string, double> overlay{{"n_init", 5.7}};
    auto out = seedround::write_seed_rounded_xml(in.string(), overlay);
    CHECK(out.has_value());
    std::string body = read_file(*out);
    CHECK(body.find("concentration=\"6\"") != std::string::npos); // 5.7 -> 6, not 1.9 -> 2
    std::filesystem::remove(in);
    std::filesystem::remove(*out);
    return 0;
}

static int test_integer_only_returns_nullopt() {
    // No fractional seed → byte-identical → nullopt (fast path, no temp file).
    std::string xml = std::string(XML_HEAD) +
                      "    <Parameter id=\"k\" value=\"100\"/>\n" + XML_MID +
                      "    <Species id=\"S1\" concentration=\"5000\" name=\"X()\"/>\n"
                      "    <Species id=\"S2\" concentration=\"k\" name=\"Y()\"/>\n" + XML_TAIL;
    auto in = write_temp(xml);
    auto out = seedround::write_seed_rounded_xml(in.string(), {});
    CHECK(!out.has_value());
    std::filesystem::remove(in);
    return 0;
}

static int test_unresolvable_token_left_verbatim() {
    // A compound expression bngsim does not resolve is left for the engine to
    // handle exactly as before (fail-safe): no rounding, nullopt.
    std::string xml = std::string(XML_HEAD) +
                      "    <Parameter id=\"a\" value=\"1.0\"/>\n" + XML_MID +
                      "    <Species id=\"S1\" concentration=\"a+0.5\" name=\"X()\"/>\n" + XML_TAIL;
    auto in = write_temp(xml);
    auto out = seedround::write_seed_rounded_xml(in.string(), {});
    CHECK(!out.has_value());
    std::filesystem::remove(in);
    return 0;
}

static int test_missing_source_is_fail_safe() {
    auto out = seedround::write_seed_rounded_xml("/nonexistent/path/to/model.xml", {});
    CHECK(!out.has_value());
    return 0;
}

int main() {
    std::cout << "Running seed-count rounding tests (GH #51)\n";
    RUN_TEST(test_round_half_up_count_matrix);
    RUN_TEST(test_literal_fractional_rounds);
    RUN_TEST(test_param_id_concentration_rounds);
    RUN_TEST(test_overlay_wins_over_xml_value);
    RUN_TEST(test_integer_only_returns_nullopt);
    RUN_TEST(test_unresolvable_token_left_verbatim);
    RUN_TEST(test_missing_source_is_fail_safe);
    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
