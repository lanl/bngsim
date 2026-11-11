// bngsim/tests/test_dense_eigenvalues.cpp — issue #78
//
// Correctness gate for the eigensolver the steady-state stability certificate
// runs on the Jacobian at an accepted Newton root. The certificate reduces to
// one question — is any eigenvalue's real part positive — so these cases pin
// both the spectrum and that verdict, on the shapes a reaction-network Jacobian
// actually takes: badly scaled, structurally sparse, heavily repeated
// eigenvalues, defective blocks, and zero rows/columns from fixed species and
// write-only accumulators.
//
// The Gardner toggle case is the issue's own reproducer: its saddle Jacobian
// must come back with a positive eigenvalue, because that is the number the
// solver refuses a root on.

#include <bngsim/dense_eigenvalues.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg)                                                                           \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "  FAIL: " << msg << " [" << __FILE__ << ":" << __LINE__ << "]"           \
                      << std::endl;                                                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

#define RUN_TEST(func)                                                                             \
    do {                                                                                           \
        ++tests_run;                                                                               \
        std::cout << "  " << #func << "... " << std::flush;                                        \
        int _rc = func();                                                                          \
        if (_rc == 0) {                                                                            \
            ++tests_passed;                                                                        \
            std::cout << "OK" << std::endl;                                                        \
        } else {                                                                                   \
            std::cout << "FAILED" << std::endl;                                                    \
        }                                                                                          \
    } while (0)

namespace {

struct Spectrum {
    bool ok = false;
    std::vector<double> re, im;
    double max_re = 0.0;
    double radius = 0.0;
};

// Eigenvalues of a column-major matrix (copied, since the solver destroys it).
Spectrum spectrum_of(const std::vector<double> &a, int n) {
    Spectrum s;
    std::vector<double> work = a;
    s.re.assign(static_cast<std::size_t>(n), 0.0);
    s.im.assign(static_cast<std::size_t>(n), 0.0);
    s.ok = bngsim::dense_eigenvalues(work.data(), n, s.re.data(), s.im.data());
    s.max_re = -1e300;
    for (int i = 0; i < n; ++i) {
        s.max_re = std::max(s.max_re, s.re[i]);
        s.radius = std::max(s.radius, std::hypot(s.re[i], s.im[i]));
    }
    return s;
}

// Is `want` (a real eigenvalue) present in the computed spectrum, to `tol`?
bool has_real(const Spectrum &s, double want, double tol) {
    for (std::size_t i = 0; i < s.re.size(); ++i) {
        if (std::abs(s.re[i] - want) <= tol && std::abs(s.im[i]) <= tol) {
            return true;
        }
    }
    return false;
}

// Is a complex pair re ± i·im present?
bool has_pair(const Spectrum &s, double re, double im, double tol) {
    bool plus = false, minus = false;
    for (std::size_t i = 0; i < s.re.size(); ++i) {
        if (std::abs(s.re[i] - re) > tol)
            continue;
        if (std::abs(s.im[i] - im) <= tol)
            plus = true;
        if (std::abs(s.im[i] + im) <= tol)
            minus = true;
    }
    return plus && minus;
}

// Column-major setter, so the test data below reads row by row.
void set_rows(std::vector<double> &a, int n, std::initializer_list<double> rows) {
    a.assign(static_cast<std::size_t>(n) * n, 0.0);
    int k = 0;
    for (double v : rows) {
        a[static_cast<std::size_t>(k % n) * n + (k / n)] =
            v; // row-major input → column-major store
        ++k;
    }
}

// ── The issue's own matrix ───────────────────────────────────────────────────
// Jacobian of the Gardner toggle at the saddle it returned for
// alpha_2 = 53.526315789473685: eigenvalues +0.40643001 and -2.40643001.
int test_gardner_saddle() {
    std::vector<double> a;
    set_rows(a, 2,
             {-1.0, -31.606656689891912, //
              -0.062583189401426, -1.0});
    Spectrum s = spectrum_of(a, 2);
    CHECK(s.ok, "eigensolver declined the 2x2 saddle Jacobian");
    CHECK(has_real(s, 0.40643001328876216, 1e-9), "missing the unstable eigenvalue +0.40643");
    CHECK(has_real(s, -2.40643001328876216, 1e-9), "missing the stable eigenvalue -2.40643");
    CHECK(s.max_re > 1e-6 * s.radius, "the saddle must read as unstable");
    return 0;
}

// The same model's stable branch — the root integration reaches at that dose.
// Jacobian at [0.00759620791, 53.1227841]: eigenvalues -0.86272 and -1.13728.
int test_gardner_stable_branch() {
    std::vector<double> a;
    set_rows(a, 2,
             {-1.0, -0.000357466138807165, //
              -52.722294565853275, -1.0});
    Spectrum s = spectrum_of(a, 2);
    CHECK(s.ok, "eigensolver declined the stable-branch Jacobian");
    CHECK(has_real(s, -0.862717682, 1e-8), "missing eigenvalue -0.86272");
    CHECK(has_real(s, -1.137282318, 1e-8), "missing eigenvalue -1.13728");
    CHECK(!(s.max_re > 1e-6 * s.radius), "the stable branch must not read as unstable");
    return 0;
}

// ── Shapes with a spectrum known in closed form ──────────────────────────────

// Upper triangular: eigenvalues are the diagonal, whatever the coupling above it.
int test_triangular() {
    const int n = 6;
    std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
    const double diag[6] = {-1e-9, -3.5, 2.75, -100.0, 0.0, -0.25};
    for (int i = 0; i < n; ++i) {
        a[static_cast<std::size_t>(i) * n + i] = diag[i];
        for (int j = i + 1; j < n; ++j)
            a[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * i - 0.25 * j;
    }
    Spectrum s = spectrum_of(a, n);
    CHECK(s.ok, "eigensolver declined a triangular matrix");
    for (int i = 0; i < n; ++i)
        CHECK(has_real(s, diag[i], 1e-9 * std::max(1.0, std::abs(diag[i]))),
              "missing a diagonal eigenvalue of a triangular matrix");
    CHECK(s.max_re > 1e-6 * s.radius, "+2.75 on the diagonal is an unstable eigenvalue");
    return 0;
}

// A damped oscillator block: eigenvalues -0.5 ± 2i. A certificate built on the
// spectral radius alone would call this unstable; the real part is what decides.
int test_complex_pair_is_stable() {
    std::vector<double> a;
    set_rows(a, 2,
             {-0.5, -2.0, //
              2.0, -0.5});
    Spectrum s = spectrum_of(a, 2);
    CHECK(s.ok, "eigensolver declined a rotation block");
    CHECK(has_pair(s, -0.5, 2.0, 1e-12), "expected the pair -0.5 +/- 2i");
    CHECK(!(s.max_re > 1e-6 * s.radius), "a damped oscillation is a stable steady state");
    return 0;
}

// The unstable focus a Hopf bifurcation leaves behind: +0.25 ± 3i. No real
// eigenvalue is positive and the determinant's sign is the Hurwitz one, so only
// the spectrum catches it.
int test_unstable_focus() {
    std::vector<double> a;
    set_rows(a, 2,
             {0.25, -3.0, //
              3.0, 0.25});
    Spectrum s = spectrum_of(a, 2);
    CHECK(s.ok, "eigensolver declined an unstable focus");
    CHECK(has_pair(s, 0.25, 3.0, 1e-12), "expected the pair +0.25 +/- 3i");
    CHECK(s.max_re > 1e-6 * s.radius, "an unstable focus must read as unstable");
    return 0;
}

// Conserved system shape: a zero row/column pair (a fixed species and a
// write-only accumulator) on top of a stable block. The zeros must land AT zero,
// not just near it, or the relative threshold would trip on them.
int test_zero_rows_and_columns() {
    const int n = 4;
    std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
    // Species 0: fixed (zero row). Species 3: pure sink (zero column).
    a[static_cast<std::size_t>(1) * n + 1] = -2.0;
    a[static_cast<std::size_t>(2) * n + 2] = -5.0;
    a[static_cast<std::size_t>(1) * n + 2] = 1.5; // coupling into species 2
    a[static_cast<std::size_t>(2) * n + 3] = 7.0; // the sink is produced
    Spectrum s = spectrum_of(a, n);
    CHECK(s.ok, "eigensolver declined a matrix with zero rows/columns");
    CHECK(has_real(s, -2.0, 1e-12), "missing eigenvalue -2");
    CHECK(has_real(s, -5.0, 1e-12), "missing eigenvalue -5");
    CHECK(std::abs(s.max_re) < 1e-12 * s.radius, "the structural zeros must be exactly zero");
    CHECK(!(s.max_re > 1e-6 * s.radius), "structural zeros are marginal, not unstable");
    return 0;
}

// Issue #1529's Jacobian: a $-fixed source S (a zero row) feeds X at k, X decays
// at 1 and is read by nothing, and S also feeds a pair Y <-> Z. Block lower
// triangular, so the spectrum is exactly 0 (S), -1 (X) and (-3 +- sqrt 5)/2.
// The zero is isolated, but its condition number is k: left inside the QR
// iteration it came back as +6.9e-05 at k = 1e7 (+3.1e-05 on another libm),
// and -1 as -1.000069, so a stable root was certified unstable.
std::vector<double> isolated_source_jacobian(double k, double s_diag) {
    const int n = 4; // S, X, Y, Z
    std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
    auto at = [&](int i, int j) -> double & {
        return a[static_cast<std::size_t>(j) * n + static_cast<std::size_t>(i)];
    };
    at(0, 0) = s_diag; // 0 for the $-fixed S; a slow decay for a free one
    at(1, 0) = k;
    at(1, 1) = -1.0;
    at(2, 0) = 1.0;
    at(2, 2) = -1.0;
    at(2, 3) = 1.0;
    at(3, 2) = 1.0;
    at(3, 3) = -2.0;
    return a;
}

bool spectrum_is_exactly(const Spectrum &s, double s_diag, double tol) {
    const double golden_lo = (-3.0 - std::sqrt(5.0)) / 2.0;
    const double golden_hi = (-3.0 + std::sqrt(5.0)) / 2.0;
    return has_real(s, s_diag, tol) && has_real(s, -1.0, tol) && has_real(s, golden_lo, tol) &&
           has_real(s, golden_hi, tol);
}

int test_an_isolated_zero_feeding_an_unread_species() {
    const auto a = isolated_source_jacobian(1e7, 0.0);
    Spectrum s = spectrum_of(a, 4);
    CHECK(s.ok, "eigensolver declined the isolated-source Jacobian");
    CHECK(spectrum_is_exactly(s, 0.0, 1e-12 * s.radius), "spectrum is not the closed form");
    CHECK(s.max_re == 0.0, "the isolated zero must come back exactly zero");
    CHECK(!(s.max_re > 1e-6 * s.radius), "a stable root must not read as unstable");
    return 0;
}

// Over the issue's scan, k from 1e3 to 1e12: the error grew with k, and where
// the label flipped depended on the sign of the roundoff.
int test_an_isolated_zero_over_twelve_decades_of_k() {
    for (int e = 3; e <= 12; ++e) {
        const double k = std::pow(10.0, e);
        Spectrum s = spectrum_of(isolated_source_jacobian(k, 0.0), 4);
        CHECK(s.ok, "eigensolver declined k = 1e" << e);
        CHECK(spectrum_is_exactly(s, 0.0, 1e-12 * s.radius), "spectrum off at k = 1e" << e);
        CHECK(s.max_re == 0.0, "isolated zero not exact at k = 1e" << e);
    }
    return 0;
}

// The sibling with no fixed species: S free, decaying at 1e-9. Its row is still
// isolated, and its eigenvalue, nine decades under the radius, comes back as
// the diagonal it is.
int test_an_isolated_slow_decay_is_read_exactly() {
    Spectrum s = spectrum_of(isolated_source_jacobian(1e7, -1e-9), 4);
    CHECK(s.ok, "eigensolver declined the slow-decay sibling");
    CHECK(has_real(s, -1e-9, 1e-21), "the isolated -1e-9 must be the diagonal it is");
    CHECK(spectrum_is_exactly(s, -1e-9, 1e-12 * s.radius), "spectrum is not the closed form");
    CHECK(s.max_re < 0.0, "every eigenvalue is negative");
    return 0;
}

// Isolation in every place: the same matrix under each of the 24 orderings of
// its species gives the same exact spectrum, so a row or column found at any
// index is swapped out correctly.
int test_isolation_in_every_ordering() {
    const int n = 4;
    const auto base = isolated_source_jacobian(1e7, 0.0);
    std::vector<int> perm = {0, 1, 2, 3};
    int orderings = 0;
    do {
        std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                a[static_cast<std::size_t>(j) * n + static_cast<std::size_t>(i)] =
                    base[static_cast<std::size_t>(perm[j]) * n + static_cast<std::size_t>(perm[i])];
        Spectrum s = spectrum_of(a, n);
        CHECK(s.ok, "eigensolver declined an ordering");
        CHECK(spectrum_is_exactly(s, 0.0, 1e-12 * s.radius), "spectrum off in an ordering");
        CHECK(s.max_re == 0.0, "isolated zero not exact in an ordering");
        ++orderings;
    } while (std::next_permutation(perm.begin(), perm.end()));
    CHECK(orderings == 24, "expected 24 orderings");
    return 0;
}

// The column half of the isolation: X is read by nothing (an isolated column)
// but reads a coupled pair S <-> Y strongly, so no row is isolated. Its
// eigenvalue, 0, is not ill-conditioned the way the row case is: without the
// column step it came back at the roundoff level, but on either side of zero
// (+2.3e-16 at k = 1e9). Isolated, it is the diagonal, exactly.
int test_an_isolated_column_is_read_exactly() {
    const int n = 3; // S, Y, X
    for (int e = 3; e <= 12; e += 3) {
        const double k = std::pow(10.0, e);
        std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
        auto at = [&](int i, int j) -> double & {
            return a[static_cast<std::size_t>(j) * n + static_cast<std::size_t>(i)];
        };
        at(0, 0) = -1.0;
        at(0, 1) = 1.0;
        at(1, 0) = 1.0;
        at(1, 1) = -2.0;
        at(2, 0) = k;
        at(2, 1) = k;
        Spectrum s = spectrum_of(a, n);
        CHECK(s.ok, "eigensolver declined the isolated-column matrix at k = 1e" << e);
        CHECK(has_real(s, (-3.0 - std::sqrt(5.0)) / 2.0, 1e-12 * s.radius), "lower root off");
        CHECK(has_real(s, (-3.0 + std::sqrt(5.0)) / 2.0, 1e-12 * s.radius), "upper root off");
        CHECK(s.max_re == 0.0, "the isolated column's zero must be exact at k = 1e" << e);
    }
    return 0;
}

// Rate constants spanning eight decades — what balancing is there for. The slow
// mode is what a stability verdict turns on and it must survive the fast one's
// roundoff: at a spectral radius of 1e6, an eigenvalue at -1e-2 sits eight
// decades down, and it comes back to twelve digits.
//
// (The limit is real and this case is near it: push the slow mode to -1e-8, and
// no method resolves it — a perturbation of eps·1e6 = 1e-10 swamps it. That is
// what the certificate's threshold is relative to the spectral radius for.)
int test_badly_scaled() {
    const int n = 3;
    std::vector<double> a;
    set_rows(a, n,
             {-1e6, 1e6, 0.0,   //
              1e-2, -1e-2, 0.0, //
              0.0, 5e2, -1e-2});
    Spectrum s = spectrum_of(a, n);
    CHECK(s.ok, "eigensolver declined a badly scaled matrix");
    CHECK(has_real(s, -1e-2, 1e-10), "the slow mode (-1e-2) must survive the fast one");
    CHECK(has_real(s, -1.00000001e6, 1e-2), "missing the fast mode (-1e6)");
    CHECK(!(s.max_re > 1e-6 * s.radius), "all three modes decay or are marginal");
    return 0;
}

// Heavily repeated eigenvalues: a reaction network where a hundred species share
// one degradation constant is the common case, and it is the case that stalls a
// naive QR iteration (this is why the sweep budget is LAPACK's 30·max(10,n)
// rather than a per-root cap).
int test_repeated_eigenvalues() {
    const int n = 120;
    std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        a[static_cast<std::size_t>(i) * n + i] = -0.1;
        if (i + 1 < n)
            a[static_cast<std::size_t>(i + 1) * n + i] = 0.1; // one-way chain
    }
    Spectrum s = spectrum_of(a, n);
    CHECK(s.ok, "eigensolver declined a 120x120 matrix with one repeated eigenvalue");
    for (int i = 0; i < n; ++i)
        CHECK(std::abs(s.re[i] + 0.1) < 1e-9 && std::abs(s.im[i]) < 1e-9,
              "every eigenvalue of the chain is -0.1");
    return 0;
}

// Degenerate edges: the certificate has to get an answer or a clean refusal, not
// undefined behavior.
int test_edge_sizes() {
    std::vector<double> one{3.25};
    Spectrum s1 = spectrum_of(one, 1);
    CHECK(s1.ok && std::abs(s1.re[0] - 3.25) < 1e-15 && s1.im[0] == 0.0, "1x1 spectrum");

    std::vector<double> zeros(9, 0.0);
    Spectrum sz = spectrum_of(zeros, 3);
    CHECK(sz.ok && sz.radius == 0.0, "the zero matrix has a zero spectrum");

    std::vector<double> nan_mat{1.0, 0.0, 0.0, std::nan("")};
    std::vector<double> re(2), im(2);
    CHECK(!bngsim::dense_eigenvalues(nan_mat.data(), 2, re.data(), im.data()),
          "a non-finite entry must be declined, not laundered into a spectrum");

    std::vector<double> empty;
    CHECK(bngsim::dense_eigenvalues(empty.data(), 0, nullptr, nullptr), "n=0 is vacuously fine");
    return 0;
}

// A similarity transform cannot move the spectrum: run a known-diagonal matrix
// through a non-orthogonal change of basis and ask for the diagonal back. This
// exercises balance + Hessenberg + QR on a dense matrix with no exploitable
// structure, without needing a reference implementation to compare against.
int test_similarity_invariance() {
    const int n = 5;
    const double lam[5] = {-4.0, -1.0, -0.5, -0.125, 0.375};
    std::vector<double> d(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        d[static_cast<std::size_t>(i) * n + i] = lam[i];
    // S = I + strictly-lower N (unit lower triangular, so S⁻¹ = I - N + N² - …
    // is exact in integers); apply A ← S·D·S⁻¹ by row/column operations.
    std::vector<double> a = d;
    for (int k = 0; k < n - 1; ++k) {
        const double m = 1.0 + 0.5 * k;
        // Row_{k+1} += m·Row_k, then Col_k -= m·Col_{k+1} (the inverse operation).
        for (int j = 0; j < n; ++j)
            a[static_cast<std::size_t>(j) * n + (k + 1)] +=
                m * a[static_cast<std::size_t>(j) * n + k];
        for (int i = 0; i < n; ++i)
            a[static_cast<std::size_t>(k) * n + i] -=
                m * a[static_cast<std::size_t>(k + 1) * n + i];
    }
    Spectrum s = spectrum_of(a, n);
    CHECK(s.ok, "eigensolver declined a similarity-transformed diagonal matrix");
    for (int i = 0; i < n; ++i)
        CHECK(has_real(s, lam[i], 1e-10), "a similarity transform moved an eigenvalue");
    CHECK(s.max_re > 1e-6 * s.radius, "+0.375 is still there after the change of basis");
    return 0;
}

} // namespace

int main() {
    std::cout << "Running dense eigenvalue tests (issue #78)..." << std::endl;
    RUN_TEST(test_gardner_saddle);
    RUN_TEST(test_gardner_stable_branch);
    RUN_TEST(test_triangular);
    RUN_TEST(test_complex_pair_is_stable);
    RUN_TEST(test_unstable_focus);
    RUN_TEST(test_zero_rows_and_columns);
    RUN_TEST(test_an_isolated_zero_feeding_an_unread_species);
    RUN_TEST(test_an_isolated_zero_over_twelve_decades_of_k);
    RUN_TEST(test_an_isolated_slow_decay_is_read_exactly);
    RUN_TEST(test_isolation_in_every_ordering);
    RUN_TEST(test_an_isolated_column_is_read_exactly);
    RUN_TEST(test_badly_scaled);
    RUN_TEST(test_repeated_eigenvalues);
    RUN_TEST(test_edge_sizes);
    RUN_TEST(test_similarity_invariance);
    std::cout << tests_passed << "/" << tests_run << " passed" << std::endl;
    return (tests_passed == tests_run) ? 0 : 1;
}
