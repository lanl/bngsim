// bngsim/src/dense_eigenvalues.cpp -- eigenvalues of a small dense real matrix
// (issue #78). See the header for why this is not a LAPACK call.
//
// The sequence is the standard one:
//
//   balance      a diagonal similarity that equalizes row and column norms.
//                Not cosmetic here: a reaction-network Jacobian mixes rate
//                constants spanning many decades, and the QR iteration's
//                accuracy is governed by the norm of the matrix it runs on.
//   hessenberg   Householder reflectors, similarity, no Q accumulated (only
//                eigenvalues are wanted).
//   francis_qr   the double-shift implicit QR iteration on the Hessenberg
//                form, deflating one 1×1 or 2×2 block at a time.
//
// All three are similarity transforms, so every step preserves the spectrum
// exactly (in exact arithmetic); the whole sequence is backward stable — the
// computed eigenvalues are exact for A + E with ||E|| = O(eps·||A||).

#include "bngsim/dense_eigenvalues.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace bngsim {

namespace {

// Column-major element access, the layout every dense matrix in this library
// uses (SUNDenseMatrix, the codegen Jacobian, compute_ss_sensitivity's J).
inline double &el(double *a, int n, int i, int j) {
    return a[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
             static_cast<std::size_t>(i)];
}

// Diagonal similarity D⁻¹AD with D a power of the radix, so it introduces no
// rounding error at all. Iterates until no row/column pair is improved by more
// than 5%, the EISPACK `balanc` criterion.
void balance(double *a, int n) {
    constexpr double radix = 2.0;
    constexpr double radix2 = radix * radix;
    bool converged = false;
    while (!converged) {
        converged = true;
        for (int i = 0; i < n; ++i) {
            double col = 0.0; // ||A[:,i]||_1 off the diagonal
            double row = 0.0; // ||A[i,:]||_1 off the diagonal
            for (int j = 0; j < n; ++j) {
                if (j == i)
                    continue;
                col += std::abs(el(a, n, j, i));
                row += std::abs(el(a, n, i, j));
            }
            if (col == 0.0 || row == 0.0)
                continue; // an isolated eigenvalue; nothing to scale
            double g = row / radix;
            double f = 1.0;
            const double s = col + row;
            while (col < g) {
                f *= radix;
                col *= radix2;
            }
            g = row * radix;
            while (col > g) {
                f /= radix;
                col /= radix2;
            }
            if ((col + row) / f < 0.95 * s) {
                converged = false;
                const double gi = 1.0 / f;
                for (int j = 0; j < n; ++j)
                    el(a, n, i, j) *= gi;
                for (int j = 0; j < n; ++j)
                    el(a, n, j, i) *= f;
            }
        }
    }
}

// Householder reduction to upper Hessenberg form (similarity, Q discarded).
void hessenberg(double *a, int n) {
    std::vector<double> v(static_cast<std::size_t>(n), 0.0);
    for (int k = 0; k + 2 < n; ++k) {
        double norm2 = 0.0;
        for (int i = k + 1; i < n; ++i)
            norm2 += el(a, n, i, k) * el(a, n, i, k);
        if (norm2 == 0.0)
            continue; // column already Hessenberg
        const double norm = std::sqrt(norm2);
        const double alpha = (el(a, n, k + 1, k) >= 0.0) ? -norm : norm;

        double vnorm2 = 0.0;
        for (int i = k + 1; i < n; ++i)
            v[static_cast<std::size_t>(i)] = el(a, n, i, k);
        v[static_cast<std::size_t>(k + 1)] -= alpha;
        for (int i = k + 1; i < n; ++i)
            vnorm2 += v[static_cast<std::size_t>(i)] * v[static_cast<std::size_t>(i)];
        if (vnorm2 == 0.0)
            continue;
        const double beta = 2.0 / vnorm2;

        // P·A over rows k+1..n-1. Columns below k are already zero there.
        for (int j = k; j < n; ++j) {
            double s = 0.0;
            for (int i = k + 1; i < n; ++i)
                s += v[static_cast<std::size_t>(i)] * el(a, n, i, j);
            s *= beta;
            for (int i = k + 1; i < n; ++i)
                el(a, n, i, j) -= s * v[static_cast<std::size_t>(i)];
        }
        // (P·A)·P over columns k+1..n-1.
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = k + 1; j < n; ++j)
                s += el(a, n, i, j) * v[static_cast<std::size_t>(j)];
            s *= beta;
            for (int j = k + 1; j < n; ++j)
                el(a, n, i, j) -= s * v[static_cast<std::size_t>(j)];
        }
        for (int i = k + 2; i < n; ++i)
            el(a, n, i, k) = 0.0; // exact zeros, not roundoff residue
    }
}

// Francis double-shift QR on an upper Hessenberg matrix, eigenvalues only.
// Deflates from the bottom; `t` accumulates the exceptional shifts applied to
// the whole active block so they can be added back to each root.
bool francis_qr(double *a, int n, double *wr, double *wi) {
    // Total sweep budget for the whole matrix, LAPACK dlahqr's rule
    // (30·max(10, n)) rather than a per-root cap. A spectrum with high
    // multiplicities — the norm for a reaction network, where a hundred species
    // can share one degradation constant — spends most of its sweeps on a few
    // stubborn roots and finishes the rest in two or three each. The 593-species
    // `ode/before_bunching` Jacobian needs ~190 sweeps on one root and converges
    // to the exact spectrum; a per-root cap of 60 or 100 declines it outright.
    const long max_sweeps = 30L * std::max(10, n);
    long sweeps = 0;

    double anorm = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = std::max(i - 1, 0); j < n; ++j)
            anorm += std::abs(el(a, n, i, j));

    int nn = n - 1;
    double t = 0.0;
    while (nn >= 0) {
        int its = 0;
        int l = 0;
        do {
            // Look for a negligible subdiagonal entry to split the block at.
            for (l = nn; l >= 1; --l) {
                double s = std::abs(el(a, n, l - 1, l - 1)) + std::abs(el(a, n, l, l));
                if (s == 0.0)
                    s = anorm;
                if (std::abs(el(a, n, l, l - 1)) + s == s) {
                    el(a, n, l, l - 1) = 0.0;
                    break;
                }
            }
            double x = el(a, n, nn, nn);
            if (l == nn) {
                // One real root.
                wr[nn] = x + t;
                wi[nn] = 0.0;
                --nn;
            } else {
                double y = el(a, n, nn - 1, nn - 1);
                double w = el(a, n, nn, nn - 1) * el(a, n, nn - 1, nn);
                if (l == nn - 1) {
                    // A 2×2 block: solve its characteristic quadratic.
                    const double p = 0.5 * (y - x);
                    const double q = p * p + w;
                    double z = std::sqrt(std::abs(q));
                    x += t;
                    if (q >= 0.0) {
                        z = p + std::copysign(z, p);
                        wr[nn - 1] = wr[nn] = x + z;
                        if (z != 0.0)
                            wr[nn] = x - w / z;
                        wi[nn - 1] = wi[nn] = 0.0;
                    } else {
                        wr[nn - 1] = wr[nn] = x + p;
                        wi[nn] = z;
                        wi[nn - 1] = -z;
                    }
                    nn -= 2;
                } else {
                    if (++sweeps > max_sweeps)
                        return false;
                    double p = 0.0, q = 0.0, r = 0.0, z = 0.0, s = 0.0;
                    if (its > 0 && its % 10 == 0) {
                        // Exceptional shift: the Wilkinson shift has stalled.
                        t += x;
                        for (int i = 0; i <= nn; ++i)
                            el(a, n, i, i) -= x;
                        s = std::abs(el(a, n, nn, nn - 1)) + std::abs(el(a, n, nn - 1, nn - 2));
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    // Find two consecutive small subdiagonals to start the bulge at.
                    int m = l;
                    for (m = nn - 2; m >= l; --m) {
                        z = el(a, n, m, m);
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / el(a, n, m + 1, m) + el(a, n, m, m + 1);
                        q = el(a, n, m + 1, m + 1) - z - r - s;
                        r = el(a, n, m + 2, m + 1);
                        s = std::abs(p) + std::abs(q) + std::abs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l)
                            break;
                        const double u = std::abs(el(a, n, m, m - 1)) * (std::abs(q) + std::abs(r));
                        const double v =
                            std::abs(p) * (std::abs(el(a, n, m - 1, m - 1)) + std::abs(z) +
                                           std::abs(el(a, n, m + 1, m + 1)));
                        if (u + v == v)
                            break;
                    }
                    for (int i = m + 2; i <= nn; ++i) {
                        el(a, n, i, i - 2) = 0.0;
                        if (i != m + 2)
                            el(a, n, i, i - 3) = 0.0;
                    }
                    // Chase the bulge down the subdiagonal.
                    for (int k = m; k <= nn - 1; ++k) {
                        if (k != m) {
                            p = el(a, n, k, k - 1);
                            q = el(a, n, k + 1, k - 1);
                            r = (k != nn - 1) ? el(a, n, k + 2, k - 1) : 0.0;
                            x = std::abs(p) + std::abs(q) + std::abs(r);
                            if (x != 0.0) {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        s = std::copysign(std::sqrt(p * p + q * q + r * r), p);
                        if (s == 0.0)
                            continue;
                        if (k == m) {
                            if (l != m)
                                el(a, n, k, k - 1) = -el(a, n, k, k - 1);
                        } else {
                            el(a, n, k, k - 1) = -s * x;
                        }
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;
                        for (int j = k; j <= nn; ++j) { // row modification
                            p = el(a, n, k, j) + q * el(a, n, k + 1, j);
                            if (k != nn - 1) {
                                p += r * el(a, n, k + 2, j);
                                el(a, n, k + 2, j) -= p * z;
                            }
                            el(a, n, k + 1, j) -= p * y;
                            el(a, n, k, j) -= p * x;
                        }
                        const int mmin = std::min(nn, k + 3);
                        for (int i = l; i <= mmin; ++i) { // column modification
                            p = x * el(a, n, i, k) + y * el(a, n, i, k + 1);
                            if (k != nn - 1) {
                                p += z * el(a, n, i, k + 2);
                                el(a, n, i, k + 2) -= p * r;
                            }
                            el(a, n, i, k + 1) -= p * q;
                            el(a, n, i, k) -= p;
                        }
                    }
                }
            }
        } while (l < nn - 1);
    }
    return true;
}

} // namespace

bool dense_eigenvalues(double *a, int n, double *wr, double *wi) {
    if (n <= 0)
        return true;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n) * static_cast<std::size_t>(n); ++i) {
        if (!std::isfinite(a[i]))
            return false;
    }
    if (n == 1) {
        wr[0] = a[0];
        wi[0] = 0.0;
        return true;
    }
    balance(a, n);
    hessenberg(a, n);
    if (!francis_qr(a, n, wr, wi))
        return false;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(wr[i]) || !std::isfinite(wi[i]))
            return false;
    }
    return true;
}

} // namespace bngsim
