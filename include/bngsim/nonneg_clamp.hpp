// bngsim/include/bngsim/nonneg_clamp.hpp -- which state slots the GH #135
// non-finite-RHS retry may clamp to 0, and the clamp itself.
//
// One rule, two integrators (issue #1357). The time course (cvode_simulator.cpp)
// has retried a non-finite RHS and Jacobian on the nonnegative-clamped state
// since GH #135, over the slots issue #706 allows. The steady-state run test
// (steady_state.cpp) now retries the same way, so the slots, the clamp, its
// second reading near 0 (issue #1340) and the reading that triggers them are
// defined here, once.
#pragma once

#include "bngsim/model.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace bngsim {

// How far below 0, in a slot's own atol, no rounding of 0 reaches: the time
// course's spare_at_pole (cvode_simulator.cpp, where the reason is given) and
// restore_where_finite both read it.
inline constexpr double kClampNoiseAtols = 1e6;

// 1 for a slot that is a concentration by construction, 0 for one that may be
// negative by design (issue #706): a fixed species, one not reported as a
// species (a promoted parameter or compartment), one declared with a negative
// initial value, and the target of a rate rule.
inline std::vector<char> nonneg_clamp_mask_of(const NetworkModel &model) {
    const int ns = model.n_species();
    std::vector<char> mask(ns, 1);
    const auto &species = model.species();
    for (int i = 0; i < ns; ++i) {
        const Species &sp = species[i];
        if (sp.fixed || !sp.reported || sp.declared_negative)
            mask[i] = 0;
    }
    for (const Reaction &rxn : model.reactions()) {
        if (!rxn.is_rate_rule_ode)
            continue;
        for (int si : rxn.product_indices) {
            if (si >= 1 && si <= ns)
                mask[si - 1] = 0;
        }
    }
    return mask;
}

// `y` with every slot the mask allows, and that is not positive, put at 0, into
// `out` (sized to the mask), but for the slots `spare` marks: the time course's
// spare_at_pole, which the run test does not use.
inline void clamp_nonneg_into(const std::vector<char> &mask, const double *y,
                              std::vector<double> &out, const char *spare = nullptr) {
    const std::size_t n = mask.size();
    if (out.size() != n)
        out.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = (y[i] > 0.0 || !mask[i] || (spare != nullptr && spare[i])) ? y[i] : 0.0;
}

// The slots each row of f reads, from the Jacobian's sparsity pattern (CSC,
// one column per slot read), read by rows. Empty where the pattern does not
// cover `n` slots, and then the clamp is not narrowed (issue #1661).
struct SlotsReadByRow {
    std::vector<int64_t> row_ptr;
    std::vector<int> cols;
    bool covers(std::size_t n) const { return row_ptr.size() == n + 1; }
};

inline SlotsReadByRow slots_read_by_row(const JacobianSparsity &sp, int n) {
    SlotsReadByRow out;
    if (n <= 0 || sp.n != n || sp.col_ptrs.size() != static_cast<std::size_t>(n) + 1)
        return out;
    const auto un = static_cast<std::size_t>(n);
    const auto nnz = static_cast<std::size_t>(sp.col_ptrs[un]);
    if (sp.row_indices.size() < nnz)
        return out;
    std::vector<int64_t> row_ptr(un + 1, 0);
    for (std::size_t k = 0; k < nnz; ++k) {
        const int64_t r = sp.row_indices[k];
        if (r < 0 || r >= n)
            return out;
        ++row_ptr[static_cast<std::size_t>(r) + 1];
    }
    for (std::size_t r = 0; r < un; ++r)
        row_ptr[r + 1] += row_ptr[r];
    out.cols.assign(nnz, 0);
    std::vector<int64_t> next(row_ptr.begin(), row_ptr.end() - 1);
    for (std::size_t c = 0; c < un; ++c) {
        for (int64_t k = sp.col_ptrs[c]; k < sp.col_ptrs[c + 1]; ++k) {
            const auto r = static_cast<std::size_t>(sp.row_indices[static_cast<std::size_t>(k)]);
            out.cols[static_cast<std::size_t>(next[r]++)] = static_cast<int>(c);
        }
    }
    out.row_ptr = std::move(row_ptr);
    return out;
}

// The narrowed clamp (issue #1661): into `out`, `y` with each slot
// clamp_nonneg_into would put at 0 put there only where a row `bad_row` marks
// reads it, and into `read` the slots those rows read (for
// read_near_zero_read_into). Where f or the Jacobian is not finite, a slot no
// such row reads is not what makes it so: clamping it cannot make the reading
// finite, and only moves the rows that do read it. A species negative where
// the model has it so, but not declared negative (a voltage written to -61 by
// set_concentration, or one declared at 10 that its own law takes below 0), was
// read at 0 by every retry once another species tipped a law non-finite, and
// followed the wrong ODE with no warning: V(40) = -328.6 for -69.8. Where
// `rows` or `bad_row` does not cover the mask, every slot is read, and `out`
// is clamp_nonneg_into's state.
inline void clamp_read_nonneg_into(const std::vector<char> &mask, const SlotsReadByRow &rows,
                                   const std::vector<char> &bad_row, const double *y,
                                   std::vector<double> &out, std::vector<char> &read,
                                   const char *spare = nullptr) {
    const std::size_t n = mask.size();
    clamp_nonneg_into(mask, y, out, spare);
    if (!rows.covers(n) || bad_row.size() != n) {
        read.assign(n, 1);
        return;
    }
    read.assign(n, 0);
    for (std::size_t r = 0; r < n; ++r) {
        if (!bad_row[r])
            continue;
        for (int64_t k = rows.row_ptr[r]; k < rows.row_ptr[r + 1]; ++k)
            read[static_cast<std::size_t>(rows.cols[static_cast<std::size_t>(k)])] = 1;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!read[i])
            out[i] = y[i];
    }
}

// The rows of f that are not finite, into `bad_row`.
inline void nonfinite_rows_of_f(const double *f, std::size_t n, std::vector<char> &bad_row) {
    bad_row.assign(n, 0);
    for (std::size_t r = 0; r < n; ++r)
        bad_row[r] = std::isfinite(f[r]) ? 0 : 1;
}

// The rows of a dense column-major n x n Jacobian with an entry that is not
// finite, into `bad_row`.
inline void nonfinite_rows_of_dense(const double *jac, std::size_t n, std::vector<char> &bad_row) {
    bad_row.assign(n, 0);
    for (std::size_t c = 0; c < n; ++c) {
        for (std::size_t r = 0; r < n; ++r) {
            if (!std::isfinite(jac[c * n + r]))
                bad_row[r] = 1;
        }
    }
}

// The rows of a Jacobian held as values over the sparsity pattern `sp` (by its
// data index) with an entry that is not finite, into `bad_row`.
inline void nonfinite_rows_of_pattern(const double *vals, const JacobianSparsity &sp, std::size_t n,
                                      std::vector<char> &bad_row) {
    bad_row.assign(n, 0);
    if (sp.col_ptrs.size() != n + 1)
        return;
    for (int64_t k = 0; k < sp.col_ptrs[n]; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        const int64_t r = sp.row_indices[uk];
        if (!std::isfinite(vals[uk]) && r >= 0 && static_cast<std::size_t>(r) < n)
            bad_row[static_cast<std::size_t>(r)] = 1;
    }
}

// The second reading of the retry (issue #1340), for where the clamp above
// leaves f or the Jacobian non-finite: in `out`, which holds that clamp's
// reading of `y`, each slot the mask leaves alone that is below 0 by less than
// its own `atol` put at 0. Such a slot may be negative by design (a membrane
// potential at -61), but inside its atol of 0 it is the boundary value with
// rounding under it: MODEL1812040005's P, a rate-rule target at an exact
// equilibrium of 0, which one order of the RHS's sums puts at exactly 0 and the
// other at -3.6e-27, where `a*P^z` is NaN. Taken second, so a slot negative by
// design is moved only where the reading has no value with it as the run has
// it: beside a masked slot whose clamp alone makes f finite, it is read at its
// value. The sensitivity RHS reads its state at the same band (GH #392). True
// where a slot moved; an `atol` not sized to the mask moves none.
inline bool read_near_zero_into(const std::vector<char> &mask, const double *y,
                                const std::vector<double> &atol, std::vector<double> &out) {
    const std::size_t n = mask.size();
    if (atol.size() != n || out.size() != n)
        return false;
    bool moved = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (!mask[i] && y[i] < 0.0 && y[i] > -atol[i]) {
            out[i] = 0.0;
            moved = true;
        }
    }
    return moved;
}

// The second reading narrowed as the first is (issue #1661): read_near_zero_into
// over the slots `read` marks only, the ones a non-finite row reads, with `out`
// holding clamp_read_nonneg_into's state. A slot in the band that no such row
// reads is not what keeps the reading from being finite either.
inline bool read_near_zero_read_into(const std::vector<char> &mask, const std::vector<char> &read,
                                     const double *y, const std::vector<double> &atol,
                                     std::vector<double> &out) {
    const std::size_t n = mask.size();
    if (atol.size() != n || out.size() != n || read.size() != n)
        return false;
    bool moved = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (read[i] && !mask[i] && y[i] < 0.0 && y[i] > -atol[i]) {
            out[i] = 0.0;
            moved = true;
        }
    }
    return moved;
}

// What an evaluation of the right-hand side or of a closed-form Jacobian read:
// all finite, a NaN and no infinity, or an infinity. The time course retries
// either non-finite reading; the run test retries only the NaN one.
enum class ClampReading { finite, nan_only, infinite };

inline ClampReading read_values(const double *v, std::size_t n) {
    bool nan = false;
    for (std::size_t k = 0; k < n; ++k) {
        if (std::isinf(v[k]))
            return ClampReading::infinite;
        nan = nan || std::isnan(v[k]);
    }
    return nan ? ClampReading::nan_only : ClampReading::finite;
}

// The narrowed reading's last step (issue #1661): a slot it reads at 0 that
// is below 0 by more than kClampNoiseAtols of its atol is put back at its
// value wherever the reading stays finite without it at 0. A row that is
// non-finite reads every slot in it, not only the one that makes it so:
// k*sqrt(S)*(1 + V) reads V, and V at -61 was read at 0 with S. A slot nearer
// 0 is not tried: that is where the run's own error puts a species it empties,
// and reading it at 0 is what the retry is for. Tried there too, a glycolysis
// model's species at -1e-8 (MODEL5952308332) were put back, which took 8% more
// right-hand sides and moved nothing past the run's own error.
//
// A slot put back once is put back at any depth below 0 for the rest of the
// run (`kept`, bit 1). Without that the right-hand side jumps where the slot
// crosses the threshold: at atol 1e-4 (a threshold of -100), V relaxing to
// E = -70 read dV/dt = g E = -7 above -100 and +3 below it, and the run held V
// at -100 for 8.9 million steps. Bit 2 marks the slots put back by the reading
// being tried (put_back_all), until it is kept or undone; bit 4 one the last
// try needed at 0, which the first try of the next call leaves there.
inline bool put_back_candidate(double y, double out, double atol, char kept) {
    return out == 0.0 && y < 0.0 && (y < -kClampNoiseAtols * atol || (kept & 1) != 0);
}

// Every candidate in `out` put back at its value, but those marked needed
// where `skip_needed`, each marked in `kept` (bit 2); how many.
inline std::size_t put_back_all(const double *y, std::vector<double> &out,
                                const std::vector<double> &atol, std::vector<char> &kept,
                                bool skip_needed = false) {
    std::size_t moved = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (put_back_candidate(y[i], out[i], atol[i], kept[i]) &&
            !(skip_needed && (kept[i] & 4) != 0)) {
            out[i] = y[i];
            kept[i] = static_cast<char>(kept[i] | 2);
            ++moved;
        }
    }
    return moved;
}

// The slots put_back_all marked, kept (true) or read at 0 again (false).
inline void settle_put_back(std::vector<double> &out, std::vector<char> &kept, bool keep) {
    for (std::size_t i = 0; i < out.size(); ++i) {
        if ((kept[i] & 2) == 0)
            continue;
        if (!keep)
            out[i] = 0.0;
        kept[i] = keep ? 1 : static_cast<char>(kept[i] & 5);
    }
}

// All the candidates at once first, but any the last call needed at 0: one
// reading, where none of them is needed at 0. Then each candidate still at 0
// on its own, those the last call needed there among them. A right-hand side
// read with k candidates costs 2 readings where it reads them all, up to m + 1
// more where it needs m of them at 0 as the call before did (each is tried
// again, and the reading taken once more at the state kept), and k + 1
// otherwise: at 5,000 species with 200 of them at -61, one at a time took
// 17.8 s for main's 0.94. `out` holds the reading's state and `read_at` was
// finite there; on return `out` is the state finally read, and `read_at` has
// left its reading at that state.
template <class ReadAt>
inline void restore_where_finite(const double *y, std::vector<double> &out,
                                 const std::vector<double> &atol, std::vector<char> &kept,
                                 ReadAt &&read_at) {
    const std::size_t n = out.size();
    if (atol.size() != n)
        return;
    if (kept.size() != n)
        kept.assign(n, 0);
    bool last_reverted = false;
    if (put_back_all(y, out, atol, kept, /*skip_needed=*/true) > 0) {
        last_reverted = read_at(out.data()) != ClampReading::finite;
        settle_put_back(out, kept, !last_reverted);
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!put_back_candidate(y[i], out[i], atol[i], kept[i]))
            continue;
        out[i] = y[i];
        last_reverted = read_at(out.data()) != ClampReading::finite;
        if (last_reverted) {
            out[i] = 0.0;
            kept[i] = static_cast<char>(kept[i] | 4);
        } else {
            kept[i] = 1;
        }
    }
    if (last_reverted)
        read_at(out.data());
}

// The retry's narrowed readings (issue #1661), for the time course
// (cvode_simulator.cpp's read_narrowed) and the run test (steady_state.cpp's
// ss_read_narrowed) alike: `y` with only the slots a row `bad_row` marks reads
// put at 0 (clamp_read_nonneg_into), and where that is not finite, with only
// the band slots such a row reads put at 0 too (read_near_zero_read_into).
// Where one is finite, restore_where_finite puts back what it can. Where
// neither is, the same two readings once more with every candidate put back
// first: a law that needs one slot at its value and another at 0, as
// k*P^1.5*log(-V) does with V at -61, is 0 * -inf read with both at 0. The
// band follows only a NaN where `band_after_infinite` is false: the run test
// refuses an infinity the clamp leaves, as it did before (ss_eval_rhs). True
// where a reading is finite, with `out` the state read and `read_at` left
// there; false otherwise, and the caller reads as before.
template <class ReadAt>
inline bool read_narrowed_into(const std::vector<char> &mask, const SlotsReadByRow &rows,
                               const std::vector<char> &bad_row, const double *y,
                               const std::vector<double> &atol, std::vector<double> &out,
                               std::vector<char> &read, std::vector<char> &kept, const char *spare,
                               bool band_after_infinite, ReadAt &&read_at) {
    const std::size_t n = mask.size();
    if (kept.size() != n)
        kept.assign(n, 0);
    auto finite_at = [&] {
        const ClampReading first = read_at(out.data());
        if (first == ClampReading::finite)
            return true;
        if (!band_after_infinite && first != ClampReading::nan_only)
            return false;
        return read_near_zero_read_into(mask, read, y, atol, out) &&
               read_at(out.data()) == ClampReading::finite;
    };
    clamp_read_nonneg_into(mask, rows, bad_row, y, out, read, spare);
    if (finite_at()) {
        restore_where_finite(y, out, atol, kept, read_at);
        return true;
    }
    if (atol.size() != n)
        return false;
    clamp_read_nonneg_into(mask, rows, bad_row, y, out, read, spare);
    if (put_back_all(y, out, atol, kept) == 0)
        return false;
    const bool finite = finite_at();
    settle_put_back(out, kept, finite);
    return finite;
}

} // namespace bngsim
