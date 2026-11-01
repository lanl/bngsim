// bngsim/src/model_dual.cpp — the right-hand side and the functions on dual
// numbers, for derivatives that are exact where there is no closed form
// (issue #1330)
//
// A difference quotient of the right-hand side steps each species by 1.5e-8 of
// the larger of itself and the state's largest concentration, and reads a
// secant. Beside a species at 1e8 that step is 1.5 in a species at 0, and a law
// can be written whose secants agree at every pair of steps a check takes: a
// Hill law at 0 with K = h/√2 gave dG*/ds = 2.13 for 0, a kink just past the
// state 4.76 for 0, with jacobian="fd" or one law the closed form declines.
// Dual numbers give the derivative itself, of each law as the interpreter
// evaluates it, branch by branch (bngsim/dual.hpp).
//
// The evaluation mirrors NetworkModel::compute_derivs_core (model.cpp) step for
// step: observables, the functions in their strongly connected order, each
// reaction's rate as compute_rxn_rate forms it on the ODE path, and the net
// stoichiometry. Each value is formed as model.cpp forms it, statement for
// statement, so that a compiler that fuses a multiply into an add does so on
// both; the derivative is carried beside it. Every fill then checks the values
// against compute_derivs() at the same state (dual_matches_rhs_), so a part of
// the model this mirror does not follow is refused, not differentiated wrong.

#include "bngsim/dual.hpp"
#include "bngsim/expression.hpp"
#include "bngsim/mm_jacobian.hpp"
#include "bngsim/model.hpp"
#include "model_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bngsim {

struct NetworkModel::DualShadow {
    // What the model's expressions were when this was built: the evaluator id
    // of each parameter and function, and each table's address. Built again
    // where any of them moves (set_param detaching an expression,
    // set_function_eval_expression, a table added after load).
    std::vector<std::intptr_t> key;
    std::string unsupported;

    std::unique_ptr<ExprTkDualEvaluator> eval;
    std::vector<Dual> params;      // parallel to Impl::parameters
    std::vector<Dual> observables; // parallel to Impl::observables
    std::vector<Dual> species;     // the species names expressions may read
    Dual time;
    std::vector<int> param_ids;    // dual expression of a derived parameter, or -1
    std::vector<int> function_ids; // dual expression of a function, or -1

    // The last evaluation: f's values and derivatives, each species' terms
    // summed in absolute value (the scale a value is rounded on), and each
    // function's dual (NaN where it was not evaluated).
    std::vector<double> f, df, gross;
    std::vector<Dual> funcs;
    // Scratch for compute_derivs at the same state.
    std::vector<double> rhs;
};

namespace {

std::vector<std::intptr_t> dual_shadow_key(const NetworkModel::Impl &impl) {
    std::vector<std::intptr_t> key;
    key.reserve(impl.parameters.size() + impl.functions.size() + impl.table_functions.size() + 3);
    key.push_back(impl.uses_rateof ? 1 : 0);
    key.push_back(static_cast<std::intptr_t>(impl.parameters.size()));
    for (const auto &p : impl.parameters) {
        key.push_back(p.evaluator_id);
    }
    key.push_back(static_cast<std::intptr_t>(impl.functions.size()));
    for (const auto &fn : impl.functions) {
        key.push_back(fn.evaluator_id);
    }
    for (const auto &tf : impl.table_functions) {
        key.push_back(reinterpret_cast<std::intptr_t>(tf.get()));
    }
    return key;
}

// A table's value and slope at its index, as a dual (TableFunction): the
// interval evaluate_at reads, and at a knot the index moves across, the tie.
Dual dual_table(const TableFunction &tf, const Dual &x) {
    const double value = tf.evaluate_at(x.v);
    // The piece: a step table's value, a linear table's slope (which changes
    // only at a corner).
    if (std::isfinite(x.v)) {
        dual_note_branch(tf.method() == InterpolationMethod::Step ? value : tf.slope_at(x.v));
    }
    if (x.d == 0.0 || !std::isfinite(x.v)) {
        return Dual(value, x.d == 0.0 ? 0.0 : std::numeric_limits<double>::quiet_NaN());
    }
    if (!tf.is_knot(x.v)) {
        return Dual(value, tf.slope_at(x.v) * x.d);
    }
    // On a knot: the slope of the interval to its right is the one the value
    // reads; a step table jumps there.
    const bool corner =
        tf.method() == InterpolationMethod::Step || tf.slope_at(x.v) != tf.left_slope_at(x.v);
    if (!corner) {
        return Dual(value, tf.slope_at(x.v) * x.d);
    }
    DualContext &ctx = dual_context();
    ctx.tied = true;
    // A linear table is continuous at its knots: a side's value is the knot's,
    // not the one a rounding past it reads (slope·ulp, which beside a species
    // whose terms are nothing would read as a jump).
    const auto side = [&](bool up) {
        const double past = std::nextafter(x.v, up ? std::numeric_limits<double>::infinity()
                                                   : -std::numeric_limits<double>::infinity());
        const double side_value =
            tf.method() == InterpolationMethod::Step ? tf.evaluate_at(past) : value;
        return Dual(side_value, (up ? tf.slope_at(x.v) : tf.left_slope_at(x.v)) * x.d);
    };
    switch (ctx.tie) {
    case DualTie::Above:
        return side(x.d > 0.0);
    case DualTie::Below:
        return side(x.d < 0.0);
    case DualTie::Value:
    default:
        return Dual(value, tf.slope_at(x.v) * x.d);
    }
}

} // namespace

NetworkModel::DualShadow *NetworkModel::dual_shadow_() {
    // The difference quotients these replace still run for a model with no
    // dual evaluation; this switch puts any model on them, so that their
    // checks stay tested (python/tests/test_differenced_jacobian_agreement.py).
    if (std::getenv("BNGSIM_NO_DUAL_NUMBERS") != nullptr) {
        thread_local DualShadow off;
        off.unsupported = "BNGSIM_NO_DUAL_NUMBERS is set";
        return &off;
    }
    std::vector<std::intptr_t> key = dual_shadow_key(*impl_);
    if (impl_->dual_shadow && impl_->dual_shadow->key == key) {
        return impl_->dual_shadow.get();
    }
    auto shadow = std::make_shared<DualShadow>();
    shadow->key = std::move(key);
    auto &s = *shadow;
    auto &impl = *impl_;
    const std::size_t np = impl.parameters.size();
    const std::size_t no = impl.observables.size();
    const std::size_t ns = impl.species.size();
    s.params.assign(np, Dual());
    s.observables.assign(no, Dual());
    s.species.assign(ns, Dual());
    s.param_ids.assign(np, -1);
    s.function_ids.assign(impl.functions.size(), -1);
    s.funcs.assign(impl.functions.size(), Dual());
    s.f.assign(ns, 0.0);
    s.df.assign(ns, 0.0);
    s.gross.assign(ns, 0.0);

    const auto refuse = [&](std::string why) {
        s.unsupported = std::move(why);
        s.eval.reset();
        impl_->dual_shadow = shadow;
        return impl_->dual_shadow.get();
    };

    if (impl.uses_rateof) {
        return refuse("it reads rateOf(), whose derivative is the derivative of the right-hand "
                      "side itself");
    }
    const auto &starts = impl.shared->function_scc_starts;
    for (std::size_t g = 0; g + 1 < starts.size(); ++g) {
        if (starts[g + 1] - starts[g] != 1) {
            return refuse("a group of its functions read each other, which no order of "
                          "evaluation resolves (issue #621)");
        }
    }

    // NetworkModel::clone()'s order of registration, name for name, so that
    // each key is the one the interpreter's evaluator gave the same name.
    s.eval = std::make_unique<ExprTkDualEvaluator>();
    s.eval->set_time_ptr(&s.time);
    for (std::size_t i = 0; i < np; ++i) {
        s.eval->try_define_variable(impl.parameters[i].name, &s.params[i]);
    }
    for (std::size_t j = 0; j < no; ++j) {
        s.eval->try_define_variable(impl.observables[j].name, &s.observables[j]);
    }
    for (std::size_t i = 0; i < ns; ++i) {
        s.eval->try_define_variable(impl.species[i].name, &s.species[i]);
    }
    try {
        for (std::size_t i = 0; i < np; ++i) {
            const auto &p = impl.parameters[i];
            if (!p.expression.empty() && p.evaluator_id >= 0) {
                s.param_ids[i] =
                    s.eval->compile_preprocessed(impl.evaluator->preprocessed_expr(p.evaluator_id));
            }
        }
        // A table reads its index through a pointer into the model: the clock,
        // a parameter or an observable. Its dual reads the same slot here.
        for (const auto &tf_ptr : impl.table_functions) {
            const TableFunction *tf = tf_ptr.get();
            const double *index = tf->index_ptr();
            const Dual *slot = nullptr;
            if (index == &impl.current_time) {
                slot = &s.time;
            }
            for (std::size_t i = 0; slot == nullptr && i < np; ++i) {
                if (index == &impl.parameters[i].value) {
                    slot = &s.params[i];
                }
            }
            for (std::size_t j = 0; slot == nullptr && j < no; ++j) {
                if (index == &impl.observables[j].total) {
                    slot = &s.observables[j];
                }
            }
            if (slot == nullptr) {
                return refuse("table function '" + tf->name() +
                              "' reads an index the dual evaluation does not hold");
            }
            if (!s.eval->define_function("tfun_" + tf->name(),
                                         [tf, slot]() { return dual_table(*tf, *slot); })) {
                return refuse("table function '" + tf->name() + "' could not be registered");
            }
        }
        for (std::size_t k = 0; k < impl.functions.size(); ++k) {
            const auto &fn = impl.functions[k];
            if (fn.evaluator_id >= 0) {
                s.function_ids[k] = s.eval->compile_preprocessed(
                    impl.evaluator->preprocessed_expr(fn.evaluator_id));
            }
        }
    } catch (const std::exception &e) {
        return refuse(std::string("an expression of it does not compile on dual numbers: ") +
                      e.what());
    }
    impl_->dual_shadow = shadow;
    return impl_->dual_shadow.get();
}

std::string NetworkModel::dual_unsupported_reason() { return dual_shadow_()->unsupported; }

// One evaluation on duals at (t, conc), the species moving along `dconc` (null:
// none do) and the parameter `seed_param` along itself (-1: none). Fills
// shadow.f / df / gross and shadow.funcs; with `functions_only`, the functions
// alone. False where the model has no dual evaluation.
bool NetworkModel::dual_evaluate_(DualShadow &s, double t, const double *conc, const double *dconc,
                                  int seed_param, bool functions_only, bool with_deferred) {
    if (!s.eval) {
        return false;
    }
    auto &impl = *impl_;
    const int ns = n_species();
    const int np = static_cast<int>(impl.parameters.size());

    for (int i = 0; i < np; ++i) {
        s.params[static_cast<std::size_t>(i)] =
            Dual(impl.parameters[static_cast<std::size_t>(i)].value, i == seed_param ? 1.0 : 0.0);
    }
    // A parameter derived from the one that moves moves with it, in the order
    // refresh_derived_params() takes, the moving one held against its own
    // expression as sync_params(held) holds it.
    if (seed_param >= 0) {
        for (const int pi : impl.shared->derived_param_order) {
            const auto &p = impl.parameters[static_cast<std::size_t>(pi)];
            const int id = s.param_ids[static_cast<std::size_t>(pi)];
            if (pi != seed_param && p.is_expression && p.evaluator_id >= 0 && id >= 0) {
                s.params[static_cast<std::size_t>(pi)] = s.eval->evaluate(id);
            }
        }
    }
    // A species name an expression reads is the Species slot the interpreter
    // binds, which compute_derivs does not move: a constant here as there.
    for (int i = 0; i < ns; ++i) {
        s.species[static_cast<std::size_t>(i)] =
            Dual(impl.species[static_cast<std::size_t>(i)].concentration);
    }
    s.time = Dual(t);

    const auto &species_list = impl.species;
    // The state as a rate law reads it: amount for an amount-valued species
    // (compute_species_factor_ode, update_observables).
    const auto read_species = [&](int si) {
        double v = conc[si];
        double d = dconc != nullptr ? dconc[si] : 0.0;
        if (species_list[static_cast<std::size_t>(si)].amount_valued) {
            v *= species_list[static_cast<std::size_t>(si)].volume_factor;
            d *= species_list[static_cast<std::size_t>(si)].volume_factor;
        }
        return Dual(v, d);
    };

    const bool evaluate_functions_here = impl.has_functions || functions_only;
    std::fill(s.funcs.begin(), s.funcs.end(), Dual(std::numeric_limits<double>::quiet_NaN()));
    if (evaluate_functions_here) {
        // update_observables, statement for statement.
        for (std::size_t j = 0; j < impl.observables.size(); ++j) {
            double total = 0.0;
            double total_d = 0.0;
            for (const auto &entry : impl.observables[j].entries) {
                const int idx0 = entry.species_index - 1;
                if (idx0 >= 0 && idx0 < ns) {
                    const Dual v = read_species(idx0);
                    total += entry.factor * v.v;
                    total_d += entry.factor * v.d;
                }
            }
            s.observables[j] = Dual(total, total_d);
        }
        // evaluate_functions: one function a group, as dual_shadow_ checked.
        const auto &bindings = impl.shared->var_param_bindings;
        for (const auto &[func_idx, param_idx] : bindings) {
            const auto &fn = impl.functions[static_cast<std::size_t>(func_idx)];
            const int id = s.function_ids[static_cast<std::size_t>(func_idx)];
            if (fn.evaluator_id < 0 || id < 0 || (fn.deferred && !with_deferred)) {
                continue;
            }
            const Dual val = s.eval->evaluate(id);
            s.params[static_cast<std::size_t>(param_idx)] = val;
            s.funcs[static_cast<std::size_t>(func_idx)] = val;
        }
    }
    if (functions_only) {
        return true;
    }

    std::fill(s.f.begin(), s.f.end(), 0.0);
    std::fill(s.df.begin(), s.df.end(), 0.0);
    std::fill(s.gross.begin(), s.gross.end(), 0.0);
    const auto &params = s.params;
    const int nr = n_reactions();
    for (int r = 0; r < nr; ++r) {
        const auto &rxn = impl.shared->reactions[static_cast<std::size_t>(r)];
        // compute_rxn_rate, ODE path.
        double rate = 0.0;
        double rate_d = 0.0;
        switch (rxn.rate_law_type) {
        case RateLawType::Elementary:
        case RateLawType::Functional: {
            const int k_idx = rxn.rate_param_idx0;
            if (k_idx < 0 || k_idx >= np) {
                return false; // compute_derivs throws on it
            }
            const Dual k = params[static_cast<std::size_t>(k_idx)];
            rate = k.v * rxn.stat_factor;
            rate_d = k.d * rxn.stat_factor;
            if (rxn.apply_species_factor) {
                // The product, and its derivative by the product rule.
                double factor = 1.0;
                double factor_d = 0.0;
                for (const int ri : rxn.reactant_indices) {
                    const int si = ri - 1;
                    if (si >= 0 && si < ns) {
                        const Dual v = read_species(si);
                        factor_d = dual_product_derivative(factor, factor_d, v.v, v.d);
                        factor *= v.v;
                    }
                }
                rate_d = dual_product_derivative(rate, rate_d, factor, factor_d);
                rate *= factor;
            }
            break;
        }
        case RateLawType::MichaelisMenten: {
            if (rxn.rate_law_param_indices.size() < 2) {
                return false;
            }
            const int kcat_idx = rxn.rate_law_param_indices[0] - 1;
            const int km_idx = rxn.rate_law_param_indices[1] - 1;
            if (kcat_idx < 0 || kcat_idx >= np || km_idx < 0 || km_idx >= np) {
                return false;
            }
            const Dual kcat = params[static_cast<std::size_t>(kcat_idx)];
            const Dual Km = params[static_cast<std::size_t>(km_idx)];
            if (rxn.reactant_indices.size() >= 2) {
                const int e_si = rxn.reactant_indices[0] - 1;
                const int s_si = rxn.reactant_indices[1] - 1;
                if (e_si >= 0 && e_si < ns && s_si >= 0 && s_si < ns) {
                    // compute_rxn_rate reads conc[] here, not the amount.
                    const Dual E(conc[e_si], dconc != nullptr ? dconc[e_si] : 0.0);
                    const Dual S(conc[s_si], dconc != nullptr ? dconc[s_si] : 0.0);
                    const MMTqssa q = mm_tqssa(Km.v, E.v, S.v);
                    rate = q.KpsF > 0.0 ? kcat.v * rxn.stat_factor * q.sFree * E.v / q.KpsF : 0.0;
                    rate += (kcat.v - kcat.v) + (Km.v - Km.v) + (E.v - E.v) + (S.v - S.v);
                    // The partials of mm_jacobian.hpp, and ∂/∂Km = −rate/D,
                    // ∂/∂kcat = rate/kcat: 0 where the rate is degenerate.
                    if (q.KpsF > 0.0 && q.D > 0.0) {
                        const double C = kcat.v * rxn.stat_factor;
                        if (E.d != 0.0) {
                            rate_d += C * q.sFree / q.D * E.d;
                        }
                        if (S.d != 0.0) {
                            rate_d += C * E.v * Km.v / (q.KpsF * q.D) * S.d;
                        }
                        if (Km.d != 0.0) {
                            rate_d += -rate / q.D * Km.d;
                        }
                        if (kcat.d != 0.0) {
                            rate_d += rxn.stat_factor * q.sFree * E.v / q.KpsF * kcat.d;
                        }
                    }
                }
            }
            break;
        }
        }

        if (rate == 0.0 && rate_d == 0.0) {
            continue;
        }

        if (rxn.per_species_volume_scaling) {
            const auto divisor = [&](int si) {
                const int lv = species_list[static_cast<std::size_t>(si)].ode_live_volume_idx0;
                if (lv >= 0 && lv < ns) {
                    const double v_live = conc[lv];
                    if (v_live > 0.0) {
                        return Dual(v_live, dconc != nullptr ? dconc[lv] : 0.0);
                    }
                }
                return Dual(species_list[static_cast<std::size_t>(si)].volume_factor);
            };
            for (const auto &[si, m] : rxn.net_loss) {
                if (si < ns) {
                    const Dual term = Dual(rate, rate_d) / divisor(si);
                    s.f[static_cast<std::size_t>(si)] -= m == 1.0 ? term.v : m * term.v;
                    s.df[static_cast<std::size_t>(si)] -= m == 1.0 ? term.d : m * term.d;
                    s.gross[static_cast<std::size_t>(si)] += std::abs(m * term.v);
                }
            }
            for (const auto &[si, m] : rxn.net_gain) {
                if (si < ns) {
                    const Dual term = Dual(rate, rate_d) / divisor(si);
                    s.f[static_cast<std::size_t>(si)] += m == 1.0 ? term.v : m * term.v;
                    s.df[static_cast<std::size_t>(si)] += m == 1.0 ? term.d : m * term.d;
                    s.gross[static_cast<std::size_t>(si)] += std::abs(m * term.v);
                }
            }
            continue;
        }
        for (const auto &[si, m] : rxn.net_loss) {
            if (si < ns) {
                if (m == 1.0) {
                    s.f[static_cast<std::size_t>(si)] -= rate;
                    s.df[static_cast<std::size_t>(si)] -= rate_d;
                } else {
                    s.f[static_cast<std::size_t>(si)] -= m * rate;
                    s.df[static_cast<std::size_t>(si)] -= m * rate_d;
                }
                s.gross[static_cast<std::size_t>(si)] += std::abs(m * rate);
            }
        }
        for (const auto &[si, m] : rxn.net_gain) {
            if (si < ns) {
                s.f[static_cast<std::size_t>(si)] += m == 1.0 ? rate : m * rate;
                s.df[static_cast<std::size_t>(si)] += m == 1.0 ? rate_d : m * rate_d;
                s.gross[static_cast<std::size_t>(si)] += std::abs(m * rate);
            }
        }
    }
    for (const auto &sp : species_list) {
        if (sp.fixed) {
            s.f[static_cast<std::size_t>(sp.index - 1)] = 0.0;
            s.df[static_cast<std::size_t>(sp.index - 1)] = 0.0;
        }
    }
    return true;
}

// Whether the duals' values at (t, conc) are compute_derivs()'s, to a few
// roundings of each species' terms: the check that this file follows the
// right-hand side, run at every fill. A value that is no number must be no
// number on both.
bool NetworkModel::dual_matches_rhs_(DualShadow &s, double t, const double *conc) {
    const int ns = n_species();
    s.rhs.assign(static_cast<std::size_t>(ns), 0.0);
    compute_derivs(t, conc, s.rhs.data());
    for (int i = 0; i < ns; ++i) {
        const double a = s.rhs[static_cast<std::size_t>(i)];
        const double b = s.f[static_cast<std::size_t>(i)];
        if (std::isnan(a) || std::isnan(b)) {
            if (std::isnan(a) != std::isnan(b)) {
                return false;
            }
            continue;
        }
        if (a == b) {
            continue;
        }
        if (!(std::abs(a - b) <= 1e-10 * s.gross[static_cast<std::size_t>(i)])) {
            return false;
        }
    }
    return true;
}

namespace {

// Run `body` with the dual context set to `tie` and cleared of ties, and give
// back whether one was met; the context is put back as it was.
template <class Body> bool with_tie(DualTie tie, bool *tied, Body &&body) {
    DualContext &ctx = dual_context();
    const DualContext saved = ctx;
    ctx.tie = tie;
    ctx.tied = false;
    bool ok = false;
    try {
        ok = body();
    } catch (...) {
        ctx = saved;
        throw;
    }
    const bool met = ctx.tied;
    ctx = saved;
    if (tied != nullptr) {
        *tied = met;
    }
    return ok;
}

} // namespace

namespace {

// Whether an evaluation from one side of a tie gives the values the state has:
// each the same to a few roundings of `scale` (a species' terms, or the value
// itself), no number where it is none. Where it does not, a law jumps there —
// `if(q > 0, 10, 0)` at q = 0 — and has no derivative along the direction, the
// same from either side or not.
bool dual_same_value(double base, double side, double scale) {
    if (std::isnan(base) || std::isnan(side)) {
        return std::isnan(base) && std::isnan(side);
    }
    return base == side || std::abs(side - base) <= 1e-12 * scale;
}

} // namespace

bool NetworkModel::dual_branches(double t, const double *conc, std::vector<DualBranch> &out) {
    DualShadow *s = dual_shadow_();
    out.clear();
    if (!s->eval) {
        return false;
    }
    return with_tie(DualTie::Value, nullptr, [&] {
        dual_context().branches = &out;
        return dual_evaluate_(*s, t, conc, nullptr, -1, false, false);
    });
}

bool NetworkModel::fill_dense_dual_jacobian(double t, const double *conc, double *jac, DualTie tie,
                                            bool *tied, bool *jumped,
                                            std::vector<char> *jumped_columns) {
    DualShadow *s = dual_shadow_();
    if (!s->eval) {
        return false;
    }
    const int ns = n_species();
    bool any_jump = false;
    std::vector<char> column_jumps(static_cast<std::size_t>(ns), 0);
    std::vector<double> seed(static_cast<std::size_t>(ns), 0.0);
    std::vector<double> J(static_cast<std::size_t>(ns) * static_cast<std::size_t>(ns), 0.0);
    const bool ok = with_tie(tie, tied, [&] {
        // The values at the state, with no direction: checked against the
        // right-hand side, and what an evaluation from a side is held to.
        if (!dual_evaluate_(*s, t, conc, nullptr, -1, false, false) ||
            !dual_matches_rhs_(*s, t, conc)) {
            return false;
        }
        const std::vector<double> base = s->f;
        const std::vector<double> scale = s->gross;
        for (int j = 0; j < ns; ++j) {
            seed[static_cast<std::size_t>(j)] = 1.0;
            const bool done = dual_evaluate_(*s, t, conc, seed.data(), -1, false, false);
            seed[static_cast<std::size_t>(j)] = 0.0;
            if (!done) {
                return false;
            }
            // A law whose value jumps where species j crosses the state: the
            // column is the derivative of the branch on that side, as the
            // column at the state is the branch's at the state, and the column
            // is reported (`jumped_columns`). A steady state that species j
            // moves away from as a parameter moves crosses the jump and has no
            // derivative; one held where it is (a state parked on the jump by
            // laws of its own) does (ss_confirm_dual_sides).
            bool jumps = false;
            for (int i = 0; tie != DualTie::Value && i < ns && !jumps; ++i) {
                jumps = !dual_same_value(base[static_cast<std::size_t>(i)],
                                         s->f[static_cast<std::size_t>(i)],
                                         scale[static_cast<std::size_t>(i)]);
            }
            column_jumps[static_cast<std::size_t>(j)] = jumps ? 1 : 0;
            any_jump = any_jump || jumps;
            std::copy(s->df.begin(), s->df.end(), J.begin() + static_cast<std::ptrdiff_t>(j) * ns);
        }
        return true;
    });
    if (!ok) {
        return false;
    }
    std::copy(J.begin(), J.end(), jac);
    if (jumped != nullptr) {
        *jumped = any_jump;
    }
    if (jumped_columns != nullptr) {
        *jumped_columns = std::move(column_jumps);
    }
    return true;
}

bool NetworkModel::fill_dual_dfdp(double t, const double *conc, int param_idx0, double *out,
                                  DualTie tie, bool *tied) {
    DualShadow *s = dual_shadow_();
    if (!s->eval || param_idx0 < 0 || param_idx0 >= static_cast<int>(impl_->parameters.size())) {
        return false;
    }
    const int ns = n_species();
    bool jumps = false;
    const bool ok = with_tie(tie, tied, [&] {
        if (!dual_evaluate_(*s, t, conc, nullptr, -1, false, false) ||
            !dual_matches_rhs_(*s, t, conc)) {
            return false;
        }
        const std::vector<double> base = s->f;
        const std::vector<double> scale = s->gross;
        if (!dual_evaluate_(*s, t, conc, nullptr, param_idx0, false, false)) {
            return false;
        }
        for (int i = 0; tie != DualTie::Value && i < ns && !jumps; ++i) {
            jumps = !dual_same_value(base[static_cast<std::size_t>(i)],
                                     s->f[static_cast<std::size_t>(i)],
                                     scale[static_cast<std::size_t>(i)]);
        }
        return true;
    });
    if (!ok) {
        return false;
    }
    if (jumps) {
        std::fill(out, out + ns, std::numeric_limits<double>::quiet_NaN());
    } else {
        std::copy(s->df.begin(), s->df.begin() + ns, out);
    }
    return true;
}

bool NetworkModel::dual_function_derivatives(double t, const double *conc, const double *dconc,
                                             int param_idx0, double *out, DualTie tie, bool *tied) {
    DualShadow *s = dual_shadow_();
    if (!s->eval || param_idx0 >= static_cast<int>(impl_->parameters.size())) {
        return false;
    }
    std::vector<Dual> base;
    const bool ok = with_tie(tie, tied, [&] {
        if (!dual_evaluate_(*s, t, conc, nullptr, -1, false, false) ||
            !dual_matches_rhs_(*s, t, conc) ||
            !dual_evaluate_(*s, t, conc, nullptr, -1, true, true)) {
            return false;
        }
        base = s->funcs;
        return dual_evaluate_(*s, t, conc, dconc, param_idx0, true, true);
    });
    if (!ok) {
        return false;
    }
    for (std::size_t m = 0; m < s->funcs.size(); ++m) {
        const Dual &v = s->funcs[m];
        const double b = base[m].v;
        const bool jumps =
            tie != DualTie::Value && !dual_same_value(b, v.v, std::max(std::abs(b), std::abs(v.v)));
        out[m] = std::isfinite(v.v) && !jumps ? v.d : std::numeric_limits<double>::quiet_NaN();
    }
    return true;
}

} // namespace bngsim
