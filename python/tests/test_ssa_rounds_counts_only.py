"""SSA/PSA round molecule counts, and nothing else (issue #718).

The SBML loader gives a state slot to quantities that are not molecule counts:
a parameter or compartment that an event assigns (GH #71) and a rate-rule
target (GH #81). The SSA rounded every slot to a whole number at the start of
every run, so a rate constant of 0.5 ran as 1 and 0.4 as 0, and a rate-rule
variable was put back on a whole number at every ``run_until`` leg, with no
warning after Simulator construction.

Oracles are closed forms that do not depend on the SSA: for zeroth- and
first-order kinetics the SSA mean equals the ODE solution exactly.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

METHODS = [
    pytest.param("ssa", {}, id="ssa"),
    pytest.param("ssa", {"codegen": False}, id="ssa-interpreted"),
    pytest.param("psa", {"poplevel": 100}, id="psa"),
]


def _ensemble(ant, t_end, method, kw, reps=400, seed=3):
    m = bngsim.Model.from_antimony_string(ant)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.SsaRoundingWarning)
        r = bngsim.Simulator(m, method=method, **kw).run_replicates(
            reps, t_span=(0, t_end), n_points=3, seed=seed, squeeze=True
        )
    assert r.ssa_diagnostics["n_rounded_populations"] == 0
    return r


def _final(r, name):
    return np.asarray(r.species)[:, -1, list(r.species_names).index(name)]


@pytest.mark.parametrize(("method", "kw"), METHODS)
@pytest.mark.parametrize("k", [0.5, 0.4])
def test_event_assigned_rate_constant_keeps_its_value(method, kw, k):
    """``A -> ∅`` at ``k·A`` from 1000, with an event (never reached) that
    assigns k, so k is promoted to a state slot. E[A(2)] = 1000·e^{-2k}. Before
    the fix k = 0.5 ran as 1 (A(2) = 135 against 368) and k = 0.4 as 0 (A never
    decayed)."""
    ant = f"compartment C=1; species A in C=1000; k={k}; J: A => ; k*A; E: at (time>=50): k=0.1;"
    r = _ensemble(ant, 2.0, method, kw)
    a = _final(r, "A")
    p = np.exp(-2 * k)
    want, se = 1000 * p, np.sqrt(1000 * p * (1 - p) / a.size)
    assert abs(a.mean() - want) <= 4.5 * se, (a.mean(), want, se)


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_event_assigned_zeroth_order_rate(method, kw):
    """``∅ -> S`` at X = 0.3, X event-assigned: S(10) is Poisson(3). X was
    rounded to 0 and S stayed 0."""
    r = _ensemble("species S=0; X=0.3; J1: => S; X; E: at (time>=50): X=1;", 10.0, method, kw)
    s = _final(r, "S")
    assert abs(s.mean() - 3.0) <= 4.5 * np.sqrt(3.0 / s.size), s.mean()


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_rate_rule_target_starts_at_its_value(method, kw):
    """``X' = 0.1`` from X = 1.4 drives ``∅ -> S`` at X: E[S(10)] = 14 + 5 = 19.
    X started at 1 instead (S = 14.9)."""
    r = _ensemble("species S=0; X=1.4; X'=0.1; J1: => S; X;", 10.0, method, kw)
    assert np.allclose(np.asarray(r.species)[:, 0, list(r.species_names).index("X")], 1.4)
    s = _final(r, "S")
    assert abs(s.mean() - 19.0) <= 4.5 * np.sqrt(19.0 / s.size), s.mean()


@pytest.mark.parametrize(("method", "kw"), METHODS)
def test_rate_rule_compartment_keeps_its_size(method, kw):
    """A rate-rule compartment of size 0.4 was rounded to 0, which also turned
    off the live-volume propensity correction. It starts at its declared size
    and grows as its rule says."""
    ant = "compartment Cc = 0.4; Cc' = 0.1; species A in Cc = 1000; J: A => ; k*A*Cc; k = 0.1;"
    m = bngsim.Model.from_antimony_string(ant)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.SsaRoundingWarning)
        r = bngsim.Simulator(m, method=method, **kw).run(t_span=(0, 2.0), n_points=3, seed=1)
    cc = np.asarray(r.species)[:, list(r.species_names).index("Cc")]
    np.testing.assert_allclose(cc, [0.4, 0.5, 0.6], rtol=1e-12)


def test_run_until_legs_carry_a_rate_rule_variable():
    """``X' = 1`` stepped in 0.3-long ``run_until`` legs. X was re-rounded to 0
    at every leg, so it never passed 0.3. Each leg must also hand the next one
    the value it reported at its end: the state was written back as of the last
    event, up to one sub-step before t_end, and X reached 3.898 at t = 3.9."""
    m = bngsim.Model.from_antimony_string("species S=0; X=0; X'=1; J1: => S; 0.5*X;")
    sim = bngsim.Simulator(m, method="ssa")
    names = None
    for k in range(1, 14):
        with warnings.catch_warnings():
            warnings.simplefilter("error", bngsim.SsaRoundingWarning)
            r = sim.run_until(0.3 * k, n_points=2, seed=100 + k)
        names = names or list(r.species_names)
        x = np.asarray(r.species)[:, names.index("X")]
        np.testing.assert_allclose(x, [0.3 * (k - 1), 0.3 * k], rtol=1e-12, atol=1e-12)


def test_validator_does_not_call_a_parameter_a_population():
    """The pre-flight validator applies the engine's rule: an event-assigned
    rate constant is not a population, a fractional molecule count still is."""
    m = bngsim.Model.from_antimony_string(
        "compartment C=1; species A in C=2.5; k=0.5; J: A => ; k*A; E: at (time>=50): k=0.1;"
    )
    flagged = {
        i.location for i in m.validate_for_ssa() if i.code == "non_integer_initial_population"
    }
    assert flagged == {"species:A"}, flagged


# ── Rounding a molecule count is reported on every leg ───────────────────────


def _net(tmp_path, a0):
    path = tmp_path / "decay.net"
    path.write_text(
        "begin parameters\n    1 k 0.1\nend parameters\n"
        f"begin species\n    1 A() {a0}\nend species\n"
        "begin reactions\n    1 1 0 k\nend reactions\n"
        "begin groups\n    1 Atot 1\nend groups\n"
    )
    return bngsim.Model.from_net(str(path))


def test_rounding_a_count_is_reported(tmp_path):
    """A fractional count is still rounded, as run_network rounds it (the
    GH #118 disposition depends on that), and each run that rounds says so."""
    m = _net(tmp_path, 5.7)
    sim = bngsim.Simulator(m, method="ssa")
    with pytest.warns(bngsim.SsaRoundingWarning, match=r"1 molecule count.*first: A\(\)"):
        r = sim.run(t_span=(0, 1), n_points=2, seed=1)
    assert r.species[0][0] == 6.0
    assert r.ssa_diagnostics["n_rounded_populations"] == 1
    assert r.ssa_diagnostics["first_rounded_species"] == "A()"


def test_rounding_on_a_later_leg_is_reported(tmp_path):
    """A count set between ``run_until`` legs is rounded at the next leg; that
    used to happen with no notice at all."""
    m = _net(tmp_path, 10)
    sim = bngsim.Simulator(m, method="ssa")
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.SsaRoundingWarning)
        sim.run_until(1.0, n_points=2, seed=1)
    m.set_concentration("A()", 3.4)
    with pytest.warns(bngsim.SsaRoundingWarning):
        r = sim.run_until(2.0, n_points=2, seed=2)
    assert r.species[0][0] == 3.0


def test_replicates_sum_the_rounding_count(tmp_path):
    m = _net(tmp_path, 5.7)
    with pytest.warns(bngsim.SsaRoundingWarning):
        r = bngsim.Simulator(m, method="ssa").run_replicates(
            4, t_span=(0, 1), n_points=2, seed=1, squeeze=True
        )
    assert r.ssa_diagnostics["n_rounded_populations"] == 4
