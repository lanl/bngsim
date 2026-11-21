"""A concentration carried to a pole is not clamped to 0 to read a rate law.

Where the right-hand side comes back non-finite, GH #135 evaluates it again on
the state with every concentration below 0 set to 0. That is the boundary value
for a species the solver's own error put a hair below 0 where a law that reads
it has no value (sqrt, a fractional power: NaN there, with their limit at 0). A
pole is no such edge.

BIOMD0000000338 (Wajima 2009, BioModels, CC0) has XIIa and K activating each
other through ``v*S*E/(k + E)`` with k = 1, so each has a pole at -1, and XIIa =
K = 0 is an exact, unstable solution. Before #1485 the dense LU's rounding could
tip XIIa below 0, and the model's own laws then carried it down to the pole.
There the RHS is -inf, and clamped it read 0 for XIIa and for K, which held both
at -1. On main (d1b27d5d, arm64) the dense run over t 0..0.08 returned XIIa = -1
and K = -0.997 with no error, 1e9 atols below 0. Before #1302 the x86-64 CI's run
over 0..2 did the same, and its agreement with KLU (XIIa = 0, the exact solution)
passed the path-agreement check. Since #1485 the dense solve keeps XIIa = K = 0
exactly, as KLU does, so the test starts XIIa a hair below 0 itself.

Where the value read is infinite, a slot the solver's step started more than
1e6 atols below 0 is now left as it is, and where that is what keeps the value
non-finite, the run is refused, naming the slot. A NaN is clamped as before.

The oracle is the property itself: a run either raises, naming the
concentration, or returns no concentration carried 1e6 atols below 0 to a pole.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
BIOMD338 = DATA / "BIOMD0000000338.xml"
# Two corpus models whose runs met a pole or a far-off iterate the clamp used to
# read (BioModels, CC0; each at its rr-parity job's own window and tolerances).
BIOMD404 = DATA / "BIOMD0000000404.xml"  # Bray 1993, chemotaxis
BIOMD810 = DATA / "BIOMD0000000810.xml"  # Ganguli 2018, tumour immunology


# The compiled right-hand side has its own retry (cvode_codegen_rhs_body). A
# model this small runs interpreted unless it is asked for, so the cases below
# ask for it where a C compiler is there.
_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
_COMPILED = pytest.param(
    {"codegen": True}, marks=pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")
)


def _took_named_route(sim, kw):
    # The route the run took, where the case names one: a compiled case that
    # fell back to the interpreter would test the interpreted retry twice.
    return "codegen" not in kw or (sim.codegen_backend != "exprtk") == kw["codegen"]


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _run_338(xiia=0.0, *, t_end=0.08, n_points=81, **kw):
    model = bngsim.Model.from_sbml(str(BIOMD338))
    model.set_concentration("XIIa", xiia)
    sim = bngsim.Simulator(model, method="ode", force_dense_linear_solver=True, **kw)
    return sim.run(t_span=(0.0, t_end), n_points=n_points, rtol=1e-6, atol=1e-9, max_steps=100000)


# A hair below 0: 1e-3 of XIIa's atol. The laws carry it from there to
# the pole at -1 by t = 0.0096 on every Jacobian, and on KLU too.
TIPPED = -1e-12


@pytest.mark.parametrize("jacobian", ["fd", "auto", "analytical"])
def test_338_carried_to_its_pole_raises_rather_than_returning_it(jacobian):
    # The refusal names XIIa, whose own reading is what is infinite (K, at
    # -0.996, is short of its pole, and its law finite).
    with pytest.raises(bngsim.SimulationError) as info:
        _run_338(TIPPED, jacobian=jacobian)
    msg = str(info.value)
    assert "carried concentration 'XIIa'" in msg, msg
    assert "times its atol below 0" in msg, msg
    assert "'K' to" not in msg, msg


@pytest.mark.parametrize("jacobian", ["fd", "auto", "analytical"])
def test_338_on_dense_keeps_the_exact_equilibrium_at_0(jacobian):
    # #1485: XIIa's and K's rows of the Newton matrix read only each other, and
    # their right-hand side is 0, so the dense solve gives them exactly 0, as
    # KLU's block-triangular form does. Before it, the LU's rounding tipped XIIa
    # off 0: on arm64, up to 72 and Fg = 0.73 at t = 2 on this window, against
    # KLU's 311. #1304's test in test_differential_shapes.py checks the same on
    # this file and on 339's, both tracked, so CI runs it.
    result = _run_338(t_end=2.0, n_points=101, jacobian=jacobian)
    species = np.asarray(result.species, dtype=float)
    names = list(result.species_names)
    assert np.all(species[:, names.index("XIIa")] == 0.0)
    assert np.all(species[:, names.index("K")] == 0.0)


def test_338_on_klu_keeps_the_exact_equilibrium_at_0():
    # The default run (KLU: 54 species) never tips XIIa: XIIa = K = 0 is the exact
    # solution, and nothing here touches a run whose RHS stays finite.
    sim = bngsim.Simulator(bngsim.Model.from_sbml(str(BIOMD338)), method="ode")
    result = sim.run(t_span=(0.0, 2.0), n_points=101, rtol=1e-6, atol=1e-9)
    species = np.asarray(result.species, dtype=float)
    names = list(result.species_names)
    assert np.all(species[:, names.index("XIIa")] == 0.0)
    assert np.all(species[:, names.index("K")] == 0.0)


def _run_404(**kw):
    sim = bngsim.Simulator(bngsim.Model.from_sbml(str(BIOMD404)), method="ode", **kw)
    result = sim.run(t_span=(0.0, 10.0), n_points=1001, rtol=1e-6, atol=1e-9, max_steps=100000)
    return sim, np.asarray(result.species, dtype=float)


def test_404_a_failed_closed_form_attempt_still_retries_on_the_difference_quotient():
    # On arm64 the closed-form attempt collapses its step to 1e-156 with MYpYpYp
    # at -1.6e147; whatever ends it, the GH #176 retry runs, and its answer is
    # the difference quotient's, bit for bit. On x86-64 the attempt succeeds,
    # within 3e-7 of that answer.
    sim, species = _run_404()
    _, fd = _run_404(jacobian="fd")
    if sim.jacobian_strategy == "fd":
        assert np.array_equal(species, fd)
    else:
        assert np.abs(species - fd).max() < 1e-5 * np.abs(fd).max()
    assert np.all(np.isfinite(species)) and species.min() > -1e-6


def _run_810(**kw):
    sim = bngsim.Simulator(bngsim.Model.from_sbml(str(BIOMD810)), method="ode", **kw)
    return sim.run(t_span=(0.0, 50.0), n_points=101, rtol=1e-5, atol=1e-9, max_steps=100000)


def test_810_on_the_difference_quotient_is_not_returned_parked_at_its_pole():
    # On arm64 main returned a cell of 3.3e26 here, where the default run (which
    # agrees with RoadRunner) has 3.4: the run carried Interferon_gamma to -3.02,
    # the pole of `/(Interferon_gamma + k4)`, and the clamp read it at 0 there. It
    # is refused, naming it. On x86-64 the run crosses that pole without landing
    # on it and returns 9.9e15 on main and here alike (issue #1339): nothing it
    # returns is below 0, which is all this asserts there.
    try:
        result = _run_810(jacobian="fd")
    except bngsim.SimulationError as exc:
        assert "carried concentration 'Interferon_gamma'" in str(exc), str(exc)
        return
    assert np.asarray(result.species, dtype=float).min() >= -1e6 * 1e-9


def test_810_by_default_is_untouched():
    species = np.asarray(_run_810().species, dtype=float)
    assert np.all(np.isfinite(species))
    assert species.min() >= 0.0 and species.max() < 1e10


# A NaN is clamped as before: its value at 0 is the law's own limit, and the
# depth of an in-tolerance undershoot has no scale to be judged on.
HILL = "species S = 100; species P = 0\nR1: S -> P; 50*S^2.5/(0.01^2.5 + S^2.5)\n"


def test_a_predictor_overshooting_a_fractional_power_is_clamped_as_before():
    # Predicted to S = -9.7 (1e9 atols) on the straight line of its saturated
    # phase; S itself never leaves 0. bngsim's defaults (rtol = atol = 1e-8).
    from scipy.integrate import solve_ivp

    def g(s):
        s = max(s, 0.0)
        return 50.0 * s**2.5 / (0.01**2.5 + s**2.5)

    t = np.linspace(0.0, 20.0, 101)
    ref = solve_ivp(
        lambda _t, y: [-g(y[0]), g(y[0])],
        (0.0, 20.0),
        [100.0, 0.0],
        method="LSODA",
        t_eval=t,
        rtol=1e-12,
        atol=1e-14,
    ).y.T
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(HILL), "ode")
    result = sim.run(t_span=(0.0, 20.0), n_points=101)
    species = np.asarray(result.species, dtype=float)
    names = list(result.species_names)
    assert np.abs(species[:, names.index("S")] - ref[:, 0]).max() < 1e-6
    assert np.abs(species[:, names.index("P")] - ref[:, 1]).max() < 1e-6


def test_an_extinction_that_stops_rtols_of_its_start_below_0_is_clamped():
    # X' = -k X^a empties X at t = 2. From X0 = 1 it stops 8.2 rtols of that below
    # 0, 8e7 atols at atol 1e-12: inside the run's tolerance.
    x0, a, rtol, atol = 1.0, 0.05, 1e-5, 1e-12
    k = x0 ** (1.0 - a) / ((1.0 - a) * 2.0)
    text = f"species X = {x0}; species Y = 0\nR1: X -> ; {k!r}*X^{a}\nR2: -> Y; X^{a}\n"
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), "ode")
    result = sim.run(t_span=(0.0, 5.0), n_points=51, rtol=rtol, atol=atol, max_steps=100000)
    t = np.asarray(result.time)
    exact = np.maximum(0.0, x0 ** (1.0 - a) - (1.0 - a) * k * t) ** (1.0 / (1.0 - a))
    x = np.asarray(result.species, dtype=float)[:, list(result.species_names).index("X")]
    assert np.abs(x - exact).max() / x0 < 10 * rtol
    assert x.min() < -1e6 * atol  # the depth this guards


def test_one_species_overflow_does_not_get_another_refused():
    # X empties through X^0.05 and stops far below 0, where its laws are only NaN
    # (clamped as on main). W empties through W^0.5, and exp(-0.01*(A/W - 1))
    # (BIOMD0000000554's form) overflows at W's undershoot. Spared only where its
    # own reading is infinite, X is read at 0 as on main and the run returns.
    x0, ax, w0, aw = 10.0, 0.05, 1e-3, 0.5
    kx = x0 ** (1 - ax) / ((1 - ax) * 2.0)
    kw = w0 ** (1 - aw) / ((1 - aw) * 3.0)
    text = (
        f"species X = {x0}; species W = {w0}; species Y = 0; species Z = 0; A = 1\n"
        f"R1: X -> ; {kx!r}*X^{ax}\nR2: -> Y; X^{ax}\n"
        f"R3: W -> ; {kw!r}*W^{aw}\nR5: -> Z; exp(-0.01*(A/W - 1))\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), "ode")
    result = sim.run(t_span=(0.0, 5.0), n_points=51, rtol=1e-6, atol=1e-10, max_steps=100000)
    t = np.asarray(result.time)
    x = np.asarray(result.species, dtype=float)[:, list(result.species_names).index("X")]
    exact = np.maximum(0.0, x0 ** (1 - ax) - (1 - ax) * kx * t) ** (1 / (1 - ax))
    assert np.abs(x - exact).max() / x0 < 20 * 1e-6


# ─── An overflow the run carried a species to (issue #1679) ──────────────────
#
# X' = -(1 + X^2) from X = 1 is X(t) = tan(pi/4 - t): it crosses 0 at pi/4 and
# has no value from t* = 3 pi/4 on. Near -1e154 X*X overflows; the retry read X
# at 0 there, where the law is -1, and the run returned X(3) = -3.5e+286 with
# no error. The same run was refused with X declared negative, and where it
# leaves through plus infinity. An overflow in the slot's own law that is finite
# one atol below 0 is no edge at 0, so the slot is left where it is and CVODE
# cuts the step: the run ends at t*.
#
# The models are .net text, so the file needs no Antimony for these.


def _blowup_net(law, x0=1.0, k=1.0, bystander=False):
    # X' = f(), with f written over the observable Xtot = X; Y, where asked for,
    # decays beside it at 0.1 and reads nothing of X.
    y_species = "    2 Y() 1\n" if bystander else ""
    y_decay = "    2 2 0 d\n" if bystander else ""
    return (
        f"begin parameters\n    1 X0 {x0!r}\n    2 k {k!r}\n    3 d 0.1\nend parameters\n"
        f"begin functions\n    1 f() {law}\nend functions\n"
        f"begin species\n    1 X() X0\n{y_species}end species\n"
        f"begin reactions\n    1 0 1 f\n{y_decay}end reactions\n"
        "begin groups\n    1 Xtot 1\nend groups\n"
    )


def _blowup_sim(tmp_path, law, *, x0=1.0, k=1.0, bystander=False, **kw):
    net = tmp_path / "blowup.net"
    net.write_text(_blowup_net(law, x0, k, bystander))
    return bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode", **kw)


def _refused_at(info):
    # The time the failure message gives for the non-finite reading.
    import re

    msg = str(info.value)
    found = re.search(r"non-finite value at t=([0-9.eE+-]+?)\.? ", msg)
    assert found, msg
    # CVODE's own failure, after cutting the step until none is left: not the
    # refusal that ends a run at a pole, which says where the run "has carried"
    # a species.
    assert "has carried" not in msg, msg
    return float(found.group(1))


QUADRATIC = "-(k+Xtot*Xtot)"  # X' = -(k + X^2)
T_STAR = 3.0 * np.pi / 4.0  # from X = 1 at k = 1


@pytest.mark.parametrize(
    "kw",
    [{}, {"jacobian": "analytical"}, {"jacobian": "fd"}, {"codegen": False}, _COMPILED],
    ids=["default", "analytical", "fd", "interpreted", "compiled"],
)
def test_a_species_run_to_minus_infinity_is_refused_at_its_blowup(tmp_path, kw):
    # Main returned [1, -0.218, -2.69, -3.5e+286] here on the closed-form
    # Jacobian, compiled and interpreted, and refused it on the difference
    # quotient.
    sim = _blowup_sim(tmp_path, QUADRATIC, **kw)
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run((0.0, 3.0), 4)
    assert _took_named_route(sim, kw)
    assert abs(_refused_at(info) - T_STAR) < 1e-4
    # The failure names the law and the way it left.
    assert "f() = -(k+Xtot*Xtot) -> -inf" in str(info.value), str(info.value)


def test_it_is_refused_where_the_same_law_declared_negative_is(tmp_path):
    # From X = -0.5 the slot is declared negative and never was clamped: the
    # blow-up is at pi/2 - atan(0.5), and the failure is the same kind.
    declared = _blowup_sim(tmp_path, QUADRATIC, x0=-0.5)
    with pytest.raises(bngsim.SimulationError) as info_declared:
        declared.run((0.0, 3.0), 4)
    assert abs(_refused_at(info_declared) - (np.pi / 2 - np.arctan(0.5))) < 1e-4
    carried = _blowup_sim(tmp_path, QUADRATIC)
    with pytest.raises(bngsim.SimulationError) as info_carried:
        carried.run((0.0, 3.0), 4)
    for info in (info_declared, info_carried):
        # "The compiled RHS" where the run is compiled (a forced threshold or
        # JIT): the same refusal, named for the right-hand side that gave it.
        assert re.search(r"The (compiled )?RHS returned a non-finite value", str(info.value)), (
            info.value
        )


def test_a_run_started_where_its_law_overflows_is_refused_on_its_first_step(tmp_path):
    # X written to -1e200, where X*X is infinite, and far past where it first
    # overflows (1.3e154): the law is finite one atol below 0, not halfway to 0.
    # Main read X at 0 and returned it unmoved at -1e200; declared at -1e200 the
    # same model was refused at t = 0. It is refused on the first step now
    # (2e-5): the reading CVODE takes before any step has no step to be judged
    # by, and is clamped as before.
    sim = _blowup_sim(tmp_path, QUADRATIC)
    sim.model.set_concentration("X()", -1e200)
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run((0.0, 1.0), 3)
    assert _refused_at(info) < 1e-3


@pytest.mark.parametrize("t_end", [1.0, 2.0, 2.3])
def test_up_to_its_blowup_the_run_is_the_closed_form(tmp_path, t_end):
    # Below 0 all the way from pi/4, and finite: nothing is retried, and at
    # 2.3 X is -17.8. bngsim's defaults (rtol = atol = 1e-8); the error grows
    # as the solution steepens, to 6e-6 of X a tenth short of the blow-up.
    result = _blowup_sim(tmp_path, QUADRATIC).run((0.0, t_end), 11)
    t = np.asarray(result.time)
    x = np.asarray(result.species, dtype=float)[:, 0]
    exact = np.tan(np.pi / 4 - t)
    assert np.abs(x - exact).max() < 1e-4 * np.abs(exact).max()


def test_an_exponential_that_overflows_below_zero_is_refused(tmp_path):
    # X' = -(1 + exp(-X)): exp(X) = (e + 1) exp(-t) - 1, which reaches 0 at
    # t* = ln(e + 1) = 1.3133. Main returned X(2) = -711.16 and X(3) = -713.16,
    # the slope -1 the law has at 0, with Y beside it or without.
    sim = _blowup_sim(tmp_path, "-(k+exp(-Xtot))", bystander=True)
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run((0.0, 3.0), 4)
    assert abs(_refused_at(info) - np.log(np.e + 1.0)) < 1e-4


def test_the_issues_growth_law_below_its_threshold_is_refused(tmp_path):
    # X' = X (1 - X) - 0.01 from 0.001, below the lower root r1 = 0.0101: it
    # leaves through minus infinity at ln((r2 - X0)/(r1 - X0))/(r2 - r1) = 4.785.
    # Main returned -1.8e+295 at t = 10.
    r1, r2 = (1 - np.sqrt(0.96)) / 2, (1 + np.sqrt(0.96)) / 2
    t_star = np.log((r2 - 0.001) / (r1 - 0.001)) / (r2 - r1)
    sim = _blowup_sim(tmp_path, "Xtot*(1-Xtot)-0.01*k", x0=0.001)
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run((0.0, 10.0), 5)
    assert abs(_refused_at(info) - t_star) < 1e-4


def test_a_batch_row_run_to_minus_infinity_is_refused(tmp_path):
    # Main returned the k = 1 row, -3.5e+286 at its end, and refused the k = 2
    # row at its own blow-up, 1.546: which of the two it did went by the row.
    sim = _blowup_sim(tmp_path, QUADRATIC)
    with pytest.raises(bngsim.SimulationError, match="Batch simulation 0 failed"):
        sim.run_batch((0.0, 3.0), 4, params=[{"k": 1.0}, {"k": 2.0}])


def test_a_run_to_steady_state_is_refused_too(tmp_path):
    sim = _blowup_sim(tmp_path, QUADRATIC)
    with pytest.raises(bngsim.SimulationError) as info:
        sim.run((0.0, 3.0), 4, steady_state=True)
    assert abs(_refused_at(info) - T_STAR) < 1e-4


@pytest.mark.parametrize(
    ("law", "start"),
    [("exp(-1/Wtot)", -1e-3), ("Wtot*Wtot*sqrt(Wtot*(Wtot + 1))", -1e200)],
    ids=["infinite up to 0", "NaN below 0"],
)
def test_an_overflow_with_an_edge_at_0_is_clamped_as_before(tmp_path, law, start):
    # Two laws whose value at 0, which is 0, is their limit from above, each
    # infinite where W starts and not finite one atol below 0:
    # - exp(-1/W) overflows for every W in (-1.4e-3, 0), all the way up to 0;
    # - W^2 sqrt(W (W + 1)) overflows at -1e200 and is NaN for W in (-1, 0).
    # A run that starts W there, far more than 1e6 atols below 0, reads it at 0
    # as on main: Z, fed by that law alone, stays at 0, and W, which nothing
    # moves, stays where it was put.
    net = tmp_path / "edge.net"
    net.write_text(
        "begin parameters\n    1 W0 1\nend parameters\n"
        f"begin functions\n    1 source() {law}\nend functions\n"
        "begin species\n    1 W() W0\n    2 Z() 0\nend species\n"
        "begin reactions\n    1 0 2 source\nend reactions\n"
        "begin groups\n    1 Wtot 1\nend groups\n"
    )
    model = bngsim.Model.from_net(str(net))
    model.set_concentration("W()", start)
    sim = bngsim.Simulator(model, "ode")
    species = np.asarray(sim.run((0.0, 5.0), 6, rtol=1e-6, atol=1e-12).species, dtype=float)
    assert np.all(species[:, 0] == start)
    assert np.all(species[:, 1] == 0.0)


@pytest.mark.parametrize(
    "kw",
    [{}, {"jacobian": "analytical"}, {"jacobian": "fd"}, {"codegen": False}, _COMPILED],
    ids=["default", "analytical", "fd", "interpreted", "compiled"],
)
def test_another_species_overflow_at_an_undershoot_is_clamped_as_before(tmp_path, kw):
    # W at -8e-5 is 8e7 atols below 0 at atol 1e-12, but inside the tolerance of
    # a species emptied from 1 at rtol 1e-5 (the extinction case above).
    # exp(-W/1e-7), which feeds Z, is exp(800) = inf there and finite one atol
    # below 0. W's own law (nothing moves it) is finite, so this is no solution
    # carried out through minus infinity: W is read at 0, where the law is 1, and
    # the run returns Z = t, as on main. Kept at -8e-5 it was refused at its
    # first step.
    net = tmp_path / "undershoot.net"
    net.write_text(
        "begin parameters\n    1 W0 1\nend parameters\n"
        "begin functions\n    1 source() exp(-Wtot/1e-7)\nend functions\n"
        "begin species\n    1 W() W0\n    2 Z() 0\nend species\n"
        "begin reactions\n    1 0 2 source\nend reactions\n"
        "begin groups\n    1 Wtot 1\nend groups\n"
    )
    model = bngsim.Model.from_net(str(net))
    model.set_concentration("W()", -8e-5)
    sim = bngsim.Simulator(model, "ode", **kw)
    result = sim.run((0.0, 1.0), 3, rtol=1e-6, atol=1e-12)
    assert _took_named_route(sim, kw)
    species = np.asarray(result.species, dtype=float)
    assert np.all(species[:, 0] == -8e-5)
    assert np.abs(species[:, 1] - np.asarray(result.time)).max() < 1e-6


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"jacobian": "analytical"},
        {"jacobian": "fd"},
        {"codegen": False},
        {"jacobian": "analytical", "force_sparse_linear_solver": True},
        _COMPILED,
    ],
    ids=["default", "analytical", "fd", "interpreted", "analytical-klu", "compiled"],
)
def test_an_undershoot_whose_own_law_overflows_is_clamped_as_before(tmp_path, kw):
    # The case above with W the reactant: W -> Z at W*exp(-W/1e-7), so W's own
    # derivative, -W*exp(-W/1e-7), is infinite too at -8e-5, but +inf: the law
    # pushes W back up toward 0, where it is 0. That is the undershoot the clamp
    # is for, not a solution carried out through minus infinity, which its own
    # law drives on down. Main read W at 0 and returned the extinct run, W held
    # and Z = 0; marked carried, the step was cut until CVODE gave up at t = 0.5
    # (ultra review of #1751).
    if kw.get("force_sparse_linear_solver") and not getattr(bngsim, "HAS_KLU", False):
        pytest.skip("KLU sparse solver not built")
    net = tmp_path / "undershoot.net"
    net.write_text(
        "begin parameters\n    1 W0 1\nend parameters\n"
        "begin functions\n    1 source() exp(-Wtot/1e-7)\nend functions\n"
        "begin species\n    1 W() W0\n    2 Z() 0\nend species\n"
        "begin reactions\n    1 1 2 source\nend reactions\n"
        "begin groups\n    1 Wtot 1\nend groups\n"
    )
    model = bngsim.Model.from_net(str(net))
    model.set_concentration("W()", -8e-5)
    sim = bngsim.Simulator(model, "ode", **kw)
    result = sim.run((0.0, 1.0), 3, rtol=1e-6, atol=1e-12)
    assert _took_named_route(sim, kw)
    species = np.asarray(result.species, dtype=float)
    assert np.all(species[:, 0] == -8e-5)
    assert np.all(species[:, 1] == 0.0)


@pytest.mark.parametrize("kw", [{"codegen": False}, _COMPILED], ids=["interpreted", "compiled"])
def test_its_own_law_infinite_up_to_0_is_clamped_as_before(tmp_path, kw):
    # X' = -exp(-1/X), X written to -1e-3: the infinity is in X's own row and
    # drives it on down, as a solution carried out through minus infinity does,
    # but it reaches all the way up to 0 (exp(1/atol) one atol below it), so 0 is
    # the law's own edge: its value there, -exp(-inf) = 0, is its limit from
    # above. That is what the reading one atol below 0 is for. X is read at 0,
    # nothing moves it, and the run returns it where it was put, as on main.
    # Asked at 0 itself the law is finite, X would count as carried, and the
    # run would be refused.
    sim = _blowup_sim(tmp_path, "-exp(-1/Xtot)", **kw)
    sim.model.set_concentration("X()", -1e-3)
    result = sim.run((0.0, 5.0), 6, rtol=1e-6, atol=1e-12)
    assert _took_named_route(sim, kw)
    assert np.all(np.asarray(result.species, dtype=float)[:, 0] == -1e-3)
