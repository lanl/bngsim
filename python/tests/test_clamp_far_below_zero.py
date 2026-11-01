"""A concentration carried to a pole is not clamped to 0 to read a rate law.

Where the right-hand side comes back non-finite, GH #135 evaluates it again on
the state with every concentration below 0 set to 0. That is the boundary value
for a species the solver's own error put a hair below 0 where a law that reads
it has no value (sqrt, a fractional power: NaN there, with their limit at 0). A
pole is no such edge.

BIOMD0000000338 (Wajima 2009, BioModels, CC0) has XIIa and K activating each
other through ``v*S*E/(k + E)`` with k = 1, so each has a pole at -1, and XIIa =
K = 0 is an exact, unstable solution. The dense LU's rounding can tip XIIa below
0, and the model's own laws then carry it down to the pole. There the RHS is
-inf, and clamped it read 0 for XIIa and for K, which held both at -1. On main
(d1b27d5d, arm64) the dense run over t 0..0.08 returned XIIa = -1 and K = -0.997
with no error, 1e9 atols below 0. Before #1302 the x86-64 CI's run over 0..2 did
the same, and its agreement with KLU (XIIa = 0, the exact solution) passed the
path-agreement check.

Where the value read is infinite, a slot the solver's step started more than
1e6 atols below 0 is now left as it is, and where that is what keeps the value
non-finite, the run is refused, naming the slot. A NaN is clamped as before.

The oracle is the property itself: a run either raises, naming the
concentration, or returns no concentration carried 1e6 atols below 0 to a pole.
"""

from __future__ import annotations

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


@pytest.fixture(autouse=True)
def _cold_codegen_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")


def _run_338(**kw):
    sim = bngsim.Simulator(
        bngsim.Model.from_sbml(str(BIOMD338)), method="ode", force_dense_linear_solver=True, **kw
    )
    return sim.run(t_span=(0.0, 0.08), n_points=81, rtol=1e-6, atol=1e-9, max_steps=100000)


@pytest.mark.parametrize("jacobian", ["fd", "auto", "analytical"])
def test_338_carried_to_its_pole_raises_rather_than_returning_it(jacobian):
    # Where rounding tips XIIa below 0 depends on the platform's last bits: on
    # arm64 this window takes it to the pole, on x86-64 the difference-quotient
    # runs burst positive. Either way no concentration comes back 1e6 atols below
    # 0, and a refusal names XIIa, whose own reading is what is infinite (K, at
    # -0.996, is short of its pole, and its law finite).
    try:
        result = _run_338(jacobian=jacobian)
    except bngsim.SimulationError as exc:
        msg = str(exc)
        assert "carried concentration 'XIIa'" in msg, msg
        assert "times its atol below 0" in msg, msg
        assert "'K' to" not in msg, msg
        return
    species = np.asarray(result.species, dtype=float)
    lowest = species.min(axis=0)
    names = list(result.species_names)
    worst = int(np.argmin(lowest))
    assert lowest[worst] >= -1e6 * 1e-9, f"{names[worst]} returned at {lowest[worst]}"


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
