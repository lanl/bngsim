"""Issue #1660: a batch row, or a scan point, reads the default atol from its own start.

Where no caller chose a per-species atol, run() holds each species to ``rtol`` of
its scale in the state it starts from, under the scalar (#1420). run_batch and
parameter_scan read that state once, from the model they were called on, and
held every row to it. A row that starts a growing species decades below that
state was then held to a tolerance above the species: a logistic grower declared
at 1, dosed at X0 = 1e-9, returned X(20) = -4.6e-7 for 0.326674, -1.5e+292 at
``n_points=41``, and no grid of 2..301 points came within 1% of it.

Each expected value is the closed form: dX/dt = k X (1 - X) from X0 gives
X(t) = X0 e^(kt) / (1 + X0 (e^(kt) - 1)).
"""

from __future__ import annotations

import math
import warnings

import bngsim
import numpy as np
import pytest

ANT = "model m\n X0 = {x0}; k = 1; species X; X = X0;\n J: -> X; k*X*(1 - X);\nend"
DOSE, T = 1e-9, 20.0


def _logistic(x0, t, k=1.0):
    e = math.exp(k * t)
    return x0 * e / (1 + x0 * (e - 1))


def _sim(declared=1, **kw):
    pytest.importorskip("antimony")
    model = bngsim.Model.from_antimony_string(ANT.format(x0=declared))
    return bngsim.Simulator(model, method="ode", **kw)


def _rel(x, exact):
    return abs(float(x) - exact) / exact


@pytest.mark.parametrize("n_points", [2, 21, 41, 101])
def test_a_row_dosed_below_the_model_follows_its_own_ode(n_points):
    """The issue's row, on grids main returned -4.6e-7, 0.554, -1.5e+292 and
    a wrong value on."""
    row = _sim().run_batch(t_span=(0, T), n_points=n_points, params=[{"X0": DOSE}])[0]
    x = np.asarray(row.species)[:, 0]
    times = np.asarray(row.time)
    exact = np.array([_logistic(DOSE, t) for t in times])
    assert x[0] == DOSE
    assert np.max(np.abs(x - exact)) / exact.max() < 1e-5


def test_a_scalar_the_caller_chose_is_read_from_the_row_too():
    """A caller's scalar is a ceiling as the default is (#1420): at rtol 1e-10,
    atol 1e-12, main held the row to 1e-12 and was 2.3e-3 off."""
    row = _sim().run_batch(
        t_span=(0, T), n_points=2, params=[{"X0": DOSE}], rtol=1e-10, atol=1e-12
    )[0]
    assert _rel(row.species[-1, 0], _logistic(DOSE, T)) < 1e-6


def test_a_row_after_a_run_is_read_from_its_own_start_not_where_the_run_ended():
    """Rows start from the initial conditions. Main took the tolerance from the
    state a run left (X near 1): 0.622 for 0.327."""
    sim = _sim(declared=DOSE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # #1362's: the model is not at its start
        sim.run(t_span=(0, T), n_points=2)
        row = sim.run_batch(t_span=(0, T), n_points=2, params=[{"X0": DOSE}])[0]
    assert _rel(row.species[-1, 0], _logistic(DOSE, T)) < 1e-5


@pytest.mark.parametrize("n_points", [2, 41])
def test_a_row_is_the_run_from_its_start(n_points):
    """GH #203's contract: a batch row and a run() at that point give the same
    numbers, bit for bit."""
    row = _sim().run_batch(t_span=(0, T), n_points=n_points, params=[{"X0": DOSE}])[0]
    sim = _sim()
    sim.model.set_param("X0", DOSE)
    alone = sim.run(t_span=(0, T), n_points=n_points)
    assert np.array_equal(np.asarray(row.species), np.asarray(alone.species))


def test_rows_at_different_doses_each_follow_their_own_ode_threaded_or_not():
    doses = [1e-12, 1e-9, 1e-6, 1e-3, 1.0]
    params = [{"X0": d} for d in doses]
    serial = _sim().run_batch(t_span=(0, T), n_points=2, params=params)
    threaded = _sim().run_batch(t_span=(0, T), n_points=2, params=params, num_processors=3)
    for dose, a, b in zip(doses, serial, threaded, strict=True):
        assert _rel(a.species[-1, 0], _logistic(dose, T)) < 1e-5, dose
        assert np.array_equal(np.asarray(a.species), np.asarray(b.species))


@pytest.mark.parametrize("call", ["run_batch", "parameter_scan"])
def test_a_row_above_the_model_is_held_no_looser_than_the_model(call):
    """Each species takes the tighter of the model's reading and the row's:
    the model at X = 1e-9 holds X to 1e-17, and a row dosed at X0 = 0.5 keeps
    that, the numbers of a run held to the model's own vector. Read from the
    row's start alone, it would take the scalar, which gives other numbers."""
    sim = _sim(declared=DOSE)
    own = sim._run_atol(None, sim._rtol)
    if call == "run_batch":
        out = sim.run_batch(t_span=(0, T), n_points=21, params=[{"X0": 0.5}])[0]
    else:
        out = sim.parameter_scan("X0", [0.5], t_span=(0, T), n_points=21)[0]
    held = _sim(declared=0.5).run(t_span=(0, T), n_points=21, atol=list(own.vector))
    plain = _sim(declared=0.5).run(t_span=(0, T), n_points=21)
    assert not np.array_equal(np.asarray(held.species), np.asarray(plain.species))
    assert np.array_equal(np.asarray(out.species), np.asarray(held.species))


def test_a_row_at_the_models_start_keeps_the_tolerance_it_had():
    """A row that moves only a rate constant starts where the model does, and is
    held to the model's tolerance, as before: the same numbers as a run() there."""
    sim = _sim(declared=DOSE)
    row = sim.run_batch(t_span=(0, T), n_points=21, params=[{"k": 0.9}])[0]
    alone = _sim(declared=DOSE)
    alone.model.set_param("k", 0.9)
    ref = alone.run(t_span=(0, T), n_points=21)
    assert np.array_equal(np.asarray(row.species), np.asarray(ref.species))
    assert _rel(row.species[-1, 0], _logistic(DOSE, T, k=0.9)) < 1e-5


@pytest.mark.parametrize("n_points", [2, 41])
def test_a_scan_point_dosed_below_the_model_follows_its_own_ode(n_points):
    """parameter_scan froze the tolerance at the invocation state, as run_batch
    did: the same -4.6e-7 and -1.5e+292."""
    point = _sim().parameter_scan("X0", [DOSE], t_span=(0, T), n_points=n_points)[0]
    assert _rel(point.species[-1, 0], _logistic(DOSE, T)) < 1e-5


def test_scan_points_each_follow_their_own_ode():
    doses = [1.0, 1e-9, 1e-3, 1e-12]
    points = _sim().parameter_scan("X0", doses, t_span=(0, T), n_points=2)
    for dose, point in zip(doses, points, strict=True):
        assert _rel(point.species[-1, 0], _logistic(dose, T)) < 1e-5, dose


def test_a_scan_point_is_the_run_from_its_start():
    point = _sim().parameter_scan("X0", [DOSE], t_span=(0, T), n_points=41)[0]
    sim = _sim()
    sim.model.set_param("X0", DOSE)
    alone = sim.run(t_span=(0, T), n_points=41)
    assert np.array_equal(np.asarray(point.species), np.asarray(alone.species))


def test_a_scan_leaves_the_model_as_it_found_it():
    sim = _sim()
    sim.parameter_scan("X0", [DOSE, 1e-3], t_span=(0, T), n_points=2)
    assert sim.model.get_param("X0") == 1
    assert list(sim.model.get_state()) == [1.0]


def test_a_continuation_point_reads_the_tolerance_from_where_it_starts():
    """bifurcate (parameter_scan with reset_conc=False) starts each point where
    the last ended. X' = -X from 1, four points of t = 10: X = e^(-10 n). Held
    to the tolerance of the first start, main returned -1.25e-8 for 2.06e-9 at
    the second point and 2.97e-11 for 4.25e-18 at the fourth."""
    pytest.importorskip("antimony")
    src = "model m\n species X; X = 1; k = 1;\n J: X -> ; k*X;\nend"
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode")
    points = sim.bifurcate("k", [1.0] * 4, t_span=(0, 10), n_points=11)
    for n, point in enumerate(points, start=1):
        assert _rel(point.species[-1, 0], math.exp(-10 * n)) < 1e-2, n


DECAY = "model m\n X0 = {x0}; species X; X = X0;\n J: X -> ; X;\nend"


@pytest.mark.parametrize("call", ["run_batch", "parameter_scan"])
def test_a_row_held_to_its_own_scale_is_not_warned_about(call):
    """AtolScaleWarning is asked against the tolerance each row ran at: a row
    at X0 = 1e-12 runs at 1e-20, and its X is not below that. Asked against the
    model's own, 1e-8, it warned of a species the row did hold."""
    pytest.importorskip("antimony")
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(DECAY.format(x0=1)), method="ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if call == "run_batch":
            out = sim.run_batch(t_span=(0, 5), n_points=6, params=[{"X0": 1e-12}])[0]
        else:
            out = sim.parameter_scan("X0", [1e-12], t_span=(0, 5), n_points=6)[0]
    assert not [c for c in caught if issubclass(c.category, bngsim.AtolScaleWarning)]
    assert _rel(out.species[-1, 0], 1e-12 * math.exp(-5)) < 1e-5


@pytest.mark.parametrize("call", ["run_batch", "parameter_scan"])
@pytest.mark.parametrize("atol", ["auto", "tracking"])
def test_auto_is_read_from_the_starts_of_the_rows(call, atol):
    """``"auto"``, and the tracking ceiling it gives, are still one vector for
    the whole call (#196), but the tightest any row's start asks for: read from
    the model at X = 1, main returned the issue's -4.6e-7 for the row at 1e-9.
    The row at 1 is held to the same vector, and is right too. A tracking
    tolerance follows each species' size below its ceiling, so it was right on
    main too: a control that the tighter ceiling does not break it."""
    sim = _sim()
    doses = [1.0, DOSE]
    if call == "run_batch":
        out = sim.run_batch(
            t_span=(0, T), n_points=2, params=[{"X0": d} for d in doses], atol=atol
        )
    else:
        out = sim.parameter_scan("X0", doses, t_span=(0, T), n_points=2, atol=atol)
    for dose, row in zip(doses, out, strict=True):
        assert _rel(row.species[-1, 0], _logistic(dose, T)) < 1e-5, dose


def test_auto_after_a_run_is_read_from_where_the_rows_start():
    """Rows start from the initial conditions; main read "auto" from where a
    run had left the model."""
    sim = _sim(declared=DOSE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # #1362's: the model is not at its start
        sim.run(t_span=(0, T), n_points=2)
        row = sim.run_batch(t_span=(0, T), n_points=2, params=[{"k": 1.0}], atol="auto")[0]
    assert _rel(row.species[-1, 0], _logistic(DOSE, T)) < 1e-5


@pytest.mark.parametrize("call", ["run_batch", "parameter_scan"])
def test_auto_for_rows_at_the_models_start_is_the_vector_it_was(call):
    """Rows that move only a rate constant start where the model is: their
    "auto" is the one derived from the model's state, bit for bit."""
    sim = _sim(declared=1e-3)
    frozen = list(sim.auto_atol())
    ks = [0.5, 1.5]
    if call == "run_batch":
        rows = [{"k": k} for k in ks]
        auto = sim.run_batch(t_span=(0, T), n_points=21, params=rows, atol="auto")
        given = _sim(declared=1e-3).run_batch(t_span=(0, T), n_points=21, params=rows, atol=frozen)
    else:
        auto = sim.parameter_scan("k", ks, t_span=(0, T), n_points=21, atol="auto")
        given = _sim(declared=1e-3).parameter_scan(
            "k", ks, t_span=(0, T), n_points=21, atol=frozen
        )
    for a, b in zip(auto, given, strict=True):
        assert np.array_equal(np.asarray(a.species), np.asarray(b.species))


def test_auto_reads_the_tightest_start_species_by_species():
    """Two species, each smallest in a different row: the batch's vector takes
    each species' own smallest, rtol of it."""
    pytest.importorskip("antimony")
    src = "model m\n a = 1; b = 1; species X, Y; X = a; Y = b;\n J1: X -> ; X; J2: Y -> ; Y;\nend"
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(src), method="ode")
    rows = [{"a": 1e-6, "b": 1.0}, {"a": 1.0, "b": 1e-9}]
    with sim._auto_from(lambda: [np.array([1e-6, 1.0]), np.array([1.0, 1e-9])]):
        vector = sim._atol_vector_from("auto", 1e-6, where="t")
    assert vector == pytest.approx([1e-12, 1e-15], rel=1e-12)
    out = sim.run_batch(t_span=(0, 5), n_points=2, params=rows, atol="auto")
    assert _rel(out[0].species[-1, 0], 1e-6 * math.exp(-5)) < 1e-5
    assert _rel(out[1].species[-1, 1], 1e-9 * math.exp(-5)) < 1e-5


def test_a_sensitivity_row_dosed_below_the_model_is_right():
    """A row with forward sensitivities from a compiled ∂f/∂p reads the default
    as a ceiling too; dX/dk = X0 (1 - X0) t e^(kt) / (1 + X0 (e^(kt) - 1))^2."""
    sim = _sim(sensitivity_params=["k"])
    if not sim._codegen_provides_sens_rhs():
        pytest.skip("codegen compile unavailable: no compiled sensitivity RHS on this install")
    row = sim.run_batch(t_span=(0, T), n_points=2, params=[{"X0": DOSE}])[0]
    e = math.exp(T)
    exact = DOSE * (1 - DOSE) * T * e / (1 + DOSE * (e - 1)) ** 2
    assert _rel(row.species[-1, 0], _logistic(DOSE, T)) < 1e-5
    assert _rel(np.asarray(row.sensitivities)[-1, 0, 0], exact) < 1e-4
