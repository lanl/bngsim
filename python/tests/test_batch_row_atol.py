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

DOSE, T = 1e-9, 20.0


def _net(parameters, species, reactions):
    """A ``.net`` model's text. Every model here is one, so the file runs on
    each leg, the ones with no Antimony extra among them."""

    def block(name, items):
        rows = "".join(f" {i} {item}\n" for i, item in enumerate(items, start=1))
        return f"begin {name}\n{rows}end {name}\n"

    return (
        block("parameters", parameters) + block("species", species) + block("reactions", reactions)
    )


@pytest.fixture(autouse=True)
def _models_dir(tmp_path):
    """Where this test's models are written."""
    global _DIR
    _DIR = tmp_path


def _load(text, **kw):
    path = _DIR / f"model_{len(list(_DIR.glob('model_*.net')))}.net"
    path.write_text(text)
    return bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", **kw)


# Logistic growth, X' = k X (1 - X): X -> 2X and 2X -> X, each at k.
def _logistic_net(x0):
    return _net([f"X0 {x0}", "k 1"], ["X() X0"], ["1 1,1 k", "1,1 1 k"])


def _logistic(x0, t, k=1.0):
    e = math.exp(k * t)
    return x0 * e / (1 + x0 * (e - 1))


def _sim(declared=1, **kw):
    return _load(_logistic_net(declared), **kw)


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
    atol 1e-12, main held the row to 1e-12 and was 2.3e-3 off. run() from the
    row's start is 4.7e-8 off; 1e-5 leaves room for another platform's steps."""
    row = _sim().run_batch(
        t_span=(0, T), n_points=2, params=[{"X0": DOSE}], rtol=1e-10, atol=1e-12
    )[0]
    assert _rel(row.species[-1, 0], _logistic(DOSE, T)) < 1e-5


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
    sim = _load(_net(["k 1"], ["X() 1"], ["1 0 k"]))
    points = sim.bifurcate("k", [1.0] * 4, t_span=(0, 10), n_points=11)
    for n, point in enumerate(points, start=1):
        assert _rel(point.species[-1, 0], math.exp(-10 * n)) < 1e-2, n


# X' = -X from X0.
DECAY = _net(["X0 1", "kd 1"], ["X() X0"], ["1 0 kd"])


@pytest.mark.parametrize("call", ["run_batch", "parameter_scan"])
def test_a_row_held_to_its_own_scale_is_not_warned_about(call):
    """AtolScaleWarning is asked against the tolerance each row ran at: a row
    at X0 = 1e-12 runs at 1e-20, and its X is not below that. Asked against the
    model's own, 1e-8, it warned of a species the row did hold."""
    sim = _load(DECAY)
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
    sim = _load(_net(["a 1", "b 1", "kd 1"], ["X() a", "Y() b"], ["1 0 kd", "2 0 kd"]))
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


def test_the_starts_auto_reads_are_each_rows_own():
    """Each row's start is read on one scratch clone with the row's writes put
    back after it. X = a*b: a row writing a alone and one writing b alone each
    start X at 1e-6; had the first row's a stayed behind, the second would read
    X = 1e-12, smaller than any row's start, and the batch's "auto" with it."""
    sim = _load(_net(["a 1", "b 1", "ab a*b", "kd 1"], ["X() ab"], ["1 0 kd"]))
    rows = [{"a": 1e-6}, {"b": 1e-6}, {}, {"a": 2.0, "b": 3.0}]
    expected = []
    for row in rows:
        clone = sim.model.clone()
        clone.set_params(row)
        clone.reset()
        expected.append(list(clone.get_state()))
    assert (
        [list(s) for s in sim._batch_row_starts(rows)]
        == expected
        == [[1e-6], [1e-6], [1.0], [6.0]]
    )
    assert list(sim.model.get_state()) == [1.0]  # the batch's model is untouched


# X -> 2X at k from X() = Xt, a derived parameter 2*X0: X(t) = Xt e^(kt).
DERIVED = """begin parameters
 1 X0 1
 2 sc 2
 3 Xt X0*sc
 4 k 1
end parameters
begin species
 1 X() Xt
end species
begin reactions
 1 1 1,1 k
end reactions
"""


def test_a_derived_initial_condition_a_row_overrode_is_put_back(tmp_path):
    """Row 0 writes Xt, overriding its expression, and X0 under it. Put back in
    one write, Xt stayed pinned at 2, row 1's start read 2 for its 2e-9, and
    under "auto" row 1 returned -9.2e-7 for 0.97. Growth is exponential; the
    bound is the logistic's, run() from 2e-9 being 3.1e-6 off."""
    path = tmp_path / "derived.net"
    path.write_text(DERIVED)
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    rows = [{"Xt": 3.0, "X0": 7.0}, {"X0": DOSE}]
    assert [list(s) for s in sim._batch_row_starts(rows)] == [[3.0], [2 * DOSE]]
    out = sim.run_batch(t_span=(0, T), n_points=2, params=rows, atol="auto")
    assert _rel(out[1].species[-1, 0], 2 * DOSE * math.exp(T)) < 1e-5


# ── A dose a hook writes, a continuation's first point, after a run ──────────


def _dose_x(model, value):
    model.set_concentration("X()", DOSE)


@pytest.mark.parametrize("atol", [None, "auto"])
def test_a_dose_an_on_point_hook_writes_is_read_at_the_point(atol):
    """The hook writes the point's start after the scan has read its own. Each
    point reads the default from there, as run() would; ``"auto"``, read from
    the points' starts before any hook ran, returned the issue's -4.6e-7."""
    kw = {} if atol is None else {"atol": atol}
    point = _sim().parameter_scan("k", [1.0], t_span=(0, T), n_points=2, on_point=_dose_x, **kw)[0]
    assert _rel(point.species[-1, 0], _logistic(DOSE, T)) < 1e-5


def test_a_point_its_hook_doses_is_judged_by_its_own_tolerance():
    """At k = 0.01 X stays near the hook's 1e-9 (1.2e-9 at t = 20). The point
    runs at 1e-17 for it, and the scale warning reads the point's tolerance,
    not the model's 1e-8, below which X stays throughout."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.AtolScaleWarning)
        point = _sim().parameter_scan("k", [0.01], t_span=(0, T), n_points=2, on_point=_dose_x)[0]
    assert _rel(point.species[-1, 0], _logistic(DOSE, T, k=0.01)) < 1e-5


@pytest.mark.parametrize("atol", [None, "auto"])
@pytest.mark.parametrize("call", ["bifurcate", "parameter_scan"])
def test_a_continuations_first_point_is_read_where_the_value_puts_it(call, atol):
    """A continuation's later points carry on from the one before; its first
    starts where the scanned value puts it. ``"auto"`` read from the state at
    the call returned -4.6e-7."""
    kw = {} if atol is None else {"atol": atol}
    sim = _sim()
    if call == "bifurcate":
        (point,) = sim.bifurcate("X0", [DOSE], t_span=(0, T), n_points=2, **kw)
    else:
        (point,) = sim.parameter_scan(
            "X0", [DOSE], t_span=(0, T), n_points=2, reset_conc=False, **kw
        )
    assert _rel(point.species[-1, 0], _logistic(DOSE, T)) < 1e-5


def test_a_batch_after_a_run_is_the_batch_on_a_fresh_model():
    """Rows start from the initial conditions, run or no run. The model's side
    of the tighter-of-two was read from its live state: after X -> Y was run to
    t = 30 (X near 1e-13), rows over k took 309 and 429 steps for 153 and 176,
    and gave other numbers."""
    src = _net(["X0 1", "k 1"], ["X() X0", "Y() 0"], ["1 2 k"])

    def sim():
        return _load(src)

    rows = [{"k": 0.5}, {"k": 2.0}]
    fresh = sim().run_batch(t_span=(0, 30), n_points=31, params=rows)
    ran = sim()
    ran.run(t_span=(0, 30), n_points=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # #1362's: the model is not at its start
        after = ran.run_batch(t_span=(0, 30), n_points=31, params=rows)
    for a, b in zip(fresh, after, strict=True):
        assert a.solver_stats["n_steps"] == b.solver_stats["n_steps"]
        assert np.array_equal(np.asarray(a.species), np.asarray(b.species))


# ── What the rows are held to, not what they compute ─────────────────────────


def test_a_wrong_length_vector_is_refused_before_any_row_or_point():
    """Refused by name up front: a threaded batch raises the ValueError, not a
    row's "Batch simulation 0 failed", and a scan's hook never runs."""
    sim = _sim()

    def a_point_started(*args, **kwargs):
        raise AssertionError("a point started before the refusal")

    with pytest.raises(ValueError, match=r"^run_batch\(atol=\.\.\.\)"):
        sim.run_batch(
            t_span=(0, T), n_points=2, params=[{"X0": DOSE}], atol=[1e-8] * 2, num_processors=2
        )
    with pytest.raises(ValueError, match=r"parameter_scan\(atol=\.\.\.\)"):
        sim.parameter_scan(
            "X0", [DOSE], t_span=(0, T), n_points=2, atol=[1e-8] * 2, on_point=a_point_started
        )


# k2 follows 2*k1 until a write pins it (issue #188); A -> B at k2.
ATTACHED = """begin parameters
 1 k1 1
 2 k2 2*k1
end parameters
begin species
 1 A() 10
 2 B() 0
end species
begin reactions
 1 1 2 k2
end reactions
"""


def test_a_row_run_by_a_simulator_built_for_its_attachment_reads_the_batchs_tolerances(
    tmp_path,
):
    """A row that pins k2 needs other sensitivity code and runs on a Simulator
    built for it (issue #961), which has none of the batch's set_tolerances. It
    reads the batch's: the row is the run() of its model at them, to the bit,
    where the built one's default 1e-8 moves it."""
    path = tmp_path / "attached.net"
    path.write_text(ATTACHED)

    def sensitive(model):
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1", "k2"])
        sim.set_tolerances(rtol=1e-6, atol=1e-4)
        return sim

    span = {"t_span": (0.0, 1.0), "n_points": 11}
    (row,) = sensitive(bngsim.Model.from_net(str(path))).run_batch(params=[{"k2": 5.0}], **span)
    alone = bngsim.Model.from_net(str(path))
    alone.set_param("k2", 5.0)
    run = sensitive(alone).run(**span)
    assert run.solver_stats["n_steps"] == row.solver_stats["n_steps"]
    assert np.array_equal(np.asarray(run.species), np.asarray(row.species))
