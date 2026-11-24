"""SSA and PSA read a concentration species at its compartment's live size.

Under SSA a concentration species in a compartment whose size moves (an event
resizes it, or a rate rule or an assignment rule sets it) is stored as its
count over the load-time size, so that a resize leaves the count alone (#741).
Each of these read that stored value where the size had moved, with no error:

* #1463: a run starts from the model's concentrations, and turned them into
  counts at the load-time size. A written start size was lost: c written to 3
  from 2 started X = 50 at 100 molecules for 150. The same for a compartment an
  assignment rule sizes, from a parameter the rule reads.
* #1464: the observables and expressions were computed on count/V_static, so a
  rule ``B := 2*X`` read 64 for 32 after c: 2 -> 4, and in ``c := 1 + time`` the
  observable S and ``p := 2*S`` never moved.
* #1659: an event's concentration assignment in a compartment an assignment rule
  sizes was stored over the load-time size: S = 50 at t = 2 in ``c := 1 + time``
  read [S](3) 12.5 for 37.5.
* #1754: what reads the state an SSA run left read the stored value as a
  concentration. An ODE run from 150 molecules at c = 4 (stored as 75, over the
  load-time 2) started from an amount of 300; so did a steady-state solve, and
  get_concentration reported 75 for [X] = 37.5. set_concentration stored its
  value as the count over the load-time size: [X] = 10 at c = 4 started the next
  SSA run at 20 molecules for 40.

Every model here has no reaction, or one only to show the run moves on, so an
SSA run is deterministic where it is compared to a closed form; the expected
values are the ODE run's and libRoadRunner's.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

pytest.importorskip("antimony")

STOCHASTIC = ("ssa", "psa")


def _run(model, method, **kw):
    sim_kw = {"poplevel": 100} if method == "psa" else {}
    run_kw = {"seed": 1} if method in STOCHASTIC else {"rtol": 1e-10, "atol": 1e-12}
    run_kw.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Simulator(model, method=method, **sim_kw).run(**run_kw)


def _rtol(method):
    """Closed forms hold to rounding under SSA/PSA; the ODE run to its tolerances."""
    return 1e-6 if method == "ode" else 1e-9


def _sel(result, *names):
    return np.asarray(result.as_roadrunner(list(names)))


def _col(result, block, name):
    names = {
        "species": result.species_names,
        "observables": result.observable_names,
        "expressions": result.expression_names,
    }[block]
    return np.asarray(getattr(result, block))[:, list(names).index(name)]


# ── #1463: the start size ────────────────────────────────────────────────────

RESIZED = "compartment c = 2;\nspecies X in c; X = 50;\nE1: at time > 1: c = 4;\n"
RATE_SIZED = "compartment c = 2;\nspecies X in c; X = 50;\nc' = 0.5;\n"


@pytest.mark.parametrize("method", ["ode", *STOCHASTIC])
@pytest.mark.parametrize(
    "text, sizes",
    [
        pytest.param(RESIZED, [3.0, 3.0, 4.0, 4.0], id="event-resized"),
        pytest.param(RATE_SIZED, [3.0, 3.5, 4.0, 4.5], id="rate rule"),
    ],
)
def test_a_written_start_size_reaches_the_starting_count(method, text, sizes):
    """c written to 3 over its load-time 2: the amount is 50*3 = 150 throughout
    (no reaction), and [X] = 150 over the size. SSA started from 100."""
    m = bngsim.Model.from_antimony_string(text)
    m.set_concentration("c", 3.0)  # a state slot: set_param refuses it
    out = _sel(_run(m, method, t_span=(0, 3), n_points=4), "X", "[X]")
    np.testing.assert_allclose(out[:, 0], 150.0, rtol=_rtol(method))
    np.testing.assert_allclose(out[:, 1], 150.0 / np.array(sizes), rtol=_rtol(method))


@pytest.mark.parametrize("method", ["ode", *STOCHASTIC])
def test_a_written_parameter_reaches_the_count_in_a_rule_sized_compartment(method):
    """``c := V0 + time`` with V0 written to 3: S = 10 is an amount of 30, kept
    as the size grows; SSA read it at the load-time size 1 (amount 10)."""
    m = bngsim.Model.from_antimony_string(
        "V0 = 1; compartment c; c := V0 + time; species S in c = 10; p := 2*S;"
    )
    m.set_param("V0", 3.0)
    r = _run(m, method, t_span=(0, 4), n_points=5)
    t = np.arange(5.0)
    np.testing.assert_allclose(_sel(r, "S")[:, 0], 30.0, rtol=_rtol(method))
    np.testing.assert_allclose(_sel(r, "[S]")[:, 0], 30.0 / (3.0 + t), rtol=_rtol(method))
    np.testing.assert_allclose(_col(r, "expressions", "p"), 60.0 / (3.0 + t), rtol=_rtol(method))


def _resized_after_a_run(method):
    """The model after a first run: c = 4 and the amount 150, from c written to 3."""
    m = bngsim.Model.from_antimony_string(RESIZED)
    m.set_concentration("c", 3.0)
    _run(m, method, t_span=(0, 3), n_points=4)
    return m


@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize(
    "go_on",
    [
        pytest.param(lambda m, method: _run(m, method, t_span=(3, 5), n_points=3), id="run again"),
        pytest.param(
            lambda m, method: _run(m, method, t_span=(0, 2), n_points=3), id="run again from 0"
        ),
        pytest.param(
            lambda m, method: (m.save_concentrations(), m.reset(), _run(m, method, t_span=(3, 5)))[
                -1
            ],
            id="save_concentrations, reset",
        ),
        pytest.param(
            lambda m, method: (
                m.save_concentrations("end"),
                m.reset(),
                m.restore_concentrations("end"),
                _run(m, method, t_span=(3, 5)),
            )[-1],
            id="named snapshot",
        ),
        pytest.param(
            lambda m, method: (m.set_state(m.get_state()), _run(m, method, t_span=(3, 5)))[-1],
            id="set_state",
        ),
        pytest.param(
            lambda m, method: _run(m.clone(), method, t_span=(3, 5), n_points=3), id="clone"
        ),
    ],
)
def test_a_run_that_goes_on_from_an_ssa_state_keeps_its_count(method, go_on):
    """What an SSA run left is a count already: a run that starts from it (or
    from a copy of it) is not rescaled to the size again. Read as a
    concentration at c = 4, the 150 molecules would become 300."""
    r = go_on(_resized_after_a_run(method), method)
    np.testing.assert_allclose(_sel(r, "X")[:, 0], 150.0, rtol=1e-9)


@pytest.mark.parametrize("method", STOCHASTIC)
def test_reset_to_a_saved_concentration_counts_it_again(method):
    """A baseline saved before any SSA run is concentrations: after a run, a
    reset to it is counted at the size again (150), not read as the counts the
    run left (100 at the load-time size)."""
    m = bngsim.Model.from_antimony_string(RESIZED)
    m.set_concentration("c", 3.0)
    m.save_concentrations()
    _run(m, method, t_span=(0, 3), n_points=4)
    m.reset()
    np.testing.assert_allclose(_sel(_run(m, method, t_span=(0, 3)), "X")[:, 0], 150.0, rtol=1e-9)


@pytest.mark.parametrize("method", STOCHASTIC)
def test_replicates_and_until_legs_keep_the_count(method):
    m = _resized_after_a_run(method)
    sim_kw = {"poplevel": 100} if method == "psa" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(m, method=method, **sim_kw)
        reps = sim.run_replicates(3, t_span=(3, 5), n_points=3, seed=1)
        legs = [sim.run_until(t, seed=1) for t in (4.0, 5.0)]
    for r in [*reps, *legs]:
        np.testing.assert_allclose(_sel(r, "X")[:, 0], 150.0, rtol=1e-9)


@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize("saved_after_a_run", [True, False], ids=["counts", "concentrations"])
def test_a_scan_reset_to_a_snapshot_reads_it_as_saved(method, saved_after_a_run):
    """parameter_scan(reset_to=label) starts each point from the snapshot, read
    as it was saved: the counts a run left, or concentrations at c = 3. Both
    are 150 molecules. The flag the scan's invocation held was put back over
    the snapshot's, and the points read 300 (counts as concentrations at c =
    4) or 100 (concentrations as counts)."""
    m = bngsim.Model.from_antimony_string(RESIZED + "k = 1;\n")
    m.set_concentration("c", 3.0)
    if saved_after_a_run:
        _run(m, method, t_span=(0, 3), n_points=4)
        m.save_concentrations("start")
        m.reset()
    else:
        m.save_concentrations("start")
        _run(m, method, t_span=(0, 3), n_points=4)
    sim_kw = {"poplevel": 100} if method == "psa" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(m, method=method, **sim_kw)
        points = sim.parameter_scan("k", [1.0, 2.0], t_span=(3, 5), n_points=3, reset_to="start")
    for r in points:
        np.testing.assert_allclose(_sel(r, "X")[:, 0], 150.0, rtol=1e-9)


@pytest.mark.parametrize("method", STOCHASTIC)
def test_an_ode_run_between_two_ssa_runs_leaves_concentrations(method):
    """SSA to t = 0.5, ODE to 3 (which leaves [X] = 25 at c = 4), then SSA: the
    last run counts 25*4 = 100. Read as the counts the first run left, it was
    50. (The size is still the load-time one when the ODE run starts here; one
    off it is in test_an_ode_run_from_an_ssa_state_keeps_its_amount.)"""
    m = bngsim.Model.from_antimony_string(RESIZED)
    _run(m, method, t_span=(0, 0.5), n_points=2)
    _run(m, "ode", t_span=(0.5, 3), n_points=2)
    r = _run(m, method, t_span=(3, 5), n_points=3)
    np.testing.assert_allclose(_sel(r, "X")[:, 0], 100.0, rtol=1e-6)


@pytest.mark.parametrize("method", STOCHASTIC)
def test_an_ode_state_is_a_concentration(method):
    """An ODE run leaves a concentration (37.5 at c = 4): an SSA run that goes
    on from it counts 37.5*4 = 150, as the ODE's amount is."""
    m = bngsim.Model.from_antimony_string(RESIZED)
    m.set_concentration("c", 3.0)
    _run(m, "ode", t_span=(0, 3), n_points=4)
    r = _run(m, method, t_span=(3, 5), n_points=3)
    np.testing.assert_allclose(_sel(r, "X")[:, 0], 150.0, rtol=1e-9)


def test_the_resized_mean_tracks_the_ode():
    """The issue's model, with a decay: 300 seeds, the mean within 5 standard
    errors of the ODE at every row (the start, 100 for 150, was 50 off)."""
    text = RESIZED + "k2 = 0.5;\nR2: X -> ; k2*X;\n"

    def one(method, seed=None):
        m = bngsim.Model.from_antimony_string(text)
        m.set_concentration("c", 3.0)
        kw = {"seed": seed} if seed is not None else {}
        return _sel(_run(m, method, t_span=(0, 3), n_points=4, **kw), "X")[:, 0]

    ode = one("ode")
    runs = np.array([one("ssa", s + 1) for s in range(300)])
    se = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    assert runs[0, 0] == 150.0
    assert np.all(np.abs(runs.mean(axis=0) - ode) <= 5.0 * np.maximum(se, 1e-9))


# ── #1464: observables and expressions ───────────────────────────────────────


@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize(
    "resize, sizes",
    [
        pytest.param("E1: at time > 1: c = 4;", [2.0, 2.0, 4.0, 4.0], id="event"),
        pytest.param("c' = 1;", [2.0, 3.0, 4.0, 5.0], id="rate rule"),
    ],
)
@pytest.mark.parametrize(
    "rule, of",
    [
        pytest.param("2*X", lambda x: 2.0 * x, id="observable"),
        pytest.param("X*X/50", lambda x: x * x / 50.0, id="expression"),
        pytest.param("2*X + 0*time", lambda x: 2.0 * x, id="expression of time"),
    ],
)
def test_a_rule_reads_the_live_concentration(method, resize, sizes, rule, of):
    """``B := rule`` in the resized compartment, with X = 50 held (no reaction):
    [X] = 100/c, and B is the rule of it. Observable or expression, it read [X]
    as 100/2 = 50 after the size moved, as did the observable X."""
    text = f"compartment c = 2;\nspecies B in c; B := {rule};\nspecies X in c; X = 50;\n{resize}\n"
    r = _run(bngsim.Model.from_antimony_string(text), method, t_span=(0, 3), n_points=4)
    x = 100.0 / np.array(sizes)
    np.testing.assert_allclose(_sel(r, "[X]")[:, 0], x, rtol=1e-9)
    np.testing.assert_allclose(_sel(r, "[B]")[:, 0], of(x), rtol=1e-9)
    np.testing.assert_allclose(_col(r, "observables", "X"), x, rtol=1e-9)


@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize(
    "size", ["compartment c; c := 1 + time;", "compartment c = 1; c' = 1;"], ids=["rule", "rate"]
)
def test_outputs_in_a_moving_compartment_follow_its_size(method, size):
    """The amount stays 10 in c(t) = 1 + t, so [S] = 10/(1 + t) and p = B =
    20/(1 + t); SSA held p, B and the observable S at their t = 0 values."""
    m = bngsim.Model.from_antimony_string(
        size + " species S in c = 10; p := 2*S; compartment d = 1; species B in d; B := 2*S;"
    )
    r = _run(m, method, t_span=(0, 4), n_points=5)
    t = np.arange(5.0)
    np.testing.assert_allclose(_col(r, "expressions", "p"), 20.0 / (1.0 + t), rtol=1e-9)
    np.testing.assert_allclose(_col(r, "species", "B"), 20.0 / (1.0 + t), rtol=1e-9)
    np.testing.assert_allclose(_col(r, "observables", "S"), 10.0 / (1.0 + t), rtol=1e-9)
    np.testing.assert_allclose(_sel(r, "S")[:, 0], 10.0, rtol=1e-9)


@pytest.mark.parametrize("method", STOCHASTIC)
def test_the_run_goes_on_from_the_stored_count(method):
    """The live outputs are for the rows only: the run itself goes on from the
    stored count. A decay in the resized compartment keeps its rate (mass
    action on the count), so the mean amount tracks the ODE's after the event."""
    text = RESIZED + "k2 = 0.5;\nR2: X -> ; k2*X;\nspecies B in c; B := 2*X;\n"
    ode = _run(bngsim.Model.from_antimony_string(text), "ode", t_span=(0, 3), n_points=4)
    runs = np.array(
        [
            _sel(
                _run(
                    bngsim.Model.from_antimony_string(text),
                    method,
                    t_span=(0, 3),
                    n_points=4,
                    seed=s + 1,
                ),
                "X",
                "[B]",
                "[X]",
            )
            for s in range(200)
        ]
    )
    np.testing.assert_allclose(runs[:, :, 1], 2.0 * runs[:, :, 2], rtol=1e-12)
    mean, se = runs[:, :, 0].mean(axis=0), runs[:, :, 0].std(axis=0, ddof=1) / np.sqrt(200)
    assert np.all(np.abs(mean - _sel(ode, "X")[:, 0]) <= 5.0 * np.maximum(se, 1e-9))


# ── #1659: an event's write in a rule-sized compartment ──────────────────────


@pytest.mark.parametrize("method", ["ode", *STOCHASTIC])
@pytest.mark.parametrize(
    "size", ["compartment c; c := 1 + time;", "compartment c = 1; c' = 1;"], ids=["rule", "rate"]
)
def test_an_event_assigns_a_concentration_at_the_live_size(method, size):
    """S = 50 at t = 2, where c = 3: the amount is 150 from there, and [S] =
    150/(1 + t). SSA stored 50 (over the load-time size 1) and read 12.5 at t =
    3 for 37.5."""
    m = bngsim.Model.from_antimony_string(
        size + " species S in c = 10; E1: at (time >= 2): S = 50;"
    )
    out = _sel(_run(m, method, t_span=(0, 4), n_points=5), "[S]", "S")
    np.testing.assert_allclose(out[:, 0], [10.0, 5.0, 50.0, 37.5, 30.0], rtol=_rtol(method))
    np.testing.assert_allclose(out[:, 1], [10.0, 10.0, 150.0, 150.0, 150.0], rtol=_rtol(method))


# ── #1754: what reads an SSA run's state ─────────────────────────────────────

RULE_SIZED = "V0 = 1; compartment c; c := V0 + time; species S in c = 10;"


def _rate_sized_after_a_run(method):
    """c' = 0.5 from c written to 3, after a run to t = 3: c = 4.5, amount 150."""
    m = bngsim.Model.from_antimony_string(RATE_SIZED)
    m.set_concentration("c", 3.0)
    _run(m, method, t_span=(0, 3), n_points=4)
    return m


def _rule_sized_after_a_run(method):
    """``c := V0 + time`` with V0 written to 3, after a run to t = 4: c = 7, the
    parameter the rule writes, and the amount 30."""
    m = bngsim.Model.from_antimony_string(RULE_SIZED)
    m.set_param("V0", 3.0)
    _run(m, method, t_span=(0, 4), n_points=5)
    return m


# (the model after an SSA run, its species, the run's end, the amount, what
# the run stored: the amount over the load-time size, and the concentration)
AFTER_AN_SSA_RUN = [
    pytest.param(_resized_after_a_run, "X", 3.0, 150.0, 75.0, 37.5, id="event-resized"),
    pytest.param(_rate_sized_after_a_run, "X", 3.0, 150.0, 75.0, 150.0 / 4.5, id="rate rule"),
    pytest.param(_rule_sized_after_a_run, "S", 4.0, 30.0, 30.0, 30.0 / 7.0, id="rule-sized"),
]


@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize("after, name, t_end, amount, stored, conc", AFTER_AN_SSA_RUN)
def test_get_concentration_reads_an_ssa_state_at_the_live_size(
    method, after, name, t_end, amount, stored, conc
):
    """150 molecules at c = 4 are [X] = 37.5: get_concentration reported the
    value the run stored, the count over the load-time size (75). get_state
    still returns that."""
    m = after(method)
    assert m.get_concentration(name) == pytest.approx(conc, rel=1e-12)
    assert m.get_state()[m.species_names.index(name)] == pytest.approx(stored, rel=1e-12)


@pytest.mark.parametrize("from_zero", [False, True], ids=["where it ended", "from 0"])
@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize("after, name, t_end, amount, stored, conc", AFTER_AN_SSA_RUN)
def test_an_ode_run_from_an_ssa_state_keeps_its_amount(
    method, from_zero, after, name, t_end, amount, stored, conc
):
    """An ODE run from 150 molecules at c = 4 read the stored 75 as [X], and
    integrated an amount of 300, which the next SSA run went on from. It starts
    from the amount the SSA run left, and leaves it to the next run. So does a
    run that sets the clock back, as an SSA run does: in ``c := V0 + time`` the
    30 molecules are [S] = 10 at t = 0, where c = 3."""
    m = after(method)
    t0 = 0.0 if from_zero else t_end
    ode = _run(m, "ode", t_span=(t0, t0 + 2), n_points=3)
    np.testing.assert_allclose(_sel(ode, name)[:, 0], amount, rtol=1e-6)
    if not from_zero:
        assert _sel(ode, f"[{name}]")[0, 0] == pytest.approx(conc, rel=1e-6)
    again = _run(m, method, t_span=(t0 + 2, t0 + 3), n_points=2)
    np.testing.assert_allclose(_sel(again, name)[:, 0], amount, rtol=1e-9)


@pytest.mark.parametrize("then", ["same", "ode"])
@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize("after, name, t_end, amount, stored, conc", AFTER_AN_SSA_RUN)
def test_set_concentration_over_an_ssa_state_is_a_concentration(
    method, then, after, name, t_end, amount, stored, conc
):
    """[X] = 10 at c = 4 is 40 molecules. Written over an SSA run's state, the
    10 was stored as the count over the load-time size, and the next SSA run
    started from 20. The ODE rows held before too, the raw write and the ODE
    run's raw read cancelling; they keep a fix of one without the other out."""
    m = after(method)
    m.set_concentration(name, 10.0)
    assert m.get_concentration(name) == pytest.approx(10.0, rel=1e-12)
    r = _run(m, method if then == "same" else "ode", t_span=(t_end, t_end + 2), n_points=3)
    np.testing.assert_allclose(_sel(r, name)[:, 0], 10.0 * amount / conc, rtol=1e-6)


@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize("between", ["ODE Simulator", "V0 written"])
def test_a_rule_sized_state_is_read_at_the_size_the_next_run_reads(method, between):
    """``c := V0 + time`` after a run to t = 4: 30 molecules at c = 7. Building
    an ODE Simulator evaluates the rule at t = 0, which left its parameter at
    3, and writing V0 = 5 makes c = 9 without evaluating it, which left the
    parameter at 7. get_concentration and set_concentration read the size the
    next run reads, not the parameter: they read [S] = 10 for 30/7 after the
    Simulator and 30/7 for 30/9 after the write, and stored [S] = 10 as 30 and
    70 molecules for 70 and 90. The read leaves the model as it was."""
    m = _rule_sized_after_a_run(method)
    if between == "ODE Simulator":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bngsim.Simulator(m, method="ode")
        size = 7.0
    else:
        m.set_param("V0", 5.0)
        size = 9.0
    held, state = m.get_param("c"), m.get_state()
    assert m.get_concentration("S") == pytest.approx(30.0 / size, rel=1e-12)
    assert m.get_param("c") == held
    np.testing.assert_array_equal(m.get_state(), state)
    for then in (method, "ode"):
        r = _run(m.clone(), then, t_span=(4, 5), n_points=2)
        assert _sel(r, "[S]")[0, 0] == pytest.approx(30.0 / size, rel=1e-9)
    m.set_concentration("S", 10.0)
    for then in (method, "ode"):
        r = _run(m.clone(), then, t_span=(4, 5), n_points=2)
        np.testing.assert_allclose(_sel(r, "S")[:, 0], 10.0 * size, rtol=1e-6)


@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize("ss_method", ["integration", "newton"])
def test_a_steady_state_from_an_ssa_state_keeps_its_amount(method, ss_method):
    """c' = 4 - c from 2 keeps X's 100 molecules, so [X] settles at 100/4 = 25.
    From an SSA run's state at t = 3 (c = 3.90) the solve read the stored 50 as
    [X] there, an amount of 195, and settled at 48.76. The model is left as
    the run stored it, as every solve leaves it."""
    m = bngsim.Model.from_antimony_string(
        "compartment c = 2;\nc' = 4 - c;\nspecies X in c; X = 50;\n"
    )
    _run(m, method, t_span=(0, 3), n_points=4)
    before = m.get_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = bngsim.Simulator(m, method="ode").steady_state(method=ss_method)
    assert ss.converged
    assert ss["X"] == pytest.approx(25.0, rel=1e-5)
    np.testing.assert_array_equal(m.get_state(), before)
    assert m._core.ssa_counts_stored


@pytest.mark.parametrize("method", STOCHASTIC)
@pytest.mark.parametrize("saved_after_a_run", [True, False], ids=["counts", "concentrations"])
def test_a_snapshot_restores_the_state_as_it_was_stored(method, saved_after_a_run):
    """Simulator.snapshot() keeps the state as stored, with the mark saying
    whether it is an SSA run's counts, and restore() puts both back. Kept as
    the concentrations get_concentration reads, an SSA run's state restored
    after an ODE run was read as counts again, 75 molecules for 150; and
    concentrations restored over an SSA run's state were stored as counts at
    the size the compartment had then, 300."""
    m = bngsim.Model.from_antimony_string(RESIZED)
    m.set_concentration("c", 3.0)
    sim_kw = {"poplevel": 100} if method == "psa" else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = bngsim.Simulator(m, method=method, **sim_kw)
        if saved_after_a_run:
            sim.run(t_span=(0, 3), n_points=4, seed=1)
            snap = sim.snapshot()
            _run(m, "ode", t_span=(3, 5), n_points=3)
        else:
            snap = sim.snapshot()
            sim.run(t_span=(0, 3), n_points=4, seed=1)
        sim.restore(snap)
        r = sim.run(t_span=(3, 5) if saved_after_a_run else (0, 3), n_points=3, seed=1)
    np.testing.assert_allclose(_sel(r, "X")[:, 0], 150.0, rtol=1e-9)


@pytest.mark.parametrize("method", STOCHASTIC)
def test_the_ssa_check_reads_an_ssa_state_as_its_counts(method):
    """101 molecules at c = 4 are [X] = 25.25, which over the load-time size 2
    is 50.5 molecules: the check before an SSA run reads the count the run
    stored, and finds no fractional population in it."""
    m = bngsim.Model.from_antimony_string(
        "compartment c = 2;\nspecies X in c; X = 50.5;\nE1: at time > 1: c = 4;\n"
    )
    _run(m, method, t_span=(0, 3), n_points=4)
    assert m.get_concentration("X") == pytest.approx(25.25, rel=1e-12)
    codes = [issue.code for issue in bngsim.validate_for_ssa(m)]
    assert "non_integer_initial_population" not in codes
