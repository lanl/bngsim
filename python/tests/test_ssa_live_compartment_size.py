"""SSA and PSA read a concentration species at its compartment's live size.

Under SSA a concentration species in a compartment whose size moves (an event
resizes it, or a rate rule or an assignment rule sets it) is stored as its
count over the load-time size, so that a resize leaves the count alone (#741).
Three places read that stored value where the size had moved, with no error:

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
