"""``steady_state()`` on a model whose rates read the time, or that has an
event (issue #710).

Both steady-state solvers look for a root of ``f(y)`` and read the right-hand
side at ``t = 0``: the residual, the Newton step, the Jacobian and ``∂f/∂p``.
The march integrates at the true time and tests convergence at ``t = 0``, and
fires no event. So ``A -> B`` at ``k·(1 − e^(−t))``, whose rate is 0 at
``t = 0``, came back ``[1, 0]`` with ``converged=True`` after one step, where A
goes to 0; and ``A' = kp − kd·A`` with an event that sets ``kp = 1`` at
``t = 2`` came back ``A = 0`` for ``1/kd``, with ``dA/dkd = 0`` for
``−1/kd²``. Nothing was logged.

A rate that reads the time has no ``f(y) = 0`` without a time being chosen, and
an event needs a trajectory. Both are refused now, by name, as is a model
whose species or compartment size is reported through a rule on the time: that
came back at its value at ``t = 0`` beside a state marked converged.

A function that reads the time and that only reports, ``minutes := time/60``,
was refused with them, though the state does not depend on it (issue #961).
The state is solved for now, and the function reported as NaN, having no
steady value.

The early stop of ``run(steady_state=True)`` tests ``‖f(t, y)‖`` at an output
point, which for such a model says nothing of the next: a rate switched on at
``t = 5`` is 0 at ``t = 1``, and the run stopped there. It goes to the end of
its span now.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

RAMPED = """begin parameters
    1 k 1
    2 A0 1
end parameters
begin functions
    1 kt() {law}
    2 seen() 2*time()
end functions
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 {rate}
end reactions
"""

EVENT = (
    "species A; A = 1; kp = 0; kd = 1\nJ0: -> A; kp\nJ1: A -> ; kd*A\nE1: at (time >= 2): kp = 1\n"
)


def _ramped(tmp_path, law="k*(1-exp(-time()))", rate="kt"):
    path = tmp_path / "ramped.net"
    path.write_text(RAMPED.format(law=law, rate=rate))
    return bngsim.Model.from_net(str(path))


def _refused(call, what):
    with pytest.raises(bngsim.SimulationError, match=rf"{what}.*\(issue #710\)"):
        call()


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_rate_that_reads_the_time_is_refused(tmp_path, method):
    """A(t) = exp(−k·(t − 1 + e^(−t))) goes to 0. It came back [1, 0],
    converged, with a residual of 0."""
    sim = bngsim.Simulator(_ramped(tmp_path), method="ode")
    _refused(lambda: sim.steady_state(method=method), "it reads the time, through")


def test_a_rate_that_reads_the_time_through_a_derived_rate_is_refused(tmp_path):
    """The function is read through another."""
    path = tmp_path / "nested.net"
    path.write_text(
        RAMPED.format(law="k*(1-exp(-time()))", rate="outer").replace(
            "    2 seen() 2*time()\n", "    2 seen() 2*time()\n    3 outer() 0.5*kt()\n"
        )
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    _refused(sim.steady_state, "it reads the time, through")


def test_a_rate_that_reads_the_time_in_a_later_reaction_is_refused(tmp_path):
    """Every reaction is asked, not the first alone."""
    path = tmp_path / "later.net"
    text = RAMPED.format(law="k*(1-exp(-time()))", rate="kt")
    assert "    1 1 2 kt\n" in text
    path.write_text(text.replace("    1 1 2 kt\n", "    1 1 2 k\n    2 2 1 kt\n"))
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    _refused(sim.steady_state, "it reads the time, through")


def test_a_table_indexed_by_time_is_refused(tmp_path):
    law = "k*tfun([0,1,2,50],[0,0.5,1,1],time)"
    sim = bngsim.Simulator(_ramped(tmp_path, law=law), method="ode")
    _refused(sim.steady_state, "it reads the time, through")


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_model_with_an_event_is_refused(method):
    """After the event A' = 1 − kd·A, so A goes to 1/kd. It came back 4.3e-10
    by integration and 0 by Newton, converged."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), method="ode")
    _refused(lambda: sim.steady_state(method=method), "1 event")


def test_the_sensitivity_of_a_model_with_an_event_is_refused():
    """dA/dkd came back 0 for −1/kd²."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), method="ode")
    _refused(lambda: sim.steady_state(sensitivity_params=["kd"]), "1 event")


# A compartment size that switches on a species, with no event of the model's
# own: the loader keeps the amounts in it across each jump with an event (issue
# #1054), one each way.
SIZE_JUMP = (
    "compartment c = 1; compartment cv; cv := piecewise(1 + 0.1*S, S > 3, 1 + 0.2*S)\n"
    "species S in c = 10; species Sn in cv = 5; k = 0.5\nJ0: S -> ; k*S*c\n"
)


def test_a_compartment_size_that_jumps_is_refused_by_name():
    """The refusal said "it has 2 events" of a model with none."""
    from bngsim._bngsim_core import SteadyStateOptions, find_steady_state

    model = bngsim.Model.from_antimony_string(SIZE_JUMP)
    assert model.n_events == 2
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.SimulationError, match=r"\(issue #710\)") as err:
        sim.steady_state()
    assert "the size of compartment cv, which an assignment rule sets, jumps" in str(err.value)
    assert "event" not in str(err.value)
    with pytest.raises(RuntimeError, match=r"the size of compartment 'cv'.*issue #710"):
        find_steady_state(model._core, SteadyStateOptions())


def test_an_event_beside_a_compartment_size_that_jumps_is_counted_alone():
    model = bngsim.Model.from_antimony_string(SIZE_JUMP + "E1: at (time >= 1): k = 0.4\n")
    assert model.n_events == 3
    _refused(bngsim.Simulator(model, method="ode").steady_state, r"it has 1 event\.")


def test_a_batch_is_refused_for_both(tmp_path):
    """steady_state_batch came back converged at [1, 6e-9] and at A = 0."""
    ramped = bngsim.Simulator(_ramped(tmp_path), method="ode")
    _refused(
        lambda: ramped.steady_state_batch([{"k": 1.0}, {"k": 2.0}]), "it reads the time, through"
    )
    event = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), method="ode")
    _refused(lambda: event.steady_state_batch([{"kd": 1.0}, {"kd": 2.0}]), "1 event")


def test_the_run_the_message_points_at_reaches_the_steady_state(tmp_path):
    """Control. ``run()`` integrates at the true time, with the event."""
    ramped = bngsim.Simulator(_ramped(tmp_path), method="ode")
    out = ramped.run(t_span=(0.0, 200.0), n_points=201)
    np.testing.assert_allclose(np.asarray(out.species)[-1], [0.0, 1.0], atol=1e-6)
    event = bngsim.Simulator(bngsim.Model.from_antimony_string(EVENT), method="ode")
    out = event.run(t_span=(0.0, 200.0), n_points=201)
    names = list(out.species_names)
    assert np.asarray(out.species)[-1, names.index("A")] == pytest.approx(1.0, abs=1e-6)


def _expression(result, name):
    return float(np.asarray(result._expression_values)[list(result.expression_names).index(name)])


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_function_that_reads_the_time_and_only_reports_is_nan(tmp_path, method):
    """``seen() = 2*time()`` is read by no reaction: A -> B at k settles at
    [0, 1] whatever the time. #710 refused it, and before that it came back 0,
    its value at t = 0 (issue #961)."""
    sim = bngsim.Simulator(_ramped(tmp_path, law="k"), method="ode")
    out = sim.steady_state(method=method)
    assert out.converged
    np.testing.assert_allclose(out.concentrations, [0.0, 1.0], atol=1e-8)
    assert np.isnan(_expression(out, "seen"))
    assert _expression(out, "kt") == pytest.approx(1.0)


def test_its_sensitivity_row_is_nan_and_the_state_s_is_not(tmp_path):
    """dB/dk = 0 at a steady state of A -> B; seen's row has no value."""
    sim = bngsim.Simulator(_ramped(tmp_path, law="k"), method="ode")
    out = sim.steady_state(sensitivity_params=["k"])
    np.testing.assert_allclose(np.asarray(out.sensitivity).ravel(), [0.0, 0.0], atol=1e-8)
    names = list(out.expression_names)
    rows = np.asarray(out.sensitivities_expressions)
    assert np.isnan(rows[names.index("seen")]).all()
    assert np.isfinite(rows[names.index("kt")]).all()
    with pytest.raises(ValueError, match=r"'expression:seen'.*reads the time.*issue #961"):
        out.output_sensitivities("expression:seen")


def test_a_batch_of_such_a_model_is_solved(tmp_path):
    rows = bngsim.Simulator(_ramped(tmp_path, law="k"), method="ode").steady_state_batch(
        [{"k": 1.0}, {"k": 2.0}]
    )
    for row in rows:
        np.testing.assert_allclose(row.concentrations, [0.0, 1.0], atol=1e-8)
        assert np.isnan(_expression(row, "seen"))


def test_a_function_that_reads_the_time_through_another_is_nan(tmp_path):
    path = tmp_path / "nested_report.net"
    path.write_text(
        RAMPED.format(law="k", rate="kt").replace(
            "    2 seen() 2*time()\n", "    2 seen() 2*time()\n    3 later() 1 + seen()\n"
        )
    )
    out = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state()
    assert out.converged
    assert np.isnan(_expression(out, "later"))


def test_a_table_on_the_time_that_only_reports_is_nan(tmp_path):
    path = tmp_path / "table_report.net"
    path.write_text(
        RAMPED.format(law="k", rate="kt").replace(
            "    2 seen() 2*time()\n", "    2 seen() tfun([0,1,2,50],[0,0.5,1,1],time)\n"
        )
    )
    out = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").steady_state()
    np.testing.assert_allclose(out.concentrations, [0.0, 1.0], atol=1e-8)
    assert np.isnan(_expression(out, "seen"))


def test_an_sbml_rule_on_the_time_that_only_reports_is_nan():
    """BIOMD0000000739 and 951 report the time in minutes, ``minutes :=
    time/60``, and nothing reads it."""
    text = "species A, B; A = 1; B = 0; kf = 1; kr = 0.5; minutes := time/60\n"
    text += "J1: A -> B; kf*A; J2: B -> A; kr*B\n"
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    out = sim.steady_state()
    names = list(out.species_names)
    got = np.asarray(out.concentrations)
    assert got[names.index("A")] == pytest.approx(1 / 3, rel=1e-6)
    assert got[names.index("B")] == pytest.approx(2 / 3, rel=1e-6)
    assert np.isnan(_expression(out, "minutes"))


def test_the_core_solver_reports_such_a_function_as_nan(tmp_path):
    from bngsim._bngsim_core import SteadyStateOptions, find_steady_state

    core = _ramped(tmp_path, law="k")._core
    assert core.functions_use_time and not core.rate_laws_read_clock
    assert core.functions_reading_clock == ["seen"]
    out = find_steady_state(core, SteadyStateOptions())
    values = dict(zip(out.expression_names, out.expression_values, strict=False))
    assert np.isnan(values["seen"]) and values["kt"] == pytest.approx(1.0)


def test_the_time_written_bare_is_refused(tmp_path):
    sim = bngsim.Simulator(_ramped(tmp_path, law="k*(1-exp(-time))"), method="ode")
    _refused(sim.steady_state, "it reads the time, through")


RULE_ON_TIME = (
    "species A, B, S; A = 1; B = 0; kf = 1; kr = 0.5\n"
    "S := B*(1 - exp(-time)); J1: A -> B; kf*A; J2: B -> A; kr*B\n"
)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_a_species_set_by_a_rule_on_the_time_is_refused(method):
    """No rate reads the time, and S does: it came back 0 for 2/3, converged,
    with dS/dkf = 0 for 2/9."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(RULE_ON_TIME), method="ode")
    _refused(lambda: sim.steady_state(method=method), "it reads the time, through")
    _refused(lambda: sim.steady_state(sensitivity_params=["kf"]), "it reads the time, through")
    _refused(lambda: sim.steady_state_batch([{"kf": 1.0}]), "it reads the time, through")


def test_a_kinetic_law_that_reads_the_time_in_sbml_is_refused():
    text = "species A, B; A = 1; B = 0; k = 1\nJ1: A -> B; k*(1 - exp(-time))*A\n"
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    _refused(sim.steady_state, "it reads the time, through")


def test_a_compartment_sized_by_a_rule_on_the_time_is_refused():
    """A and B are amounts in a compartment that grows from 1 to 2: their
    concentrations came back 0.5 each, for 0.25."""
    text = (
        "compartment C; C := 1 + time/(1+time); substanceOnly species A in C, B in C;"
        " A = 1; B = 0; k = 1\nJ: A -> B; k*A\nJ2: B -> A; k*B\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    _refused(sim.steady_state, "it reads the time, through")


def test_a_function_given_a_body_that_reads_the_time_is_refused(tmp_path):
    """The engine's hook for replacing how a function is evaluated: a model
    built with ``kt() = k`` and given ``k*(1 - exp(-time()))`` afterwards
    reads the time as one built with it does."""
    model = _ramped(tmp_path, law="k").clone()
    model._core.set_function_eval_expression("seen", "2*k")
    model._core.set_function_eval_expression("kt", "k*(1-exp(-time()))")
    assert model._core.functions_use_time
    _refused(bngsim.Simulator(model, method="ode").steady_state, "it reads the time, through")


def test_a_function_given_a_body_that_reads_no_time_is_not_marked(tmp_path):
    """Control. A name that only holds the letters, ``lifetime``, is not the
    time."""
    path = tmp_path / "ab.net"
    path.write_text(
        "begin parameters\n 1 k 1\n 2 lifetime 2\n 3 timeout 3\nend parameters\n"
        "begin functions\n 1 kt() k\nend functions\n"
        "begin species\n 1 A() 1\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 kt\nend reactions\n"
    )
    model = bngsim.Model.from_net(str(path))
    model._core.set_function_eval_expression("kt", "k*lifetime/timeout")
    assert not model._core.functions_use_time
    out = bngsim.Simulator(model, method="ode").steady_state()
    assert out.converged


def test_the_core_solver_refuses_too(tmp_path):
    """The solver itself, under the Python layer."""
    from bngsim._bngsim_core import SteadyStateOptions, find_steady_state

    for model in (_ramped(tmp_path), bngsim.Model.from_antimony_string(EVENT)):
        with pytest.raises(RuntimeError, match=r"issue #710"):
            find_steady_state(model._core, SteadyStateOptions())


# ─── The early stop of a time course ────────────────────────────────────────

SWITCHED = {
    "a-rate-switched-on": "if(time()>5,k,0)",
    "a-table-on-the-time": "k*tfun([0,50,51,1e9],[0,0,1,1],time)",
    # No condition and no knot: the run is on the solver's warm path, which
    # has an early stop of its own. The rate is 1.5e-13·k at t = 1.
    "a-smooth-rate-that-starts-at-nothing": "k*time()^8/(40^8+time()^8)",
}


def _early(sim, **kw):
    out = sim.run(t_span=(0.0, 200.0), n_points=201, steady_state=True, **kw)
    return out, np.asarray(out.species)


@pytest.mark.parametrize("case", sorted(SWITCHED))
def test_a_time_course_does_not_stop_before_a_rate_is_switched_on(tmp_path, case):
    """``A -> B`` at a rate that is 0 until t = 5, or t = 50: ‖f‖ is 0 at the
    first output point, and the run stopped at t = 1 with [1, 0] marked
    steady, for [0, 1]."""
    out, species = _early(bngsim.Simulator(_ramped(tmp_path, law=SWITCHED[case]), method="ode"))
    assert len(species) == 201 and out.time[-1] == 200.0
    assert not out.solver_stats["steady_state_reached"]
    np.testing.assert_allclose(species[-1], [0.0, 1.0], atol=1e-6)


def test_a_time_course_does_not_stop_before_an_event():
    """``A' = kp − kd·A`` from 1 with kp = 0 until an event at t = 50 sets it to
    1: the run stopped at t = 19 with A = 0, for 1."""
    text = EVENT.replace("time >= 2", "time >= 50")
    out, species = _early(bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode"))
    assert len(species) == 201 and not out.solver_stats["steady_state_reached"]
    assert species[-1, list(out.species_names).index("A")] == pytest.approx(1.0, abs=1e-6)


def test_a_batch_row_does_not_stop_early_either(tmp_path):
    sim = bngsim.Simulator(_ramped(tmp_path, law=SWITCHED["a-rate-switched-on"]), method="ode")
    rows = sim.run_batch(
        params=[{"k": 1.0}, {"k": 2.0}], t_span=(0.0, 200.0), n_points=201, steady_state=True
    )
    for row in rows:
        assert len(np.asarray(row.species)) == 201
        np.testing.assert_allclose(np.asarray(row.species)[-1], [0.0, 1.0], atol=1e-6)


def test_a_time_course_of_a_model_that_reads_no_time_stops_early(tmp_path):
    """Control. A -> B at k, with neither an event nor the time anywhere."""
    path = tmp_path / "plain.net"
    path.write_text(
        "begin parameters\n 1 k 1\nend parameters\n"
        "begin species\n 1 A() 1\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 k\nend reactions\n"
    )
    out, species = _early(bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode"))
    assert len(species) < 201 and out.solver_stats["steady_state_reached"]
    np.testing.assert_allclose(species[-1], [0.0, 1.0], atol=1e-6)
    # And with a sensitivity requested, which takes the solver's other path.
    sens = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["k"]
    )
    out, species = _early(sens)
    assert len(species) < 201 and out.solver_stats["steady_state_reached"]


@pytest.mark.parametrize("sensitivity", [False, True], ids=["plain", "with a sensitivity"])
def test_a_time_course_stops_early_beside_a_function_that_only_reports(tmp_path, sensitivity):
    """``seen() = 2*time()`` is read by no rate: A -> B settles as it would
    without it, and the run stops there. #710 ran it to the end of its span
    (issue #961). The rows it reports read the time where they were taken."""
    kw = {"sensitivity_params": ["k"]} if sensitivity else {}
    sim = bngsim.Simulator(_ramped(tmp_path, law="k"), method="ode", **kw)
    out, species = _early(sim)
    assert len(species) < 201 and out.solver_stats["steady_state_reached"]
    np.testing.assert_allclose(species[-1], [0.0, 1.0], atol=1e-6)
    seen = np.asarray(out.expressions)[:, list(out.expression_names).index("seen")]
    np.testing.assert_allclose(seen, 2 * np.asarray(out.time), rtol=1e-12)


def test_a_batch_row_stops_early_beside_a_function_that_only_reports(tmp_path):
    rows = bngsim.Simulator(_ramped(tmp_path, law="k"), method="ode").run_batch(
        params=[{"k": 1.0}, {"k": 2.0}], t_span=(0.0, 200.0), n_points=201, steady_state=True
    )
    for row in rows:
        assert len(np.asarray(row.species)) < 201


@pytest.mark.parametrize(
    "text",
    [
        RULE_ON_TIME,
        "compartment C; C := 1 + time/(1+time); substanceOnly species A in C, B in C;"
        " A = 1; B = 0; k = 1\nJ: A -> B; k*A\nJ2: B -> A; k*B\n",
    ],
    ids=["a species set by a rule on the time", "a compartment sized by one"],
)
def test_a_time_course_does_not_stop_while_a_reported_state_reads_the_time(text):
    """No rate reads the time, and what is reported as a state does: S, or the
    concentrations in C, are still moving where ``‖f(y)‖`` has settled."""
    out, species = _early(bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode"))
    assert len(species) == 201 and not out.solver_stats["steady_state_reached"]
    rows = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode").run_batch(
        params=[{"kf" if "kf" in text else "k": 1.0}],
        t_span=(0.0, 200.0),
        n_points=201,
        steady_state=True,
    )
    assert all(len(np.asarray(row.species)) == 201 for row in rows)


def test_a_rate_that_reads_its_own_derivative_is_not_refused():
    """Control. ``rateOf`` reads the state's derivatives, which are 0 at a
    steady state whatever the time: B settles at 1 + rateOf(A) = 1."""
    text = (
        "species A, B; A = 3; B = 0; kp = 1; kd = 0.5\n"
        "J0: -> A; kp\nJ1: A -> ; kd*A\nJ2: -> B; 1 + rateOf(A)\nJ3: B -> ; B\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(text), method="ode")
    out = sim.steady_state()
    assert out.converged
    names = list(out.species_names)
    got = np.asarray(out.concentrations)
    assert got[names.index("A")] == pytest.approx(2.0, rel=1e-6)
    assert got[names.index("B")] == pytest.approx(1.0, rel=1e-6)


def test_a_model_with_neither_is_solved_as_it_was(tmp_path):
    """Control."""
    path = tmp_path / "ab.net"
    path.write_text(
        "begin parameters\n 1 kf 1\n 2 kr 0.5\nend parameters\n"
        "begin species\n 1 A() 3\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 kf\n 2 2 1 kr\nend reactions\n"
    )
    sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode")
    out = sim.steady_state(sensitivity_params=["kf"])
    np.testing.assert_allclose(out.concentrations, [1.0, 2.0], rtol=1e-7)
    rows = sim.steady_state_batch([{"kf": 1.0}, {"kf": 2.0}])
    assert len(rows) == 2
