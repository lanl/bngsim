"""Issue #1369: run_batch and run_replicates meet the stop conditions run() does.

``add_stop_condition`` was honoured by ``run()`` alone: ``run_batch`` and
``run_replicates`` returned every row to t_end, past the condition, with no
word. Each row is now checked as ``run()`` checks it. A row that meets a
condition comes back cut at the first sample it holds at, with
``Result.stop_condition`` naming it, beside the rows that ran to the end, and
nothing is raised: one row's stop is not the batch's. parameter_scan, which
raised out of the scan at the first point that met one, reports per point too.

A() decays at k from 100, observed as A_tot; the condition is A_tot < 10, met
at t = ln(10)/k, about 23 at k = 0.1 and never by t = 100 at k = 0.001.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = (
    "begin parameters\n 1 k 0.1\nend parameters\n"
    "begin species\n 1 A() 100\nend species\n"
    "begin reactions\n 1 1 0 k\nend reactions\n"
    "begin groups\n 1 A_tot 1\nend groups\n"
)
KW = dict(t_span=(0, 100), n_points=101, seed=1)


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "decay.net"
    path.write_text(NET)
    return str(path)


def _sim(net: str, method: str, condition="A_tot < 10", label="low") -> bngsim.Simulator:
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method=method)
    sim.add_stop_condition(condition, label=label)
    return sim


def _met_at_its_end(result) -> bool:
    """The row holds A_tot >= 10 at every sample but its last."""
    a = np.asarray(result.observables["A_tot"])
    return bool(a[-1] < 10.0 and np.all(a[:-1] >= 10.0))


@pytest.mark.parametrize("method", ["ode", "ssa"])
def test_a_batch_row_that_meets_a_condition_is_cut_there(net, method):
    """Before the fix, both rows ran to t = 100 and stop_condition did not exist."""
    rows = _sim(net, method).run_batch(params=[{"k": 0.1}, {"k": 0.001}], **KW)
    met, ran = rows
    assert met.stop_condition == "low"
    assert met.n_times < 101 and _met_at_its_end(met)
    assert ran.stop_condition is None
    assert ran.n_times == 101 and float(ran.time[-1]) == 100.0


def test_an_ode_row_is_cut_where_run_is(net):
    """The row is cut at run()'s sample, and holds run()'s values there. The
    values are held to 1e-12, not to the bit: the row is solved by the batch's
    own option builder, which the run-versus-batch contract holds to a
    tolerance (test_hpc_contract.py)."""
    with pytest.raises(bngsim.StopConditionMet) as raised:
        _sim(net, "ode").run(**KW)
    row = _sim(net, "ode").run_batch(params=[{"k": 0.1}], **KW)[0]
    np.testing.assert_array_equal(row.time, raised.value.result.time)
    np.testing.assert_allclose(row.species, raised.value.result.species, rtol=1e-12, atol=0)
    assert raised.value.result.stop_condition == row.stop_condition == "low"


@pytest.mark.parametrize("processors", [None, 2], ids=["sequential", "threads"])
def test_every_replicate_is_cut_where_it_meets_the_condition(net, processors):
    reps = _sim(net, "ssa").run_replicates(3, num_processors=processors, **KW)
    assert [r.stop_condition for r in reps] == ["low", "low", "low"]
    assert all(_met_at_its_end(r) for r in reps)


@pytest.mark.parametrize("call", ["batch", "replicates", "scan"])
def test_squeeze_is_refused_before_any_row_runs(net, call, monkeypatch):
    """Refused only once a row had been cut, squeeze=True threw away every row
    already solved; it is refused up front, as steady_state=True is with it."""
    sim = _sim(net, "ssa")

    def a_row_ran(*args, **kwargs):
        raise AssertionError("a row ran before the refusal")

    # Simulator has __slots__: patch the class. Each is the first per-row step
    # of its call.
    monkeypatch.setattr(bngsim.Simulator, "_run_single_batch", a_row_ran)
    monkeypatch.setattr(bngsim.Simulator, "_apply_ssa_breakpoints", a_row_ran)
    monkeypatch.setattr(bngsim.Simulator, "run", a_row_ran)
    with pytest.raises(ValueError, match=r"squeeze=True\) is refused while stop conditions"):
        if call == "batch":
            sim.run_batch(params=[{"k": 0.1}, {"k": 0.001}], squeeze=True, **KW)
        elif call == "replicates":
            sim.run_replicates(2, squeeze=True, **KW)
        else:
            sim.parameter_scan("k", [0.1, 0.001], squeeze=True, **KW)


def test_a_scan_point_that_meets_a_condition_is_cut_there(net):
    """parameter_scan raised StopConditionMet out of the scan at the first point
    that met a condition, and every other point was lost."""
    met, ran = _sim(net, "ode").parameter_scan("k", [0.1, 0.001], **KW)
    assert met.stop_condition == "low" and _met_at_its_end(met)
    assert ran.stop_condition is None and ran.n_times == 101
    assert [r.custom_attrs["scan_value"] for r in (met, ran)] == [0.1, 0.001]


def test_a_continuation_point_starts_where_a_cut_point_left_the_model(net):
    """A condition does not interrupt the solve, so the point after a cut one
    continues from the end of the span: A(100) = 100 e^-10, under 10 from its
    first sample."""
    first, second = _sim(net, "ode").bifurcate("k", [0.1, 0.001], **KW)
    assert first.stop_condition == "low" and first.n_times < 101
    assert second.stop_condition == "low" and second.n_times == 1


def test_result_squeeze_refuses_a_row_that_met_a_condition(net):
    rows = _sim(net, "ode").run_batch(params=[{"k": 0.1}, {"k": 0.001}], **KW)
    with pytest.raises(ValueError, match=r"met a stop condition \(1 of 2: rows 0\)"):
        bngsim.Result.squeeze(rows)


def test_squeeze_stacks_rows_none_met(net):
    """Control: the rows of a conditioned batch stack where none met it, and
    squeeze=True stacks with no condition registered."""
    params = [{"k": 0.001}, {"k": 0.002}]
    batch = bngsim.Result.squeeze(_sim(net, "ode").run_batch(params=params, **KW))
    assert batch.species.shape == (2, 101, 1)
    assert batch.stop_condition is None
    plain = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    assert plain.run_batch(params=params, squeeze=True, **KW).species.shape == (2, 101, 1)


def test_a_threaded_batch_cuts_its_rows_as_a_sequential_one(net):
    params = [{"k": 0.1}, {"k": 0.001}, {"k": 0.2}]
    seq = _sim(net, "ode").run_batch(params=params, **KW)
    par = _sim(net, "ode").run_batch(params=params, num_processors=2, **KW)
    assert [r.stop_condition for r in par] == [r.stop_condition for r in seq]
    assert [r.stop_condition for r in seq] == ["low", None, "low"]
    for a, b in zip(par, seq, strict=True):
        np.testing.assert_array_equal(a.time, b.time)


def test_a_cut_row_refuses_state_and_says_why(net):
    """A cut row carries the reported blocks only, as run()'s raised result does;
    the refusal names the stop rather than a load or a stack."""
    met, ran = _sim(net, "ode").run_batch(params=[{"k": 0.1}, {"k": 0.001}], **KW)
    assert ran.state.shape == (101, 1)
    with pytest.raises(ValueError, match=r"cut at stop condition 'low'"):
        _ = met.state


def test_a_cut_row_keeps_its_condition_through_save_and_load(net, tmp_path):
    pytest.importorskip("h5py")
    met, ran = _sim(net, "ode").run_batch(params=[{"k": 0.1}, {"k": 0.001}], **KW)
    met.save(tmp_path / "met.h5")
    ran.save(tmp_path / "ran.h5")
    assert bngsim.Result.load(tmp_path / "met.h5").stop_condition == "low"
    assert bngsim.Result.load(tmp_path / "ran.h5").stop_condition is None


def test_to_xarray_carries_the_condition(net):
    pytest.importorskip("xarray")
    met, ran = _sim(net, "ode").run_batch(params=[{"k": 0.1}, {"k": 0.001}], **KW)
    assert met.to_xarray().attrs["stop_condition"] == "low"
    assert "stop_condition" not in ran.to_xarray().attrs


@pytest.mark.parametrize("condition", ["A__ < 10", "A_tot < 10 and t > 0"])
def test_an_expression_reads_species_and_time_as_before(net, condition):
    """An expression is evaluated against the names it reads only; a species, by
    its sanitized name, and the time still read as they did."""
    want = _sim(net, "ode").run_batch(params=[{"k": 0.1}], **KW)[0]
    got = _sim(net, "ode", condition=condition).run_batch(params=[{"k": 0.1}], **KW)[0]
    assert got.stop_condition == "low"
    np.testing.assert_array_equal(got.time, want.time)


def test_a_condition_on_the_time_alone_cuts_there(net):
    row = _sim(net, "ode", condition="t >= 50").run_batch(params=[{"k": 0.001}], **KW)[0]
    assert row.n_times == 51 and float(row.time[-1]) == 50.0


def test_without_conditions_rows_run_to_the_end(net):
    """Control: nothing registered, nothing cut."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    rows = sim.run_batch(params=[{"k": 0.1}], **KW)
    assert rows[0].n_times == 101 and rows[0].stop_condition is None
    assert sim.run(**KW).stop_condition is None


def test_the_first_condition_added_that_a_row_meets_names_it(net):
    """Conditions are checked in the order they were added; an unlabelled one
    reads as its expression; a callable is called with the row."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    sim.add_stop_condition("A_tot < 50")
    sim.add_stop_condition(lambda r: True, label="always")
    rows = sim.run_batch(params=[{"k": 0.1}, {"k": 0.001}], **KW)
    assert rows[0].stop_condition == "A_tot < 50"
    assert float(rows[0].observables["A_tot"][-1]) < 50.0
    assert rows[1].stop_condition == "always"
