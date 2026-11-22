"""``Result.rerun()`` (issue #1703): the run a result records, made again.

A rerun builds the model from the recorded source and gives it the recorded
start, so the oracle here is the original run itself: an ODE run must come back
bit for bit, and a stochastic one too, on this build. Each case is a way a run
can start or be set up that the record has to carry for that to hold: a live
state after a ``set_param``, a delayed event still pending, each kind of
absolute tolerance, a Jacobian retried on a difference quotient, a batch row, a
replicate, a scan point, a ``run_until`` step. Where the record cannot promise
the same run, the rerun has to refuse with a code, not run something else.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ExactAtol, Model, Result, Simulator, TrackingAtol
from bngsim._exceptions import ModelError, UsageError
from bngsim.convert._bng2 import find_bng2

NET = """\
begin parameters
    1 kf 0.5
    2 kr 0.25
    3 A0 100
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""

# B jumps to 100 one time unit after t = 0.5: a run over (0, 1) leaves the
# execution pending, and the run that continues from t = 1 applies it at 1.5.
DELAYED = """
model delayed
  J1: A -> B; k*A
  E1: at 1 after (time >= 0.5): B = 100
  k = 0.3
  A = 10
  B = 0
end
"""

# S empties at t = 2, where d/dS sqrt(S) is infinite: the analytical attempt
# fails and the run is made again on a difference quotient (GH #176, #867).
SINK = """
model sink
  J1: S -> ; k*sqrt(S)
  k = 1
  S = 1
end
"""


@pytest.fixture
def net_path(tmp_path: Path) -> Path:
    path = tmp_path / "ab.net"
    path.write_text(NET, encoding="utf-8")
    return path


def _same(a: Result, b: Result) -> None:
    """*b* holds *a*'s numbers, bit for bit."""
    np.testing.assert_array_equal(b.time, a.time)
    np.testing.assert_array_equal(np.asarray(b.species), np.asarray(a.species))
    np.testing.assert_array_equal(np.asarray(b.observables), np.asarray(a.observables))


# ─── The same numbers ──────────────────────────────────────────────────


def test_a_run_from_a_live_state_after_a_set_param_comes_back_bit_for_bit(
    net_path: Path,
) -> None:
    model = Model.from_net(net_path)
    sim = Simulator(model)
    model.set_param("kf", 0.9)
    sim.run(t_span=(0, 3), n_points=4)
    first = sim.run(t_span=(3, 8), n_points=11, start="live", rtol=1e-9)
    again = first.rerun()
    _same(first, again)
    assert again.record["run"]["parameters"] == first.record["run"]["parameters"]
    assert again.record["run"]["start_state_sha256"] == first.start_state_sha256
    assert again.notices == []


def test_the_model_the_run_used_is_neither_read_nor_changed(net_path: Path) -> None:
    """A rerun inherits nothing a later run left behind (issue #1701), and
    leaves the model a session is using as it was."""
    model = Model.from_net(net_path)
    sim = Simulator(model)
    first = sim.run(t_span=(0, 5), n_points=6)
    model.set_param("kf", 7.0)
    sim.run(t_span=(5, 9), n_points=3)
    state, kf = model.get_state(), model.get_param("kf")

    _same(first, first.rerun())
    np.testing.assert_array_equal(model.get_state(), state)
    assert model.get_param("kf") == kf
    # And the Simulator's next run continues from where it was.
    assert sim.run(t_span=(9, 10), n_points=2).started_from == "live"


def test_a_delayed_event_pending_at_the_start_is_carried() -> None:
    """The second run starts with B's jump pending at t = 1.5. A rerun from the
    state alone would never apply it."""
    sim = Simulator(Model.from_antimony_string(DELAYED))
    sim.run(t_span=(0, 1), n_points=3)
    second = sim.run(t_span=(1, 3), n_points=5)
    carry = second.record["run"]["start_carry"]
    assert carry == {
        "events": {
            "t": 1.0,
            "triggers": [True],
            "pending": [{"event": 0, "apply_time": 1.5, "values": [100.0]}],
        },
        "ssa_counts": False,
    }
    assert second.observables["B"][-1] > 100
    _same(second, second.rerun())
    # The record of a fresh start holds no carry.
    first = Simulator(Model.from_antimony_string(DELAYED)).run(t_span=(0, 1), n_points=2)
    assert first.record["run"]["start_carry"] is None


@pytest.mark.parametrize("method", ["ssa", "psa"])
def test_a_stochastic_run_comes_back_with_its_seed(net_path: Path, method: str) -> None:
    kw = {"poplevel": 20} if method == "psa" else {}
    sim = Simulator(Model.from_net(net_path), method=method, **kw)
    sim.run(t_span=(0, 2), n_points=3)
    second = sim.run(t_span=(2, 6), n_points=9)
    _same(second, second.rerun())
    other = second.rerun(seed=second.seed + 1)
    assert not np.array_equal(np.asarray(other.species), np.asarray(second.species))
    assert other.record["run"]["rerun_of"]["changed"] == {"seed": second.seed + 1}


@pytest.mark.parametrize(
    "atol",
    [
        None,
        1e-6,
        "auto",
        "tracking",
        TrackingAtol(decades=4),
        ExactAtol(1e-9),
        [1e-7, 1e-9],
    ],
    ids=["default", "scalar", "auto", "tracking", "tracking4", "exact", "vector"],
)
def test_every_absolute_tolerance_is_given_back_as_it_was_applied(
    net_path: Path, atol: object
) -> None:
    """A scalar is read as a ceiling over the start's scales (#1420), and a
    tracking ceiling is made from the start: given again as the call gave it,
    either would be a second reading. The record's applied values go back."""
    model = Model.from_net(net_path)
    sim = Simulator(model)
    sim.run(t_span=(0, 1), n_points=2)
    model.set_concentration("B()", 1e-6)
    r = sim.run(t_span=(1, 6), n_points=6, rtol=1e-7, atol=atol)
    again = r.rerun()
    _same(r, again)
    assert again.record["run"]["atol"] == r.record["run"]["atol"]


def test_tolerances_set_on_the_simulator_and_sample_times(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path))
    sim.set_tolerances(rtol=1e-9, atol=1e-11)
    r = sim.run(sample_times=[0.0, 0.3, 2.0, 5.0])
    again = r.rerun()
    _same(r, again)
    assert again.record["run"]["sample_times"] == [0.0, 0.3, 2.0, 5.0]


def test_a_steady_state_run_and_a_step_bound(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 500), n_points=4, steady_state=True)
    _same(r, r.rerun())
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 5), n_points=4, max_step=0.1)
    _same(r, r.rerun())


def test_the_jacobian_a_retried_run_used_is_the_one_used_again() -> None:
    """The run's analytical attempt failed and it was made again on a
    difference quotient; a rerun that tried the analytical one first would
    take another path. So would a later run, which kept the quotient."""
    sim = Simulator(Model.from_antimony_string(SINK))
    r = sim.run(t_span=(0, 40), n_points=5)
    assert r.record["backend"]["jacobian_retried"] is True
    again = r.rerun()
    _same(r, again)
    assert again.record["backend"]["jacobian"] == "difference_quotient"
    assert "rerun.different_backend" not in [n["code"] for n in again.notices]
    kept = sim.run(t_span=(0, 40), n_points=5, start="initial")
    _same(kept, kept.rerun())


def test_an_interpreted_run_stays_interpreted(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path), codegen=False).run(t_span=(0, 5), n_points=4)
    assert r.record["backend"]["rhs"] == "interpreted"
    again = r.rerun()
    assert again.record["backend"]["rhs"] == "interpreted"
    _same(r, again)


@pytest.mark.parametrize("solver", ["dense", "fd"])
def test_the_requested_linear_solver_and_jacobian(net_path: Path, solver: str) -> None:
    kw = {"force_dense_linear_solver": True} if solver == "dense" else {"jacobian": "fd"}
    r = Simulator(Model.from_net(net_path), **kw).run(t_span=(0, 5), n_points=4)
    again = r.rerun()
    _same(r, again)
    assert again.record["backend"] == {**r.record["backend"]}


def test_a_sensitivity_run_comes_back_with_its_sensitivities(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path), sensitivity_params=["kf", "A0"])
    r = sim.run(t_span=(0, 5), n_points=4)
    again = r.rerun()
    _same(r, again)
    np.testing.assert_array_equal(np.asarray(again.sensitivities), np.asarray(r.sensitivities))


def test_rows_of_a_batch_replicates_and_a_scan_and_run_until_steps(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path))
    for row in sim.run_batch((0, 5), 4, params=[{"kf": 0.2}, {"kf": 2.0}], squeeze=False):
        _same(row, row.rerun())
    for point in sim.parameter_scan("kf", [0.2, 2.0], t_span=(0, 5), n_points=4):
        _same(point, point.rerun())
    for step in [sim.run_until(t) for t in (1.0, 2.5)]:
        _same(step, step.rerun())
    ssa = Simulator(Model.from_net(net_path), method="ssa")
    for rep in ssa.run_replicates(2, t_span=(0, 5), n_points=4, squeeze=False):
        _same(rep, rep.rerun())


def test_a_model_from_text_reruns_and_so_does_its_saved_result(tmp_path: Path) -> None:
    """A model loaded from a string has no file: the text is kept, and a saved
    result keeps it too."""
    pytest.importorskip("h5py")
    sim = Simulator(Model.from_antimony_string(DELAYED))
    sim.run(t_span=(0, 1), n_points=2)
    second = sim.run(t_span=(1, 3), n_points=5)
    assert second.record["model"]["path"] is None
    _same(second, second.rerun())
    second.save(tmp_path / "r.h5")
    loaded = Result.load(tmp_path / "r.h5")
    _same(second, loaded.rerun())


def test_a_saved_result_of_a_file_model_reruns(net_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    model = Model.from_net(net_path)
    sim = Simulator(model)
    model.set_param("kr", 0.1)
    sim.run(t_span=(0, 2), n_points=2)
    r = sim.run(t_span=(2, 7), n_points=6, atol="auto")
    r.save(tmp_path / "r.h5")
    _same(r, Result.load(tmp_path / "r.h5").rerun())


def test_a_derived_parameter_set_by_hand_is_set_again() -> None:
    """``k2 = 2*k`` follows ``k`` unless set itself; the rerun sets only what
    differs, the primary parameter first, so neither reading is lost."""
    text = "model p\n  J1: A -> B; k2*A\n  k = 0.5\n  k2 = 2*k\n  A = 10\nend\n"
    model = Model.from_antimony_string(text)
    model.set_param("k", 0.9)
    r = Simulator(model).run(t_span=(0, 3), n_points=4)
    _same(r, r.rerun())
    model.set_param("k2", 5.0)
    r = Simulator(model).run(t_span=(0, 3), n_points=4, start="initial")
    assert r.record["run"]["parameters"]["k2"] == 5.0
    _same(r, r.rerun())


# ─── Changing a setting ────────────────────────────────────────────────


def test_a_changed_setting_is_recorded_and_the_rest_are_kept(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 5), n_points=4, atol="auto")
    tight = r.rerun(rtol=1e-11)
    rec, run = r.record, tight.record["run"]
    assert run["rtol"] == 1e-11
    # The absolute tolerances the run was given, kept.
    assert run["atol"]["per_species"] == rec["run"]["atol"]["per_species"]
    expected = hashlib.sha256(
        json.dumps(rec, sort_keys=True, default=repr).encode("utf-8")
    ).hexdigest()
    assert run["rerun_of"] == {"record_sha256": expected, "changed": {"rtol": 1e-11}}
    assert r.record["run"]["rerun_of"] is None
    np.testing.assert_allclose(tight.species, r.species, rtol=1e-6)
    assert not np.array_equal(np.asarray(tight.species), np.asarray(r.species))


def test_a_new_method_drops_the_old_ones_settings(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 5), n_points=4)
    ssa = r.rerun(method="ssa", seed=3)
    run = ssa.record["run"]
    assert run["method"] == "ssa" and run["seed"] == 3
    assert run["rtol"] is None and run["atol"] is None
    assert run["rerun_of"]["changed"] == {"method": "ssa", "seed": 3}


def test_new_times(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 5), n_points=4)
    assert r.rerun(n_points=11).n_times == 11
    assert r.rerun(sample_times=[0.0, 1.0]).n_times == 2
    longer = r.rerun(t_span=(0, 10))
    assert longer.time[-1] == 10.0
    assert longer.record["run"]["rerun_of"]["changed"] == {"t_span": [0.0, 10.0]}


# ─── What it refuses ───────────────────────────────────────────────────


def _code(exc: pytest.ExceptionInfo) -> tuple[str, dict]:
    return exc.value.code, exc.value.details


def test_a_setting_it_does_not_change(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    with pytest.raises(UsageError) as e:
        r.rerun(rtoll=1e-9)
    assert e.value.code == "rerun.unknown_setting"
    assert e.value.details["settings"] == ["rtoll"]
    with pytest.raises(UsageError) as e:
        r.rerun(n_points=3, sample_times=[0.0, 1.0])
    assert e.value.code == "usage.conflicting_arguments"


def test_a_model_file_changed_or_gone(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    net_path.write_text(NET.replace("1 kf 0.5", "1 kf 0.6"), encoding="utf-8")
    with pytest.raises(ModelError) as e:
        r.rerun()
    assert e.value.code == "rerun.model_changed"
    assert e.value.details["recorded_sha256"] == r.record["model"]["sha256"]
    net_path.unlink()
    with pytest.raises(ModelError) as e:
        r.rerun()
    assert _code(e) == ("rerun.model_missing", {"path": str(net_path.resolve())})


def test_a_stack_and_calls_that_record_no_run(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path))
    stack = sim.run_batch((0, 1), 2, params=[{"kf": 0.2}, {"kf": 2.0}], squeeze=True)
    with pytest.raises(UsageError) as e:
        stack.rerun()
    assert _code(e) == ("rerun.unsupported", {"reason": "stack"})
    whole = sim.compute_all_sensitivities(t_span=(0, 1), n_points=3)
    with pytest.raises(UsageError) as e:
        whole.rerun()
    assert _code(e) == ("rerun.not_recorded", {"missing": "run"})


def test_a_model_built_in_python() -> None:
    builder = bngsim.ModelBuilder()
    builder.add_parameter("k", 0.5)
    builder.add_species("A", 10.0)
    builder.add_reaction(["A"], [], rate="k")
    r = Simulator(builder.build()).run(t_span=(0, 1), n_points=2)
    with pytest.raises(UsageError) as e:
        r.rerun()
    assert _code(e) == ("rerun.not_recorded", {"missing": "model source"})


def test_a_run_with_stop_conditions(net_path: Path) -> None:
    sim = Simulator(Model.from_net(net_path))
    sim.add_stop_condition("Atot < 1")
    r = sim.run(t_span=(0, 1), n_points=2)
    with pytest.raises(UsageError) as e:
        r.rerun()
    assert _code(e) == (
        "rerun.unsupported",
        {"reason": "stop_conditions", "stop_conditions": ["Atot < 1"]},
    )


def test_carried_and_declared_sensitivities(net_path: Path) -> None:
    model = Model.from_net(net_path)
    sim = Simulator(model, sensitivity_params=["kf"])
    sim.run(t_span=(0, 1), n_points=2)
    carried = sim.run(t_span=(1, 2), n_points=2, carry_sensitivities=True)
    with pytest.raises(UsageError) as e:
        carried.rerun()
    assert _code(e) == ("rerun.unsupported", {"reason": "carry_sensitivities"})

    model = Model.from_net(net_path)
    model.declare_ic_sensitivity({"A()": {"kf": 2.0}})
    declared = Simulator(model, sensitivity_params=["kf"]).run(t_span=(0, 1), n_points=2)
    assert declared.record["run"]["declared_ic_sensitivity"] is True
    with pytest.raises(UsageError) as e:
        declared.rerun()
    assert _code(e) == ("rerun.unsupported", {"reason": "declared_ic_sensitivity"})


def test_a_model_given_table_data_after_its_load(tmp_path: Path) -> None:
    text = """begin parameters
    1 k 1
end parameters
begin functions
    1 f() tfun([0, 10], [1, 2], time)
end functions
begin species
    1 A() 10
end species
begin reactions
    1 1 0 f
end reactions
"""
    path = tmp_path / "t.net"
    path.write_text(text, encoding="utf-8")
    model = Model.from_net(path)
    name = model.table_function_names[0]
    model.add_table_function(name, times=[0.0, 10.0], values=[3.0, 4.0])
    r = Simulator(model).run(t_span=(0, 1), n_points=2)
    assert r.record["model"]["edits"] == ["add_table_function"]
    with pytest.raises(UsageError) as e:
        r.rerun()
    assert _code(e) == (
        "rerun.unsupported",
        {"reason": "model_edited", "edits": ["add_table_function"]},
    )


@pytest.mark.skipif(not bngsim.HAS_NFSIM, reason="NFsim not compiled in")
def test_a_network_free_run() -> None:
    from bngsim.examples import path as example_path

    model = Model.load(example_path("binding", "net"))
    r = Simulator(model, method="nf", xml_path=str(example_path("binding", "bng-xml"))).run(
        t_span=(0, 1), n_points=2
    )
    with pytest.raises(UsageError) as e:
        r.rerun()
    assert _code(e) == ("rerun.unsupported", {"reason": "network_free", "method": "nfsim"})


# ─── What it says ──────────────────────────────────────────────────────


def _with_record_part(result: Result, part: str, value: dict) -> Result:
    """*result* with its record's *part* replaced, as a file written elsewhere
    would hold it."""
    from bngsim import _record

    parts = _record.frozen_parts(result)
    parts[part] = {**parts[part], **value}
    result._record_frozen = parts
    result._source_text = None
    return result


def test_another_build_runs_and_says_so(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    _with_record_part(r, "software", {"bngsim": "0.0.1", "build_commit": "f" * 40})
    with pytest.warns(UserWarning, match="0.0.1"):
        again = r.rerun()
    assert [n["code"] for n in again.notices] == ["rerun.different_build"]
    _same(r, again)


def test_other_code_runs_and_says_so(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    _with_record_part(r, "backend", {"linear_solver": "lapack_dense"})
    with pytest.warns(UserWarning, match="linear_solver"):
        again = r.rerun()
    assert [n["code"] for n in again.notices] == ["rerun.different_backend"]


def test_a_record_lacking_a_part(net_path: Path) -> None:
    r = Simulator(Model.from_net(net_path)).run(t_span=(0, 1), n_points=2)
    from bngsim import _record

    parts = _record.frozen_parts(r)
    del parts["run"]["start_carry"]
    r._record_frozen = parts
    with pytest.raises(UsageError) as e:
        r.rerun()
    assert _code(e) == ("rerun.not_recorded", {"missing": "run.start_carry"})


@pytest.mark.skipif(
    find_bng2() is None or shutil.which("perl") is None, reason="BNG2.pl / perl not available"
)
def test_a_bngl_model_reruns_from_its_file() -> None:
    """Built again from the .bngl, by way of the network BNG2.pl writes."""
    from bngsim.examples import path as example_path

    model = Model.load(example_path("binding", "bngl"))
    sim = Simulator(model)
    sim.run(t_span=(0, 1), n_points=2)
    r = sim.run(t_span=(1, 4), n_points=4)
    _same(r, r.rerun())
