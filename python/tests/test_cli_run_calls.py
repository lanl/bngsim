"""The ``bngsim`` commands for the run calls beyond ``run()`` (issue #1890).

``run-until``, ``run-batch``, ``parameter-scan``, ``steady-state``,
``steady-state-batch``, ``steady-states`` and ``run-bngl``. Each is a thin layer
over one Python call, so each test holds what the command prints to what the
call itself returns, made here directly. The rest is the contract they share
with ``bngsim run``: one JSON object on standard output, arrays only to
``--out``, a wall-clock limit on every call, an option the call would not use
refused, and an exit code for each outcome.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import Model, Result, Simulator, StopConditionMet, _cli, _steady_form, examples

CHAIN = str(examples.path("chain", "sbml"))
BIRTH_DEATH = str(examples.path("birth_death", "net"))
SCHLOGL = str(examples.path("schlogl", "sbml"))


def _bngsim(*args: str, script: str | None = None, cwd=None) -> subprocess.CompletedProcess:
    if script is None:
        cmd = [sys.executable, "-m", "bngsim", *args]
    else:
        code = script + f"\nfrom bngsim._cli import main\nraise SystemExit(main({list(args)!r}))\n"
        cmd = [sys.executable, "-c", code]
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", timeout=300, cwd=cwd
    )


def _json(run: subprocess.CompletedProcess, exit_code: int = 0) -> dict:
    assert run.returncode == exit_code, (run.returncode, run.stderr)
    return json.loads(run.stdout)  # raises on anything before or after the object


def _error(run: subprocess.CompletedProcess, exit_code: int, code: str) -> dict:
    assert run.returncode == exit_code, (run.returncode, run.stderr)
    assert run.stdout == ""
    error = json.loads(run.stderr)
    assert error["code"] == code, error
    return error


def _plain(value):
    """``value`` as the command's JSON writes it: a non-finite float as a string."""
    return json.loads(_cli._dumps(value))


ROWS = [{"k1": 0.5}, {"k1": 1.0, "k2": 0.2}, {"k1": 3.0}]


# ─── run-until ───────────────────────────────────────────────────────────────


def test_run_until_a_condition_met_exits_0_with_the_result_it_cut():
    d = _json(_bngsim("run-until", CHAIN, "--t-end", "50", "--stop-when", "C > 50", "--json"))
    assert list(d) == ["model", "record", "digest", "out", "set", "stop_condition"]
    sim = Simulator(Model.load(CHAIN))
    sim.add_stop_condition("C > 50")
    with pytest.raises(StopConditionMet) as met:
        sim.run_until(50.0)
    assert d["stop_condition"] == met.value.condition == "C > 50"
    assert d["digest"] == _plain(met.value.result.digest())
    assert d["digest"]["t_end"] == float(met.value.result.time[-1]) < 50.0
    assert d["record"]["run"]["timeout"] is None  # run_until takes none: the command's own


def test_run_until_with_no_condition_met_is_the_whole_run():
    args = ("run-until", CHAIN, "--t-end", "8", "--n-points", "9", "--stop-when", "C > 1000")
    d = _json(_bngsim(*args, "--species", "--json"))
    sim = Simulator(Model.load(CHAIN))
    r = sim.run_until(8.0, n_points=9)
    assert d["stop_condition"] is None
    assert d["digest"] == _plain(r.digest(species=True))
    text = _bngsim(*args)
    assert text.returncode == 0 and "stopped" not in text.stdout


def test_run_until_conditions_are_checked_in_order_and_seeded():
    # Two conditions: the first one added that holds cuts the run, even where
    # a later one held earlier; and a stochastic run takes its seed.
    args = ("run-until", BIRTH_DEATH, "--method", "ssa", "--t-end", "20", "--seed", "5")
    d = _json(_bngsim(*args, "--stop-when", "A > 9000", "--stop-when", "A > 10", "--json"))
    sim = Simulator(Model.load(BIRTH_DEATH), method="ssa")
    sim.add_stop_condition("A > 9000")
    sim.add_stop_condition("A > 10")
    with pytest.raises(StopConditionMet) as met:
        sim.run_until(20.0, seed=5)
    assert d["stop_condition"] == met.value.condition == "A > 9000"
    assert d["digest"] == _plain(met.value.result.digest())
    assert d["record"]["run"]["seed"] == 5
    text = _bngsim(*args, "--stop-when", "A > 9000")
    assert text.returncode == 0 and "stopped: A > 9000 held at t=" in text.stdout


def test_run_until_sets_and_writes_the_cut_run(tmp_path):
    out = tmp_path / "cut.csv"
    d = _json(_bngsim("run-until", CHAIN, "--t-end", "30", "--set", "k1=2", "--set", "A=50",
                      "--stop-when", "C > 20", "--out", str(out), "--json"))  # fmt: skip
    m = Model.load(CHAIN)
    m.set_param("k1", 2.0)
    m.set_concentration("A", 50.0)
    m.save_concentrations()
    sim = Simulator(m)
    sim.add_stop_condition("C > 20")
    with pytest.raises(StopConditionMet) as met:
        sim.run_until(30.0)
    cut = met.value.result
    assert d["digest"] == _plain(cut.digest())
    assert d["set"] == {"k1": 2.0, "A": 50.0}
    back = np.loadtxt(out, delimiter=",", skiprows=1)
    assert back.shape == (cut.n_times, 1 + len(cut.observable_names))
    assert np.allclose(back[:, 1:], np.asarray(cut.observables), rtol=1e-11)


def test_a_condition_that_cannot_be_read_is_a_usage_error():
    error = _error(
        _bngsim("run-until", CHAIN, "--t-end", "5", "--stop-when", "nope > 1", "--json"),
        _cli.EXIT_USAGE,
        "usage.invalid_argument",
    )
    assert error["details"]["argument"] == "--stop-when"
    assert error["details"]["value"] == "nope > 1"


# ─── run-batch ───────────────────────────────────────────────────────────────


def _batch(model: str = CHAIN, rows=ROWS, **kw) -> Result:
    sim = Simulator(Model.load(model), method=kw.pop("method", "ode"))
    return Result.squeeze(list(sim.run_batch((0.0, 20.0), 21, params=rows, **kw)))


def test_run_batch_digests_the_rows_as_a_stack():
    d = _json(_bngsim("run-batch", CHAIN, "--t-end", "20", "--n-points", "21",
                      "--params", json.dumps(ROWS), "--json"))  # fmt: skip
    assert list(d) == ["model", "record", "digest", "out", "set"]
    r = _batch(timeout=600 / 3)
    assert d["digest"] == _plain(r.digest())
    assert d["digest"]["n_runs"] == 3
    # Each row may take its share of the limit, so all of them keep it.
    assert d["record"]["run"]["timeout"] == pytest.approx(200.0)
    assert "run.parameters.k1" in d["record"]["rows_differ"]


@pytest.mark.parametrize("form", ["json", "csv", "tsv"])
def test_run_batch_reads_the_rows_from_a_file(tmp_path, form):
    path = tmp_path / f"rows.{form}"
    if form == "json":
        path.write_text(json.dumps(ROWS))
        rows = ROWS
    else:
        sep = "," if form == "csv" else "\t"
        path.write_text(f"k1{sep}k2\n0.5{sep}0.1\n\n2{sep}0.3\n")
        rows = [{"k1": 0.5, "k2": 0.1}, {"k1": 2.0, "k2": 0.3}]
    d = _json(_bngsim("run-batch", CHAIN, "--t-end", "20", "--n-points", "21",
                      "--params", str(path), "--species", "--json"))  # fmt: skip
    r = _batch(rows=rows, timeout=600 / len(rows))
    assert d["digest"] == _plain(r.digest(species=True))


def test_run_batch_seeds_a_stochastic_stack_and_writes_it(tmp_path):
    out = tmp_path / "rows.npz"
    rows = [{"ksyn": 10.0}, {"ksyn": 20.0}]
    d = _json(_bngsim("run-batch", BIRTH_DEATH, "--method", "ssa", "--t-end", "20",
                      "--n-points", "21", "--params", json.dumps(rows), "--seed", "11",
                      "--out", str(out), "--json"))  # fmt: skip
    r = _batch(BIRTH_DEATH, rows, method="ssa", seed=11, timeout=300)
    assert d["digest"] == _plain(r.digest())
    assert d["seeds"][0] == 11 and len(set(d["seeds"])) == 2
    with np.load(out, allow_pickle=False) as f:
        assert np.array_equal(f["observables"], np.asarray(r.observables))


def test_a_batch_row_past_its_share_of_the_limit_exits_5():
    rows = json.dumps([{"ksyn": 1e6}] * 4)
    run = _bngsim("run-batch", BIRTH_DEATH, "--method", "ssa", "--t-end", "1000", "--params",
                  rows, "--timeout", "1", "--seed", "1", "--json")  # fmt: skip
    error = _error(run, _cli.EXIT_TIMEOUT, "run.timeout")
    assert error["details"]["timeout"] == pytest.approx(0.25)


# ─── parameter-scan ──────────────────────────────────────────────────────────


def test_parameter_scan_digests_the_values_as_a_stack():
    d = _json(_bngsim("parameter-scan", CHAIN, "--parameter", "k1", "--par-scan-vals",
                      "0.5,1,2,4", "--t-end", "20", "--n-points", "21", "--json"))  # fmt: skip
    sim = Simulator(Model.load(CHAIN))
    rows = sim.parameter_scan("k1", [0.5, 1, 2, 4], t_span=(0, 20), n_points=21, timeout=150)
    assert d["digest"] == _plain(Result.squeeze(rows).digest())
    assert d["parameter"] == "k1"
    assert d["values"] == [0.5, 1.0, 2.0, 4.0]
    assert d["record"]["run"]["timeout"] == pytest.approx(150.0)


def test_parameter_scan_takes_a_range_and_a_continuation(tmp_path):
    out = tmp_path / "scan.npz"
    d = _json(_bngsim("parameter-scan", CHAIN, "--parameter", "k2", "--par-min", "0.1",
                      "--par-max", "10", "--n-scan-pts", "5", "--log-scale", "--no-reset-conc",
                      "--t-end", "5", "--n-points", "6", "--out", str(out), "--json"))  # fmt: skip
    sim = Simulator(Model.load(CHAIN))
    rows = sim.parameter_scan(
        "k2", par_min=0.1, par_max=10, n_scan_pts=5, log_scale=True, reset_conc=False,
        t_span=(0, 5), n_points=6, timeout=120,
    )  # fmt: skip
    r = Result.squeeze(rows)
    assert d["digest"] == _plain(r.digest())
    assert d["values"] == pytest.approx([0.1, 10**-0.5, 1.0, 10**0.5, 10.0])
    with np.load(out, allow_pickle=False) as f:
        assert np.array_equal(f["observables"], np.asarray(r.observables))


def test_a_species_set_is_where_each_value_starts_and_a_dose_still_doses(tmp_path):
    # A0 seeds A; B is set. With B saved as the initial state, a scan of A0 would
    # start every value at the snapshot's A, dropping the dose. The command sets
    # B as the state the scan starts from, and A0 still doses A.
    model = tmp_path / "dose.ant"
    model.write_text(
        "compartment c = 1; species A in c, B in c; A0 = 1; A = A0; B = 0; k = 1; J: A -> B; k*A\n"
    )
    d = _json(_bngsim("parameter-scan", str(model), "--parameter", "A0", "--par-scan-vals",
                      "1,10", "--set", "B=5", "--t-end", "1", "--species", "--json"))  # fmt: skip
    m = Model.load(str(model))
    m.set_concentration("B", 5.0)
    rows = Simulator(m).parameter_scan("A0", [1, 10], t_span=(0, 1), timeout=300)
    assert d["digest"] == _plain(Result.squeeze(rows).digest(species=True))
    first = {c["name"]: c for c in d["digest"]["species"]["columns"]}
    assert first["A"]["max"] == pytest.approx(10.0)  # the dose reached A


# ─── steady-state ────────────────────────────────────────────────────────────


def test_steady_state_prints_the_solvers_verdict_and_the_values():
    d = _json(_bngsim("steady-state", CHAIN, "--species", "--json"))
    assert list(d) == ["model", "call", "parameters", "result", "out", "set"]
    ss = Simulator(Model.load(CHAIN)).steady_state()
    f = d["result"]
    assert f["converged"] is ss.converged is True
    assert f["residual"] == ss.residual and f["method_used"] == ss.method_used
    assert f["criterion"] == ss.criterion == "scaled"
    values = {c["name"]: c["value"] for c in f["species"]["columns"]}
    assert values == dict(zip(ss.species_names, ss.concentrations.tolist(), strict=True))
    obs = {c["name"]: c["value"] for c in f["observables"]["columns"]}
    assert obs == dict(zip(ss.observable_names, ss._observable_values.tolist(), strict=True))
    assert d["call"] == {"name": "steady_state", "kwargs": {}}
    # The parameters a run's record lists, at their values.
    run = Simulator(Model.load(CHAIN)).run(t_span=(0, 1), n_points=2)
    assert d["parameters"] == run.record["run"]["parameters"]
    assert d["parameters"]["k1"] == 1.0 and d["parameters"]["k2"] == 0.5


def test_steady_state_passes_the_solver_options_and_writes_the_arrays(tmp_path):
    out = tmp_path / "ss.npz"
    args = ["steady-state", SCHLOGL, "--method", "newton", "--criterion", "residual", "--tol",
            "1e-10", "--rtol", "1e-9", "--atol", "1e-9", "--max-time", "1e5", "--max-steps",
            "50000", "--set", "k1=0.2", "--out", str(out), "--json"]  # fmt: skip
    d = _json(_bngsim(*args))
    m = Model.load(SCHLOGL)
    m.set_param("k1", 0.2)
    kw = {"tol": 1e-10, "max_time": 1e5, "criterion": "residual", "rtol": 1e-9,
          "atol": 1e-9, "max_steps": 50000, "method": "newton"}  # fmt: skip
    ss = Simulator(m).steady_state(**kw)
    assert d["call"]["kwargs"] == kw
    assert d["result"] == _plain(_steady_form.steady_state(ss))
    assert d["result"]["method_used"] == ss.method_used
    assert d["result"]["root_stability"] == ss.root_stability
    assert d["parameters"]["k1"] == 0.2
    with np.load(out, allow_pickle=False) as f:
        assert np.array_equal(f["concentrations"], ss.concentrations)
        assert list(f["species_names"]) == ss.species_names
        assert np.array_equal(f["eigenvalues"], ss.eigenvalues)
        assert bool(f["converged"]) is ss.converged


def test_a_steady_state_lists_its_largest_columns_first(tmp_path):
    # Five species and room for two: the two largest are listed, in model order.
    model = tmp_path / "five.ant"
    model.write_text(
        "compartment c = 1; species A in c, B in c, C in c, D in c, E in c;"
        " A = 1; B = 50; C = 3; D = 400; E = 2; k = 0; J: A -> B; k*A\n"
    )
    d = _json(_bngsim("steady-state", str(model), "--species", "--max-columns", "2", "--json"))
    block = d["result"]["species"]
    assert [c["name"] for c in block["columns"]] == ["B", "D"]
    assert block["n_columns"] == 5 and block["n_left_out"] == 3


def test_a_steady_state_not_converged_exits_0_and_says_so():
    # Growth with no bound has no steady state: the call returns, not converged.
    run = _bngsim("steady-state", BIRTH_DEATH, "--set", "kdeg=0", "--max-time", "10")
    assert run.returncode == 0, run.stderr
    assert "NOT CONVERGED" in run.stdout
    d = _json(_bngsim("steady-state", BIRTH_DEATH, "--set", "kdeg=0", "--max-time", "10",
                      "--json"))  # fmt: skip
    m = Model.load(BIRTH_DEATH)
    m.set_param("kdeg", 0.0)
    ss = Simulator(m).steady_state(max_time=10.0)
    assert d["result"]["converged"] is ss.converged is False
    assert d["result"]["continued_run_distance"] == _plain(ss.continued_run_distance)


# ─── steady-state-batch ──────────────────────────────────────────────────────


def test_steady_state_batch_gives_statistics_of_the_rows_that_converged(tmp_path):
    out = tmp_path / "rows.npz"
    rows = [{"k1": 0.1}, {"k1": 0.2}, {"k1": 0.3}]
    d = _json(_bngsim("steady-state-batch", SCHLOGL, "--params", json.dumps(rows),
                      "--species", "--out", str(out), "--json"))  # fmt: skip
    found = Simulator(Model.load(SCHLOGL)).steady_state_batch(rows)
    f = d["result"]
    assert f["n_rows"] == 3 and f["n_converged"] == sum(r.converged for r in found)
    by_name = {c["name"]: c for c in f["species"]["columns"]}
    kept = np.array([r.concentrations for r in found if r.converged])
    for j, name in enumerate(found[0].species_names):
        assert by_name[name]["mean"] == pytest.approx(kept[:, j].mean(), rel=1e-15)
        assert by_name[name]["min"] == kept[:, j].min()
        assert by_name[name]["max"] == kept[:, j].max()
    with np.load(out, allow_pickle=False) as a:
        assert np.array_equal(a["concentrations"], np.array([r.concentrations for r in found]))
        assert list(a["converged"]) == [r.converged for r in found]


def test_a_row_that_does_not_converge_is_named_and_left_out_of_the_statistics():
    rows = [{"kdeg": 0.5}, {"kdeg": 0.0}, {"kdeg": 2.0}]
    d = _json(_bngsim("steady-state-batch", BIRTH_DEATH, "--params", json.dumps(rows),
                      "--max-time", "200", "--json"))  # fmt: skip
    m = Model.load(BIRTH_DEATH)
    found = Simulator(m).steady_state_batch(rows, max_time=200.0)
    f = d["result"]
    assert [r.converged for r in found] == [True, False, True]
    assert f["unconverged_rows"] == [1] and f["n_unconverged"] == 1
    (col,) = f["observables"]["columns"]
    assert col["n_finite"] == 2
    assert col["mean"] == pytest.approx(np.mean([found[0]._observable_values[0],
                                                 found[2]._observable_values[0]]))  # fmt: skip


# ─── steady-states ───────────────────────────────────────────────────────────


def test_steady_states_lists_each_root_and_the_species_that_tell_them_apart(tmp_path):
    out = tmp_path / "roots.npz"
    d = _json(_bngsim("steady-states", SCHLOGL, "--n-starts", "30", "--seed", "3",
                      "--out", str(out), "--json"))  # fmt: skip
    found = Simulator(Model.load(SCHLOGL)).steady_states(n_starts=30, seed=3)
    f = d["result"]
    assert f["n_roots"] == len(found) >= 2  # Schlögl's model is bistable
    assert [r["stability"] for r in f["roots"]] == found.stability
    assert [r["multiplicity"] for r in f["roots"]] == found.multiplicity.tolist()
    by_name = {c["name"]: c["values"] for c in f["species"]["columns"]}
    for j, name in enumerate(found.species_names):
        if name in by_name:
            assert by_name[name] == found.states[:, j].tolist()
    assert d["call"]["kwargs"] == {"n_starts": 30, "seed": 3}
    with np.load(out, allow_pickle=False) as a:
        assert np.array_equal(a["states"], found.states)
        assert np.array_equal(a["basins"], found.basins)


# ─── run-bngl ────────────────────────────────────────────────────────────────

_BNGL = """begin model
begin parameters
  k1 1
  k2 0.5
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() 100
  B() 0
end seed species
begin observables
  Molecules A A()
  Molecules B B()
end observables
begin reaction rules
  A() -> B() k1
  B() -> 0 k2
end reaction rules
end model

generate_network({overwrite=>1})
simulate({method=>"ode",t_end=>10,n_steps=>10,suffix=>"first"})
parameter_scan({parameter=>"k1",par_min=>0.1,par_max=>1,n_scan_pts=>3,method=>"ode",\
t_end=>5,n_steps=>5,suffix=>"scan"})
setParameter("k2", 0.25)
simulate({method=>"ode",t_end=>4,n_steps=>4})
"""


def _bngl_runs(ran) -> list[tuple]:
    """What the command prints of each run of a ProtocolResult, made here."""
    out = []
    for run in ran:
        if run.result is not None:
            out.append((run.kind, run.label, _plain(run.result.digest()), []))
        else:
            stack = Result.squeeze(list(run.scan_results))
            out.append((run.kind, run.label, _plain(stack.digest()), list(run.scan_values)))
    return out


_HAS_BNG2 = bngsim.HAS_BNGL and shutil.which("perl") is not None


@pytest.mark.skipif(not _HAS_BNG2, reason="needs BNG2.pl on BNGPATH")
def test_run_bngl_digests_each_run_the_file_makes(tmp_path):
    path = tmp_path / "m.bngl"
    path.write_text(_BNGL)
    d = _json(_bngsim("run-bngl", str(path), "--out", str(tmp_path / "arrays"), "--json"))
    ran = bngsim.run_bngl(path)
    got = [(r["kind"], r["label"], r["digest"], r.get("values", [])) for r in d["runs"]]
    assert got == _bngl_runs(ran)
    assert sorted(p.name for p in (tmp_path / "arrays").iterdir()) == [
        "0-first.npz", "1-scan.npz", "2.npz"
    ]  # fmt: skip
    with np.load(tmp_path / "arrays" / "2.npz", allow_pickle=False) as back:
        assert np.array_equal(back["observables"], np.asarray(ran[2].result.observables))


# Model.from_bngl, in place of BNG2.pl: the .net the file generates, and the
# actions read from the file (which needs no BNG2.pl). What the command does with
# them is what it does with BNG2.pl's.
_NO_BNG2 = """
import bngsim
from pathlib import Path
from bngsim.convert import parse_bngl_protocol

NET = {net!r}

def from_bngl(path, protocol=False, **kw):
    model = bngsim.Model.load(NET)
    return (model, parse_bngl_protocol(Path(path), strict=False)) if protocol else model

bngsim.Model.from_bngl = staticmethod(from_bngl)
"""

_CHAIN_BNGL_ACTIONS = """
simulate({method=>"ode",t_end=>6,n_steps=>6,suffix=>"a"})
parameter_scan({parameter=>"k2",par_scan_vals=>[0.1,1],method=>"ode",t_end=>3,n_steps=>3})
simulate({method=>"ssa",t_end=>2,n_steps=>2,suffix=>"noisy"})
"""


def test_run_bngl_runs_the_actions_as_run_protocol_does(tmp_path):
    bngl = tmp_path / "chain.bngl"
    bngl.write_text(Path(examples.path("chain", "bngl")).read_text() + _CHAIN_BNGL_ACTIONS)
    net = str(examples.path("chain", "net"))
    run = _bngsim("run-bngl", str(bngl), "--seed", "4", "--rtol", "1e-9", "--json",
                  script=_NO_BNG2.format(net=net))  # fmt: skip
    d = _json(run)
    from bngsim.convert import parse_bngl_protocol

    ran = bngsim.run_protocol(
        Model.load(net), parse_bngl_protocol(bngl, strict=False), seed=4, rtol=1e-9
    )
    got = [(r["kind"], r["label"], r["digest"], r.get("values", [])) for r in d["runs"]]
    assert got == _bngl_runs(ran)
    assert [r["method"] for r in d["runs"]] == ["ode", "ode", "ssa"]
    assert d["out"] is None


def test_run_bngl_with_no_runs_says_so(tmp_path):
    bngl = tmp_path / "chain.bngl"
    bngl.write_text(Path(examples.path("chain", "bngl")).read_text())
    net = str(examples.path("chain", "net"))
    run = _bngsim("run-bngl", str(bngl), script=_NO_BNG2.format(net=net))
    assert run.returncode == 0, run.stderr
    assert "make no run" in run.stdout


# ─── The wall-clock limit of a call that takes none ──────────────────────────


@pytest.mark.parametrize("command", ["steady-states", "steady-state-batch"])
def test_a_call_past_the_commands_limit_exits_5_and_stops(command, tmp_path):
    if command == "steady-states":
        args = [command, CHAIN, "--n-starts", "1000000"]
    else:
        rows = tmp_path / "rows.csv"
        rows.write_text("k1\n" + "1\n" * 200_000)
        args = [command, CHAIN, "--params", str(rows)]
    start = time.monotonic()
    run = _bngsim(*args, "--timeout", "0.5", "--json")
    elapsed = time.monotonic() - start
    error = _error(run, _cli.EXIT_TIMEOUT, "run.timeout")
    assert error["details"]["timeout"] == 0.5
    assert error["details"]["elapsed"] >= 0.5
    assert "--timeout" in error["message"]
    assert elapsed < 60  # the process ended, the call still running
    text = _bngsim(*args, "--timeout", "0.5")
    assert text.returncode == _cli.EXIT_TIMEOUT and text.stdout == ""
    assert "bngsim: error [run.timeout]" in text.stderr


def test_what_stderr_held_goes_into_the_timeout_error(tmp_path):
    # B has no initial value: the load warns. A call that then runs out of time
    # reports one JSON object, the warning inside it as `log`.
    model = tmp_path / "w.ant"
    model.write_text("compartment c = 1; species A in c, B in c; A = 1; k = 1; J: A -> B; k*A\n")
    run = _bngsim("steady-states", str(model), "--n-starts", "1000000", "--timeout", "0.5",
                  "--json")  # fmt: skip
    error = _error(run, _cli.EXIT_TIMEOUT, "run.timeout")
    assert "'B'" in error["log"]


def test_a_call_inside_its_limit_is_not_stopped():
    # A limit far above the call: the watchdog is cancelled when it returns.
    d = _json(_bngsim("steady-state", CHAIN, "--timeout", "2", "--json"))
    assert d["result"]["converged"] is True


def test_run_bngl_is_bounded_too(tmp_path):
    bngl = tmp_path / "long.bngl"
    bngl.write_text(
        Path(examples.path("birth_death", "bngl")).read_text()
        + 'simulate({method=>"ssa",t_end=>1e6,n_steps=>10})\n'
    )
    net = str(examples.path("birth_death", "net"))
    run = _bngsim("run-bngl", str(bngl), "--timeout", "0.5", "--json",
                  script=_NO_BNG2.format(net=net))  # fmt: skip
    _error(run, _cli.EXIT_TIMEOUT, "run.timeout")


# ─── An option the call would not use is refused ─────────────────────────────

_P = json.dumps([{"k1": 1.0}])


@pytest.mark.parametrize(
    "args, argument",
    [
        (["run-until", CHAIN, "--t-end", "5", "--t-start", "1"], "--t-start"),
        (["run-until", CHAIN, "--t-end", "5", "--sample-times", "0,1"], "--sample-times"),
        (["run-until", CHAIN, "--t-end", "5", "--steady-state"], "--steady-state"),
        (["run-until", CHAIN, "--t-end", "5", "--xml", "m.xml"], "--xml"),
        (["run-until", CHAIN, "--stop-when", "C > 1"], "--t-end"),
        (["run-until", CHAIN, "--t-end", "5", "--stop-when", " "], "--stop-when"),
        (["run-until", CHAIN, "--t-end", "5", "--seed", "1"], "--seed"),
        (["run-until", CHAIN, "--t-end", "5", "--method", "nf"], "--method"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", _P, "--method", "rm"], "--method"),
        (["parameter-scan", CHAIN, "--t-end", "5", "--parameter", "k1", "--par-scan-vals", "1",
          "--method", "nfsim"], "--method"),
        (["run-batch", CHAIN, "--t-end", "5"], "--params"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", _P, "--sample-times", "0,1"],
         "--sample-times"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", _P, "--steady-state"], "--steady-state"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", _P, "--replicates", "2"],
         "--replicates"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", _P, "--out", "x.csv"], "--out"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", _P, "--out-kind", "species"],
         "--out-kind"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", "[]"], "--params"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", "[{\"k1\": \"x\"}]"], "--params"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", "[{\"k1\": true}]"], "--params"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", "[{\"k1\": NaN}]"], "--params"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", "[1, 2]"], "--params"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", "{\"k1\": "], "--params"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", "rows.txt"], "--params"),
        (["run-batch", CHAIN, "--t-end", "5", "--params", "missing.csv"], "--params"),
        (["parameter-scan", CHAIN, "--t-end", "5", "--par-scan-vals", "1,2"], "--parameter"),
        (["parameter-scan", CHAIN, "--t-end", "5", "--parameter", "k1", "--par-scan-vals", "1",
          "--steady-state"], "--steady-state"),
        (["parameter-scan", CHAIN, "--t-end", "5", "--parameter", "k1", "--par-scan-vals", "1",
          "--sample-times", "0,1"], "--sample-times"),
        (["steady-state", CHAIN, "--seed", "1"], "--seed"),
        (["steady-state", CHAIN, "--t-end", "5"], "--t-end"),
        (["steady-state", CHAIN, "--out", "x.h5"], "--out"),
        (["steady-state", CHAIN, "--out", "x.csv"], "--out"),
        (["steady-state", CHAIN, "--n-workers", "2"], "--n-workers"),
        (["steady-state-batch", CHAIN], "--params"),
        (["steady-states", CHAIN, "--species"], "--species"),
        (["steady-states", CHAIN, "--rtol", "1e-6"], "--rtol"),
        (["steady-state", "model.bngl"], "--network"),
        (["run-batch", "model.bngl", "--t-end", "1", "--params", _P], "--network"),
        (["steady-state", CHAIN, "--network"], "--network"),
        (["run-bngl", CHAIN], "model"),
        (["run-bngl", "m.bngl", "--atol", "auto"], "--atol"),
        (["run-bngl", "m.bngl", "--set", "k1=2"], "--set"),
        (["run-bngl", "m.bngl", "--network"], "--network"),
    ],
)  # fmt: skip
def test_a_wrong_command_line_exits_2(args, argument, tmp_path):
    (tmp_path / "rows.txt").write_text("k1\n1\n")
    run = _bngsim(*args, "--json", cwd=tmp_path)
    error = _error(run, _cli.EXIT_USAGE, "usage.invalid_argument")
    assert error["details"].get("argument") == argument, error


def test_a_refused_option_names_the_command_and_its_usage():
    run = _bngsim("run-batch", CHAIN, "--t-end", "5", "--params", _P, "--sample-times", "0,1",
                  "--json")  # fmt: skip
    error = _error(run, _cli.EXIT_USAGE, "usage.invalid_argument")
    assert error["message"].startswith("`bngsim run-batch` takes no --sample-times;")
    assert error["details"]["usage"].startswith("usage: bngsim run-batch")


def test_the_python_name_of_a_command_names_the_command():
    error = _error(_bngsim("steady_state_batch", CHAIN, "--json"), _cli.EXIT_USAGE,
                   "usage.invalid_argument")  # fmt: skip
    assert "`bngsim steady-state-batch`" in error["message"]


def test_a_params_file_with_a_short_row_is_refused(tmp_path):
    rows = tmp_path / "rows.csv"
    rows.write_text("k1,k2\n1,2\n3\n")
    error = _error(_bngsim("run-batch", CHAIN, "--t-end", "5", "--params", str(rows), "--json"),
                   _cli.EXIT_USAGE, "usage.invalid_argument")  # fmt: skip
    assert "line 3" in error["message"]


@pytest.mark.parametrize(
    "args, argument",
    [
        (["run-batch", CHAIN, "--t-end", "5", "--params", '[{"k11": 1}]'], "--params"),
        (["steady-state-batch", CHAIN, "--params", '[{"k11": 1}]'], "--params"),
        (["parameter-scan", CHAIN, "--t-end", "5", "--parameter", "k11", "--par-scan-vals", "1"],
         "--parameter"),
    ],
)  # fmt: skip
def test_a_name_the_model_lacks_is_a_usage_error_as_in_set(args, argument):
    # The call's own refusal, with its close matches, as --set's is.
    error = _error(_bngsim(*args, "--json"), _cli.EXIT_USAGE, "usage.invalid_argument")
    assert error["details"]["argument"] == argument
    assert "k11" in error["message"] and "k1" in json.dumps(error["details"])


def test_an_out_directory_for_run_bngl_that_is_a_file_is_refused(tmp_path):
    (tmp_path / "taken").write_text("")
    error = _error(_bngsim("run-bngl", "m.bngl", "--out", str(tmp_path / "taken"), "--json"),
                   _cli.EXIT_USAGE, "usage.invalid_argument")  # fmt: skip
    assert error["details"]["argument"] == "--out"


# ─── The commands are listed, and documented ─────────────────────────────────


def test_help_lists_each_command():
    run = _bngsim("--help")
    listed = [line.split()[0] for line in run.stdout.splitlines() if line.startswith("    ")]
    for command in _cli._RUN_CALLS:
        assert command in listed, command


def test_each_command_is_in_the_documentation():
    page = (Path(__file__).parents[2] / "docs" / "reference" / "command-line.md").read_text(
        encoding="utf-8"
    )
    for command in _cli._ALL_COMMANDS:
        assert f"| `bngsim {command}" in page, command
    assert "Not commands yet" not in page
