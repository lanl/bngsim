"""``bngsim run`` and ``bngsim check`` (issue #1708, part 2).

Each command is a thin layer over a Python call, so each test holds the
command's JSON to what the call itself returns: the digest and record of
``Simulator(Model.load(MODEL)).run(...)``, and ``Result.check()``. The rest is
the command's own contract: arrays never on the terminal, a wall-clock limit
on every run, and an exit code for each outcome.
"""

from __future__ import annotations

import json
import subprocess
import sys

import bngsim
import numpy as np
import pytest
from bngsim import Model, Simulator, _cli, examples

CHAIN = str(examples.path("chain", "sbml"))
BIRTH_DEATH = str(examples.path("birth_death", "net"))


def _bngsim(*args: str, script: str | None = None) -> subprocess.CompletedProcess:
    if script is None:
        cmd = [sys.executable, "-m", "bngsim", *args]
    else:
        code = script + f"\nfrom bngsim._cli import main\nraise SystemExit(main({list(args)!r}))\n"
        cmd = [sys.executable, "-c", code]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)


def _json(run: subprocess.CompletedProcess, exit_code: int = 0) -> dict:
    assert run.returncode == exit_code, (run.returncode, run.stderr)
    return json.loads(run.stdout)


def _error(run: subprocess.CompletedProcess, exit_code: int, code: str) -> dict:
    assert run.returncode == exit_code, (run.returncode, run.stderr)
    assert run.stdout == ""
    error = json.loads(run.stderr)
    assert error["code"] == code, error
    return error


def _plain(value):
    return json.loads(json.dumps(value))


# ─── run ─────────────────────────────────────────────────────────────────────


def test_run_prints_the_record_and_the_digest_of_the_python_call():
    d = _json(_bngsim("run", CHAIN, "--t-end", "20", "--n-points", "21", "--json"))
    assert list(d) == ["model", "record", "digest", "out"]
    r = Simulator(Model.load(CHAIN), method="ode").run(t_span=(0, 20), n_points=21, timeout=600)
    assert d["digest"] == _plain(r.digest())
    assert d["record"]["run"]["method"] == "ode"
    assert d["record"]["run"]["started_from"] == "initial"
    assert d["record"]["run"]["timeout"] == 600.0  # every run has a limit
    assert "start_state" not in d["record"]["run"]
    assert d["out"] is None


def test_run_without_json_prints_the_digest_text_and_no_arrays():
    run = _bngsim("run", CHAIN, "--t-end", "20", "--n-points", "21", "--species")
    assert run.returncode == 0, run.stderr
    r = Simulator(Model.load(CHAIN), method="ode").run(t_span=(0, 20), n_points=21)
    assert run.stdout == f"chain.sbml: {r.digest(species=True)}\n"


@pytest.mark.parametrize("suffix", [".npz", ".csv", ".tsv"])
def test_run_writes_the_arrays_to_out(tmp_path, suffix):
    out = tmp_path / f"run{suffix}"
    args = ["run", CHAIN, "--t-end", "5", "--n-points", "6", "--out", str(out), "--json"]
    if suffix == ".tsv":
        args += ["--out-kind", "species"]
    d = _json(_bngsim(*args))
    assert d["out"] == str(out.resolve())
    r = Simulator(Model.load(CHAIN), method="ode").run(t_span=(0, 5), n_points=6)
    if suffix == ".npz":
        with np.load(out, allow_pickle=False) as f:
            assert np.array_equal(f["observables"], np.asarray(r.observables))
            assert np.array_equal(f["time"], r.time)
    else:
        sep = "," if suffix == ".csv" else "\t"
        back = np.loadtxt(out, delimiter=sep, skiprows=1)
        block = r.observables if suffix == ".csv" else r.species
        assert np.allclose(back[:, 1:], np.asarray(block), rtol=1e-11)


def test_run_writes_hdf5_where_h5py_is_installed(tmp_path):
    pytest.importorskip("h5py")
    out = tmp_path / "run.h5"
    _json(_bngsim("run", CHAIN, "--t-end", "5", "--n-points", "6", "--out", str(out), "--json"))
    back = bngsim.Result.load(out)
    assert back.n_times == 6 and back.record["run"]["method"] == "ode"


def test_run_takes_sample_times_and_set():
    d = _json(
        _bngsim(
            "run", CHAIN, "--sample-times", "0,1,3,7", "--set", "k1=2", "--set", "A=50", "--json"
        )
    )
    m = Model.load(CHAIN)
    m.set_param("k1", 2.0)
    m.set_concentration("A", 50.0)
    r = Simulator(m, method="ode").run(sample_times=[0, 1, 3, 7])
    assert d["digest"] == _plain(r.digest())
    assert d["digest"]["n_points"] == 4
    assert d["record"]["run"]["parameters"]["k1"] == 2.0


def test_run_seeds_a_stochastic_run():
    args = ("run", BIRTH_DEATH, "--method", "ssa", "--t-end", "5", "--n-points", "6")
    one = _json(_bngsim(*args, "--seed", "7", "--json"))
    two = _json(_bngsim(*args, "--seed", "7", "--json"))
    assert one["digest"] == two["digest"] and one["record"]["run"]["seed"] == 7
    assert one["digest"]["stochastic"] is True


def test_run_replicates_digest_a_stack(tmp_path):
    out = tmp_path / "reps.npz"
    d = _json(
        _bngsim(
            "run", BIRTH_DEATH, "--method", "ssa", "--t-end", "10", "--n-points", "11",
            "--replicates", "6", "--seed", "3", "--out", str(out), "--json",
        )
    )  # fmt: skip
    assert d["digest"]["n_runs"] == 6
    (row,) = d["digest"]["observables"]["columns"]
    with np.load(out, allow_pickle=False) as f:
        last = f["observables"][:, -1, 0]
    assert row["last_mean"] == pytest.approx(last.mean())
    assert row["last_std"] == pytest.approx(last.std(ddof=1))


def test_a_steady_state_run():
    d = _json(_bngsim("run", CHAIN, "--t-end", "2000", "--steady-state", "--json"))
    assert d["record"]["run"]["steady_state"] is True


# ─── Errors and exit codes ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "args, argument",
    [
        (["run", CHAIN], "--t-end"),
        (["run", CHAIN, "--t-end", "1", "--set", "k1"], "--set"),
        (["run", CHAIN, "--t-end", "1", "--set", "k1=x"], "--set"),
        (["run", CHAIN, "--t-end", "1", "--set", "nope=2"], "--set"),
        (["run", BIRTH_DEATH, "--t-end", "1", "--replicates", "3", "--rtol", "1e-6"], "--rtol"),
        (
            [
                "run",
                BIRTH_DEATH,
                "--method",
                "ssa",
                "--t-end",
                "1",
                "--replicates",
                "2",
                "--out",
                "x.csv",
            ],
            "--out",
        ),
        (["check", BIRTH_DEATH, "--t-end", "1", "--replicates", "2"], "--replicates"),
        (["run", "model.bngl", "--t-end", "1"], "--network"),
    ],
)
def test_a_wrong_run_command_line_exits_2(args, argument, tmp_path):
    run = subprocess.run(
        [sys.executable, "-m", "bngsim", *args, "--json"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )
    error = _error(run, _cli.EXIT_USAGE, "usage.invalid_argument")
    assert error["details"]["argument"] == argument, error


@pytest.mark.parametrize(
    "args",
    [
        ["--timeout", "0"],
        ["--timeout", "x"],
        ["--replicates", "0"],
        ["--atol", "loose"],
        ["--sample-times", "1,b"],
    ],
)
def test_bad_option_values_exit_2(args):
    run = _bngsim("run", CHAIN, "--t-end", "1", *args, "--json")
    _error(run, _cli.EXIT_USAGE, "usage.invalid_argument")


def test_an_unknown_name_in_set_names_the_close_ones():
    error = _error(
        _bngsim("run", CHAIN, "--t-end", "1", "--set", "k11=2", "--json"),
        _cli.EXIT_USAGE,
        "usage.invalid_argument",
    )
    assert "k1" in error["details"].get("close_matches", [])


_RAISE = """
import bngsim
def run(self, *a, **k):
    raise {error}
bngsim.Simulator.run = run
"""


def test_a_timed_out_run_exits_5():
    script = _RAISE.format(
        error='bngsim.SimulationTimeout("too long", timeout=600.0, elapsed=601.0)'
    )
    run = _bngsim("run", CHAIN, "--t-end", "1", "--json", script=script)
    _error(run, _cli.EXIT_TIMEOUT, "run.timeout")


def test_a_failed_run_exits_4():
    script = _RAISE.format(error='bngsim.SimulationError("CVODE failed with flag=-4")')
    run = _bngsim("run", CHAIN, "--t-end", "1", "--json", script=script)
    assert run.returncode == _cli.EXIT_RUN_FAILED, run.stderr
    assert run.stdout == ""


def test_the_timeout_option_reaches_the_run():
    d = _json(_bngsim("run", CHAIN, "--t-end", "1", "--timeout", "12.5", "--json"))
    assert d["record"]["run"]["timeout"] == 12.5


# ─── check ───────────────────────────────────────────────────────────────────


def test_check_prints_the_report_and_exits_0_when_ok():
    d = _json(_bngsim("check", CHAIN, "--t-end", "20", "--n-points", "21", "--json"))
    assert list(d) == ["model", "record", "digest", "out", "check"]
    r = Simulator(Model.load(CHAIN), method="ode").run(t_span=(0, 20), n_points=21)
    rep = r.check()
    assert d["check"]["ok"] is True
    assert [c["name"] for c in d["check"]["checks"]] == [c["name"] for c in rep["checks"]]
    assert [c["status"] for c in d["check"]["checks"]] == [c["status"] for c in rep["checks"]]


def test_check_that_is_not_ok_exits_6_with_the_report_on_stdout():
    run = _bngsim(
        "check", CHAIN, "--t-end", "20", "--n-points", "21", "--rtol", "1e-2", "--atol", "1e-2",
        "--json",
    )  # fmt: skip
    d = _json(run, _cli.EXIT_CHECK_FAILED)
    assert d["check"]["ok"] is False
    assert d["check"]["checks"][0]["status"] == "diff"
    text = _bngsim("check", CHAIN, "--t-end", "20", "--rtol", "1e-2", "--atol", "1e-2")
    assert text.returncode == _cli.EXIT_CHECK_FAILED
    assert "NOT OK" in text.stdout


def test_check_takes_named_checks_and_a_budget():
    d = _json(
        _bngsim(
            "check", CHAIN, "--t-end", "5", "--n-points", "6", "--checks", "jacobian, grid",
            "--budget", "300", "--json",
        )
    )  # fmt: skip
    assert [c["name"] for c in d["check"]["checks"]] == ["grid", "jacobian"]
    error = _error(
        _bngsim("check", CHAIN, "--t-end", "5", "--checks", "tighter", "--json"),
        _cli.EXIT_USAGE,
        "usage.invalid_argument",
    )
    assert "tighter" in error["message"]


def test_check_all_on_a_stochastic_run_reports_replicates():
    d = _json(
        _bngsim(
            "check", BIRTH_DEATH, "--method", "ssa", "--t-end", "5", "--n-points", "6",
            "--seed", "2", "--checks", "all", "--json",
        )
    )  # fmt: skip
    assert [c["name"] for c in d["check"]["checks"]] == ["replicates"]


def test_help_lists_run_and_check_and_exit_6():
    run = _bngsim("--help")
    assert "run" in run.stdout and "check" in run.stdout
    assert "6 check not ok" in run.stdout
