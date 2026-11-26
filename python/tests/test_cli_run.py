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
    assert list(d) == ["model", "record", "digest", "out", "set"]
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


@pytest.mark.parametrize("suffix", [".npz", ".csv", ".tsv", ".NPZ"])
def test_run_writes_the_arrays_to_out(tmp_path, suffix):
    out = tmp_path / f"run{suffix}"
    args = ["run", BIRTH_DEATH, "--t-end", "5", "--n-points", "6", "--out", str(out), "--json"]
    if suffix == ".tsv":
        args += ["--out-kind", "species"]
    d = _json(_bngsim(*args))
    assert d["out"] == str(out.resolve())
    r = Simulator(Model.load(BIRTH_DEATH), method="ode").run(t_span=(0, 5), n_points=6)
    if suffix.lower() == ".npz":
        with np.load(out, allow_pickle=False) as f:
            assert np.array_equal(f["observables"], np.asarray(r.observables))
            assert np.array_equal(f["time"], r.time)
        return
    sep = "," if suffix == ".csv" else "\t"
    header = out.read_text().splitlines()[0].split(sep)
    names = r.observable_names if suffix == ".csv" else r.species_names
    assert header == ["time", *names]  # A for observables, A() for species
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
    m.save_concentrations()  # the command's set species are the initial state
    r = Simulator(m, method="ode").run(sample_times=[0, 1, 3, 7])
    assert d["digest"] == _plain(r.digest())
    assert d["digest"]["n_points"] == 4
    assert d["record"]["run"]["parameters"]["k1"] == 2.0
    assert d["record"]["run"]["started_from"] == "initial"
    assert d["set"] == {"k1": 2.0, "A": 50.0}


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
    assert len(d["seeds"]) == 6 and d["seeds"][0] == 3
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
        (["run", CHAIN, "--t-end", "1", "--set", "A=nan"], "--set"),
        (["run", CHAIN, "--t-end", "1", "--set", "nope=2"], "--set"),
        (["run", BIRTH_DEATH, "--method", "ssa", "--t-end", "1", "--replicates", "3",
          "--steady-state"], "--steady-state"),
        (["run", BIRTH_DEATH, "--t-end", "1", "--replicates", "3"], "--replicates"),
        (["run", BIRTH_DEATH, "--method", "ssa", "--t-end", "1", "--replicates", "2",
          "--out", "x.csv"], "--out"),
        (["check", BIRTH_DEATH, "--t-end", "1", "--replicates", "2"], "--replicates"),
        (["run", "model.bngl", "--t-end", "1"], "--network"),
        (["run", CHAIN, "--t-end", "1", "--network"], "--network"),
        (["run", BIRTH_DEATH, "--method", "ssa", "--t-end", "1", "--rtol", "1e-9"], "--rtol"),
        (["run", BIRTH_DEATH, "--method", "ssa", "--t-end", "1", "--atol", "1e-3"], "--atol"),
        (["run", CHAIN, "--t-end", "1", "--xml", "m.xml"], "--xml"),
        (["run", CHAIN, "--t-end", "1", "--seed", "3"], "--seed"),
        (["run", CHAIN, "--t-end", "1", "--poplevel", "10"], "--poplevel"),
        (["run", CHAIN, "--method", "psa", "--t-end", "1"], "--poplevel"),
        (["run", CHAIN, "--sample-times", "0,1", "--t-end", "5"], "--t-end"),
        (["run", CHAIN, "--sample-times", "0,1", "--n-points", "5"], "--n-points"),
        (["run", CHAIN, "--sample-times", "0,1", "--t-start", "0"], "--t-start"),
        (["run", CHAIN, "--sample-times", "3,1,2"], "--sample-times"),
        (["run", CHAIN, "--t-end", "1", "--out", "x.txt"], "--out"),
        (["run", CHAIN, "--t-end", "1", "--out", "noext"], "--out"),
        (["run", CHAIN, "--t-end", "1", "--out", "nodir/x.npz"], "--out"),
        (["run", CHAIN, "--t-end", "1", "--out", "."], "--out"),
        (["run", CHAIN, "--t-end", "1", "--out", "x.npz", "--out-kind", "species"], "--out-kind"),
    ],
)  # fmt: skip
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
        ["--rtol", "nan"],
        ["--rtol", "-1"],
        ["--atol", "-1"],
        ["--atol", "nan"],
        ["--poplevel", "0"],
    ],
)
def test_bad_option_values_exit_2(args):
    run = _bngsim("run", CHAIN, "--t-end", "1", *args, "--json")
    error = _error(run, _cli.EXIT_USAGE, "usage.invalid_argument")
    assert error["details"]["argument"] == args[0]
    assert ("seconds" in error["message"]) is (args[0] == "--timeout")


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
    assert list(d) == ["model", "record", "digest", "out", "set", "check"]
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


def test_check_all_on_a_stochastic_run_reports_replicates_and_is_not_ok():
    # Its one check reports and judges nothing, so the run is not checked: the
    # report is not ok, and the command exits 6.
    d = _json(
        _bngsim(
            "check", BIRTH_DEATH, "--method", "ssa", "--t-end", "5", "--n-points", "6",
            "--seed", "2", "--checks", "all", "--json",
        ),
        exit_code=6,
    )  # fmt: skip
    assert [(c["name"], c["status"]) for c in d["check"]["checks"]] == [("replicates", "info")]
    assert d["check"]["ok"] is False


def test_help_lists_run_and_check_and_exit_6():
    run = _bngsim("--help")
    lines = [line.split() for line in run.stdout.splitlines()]
    assert ["run", "run", "a", "model;"] in [w[:4] for w in lines if w]
    assert any(w[:2] == ["check", "run"] for w in lines if w)
    assert "6 check not ok" in run.stdout


def test_a_start_time_and_a_relative_out_path(tmp_path):
    run = subprocess.run(
        [sys.executable, "-m", "bngsim", "run", CHAIN, "--t-start", "5", "--t-end", "10",
         "--n-points", "6", "--out", "rel.npz", "--json"],
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
    )  # fmt: skip
    d = _json(run)
    assert d["digest"]["t_start"] == 5.0 and d["record"]["run"]["t_start"] == 5.0
    assert d["out"] == str((tmp_path / "rel.npz").resolve())


def test_the_budget_reaches_the_check():
    # A budget spent before the first rerun: every check times out, not ok.
    d = _json(
        _bngsim("check", CHAIN, "--t-end", "5", "--budget", "1e-9", "--json"),
        _cli.EXIT_CHECK_FAILED,
    )
    assert {c["status"] for c in d["check"]["checks"]} == {"timeout"}


# ─── The review's cases ──────────────────────────────────────────────────────


def test_a_real_solver_failure_exits_4_with_one_json_object_on_stderr(tmp_path):
    model = tmp_path / "m.ant"
    model.write_text("compartment c = 2; species A in c; A = 10; k1 = 1; J: A -> ; k1*A\n")
    run = _bngsim("run", str(model), "--t-end", "1", "--set", "A=1e308", "--set", "k1=1e308",
                  "--json")  # fmt: skip
    assert run.returncode == _cli.EXIT_RUN_FAILED and run.stdout == ""
    error = json.loads(run.stderr)  # SUNDIALS' own lines are inside it, not before it
    assert error["code"].startswith("solver.") or error["code"] == "run.failed", error
    assert "log" in error and error["log"]


def test_a_real_timeout_exits_5():
    run = _bngsim("run", BIRTH_DEATH, "--method", "ssa", "--t-end", "1000", "--set", "ksyn=1e6",
                  "--timeout", "0.5", "--json")  # fmt: skip
    _error(run, _cli.EXIT_TIMEOUT, "run.timeout")


def test_replicates_share_the_time_limit():
    import time

    start = time.monotonic()
    run = _bngsim("run", BIRTH_DEATH, "--method", "ssa", "--t-end", "20", "--set", "ksyn=1e6",
                  "--timeout", "1", "--replicates", "8", "--seed", "1", "--json")  # fmt: skip
    elapsed = time.monotonic() - start
    assert run.returncode == _cli.EXIT_TIMEOUT, run.stderr
    assert json.loads(run.stderr)["details"]["timeout"] == pytest.approx(1 / 8)
    assert elapsed < 30
    d = _json(_bngsim("run", BIRTH_DEATH, "--method", "ssa", "--t-end", "1", "--replicates", "3",
                      "--timeout", "600", "--json"))  # fmt: skip
    assert d["record"]["run"]["timeout"] == pytest.approx(200.0)


def test_psa_takes_its_population_level():
    d = _json(_bngsim("run", str(examples.path("schlogl", "sbml")), "--method", "psa",
                      "--poplevel", "10", "--t-end", "5", "--seed", "1", "--json"))  # fmt: skip
    assert d["record"]["run"]["method"] == "psa"
    assert d["record"]["run"]["poplevel"] == 10.0


def test_a_network_free_run_checks_as_unsupported_not_failed():
    xml = str(examples.path("binding", "bng-xml"))
    net = str(examples.path("binding", "net"))
    run = _bngsim("check", net, "--method", "nf", "--xml", xml, "--t-end", "5", "--seed", "1",
                  "--json")  # fmt: skip
    d = _json(run, _cli.EXIT_CHECK_FAILED)  # no check could run: not ok, but no exception
    assert {c["status"] for c in d["check"]["checks"]} == {"unsupported"}


def test_a_canonical_network_free_method_takes_a_seed_and_the_xml():
    """``nf_reject`` is the method ``nf`` resolves to, and ``Simulator`` takes it.
    The command's own list held only nf, nfsim, rm and rulemonkey, so it refused
    ``--seed`` (and ``--xml``) for it as a deterministic run's."""
    xml = str(examples.path("binding", "bng-xml"))
    net = str(examples.path("binding", "net"))
    run = _bngsim("run", net, "--method", "nf_reject", "--xml", xml, "--t-end", "5",
                  "--seed", "1", "--json")  # fmt: skip
    assert _json(run)["record"]["run"]["seed"] == 1


def test_a_species_set_is_a_concentration_and_the_initial_state(tmp_path):
    # A species declared in amounts, 100 in a compartment of size 2: --set A=7
    # is Model.set_concentration's 7, and the run starts from it.
    model = tmp_path / "amt.ant"
    model.write_text(
        "compartment c = 2; species A in c; substanceOnly species A; A = 100; k = 0.1;"
        " J: A -> ; k*A\n"
    )
    d = _json(_bngsim("run", str(model), "--t-end", "1", "--set", "A=7", "--species", "--json"))
    m = Model.load(str(model))
    m.set_concentration("A", 7.0)
    m.save_concentrations()
    r = Simulator(m, method="ode").run(t_span=(0, 1), n_points=101)
    assert d["digest"] == _plain(r.digest(species=True))
    assert d["record"]["run"]["started_from"] == "initial"


def test_a_derived_parameter_in_set_is_a_usage_error(tmp_path):
    model = tmp_path / "d.ant"
    model.write_text(
        "compartment c = 1; species A in c; A = 1; k1 = 1; kd := 2*k1; J: A -> ; kd*A\n"
    )
    error = _error(_bngsim("run", str(model), "--t-end", "1", "--set", "kd=5", "--json"),
                   _cli.EXIT_USAGE, "usage.invalid_argument")  # fmt: skip
    assert error["details"]["argument"] == "--set"


def test_h5_without_h5py_is_refused_before_the_run(tmp_path):
    script = """
import sys, bngsim
sys.modules['h5py'] = None
def run(self, *a, **k):
    raise AssertionError('ran')
bngsim.Simulator.run = run
"""
    run = _bngsim("run", CHAIN, "--t-end", "1", "--out", str(tmp_path / "x.h5"), "--json",
                  script=script)  # fmt: skip
    _error(run, _cli.EXIT_ERROR, "env.missing_dependency")


def test_one_replicate_is_one_run():
    # Result.squeeze of one run is that run: a digest of one run, with its seed.
    d = _json(_bngsim("run", BIRTH_DEATH, "--method", "ssa", "--t-end", "2", "--replicates", "1",
                      "--seed", "4", "--json"))  # fmt: skip
    assert d["digest"]["n_runs"] is None and d["seeds"] == [4]


def test_an_out_that_is_a_directory_is_refused(tmp_path):
    (tmp_path / "x.npz").mkdir()
    error = _error(_bngsim("run", CHAIN, "--t-end", "1", "--out", str(tmp_path / "x.npz"),
                           "--json"), _cli.EXIT_USAGE, "usage.invalid_argument")  # fmt: skip
    assert "directory" in error["message"]


def test_a_warning_of_a_successful_json_run_still_reaches_stderr(tmp_path):
    # B has no initial value: the load warns. With --json the warning is held
    # while the command works, and written to stderr after it succeeds.
    model = tmp_path / "w.ant"
    model.write_text("compartment c = 1; species A in c, B in c; A = 1; k = 1; J: A -> B; k*A\n")
    run = _bngsim("run", str(model), "--t-end", "1", "--json")
    d = _json(run)
    assert d["digest"]["n_points"] == 101
    assert "'B'" in run.stderr and "initial" in run.stderr


def test_an_exit_mid_command_still_writes_what_was_held(tmp_path):
    # A run that writes to stderr and then exits: under --json the held text
    # reaches stderr before the process stops, as without --json.
    script = """
import sys, bngsim
def run(self, *a, **k):
    print('held line', file=sys.stderr, flush=True)
    raise SystemExit(7)
bngsim.Simulator.run = run
"""
    run = _bngsim("run", CHAIN, "--t-end", "1", "--json", script=script)
    assert run.returncode == 7
    assert "held line" in run.stderr
