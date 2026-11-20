"""Issue #1680: SUNDIALS' warnings go to standard error, not standard output.

SUNDIALS' default logger writes its warnings to the C-level standard output.
They pass by ``sys.stdout``, ``warnings`` and ``logging``, so a run that is
right could put text into the output of the script that called it: ``A -> 0``
at k = 1e9 from t = 1e6, which returns the right answer, wrote eleven
``[WARNING] ... t + h = t on the next step`` lines there. bngsim now sends each
context's warnings to standard error, where SUNDIALS already sends its errors,
unless ``SUNLOGGER_WARNING_FILENAME`` says where the caller wants them.

A run that drops its own warnings (a model with discontinuity or state-switch
roots, in a time course or a steady-state march) drops them only where the
caller has not set that variable: where it is set, they go where it says.

Each case runs in a child process whose standard output (fd 1) and standard
error (fd 2) are captured apart, since nothing in the parent can see either.
Each needs a run that draws a warning; a module probe asks for one with
``SUNLOGGER_WARNING_FILENAME=stderr`` and skips the cases whose run draws none
on this host (a SUNDIALS logging below warnings), rather than fail them.
"""

from __future__ import annotations

import os
import subprocess
import sys

import bngsim
import pytest

# A -> 0 at k = 1e9 over (1e6, 1e6 + 1): right ([1, 0, 0]), and CVODE's first
# steps are shorter than the doubles are apart at 1e6, which draws the warning.
DECAY = (
    "begin parameters\n 1 k 1e9\nend parameters\n"
    "begin species\n 1 A() 1\nend species\n"
    "begin reactions\n 1 1 0 k\nend reactions\n"
)
# X' = X^2 from 1 has no solution past t = 1: the run fails, with SUNDIALS'
# [ERROR] line beside the warnings.
SQUARE = (
    "begin parameters\n 1 k 1\nend parameters\n"
    "begin functions\n 1 f() k*Xtot*Xtot\nend functions\n"
    "begin species\n 1 X() 1\nend species\n"
    "begin reactions\n 1 0 1 f\nend reactions\n"
    "begin groups\n 1 Xtot 1\nend groups\n"
)
# Y is made at rate if(E > 0, k, 0), and nothing makes E, so the state-switch
# root's residual E is zero from the start: CVODE warns "some root functions
# identically 0" on every run, whatever the step sizes (Z decays so that a
# steady-state march takes steps). bngsim drops a root model's warnings itself.
ROOTED = (
    "begin parameters\n 1 k 1.0\nend parameters\n"
    "begin functions\n 1 r() if(Eobs>0,k,0)\nend functions\n"
    "begin species\n 1 E() 0\n 2 Y() 0\n 3 Z() 1\nend species\n"
    "begin reactions\n 1 0 2 r\n 2 3 0 k\nend reactions\n"
    "begin groups\n 1 Eobs 1\nend groups\n"
)
CHILD = r"""
import sys, warnings
import bngsim
warnings.simplefilter("ignore")
sim = bngsim.Simulator(bngsim.Model.from_net(sys.argv[1]), method="ode")
span = (float(sys.argv[2]), float(sys.argv[3]))
try:
    if sys.argv[4] in ("integration", "newton"):
        state = sim.steady_state(method=sys.argv[4])
        print("returned", [round(float(v), 6) + 0.0 for v in state.concentrations])
    elif sys.argv[4] == "batch":
        rows = sim.run_batch(t_span=span, n_points=3, params=[{"k": 1e9}, {"k": 2e9}],
                             num_processors=2)
        print("returned", [float(v) for row in rows for v in row.species[:, 0]])
    else:
        print("returned", [float(v) for v in sim.run(t_span=span, n_points=3).species[:, 0]])
except bngsim.SimulationError:
    print("raised SimulationError")
"""


def _child(tmp_path, model, t0, t1, call="run", **env):
    """What the child printed, and its [WARNING] and [ERROR] lines per stream."""
    net = tmp_path / "model.net"
    net.write_text(model)
    environ = {k: v for k, v in os.environ.items() if not k.startswith("SUNLOGGER_")}
    environ.update(env)
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(net), repr(t0), repr(t1), call],
        capture_output=True,
        text=True,
        env=environ,
        timeout=300,
    )
    own = [line for line in done.stdout.splitlines() if not line.startswith("[")]
    assert own, done.stderr[-2000:]

    def lines(text, tag):
        return sum(line.startswith(f"[{tag}]") for line in text.splitlines())

    warned = {"stdout": lines(done.stdout, "WARNING"), "stderr": lines(done.stderr, "WARNING")}
    errors = {"stdout": lines(done.stdout, "ERROR"), "stderr": lines(done.stderr, "ERROR")}
    return own[-1], warned, errors


_RUNS = {
    "decay": (DECAY, 1e6, 1e6 + 1, "run"),
    "square": (SQUARE, 0.0, 2.0, "run"),
}


@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    """How many warnings each run draws here, asked for on standard error.

    SUNDIALS follows ``SUNLOGGER_WARNING_FILENAME`` itself, and bngsim leaves
    it as set, so this counts what the run draws, wherever bngsim would put it.
    """
    counts = {}
    for name, (model, t0, t1, call) in _RUNS.items():
        where = tmp_path_factory.mktemp(name)
        _, warned, _ = _child(where, model, t0, t1, call, SUNLOGGER_WARNING_FILENAME="stderr")
        counts[name] = warned["stderr"]
    return counts


def _needs(drawn, *names):
    """Skip unless one of the named runs draws a warning here."""
    if not any(drawn[name] for name in names):
        pytest.skip(f"SUNDIALS draws no warning on this host for the {' or '.join(names)} run")


def test_a_healthy_runs_warnings_go_to_standard_error(tmp_path, drawn):
    _needs(drawn, "decay")
    out, warned, errors = _child(tmp_path, DECAY, 1e6, 1e6 + 1)
    assert out == "returned [1.0, 0.0, 0.0]"
    assert warned["stdout"] == 0
    assert warned["stderr"] > 0  # moved, not dropped
    assert errors == {"stdout": 0, "stderr": 0}


def test_a_failing_runs_warnings_join_its_error_on_standard_error(tmp_path, drawn):
    _needs(drawn, "square")
    out, warned, errors = _child(tmp_path, SQUARE, 0.0, 2.0)
    assert out == "raised SimulationError"
    assert warned["stdout"] == 0 and warned["stderr"] > 0
    assert errors == {"stdout": 0, "stderr": 1}  # as before (#447)


def test_each_thread_of_a_batch_sends_its_warnings_to_standard_error(tmp_path, drawn):
    _needs(drawn, "decay")
    # A batch row runs on its own context.
    out, warned, _ = _child(tmp_path, DECAY, 1e6, 1e6 + 1, call="batch")
    assert out.startswith("returned [1.0, ")
    assert warned["stdout"] == 0 and warned["stderr"] > 0


@pytest.mark.parametrize(
    ("where", "stdout", "stderr"),
    [("stdout", True, False), ("stderr", False, True), (os.devnull, False, False)],
    ids=["stdout", "stderr", "the null device"],
)
def test_the_callers_own_choice_of_where_they_go_is_kept(tmp_path, drawn, where, stdout, stderr):
    _needs(drawn, "decay")  # so that the null device's "none" means dropped
    # SUNLOGGER_WARNING_FILENAME is SUNDIALS' own variable, read when a logger
    # is made: "stdout" puts the warnings back where they were.
    out, warned, _ = _child(tmp_path, DECAY, 1e6, 1e6 + 1, SUNLOGGER_WARNING_FILENAME=where)
    assert out == "returned [1.0, 0.0, 0.0]"
    assert (warned["stdout"] > 0) is stdout
    assert (warned["stderr"] > 0) is stderr


def test_an_empty_choice_is_no_choice(tmp_path, drawn):
    _needs(drawn, "decay")
    out, warned, _ = _child(tmp_path, DECAY, 1e6, 1e6 + 1, SUNLOGGER_WARNING_FILENAME="")
    assert out == "returned [1.0, 0.0, 0.0]"
    assert warned["stdout"] == 0 and warned["stderr"] > 0


def test_the_error_line_still_follows_its_own_variable(tmp_path, drawn):
    _needs(drawn, "square")
    out, warned, errors = _child(tmp_path, SQUARE, 0.0, 2.0, SUNLOGGER_ERROR_FILENAME=os.devnull)
    assert out == "raised SimulationError"
    assert errors == {"stdout": 0, "stderr": 0}
    assert warned["stdout"] == 0 and warned["stderr"] > 0


@pytest.mark.parametrize("call", ["run", "integration", "newton"])
def test_a_root_model_drops_its_warnings_unless_the_caller_says_where(tmp_path, drawn, call):
    # register_roots (a time course) and the steady-state march's roots send
    # the context's warnings to the null device after SunContextGuard has sent
    # them to standard error; a caller who set SUNLOGGER_WARNING_FILENAME has
    # said where they go, and that holds there too. The root's warning does not
    # hang on step sizes, so a SUNDIALS that logs warnings at all draws it; it
    # is not probed itself, since the probe would go through the sites tested.
    _needs(drawn, "decay", "square")
    want = "returned [0.0, 0.0, " if call == "run" else "returned [0.0, 0.0, 0.0]"
    out, warned, _ = _child(tmp_path, ROOTED, 0.0, 10.0, call)
    assert out.startswith(want), out
    assert warned == {"stdout": 0, "stderr": 0}  # dropped, as before #1680
    out, warned, _ = _child(tmp_path, ROOTED, 0.0, 10.0, call, SUNLOGGER_WARNING_FILENAME="stdout")
    assert out.startswith(want), out
    assert warned["stdout"] > 0 and warned["stderr"] == 0
    out, warned, _ = _child(tmp_path, ROOTED, 0.0, 10.0, call, SUNLOGGER_WARNING_FILENAME="stderr")
    assert warned["stdout"] == 0 and warned["stderr"] > 0


@pytest.mark.parametrize(
    ("variable", "call"),
    [
        ("SUNLOGGER_WARNING_FILENAME", "run"),
        ("SUNLOGGER_ERROR_FILENAME", "run"),
        ("SUNLOGGER_WARNING_FILENAME", "sensitivities"),
        ("SUNLOGGER_WARNING_FILENAME", "integration"),
        ("SUNLOGGER_WARNING_FILENAME", "newton"),
    ],
)
def test_a_log_file_sundials_cannot_open_is_named_in_the_error(
    tmp_path, monkeypatch, drawn, variable, call
):
    # SUNDIALS opens the file the variable names when it makes a solver
    # context, and makes none where it cannot, so every run and solve fails.
    # The error said only "SUNContext_Create failed"; it names the variable
    # and its value. A SUNDIALS that logs warnings opens both files.
    _needs(drawn, "decay", "square")
    where = tmp_path / "no such directory" / "sundials.log"
    net = tmp_path / "model.net"
    net.write_text(DECAY)
    monkeypatch.setenv(variable, str(where))
    kw = {"sensitivity_params": ["k"]} if call == "sensitivities" else {}
    sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode", **kw)
    with pytest.raises(bngsim.SimulationError) as caught:
        if call in ("integration", "newton"):
            sim.steady_state(method=call)
        else:
            sim.run(t_span=(0.0, 1.0), n_points=3)
    assert "SUNContext_Create failed" in str(caught.value)
    assert f"{variable}={where}" in str(caught.value)


def test_a_log_file_name_that_is_not_utf8_is_shown_in_the_error(tmp_path, monkeypatch, drawn):
    # The error quotes the variable's value, and a file name is bytes: one that
    # is not UTF-8 text would have turned the SimulationError into a bare
    # UnicodeDecodeError on its way to Python (issue #1653). The byte is shown
    # as Python's "backslashreplace" shows it. "\udcff" is how Python holds
    # the byte 0xff of a POSIX file name; elsewhere the name need only be named.
    _needs(drawn, "decay", "square")
    net = tmp_path / "model.net"
    net.write_text(DECAY)
    monkeypatch.setenv("SUNLOGGER_WARNING_FILENAME", f"{tmp_path}/no\udcff directory/w.log")
    sim = bngsim.Simulator(bngsim.Model.from_net(str(net)), method="ode")
    with pytest.raises(bngsim.SimulationError) as caught:
        sim.run(t_span=(0.0, 1.0), n_points=3)
    assert "SUNLOGGER_WARNING_FILENAME=" in str(caught.value)
    if os.name == "posix":
        assert f"{tmp_path}/no\\xff directory/w.log" in str(caught.value)
