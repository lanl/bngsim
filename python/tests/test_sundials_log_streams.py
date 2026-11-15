"""Issue #1680: SUNDIALS' warnings go to standard error, not standard output.

SUNDIALS' default logger writes its warnings to the C-level standard output.
They pass by ``sys.stdout``, ``warnings`` and ``logging``, so a run that is
right could put text into the output of the script that called it: ``A -> 0``
at k = 1e9 from t = 1e6, which returns the right answer, wrote eleven
``[WARNING] ... t + h = t on the next step`` lines there. bngsim now sends each
context's warnings to standard error, where SUNDIALS already sends its errors,
unless ``SUNLOGGER_WARNING_FILENAME`` says where the caller wants them.

Each case runs in a child process whose standard output (fd 1) and standard
error (fd 2) are captured apart, since nothing in the parent can see either.
"""

from __future__ import annotations

import os
import subprocess
import sys

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
CHILD = r"""
import sys, warnings
import bngsim
warnings.simplefilter("ignore")
sim = bngsim.Simulator(bngsim.Model.from_net(sys.argv[1]), method="ode")
span = (float(sys.argv[2]), float(sys.argv[3]))
try:
    if sys.argv[4] == "batch":
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


def test_a_healthy_runs_warnings_go_to_standard_error(tmp_path):
    out, warned, errors = _child(tmp_path, DECAY, 1e6, 1e6 + 1)
    assert out == "returned [1.0, 0.0, 0.0]"
    assert warned["stdout"] == 0
    assert warned["stderr"] > 0  # moved, not dropped
    assert errors == {"stdout": 0, "stderr": 0}


def test_a_failing_runs_warnings_join_its_error_on_standard_error(tmp_path):
    out, warned, errors = _child(tmp_path, SQUARE, 0.0, 2.0)
    assert out == "raised SimulationError"
    assert warned["stdout"] == 0 and warned["stderr"] > 0
    assert errors == {"stdout": 0, "stderr": 1}  # as before (#447)


def test_each_thread_of_a_batch_sends_its_warnings_to_standard_error(tmp_path):
    # A batch row runs on its own context.
    out, warned, _ = _child(tmp_path, DECAY, 1e6, 1e6 + 1, call="batch")
    assert out.startswith("returned [1.0, ")
    assert warned["stdout"] == 0 and warned["stderr"] > 0


@pytest.mark.parametrize(
    ("where", "stdout", "stderr"),
    [("stdout", True, False), ("stderr", False, True), (os.devnull, False, False)],
    ids=["stdout", "stderr", "the null device"],
)
def test_the_callers_own_choice_of_where_they_go_is_kept(tmp_path, where, stdout, stderr):
    # SUNLOGGER_WARNING_FILENAME is SUNDIALS' own variable, read when a logger
    # is made: "stdout" puts the warnings back where they were.
    out, warned, _ = _child(tmp_path, DECAY, 1e6, 1e6 + 1, SUNLOGGER_WARNING_FILENAME=where)
    assert out == "returned [1.0, 0.0, 0.0]"
    assert (warned["stdout"] > 0) is stdout
    assert (warned["stderr"] > 0) is stderr


def test_an_empty_choice_is_no_choice(tmp_path):
    out, warned, _ = _child(tmp_path, DECAY, 1e6, 1e6 + 1, SUNLOGGER_WARNING_FILENAME="")
    assert out == "returned [1.0, 0.0, 0.0]"
    assert warned["stdout"] == 0 and warned["stderr"] > 0


def test_the_error_line_still_follows_its_own_variable(tmp_path):
    out, warned, errors = _child(tmp_path, SQUARE, 0.0, 2.0, SUNLOGGER_ERROR_FILENAME=os.devnull)
    assert out == "raised SimulationError"
    assert errors == {"stdout": 0, "stderr": 0}
    assert warned["stdout"] == 0 and warned["stderr"] > 0
