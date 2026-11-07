"""Issue #1556: numpy's floating-point warnings in a stop condition, on any Python.

A stop condition runs under ``eval`` with no builtins. On Python 3.13 and later,
a warning numpy issues there (0/0, x/0, an overflow) looks ``__import__`` up in
those builtins and raises KeyError, and since #1509 that refused the run. On
3.12 the same warning was issued from inside the condition (its file is
``<stop_condition>``), which is what fails on 3.13: the tests below catch that
on 3.12 too.

Now each error numpy is set to warn of is collected and told by one
RuntimeWarning from outside the condition, and the sample is read at the
value numpy gives it: NaN is never met, +inf is met where the condition reads
``> 0``. numpy's other modes are the user's and are kept.

A() decays at k = 0.1 from 100 (A_tot = 100 exp(-t/10)) on the grid
t = 0, 1, ..., 100, so where each condition below is first met follows from
that closed form.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

NET = (
    "begin parameters\n 1 k 0.1\nend parameters\n"
    "begin species\n 1 A() 100\nend species\n"
    "begin reactions\n 1 1 0 k\nend reactions\n"
    "begin groups\n 1 A_tot 1\nend groups\n"
)
KW = dict(t_span=(0, 100), n_points=101)
NAN_EVERYWHERE = "(A_tot - A_tot) / (A_tot - A_tot)"


@pytest.fixture
def net(tmp_path):
    path = tmp_path / "decay.net"
    path.write_text(NET)
    return str(path)


def _run(net: str, condition: str) -> tuple[float | None, list[warnings.WarningMessage]]:
    """Where *condition* stops the decay, and every warning the run issued."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    sim.add_stop_condition(condition)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            sim.run(**KW)
            stop = None
        except bngsim.StopConditionMet as met:
            stop = float(met.result.time[-1])
    return stop, list(caught)


def _told(caught: list[warnings.WarningMessage], condition: str) -> list[str]:
    """The RuntimeWarnings that name *condition*; none may come from inside it."""
    inside = [str(w.message) for w in caught if w.filename == "<stop_condition>"]
    assert not inside, f"warned from inside the condition: {inside}"
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, RuntimeWarning) and repr(condition) in str(w.message)
    ]


@pytest.mark.parametrize(
    "condition, stop, kind",
    [
        # 0/0 = NaN at every sample, never met.
        (NAN_EVERYWHERE, None, "invalid value"),
        # t/t is 0/0 = NaN at t = 0 only, so this is A_tot < 10, first at t = 24
        # (10 ln 10 = 23.03).
        ("(A_tot < 10) * (t / t)", 24.0, "invalid value"),
        # 1/(t - 50) is negative before t = 50 and 1/0 = +inf there: met at 50.
        ("1 / (t - 50) > 0", 50.0, "divide by zero"),
        # A_tot**1000 overflows to +inf while A_tot > 2.03 (t < 39), and is < 1
        # once A_tot < 1 (t > 10 ln 100 = 46.05): 1.005**1000 = 175 at t = 46.
        ("A_tot ** 1000 < 1", 47.0, "overflow"),
    ],
)
def test_an_arithmetic_error_is_read_as_its_value_and_told_from_outside(
    net, condition, stop, kind
):
    """The run is not refused, stops where IEEE arithmetic on the closed form
    says, and says once, from the caller's line, which error it met."""
    got, caught = _run(net, condition)
    assert got == stop
    told = _told(caught, condition)
    assert told == [
        f"{kind} encountered in stop condition {condition!r}; each sample is read at "
        "the value numpy gives it, where NaN is not met (issue #1556)"
    ]
    assert [w.filename for w in caught if str(w.message) in told] == [__file__]


def test_an_error_past_the_sample_met_is_not_told(net):
    """(A_tot < 10) + 0 * (1 / (t - 50)) is met at t = 24, before 1/0 at t = 50:
    the samples after the one met are not read, so nothing is told."""
    condition = "(A_tot < 10) + 0 * (1 / (t - 50))"
    got, caught = _run(net, condition)
    assert got == 24.0
    assert _told(caught, condition) == []


def test_a_warning_python_cannot_deliver_from_a_condition_is_named_in_the_refusal(net):
    """Casting a complex to a real issues numpy's ComplexWarning, which is not a
    floating-point error and is shown by default: on 3.13 and later it cannot be
    delivered from inside the condition, and the refusal says why rather than
    KeyError alone. Where it runs, this is A_tot < 10, met at t = 24."""
    condition = "(A_tot + 0j).astype('float64') < 10"
    if sys.version_info < (3, 13):
        assert _run(net, condition)[0] == 24.0
        return
    with pytest.raises(
        ValueError,
        match=r"could not be evaluated at t=0: KeyError: '__import__', which a stop condition "
        r"does not have; on Python 3\.13 and later a warning issued inside one needs it "
        r"\(issues #1509, #1556\)$",
    ):
        _run(net, condition)


def test_two_kinds_of_error_are_told_in_one_warning(net):
    """0/0 at every sample and 1/0 at t = 50: both kinds, in the order met."""
    condition = "(A_tot - A_tot) / (A_tot - A_tot) + 1 / (t - 50)"
    got, caught = _run(net, condition)
    assert got is None  # NaN + anything is NaN
    (told,) = _told(caught, condition)
    assert told.startswith("invalid value and divide by zero encountered in stop condition")


@pytest.mark.parametrize("mode", ["ignore", "print"])
def test_an_error_numpy_is_set_not_to_warn_of_is_not_told(net, mode, capfd):
    """np.errstate(invalid=...) is the user's: ignored or printed, as numpy
    does it, and bngsim adds no warning of its own."""
    with np.errstate(invalid=mode):
        got, caught = _run(net, NAN_EVERYWHERE)
    assert got is None
    assert _told(caught, NAN_EVERYWHERE) == []
    printed = capfd.readouterr()
    assert ("invalid value encountered" in printed.out + printed.err) == (mode == "print")


def test_an_error_numpy_is_set_to_raise_refuses_the_run(net):
    """np.errstate(invalid="raise") makes 0/0 raise FloatingPointError, which is
    an expression that raises at a sample: refused, as #1509 has it."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    sim.add_stop_condition(NAN_EVERYWHERE)
    with (
        np.errstate(invalid="raise"),
        pytest.raises(ValueError, match="could not be evaluated at t=0: FloatingPointError"),
    ):
        sim.run(**KW)


class _Handler:
    """numpy's error callback ("call") and error log ("log") in one."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.lines: list[str] = []

    def __call__(self, kind: str, flag: int) -> None:
        self.calls.append((kind, flag))

    def write(self, message: str) -> None:
        self.lines.append(message)


@pytest.mark.parametrize("mode", ["call", "log"])
def test_an_error_numpy_is_set_to_call_or_log_for_still_reaches_the_users_handler(net, mode):
    """The user's callback or log gets each 0/0, at each of the 101 samples, and
    bngsim adds no warning: the handler is how the user asked to hear of it."""
    handler = _Handler()
    with np.errstate(invalid=mode, call=handler):
        got, caught = _run(net, NAN_EVERYWHERE)
        assert np.geterrcall() is handler  # still the user's after the run
    assert got is None
    assert _told(caught, NAN_EVERYWHERE) == []
    if mode == "call":
        assert handler.calls == [("invalid value", 8)] * 101
        assert handler.lines == []
    else:
        assert handler.calls == []
        assert len(handler.lines) == 101
        assert all("invalid value" in line for line in handler.lines)


def test_a_kind_set_to_call_goes_to_the_user_and_one_set_to_warn_is_told(net):
    """divide="call" with invalid left at "warn": the user's callback gets the
    one 1/0 (t = 50) and bngsim tells only the 0/0, which is every sample."""
    condition = "(A_tot - A_tot) / (A_tot - A_tot) + 1 / (t - 50)"
    handler = _Handler()
    with np.errstate(divide="call", invalid="warn", call=handler):
        got, caught = _run(net, condition)
    assert got is None
    assert handler.calls == [("divide by zero", 1)]
    assert [m.split(" encountered")[0] for m in _told(caught, condition)] == ["invalid value"]


@pytest.mark.parametrize(
    "mode, numpys_error",
    [
        ("call", "python callback specified for invalid value"),
        ("log", "log specified for"),
    ],
)
def test_a_kind_set_to_call_or_log_with_nowhere_to_go_refuses_the_run_as_numpy_raises(
    net, mode, numpys_error
):
    """numpy raises NameError for "call" with no callback, or "log" with no
    object to write to; that is an expression that raises, refused (#1509)."""
    sim = bngsim.Simulator(bngsim.Model.from_net(net), method="ode")
    sim.add_stop_condition(NAN_EVERYWHERE)
    with (
        np.errstate(invalid=mode, call=None),
        pytest.raises(ValueError, match=f"at t=0: NameError: {numpys_error}"),
    ):
        sim.run(**KW)


def test_a_clean_condition_warns_nothing_and_leaves_numpys_state_alone(net):
    """A_tot < 10 meets no error: met at t = 24, nothing told, and numpy's
    modes and callback are what they were before the run."""
    before = (np.geterr(), np.geterrcall())
    got, caught = _run(net, "A_tot < 10")
    assert got == 24.0
    assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == []
    assert (np.geterr(), np.geterrcall()) == before
    # And after a run that met errors, too.
    _run(net, NAN_EVERYWHERE)
    assert (np.geterr(), np.geterrcall()) == before


SMOKE = Path(__file__).resolve().parents[2] / "ci" / "wheel_smoke_stop_condition.py"


def test_the_wheels_run_the_smoke_script_and_it_passes():
    """The pull-request suites run 3.12; every CPython a wheel is built for runs
    ci/wheel_smoke_stop_condition.py through cibuildwheel's test-command. A
    script nothing runs protects nothing, and one that fails here would fail
    every wheel leg. Run from outside the repo root, as cibuildwheel does."""
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")
    pyproject = SMOKE.parents[1] / "pyproject.toml"
    command = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["cibuildwheel"][
        "test-command"
    ]
    assert "python {project}/ci/wheel_smoke_stop_condition.py" in command
    proc = subprocess.run(
        [sys.executable, str(SMOKE)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
