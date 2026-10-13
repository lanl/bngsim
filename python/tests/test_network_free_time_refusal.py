"""GH #862: NFsim refuses a rate law that reads time(); RuleMonkey runs it.

NFsim cannot prepare such a function and failed at setup with "Error preparing
function f in class GlobalFunction". It is refused before the backend is built,
naming the function and pointing at rm, ode and ssa, which read time()
correctly. A bare `time` is refused too.

RuleMonkey was refused here as well until 3.11.0, because a total propensity of
0 was final and A(s~0) -> A(s~1) at f() = if(time()>=6, k, 0) never fired
(richardposner/RuleMonkey#86). It now runs both a function that reads time() and
a table function indexed by time (issue #1022). The values are checked in
test_rulemonkey_model_clock.py, and how bngsim reads a model's use of the clock
in test_rulemonkey_clock_profile.py.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import (
    time_indexed_table_functions,
    time_reading_functions,
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
XML = DATA / "nfsim" / "time_switch.xml"


def _available(method: str) -> bool:
    if method == "nf":
        return bool(getattr(bngsim, "HAS_NFSIM", False))
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


# Literal reasons, the ones the other network-free tests give: the skip audit
# (test_skip_audit.py) reads each reason from the source and cannot read an
# f-string's.
needs_nfsim = pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
needs_rulemonkey = pytest.mark.skipif(not _available("rm"), reason="RuleMonkey not compiled in")


@needs_nfsim
def test_the_simulator_refuses_it():
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    with pytest.raises(ModelError, match=r"NFsim .* read time\(\): 'f'"):
        bngsim.Simulator(m, method="nf", xml_path=str(XML))


@needs_nfsim
def test_an_nfsim_session_refuses_it():
    with pytest.raises(ModelError, match=r"NFsim .* read time\(\): 'f'"):
        bngsim.NfsimSession(str(XML))


@needs_nfsim
def test_the_refusal_points_at_the_backends_that_run_it():
    with pytest.raises(ModelError, match=r"method='rm' .* method='ode' or method='ssa'"):
        bngsim.NfsimSession(str(XML))


@needs_rulemonkey
def test_rulemonkey_is_not_refused():
    """Building the simulator or the session raises nothing (issue #1022)."""
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    for xml in (XML, TFUN):
        bngsim.Simulator(m, method="rm", xml_path=str(xml))
        bngsim.RuleMonkeySession(str(xml)).destroy()


def test_the_scan_finds_the_function():
    assert time_reading_functions(XML) == ["f"]


TFUN = DATA / "nfsim" / "tfun_new_format" / "valid_inline_step_time.xml"


def test_a_time_indexed_table_function_does_not_read_time_itself():
    # Its clock is the ctrName attribute and its Expression reads a placeholder.
    assert time_reading_functions(TFUN) == []
    assert time_indexed_table_functions(TFUN) == ["tfun_rate"]


@needs_nfsim
def test_a_bare_time_is_refused_too(tmp_path):
    # NFsim fails to prepare f under the bare spelling as well.
    xml = tmp_path / "bare_time.xml"
    xml.write_text(XML.read_text().replace("time()", "time"), encoding="utf-8")
    assert time_reading_functions(xml) == ["f"]
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    with pytest.raises(ModelError, match=r"functions read time\(\): 'f'"):
        bngsim.Simulator(m, method="nf", xml_path=str(xml))


@needs_nfsim
def test_nfsim_still_runs_a_time_indexed_table_function():
    # NFsim reads one correctly while reactions fire; the case where nothing
    # fires is issue #892.
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    r = bngsim.Simulator(m, method="nf", xml_path=str(TFUN)).run(t_span=(0, 1), n_points=3, seed=1)
    assert r.n_times == 3


def test_a_model_with_no_time_reads_is_not_refused():
    assert time_reading_functions(DATA / "nfsim" / "first_order_switch.xml") == []
