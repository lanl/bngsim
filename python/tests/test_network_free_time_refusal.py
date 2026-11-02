"""GH #862: a rate law that reads time() is refused on NFsim.

NFsim cannot prepare f() = if(time()>=6, k, 0) and failed at setup with "Error
preparing function f in class GlobalFunction". It is now refused before the
backend is built, naming the function, and pointing at rm, ode and ssa, which
read time() correctly. A bare `time` is refused too.

RuleMonkey refused both, and a table function indexed by time, until 3.11.0
made it integrate a rate that reads the clock (issue #1022); its values are
checked in test_rulemonkey_model_time.py.

NFsim runs a table function indexed by time: it reads one at its x values
whether or not a reaction fires (#892), and integrates the rate where a linear
one ramps (#1179). test_nfsim_time_table_breakpoints.py checks the values.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import ModelError
from bngsim._network_free_checks import (
    clock_readers,
    time_indexed_table_functions,
    time_reading_functions,
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
XML = DATA / "nfsim" / "time_switch.xml"


def _has_nfsim() -> bool:
    return bool(getattr(bngsim, "HAS_NFSIM", False))


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
def test_the_simulator_refuses_it():
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    with pytest.raises(ModelError, match=r"functions read time\(\): 'f'.*method='rm'"):
        bngsim.Simulator(m, method="nf", xml_path=str(XML))


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
def test_an_nfsim_session_refuses_it():
    with pytest.raises(ModelError, match=r"NFsim .* read time\(\): 'f'"):
        bngsim.NfsimSession(str(XML))


def test_the_scan_finds_the_function():
    assert time_reading_functions(XML) == ["f"]


TFUN = DATA / "nfsim" / "tfun_new_format" / "valid_inline_step_time.xml"


def test_a_time_indexed_table_function_does_not_read_time_itself():
    # Its clock is the ctrName attribute and its Expression reads a placeholder.
    assert time_reading_functions(TFUN) == []
    assert time_indexed_table_functions(TFUN) == ["tfun_rate"]


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
def test_a_bare_time_is_refused_too(tmp_path):
    # The engine binds the clock as a bare `time` as well.
    xml = tmp_path / "bare_time.xml"
    xml.write_text(XML.read_text().replace("time()", "time"), encoding="utf-8")
    assert time_reading_functions(xml) == ["f"]
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    with pytest.raises(ModelError, match=r"functions read time\(\): 'f'"):
        bngsim.Simulator(m, method="nf", xml_path=str(xml))


def test_the_clock_readers_are_both_kinds():
    assert clock_readers(XML) == ["f"]
    assert clock_readers(TFUN) == ["tfun_rate"]


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
def test_nfsim_still_runs_a_time_indexed_table_function():
    # NFsim reads one at its x values whether or not anything fires (issue
    # #892); test_nfsim_time_table_breakpoints.py checks the values.
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    r = bngsim.Simulator(m, method="nf", xml_path=str(TFUN)).run(t_span=(0, 1), n_points=3, seed=1)
    assert r.n_times == 3


def test_a_model_with_no_time_reads_is_not_refused():
    assert time_reading_functions(DATA / "nfsim" / "first_order_switch.xml") == []


# ── How NFsim reads a table indexed by time ─────────────────────────────────

LINEAR = DATA / "nfsim" / "tfun_new_format" / "valid_inline_linear_time.xml"


def _edit(tmp_path: Path, src: Path, *edits: tuple[str, str]) -> Path:
    xml = src.read_text(encoding="utf-8")
    for old, new in edits:
        assert old in xml, old
        xml = xml.replace(old, new)
    out = tmp_path / "edited.xml"
    out.write_text(xml, encoding="utf-8")
    return out


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
def test_the_legacy_placeholder_without_a_method_is_a_step(tmp_path):
    # BNG2.pl 2.9.3 writes TFUN(t, "file") with __TFUN__VAL__ and no method,
    # which NFsim reads as a step (parseFuncXML.cpp).
    legacy = _edit(tmp_path, LINEAR, (' method="linear">', ">"), ("__TFUN_VAL__", "__TFUN__VAL__"))
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    r = bngsim.Simulator(m, method="nf", xml_path=str(legacy)).run(
        sample_times=[0.0, 0.5, 1.5], seed=1
    )
    assert list(r.expressions["tfun_rate"]) == pytest.approx([0.0, 0.0, 10.0])


@pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")
def test_a_linear_time_table_no_rate_law_reads_is_right_at_the_outputs(tmp_path):
    step = '<Function id="tfun_rate" type="TFUN" mode="inline" ctrName="time"'
    xml = _edit(
        tmp_path,
        TFUN,
        (
            step,
            '<Function id="drive" type="TFUN" mode="inline" ctrName="time" xData="0,1,2" '
            'yData="0,10,20" method="linear"><Expression>__TFUN_VAL__</Expression>'
            "<ListOfReferences/></Function>\n      " + step,
        ),
    )
    m = bngsim.Model.from_net(str(DATA / "simple_decay.net"))
    r = bngsim.Simulator(m, method="nf", xml_path=str(xml)).run(
        sample_times=[0.0, 0.5, 1.5], seed=1
    )
    assert list(r.expressions["drive"]) == pytest.approx([0.0, 5.0, 15.0])
