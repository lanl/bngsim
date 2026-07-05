"""Tests for bngsim.convert.validate_conversion — the L0–L4 conversion-
validation framework (GH #217, converter epic #211).

Covers the five escalating gates (L0 syntactic validity, L1 structural
equivalence, L2 round-trip identity, L3 numerical equivalence, L4 best-effort
symbolic equivalence) over the tracked clean ``.net`` corpus in **both**
directions, the artifact surface (``ConversionValidationReport`` / ``to_dict`` /
``summary`` / ``ok``), the level-subset selection, direction inference, the
non-gating nature of L4, the strict-refusal verdict path, the lossy conversion
that only L2/L3 (not L0/L1) catch, and the CLI entry point.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim.convert import (
    ConversionValidationReport,
    LevelResult,
    validate_conversion,
    write_sbml,
)

pytestmark = pytest.mark.skipif(
    not bngsim.HAS_LIBSBML, reason="conversion validation requires libsbml"
)

_BNGSIM = Path(__file__).resolve().parents[2]
_DATA = _BNGSIM / "tests" / "data"
_SMITH = _BNGSIM / "benchmarks" / "models" / "sbml" / "Smith2013_BIOMD0000000474_petab.xml"

# Tracked, network-clean .net models that convert faithfully (unit volume).
_CLEAN_NETS = [
    "simple_decay.net",
    "case_sensitivity.net",
    "derived_rate_const.net",
    "derived_quotient.net",
    "two_species_reversible.net",
    "t_as_observable.net",
    "fixed_species.net",
    "preequil_prod_deg.net",
    "time_dependent_func.net",
    "mm_tqssa.net",
    "obs_zero_arg_call.net",
]

_GATES = ("L0", "L1", "L2", "L3")


@pytest.fixture
def data_dir() -> Path:
    assert _DATA.is_dir(), f"Test data directory not found: {_DATA}"
    return _DATA


def _require(name: str, data_dir: Path) -> Path:
    src = data_dir / name
    if not src.is_file():
        pytest.skip(f"tracked .net not present: {src}")
    return src


# ─── net→SBML direction: all hard gates pass over the clean corpus ──────────


@pytest.mark.parametrize("name", _CLEAN_NETS)
def test_net2sbml_gates_pass(name: str, data_dir: Path) -> None:
    src = _require(name, data_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = validate_conversion(src)

    assert isinstance(report, ConversionValidationReport)
    assert report.direction == "net2sbml"
    assert report.ok, report.summary()
    for lv in _GATES:
        level = report.level(lv)
        assert level is not None and level.status == "pass", report.summary()


# ─── SBML→.net direction over the same models (round-trip via clean SBML) ───


@pytest.mark.parametrize("name", _CLEAN_NETS)
def test_sbml2net_gates_pass(name: str, data_dir: Path, tmp_path: Path) -> None:
    src = _require(name, data_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        # Produce a clean SBML source from the .net, then grade the reverse.
        model = bngsim.Model.from_net(src)
        sbml = tmp_path / (src.stem + ".xml")
        write_sbml(model, sbml, strict=True)
        report = validate_conversion(sbml)

    assert report.direction == "sbml2net"
    assert report.ok, report.summary()
    for lv in _GATES:
        assert report.level(lv).status == "pass", report.summary()


# ─── L4 best-effort symbolic equivalence ────────────────────────────────────


def test_l4_reports_algebraic_equality(data_dir: Path) -> None:
    src = _require("two_species_reversible.net", data_dir)
    report = validate_conversion(src, levels=("L4",))
    l4 = report.level("L4")
    assert l4 is not None
    assert l4.gating is False  # never blocks a conversion
    assert l4.status == "equal", l4.detail


def test_l4_punts_inconclusive_on_mm(data_dir: Path) -> None:
    # The .net MM (tQSSA) closed form is re-emitted by SBML in a non-smooth
    # (Max-clamped) algebraic form; symbolic equality cannot crack it, so L4 is
    # allowed to punt to *inconclusive* — and it must not block the conversion.
    src = _require("mm_tqssa.net", data_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = validate_conversion(src)
    assert report.level("L4").status == "inconclusive"
    assert report.ok  # gates still pass; L4 is non-gating


def test_l4_does_not_affect_overall_ok(data_dir: Path) -> None:
    src = _require("mm_tqssa.net", data_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = validate_conversion(src)
    # Overall ok is the conjunction of the gating levels only.
    assert report.ok == all(lv.ok for lv in report.levels if lv.gating)


# ─── L4 floating-point round-off tolerance ──────────────────────────────────


def test_l4_max_numeric_coeff() -> None:
    import sympy as sp
    from bngsim.convert._validate import _max_numeric_coeff

    s0, s1 = sp.symbols("_s0 _s1")
    assert _max_numeric_coeff(sp.sympify("1.78e-15*_s0*_s1 + 6.9e-18")) == pytest.approx(1.78e-15)
    assert _max_numeric_coeff(0.5 * s0 - 2 * s1) == 2.0
    assert _max_numeric_coeff(s0 - s0) == 0.0  # expands to 0


class _FakeModel:
    """A stand-in carrying just the state the symbolic verdict samples."""

    def __init__(self, state: list[float]) -> None:
        self._state = state

    def get_state(self) -> list[float]:
        return self._state


def test_l4_forgives_fp_residual_reports_equal(monkeypatch) -> None:
    """A Δ that is pure machine-precision coefficient dust (what the BNGL-float →
    MathML → back round-trip leaves) is forgiven as ``equal`` up to round-off —
    not the misleading ``not-equal`` that ignores the gate's own L2 RHS identity."""
    import sympy as sp
    from bngsim.convert import _validate

    s0, s1 = sp.symbols("_s0 _s1")
    base = {0: 2 * s0 * s1 - 3 * s0, 1: -2 * s0 * s1 + 3 * s0}
    dusted = {
        0: base[0] + sp.Float("1.78e-15") * s0 * s1,
        1: base[1] - sp.Float("4.4e-16") * s0,
    }
    rhs = iter([(base, 2), (dusted, 2)])
    monkeypatch.setattr(_validate, "_symbolic_rhs", lambda _m: next(rhs))
    status, detail = _validate._symbolic_verdict(_FakeModel([1.0, 2.0]), _FakeModel([1.0, 2.0]))
    assert status == "equal"
    assert "round-off" in detail


def test_l4_genuine_difference_still_not_equal(monkeypatch) -> None:
    """A magnitude-carrying Δ (a whole extra term) evaluates nonzero → the
    numeric adjudication confirms a real difference, reported ``not-equal``."""
    import sympy as sp
    from bngsim.convert import _validate

    s0, s1 = sp.symbols("_s0 _s1")
    a = {0: 2 * s0 * s1, 1: -2 * s0 * s1}
    b = {0: 2 * s0 * s1 + sp.Rational(1, 2) * s0, 1: -2 * s0 * s1}
    rhs = iter([(a, 2), (b, 2)])
    monkeypatch.setattr(_validate, "_symbolic_rhs", lambda _m: next(rhs))
    status, detail = _validate._symbolic_verdict(_FakeModel([1.0, 2.0]), _FakeModel([1.0, 2.0]))
    assert status == "not-equal"
    assert "numerically confirmed" in detail


def test_l4_cancellation_residual_is_inconclusive_not_not_equal(monkeypatch) -> None:
    """A residual with huge coefficients that *cancel numerically* (the
    catastrophic-cancellation case the per-coefficient screen mis-reads, like the
    bng_parity ``rab_rab7_ox`` 5.79e77 residual that L2 confirms is ~1e-16) must
    not be ``not-equal``: the screen fails but numeric adjudication shows ~0 →
    honest ``inconclusive``, never a false alarm."""
    import sympy as sp
    from bngsim.convert import _validate

    s0, s1 = sp.symbols("_s0 _s1")
    big = sp.Float("5.79e77")
    # Symbolically distinct (big*s0 vs big*s1) so simplify won't reduce it, but
    # equal wherever s0 == s1 — which is the sampled state below.
    a = {0: big * s0}
    b = {0: big * s1}
    rhs = iter([(a, 2), (b, 2)])
    monkeypatch.setattr(_validate, "_symbolic_rhs", lambda _m: next(rhs))
    status, detail = _validate._symbolic_verdict(_FakeModel([1.0, 1.0]), _FakeModel([1.0, 1.0]))
    assert status == "inconclusive", detail
    assert "could not be symbolically reduced" in detail


# ─── Lossy conversion: L0/L1 pass but L2/L3 catch the semantic loss ─────────


@pytest.mark.skipif(not _SMITH.is_file(), reason="Smith2013 benchmark not present")
def test_lossy_conversion_caught_by_numeric_gates() -> None:
    # Smith2013 has amount-valued species in non-unit volumes; plain .net cannot
    # carry the volume, so a best-effort (lossy) conversion is structurally
    # valid yet numerically wrong — exactly what L2/L3 exist to catch.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = validate_conversion(_SMITH, strict=False, levels=_GATES)
    assert report.level("L0").status == "pass"
    assert report.level("L1").status == "pass"
    assert report.level("L2").status == "fail"
    assert report.level("L3").status == "fail"
    assert not report.ok


@pytest.mark.skipif(not _SMITH.is_file(), reason="Smith2013 benchmark not present")
def test_strict_refusal_returns_failed_report_not_exception() -> None:
    # Under strict mode the converter refuses an unfaithful conversion; the
    # validator must surface that as a failed verdict, never crash.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        report = validate_conversion(_SMITH, strict=True, levels=_GATES)
    assert not report.ok
    assert report.target_path is None
    for lv in _GATES:
        assert report.level(lv).status == "fail"
        assert "refused under strict mode" in report.level(lv).detail


# ─── Report artifact surface ────────────────────────────────────────────────


def test_to_dict_is_json_serializable(data_dir: Path) -> None:
    import json

    src = _require("simple_decay.net", data_dir)
    report = validate_conversion(src)
    blob = json.dumps(report.to_dict())  # must not raise
    parsed = json.loads(blob)
    assert parsed["direction"] == "net2sbml"
    assert parsed["ok"] is True
    assert {lv["level"] for lv in parsed["levels"]} == {"L0", "L1", "L2", "L3", "L4"}


def test_summary_lists_every_level(data_dir: Path) -> None:
    src = _require("simple_decay.net", data_dir)
    report = validate_conversion(src)
    text = report.summary()
    for lv in ("L0", "L1", "L2", "L3", "L4"):
        assert lv in text
    assert "PASS" in text


# ─── Level selection and direction inference ────────────────────────────────


def test_level_subset_runs_only_requested(data_dir: Path) -> None:
    src = _require("simple_decay.net", data_dir)
    report = validate_conversion(src, levels=("L0", "L3"))
    assert {lv.level for lv in report.levels} == {"L0", "L3"}


def test_unknown_level_rejected(data_dir: Path) -> None:
    src = _require("simple_decay.net", data_dir)
    with pytest.raises(ValueError, match="unknown level"):
        validate_conversion(src, levels=("L9",))


def test_direction_inference_and_override(data_dir: Path, tmp_path: Path) -> None:
    src = _require("simple_decay.net", data_dir)
    assert validate_conversion(src).direction == "net2sbml"
    # Explicit direction is honored.
    assert validate_conversion(src, direction="net2sbml").direction == "net2sbml"


def test_unknown_suffix_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "model.bogus"
    bogus.write_text("nonsense")
    with pytest.raises(ValueError, match="cannot infer conversion direction"):
        validate_conversion(bogus)


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_conversion(tmp_path / "does_not_exist.net")


# ─── LevelResult semantics ──────────────────────────────────────────────────


def test_levelresult_ok_semantics() -> None:
    # A failing gate is not ok; a failing non-gating level is still ok.
    assert LevelResult("L1", "x", True, "pass", "").ok
    assert not LevelResult("L1", "x", True, "fail", "").ok
    assert LevelResult("L4", "x", False, "not-equal", "").ok
    assert LevelResult("L4", "x", False, "inconclusive", "").ok


# ─── CLI ────────────────────────────────────────────────────────────────────


def test_cli_pass_exit_zero(data_dir: Path, capsys: pytest.CaptureFixture) -> None:
    from bngsim.convert._cli import validate_main

    src = _require("two_species_reversible.net", data_dir)
    rc = validate_main([str(src)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out


def test_cli_json_output(data_dir: Path, capsys: pytest.CaptureFixture) -> None:
    import json

    from bngsim.convert._cli import validate_main

    src = _require("simple_decay.net", data_dir)
    rc = validate_main([str(src), "--json", "--levels", "L0,L1"])
    out = capsys.readouterr().out
    assert rc == 0
    blob = json.loads(out)
    assert {lv["level"] for lv in blob["levels"]} == {"L0", "L1"}


def test_cli_missing_file_exit_two(capsys: pytest.CaptureFixture) -> None:
    from bngsim.convert._cli import validate_main

    rc = validate_main(["/no/such/model.net"])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err
