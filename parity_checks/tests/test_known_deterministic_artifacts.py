"""The ODE known-artifact gate reports the number it judged by (GH #69, #872).

``annotate_known_artifact`` excuses a DIFF on a catalogued model while its
``max_abs`` stays within the entry's ``max_abs_bound``. After #996 the CI
nightlies dropped ``proliferation``'s tag on both ODE arms, and nothing recorded
how far past its bound of 1.0 the model had gone: the report keeps ``max_rel``,
not ``max_abs``. The row comment now states this run's ``max_abs`` and the bound
on both paths, near its start, so the nightly's 200-character verdict note keeps
them. These tests pin that without running any engine.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import verdicts as V

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import bng_ode_run as bor  # noqa: E402

RAW = "Differs from run_network beyond tolerance at 211/30003 points (3 species), full trajectory."
STEM = "proliferation"
BOUND = bor.KNOWN_DETERMINISTIC_ARTIFACTS[STEM]["max_abs_bound"]


def _annotate(max_abs, status="diff", stem=STEM):
    res = {"comment": RAW}
    return bor.annotate_known_artifact(res, status, stem, max_abs)


def test_within_the_bound_is_excused_and_says_by_how_much():
    out = _annotate(0.517)
    assert out["subclass"] == "known_artifact"
    note = V._note(out["comment"])
    assert "max_abs 0.517" in note and f"bound {BOUND:g}" in note
    assert RAW in out["comment"]


def test_past_the_bound_scores_and_says_by_how_much():
    over = BOUND * 1.37
    out = _annotate(over)
    assert "subclass" not in out
    note = V._note(out["comment"])
    assert f"max_abs {over:.3g}" in note and f"bound {BOUND:g}" in note
    assert RAW in out["comment"]


def test_the_bound_itself_is_excused():
    assert _annotate(BOUND)["subclass"] == "known_artifact"


@pytest.mark.parametrize("bad", [None, math.inf, math.nan])
def test_a_missing_or_non_finite_divergence_is_never_excused(bad):
    out = _annotate(bad)
    assert "subclass" not in out
    assert "max_abs" in V._note(out["comment"])


@pytest.mark.parametrize(
    ("status", "stem"),
    [("pass", STEM), ("exception", STEM), ("diff", "not_a_catalogued_model")],
)
def test_other_rows_are_left_alone(status, stem):
    out = _annotate(0.5, status=status, stem=stem)
    assert out == {"comment": RAW}
