"""GH #183: known reference-side block_same_complex_binding NF divergences.

A DIFF on a catalogued NF ring-former (the legacy NFsim v1.14.3 has no ``-bscb``)
is a reference-side bug, not a bngsim defect. The stochastic matrix keeps the honest
DIFF verdict but tags the row ``subclass="ref_bscb"`` so it renders KNOWN REF
(non-scoring) instead of being read as a bngsim bug. These tests pin that tagging and
its rendering without running any engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import bng_stoch_run as bsr  # noqa: E402
import generate_bng_matrix as gbm  # noqa: E402


def test_egfr_is_catalogued_with_issue_and_reason():
    art = bsr.NF_KNOWN_REF.get("EGFR_oligo_v2")
    assert art is not None, "EGFR_oligo_v2 must be a known reference-side bscb divergence"
    assert art["issue"] == "internal#183"
    assert "bscb" in art["reason"] or "-bscb" in art["reason"]


def test_annotate_tags_known_nf_diff_keeps_verdict():
    res = {"comment": "frac_pass 0.93", "status": "diff"}
    out = bsr.annotate_ref_bscb(res, "diff", "nf", "EGFR_oligo_v2")
    # Verdict is unchanged — still an honest DIFF.
    assert out["status"] == "diff"
    # But sub-classified non-scoring (the "KNOWN REF" label is rendered from this
    # subclass, not spelled into the comment), with a plain-English explanation that
    # attributes the difference to the legacy engine, and the original comment preserved.
    assert out["subclass"] == "ref_bscb"
    assert "NFsim" in out["comment"] and "bngsim is correct" in out["comment"]
    assert "frac_pass 0.93" in out["comment"]  # original comment preserved


def test_annotate_is_noop_for_uncatalogued_or_wrong_track():
    # Uncatalogued model: untouched.
    r1 = bsr.annotate_ref_bscb({"comment": "c"}, "diff", "nf", "some_other_model")
    assert "subclass" not in r1
    # Catalogued model but SSA track (bscb is NF-only): untouched.
    r2 = bsr.annotate_ref_bscb({"comment": "c"}, "diff", "ssa", "EGFR_oligo_v2")
    assert "subclass" not in r2
    # Catalogued NF model but a PASS (not a DIFF): untouched.
    r3 = bsr.annotate_ref_bscb({"comment": "c"}, "pass", "nf", "EGFR_oligo_v2")
    assert "subclass" not in r3


def test_classify_row_stoch_renders_known_ref_distinctly():
    # ref_bscb DIFF -> KNOWN REF in the TRIAGED (yellow) bucket, NOT FAILED (red) --
    # a DIFF explained away from bngsim, grouped with the other attributed-away cases
    # (matches the docstring/legend and the rr_parity/amici sibling matrices).
    cls, cat = gbm.classify_row_stoch("DIFF", "ref_bscb")
    assert (cls, cat) == ("status-triaged", "KNOWN REF")
    # A plain DIFF is still FAILED — the change is scoped to the tagged subclass.
    assert gbm.classify_row_stoch("DIFF", None) == ("status-failed", "FAILED")
    assert gbm.classify_row_stoch("DIFF") == ("status-failed", "FAILED")
    # An unrelated subclass on a DIFF does not get the KNOWN REF treatment.
    assert gbm.classify_row_stoch("DIFF", "something_else") == ("status-failed", "FAILED")
