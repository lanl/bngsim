"""Lock the rr_parity per-model override mechanism (`overrides.py` + the
runner's `_job_overrides` reader).

These are pure plumbing — config-shaped, no math — so the contracts are about
faithful round-tripping of the authored allow-list into the runner:

overrides_for
  * a seeded KNOWN_ARTIFACT key yields a known_artifact record carrying its
    reason; an unknown key yields nothing
  * a TOL_OVERRIDES key yields a tol record applied to both engines; when both
    a tol and a known_artifact exist for one key the tol is emitted first
    (apply-before-run vs reclassify-after-run ordering)
  * the key is regime-scoped: an ode override does not leak onto the ssa job

stale_keys
  * flags an authored key whose model is absent from the built set, and stays
    silent when every key is matched

_job_overrides (runner side)
  * reads a tol Override into a {rtol, atol} dict, and a known_artifact /
    invalid_reference Override each into a (reason, issue) pair; a job with no
    overrides yields (None, None, None) — so a tol override actually reaches the
    integrator and a reclassification override actually reaches its reclassifier
"""

from __future__ import annotations

import overrides as ov
from _core import Job, Oracle, Override
from rr_run import _job_overrides


# --------------------------------------------------------------------------- #
# overrides_for
# --------------------------------------------------------------------------- #
def test_seeded_known_artifact_round_trips():
    """A committed KNOWN_ARTIFACT key must surface as a known_artifact record
    carrying its reason (the entry the runner reclassifies DIFF→PASS). Keyed off
    a live dict entry so it survives churn like the #23 BIOMD2-ssa retirement."""
    key = next(iter(ov.KNOWN_ARTIFACT))
    model_id, method = key.rsplit(":", 1)
    recs = ov.overrides_for(model_id, method)
    arts = [r for r in recs if r["field"] == "known_artifact"]
    assert len(arts) == 1
    assert arts[0]["reason"]  # non-empty rationale
    assert "value" in arts[0] and "issue" in arts[0]["value"]


def test_unknown_key_has_no_overrides():
    assert ov.overrides_for("BIOMD9999999999", "ode") == []
    # Overrides are regime-scoped: a seeded KNOWN_ARTIFACT entry must not leak
    # onto the same model's other-regime job.
    key = next(iter(ov.KNOWN_ARTIFACT))
    model_id, method = key.rsplit(":", 1)
    other = "ssa" if method == "ode" else "ode"
    assert ov.overrides_for(model_id, other) == []


def test_seeded_invalid_reference_round_trips():
    """The committed MODEL2002070001 ODE entry (RR runs but non-finite) must
    surface as an invalid_reference record carrying its reason — the entry the
    runner reclassifies to BAD_TEST while the premise holds."""
    recs = ov.overrides_for("MODEL2002070001", "ode")
    inval = [r for r in recs if r["field"] == "invalid_reference"]
    assert len(inval) == 1
    assert inval[0]["reason"]  # non-empty rationale
    assert "value" in inval[0] and "issue" in inval[0]["value"]


def test_tol_override_emitted_first_and_for_both_engines(monkeypatch):
    """A TOL_OVERRIDES entry yields a tol record with rtol/atol; if a key also
    has a known_artifact, the tol record precedes it (apply-before-run, then
    reclassify-after-run). Injected onto a live known_artifact key so the test
    does not depend on whether a real tol+artifact co-occurrence has been triaged
    yet."""
    key = next(iter(ov.KNOWN_ARTIFACT))  # a key that already carries a known_artifact
    model_id, method = key.rsplit(":", 1)
    monkeypatch.setitem(
        ov.TOL_OVERRIDES,
        key,
        {"rtol": 1e-12, "atol": 1e-12, "reason": "test: ill-conditioned"},
    )
    recs = ov.overrides_for(model_id, method)
    fields = [r["field"] for r in recs]
    assert fields == ["tol", "known_artifact"]  # tol first
    tol = recs[0]
    assert tol["value"] == {"rtol": 1e-12, "atol": 1e-12}
    assert tol["reason"]


# --------------------------------------------------------------------------- #
# stale_keys
# --------------------------------------------------------------------------- #
def test_stale_keys_flags_unmatched_and_stays_silent_when_matched():
    """An authored key is stale iff its model is absent from the built set for
    that regime; a key never counts against the other regime. (The BIOMD2 ssa
    entry that used to seed this test was retired in #23, leaving only :ode
    override keys.)"""
    ode_keys = sorted(k for k in ov.ALL_KEYS if k.endswith(":ode"))
    assert ode_keys, "expected at least one :ode override key"
    example = ode_keys[0]
    model_id = example.rsplit(":", 1)[0]
    # model present in the built set → not stale
    assert example not in ov.stale_keys({model_id}, "ode")
    # model absent → flagged
    assert example in ov.stale_keys({"X", "Y"}, "ode")
    # wrong regime: an ode key never counts as an ssa stale key
    assert example not in ov.stale_keys({"X"}, "ssa")
    # every ode override model matched → silent
    ode_models = {k.rsplit(":", 1)[0] for k in ov.ALL_KEYS if k.endswith(":ode")}
    assert ov.stale_keys(ode_models, "ode") == []


# --------------------------------------------------------------------------- #
# _job_overrides (runner reader)
# --------------------------------------------------------------------------- #
def _job(overrides):
    return Job(
        model_id="M",
        input_format="sbml",
        method="ode",
        reference_engine="roadrunner",
        model="models/M/M.xml",
        oracle=Oracle(metric="max_rel_err", tol=1e-4),
        params={},
        overrides=overrides,
    )


def test_job_overrides_reads_tol_and_artifact():
    job = _job(
        [
            Override(field="tol", value={"rtol": 1e-11, "atol": 1e-13}, reason="ill-cond"),
            Override(field="known_artifact", value={"issue": "GH-123"}, reason="RR bug"),
        ]
    )
    tol, artifact, invalid_ref, adjudication = _job_overrides(job)
    assert tol == {"rtol": 1e-11, "atol": 1e-13}
    assert artifact == ("RR bug", "GH-123")
    assert invalid_ref is None
    assert adjudication is None


def test_job_overrides_empty_is_all_none():
    assert _job_overrides(_job([])) == (None, None, None, None)


def test_job_overrides_artifact_without_issue():
    job = _job([Override(field="known_artifact", value={"issue": None}, reason="degenerate")])
    tol, artifact, invalid_ref, adjudication = _job_overrides(job)
    assert tol is None
    assert artifact == ("degenerate", None)
    assert invalid_ref is None
    assert adjudication is None


def test_job_overrides_reads_invalid_reference():
    job = _job(
        [Override(field="invalid_reference", value={"issue": "GH-9"}, reason="RR non-finite")]
    )
    tol, artifact, invalid_ref, adjudication = _job_overrides(job)
    assert tol is None and artifact is None
    assert invalid_ref == ("RR non-finite", "GH-9")
    assert adjudication is None


def test_job_overrides_reads_no_oracle_adjudicated():
    job = _job(
        [
            Override(
                field="no_oracle_adjudicated",
                value={"issue": "GH #117", "verdict": "confirm"},
                reason="scipy BDF reproduces to max_rel=0",
            )
        ]
    )
    tol, artifact, invalid_ref, adjudication = _job_overrides(job)
    assert tol is None and artifact is None and invalid_ref is None
    assert adjudication == ("confirm", "scipy BDF reproduces to max_rel=0", "GH #117")
