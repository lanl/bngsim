"""Lock the AMICI-calibrated disposition mechanism (issue #380).

`amici_parity` honors only the engine-agnostic ``tol`` overrides from the shared
manifest, on purpose: rr_parity's are calibrated against RoadRunner and applying
them here could mask a genuine bngsim-vs-AMICI difference. The gap that left is
that there was no amici-calibrated path either, so a row where **AMICI** is the
wrong engine scored as a bngsim ``DIFF``. ``amici_dispositions`` is that path, and
what makes it safe rather than a fudge is the set of contracts below.

The mechanism (``apply_disposition``)
  * an INVALID_REFERENCE DIFF becomes REFERENCE_FAILED with the
    ``invalid_result`` refusal class — non-scoring, but NOT a PASS: the engines
    did not agree and the row must not claim they did
  * a COMPARISON_ARTIFACT DIFF becomes PASS, reason recorded — never silent
  * both hold ONLY over a DIFF: a row that agrees on its own is STALE, and a row
    that produced no comparison at all (raised / timed out / no oracle) is
    *inapplicable* rather than stale, because a timeout falsifies nothing
  * an entry whose recorded ``max_rel`` no longer matches the fresh one is still
    applied but flagged DRIFTED — the guard against an entry silently absorbing a
    different, real defect on the same model
  * the disposition never invents a ``reference_refusal`` for a row it did not
    reclassify, so an auto-derived class survives it

The authored entries
  * every key names a model the corpus actually builds (the staleness check
    rr_parity does at manifest-build time; this module is read at run time, so it
    is checked here instead)
  * every entry carries a reason and an issue reference, and no key is authored
    in both dicts — a row cannot be both "the reference is wrong" and "neither
    engine is wrong"
  * the three #325 rows this issue exists to record are present, with the
    disposition each was triaged to

The renderer
  * a REFERENCE_FAILED reached by disposition reads ``REF INVALID``, not the same
    gray badge as an AMICI crash — the two are materially different facts
"""

from __future__ import annotations

import json
from pathlib import Path

import amici_dispositions as ad
import generate_amici_sens_matrix as gsm
import pytest
from _core import CLEAN, Outcome, read_manifest

_PC = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# disposition_for / dispositions_for_regime
# --------------------------------------------------------------------------- #
def _single_regime_keys() -> list[str]:
    """Authored keys whose model is authored on exactly ONE regime.

    A model may legitimately be authored on both — MODEL0910846879's defect is in
    the state trajectory, so it disposes of the ``ode`` row and the ``sens`` row
    that is downstream of it (issue #382). Those are the wrong witnesses for the
    leakage check below, which is about a disposition NOT reaching a regime it was
    not authored for.
    """
    per_model: dict[str, set[str]] = {}
    for key in ad.ALL_KEYS:
        model_id, _, regime = key.rpartition(":")
        per_model.setdefault(model_id, set()).add(regime)
    return [f"{m}:{next(iter(r))}" for m, r in per_model.items() if len(r) == 1]


def test_lookup_is_regime_scoped_and_empty_for_unknown_models():
    assert ad.disposition_for("BIOMD9999999999", "sens") is None
    # A sens disposition must not leak onto the same model's ODE job: the ODE and
    # sensitivity jobs compare different quantities, and an AMICI defect in one
    # says nothing about the other.
    single = _single_regime_keys()
    assert single, "no single-regime entry left to test leakage with"
    for key in single:
        model_id, regime = key.rsplit(":", 1)
        assert ad.disposition_for(model_id, regime) is not None
        other = "ode" if regime == "sens" else "sens"
        assert ad.disposition_for(model_id, other) is None, key


def test_a_model_authored_on_both_regimes_gets_each_regimes_own_entry():
    """The other side of the scoping rule: authoring both is allowed, and the two
    must not collapse into one. MODEL0910846879's ``ode`` entry carries the closed
    form; its ``sens`` entry points at that one rather than restating it, so
    returning the wrong regime's entry would silently swap the evidence."""
    both = [
        m
        for m in {k.rpartition(":")[0] for k in ad.ALL_KEYS}
        if ad.disposition_for(m, "ode") and ad.disposition_for(m, "sens")
    ]
    assert "MODEL0910846879" in both
    for model_id in both:
        assert ad.disposition_for(model_id, "ode") != ad.disposition_for(model_id, "sens")


def test_a_key_authored_in_both_dicts_raises_rather_than_picking_one(monkeypatch):
    """The two dispositions make contradictory claims about who is at fault, so a
    key in both is an authoring error, not a precedence question."""
    key = next(iter(ad.INVALID_REFERENCE))
    model_id, regime = key.rsplit(":", 1)
    monkeypatch.setitem(ad.COMPARISON_ARTIFACT, key, {"issue": None, "reason": "test"})
    with pytest.raises(ValueError, match="never both"):
        ad.disposition_for(model_id, regime)


def test_regime_map_is_keyed_by_bare_model_id():
    """What the runners hold per result row is a model id, not a composite key.

    Membership is decided per *model*, not per key: a model authored on both
    regimes appears in the sens map because one of its keys is a sens key, and
    reading that off a single key would call it a leak.
    """
    m = ad.dispositions_for_regime("sens")
    sens_models = {k.rpartition(":")[0] for k in ad.ALL_KEYS if k.endswith(":sens")}
    for key in ad.ALL_KEYS:
        model_id = key.rpartition(":")[0]
        assert (model_id in m) is (model_id in sens_models), key
    assert all(":" not in k for k in m)


# --------------------------------------------------------------------------- #
# apply_disposition — the reclassifier
# --------------------------------------------------------------------------- #
_INVALID = {"kind": "invalid_reference", "reason": "AMICI is inert", "issue": "GH #380"}
_ARTIFACT = {"kind": "comparison_artifact", "reason": "edge cell", "issue": "GH #380"}


def test_invalid_reference_diff_becomes_a_non_scoring_reference_failed():
    outcome, comment, refusal, state = ad.apply_disposition(_INVALID, Outcome.DIFF, 1.0, "20 sp")
    assert outcome == Outcome.REFERENCE_FAILED
    assert outcome in CLEAN  # non-scoring
    assert refusal == ad.INVALID_RESULT_REFUSAL
    assert state == "applied"
    # The reason and the natural verdict both survive onto the row: a reader must
    # be able to see WHY it is non-scoring and WHAT it was.
    assert "AMICI is inert" in comment and "GH #380" in comment
    assert "was DIFF at max_rel=1" in comment
    assert "20 sp" in comment


def test_invalid_reference_is_not_reclassified_to_pass():
    """The whole point of choosing REFERENCE_FAILED: PASS would assert the two
    engines agreed, and they did not. This is the mirror of #319/#323 — name the
    gap, do not claim the win."""
    outcome, _c, _r, _s = ad.apply_disposition(_INVALID, Outcome.DIFF, 1.0, "")
    assert outcome != Outcome.PASS


def test_comparison_artifact_diff_becomes_a_pass_with_its_reason():
    outcome, comment, refusal, state = ad.apply_disposition(_ARTIFACT, Outcome.DIFF, 0.17, "")
    assert outcome == Outcome.PASS
    assert state == "applied"
    # No refusal invented: this row has a usable oracle, it just could not resolve
    # one cell.
    assert refusal is None
    assert "edge cell" in comment


def test_a_row_that_agrees_on_its_own_is_stale_not_silently_kept():
    for disp in (_INVALID, _ARTIFACT):
        outcome, comment, refusal, state = ad.apply_disposition(disp, Outcome.PASS, 0.0, "")
        assert (outcome, state, refusal) == (Outcome.PASS, "stale", None)
        assert "STALE" in comment and "prune" in comment


@pytest.mark.parametrize(
    "outcome",
    [
        Outcome.EXCEPTION,
        Outcome.TIMEOUT,
        Outcome.REFERENCE_FAILED,
        Outcome.BAD_TEST,
        Outcome.UNSUPPORTED,
    ],
)
def test_a_row_with_no_comparison_is_inapplicable_not_stale(outcome):
    """Staleness means the premise was falsified. A row that raised or timed out
    produced no comparison at all, so it falsifies nothing — telling the author to
    prune a still-valid entry because AMICI happened to time out would be wrong."""
    got, comment, refusal, state = ad.apply_disposition(_INVALID, outcome, None, "")
    assert (got, state, refusal) == (outcome, "inapplicable", None)
    assert "STALE" not in comment
    assert "did not apply" in comment


def test_an_inapplicable_row_keeps_its_auto_derived_refusal():
    """A REFERENCE_FAILED row already carries a refusal class derived from AMICI's
    exception. The disposition returns None so the caller leaves it alone —
    otherwise a genuine ``compile`` failure would be relabelled ``invalid_result``
    and read as a settled, evidence-backed attribution it is not."""
    _o, _c, refusal, _s = ad.apply_disposition(_INVALID, Outcome.REFERENCE_FAILED, None, "")
    assert refusal is None


# --------------------------------------------------------------------------- #
# The drift guard
# --------------------------------------------------------------------------- #
class TestDrift:
    """An entry attributes ONE observed disagreement. If the fresh metric no
    longer looks like the recorded one, the attribution is no longer known to
    cover what the row is now doing — the row is still disposed of (a red DIFF on
    a model with a documented AMICI defect is not an improvement) but the comment
    says so, so a re-triage is possible."""

    def test_a_matching_metric_says_nothing_extra(self):
        disp = {**_INVALID, "max_rel": 1.0}
        _o, comment, _r, _s = ad.apply_disposition(disp, Outcome.DIFF, 1.0, "")
        assert "DRIFTED" not in comment

    def test_a_metric_that_moved_by_more_than_2x_is_flagged_but_still_applied(self):
        disp = {**_INVALID, "max_rel": 1.0}
        outcome, comment, _r, state = ad.apply_disposition(disp, Outcome.DIFF, 0.1, "")
        assert (outcome, state) == (Outcome.REFERENCE_FAILED, "applied")
        assert "DRIFTED" in comment

    def test_wobble_inside_the_factor_of_two_band_is_not_drift(self):
        # These metrics move with integration tolerance; a tight band would cry
        # wolf on every re-run.
        disp = {**_ARTIFACT, "max_rel": 0.1746}
        for fresh in (0.1, 0.3):
            _o, comment, _r, _s = ad.apply_disposition(disp, Outcome.DIFF, fresh, "")
            assert "DRIFTED" not in comment

    def test_an_entry_with_no_recorded_metric_never_drifts(self):
        _o, comment, _r, _s = ad.apply_disposition(_INVALID, Outcome.DIFF, 0.5, "")
        assert "DRIFTED" not in comment

    @pytest.mark.parametrize(
        "recorded,fresh,drifted",
        [
            (1.0, float("inf"), True),  # finite -> non-finite is a change of KIND
            (1.0, float("nan"), True),
            (float("inf"), 1.0, True),
            (float("inf"), float("inf"), False),
            (1.0, 0.0, True),  # a zero on one side only is not a matter of degree
            (0.0, 0.0, False),
        ],
    )
    def test_non_finite_and_zero_crossings(self, recorded, fresh, drifted):
        assert ad._drifted(recorded, fresh) is drifted


# --------------------------------------------------------------------------- #
# The authored entries
# --------------------------------------------------------------------------- #
def _corpus_model_ids() -> set[str]:
    _meta, jobs = read_manifest(_PC / "rr_parity" / "ode_jobs.json")
    return {j.model_id for j in jobs}


def test_every_authored_key_names_a_model_the_corpus_builds():
    """rr_parity catches a stale override when it builds its manifest. These are
    read at run time and never baked in, so nothing would catch a typo'd or
    retired model id but this."""
    ids = _corpus_model_ids()
    for regime in ad.REGIMES:
        assert ad.stale_keys(ids, regime) == []


def test_stale_keys_actually_flags_an_absent_model():
    assert ad.stale_keys(set(), "sens") == sorted(k for k in ad.ALL_KEYS if k.endswith(":sens"))


def test_every_entry_cites_a_reason_and_an_issue():
    """A disposition with no rationale is a silent fudge, and one with no tracker
    reference cannot be re-litigated when the evidence changes."""
    for src in (ad.INVALID_REFERENCE, ad.COMPARISON_ARTIFACT):
        for key, entry in src.items():
            assert entry.get("reason", "").strip(), key
            assert entry.get("issue"), key


def test_the_triaged_rows_from_325_are_recorded_on_the_right_disposition():
    """The #380 rows that are still DIFFs, each on the disposition its triage
    reached: the AMICI defect that has no saltation term, and the row where
    neither engine is wrong."""
    assert "MODEL2105110001:sens" in ad.INVALID_REFERENCE
    assert "BIOMD0000000117:sens" in ad.COMPARISON_ARTIFACT


def test_the_row_383_fixed_is_not_authored():
    """#380's third row, MODEL1607210000, PASSes at max_rel=0 since #383 seeded the
    initialAssignment initial-condition term. An entry for it would be dead weight
    the runner could only flag STALE, so the retirement is recorded as a NOTE in
    the module instead — the same treatment rr_parity gives a retired override."""
    assert "MODEL1607210000:sens" not in ad.ALL_KEYS


# The oracles an invalid-reference claim may rest on. RoadRunner is the usual
# one — a third engine sharing no code with either party. A CLOSED FORM is the
# other, and it is strictly stronger: it is not an engine's opinion at all, so
# there is no third implementation left to be wrong. Both are independent of
# bngsim, which is the property this list exists to enforce; what is excluded is
# an entry whose only support is bngsim's own answer.
_INDEPENDENT_ORACLES = ("RoadRunner", "CLOSED FORM")


def test_the_evidence_names_an_oracle_that_is_not_bngsim():
    """bngsim must never adjudicate its own case: an invalid-reference claim is
    only as good as an oracle that is not bngsim.

    Widened from "names RoadRunner" when MODEL0910846879 was authored against the
    model's closed form (issue #382). That model has no reactions and one state,
    so it reduces exactly and there is nothing for a third engine to add — but
    the rule the RoadRunner check stood for is unchanged, and a claim citing
    neither still fails here."""
    for key, entry in ad.INVALID_REFERENCE.items():
        reason = entry["reason"].lower()
        assert any(o.lower() in reason for o in _INDEPENDENT_ORACLES), key


def test_the_closed_form_that_disposes_of_MODEL0910846879_still_holds():
    """The evidence itself, re-derived rather than quoted.

    An INVALID_REFERENCE entry is only as good as its oracle, and a reason string
    is just a claim about one. This recomputes MODEL0910846879's closed form from
    the SBML values and checks bngsim against it, so the entry cannot outlive the
    fact it rests on: if bngsim ever stops matching, this fails and the
    disposition must be re-litigated rather than quietly keeping a real bngsim
    defect non-scoring.

    The model has no reactions and one state (``TVD``, a rateRule target), and
    every assignment rule reduces over constants, so the whole trajectory is
    ``TVZ + (TVD0 - TVZ)*exp(-t/TVDDL)``. Deliberately NOT compared against AMICI
    here: this test is about the oracle, and the AMICI half of the evidence is
    that it returns ``TVD0*exp(-t/30)`` instead, which is this same form with
    ``TVZ = 0``.
    """
    np = pytest.importorskip("numpy")
    bngsim = pytest.importorskip("bngsim")
    path = _PC / "rr_parity" / "models" / "MODEL0910846879" / "MODEL0910846879.xml"
    if not path.is_file():
        pytest.skip("rr_parity corpus model MODEL0910846879 not present")

    # Straight from the <listOfParameters> of the model file.
    ADHC, ANM, POT, Z10, Z11 = 1.0, 0.987545, 35.1148, 45.0, 0.01
    ANMSLT, AHTHM, ANMTM, DR, TVDDL = 2.0, 2.0, 1.5, 0.0, 30.0
    TVD0 = 0.000980838

    ANMSML = (ANM - 1.0) * ANMSLT + 1.0
    STH1 = (Z10 - POT) ** 2.0 * Z11 * ANMSML
    STH = 0.8 if STH1 < 0.8 else (8.0 if STH1 > 8.0 else STH1)
    AHCM = (ADHC - 1.0) * AHTHM + 1.0
    ANMTH = (ANM - 1.0) * ANMTM * 0.001
    AHTH1 = AHCM * STH * 0.001
    AHTH = 0.0 if AHTH1 < 0.0 else AHTH1
    TVZ1 = ANMTH + AHTH
    TVZ = 0.0 if TVZ1 < 0.0 else TVZ1

    # The piecewise AMICI holds at 0: its operand is positive, so the otherwise
    # branch is the one that is live, for the whole run.
    assert TVZ1 > 0.0
    assert TVZ == TVZ1

    # dTVD/dt = (TVZ + DR - TVD)/TVDDL relaxes to TVZ + DR with time constant
    # TVDDL. DR is 0 in this model, but it is carried rather than dropped so the
    # form stays the model's and not a simplification of it.
    steady = TVZ + DR

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_sbml(str(path))
        run = bngsim.Simulator(model, method="ode").run((0.0, 100.0), 101, rtol=1e-10, atol=1e-14)
    t = np.asarray(run.time)
    got = np.asarray(run.species)[:, 0]
    exact = steady + (TVD0 - steady) * np.exp(-t / TVDDL)
    assert np.max(np.abs(got - exact) / np.abs(exact)) < 1e-7

    # And the trajectory AMICI actually returns is the SAME form with TVZ = 0,
    # which is what makes "AMICI holds TVZ at 0" a measurement and not a guess.
    amici_shape = DR + (TVD0 - DR) * np.exp(-t / TVDDL)
    assert amici_shape[-1] == pytest.approx(3.499040825e-5, rel=1e-9)
    assert np.max(np.abs(got - amici_shape) / np.abs(got)) > 0.9


def test_an_entry_supported_only_by_bngsim_is_rejected(monkeypatch):
    """The guard has teeth: it is the *citation* that is required, not a
    formality. An entry whose reason rests on bngsim agreeing with itself names
    none of the accepted oracles and must fail."""
    monkeypatch.setattr(
        ad,
        "INVALID_REFERENCE",
        {"FAKE:sens": {"issue": "GH #0", "max_rel": 1.0, "reason": "bngsim disagrees with it."}},
    )
    with pytest.raises(AssertionError):
        test_the_evidence_names_an_oracle_that_is_not_bngsim()


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
def test_a_disposed_reference_failed_reads_as_ref_invalid():
    """AMICI answering wrongly and AMICI not answering at all are different facts;
    the shared gray REFERENCE_FAILED badge flattens them."""
    cls, badge = gsm.sens_row_class("REFERENCE_FAILED", {}, ad.INVALID_RESULT_REFUSAL)
    assert badge == "REF INVALID"
    assert cls == gsm.sens_row_class("REFERENCE_FAILED", {})[0]  # same gray styling


def test_an_amici_crash_still_reads_as_reference_failed():
    assert gsm.sens_row_class("REFERENCE_FAILED", {}, "compile")[1] == "REFERENCE_FAILED"
    assert gsm.sens_row_class("REFERENCE_FAILED", {}, None)[1] == "REFERENCE_FAILED"


def test_the_page_counts_and_explains_the_disposed_rows(tmp_path):
    report = tmp_path / "report_sens.json"
    report.write_text(
        json.dumps(
            {
                "_meta": {"tally": {"REFERENCE_FAILED": 1}, "n_models": 1},
                "results": [
                    {
                        "model_id": "MODEL1607210000",
                        "method": "sens/staggered",
                        "outcome": "REFERENCE_FAILED",
                        "reference_refusal": ad.INVALID_RESULT_REFUSAL,
                        "comment": "invalid reference (GH #380): 30 of 31 inert",
                        "extra": {"sens_method": "staggered", "n_params": 20},
                    }
                ],
            }
        )
    )
    out = tmp_path / "m.html"
    gsm.generate_html(report, out)
    html = out.read_text()
    assert "REF INVALID" in html
    assert "ref invalid" in html  # the summary card
    assert "30 of 31 inert" in html  # the evidence reaches the page
