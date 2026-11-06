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
  * a NO_GROUND_TRUTH DIFF *or* PASS becomes BAD_TEST (issue #1542) — never
    REFERENCE_FAILED, which the ODE matrix draws as a bngsim PASS, and never
    STALE, because the entry asserts a property of the model; a row with no
    comparison is inapplicable to it as to the others, so a bngsim EXCEPTION
    stays a visible, scoring row
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
    in two dicts — a row cannot be two of "the reference is wrong", "neither
    engine is wrong" and "there is no answer to be right about"
  * a NO_GROUND_TRUTH entry cites an oracle that is neither engine, and
    MODEL1112050001's closed form is re-derived from the SBML text
  * the three #325 rows this issue exists to record are present, with the
    disposition each was triaged to

The renderers
  * a REFERENCE_FAILED reached by disposition reads ``REF INVALID``, not the same
    gray badge as an AMICI crash — the two are materially different facts
  * a BAD_TEST reached by NO_GROUND_TRUTH reads ``NO TRUTH`` on the ODE matrix,
    not as two engines that refused the model
"""

from __future__ import annotations

import json
from pathlib import Path

import amici_dispositions as ad
import generate_amici_matrix as gam
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


@pytest.mark.parametrize("other", ["COMPARISON_ARTIFACT", "NO_GROUND_TRUTH"])
def test_a_key_authored_in_both_dicts_raises_rather_than_picking_one(monkeypatch, other):
    """The dispositions make contradictory claims about who is at fault, and
    whether there is an answer at all, so a key in two of them is an authoring
    error, not a precedence question."""
    key = next(iter(ad.INVALID_REFERENCE))
    model_id, regime = key.rsplit(":", 1)
    monkeypatch.setitem(getattr(ad, other), key, {"issue": None, "reason": "test"})
    with pytest.raises(ValueError, match="one disposition"):
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
_NO_TRUTH = {"kind": "no_ground_truth", "reason": "x9 leaves the doubles", "issue": "GH #1542"}


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
    prune a still-valid entry because AMICI happened to time out would be wrong.

    NO_GROUND_TRUTH included: it disposes of a comparison and nothing else, so a
    bngsim EXCEPTION on such a model keeps scoring."""
    for disp in (_INVALID, _ARTIFACT, _NO_TRUTH):
        got, comment, refusal, state = ad.apply_disposition(disp, outcome, None, "")
        assert (got, state, refusal) == (outcome, "inapplicable", None)
        assert "STALE" not in comment
        assert "did not apply" in comment


@pytest.mark.parametrize("natural,value", [(Outcome.DIFF, 1.0), (Outcome.PASS, 0.0)])
def test_no_ground_truth_disposes_of_a_diff_and_a_pass_alike(natural, value):
    """On a model with no solution, a DIFF is two non-solutions that rounded apart
    and a PASS is two that rounded alike. Neither says anything about bngsim, so
    both become the same non-scoring BAD_TEST, with the reason and the natural
    verdict on the row."""
    outcome, comment, refusal, state = ad.apply_disposition(_NO_TRUTH, natural, value, "29 sp")
    assert outcome == Outcome.BAD_TEST
    assert outcome in CLEAN  # non-scoring
    assert (refusal, state) == (None, "applied")
    assert comment.startswith(ad.NO_GROUND_TRUTH_LABEL)
    assert "x9 leaves the doubles" in comment and "GH #1542" in comment
    assert f"was {natural} at max_rel={value:.4g}" in comment
    assert "29 sp" in comment


def test_no_ground_truth_is_neither_reference_failed_nor_pass():
    """REFERENCE_FAILED says AMICI could not anchor a run bngsim made, and the ODE
    matrix draws it as a bngsim PASS; PASS says the engines agreed about the
    model. bngsim's run is no more a solution than AMICI's, so neither fits."""
    for natural in (Outcome.DIFF, Outcome.PASS):
        outcome, _c, _r, _s = ad.apply_disposition(_NO_TRUTH, natural, 1.0, "")
        assert outcome not in (Outcome.REFERENCE_FAILED, Outcome.PASS)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_a_no_ground_truth_entry_leaves_a_non_finite_diff_scoring(value):
    """A non-finite max_rel is a time-grid or species mismatch, or non-finite
    output, which a loader regression on the model would produce. It is not two
    non-solutions rounding apart, so it stays a scoring DIFF that says why."""
    outcome, comment, refusal, state = ad.apply_disposition(_NO_TRUTH, Outcome.DIFF, value, "")
    assert (outcome, refusal, state) == (Outcome.DIFF, None, "inapplicable")
    assert "did not apply" in comment and "re-triage" in comment


def test_a_no_ground_truth_pass_is_applied_not_stale():
    """What the entry asserts is a property of the model. A PASS cannot falsify
    it, so it must not ask the author to prune the entry."""
    _o, comment, _r, state = ad.apply_disposition(_NO_TRUTH, Outcome.PASS, 0.0, "")
    assert state == "applied"
    assert "STALE" not in comment and "prune" not in comment


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
    for src in (ad.INVALID_REFERENCE, ad.COMPARISON_ARTIFACT, ad.NO_GROUND_TRUTH):
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


# What a no-ground-truth claim may rest on: not bngsim and not AMICI, and since
# both run CVODES, not RoadRunner's CVODE either. A CLOSED FORM, or an integrator
# that shares no code with SUNDIALS (scipy's) driven where it tracks the solution.
_GROUND_TRUTH_ORACLES = ("CLOSED FORM", "scipy")


def test_a_no_ground_truth_entry_cites_an_oracle_that_is_neither_engine():
    """ "Neither engine has an answer" cannot be shown by either engine alone."""
    for key, entry in ad.NO_GROUND_TRUTH.items():
        reason = entry["reason"].lower()
        assert any(o.lower() in reason for o in _GROUND_TRUTH_ORACLES), key


def test_a_no_ground_truth_entry_resting_on_the_engines_is_rejected(monkeypatch):
    monkeypatch.setattr(
        ad,
        "NO_GROUND_TRUTH",
        {"FAKE:ode": {"issue": "GH #0", "reason": "bngsim and AMICI and RoadRunner all fail."}},
    )
    with pytest.raises(AssertionError):
        test_a_no_ground_truth_entry_cites_an_oracle_that_is_neither_engine()


def test_the_closed_form_that_leaves_MODEL1112050001_without_a_solution_still_holds():
    """The NO_GROUND_TRUTH evidence for MODEL1112050001, re-derived (issue #1542).

    From the SBML text: ``x8`` drains at the constant ``k6*PTP*k8``, which reaches
    ``x4 + x5`` through ``k20*x8``; ``x9`` grows at ``-k7*(x4+x5)/IRp``; and
    ``x9 + x10 + x12`` and ``x11 + x12`` are conserved. Together these give
    ``x9 = x9(0)*exp(A*t^3)`` early, with ``A = k7*k20*k6*PTP*k8/(6*IRp)``, which
    leaves the doubles long before the job's ``t_end``.

    ``x9`` alone would pass the largest double near t = 1.1e-6; the model gets
    there sooner, through ``k9``'s ``x12/PI3K`` with ``PI3K = 5e-15``, which
    drives ``x13`` and ``x14`` past 1e10 by t = 3.363e-7.

    The numbers are then checked against bngsim twice, as the MODEL0910846879
    test does against its closed form. With atol far below ``x9``, bngsim must
    follow the law, so the law is the model's and not a misreading of it. At the
    suite's tolerance, bngsim must hold ``x9`` on its repelling equilibrium,
    which is the entry's account of what the engines return. If bngsim ever
    stops doing that (it follows ``x9`` and fails, say), this fails and the entry
    must be re-litigated.
    """
    import math
    import re
    import sys
    import warnings

    np = pytest.importorskip("numpy")
    sympy = pytest.importorskip("sympy")
    libsbml = pytest.importorskip("libsbml")
    bngsim = pytest.importorskip("bngsim")
    import _amici_common as ac

    path = _PC / "rr_parity" / "models" / "MODEL1112050001" / "MODEL1112050001.xml"
    if not path.is_file():
        pytest.skip("rr_parity corpus model MODEL1112050001 not present")
    model = libsbml.readSBMLFromFile(str(path)).getModel()

    def expr(math_ast):
        text = libsbml.formulaToL3String(math_ast).replace("^", "**")
        names = set(re.findall(r"[A-Za-z_]\w*", text))
        return sympy.sympify(text, locals={n: sympy.Symbol(n) for n in names})

    rate = {r.getVariable(): expr(r.getMath()) for r in model.getListOfRules() if r.isRate()}
    S = sympy.Symbol
    x4, x5, x8, x9 = S("x4"), S("x5"), S("x8"), S("x9")
    k6, k7, k8, k20, PTP, IRp = S("k6"), S("k7"), S("k8"), S("k20"), S("PTP"), S("IRp")

    # The structure the closed form rests on.
    assert sympy.simplify(rate["x9"] + rate["x10"] + rate["x12"]) == 0
    assert sympy.simplify(rate["x11"] + rate["x12"]) == 0
    assert sympy.expand(rate["x8"].subs({x5: 0, x8: 0})) == sympy.expand(-k6 * PTP * k8)
    assert sympy.diff(rate["x4"] + rate["x5"], x8) == k20
    assert sympy.simplify(sympy.diff(rate["x9"], x9) + k7 * (x4 + x5) / IRp) == 0

    value = {p.getId(): p.getValue() for p in model.getListOfParameters() if p.isSetValue()}
    (k8_rule,) = [
        r for r in model.getListOfRules() if r.isAssignment() and r.getVariable() == "k8"
    ]
    value["k8"] = float(expr(k8_rule.getMath()))
    assert value["k8"] == pytest.approx(7.06e12, rel=1e-3)
    assert value["IRp"] == pytest.approx(8.97e-13)
    # What overflows first: k9 reads x12/PI3K, and x12 follows -x9.
    (k9_rule,) = [
        r for r in model.getListOfRules() if r.isAssignment() and r.getVariable() == "k9"
    ]
    assert sympy.diff(expr(k9_rule.getMath()), S("x12")) == (S("k9sim") - S("k9bas")) / S("PI3K")
    assert value["PI3K"] == pytest.approx(5e-15)
    x9_0 = model.getSpecies("x9").getInitialConcentration()
    # x9 starts AT the suite's atol, which is why neither engine's error test sees it.
    assert x9_0 == ac.DEFAULT_ATOL == 1e-12

    A = value["k7"] * value["k20"] * value["k6"] * value["PTP"] * value["k8"] / (6 * value["IRp"])
    assert pytest.approx(5.29e20, rel=1e-2) == A
    law = lambda t: x9_0 * math.exp(A * t**3)  # noqa: E731
    t_overflow = ((math.log(sys.float_info.max) - math.log(x9_0)) / A) ** (1 / 3)
    assert t_overflow < 2e-6
    (job,) = [
        j
        for j in read_manifest(_PC / "rr_parity" / "ode_jobs.json")[1]
        if j.model_id == "MODEL1112050001"
    ]
    assert job.params["t_end"] > t_overflow * 1e6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = bngsim.Model.from_sbml(str(path))
        tracked = bngsim.Simulator(m, method="ode").run(
            sample_times=[0.0, 3e-7, 3.363e-7], rtol=1e-8, atol=1e-24
        )
        m.reset()
        suite = bngsim.Simulator(m, method="ode").run(
            (0.0, 100.0), 101, rtol=ac.DEFAULT_RTOL, atol=ac.DEFAULT_ATOL
        )
    names = list(tracked.species_names)
    got = np.asarray(tracked.species)[1, names.index("x9")]
    assert got == pytest.approx(law(3e-7), rel=1e-4)
    # x14 is past 1e10 by t = 3.363e-7 and climbing super-exponentially.
    assert np.asarray(tracked.species)[2, names.index("x14")] > 1e10

    y = dict(zip(names, np.asarray(suite.species)[1], strict=False))  # t = 1
    a = value["k7"] * abs(y["x4"] + y["x5"]) / value["IRp"]
    held = -value["k22"] * value["PTP"] * y["x10"] / a
    assert y["x9"] < 0 < y["x10"]
    assert y["x9"] == pytest.approx(held, rel=1e-3)


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


def test_a_no_ground_truth_bad_test_is_recognised_and_nothing_else_is():
    label = ad.NO_GROUND_TRUTH_LABEL
    assert gam.is_no_ground_truth({"outcome": "BAD_TEST", "comment": f"{label} (GH #1542): x"})
    assert not gam.is_no_ground_truth({"outcome": "BAD_TEST", "comment": "both raised"})
    assert not gam.is_no_ground_truth({"outcome": "BAD_TEST"})
    # The label alone does not make a row one: only the BAD_TEST the disposition made.
    assert not gam.is_no_ground_truth({"outcome": "PASS", "comment": f"{label}: x"})


def test_the_ode_page_draws_a_no_ground_truth_row_as_that(tmp_path):
    """Two engines that ran are not two engines that refused: the default BAD_TEST
    footnote would say both rejected the model, which neither did."""
    label = ad.NO_GROUND_TRUTH_LABEL
    rows = [
        {
            "model_id": "MODEL1112050001",
            "method": "ode",
            "reference_engine": "amici",
            "outcome": "BAD_TEST",
            "comment": f"{label} (GH #1542): x9 leaves the doubles | was DIFF at max_rel=1",
        },
        {
            "model_id": "MODEL0000000001",
            "method": "ode",
            "reference_engine": "amici",
            "outcome": "BAD_TEST",
            "comment": "both raised",
        },
    ]
    report = tmp_path / "report_ode.json"
    meta = {
        "tally": {"BAD_TEST": 2},
        "n_jobs": 2,
        "elapsed_sec": 1.0,
        "versions": {"bngsim": "0", "amici": "0", "python": "3", "platform": "x"},
    }
    report.write_text(json.dumps({"_meta": meta, "results": rows}))
    out = tmp_path / "m.html"
    gam.generate_html(report, out)
    html = out.read_text()
    assert html.count(">NO TRUTH<") == 2  # both engines' badges on the one row
    assert "<strong>NO TRUTH:</strong>" in html  # and the badge legend explains it
    assert html.count("no solution over this horizon") == 1  # bngsim's cell
    assert html.count("no solution to check it against") == 1  # AMICI's cell
    assert html.count("Both engines rejected this model") == 2  # the other row's two cells


def test_the_runners_regime_map_carries_the_no_ground_truth_entry():
    """The runners read ``dispositions_for_regime``, not ``disposition_for``: an
    entry missing from that map is never applied, however it is authored."""
    disp = ad.dispositions_for_regime("ode")["MODEL1112050001"]
    assert disp["kind"] == "no_ground_truth"
    assert "MODEL1112050001" not in ad.dispositions_for_regime("sens")


def test_rr_parity_does_not_confirm_bngsim_on_a_model_with_no_ground_truth():
    """One model, one finding, in both suites. A model this module says has no
    solution cannot be one rr_parity's NO_ORACLE_ADJUDICATED calls a confirmed
    bngsim solve; it must read ``uncoverable`` there (issue #1542 re-adjudicated
    MODEL1112050001 from ``confirm``).

    Loaded by path: bng_parity has an ``overrides`` module too, and a bare import
    resolves to whichever directory reached ``sys.path`` first."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rr_parity_overrides_1542", _PC / "rr_parity" / "overrides.py"
    )
    ov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ov)

    for key in ad.NO_GROUND_TRUTH:
        entry = ov.NO_ORACLE_ADJUDICATED.get(key)
        if entry is not None:
            assert entry["verdict"] == "uncoverable", key
    assert ov.NO_ORACLE_ADJUDICATED["MODEL1112050001:ode"]["verdict"] == "uncoverable"


def test_no_ground_truth_is_authored_for_ode_only():
    """The sensitivity matrix does not draw a NO_GROUND_TRUTH row yet, so a
    ``:sens`` entry would render as an ordinary BAD_TEST, "both engines failed".
    Teach generate_amici_sens_matrix.py first, then drop this."""
    assert all(k.endswith(":ode") for k in ad.NO_GROUND_TRUTH)
