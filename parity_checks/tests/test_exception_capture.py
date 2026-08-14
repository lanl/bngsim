"""Issue #324 — a capped exception must still say what class of failure it was.

The report is the durable artifact: the sensitivity sweep behind it is ~4 hours
at 4 workers, so a census that cannot be answered from the report can only be
answered by re-running the sweep. Every captured exception is capped so 2,646
rows stay readable, and the cap used to be a plain head cut.

That is exactly the wrong end to keep. Several bngsim refusals enumerate model
symbols *before* naming the fault — the under-specified-model refusal (#323)
reads ``Parameters 'A', 'B', ... have no value attribute and no
initialAssignment, but are referenced by ...`` — so on a model with a long
parameter list all 400 characters were names and the diagnostic never appeared.
Three models (`MODEL0848342500`, `MODEL7980735163`, `MODEL9808533471`) could not
be classified from the report at all.

Two changes, tested here:

* the cap drops the **middle**, marking how much went, so a trailing diagnostic
  survives an arbitrarily long enumeration at the same 400-character budget; and
* every row carries ``exception_class`` — ``"<phase>:<ExceptionType>"`` — a key
  that is stable across models, so a census groups on it instead of parsing
  prose that a cap may have cut anywhere.

The second change has a consequence worth its own test: the AMICI refusal
subclass is now decided in the worker, against the **full** message, rather than
in the parent against whatever survived the cap.
"""

from __future__ import annotations

from pathlib import Path

import amici_run
import amici_sens_run
import pytest
from _core import EXCEPTION_TEXT_LIMIT, capture_exception
from _core.schema import JobResult

_MODELS_DIR = Path(__file__).resolve().parents[1] / "rr_parity" / "models"


class _ModelError(Exception):
    """Stands in for bngsim's ModelError — the capture keys on type name only."""


def _long_refusal(n: int = 30, prefix: str = "Ca_SR_DS_Calcium_Concentration") -> _ModelError:
    """#323's message shape: a long enumeration, then the clause that names it."""
    names = ", ".join(f"'{prefix}_{i}'" for i in range(n))
    return _ModelError(
        f"Parameters {names} have no value attribute and no initialAssignment, "
        f"but are referenced by the rate law of reaction J17."
    )


# ── The cap keeps the end of the message ────────────────────────────────────


def test_the_trailing_diagnostic_survives_a_long_enumeration():
    """The issue's own reproducer: the phrase naming the failure class is last."""
    cap = capture_exception("bngsim-params", _long_refusal())

    assert "have no value attribute and no initialAssignment" in cap.text
    # ...and the head cut this replaces did not keep it, so the test is about the
    # change rather than about this fixture happening to be short.
    head_cut = f"bngsim-params: {type(_ModelError()).__name__}: {_long_refusal()}"[
        :EXCEPTION_TEXT_LIMIT
    ]
    assert "no initialAssignment" not in head_cut


def test_the_cap_is_the_same_size_it_always_was():
    """Middle-elision buys the tail at no cost in report size."""
    cap = capture_exception("bngsim-params", _long_refusal())
    assert len(cap.text) == EXCEPTION_TEXT_LIMIT


def test_an_elided_message_says_it_was_elided_and_by_how_much():
    """A reader must never mistake a cut message for one that reads that way.

    The count is checked to be the *real* one — head + elided + tail has to
    reconstruct the original length — so the marker cannot drift into a
    decorative number that says nothing.
    """
    exc = _long_refusal()
    cap = capture_exception("bngsim-params", exc)
    full = cap.full

    head, rest = cap.text.split(" ...[", 1)
    n_elided_txt, tail = rest.split(" chars elided]... ", 1)
    assert len(head) + int(n_elided_txt) + len(tail) == len(full)
    assert full.startswith(head) and full.endswith(tail)


def test_a_short_message_is_untouched():
    """No marker, no truncation — the common case must stay verbatim."""
    cap = capture_exception("compare", ValueError("shapes (3,) and (4,) differ"))
    assert cap.text == "compare: ValueError: shapes (3,) and (4,) differ"
    assert "elided" not in cap.text


def test_a_message_exactly_at_the_limit_is_untouched():
    """The boundary, so an off-by-one cannot elide a message that already fits."""
    exc = ValueError("x" * (EXCEPTION_TEXT_LIMIT - len("p: ValueError: ")))
    cap = capture_exception("p", exc)
    assert len(cap.text) == EXCEPTION_TEXT_LIMIT
    assert "elided" not in cap.text


def test_a_pathologically_small_limit_degrades_to_a_head_cut():
    """No limit leaves room for a marker plus two characters; do not crash."""
    cap = capture_exception("p", _long_refusal(), limit=8)
    assert len(cap.text) == 8


# ── The grouping key ────────────────────────────────────────────────────────


def test_the_class_is_stable_across_models_that_fail_the_same_way():
    """The whole point: two models whose symbol names differ group together.

    Their `text` cannot do this — the variable part is unbounded and lands
    wherever the cap falls — which is why three models had to be grouped by hand.
    """
    a = capture_exception("bngsim-params", _long_refusal(30, "Ca_SR_DS_Calcium"))
    b = capture_exception("bngsim-params", _long_refusal(80, "totally_different_symbol"))

    assert a.cls == b.cls == "bngsim-params:_ModelError"
    assert a.text != b.text


def test_the_class_separates_phases_of_the_same_exception_type():
    """A RuntimeError from the AMICI build is not one from the comparison."""
    exc = RuntimeError("boom")
    assert capture_exception("amici-build", exc).cls == "amici-build:RuntimeError"
    assert capture_exception("compare", exc).cls == "compare:RuntimeError"


def test_full_is_uncapped_and_never_the_reported_text():
    """`full` exists for classifiers; the report gets `text`."""
    cap = capture_exception("amici", _long_refusal())
    assert len(cap.full) > EXCEPTION_TEXT_LIMIT
    assert cap.text != cap.full
    assert cap.full.endswith("reaction J17.")


def test_the_report_row_carries_the_class():
    """`JobResult` round-trips it, and an older report without it still loads."""
    row = JobResult(
        model_id="M",
        method="sens/staggered",
        reference_engine="amici",
        outcome="UNSUPPORTED",
        exception="bngsim-params: ModelError: ...",
        exception_class="bngsim-params:ModelError",
    )
    assert row.to_dict()["exception_class"] == "bngsim-params:ModelError"
    assert JobResult.from_dict(row.to_dict()).exception_class == "bngsim-params:ModelError"

    legacy = {k: v for k, v in row.to_dict().items() if k != "exception_class"}
    assert JobResult.from_dict(legacy).exception_class is None


# ── The class tracks the text through the attribution ───────────────────────


@pytest.mark.parametrize("mod", [amici_sens_run, amici_run], ids=["sens", "ode"])
class TestClassifyFailureCarriesTheClass:
    """Whichever sides `_classify_failure` keeps in the text, it keeps in the key."""

    def test_bngsim_only(self, mod):
        status, text, cls = mod._classify_failure("bn", "", False, "bngsim:ValueError", "")
        assert (status, text, cls) == ("exception", "bn", "bngsim:ValueError")

    def test_reference_only(self, mod):
        status, text, cls = mod._classify_failure("", "am", False, "", "amici:RuntimeError")
        assert (status, text, cls) == ("reference_failed", "am", "amici:RuntimeError")

    def test_both_raised_keeps_both_in_reference_first_order(self, mod):
        status, text, cls = mod._classify_failure(
            "bn", "am", False, "bngsim:ValueError", "amici:RuntimeError"
        )
        assert status == "bad_test"
        assert text == "am || bn"
        assert cls == "amici:RuntimeError || bngsim:ValueError"

    def test_a_declared_refusal_wins_and_keeps_the_reference_alongside(self, mod):
        status, text, cls = mod._classify_failure(
            "bn", "am", True, "bngsim:SensitivityUnsupportedError", "amici:RuntimeError"
        )
        assert status == "unsupported"
        assert text == "bn || am"
        assert cls == "bngsim:SensitivityUnsupportedError || amici:RuntimeError"

    def test_a_declared_refusal_alone_does_not_join_an_empty_key(self, mod):
        status, text, cls = mod._classify_failure(
            "bn", "", True, "bngsim:SensitivityUnsupportedError", ""
        )
        assert (status, text, cls) == (
            "unsupported",
            "bn",
            "bngsim:SensitivityUnsupportedError",
        )

    def test_classes_are_optional(self, mod):
        """A caller with no keys still gets the status and the text."""
        status, text, cls = mod._classify_failure("bn", "am")
        assert (status, text, cls) == ("bad_test", "am || bn", "")


# ── The refusal subclass no longer depends on where the cap fell ────────────


def test_the_refusal_subclass_reads_the_full_message_not_the_capped_one():
    """An AMICI keyword past the head budget used to decide the subclass by luck.

    ``classify_reference_refusal`` keys on phrases like ``cvode`` /
    ``too_much_work``. Under a head cut those had to sit inside the first 400
    characters to be seen; under middle-elision the head is smaller still. So the
    worker classifies from ``CapturedException.full`` and records the verdict,
    and this pins that the two answers actually differ on a realistic message.
    """
    import _amici_common as ac

    padding = "x" * 600
    exc = RuntimeError(f"AMICI failed. {padding} Reason: CVODES returned TOO_MUCH_WORK")
    cap = capture_exception("amici", exc)

    assert ac.classify_reference_refusal(cap.full) == "integrator"
    assert ac.classify_reference_refusal(cap.text) == "integrator"

    # ...and where the keyword lands squarely in the elided middle, only the full
    # text can still answer — which is why the worker keeps it.
    buried = RuntimeError(f"AMICI failed. {padding} CVODES trouble. {padding} Giving up.")
    buried_cap = capture_exception("amici", buried)
    assert ac.classify_reference_refusal(buried_cap.full) == "integrator"
    assert ac.classify_reference_refusal(buried_cap.text) == "other"


# ── The three models the issue could not classify from the report ───────────

# Gated on the file, not the directory: the corpus is gitignored, so `models/`
# exists and is empty in a fresh worktree and in CI (GH #192).
_UNCLASSIFIABLE = ["MODEL0848342500", "MODEL7980735163", "MODEL9808533471"]


@pytest.mark.parametrize("model_id", _UNCLASSIFIABLE)
def test_the_three_models_the_issue_names_are_now_classifiable(model_id):
    """#324's evidence, end to end on the models that produced it.

    Each raises #323's under-specified-model refusal with a parameter list long
    enough (892–1173 characters of message) that the head cut ended mid-symbol.
    Two things a reader needs are now in the row: the closing clause naming the
    remedy — including the environment escape hatch, which identifies the refusal
    on sight — and a key identical across all three, which is the grouping that
    had to be done by hand against the source.

    The enumeration is long enough here that the *opening* clause ("... have no
    value attribute and no initialAssignment") still falls in the elided middle.
    That is the split working as designed rather than a shortfall: the head keeps
    which model and which symbols, the tail keeps what to do about it, and
    `exception_class` is what a census groups on either way.
    """
    path = _MODELS_DIR / model_id / f"{model_id}.xml"
    if not path.exists():
        pytest.skip(f"corpus model not present: {path}")

    import bngsim._sbml_loader as loader

    with pytest.raises(Exception) as exc_info:  # noqa: B017 - the type is the assertion below
        loader.load_sbml_string(path.read_text())
    cap = capture_exception("bngsim-params", exc_info.value)

    assert cap.cls == "bngsim-params:UnderSpecifiedModelError"
    assert len(cap.full) > EXCEPTION_TEXT_LIMIT, "the fixture must actually be truncated"
    assert "add an initialAssignment" in cap.text
    assert "BNGSIM_ALLOW_UNSET_PARAMS=1" in cap.text

    # The head cut this replaces kept none of that — it ended inside the symbol
    # list, which is why these three had to be classified against their source.
    old = cap.full[:EXCEPTION_TEXT_LIMIT]
    assert "initialAssignment" not in old
    assert "BNGSIM_ALLOW_UNSET_PARAMS=1" not in old
