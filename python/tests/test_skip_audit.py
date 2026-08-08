"""The skip audit's own classifier (GH #222).

``conftest._is_corpus_absence`` decides which skips get collapsed into the one
"model corpus absent" row and counted in the footer. It exists because that family
both fragments (each gate names its own model, so ~48 skips arrive as ~34 rows of
`1`) and understates (a test parametrized over the corpus contributes one skip
however many models it would have covered).

These tests pin the RULE — absence AND a corpus name — rather than a list of
phrasings, because the phrasings drift and a list that rots is exactly what the
audit above exists to prevent. The strings below are the ones the suite actually
emits today, harvested from a corpus-less checkout; they are examples of the rule,
not the definition of it.
"""

from __future__ import annotations

import pytest

# Emitted today by a checkout with no rr_parity / benchmark corpus.
CORPUS_ABSENCE = [
    "rr_parity corpus model BIOMD0000000150 not present",
    "rr_parity corpus model not present: /some/path/BIOMD0000000474/model.xml",
    "rr_parity corpus not present",
    "rr_parity corpus not present: /some/path",
    "BIOMD879 SBML not present",
    "MODEL1708310001 SBML not present",
    "Mitra2019 JNK benchmark nets not present",
    "benchmark bngl corpus not present",
    "benchmark model not present: /some/path/repro.net",
]

# Skips with other causes. A build variant or an optional dependency is a
# statement about this BUILD, not about a missing corpus, and folding either into
# the corpus count would make the footer's claim false.
NOT_CORPUS_ABSENCE = [
    "bngsim built without the MIR backend (configure with -DBNGSIM_ENABLE_MIR=ON)",
    "requires a build without SuiteSparse/KLU",
    "KLU not compiled",
    "RuleMonkey compiled in",
    "BNG2.pl / perl not available",
    "could not import h5py",
    "roadrunner not installed",
    "stale-binary check explicitly bypassed via env",
    "No CMake build dir; skipping CMakeCache cross-check.",
    "abc.xml not at /Users/x/Code/PyBNF/tests/bngl_files/abc.xml",
]


@pytest.mark.parametrize("reason", CORPUS_ABSENCE)
def test_a_corpus_absence_is_classified_as_one(reason, skip_audit):
    assert skip_audit.is_corpus_absence(reason)


@pytest.mark.parametrize("reason", NOT_CORPUS_ABSENCE)
def test_another_cause_is_not_folded_into_the_corpus_count(reason, skip_audit):
    assert not skip_audit.is_corpus_absence(reason)


class TestTheRuleIsConjunctive:
    """Both halves are load-bearing, and each fails a real reason string alone."""

    def test_absence_alone_is_not_enough(self, skip_audit):
        # A one-off fixture, nothing to do with a corpus: counting it would put a
        # test in the footer that `materialize.py` does not bring back.
        assert not skip_audit.is_corpus_absence("egfr_net.net not present")
        assert not skip_audit.is_corpus_absence("fixture not present: /tmp/x.net")

    def test_a_corpus_name_alone_is_not_enough(self, skip_audit):
        # Says "sbml", reports a build variant. Matching on the name alone would
        # sweep this into a number that claims a corpus would fix it.
        assert not skip_audit.is_corpus_absence("SBML loader compiled out of this build")


def test_every_corpus_absence_is_already_a_declared_skip(skip_audit):
    """The invariant that keeps strict mode honest.

    ``BNGSIM_SKIP_AUDIT=strict`` fails a run for any skip whose reason matches no
    ``_DECLARED_SKIPS`` entry, and python-tests.yml sets it on two platforms that
    never have the corpus. So a corpus-absence reason MUST stay declared — the
    footer is there to make it visible, not to make it fatal. If these ever
    diverge, every CI leg and every worktree push turns red at once, which is the
    cry-wolf outcome conftest's own note argues against.
    """
    for reason in CORPUS_ABSENCE:
        assert any(
            pattern.lower() in reason.lower() for pattern, _ in skip_audit.declared_skips
        ), f"{reason!r} is classified as corpus absence but is not a declared skip"
