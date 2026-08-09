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

import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

# A hole in an f-string reason, e.g. f"{net_name} not present". Matching is a
# substring test, so the runtime text is unknowable but the literal parts still
# have to be declared; a sentinel keeps the two halves from being glued into a
# substring that exists in neither.
_HOLE = "\x00"

_SKIP_CALLS = frozenset({"pytest.skip", "skip"})
_SKIPIF_MARKS = frozenset({"pytest.mark.skipif", "mark.skipif"})
# Trap 1 (#179): pytest.importorskip("sympy") GENERATES "could not import
# 'sympy'" at run time, which is declared — but the source only shows the bare
# literal "sympy", so scanning these call sites invents failures. #179 measured
# that skipping this step roughly triples the apparent problem size.
#
# With a sub-case the issue did not name and a first pass at this check missed:
# importorskip takes an OPTIONAL `reason=`, and when it is given the reason is
# hand-written after all — the generated text is never produced. Ignoring the
# call site wholesale therefore misses a real reason. There is exactly one such
# site today (test_vivarium.py), and it went undeclared long enough for a strict
# run to catch what this scan had just pronounced clean.
_IMPORTORSKIP = frozenset({"pytest.importorskip", "importorskip"})
# Trap 2 (#179): an xfail reason never reaches the audit, because an xfail is not
# a skip. Scanning `reason=` without distinguishing the mark sweeps them in.
_XFAIL_MARKS = frozenset({"pytest.mark.xfail", "mark.xfail"})


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_text(node: ast.AST | None) -> str | None:
    """The fixed text of a reason expression, or None if there is none."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else _HOLE
            for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _literal_text(node.left)
    return None


def _hand_written_skip_reasons() -> dict[str, list[str]]:
    """Every skip reason written as a literal under ``python/tests/``.

    AST rather than regex so the two traps are decided by what the node *is*:
    a `reason=` on `skipif` is a skip reason and the identical kwarg on `xfail`
    is not, and no regex over `reason=` can tell them apart.
    """
    found: dict[str, list[str]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _dotted(node.func)
            explicit_reason = any(kw.arg == "reason" for kw in node.keywords)
            if name in _XFAIL_MARKS:
                continue
            if name in _IMPORTORSKIP and not explicit_reason:
                continue
            if name in _SKIP_CALLS:
                arg = node.args[0] if node.args else None
            elif name in _SKIPIF_MARKS or name in _IMPORTORSKIP:
                arg = None
            else:
                continue
            for kw in node.keywords:
                if kw.arg == "reason":
                    arg = kw.value
            text = _literal_text(arg)
            if text and text.strip(_HOLE).strip():
                found.setdefault(text, []).append(f"{path.name}:{node.lineno}")
    return found


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


class TestDeclarationsCoverTheReasonsTestsActuallyEmit:
    """The drift check (#179).

    ``_DECLARED_SKIPS`` is hand-maintained and nothing compared it against the
    reason strings the suite emits, in either direction. It had drifted to 25
    undeclared reasons across 47 files — none of which fired in the default
    build, which is exactly why nobody saw them: they are build-variant reasons,
    and the variants they describe are the ones the OTHER CI legs use. That is
    what blocked turning strict mode on anywhere but the default build.
    """

    def test_every_hand_written_reason_is_declared(self, skip_audit):
        """Forward direction: a new skip has to be justified in a diff."""
        undeclared = {
            reason: sites
            for reason, sites in _hand_written_skip_reasons().items()
            if skip_audit.tier_of(reason) is None
        }
        assert not undeclared, "skip reasons matching no _DECLARED_SKIPS pattern:\n" + "\n".join(
            f"  {reason!r}  ({', '.join(sites[:4])})"
            for reason, sites in sorted(undeclared.items())
        )

    def test_every_declared_pattern_still_matches_something(self, skip_audit):
        """Reverse direction: a declaration outliving its skip is a licence
        nobody is using, and it reads as a decision somebody made about a
        condition that no longer exists. ``scipy`` was one — subsumed by
        ``could not import`` and matching no hand-written reason — and removing
        it is half of what #179 asked for.

        ``pytest.importorskip`` counts as a match: its reasons are real, they
        are simply generated at run time rather than written down, so a pattern
        justified only by them is doing its job.
        """
        hand_written = _hand_written_skip_reasons()
        generated = self._importorskip_reasons()
        unmatched = [
            pattern
            for pattern, _tier, _why in skip_audit.declared_skips
            if not any(pattern.lower() in reason.lower() for reason in (*hand_written, *generated))
        ]
        assert not unmatched, (
            "declared patterns that no skip in this tree can produce: "
            f"{unmatched} — delete them, or fix the test that stopped emitting them"
        )

    @staticmethod
    def _importorskip_reasons() -> list[str]:
        """What ``pytest.importorskip("x")`` reports when x is absent."""
        names: list[str] = []
        for path in sorted(TESTS_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and _dotted(node.func) in _IMPORTORSKIP
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    names.append(str(node.args[0].value))
        return [f"could not import {name!r}" for name in names]


class TestTheScannerHandlesBothTrapsFromTheIssue:
    """The scan is the thing asserting the invariant, so its own two known
    failure modes are pinned here. Both cost #179's author time, and both fail
    SILENTLY — trap 1 by inventing undeclared reasons, trap 2 by counting
    something that never reaches the audit at all.
    """

    def _reasons_in(self, tmp_path, source: str) -> dict[str, list[str]]:
        global TESTS_DIR  # noqa: PLW0603 - point the scanner at a one-file tree
        (tmp_path / "test_sample.py").write_text(source, encoding="utf-8")
        original, TESTS_DIR = TESTS_DIR, tmp_path
        try:
            return _hand_written_skip_reasons()
        finally:
            TESTS_DIR = original

    def test_importorskip_is_not_read_as_a_hand_written_reason(self, tmp_path):
        found = self._reasons_in(
            tmp_path,
            'import pytest\npytest.importorskip("sympy")\n',
        )
        assert found == {}, f"importorskip's argument was scanned as a reason: {found}"

    def test_an_explicit_reason_on_importorskip_is_hand_written_after_all(self, tmp_path):
        """The sub-case that got past the first version of this scan: pass
        ``reason=`` and the generated text is never produced, so the string in
        the source is the one the audit will see and must be declared."""
        found = self._reasons_in(
            tmp_path,
            "import pytest\n"
            'pytest.importorskip("vivarium", reason="vivarium-core not installed")\n',
        )
        assert list(found) == ["vivarium-core not installed"]

    def test_an_xfail_reason_is_not_a_skip_reason(self, tmp_path):
        found = self._reasons_in(
            tmp_path,
            'import pytest\n@pytest.mark.xfail(reason="known bad")\ndef test_x(): ...\n',
        )
        assert found == {}, f"an xfail reason was scanned as a skip reason: {found}"

    def test_skipif_and_skip_reasons_are_both_found(self, tmp_path):
        found = self._reasons_in(
            tmp_path,
            "import pytest\n"
            '@pytest.mark.skipif(True, reason="from skipif")\n'
            "def test_a(): ...\n"
            'def test_b(): pytest.skip("from skip")\n',
        )
        assert set(found) == {"from skipif", "from skip"}

    def test_an_f_string_reason_keeps_its_literal_parts(self, tmp_path):
        """A reason with a runtime hole still has to be declared on the fixed
        text around it — that is all a substring match can ever see."""
        found = self._reasons_in(
            tmp_path,
            'import pytest\ndef test_a(): pytest.skip(f"{name} corpus not present")\n',
        )
        assert list(found) == ["\x00 corpus not present"]


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
        assert skip_audit.tier_of(reason) == skip_audit.ANYWHERE, (
            f"{reason!r} is classified as corpus absence but is not declared as a "
            f"skip that is legitimate anywhere (tier: {skip_audit.tier_of(reason)})"
        )
