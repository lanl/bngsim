"""Single source of truth for the SBML constructs bngsim refuses under ODE and
their SBML Test Suite feature tags (GH #113, GH #241).

bngsim integrates ODEs; it has no DDE solver (``delay()``), no DAE constraint
solver (a non-empty ``AlgebraicRule``), and no fast-equilibrium solver
(``fast="true"``). It refuses all three loud-by-default rather than silently
approximate them (see :mod:`bngsim._sbml_loader` and :mod:`bngsim._simulator`).

Two consumers read the *same* definitions from here so they can never drift:

* the loader refusal messages take their human-readable construct **labels**
  from :data:`CONSTRUCTS` (the loader raises with ``"delay"`` / ``"AlgebraicRule"``
  in the text; the simulator raises with ``"fast"``);
* the SBML Test Suite test-runner manifest (GH #241) declares the **suite tags**
  from here as "unsupported", so the *official* SBML Test Suite grading counts a
  refused case as ``Unsupported`` ("tool does not support the relevant tag(s)")
  rather than ``Error``.

The manifest is a *committed* artifact
(``benchmarks/suites/sbml_test_suite/testrunner/bngsim-unsupported-tags.txt``);
:func:`manifest_text` is its byte-stable generator and a unit test asserts the
two are equal, so the declared boundary is reviewed like code.

Deliberate NON-declarations (documented, not gamed — see the boundary notes in
``README.md`` and GH #231 / GH #242):

* ``CSymbolAvogadro`` is NOT declared. bngsim uses the SI-exact Avogadro
  constant (``6.02214076e23``); three suite cases (00960/00961/01323) encode the
  older rounded value and are therefore an honest ``NoMatch``, not a construct
  bngsim cannot solve.
* ``RandomEventExecution`` is NOT declared. bngsim supports it
  reproducibly-per-seed (GH #242); against a single stochastic reference it may
  ``NoMatch``, which is again an honest numeric outcome, not a capability gap.
"""

from __future__ import annotations

from typing import NamedTuple


class UnsupportedConstruct(NamedTuple):
    """One construct bngsim refuses under ODE.

    ``label`` is the substring that appears in the refusal message (existing
    tests match on ``"delay"`` / ``"AlgebraicRule"`` / ``"fast"`` — do not
    change these); ``reason`` is a one-line human explanation; ``suite_tags``
    are the SBML Test Suite component/test tags that mark a case as using the
    construct (declared unsupported in the manifest).
    """

    label: str
    reason: str
    suite_tags: tuple[str, ...]


# The three refused constructs. ``label`` values are load-bearing: they are the
# substrings the refusal-message tests assert on AND the loader/simulator emit
# them verbatim.
CONSTRUCTS: tuple[UnsupportedConstruct, ...] = (
    UnsupportedConstruct(
        label="delay",
        reason=(
            "delay() turns the system into a delay-differential equation (DDE); "
            "bngsim has no DDE solver"
        ),
        suite_tags=("CSymbolDelay",),
    ),
    UnsupportedConstruct(
        label="AlgebraicRule",
        reason=(
            "a non-empty AlgebraicRule is a DAE constraint defining a variable "
            "implicitly; bngsim has no DAE solver"
        ),
        suite_tags=("AlgebraicRule",),
    ),
    UnsupportedConstruct(
        label="fast",
        reason=(
            'fast="true" declares a fast-equilibrium constraint; bngsim has no '
            "fast-equilibrium constraint solver"
        ),
        suite_tags=("FastReaction", "MultipleFastReactions"),
    ),
)

# Named label constants so the loader references the SSOT rather than repeating
# the literal strings (keeping the refusal text and the manifest tags in sync).
DELAY = CONSTRUCTS[0].label
ALGEBRAIC_RULE = CONSTRUCTS[1].label
FAST = CONSTRUCTS[2].label

# Component tags for test types bngsim does not attempt at all as a time-course
# ODE engine. ``fbc`` (flux-balance-constraints) cases are ``FluxBalanceSteadyState``
# tests, not ``TimeCourse``; declaring the bare ``fbc`` prefix matches every
# ``fbc:*`` tag (the official runner splits a tag on ':' and prefix-matches), so
# those cases score as ``Unsupported`` rather than ``Error``. This is not a
# load-time refusal — the harness never loads a non-TimeCourse case — so it lives
# here (manifest-only) instead of in :data:`CONSTRUCTS`.
NON_TIMECOURSE_TAGS: tuple[str, ...] = ("fbc",)


def construct_labels() -> tuple[str, ...]:
    """The refusal-message labels, in declaration order."""
    return tuple(c.label for c in CONSTRUCTS)


def unsupported_tags() -> list[str]:
    """The sorted, de-duplicated SBML Test Suite tags bngsim declares unsupported.

    This is exactly the set fed to the official runner (and to the local
    :mod:`score` reimplementation) as the wrapper's unsupported-tags list.
    """
    tags: set[str] = set(NON_TIMECOURSE_TAGS)
    for c in CONSTRUCTS:
        tags.update(c.suite_tags)
    return sorted(tags)


def manifest_text() -> str:
    """The byte-stable body of the committed ``bngsim-unsupported-tags.txt``.

    Format: a ``#``-comment header explaining provenance, then one tag per line
    (blank lines and ``#`` comments are ignored by the parser). The trailing
    comma-joined line is the exact string to paste into the SBML Test Suite
    runner's "unsupported tags" field.
    """
    lines = [
        "# bngsim — SBML Test Suite unsupported-tag manifest (GH #241).",
        "#",
        "# GENERATED from bngsim._sbml_unsupported.manifest_text() by",
        "# testrunner/gen_manifest.py — do not edit by hand; a unit test",
        "# (test_sbml_unsupported_manifest.py) asserts this file matches the SSOT.",
        "#",
        "# Each tag marks a construct/test-type bngsim refuses under ODE, so the",
        "# official grading reports those cases as 'Unsupported' rather than",
        "# 'Error'. Reasons:",
    ]
    for c in CONSTRUCTS:
        joined = ", ".join(c.suite_tags)
        lines.append(f"#   {joined}: {c.reason}")
    lines.append(
        "#   "
        + ", ".join(NON_TIMECOURSE_TAGS)
        + ": flux-balance (FluxBalanceSteadyState) cases are not TimeCourse; "
        "bngsim is a time-course ODE engine"
    )
    lines.append("#")
    lines.append(
        "# NOT declared (documented honest NoMatch, not a capability gap): "
        "CSymbolAvogadro (SI-exact constant), RandomEventExecution (supported "
        "reproducibly-per-seed)."
    )
    lines.append("#")
    lines.append("# Paste-ready comma-joined form for the runner UI:")
    lines.append("#   " + ", ".join(unsupported_tags()))
    lines.append("")
    lines.extend(unsupported_tags())
    return "\n".join(lines) + "\n"


def parse_manifest(text: str) -> list[str]:
    """Parse a manifest file body back into its sorted tag list.

    Skips blank lines and ``#`` comments; returns the tags in file order (which
    :func:`manifest_text` writes sorted). Lets the local scorer read the exact
    committed artifact rather than re-deriving the set.
    """
    tags: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tags.append(line)
    return tags
