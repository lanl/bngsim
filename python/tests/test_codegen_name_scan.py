"""GH #165 — the derived-expression name scan is linear in the expression.

``_prepare_derived_expr`` has to know which parameters an expression names before
it can differentiate w.r.t. them. It used to ask that question one parameter at a
time, with a freshly interpolated ``re.search(rf"\\b{name}\\b", expr)`` per
candidate — O(#parameters) *pattern compilations* per expression, because a real
model's parameter count outruns ``re``'s internal 512-entry pattern cache and
every search then recompiles. On ``Smith_BMCSystBiol2013`` (922 parameters, 89
derived expressions) that was ~82,000 throwaway compilations per pass and ~0.9 s
of a 1.9 s ``Simulator(...)`` construction — paid on *every* construction, since
the .so cache key is a hash of the generated source.

That cost is why GH #161 read as a regression: it removed an early decline, so a
cross-compartment model reached this scan a second time.

Two things are pinned here, because the fix is an equivalence argument rather
than a new behaviour:

* the scan returns exactly what the per-name ``\\b…\\b`` search returned, over
  the overlap cases the word-boundary anchors exist for; and
* an identifier-named parameter set costs **no** per-name regex at all, which is
  the property that made it fast. Counting calls rather than seconds keeps that
  deterministic.
"""

from __future__ import annotations

import re

import pytest
from bngsim._codegen import _names_referenced_in


def _reference(text: str, names) -> list[str]:
    """The pre-#165 implementation, verbatim in behaviour."""
    return sorted(n for n in names if re.search(rf"\b{re.escape(n)}\b", text))


# (text, names) — every case the \b anchors are there for.
EQUIVALENCE_CASES = [
    ("k1*A + kminus1*B", {"k1", "kminus1", "k2"}),
    # A name that is a prefix of the identifier actually present must not match.
    ("foo_bar*2", {"foo", "foo_bar", "bar"}),
    ("Insulin + t_ins", {"ins", "Ins", "t_ins", "Insulin"}),
    # A digit is a word character, so \b does not hold inside ``2x`` and the
    # parameter ``x`` is NOT referenced there — a tokenizer anchored on
    # ``[A-Za-z_]\w*`` instead of on maximal ``\w+`` runs would say it is.
    ("2x + x3", {"x", "x3"}),
    # Function-call and index syntax puts non-word characters at the boundary.
    ("if(sel >= 1, kA, kB)", {"sel", "kA", "kB", "if"}),
    ("exp(-k*time)", {"k", "time", "exp"}),
    # Names carrying a non-word character (no SBML/BNGL identifier does, but the
    # scan is not allowed to assume it) — these keep the per-name search.
    ("a.b + c", {"a.b", "a", "b", "c"}),
    ("x-y*2", {"x-y", "x", "y"}),
    ("rate[0] + rate", {"rate[0]", "rate"}),
    # Nothing referenced at all, and the empty-expression edge.
    ("42", {"k1", "k2"}),
    ("", {"k1"}),
    ("k1", set()),
]


@pytest.mark.parametrize("text,names", EQUIVALENCE_CASES)
def test_matches_the_per_name_word_boundary_search(text, names):
    assert _names_referenced_in(text, names) == _reference(text, names)


def test_result_is_sorted_and_deduplicated():
    names = {"kb", "ka", "kc"}
    assert _names_referenced_in("ka + kb + ka", names) == ["ka", "kb"]


def test_identifier_names_cost_no_per_name_regex(monkeypatch):
    """The property the fix rests on: one tokenizing pass, not one regex per name.

    ``_HAS_NON_WORD_RE`` and ``_WORD_RUN_RE`` are module-level compiled patterns,
    so neither reaches ``re.search`` — a call here means a name was scanned
    individually, which is exactly the O(#parameters) behaviour being retired.
    """
    calls = []
    real_search = re.search
    monkeypatch.setattr(
        re, "search", lambda p, s, *a, **k: calls.append(p) or real_search(p, s, *a, **k)
    )

    names = {f"k{i}" for i in range(2000)} | {"binding_rate", "k_off"}
    assert _names_referenced_in("k7*A + binding_rate*B", names) == ["binding_rate", "k7"]
    assert calls == []


def test_a_non_identifier_name_still_takes_the_per_name_search(monkeypatch):
    """The fallback is reached only for the names that need it."""
    calls = []
    real_search = re.search
    monkeypatch.setattr(
        re, "search", lambda p, s, *a, **k: calls.append(p) or real_search(p, s, *a, **k)
    )

    names = {f"k{i}" for i in range(500)} | {"a.b"}
    assert _names_referenced_in("k3 + a.b", names) == ["a.b", "k3"]
    assert len(calls) == 1
