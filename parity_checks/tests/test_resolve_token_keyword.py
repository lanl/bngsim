"""GH #182: actions-block value expressions must cope with Python-keyword params.

A BNGL parameter may be legally named with a Python keyword (e.g. ``lambda``). The
model is well-formed — it is the parity suite's ``eval``-based token resolver that has
to cope. ``_resolve_token`` reuses bngsim's own keyword-alias machinery
(``_PY_KEYWORD_PARAM_NAMES`` / ``_alias_keyword_param``, the same map the model-block
codegen/Jacobian apply) so the value resolves identically. Mirrors modified_THEM_v1_1.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bng_parity"))

import _bng_common as bc  # noqa: E402


def test_lambda_param_in_value_expression_resolves():
    # modified_THEM_v1_1: setConcentration value "(1/3)*(5.5/11)*Lcpc/(1+lambda)".
    params = {"Lcpc": 6624.354, "lambda": 0.12133333333333333}
    got = bc._resolve_token('"(1/3)*(5.5/11)*Lcpc/(1+lambda)"', params)
    expect = (1 / 3) * (5.5 / 11) * 6624.354 / (1 + 0.12133333333333333)
    assert got is not None
    assert abs(got - expect) < 1e-9


def test_bare_keyword_param_resolves():
    # A bare `lambda` parameter reference (not just inside an expression) resolves.
    assert bc._resolve_token("lambda", {"lambda": 2.5}) == 2.5
    assert bc._resolve_token("2*lambda + 1", {"lambda": 3.0}) == 7.0


def test_non_keyword_expression_is_unaffected():
    # The common path (no keyword params) is unchanged.
    assert bc._resolve_token("10*tend", {"tend": 4.0}) == 40.0
    assert bc._resolve_token("3.5", {}) == 3.5
    assert bc._resolve_token("nope", {"other": 1.0}) is None


def test_keyword_alias_does_not_leak_into_result():
    # The alias substitution is internal — the resolved value is the real number, and a
    # second distinct keyword param in the same expression also resolves.
    params = {"class": 4.0, "lambda": 6.0}
    assert bc._resolve_token("class + lambda", params) == 10.0
