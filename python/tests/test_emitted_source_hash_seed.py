"""The emitted source of a model is the same for every hash seed (issue #550).

Every emitter prints a sum through sympy's ``as_ordered_terms``, which sorts
the terms by their monomials over the sum's generators. sympy collects the
generators in a set and sorts it by ``default_sort_key``, and two generators
that differ only in the type of a number, ``X`` and the ``X**1.0`` that
differentiating ``X**2.0`` leaves, have keys of which neither is less than the
other. They stay in the order the set gave them, which follows
``PYTHONHASHSEED``: one corpus model, MODEL0847869198, had two sensitivity
sources, 8 lines of 2,094 apart, and which of them a run compiled changed from
process to process.

``bngsim._term_order.ordered_terms`` breaks the tie by the generators'
``srepr``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sp = pytest.importorskip("sympy")

from bngsim._term_order import ccode, ordered_terms  # noqa: E402

# The rate of A -> B is a*Aobs^2.0 + b*Aobs^2. Its derivative in Aobs is
# 2.0*a*Aobs**1.0 + 2*b*Aobs: two terms, over the generators Aobs**1.0, Aobs,
# a and b, the first two of which tie.
NET = """begin parameters
    1 a   1.5
    2 b   0.5
    3 kd  0.1
end parameters
begin functions
    1 made() a*Aobs^2.0+b*Aobs^2
end functions
begin species
    1 A() 1
    2 B() 0
end species
begin reactions
    1 1 2 made
    2 2 1 kd
end reactions
begin groups
    1 Aobs  1
end groups
"""

_EMIT = """
import hashlib, logging, sys
logging.disable(logging.CRITICAL)
import bngsim
from bngsim import _codegen as cg
model = bngsim.Model.from_net(sys.argv[1])
source = cg.generate_combined_from_model(model, emit_output_sens=True)[0]
line = next(l for l in source.splitlines() if "/* made */" in l and "obs_sens_c" in l)
print(hashlib.sha256(source.encode()).hexdigest(), "|", line.strip())
"""

# On main, under CPython 3.12, seeds 0 and 1 print the term in a first and
# seeds 3 and 5 the term in b first.
_SEEDS = (0, 1, 3, 5, 7, 8)


def _emitted_under(seed: int, net: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", _EMIT, net],
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.strip().splitlines()[-1]


def test_the_combined_source_is_the_same_under_every_hash_seed(tmp_path):
    """Six processes, six seeds, one source. On main there were two, with
    ``2.0*pow(obs[0], 1.0)*p[0] + 2.0*obs[0]*p[1]`` in one and the two terms
    the other way round in the other."""
    net = tmp_path / "tie.net"
    net.write_text(NET)
    emitted = {seed: _emitted_under(seed, str(net)) for seed in _SEEDS}
    assert len(set(emitted.values())) == 1, emitted
    line = emitted[_SEEDS[0]].split("|", 1)[1]
    assert "pow(obs[0], 1.0)" in line and "obs[0]*p[1]" in line


def _tied_sum():
    """2.0*p*A**1.0 + 2*q*A. The generators sort as A and A**1.0, tied, then p
    and q, so the tied pair leads the monomials and decides the order."""
    x, p, q = sp.symbols("A p q")
    whole = sp.Float(2.0) * p * sp.Pow(x, sp.Float(1.0), evaluate=False)
    return whole + 2 * q * x, x


def test_two_generators_that_tie_are_what_the_model_has():
    """Control. What the fix rests on: ``A`` and ``A**1.0`` are different
    expressions whose sort keys are not equal and of which neither is less
    than the other, so ``sorted`` keeps them in the order it was given."""
    expr, x = _tied_sum()
    power = sp.Pow(x, sp.Float(1.0), evaluate=False)
    assert power != x
    first, second = sp.default_sort_key(x), sp.default_sort_key(power)
    assert first != second
    assert not first < second and not second < first
    assert {g for g in expr.as_terms()[1] if g.has(x)} == {x, power}


def test_a_tie_is_broken_by_what_the_generators_are():
    """The term in ``A**1.0`` is printed first, whichever of the two the set
    gave first: ``Pow(...)`` is ahead of ``Symbol(...)`` in ``srepr``, and the
    first generator's power leads the monomial order. sympy's own order for
    this sum is one or the other with the seed of the process."""
    expr, x = _tied_sum()
    terms = ordered_terms(expr)
    assert len(terms) == 2
    assert terms[0].has(sp.Float(1.0)) and not terms[1].has(sp.Float(1.0))
    assert ccode(expr) == "2.0*pow(A, 1.0)*p + 2*A*q"


def _from_every_set_order(expr, monkeypatch):
    """``(ordered_terms(expr), expr.as_ordered_terms())`` with ``as_terms``
    made to sort its generators from each order a set could hand them over
    in, as a hash seed decides it."""
    import itertools

    terms, gens = expr.as_terms()
    seen = []
    for handed in itertools.permutations(gens):
        other = sorted(handed, key=sp.default_sort_key)
        to = [other.index(g) for g in gens]
        moved = []
        for term, (coeff, monom, ncpart) in terms:
            at = [0] * len(gens)
            for i, power in zip(to, monom, strict=True):
                at[i] = power
            moved.append((term, (coeff, tuple(at), ncpart)))
        with monkeypatch.context() as patch:
            patch.setattr(type(expr), "as_terms", lambda self, m=moved, o=other: (m, o))
            seen.append((ordered_terms(expr), expr.as_ordered_terms()))
    return seen


def test_the_generators_are_asked_for_in_either_order(monkeypatch):
    """The tied sum with its generators sorted from each of the 24 orders a
    set could give them in: sympy's own order of the terms comes out both
    ways, and this one comes out one way."""
    expr, _ = _tied_sum()
    seen = _from_every_set_order(expr, monkeypatch)
    assert len(seen) == 24
    assert len({tuple(native) for _, native in seen}) == 2
    assert len({tuple(ours) for ours, _ in seen}) == 1
    assert seen[0][0] == ordered_terms(expr)


def test_a_tie_that_is_not_between_neighbours(monkeypatch):
    """``<`` on sympy's sort keys is not an order: ``g(1, a)`` is below
    ``g(1, b)``, and ``g(1.0, c)`` ties with both, since the comparison stops
    at the 1 and the 1.0. ``sorted`` leaves the three in an order that depends
    on all of the order it was given, and breaking the tie between neighbours
    only, which a first version of this did, left MODEL0847999575 with two
    sources (36 such sums). Sorted from each of the six orders of its three
    generators, the sum has one order of its terms here and more in sympy."""
    a, b, c = sp.symbols("a b c")
    g = sp.Function("g")
    low, tied, high = g(1, a), g(sp.Float(1.0), c), g(1, b)
    expr = 2 * low + 3 * tied + 5 * high
    k_low, k_tied, k_high = (sp.default_sort_key(x) for x in (low, tied, high))
    assert k_low < k_high
    assert not k_low < k_tied and not k_tied < k_low
    assert not k_tied < k_high and not k_high < k_tied
    seen = _from_every_set_order(expr, monkeypatch)
    assert len(seen) == 6
    assert len({tuple(native) for _, native in seen}) > 1
    assert len({tuple(ours) for ours, _ in seen}) == 1


@pytest.mark.parametrize(
    "text",
    [
        "a*x**2 + b*x + c",
        "1 - 2*x",
        "x - 1",
        "-x*y + 3*x**2*y - y**3 + 2",
        "exp(-k*t)*a - b*log(x) + x/(1 + x)",
        "a*(x + y)**2 - (x - y)*(a + b) + 1.5",
        "x**2.5 + x**0.5 - 2.0*x",
        "-1.0*a*b + a*b**2 - 3",
        "x",
        "2*x",
    ],
)
def test_a_sum_with_no_tie_is_ordered_as_sympy_orders_it(text):
    """Control. Without a tie the order is sympy's own, term for term, so the
    text an emitter prints for such a sum is what it was."""
    expr = sp.sympify(text)
    assert ordered_terms(expr) == expr.as_ordered_terms()
    assert ccode(expr) == sp.ccode(expr)


# ── A symbol picked out of a set ─────────────────────────────────────────────
#
# ``_rewrite_saturating_ratio`` divides ``f^m`` out of ``f^m/(rest + f)^m``, and
# finds m from the slopes of the two exponents against a symbol of theirs. It
# took the first symbol ``free_symbols`` handed it, a set, and for an exponent
# with a condition in it the answer is not the same against every symbol:
# against the one the condition holds, the ratio of the slopes is a product of
# two conditionals that ``cancel`` leaves as it is. MODEL1006230049 was
# rewritten under one seed and not under another, with 10 lines of 1,571
# different.

_PIVOT = """
import sympy as sp
from bngsim._jacobian import _whole_power_offset
pH, t, T1, R = sp.symbols("pH_calc t T1 R")
held = sp.Piecewise((pH, t <= 1.0), (7.3, True))
term = -held + 3.5 + 0.35 * (0.003 - 1 / T1) / R
num = -2 * held + 7.0 + 0.7 * (0.003 - 1 / T1) / R
print(_whole_power_offset(num, term, sp))
"""


def test_a_count_is_found_whichever_symbol_a_set_hands_over_first():
    """``num = 2·term`` for two exponents in pH_calc (under a condition on the
    time), T1 and R. Six processes, six seeds, one answer, (2, 0). On main,
    under CPython 3.12, seeds 0, 2, 5 and 9 take pH_calc first and answer
    None."""
    answers = {}
    for seed in (0, 1, 2, 3, 5, 9):
        done = subprocess.run(
            [sys.executable, "-c", _PIVOT],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        answers[seed] = done.stdout.strip().splitlines()[-1]
    assert set(answers.values()) == {"(2, 0)"}, answers


def test_exponents_that_are_not_parallel_are_refused_against_every_symbol():
    """Control. ``num = 2·term + R``: the slopes against T1 give 2 and the
    leftover keeps R, so there is no count, whichever symbol is asked first."""
    from bngsim._jacobian import _whole_power_offset

    t1, r = sp.symbols("T1 R")
    term = 3.5 + 0.35 * (0.003 - 1 / t1) / r
    assert _whole_power_offset(2 * term + r, term, sp) is None
    assert _whole_power_offset(2 * term + 1, term, sp) == (2, 1)
