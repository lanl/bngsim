"""The emitted source of a model is the same for every hash seed (issue #550).

Every emitter prints a sum through sympy's ``as_ordered_terms``, which sorts
the terms by their monomials over the sum's generators. sympy collects the
generators in a set and sorts it by ``default_sort_key``, and two generators
that differ only in the type of a number, ``X`` and the ``X**1.0`` that
differentiating ``X**2.0`` leaves, have keys of which neither is less than the
other. They stay in the order the set gave them, which follows
``PYTHONHASHSEED``: one corpus model, MODEL0847869198, had two sensitivity
sources, 8 lines of 2,099 apart, and which of them a run compiled changed from
process to process.

``bngsim._term_order`` computes the keys the orders are taken from, with no
ties in them, and the emitters print by those.

Several tests here ask the module for an order and compare it with sympy's
own. On main there is no such module: with sympy's orders standing in for it,
those that say an expression with no tie is printed as it was pass under every
seed, and those that say a tie is broken fail under some.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

sp = pytest.importorskip("sympy")


def ordered_terms(expr):
    """``bngsim._term_order.ordered_terms``, imported where it is used so that
    the controls run on a build that does not have the module."""
    from bngsim._term_order import ordered_terms

    return ordered_terms(expr)


def ccode(expr):
    from bngsim._term_order import ccode

    return ccode(expr)


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


# A parameter that is an expression of others, kdd = pp·aa^2.0 + qq·aa^2. Its
# partial in aa is printed by another path (sympy's C printer, for the chain
# rule through a derived parameter), with the tied pair aa and aa**1.0 ahead of
# pp and qq.
DERIVED = """begin parameters
    1 aa  0.7
    2 pp  1.5
    3 qq  0.5
    4 kdd pp*aa^2.0+qq*aa^2
end parameters
begin species
    1 A() 1
    2 B() 0
end species
begin reactions
    1 1 2 kdd
    2 2 1 pp
end reactions
begin groups
    1 Aobs  1
end groups
"""

_EMIT_DERIVED = _EMIT.replace(
    'if "/* made */" in l and "obs_sens_c" in l', 'if "pow(p[0], 1.0)" in l'
)


def test_the_partials_of_a_derived_parameter_are_the_same_under_every_hash_seed(tmp_path):
    """On main, under CPython 3.12, seeds 0 and 1 print
    ``2.0*pow(p[0], 1.0)*p[1] + 2*p[0]*p[2]`` and seeds 2 and 7 the two terms
    the other way round."""
    net = tmp_path / "derived.net"
    net.write_text(DERIVED)
    emitted = {}
    for seed in (0, 1, 2, 7):
        done = subprocess.run(
            [sys.executable, "-c", _EMIT_DERIVED, str(net)],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        emitted[seed] = done.stdout.strip().splitlines()[-1]
    assert len(set(emitted.values())) == 1, emitted
    assert "2.0*pow(p[0], 1.0)*p[1] + 2*p[0]*p[2]" in emitted[0]


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
    gave first. The keys of ``A**1.0`` and ``A`` differ where one has the
    exponent ``Float(1.0)`` and the other ``Integer(1)``, and ``Float`` is
    ahead of ``Integer`` in the order of the types' names; the first
    generator's power leads the monomial order. sympy's own order for this sum
    is one or the other with the seed of the process."""
    expr, x = _tied_sum()
    terms = ordered_terms(expr)
    assert len(terms) == 2
    assert terms[0].has(sp.Float(1.0)) and not terms[1].has(sp.Float(1.0))
    assert ccode(expr) == "2.0*pow(A, 1.0)*p + 2*A*q"


def _from_every_set_order(expr, monkeypatch, read=None):
    """``(ordered_terms(expr), expr.as_ordered_terms())``, or ``read(expr)``,
    with ``as_terms`` made to sort its generators from each order a set could
    hand them over in, as a hash seed decides it."""
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
        from bngsim import _term_order

        _term_order._forget()  # what an earlier order left memoized
        with monkeypatch.context() as patch:
            patch.setattr(type(expr), "as_terms", lambda self, m=moved, o=other: (m, o))
            seen.append(read(expr) if read else (ordered_terms(expr), expr.as_ordered_terms()))
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


def test_the_order_of_two_keys():
    """What the generators are sorted by: tuples compared entry by entry, as
    ``<`` compares them where it can, a shorter one ahead of one it begins,
    and two numbers of one value and two types in the order of the types'
    names. Unlike ``<`` on the keys, it has no ties but between equals."""
    from bngsim._term_order import _key_cmp

    one, real = sp.Integer(1), sp.Float(1.0)
    assert _key_cmp((1, "a"), (1, "b")) < 0 < _key_cmp((2, "a"), (1, "b"))
    assert _key_cmp((1, "a"), (1, "a")) == 0
    assert _key_cmp((1,), (1, "a")) < 0 < _key_cmp((1, "a"), (1,))
    assert not one < real and not real < one and one != real
    assert _key_cmp(real, one) < 0 < _key_cmp(one, real)
    assert _key_cmp((real, "z"), (one, "a")) < 0 < _key_cmp((one, "a"), (real, "z"))
    assert _key_cmp(sp.Float(0.5), one) < 0 < _key_cmp(sp.Float(1.5), one)


def test_two_generators_with_one_key_are_told_apart_by_what_they_are(monkeypatch):
    """A symbol and its namesake with an assumption are two expressions with
    one sort key. Nothing bngsim builds has such a pair, and sympy's order of
    them is the set's; here it is that of their ``srepr``."""
    plain, positive, z = sp.Symbol("x"), sp.Symbol("x", positive=True), sp.Symbol("z")
    assert plain != positive and sp.default_sort_key(plain) == sp.default_sort_key(positive)
    expr = 2 * plain + 3 * z * positive
    seen = _from_every_set_order(expr, monkeypatch)
    assert len({tuple(native) for _, native in seen}) > 1
    assert len({tuple(ours) for ours, _ in seen}) == 1


@pytest.mark.parametrize(
    "emitter, text",
    [("exprtk", "2.0*((A)^(1.0))*p + 2*A*q"), ("c", "2.0*pow(A, 1.0)*p + 2.0*A*q")],
)
def test_both_printers_of_the_jacobian_print_the_tied_sum_one_way(monkeypatch, emitter, text):
    """``sympy_to_exprtk`` writes the Jacobian the interpreter evaluates and
    ``sympy_to_c`` the one that is compiled. Each has a printer of its own, and
    each printed the tied sum either way round, with the order its set of
    generators came in."""
    from bngsim import _jacobian

    def emit(expr):
        if emitter == "exprtk":
            return _jacobian.sympy_to_exprtk(expr)
        return _jacobian.sympy_to_c(expr, lambda name: name)

    expr, _ = _tied_sum()
    seen = _from_every_set_order(expr, monkeypatch, read=emit)
    assert len(seen) == 24 and set(seen) == {text}


def test_an_edit_to_the_term_order_changes_the_cache_key(tmp_path):
    """The order of the terms is part of what is emitted, so the module that
    decides it is one of those the codegen cache key is a digest of: an edit
    to it must not be met by a library built before it."""
    from bngsim import _codegen as cg

    assert "_term_order" in cg._CODEGEN_SOURCE_MODULES
    for name in (*cg._CODEGEN_SOURCE_MODULES, "_term_order"):
        (tmp_path / f"{name}.py").write_text(f"# {name}\n")
    before = cg._compute_codegen_source_digest(tmp_path)
    (tmp_path / "_term_order.py").write_text("# _term_order\n# another order\n")
    assert before != "" and cg._compute_codegen_source_digest(tmp_path) != before


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
        "k*(a + x)*(b + y)*(a + 2.5*y)",
        "-k*(a + x)**2*exp(b + y)/(c + x*y)",
        "Max(a*x, b + y, 1) - Min(x + y, 2*a)",
        "k*(a*x + b)**(c + 1) + k*(b*x + a)**(c + 1)",
        "Max(x*y*a*b, c + k) + Min(a*b*c*x, y + k)",
    ],
)
def test_a_sum_with_no_tie_is_ordered_as_sympy_orders_it(text):
    """Control (of the module against sympy's own). What the fix must not
    change: without a tie the order is sympy's own, term for term and factor
    for factor, and so are the key and the text an emitter prints for such an
    expression."""
    from bngsim import _term_order

    expr = sp.sympify(text)
    assert ordered_terms(expr) == expr.as_ordered_terms()
    assert ccode(expr) == sp.ccode(expr)
    assert _term_order.srepr(expr) == sp.srepr(expr)
    for part in sp.preorder_traversal(expr):
        assert _term_order.stable_key(part) == part.sort_key()
        if part.is_Mul:
            assert _term_order.ordered_factors(part) == part.as_ordered_factors()


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


def test_a_symbol_that_gives_no_count_is_passed_over_for_the_next():
    """The same exponents with the conditional's symbol named so that it is
    asked first, ``A_calc`` ahead of ``R`` and ``T1``. Against it the ratio of
    the slopes is not a number, and the next symbol is asked: (2, 0), under
    six seeds in six processes. Taking the first symbol in the order of the
    names would have answered None for every seed, where main answers it for
    some."""
    answers = {}
    for seed in (0, 1, 2, 3, 5, 9):
        done = subprocess.run(
            [sys.executable, "-c", _PIVOT.replace("pH_calc", "A_calc")],
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


def test_a_mismatch_costs_one_cancel(monkeypatch):
    """Control. Exponents that are not parallel, in three symbols and with no
    condition in them: the ratio of the slopes against the first symbol is no
    count, and it is every symbol's answer, so no other is asked. (Asking each
    in turn cost MODEL1006230049 3 s of 90; main asks one, as this does.)"""
    from bngsim._jacobian import _whole_power_offset

    t1, r, u = sp.symbols("T1 R U")
    term = 3.5 + 0.35 * (0.003 - 1 / t1) / r + u
    calls = []
    cancel = sp.cancel
    monkeypatch.setattr(sp, "cancel", lambda e: (calls.append(e), cancel(e))[1])
    assert _whole_power_offset(2 * term + r * u, term, sp) is None
    assert len(calls) == 1


# ── A key that is made of a tied sum ─────────────────────────────────────────
#
# The key of a sum is made of its ordered terms, so whatever holds a tied sum
# has a key that follows the seed, and so has every order taken from keys.

_PRINTED = """
import sympy as sp
from bngsim._jacobian import sympy_to_c, sympy_to_exprtk
a, b, c, k, X, Y = sp.symbols("a b c k X Y")
tied = 2.0*a*X**1.0 + 2*b*X
other = 2*a*X + c*Y
cases = [
    k*tied**2 + k*other**2,
    k*sp.exp(tied) + k*sp.exp(other),
    sp.exp(tied*(2*b*X + c*Y)) + sp.exp(other*(2*b*X + c*Y)),
    k*tied*(2*b*X + c*Y),
    -k*tied*(2*b*X + c*Y)/(a + X),
    sp.Min(a*X**2.0 + 1, b*X**2 + 1.0),
    sp.Max(a*X**2.0 + 1, b*X**2 + 1.0, tied),
    sp.And(2.0*X**2 > Y, 2*X**2.0 > Y),
    sp.Or(2.0*X**2 > Y, 2*X**2.0 > Y),
]
for e in cases:
    print(PRINTERS)
"""


def _printed_under(seed: int, script: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def test_what_holds_a_tied_sum_is_printed_one_way_under_every_hash_seed():
    """A sum of two powers of sums, of two exponentials of sums; a product of
    two bracketed sums, and the same with a sign and a denominator; ``min``,
    ``max``, ``and`` and ``or`` of expressions that tie. Each through both
    printers of the Jacobian, under six seeds in six processes: one text. On
    main the two terms, the two factors and the two arguments come out either
    way round."""
    script = _PRINTED.replace(
        "PRINTERS", 'sympy_to_c(e, lambda name: name), "|", sympy_to_exprtk(e)'
    )
    printed = {seed: _printed_under(seed, script) for seed in _SEEDS}
    assert len(set(printed.values())) == 1, printed
    lines = printed[_SEEDS[0]].splitlines()
    assert len(lines) == 9 and "None" not in printed[_SEEDS[0]]


def test_the_c_printer_of_a_derived_parameter_and_srepr_are_one_way_too():
    """The same expressions through ``_term_order.ccode``, which prints the
    partials of a derived parameter, and ``_term_order.srepr``, which is what
    sets of expressions are sorted by where their order is emitted."""
    script = _PRINTED.replace(
        "from bngsim._jacobian import sympy_to_c, sympy_to_exprtk",
        "from bngsim._term_order import ccode, srepr",
    ).replace("PRINTERS", 'ccode(e), "|", srepr(e)')
    printed = {seed: _printed_under(seed, script) for seed in _SEEDS}
    assert len(set(printed.values())) == 1, printed


def test_sympys_own_srepr_of_a_tied_sum_follows_the_seed():
    """Control. ``srepr`` prints a sum's terms in the order of
    ``as_ordered_terms``, so it is no key to sort a set by: under these six
    seeds it gives the tied sum both ways."""
    script = (
        "import sympy as sp\n"
        "a, b, X = sp.symbols('a b X')\n"
        "print(sp.srepr(2.0*a*X**1.0 + 2*b*X))\n"
    )
    assert len({_printed_under(seed, script) for seed in (0, 1, 2, 3, 5, 9)}) == 2


# The rate of A -> B holds two tied sums: as the bases of two squares that are
# added, and as two factors of one product.
_WITH_B = NET.replace("    1 Aobs  1\nend groups", "    1 Aobs  1\n    2 Bobs  2\nend groups")
NESTED = _WITH_B.replace(
    "a*Aobs^2.0+b*Aobs^2", "a*(a*Aobs^2.0+b*Aobs^2)^2+a*(a*Aobs^2+b*Aobs*Bobs)^2"
)
FACTORS = _WITH_B.replace("a*Aobs^2.0+b*Aobs^2", "a*(a*Aobs^2.0+b*Aobs^2)^2*(4*b*Aobs+a*Bobs)")

_EMIT_WHOLE = _EMIT.replace(
    'line = next(l for l in source.splitlines() if "/* made */" in l and "obs_sens_c" in l)\n',
    "",
).replace(', "|", line.strip()', "")


@pytest.mark.parametrize("net", ["NESTED", "FACTORS"])
def test_a_model_with_a_tied_sum_inside_another_has_one_source(tmp_path, net):
    """Six processes, six seeds, one combined source, for a rate law with a
    tied sum under a power in each of two terms, and for one with two sums
    that hold a tie as factors of one product. On main each has two."""
    text = globals()[net]
    assert text.count("Bobs") >= 2 and "^2.0" in text
    path = tmp_path / "model.net"
    path.write_text(text)
    emitted = set()
    for seed in _SEEDS:
        done = subprocess.run(
            [sys.executable, "-c", _EMIT_WHOLE, str(path)],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        emitted.add(done.stdout.strip().splitlines()[-1])
    assert len(emitted) == 1, emitted


def test_arguments_from_a_set_have_one_order_from_any():
    """``ordered_args`` and ``in_order`` from each of the six orders three
    items can be handed over in, two of which tie and one of which holds a
    tied sum: one order."""
    import itertools

    from bngsim import _term_order

    a, b, X = sp.symbols("a b X")
    tied = 2.0 * a * X**1.0 + 2 * b * X
    items = [X**2.0 + 1, X**2 + 1.0, sp.exp(tied)]
    for order in (_term_order.ordered_args, _term_order.in_order):
        assert len({tuple(order(p)) for p in itertools.permutations(items)}) == 1


@pytest.mark.parametrize("kind", ["Min", "Max", "And", "Or"])
def test_the_key_of_a_set_like_operation_is_one_whichever_way_its_arguments_are_kept(kind):
    """sympy keeps the arguments of ``Min``, ``Max``, ``And`` and ``Or`` in the
    order ``ordered`` gave them from a set, and its key for the operation is
    made of them in that order. With two arguments that tie, the same
    operation has two keys there and one here."""
    from bngsim import _term_order

    x, y = sp.symbols("X Y")
    if kind in ("Min", "Max"):
        first, second = x**2.0 + 1, x**2 + 1.0
    else:
        first, second = 2.0 * x**2 > y, 2 * x**2.0 > y
    cls = getattr(sp, kind)
    one, other = sp.Basic.__new__(cls, first, second), sp.Basic.__new__(cls, second, first)
    assert one.args == other.args[::-1]
    assert one.sort_key() != other.sort_key()
    _term_order._forget()
    assert _term_order.stable_key(one) == _term_order.stable_key(other)


@pytest.mark.parametrize(
    ("text", "c", "exprtk"),
    [
        ("1 - 4*p", "1.0 - 4.0*p", "1 - 4*p"),
        ("-x*y/2", "-x*y/2.0", "-x*y/2"),
        ("-2*x/p**2", "-2.0*x/((p)*(p))", "-2*x/((p)^(2))"),
        (
            "x/2 - (y - z)/(2*(1 - 4*p))",
            "x/2.0 - (y - z)/(2.0 - 8.0*p)",
            "x/2 - (y - z)/(2 - 8*p)",
        ),
        ("-(y - z)/2", "-y/2.0 + z/2.0", "-y/2 + z/2"),
        ("-1.5*x*(y + z)/(p + 1)", "-1.5*x*(y + z)/(p + 1.0)", "-1.5*x*(y + z)/(p + 1)"),
    ],
)
def test_a_product_with_a_sign_is_printed_as_it_was(text, c, exprtk):
    """Control. The text the two printers of the Jacobian give a product with
    a negative coefficient, which is main's. sympy hands such a coefficient
    over as -1 and its size, and a product given back to its printer that way
    is printed ``-1.0*4.0*p``: a first version of the factor order did, and
    changed the sensitivity source of 117 corpus models of 203."""
    from bngsim._jacobian import sympy_to_c, sympy_to_exprtk

    expr = sp.sympify(text)
    assert sympy_to_c(expr, lambda name: name) == c
    assert sympy_to_exprtk(expr) == exprtk


def test_a_product_that_was_never_evaluated_is_printed_in_the_order_it_was_built():
    """Control. sympy prints a product built with ``evaluate=False``, one with
    a number that is not its first argument, as its ``args`` have it, and the
    emitters build such products (a rate times ``1.0`` times a power). They
    are printed as they were: sorting their factors as the others' are
    changed the source of 71 corpus models of 203."""
    from bngsim._jacobian import sympy_to_c, sympy_to_exprtk

    p, q, x, y = sp.symbols("p q x y")
    built = [
        (
            sp.Mul(p, q, sp.Integer(1), x**y, evaluate=False),
            "p*q*1.0*pow(x, y)",
            "p*q*1*((x)^(y))",
        ),
        (sp.Mul(sp.S.One, y, x, evaluate=False), "1.0*y*x", "1*y*x"),
        (sp.Mul(y, 2, x, evaluate=False), "y*2.0*x", "y*2*x"),
    ]
    for expr, c, exprtk in built:
        assert sympy_to_c(expr, lambda name: name) == c
        assert sympy_to_exprtk(expr) == exprtk


def test_expressions_with_no_tie_are_printed_as_sympy_prints_them():
    """Control (of the module against sympy's own). Four hundred expressions
    with no tie in them, built at random from
    sums, products, powers, ``Max`` and products that were never evaluated,
    with negative, rational and floating coefficients: the key, the C text
    and ``srepr`` are sympy's own. (Its own are the same under every seed for
    these, which is what makes them something to compare with.) A product
    with several numbers in it, ``1*1*(-1.0)*p``, is among them: sympy takes
    the sign off and builds the product again before it orders the factors,
    and doing that twice lost one of the ones to the coefficient."""
    import random

    from bngsim import _term_order

    rng = random.Random(550)
    a, b, c, k, x, y = sp.symbols("a b c k x y")
    atoms = [a, b, c, k, x, y, sp.Integer(2), sp.Integer(-3), sp.Rational(1, 3), sp.Float(2.5)]
    atoms += [sp.Float(-1.5), sp.Integer(1)]

    def build(depth=0):
        r = rng.random()
        if depth > 3 or r < 0.3:
            return rng.choice(atoms)
        if r < 0.5:
            return build(depth + 1) + build(depth + 1)
        if r < 0.7:
            return build(depth + 1) * build(depth + 1)
        if r < 0.8:
            return sp.Mul(*(build(depth + 1) for _ in range(3)), evaluate=False)
        if r < 0.9:
            return build(depth + 1) ** rng.choice([2, -1, 3])
        if r < 0.95:
            return sp.Max(build(depth + 1), build(depth + 1))
        return sp.exp(build(depth + 1))

    cases = [sp.Mul(1, 1, sp.Float(-1.0), a, b, 1 / (1 + x), evaluate=False)]
    while len(cases) < 401:
        try:
            cases.append(build())
        except (ValueError, TypeError):  # a Max of what cannot be compared
            continue
    for expr in cases:
        if not isinstance(expr, sp.Basic) or expr.has(sp.zoo, sp.nan):
            continue
        assert _term_order.stable_key(expr) == expr.sort_key(), expr
        assert _term_order.ccode(expr) == sp.ccode(expr), expr
        assert _term_order.srepr(expr) == sp.srepr(expr), expr


# ── The factors of a product, and who is given them ──────────────────────────
#
# sympy's ``_print_Mul`` asks the product for its factors, so the method it
# calls is what answers with this module's order: for a ``_print_Mul`` on a
# thread where one of this module's printers is printing, and for no one else.


class _Asked(Exception):
    """This module's order of factors was asked for."""


def _printing_with(monkeypatch, inside):
    """Run ``inside()`` from within one of this module's printers, with this
    module's order of factors replaced by one that raises."""
    from bngsim import _term_order
    from sympy.printing.repr import ReprPrinter

    def raises(expr):
        raise _Asked

    class Hooked(_term_order.SeedFreeTermOrder, ReprPrinter):
        def _print_Symbol(self, expr):
            return str(inside())

    monkeypatch.setattr(_term_order, "ordered_factors", raises)
    return Hooked().doprint(sp.Symbol("hook"))


def test_a_printer_of_this_module_is_given_this_modules_factors(monkeypatch):
    """The wiring, each way: a product printed by one of this module's
    printers has its factors from this module, and one printed by sympy's own
    printer has sympy's."""
    from bngsim import _term_order

    a, b, c = sp.symbols("a b c")
    product = a * (b + c) * (a + c)
    theirs = sp.srepr(product)
    monkeypatch.setattr(
        _term_order, "ordered_factors", lambda expr: (_ for _ in ()).throw(_Asked())
    )
    with pytest.raises(_Asked):
        _term_order.srepr(product)
    with pytest.raises(_Asked):
        _term_order.ccode(product)
    assert sp.srepr(product) == theirs and sp.ccode(product) == "a*(a + c)*(b + c)"


def test_a_product_asked_for_its_factors_by_anything_else_is_answered_by_sympy(monkeypatch):
    """While a printer of this module is printing, a product asked for its
    factors by something that is no ``_print_Mul`` has sympy's own: a direct
    call, and ``sort_key``, whose result sympy keeps for the rest of the
    process. A first version changed the method for every caller while a
    product was being printed, and a key computed then was served afterwards
    (6f8fba16 for f36b7c69 under seed 0). Nor is an order asked for by name
    this module's, from a ``_print_Mul`` or not."""
    a, b, c = sp.symbols("a b c")
    product = sp.Mul(2, a, (b + c) ** 2, (1.0 * a + c), evaluate=False)
    fresh = sp.Mul(3, a, (b + 2 * c) ** 2, (1.0 * b + c), evaluate=False)
    want = product.as_ordered_factors(), product.as_ordered_factors(order="rev-lex")
    got = []

    def inside():
        def _print_Mul():  # an order by name, asked for from a function of this name
            return fresh.as_ordered_factors(order="lex")

        _print_Mul()
        got.append((product.as_ordered_factors(), product.as_ordered_factors(order="rev-lex")))
        got.append(fresh.sort_key())
        return "done"

    assert _printing_with(monkeypatch, inside) == "done"
    assert got == [want, fresh.sort_key()]


def test_a_print_that_raises_leaves_no_printer_inside(monkeypatch):
    """A printer that is left by an exception, a ``KeyboardInterrupt``
    included, is no longer inside: sympy's own printer on the same thread has
    sympy's factors afterwards. (The first version put sympy's method back on
    the way out, and an interrupt between its two steps left it changed: 2 of
    147.)"""
    from bngsim import _term_order

    a, b, c = sp.symbols("a b c")
    product = a * (b + c) * (a + c)
    for error in (ValueError, KeyboardInterrupt):

        def inside(error=error):
            raise error

        with pytest.raises(error):
            _printing_with(monkeypatch, inside)
        assert getattr(_term_order._here, "printing", 0) == 0
        assert sp.srepr(product).startswith("Mul(")  # ordered_factors raises if it is asked


def test_another_threads_printing_is_not_answered(monkeypatch):
    """While one thread is inside a printer of this module, sympy's own
    printer on another thread has sympy's factors."""
    import threading

    a, b, c = sp.symbols("a b c")
    product = a * (b + c) * (a + c)
    seen = []

    def inside():
        other = threading.Thread(target=lambda: seen.append(sp.srepr(product)))
        other.start()
        other.join(60)
        return "done"

    assert _printing_with(monkeypatch, inside) == "done"
    assert seen == [sp.srepr(product)]


def test_a_nest_as_deep_as_main_prints_is_printed():
    """Control. A product of a sum of a product, 175 deep, which main prints:
    from a script it stops at 197, and under pytest, which is some frames
    down already, at 190. The version of this change that wrapped
    ``_print_Mul`` stopped at 165 from a script: a frame more for each product
    on the way down."""
    from bngsim._jacobian import sympy_to_c

    x, a = sp.symbols("x a")
    expr = x
    for i in range(175):
        expr = sp.Symbol(f"k{i % 7}") * (expr + a)
    text = sympy_to_c(expr, lambda name: name)
    assert text is not None and text.count("(") >= 175


# ── A count that rounding left beside itself ─────────────────────────────────


def test_a_count_within_rounding_is_one_answer_from_every_symbol():
    """``num = 3·term`` with floats in the exponents: against ``r`` the ratio
    of the slopes is 3, and against ``q``, where 3·0.35 was multiplied out, it
    is 3.0000000000000004 (or 2.9999999999999996). That was no count, so the
    answer was (3, 0) or None by which symbol was asked first: 52 of 2,800
    pairs of exponents. Every order of the symbols gives (3, 0)."""
    import itertools

    from bngsim._jacobian import _whole_power_offset, _whole_power_offset_asking

    p, q, r, t = sp.symbols("p q r t")
    term = -sp.Piecewise((p, t <= 1.0), (7.3, True)) + 3.5 + (0.00105 - 0.35 / q) / r
    num = 3 * term
    against_q = sp.cancel(sp.diff(num, q) / sp.diff(term, q))
    assert against_q.is_number and float(against_q) != 3.0 and abs(float(against_q) - 3) < 1e-14
    answers = {
        _whole_power_offset_asking(num, term, list(order), sp)
        for order in itertools.permutations([p, q, r, t])
    }
    assert answers == {(3, 0)}
    assert _whole_power_offset(num, term, sp) == (3, 0)


def test_a_ratio_that_is_beside_a_count_and_is_not_it_is_refused():
    """Control. A ratio of 3.000001 is no count, and one that rounding would
    take for 3 is refused by what is left over: ``3·term + 1e-12·q`` keeps q."""
    from bngsim._jacobian import _whole_power_offset

    q, r = sp.symbols("q r")
    term = 3.5 + (0.00105 - 0.35 / q) / r
    assert _whole_power_offset(3.000001 * term, term, sp) is None
    assert _whole_power_offset(3 * term + 1e-12 * q, term, sp) is None


def test_what_is_within_rounding_of_a_count():
    """A number within 1e-9 of a count, 1 or more, is that count, and nothing
    else is one."""
    from bngsim._jacobian import _count_within_rounding

    assert _count_within_rounding(sp.Float(3.0000000000000004)) == 3
    assert _count_within_rounding(sp.Float(2.9999999999999996)) == 3
    assert _count_within_rounding(sp.Float(3.00001)) is None
    assert _count_within_rounding(sp.Float(0.4)) is None
    assert _count_within_rounding(sp.Symbol("q")) is None


# ── A comoving case, under every seed ────────────────────────────────────────

_COMOVING = """begin parameters
    1 k0    0.1
    2 k1    2.0
    3 a     1.1
    4 on    3.0
    5 D     4.0
    6 kdeg  0.3
    7 _rateLaw1 1
    8 w01   2.0
    9 w1    1.0
end parameters
begin functions
    1 s() (t-on/(w01-w1))/D
    2 prod() k0+if(t>=on/(w01-w1),if(t<=(on/(w01-w1)+D),k1*s()*((1-s())^(a-1)),0),0)
end functions
begin species
    1 X() 0
    2 Tc() 0
end species
begin reactions
    1 0 1 prod
    2 1 0 kdeg
    3 0 2 _rateLaw1
end reactions
begin groups
    1 t 2
end groups
"""

_EMIT_SENS = """
import hashlib, logging, sys
logging.disable(logging.CRITICAL)
import bngsim
from bngsim import _codegen as cg
core = bngsim.Model.from_net(sys.argv[1])._core
source = cg.generate_sens_from_model(core, functional=True, emit_term_scale=True)
print(hashlib.sha256(source.encode()).hexdigest(), source.count("*c_out ="))
"""


def test_a_model_has_its_comoving_case_under_every_hash_seed(tmp_path):
    """A window whose onset is ``on/(w01 - w1)``. The shift of its crossings
    in ``on`` is ``1/(w01 - w1)``, which ``cancel`` returned as that or as
    ``-1/(-w01 + w1)`` by the seed, sympy's order of the two names being a
    tie. Under the second sympy does not come back from the law's conditional
    and the plan was dropped, so the model had its two comoving cases under
    seeds 6 and 7 and none under 0 to 5, where dX/dD was the plain column,
    0.37% off."""
    path = tmp_path / "model.net"
    path.write_text(_COMOVING)
    emitted = {}
    for seed in (0, 1, 6, 7):
        done = subprocess.run(
            [sys.executable, "-c", _EMIT_SENS, str(path)],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        emitted[seed] = done.stdout.strip().splitlines()[-1]
    assert len(set(emitted.values())) == 1, emitted
    assert emitted[0].split()[1] == "2", emitted


# ── What the emitters sort by ────────────────────────────────────────────────


def test_no_emitter_sorts_by_a_key_of_sympys_that_can_follow_the_seed():
    """``sp.srepr`` and ``default_sort_key`` of an expression that holds a
    tied sum follow the seed, so neither is a key to put a set of expressions
    in order by where the order is written out. The emitters sorted the
    conditionals of an exponent, the relationals of a law and the shifts of a
    parameter by ``srepr``, and the steps of a staircase by
    ``default_sort_key``. (A symbol's ``srepr`` is its name, and may be
    sorted by.)"""
    import ast
    import inspect

    from bngsim import _codegen, _jacobian, _switch_sensitivity

    found = []
    for module in (_codegen, _jacobian, _switch_sensitivity):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name not in ("sorted", "sort", "min", "max"):
                continue
            for keyword in node.keywords:
                if keyword.arg != "key":
                    continue
                text = ast.unparse(keyword.value)
                by_symbol_name = "s.name, sp.srepr(s)" in text
                if ("sp.srepr" in text or "default_sort_key" in text) and not by_symbol_name:
                    found.append(f"{module.__name__}:{node.lineno}: {text}")
    assert found == [
        # Guards that are one clock against one number: no sum to tie.
        "bngsim._switch_sensitivity:" + found[0].split(":")[1] + ": sp.srepr"
    ], found
    # And what they are sorted by: nothing else of sympy's (``str`` and
    # ``sort_key`` follow the seed as ``srepr`` does).
    assert inspect.getsource(_codegen).count("key=_term_order.srepr") == 4
    assert inspect.getsource(_switch_sensitivity).count("_term_order.in_order(by_count[n])") == 1


def test_the_cache_does_not_serve_a_source_built_before_the_orders_changed():
    """The version the codegen cache is keyed by is past the one main has."""
    from bngsim import _codegen

    assert int(_codegen._CODEGEN_VERSION) >= 39
