"""The order a sum's terms are printed in, the same for every hash seed (issue #550).

Every emitter prints a sympy ``Add`` through ``Printer._as_ordered_terms``,
which for the default order is ``Expr.as_ordered_terms``. That sorts the terms
by their monomials over the sum's generators, and it takes the order of the
generators from ``Expr.as_terms``, which collects them in a ``set`` and sorts
the set by ``default_sort_key``.

Two generators that differ only in the type of a number in them have keys
that are not equal and of which neither is less than the other: ``F`` and the
``F**1.0`` that differentiating ``F**2.0`` leaves, or ``1 + x`` and
``1.0 + x``. ``sorted`` leaves such a pair in the order the set gave it, which
is the order of their hashes, and a ``Symbol`` hashes its name: the order
follows ``PYTHONHASHSEED``, which CPython draws for each process. The
monomials follow the generators and the printed order of the terms follows the
monomials, so one model had two sensitivity sources, 8 lines of 2,094 apart
(MODEL0847869198: seeds 0, 1, 2, 5 and 6 one, seeds 3, 4 and 7 the other). The
two are the same sum, added up in a different order: they differ in the last
bits of a column, and which of them a cache holds is whichever was built
first.

``ordered_terms`` is ``as_ordered_terms`` with such a tie broken by the
generators' ``srepr``, which does not read a hash. A sum with no tie among
its generators comes back exactly as sympy orders it, so its printed text is
what it was.
"""

from __future__ import annotations


def _untied(gens: list) -> list:
    """``gens``, which sympy sorted by ``default_sort_key``, with each run of
    tied neighbours put in the order of their ``srepr``."""
    from sympy import default_sort_key, srepr

    keys = [default_sort_key(g) for g in gens]
    fixed = list(gens)
    n = len(gens)
    i = 0
    while i < n:
        j = i + 1
        # Sorted, so a key that is not above the one before it is tied with it.
        while j < n and not keys[j - 1] < keys[j]:
            j += 1
        if j - i > 1:
            fixed[i:j] = sorted(gens[i:j], key=srepr)
        i = j
    return fixed


def ordered_terms(expr, order=None) -> list:
    """The terms of ``expr`` as ``expr.as_ordered_terms(order=order)`` gives
    them, with the generators its monomials are taken over in an order that
    does not depend on the hash seed.

    This is ``Expr.as_ordered_terms`` line for line but for the generators, so
    that ``as_terms``, the expensive part, is still called once a sum.
    """
    from sympy import Add, Mul
    from sympy.core.numbers import Number, NumberSymbol

    if order is None and expr.is_Add:
        # sympy's special case: a positive number and a negative multiple of
        # something, ``1 - 2*x``, are left in that order.
        def is_not_a_number(x):
            return not isinstance(x, (Number, NumberSymbol))

        add_args = sorted(Add.make_args(expr), key=is_not_a_number)
        if (
            len(add_args) == 2
            and isinstance(add_args[0], (Number, NumberSymbol))
            and isinstance(add_args[1], Mul)
        ):
            mul_args = sorted(Mul.make_args(add_args[1]), key=is_not_a_number)
            if (
                len(mul_args) == 2
                and isinstance(mul_args[0], Number)
                and add_args[0].is_positive
                and mul_args[0].is_negative
            ):
                return add_args

    key, reverse = expr._parse_order(order)
    terms, gens = expr.as_terms()
    if any(term.is_Order for term, _ in terms):
        # A series remainder, which no rate law holds: sympy's own path.
        return expr.as_ordered_terms(order=order)

    fixed = _untied(gens)
    if any(a is not b for a, b in zip(fixed, gens, strict=True)):
        place = {g: i for i, g in enumerate(fixed)}
        to = [place[g] for g in gens]
        moved = []
        for term, (coeff, monom, ncpart) in terms:
            at = [0] * len(gens)
            for i, power in zip(to, monom, strict=True):
                at[i] = power
            moved.append((term, (coeff, tuple(at), ncpart)))
        terms = moved
    # Stable, and ``terms`` is in the order of ``expr.args``, which sympy
    # sorts by structure: two terms of one monomial and one coefficient stay
    # in that order.
    return [term for term, _ in sorted(terms, key=key, reverse=reverse)]


class SeedFreeTermOrder:
    """Mix into a sympy printer, ahead of it, to print each sum's terms in
    :func:`ordered_terms`' order."""

    def _as_ordered_terms(self, expr, order=None):
        order = order or self.order  # type: ignore[attr-defined]
        if order in ("old", "none"):
            return super()._as_ordered_terms(expr, order=order)  # type: ignore[misc]
        return ordered_terms(expr, order)


_c99_printer: list = []


def ccode(expr) -> str:
    """``sympy.ccode(expr)``, with each sum's terms in :func:`ordered_terms`'
    order."""
    if not _c99_printer:
        from sympy.printing.c import C99CodePrinter

        class _C99(SeedFreeTermOrder, C99CodePrinter):
            pass

        _c99_printer.append(_C99)
    return _c99_printer[0]().doprint(expr)
