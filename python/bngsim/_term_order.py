"""The order sums, products and set-like operations are printed in, the same
for every hash seed (issue #550).

sympy prints the terms of an ``Add`` and the factors of a ``Mul`` in sorted
order, and keeps the arguments of ``Min``, ``Max``, ``And`` and ``Or`` sorted.
Each of those sorts can follow ``PYTHONHASHSEED``, which CPython draws for each
process, and all for one reason.

``Expr.as_ordered_terms`` sorts a sum's terms by their monomials over the
sum's generators, and takes the order of the generators from
``Expr.as_terms``, which collects them in a ``set`` and sorts the set by
``default_sort_key``. Two generators that differ only in the type of a number
in them have keys that are not equal and of which neither is less than the
other: ``F`` and the ``F**1.0`` that differentiating ``F**2.0`` leaves, or
``1 + x`` and ``1.0 + x``. ``sorted`` leaves such a pair in the order the set
gave it, which is the order of their hashes, and a ``Symbol`` hashes its name.
The monomials follow the generators and the printed order of the terms follows
the monomials, so one model had two sensitivity sources, 8 lines of 2,099 apart
(MODEL0847869198: seeds 0, 1, 2, 5 and 6 one, seeds 3, 4 and 7 the other). The
two are the same sum, added up in a different order: they differ in the last
bits of a column, and which of them a cache holds is whichever was built
first.

That is the order of one sum. But the sort key of a sum is made of its
ordered terms (``Expr.sort_key``), so the key of anything that holds such a
sum follows the seed as well, and with it every order that is taken from keys:
the generators of an enclosing sum, the factors of a product
(``Mul.as_ordered_factors``), and the arguments of ``Min``, ``Max``, ``And``
and ``Or``, which sympy puts through ``ordered`` from a set.
``k*(2.0*a*X**1.0 + 2*b*X)**2 + k*(2*a*X + c*Y)**2`` printed its two terms
either way round, and ``k*(2.0*a*X**1.0 + 2*b*X)*(2*b*X + c*Y)`` its two
bracketed factors. ``srepr`` is no way out: it prints a sum's terms in the
same order.

So the keys are computed here. :func:`stable_key` is sympy's ``sort_key``
with the orders inside it taken from this module, and the three orders are
taken from it:

- :func:`ordered_terms`, ``Expr.as_ordered_terms`` with the generators in an
  order that has no ties;
- :func:`ordered_factors`, ``Mul.as_ordered_factors``;
- :func:`ordered_args`, ``ordered`` for the arguments of a set-like
  operation.

Where no key in an expression follows the seed, each is the order sympy
gives, and the printed text is what it was.
"""

from __future__ import annotations

from functools import cmp_to_key, lru_cache

_MEMO = 1 << 17


def _cmp(a, b) -> int:
    return (a > b) - (a < b)


def _key_cmp(a, b) -> int:
    """Order two sort keys as tuples are ordered, with one difference: two
    entries that are not equal and of which neither is less than the other,
    ``Integer(1)`` and ``Float(1.0)``, are put in the order of their types'
    names and then of what they print as.

    Where ``a < b`` holds for the keys as they are, this agrees. Unlike
    ``<``, it is an order: with ``<`` alone, ``(1, x)`` is below ``(1, y)``
    and ``(1.0, z)`` is tied with both.
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        for x, y in zip(a, b, strict=False):
            c = _key_cmp(x, y)
            if c:
                return c
        return _cmp(len(a), len(b))
    try:
        if a == b:
            return 0
        if a < b:
            return -1
        if b < a:
            return 1
    except TypeError:
        pass
    from sympy import Basic

    def what(x):
        return type(x).__name__, srepr(x) if isinstance(x, Basic) else repr(x)

    return _cmp(what(a), what(b))


@lru_cache(maxsize=_MEMO)
def _follows_the_seed(expr) -> bool:
    """Whether sympy's sort key of ``expr`` can follow the hash seed: it holds
    a sum, or an operation whose arguments sympy keeps in a set."""
    from sympy import Add
    from sympy.core.operations import LatticeOp
    from sympy.functions.elementary.miscellaneous import MinMaxBase

    return expr.has(Add, LatticeOp, MinMaxBase)


@lru_cache(maxsize=_MEMO)
def stable_key(expr):
    """``expr.sort_key()``, with the orders inside it taken from this module.

    This is ``Expr.sort_key`` and ``Basic.sort_key`` line for line, but for
    three orders: the terms of a sum, the factors of a product, and the
    arguments of ``Min``, ``Max``, ``And`` and ``Or``. An expression that holds
    none of those has sympy's own key, and is not taken apart here.
    """
    from sympy import Basic, Expr, Function, S, default_sort_key
    from sympy.core.operations import LatticeOp
    from sympy.functions.elementary.miscellaneous import MinMaxBase

    if not isinstance(expr, Basic):
        return default_sort_key(expr)
    if not _follows_the_seed(expr):
        return expr.sort_key()
    own = type(expr).sort_key
    if isinstance(expr, Expr) and own is Expr.sort_key:
        coeff, inner = expr.as_coeff_Mul()
        if inner.is_Pow:
            base, exp = inner.as_base_exp()
            if base is S.Exp1:
                base, exp = Function("exp")(exp), S.One
            inner = base
        else:
            exp = S.One
        if inner.is_Dummy:
            args: tuple = (inner.sort_key(),)
        elif inner.is_Atom:
            args = (str(inner),)
        else:
            if inner.is_Add:
                parts = ordered_terms(inner)
            elif inner.is_Mul:
                parts = ordered_factors(inner)
            elif isinstance(inner, MinMaxBase):
                parts = ordered_args(inner.args)
            else:
                parts = inner.args
            args = tuple(stable_key(part) for part in parts)
        return inner.class_key(), (len(args), args), stable_key(exp), coeff
    if own is Basic.sort_key:
        parts = ordered_args(expr.args) if isinstance(expr, LatticeOp) else expr._sorted_args
        args = tuple(stable_key(part) if isinstance(part, Basic) else part for part in parts)
        return expr.class_key(), (len(args), args), S.One.sort_key(), S.One
    return expr.sort_key()


def _forget() -> None:
    """Drop what is memoized here (the keys and the default order of each
    sum's terms)."""
    for memo in (_follows_the_seed, stable_key, _default_order_terms):
        memo.cache_clear()


def _nodes(expr) -> float:
    """The number of nodes in ``expr`` as sympy's ``ordered`` counts them, a
    ``Float`` for half of one: what it sorts by ahead of the keys."""
    if getattr(expr, "is_Float", False):
        return 0.5
    return 1 + sum(_nodes(arg) for arg in getattr(expr, "args", ()))


def _in_order(items: list, keys: list) -> list:
    """``items`` sorted by ``keys`` in an order that does not depend on the
    order they are given in.

    Where each key is below the next, that is the order they are in, and the
    list comes back as it is: ``<`` being true agrees with :func:`_key_cmp`,
    which is an order, so the list is sorted by it. Otherwise two of them tie,
    and they are all sorted again by :func:`_key_cmp` and then by what they
    print as (:func:`srepr`). (A tie is not confined to its neighbours: ``<``
    on these keys is not transitive, and what ``sorted`` makes of such a list
    depends on all of the order it was given.)
    """
    try:
        if all(keys[i] < keys[i + 1] for i in range(len(keys) - 1)):
            return items
    except TypeError:
        pass

    def by_key_then_structure(i, j):
        return _key_cmp(keys[i], keys[j]) or _cmp(srepr(items[i]), srepr(items[j]))

    return [items[i] for i in sorted(range(len(items)), key=cmp_to_key(by_key_then_structure))]


def in_order(items) -> list:
    """``items`` in the order of their keys (:func:`stable_key`), with no
    ties: the same list from any order of the same items, a set's included."""
    items = list(items)
    return _in_order(items, [stable_key(item) for item in items])


def _untied(gens: list) -> list:
    """``gens``, which sympy sorted by ``default_sort_key`` from a set, in an
    order that does not depend on the set's."""
    return _in_order(gens, [stable_key(g) for g in gens])


def ordered_args(args) -> list:
    """The arguments of ``Min``, ``Max``, ``And`` or ``Or`` in the order
    sympy's ``ordered`` puts them in, by node count and then by key, with the
    keys from :func:`stable_key` and no ties."""
    items = list(args)
    return _in_order(items, [(_nodes(a), stable_key(a)) for a in items])


def ordered_factors(expr, whole_coefficient: bool = False) -> list:
    """``expr.as_ordered_factors()`` with the keys from :func:`stable_key`.

    The commutative factors are sorted from the order of ``expr.args``, which
    sympy fixes by structure and not by hash, so the sort itself is sympy's:
    two factors whose keys tie stay in that order, in every process.

    sympy hands a negative coefficient over as -1 and its size, ``-4*p`` as
    ``[-1, 4, p]``. ``whole_coefficient`` leaves it whole, for a product that
    is to be printed: with a number beside its first factor a printer takes
    it for one that was never evaluated, and prints ``-1*4*p``.
    """
    commutative, rest = expr.args_cnc(split_1=not whole_coefficient)
    commutative.sort(key=stable_key)
    return commutative + rest


def ordered_terms(expr, order=None) -> list:
    """The terms of ``expr`` as ``expr.as_ordered_terms(order=order)`` gives
    them, with the generators its monomials are taken over in an order that
    does not depend on the hash seed.

    This is ``Expr.as_ordered_terms`` line for line but for the generators, so
    that ``as_terms``, the expensive part, is still called once a sum (and,
    for the default order, once for the sum's key and its printing together).
    """
    if order is None:
        return list(_default_order_terms(expr))
    return _ordered_terms(expr, order)


@lru_cache(maxsize=_MEMO)
def _default_order_terms(expr) -> tuple:
    return tuple(_ordered_terms(expr, None))


def _ordered_terms(expr, order) -> list:
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
    if fixed is not gens and any(a is not b for a, b in zip(fixed, gens, strict=True)):
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


def _never_evaluated(expr) -> bool:
    """sympy's test, in ``StrPrinter._print_Mul``, for a product that was
    built unevaluated, ``p*q*1*x**n``: a number that is not its first
    argument, or a first argument of 1. It prints such a product as its
    ``args`` have it, which is the order it was built in."""
    from sympy import S
    from sympy.core.numbers import Number

    args = expr.args
    return args[0] is S.One or any(
        isinstance(a, Number) or (a.is_Pow and all(ai.is_Integer for ai in a.args))
        for a in args[1:]
    )


class SeedFreeTermOrder:
    """Mix into a sympy printer, ahead of it, to print each sum's terms, each
    product's factors and the arguments of ``Min``, ``Max``, ``And`` and
    ``Or`` in this module's orders.

    The printer is given ``order="none"``, under which sympy sorts nothing
    and prints a product's factors as its ``args`` have them: the product it
    is handed has them in :func:`ordered_factors`' order. (sympy's
    ``_print_Mul`` asks ``as_ordered_factors`` of a product it has rebuilt
    without its sign, which a subclass of ``Mul`` would not survive.)
    """

    def __init__(self, settings=None):
        settings = dict(settings or {})
        settings["order"] = "none"
        super().__init__(settings)

    def _as_ordered_terms(self, expr, order=None):
        return ordered_terms(expr)

    def _print_Mul(self, expr):
        from sympy import Mul
        from sympy.printing.str import StrPrinter

        theirs = getattr(super()._print_Mul, "__func__", None)  # type: ignore[misc]
        if theirs is StrPrinter._print_Mul and _never_evaluated(expr):
            # sympy prints such a product as its args have it, in no order of
            # its own, and so it is printed here.
            return super()._print_Mul(expr)  # type: ignore[misc]
        factors = Mul._from_args(ordered_factors(expr, whole_coefficient=True))
        if not factors.is_Mul:
            return self._print(factors)  # type: ignore[attr-defined]
        return super()._print_Mul(factors)  # type: ignore[misc]


_printers: dict = {}


def ccode(expr) -> str:
    """``sympy.ccode(expr)``, in this module's orders."""
    if "c99" not in _printers:
        from sympy.printing.c import C99CodePrinter

        class _C99(SeedFreeTermOrder, C99CodePrinter):
            def _print_Max(self, expr):
                return self._nest(expr, "fmax")

            def _print_Min(self, expr):
                return self._nest(expr, "fmin")

            def _print_And(self, expr):
                return self._joined(expr, "and")

            def _print_Or(self, expr):
                return self._joined(expr, "or")

            def _joined(self, expr, operator):
                # sympy's own shape, its arguments sorted by key.
                from sympy.printing.precedence import precedence

                level = precedence(expr)
                parts = (self.parenthesize(a, level) for a in in_order(expr.args))
                return f" {self._operators[operator]} ".join(parts)

            def _nest(self, expr, name):
                # sympy's own shape, fmax(a, fmax(b, c)), from the first
                # argument in.
                printed = [self._print(a) for a in ordered_args(expr.args)]
                out = printed[-1]
                for inner in reversed(printed[:-1]):
                    out = f"{name}({inner}, {out})"
                return out

        _printers["c99"] = _C99
    return _printers["c99"]().doprint(expr)


def srepr(expr) -> str:
    """``sympy.srepr(expr)``, in this module's orders: a description of an
    expression's structure that is the same in every process. sympy's own
    prints a sum's terms as ``as_ordered_terms`` gives them."""
    if "repr" not in _printers:
        from sympy.printing.repr import ReprPrinter

        class _Repr(SeedFreeTermOrder, ReprPrinter):
            def _print_Mul(self, expr, order=None):
                # As sympy's: the factors as as_ordered_factors hands them
                # over, a negative coefficient as -1 and its size.
                args = ", ".join(self._print(a) for a in ordered_factors(expr))
                return f"{type(expr).__name__}({args})"

            def _set_like(self, expr):
                args = ", ".join(self._print(a) for a in ordered_args(expr.args))
                return f"{type(expr).__name__}({args})"

            _print_And = _print_Or = _print_Min = _print_Max = _set_like

        _printers["repr"] = _Repr
    return _printers["repr"]().doprint(expr)
