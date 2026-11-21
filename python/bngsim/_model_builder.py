"""Build a model in Python, one species, parameter and reaction at a time (issue #1706).

Every loader builds its model through the C++ ``ModelBuilder``.
:class:`ModelBuilder` is the same builder for a model written in Python: it
names species by name, gives each species an observable of the same name, and
returns a :class:`bngsim.Model` that runs like a loaded one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from bngsim._bngsim_core import ModelBuilder as _CoreBuilder
from bngsim._model import Model

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ModelBuilder:
    """Build a reaction network in Python.

    Add parameters, species, reactions, and optionally observables, functions,
    and events, then call :meth:`build` for a :class:`bngsim.Model`. Every
    species gets an observable of the same name, so an expression can name a
    species.

    Examples
    --------
    A + B <-> C:

    >>> import bngsim
    >>> b = bngsim.ModelBuilder()
    >>> b.add_parameter("kf", 0.01)
    >>> b.add_parameter("kr", 0.1)
    >>> b.add_species("A", 100)
    >>> b.add_species("B", 80)
    >>> b.add_species("C", 0)
    >>> b.add_reaction(["A", "B"], ["C"], rate="kf")
    >>> b.add_reaction(["C"], ["A", "B"], rate="kr")
    >>> model = b.build()
    >>> model.species_names
    ['A', 'B', 'C']
    """

    def __init__(self) -> None:
        self._core = _CoreBuilder()
        self._species: dict[str, int] = {}
        self._names: set[str] = set()
        # Names a reaction's rate can name directly: parameters and functions.
        self._rate_names: set[str] = set()
        self._n_hidden = 0
        # Each hidden function's name, and the rate= or law= it was made from.
        self._hidden: dict[str, str] = {}
        self._built = False

    def _new_name(self, name: str, what: str) -> None:
        if self._built:
            from bngsim._exceptions import BngsimError

            raise BngsimError(
                "this ModelBuilder has built its model; start a new one",
                code="usage.not_supported",
            )
        if not isinstance(name, str) or not _NAME.match(name):
            from bngsim._exceptions import InvalidModelError

            raise InvalidModelError(
                f"{what} name {name!r} is not a name: use letters, digits and "
                "underscores, starting with a letter or underscore"
            )
        if name in self._names:
            from bngsim._exceptions import InvalidModelError

            raise InvalidModelError(f"{name!r} is already defined")
        self._names.add(name)

    def _index(self, name: str) -> int:
        try:
            return self._species[name]
        except KeyError:
            from bngsim._exceptions import InvalidModelError

            raise InvalidModelError(
                f"no species named {name!r}; add it with add_species"
            ) from None

    def _hidden_name(self, stem: str) -> str:
        while True:
            self._n_hidden += 1
            name = f"_{stem}{self._n_hidden}"
            if name not in self._names:
                self._names.add(name)
                return name

    def _hidden_function(self, expression: str, what: str) -> str:
        name = self._hidden_name("rate")
        self._core.add_function(name, expression)
        self._hidden[name] = f"{what}={expression!r}"
        return name

    def add_parameter(self, name: str, value: float) -> None:
        """Add a parameter with a value."""
        self._new_name(name, "parameter")
        self._core.add_parameter(name, float(value))
        self._rate_names.add(name)

    def add_species(self, name: str, initial: float = 0.0, *, fixed: bool = False) -> None:
        """Add a species with its initial amount, and an observable of the same name.

        A ``fixed`` species keeps its amount: reactions read it but don't change it.
        """
        self._new_name(name, "species")
        index = self._core.add_species(name, float(initial), fixed)
        self._species[name] = index
        self._core.add_observable(name, [(index, 1.0)])

    def add_observable(self, name: str, species: Iterable[str] | Mapping[str, float]) -> None:
        """Add an observable: a sum of species, or a weighted sum given as a dict."""
        self._new_name(name, "observable")
        weights = species if isinstance(species, Mapping) else {s: 1.0 for s in species}
        self._core.add_observable(name, [(self._index(s), float(w)) for s, w in weights.items()])

    def add_function(self, name: str, expression: str) -> None:
        """Add a function of parameters, species, observables, other functions and ``time()``."""
        self._new_name(name, "function")
        self._core.add_function(name, expression)
        self._rate_names.add(name)

    def add_reaction(
        self,
        reactants: Iterable[str],
        products: Iterable[str],
        *,
        rate: str | float | None = None,
        law: str | float | None = None,
    ) -> None:
        """Add a reaction. Give either ``rate`` or ``law``.

        ``rate`` is a mass-action rate constant: the reaction's rate is ``rate``
        times the amount of each reactant, so ``A + A`` at ``rate="k"`` runs at
        ``k*A*A`` (``k*A*(A - 1)`` in a stochastic run, as stochastic mass
        action counts pairs). It can be a number, name a parameter, function,
        observable or species, or be an expression.

        ``law`` is the whole rate, as an expression, not multiplied by the
        reactants: ``law="Vmax*S/(Km + S)"``. A stochastic run uses it as
        written, so a law that stays above zero when too few reactants remain
        can fire anyway and drive a count below zero. Use ``rate`` for mass
        action.

        A species listed twice is a reactant or product twice. An empty list is
        nothing: ``add_reaction([], ["A"], rate="k")`` makes A at rate ``k``.
        """
        if (rate is None) == (law is None):
            from bngsim._exceptions import UsageError

            if rate is None:  # neither given
                raise UsageError(
                    "give add_reaction either rate= or law=, not both",
                    code="usage.missing_argument",
                    details={"argument": "rate"},
                )
            raise UsageError(
                "give add_reaction either rate= or law=, not both",
                code="usage.conflicting_arguments",
            )
        if isinstance(rate, (int, float)):
            rate = repr(float(rate))
        if isinstance(law, (int, float)):
            law = repr(float(law))
        if self._built:
            from bngsim._exceptions import BngsimError

            raise BngsimError(
                "this ModelBuilder has built its model; start a new one",
                code="usage.not_supported",
            )
        r = [self._index(s) for s in _names(reactants, "reactants")]
        p = [self._index(s) for s in _names(products, "products")]
        if law is not None:
            self._core.add_reaction(
                r, p, "functional", self._hidden_function(law, "law"), apply_species_factor=False
            )
            return
        assert rate is not None
        if rate in self._rate_names:
            # A parameter or a function: the core reads either by its name.
            self._core.add_reaction(r, p, "elementary", rate)
        else:
            self._core.add_reaction(r, p, "functional", self._hidden_function(rate, "rate"))

    def add_event(
        self,
        trigger: str,
        assignments: Mapping[str, str | float],
        *,
        delay: float = 0.0,
        name: str | None = None,
    ) -> None:
        """Add an event: when ``trigger`` becomes true, set species to new values.

        ``trigger`` is a condition such as ``"time() >= 5"``. Each assignment
        sets a species to an expression, which can read the species' own value:
        ``{"A": "A + 100"}`` adds 100 to A.
        """
        if self._built:
            from bngsim._exceptions import BngsimError

            raise BngsimError(
                "this ModelBuilder has built its model; start a new one",
                code="usage.not_supported",
            )
        if name is None:
            name = self._hidden_name("event")
        else:
            self._new_name(name, "event")
        self._core.add_event(
            name,
            trigger,
            [(self._index(s), str(value)) for s, value in assignments.items()],
            delay=float(delay),
        )

    def build(self) -> Model:
        """The model. A builder builds one model."""
        if self._built:
            from bngsim._exceptions import BngsimError

            raise BngsimError(
                "this ModelBuilder has built its model; start a new one",
                code="usage.not_supported",
            )
        self._built = True
        try:
            core = self._core.build()
        except RuntimeError as exc:
            message = str(exc)
            for name, source in self._hidden.items():
                if f"'{name}'" in message:
                    symbol = re.search(r"Undefined symbol: '([^']+)'", message)
                    what = f"{symbol.group(1)!r}, which is" if symbol else "something"
                    from bngsim._exceptions import InvalidModelError

                    raise InvalidModelError(
                        f"a reaction's {source} names {what} not defined: add it as a "
                        "parameter, species, observable or function"
                    ) from None
            raise
        return Model(_core=core)


def _names(species: Iterable[str], what: str) -> list[str]:
    if isinstance(species, str):
        from bngsim._exceptions import ArgumentTypeError

        raise ArgumentTypeError(
            f"{what} is a list of species names, such as ['A', 'B'], not a string"
        )
    return list(species)
