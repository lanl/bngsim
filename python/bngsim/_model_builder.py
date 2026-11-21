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
        self._n_hidden = 0
        self._built = False

    def _new_name(self, name: str, what: str) -> None:
        if self._built:
            raise RuntimeError("this ModelBuilder has built its model; start a new one")
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(
                f"{what} name {name!r} is not a name: use letters, digits and "
                "underscores, starting with a letter or underscore"
            )
        if name in self._names:
            raise ValueError(f"{name!r} is already defined")
        self._names.add(name)

    def _index(self, name: str) -> int:
        try:
            return self._species[name]
        except KeyError:
            raise ValueError(f"no species named {name!r}; add it with add_species") from None

    def _hidden_function(self, expression: str) -> str:
        self._n_hidden += 1
        name = f"_rate{self._n_hidden}"
        self._core.add_function(name, expression)
        return name

    def add_parameter(self, name: str, value: float) -> None:
        """Add a parameter with a value."""
        self._new_name(name, "parameter")
        self._core.add_parameter(name, float(value))

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

    def add_reaction(
        self,
        reactants: Iterable[str],
        products: Iterable[str],
        *,
        rate: str | None = None,
        law: str | None = None,
    ) -> None:
        """Add a reaction. Give either ``rate`` or ``law``.

        ``rate`` is a mass-action rate constant: the reaction's rate is ``rate``
        times the amount of each reactant, so ``A + A`` at ``rate="k"`` runs at
        ``k*A*A``. It can name a parameter or a function, or be an expression.

        ``law`` is the whole rate, as an expression, not multiplied by the
        reactants: ``law="Vmax*S/(Km + S)"``.

        A species listed twice is a reactant or product twice. An empty list is
        nothing: ``add_reaction([], ["A"], rate="k")`` makes A at rate ``k``.
        """
        if (rate is None) == (law is None):
            raise ValueError("give add_reaction either rate= or law=, not both")
        if self._built:
            raise RuntimeError("this ModelBuilder has built its model; start a new one")
        r = [self._index(s) for s in _names(reactants, "reactants")]
        p = [self._index(s) for s in _names(products, "products")]
        if law is not None:
            self._core.add_reaction(
                r, p, "functional", self._hidden_function(law), apply_species_factor=False
            )
            return
        assert rate is not None
        if _NAME.match(rate) and rate in self._names and rate not in self._species:
            # A parameter or a function: the core reads either by its name.
            self._core.add_reaction(r, p, "elementary", rate)
        else:
            self._core.add_reaction(r, p, "functional", self._hidden_function(rate))

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
            raise RuntimeError("this ModelBuilder has built its model; start a new one")
        if name is None:
            self._n_hidden += 1
            name = f"_event{self._n_hidden}"
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
            raise RuntimeError("this ModelBuilder has built its model; start a new one")
        self._built = True
        return Model(_core=self._core.build())


def _names(species: Iterable[str], what: str) -> list[str]:
    if isinstance(species, str):
        raise TypeError(f"{what} is a list of species names, such as ['A', 'B'], not a string")
    return list(species)
