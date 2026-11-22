"""Small example models that ship with BNGsim, with their exact solutions.

Each model is small, shows one feature, and has an exact solution, so you can
run it and check BNGsim's answer on your own machine::

    import numpy as np
    import bngsim

    model = bngsim.examples.load("binding")
    result = bngsim.Simulator(model).run(t_span=(0, 10), n_points=101)
    exact = bngsim.examples.exact("binding", result.time)
    np.allclose(result.observables["C"], exact["C"], rtol=1e-6)  # True

=============  ============================================  ==========================
Name           Model                                         Shows
=============  ============================================  ==========================
binding        A + B <-> C (A(b) + B(a) <-> A(b!1).B(a!1))  every format; network-free
decay          A -> 0                                        ODE, SSA, sensitivities
chain          A -> B -> C                                   a decay chain
birth_death    0 -> A -> 0, large numbers                    SSA vs. PSA
bolus          A -> 0, with a dose at t = 5                  events
schlogl        a bistable switch                             steady states
=============  ============================================  ==========================

Formats: ``"net"``, ``"bngl"``, ``"bng-xml"`` (for ``method="nf"`` or
``"rm"``, as ``xml_path=``), ``"sbml"``, ``"antimony"`` and ``"python"`` (a
:class:`bngsim.ModelBuilder` script). :func:`formats` says which each model
has. Every format of a model is the same model, with the same observables.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from bngsim._model import Model
from bngsim.examples import _exact

_DIR = Path(__file__).parent / "models"

#: Each format's file suffix, in the order :func:`load` tries them.
_SUFFIX = {
    "net": ".net",
    "sbml": ".sbml",
    "python": ".py",
    "antimony": ".ant",
    "bngl": ".bngl",
    "bng-xml": "_bng.xml",
}

_NAMES = ("binding", "decay", "chain", "birth_death", "bolus", "schlogl")


def names() -> list[str]:
    """The example models' names."""
    return list(_NAMES)


def _check(name: str) -> None:
    if name not in _NAMES:
        from bngsim._exceptions import UsageError

        raise UsageError(
            f"no example named {name!r}; the examples are {', '.join(_NAMES)}",
            details={"name": name},
        )


def describe(name: str) -> str:
    """What the model is, in a sentence or two."""
    _check(name)
    doc = importlib.import_module(f"bngsim.examples.models.{name}").__doc__ or ""
    return doc.strip()


def formats(name: str) -> list[str]:
    """The formats this model comes in."""
    _check(name)
    return [f for f, suffix in _SUFFIX.items() if (_DIR / f"{name}{suffix}").is_file()]


def path(name: str, format: str = "net") -> Path:
    """The model's file in one format, for tools that want a path."""
    _check(name)
    if format not in _SUFFIX:
        from bngsim._exceptions import UsageError

        raise UsageError(f"format {format!r} is not one of {', '.join(_SUFFIX)}")
    file = _DIR / f"{name}{_SUFFIX[format]}"
    if not file.is_file():
        from bngsim._exceptions import UsageError

        raise UsageError(
            f"{name!r} does not come in {format!r}; it comes in {', '.join(formats(name))}"
        )
    return file


def load(name: str, format: str | None = None) -> Model:
    """The model as a :class:`bngsim.Model`.

    With no ``format``, the first of ``"net"``, ``"sbml"`` and ``"python"`` the
    model comes in, none of which needs an extra. ``"antimony"`` needs
    ``bngsim[antimony]``, and ``"bngl"`` needs ``bngsim[bngl]`` and Perl.
    ``"bng-xml"`` is not a model to load: pass ``path(name, "bng-xml")`` to a
    network-free simulator as ``xml_path``.
    """
    _check(name)
    if format is None:
        format = next(f for f in ("net", "sbml", "python") if f in formats(name))
    if format == "bng-xml":
        from bngsim._exceptions import UsageError

        raise UsageError(
            "a BNG-XML file is read by the network-free engines, not loaded as a "
            f"model: load the network with bngsim.examples.load({name!r}, 'net'), then "
            f"bngsim.Simulator(model, method='nf', xml_path=bngsim.examples.path({name!r}, "
            "'bng-xml'))"
        )
    if format == "python":
        path(name, "python")
        built: Model = importlib.import_module(f"bngsim.examples.models.{name}").build()
        return built
    return Model.load(path(name, format))


def exact(name: str, t, **params: float) -> dict[str, np.ndarray]:
    """The exact solution at times ``t``: each observable as an array.

    ``params`` overrides the model's parameters and initial amounts, by name
    (``bngsim.examples.exact("decay", t, k=1.0)``); the defaults are the
    values in the model's files. For a model whose reactions are all first
    order (``decay``, ``chain``, ``birth_death``, ``bolus``), this is also the
    mean of a stochastic (SSA) run.
    """
    _check(name)
    if name not in _exact.SOLUTIONS:
        from bngsim._exceptions import UsageError

        raise UsageError(
            f"{name!r} has no closed-form time course; "
            f"bngsim.examples.steady_states({name!r}) has its exact steady states"
        )
    return _exact.SOLUTIONS[name](np.asarray(t, dtype=float), **_params(name, params))


def steady_states(name: str, **params: float) -> list[dict[str, float]]:
    """The exact steady states: each a dict of observable values and ``"stable"``."""
    _check(name)
    return _exact.steady_states(name, _params(name, params))


def _params(name: str, params: dict[str, float]) -> dict[str, float]:
    defaults = _exact.DEFAULTS[name]
    unknown = set(params) - set(defaults)
    if unknown:
        from bngsim._exceptions import UsageError

        raise UsageError(
            f"{name!r} has no parameter {', '.join(sorted(unknown))}; "
            f"its parameters are {', '.join(defaults)}"
        )
    return {**defaults, **params}


__all__ = ["describe", "exact", "formats", "load", "names", "path", "steady_states"]
