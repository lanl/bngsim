"""The model's parameter values, handed to a network-free backend (issue #1360).

NFsim and RuleMonkey are built from the BNG XML alone, so a ``Model.set_param``
reached neither: with the only phosphorylation rule switched off (``kph=0``),
SSA gave RP = 0 and nf 121, rm 127, with no warning. Each run now hands the
backend, as its own parameter overrides, every parameter the model holds at a
value other than the one it was loaded with. Both backends re-derive the
parameters an override feeds and the seed amounts that read them (NFsim, issue
#29; RuleMonkey, GH #44).

Other than the one it was loaded with, and not other than the XML's: BNG2.pl's
``writeXML`` prints ``value=`` as ``%.8g`` (``BNGOutput.pm``), so an unchanged
``NA = 6.02214076e23`` reads ``6.0221408e+23`` there, and a derived ``1/(NA*V)``
differs from the model's in its ninth digit. Compared with the XML, every such
parameter would be pushed, and a model nobody changed would run on other rate
constants than it ran on before, and than BNG2.pl's ``simulate_nf`` runs it on.
Compared with the XML's value as printed, a change in the ninth digit would be
dropped. The model's own values at load (``Model._loaded_params``) are exact.

Only the parameters the XML declares are pushed: a backend refuses a name it
does not have. When the XML cannot be read, which those are is not known, and a
run that changed any is refused rather than run on the XML's values.

A derived parameter the model holds at a value written to it goes with them,
even at its value at load. ``Model.set_param`` on one overrides its expression,
and the backend, handed the parameters the expression reads, would re-derive it
from them: with ``kph = 0`` and ``kph2 = kph*kv*NA*V/3`` then written back to its
value at load, the model held ``kph2 = 1/6`` and SSA ended at RP = 59, while nf
and rm were handed ``kph = 0`` alone and ran no RP.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from bngsim._exceptions import ModelError
from bngsim._model import _names_in, _unclamped


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def xml_parameter_names(xml_path: str | Path) -> frozenset[str] | None:
    """The ids of the model's ``<Parameter>`` elements in *xml_path*, or
    ``None`` when the file cannot be parsed."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    return frozenset(
        name
        for lst in root.iter()
        if _local(lst.tag) == "ListOfParameters"
        for p in lst
        if _local(p.tag) == "Parameter" and (name := p.get("id"))
    )


def parameter_overrides(model: Any, xml_names: frozenset[str] | None) -> dict[str, float]:
    """The parameters of *model* the XML declares, with the model's values: each
    one the model holds at a value other than its value at load, and with them
    each derived one the model holds at a value written to it.

    Raises :class:`ModelError` when *xml_names* is ``None`` (the XML could not
    be read) and the model holds a parameter at a value other than its value at
    load."""
    core = model._core
    loaded = model._loaded_params
    names = list(core.param_names)
    values = [float(core.get_param(name)) for name in names]
    changed = {
        name: value
        for name, value, internal in zip(names, values, core.param_is_internal, strict=True)
        if not internal and loaded.get(name) != value
    }
    if xml_names is None:
        if changed:
            raise ModelError(
                "The network-free backend runs on the XML's parameters unless it is "
                f"handed the model's, and this model has changed {sorted(changed)}, but "
                "the XML could not be read to find which of them it declares. A run "
                "would drop every one (issue #1360). Pass an xml_path that is "
                "well-formed XML."
            )
        return {}
    out = {name: value for name, value in changed.items() if name in xml_names}
    if out:
        known = set(names)
        for name, value, derived, expression in zip(
            names, values, core.param_is_expression, core.param_expressions, strict=True
        ):
            # An overridden derived parameter: not derived now, while its
            # expression names another parameter (Model.param_is_expression).
            if name in xml_names and not derived and known.intersection(_names_in(expression)):
                out.setdefault(name, value)
    return out


# --------------------------------------------------------------------------- #
# Seed amounts (issue #1373)
# --------------------------------------------------------------------------- #
# The species-side twin of the above: a Model.set_concentration reached neither
# backend, so a run started from the XML's amounts (L0 = 300 for the 0 set).
# Each run now hands the backend the amount of every species the model holds at
# an amount other than its amount at load, by the id of the XML's <Species>
# with that BNGL name (BNG2.pl writes the same canonical string as the .net's
# species line, ``@c::`` included). The model's amount at load, not the XML's,
# for the reason given for parameters above.
#
# Not the same string for a fixed species: the .net loader drops its ``$`` and
# the XML keeps it, so the XML's names are read the way the loader reads them.
# Unmatched, a $X(a) set to 10 was refused as no seed species, and one set to 0
# dropped as one, so the run started from the XML's 50.
#
# And a species can sit at its amount at load while a parameter its seed reads
# is handed on changed: after save_concentrations(), or set_concentration of
# the same 300, set_param("L0", 0) leaves the model at L(r) = 300, as SSA starts
# it, and the backend, handed L0 = 0 alone, re-derived the seed as 0. So
# wherever a parameter is handed on, every seed species goes with it at the
# model's amount.


def xml_seed_species(xml_path: str | Path) -> dict[str, str | None] | None:
    """The model's seed species in *xml_path*: BNGL name, read as the .net
    loader reads it, -> ``<Species id=>``, or ``None`` for a name more than one
    seed species has. ``None`` when the file cannot be parsed."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except (ET.ParseError, OSError):
        return None
    out: dict[str, str | None] = {}
    for model in root:
        if _local(model.tag) != "model":
            continue
        for lst in model:
            if _local(lst.tag) != "ListOfSpecies":
                continue
            for sp in lst:
                name, sid = sp.get("name"), sp.get("id")
                if _local(sp.tag) != "Species" or not name or not sid:
                    continue
                name = _unclamped(name)
                out[name] = None if name in out else sid
    return out


def species_amounts(
    model: Any, seeds: dict[str, str | None] | None, method: str, *, every_seed: bool = False
) -> dict[str, float]:
    """``<Species id=>`` -> amount, for each species *model* holds at an amount
    other than its amount at load, and with *every_seed* (a parameter is handed
    on, which a seed may read) for every seed species. Raises ``ValueError`` for
    one the backend cannot be told: no seed species of that name (the backend
    starts it at 0), more than one, or an amount that rounds to no count.
    Raises :class:`ModelError` when *seeds* is ``None`` (the XML could not be
    read) and the model holds a species at an amount other than its amount at
    load."""
    names = model._core.species_names
    state = model._core.get_state()
    loaded = model._loaded_state
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        amount = float(state[i])
        changed = not (i < len(loaded) and loaded[i] == amount)
        if seeds is None:
            if changed:
                raise ModelError(
                    f"method={method!r} starts from the XML's species amounts unless it "
                    f"is handed the model's, and this model holds {name!r} at "
                    f"{amount:g}, but the XML could not be read to find its seed "
                    "species (issue #1373). Pass an xml_path that is well-formed XML."
                )
            continue
        sid = seeds.get(name)
        if not changed and not (every_seed and sid is not None):
            continue
        if sid is not None and math.isfinite(amount) and amount > -0.5:
            # Rounded half-up by the backend, as every seed is. Below 0 only by
            # what an ODE run can leave (-1e-17), which SSA rounds to 0 and the
            # backend refuses.
            out[sid] = max(amount, 0.0)
            continue
        if sid is None and name not in seeds and amount == 0.0:
            # Not a seed species, so the backend starts it at 0 as asked.
            continue
        if sid is not None:
            why = "it rounds to no number of molecules."
        elif name in seeds:
            why = (
                "more than one of the XML's seed species has that name. Network-free "
                "runs are seeded from the XML's <Species> list, by name."
            )
        else:
            # Most often a complex an SSA, PSA or ODE run on this same model left
            # behind: the run writes its end state into the model, and the XML,
            # written without the network, seeds only the seed species.
            why = (
                "the XML has no seed species of that name, so the run would start it "
                "at 0. Network-free runs are seeded from the XML's <Species> list. If "
                "an SSA, PSA or ODE run on this model left it there, give that run "
                "model.clone() so this model stays where it was, or call "
                "model.reset() first, which returns to the initial amounts (or to an "
                "unlabeled save_concentrations()). Otherwise write the XML with this "
                "species as a seed species, or start it from the network with "
                "method='ssa'."
            )
        raise ValueError(
            f"method={method!r} cannot start species {name!r} at the amount the model "
            f"holds, {amount:g}: {why}"
        )
    return out
