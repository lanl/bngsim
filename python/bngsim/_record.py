"""What a Result records of the run that made it (issue #1703).

:attr:`bngsim.Result.record` is built here from three things the run leaves on
the Result: what the Simulator was asked and what it did (``_run_info``, a
small dict the run fills in), the parameter values and absolute tolerances it
started with (two arrays), and what the model is (:class:`ModelInfo`, made once
per model and shared by its clones). The dict itself is built when it is read,
so a ``run_until`` loop of short steps pays for none it does not read.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bngsim._model import Model
    from bngsim._result import Result

#: Incremented by a change that removes or renames a key, or changes what one
#: holds. A key added is no change of version.
SCHEMA_VERSION = 1

#: The methods whose amounts are numbers of molecules.
STOCHASTIC = frozenset({"ssa", "psa", "nfsim", "rulemonkey"})

#: ``solver_stats["linear_solver"]``, by name (``LinearSolverKind``,
#: include/bngsim/result.hpp).
LINEAR_SOLVERS = {0: "dense", 1: "klu", 2: "lapack_dense"}

#: The parts of a record that a run, a squeeze or a file holds; the rest are
#: read from the Result's own fields whenever the record is built.
_PARTS = ("software", "model", "run", "backend", "time", "columns")


class ModelInfo(NamedTuple):
    """What a record says of the model, made once per model (:func:`model_info`)."""

    #: ``{"format", "path", "sha256", ...}`` as :meth:`Model.describe` gives it.
    source: dict[str, Any]
    #: Loaded from SBML or Antimony, whose species are amounts in compartments.
    sbml: bool
    #: Each SBML species by the model's name: (compartment, declared as an
    #: amount, substance unit id). Empty for every other model.
    species: dict[str, tuple[str, bool, str | None]]
    #: Each SBML observable that reports one species alone, and that species.
    observables: dict[str, str]
    #: Each SBML compartment: (size unit id, whether a rule or event moves it).
    compartments: dict[str, tuple[str | None, bool]]
    time_units: str | None
    #: The parameters a record lists, those that are not synthesized slots, and
    #: where each is in ``core.param_names``.
    param_names: tuple[str, ...]
    param_index: NDArray[np.intp]
    #: ``len(core.param_names)`` when this was made.
    n_params: int


def model_info(model: Model) -> ModelInfo:
    """*model*'s :class:`ModelInfo`, made on the first run and kept."""
    info = model._record_info
    if info is not None:
        return info
    core = model._core
    source = (
        dict(model._source)
        if model._source is not None
        else {"format": None, "path": None, "sha256": None}
    )
    names = list(core.param_names)
    internal = list(getattr(core, "param_is_internal", [False] * len(names)))
    keep = [i for i, hidden in enumerate(internal) if not hidden]
    species: dict[str, tuple[str, bool, str | None]] = {}
    observables: dict[str, str] = {}
    compartments: dict[str, tuple[str | None, bool]] = {}
    time_units = None
    record = model._sbml_record
    if record is not None:
        units = record.get("units") or {}
        species_units = units.get("species", {})
        species_names = list(core.species_names)
        cd = core.codegen_data()
        amount_valued = {
            species_names[i]: bool(sp.get("amount_valued")) for i, sp in enumerate(cd["species"])
        }
        for name, comp in record["species"].items():
            species[name] = (comp, amount_valued.get(name, False), species_units.get(name))
        for obs in cd["observables"]:
            entries = list(obs["entries"])
            if len(entries) == 1 and float(entries[0][1]) == 1.0:
                read = species_names[int(entries[0][0])]
                if read == obs["name"] and read in species:
                    observables[obs["name"]] = read
        comp_units = units.get("compartments", {})
        for name, entry in record["compartments"].items():
            compartments[name] = (comp_units.get(name), bool(entry["size_set_by"]))
        time_units = units.get("time")
    info = ModelInfo(
        source=source,
        sbml=record is not None,
        species=species,
        observables=observables,
        compartments=compartments,
        time_units=time_units,
        param_names=tuple(names[i] for i in keep),
        param_index=np.asarray(keep, dtype=np.intp),
        n_params=len(names),
    )
    model._record_info = info
    return info


def parameter_values(model: Model) -> tuple[ModelInfo, NDArray[np.float64]]:
    """*model*'s :class:`ModelInfo` and the values of the parameters it lists,
    as they stand now."""
    info = model_info(model)
    values = model._core.get_param_values()
    if values.shape[0] != info.n_params:  # pragma: no cover - parameters are fixed at build
        model._record_info = None
        info = model_info(model)
    return info, values[info.param_index]


def atol_given(atol: Any) -> Any:
    """The ``atol=`` a caller passed, as JSON can hold it: ``None`` where none
    was, a number, a word (``"auto"``, ``"tracking"``), a list, or the
    ``repr`` of a tolerance object."""
    if atol is None or isinstance(atol, (str, bool)):
        return atol
    if isinstance(atol, (int, float, np.floating, np.integer)):
        return float(atol)
    if isinstance(atol, (list, tuple, np.ndarray)):
        return [float(x) for x in np.asarray(atol, dtype=float).ravel()]
    return repr(atol)


def _size_of(
    comp: str, params: dict[str, float | None], fallback: float | None, varies: bool
) -> float | None:
    if varies:
        return None
    if comp in params:
        return params[comp]
    return fallback


def columns(result: Result, method: str | None) -> dict[str, Any] | None:
    """What each species and observable column of *result* holds."""
    info = result._model_info
    if info is None or method is None:
        return None
    stochastic = method in STOCHASTIC
    # A network's numbers are what its file holds, which states no unit, and a
    # stochastic method reads them as molecules. An SBML model's state that is
    # no species (a size or parameter a rule moves) is no amount at all.
    plain = None if info.sbml else ("count" if stochastic else None)
    vf = result._species_volume_factors
    # One size per species; a stack holds one list per row, and takes its
    # sizes from the parameters its rows agree on instead.
    flat: list[float] | None = vf if vf and not isinstance(vf[0], list) else None  # type: ignore[assignment]
    params = run_parameters(result) or {}
    species: list[dict[str, Any]] = []
    sizes: dict[str, float | None] = {}
    for i, name in enumerate(result._species_names):
        meta = info.species.get(name)
        if meta is None:
            species.append(
                {"name": name, "quantity": plain, "compartment": None, "substance_units": None}
            )
            continue
        comp, _amount, units = meta
        species.append(
            {
                "name": name,
                "quantity": "concentration",
                "compartment": comp,
                "substance_units": units,
            }
        )
        if comp not in sizes and flat is not None and i < len(flat):
            sizes[comp] = float(flat[i])
    observables: list[dict[str, Any]] = []
    for name in result._observable_names:
        read = info.observables.get(name)
        if read is None:
            quantity = plain
            observables.append({"name": name, "quantity": quantity, "compartment": None})
            continue
        comp, amount, _units = info.species[read]
        # The file's own quantity: an amount (a count, read stochastically)
        # where it declares the species in amounts.
        quantity = ("count" if stochastic else "amount") if amount else "concentration"
        observables.append({"name": name, "quantity": quantity, "compartment": comp})
    compartments = {
        name: {
            "size": _size_of(name, params, sizes.get(name), varies),
            "size_varies": varies,
            "units": units,
        }
        for name, (units, varies) in info.compartments.items()
    }
    return {"species": species, "observables": observables, "compartments": compartments}


def run_parameters(result: Result) -> dict[str, float | None] | None:
    """The parameter values *result*'s run started with, by name: ``None``
    for one the rows of a stack started with different values of."""
    held = result._run_params
    if held is None:
        return None
    names, values = held
    return dict(zip(names, (None if v is None else float(v) for v in values), strict=True))


def live_parts(result: Result) -> dict[str, Any]:
    """The parts of *result*'s record its run left on it."""
    from bngsim._build_provenance import _build_commit
    from bngsim._version import __version__

    info = result._model_info
    run_info = result._run_info
    run = None
    backend = None
    if run_info is not None:
        run = {k: copy.deepcopy(v) for k, v in run_info.items() if k != "backend"}
        atol = result._run_atol
        if run.get("atol") is not None:
            run["atol"]["per_species"] = None if atol is None else [float(x) for x in atol]
        run["parameters"] = run_parameters(result)
        backend = copy.deepcopy(run_info.get("backend"))
    return {
        "software": {"bngsim": __version__, "build_commit": _build_commit()},
        "model": None if info is None else copy.deepcopy(info.source),
        "run": run,
        "backend": backend,
        "time": None if info is None else {"units": info.time_units},
        "columns": columns(result, None if run_info is None else run_info.get("method")),
    }


def record(result: Result, *, start_state: bool = True) -> dict[str, Any]:
    """:attr:`Result.record`; ``start_state=False`` leaves the start vector out,
    as :meth:`Result.summary` does."""
    parts = frozen_parts(result) or {}
    out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    out.update((k, parts.get(k)) for k in _PARTS)
    run = out["run"]
    if run is not None:
        run["seed"] = result._seed
        run["started_from"] = result._started_from
        run["start_state_sha256"] = result.start_state_sha256
        if start_state:
            held = result._start_state
            run["start_state"] = None if held is None else [float(x) for x in held]
    out["notices"] = [n.as_dict() for n in result._notices]
    return out


def frozen_parts(result: Result) -> dict[str, Any] | None:
    """The parts of *result*'s record a file keeps, or ``None`` where it has
    none."""
    if result._record_frozen is not None:
        return copy.deepcopy(result._record_frozen)
    if result._run_info is None and result._model_info is None:
        return None
    return live_parts(result)


_DIFFER = object()


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _agree(values: list[Any]) -> Any:
    """What *values* agree on: a dict key by key, and anything else whole,
    ``None`` where they differ."""
    first = values[0]
    if all(isinstance(v, dict) for v in values):
        keys = list(dict.fromkeys(k for v in values for k in v))
        return {k: _agree([v.get(k, _DIFFER) for v in values]) for k in keys}
    if any(v is _DIFFER for v in values):
        return None
    return first if all(_same(v, first) for v in values[1:]) else None


def _same_model(a: ModelInfo | None, b: ModelInfo | None) -> bool:
    """Whether two records describe one model: the same object, or two made
    alike (rows run on clones made before their parent's was)."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    return all(
        getattr(a, f) == getattr(b, f) for f in ModelInfo._fields if f != "param_index"
    ) and np.array_equal(a.param_index, b.param_index)


def _same_array(arrays: list[Any]) -> Any:
    """The array every entry of *arrays* holds value for value, else ``None``."""
    first = arrays[0]
    if first is None:
        return None
    for a in arrays[1:]:
        if a is None or not np.array_equal(
            np.asarray(a, dtype=float), np.asarray(first, dtype=float), equal_nan=True
        ):
            return None
    return first


def squeeze_into(out: Result, results: list[Result]) -> None:
    """Give *out*, the stack of *results*, the record parts they agree on: each
    value they all hold alike, key by key, and ``None`` where they differ (a
    batch row's own parameter, its wall time). Nothing where a row recorded no
    run."""
    if not results or any(r._run_info is None for r in results):
        return
    if any(r._record_frozen is not None for r in results):
        # Rows loaded from files: merge what each file holds.
        out._record_frozen = _agree(
            [copy.deepcopy(r._record_frozen) or live_parts(r) for r in results]
        )
        return
    first = results[0]._model_info
    if all(_same_model(r._model_info, first) for r in results):
        out._model_info = first
    out._run_info = _agree([r._run_info for r in results])
    out._run_atol = _same_array([r._run_atol for r in results])
    held = [r._run_params for r in results]
    if all(h is not None and h[0] == held[0][0] for h in held):  # type: ignore[index]
        names = held[0][0]  # type: ignore[index]
        stack = np.vstack([np.asarray(h[1], dtype=np.float64) for h in held])  # type: ignore[index]
        alike = np.all((stack == stack[0]) | (np.isnan(stack) & np.isnan(stack[0])), axis=0)
        out._run_params = (
            names,
            [float(v) if a else None for v, a in zip(stack[0], alike, strict=True)],
        )
