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
    #: The text the model was loaded from, where it was loaded from a string:
    #: what :meth:`Result.rerun` builds it again from. Not in the record.
    source_text: str | bytes | None = None
    #: The calls that changed the model's equations or data since its load
    #: (``Model._edits``), which its source does not hold.
    edits: tuple[str, ...] = ()
    #: The sha256 of the model's table functions' data (:func:`tables_sha256`):
    #: a table read from a file of its own is in no source's hash.
    tables_sha256: str | None = None
    #: Which parameters were attached to an expression at load
    #: (``Model._loaded_attachment``).
    loaded_attachment: tuple[bool, ...] = ()


class RunCapture(NamedTuple):
    """What a run leaves on its Result for the record (``Result._run_info``):
    the values as the run had them, made into the record's ``run`` and
    ``backend`` dicts only when the record is read (:func:`run_and_backend`),
    so a run_until loop of short steps pays for a tuple and no dicts."""

    method: str
    requested_method: str
    t_start: float
    t_end: float
    n_points: int
    sample_times: Any
    start: str | None
    rtol: float | None
    atol_given: Any
    atol_scalar: float | None
    atol_decades: float | None
    max_steps: int | None
    max_step: float | None
    steady_state: bool
    steady_state_tol: float | None
    carry_sensitivities: bool
    sensitivity_params: tuple[str, ...]
    sensitivity_ic: tuple[str, ...]
    sensitivity_method: str | None
    poplevel: float | None
    stop_conditions: tuple[str, ...]
    xml: dict[str, Any] | None
    timeout: float | None
    wall_time_s: float
    rhs: str | None
    jacobian: str | None
    jacobian_requested: str | None
    jacobian_retried: bool | None
    linear_solver: int | None
    linear_solver_requested: str | None
    sensitivity_rhs: str | None
    propensities: str | None
    #: The event state the run started from, ``(core.event_carry(),
    #: core.ssa_counts_stored)``, as :func:`start_carry` takes it.
    start_carry: Any = None
    #: Whether :meth:`Model.declare_ic_sensitivity` had declared rows the run
    #: seeded its sensitivities from, which no record holds.
    declared_ic_sensitivity: bool = False
    #: Of a :meth:`Result.rerun`: ``{"record_sha256", "changed"}``.
    rerun_of: dict[str, Any] | None = None
    #: The Simulator's ``codegen=``, ``strict_ssa=`` and ``reaction_stats=``.
    codegen_requested: bool | None = None
    strict_ssa: bool = True
    reaction_stats: bool = False
    #: Which parameters were attached to an expression when the run started
    #: (:func:`attachment`); empty where none was at load.
    attachment: tuple[bool, ...] = ()


def _num(x: Any, kind: type = float) -> Any:
    return None if x is None else kind(x)


def run_and_backend(info: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The record's ``run`` and ``backend`` parts of a :class:`RunCapture`,
    or of the dict form a stack of them merged into."""
    if isinstance(info, dict):
        return copy.deepcopy(info["run"]), copy.deepcopy(info["backend"])
    c: RunCapture = info
    run = {
        "method": c.method,
        "requested_method": c.requested_method,
        "t_start": float(c.t_start),
        "t_end": float(c.t_end),
        "n_points": int(c.n_points),
        "sample_times": None if c.sample_times is None else [float(t) for t in c.sample_times],
        "start": c.start,
        "rtol": _num(c.rtol),
        "atol": None
        if c.atol_scalar is None
        else {
            "given": copy.deepcopy(c.atol_given),
            "scalar": float(c.atol_scalar),
            "tracking_decades": float(c.atol_decades or 0.0),
        },
        "max_steps": _num(c.max_steps, int),
        "max_step": _num(c.max_step),
        "steady_state": bool(c.steady_state),
        "steady_state_tol": _num(c.steady_state_tol),
        "carry_sensitivities": bool(c.carry_sensitivities),
        "sensitivity_params": list(c.sensitivity_params),
        "sensitivity_ic": list(c.sensitivity_ic),
        "sensitivity_method": c.sensitivity_method,
        "poplevel": _num(c.poplevel),
        "stop_conditions": list(c.stop_conditions),
        "xml": None if c.xml is None else dict(c.xml),
        "timeout": _num(c.timeout),
        "wall_time_s": float(c.wall_time_s),
        "start_carry": start_carry(c.start_carry),
        "strict_ssa": bool(c.strict_ssa),
        "reaction_stats": bool(c.reaction_stats),
        "declared_ic_sensitivity": bool(c.declared_ic_sensitivity),
        "rerun_of": copy.deepcopy(c.rerun_of),
    }
    backend = {
        "rhs": c.rhs,
        "jacobian": c.jacobian,
        "jacobian_requested": c.jacobian_requested,
        "jacobian_retried": c.jacobian_retried,
        "linear_solver": None if c.linear_solver is None else LINEAR_SOLVERS.get(c.linear_solver),
        "linear_solver_requested": c.linear_solver_requested,
        "rhs_requested": c.codegen_requested,
        "sensitivity_rhs": c.sensitivity_rhs,
        "propensities": c.propensities,
    }
    return run, backend


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
        source_text=model._source_text,
        edits=tuple(model._edits),
        tables_sha256=tables_sha256(model),
        loaded_attachment=tuple(model._loaded_attachment),
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


def tables_sha256(model: Model) -> str | None:
    """The sha256 of *model*'s table functions' data, as JSON, or ``None``
    where it has none."""
    tables = list(getattr(model._core, "table_function_tables", ()))
    if not tables:
        return None
    import hashlib
    import json

    keep = ("name", "index_kind", "method", "xs", "ys")
    text = json.dumps([{k: t.get(k) for k in keep} for t in tables], sort_keys=True, default=list)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attachment(model: Model) -> tuple[bool, ...]:
    """Which of *model*'s parameters are attached to an expression now, for a
    run's record; ``()`` where none was at load, or no write to its
    parameters may have pinned one since: the core's list costs a run some
    3 ns a parameter, which a model of 20,000 noticed."""
    if not model._loaded_attachment or not model._maybe_pinned:
        return ()
    return tuple(model._core.param_is_expression)


def pinned_parameters(info: ModelInfo | None, run_info: Any) -> list[str] | None:
    """The parameters a record lists that were attached to an expression at
    load and pinned to a number when the run started, or ``None`` where that
    was not recorded."""
    if isinstance(run_info, dict):
        return copy.deepcopy(run_info["run"].get("parameters_pinned"))
    if info is None:
        return None
    if not info.loaded_attachment:
        return []
    now, then = run_info.attachment, info.loaded_attachment
    return [
        info.param_names[k]
        for k, i in enumerate(info.param_index)
        if i < len(then) and i < len(now) and then[i] and not now[i]
    ]


def start_carry(state: Any) -> dict[str, Any] | None:
    """The event state a run starts from, as a record holds it: ``None`` where
    there is none (a fresh start), else ``{"events", "ssa_counts"}``.
    ``events`` is ``None`` or ``{"t", "triggers", "pending"}``, each pending
    execution ``{"event", "apply_time", "values"}``: the delayed executions not
    yet applied, which a run that starts at ``t`` applies, and each trigger's
    truth at ``t``. ``ssa_counts`` says the state is an SSA run's counts.
    *state* is ``(core.event_carry(), core.ssa_counts_stored)``."""
    if state is None or isinstance(state, dict):
        return copy.deepcopy(state)
    carry, counts = state
    events = None
    if carry is not None:
        t, triggers, pending = carry
        events = {
            "t": float(t),
            "triggers": [bool(x) for x in triggers],
            "pending": [
                {"event": int(i), "apply_time": float(at), "values": [float(v) for v in vals]}
                for i, at, vals in pending
            ],
        }
    if events is None and not counts:
        return None
    return {"events": events, "ssa_counts": bool(counts)}


def carry_state(held: dict[str, Any] | None) -> Any:
    """What ``core.set_event_carry_state`` takes for a record's
    ``start_carry`` (:func:`start_carry`'s inverse)."""
    if held is None:
        return None
    events = held["events"]
    carry = None
    if events is not None:
        carry = (
            events["t"],
            list(events["triggers"]),
            [(p["event"], p["apply_time"], list(p["values"])) for p in events["pending"]],
        )
    return (carry, [], -math.inf, bool(held["ssa_counts"]))


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
        run, backend = run_and_backend(run_info)
        atol = result._run_atol
        if run.get("atol") is not None:
            run["atol"]["per_species"] = None if atol is None else [float(x) for x in atol]
        run["parameters"] = run_parameters(result)
        run["parameters_pinned"] = pinned_parameters(info, run_info)
    model = None
    if info is not None:
        model = copy.deepcopy(info.source)
        model["edits"] = list(info.edits)
        model["tables_sha256"] = info.tables_sha256
    return {
        "software": {"bngsim": __version__, "build_commit": _build_commit()},
        "model": model,
        "run": run,
        "backend": backend,
        "time": None if info is None else {"units": info.time_units},
        "columns": columns(result, None if run is None else run["method"]),
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
    # Of a stack, where its rows differ: the paths whose None says so.
    differs = result._record_differs
    out["rows_differ"] = None if differs is None else list(differs)
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


def _agree(values: list[Any], path: str, differ: list[str]) -> Any:
    """What *values* agree on: a dict key by key, and anything else whole,
    ``None`` where they differ, whose *path* is then added to *differ*."""
    first = values[0]
    if all(isinstance(v, dict) for v in values):
        keys = list(dict.fromkeys(k for v in values for k in v))
        return {
            k: _agree([v.get(k, _DIFFER) for v in values], f"{path}.{k}" if path else k, differ)
            for k in keys
        }
    if any(v is _DIFFER for v in values) or not all(_same(v, first) for v in values[1:]):
        differ.append(path)
        return None
    return first


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
    batch row's own parameter, its wall time), with where they differ
    (``_record_differs``). Nothing where a row recorded no run."""
    if not results:
        return
    differ: list[str] = []
    for key, attr in (
        ("run.seed", "_seed"),
        ("run.started_from", "_started_from"),
        ("run.start_state_sha256", "start_state_sha256"),
    ):
        if len({getattr(r, attr) for r in results}) > 1:
            differ.append(key)
    if "run.start_state_sha256" in differ:
        differ.append("run.start_state")
    first = results[0]._model_info
    if any(r._record_frozen is not None for r in results) or not all(
        _same_model(r._model_info, first) for r in results
    ):
        # Rows loaded from files, alone or beside live ones, or of different
        # models: what each record holds, merged key by key.
        parts = [frozen_parts(r) for r in results]
        if any(part is None for part in parts):
            return
        out._record_frozen = _agree(parts, "", differ)
        out._record_differs = tuple(dict.fromkeys(differ))
        return
    if any(r._run_info is None for r in results):
        return
    out._model_info = first
    both = [run_and_backend(r._run_info) for r in results]
    out._run_info = {
        "run": _agree([b[0] for b in both], "run", differ),
        "backend": _agree([b[1] for b in both], "backend", differ),
    }
    pins = [pinned_parameters(r._model_info, r._run_info) for r in results]
    if all(p == pins[0] for p in pins):
        out._run_info["run"]["parameters_pinned"] = pins[0]
    else:
        out._run_info["run"]["parameters_pinned"] = None
        differ.append("run.parameters_pinned")
    out._run_atol = _same_array([r._run_atol for r in results])
    if out._run_atol is None and any(r._run_atol is not None for r in results):
        differ.append("run.atol.per_species")
    held = [r._run_params for r in results]
    if all(h is not None and h[0] == held[0][0] for h in held):  # type: ignore[index]
        names = held[0][0]  # type: ignore[index]
        stack = np.vstack([np.asarray(h[1], dtype=np.float64) for h in held])  # type: ignore[index]
        alike = np.all((stack == stack[0]) | (np.isnan(stack) & np.isnan(stack[0])), axis=0)
        out._run_params = (
            names,
            [float(v) if a else None for v, a in zip(stack[0], alike, strict=True)],
        )
        compartments = first.compartments if first is not None else {}
        for name, a in zip(names, alike, strict=True):
            if not a:
                differ.append(f"run.parameters.{name}")
                if name in compartments:
                    differ.append(f"columns.compartments.{name}.size")
    out._record_differs = tuple(dict.fromkeys(differ))
