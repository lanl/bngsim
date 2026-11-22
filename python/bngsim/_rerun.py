"""``Result.rerun()``: the run a Result records, made again (issue #1703).

A Result's record holds every input of its run: the model's source and its
sha256, the parameter values and the state the run started from, the event
state it continued, the method, the seed, the sample times, the tolerances as
the solver was given them and the code that ran. So "run it again" has one
reading on a Result, where on a Simulator it has several (from the initial
conditions or from where the last run started, with the same seed or a new
one, with the parameters as they were or as they are).

The rerun builds the model again from its source, a file whose sha256 must
still match or the text a string load kept, and never reads or changes the
model the run used. Where the record cannot promise the same run (a part it
lacks, a model changed on disk, a run it does not hold all of), the rerun
refuses with a code rather than run something else.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim import _notices, _record

if TYPE_CHECKING:
    from bngsim._model import Model
    from bngsim._result import Result

#: The settings a rerun can change, by the keyword :meth:`Simulator.run` or
#: :class:`Simulator` takes them as.
SETTINGS = (
    "t_span",
    "n_points",
    "sample_times",
    "seed",
    "rtol",
    "atol",
    "max_steps",
    "max_step",
    "timeout",
    "steady_state",
    "steady_state_tol",
    "method",
    "jacobian",
    "codegen",
)

#: The settings that are an ODE run's alone, left out where a rerun changes
#: the method away from ``"ode"`` and does not name them.
_ODE_ONLY = ("rtol", "atol", "max_steps", "max_step", "steady_state", "steady_state_tol")

#: Methods a record does not hold all of: a network-free run reads a BNG-XML
#: file and a session the record does not keep.
_NETWORK_FREE = frozenset({"nfsim", "rulemonkey"})


def _missing(part: str) -> Exception:
    from bngsim._exceptions import UsageError

    return UsageError(
        f"This result's record has no {part}, which a rerun needs to make the same run "
        "again: it was made by a call that records no run (compute_all_sensitivities, a "
        "network-free session) or loaded from a file written before bngsim recorded it. "
        "Run the model again with Simulator.run.",
        code="rerun.not_recorded",
        details={"missing": part},
        remedies=[],
    )


def _unsupported(reason: str, message: str, **details: Any) -> Exception:
    from bngsim._exceptions import UsageError

    return UsageError(
        message, code="rerun.unsupported", details={"reason": reason, **details}, remedies=[]
    )


def _record_sha256(rec: dict[str, Any]) -> str:
    """The sha256 of a record, as JSON with its keys sorted."""
    text = json.dumps(rec, sort_keys=True, default=repr, allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def source_text(result: Result) -> str | bytes | None:
    """The text *result*'s model was loaded from, where it came from a string."""
    if result._model_info is not None:
        return result._model_info.source_text
    return result._source_text


def _check_refusals(result: Result, rec: dict[str, Any]) -> None:
    """Refuse a result whose record cannot promise the same run."""
    if rec["rows_differ"] is not None:
        raise _unsupported(
            "stack",
            "This result is a stack of several runs (run_batch, run_replicates, "
            "parameter_scan), and a rerun makes one run. Rerun one row: each row's "
            "Result has a record of its own.",
        )
    run = rec["run"]
    if run is None:
        raise _missing("run")
    model = rec["model"]
    if model is None or model.get("format") is None:
        raise _missing("model source")
    if run["method"] in _NETWORK_FREE:
        raise _unsupported(
            "network_free",
            f"A {run['method']} run reads a BNG-XML file and keeps a session the record "
            "does not hold, so a rerun cannot promise the same run. Run it again with "
            "Simulator.run.",
            method=run["method"],
        )
    if run.get("stop_conditions"):
        raise _unsupported(
            "stop_conditions",
            "This run had stop conditions, which its record names but cannot rebuild "
            f"({run['stop_conditions']}). Register them on a Simulator and run it again.",
            stop_conditions=list(run["stop_conditions"]),
        )
    if run.get("carry_sensitivities"):
        raise _unsupported(
            "carry_sensitivities",
            "This run carried sensitivities from an earlier run (carry_sensitivities=True), "
            "whose dx/dθ its record does not hold.",
        )
    if run.get("declared_ic_sensitivity"):
        raise _unsupported(
            "declared_ic_sensitivity",
            "This run seeded its sensitivities from rows given to "
            "Model.declare_ic_sensitivity, which its record does not hold.",
        )
    edits = model.get("edits")
    if edits:
        raise _unsupported(
            "model_edited",
            f"The model was changed after its load by {', '.join(edits)}, which its "
            "source does not hold, so a model built from the source is not the one "
            "this run used.",
            edits=list(edits),
        )
    for key in ("parameters", "start_state", "t_start", "t_end", "n_points"):
        if run.get(key) is None:
            raise _missing(f"run.{key}")
    if "start_carry" not in run:
        raise _missing("run.start_carry")
    for name, value in run["parameters"].items():
        if value is None:
            raise _missing(f"run.parameters.{name}")
    if run["method"] == "ode":
        for key in ("rtol", "atol", "max_steps"):
            if run.get(key) is None:
                raise _missing(f"run.{key}")
        if rec["backend"] is None:
            raise _missing("backend")


def build_model(result: Result, source: dict[str, Any]) -> Model:
    """The model *source* describes, built again, its file's sha256 checked."""
    from bngsim._exceptions import ModelError
    from bngsim._model import Model

    fmt = source["format"]
    path = source.get("path")
    sha = source.get("sha256")
    sizes = source.get("compartment_sizes")
    if path is not None:
        try:
            data = Path(path).read_bytes()
        except OSError as e:
            raise ModelError(
                f"The model file this run used, {path}, cannot be read ({e}), so it cannot "
                "be run again.",
                code="rerun.model_missing",
                details={"path": path},
                remedies=[],
            ) from e
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha:
            raise ModelError(
                f"The model file this run used, {path}, has changed since the run "
                f"(sha256 {sha} then, {digest} now), so a rerun would run another model.",
                code="rerun.model_changed",
                details={"path": path, "recorded_sha256": sha, "sha256": digest},
                remedies=[],
            )
        if fmt == "net":
            return Model.from_net(path)
        if fmt == "bngl":
            return Model.from_bngl(path)
        if fmt == "sbml":
            return Model.from_sbml(path, compartment_sizes=sizes)
        if fmt == "antimony":
            return Model.from_antimony(path, compartment_sizes=sizes)
        raise _missing(f"model loader for format {fmt!r}")  # pragma: no cover
    text = source_text(result)
    if text is None:
        raise _missing("model source text")
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    if hashlib.sha256(data).hexdigest() != sha:  # pragma: no cover - kept beside its hash
        raise _missing("model source text matching its sha256")
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if fmt == "sbml":
        return Model.from_sbml_string(text, compartment_sizes=sizes)
    if fmt == "antimony":
        return Model.from_antimony_string(text, compartment_sizes=sizes)
    raise _missing(f"model loader for text of format {fmt!r}")  # pragma: no cover


def _put_start(model: Model, run: dict[str, Any]) -> None:
    """Give *model* the parameter values, state and event state *run* started
    with, and refuse where the model cannot hold them."""
    from bngsim._exceptions import ModelError

    core = model._core
    names = set(core.param_names)
    params: dict[str, float] = run["parameters"]
    unknown = sorted(n for n in params if n not in names)
    if unknown:
        raise ModelError(
            f"The model built from the recorded source has no parameter {unknown[0]!r}, "
            "which the run had, so it is not the model the run used.",
            code="rerun.model_changed",
            details={"parameters": unknown},
            remedies=[],
        )
    # Only the values that differ: a set_param pins a derived parameter's value
    # and detaches it from its expression.
    differ = {n: float(v) for n, v in params.items() if not _same(core.get_param(n), float(v))}
    for name in [n for n in model.primary_param_names if n in differ]:
        model.set_param(name, differ[name])
    left = {n: v for n, v in differ.items() if not _same(core.get_param(n), v)}
    if left:
        model.set_params(left)
    state = np.asarray(run["start_state"], dtype=np.float64)
    if state.shape[0] != model.n_species:
        raise ModelError(
            f"The model built from the recorded source has {model.n_species} species and "
            f"the run started from {state.shape[0]}, so it is not the model the run used.",
            code="rerun.model_changed",
            details={"n_species": model.n_species, "recorded": int(state.shape[0])},
            remedies=[],
        )
    now = np.asarray(core.get_state(), dtype=np.float64)
    if not np.array_equal(now, state, equal_nan=True):
        model.set_state(state)
    carry = _record.carry_state(run["start_carry"])
    if carry is not None:
        core.set_event_carry_state(carry)


def _simulator_options(
    run: dict[str, Any], backend: dict[str, Any] | None, method: str, changes: dict[str, Any]
) -> dict[str, Any]:
    """The :class:`Simulator` keywords that make *run* again under *method*."""
    same = method == run["method"]
    opts: dict[str, Any] = {}
    if method == "ode" and same and backend is not None:
        # The Jacobian the run used: a difference quotient where it retried on
        # one, or where the Simulator kept one from an earlier retry.
        used = backend.get("jacobian")
        requested = backend.get("jacobian_requested") or "auto"
        opts["jacobian"] = "fd" if used == "difference_quotient" else requested
        solver = backend.get("linear_solver_requested")
        opts["force_dense_linear_solver"] = solver == "dense"
        opts["force_sparse_linear_solver"] = solver == "sparse"
        if backend.get("rhs") == "interpreted":
            opts["codegen"] = False
    if method == "ode" and (run.get("sensitivity_params") or run.get("sensitivity_ic")):
        opts["sensitivity_params"] = list(run["sensitivity_params"]) or None
        opts["sensitivity_ic"] = list(run["sensitivity_ic"]) or None
        if run.get("sensitivity_method"):
            opts["sensitivity_method"] = run["sensitivity_method"]
    if method == "psa" and same and run.get("poplevel") is not None:
        opts["poplevel"] = run["poplevel"]
    for key in ("jacobian", "codegen"):
        if key in changes:
            opts[key] = changes[key]
    # A stochastic run's propensities, interpreted where its code was.
    if (
        same
        and "codegen" not in changes
        and backend is not None
        and backend.get("propensities") == "interpreted"
    ):
        opts["codegen"] = False
    return opts


def _run_options(run: dict[str, Any], method: str, changes: dict[str, Any]) -> dict[str, Any]:
    """The :meth:`Simulator.run` keywords that make *run* again under *method*."""
    from bngsim._simulator import _Recorded

    opts: dict[str, Any] = {"t_span": (run["t_start"], run["t_end"])}
    if run.get("sample_times") is not None:
        opts["sample_times"] = list(run["sample_times"])
    else:
        opts["n_points"] = run["n_points"]
    opts["seed"] = run.get("seed")
    if run.get("timeout") is not None:
        opts["timeout"] = run["timeout"]
    if method == "ode" and run["method"] == "ode":
        opts["rtol"] = run["rtol"]
        atol = run["atol"]
        opts["atol"] = _Recorded(
            float(atol["scalar"]),
            None if atol.get("per_species") is None else [float(x) for x in atol["per_species"]],
            float(atol.get("tracking_decades") or 0.0),
            atol.get("given"),
        )
        opts["max_steps"] = run["max_steps"]
        # None: no bound, which a max_step <= 0 asks for; left out, the run
        # would take the model's own bound (a periodic schedule's, GH #88).
        opts["max_step"] = run["max_step"] if run.get("max_step") is not None else 0.0
        opts["steady_state"] = bool(run.get("steady_state"))
        if run.get("steady_state_tol") is not None:
            opts["steady_state_tol"] = run["steady_state_tol"]
    for key in SETTINGS:
        if key in changes and key not in ("method", "jacobian", "codegen"):
            opts[key] = changes[key]
    if "t_span" in changes:
        opts["t_span"] = tuple(changes["t_span"])
    if "sample_times" in changes:
        opts.pop("n_points", None)
    elif "n_points" in changes:
        opts.pop("sample_times", None)
    if method != "ode":
        for key in _ODE_ONLY:
            if key not in changes:
                opts.pop(key, None)
    return opts


def _changed(changes: dict[str, Any]) -> dict[str, Any]:
    """*changes* as a record holds them."""
    out: dict[str, Any] = {}
    for key, value in changes.items():
        if key == "atol":
            out[key] = _record.atol_given(value)
        elif key in ("t_span", "sample_times"):
            out[key] = [float(x) for x in value]
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


#: What a rerun's notices compare between the two records' backends.
_BACKEND = ("rhs", "jacobian", "linear_solver", "sensitivity_rhs", "propensities")


@_notices.collected
def rerun(result: Result, **changes: Any) -> Result:
    """:meth:`Result.rerun`."""
    from bngsim._error_codes import pass_argument
    from bngsim._exceptions import UsageError
    from bngsim._simulator import Simulator

    unknown = sorted(set(changes) - set(SETTINGS))
    if unknown:
        raise UsageError(
            f"Result.rerun() does not take {unknown[0]!r}. It changes these settings and "
            f"keeps the rest: {', '.join(SETTINGS)}.",
            code="rerun.unknown_setting",
            details={"settings": unknown, "accepted": list(SETTINGS)},
            remedies=[pass_argument(name) for name in SETTINGS],
        )
    if "sample_times" in changes and "n_points" in changes:
        raise UsageError(
            "Result.rerun() takes sample_times or n_points, not both.",
            code="usage.conflicting_arguments",
            details={"arguments": ["sample_times", "n_points"]},
            remedies=[],
        )
    rec = result.record
    _check_refusals(result, rec)
    run = rec["run"]
    method = changes.get("method", run["method"])
    model = build_model(result, rec["model"])
    _put_start(model, run)
    options = _simulator_options(run, rec["backend"], method, changes)
    sim = Simulator(model, method=method, **options)
    again = sim.run(start="live", **_run_options(run, method, changes))

    again._run_info = again._run_info._replace(
        rerun_of={"record_sha256": _record_sha256(rec), "changed": _changed(changes)}
    )
    software = rec["software"] or {}
    now = again.record["software"]
    built = (software.get("bngsim"), software.get("build_commit"))
    if software and built != (now["bngsim"], now["build_commit"]):
        _notices.warn(
            f"This rerun ran on bngsim {now['bngsim']} (build {now['build_commit']}), and "
            f"the run it repeats on {software.get('bngsim')} (build "
            f"{software.get('build_commit')}): a stochastic run reproduces exactly only on "
            "the same build, and an ODE run may differ in its last digits.",
            UserWarning,
            code="rerun.different_build",
            stacklevel=_notices.OUTSIDE,
        )
    if method == run["method"] and not ({"jacobian", "codegen"} & set(changes)):
        then = rec["backend"] or {}
        ran = again.record["backend"] or {}
        moved = [k for k in _BACKEND if then.get(k) != ran.get(k)]
        if moved:
            _notices.warn(
                "This rerun ran other code than the run it repeats: "
                + ", ".join(f"{k} {ran.get(k)!r} for {then.get(k)!r}" for k in moved)
                + ". Its numbers may differ in their last digits.",
                UserWarning,
                code="rerun.different_backend",
                stacklevel=_notices.OUTSIDE,
            )
    return again
