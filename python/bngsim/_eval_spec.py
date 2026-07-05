"""bngsim._eval_spec — serializable single-evaluation kernel (GH #203).

The HPC scheduler-free contract: bngsim is a clean, *stateless* single-evaluation
kernel; PyBNF (or any other frontend) owns the scheduler — multistart, bootstrap,
profile likelihood, Slurm/MPI fan-out. To distribute thousands of independent
evaluations, the driver needs a serializable description of *one* evaluation that
any worker process can materialize and run deterministically, plus checkpoint and
restart support.

:class:`EvaluationSpec` is that description: a frozen, JSON-serializable record of
``(model source, parameter vector, time grid, sensitivity set, solver options,
output selectors)``. It carries no live objects, no mutable optimization state,
and no objective/noise/loss layer (those are the frontend's — bngsim returns the
raw output + sensitivity *primitives*; see GH #194). ``evaluate()`` reconstructs a
:class:`~bngsim.Simulator` and runs it; for fixed inputs the result is
deterministic, so the same spec evaluated on any node yields the same arrays.

Pair a spec with :meth:`bngsim.Result.summary` for a compact what-came-back record
(full arrays persist via :meth:`bngsim.Result.save`).

Example
-------
>>> import bngsim
>>> spec = bngsim.EvaluationSpec(
...     model_source="model.net",
...     model_format="net",
...     t_span=(0.0, 100.0),
...     n_points=101,
...     params={"k1": 0.5},
...     sensitivity_params=("k1",),
...     outputs=("observable:Atot",),
... )
>>> blob = spec.to_json()                 # ship to a worker / checkpoint
>>> same = bngsim.EvaluationSpec.from_json(blob)
>>> result = same.evaluate()              # deterministic for fixed inputs
>>> grad = result.output_sensitivities(same.outputs)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # avoid import cycles at module load
    from bngsim._model import Model
    from bngsim._result import Result
    from bngsim._simulator import Simulator

# Model-source kinds EvaluationSpec knows how to materialize. The ``*_string``
# variants carry the model text inline in ``model_source`` (no filesystem); the
# others carry a path. Kept in lock-step with the Model.from_* loaders.
_PATH_FORMATS: frozenset[str] = frozenset({"net", "sbml", "antimony"})
_STRING_FORMATS: frozenset[str] = frozenset({"sbml_string", "antimony_string"})
_VALID_FORMATS: frozenset[str] = _PATH_FORMATS | _STRING_FORMATS


@dataclass(frozen=True)
class EvaluationSpec:
    """A serializable, stateless description of one bngsim evaluation.

    Parameters
    ----------
    model_source : str
        For a path format (``net``/``sbml``/``antimony``), the model file path.
        For a ``*_string`` format, the inline model text.
    model_format : str
        One of ``"net"``, ``"sbml"``, ``"antimony"``, ``"sbml_string"``,
        ``"antimony_string"``.
    method : str
        Simulation method. Default ``"ode"`` (the only method with sensitivities).
    t_span : tuple[float, float]
        ``(t_start, t_end)`` integration interval.
    n_points : int
        Number of output time points (including ``t_start``).
    params : Mapping[str, float]
        Parameter overrides applied to the model before running (the θ vector).
    sensitivity_params : Sequence[str]
        Parameter names to compute forward output sensitivities for. Empty
        disables parameter sensitivities.
    sensitivity_ic : Sequence[str]
        Species names whose initial conditions to differentiate against.
    sensitivity_method : str
        CVODES sensitivity method (``"staggered"`` / ``"simultaneous"``).
    outputs : Sequence[str]
        Output selectors (``species:``/``observable:``/``expression:`` …) that
        the caller intends to read off the result. Recorded for provenance; not
        applied during integration.
    rtol, atol : float, optional
        Solver tolerances. ``None`` uses the Simulator defaults.
    max_steps : int, optional
        Max internal solver steps per output point. ``None`` uses the default.
    model_sha256 : str, optional
        Hex SHA-256 of the model source (file bytes for a path format, the
        UTF-8 text for a ``*_string`` format). When set, :meth:`build_model`
        verifies the live source matches and raises on mismatch — a cluster
        integrity guard against a stale/edited model file on a shared filesystem.
    """

    model_source: str
    model_format: str = "net"
    method: str = "ode"
    t_span: tuple[float, float] = (0.0, 100.0)
    n_points: int = 101
    params: Mapping[str, float] = field(default_factory=dict)
    sensitivity_params: Sequence[str] = ()
    sensitivity_ic: Sequence[str] = ()
    sensitivity_method: str = "staggered"
    outputs: Sequence[str] = ()
    rtol: float | None = None
    atol: float | None = None
    max_steps: int | None = None
    model_sha256: str | None = None

    def __post_init__(self) -> None:
        # Normalize to canonical, immutable container types so two specs built
        # from a dict vs from kwargs compare equal and serialize identically.
        # Frozen dataclass ⇒ assign via object.__setattr__.
        if self.model_format not in _VALID_FORMATS:
            raise ValueError(
                f"Unknown model_format {self.model_format!r}. "
                f"Expected one of {sorted(_VALID_FORMATS)}."
            )
        ts = tuple(float(x) for x in self.t_span)
        if len(ts) != 2:
            raise ValueError(f"t_span must be a 2-tuple (t_start, t_end), got {self.t_span!r}.")
        object.__setattr__(self, "t_span", ts)
        object.__setattr__(self, "n_points", int(self.n_points))
        object.__setattr__(
            self, "params", {str(k): float(v) for k, v in dict(self.params).items()}
        )
        object.__setattr__(self, "sensitivity_params", tuple(self.sensitivity_params))
        object.__setattr__(self, "sensitivity_ic", tuple(self.sensitivity_ic))
        object.__setattr__(self, "outputs", tuple(self.outputs))

    # ─── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-encodable dict of this spec.

        ``params`` is emitted sorted by name and sequences as lists, so the dict
        (and :meth:`to_json`) is byte-stable for a fixed spec — usable directly
        as a content key for caching/deduplication.
        """
        return {
            "model_source": self.model_source,
            "model_format": self.model_format,
            "method": self.method,
            "t_span": list(self.t_span),
            "n_points": self.n_points,
            "params": {k: self.params[k] for k in sorted(self.params)},
            "sensitivity_params": list(self.sensitivity_params),
            "sensitivity_ic": list(self.sensitivity_ic),
            "sensitivity_method": self.sensitivity_method,
            "outputs": list(self.outputs),
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
            "model_sha256": self.model_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationSpec:
        """Reconstruct a spec from a :meth:`to_dict` mapping (extra keys rejected)."""
        known = {
            "model_source",
            "model_format",
            "method",
            "t_span",
            "n_points",
            "params",
            "sensitivity_params",
            "sensitivity_ic",
            "sensitivity_method",
            "outputs",
            "rtol",
            "atol",
            "max_steps",
            "model_sha256",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown EvaluationSpec field(s): {sorted(unknown)}")
        kwargs = {k: data[k] for k in known if k in data}
        if "t_span" in kwargs:
            kwargs["t_span"] = tuple(kwargs["t_span"])
        return cls(**kwargs)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to a JSON string (keys sorted for byte-stability)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> EvaluationSpec:
        """Deserialize from a :meth:`to_json` string."""
        return cls.from_dict(json.loads(text))

    def with_params(self, params: Mapping[str, float], *, merge: bool = False) -> EvaluationSpec:
        """Return a copy with ``params`` replaced (or merged when ``merge=True``).

        The ergonomic path for a parameter sweep / multistart: serialize one base
        spec, then stamp each θ row through ``with_params`` on the worker. Returns
        a new frozen instance; the original is untouched.
        """
        new_params = {**self.params, **dict(params)} if merge else dict(params)
        return replace(self, params=new_params)

    # ─── Source integrity ──────────────────────────────────────────

    def compute_source_sha256(self) -> str:
        """SHA-256 of the model source (file bytes for a path, UTF-8 text inline)."""
        h = hashlib.sha256()
        if self.model_format in _PATH_FORMATS:
            h.update(Path(self.model_source).read_bytes())
        else:
            h.update(self.model_source.encode("utf-8"))
        return h.hexdigest()

    # ─── Materialization ───────────────────────────────────────────

    def build_model(self) -> Model:
        """Load the model from :attr:`model_source` per :attr:`model_format`.

        When :attr:`model_sha256` is set, the live source is hashed and compared
        first; a mismatch raises :class:`ValueError` (cluster integrity guard).
        """
        from bngsim._model import Model

        if self.model_sha256 is not None:
            actual = self.compute_source_sha256()
            if actual != self.model_sha256:
                raise ValueError(
                    "EvaluationSpec model source SHA-256 mismatch: expected "
                    f"{self.model_sha256}, got {actual}. The model at "
                    f"{self.model_source!r} differs from the one this spec was "
                    "built against (stale or edited artifact on a shared filesystem)."
                )

        if self.model_format == "net":
            return Model.from_net(self.model_source)
        if self.model_format == "sbml":
            return Model.from_sbml(self.model_source)
        if self.model_format == "antimony":
            return Model.from_antimony(self.model_source)
        if self.model_format == "sbml_string":
            return Model.from_sbml_string(self.model_source)
        if self.model_format == "antimony_string":
            return Model.from_antimony_string(self.model_source)
        # __post_init__ already validated the format; defensive only.
        raise ValueError(f"Unknown model_format {self.model_format!r}.")

    def build_simulator(self) -> Simulator:
        """Build a :class:`~bngsim.Simulator` with θ applied and sensitivities wired."""
        from bngsim._simulator import Simulator

        model = self.build_model()
        if self.params:
            model.set_params(dict(self.params))
        return Simulator(
            model,
            method=self.method,
            sensitivity_params=list(self.sensitivity_params) or None,
            sensitivity_ic=list(self.sensitivity_ic) or None,
            sensitivity_method=self.sensitivity_method,
        )

    def evaluate(self) -> Result:
        """Materialize and run this evaluation, returning the :class:`~bngsim.Result`.

        Deterministic for fixed inputs (model, θ, sensitivity set, solver
        options). Read named outputs via :meth:`Result.outputs` and gradients via
        :meth:`Result.output_sensitivities` with :attr:`outputs` as selectors.
        """
        sim = self.build_simulator()
        run_kwargs: dict[str, Any] = {"t_span": self.t_span, "n_points": self.n_points}
        if self.rtol is not None:
            run_kwargs["rtol"] = self.rtol
        if self.atol is not None:
            run_kwargs["atol"] = self.atol
        if self.max_steps is not None:
            run_kwargs["max_steps"] = self.max_steps
        return sim.run(**run_kwargs)
