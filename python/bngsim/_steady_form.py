"""The JSON forms of the steady-state calls' answers, for the ``bngsim`` command.

:class:`~bngsim.SteadyStateResult` and :class:`~bngsim.SteadyStates` hold arrays
(a concentration per species, a spectrum per root), and the command writes no
array to standard output. These are their bounded forms: the solver's verdict
and how it was reached, and at most ``max_columns`` columns of each kind, the
rest to ``--out``. Nothing here reads a result beyond what it holds, and nothing
calls back into the solve.

Columns are chosen as :meth:`bngsim.Result.digest` chooses them, and listed in
the model's order: any that is not finite first, then for one steady state the
largest in size, for many states (a batch's rows, a model's roots) the widest
spread across them.

Numbers are written as the result holds them. A ``newton_distance`` of NaN (no
Newton step was taken) and one of infinity (the Jacobian is singular there) are
different answers, so they are not both ``None``: the command writes them as
the strings ``"nan"`` and ``"inf"``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim._digest import _rank

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

#: Incremented by a change that removes or renames a key, or changes what one
#: holds. A key added is no change of version.
FORM_VERSION = 1

#: The statistics across rows a batch gives of each column, by key.
QUANTILES = (("q05", 0.05), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q95", 0.95))


def _float(x: Any) -> float:
    return float(x)


def _names_values(ss: Any, kind: str) -> tuple[list[str], NDArray[np.float64]]:
    """A steady state's values of one kind of column, and their names."""
    if kind == "species":
        return list(ss.species_names), np.asarray(ss.concentrations, dtype=np.float64)
    return list(ss.observable_names), np.asarray(ss._observable_values, dtype=np.float64)


def _kinds(ss: Any, species: bool) -> list[str]:
    """Observables where the model has any, else species, as a digest lists them;
    species too with ``species=True``."""
    kinds = ["observables"] if len(ss.observable_names) else ["species"]
    if species and "species" not in kinds:
        kinds.append("species")
    return kinds


def _chosen(values: NDArray[np.float64], by: NDArray[np.float64], max_columns: int) -> list[int]:
    """The columns to list: any not finite first, then the largest ``by``."""
    finite = np.isfinite(values)
    nonfinite = ~(finite if values.ndim == 1 else finite.all(axis=0))
    return _rank(nonfinite, np.where(np.isfinite(by), by, np.inf), max_columns)


def _eigen(eigenvalues: Any) -> dict[str, Any]:
    ev = np.asarray(eigenvalues, dtype=np.complex128)
    return {
        "n_eigenvalues": int(ev.size),
        "largest_real_part": _float(ev.real.max()) if ev.size else None,
    }


def _verdict(ss: Any) -> dict[str, Any]:
    """What one solve concluded and how it got there."""
    species = list(ss.species_names)
    return {
        "converged": bool(ss.converged),
        "criterion": ss.criterion,
        "method_used": ss.method_used,
        "residual": _float(ss.residual),
        "newton_distance": _float(ss.newton_distance),
        "continued_run_distance": _float(ss.continued_run_distance),
        "root_stability": ss.root_stability,
        "n_unstable_roots_rejected": int(ss.n_unstable_roots_rejected),
        **_eigen(ss.eigenvalues),
        "n_steps": int(ss.n_steps),
        "n_rhs_evals": int(ss.n_rhs_evals),
        "n_state_switch_roots": int(ss.n_state_switch_roots),
        "state_switch_roots_dropped": bool(ss.state_switch_roots_dropped),
        "n_residual_species": int(ss.n_residual_species),
        "excluded_species": [species[i] for i in ss.excluded_species if 0 <= i < len(species)],
        "unconverged_pure_sinks": list(ss.unconverged_pure_sinks),
        "rhs_backend": ss.rhs_backend,
        "solver_jacobian_source": ss.solver_jacobian_source,
        "solver_jacobian_retried": bool(ss.solver_jacobian_retried),
        "linear_solver": ss.linear_solver,
    }


def steady_state(ss: Any, *, species: bool = False, max_columns: int = 20) -> dict[str, Any]:
    """The JSON form of one :class:`~bngsim.SteadyStateResult`: its verdict, and
    the value of each listed column at the state returned."""
    out: dict[str, Any] = {"form_version": FORM_VERSION, **_verdict(ss)}
    kinds = _kinds(ss, species)
    out["kinds"] = kinds
    for kind in kinds:
        names, values = _names_values(ss, kind)
        chosen = _chosen(values, np.abs(values), max_columns)
        out[kind] = {
            "n_columns": len(names),
            "n_left_out": len(names) - len(chosen),
            "columns": [{"name": names[j], "value": _float(values[j])} for j in chosen],
        }
    return out


def _stats(column: NDArray[np.float64]) -> dict[str, Any]:
    """Statistics of one column across the rows it is given."""
    ok = column[np.isfinite(column)]
    stats: dict[str, Any] = {"n_finite": int(ok.size)}
    keys = ["mean", "std", "min", *(k for k, _ in QUANTILES), "max"]
    if ok.size == 0:
        stats.update(dict.fromkeys(keys))
        return stats
    stats["mean"] = _float(ok.mean())
    stats["std"] = _float(ok.std(ddof=1)) if ok.size > 1 else None
    stats["min"] = _float(ok.min())
    for (key, _), v in zip(QUANTILES, np.quantile(ok, [q for _, q in QUANTILES]), strict=True):
        stats[key] = _float(v)
    stats["max"] = _float(ok.max())
    return stats


def _counts(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def steady_state_batch(
    rows: Sequence[Any], *, species: bool = False, max_columns: int = 20
) -> dict[str, Any]:
    """The JSON form of :meth:`~bngsim.Simulator.steady_state_batch`'s rows:
    how many converged and which did not, how each was reached, and statistics
    of each listed column across the rows that converged. A row that did not
    converge holds no steady state, so its values are in no statistic."""
    converged = [bool(r.converged) for r in rows]
    failed = [i for i, ok in enumerate(converged) if not ok]
    out: dict[str, Any] = {
        "form_version": FORM_VERSION,
        "n_rows": len(rows),
        "n_converged": sum(converged),
        "unconverged_rows": failed[:max_columns],
        "n_unconverged": len(failed),
        "method_used": _counts([r.method_used for r in rows]),
        "root_stability": _counts([r.root_stability for r in rows]),
        "largest_residual": _float(max(float(r.residual) for r in rows)) if rows else None,
    }
    kinds = _kinds(rows[0], species) if rows else []
    out["kinds"] = kinds
    for kind in kinds:
        names = _names_values(rows[0], kind)[0]
        block = np.array([_names_values(r, kind)[1] for r in rows], dtype=np.float64)
        block = block.reshape(len(rows), len(names))
        kept = block[np.asarray(converged, dtype=bool)]
        if kept.shape[0]:
            spread = np.where(np.isfinite(kept), kept, np.nan)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                # A column with no finite value: nanmax says so; _chosen lists it.
                warnings.simplefilter("ignore", RuntimeWarning)
                span = np.nanmax(spread, axis=0) - np.nanmin(spread, axis=0)
            chosen = _chosen(kept, span, max_columns)
        else:
            chosen = list(range(min(max_columns, len(names))))
        out[kind] = {
            "n_columns": len(names),
            "n_left_out": len(names) - len(chosen),
            "columns": [{"name": names[j], **_stats(kept[:, j])} for j in chosen],
        }
    return out


def steady_states(found: Any, *, max_columns: int = 20) -> dict[str, Any]:
    """The JSON form of :meth:`~bngsim.Simulator.steady_states`' distinct roots:
    each root's stability, how many starts reached it and its residual, and
    each listed species' value at every root. The species listed are those
    that differ most from root to root, which is what tells the roots apart."""
    states = np.asarray(found.states, dtype=np.float64)
    names = list(found.species_names)
    n_roots = int(states.shape[0])
    roots = []
    for k in range(n_roots):
        roots.append(
            {
                "stability": found.stability[k],
                "multiplicity": int(found.multiplicity[k]),
                "residual": _float(found.residuals[k]),
                **_eigen(found.eigenvalues[k]),
            }
        )
    if n_roots:
        with np.errstate(all="ignore"):
            span = states.max(axis=0) - states.min(axis=0)
        chosen = _chosen(states, span, max_columns)
    else:
        chosen = []
    return {
        "form_version": FORM_VERSION,
        "n_roots": n_roots,
        "n_starts": int(np.asarray(found.basins).size),
        "n_failed": int(found.n_failed),
        "stability": _counts(found.stability),
        "roots": roots,
        "species": {
            "n_columns": len(names),
            "n_left_out": len(names) - len(chosen),
            "columns": [
                {"name": names[j], "values": [_float(v) for v in states[:, j]]} for j in chosen
            ],
        },
    }


# ─── The arrays, for --out ───────────────────────────────────────────────────


def _str_array(values: Sequence[Any]) -> NDArray[Any]:
    """Text as NumPy's own string type, which ``np.load`` reads without pickle."""
    return np.array([str(v) for v in values], dtype=str)


def steady_state_arrays(ss: Any) -> dict[str, Any]:
    """Every array of one steady state, by the name ``--out`` writes it under."""
    return {
        "species_names": _str_array(ss.species_names),
        "concentrations": np.asarray(ss.concentrations, dtype=np.float64),
        "observable_names": _str_array(ss.observable_names),
        "observables": np.asarray(ss._observable_values, dtype=np.float64),
        "eigenvalues": np.asarray(ss.eigenvalues, dtype=np.complex128),
        "converged": np.asarray(bool(ss.converged)),
        "residual": np.asarray(float(ss.residual)),
    }


def steady_state_batch_arrays(rows: Sequence[Any]) -> dict[str, Any]:
    """Every array of a batch's rows, one row per parameter set."""
    first = rows[0]
    n_species, n_obs = len(first.species_names), len(first.observable_names)
    return {
        "species_names": _str_array(first.species_names),
        "concentrations": np.array(
            [np.asarray(r.concentrations, dtype=np.float64) for r in rows]
        ).reshape(len(rows), n_species),
        "observable_names": _str_array(first.observable_names),
        "observables": np.array(
            [np.asarray(r._observable_values, dtype=np.float64) for r in rows]
        ).reshape(len(rows), n_obs),
        "converged": np.array([bool(r.converged) for r in rows], dtype=bool),
        "residual": np.array([float(r.residual) for r in rows], dtype=np.float64),
        "method_used": _str_array([r.method_used for r in rows]),
        "root_stability": _str_array([r.root_stability for r in rows]),
    }


def steady_states_arrays(found: Any) -> dict[str, Any]:
    """Every array of a model's distinct roots. Each root's spectrum is left
    out (they may differ in length); its largest real part is kept."""
    largest = [
        float(np.asarray(e).real.max()) if np.asarray(e).size else float("nan")
        for e in found.eigenvalues
    ]
    return {
        "species_names": _str_array(found.species_names),
        "states": np.asarray(found.states, dtype=np.float64),
        "stability": _str_array(found.stability),
        "multiplicity": np.asarray(found.multiplicity, dtype=np.int64),
        "basins": np.asarray(found.basins, dtype=np.int64),
        "residuals": np.asarray(found.residuals, dtype=np.float64),
        "largest_real_part": np.asarray(largest, dtype=np.float64),
    }
