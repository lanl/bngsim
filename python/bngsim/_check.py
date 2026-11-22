"""A one-call check of a run (issue #1707).

:meth:`bngsim.Result.check` makes the run a result records again with one
setting changed (:meth:`Result.rerun`, issue #1703) and says whether the answer
moved. Each rerun starts from the state the run started from, on a fresh model,
so a check costs the session nothing and compares like with like.

The rule two paths of one run are held to is the developer-side differential
harness's (``parity_checks/differential``, issue #1257), which imports it from
here, so a verdict means the same in a user's session and in the nightly:

- :func:`gap`: the largest cell-wise difference, past an absolute
  :func:`floor` of ``PATH_RTOL_FACTOR`` times the run's atol (at most
  ``PATH_CEILING`` of the peak), over the reference's peak;
- the run's own drift: the gap between the run and the run at a tenfold tighter
  ``rtol`` and ``atol``;
- a path passes when its gap is at most ``max(min(PATH_RTOL_FACTOR·rtol,
  PATH_CEILING), DRIFT_FACTOR·drift)``: two correct paths each err by about
  ``rtol``, scaled by the problem's conditioning, which the drift measures, and
  never by more than ``PATH_CEILING`` of the peak for the rtol alone;
- a run whose drift is past ``UNJUDGEABLE_DRIFT`` is not converged at its own
  tolerance, and no path is judged against it.

Verdicts: ``pass``, ``diff``, ``skip`` (with the reason), ``unsupported`` (the
check does not apply to this run), ``exception`` (the changed run raised),
``no_base`` (the run is not converged enough to judge a path against) and
``timeout`` (the ``budget`` ran out).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim import _record
from bngsim._exceptions import UsageError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bngsim._result import Result

#: Incremented by a change that removes or renames a key, or changes what one
#: holds. A key added is no change of version.
CHECK_VERSION = 1

#: How many times the run's rtol two correct paths may differ by, before the
#: run's own drift is counted: their errors are each about rtol, scaled by the
#: problem's conditioning, which a tolerance does not bound.
PATH_RTOL_FACTOR = 1e3
#: And how far past the run's own drift at a tenfold tighter tolerance.
DRIFT_FACTOR = 10.0
#: A run that moves by more than this at a tenfold tighter tolerance is not
#: converged at its own: no path is judged against it.
UNJUDGEABLE_DRIFT = 1e-3
#: The most the ``PATH_RTOL_FACTOR·rtol`` term may forgive, as a share of the
#: peak. At the nightly's rtol of 1e-8 it is 1e-5 and this never bites; at a
#: caller's rtol of 1e-3 the term alone is the whole peak, and no path could
#: fail.
PATH_CEILING = 1e-2
#: The grid check's output points, as a multiple of the run's intervals.
GRID_FACTOR = 8
#: How many reruns the replicates check makes of a stochastic run.
DEFAULT_REPLICATES = 10

#: The checks, in the order they run; ``tolerance`` first, as the others are
#: judged against its drift.
DETERMINISTIC = ("tolerance", "grid", "jacobian", "backend", "second_engine")
STOCHASTIC = ("replicates",)
#: What ``checks=None`` runs: the cheap checks that catch the common case.
DEFAULT_DETERMINISTIC = ("tolerance", "grid")
DEFAULT_STOCHASTIC = ("replicates",)

_STATUSES = ("pass", "diff", "skip", "unsupported", "exception", "no_base", "timeout")
_FAILING = ("diff", "exception")


def gap(x: NDArray[np.float64], ref: NDArray[np.float64], atol: float) -> float:
    """``max |x - ref|`` over the reference's peak, past an absolute floor of
    ``atol``: 0 where every cell is within atol. A shape or finiteness mismatch
    is ``inf``."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    if x.shape != ref.shape or not np.array_equal(np.isfinite(x), np.isfinite(ref)):
        return float("inf")
    finite = np.isfinite(ref)
    if not finite.any():
        return 0.0
    over = np.abs(x - ref)[finite] - atol
    peak = float(np.abs(ref[finite]).max())
    return float(max(over.max(), 0.0) / max(peak, atol))


def floor(atol: float, ref: NDArray[np.float64]) -> float:
    """The absolute difference :func:`gap` forgives two paths of a run at
    *atol* whose reference is *ref*: ``PATH_RTOL_FACTOR·atol``, but at most
    ``PATH_CEILING`` of the reference's peak (at a caller's atol of 1e-2 the
    first alone is 10 on a peak of 100)."""
    ref = np.asarray(ref, dtype=float)
    finite = np.abs(ref[np.isfinite(ref)])
    peak = float(finite.max()) if finite.size else 0.0
    cap = PATH_CEILING * peak
    return min(PATH_RTOL_FACTOR * atol, cap) if cap > 0 else PATH_RTOL_FACTOR * atol


def bound(rtol: float, drift: float) -> float:
    """How far two correct paths of a run at *rtol* whose own drift is *drift*
    may be apart, as :func:`gap` measures it."""
    return max(min(PATH_RTOL_FACTOR * rtol, PATH_CEILING), DRIFT_FACTOR * drift)


def _worst(x: NDArray[np.float64], ref: NDArray[np.float64], names: list[str]) -> str | None:
    """The column of the largest difference, by name."""
    if x.shape != ref.shape or not names:
        return None
    d = np.abs(np.asarray(x, dtype=float) - np.asarray(ref, dtype=float))
    d = np.where(np.isfinite(d), d, np.inf)
    if not d.size:
        return None
    j = int(np.unravel_index(int(np.argmax(d)), d.shape)[-1])
    return names[j] if j < len(names) else None


def _num(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


class Report(dict):  # type: ignore[type-arg]
    """:meth:`Result.check`'s dict, whose ``str()`` is its plain-text form.

    It is a ``dict`` in every other way: ``json.dumps(report, allow_nan=False)``
    writes it.
    """

    __slots__ = ()

    def __str__(self) -> str:
        return text(self)

    def text(self) -> str:
        """The plain-text form, as ``str()`` gives it."""
        return text(self)


class _Out(Exception):
    """The budget ran out before a check could start or finish."""


def check(
    result: Result,
    *,
    checks: Any = None,
    budget: float | None = None,
    replicates: int = DEFAULT_REPLICATES,
) -> Report:
    """:meth:`Result.check`."""
    if result._species.ndim == 3:
        raise UsageError(
            "Result.check() checks one run; this is a stack of runs. Check each row "
            "(the list run_batch or run_replicates returns without squeeze=True).",
            code="usage.invalid_argument",
        )
    if budget is not None and not (isinstance(budget, int | float) and budget > 0):
        raise UsageError(f"check(budget=...) takes seconds > 0 or None, got {budget!r}")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise UsageError(
            f"check(replicates=...) takes a whole number of 2 or more, got {replicates!r}"
        )
    rec = result.record
    run = rec.get("run")
    method = None if run is None else run.get("method")
    stochastic = None if method is None else method in _record.STOCHASTIC
    known = DETERMINISTIC + STOCHASTIC
    if checks is None:
        names = list(DEFAULT_STOCHASTIC if stochastic else DEFAULT_DETERMINISTIC)
    elif checks == "all":
        names = list(STOCHASTIC if stochastic else DETERMINISTIC)
    else:
        names = [checks] if isinstance(checks, str) else list(checks)
        bad = [n for n in names if n not in known]
        if bad:
            raise UsageError(
                f"check(checks=...) does not know {bad[0]!r}: it takes None (the default "
                f"set), 'all', or names from {', '.join(known)}.",
                code="usage.invalid_argument",
                details={"checks": bad, "accepted": list(known)},
            )
        names = list(dict.fromkeys(names))
    deadline = None if budget is None else time.monotonic() + float(budget)

    out = Report()
    out["check_version"] = CHECK_VERSION
    out["ok"] = False
    out["method"] = method
    out["stochastic"] = stochastic
    out["rtol"] = None
    out["atol"] = None
    out["drift"] = None
    out["bound"] = None
    out["checks"] = []

    if run is None:
        for name in names:
            out["checks"].append(
                _verdict(name, "unsupported", note="the result records no run to make again")
            )
        return _finish(out)

    ctx = _Context(result, rec, deadline)
    out["rtol"] = ctx.rtol
    out["atol"] = ctx.atol
    order = [n for n in known if n in names]
    for name in order:
        v = ctx.run_check(name, replicates)
        out["checks"].append(v)
    out["drift"] = _num(ctx.drift)
    out["bound"] = _num(ctx.limit)
    return _finish(out)


def _finish(out: Report) -> Report:
    statuses = [c["status"] for c in out["checks"]]
    out["ok"] = bool(statuses) and "pass" in statuses and not any(s in _FAILING for s in statuses)
    out["n_pass"] = statuses.count("pass")
    out["n_diff"] = statuses.count("diff")
    out["next"] = _next(out)
    return out


def _next(out: Report) -> str:
    """What to do next, in one sentence."""
    diffs = [c for c in out["checks"] if c["status"] == "diff"]
    if diffs and diffs[0]["name"] == "tolerance":
        return (
            "The answer moves when the tolerance is tightened: rerun at a tighter rtol and "
            "atol (result.rerun(rtol=..., atol=...)) until it does not, and report that run."
        )
    if diffs and diffs[0]["name"] == "second_engine":
        return (
            "bngsim and libRoadRunner disagree on this model: one of them reads it "
            "differently. Report it at https://github.com/lanl/bngsim/issues with "
            "json.dumps(report) and the model file."
        )
    if diffs:
        return (
            f"Two bngsim paths disagree ({diffs[0]['name']}): on a model bngsim accepts, that "
            "is a bug worth reporting at https://github.com/lanl/bngsim/issues, with "
            "json.dumps(report), result.record and the model file. Do not report the "
            "number as resolved."
        )
    if any(c["status"] == "exception" for c in out["checks"]):
        return "A changed run raised: read its note before relying on the result."
    if any(c["status"] == "no_base" for c in out["checks"]):
        return (
            "The run is not converged at its own tolerance, so no path could be judged: "
            "rerun at a tighter rtol and atol and check that."
        )
    if not out["ok"]:
        return "No check could run on this result: see each check's note."
    return "No check moved the answer past its bound."


def _verdict(
    name: str,
    status: str,
    value: float | None = None,
    note: str = "",
    *,
    worst: str | None = None,
    changed: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert status in _STATUSES, status
    v: dict[str, Any] = {
        "name": name,
        "status": status,
        "value": _num(value),
        "worst_column": worst,
        "changed": changed or {},
        "note": note,
    }
    if extra:
        v.update(extra)
    return v


class _Context:
    """The run a result records, and what its checks share: the tolerances, the
    drift and the bound."""

    def __init__(self, result: Result, rec: dict[str, Any], deadline: float | None) -> None:
        self.result = result
        self.rec = rec
        self.run = rec["run"]
        self.deadline = deadline
        self.method = self.run.get("method")
        self.stochastic = self.method in _record.STOCHASTIC
        self.x = np.asarray(result._species, dtype=float)
        self.names = list(result._species_names)
        atol = self.run.get("atol") or {}
        self.rtol: float | None = self.run.get("rtol")
        per = atol.get("per_species")
        self.per_species = None if per is None else [float(a) for a in per]
        self.atol: float | None = atol.get("scalar")
        self.drift: float | None = None
        self.tight_x: NDArray[np.float64] = self.x
        self.limit: float | None = None
        self.tol_error = ""

    # --- running ---------------------------------------------------------

    def _left(self) -> float | None:
        if self.deadline is None:
            return None
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise _Out
        return left

    def rerun(self, **changes: Any) -> Result:
        left = self._left()
        if left is not None:
            changes["timeout"] = left
        try:
            return self.result.rerun(**changes)
        except Exception as e:
            if self.deadline is not None and _timed_out(e):
                raise _Out from e
            raise

    def run_check(self, name: str, replicates: int) -> dict[str, Any]:
        fn = getattr(self, f"check_{name}")
        try:
            self._left()
            return fn() if name != "replicates" else fn(replicates)
        except _Out:
            return _verdict(name, "timeout", note="the budget ran out before this check finished")
        except Exception as e:  # noqa: BLE001 - a check that raises is a finding
            return _verdict(name, "exception", note=f"{type(e).__name__}: {_first_line(e)}")

    # --- the reference ---------------------------------------------------

    def _deterministic(self, name: str) -> dict[str, Any] | None:
        if self.stochastic:
            return _verdict(
                name,
                "unsupported",
                note=f"a {self.method} run is stochastic: see the replicates check",
            )
        if self.method != "ode":
            return _verdict(name, "unsupported", note=f"the run's method is {self.method!r}")
        if self.rtol is None or self.atol is None:
            return _verdict(name, "unsupported", note="the record gives no rtol or atol")
        return None

    def _floor(self) -> float:
        assert self.atol is not None
        return floor(float(self.atol), self.x)

    def _tighter(self, by: float) -> dict[str, Any]:
        assert self.rtol is not None and self.atol is not None
        if self.per_species is not None:
            atol: Any = [a / by for a in self.per_species]
        else:
            atol = float(self.atol) / by
        return {"rtol": float(self.rtol) / by, "atol": atol}

    def _ensure_drift(self) -> str | None:
        """The run's own drift, made once; the reason there is none, or None."""
        if self.drift is None and not self.tol_error:
            try:
                tight = self.rerun(**self._tighter(10.0))
                self.tight_x = np.asarray(tight._species, dtype=float)
                self.drift = gap(self.tight_x, self.x, self._floor())
                assert self.rtol is not None
                self.limit = bound(float(self.rtol), self.drift)
            except _Out:
                raise
            except Exception as e:  # noqa: BLE001 - recorded on every check's note
                self.tol_error = f"{type(e).__name__}: {_first_line(e)}"
        return self.tol_error or None

    def compare(
        self, name: str, x: NDArray[np.float64], changed: dict[str, Any]
    ) -> dict[str, Any]:
        err = self._ensure_drift()
        if err:
            return _verdict(
                name,
                "no_base",
                note=f"the run at a tenfold tighter tolerance raised ({err})",
                changed=changed,
            )
        assert self.drift is not None and self.limit is not None
        if not self.drift <= UNJUDGEABLE_DRIFT:
            return _verdict(
                name,
                "no_base",
                note=f"the run itself moves by {self.drift:.2e} of its peak at a tenfold tighter "
                f"tolerance, past {UNJUDGEABLE_DRIFT:g}: no path is judged against it",
                changed=changed,
            )
        g = gap(x, self.x, self._floor())
        worst = _worst(x, self.x, self.names) if g > 0 else None
        status = "pass" if g <= self.limit else "diff"
        note = (
            f"{g:.2e} of the trajectory's peak "
            f"(bound {self.limit:.1e}, own drift {self.drift:.1e})"
        )
        return _verdict(
            name, status, g, note, worst=worst, changed=changed, extra={"bound": _num(self.limit)}
        )

    # --- the checks ------------------------------------------------------

    def check_tolerance(self) -> dict[str, Any]:
        skip = self._deterministic("tolerance")
        if skip:
            return skip
        err = self._ensure_drift()
        changed = self._tighter(10.0)
        if err:
            return _verdict(
                "tolerance",
                "exception",
                note=f"the run at a tenfold tighter tolerance raised: {err}",
                changed=changed,
            )
        assert self.drift is not None and self.atol is not None
        # The move at the run's own atol, not the paths' floor of 1e3·atol: a
        # species that moves by 0.05 where atol is 1e-4 has moved.
        moved = gap(self.tight_x, self.x, float(self.atol))
        limit = UNJUDGEABLE_DRIFT
        extra: dict[str, Any] = {"bound": _num(limit), "value_100": None}
        # A hundredfold tighter too, where the budget allows: whether the
        # answer keeps moving, or has stopped.
        try:
            tighter = self.rerun(**self._tighter(100.0))
            extra["value_100"] = _num(
                gap(np.asarray(tighter._species, dtype=float), self.x, float(self.atol))
            )
        except _Out:
            pass
        except Exception:  # noqa: BLE001 - the tenfold run decides; this one adds
            pass
        status = "pass" if moved <= limit else "diff"
        note = (
            f"the answer moves by {moved:.2e} of its peak at rtol/10 and atol/10 (bound {limit:g})"
        )
        if extra["value_100"] is not None:
            note += f", by {extra['value_100']:.2e} at rtol/100"
        worst = _worst(self.tight_x, self.x, self.names) if moved > 0 else None
        return _verdict(
            "tolerance", status, moved, note, worst=worst, changed=changed, extra=extra
        )

    def check_grid(self) -> dict[str, Any]:
        skip = self._deterministic("grid")
        if skip:
            return skip
        t = np.asarray(self.result._time, dtype=float)
        if t.size < 2:
            return _verdict("grid", "unsupported", note="the run has fewer than two time points")
        if self.run.get("sample_times") is not None:
            fine = np.concatenate(
                [
                    np.linspace(a, b, GRID_FACTOR + 1)[:-1]
                    for a, b in zip(t[:-1], t[1:], strict=True)
                ]
                + [t[-1:]]
            )
            changed: dict[str, Any] = {"sample_times": f"{GRID_FACTOR}x the points"}
            again = self.rerun(sample_times=[float(v) for v in fine])
        else:
            n = (int(self.run["n_points"]) - 1) * GRID_FACTOR + 1
            changed = {"n_points": n}
            again = self.rerun(n_points=n)
        x = np.asarray(again._species, dtype=float)[::GRID_FACTOR]
        return self.compare("grid", x, changed)

    def check_jacobian(self) -> dict[str, Any]:
        skip = self._deterministic("jacobian")
        if skip:
            return skip
        backend = self.rec.get("backend") or {}
        if backend.get("jacobian") == "fd":
            return _verdict(
                "jacobian", "skip", note="the run already used a difference-quotient Jacobian"
            )
        changed = {"jacobian": "fd"}
        again = self.rerun(jacobian="fd")
        return self.compare("jacobian", np.asarray(again._species, dtype=float), changed)

    def check_backend(self) -> dict[str, Any]:
        skip = self._deterministic("backend")
        if skip:
            return skip
        rhs = (self.rec.get("backend") or {}).get("rhs")
        if rhs == "interpreted":
            from bngsim._exceptions import CodegenToolchainError

            changed: dict[str, Any] = {"codegen": True}
            try:
                again = self.rerun(codegen=True)
            except CodegenToolchainError as e:
                return _verdict("backend", "skip", note=f"no compiled path here: {_first_line(e)}")
            used = (again.record.get("backend") or {}).get("rhs")
            if used == "interpreted":
                return _verdict(
                    "backend",
                    "skip",
                    note="codegen=True ran interpreted: no compiled path to compare",
                )
        else:
            changed = {"codegen": False}
            again = self.rerun(codegen=False)
        return self.compare("backend", np.asarray(again._species, dtype=float), changed)

    def check_second_engine(self) -> dict[str, Any]:
        skip = self._deterministic("second_engine")
        if skip:
            return skip
        return _second_engine(self)

    def check_replicates(self, n: int) -> dict[str, Any]:
        if not self.stochastic:
            return _verdict("replicates", "unsupported", note="the run is deterministic")
        seed = self.run.get("seed")
        if seed is None:
            return _verdict("replicates", "unsupported", note="the record gives no seed")
        obs = np.asarray(self.result._observables, dtype=float)
        names = list(self.result._observable_names)
        if not names:
            obs, names = self.x, self.names
        last = np.empty((n, len(names)))
        for i in range(n):
            again = self.rerun(seed=(int(seed) + 1 + i) % (2**31 - 1))
            block = again._observables if self.result._observable_names else again._species
            last[i] = np.asarray(block, dtype=float)[-1]
        mean = last.mean(axis=0)
        se = last.std(axis=0, ddof=1) / math.sqrt(n)
        mine = obs[-1]
        z = np.where(se > 0, (mine - mean) / np.where(se > 0, se, 1.0), 0.0)
        cols = [
            {
                "name": names[j],
                "run_last": _num(mine[j]),
                "mean_last": _num(mean[j]),
                "se_last": _num(se[j]),
                "z": _num(z[j]),
            }
            for j in range(len(names))
        ]
        worst = int(np.argmax(np.abs(z))) if len(names) else None
        note = (
            f"{n} reruns at seeds {int(seed) + 1} to {int(seed) + n}: at the last time each "
            "observable's mean over them, its standard error, and how many standard errors "
            "this run is from it; one trajectory is one draw, so report a mean over replicates"
        )
        return _verdict(
            "replicates",
            "pass",
            None if worst is None else float(abs(z[worst])),
            note,
            worst=None if worst is None else names[worst],
            changed={"seed": f"{n} others"},
            extra={"n_replicates": n, "columns": cols},
        )


def _second_engine(ctx: _Context) -> dict[str, Any]:
    """The run made again in libRoadRunner, from the model file, where it can be
    made the same: an SBML or Antimony model, unchanged since its load, run
    from its initial state at the file's own parameters."""
    name = "second_engine"
    model = ctx.rec.get("model") or {}
    fmt = model.get("format")
    if fmt not in ("sbml", "antimony"):
        return _verdict(
            name,
            "unsupported",
            note=f"libRoadRunner reads SBML and Antimony; this model is {fmt!r}",
        )
    try:
        import roadrunner
    except ImportError:
        return _verdict(
            name,
            "skip",
            note="libRoadRunner is not installed (pip install libroadrunner): the model as "
            "bngsim read it was not checked against an independent reader",
        )
    if ctx.run.get("started_from") != "initial":
        return _verdict(name, "skip", note="the run did not start from the model's initial state")
    if model.get("edits"):
        return _verdict(name, "skip", note="the model was changed after its load")
    if ctx.run.get("sample_times") is not None or ctx.run.get("steady_state"):
        return _verdict(
            name, "skip", note="only a plain t_span run is made again in libRoadRunner"
        )
    if ctx.run.get("parameters_pinned"):
        return _verdict(name, "skip", note="a derived parameter was pinned by set_param")
    path = model.get("path")
    import bngsim

    try:
        fresh = bngsim.Model.load(path) if path else None
    except Exception as e:  # noqa: BLE001
        return _verdict(
            name, "skip", note=f"the model file could not be read again: {_first_line(e)}"
        )
    if fresh is None:
        return _verdict(
            name, "skip", note="the record names no model file (a model loaded from text)"
        )
    held = ctx.run.get("parameters") or {}
    for p in fresh.param_names:
        if p in held and held[p] is not None:
            a, b = float(held[p]), float(fresh.get_param(p))
            if not (a == b or (math.isnan(a) and math.isnan(b))):
                return _verdict(
                    name, "skip", note=f"the run's parameter {p} = {a:g} is not the file's {b:g}"
                )
    try:
        if fmt == "antimony":
            import antimony

            if antimony.loadAntimonyFile(path) < 0:
                raise RuntimeError(antimony.getLastError())
            sbml = antimony.getSBMLString(antimony.getMainModuleName())
            rr = roadrunner.RoadRunner(sbml)
        else:
            rr = roadrunner.RoadRunner(path)
        ids = list(rr.model.getFloatingSpeciesIds()) + list(rr.model.getBoundarySpeciesIds())
        missing = [s for s in ctx.names if s not in ids]
        if missing:
            return _verdict(name, "skip", note=f"libRoadRunner has no species {missing[0]!r}")
        assert ctx.rtol is not None and ctx.atol is not None
        rr.integrator.relative_tolerance = float(ctx.rtol)
        rr.integrator.absolute_tolerance = float(ctx.atol)
        sel = ["time"] + [f"[{s}]" for s in ctx.names]
        out = rr.simulate(
            float(ctx.run["t_start"]), float(ctx.run["t_end"]), int(ctx.run["n_points"]), sel
        )
        x = np.asarray(out, dtype=float)[:, 1:]
    except Exception as e:  # noqa: BLE001 - the other engine's failure is not bngsim's
        return _verdict(
            name,
            "skip",
            note=f"libRoadRunner could not run it: {type(e).__name__}: {_first_line(e)}",
        )
    v = ctx.compare(name, x, {"engine": f"libRoadRunner {roadrunner.__version__}"})
    return v


def _timed_out(e: BaseException) -> bool:
    from bngsim._exceptions import BngsimError

    code = getattr(e, "code", "") if isinstance(e, BngsimError) else ""
    return "timeout" in str(code) or "timed out" in str(e).lower()


def _first_line(e: BaseException) -> str:
    return str(e).splitlines()[0][:300] if str(e) else ""


def _g(x: Any) -> str:
    return "?" if x is None else f"{x:.3g}"


def text(r: dict[str, Any]) -> str:
    """The plain-text form of a report: a verdict line, one line per check,
    and what to do next."""
    lines = []
    verdict = "OK" if r.get("ok") else "NOT OK"
    head = f"{verdict}: {r.get('n_pass', 0)} passed, {r.get('n_diff', 0)} moved the answer"
    if r.get("method"):
        head += f" ({r['method']} run"
        if r.get("rtol") is not None:
            head += f", rtol {_g(r['rtol'])}"
        head += ")"
    lines.append(head)
    for c in r.get("checks", []):
        line = f"  {c['name']}: {c['status']}"
        if c.get("note"):
            line += f": {c['note']}"
        if c.get("worst_column"):
            line += f"; most in {c['worst_column']}"
        lines.append(line)
    if r.get("next"):
        lines.append(f"next: {r['next']}")
    return "\n".join(lines)
