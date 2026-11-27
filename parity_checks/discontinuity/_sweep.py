"""The sweep driver: run a case's points through bngsim and judge each one
against its reference (issue #1912).

One function per job:

* ``run_point(case, point)``: one run, one ``Row``. The model is loaded once
  per process, and each point sets its parameters with ``set_param`` and resets
  the state, so one compile serves the whole case.
* ``run_row(case, row)``: every point of one parameter row, which share one
  reference solve, and the row's convergence rows.
* ``check_spec(case)``: the case's spec against the model text, by bngsim's
  own right-hand side at states away from every surface.
* ``slice_params(cases)``: the PR slice, as pytest parameters, with each known
  failure as a strict expected failure.

A row's outcome is the one taxonomy every parity suite shares
(``_core.Outcome``): PASS, DIFF (a wrong answer, or a run that returned where
a refusal was expected), EXCEPTION (bngsim raised; ``subclass`` is ``solver``
for a CVODE failure), TIMEOUT, UNSUPPORTED (a typed refusal: the run declined
loudly) and REFERENCE_FAILED (the reference could not answer: not a finding).
"""

from __future__ import annotations

import builtins
import importlib.util
import tempfile
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import _judge as J
import bngsim
import numpy as np
from _case import Case, Offset, Point, SteadyState
from _reference import NoReference, ReferenceUnavailable, window_peak

#: The refusals bngsim declares: a run that declines loudly is not a wrong
#: answer.
REFUSALS = (bngsim.SensitivityUnsupportedError, bngsim.CodegenDeclinedError)
#: What each ODE engine passes as ``Simulator(codegen=...)``: ``None`` leaves the
#: choice to bngsim, as every seed case runs.
CODEGEN = {"ode": None, "ode-compiled": True, "ode-interpreted": False}
#: The reason a diffrax point skips where JAX is absent: one ``python/tests``'
#: conftest already declares.
NO_JAX = "JAX not installed"


class EngineUnavailable(RuntimeError):
    """The engine a point asks for is not installed here (JAX, for diffrax).
    The PR slice skips the point; the nightly provisions every engine."""


def resolve(raises) -> type[BaseException]:
    """An exception class from a ``Known.raises`` or an ``Owed.refusal``: a
    class as it is, or a name, looked up in bngsim and then in the builtins."""
    if isinstance(raises, type) and issubclass(raises, BaseException):
        return raises
    cls = getattr(bngsim, str(raises), None) or getattr(builtins, str(raises), None)
    if not (isinstance(cls, type) and issubclass(cls, BaseException)):
        raise ValueError(f"{raises!r} names no exception class in bngsim or the builtins")
    return cls


#: The plain run the spec check reads its states from: an ``Offset`` grid, so
#: it holds a state either side of every crossing, ``SPEC_SHIFT`` of
#: ``max(1, |t*|)`` away.
SPEC_RTOL = 1e-8
SPEC_POINTS = 31
SPEC_SHIFT = 1e-4
#: States within this of a surface are left out of the spec check: there the
#: live comparison bngsim reads and the spec's branch variable may differ.
SPEC_MARGIN = 1e-9
SPEC_RTOL_MATCH = 1e-9
#: The flows are compared at the row's own values and again with every
#: parameter the spec reads moved by these factors (each parameter by a
#: different one), so a spec that agrees with the text only where a parameter
#: is 1 (``kx**2`` for ``kx``) is caught.
SPEC_FACTORS = (1.17, 0.83, 1.31)


@dataclass
class Row:
    """One row of the report: a point, or a convergence group."""

    model_id: str
    method: str
    outcome: str
    note: str = ""
    subclass: str | None = None
    seconds: float | None = None
    work: dict | None = None
    backend: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model_id}|{self.method}"

    def to_result(self, versions: dict | None = None) -> dict:
        """A ``_core.JobResult``'s fields."""
        failed = self.outcome in ("EXCEPTION", "TIMEOUT", "UNSUPPORTED", "REFERENCE_FAILED")
        timing = None
        if self.seconds is not None or self.work:
            bn: dict = {}
            if self.seconds is not None:
                bn["integrate_cold_sec"] = round(self.seconds, 6)
            if self.work:
                bn["work"] = self.work
            if self.backend:
                bn["config"] = {"codegen": self.backend}
            timing = {"bngsim": bn}
        return {
            "model_id": self.model_id,
            "method": self.method,
            "reference_engine": f"discontinuity:{self.extra.get('reference', '')}",
            "outcome": self.outcome,
            "exception": self.note if failed else "",
            "comment": "" if failed else self.note,
            "subclass": self.subclass,
            "wall_sec": None if self.seconds is None else round(self.seconds, 6),
            "versions": dict(versions or {}),
            "timing": timing,
            "extra": self.extra or None,
        }


# ── Loading ─────────────────────────────────────────────────────────────────

_MODELS: dict[tuple[str, str, str], bngsim.Model] = {}


def load(case: Case) -> bngsim.Model:
    """The case's model, loaded once per process (per name and text, so a
    case rewritten under the same name is loaded afresh)."""
    key = (case.name, case.fmt, case.model)
    if key not in _MODELS:
        if case.fmt == "net":
            with tempfile.TemporaryDirectory(prefix="discontinuity-") as d:
                path = Path(d) / f"{case.name}.net"
                path.write_text(case.model)
                _MODELS[key] = bngsim.Model.from_net(str(path))
        elif case.fmt == "antimony":
            _MODELS[key] = bngsim.Model.from_antimony_string(case.model)
        else:
            _MODELS[key] = bngsim.Model.from_sbml_string(case.model)
    return _MODELS[key]


def prepare(case: Case, values: dict[str, float]) -> bngsim.Model:
    """The model at these values, reset to the initial state they imply."""
    model = load(case)
    for k, v in values.items():
        model.set_param(k, v)
    model.reset()
    return model


# ── One point ───────────────────────────────────────────────────────────────


def _failure(e: BaseException) -> tuple[str, str | None]:
    if isinstance(e, bngsim.SimulationTimeout):
        return "TIMEOUT", None
    if isinstance(e, REFUSALS):
        return "UNSUPPORTED", type(e).__name__
    code = getattr(e, "code", None) or ""
    if isinstance(e, bngsim.BngsimError):
        return "EXCEPTION", "solver" if code.startswith("solver.") else type(e).__name__
    return "EXCEPTION", f"harness:{type(e).__name__}"


def _first_line(e: BaseException) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0] if text else ''}"


@dataclass
class Ran:
    """What one run reported, whichever engine ran it."""

    times: np.ndarray
    species: np.ndarray  # (n_t, n_species)
    names: list[str]
    sens: np.ndarray | None  # (n_t, n_species, n_params)
    sens_params: list[str]
    converged: bool
    work: dict
    backend: str | None


def execute(case: Case, model: bngsim.Model, p: Point, request: dict, atol: float) -> Ran:
    """Run one point on its engine: a time course or a steady state with
    ``Simulator(method="ode")``, or a time course with ``run_diffrax``."""
    window = case.window(case.values(p.row))
    if isinstance(p.grid, SteadyState):
        sim = bngsim.Simulator(model, method="ode", codegen=CODEGEN[p.engine])
        ss = sim.steady_state(
            method=p.grid.method, rtol=p.rtol, atol=atol, mask=list(case.steady_mask) or None
        )
        return Ran(
            times=np.array([np.inf]),
            species=np.asarray(ss.concentrations, dtype=float)[None, :],
            names=list(ss.species_names),
            sens=None,
            sens_params=[],
            converged=bool(ss.converged),
            work={"n_steps": int(ss.n_steps), "n_rhs_evals": int(ss.n_rhs_evals)},
            backend=sim.codegen_backend,
        )
    if p.engine == "diffrax":
        from bngsim._jax_rhs import jax_available

        if not jax_available() or importlib.util.find_spec("diffrax") is None:
            raise EngineUnavailable(NO_JAX)
        from bngsim._diffrax_solver import run_diffrax

        out = run_diffrax(
            model,
            t_start=window[0],
            t_end=window[1],
            n_points=int(request["n_points"]),
            rtol=p.rtol,
            atol=atol,
        )
        return Ran(
            times=np.asarray(out["time"], dtype=float),
            species=np.asarray(out["species"], dtype=float),
            names=list(out["species_names"]),
            sens=None,
            sens_params=[],
            converged=True,
            work={"n_steps": int(out["n_steps"])},
            backend="diffrax",
        )
    sim = bngsim.Simulator(
        model, method="ode", sensitivity_params=list(p.sens) or None, codegen=CODEGEN[p.engine]
    )
    res = sim.run(t_span=window, rtol=p.rtol, atol=atol, timeout=case.timeout, **request)
    return Ran(
        times=np.asarray(res.time, dtype=float),
        species=np.asarray(res.species, dtype=float),
        names=list(res.species_names),
        sens=np.asarray(res.sensitivities, dtype=float) if p.sens else None,
        sens_params=list(res.sensitivity_params) if p.sens else [],
        converged=True,
        work=dict(res.solver_stats),
        backend=sim.codegen_backend,
    )


def run_point(case: Case, p: Point, *, strict: bool = False) -> Row:
    """Run one point and judge it. With ``strict`` (the PR slice) bngsim's
    exceptions and the reference's refusals propagate, so a test sees their
    types; otherwise they become rows.

    Each stage fails as itself: preparing the model is the harness's (a
    ``set_param`` that raises is never an expected refusal), the run is
    bngsim's, and a reference that raises anything but
    ``ReferenceUnavailable`` costs this row, not its whole job.

    A point the reference says owes a refusal (``Owed``) passes when the run
    raises it, and is DIFF when the run returns: there is no number to judge.
    A steady state must also report ``converged=True``."""
    ref = case.reference
    values = case.values(p.row)
    window = case.window(values)
    row = Row(case.row_id(p.row), p.method, "PASS", extra={"reference": ref.kind})
    known = case.known.get(case.key(p))
    if known is not None:
        row.extra["known"] = known.issue
    try:
        crossings = ref.crossings(values, window)
        owed_fn = getattr(ref, "owed", None)
        owed = owed_fn(values, p.sens) if owed_fn is not None else None
    except Exception as e:
        if strict:
            raise
        return _reference_failed(row, "crossings", e)
    row.extra["crossings"] = [float(c) for c in crossings]
    request = p.grid.request(window, crossings)
    atol = p.rtol * case.atol_ratio
    try:
        model = prepare(case, values)
        refusal = resolve(owed.refusal) if owed is not None else None
    except Exception as e:
        if strict:
            raise
        row.subclass = f"harness:{type(e).__name__}"
        return _set(row, "EXCEPTION", f"preparing the model: {_first_line(e)}")
    seen: list[str] = []
    try:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = execute(case, model, p, request, atol)
        row.seconds = time.perf_counter() - t0
        seen = sorted({type(w.message).__name__ for w in caught if isinstance(w.message, Warning)})
    except EngineUnavailable as e:
        if strict:
            raise
        row.subclass = "harness:EngineUnavailable"
        return _set(row, "EXCEPTION", str(e))
    except Exception as e:
        if refusal is not None and isinstance(e, refusal):
            return _set(row, "PASS", f"refused as owed ({owed.reason}): {_first_line(e)}")
        if strict:
            raise
        outcome, sub = _failure(e)
        row.subclass = sub
        return _set(row, outcome, _first_line(e))
    row.work = res.work
    row.backend = res.backend
    if seen:
        row.extra["warnings"] = seen
    if owed is not None:
        return _set(row, "DIFF", f"returned where {refusal.__name__} was expected: {owed.reason}")
    if not isinstance(p.grid, SteadyState):
        moved = grid_mismatch(request, res.times, window)
        if moved:
            return _set(row, "DIFF", f"the {p.grid.label} grid: {moved}")
    names = res.names
    if isinstance(ref, NoReference):
        sp = {n: res.species[:, i] for i, n in enumerate(names)}
        outcome, note = J.judge_invariants(
            ref.invariants,
            res.times,
            sp,
            values,
            rtol=p.rtol,
            atol=atol,
            k_state=case.judge.k_state,
        )
        return _set(row, outcome, note)
    idx = [names.index(s) for s in case.observe]
    x = res.species[:, idx]
    S = None
    if p.sens:
        cols = [res.sens_params.index(q) for q in p.sens]
        S = res.sens[:, idx, :][:, :, cols]
    try:
        sol = ref.solve(values, window, res.times, case.observe, p.sens)
        peak = (
            window_peak(ref, values, window, case.observe, p.sens)
            if case.judge.peak == "window" and not isinstance(p.grid, SteadyState)
            else None
        )
    except Exception as e:
        if strict:
            raise
        return _reference_failed(row, "solve", e)
    v = J.judge(
        sol,
        x,
        S,
        rtol=p.rtol,
        atol=atol,
        observe=case.observe,
        sens=p.sens,
        params=values,
        k_state=case.judge.k_state,
        k_sens=case.judge.k_sens,
        window_peak=peak,
    )
    row.extra.update(
        {
            "state_ratio": _round(v.state_ratio),
            "sens_ratio": None if v.sens_ratio is None else _round(v.sens_ratio),
        }
    )
    if v.sides:
        row.extra["sides"] = v.sides
    if v.unresolved:
        row.extra["unresolved"] = v.unresolved
    if not res.converged:
        return _set(row, "DIFF", f"steady_state reports converged=False; {v.note}")
    return _set(row, v.outcome, v.note)


def _reference_failed(row: Row, stage: str, e: BaseException) -> Row:
    """A reference that declines is REFERENCE_FAILED; one that breaks is a
    fault in the suite, EXCEPTION, on this row alone."""
    if isinstance(e, ReferenceUnavailable):
        return _set(row, "REFERENCE_FAILED", f"{stage}: {e}")
    row.subclass = f"reference:{type(e).__name__}"
    return _set(row, "EXCEPTION", f"the reference's {stage} raised {_first_line(e)}")


def grid_mismatch(request: dict, times, t_span) -> str | None:
    """How the run's times differ from those it was asked for, or None: the
    sample times to the bit, or ``n_points`` rows from the window's start to
    its end."""
    got = np.asarray(times, float)
    if "sample_times" in request:
        want = np.asarray(request["sample_times"], float)
    else:
        n = int(request["n_points"])
        if len(got) != n:
            return f"{len(got)} rows reported where {n} were asked for"
        want = got.copy()
        want[[0, -1]] = [float(t_span[0]), float(t_span[1])]
    if got.shape != want.shape:
        return f"{len(got)} rows reported where {len(want)} were asked for"
    if np.any(np.diff(got) <= 0):
        return f"the rows' times do not increase: {got.tolist()[:8]}"
    off = np.flatnonzero(got != want)
    if not len(off):
        return None
    i = int(off[0])
    return (
        f"the row asked for at t={float(want[i])!r} came back at t={float(got[i])!r} "
        f"({len(off)} rows moved)"
    )


def _set(row: Row, outcome: str, note: str) -> Row:
    row.outcome = outcome
    row.note = note
    known = row.extra.get("known")
    if known is not None:
        row.note = f"known #{known}: {note}"
    return row


def _round(x: float) -> float:
    return float(f"{x:.4g}") if np.isfinite(x) else x


# ── Rows of points, and convergence ─────────────────────────────────────────


def converge_row(case: Case, group: tuple, judged: list[tuple[Point, Row]]) -> Row:
    """The convergence row of one group, from its points and their rows."""
    c = J.converge([(p.rtol, _ratio(r)) for p, r in judged])
    model_id, method = case.converge_key(group).split("|", 1)
    out = Row(model_id, method, c.outcome, c.note, extra={"reference": case.reference.kind})
    known = case.known.get(case.converge_key(group))
    if known is not None:
        out.extra["known"] = known.issue
        out.note = f"known #{known.issue}: {out.note}"
    return out


def _ratio(r: Row) -> float | None:
    if r.outcome not in ("PASS", "DIFF") or "state_ratio" not in r.extra:
        return None
    return max(r.extra["state_ratio"], r.extra.get("sens_ratio") or 0.0)


def run_row(case: Case, row: tuple, *, on_point=None) -> list[Row]:
    """Every point of one parameter row, then its convergence rows. Each
    column set is compiled first, untimed, so a point's time is its run's and
    not the compiler's."""
    warm(case, row)
    points = case.points(row)
    rows = []
    for p in points:
        r = run_point(case, p)
        rows.append(r)
        if on_point:
            on_point(r)
    if case.judge.converge:
        by_point = dict(zip(points, rows, strict=True))
        for group, pts in case.groups(points).items():
            rows.append(converge_row(case, group, [(p, by_point[p]) for p in pts]))
    return rows


def warm(case: Case, row: tuple = ()) -> None:
    """Compile the case's code for every column set and engine it sweeps, by a
    short run of each. What fails here is left for the points themselves to
    report."""
    t0, t1 = case.window(case.values(row))
    short = (t0, t0 + 1e-3 * (t1 - t0))
    for sens in case.axes.sens:
        for engine in case.axes.engines:
            try:
                model = prepare(case, case.values(row))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if engine == "diffrax":
                        from bngsim._diffrax_solver import run_diffrax

                        run_diffrax(model, t_start=short[0], t_end=short[1], n_points=2)
                        continue
                    sim = bngsim.Simulator(
                        model,
                        method="ode",
                        sensitivity_params=list(sens) or None,
                        codegen=CODEGEN[engine],
                    )
                    sim.run(t_span=short, n_points=2, rtol=1e-6, timeout=case.timeout)
            except Exception:  # noqa: BLE001, S112 - the points report it
                continue


# ── The spec against the text ───────────────────────────────────────────────


def check_spec(case: Case) -> list[str]:
    """Where the case's spec and its model text disagree; [] when they agree.

    For every parameter row: the spec's states must be the model's species,
    its start the model's initial state, and its flows bngsim's right-hand
    side, at the states a plain run passes through (a state either side of
    every crossing among them), away from every surface, at the row's values
    and again at values moved by ``SPEC_FACTORS``. Every surface must be
    compared on both of its sides somewhere."""
    spec = case.spec
    if spec is None:
        return []
    names = list(load(case).species_names)
    if set(names) != set(spec.states):
        return [
            f"the spec's states {sorted(spec.states)} are not the model's species {sorted(names)}"
        ]
    order = [names.index(s) for s in spec.states]
    problems: list[str] = []
    sides: list[set[int]] = [set() for _ in spec.surfaces]
    for row in case.spec_check_rows():
        values = case.values(row)
        where = case.row_id(row)
        model = prepare(case, values)
        start = np.asarray(model.get_state(), dtype=float)[order]
        want = spec.start(values)
        if not np.allclose(start, want, rtol=1e-14, atol=0):
            problems.append(
                f"{where}: the model starts at {start.tolist()}, the spec at {want.tolist()}"
            )
        window = case.window(values)
        try:
            crossings = case.reference.crossings(values, window)
        except Exception as e:  # noqa: BLE001 - a reference that cannot say is a problem too
            problems.append(f"{where}: the reference gives no crossings: {_first_line(e)}")
            crossings = ()
        request = Offset(SPEC_POINTS, shift=SPEC_SHIFT).request(window, crossings)
        res = bngsim.Simulator(model, method="ode").run(
            t_span=window, rtol=SPEC_RTOL, atol=SPEC_RTOL * 1e-2, **request
        )
        states = list(
            zip(np.asarray(res.time, float), np.asarray(res.species, float), strict=True)
        )
        settings = [values] + [
            {
                **values,
                **{
                    q: values[q] * SPEC_FACTORS[(i + j) % len(SPEC_FACTORS)]
                    for j, q in enumerate(spec.params)
                },
            }
            for i in range(len(SPEC_FACTORS))
        ]
        for k, at in enumerate(settings):
            for q, v in at.items():
                model.set_param(q, v)
            label = "its values" if k == 0 else "moved values"
            checked = 0
            for t, y_model in states:
                y = y_model[order]
                g = spec.g_values(t, y, at)
                if any(abs(v) <= SPEC_MARGIN for v in g):
                    continue
                s = [int(v > 0) for v in g]
                for i, b in enumerate(s):
                    sides[i].add(b)
                f_spec = spec.flows(t, y, at, s)
                f_model = np.asarray(model.rhs(y_model, float(t)), dtype=float)[order]
                scale = max(1.0, float(np.max(np.abs(f_spec))))
                if not np.allclose(f_model, f_spec, rtol=SPEC_RTOL_MATCH, atol=1e-12 * scale):
                    problems.append(
                        f"{where} at {label}, t={t:.10g}, branches {s}: bngsim's flows "
                        f"{f_model.tolist()}, the spec's {f_spec.tolist()}"
                    )
                checked += 1
            if checked < len(states) // 2:
                problems.append(
                    f"{where} at {label}: only {checked} of {len(states)} states lay away from every surface"
                )
        prepare(case, values)
    for surf, seen in zip(spec.surfaces, sides, strict=True):
        if seen != {0, 1}:
            problems.append(f"the surface of {surf.switch!r} was never compared on both its sides")
    return problems


# ── The PR slice ────────────────────────────────────────────────────────────


def slice_params(cases: Iterable[Case]) -> list:
    """The PR slice's points as ``pytest.param(case, point)``, with each known
    failure marked strict, so its fix turns the run red."""
    import pytest

    out = []
    for case in cases:
        for p in case.slice:
            marks = ()
            known = case.known.get(case.key(p))
            if known is not None:
                marks = (_xfail(known),)
            out.append(pytest.param(case, p, id=case.key(p), marks=marks))
    return out


def slice_groups(cases: Iterable[Case]) -> list:
    """The slice's convergence groups: those holding two tolerances or more."""
    import pytest

    out = []
    for case in cases:
        if not case.judge.converge:
            continue
        every_rtol = set(map(float, case.axes.rtol))
        for group, pts in case.groups(list(case.slice)).items():
            key = case.converge_key(group)
            # A known failure of a convergence row describes the full sweep's
            # group; the slice judges only the tolerances it holds, so the mark
            # applies only where those are all of them.
            known = case.known.get(key) if {p.rtol for p in pts} == every_rtol else None
            marks = (_xfail(known),) if known is not None else ()
            out.append(pytest.param(case, group, tuple(pts), id=key, marks=marks))
    return out


def _xfail(known):
    import pytest

    return pytest.mark.xfail(
        strict=True,
        raises=resolve(known.raises),
        reason=f"#{known.issue}: {known.reason}".rstrip(": "),
    )
