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

import tempfile
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import _judge as J
import bngsim
import numpy as np
from _case import Case, Offset, Point
from _reference import NoReference, ReferenceUnavailable

#: The refusals bngsim declares: a run that declines loudly is not a wrong
#: answer.
REFUSALS = (bngsim.SensitivityUnsupportedError, bngsim.CodegenDeclinedError)
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


def run_point(case: Case, p: Point, *, strict: bool = False) -> Row:
    """Run one point and judge it. With ``strict`` (the PR slice) bngsim's
    exceptions and the reference's refusals propagate, so a test sees their
    types; otherwise they become rows.

    Each stage fails as itself: preparing the model is the harness's (a
    ``set_param`` that raises is never an expected refusal), the run is
    bngsim's, and a reference that raises anything but
    ``ReferenceUnavailable`` costs this row, not its whole job."""
    ref = case.reference
    values = case.values(p.row)
    row = Row(case.row_id(p.row), p.method, "PASS", extra={"reference": ref.kind})
    known = case.known.get(case.key(p))
    if known is not None:
        row.extra["known"] = known.issue
    try:
        crossings = ref.crossings(values, case.t_span)
    except Exception as e:
        if strict:
            raise
        return _reference_failed(row, "crossings", e)
    row.extra["crossings"] = [float(c) for c in crossings]
    request = p.grid.request(case.t_span, crossings)
    atol = p.rtol * case.atol_ratio
    try:
        model = prepare(case, values)
    except Exception as e:
        if strict:
            raise
        row.subclass = f"harness:{type(e).__name__}"
        return _set(row, "EXCEPTION", f"preparing the model: {_first_line(e)}")
    seen: list[str] = []
    try:
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(p.sens) or None)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sim.run(
                t_span=case.t_span, rtol=p.rtol, atol=atol, timeout=case.timeout, **request
            )
        row.seconds = time.perf_counter() - t0
        seen = sorted({type(w.message).__name__ for w in caught if isinstance(w.message, Warning)})
    except Exception as e:
        expected = isinstance(ref, NoReference) and ref.expect_refusal is not None
        if expected and isinstance(e, ref.expect_refusal):
            return _set(row, "PASS", f"refused as expected: {_first_line(e)}")
        if strict:
            raise
        outcome, sub = _failure(e)
        row.subclass = sub
        return _set(row, outcome, _first_line(e))
    row.work = dict(res.solver_stats)
    row.backend = sim.codegen_backend
    if seen:
        row.extra["warnings"] = seen
    moved = grid_mismatch(request, res.time, case.t_span)
    if moved:
        return _set(row, "DIFF", f"the {p.grid.label} grid: {moved}")
    names = list(res.species_names)
    all_species = np.asarray(res.species, dtype=float)
    if isinstance(ref, NoReference):
        if ref.expect_refusal is not None:
            return _set(row, "DIFF", f"returned where {ref.expect_refusal.__name__} was expected")
        sp = {n: all_species[:, i] for i, n in enumerate(names)}
        outcome, note = J.judge_invariants(
            ref.invariants,
            res.time,
            sp,
            values,
            rtol=p.rtol,
            atol=atol,
            k_state=case.judge.k_state,
        )
        return _set(row, outcome, note)
    idx = [names.index(s) for s in case.observe]
    x = all_species[:, idx]
    S = None
    if p.sens:
        cols = [list(res.sensitivity_params).index(q) for q in p.sens]
        S = np.asarray(res.sensitivities, dtype=float)[:, idx, :][:, :, cols]
    try:
        sol = ref.solve(values, case.t_span, np.asarray(res.time, float), case.observe, p.sens)
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
    """Compile the case's code for every column set it sweeps, by a short run
    of each. What fails here is left for the points themselves to report."""
    t0, t1 = map(float, case.t_span)
    for sens in case.axes.sens:
        try:
            model = prepare(case, case.values(row))
            sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(sens) or None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim.run(
                    t_span=(t0, t0 + 1e-3 * (t1 - t0)), n_points=2, rtol=1e-6, timeout=case.timeout
                )
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
    for row in case.rows():
        values = case.values(row)
        where = case.row_id(row)
        model = prepare(case, values)
        start = np.asarray(model.get_state(), dtype=float)[order]
        want = spec.start(values)
        if not np.allclose(start, want, rtol=1e-14, atol=0):
            problems.append(
                f"{where}: the model starts at {start.tolist()}, the spec at {want.tolist()}"
            )
        try:
            crossings = case.reference.crossings(values, case.t_span)
        except Exception as e:  # noqa: BLE001 - a reference that cannot say is a problem too
            problems.append(f"{where}: the reference gives no crossings: {_first_line(e)}")
            crossings = ()
        request = Offset(SPEC_POINTS, shift=SPEC_SHIFT).request(case.t_span, crossings)
        res = bngsim.Simulator(model, method="ode").run(
            t_span=case.t_span, rtol=SPEC_RTOL, atol=SPEC_RTOL * 1e-2, **request
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
        strict=True, raises=known.raises, reason=f"#{known.issue}: {known.reason}".rstrip(": ")
    )
