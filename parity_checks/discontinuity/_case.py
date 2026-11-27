"""The discontinuity suite's case format (issue #1912, phase 0 of #1781).

A case is one model whose rate laws switch, the axes it is swept over, and a
reference that does not go through bngsim's switch machinery. The sweep driver
(``_sweep.py``) expands it into points, runs bngsim at each, and judges each
run against the reference by the rule the case states (``Judge``).

Nothing here imports bngsim: a case and its reference are data and plain
mathematics, so the reference cannot share a bug with the engine it judges.

Keys. Every point has a stable key, ``<row>|<method>``, which the nightly's
verdict files are keyed by (``verdicts.py`` reads ``model_id|method``):

* the row is the case's name and its parameter row, ``clock_step[b=-1.0]``,
  or the bare name for the base row;
* the method names the run, ``rtol=1e-10,grid=u31,sens=kx+c``, or
  ``converge,grid=u31,sens=kx+c`` for the row that judges one grid and
  column set across the rtol axis.

Renaming a case, a parameter or a grid renames its rows, which the nightly
reports as rows dropped and rows added; that is deliberate.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

#: The families phase 0 builds (#1781's list). A case names one.
FAMILIES = (
    "pulse",
    "step",
    "sawtooth",
    "bend",
    "clock",
    "coincident",
    "event",
    "residual",
    "slide",
    "parked",
)
#: What a switch's condition reads: time and parameters only (its crossing
#: time is known before the run), or the state. #1922's pass is to classify
#: every switch this way from its expression; the suite's declarations are
#: the answers it will be checked against.
READS = ("time", "state")
#: Whether the rate law jumps at the switch or only its derivative does.
EFFECTS = ("jump", "bend")
#: A BNGL model enters as the ``.net`` BNG2.pl writes for it, so that no run of
#: the suite needs BNG2.pl (most CI legs have none). Antimony is read by
#: ``Model.from_antimony_string``, through SBML, so its events are SBML's.
FORMATS = ("net", "sbml", "antimony")
#: How a point is run. ``ode`` is ``Simulator(method="ode")`` with codegen left
#: to its own choice, as every seed case runs; ``ode-compiled`` and
#: ``ode-interpreted`` force codegen on and off; ``diffrax`` is the experimental
#: JAX integrator ``run_diffrax`` (plain runs on ``Uniform`` grids only, as it
#: takes ``n_points`` and integrates no columns). Keys name a point's engine
#: only when it is not ``ode``, so the seed cases' keys are unchanged.
ENGINES = ("ode", "ode-compiled", "ode-interpreted", "diffrax")
#: What a column is scaled by (``Judge.peak``): its largest magnitude over the
#: samples, or over the whole window.
PEAKS = ("samples", "window")
#: The tolerances every case is swept over unless it says otherwise.
RTOLS = (1e-6, 1e-8, 1e-10, 1e-12)
#: The most points a case may put in the PR slice. A count, not a wall-clock
#: budget, so the rule is the same on every runner.
SLICE_CAP = 8


class CaseError(ValueError):
    """A case is written wrongly: a fault in the suite, not in bngsim."""


def num(v: float) -> str:
    """A number as keys spell it: the shortest repr that reads back exactly."""
    return repr(float(v))


# ── Switches ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Switch:
    """One switch in the model's rate laws, as its expression classifies it."""

    name: str
    reads: str
    effect: str

    def __post_init__(self):
        if self.reads not in READS:
            raise CaseError(f"switch {self.name!r}: reads={self.reads!r}, not one of {READS}")
        if self.effect not in EFFECTS:
            raise CaseError(f"switch {self.name!r}: effect={self.effect!r}, not one of {EFFECTS}")


# ── Output grids ────────────────────────────────────────────────────────────
#
# A grid turns the window and the reference's crossing times into the run's
# output keywords. The reference is then evaluated at the times the run
# reports, so a grid never has to agree with bngsim on how n_points spaces.


@dataclass(frozen=True)
class Uniform:
    """``n_points=n``: the run's own grid, which knows nothing of the crossings.
    Only its row count and both ends are checked: the reference is read at the
    interior times the run reports, so a row recorded early is judged where it
    was recorded. ``Sampled`` asks for the same times and checks every one."""

    n: int

    @property
    def label(self) -> str:
        return f"u{self.n}"

    def request(self, t_span, crossings) -> dict:
        return {"n_points": self.n}


@dataclass(frozen=True)
class Aligned:
    """n uniform points, and a sample exactly at every crossing in the window."""

    n: int

    @property
    def label(self) -> str:
        return f"aligned{self.n}"

    def request(self, t_span, crossings) -> dict:
        t0, t1 = map(float, t_span)
        times = {float(t) for t in _linspace(t0, t1, self.n)}
        times |= {float(c) for c in crossings if t0 < c < t1}
        return {"sample_times": sorted(times)}


@dataclass(frozen=True)
class Offset:
    """n uniform points, and one a hair before and one a hair after every
    crossing in the window: ``shift`` of ``max(1, |t*|)`` away, far past the
    double's spacing, so each lies on one side for certain."""

    n: int
    shift: float = 1e-9

    @property
    def label(self) -> str:
        return f"offset{self.n}"

    def request(self, t_span, crossings) -> dict:
        t0, t1 = map(float, t_span)
        times = {float(t) for t in _linspace(t0, t1, self.n)}
        for c in crossings:
            d = self.shift * max(1.0, abs(c))
            times |= {t for t in (c - d, c + d) if t0 < t < t1}
        return {"sample_times": sorted(times)}


@dataclass(frozen=True)
class Sampled:
    """n uniform points as ``sample_times`` and nothing added at a crossing:
    the times a user asks for, each of which must come back to the bit (#1799:
    a sensitivity run recorded the row asked for at 3e-11 from 2.95e-11)."""

    n: int

    @property
    def label(self) -> str:
        return f"sampled{self.n}"

    def request(self, t_span, crossings) -> dict:
        return {"sample_times": _linspace(*map(float, t_span), self.n)}


#: How ``Simulator.steady_state`` may be asked.
STEADY_METHODS = ("integration", "newton")


@dataclass(frozen=True)
class SteadyState:
    """Not a time course: ``Simulator.steady_state(method=...)``, judged against
    the reference's limit as t goes to infinity (a ``Limit``). The run must also
    report ``converged=True``. A case's grids are all of this kind or none."""

    method: str

    @property
    def label(self) -> str:
        return f"steady-{self.method}"

    def request(self, t_span, crossings) -> dict:
        return {"method": self.method}


Grid = Uniform | Aligned | Offset | Sampled | SteadyState
#: The grids that ask for a time course.
COURSES = (Uniform, Aligned, Offset, Sampled)


def _linspace(t0: float, t1: float, n: int) -> list[float]:
    if n < 2:
        raise CaseError(f"a grid needs at least 2 points, not {n}")
    return [t0 + (t1 - t0) * i / (n - 1) for i in range(n - 1)] + [t1]


# ── Axes and points ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Axes:
    """What a case is swept over. ``params`` is a list of rows, each a set of
    overrides of the case's base values (parameters are often coupled, so they
    are not a product); the other axes are. ``()`` in ``sens`` is a plain run."""

    params: tuple[Mapping[str, float], ...] = ({},)
    rtol: tuple[float, ...] = RTOLS
    grids: tuple[Grid, ...] = (Uniform(4), Uniform(31))
    sens: tuple[tuple[str, ...], ...] = ((),)
    engines: tuple[str, ...] = ("ode",)


@dataclass(frozen=True)
class Point:
    """One run: a parameter row, a tolerance, a grid, a set of columns and an
    engine."""

    row: tuple[tuple[str, float], ...]
    rtol: float
    grid: Grid
    sens: tuple[str, ...]
    engine: str = "ode"

    @property
    def method(self) -> str:
        return (
            f"rtol={self.rtol:g},grid={self.grid.label},sens={_sens_label(self.sens)}"
            f"{_engine_label(self.engine)}"
        )

    @property
    def group(self) -> tuple:
        """The points one convergence row judges together: all but the rtol."""
        return (self.row, self.grid, self.sens, self.engine)


def point(
    rtol: float, grid: Grid, sens: tuple[str, ...] = (), engine: str = "ode", **row: float
) -> Point:
    """A point by hand, for a case's slice: ``point(1e-10, Uniform(31), ("kx",), b=-1.0)``."""
    return Point(_row(row), float(rtol), grid, tuple(sens), engine)


def _row(overrides: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(k), float(v)) for k, v in overrides.items()))


def _sens_label(sens: tuple[str, ...]) -> str:
    return "+".join(sens) or "none"


def _engine_label(engine: str) -> str:
    return "" if engine == "ode" else f",engine={engine}"


def converge_method(group: tuple) -> str:
    _, grid, sens, engine = group
    return f"converge,grid={grid.label},sens={_sens_label(sens)}{_engine_label(engine)}"


# ── The judge and known failures ────────────────────────────────────────────


@dataclass(frozen=True)
class Judge:
    """How far a run may stray from its reference, per column, in units of
    ``atol + rtol·peak`` (see ``_judge.py``). The default is ``bngsim._check``'s
    PATH_RTOL_FACTOR: what two correct paths may differ by. A case that needs a
    looser one says why in its ``note``. ``converge`` adds the row that asks
    the error to fall as rtol tightens.

    ``peak`` is what a column's tolerance is scaled by. ``"samples"`` (the
    default) is its largest magnitude over the run's samples. ``"window"`` also
    takes the reference's largest over the whole window: both limits at every
    crossing and a dense scan. A narrow pulse's column runs 1e4 to 1e5 times its
    end value inside the pulse, CVODES holds it to rtol of that running size
    there, and a grid with no sample inside the pulse would judge it against its
    end value (#1779's triangle and sawtooth; a corner-free pulse of the same
    width does no better, see #1777). A decaying column is right to be judged
    by its samples: its early errors decay with it."""

    k_state: float = 1e3
    k_sens: float = 1e3
    converge: bool = True
    peak: str = "samples"


@dataclass(frozen=True)
class Known:
    """A point that fails on main for a filed bug: a strict expected failure in
    the PR slice (``raises`` is the narrowest type it fails with), and a
    non-passing row in the committed nightly baseline. Fixing the bug turns
    both red until the mark is removed.

    ``raises`` is a type, or the name of one of bngsim's exception classes
    (``"SimulationError"``, ``"SensitivityUnsupportedError"``,
    ``"SimulationTimeout"`` for a stall cut off by the case's ``timeout``):
    ``_cases`` imports nothing from bngsim, and the driver resolves the name."""

    issue: int
    raises: type[BaseException] | str = AssertionError
    reason: str = ""


# ── The case ────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class Case:
    """One model, its sweep, and its reference.

    * ``model``: the model text, in ``fmt`` (a ``.net``, SBML or Antimony).
      Every value the sweep moves is a model parameter, set with ``set_param``
      on a reset model, so one compile serves the whole case.
    * ``params``: the base value of every parameter the run sets, derived ones
      included; ``derive`` computes those from the rest (a threshold placed so
      the crossing sits at a chosen time).
    * ``t_span``: the window, or a function of a row's values giving it (#1836's
      late switch needs a window that ends just past it, at every span).
    * ``observe``: the species judged, by their names in the model.
    * ``reference``: a ``ClosedForm``, ``Limit``, ``Branchwise``, ``Frozen`` or
      ``NoReference`` (``_reference.py``).
    * ``spec``: the model written out by hand with explicit branch variables
      (``_branchwise.Spec``). The slice checks it against the text, and a
      branch-by-branch reference integrates it. A model with events has none:
      the spec holds no resets, and its controls (points that pass) stand in.
    * ``slice``: the points the PR slice runs, at most ``SLICE_CAP``.
    * ``steady_mask``: for ``SteadyState`` points, the species the convergence
      test reads (``steady_state(mask=...)``); empty for all of them.
    * ``spec_rows``: the parameter rows the spec check reads, every row when
      None. The spec is one model whatever the row, and the check moves every
      parameter it reads anyway, so a case whose rows stall a plain run (#1779,
      #1802) names a row that runs.
    """

    name: str
    family: str
    model: str
    fmt: str
    switches: tuple[Switch, ...]
    params: Mapping[str, float]
    t_span: tuple[float, float] | Callable[[dict[str, float]], tuple[float, float]]
    observe: tuple[str, ...]
    reference: object
    axes: Axes
    slice: tuple[Point, ...]
    spec: object | None = None
    derive: Callable[[dict[str, float]], Mapping[str, float]] | None = None
    atol_ratio: float = 1e-2
    judge: Judge = field(default_factory=Judge)
    known: Mapping[str, Known] = field(default_factory=dict)
    timeout: float = 60.0
    steady_mask: tuple[str, ...] = ()
    spec_rows: tuple[Mapping[str, float], ...] | None = None
    note: str = ""

    # Values and keys

    def values(self, row: tuple[tuple[str, float], ...] = ()) -> dict[str, float]:
        """Every parameter the run sets, at this row."""
        out = {k: float(v) for k, v in self.params.items()}
        out.update(dict(row))
        if self.derive is not None:
            out.update({k: float(v) for k, v in self.derive(dict(out)).items()})
        return out

    def window(self, values: Mapping[str, float] | None = None) -> tuple[float, float]:
        """The run's window at these values (the base row's when None)."""
        span = self.t_span(dict(values or self.values())) if callable(self.t_span) else self.t_span
        t0, t1 = span
        return float(t0), float(t1)

    @property
    def steady(self) -> bool:
        """Whether the case's points are steady states, not time courses."""
        return any(isinstance(g, SteadyState) for g in self.axes.grids)

    def row_id(self, row: tuple[tuple[str, float], ...]) -> str:
        if not row:
            return self.name
        return f"{self.name}[{','.join(f'{k}={num(v)}' for k, v in row)}]"

    def key(self, p: Point) -> str:
        return f"{self.row_id(p.row)}|{p.method}"

    def converge_key(self, group: tuple) -> str:
        return f"{self.row_id(group[0])}|{converge_method(group)}"

    def rows(self) -> list[tuple[tuple[str, float], ...]]:
        return [_row(r) for r in self.axes.params]

    def spec_check_rows(self) -> list[tuple[tuple[str, float], ...]]:
        """The rows the spec check reads (``spec_rows``)."""
        return self.rows() if self.spec_rows is None else [_row(r) for r in self.spec_rows]

    def points(self, row: tuple[tuple[str, float], ...] | None = None) -> list[Point]:
        """The sweep, in a fixed order: rows, then sens, engines, grids, rtol
        (loosest first)."""
        rows = self.rows() if row is None else [row]
        return [
            Point(r, float(rtol), grid, tuple(sens), engine)
            for r, sens, engine, grid, rtol in itertools.product(
                rows,
                self.axes.sens,
                self.axes.engines,
                self.axes.grids,
                sorted(self.axes.rtol, reverse=True),
            )
        ]

    def groups(self, points: list[Point]) -> dict[tuple, list[Point]]:
        """``points`` by convergence group, each loosest rtol first; only the
        groups with two tolerances or more can be judged for convergence."""
        out: dict[tuple, list[Point]] = {}
        for p in points:
            out.setdefault(p.group, []).append(p)
        return {
            g: sorted(ps, key=lambda p: -p.rtol)
            for g, ps in out.items()
            if len({p.rtol for p in ps}) >= 2
        }

    def all_keys(self) -> set[str]:
        """Every key the full sweep reports."""
        pts = self.points()
        out = {self.key(p) for p in pts}
        if self.judge.converge:
            out |= {self.converge_key(g) for g in self.groups(pts)}
        return out

    # Validation

    def validate(self) -> None:
        """Raise ``CaseError`` for a case written wrongly."""
        where = f"case {self.name!r}"
        if not self.name or any(c in self.name for c in "|[],= "):
            raise CaseError(f"{where}: a name may not hold '|', '[', ']', ',', '=' or a space")
        if self.family not in FAMILIES:
            raise CaseError(f"{where}: family {self.family!r} is not one of {FAMILIES}")
        if self.fmt not in FORMATS:
            raise CaseError(f"{where}: fmt {self.fmt!r} is not one of {FORMATS}")
        # A smooth pulse (#1795's Gaussian) has no switch for #1922 to find, and
        # no crossing; such a case says in its note why it is here.
        if not self.switches and (
            self.reference.crossings(self.values(), self.window()) or not self.note
        ):
            raise CaseError(f"{where}: names no switch")
        if len({s.name for s in self.switches}) != len(self.switches):
            raise CaseError(f"{where}: two switches share a name")
        if not self.observe:
            raise CaseError(f"{where}: observes no species")
        for row in self.axes.params:
            unknown = set(row) - set(self.params)
            if unknown:
                raise CaseError(f"{where}: row {dict(row)} sets {sorted(unknown)}, not in params")
        for row in self.rows():
            t0, t1 = self.window(self.values(row))
            if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
                raise CaseError(
                    f"{where}: the window ({t0}, {t1}) at {self.row_id(row)} is not a forward window"
                )
        self._validate_axes(where)
        if self.derive is not None:
            unknown = set(self.derive(dict(self.params))) - set(self.params)
            if unknown:
                raise CaseError(f"{where}: derive sets {sorted(unknown)}, not in params")
        for sens in self.axes.sens:
            unknown = set(sens) - set(self.params)
            if unknown:
                raise CaseError(f"{where}: columns {sorted(unknown)} are not parameters")
            if len(set(sens)) != len(sens):
                raise CaseError(f"{where}: sens {sens} names a column twice")
        if not self.axes.rtol or any(not (0 < r < 1) for r in self.axes.rtol):
            raise CaseError(f"{where}: rtol {self.axes.rtol} must lie in (0, 1)")
        labels = [g.label for g in self.axes.grids]
        if not labels or len(set(labels)) != len(labels):
            raise CaseError(f"{where}: grids {labels} must be given, once each")
        if not (self.atol_ratio > 0):
            raise CaseError(f"{where}: atol_ratio must be positive")
        sweep = set(self.points())
        if not self.slice:
            raise CaseError(f"{where}: puts no point in the PR slice")
        if len(self.slice) > SLICE_CAP:
            raise CaseError(
                f"{where}: {len(self.slice)} slice points, past the cap of {SLICE_CAP}"
            )
        outside = [self.key(p) for p in self.slice if p not in sweep]
        if outside:
            raise CaseError(f"{where}: slice points outside its sweep: {outside}")
        if len(set(self.slice)) != len(self.slice):
            raise CaseError(f"{where}: a slice point is listed twice")
        # The slice touches every tolerance, grid and column set the case
        # sweeps, so a PR that breaks one of them sees it. Parameter rows are
        # exempt: a case may sweep more of them than the cap allows.
        for axis, have, want in (
            ("rtol", {p.rtol for p in self.slice}, set(map(float, self.axes.rtol))),
            ("grid", {p.grid.label for p in self.slice}, set(labels)),
            ("sens", {p.sens for p in self.slice}, set(map(tuple, self.axes.sens))),
            ("engine", {p.engine for p in self.slice}, set(self.axes.engines)),
        ):
            if want - have:
                raise CaseError(
                    f"{where}: its slice never takes {axis} {sorted(want - have, key=str)}"
                )
        stray = set(self.known) - self.all_keys()
        if stray:
            raise CaseError(f"{where}: known failures at keys the sweep has not: {sorted(stray)}")
        for k, v in self.known.items():
            if not isinstance(v, Known) or v.issue <= 0:
                raise CaseError(f"{where}: known[{k!r}] must be a Known naming its issue")
            if isinstance(v.raises, str):
                if not v.raises.isidentifier():
                    raise CaseError(f"{where}: known[{k!r}] raises {v.raises!r}, not a class name")
            elif not (isinstance(v.raises, type) and issubclass(v.raises, BaseException)):
                raise CaseError(
                    f"{where}: known[{k!r}] raises {v.raises!r}, not an exception type"
                )
        check = getattr(self.reference, "check_case", None)
        if check is not None:
            check(self)
        if self.spec is not None:
            self.spec.check_case(self)

    def _validate_axes(self, where: str) -> None:
        """The engines, steady states and judge a case asks for fit together."""
        engines = self.axes.engines
        if not engines or len(set(engines)) != len(engines):
            raise CaseError(f"{where}: engines {engines} must be given, once each")
        bad = [e for e in engines if e not in ENGINES]
        if bad:
            raise CaseError(f"{where}: engines {bad} are not among {ENGINES}")
        if self.judge.peak not in PEAKS:
            raise CaseError(f"{where}: judge peak {self.judge.peak!r} is not one of {PEAKS}")
        grids = self.axes.grids
        plain_uniform = not any(self.axes.sens) and all(isinstance(g, Uniform) for g in grids)
        if "diffrax" in engines and not plain_uniform:
            raise CaseError(
                f"{where}: run_diffrax takes n_points and integrates no columns, so a "
                "case with the diffrax engine sweeps Uniform grids and plain runs only"
            )
        steady = [g for g in grids if isinstance(g, SteadyState)]
        if steady:
            if len(steady) != len(grids):
                raise CaseError(f"{where}: mixes steady states and time courses in its grids")
            if any(g.method not in STEADY_METHODS for g in steady):
                raise CaseError(f"{where}: a steady-state method is not one of {STEADY_METHODS}")
            if any(self.axes.sens) or any(
                e not in ("ode", "ode-compiled", "ode-interpreted") for e in engines
            ):
                raise CaseError(f"{where}: steady states are plain runs of an ODE engine here")
            if getattr(self.reference, "kind", "") != "limit":
                raise CaseError(f"{where}: a steady state is judged against a Limit reference")
            if self.judge.converge:
                raise CaseError(
                    f"{where}: a steady state's error is set by steady_state's tol, not "
                    "its rtol, so its case turns the convergence rows off"
                )
        elif getattr(self.reference, "kind", "") == "limit":
            raise CaseError(f"{where}: a Limit reference judges SteadyState grids only")
        elif self.steady_mask:
            raise CaseError(f"{where}: steady_mask applies to SteadyState grids only")
        if self.steady_mask and not set(self.observe) <= set(self.steady_mask):
            raise CaseError(f"{where}: every observed species must be in steady_mask")
        if self.spec_rows is not None:
            stray = [dict(r) for r in self.spec_check_rows() if r not in self.rows()]
            if not self.spec_rows or stray:
                raise CaseError(f"{where}: spec_rows {stray or '()'} are not rows it sweeps")
