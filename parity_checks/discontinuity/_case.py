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
#: the suite needs BNG2.pl (most CI legs have none).
FORMATS = ("net", "sbml")
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
    """``n_points=n``: the run's own grid, which knows nothing of the crossings."""

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


Grid = Uniform | Aligned | Offset


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


@dataclass(frozen=True)
class Point:
    """One run: a parameter row, a tolerance, a grid and a set of columns."""

    row: tuple[tuple[str, float], ...]
    rtol: float
    grid: Grid
    sens: tuple[str, ...]

    @property
    def method(self) -> str:
        return f"rtol={self.rtol:g},grid={self.grid.label},sens={_sens_label(self.sens)}"

    @property
    def group(self) -> tuple:
        """The points one convergence row judges together: all but the rtol."""
        return (self.row, self.grid, self.sens)


def point(rtol: float, grid: Grid, sens: tuple[str, ...] = (), **row: float) -> Point:
    """A point by hand, for a case's slice: ``point(1e-10, Uniform(31), ("kx",), b=-1.0)``."""
    return Point(_row(row), float(rtol), grid, tuple(sens))


def _row(overrides: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(k), float(v)) for k, v in overrides.items()))


def _sens_label(sens: tuple[str, ...]) -> str:
    return "+".join(sens) or "none"


def converge_method(group: tuple) -> str:
    _, grid, sens = group
    return f"converge,grid={grid.label},sens={_sens_label(sens)}"


# ── The judge and known failures ────────────────────────────────────────────


@dataclass(frozen=True)
class Judge:
    """How far a run may stray from its reference, per column, in units of
    ``atol + rtol·peak`` (see ``_judge.py``). The default is ``bngsim._check``'s
    PATH_RTOL_FACTOR: what two correct paths may differ by. A case that needs a
    looser one says why in its ``note``. ``converge`` adds the row that asks
    the error to fall as rtol tightens."""

    k_state: float = 1e3
    k_sens: float = 1e3
    converge: bool = True


@dataclass(frozen=True)
class Known:
    """A point that fails on main for a filed bug: a strict expected failure in
    the PR slice (``raises`` is the narrowest type it fails with), and a
    non-passing row in the committed nightly baseline. Fixing the bug turns
    both red until the mark is removed."""

    issue: int
    raises: type[BaseException] = AssertionError
    reason: str = ""


# ── The case ────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class Case:
    """One model, its sweep, and its reference.

    * ``model``: the model text, in ``fmt`` (a ``.net`` or SBML). Every value the sweep moves is a
      model parameter, set with ``set_param`` on a reset model, so one compile
      serves the whole case.
    * ``params``: the base value of every parameter the run sets, derived ones
      included; ``derive`` computes those from the rest (a threshold placed so
      the crossing sits at a chosen time).
    * ``observe``: the species judged, by their names in the model.
    * ``reference``: a ``ClosedForm``, ``Branchwise``, ``Frozen`` or
      ``NoReference`` (``_reference.py``).
    * ``spec``: the model written out by hand with explicit branch variables
      (``_branchwise.Spec``). The slice checks it against the text, and a
      branch-by-branch reference integrates it.
    * ``slice``: the points the PR slice runs, at most ``SLICE_CAP``.
    """

    name: str
    family: str
    model: str
    fmt: str
    switches: tuple[Switch, ...]
    params: Mapping[str, float]
    t_span: tuple[float, float]
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
    note: str = ""

    # Values and keys

    def values(self, row: tuple[tuple[str, float], ...] = ()) -> dict[str, float]:
        """Every parameter the run sets, at this row."""
        out = {k: float(v) for k, v in self.params.items()}
        out.update(dict(row))
        if self.derive is not None:
            out.update({k: float(v) for k, v in self.derive(dict(out)).items()})
        return out

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

    def points(self, row: tuple[tuple[str, float], ...] | None = None) -> list[Point]:
        """The sweep, in a fixed order: rows, then sens, grids, rtol (loosest first)."""
        rows = self.rows() if row is None else [row]
        return [
            Point(r, float(rtol), grid, tuple(sens))
            for r, sens, grid, rtol in itertools.product(
                rows, self.axes.sens, self.axes.grids, sorted(self.axes.rtol, reverse=True)
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
        if not self.switches:
            raise CaseError(f"{where}: names no switch")
        if len({s.name for s in self.switches}) != len(self.switches):
            raise CaseError(f"{where}: two switches share a name")
        t0, t1 = map(float, self.t_span)
        if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
            raise CaseError(f"{where}: t_span {self.t_span} is not a forward window")
        if not self.observe:
            raise CaseError(f"{where}: observes no species")
        for row in self.axes.params:
            unknown = set(row) - set(self.params)
            if unknown:
                raise CaseError(f"{where}: row {dict(row)} sets {sorted(unknown)}, not in params")
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
        check = getattr(self.reference, "check_case", None)
        if check is not None:
            check(self)
        if self.spec is not None:
            self.spec.check_case(self)
