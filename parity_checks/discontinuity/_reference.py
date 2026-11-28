"""What a reference gives, and the two simplest kinds (issue #1912).

Every reference answers ``solve(params, t_span, times, observe, sens)`` with
a ``Solution``: the observed species and their sensitivity columns at those
times, each as a band ``[lo, hi]`` (a point reference has ``lo == hi``; a
validated enclosure does not), the crossing times, and the reference's own
accuracy floor.

**Both sides of a crossing.** A sample taken exactly at a crossing time has
two right answers: the columns jump there (and a species does at an event),
so the value depends on which side is reported. Which side bngsim owes is
for the contract to fix (#1916). Until then each solution gives both limits
at such a time, ``*_left`` and the plain fields (the right-hand limit), and
the judge accepts either and records which one bngsim gave. Away from a
crossing the two are the same.

**Independence.** No reference imports bngsim (a frame test checks this in
a clean interpreter), and bngsim must never import one: the phase-3 engine's
saltation code in particular must not reuse ``_branchwise.saltation``, or the
reference would stop being a second opinion.

The kinds:

* ``ClosedForm`` (here): formulas written by hand, evaluated live. Events
  are written into them too: a species an event resets jumps at the event's
  time, read through ``past()`` like any other indicator, so both of its limits
  are there.
* ``Limit`` (here): a steady state written by hand, for ``SteadyState`` grids.
* ``Branchwise`` (``_branchwise.py``): the case's spec integrated one smooth
  branch at a time at 30 digits, with crossings located on each branch.
* ``Frozen`` (``_frozen.py``): committed enclosures from validated
  integrators (CAPD, Ariadne), recorded offline.
* ``NoReference`` (here): no reference exists yet; only invariants and an
  expected refusal are checked.

**A refusal owed.** Where a column has no derivative (the run's answer jumps
as the parameter crosses a value, or has a kink there), no number is right,
and the run owes a typed refusal. A ``ClosedForm`` says where with ``owes``,
an ``Owed`` naming the refusal by its class name, since no reference imports
bngsim; the driver resolves the name and holds the run to it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

#: A sample within this many ulps of a crossing time is taken to be at it.
AT_CROSSING_ULPS = 4
#: The relative accuracy, against a column's peak, of a formula evaluated in
#: doubles: a few roundings of an exponential of an exposure. The seed cases'
#: closed forms agree with the 30-digit branchwise reference to about 2e-16.
CLOSED_FORM_FLOOR = 1e-15
#: The uniform times a window peak is scanned at, beside both limits at every
#: crossing (``window_peak``).
PEAK_SCAN = 2001
#: The refusal a column owes where it has no derivative, unless said otherwise:
#: bngsim's base class of every sensitivity refusal.
SENSITIVITY_REFUSAL = "SensitivityUnsupportedError"


class ReferenceUnavailable(RuntimeError):
    """The reference cannot answer at this point (a sliding surface, a stale
    record, an unrecorded time). The row is REFERENCE_FAILED, not a finding."""


@dataclass
class Band:
    """``lo <= true value <= hi``, elementwise."""

    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def point(cls, a) -> Band:
        a = np.asarray(a, dtype=float)
        return cls(a, a)

    @property
    def shape(self):
        return self.lo.shape

    def distance(self, x) -> np.ndarray:
        """How far ``x`` lies outside the band (0 inside); inf where either is
        not finite."""
        x = np.asarray(x, dtype=float)
        d = np.maximum(np.maximum(self.lo - x, x - self.hi), 0.0)
        ok = np.isfinite(x) & np.isfinite(self.lo) & np.isfinite(self.hi)
        return np.where(ok, d, np.inf)

    def magnitude(self) -> np.ndarray:
        return np.maximum(np.abs(self.lo), np.abs(self.hi))


@dataclass
class Solution:
    """A reference at the run's sample times.

    ``species``: (n_t, n_obs); ``sens``: (n_t, n_obs, n_cols) or None. The
    plain fields are right-hand limits, ``*_left`` the left-hand ones; they
    differ only on the rows ``at_crossing`` marks."""

    times: np.ndarray
    species: Band
    species_left: Band
    sens: Band | None
    sens_left: Band | None
    crossings: tuple[float, ...]
    at_crossing: np.ndarray
    floor: float
    kind: str
    note: str = ""


def at_crossing(times, crossings) -> np.ndarray:
    """Which sample times lie within ``AT_CROSSING_ULPS`` of a crossing."""
    t = np.asarray(times, dtype=float)
    out = np.zeros(t.shape, dtype=bool)
    for c in crossings:
        out |= np.abs(t - float(c)) <= AT_CROSSING_ULPS * np.spacing(abs(float(c)))
    return out


def past(t, t_star: float, side: str, ulps: int = AT_CROSSING_ULPS) -> np.ndarray:
    """``t > t*``, with a time at ``t*`` itself on the side asked for:
    ``"right"`` counts it as past, ``"left"`` does not. A closed form reads
    every indicator through this, so its two limits at a crossing are the two
    sides of that indicator. A time within ``ulps`` of ``t*`` is at it. A model
    whose instants lie a double apart (#1627: 0.3 and 0.1*3) passes 0, so that
    a sample on one of them is not taken to be on the other too."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    t = np.asarray(t, dtype=float)
    near = np.abs(t - t_star) <= ulps * np.spacing(abs(float(t_star)))
    return np.where(near, side == "right", t > t_star)


# ── Closed forms ────────────────────────────────────────────────────────────

#: ``fn(params, t, side) -> (species, columns)``: ``species`` maps a species
#: name to its values at ``t``; ``columns`` maps ``(species, parameter)`` to
#: d(species)/d(parameter). ``side`` is passed to every ``past()``.
ClosedFormFn = Callable[
    [dict, np.ndarray, str], tuple[dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]
]


@dataclass(frozen=True)
class Owed:
    """No column has a derivative here, so the run owes a refusal: one of
    bngsim's exception classes by name, or a class (it or a subclass of it)."""

    reason: str
    refusal: str | type = SENSITIVITY_REFUSAL


#: ``owes(params, sens) -> Owed | None``: whether a point with these values and
#: columns owes a refusal (``sens`` is ``()`` for a plain run).
OwesFn = Callable[[dict, tuple[str, ...]], "Owed | None"]


@dataclass
class ClosedForm:
    """Formulas written by hand. ``crossings(params)`` lists the crossing
    times, which the grids and the two-sided rows need. ``owes`` says where a
    point owes a refusal instead of a value (``Owed``). ``peaks(params)`` lists
    stretches of time where a column may peak away from every crossing (a
    smooth pulse's middle), which ``window_peak`` scans densely."""

    fn: ClosedFormFn
    crossings_fn: Callable[[dict], Sequence[float]]
    floor: float = CLOSED_FORM_FLOOR
    owes: OwesFn | None = None
    peaks_fn: Callable[[dict], Sequence[tuple[float, float]]] | None = None
    kind: str = field(default="closed", init=False)

    def crossings(self, params: dict, t_span) -> tuple[float, ...]:
        t0, t1 = map(float, t_span)
        return tuple(sorted(c for c in map(float, self.crossings_fn(params)) if t0 < c <= t1))

    def owed(self, params: dict, sens: Sequence[str]) -> Owed | None:
        return None if self.owes is None else self.owes(dict(params), tuple(sens))

    def peaks(self, params: dict, t_span) -> tuple[tuple[float, float], ...]:
        """The stretches ``peaks_fn`` names, clipped to the window."""
        if self.peaks_fn is None:
            return ()
        t0, t1 = map(float, t_span)
        out = []
        for lo, hi in self.peaks_fn(dict(params)):
            lo, hi = max(t0, float(lo)), min(t1, float(hi))
            if hi > lo:
                out.append((lo, hi))
        return tuple(out)

    def solve(
        self, params: dict, t_span, times, observe: Sequence[str], sens: Sequence[str]
    ) -> Solution:
        t = np.asarray(times, dtype=float)
        cross = self.crossings(params, t_span)
        sides = {}
        for side in ("left", "right"):
            x, cols = self.fn(dict(params), t, side)
            missing = [s for s in observe if s not in x]
            if missing:
                raise ReferenceUnavailable(f"the closed form gives no value for {missing}")
            sp = np.stack([np.broadcast_to(np.asarray(x[s], float), t.shape) for s in observe], -1)
            if sens:
                gone = [(s, q) for s in observe for q in sens if (s, q) not in cols]
                if gone:
                    raise ReferenceUnavailable(f"the closed form gives no column {gone[0]}")
                se = np.stack(
                    [
                        np.stack(
                            [
                                np.broadcast_to(np.asarray(cols[(s, q)], float), t.shape)
                                for q in sens
                            ],
                            -1,
                        )
                        for s in observe
                    ],
                    -2,
                )
            else:
                se = None
            sides[side] = (sp, se)
        (spl, sel), (spr, ser) = sides["left"], sides["right"]
        return Solution(
            times=t,
            species=Band.point(spr),
            species_left=Band.point(spl),
            sens=None if ser is None else Band.point(ser),
            sens_left=None if sel is None else Band.point(sel),
            crossings=cross,
            at_crossing=at_crossing(t, cross),
            floor=self.floor,
            kind=self.kind,
        )


# ── The peak over the window ────────────────────────────────────────────────


def window_peak(
    ref, params: dict, t_span, observe: Sequence[str], sens: Sequence[str], n: int = PEAK_SCAN
) -> tuple[np.ndarray, np.ndarray | None]:
    """Each column's largest magnitude over the whole window, from the
    reference alone: at ``n`` uniform times, at both limits of every
    crossing, and at ``n`` times across each stretch the reference names as a
    possible peak away from its crossings (``ClosedForm.peaks``). A column
    whose peak lies at a crossing (a triangle's apex, a sawtooth's jump) is
    caught there, however narrow its pulse; a smooth peak is caught to the
    resolution of the scan that covers it, so a smooth pulse narrower than
    the uniform scan (a Gaussian 1e-3 wide) needs its stretch named. Returns
    ``(species (n_obs,), columns (n_obs, n_cols) or None)``."""
    t0, t1 = map(float, t_span)
    cross = [c for c in ref.crossings(params, t_span) if t0 <= c <= t1]
    peaks = getattr(ref, "peaks", None)
    dense = [np.linspace(lo, hi, n) for lo, hi in (peaks(params, t_span) if peaks else ())]
    times = np.unique(np.concatenate([np.linspace(t0, t1, n), np.asarray(cross, float), *dense]))
    sol = ref.solve(params, t_span, times, observe, sens)
    sp = np.maximum(sol.species.magnitude(), sol.species_left.magnitude()).max(axis=0)
    if sol.sens is None:
        return sp, None
    se = np.maximum(sol.sens.magnitude(), sol.sens_left.magnitude()).max(axis=0)
    return sp, se


# ── Steady states ───────────────────────────────────────────────────────────


@dataclass
class Limit:
    """A steady state written by hand, for ``SteadyState`` grids:
    ``fn(params) -> {species: value}``, the state as t goes to infinity. It
    answers at the one "time" a steady-state run reports, infinity."""

    fn: Callable[[dict], dict[str, float]]
    floor: float = CLOSED_FORM_FLOOR
    kind: str = field(default="limit", init=False)

    def crossings(self, params: dict, t_span) -> tuple[float, ...]:
        return ()

    def owed(self, params: dict, sens: Sequence[str]) -> Owed | None:
        return None

    def solve(
        self, params: dict, t_span, times, observe: Sequence[str], sens: Sequence[str]
    ) -> Solution:
        if sens:
            raise ReferenceUnavailable("a Limit gives no sensitivity columns")
        x = self.fn(dict(params))
        missing = [s for s in observe if s not in x]
        if missing:
            raise ReferenceUnavailable(f"the limit gives no value for {missing}")
        sp = np.array([[float(x[s]) for s in observe]])
        if not all(math.isfinite(v) for v in sp.ravel()):
            raise ReferenceUnavailable("the limit is not finite")
        return Solution(
            times=np.array([np.inf]),
            species=Band.point(sp),
            species_left=Band.point(sp),
            sens=None,
            sens_left=None,
            crossings=(),
            at_crossing=np.zeros(1, dtype=bool),
            floor=self.floor,
            kind=self.kind,
        )


# ── No reference ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Invariant:
    """A quantity the run must keep whatever its trajectory: ``fn(times,
    species, params)`` returns the residual over time and the scale it is
    judged against (a conserved total, say). ``species`` maps every model
    species name to its column."""

    name: str
    fn: Callable[[np.ndarray, dict[str, np.ndarray], dict], tuple[np.ndarray, float]]


@dataclass
class NoReference:
    """No independent reference exists yet (some slides, #1915). The case
    checks only what can be checked, and says so: invariants, each judged to
    ``k_state·(atol + rtol·scale)``, and a refusal the run must raise, at
    every point (a class, or its name in bngsim, since ``_cases`` imports none
    of bngsim). Never invent a reference to fill the gap."""

    why: str
    invariants: tuple[Invariant, ...] = ()
    expect_refusal: type[BaseException] | str | None = None
    kind: str = field(default="none", init=False)

    def crossings(self, params: dict, t_span) -> tuple[float, ...]:
        return ()

    def owed(self, params: dict, sens: Sequence[str]) -> Owed | None:
        if self.expect_refusal is None:
            return None
        return Owed(self.why, self.expect_refusal)

    def check_case(self, case) -> None:
        from _case import CaseError

        if not self.why:
            raise CaseError(f"case {case.name!r}: NoReference must say why there is none")
        if not self.invariants and self.expect_refusal is None:
            raise CaseError(f"case {case.name!r}: NoReference checks nothing")
