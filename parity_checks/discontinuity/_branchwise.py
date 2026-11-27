"""The branch-by-branch reference: a case's spec integrated one smooth branch
at a time, at 30 digits (issue #1912).

A ``Spec`` is the model written out by hand in sympy with an explicit branch
variable for each switch: ``rhs(x, s, t, p)`` gives the flows with ``s[k]``
standing for "surface k's ``g`` is positive", and ``surfaces`` gives each
``g(x, t, p)``. That is #1781's target design, written once inside the test,
so the reference shares no code with bngsim's switch handling.

``Branchwise`` integrates it:

* the state and its sensitivity columns together (the forward sensitivity
  equations, from sympy's derivatives of the flows), with mpmath's Taylor
  series method at ``dps`` digits;
* holding every branch variable fixed between crossings, so each segment is
  smooth;
* locating each crossing as a root of ``g`` along the branch's own solution:
  a sign scan at fixed times across the window, then bisection to the working
  precision;
* restarting exactly at the crossing, with the state carried over and the
  columns jumped by the saltation formula (``saltation()``).

It refuses (``ReferenceUnavailable``), rather than guessing, where it cannot
follow the model: a start exactly on a surface, a crossing the flow only
grazes, a slide (the flows on both sides point into the surface), two
surfaces crossed at one instant, or a different number of crossings than the
case declares. The declared count is what catches a surface touched and left
between two scan times, which a sign scan cannot see.

Parameters and sample times enter as the exact values of their doubles, so
the reference integrates the model bngsim runs, not its decimal spelling.

mpmath's Taylor method is not validated. At 30 digits its error is many
orders below a double's, and a 40-digit solve agrees with it far below the
judge's floor (a frame test checks this). The validated tools
(``_frozen.py``) certify what this cannot: which crossings exist and in what
order.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import mpmath
import numpy as np
import sympy
from _reference import AT_CROSSING_ULPS, Band, ReferenceUnavailable, Solution, at_crossing

#: Output is rounded to doubles, so a 30-digit reference is good to a few of
#: their roundings.
BRANCHWISE_FLOOR = 4 * float(np.finfo(float).eps)
#: The most crossings one run may take before the reference calls it
#: chattering and refuses.
MAX_CROSSINGS = 64
#: Two roots within this many digits short of the working precision are one
#: instant: two surfaces crossed together (refused; a coincident case needs a
#: certified record, #1915), or a surface crossed again at once.
TIE_DIGITS = 5


@dataclass(frozen=True)
class Surface:
    """A switch's surface: its branch variable is 1 where ``g(x, t, p) > 0``."""

    switch: str
    g: Callable


@dataclass(eq=False)
class Spec:
    """A model written by hand, with explicit branch variables.

    ``states`` are the model's species names, every one. ``rhs(x, s, t, p)``
    and ``x0(p)`` return mappings from those names to sympy expressions in
    ``x[name]``, ``p[name]``, ``t`` and ``s[k]``; ``surfaces[k]`` defines
    ``s[k]``. ``params`` are the parameters they read."""

    states: tuple[str, ...]
    params: tuple[str, ...]
    rhs: Callable
    surfaces: tuple[Surface, ...]
    x0: Callable

    def check_case(self, case) -> None:
        from _case import CaseError

        where = f"case {case.name!r}"
        missing = [s for s in case.observe if s not in self.states]
        if missing:
            raise CaseError(f"{where}: observes {missing}, which its spec does not hold")
        unknown = set(self.params) - set(case.params)
        if unknown:
            raise CaseError(f"{where}: its spec reads {sorted(unknown)}, not in params")
        names = {s.name for s in case.switches}
        stray = {s.switch for s in self.surfaces} - names
        if stray:
            raise CaseError(f"{where}: spec surfaces name switches {sorted(stray)} it lacks")
        bare = names - {s.switch for s in self.surfaces}
        if bare:
            raise CaseError(f"{where}: switches {sorted(bare)} have no surface in its spec")
        sym = self.symbolic()
        known = set(sym.xs) | set(sym.ps) | set(sym.sb) | {sym.t}
        for label, expr in [
            *zip(self.states, sym.f, strict=False),
            *(("a surface", g) for g in sym.g),
        ]:
            free = expr.free_symbols - known
            if free:
                raise CaseError(f"{where}: its spec's {label} reads unknown symbols {free}")

    @functools.cache  # noqa: B019 - a Spec lives as long as the suite
    def symbolic(self) -> _Symbolic:
        n, m, k = len(self.states), len(self.params), len(self.surfaces)
        xs = tuple(sympy.symbols(f"x0:{n}")) if n else ()
        ps = tuple(sympy.symbols(f"p0:{m}")) if m else ()
        sb = tuple(sympy.symbols(f"s0:{k}")) if k else ()
        t = sympy.Symbol("t")
        X = dict(zip(self.states, xs, strict=True))
        P = dict(zip(self.params, ps, strict=True))
        flows = self.rhs(X, sb, t, P)
        start = self.x0(P)
        for what, got in (("rhs", flows), ("x0", start)):
            if set(got) != set(self.states):
                raise ValueError(f"spec {what} gives {sorted(got)}, not the states {self.states}")
        return _Symbolic(
            xs=xs,
            ps=ps,
            sb=sb,
            t=t,
            f=[sympy.sympify(flows[s]) for s in self.states],
            x0=[sympy.sympify(start[s]) for s in self.states],
            g=[sympy.sympify(surf.g(X, t, P)) for surf in self.surfaces],
        )

    # In doubles, for checking the spec against bngsim's own right-hand side.

    @functools.cache  # noqa: B019
    def _doubles(self):
        sym = self.symbolic()
        args = (sym.t, *sym.xs, *sym.ps, *sym.sb)
        return (
            sympy.lambdify(args, sym.f, modules="math"),
            sympy.lambdify(args, sym.g, modules="math"),
            sympy.lambdify(sym.ps, sym.x0, modules="math"),
        )

    def _pv(self, p: Mapping[str, float]) -> list[float]:
        return [float(p[q]) for q in self.params]

    def g_values(self, t: float, y: Sequence[float], p: Mapping[str, float]) -> list[float]:
        """Each surface's g at a state."""
        _, gfn, _ = self._doubles()
        zeros = [0] * len(self.surfaces)
        return [float(v) for v in gfn(float(t), *map(float, y), *self._pv(p), *zeros)]

    def flows(self, t: float, y: Sequence[float], p: Mapping[str, float], s) -> np.ndarray:
        """The flows at a state, on the branches ``s``."""
        ffn, _, _ = self._doubles()
        return np.asarray(ffn(float(t), *map(float, y), *self._pv(p), *s), dtype=float)

    def start(self, p: Mapping[str, float]) -> np.ndarray:
        _, _, x0fn = self._doubles()
        return np.asarray(x0fn(*self._pv(p)), dtype=float)


@dataclass(frozen=True)
class _Symbolic:
    xs: tuple
    ps: tuple
    sb: tuple
    t: sympy.Symbol
    f: list
    x0: list
    g: list


def saltation(s_minus, f_minus, f_plus, gx, gt, gp):
    """The columns just past a crossing, and dt*/dp.

    At a crossing of g(t, x, p) = 0 from the flow ``f_minus`` into
    ``f_plus``, with the columns ``s_minus`` (n x m) just before it:

        dt*/dp = -(gx . s_minus + gp) / (gt + gx . f_minus)
        s_plus = s_minus + (f_minus - f_plus) (dt*/dp)

    Plain arithmetic on nested lists, so it runs at whatever precision its
    numbers carry. Raises ZeroDivisionError for a crossing the flow grazes."""
    n, m = len(f_minus), len(gp)
    rate = gt + sum(gx[i] * f_minus[i] for i in range(n))
    if rate == 0:
        raise ZeroDivisionError("the flow grazes the surface: dg/dt is 0 at the crossing")
    tau = [-(sum(gx[i] * s_minus[i][k] for i in range(n)) + gp[k]) / rate for k in range(m)]
    s_plus = [
        [s_minus[i][k] + (f_minus[i] - f_plus[i]) * tau[k] for k in range(m)] for i in range(n)
    ]
    return s_plus, tau


@dataclass
class _Segment:
    t_lo: object  # mpf
    branches: tuple[int, ...]
    z_lo: list
    fn: Callable  # mpmath's odefun solution, for t >= t_lo


@dataclass
class Crossing:
    t: object  # mpf
    surface: int
    switch: str
    direction: int  # +1: g rose through 0
    tau: list  # dt*/dp over the run's columns


@dataclass
class _Trajectory:
    segments: list[_Segment]
    crossings: list[Crossing]
    sens: tuple[str, ...]
    pv: list
    fns: _Fns


@dataclass(eq=False)
class Branchwise:
    """The spec integrated branch by branch. ``expect`` is the number of
    crossings the window holds, or a function of the parameters giving it.
    ``certificate``, a ``Frozen`` record, certifies the crossing structure:
    every solve then checks that it found the certified crossings, in order,
    each inside its interval."""

    spec: Spec
    expect: int | Callable[[dict], int]
    dps: int = 30
    scan: int = 400
    floor: float = BRANCHWISE_FLOOR
    certificate: object | None = None
    kind: str = field(default="branchwise", init=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def check_case(self, case) -> None:
        from _case import CaseError

        if case.spec is not self.spec:
            raise CaseError(
                f"case {case.name!r}: its Branchwise reference integrates another spec"
            )
        if self.certificate is not None:
            self.certificate.check_case(case)

    def crossings(self, params: Mapping[str, float], t_span) -> tuple[float, ...]:
        return tuple(float(c.t) for c in self.trajectory(params, t_span).crossings)

    def solve(
        self, params, t_span, times, observe: Sequence[str], sens: Sequence[str]
    ) -> Solution:
        unknown = set(sens) - set(self.spec.params)
        if unknown:
            raise ReferenceUnavailable(f"the spec reads no parameter {sorted(unknown)}")
        traj = self.trajectory(params, t_span)
        t = np.asarray(times, dtype=float)
        idx = [self.spec.states.index(s) for s in observe]
        cols = [self.spec.params.index(q) for q in sens]
        n, m = len(self.spec.states), len(self.spec.params)
        cross = tuple(float(c.t) for c in traj.crossings)
        right = np.empty((len(t), n + n * m))
        left = np.empty_like(right)
        with mpmath.workdps(self.dps):
            for i, ti in enumerate(t):
                k_left, k_right = _sides(cross, ti)
                right[i] = [float(v) for v in self._at(traj, k_right, ti)]
                left[i] = (
                    right[i]
                    if k_left == k_right
                    else [float(v) for v in self._at(traj, k_left, ti)]
                )

        def split(z):
            sp = z[:, idx]
            se = z[:, n:].reshape(len(t), n, m)[:, idx, :][:, :, cols] if cols else None
            return sp, se

        spr, ser = split(right)
        spl, sel = split(left)
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
            note=f"{len(cross)} crossings",
        )

    def trajectory(self, params: Mapping[str, float], t_span) -> _Trajectory:
        """The run with the columns of every parameter the spec reads, so one
        integration serves every column set asked of this row."""
        key = (_pkey(params, self.spec.params), _tkey(t_span))
        if key not in self._cache:
            self._cache[key] = self._integrate(params, t_span, tuple(self.spec.params))
        return self._cache[key]

    # The integration.

    def _integrate(self, params, t_span, sens) -> _Trajectory:
        fns = _compiled(self.spec, sens)
        n, m = len(self.spec.states), len(sens)
        surfaces = self.spec.surfaces
        with mpmath.workdps(self.dps):
            mpf = mpmath.mpf
            tie = mpf(10) ** (TIE_DIGITS - self.dps)
            pv = [mpf(float(params[q])) for q in self.spec.params]
            t0, t1 = mpf(float(t_span[0])), mpf(float(t_span[1]))
            x0 = list(fns.x0(*pv))
            s0 = fns.x0_dp(*pv) if m else []
            z = x0 + [s0[i][k] for i in range(n) for k in range(m)]
            g0 = fns.g(t0, *x0, *pv)
            if any(v == 0 for v in g0):
                raise ReferenceUnavailable(
                    "the run starts exactly on a surface: which branch holds there is the "
                    "contract's to say (#1916), not this reference's"
                )
            branches = tuple(int(v > 0) for v in g0)
            scan = [t0 + (t1 - t0) * mpf(i) / self.scan for i in range(1, self.scan + 1)]
            segments: list[_Segment] = []
            crossings: list[Crossing] = []
            t_lo = t0
            while True:
                fn = mpmath.odefun(_rhs(fns, pv, branches), t_lo, list(z))
                segments.append(_Segment(t_lo, branches, list(z), fn))
                hit = self._first_crossing(fns, fn, t_lo, scan, pv, branches, n, tie)
                if hit is None:
                    break
                k, t_star = hit
                if crossings and t_star - t_lo <= tie * max(1, abs(t_star)):
                    raise ReferenceUnavailable(
                        f"{surfaces[k].switch!r} is crossed again at once after t = {float(t_lo)!r}: chattering"
                    )
                zs = fn(t_star)
                x_star = list(zs[:n])
                s_minus = [[zs[n + i * m + q] for q in range(m)] for i in range(n)]
                flipped = tuple(1 - b if j == k else b for j, b in enumerate(branches))
                f_minus = fns.f(t_star, *x_star, *pv, *branches)
                f_plus = fns.f(t_star, *x_star, *pv, *flipped)
                gx, gt = fns.gx[k](t_star, *x_star, *pv), fns.gt[k](t_star, *x_star, *pv)
                gp = fns.gp[k](t_star, *x_star, *pv) if m else []
                rate_minus = gt + sum(gx[i] * f_minus[i] for i in range(n))
                rate_plus = gt + sum(gx[i] * f_plus[i] for i in range(n))
                size = abs(gt) + sum(abs(gx[i] * f_minus[i]) for i in range(n))
                if size == 0 or abs(rate_minus) <= tie * size:
                    raise ReferenceUnavailable(
                        f"the flow grazes {surfaces[k].switch!r} at t = {float(t_star)!r}"
                    )
                if rate_plus == 0 or (rate_plus > 0) != (rate_minus > 0):
                    raise ReferenceUnavailable(
                        f"the flows on both sides of {surfaces[k].switch!r} point into it at "
                        f"t = {float(t_star)!r}: a slide, which needs Filippov's rule (#1915)"
                    )
                s_plus, tau = saltation(s_minus, f_minus, f_plus, gx, gt, gp) if m else ([], [])
                direction = 1 if rate_minus > 0 else -1
                crossings.append(Crossing(t_star, k, surfaces[k].switch, direction, tau))
                if len(crossings) > MAX_CROSSINGS:
                    raise ReferenceUnavailable(f"more than {MAX_CROSSINGS} crossings: chattering")
                z = x_star + [s_plus[i][q] for i in range(n) for q in range(m)]
                branches, t_lo = flipped, t_star
        expected = self.expect(dict(params)) if callable(self.expect) else self.expect
        if len(crossings) != expected:
            raise ReferenceUnavailable(
                f"found {len(crossings)} crossings where the case expects {expected}: a surface "
                "touched and left between two scan times would read this way"
            )
        if self.certificate is not None:
            self.certificate.certify(params, [(float(c.t), c.switch) for c in crossings])
        return _Trajectory(segments, crossings, sens, pv, fns)

    def _first_crossing(self, fns, fn, t_lo, scan, pv, branches, n, tie):
        """The earliest surface crossed after ``t_lo`` on this branch, as
        ``(surface, t*)``, or None. At ``t_lo`` each g has the sign its branch
        variable says, which a g evaluated on the surface itself (0 to the
        last digits) would not reliably give."""
        signs = [1 if b else -1 for b in branches]
        a = t_lo
        for b in scan:
            # A crossing that fell on a scan time restarted exactly there
            # (below), so that scan time is behind this segment.
            if b <= t_lo:
                continue
            gb = fns.g(b, *fn(b)[:n], *pv)
            changed = [k for k, v in enumerate(gb) if v == 0 or (v > 0) != (signs[k] > 0)]
            if changed:
                # A g exactly 0 at the scan time has its root there, exactly.
                roots = sorted(
                    (b if gb[k] == 0 else self._bisect(fns, fn, k, a, b, signs[k], pv, n), k)
                    for k in changed
                )
                (t_star, k), rest = roots[0], roots[1:]
                if rest and rest[0][0] - t_star <= tie * max(1, abs(t_star)):
                    raise ReferenceUnavailable(
                        f"two surfaces cross at one instant, t = {float(t_star)!r}: a coincident "
                        "case needs a certified record (#1915)"
                    )
                return k, t_star
            a = b
        return None

    def _bisect(self, fns, fn, k, a, b, sign_a, pv, n):
        """g_k's root in (a, b], where g_k has the sign ``sign_a`` at a and not at b."""
        tol = mpmath.mpf(10) ** (2 - self.dps) * max(1, abs(b))
        lo, hi = a, b
        while hi - lo > tol:
            mid = (lo + hi) / 2
            v = fns.g(mid, *fn(mid)[:n], *pv)[k]
            if v != 0 and (v > 0) == (sign_a > 0):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @staticmethod
    def _at(traj: _Trajectory, k: int, t: float) -> list:
        seg = traj.segments[k]
        tm = mpmath.mpf(float(t))
        if tm >= seg.t_lo:
            return seg.fn(tm)
        # A few ulps before the segment's start, which mpmath cannot integrate
        # back to: one Taylor term is exact far below a double there.
        dz = traj.fns.aug(seg.t_lo, *seg.z_lo, *traj.pv, *seg.branches)
        return [z + d * (tm - seg.t_lo) for z, d in zip(seg.z_lo, dz, strict=True)]


def _sides(cross: Sequence[float], t: float) -> tuple[int, int]:
    """The segments holding a sample's left and right limits: the same one
    away from a crossing, the two either side of it at one."""
    for j, c in enumerate(cross):
        if abs(t - c) <= AT_CROSSING_ULPS * np.spacing(abs(c)):
            return j, j + 1
    k = sum(1 for c in cross if t > c)
    return k, k


def _rhs(fns, pv, branches):
    def rhs(t, z):
        return fns.aug(t, *z, *pv, *branches)

    return rhs


def _pkey(params: Mapping[str, float], names: Sequence[str]) -> tuple:
    return tuple((q, float(params[q])) for q in names)


def _tkey(t_span) -> tuple[float, float]:
    return (float(t_span[0]), float(t_span[1]))


@dataclass(frozen=True)
class _Fns:
    f: Callable
    g: Callable
    gx: list
    gt: list
    gp: list
    aug: Callable
    x0: Callable
    x0_dp: Callable


@functools.lru_cache(maxsize=64)
def _compiled(spec: Spec, sens: tuple[str, ...]) -> _Fns:
    """The spec's functions lambdified to mpmath, for one set of columns."""
    sym = spec.symbolic()
    n, m = len(sym.xs), len(sens)
    q = [sym.ps[spec.params.index(name)] for name in sens]
    S = sympy.Matrix(n, m, lambda i, k: sympy.Symbol(f"S{i}_{k}"))
    F = sympy.Matrix(sym.f)
    dS = F.jacobian(sympy.Matrix(sym.xs)) * S + F.jacobian(sympy.Matrix(q)) if m else None
    zsyms = [*sym.xs, *[S[i, k] for i in range(n) for k in range(m)]]
    aug = [*sym.f, *([dS[i, k] for i in range(n) for k in range(m)] if m else [])]
    base = (sym.t, *sym.xs, *sym.ps)
    lam = functools.partial(sympy.lambdify, modules="mpmath", cse=True)
    x0_dp = sympy.Matrix(sym.x0).jacobian(sympy.Matrix(q)).tolist() if m else []
    return _Fns(
        f=lam((*base, *sym.sb), sym.f),
        g=lam(base, sym.g),
        gx=[lam(base, [sympy.diff(g, x) for x in sym.xs]) for g in sym.g],
        gt=[lam(base, sympy.diff(g, sym.t)) for g in sym.g],
        gp=[lam(base, [sympy.diff(g, qq) for qq in q]) for g in sym.g],
        aug=lam((sym.t, *zsyms, *sym.ps, *sym.sb), aug),
        x0=lam(sym.ps, sym.x0),
        x0_dp=lam(sym.ps, x0_dp),
    )
