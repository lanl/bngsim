"""Closed forms for the suite's references (issue #1912).

Each function here returns a ``ClosedForm``: species and sensitivity columns at
any times, written by hand, with every indicator read through ``past()`` so a
sample at a crossing time has both of its limits. No bngsim here.

* ``window_decay``: a species decaying at a rate switched on at ``t_on`` and
  off at ``t_off`` (two time switches; the columns of both times jump).
* ``clock_exposure``: #1776's pulses on ``u = a*time() - b*Tobs``, where
  ``Tobs`` counts a clock ``0 -> T`` at ``k_clock``, so u runs at
  ``r = a - b*k_clock`` and the exposure is ``kx·H(r·t)/r``. Lifted from
  ``python/tests/test_clock_carried_crossing.py``'s ``_exact``, which gave the
  end point only, and extended to every time and to the columns of a, b and c.
* ``decay_bend``: a source ``kp*max(0, th - X)`` on a decaying X: a bend on
  the state, whose crossing time ``ln(X0/th)/kd`` is known.

Below those, one closed form per #1913 case, each its case's model written
out (an event as a jump of the species it resets), and two (``cube_switch``,
``counter_onset``) taken at 30 digits by mpmath where a double would not do.

Last, #1914's families: ``pulse_exposure`` (the pulse shapes on a condition
that reads ``time()``, a counter clock, or #1776's residual), ``level_step``,
and three on a decaying state (``state_step``, ``min_bend``, ``abs_bend``).
"""

from __future__ import annotations

import math

import numpy as np
from _reference import ClosedForm, Owed, Permitted, past


def window_decay(
    src: str, dst: str, x0: float, k: str = "kx", on: str = "t_on", off: str = "t_off"
):
    """``src -> dst`` at ``k·(if(time()>on,1,0) - if(time()>off,1,0))``, from
    ``src = x0``, ``dst = 0``."""

    def fn(p, t, side):
        kv, t_on, t_off = p[k], p[on], p[off]
        a, b = past(t, t_on, side), past(t, t_off, side)
        span = np.where(a, np.minimum(t, t_off) - t_on, 0.0)
        X = x0 * np.exp(-kv * span)
        dE = {k: span, on: np.where(a, -kv, 0.0), off: np.where(b, kv, 0.0)}
        cols = {}
        for q, d in dE.items():
            cols[(src, q)] = -X * d
            cols[(dst, q)] = X * d
        return {src: X, dst: x0 - X}, cols

    return ClosedForm(fn, lambda p: (p[on], p[off]))


#: The pulse shapes on u - c, each as (density h, its integral H, the offsets
#: of u - c where a switch sits, the offset where h jumps or None).
def _h_step(v, on, w):
    return np.where(on, 1.0, 0.0)


def _H_step(v, on, w):
    return np.where(on, np.maximum(v, 0.0), 0.0)


def _h_triangle(v, on, w):
    return np.maximum(0.0, 1.0 - np.abs(v) / (w / 2)) / w


def _H_triangle(v, on, w):
    lo = ((v + w / 2) / w) ** 2
    hi = 0.25 + v / w - (v / w) ** 2
    return np.where(v <= -w / 2, 0.0, np.where(v <= 0, lo, np.where(v <= w / 2, hi, 0.5)))


def _h_sawtooth(v, on, w):
    return np.where(on & (v < w), (2.0 / w) * (1.0 - np.clip(v, 0.0, w) / w), 0.0)


def _H_sawtooth(v, on, w):
    vv = np.clip(v, 0.0, w)
    return np.where(on, 2 * vv / w - (vv / w) ** 2, 0.0)


SHAPES = {
    # name: (h, H, switch offsets in u - c, in units of w where given as a pair)
    "step": (_h_step, _H_step, lambda w: (0.0,)),
    "triangle": (_h_triangle, _H_triangle, lambda w: (-w / 2, 0.0, w / 2)),
    "sawtooth": (_h_sawtooth, _H_sawtooth, lambda w: (0.0, w)),
}


def clock_exposure(shape: str, x0: float = 10.0):
    """``X -> Y`` at ``kx·h(u)`` with ``u = a*time() - b*Tobs``, the clock
    ``0 -> T`` at ``k_clock`` from ``T = 0``, and ``X = x0``. The pulse must
    lie wholly ahead of the start (its first switch at ``u = c + offset > 0``;
    the integrals start from 0 there), the clock must carry u forward
    (``r = a - b*k_clock > 0``), and a triangle or sawtooth needs ``w > 0``."""
    h, H, offsets = SHAPES[shape]

    def fn(p, t, side):
        kx, kc, a, b, c = (p[q] for q in ("kx", "k_clock", "a", "b", "c"))
        w = p.get("w", 0.0)
        r = a - b * kc
        if shape != "step" and not w > 0:
            raise ValueError(f"a {shape} needs w > 0, not {w}")
        if r <= 0 or not c + min(offsets(w)) > 0:
            raise ValueError(
                f"clock_exposure needs r > 0 and the pulse ahead of t = 0, not r = {r}, "
                f"c = {c}, w = {w}"
            )
        on = past(t, c / r, side)
        v = r * t - c
        hv, Hv = h(v, on, w), H(v, on, w)
        E = kx * Hv / r
        dE_dr = -kx * Hv / r**2 + kx * hv * t / r
        dE = {
            "kx": Hv / r,
            "k_clock": dE_dr * -b,
            "a": dE_dr,
            "b": dE_dr * -kc,
            "c": -kx * hv / r,
        }
        X = x0 * np.exp(-E)
        cols = {}
        for q, d in dE.items():
            cols[("X()", q)] = -X * d
            cols[("Y()", q)] = X * d
            cols[("T()", q)] = t if q == "k_clock" else np.zeros_like(t)
        return {"T()": kc * t, "X()": X, "Y()": x0 - X}, cols

    def crossings(p):
        r = p["a"] - p["b"] * p["k_clock"]
        return tuple((p["c"] + o) / r for o in offsets(p.get("w", 0.0)))

    return ClosedForm(fn, crossings)


def decay_bend(x0: float = 10.0):
    """``X -> 0`` at ``kd``, and ``0 -> Y`` at ``kp*max(0, th - X)``, from
    ``X = x0 > th``: Y starts once X falls through ``th``, at
    ``t* = ln(x0/th)/kd``, with no jump (a bend)."""

    def fn(p, t, side):
        kd, kp, th = p["kd"], p["kp"], p["th"]
        ts = np.log(x0 / th) / kd
        on = past(t, ts, side)
        X = x0 * np.exp(-kd * t)
        Y = np.where(on, kp * (th * (t - ts) + (X - th) / kd), 0.0)
        cols = {
            ("X()", "kd"): -t * X,
            ("X()", "kp"): np.zeros_like(t),
            ("X()", "th"): np.zeros_like(t),
            ("Y()", "kp"): np.where(on, Y / kp, 0.0),
            ("Y()", "th"): np.where(on, kp * (t - ts), 0.0),
            ("Y()", "kd"): np.where(on, kp * (th * ts / kd - t * X / kd - (X - th) / kd**2), 0.0),
        }
        return {"X()": X, "Y()": Y}, cols

    return ClosedForm(fn, lambda p: (np.log(x0 / p["th"]) / p["kd"],))


# ── #1913's cases: one closed form per filed bug's model ────────────────────
#
# Each is the model of its case in ``_cases``, written out by hand. Events are
# written in as jumps of the species they reset, through ``past()``, so a
# sample at an event's time has both of its limits.


def _ind(t, t_star, side, ulps=None):
    """1 from ``t_star`` on (a trigger that holds at ``t_star`` itself), as a
    float array; the left limit at ``t_star`` is 0, the right 1."""
    kw = {} if ulps is None else {"ulps": ulps}
    return past(t, t_star, side, **kw).astype(float)


def event_touch_tie():
    """``X' = -k X`` from 1; E1 adds 1 at ``tau``, E2 doubles X at 2.5, where
    ``sin(2πt/10)`` touches 1. Off the tie, the closed form; at ``tau = 2.5`` the
    order decides X(6), which jumps by 0.70 as tau passes it, so dX/dtau has no
    value there and a refusal is owed (``owes``)."""
    T2 = 2.5

    def fn(p, t, side):
        k, tau = p["k"], p["tau"]
        e = np.exp(-k * t)
        a, b = _ind(t, tau, side), _ind(t, T2, side)
        boost = np.exp(-k * (t - tau))
        if tau < T2:  # E1 first: (e + boost) once past tau, doubled from 2.5
            X = (e + a * boost) * (1 + b)
            dtau = a * (1 + b) * k * boost
        else:  # E2 first: doubled from 2.5, then plus boost from tau
            X = e * (1 + b) + a * boost
            dtau = a * k * boost
        return {"X": X}, {("X", "tau"): dtau}

    def owes(p, sens):
        if sens and abs(p["tau"] - T2) <= 4 * np.spacing(T2):
            return Owed(
                "E1 at tau and E2 at 2.5 fire on one instant, and X(6) jumps by 0.70 as "
                "tau passes 2.5: dX/dtau has no value"
            )
        return None

    return ClosedForm(fn, lambda p: (p["tau"], T2), owes=owes)


def event_unlisted_value():
    """``Y' = 1`` while ``time - y*floor(time/y) < 0.05``, with ``y = 100`` until F
    sets it to ``x + 3 = 3`` at 1.5 (F's value is read at its trigger, t = 1,
    before F2 sets x at 1.2). So Y gains 0.05 in each window: [0, 0.05), then
    [3j, 3j + 0.05) from 3 on."""

    def windows(t1):
        out = [(0.0, 0.05)]
        j = 1
        while 3.0 * j < t1:
            out.append((3.0 * j, 3.0 * j + 0.05))
            j += 1
        return out

    def fn(p, t, side):
        Y = np.zeros_like(t, dtype=float)
        for lo, hi in windows(float(np.max(t)) if t.size else 0.0):
            Y += np.clip(t - lo, 0.0, hi - lo)
        return {"Y": Y}, {}

    def crossings(p):
        return (0.05, 1.0, 1.2, 1.5, *(x for w in windows(20.0)[1:] for x in w))

    return ClosedForm(fn, crossings)


def event_double_apart():
    """G adds 1 to g at ``time == 0.3``, H to h at ``time == 0.1*k``, E adds amt
    to D at ``time == 0.1*j``, and F sets amt from 1 to 2 at ``time >= 0.3``.
    Each trigger holds at its own double (0.1*3 is 0.30000000000000004, beside
    0.3), and they fire in time order, so F precedes E. The indicators are
    exact (``ulps=0``): a sample on 0.3 has G fired and H not yet."""

    def fn(p, t, side):
        th, te = 0.1 * p["k"], 0.1 * p["j"]
        amt = 2.0 if te >= 0.3 else 1.0
        return {
            "g": _ind(t, 0.3, side, ulps=0),
            "h": _ind(t, th, side, ulps=0),
            "D": amt * _ind(t, te, side, ulps=0),
        }, {}

    return ClosedForm(fn, lambda p: (0.3, 0.1 * p["k"], 0.1 * p["j"]))


def event_row_short(t_event: float = 2.95e-11):
    """#765's model: ``X' = -k X`` from 1, and an event sets Y = 1 at
    ``time >= t_event``."""

    def fn(p, t, side):
        k = p["k"]
        X = np.exp(-k * t)
        Y = _ind(t, t_event, side)
        return {"X": X, "Y": Y}, {("X", "k"): -t * X, ("Y", "k"): np.zeros_like(t)}

    return ClosedForm(fn, lambda p: (t_event,))


def event_trigger_step():
    """``B' = C' = 1`` from 0; the event resets B to 0 when
    ``C + piecewise(0, time < T0 + 1, 5) >= T0 + 1``, which first holds at
    ``t* = T0 + 1`` (C reaches it there, and the step lifts the left side by 5
    at the same instant). So B = t - t* past t*, and dB/dT0 = -1 there."""

    def fn(p, t, side):
        ts = p["T0"] + 1.0
        on = _ind(t, ts, side)
        return {"B": t - on * ts, "C": t}, {("B", "T0"): -on, ("C", "T0"): np.zeros_like(t)}

    return ClosedForm(fn, lambda p: (p["T0"] + 1.0,))


def table_pulse(height: float = 5.0, t_on: float = 10.0, t_off: float = 12.0):
    """``A' = F``, a step table that is ``height`` on [t_on, t_off) and 0
    elsewhere."""

    def fn(p, t, side):
        return {"A()": height * np.clip(t - t_on, 0.0, t_off - t_on)}, {}

    return ClosedForm(fn, lambda p: (t_on, t_off))


def periodic_window(thr: float = 0.99, omega: float = 10.0):
    """``X' = c·[sin(ω t) > thr] - kd X`` from 0. The window's edges are fixed
    numbers, (asin(thr) + 2πm)/ω and (π - asin(thr) + 2πm)/ω, so every column
    exists: no parameter moves a crossing."""
    a0 = math.asin(thr)

    def edges(t1):
        out, m = [], 0
        while True:
            lo = (a0 + 2 * math.pi * m) / omega
            if lo >= t1:
                return out
            out.append((lo, (math.pi - a0 + 2 * math.pi * m) / omega))
            m += 1

    def fn(p, t, side):
        c, kd = p["c"], p["kd"]
        X = np.zeros_like(t, dtype=float)
        dkd = np.zeros_like(t, dtype=float)

        def G(u):
            return u * np.exp(-kd * u) / kd + np.exp(-kd * u) / kd**2

        for lo, hi in edges(float(np.max(t)) if t.size else 0.0):
            inside = t > lo
            b = np.minimum(hi, t)
            X += np.where(inside, c * (np.exp(-kd * (t - b)) - np.exp(-kd * (t - lo))) / kd, 0.0)
            dkd += np.where(inside, c * (G(t - lo) - G(t - b)), 0.0)
        return {"X()": X}, {("X()", "c"): X / c, ("X()", "kd"): dkd}

    return ClosedForm(fn, lambda p: tuple(x for e in edges(10.0) for x in e))


def gaussian_pulse():
    """``X' = kx/(s√π)·exp(-((t - c)/s)²)`` from 0: a pulse of area kx, so
    ``X = kx/2·(erf((t - c)/s) + erf(c/s))``. It has no switch, and no crossing."""
    erf = np.vectorize(math.erf)

    def fn(p, t, side):
        kx, c, s = p["kx"], p["c"], p["s"]
        return {"X()": kx / 2 * (erf((t - c) / s) + math.erf(c / s))}, {}

    return ClosedForm(fn, lambda p: ())


def table_step_switch_time():
    """``X' = a``; Y is made at ``k X`` from ``tau`` on, and W at ``q X tf``, where
    the step table tf is 0 before 3 and 1 from 3. W does not read tau, so its
    column is 0 everywhere, also with tau on the table's breakpoint."""

    def fn(p, t, side):
        a, k, q, tau = p["a"], p["k"], p["q"], p["tau"]
        y_on, w_on = _ind(t, tau, side), _ind(t, 3.0, side)
        X = a * t
        Y = y_on * k * a * (t**2 - tau**2) / 2
        W = w_on * q * a * (t**2 - 9.0) / 2
        zero = np.zeros_like(t)
        return {"X()": X, "Y()": Y, "W()": W}, {
            ("X()", "tau"): zero,
            ("Y()", "tau"): -y_on * k * a * tau,
            ("W()", "tau"): zero,
        }

    return ClosedForm(fn, lambda p: (p["tau"], 3.0))


def table_kink_switch_time():
    """``Y' = r·gate·g``, with gate 0.25 before ``tau`` and 1 from it, and the step
    table g 0.5 before 3 and 1 from 3. Past tau, dY/dtau = -0.75·r·g(tau): -0.375
    for tau < 3 and -0.75 for tau > 3. At tau = 3 the two sides differ, so the
    column has no value there and a refusal is owed."""

    def G(x):  # the integral of g from 0
        return 0.5 * np.minimum(x, 3.0) + np.maximum(x - 3.0, 0.0)

    def fn(p, t, side):
        r, tau = p["r"], p["tau"]
        on = _ind(t, tau, side)
        Y = r * (0.25 * G(np.minimum(t, tau)) + on * (G(t) - G(np.minimum(t, tau))))
        g_tau = 1.0 if tau >= 3.0 else 0.5
        return {"Y()": Y}, {("Y()", "tau"): -0.75 * r * g_tau * on}

    def owes(p, sens):
        if sens and abs(p["tau"] - 3.0) <= 4 * np.spacing(3.0):
            return Owed(
                "tau sits on the table's breakpoint, where dY/dtau is -0.375 from below "
                "and -0.75 from above"
            )
        return None

    return ClosedForm(fn, lambda p: (p["tau"], 3.0), owes=owes)


def parked_zero_window(a0: float = 10.0):
    """``A' = -a A`` from a0; B is made at ``kp·max(0, q - A)``, so it sits at
    exactly 0 until ``t* = ln(a0/q)/a``; Y is made at kb while B is 0. So
    ``Y = kb·min(t, t*)``, and the exit time's columns reach Y through t*."""

    def fn(p, t, side):
        a, q, kb, kp = p["a"], p["q"], p["kb"], p["kp"]
        ts = math.log(a0 / q) / a
        on = past(t, ts, side)
        A = a0 * np.exp(-a * t)
        B = np.where(on, kp * (q * (t - ts) + (A - q) / a), 0.0)
        Y = kb * np.minimum(t, ts)
        dts_da, dts_dq = -math.log(a0 / q) / a**2, -1.0 / (q * a)
        zero = np.zeros_like(t)
        return {"A()": A, "B()": B, "Y()": Y}, {
            ("A()", "a"): -t * A,
            ("A()", "q"): zero,
            ("A()", "kb"): zero,
            ("B()", "a"): np.where(on, kp * (q * ts / a - t * A / a - (A - q) / a**2), 0.0),
            ("B()", "q"): np.where(on, kp * (t - ts), 0.0),
            ("B()", "kb"): zero,
            ("Y()", "a"): np.where(on, kb * dts_da, 0.0),
            ("Y()", "q"): np.where(on, kb * dts_dq, 0.0),
            ("Y()", "kb"): np.minimum(t, ts),
        }

    return ClosedForm(fn, lambda p: (math.log(a0 / p["q"]) / p["a"],))


def cube_switch():
    """``A = 1 + eps·t``; Y is made at kb once ``A³ > c``, at
    ``t* = (c^(1/3) - 1)/eps``. The cube root is taken at 30 digits. In doubles
    its rounding would move t* by up to 2e-8 at eps = 1e-8: inside the case's
    k_state allowance, but the reference need not spend any of it."""
    import mpmath

    def t_star(p):
        with mpmath.workdps(30):
            return float((mpmath.cbrt(mpmath.mpf(p["c"])) - 1) / mpmath.mpf(p["eps"]))

    def fn(p, t, side):
        ts = t_star(p)
        return {
            "A()": 1.0 + p["eps"] * t,
            "Y()": np.where(past(t, ts, side), p["kb"] * (t - ts), 0.0),
        }, {}

    return ClosedForm(fn, lambda p: (t_star(p),))


def late_switch():
    """A counter ``C' = rc`` from c0, and ``X' = k(1 - X)`` once ``C >= sigma``, at
    ``t* = (sigma - c0)/rc``."""

    def fn(p, t, side):
        c0, sigma, k, rc = p["c0"], p["sigma"], p["k"], p["rc"]
        ts = (sigma - c0) / rc
        on = past(t, ts, side)
        e = np.exp(-k * np.where(on, t - ts, 0.0))
        return {"C()": c0 + rc * t, "X()": np.where(on, 1.0 - e, 0.0)}, {
            ("C()", "sigma"): np.zeros_like(t),
            ("X()", "sigma"): np.where(on, -k * e / rc, 0.0),
        }

    return ClosedForm(fn, lambda p: ((p["sigma"] - p["c0"]) / p["rc"],))


def counter_onset():
    """``Tc' = 1`` from 0 (so Tc = t); ``X' = k0 + P - kdeg X`` from 0, where the
    pulse ``P = k1 s^(a-1)(1 - s)``, ``s = (Tc - on)/D``, runs while
    ``on <= Tc <= on + D``. The pulse's integral is taken by mpmath's quadrature
    at 30 digits (s^(a-1) defeats a Taylor method at the onset). Moving ``on``
    moves the whole pulse, so ``dX/don = kdeg·X_pulse - P(t)``: the pulse's part
    of X grows with kdeg for each unit it starts later, less the rate it adds at
    the moving front."""
    import mpmath

    def pulse_part(p, t):
        a, D, k1, kdeg, on = (mpmath.mpf(p[q]) for q in ("a", "D", "k1", "kdeg", "on"))
        tt = mpmath.mpf(float(t))
        if tt <= on:
            return mpmath.mpf(0)
        sig = min((tt - on) / D, mpmath.mpf(1))
        f = lambda s: mpmath.exp(-kdeg * (tt - on - D * s)) * k1 * s ** (a - 1) * (1 - s) * D  # noqa: E731
        return mpmath.quad(f, [0, sig])

    def fn(p, t, side):
        k0, kdeg, a, D, k1, on = (p[q] for q in ("k0", "kdeg", "a", "D", "k1", "on"))
        with mpmath.workdps(30):
            xp = np.array([float(pulse_part(p, ti)) for ti in np.ravel(t)]).reshape(np.shape(t))
        s = (t - on) / D
        inside = (t > on) & (t < on + D)
        P = np.where(inside, k1 * np.clip(s, 0.0, 1.0) ** (a - 1) * (1 - s), 0.0)
        X = k0 * (1 - np.exp(-kdeg * t)) / kdeg + xp
        return {"Tc": t, "X": X}, {("Tc", "on"): np.zeros_like(t), ("X", "on"): kdeg * xp - P}

    return ClosedForm(fn, lambda p: (p["on"], p["on"] + p["D"]))


def slow_threshold():
    """A rises at r from a0 while A < 5, reaching it at ``t* = (5 - a0)/r``; past
    5, ``A' = kf - kf/6·A`` relaxes it to 6. Y sits at its rest value 0.01 while
    A < 5 and decays at ky past t*."""

    def fn(p, t, side):
        r, kf, ky, a0 = p["r"], p["kf"], p["ky"], p["a0"]
        ts = (5.0 - a0) / r
        on = past(t, ts, side)
        dt = np.where(on, t - ts, 0.0)
        A = np.where(on, 6.0 - np.exp(-kf / 6 * dt), a0 + r * t)
        Y = np.where(on, 0.01 * np.exp(-ky * dt), 0.01)
        return {"A()": A, "Y()": Y}, {}

    return ClosedForm(fn, lambda p: ((5.0 - p["a0"]) / p["r"],))


# ── #1914: the families, each swept ─────────────────────────────────────────
#
# Every pulse, step and sawtooth case reads one condition variable that is
# linear in t: u = a*t - b*T, with a clock T = T0 + k_clock*t. So u runs at
# r = a - b*k_clock from u(0) = -b*T0, and u - c reaches an offset o at
# t = (c + b*T0 + o)/r, a formula. What the condition reads fixes a and b. A
# case that reads time() keeps the clock as a witness: T's exact value says
# which time the state the run reports belongs to.
#
# The pulse a shape h makes on X -> Y is an exposure E = kx*(H(v) - H(v0))/r
# over v = u - c, from v0 = -c - b*T0 at t = 0, with H the integral of h. Every
# shape with a width w is a scale family in it, h(v) = phi(v/w)/w, so
# dH/dw = -(v/w)*h(v). Each column is the chain rule through v, v0 and r.

#: What a condition reads: (a, b), or None where the case's parameters a and b
#: give them (#1776's residual a*time() - b*Tobs).
LINES = {"time": (1.0, 0.0), "clock": (0.0, -1.0), "coupled": None}


class _Line:
    """u = a*t - b*(T0 + k_clock*t) at a row's values, and the derivatives of
    v = u - c, v0 = v(0) and r along it, by parameter."""

    def __init__(self, p: dict, reads: str):
        fixed = LINES[reads]
        self.reads = reads
        self.a, self.b = fixed if fixed is not None else (p["a"], p["b"])
        self.kc, self.T0, self.c = p.get("k_clock", 0.0), p.get("T0", 0.0), p["c"]
        self.r = self.a - self.b * self.kc
        # The parameters the line moves with: a and b only where they are
        # the case's; k_clock and T0 where the model has a clock.
        self.params = ["c"] + [q for q in ("k_clock", "T0") if q in p]
        if fixed is None:
            self.params += ["a", "b"]

    def at(self, o: float) -> float:
        """The time u - c reaches o."""
        return (self.c + self.b * self.T0 + o) / self.r

    def v(self, t):
        return self.r * t - (self.c + self.b * self.T0)

    @property
    def v0(self) -> float:
        return -(self.c + self.b * self.T0)

    def dv(self, q: str, t):
        zero = np.zeros_like(t)
        return {
            "c": zero - 1.0,
            "k_clock": -self.b * t,
            "T0": zero - self.b,
            "a": t,
            "b": -(self.T0 + self.kc * t),
        }[q]

    def dv0(self, q: str) -> float:
        return {"c": -1.0, "k_clock": 0.0, "T0": -self.b, "a": 0.0, "b": -self.T0}[q]

    def dr(self, q: str) -> float:
        return {"c": 0.0, "k_clock": -self.b, "T0": 0.0, "a": 1.0, "b": -self.kc}[q]


def _p_window(v, at, w):
    return np.where(at(0.0) & ~at(w), 1.0 / w, 0.0)


def _P_window(v, at, w):
    return np.clip(v, 0.0, w) / w


def _p_gauss(v, at, w):
    s = w / 4
    return np.exp(-((v / s) ** 2)) / (2 * s * math.sqrt(math.pi))


_erf = np.vectorize(math.erf, otypes=[float])


def _P_gauss(v, at, w):
    return (1.0 + _erf(np.asarray(v, dtype=float) / (w / 4))) / 4


#: Each shape over v = u - c: (h, H, the offsets of v where a switch sits, the
#: offset of its middle, whether it has a width). ``at(o)`` is the indicator
#: that the crossing at offset o is past, read through ``past()`` so a sample
#: on it has both limits. The step, triangle and sawtooth are #1776's
#: (``SHAPES``); the window is a rectangle of area 1 on [0, w), and the
#: Gaussian has the triangle's area, 1/2, and s = w/4: the corner-free control.
PULSES = {
    "step": (
        lambda v, at, w: np.where(at(0.0), 1.0, 0.0),
        lambda v, at, w: np.where(at(0.0), np.maximum(v, 0.0), 0.0),
        lambda w: (0.0,),
        None,
        False,
    ),
    "window": (_p_window, _P_window, lambda w: (0.0, w), lambda w: w / 2, True),
    "triangle": (
        lambda v, at, w: _h_triangle(v, None, w),
        lambda v, at, w: _H_triangle(v, None, w),
        lambda w: (-w / 2, 0.0, w / 2),
        lambda w: 0.0,
        True,
    ),
    "sawtooth": (
        lambda v, at, w: _h_sawtooth(v, at(0.0), w),
        lambda v, at, w: _H_sawtooth(v, at(0.0), w),
        lambda w: (0.0, w),
        lambda w: w / 2,
        True,
    ),
    "gauss": (_p_gauss, _P_gauss, lambda w: (), lambda w: 0.0, True),
}


def _unit_exposure(p: dict, t, side: str, shape: str, reads: str):
    """``U = (H(v) - H(v0))/r`` and its derivatives by every parameter it
    moves with (the line's, and w), so that a pulse of height kx is ``kx*U``."""
    h, H, offsets, _, has_w = PULSES[shape]
    L = _Line(p, reads)
    w = float(p.get("w", 0.0))
    if has_w and not w > 0:
        raise ValueError(f"a {shape} needs w > 0, not {w}")
    if not L.r > 0:
        raise ValueError(f"the condition must move forward: r = {L.r}")
    offs = offsets(w)
    if offs and not L.v0 < min(offs):
        raise ValueError(
            f"the {shape} must lie wholly ahead of t = 0: it starts at v = {min(offs)}, "
            f"and v(0) = {L.v0}"
        )

    def at(o):
        return past(t, L.at(o), side)

    def never(o):
        return np.zeros(np.shape(L.v0), dtype=bool)

    v, v0 = L.v(t), np.asarray(L.v0)
    hv, h0 = h(v, at, w), h(v0, never, w)
    U = (H(v, at, w) - H(v0, never, w)) / L.r
    dU = {q: (hv * L.dv(q, t) - h0 * L.dv0(q)) / L.r - U / L.r * L.dr(q) for q in L.params}
    if has_w:
        dU["w"] = (-(v / w) * hv + (v0 / w) * h0) / L.r
    return L, U, dU


def _xy(p, t, L, E, dE, x0):
    """X -> Y at exposure E, and the witness clock T where the model has one."""
    X = x0 * np.exp(-E)
    species = {"X()": X, "Y()": x0 - X}
    cols = {}
    for q, d in dE.items():
        cols[("X()", q)] = -X * d
        cols[("Y()", q)] = X * d
    if "k_clock" in p:
        species["T()"] = L.T0 + L.kc * t
        for q in dE:
            cols[("T()", q)] = t if q == "k_clock" else np.ones_like(t) if q == "T0" else 0 * t
    return species, cols


def pulse_exposure(shape: str, reads: str, x0: float = 10.0):
    """``X -> Y`` at ``kx*h(u - c)`` for one of ``PULSES``, on a condition that
    reads ``time()``, the clock ``Tobs`` (from ``T0`` at ``k_clock``), or
    #1776's ``a*time() - b*Tobs``, from ``X = x0``. A shape of compact support
    must lie wholly ahead of t = 0; the Gaussian's tail before 0 is taken off.
    Agrees with ``clock_exposure`` where both apply (a frame test holds them
    together), and names the stretch around each pulse's middle for
    ``window_peak``, which a Gaussian narrower than its uniform scan needs."""
    _, _, offsets, centre, has_w = PULSES[shape]

    def fn(p, t, side):
        L, U, dU = _unit_exposure(p, t, side, shape, reads)
        kx = p["kx"]
        dE = {"kx": U, **{q: kx * d for q, d in dU.items()}}
        return _xy(p, t, L, kx * U, dE, x0)

    def crossings(p):
        L = _Line(p, reads)
        return tuple(L.at(o) for o in offsets(p.get("w", 0.0)))

    def peaks(p):
        L, w = _Line(p, reads), p.get("w", 0.0)
        return ((L.at(centre(w) - 2 * w), L.at(centre(w) + 2 * w)),)

    return ClosedForm(fn, crossings, peaks_fn=peaks if has_w else None)


def level_step(reads: str, x0: float = 10.0):
    """``X -> Y`` at ``if(u > c, k1, k0)``: a rate that steps from k0 to k1
    (up or down) as the condition crosses c. E = k0*t + (k1 - k0)*U, with U
    the unit step's exposure."""

    def fn(p, t, side):
        L, U, dU = _unit_exposure(p, t, side, "step", reads)
        k0, k1 = p["k0"], p["k1"]
        dE = {"k0": t - U, "k1": U, **{q: (k1 - k0) * d for q, d in dU.items()}}
        return _xy(p, t, L, k0 * t + (k1 - k0) * U, dE, x0)

    return ClosedForm(fn, lambda p: (_Line(p, reads).at(0.0),))


def state_step(x0: float = 10.0):
    """``X -> Y`` at ``if(X > th, k1, k2)``: X switches its own decay rate as
    it falls through th, at ``t* = ln(x0/th)/k1``. Past t*,
    ``X = th*exp(-k2(t - t*))``, so its k1 column jumps there by the saltation
    term, ``-th*t*(k2/k1 - 1)``, unless k1 = k2."""

    def fn(p, t, side):
        k1, k2, th = p["k1"], p["k2"], p["th"]
        ts = math.log(x0 / th) / k1
        on = past(t, ts, side)
        X = np.where(on, th * np.exp(-k2 * (t - ts)), x0 * np.exp(-k1 * t))
        dX = {
            "k1": np.where(on, -X * k2 * ts / k1, -t * X),
            "k2": np.where(on, -(t - ts) * X, 0.0),
            "th": np.where(on, X / th * (1 - k2 / k1), 0.0),
        }
        cols = {}
        for q, d in dX.items():
            cols[("X()", q)] = d
            cols[("Y()", q)] = -d
        return {"X()": X, "Y()": x0 - X}, cols

    return ClosedForm(fn, lambda p: (math.log(x0 / p["th"]) / p["k1"],))


def min_bend(x0: float = 10.0):
    """``X -> 0`` at kd, and ``0 -> Y`` at ``min(vmax, k*X)``: Y is made at
    vmax until k*X falls through it at ``t* = ln(k*x0/vmax)/kd``, then at k*X.
    Where k*x0 <= vmax there is no crossing, and Y is made at k*X throughout.
    The rate has a corner at t*, not a jump, so every column is continuous."""

    def t_star(p):
        k, kd, vmax = p["k"], p["kd"], p["vmax"]
        return math.log(k * x0 / vmax) / kd if k * x0 > vmax else None

    def fn(p, t, side):
        k, kd, vmax = p["k"], p["kd"], p["vmax"]
        X = x0 * np.exp(-kd * t)
        ts = t_star(p)
        zero = np.zeros_like(t)
        if ts is None:  # k*X stays below vmax: Y' = k*X throughout
            Y = k * (x0 - X) / kd
            dY = {
                "vmax": zero,
                "k": Y / k,
                "kd": k * (X * (t / kd + 1 / kd**2) - x0 / kd**2),
            }
        else:
            on = past(t, ts, side)
            Y = np.where(on, vmax * ts + (vmax - k * X) / kd, vmax * t)
            dY = {
                "vmax": np.where(on, ts, t),
                "k": np.where(on, (vmax / k - X) / kd, 0.0),
                "kd": np.where(
                    on, k * X * (t / kd + 1 / kd**2) - vmax * (ts / kd + 1 / kd**2), 0.0
                ),
            }
        cols = {("Y()", q): d for q, d in dY.items()}
        cols.update({("X()", "kd"): -t * X, ("X()", "k"): zero, ("X()", "vmax"): zero})
        return {"X()": X, "Y()": Y}, cols

    return ClosedForm(fn, lambda p: () if t_star(p) is None else (t_star(p),))


def abs_bend(x0: float = 10.0):
    """``X -> 0`` at kd, and ``0 -> Y`` at ``kx*abs(X - th)``, from ``X = x0 > th``:
    the rate falls to 0 as X passes th at ``t* = ln(x0/th)/kd`` and rises again,
    a corner of abs. So ``Y = kx*((x0 - X)/kd - th*t)`` before t* and
    ``kx*((x0 - 2*th + X)/kd + th*(t - 2*t*))`` after it."""

    def fn(p, t, side):
        kx, th, kd = p["kx"], p["th"], p["kd"]
        ts = math.log(x0 / th) / kd
        on = past(t, ts, side)
        X = x0 * np.exp(-kd * t)
        inner = np.where(on, (x0 - 2 * th + X) / kd + th * (t - 2 * ts), (x0 - X) / kd - th * t)
        dY = {
            "kx": inner,
            "th": kx * np.where(on, t - 2 * ts, -t),
            "kd": kx
            * np.where(
                on,
                -(x0 - 2 * th + X) / kd**2 - t * X / kd + 2 * th * ts / kd,
                -(x0 - X) / kd**2 + t * X / kd,
            ),
        }
        zero = np.zeros_like(t)
        cols = {("Y()", q): d for q, d in dY.items()}
        cols.update({("X()", "kd"): -t * X, ("X()", "kx"): zero, ("X()", "th"): zero})
        return {"X()": X, "Y()": kx * inner}, cols

    return ClosedForm(fn, lambda p: (math.log(x0 / p["th"]) / p["kd"],))


# ── #1915: coincident surfaces and events that set or move the state ────────
#
# An indicator on time alone is exact (``past(..., ulps=0)``): instants a double
# apart (#1627) are two instants, so a sample on one is not taken to be on the
# other. An indicator on the state reads ``past()``'s default blur, since the
# run locates such a crossing only to its own accuracy.
# Each model keeps the witness clock T = k_clock*t (kc in the Antimony ones).

#: Two instants this many ulps apart or fewer are an ulp from a tie: where the
#: tie's columns have no value, a run may decline the columns there too
#: (``Permitted``); whether such instants are kept apart is #1916's to say.
NEAR_TIE_ULPS = 4
#: The refusal a run may give an ulp from such a tie: bngsim's refusal of the
#: parameters whose columns two switches on one instant leave without a value,
#: and no other (#2004's refusal, of columns that do have one, is not excused).
TIE_REFUSAL = "SensitivityRefusedForParametersError"


def _exact(t, t_star, side):
    return past(t, t_star, side, ulps=0)


def _near_tie(u: float, v: float) -> bool:
    return 0 < abs(u - v) <= NEAR_TIE_ULPS * np.spacing(max(abs(u), abs(v)))


def _xy_witness(p, t, E, dE, x0, kc="k_clock", x="X()", y="Y()", T="T()"):
    """X -> Y at exposure E, beside the witness clock T = kc*t."""
    X = x0 * np.exp(-E)
    species = {x: X, y: x0 - X, T: p[kc] * t}
    cols = {}
    for q, d in dE.items():
        cols[(x, q)] = -X * d
        cols[(y, q)] = X * d
        cols[(T, q)] = np.zeros_like(t)
    return species, cols


def coincident_time(interacting: bool, x0: float = 10.0):
    """``X -> Y`` at ``k0 + d1*[t>c1] + d2*[t>c2]`` (additive), or at
    ``k0 + d1*[t>c1 && t>c2]`` (interacting). At c1 = c2 the additive law's
    columns exist: each switch adds its own term, so the order of the two does
    not matter. The interacting law's do not: dX/dc1 is 0 from below and
    X*d1 from above, so a run owes a refusal of the c columns there, and may
    decline them an ulp from it."""

    def fn(p, t, side):
        k0, d1, c1, c2 = p["k0"], p["d1"], p["c1"], p["c2"]
        i1, i2 = _exact(t, c1, side), _exact(t, c2, side)
        if interacting:
            m = max(c1, c2)
            on = i1 & i2
            E = k0 * t + d1 * np.where(on, t - m, 0.0)
            dE = {
                "k0": t,
                "d1": np.where(on, t - m, 0.0),
                "c1": np.where(on & (c1 > c2), -d1, 0.0),
                "c2": np.where(on & (c2 > c1), -d1, 0.0),
            }
        else:
            d2 = p["d2"]
            E = k0 * t + d1 * np.where(i1, t - c1, 0.0) + d2 * np.where(i2, t - c2, 0.0)
            dE = {
                "k0": t,
                "d1": np.where(i1, t - c1, 0.0),
                "d2": np.where(i2, t - c2, 0.0),
                "c1": np.where(i1, -d1, 0.0),
                "c2": np.where(i2, -d2, 0.0),
            }
        return _xy_witness(p, t, E, dE, x0)

    def owes(p, sens):
        if interacting and p["d1"] != 0 and p["c1"] == p["c2"] and {"c1", "c2"} & set(sens):
            return Owed(
                "both switches turn the one term on at the same instant: dX/dc1 is 0 from "
                "below and X*d1 from above, so it has no value"
            )
        return None

    def permits(p, sens):
        if interacting and _near_tie(p["c1"], p["c2"]) and {"c1", "c2"} & set(sens):
            return Permitted("an ulp from a tie at which the c columns have no value", TIE_REFUSAL)
        return None

    return ClosedForm(fn, lambda p: (p["c1"], p["c2"]), owes=owes, permits=permits)


def coincident_state(x0: float = 10.0):
    """``X -> Y`` at ``k0 + d1*[X>th1] + d2*[X>th2]`` from ``X = x0`` above both:
    X falls through the higher threshold first, at ``t_hi = ln(x0/hi)/kA``,
    then through the lower at ``t_lo = t_hi + ln(hi/lo)/kB``, where kA is the
    rate above both, kB between and kC = k0 below. At th1 = th2 the run goes
    from kA to kC at once, and the th columns have no value there (from either
    side the crossing time moves with the rate between, which differs), so a
    run owes their refusal; the rate columns exist.

    Unlike the time switches, a state switch's crossing is located only to the
    run's own accuracy, about rtol*X/|X'|, far wider than a double: two crossings
    within ``AT_CROSSING_ULPS`` of a row are one instant on it, so its limits
    there are before both and after both."""

    def parts(p):
        k0, d1, d2, th1, th2 = (p[q] for q in ("k0", "d1", "d2", "th1", "th2"))
        if not x0 > max(th1, th2):
            raise ValueError(f"x0 = {x0} must lie above both thresholds")
        first1 = th1 >= th2  # th1 is crossed first (or with th2)
        hi, lo = (th1, th2) if first1 else (th2, th1)
        kA, kC = k0 + d1 + d2, k0
        kB = k0 + (d2 if first1 else d1)  # above the lower threshold only
        t_hi = math.log(x0 / hi) / kA
        t_lo = t_hi + math.log(hi / lo) / kB
        # d/dq of (ln hi, ln lo, kA, kB, kC), by parameter.
        dhi = {"th1": 1 / hi if first1 else 0.0, "th2": 0.0 if first1 else 1 / hi}
        dlo = {"th1": 0.0 if first1 else 1 / lo, "th2": 1 / lo if first1 else 0.0}
        dkA = {"k0": 1.0, "d1": 1.0, "d2": 1.0}
        dkB = {"k0": 1.0, "d2" if first1 else "d1": 1.0}
        dkC = {"k0": 1.0}
        return hi, lo, kA, kB, kC, t_hi, t_lo, dhi, dlo, dkA, dkB, dkC

    def fn(p, t, side):
        hi, lo, kA, kB, kC, t_hi, t_lo, dhi, dlo, dkA, dkB, dkC = parts(p)
        a, b = past(t, t_hi, side), past(t, t_lo, side)
        X = np.where(
            b,
            lo * np.exp(-kC * (t - t_lo)),
            np.where(a, hi * np.exp(-kB * (t - t_hi)), x0 * np.exp(-kA * t)),
        )
        dE = {}
        for q in ("k0", "d1", "d2", "th1", "th2"):
            g_hi, g_lo = dhi.get(q, 0.0), dlo.get(q, 0.0)
            a_, b_, c_ = dkA.get(q, 0.0), dkB.get(q, 0.0), dkC.get(q, 0.0)
            dt_hi = -g_hi / kA - t_hi * a_ / kA
            dt_lo = dt_hi + ((g_hi - g_lo) * kB - math.log(hi / lo) * b_) / kB**2
            dlnX = np.where(
                b,
                g_lo - c_ * (t - t_lo) + kC * dt_lo,
                np.where(a, g_hi - b_ * (t - t_hi) + kB * dt_hi, -a_ * t),
            )
            dE[q] = -dlnX  # X = x0*exp(-E)
        return _xy_witness(p, t, -np.log(X / x0), dE, x0)

    def owes(p, sens):
        if p["d1"] != 0 and p["d2"] != 0 and p["th1"] == p["th2"] and {"th1", "th2"} & set(sens):
            return Owed(
                "both thresholds are crossed at one instant: from either side the crossing "
                "time moves with the rate between them, which differs, so dX/dth has no value"
            )
        return None

    def permits(p, sens):
        if _near_tie(p["th1"], p["th2"]) and {"th1", "th2"} & set(sens):
            return Permitted(
                "an ulp from a tie at which the th columns have no value", TIE_REFUSAL
            )
        return None

    def crossings(p):
        return parts(p)[5:7]

    return ClosedForm(fn, crossings, owes=owes, permits=permits)


def switch_event(x0: float = 10.0):
    """``X' = -piecewise(k1, time > c, k0)*X`` from x0, and an event at
    ``time >= te`` that sets ``X = a*X + d`` (a = 1 adds d, a = 0 sets X to d).
    With R(t) = k0*min(t, c) + k1*(t - c)+, X = x0*exp(-R(t)) before te and
    ``a*x0*exp(-R(t)) + d*exp(-(R(t) - R(te)))`` after it. At te = c, dX/dte
    is d*e^(..)*k0 from below and *k1 from above, so the c and te columns have
    no value: a run owes their refusal, and may decline them an ulp from it."""

    def fn(p, t, side):
        k0, k1, c, te, a, d = (p[q] for q in ("k0", "k1", "c", "te", "a", "d"))
        ic, ie = _exact(t, c, side), _exact(t, te, side)
        R = k0 * np.minimum(t, c) + k1 * np.where(ic, t - c, 0.0)
        dR = {
            "k0": np.minimum(t, c),
            "k1": np.where(ic, t - c, 0.0),
            "c": np.where(ic, k0 - k1, 0.0),
        }
        Re = k0 * min(te, c) + k1 * max(te - c, 0.0)
        dRe = {"k0": min(te, c), "k1": max(te - c, 0.0), "c": (k0 - k1) if te > c else 0.0}
        dRe["te"] = k1 if te >= c else k0
        A = a * x0 * np.exp(-R)
        B = d * np.exp(-(R - Re))
        X = np.where(ie, A + B, x0 * np.exp(-R))
        cols = {}
        for q in ("k0", "k1", "c", "te", "a", "d"):
            r_t = dR.get(q, 0.0)
            before = -X * r_t
            after = -(A + B) * r_t + B * dRe.get(q, 0.0)
            if q == "a":
                after = after + x0 * np.exp(-R)
            if q == "d":
                after = after + np.exp(-(R - Re))
            cols[("X", q)] = np.broadcast_to(np.where(ie, after, before), t.shape) * 1.0
            cols[("T", q)] = np.zeros_like(t)
        return {"X": X, "T": p["kc"] * t}, cols

    def owes(p, sens):
        if p["d"] != 0 and p["k0"] != p["k1"] and p["c"] == p["te"] and {"c", "te"} & set(sens):
            return Owed(
                "the event and the switch share their instant: dX/dte is d*k0*e^(..) from "
                "below and d*k1*e^(..) from above, so it has no value"
            )
        return None

    def permits(p, sens):
        if _near_tie(p["c"], p["te"]) and {"c", "te"} & set(sens):
            return Permitted(
                "an ulp from a tie at which the c and te columns have no value", TIE_REFUSAL
            )
        return None

    return ClosedForm(fn, lambda p: (p["c"], p["te"]), owes=owes, permits=permits)


def state_reset(t_end: float, x0: float = 10.0):
    """``X' = -k*X`` from x0, and an event that sets ``X = a*X + d`` each time X
    falls through th: at ``t1 = ln(x0/th)/k`` and every ``D = ln(P/th)/k`` after
    it, ``P = a*th + d`` being X just past each firing. Between firings j and
    j + 1, ``X = P*exp(-k(t - t_j))``, so each column is X times
    d(ln P)/dq - (dk/dq)(t - t_j) + k*dt_j/dq. ``t_end`` bounds the firings
    listed as crossings."""

    def times(p):
        k, th, a, d = p["k"], p["th"], p["a"], p["d"]
        P = a * th + d
        if not (x0 > th and th < P):
            raise ValueError(f"needs x0 > th and a*th + d > th, not x0 = {x0}, P = {P}, th = {th}")
        t1, D = math.log(x0 / th) / k, math.log(P / th) / k
        out, j = [], 0
        while t1 + j * D <= t_end:
            out.append(t1 + j * D)
            j += 1
        return P, t1, D, out

    def fn(p, t, side):
        k, th, a = p["k"], p["th"], p["a"]
        P, t1, D, fire = times(p)
        dt1 = {"k": -t1 / k, "th": -1 / (k * th)}
        dD = {"k": -D / k, "th": (a / P - 1 / th) / k, "a": (th / P) / k, "d": (1 / P) / k}
        dlnP = {"th": a / P, "a": th / P, "d": 1 / P}
        n = sum(past(t, tj, side).astype(int) for tj in fire) if fire else np.zeros(t.shape, int)
        n = np.asarray(n)
        tj = np.where(n > 0, t1 + (n - 1) * D, 0.0)
        X = np.where(n > 0, P * np.exp(-k * (t - tj)), x0 * np.exp(-k * t))
        cols = {}
        for q in ("k", "th", "a", "d"):
            dtj = dt1.get(q, 0.0) + (n - 1) * dD.get(q, 0.0)
            after = X * (dlnP.get(q, 0.0) - (t - tj) * (q == "k") + k * dtj)
            before = -t * X if q == "k" else np.zeros_like(t)
            cols[("X", q)] = np.where(n > 0, after, before)
            cols[("T", q)] = np.zeros_like(t)
        return {"X": X, "T": p["kc"] * t}, cols

    return ClosedForm(fn, lambda p: tuple(times(p)[3]))


def cross_switch(x0: float = 10.0):
    """``X' = -piecewise(k1, X > th, k2)*X`` from x0 > th, and an event at
    ``time >= tau`` that sets ``X = xr``. Before tau, X falls through th at
    ``t1 = ln(x0/th)/k1`` if that comes first. From tau, X starts again at xr:
    above th it falls through th again at ``t2 = tau + ln(xr/th)/k1``; below it,
    it stays below. The event moves the state across the surface with no root,
    so the branch it lands on is read from the state it sets."""

    def parts(p):
        k1, th, tau, xr = p["k1"], p["th"], p["tau"], p["xr"]
        if not x0 > th or xr == th:
            raise ValueError(f"needs x0 > th and xr off th, not x0 = {x0}, xr = {xr}, th = {th}")
        t1 = math.log(x0 / th) / k1
        t2 = tau + math.log(xr / th) / k1 if xr > th else None
        return t1, t2

    def fn(p, t, side):
        k1, k2, th, tau, xr = (p[q] for q in ("k1", "k2", "th", "tau", "xr"))
        t1, t2 = parts(p)
        ie = _exact(t, tau, side)
        i1 = past(t, t1, side) & (t1 < tau)
        X_pre = np.where(i1, th * np.exp(-k2 * (t - t1)), x0 * np.exp(-k1 * t))
        if t2 is not None:
            i2 = past(t, t2, side)
            X_post = np.where(i2, th * np.exp(-k2 * (t - t2)), xr * np.exp(-k1 * (t - tau)))
        else:
            i2 = np.zeros(t.shape, bool)
            X_post = xr * np.exp(-k2 * (t - tau))
        X = np.where(ie, X_post, X_pre)
        dt1 = {"k1": -t1 / k1, "th": -1 / (k1 * th)}
        dt2 = (
            {}
            if t2 is None
            else {
                "k1": -math.log(xr / th) / k1**2,
                "th": -1 / (k1 * th),
                "tau": 1.0,
                "xr": 1 / (k1 * xr),
            }
        )
        cols = {}
        for q in ("k1", "k2", "th", "tau", "xr"):
            pre = np.where(
                i1,
                X * ((q == "th") / th - (q == "k2") * (t - t1) + k2 * dt1.get(q, 0.0)),
                -(q == "k1") * t * X,
            )
            if t2 is not None:
                post = np.where(
                    i2,
                    X * ((q == "th") / th - (q == "k2") * (t - t2) + k2 * dt2.get(q, 0.0)),
                    X * ((q == "xr") / xr - (q == "k1") * (t - tau) + k1 * (q == "tau")),
                )
            else:
                post = X * ((q == "xr") / xr - (q == "k2") * (t - tau) + k2 * (q == "tau"))
            cols[("X", q)] = np.where(ie, post, pre)
            cols[("T", q)] = np.zeros_like(t)
        return {"X": X, "T": p["kc"] * t}, cols

    def crossings(p):
        t1, t2 = parts(p)
        return tuple(c for c in ((t1 if t1 < p["tau"] else None), p["tau"], t2) if c is not None)

    return ClosedForm(fn, crossings)


def clock_reset(x0: float = 10.0):
    """A counter ``T' = kc`` from 0, ``X -> Y`` at ``kx/w`` while
    ``c < T < c + w``, and an event at ``time >= tau`` that sets T back to 0.
    So T = kc*(t - o), o = 0 before tau and tau after, and the window lies at
    ``(o + c/kc, o + (c + w)/kc)`` on each stretch, cut short by the reset
    where tau falls inside it. The exposure is kx/w times the time spent
    inside: the reset moves the crossings the clock carries."""

    def spans(p):
        kc, c, w, tau = p["kc"], p["c"], p["w"], p["tau"]
        out = []
        for s, o in enumerate((0.0, tau)):
            lo, hi = o + c / kc, o + (c + w) / kc
            end = tau if s == 0 else math.inf
            cut = s == 0 and hi > tau
            up = min(hi, end)
            if up > lo:
                dlo = {"c": 1 / kc, "kc": -c / kc**2, "tau": float(s == 1)}
                dup = (
                    {"tau": 1.0}
                    if cut
                    else {"c": 1 / kc, "w": 1 / kc, "kc": -(c + w) / kc**2, "tau": float(s == 1)}
                )
                # The time spent inside once the window has passed: w/kc, taken
                # as it is, so that past the pulse the w and c columns cancel
                # to 0 exactly rather than to the rounding of up - lo.
                width = tau - lo if cut else w / kc
                out.append((lo, up, dlo, dup, cut, width))
        return out

    def fn(p, t, side):
        kx, kc, w, tau = p["kx"], p["kc"], p["w"], p["tau"]
        E = np.zeros_like(t, dtype=float)
        dE = {q: np.zeros_like(t, dtype=float) for q in ("kx", "kc", "c", "w", "tau")}
        for lo, up, dlo, dup, _, width in spans(p):
            a, b = past(t, lo, side), past(t, up, side)
            inside = np.where(b, width, np.where(a, t - lo, 0.0))
            E += kx / w * inside
            for q in dE:
                dE[q] += (
                    kx
                    / w
                    * (np.where(b, dup.get(q, 0.0), 0.0) - np.where(a, dlo.get(q, 0.0), 0.0))
                )
        dE["kx"] = E / kx
        dE["w"] = dE["w"] - E / w
        X = x0 * np.exp(-E)
        it = _exact(t, tau, side)
        T = kc * np.where(it, t - tau, t)
        cols = {}
        for q, d in dE.items():
            cols[("X", q)] = -X * d
            cols[("Y", q)] = X * d
            cols[("T", q)] = (
                np.where(it, t - tau, t)
                if q == "kc"
                else np.where(it, -kc, 0.0)
                if q == "tau"
                else 0 * t
            )
        return {"X": X, "Y": x0 - X, "T": T}, cols

    def crossings(p):
        out = {p["tau"]}
        for lo, up, _, _, cut, _ in spans(p):
            out.add(lo)
            if not cut:
                out.add(up)
        return tuple(sorted(out))

    return ClosedForm(fn, crossings)
