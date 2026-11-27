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
"""

from __future__ import annotations

import math

import numpy as np
from _reference import ClosedForm, Owed, past


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


def _ind(t, t_star, side):
    """1 from ``t_star`` on (a trigger that holds at ``t_star`` itself), as a
    float array; the left limit at ``t_star`` is 0, the right 1."""
    return past(t, t_star, side).astype(float)


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
    0.3), and they fire in time order, so F precedes E."""

    def fn(p, t, side):
        th, te = 0.1 * p["k"], 0.1 * p["j"]
        amt = 2.0 if te >= 0.3 else 1.0
        return {
            "g": _ind(t, 0.3, side),
            "h": _ind(t, th, side),
            "D": amt * _ind(t, te, side),
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
    ``t* = (c^(1/3) - 1)/eps``. The cube root is taken at 30 digits: at eps = 1e-8
    a double's rounding of it would move t* by 2e-8."""
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
