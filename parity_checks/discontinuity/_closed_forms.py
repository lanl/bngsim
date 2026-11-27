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

import numpy as np
from _reference import ClosedForm, past


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
    lie ahead of the start (``c > 0``) and the clock must carry u forward
    (``r = a - b*k_clock > 0``)."""
    h, H, offsets = SHAPES[shape]

    def fn(p, t, side):
        kx, kc, a, b, c = (p[q] for q in ("kx", "k_clock", "a", "b", "c"))
        w = p.get("w", 0.0)
        r = a - b * kc
        if r <= 0 or c <= 0:
            raise ValueError(f"clock_exposure needs r > 0 and c > 0, not r = {r}, c = {c}")
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
