"""Exact solutions of the example models (issue #1706).

Each solution takes the time points and the model's parameters, with the
values the model files have as defaults, and returns each observable as an
array. ``test_examples.py`` checks that the defaults are the values in every
file, and that BNGsim's results match these solutions.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

# Each model's parameters and initial amounts, as its files have them.
DEFAULTS: dict[str, dict[str, float]] = {
    "binding": {"kf": 0.01, "kr": 0.1, "A0": 100.0, "B0": 80.0},
    "decay": {"k": 0.5, "A0": 100.0},
    "chain": {"k1": 1.0, "k2": 0.5, "A0": 100.0},
    "birth_death": {"ksyn": 5000.0, "kdeg": 0.5},
    "bolus": {"k": 0.5, "A0": 100.0, "dose": 100.0, "t_dose": 5.0},
    "schlogl": {"k1": 0.075, "k2": 1e-4, "k3": 1000.0, "k4": 16.5},
}


def _binding_roots(kf: float, kr: float, A0: float, B0: float) -> tuple[float, float]:
    # dC/dt = kf*(A0 - C)*(B0 - C) - kr*C = kf*(C - r1)*(C - r2), r1 < r2.
    b = kf * (A0 + B0) + kr
    disc = math.sqrt(b * b - 4 * kf * kf * A0 * B0)
    return (b - disc) / (2 * kf), (b + disc) / (2 * kf)


def binding(t, kf, kr, A0, B0):
    if kf == 0:  # nothing binds
        C = np.zeros_like(t)
        return {"A": A0 - C, "B": B0 - C, "C": C}
    r1, r2 = _binding_roots(kf, kr, A0, B0)
    e = np.exp(-kf * (r2 - r1) * t)
    C = r1 * r2 * (1 - e) / (r2 - r1 * e)
    return {"A": A0 - C, "B": B0 - C, "C": C}


def decay(t, k, A0):
    return {"A": A0 * np.exp(-k * t)}


def chain(t, k1, k2, A0):
    A = A0 * np.exp(-k1 * t)
    if k1 == k2:  # the limit of the formula below
        B = A0 * k1 * t * np.exp(-k1 * t)
    else:
        B = A0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    return {"A": A, "B": B, "C": A0 - A - B}


def birth_death(t, ksyn, kdeg):
    return {"A": ksyn / kdeg * (1 - np.exp(-kdeg * t))}


def bolus(t, k, A0, dose, t_dose):
    before = A0 * np.exp(-k * t)
    after = (A0 * np.exp(-k * t_dose) + dose) * np.exp(-k * (t - t_dose))
    return {"A": np.where(t < t_dose, before, after)}


def steady_states(name: str, p: dict[str, float]) -> list[dict[str, float]]:
    if name == "binding":
        C = _binding_roots(p["kf"], p["kr"], p["A0"], p["B0"])[0] if p["kf"] else 0.0
        return [{"A": p["A0"] - C, "B": p["B0"] - C, "C": C, "stable": True}]
    if name == "decay" or name == "bolus":
        return [{"A": 0.0, "stable": True}]
    if name == "chain":
        return [{"A": 0.0, "B": 0.0, "C": p["A0"], "stable": True}]
    if name == "birth_death":
        return [{"A": p["ksyn"] / p["kdeg"], "stable": True}]
    if name == "schlogl":
        # dX/dt = -k2*X^3 + k1*X^2 - k4*X + k3, whose real positive roots are
        # the steady states; one is stable where the slope there is negative.
        coeffs = [-p["k2"], p["k1"], -p["k4"], p["k3"]]
        slope = np.polyder(coeffs)
        roots = sorted(r.real for r in np.roots(coeffs) if abs(r.imag) < 1e-9 and r.real > 0)
        return [{"X": float(x), "stable": bool(np.polyval(slope, x) < 0)} for x in roots]
    raise ValueError(f"no example named {name!r}")


SOLUTIONS: dict[str, Callable[..., dict[str, np.ndarray]]] = {
    "binding": binding,
    "decay": decay,
    "chain": chain,
    "birth_death": birth_death,
    "bolus": bolus,
}
