#!/usr/bin/env python3
"""GH #212 Phase-1 acceptance — forward sensitivity through Smith2013's events.

Smith2013 (BIOMD0000000474 / MODEL1212210000, "Regulation of Insulin Signalling
by Oxidative Stress"; 133 species, 367 reactions) is the real-world fixed-time
multi-event model behind AMICI's event-sensitivity benchmark: its three events
are all ``geq(time, t_ins | const)`` insulin on/off pulses that reset ``Ins`` to
a constant — exactly the Phase-1 subclass (``∂t*/∂p = 0`` for any non-trigger
parameter, so the sensitivity jump collapses to ``s⁺ = J_h·s⁻ + ∂h/∂p``).

Before GH #212 bngsim hard-raised on this model the moment sensitivities were
requested. This script is the AMICI-independent acceptance: it integrates the
*coupled* state+sensitivity system straight through all three events and checks
the analytic species sensitivities ``dx/dp`` against bngsim's own central finite
difference over the same parameters — the FD guard that does not depend on any
external oracle. The events fire at ``t ≈ t_ins`` and at ``t = 2880, 2895``; the
acceptance is specifically that analytic and FD agree *across* those firings.

    python smith_event_sens.py                 # default 4 network params
    python smith_event_sens.py --params a,b,c   # explicit parameter names
    python smith_event_sens.py --n 6            # auto-pick the 6 most sensitive

Run inside ``bngsim/.venv`` (the amici+rr+bngsim environment). No AMICI is
required here; the AMICI ``sy`` cross-check is the companion in ``output_sens``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bngsim
import numpy as np

_SUITE_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _SUITE_DIR.parents[1]  # bngsim/benchmarks
SMITH = _BENCH_ROOT / "models" / "sbml" / "Smith2013_BIOMD0000000474_petab.xml"

# Trigger-time / indicator parameters: requesting these is Phase 2 (∂t*/∂p ≠ 0),
# so they are excluded from the Phase-1 acceptance set.
TRIGGER_PARAMS = {"t_ins", "indicator_jnk", "indicator_foxo"}

# Well-conditioned rate constants for the default acceptance. The leading model
# parameters are *compartment volumes* (`extracellular`, `cellsurface`, …) whose
# sensitivities are stiff and poorly scaled — they make the coupled solve crawl
# (mxstep) before any event even fires — so the default set is a handful of
# ordinary mass-action rate constants instead. Override with --params.
DEFAULT_PARAMS = ["k4", "kminus4", "k6"]

T_END = 3000.0
N_POINTS = 61
# Smith's coupled state+sensitivity system is stiff; give CVODE plenty of steps.
MAX_STEPS = 2_000_000
# Relative noise floor (mirrors output_sens.py): keep opposite-sign float noise
# on near-zero sensitivities from dominating the headline relerr.
RTOL = 5e-3
ATOL_REL = 1e-6


def _species_traj(model, params, overrides):
    m = model.clone() if hasattr(model, "clone") else bngsim.Model.from_sbml(str(SMITH))
    for name, val in overrides.items():
        m.set_param(name, val)
    sim = bngsim.Simulator(m, method="ode", sensitivity_params=list(params))
    r = sim.run(t_span=(0, T_END), n_points=N_POINTS, max_steps=MAX_STEPS)
    return r


def _fd_species_sens(model, params, p0):
    """Central-difference ``dx_species/dp`` on bngsim's own trajectory."""
    cols = []
    for p in params:
        base = float(p0[p])
        eps = 1e-6 * abs(base) if base else 1e-9
        hi = np.asarray(_species_traj(model, [], {p: base + eps}).species)
        lo = np.asarray(_species_traj(model, [], {p: base - eps}).species)
        cols.append((hi - lo) / (2 * eps))
    return np.stack(cols, axis=-1)  # (nt, n_species, n_param)


def _relerr(analytic, fd):
    scale = float(max(np.abs(analytic).max(), np.abs(fd).max(), 1e-300))
    atol = max(1e-12, ATOL_REL * scale)
    denom = np.maximum(np.abs(fd), atol)
    e = np.abs(analytic - fd) / denom
    return {
        "max": float(e.max()),
        "p95": float(np.percentile(e, 95)),
        "med": float(np.median(e)),
        "scale": scale,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="comma-separated parameter names")
    ap.add_argument("--n", type=int, default=4, help="auto-pick N most-sensitive params")
    args = ap.parse_args(argv)

    if not SMITH.exists():
        print(f"missing model: {SMITH}", file=sys.stderr)
        return 2

    model = bngsim.Model.from_sbml(str(SMITH))
    pnames = [p for p in model.param_names if p not in TRIGGER_PARAMS]
    p0 = {p: model.get_param(p) for p in model.param_names}

    if args.params:
        params = [p.strip() for p in args.params.split(",") if p.strip()]
    else:
        # Auto-pick: run once over a broad candidate set, keep the N parameters
        # whose analytic sensitivity has the largest norm (so FD has signal).
        cand = pnames[: min(len(pnames), 40)]
        r = _species_traj(model, cand, {})
        S = np.asarray(r.sensitivities)  # (nt, n_species, n_cand)
        norms = np.linalg.norm(S.reshape(-1, S.shape[-1]), axis=0)
        order = np.argsort(norms)[::-1]
        params = [cand[i] for i in order[: args.n] if norms[i] > 0]

    print(f"model: {SMITH.name}  events={model._core.n_events}  params={params}")

    r = _species_traj(model, params, {})
    t = np.asarray(r.time)
    analytic = np.asarray(r.sensitivities)  # (nt, n_species, n_param)
    assert np.all(np.isfinite(analytic)), "non-finite analytic sensitivities"

    fd = _fd_species_sens(model, params, p0)
    stats = _relerr(analytic, fd)

    # Acceptance: median relerr below the floor (FD guard) AND finite + matching
    # at the post-event tail (the discontinuities are crossed correctly).
    post_event = t >= 2895.0
    tail = _relerr(analytic[post_event], fd[post_event]) if post_event.any() else stats
    ok = stats["med"] <= RTOL and tail["med"] <= RTOL

    out = {
        "params": params,
        "n_events": model._core.n_events,
        "relerr_all": stats,
        "relerr_post_event": tail,
        "rtol": RTOL,
        "pass": bool(ok),
    }
    print(json.dumps(out, indent=2))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
