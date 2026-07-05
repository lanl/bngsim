#!/usr/bin/env python3
"""Compare CVODE settings: BNGsim vs run_network vs libRoadRunner.

The question: why do 5 orbit models fail in BNGsim but succeed in RR?
"""

import roadrunner

# ── libRoadRunner defaults ────────────────────────────────────────────
rr = roadrunner.RoadRunner()
integ = rr.integrator
print("=== libRoadRunner CVODE defaults ===")
print(f"  absolute_tolerance: {integ.absolute_tolerance}")
print(f"  relative_tolerance: {integ.relative_tolerance}")
print(f"  maximum_bdf_order: {integ.maximum_bdf_order}")
print(f"  maximum_adams_order: {integ.maximum_adams_order}")
print(f"  maximum_num_steps: {integ.maximum_num_steps}")
print(f"  maximum_time_step: {integ.maximum_time_step}")
print(f"  minimum_time_step: {integ.minimum_time_step}")
print(f"  initial_time_step: {integ.initial_time_step}")
print(f"  stiff: {integ.stiff}")
print(f"  multiple_steps: {integ.multiple_steps}")
print(f"  variable_step_size: {integ.variable_step_size}")

# Check all settings
print("\n  All settings:")
for key in sorted(integ.getExistingIntegratorNames()):
    print(f"    {key}")

# ── BNGsim defaults ──────────────────────────────────────────────────
print("\n=== BNGsim CVODE defaults ===")
print("  rtol: 1e-8")
print("  atol: 1e-8")
print("  max_steps: 10000")
print("  max_step_size: 0 (no limit)")
print("  min_step_size: NOT SET (SUNDIALS default)")
print("  initial_step_size: NOT SET (SUNDIALS auto)")
print("  method: BDF only (no Adams)")
print("  nonlinear solver: Newton")

# ── run_network defaults ──────────────────────────────────────────────
print("\n=== run_network (BNG2) defaults ===")
print("  rtol: 1e-8")
print("  atol: 1e-8")
print("  max_steps: 10000 (same as BNGsim)")
print("  max_step_size: NOT SET")
print("  method: BDF + Newton (same as BNGsim)")
print("  SUNDIALS version: 2.4.0 (ancient)")

print("\n=== KEY DIFFERENCES ===")
print("1. max_steps: RR=20000, BNGsim=10000, run_network=10000")
print("2. stiff mode: RR tries Adams first, falls back to BDF")
print("   BNGsim: BDF only (stiff solver)")
print("   run_network: BDF only (stiff solver)")
print("3. Tolerances: All three use 1e-8/1e-8 by default")
print("4. SUNDIALS version: RR=5.x, BNGsim=7.x, run_network=2.4")

# ── Test: do orbit models work with max_steps=20000? ─────────────────
print("\n=== TEST: Orbit models with max_steps=20000 ===")
import os  # noqa: E402  (after diagnostic banner above)

import bngsim  # noqa: E402

SSYS = os.path.expanduser("~/Code/ssys/test_models2")
orbits = [
    "S1987_D1_orbit_e0.1",
    "S1987_D2_orbit_e0.3",
    "S1987_D3_orbit_e0.5",
    "S1987_D4_orbit_e0.7",
    "S1987_D5_orbit_e0.9",
]
for name in orbits:
    path = os.path.join(SSYS, name + ".ant")
    model = bngsim.Model.from_antimony(path)
    sim = bngsim.Simulator(model, method="ode")
    try:
        result = sim.run(
            t_span=(0, 20),
            n_points=201,
            rtol=1e-8,
            atol=1e-8,
            max_steps=20000,
        )
        print(f"  {name}: max_steps=20000 → PASS")
    except Exception as e:
        err_msg = str(e)[:60]
        print(f"  {name}: max_steps=20000 → FAIL: {err_msg}")
        # Try max_steps=100000
        try:
            result = sim.run(
                t_span=(0, 20),
                n_points=201,
                rtol=1e-8,
                atol=1e-8,
                max_steps=100000,
            )
            print("           max_steps=100000 → PASS")
        except Exception:
            print("           max_steps=100000 → STILL FAILS")
