#!/usr/bin/env python3
"""Diagnose G2 and G3 Antimony sweep failures.

G2 hypothesis: max_steps=10000 too low (RoadRunner default=100000)
G3 hypothesis: ODE system construction differs from RoadRunner
"""

import json

import numpy as np

# Load sweep results
with open("bngsim/benchmarks/suites/antimony/results/antimony_sweep_results.json") as f:
    results = json.load(f)

g2_fails = [r for r in results if r["g1_load"] and not r["g2_ode"]]
g3_fails = [r for r in results if r.get("g2_ode") and not r.get("g3_xval")]

# ── G2: Test with max_steps=100000 ────────────────────────────────
print("=" * 60)
print("G2 DIAGNOSIS: max_steps=10000 vs 100000")
print("=" * 60)

import antimony as ant  # noqa: E402  (after data filtering banner above)
import bngsim  # noqa: E402
import roadrunner  # noqa: E402

for r in g2_fails:
    name, path = r["name"], r["path"]
    model = bngsim.Model.from_antimony(path)
    sim = bngsim.Simulator(model, method="ode")

    # Try with more steps
    try:
        result = sim.run(
            t_span=(0, 20),
            n_points=201,
            rtol=1e-10,
            atol=1e-10,
            max_steps=100000,
        )
        has_nan = np.any(np.isnan(result.species))
        if has_nan:
            print(f"  {name}: STILL FAILS (NaN) even with 100K steps")
        else:
            print(f"  {name}: PASSES with max_steps=100000!")
    except Exception as e:
        print(f"  {name}: STILL FAILS: {str(e)[:60]}")

    # Also test RoadRunner
    try:
        ant.clearPreviousLoads()
        ant.loadFile(path)
        mod = ant.getModuleNames()[-1]
        sbml = ant.getSBMLString(mod)
        rr = roadrunner.RoadRunner(sbml)
        rr.integrator.absolute_tolerance = 1e-10
        rr.integrator.relative_tolerance = 1e-10
        rr_result = rr.simulate(0, 20, 201)
        rr_data = np.array(rr_result)
        if np.any(np.isnan(rr_data)):
            print("    RoadRunner: ALSO FAILS (NaN)")
        else:
            print("    RoadRunner: succeeds")
    except Exception as e:
        print(f"    RoadRunner: fails: {str(e)[:60]}")

# ── G3: Compare RHS at t=0 ────────────────────────────────────────
print()
print("=" * 60)
print("G3 DIAGNOSIS: RHS comparison at t=0")
print("=" * 60)

for r in g3_fails[:5]:
    name, path = r["name"], r["path"]
    print(f"\n--- {name} (err={r.get('max_rel_err', 'N/A')}) ---")

    # BNGsim: get species names and initial values
    model = bngsim.Model.from_antimony(path)
    sp_names_bng = model.species_names
    n_sp = model.n_species
    print(f"  BNGsim species ({n_sp}): {sp_names_bng}")

    # RoadRunner: get species names
    ant.clearPreviousLoads()
    ant.loadFile(path)
    mod = ant.getModuleNames()[-1]
    sbml = ant.getSBMLString(mod)
    rr = roadrunner.RoadRunner(sbml)
    sp_names_rr = rr.model.getFloatingSpeciesIds()
    print(f"  RR species ({len(sp_names_rr)}): {list(sp_names_rr)}")

    # Compare ICs
    bng_ic = [model.get_concentration(n) for n in sp_names_bng]
    rr_ic = [rr.model[f"[{n}]"] for n in sp_names_rr]
    print(f"  BNGsim ICs: {bng_ic}")
    print(f"  RR ICs:     {rr_ic}")

    # Run one tiny step and compare
    sim = bngsim.Simulator(model, method="ode")
    bng_res = sim.run(
        t_span=(0, 0.001),
        n_points=2,
        rtol=1e-12,
        atol=1e-12,
    )
    rr.integrator.absolute_tolerance = 1e-12
    rr.integrator.relative_tolerance = 1e-12
    rr_res = rr.simulate(0, 0.001, 2)
    rr_data = np.array(rr_res)

    bng_final = bng_res.species[-1, :]
    print(f"  BNGsim at t=0.001: {bng_final}")

    # Match by name
    for i, bname in enumerate(sp_names_bng):
        # strip _ant_ prefix for matching
        match_name = bname.replace("_ant_", "")
        if match_name in sp_names_rr:
            ri = list(sp_names_rr).index(match_name)
            rr_val = rr_data[-1, ri + 1]  # +1 for time col
            bng_val = bng_final[i]
            diff = abs(bng_val - rr_val)
            if diff > 1e-10:
                print(f"  MISMATCH {bname}: BNG={bng_val:.8e} RR={rr_val:.8e} diff={diff:.2e}")
