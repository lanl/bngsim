#!/usr/bin/env python3
"""Diagnose the 9 remaining G3 cross-validation failures.

For each model, compare BNGsim vs RoadRunner:
  - Initial conditions
  - State at t=0.001 (one tiny step)
  - Identify exact species where mismatch occurs

Session 23 — bug hunting.
"""

import json

import antimony as ant
import bngsim
import numpy as np
import roadrunner


def main():
    with open("bngsim/benchmarks/suites/antimony/results/antimony_sweep_results.json") as f:
        results = json.load(f)

    g3_fails = [r for r in results if r.get("g2_ode") and not r.get("g3_xval")]

    print("=" * 70)
    print(f"DIAGNOSING {len(g3_fails)} G3 FAILURES")
    print("=" * 70)

    for r in g3_fails:
        name = r["name"]
        path = r["path"]
        err = r.get("max_rel_err", "?")
        print(f"\n{'=' * 60}")
        print(f"MODEL: {name} (max_rel_err={err})")
        print(f"{'=' * 60}")

        # Load in BNGsim
        model = bngsim.Model.from_antimony(path)
        sp_bng = model.species_names
        n_sp = model.n_species

        # Load in RoadRunner
        ant.clearPreviousLoads()
        ant.loadFile(path)
        mod = ant.getModuleNames()[-1]
        sbml = ant.getSBMLString(mod)
        rr = roadrunner.RoadRunner(sbml)
        sp_rr = list(rr.model.getFloatingSpeciesIds())

        print(f"  BNG species ({n_sp}): {sp_bng}")
        print(f"  RR  species ({len(sp_rr)}): {sp_rr}")

        # Compare ICs
        ic_mismatches = []
        for bn in sp_bng:
            match = bn.replace("_ant_", "")
            if match in sp_rr:
                bv = model.get_concentration(bn)
                rv = rr.model["[" + match + "]"]
                if abs(bv - rv) > 1e-10 * max(abs(rv), 1):
                    ic_mismatches.append((bn, match, bv, rv))

        if ic_mismatches:
            print("  IC MISMATCHES:")
            for bn, _rn, bv, rv in ic_mismatches:
                print(f"    {bn}: BNG={bv} RR={rv}")
        else:
            print("  ICs match.")

        # Compare at t=0.001
        try:
            sim = bngsim.Simulator(model, method="ode")
            br = sim.run(
                t_span=(0, 0.001),
                n_points=2,
                rtol=1e-12,
                atol=1e-12,
            )
            rr.integrator.absolute_tolerance = 1e-12
            rr.integrator.relative_tolerance = 1e-12
            rr_res = rr.simulate(0, 0.001, 2)
            rr_data = np.array(rr_res)

            t_mismatches = []
            for i, bn in enumerate(sp_bng):
                match = bn.replace("_ant_", "")
                if match in sp_rr:
                    ri = sp_rr.index(match)
                    bv = br.species[-1, i]
                    rv = rr_data[-1, ri + 1]
                    diff = abs(bv - rv)
                    rel = diff / max(abs(rv), 1e-12)
                    if rel > 1e-6:
                        t_mismatches.append((bn, match, bv, rv, rel))

            if t_mismatches:
                print("  t=0.001 MISMATCHES:")
                for bn, _rn, bv, rv, rel in t_mismatches:
                    print(f"    {bn}: BNG={bv:.8e} RR={rv:.8e} rel={rel:.2e}")
            else:
                print("  t=0.001 values match.")

        except Exception as e:
            print(f"  ERROR during comparison: {e}")

        # Show canonical Antimony for context
        ant.clearPreviousLoads()
        ant.loadFile(path)
        mod = ant.getModuleNames()[-1]
        s = ant.getAntimonyString(mod)
        # Show just assignment rules and key sections
        has_assign = False
        has_react = False
        for line in s.splitlines():
            stripped = line.strip()
            if "Assignment Rules" in stripped:
                has_assign = True
            if "Reactions:" in stripped:
                has_react = True
        if has_assign:
            print("  [Has Assignment Rules]")
        if has_react:
            print("  [Has Reactions]")


if __name__ == "__main__":
    main()
