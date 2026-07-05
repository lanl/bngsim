#!/usr/bin/env python3
"""Diagnose what RoadRunner returns for G3 failure models.

The question: does RR.simulate() return species data for these models?
If so, what are the column names?
"""

import antimony as ant
import numpy as np
import roadrunner

MODELS = [
    "m16_normal_dens",
    "m11_Monod_chemostat",
    "V1988a_weibull_growth",
    "V1992_blasius_equation",
    "V1992_log_ode",
    "MS2007_fermentation_yeast",
    "DN2015b_michaelis_menten",
    "RV1990_central_F",
    "m17_central_t",
]

import os  # noqa: E402  (after model-name list above)

DIRS = [
    os.path.expanduser("~/Code/ssys/test_models1"),
    os.path.expanduser("~/Code/ssys/test_models2"),
    os.path.expanduser("~/Code/ssys/test_models3"),
    os.path.expanduser("~/Code/ssys/test_models4"),
]


def find_model(name):
    for d in DIRS:
        p = os.path.join(d, name + ".ant")
        if os.path.exists(p):
            return p
    return None


def main():
    for name in MODELS:
        path = find_model(name)
        if not path:
            print(f"{name}: FILE NOT FOUND")
            continue

        print(f"\n{'=' * 60}")
        print(f"MODEL: {name}")

        ant.clearPreviousLoads()
        ant.loadFile(path)
        mod = ant.getModuleNames()[-1]
        sbml = ant.getSBMLString(mod)

        rr = roadrunner.RoadRunner(sbml)

        # What species does RR know about?
        floating = list(rr.model.getFloatingSpeciesIds())
        boundary = list(rr.model.getBoundarySpeciesIds())
        print(f"  RR floating species: {floating}")
        print(f"  RR boundary species: {boundary}")

        # Simulate and check column names
        rr.integrator.absolute_tolerance = 1e-10
        rr.integrator.relative_tolerance = 1e-10
        result = rr.simulate(0, 1, 11)
        cols = result.colnames
        data = np.array(result)
        print(f"  RR simulate columns: {cols}")
        print(f"  RR simulate shape: {data.shape}")

        # Strip brackets from column names
        sp_names = [c.replace("[", "").replace("]", "") for c in cols[1:]]
        print(f"  Extracted species: {sp_names}")

        # What does BNGsim produce?
        import bngsim

        model = bngsim.Model.from_antimony(path)
        print(f"  BNG species: {model.species_names}")

        # Show what names would match
        bng_names = model.species_names
        matched = []
        unmatched = []
        for bn in bng_names:
            mn = bn.replace("_ant_", "")
            if mn in sp_names:
                matched.append((bn, mn))
            else:
                unmatched.append(bn)
        print(f"  Matched: {matched}")
        print(f"  Unmatched BNG: {unmatched}")


if __name__ == "__main__":
    main()
