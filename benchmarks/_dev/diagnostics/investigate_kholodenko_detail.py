#!/usr/bin/env python3
"""Deep investigation of Kholodenko_2000 BNGsim vs run_network divergence.

The 17% error at t=10 is NOT phase drift — it's too early for oscillations.
Hypothesis: Observable evaluation order or mapping issue in functional rates.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np

# BioNetGen 2.9.3 install. BNGPATH gives the root; BNG2_PL / RUN_NETWORK
# override an individual tool. Default = canonical ~/Simulations install.
BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
BNG2_PL = os.environ.get("BNG2_PL", os.path.join(BNGPATH, "BNG2.pl"))
RUN_NETWORK = os.environ.get("RUN_NETWORK", os.path.join(BNGPATH, "bin", "run_network"))
RULEBENDER_WS = os.environ.get(
    "RULEBENDER_WS", os.path.expanduser("~/Simulations/RuleBender-workspace")
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    import bngsim

    # Generate .net
    src = os.path.join(RULEBENDER_WS, "RAS_oscillations", "Kholodenko_2000.bngl")
    tmpdir = tempfile.mkdtemp()
    shutil.copy2(src, tmpdir)
    subprocess.run(
        ["perl", BNG2_PL, "Kholodenko_2000.bngl"], capture_output=True, cwd=tmpdir, timeout=120
    )
    net = os.path.join(tmpdir, "Kholodenko_2000.net")

    print("=== .net file analysis ===")
    print()
    with open(net) as f:
        content = f.read()
    print(content)

    print()
    print("=== Observable → Species mapping ===")
    print("In .net groups block:")
    print("  MKKK   → species 3 = MKK()    (!) Observable 'MKKK' counts MKK molecules")
    print("  MKKK_P → species 2 = MKKK_P()")
    print("  MKK    → species 3 = MKK()")
    print("  MKK_P  → species 4 = MKK_P()")
    print()
    print("In functions:")
    print("  v1() = V1/((1+((MAPK_PP/KI)^n))*(K1+MKKK))")
    print("  MKKK here = observable = group 1 = species 3 = MKK()")
    print()

    # Run BNGsim with very short timestep to check initial behavior
    print("=== BNGsim initial state check ===")
    m = bngsim.Model.from_net(net)
    sim = bngsim.Simulator(m, method="ode")
    r = sim.run(t_span=(0, 1), n_points=11)
    bng_sp = np.asarray(r.species)
    np.asarray(r.observables) if hasattr(r, "observables") else None

    print(f"  Species at t=0: {bng_sp[0]}")
    print(f"  Species at t=1: {bng_sp[1]}")

    # Run run_network with same timestep
    prefix = os.path.join(tmpdir, "rn_detail")
    cmd = [RUN_NETWORK, "-g", net, "-o", prefix, net, "0.1", "10"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    cdat = prefix + ".cdat"
    gdat = prefix + ".gdat"

    if os.path.exists(cdat):
        rn_data = np.loadtxt(cdat, comments="#")
        rn_sp = rn_data[:, 1:]
        print(f"  RN species at t=0: {rn_sp[0]}")
        print(f"  RN species at t=1: {rn_sp[1]}")

        print()
        print("=== Species-by-species comparison at t=0.1 ===")
        sp_names = ["MKKK", "MKKK_P", "MKK", "MKK_P", "MKK_PP", "MAPK", "MAPK_P", "MAPK_PP"]
        for i, name in enumerate(sp_names):
            b = bng_sp[1, i]
            r_val = rn_sp[1, i]
            diff = abs(b - r_val)
            rel = diff / max(abs(r_val), 1e-10)
            flag = " ***" if rel > 0.001 else ""
            print(
                f"  {name:10s}: BNG={b:12.6f}  RN={r_val:12.6f}  "
                f"diff={diff:.2e}  rel={rel:.2e}{flag}"
            )

    if os.path.exists(gdat):
        print()
        print("=== run_network .gdat (observables) at t=0.1 ===")
        with open(gdat) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") or line.startswith("#"):
                    print(f"  {line}")
                if "1.0" in line.split()[0:1]:
                    break

    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
