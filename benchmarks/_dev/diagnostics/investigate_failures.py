#!/usr/bin/env python3
"""Investigate ODE validation failures: Kholodenko_2000 and LV.

Writes results to stdout. Run from benchmarks directory.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BioNetGen 2.9.3 install. BNGPATH gives the root; BNG2_PL / RUN_NETWORK
# override an individual tool. Default = canonical ~/Simulations install.
BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
BNG2_PL = os.environ.get("BNG2_PL", os.path.join(BNGPATH, "BNG2.pl"))
RUN_NETWORK = os.environ.get("RUN_NETWORK", os.path.join(BNGPATH, "bin", "run_network"))
RULEBENDER_WS = os.environ.get(
    "RULEBENDER_WS", os.path.expanduser("~/Simulations/RuleBender-workspace")
)

sys.path.insert(0, SCRIPT_DIR)


def generate_net(bngl_path):
    """Generate .net in temp dir, return path."""
    tmpdir = tempfile.mkdtemp()
    shutil.copy2(bngl_path, tmpdir)
    stem = os.path.splitext(os.path.basename(bngl_path))[0]
    subprocess.run(
        ["perl", BNG2_PL, f"{stem}.bngl"],
        capture_output=True,
        cwd=tmpdir,
        timeout=120,
    )
    net = os.path.join(tmpdir, f"{stem}.net")
    if os.path.exists(net):
        return net, tmpdir
    return None, tmpdir


def run_rn_ode(net_path, t_end, n_steps):
    """Run run_network ODE, return species array."""
    tmpdir = tempfile.mkdtemp()
    prefix = os.path.join(tmpdir, "out")
    dt = t_end / n_steps
    cmd = [RUN_NETWORK, "-g", net_path, "-o", prefix, net_path, f"{dt:.15g}", str(n_steps)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    cdat = prefix + ".cdat"
    if os.path.exists(cdat):
        data = np.loadtxt(cdat, comments="#")
        shutil.rmtree(tmpdir)
        return data[:, 1:]  # skip time column
    shutil.rmtree(tmpdir)
    return None


def investigate_lv():
    """LV: test multiple time horizons."""
    import bngsim

    print("=" * 60)
    print("  LV (Lotka-Volterra) Investigation")
    print("=" * 60)

    net = os.path.join(SCRIPT_DIR, "LV.net")

    for t_end in [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0, 10.0]:
        try:
            m = bngsim.Model.from_net(net)
            sim = bngsim.Simulator(m, method="ode")
            r = sim.run(t_span=(0, t_end), n_points=101)
            sp = np.asarray(r.species)
            stats = r.solver_stats
            print(
                f"  t_end={t_end:8.4f}: OK  "
                f"S(end)={sp[-1, 0]:10.1f}  W(end)={sp[-1, 1]:10.1f}  "
                f"steps={stats.get('n_steps', 0)}"
            )
        except Exception as e:
            print(f"  t_end={t_end:8.4f}: FAIL  {str(e)[:80]}")


def investigate_kholodenko():
    """Kholodenko MAPK cascade: compare at multiple time horizons."""
    import bngsim

    print()
    print("=" * 60)
    print("  Kholodenko_2000 (MAPK cascade) Investigation")
    print("=" * 60)

    src = os.path.join(RULEBENDER_WS, "RAS_oscillations", "Kholodenko_2000.bngl")
    net, tmpdir = generate_net(src)
    if net is None:
        print("  FAIL: Could not generate .net")
        return

    # Species index 7 = MAPK_PP (the one that diverged)
    sp_idx = 7
    sp_name = "MAPK_PP"

    for t_end in [10, 50, 100, 500, 1000, 2000, 4000]:
        n_steps = max(100, t_end)
        try:
            m = bngsim.Model.from_net(net)
            sim = bngsim.Simulator(m, method="ode")
            r = sim.run(t_span=(0, t_end), n_points=n_steps + 1)
            bng_sp = np.asarray(r.species)
        except Exception as e:
            print(f"  t_end={t_end:5d}: BNGsim FAIL: {e}")
            continue

        rn_sp = run_rn_ode(net, t_end, n_steps)
        if rn_sp is None:
            print(f"  t_end={t_end:5d}: run_network FAIL")
            continue

        # Check shapes match
        if bng_sp.shape != rn_sp.shape:
            print(f"  t_end={t_end:5d}: shape mismatch BNG={bng_sp.shape} RN={rn_sp.shape}")
            continue

        # Relative error
        denom = np.maximum(np.abs(rn_sp), 1e-8)
        rel_err = np.abs(bng_sp - rn_sp) / denom
        max_err = np.max(rel_err)

        # Find where max error occurs
        idx = np.unravel_index(np.argmax(rel_err), rel_err.shape)
        t_at_max = idx[0] * (t_end / n_steps)

        # Report MAPK_PP at end
        bng_val = bng_sp[-1, sp_idx]
        rn_val = rn_sp[-1, sp_idx]

        status = "PASS" if max_err < 1e-5 else "DRIFT" if max_err < 1 else "FAIL"
        print(
            f"  t_end={t_end:5d}: {status}  "
            f"max_rel_err={max_err:.2e} at t={t_at_max:.0f}  "
            f"{sp_name}(end): BNG={bng_val:.2f} RN={rn_val:.2f}"
        )

    shutil.rmtree(tmpdir)

    # Print Kholodenko rate law analysis
    print()
    print("  Rate law analysis:")
    print("  - All 10 reactions use FUNCTIONAL rate laws (v1..v10)")
    print("  - v1()=V1/((1+(MAPK_PP/KI)^n)*(K1+MKKK)) — MM-like, hand-written")
    print("  - NOT MichaelisMenten type — no tQSSA/sQSSA distinction applies")
    print("  - Both engines evaluate identical ExprTk/muParser expressions")
    print("  - Oscillatory system: phase drift expected from CVODE v7 vs v2.4")


if __name__ == "__main__":
    investigate_lv()
    investigate_kholodenko()
