#!/usr/bin/env python3
"""``ode`` suite runner — ODE correctness + timing: BNGsim CVODE vs run_network.

Two gates per curated BNGL model:

1. correctness — BNG2.pl generates the .net; BNGsim and run_network each
   integrate the ODE system; the trajectories are cross-validated
   (relative tolerance ~1e-5). Both engines integrate the *same* reaction
   network, so the solution is deterministic and a direct trajectory
   comparison is the right test (unlike the stochastic ssa/psa suites,
   which compare ensembles).
2. timing — a warmup + timed-run wall-clock comparison, median reported.
   Per the suite design rule, timing is only reported for a model that
   passed cross-validation.

Usage:
    python run.py                     # both gates, full sweep
    python run.py --mode correctness  # cross-validation only
    python run.py --mode timing       # timing only
    python run.py --quick             # BNGsim load+run smoke, no run_network
    python run.py --effort low        # cheap subset (cumulative tiers)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np

_BENCH_ROOT = Path(__file__).resolve().parents[2]  # bngsim/benchmarks
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402
from _effort import add_effort_arg, filter_by_effort  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = _BENCH_ROOT  # bngsim/benchmarks
RESULTS_DIR = SCRIPT_DIR / "results"

# BioNetGen 2.9.3 install. Set BNGPATH to the install root; the BNG2.pl and
# run_network paths are derived from it. BNG2_PL / RUN_NETWORK may override an
# individual tool. Defaults assume the canonical ~/Simulations/BioNetGen-2.9.3.
BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
BNG2_PL = os.environ.get("BNG2_PL", os.path.join(BNGPATH, "BNG2.pl"))
RUN_NETWORK = os.environ.get("RUN_NETWORK", os.path.join(BNGPATH, "bin", "run_network"))

BNG_TIMEOUT = 120  # seconds for BNG2.pl network generation
RN_TIMEOUT = 120  # seconds for run_network ODE
BNGSIM_TIMEOUT = 60  # seconds for BNGsim ODE

# Timing-gate protocol: discarded warmup passes then timed passes, median
# reported (shared defaults with the ssa/psa suites via _netbench).
DEFAULT_WARMUP = nb.DEFAULT_WARMUP
DEFAULT_RUNS = nb.DEFAULT_RUNS

# Model corpus -- all four source buckets are vendored in-repo under
# models/bngl/<source>/, a fixed path with no env-var override (a vendored
# corpus has no external dependency). The "ode_bench" bucket holds the
# RuleBender models and the SSA-bench models as a curated mixed-provenance
# corpus (see models/README.md).
_BNGL = BENCH_ROOT / "models" / "bngl"
_SRC_DIR = {
    "models2": _BNGL / "models2",
    "pybnf": _BNGL / "pybnf",
    "rulehub": _BNGL / "rulehub",
    "ode_bench": _BNGL / "ode",
}

# Curated ODE candidate list -- unique published models, no fit iterations,
# no Sat/Hill, no cBNGL/energy, no __FREE (or ground-truth versions).
#
# Each model carries an "effort" tier -- "low" / "medium" / "high" -- so the
# sweep can be run as a cheap subset via --effort (see _effort.py). The cost
# driver is the generated network size (the ODE state dimension): "low" =
# small networks, "medium" = moderate, "high" = large rule-based networks
# (hundreds-to-thousands of species). Size-based estimates, not a wall-clock
# calibration; refine them if a calibration run is done.
#
# "src" selects the corpus directory (see _SRC_DIR); "path" is filled in below.
# The van der Pol "oscillator" model from the old RuleBender workspace was
# dropped -- the workspace no longer carries it and no copy survives in-repo.
CURATED_MODELS = [
    # === BNG2 Models2 (distributed with BioNetGen, canonical) ===
    {
        "name": "LV",
        "src": "models2",
        "file": "LV.bngl",
        "source": "BNG2 Models2",
        "ref": "Lotka-Volterra",
        "effort": "low",
        # The model's own actions block specifies t_end=0.001 (its rate
        # constants, k1~1.3e5, set a ~1e-5 s timescale). The earlier
        # t_end=10 ran the oscillator ~1e6 cycles, where two independent
        # CVODE integrations drift out of phase and pointwise cross-
        # validation is meaningless -- a property of the oscillator, not
        # an engine bug. Use the model's design horizon.
        "t_end": 0.001,
        "n_steps": 1000,
    },
    {
        "name": "Repressilator",
        "src": "models2",
        "file": "Repressilator.bngl",
        "source": "BNG2 Models2",
        "ref": "Elowitz & Leibler 2000",
        "effort": "low",
        "t_end": 1000,
        "n_steps": 1000,
    },
    {
        "name": "blbr",
        "src": "models2",
        "file": "blbr.bngl",
        "source": "BNG2 Models2",
        "ref": "Bivalent ligand bivalent receptor",
        "effort": "low",
        "t_end": 100,
        "n_steps": 200,
    },
    {
        "name": "egfr_net",
        "src": "models2",
        "file": "egfr_net.bngl",
        "source": "BNG2 Models2",
        "ref": "Blinov et al. 2006",
        "effort": "high",
        "t_end": 120,
        "n_steps": 120,
    },
    {
        "name": "egfr_path",
        "src": "models2",
        "file": "egfr_path.bngl",
        "source": "BNG2 Models2",
        "ref": "EGFR pathway",
        "effort": "low",
        "t_end": 120,
        "n_steps": 120,
    },
    {
        "name": "fceri_ji",
        "src": "models2",
        "file": "fceri_ji.bngl",
        "source": "BNG2 Models2",
        "ref": "Faeder et al. 2003",
        "effort": "high",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "fceri_fyn",
        "src": "models2",
        "file": "fceri_fyn.bngl",
        "source": "BNG2 Models2",
        "ref": "Faeder FceRI+Fyn",
        "effort": "high",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "fceri_fyn_lig",
        "src": "models2",
        "file": "fceri_fyn_lig.bngl",
        "source": "BNG2 Models2",
        "ref": "Faeder FceRI+Fyn+Lig",
        "effort": "high",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "fceri_ji_red",
        "src": "models2",
        "file": "fceri_ji_red.bngl",
        "source": "BNG2 Models2",
        "ref": "Faeder FceRI reduced",
        "effort": "medium",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "fceri_trimer",
        "src": "models2",
        "file": "fceri_trimer.bngl",
        "source": "BNG2 Models2",
        "ref": "Faeder FceRI trimer",
        "effort": "high",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "fceri_gamma2",
        "src": "models2",
        "file": "fceri_gamma2.bngl",
        "source": "BNG2 Models2",
        "ref": "Goldstein et al. 2004",
        "effort": "high",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "fceri_fyn_trimer",
        "src": "models2",
        "file": "fceri_fyn_trimer.bngl",
        "source": "BNG2 Models2",
        "ref": "Faeder FceRI Fyn trimer",
        "effort": "high",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "fceri_lyn_745",
        "src": "models2",
        "file": "fceri_lyn_745.bngl",
        "source": "BNG2 Models2",
        "ref": "Faeder FceRI Lyn (745 sp)",
        "effort": "high",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "egfr_net_red",
        "src": "models2",
        "file": "egfr_net_red.bngl",
        "source": "BNG2 Models2",
        "ref": "Blinov EGFR reduced",
        "effort": "low",
        "t_end": 120,
        "n_steps": 120,
    },
    {
        "name": "Haugh2b",
        "src": "models2",
        "file": "Haugh2b.bngl",
        "source": "BNG2 Models2",
        "ref": "Haugh et al.",
        "effort": "low",
        "t_end": 100,
        "n_steps": 200,
    },
    {
        "name": "localfunc",
        "src": "models2",
        "file": "localfunc.bngl",
        "source": "BNG2 Models2",
        "ref": "Local functions example",
        "effort": "low",
        "t_end": 100,
        "n_steps": 100,
    },
    {
        "name": "nfkb",
        "src": "models2",
        "file": "nfkb.bngl",
        "source": "BNG2 Models2",
        "ref": "NF-kB signaling",
        "effort": "low",
        "t_end": 200,
        "n_steps": 200,
    },
    {
        "name": "SHP2_base_model",
        "src": "models2",
        "file": "SHP2_base_model.bngl",
        "source": "BNG2 Models2",
        "ref": "SHP2 phosphatase",
        "effort": "medium",
        "t_end": 100,
        "n_steps": 200,
    },
    {
        "name": "Motivating_example",
        "src": "models2",
        "file": "Motivating_example.bngl",
        "source": "BNG2 Models2",
        "ref": "BNGL tutorial",
        "effort": "medium",
        "t_end": 100,
        "n_steps": 200,
    },
    {
        "name": "toy-jim",
        "src": "models2",
        "file": "toy-jim.bngl",
        "source": "BNG2 Models2",
        "ref": "Simple toy model",
        "effort": "low",
        "t_end": 100,
        "n_steps": 100,
    },
    # === PyBNF published examples (ground-truth versions, no __FREE) ===
    {
        "name": "egfr_ground",
        "src": "pybnf",
        "file": "egfr_ground.bngl",
        "source": "PyBNF",
        "ref": "Blinov EGFR (Mitra 2019)",
        "effort": "high",
        "t_end": 120,
        "n_steps": 120,
    },
    {
        "name": "RAFi_ground",
        "src": "pybnf",
        "file": "RAFi_ground.bngl",
        "source": "PyBNF",
        "ref": "RAF inhibitor (Mitra 2019)",
        "effort": "low",
        "t_end": 10000,
        "n_steps": 200,
    },
    # === RuleHub published / contributed models ===
    {
        "name": "Dolan2015",
        "src": "rulehub",
        "file": "Dolan2015.bngl",
        "source": "RuleHub",
        "ref": "Dolan et al. 2015",
        "effort": "medium",
        "t_end": 1000,
        "n_steps": 200,
    },
    {
        "name": "Scaff-22_ground",
        "src": "rulehub",
        "file": "Scaff-22_ground.bngl",
        "source": "RuleHub",
        "ref": "MAPK scaffolding (Mitra 2019)",
        "effort": "medium",
        "t_end": 1000,
        "n_steps": 200,
    },
    {
        "name": "m1_ground",
        "src": "rulehub",
        "file": "m1_ground.bngl",
        "source": "RuleHub",
        "ref": "Three-step pathway (Mitra 2019)",
        "effort": "low",
        "t_end": 100,
        "n_steps": 200,
    },
    {
        "name": "example5_ground_truth",
        "src": "rulehub",
        "file": "example5_ground_truth.bngl",
        "source": "RuleHub",
        "ref": "Receptor model (Thomas 2016)",
        "effort": "low",
        "t_end": 100,
        "n_steps": 200,
    },
    {
        "name": "elephant_EFA",
        "src": "rulehub",
        "file": "elephant_EFA.bngl",
        "source": "RuleHub",
        "ref": "Elephant curve fitting",
        "effort": "low",
        "t_end": 10,
        "n_steps": 200,
    },
    {
        "name": "after_bunching",
        "src": "rulehub",
        "file": "after_bunching.bngl",
        "source": "RuleHub",
        "ref": "Hlavacek 2018 restructuration",
        "effort": "medium",
        "t_end": 100,
        "n_steps": 200,
    },
    {
        "name": "before_bunching",
        "src": "rulehub",
        "file": "before_bunching.bngl",
        "source": "RuleHub",
        "ref": "Hlavacek 2018 restructuration",
        "effort": "high",
        "t_end": 100,
        "n_steps": 200,
    },
    # === RuleBender published models (vendored in models/bngl/ode/) ===
    {
        "name": "Kholodenko_2000",
        "src": "ode_bench",
        "file": "Kholodenko_2000.bngl",
        "source": "RuleBender",
        "ref": "Kholodenko 2000 (MAPK cascade)",
        "effort": "low",
        "t_end": 4000,
        "n_steps": 400,
    },
    {
        "name": "SIR",
        "src": "ode_bench",
        "file": "SIR.bngl",
        "source": "RuleBender",
        "ref": "SIR compartmental model",
        "effort": "low",
        "t_end": 100,
        "n_steps": 200,
    },
    {
        "name": "RasRaf_stdBNGL_v3",
        "src": "ode_bench",
        "file": "RasRaf_stdBNGL_v3.bngl",
        "source": "RuleBender",
        "ref": "Ras-Raf standard BNGL",
        "effort": "low",
        "t_end": 1000,
        "n_steps": 200,
    },
    {
        "name": "monoBE_plusInh",
        "src": "ode_bench",
        "file": "monoBE_plusInh.bngl",
        "source": "RuleBender",
        "ref": "BRAF V600E + inhibitor",
        "effort": "low",
        "t_end": 1000,
        "n_steps": 200,
    },
    {
        "name": "CaM_Ca_interaction_v1",
        "src": "ode_bench",
        "file": "CaM_Ca_interaction_v1.bngl",
        "source": "RuleBender",
        "ref": "Calmodulin-calcium binding",
        "effort": "low",
        "t_end": 100,
        "n_steps": 200,
    },
    # === Existing SSA benchmarks (also work for ODE) ===
    {
        "name": "simple_system",
        "src": "ode_bench",
        "file": "simple_system.bngl",
        "source": "SSA bench",
        "ref": "NFsim simple binding",
        "effort": "low",
        "t_end": 10,
        "n_steps": 100,
    },
    {
        "name": "gene_expr_3stage",
        "src": "ode_bench",
        "file": "gene_expr_3stage.bngl",
        "source": "SSA bench",
        "ref": "Shahrezaei & Swain 2008",
        "effort": "low",
        "t_end": 1000,
        "n_steps": 200,
    },
    {
        "name": "tcr_signaling",
        "src": "ode_bench",
        "file": "tcr_signaling.bngl",
        "source": "SSA bench",
        "ref": "Lipniacki et al. 2008",
        "effort": "low",
        "t_end": 300,
        "n_steps": 300,
    },
    {
        "name": "erk_activation",
        "src": "ode_bench",
        "file": "erk_activation.bngl",
        "source": "SSA bench",
        "ref": "Kochanczyk et al. 2017",
        "effort": "low",
        "t_end": 8640,
        "n_steps": 200,
    },
    {
        "name": "prion_aggregation",
        "src": "ode_bench",
        "file": "prion_aggregation.bngl",
        "source": "SSA bench",
        "ref": "Rubenstein et al. 2007",
        "effort": "high",
        "t_end": 10,
        "n_steps": 200,
    },
    {
        "name": "egfr_net_bench",
        "src": "ode_bench",
        "file": "egfr_net.bngl",
        "source": "SSA bench",
        "ref": "Blinov 2006 (356 sp)",
        "effort": "high",
        "t_end": 120,
        "n_steps": 120,
    },
    {
        "name": "multisite_phos",
        "src": "ode_bench",
        "file": "multisite_phos.bngl",
        "source": "SSA bench",
        "ref": "5-site phosphorylation",
        "effort": "high",
        "t_end": 120,
        "n_steps": 120,
    },
    {
        "name": "fceri_gamma_bench",
        "src": "ode_bench",
        "file": "fceri_gamma.bngl",
        "source": "SSA bench",
        "ref": "Goldstein 2004 (3744 sp)",
        "effort": "high",
        "t_end": 200,
        "n_steps": 200,
    },
]
for _m in CURATED_MODELS:
    _m["path"] = str(_SRC_DIR[_m["src"]] / _m["file"])


def generate_net(bngl_path, work_dir):
    """Run BNG2.pl to generate .net file. Returns path to .net file or None."""
    bngl_path = Path(bngl_path)

    # Copy BNGL file to work dir
    dest_bngl = Path(work_dir) / bngl_path.name
    shutil.copy2(bngl_path, dest_bngl)

    # Also copy any .tfun files in the same directory
    for tfun in bngl_path.parent.glob("*.tfun"):
        shutil.copy2(tfun, Path(work_dir) / tfun.name)

    cmd = ["perl", BNG2_PL, str(dest_bngl)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BNG_TIMEOUT,
            cwd=work_dir,
        )
    except subprocess.TimeoutExpired:
        return None, "BNG2.pl timeout"

    # Find .net file
    stem = bngl_path.stem
    net_path = Path(work_dir) / f"{stem}.net"
    if net_path.exists():
        return str(net_path), None

    # Check for errors
    err = (proc.stdout + proc.stderr)[:500]
    return None, f"No .net file produced: {err}"


def count_species_reactions(net_path):
    """Count species and reactions from .net file headers."""
    n_species = 0
    n_reactions = 0
    in_species = False
    in_reactions = False
    try:
        with open(net_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("begin species"):
                    in_species = True
                    continue
                if line.startswith("end species"):
                    in_species = False
                    continue
                if line.startswith("begin reactions"):
                    in_reactions = True
                    continue
                if line.startswith("end reactions"):
                    in_reactions = False
                    continue
                if in_species and line and not line.startswith("#"):
                    n_species += 1
                if in_reactions and line and not line.startswith("#"):
                    n_reactions += 1
    except Exception:
        pass
    return n_species, n_reactions


def run_bngsim_ode(net_path, t_end, n_steps):
    """Run BNGsim ODE and return species trajectory + stats."""
    import bngsim

    try:
        model = bngsim.Model.from_net(net_path)
    except Exception as e:
        return None, f"Load error: {e}"

    try:
        sim = bngsim.Simulator(model, method="ode")
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1)
    except Exception as e:
        return None, f"ODE error: {e}"

    species = np.asarray(result.species)

    # Check for NaN/Inf/negative
    if np.any(np.isnan(species)):
        return species, "NaN in species"
    if np.any(np.isinf(species)):
        return species, "Inf in species"
    if np.any(species < -1e-6):
        min_val = np.min(species)
        return species, f"Negative species (min={min_val:.3g})"

    return species, None


def run_run_network_ode(net_path, t_end, n_steps):
    """Run run_network ODE and return species trajectory."""
    sample_time = t_end / n_steps

    with tempfile.TemporaryDirectory(prefix="ode_rn_") as tmpdir:
        prefix = os.path.join(tmpdir, "out")
        cmd = [
            RUN_NETWORK,
            "-g",
            net_path,
            "-o",
            prefix,
            net_path,
            f"{sample_time:.15g}",
            str(n_steps),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=RN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return None, "run_network timeout"

        if proc.returncode != 0:
            err = (proc.stdout + proc.stderr)[:300]
            return None, f"run_network error: {err}"

        # Parse .cdat file
        cdat = prefix + ".cdat"
        if not os.path.exists(cdat):
            return None, "No .cdat file produced"

        try:
            data = np.loadtxt(cdat, comments="#")
            # First column is time, rest are species
            species = data[:, 1:]
            return species, None
        except Exception as e:
            return None, f"Parse .cdat error: {e}"


def cross_validate(bng_species, rn_species, rtol=1e-5, atol_rel=1e-7):
    """Compare BNGsim vs run_network species trajectories.

    A cell passes when ``|bng - rn| <= atol + rtol * |rn|`` -- numpy's
    ``allclose`` rule, a relative tolerance plus an additive absolute floor.
    The floor ``atol`` is ``atol_rel`` times the trajectory's *peak* species
    magnitude: a species whose value sits many orders of magnitude below
    that peak is negligible to the model's behaviour, so the floor absorbs
    the cross-engine integration-tolerance noise on it. Two independent
    CVODE integrations agree only to their own tolerance floor -- measured
    here at up to ~2e-7 of the trajectory peak -- so without the floor a
    sub-1e-7-of-peak species turns a pure-noise gap into a large *relative*
    error; ``atol_rel`` 1e-7 sits just above that noise. A genuine error on
    a *significant* species still far exceeds ``atol + rtol * |rn|``.

    The reported statistic is ``max |bng - rn| / (atol + rtol * |rn|)`` --
    the worst cell's error as a multiple of its own tolerance; <= 1.0 passes.
    """
    if bng_species is None or rn_species is None:
        return False, "Missing trajectory"

    if bng_species.shape != rn_species.shape:
        return False, f"Shape mismatch: BNGsim {bng_species.shape} vs RN {rn_species.shape}"

    peak = float(np.max(np.abs(rn_species)))
    atol = atol_rel * peak if peak > 0 else atol_rel

    allowed = atol + rtol * np.abs(rn_species)
    tol_ratio = np.abs(bng_species - rn_species) / allowed
    max_ratio = float(np.max(tol_ratio))

    if max_ratio <= 1.0:
        return True, f"max_tol_ratio={max_ratio:.2e} (atol={atol:.2e}, rtol={rtol:.0e})"
    else:
        idx = np.unravel_index(int(np.argmax(tol_ratio)), tol_ratio.shape)
        return False, (
            f"max_tol_ratio={max_ratio:.2e} at t_idx={idx[0]}, sp_idx={idx[1]}: "
            f"BNG={bng_species[idx]:.6g} vs RN={rn_species[idx]:.6g} "
            f"(atol={atol:.2e}, rtol={rtol:.0e})"
        )


def timing_ode(net_path, t_end, n_steps, warmup, runs):
    """Median wall-clock comparison of the two ODE engines for one model.

    Each engine runs ``warmup`` discarded passes then ``runs`` timed passes.
    BNGsim model loading is excluded (clone + reset only, amortized by
    PyBNF); ``run_network`` timing includes the full subprocess overhead --
    the real per-simulation cost. Returns a dict with the two medians and
    their ratio (``speedup``, RN / BNGsim), or an ``error``.
    """
    import bngsim

    try:
        model = bngsim.Model.from_net(net_path)
    except Exception as e:  # noqa: BLE001
        return {"error": f"load: {e}"}

    bng_times = []
    for i in range(warmup + runs):
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ode")
        t0 = time.perf_counter()
        try:
            sim.run(t_span=(0, t_end), n_points=n_steps + 1)
        except Exception as e:  # noqa: BLE001
            return {"error": f"BNGsim ODE: {e}"}
        if i >= warmup:
            bng_times.append(time.perf_counter() - t0)

    rn_times = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        _, err = run_run_network_ode(net_path, t_end, n_steps)
        elapsed = time.perf_counter() - t0
        if err:
            return {"error": f"run_network ODE: {err}"}
        if i >= warmup:
            rn_times.append(elapsed)

    bng_med = median(bng_times)
    rn_med = median(rn_times)
    return {
        "bngsim_median": bng_med,
        "run_network_median": rn_med,
        "bngsim_all": bng_times,
        "run_network_all": rn_times,
        "speedup": (rn_med / bng_med) if (bng_med > 0 and rn_med > 0) else None,
        "error": None,
    }


def generate_markdown(payload, outpath):
    """Render the ODE correctness + timing report from collected results."""
    info = payload["machine_info"]
    lines = ["# ODE suite — BNGsim vs run_network\n", "## Machine\n"]
    for label, key in (
        ("Platform", "platform"),
        ("Processor", "processor"),
        ("Python", "python"),
        ("BNGsim", "bngsim_version"),
        ("run_network", "run_network_version"),
        ("Git commit", "git_commit"),
        ("Date", "date"),
    ):
        lines.append(f"- **{label}**: {info.get(key, 'n/a')}")
    lines.append(
        f"- **Protocol**: mode={payload['mode']}, cross-validation rtol~1e-5, "
        f"timing={payload['warmup']} warmup + {payload['runs']} timed runs, median"
    )
    lines.append("")

    lines.append("## Results\n")
    lines.append(
        "| Model | Source | Species | Rxns | Cross-validation | "
        "BNGsim ODE (s) | run_network ODE (s) | Speedup |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    speedups = []
    for r in payload["results"]:
        name = r["name"]
        sp = r.get("species", "—")
        rxn = r.get("reactions", "—")
        src = r.get("source", "—")
        status = r["status"]

        if status == "skip":
            xval = f"*(skipped: {r.get('reason', '')})*"
        elif status.startswith("fail"):
            xval = f"**{status}**: {str(r.get('error', '')).splitlines()[0][:60]}"
        elif status == "pass_quick":
            xval = "*(quick: BNGsim only)*"
        elif status == "pass_timing_only":
            xval = "*(timing mode, not cross-validated)*"
        elif status == "pass":
            xval = f"PASS ({r.get('cross_validation', '')})"
        else:
            xval = status

        tim = r.get("timing")
        if not tim or tim.get("error"):
            bt = rt = su = "—"
        else:
            bt = f"{tim['bngsim_median']:.4f}"
            rt = f"{tim['run_network_median']:.4f}"
            if tim["speedup"]:
                su = f"**{tim['speedup']:.1f}×**"
                speedups.append(tim["speedup"])
            else:
                su = "n/a"
        lines.append(f"| {name} | {src} | {sp} | {rxn} | {xval} | {bt} | {rt} | {su} |")

    lines.append("")
    geo = nb.geometric_mean(speedups)
    if geo:
        lines.append(f"**Geometric-mean speedup (cross-validated models): {geo:.1f}×**\n")
    outpath.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="ODE correctness + timing sweep")
    parser.add_argument(
        "--mode",
        choices=["correctness", "timing", "both"],
        default="both",
        help="Which gates to run (default: both -- timing only for cross-validated models).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="BNGsim load+run smoke only -- skip run_network, cross-validation and timing.",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Timing warmup runs.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Timing timed runs.")
    add_effort_arg(parser)
    args = parser.parse_args()

    # Cumulative effort threshold: 'low' < 'medium' < 'high' (= all).
    models = filter_by_effort(CURATED_MODELS, args.effort, key=lambda m: m["effort"])

    print("=" * 70)
    print("  ODE suite — BNGsim vs run_network")
    print("=" * 70)
    print(
        f"  mode={args.mode}, effort={args.effort}: {len(models)} of {len(CURATED_MODELS)} models"
    )

    results = []

    for model_cfg in models:
        name = model_cfg["name"]
        bngl_path = model_cfg["path"]
        source = model_cfg["source"]
        ref = model_cfg["ref"]
        t_end = model_cfg["t_end"]
        n_steps = model_cfg["n_steps"]

        print(f"\n--- {name} ({source}: {ref}) ---")

        if not os.path.exists(bngl_path):
            print(f"  SKIP: File not found: {bngl_path}")
            results.append({"name": name, "status": "skip", "reason": "file not found"})
            continue

        with tempfile.TemporaryDirectory(prefix=f"ode_{name}_") as tmpdir:
            # Step 1: generate .net
            net_path, err = generate_net(bngl_path, tmpdir)
            if err:
                print(f"  FAIL: .net generation: {err}")
                results.append({"name": name, "status": "fail_gennet", "error": err})
                continue

            n_sp, n_rxn = count_species_reactions(net_path)
            print(f"  .net: {n_sp} species, {n_rxn} reactions")

            # Step 2: BNGsim ODE
            bng_species, err = run_bngsim_ode(net_path, t_end, n_steps)
            if err:
                print(f"  FAIL: BNGsim ODE: {err}")
                results.append(
                    {
                        "name": name,
                        "status": "fail_bngsim",
                        "error": err,
                        "species": n_sp,
                        "reactions": n_rxn,
                    }
                )
                continue
            print(f"  BNGsim ODE: OK ({bng_species.shape})")

            if args.quick:
                print("  PASS (quick mode, no cross-validation)")
                results.append(
                    {
                        "name": name,
                        "status": "pass_quick",
                        "species": n_sp,
                        "reactions": n_rxn,
                        "source": source,
                        "ref": ref,
                        "t_end": t_end,
                        "n_steps": n_steps,
                    }
                )
                continue

            row = {
                "name": name,
                "species": n_sp,
                "reactions": n_rxn,
                "source": source,
                "ref": ref,
                "t_end": t_end,
                "n_steps": n_steps,
            }

            # Step 3: correctness gate -- cross-validate vs run_network
            xval_ok = True
            if args.mode in ("correctness", "both"):
                rn_species, err = run_run_network_ode(net_path, t_end, n_steps)
                if err:
                    print(f"  FAIL: run_network ODE: {err}")
                    results.append(
                        {
                            "name": name,
                            "status": "fail_rn",
                            "error": err,
                            "species": n_sp,
                            "reactions": n_rxn,
                        }
                    )
                    continue
                print(f"  run_network ODE: OK ({rn_species.shape})")
                passed, detail = cross_validate(bng_species, rn_species)
                row["cross_validation"] = detail
                if passed:
                    print(f"  PASS: Cross-validated ({detail})")
                    row["status"] = "pass"
                else:
                    print(f"  FAIL: Cross-validation: {detail}")
                    row["status"] = "fail_xval"
                    row["error"] = detail
                    xval_ok = False
            else:
                row["status"] = "pass_timing_only"

            # Step 4: timing gate -- only for cross-validated models
            if args.mode == "timing" or (args.mode == "both" and xval_ok):
                tim = timing_ode(net_path, t_end, n_steps, args.warmup, args.runs)
                row["timing"] = tim
                if tim.get("error"):
                    print(f"  timing: ERROR — {tim['error'][:100]}")
                else:
                    su = f"{tim['speedup']:.1f}×" if tim["speedup"] else "n/a"
                    print(
                        f"  timing: BNGsim {tim['bngsim_median']:.4f}s | "
                        f"run_network {tim['run_network_median']:.4f}s | speedup {su}"
                    )
            elif args.mode == "both" and not xval_ok:
                print("  timing: skipped (cross-validation did not pass)")

            results.append(row)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    pass_count = sum(1 for r in results if r["status"].startswith("pass"))
    fail_count = sum(1 for r in results if r["status"].startswith("fail"))
    skip_count = sum(1 for r in results if r["status"] == "skip")

    print(f"  Total: {len(results)}")
    print(f"  PASS: {pass_count}")
    print(f"  FAIL: {fail_count}")
    print(f"  SKIP: {skip_count}")

    if fail_count > 0:
        print("\n  Failures:")
        for r in results:
            if r["status"].startswith("fail"):
                print(f"    {r['name']}: {r['status']} — {r.get('error', 'unknown')[:100]}")

    # Save results
    payload = {
        "machine_info": nb.machine_info(),
        "mode": args.mode,
        "warmup": args.warmup,
        "runs": args.runs,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "ode_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    generate_markdown(payload, RESULTS_DIR / "ode_results.md")
    print(f"\nResults: {json_path}")
    print(f"Report:  {RESULTS_DIR / 'ode_results.md'}")


if __name__ == "__main__":
    main()
