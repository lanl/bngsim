"""Shared utilities for BNGsim benchmark harnesses.

Provides timing, comparison, result I/O, reporting utilities, and
shared CLI helpers used across validation and comparison harnesses.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HARNESS_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = HARNESS_DIR.parent / "benchmarks"
RESULTS_DIR = HARNESS_DIR / "results"
REPO_ROOT = HARNESS_DIR.parent.parent

# External tool paths
_DEFAULT_BNGPATH = REPO_ROOT / "BioNetGen-2.9.3"
_BNGPATH = os.environ.get(
    "BNGPATH",
    str(_DEFAULT_BNGPATH)
    if _DEFAULT_BNGPATH.exists()
    else os.path.expanduser("~/Simulations/BioNetGen-2.9.3"),
)
BNG2_PL = os.environ.get("BNG2_PL", str(Path(_BNGPATH) / "BNG2.pl"))
RUN_NETWORK = os.environ.get(
    "RUN_NETWORK",
    str(Path(_BNGPATH) / "bin" / "run_network"),
)
NFSIM_BIN = os.environ.get(
    "NFSIM",
    str(Path(_BNGPATH) / "bin" / "NFsim"),
)

# Model pools
SSYS_ANT_DIR = BENCHMARKS_DIR / "ant"  # Pool A: ~117 ssys Antimony
BIOMODELS_ANT_DIR = Path(
    os.environ.get("BIOMODELS_ANT_DIR", str(BENCHMARKS_DIR / "biomodels_ant"))
)
COMMENTARY_RESULTS = Path(
    os.environ.get(
        "COMMENTARY_RESULTS", str(REPO_ROOT / "bngsim" / "benchmarks" / "commentary_results.json")
    )
)
ODE_NET_DIR = BENCHMARKS_DIR / "ode"  # Pool C: .net files
SSA_NET_DIR = BENCHMARKS_DIR / "ssa"
PSA_NET_DIR = BENCHMARKS_DIR / "psa"
NF_DIR = BENCHMARKS_DIR / "nf"

# Suite manifests. These ODE/SSA/PSA timing manifests moved into
# benchmarks/_dev/ when run_benchmark.py was retired and the suites/
# reorg landed; the per-suite correctness runners live in
# benchmarks/suites/{ode,ssa,psa}/.
SUITE_ODE = BENCHMARKS_DIR / "_dev" / "suite_ode.json"
SUITE_SSA = BENCHMARKS_DIR / "_dev" / "suite_ssa.json"
SUITE_PSA = BENCHMARKS_DIR / "_dev" / "suite_psa.json"

# Timeouts
BNG_TIMEOUT = 120
RN_TIMEOUT = 600
BNGSIM_TIMEOUT = 300
DEFAULT_BNGSIM_ODE_TIMEOUT = 120.0

# Default benchmark protocol
DEFAULT_WARMUP = 2
DEFAULT_RUNS = 5


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def normalize_bngsim_timeout(timeout):
    """Normalize a BNGsim wall-clock timeout to ``float | None``.

    ``None`` and non-positive values disable the guard, matching
    ``bngsim.Simulator.run(timeout=...)`` semantics. String values may
    use ``none`` / ``off`` / ``disable`` to disable the guard.
    """
    if timeout is None:
        return None
    if isinstance(timeout, str):
        text = timeout.strip().lower()
        if text in ("none", "off", "disable", "disabled", "no"):
            return None
        timeout = float(text)
    else:
        timeout = float(timeout)
    return timeout if timeout > 0.0 else None


def _argparse_bngsim_timeout(value: str):
    try:
        return normalize_bngsim_timeout(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive number of seconds, or 'none'/'off' to disable"
        ) from exc


def add_bngsim_timeout_arg(parser, *, default=DEFAULT_BNGSIM_ODE_TIMEOUT):
    """Add a shared ``--bngsim-timeout`` CLI flag to a benchmark parser."""
    default_timeout = normalize_bngsim_timeout(default)
    default_label = "disabled" if default_timeout is None else f"{default_timeout:g} s"
    parser.add_argument(
        "--bngsim-timeout",
        type=_argparse_bngsim_timeout,
        default=default_timeout,
        metavar="SECONDS|none",
        help=(
            "Wall-clock guard for BNGsim ODE sim.run() calls. "
            f"Default: {default_label}. "
            "Use 'none', 'off', or any value <= 0 to disable."
        ),
    )


def format_bngsim_timeout(timeout) -> str:
    """Return a human-readable label for a normalized timeout value."""
    timeout = normalize_bngsim_timeout(timeout)
    return "disabled" if timeout is None else f"{timeout:g}s"


# ---------------------------------------------------------------------------
# Machine info
# ---------------------------------------------------------------------------


def get_machine_info() -> dict:
    """Collect machine specs for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    try:
        import bngsim

        info["bngsim_version"] = getattr(bngsim, "__version__", "unknown")
    except ImportError:
        info["bngsim_version"] = "not installed"
    try:
        import amici

        info["amici_version"] = getattr(amici, "__version__", "unknown")
    except ImportError:
        info["amici_version"] = "not installed"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=HARNESS_DIR.parent,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            cwd=HARNESS_DIR.parent,
            timeout=5,
        )
        # 0 = clean, 1 = dirty
        info["git_dirty"] = result.returncode != 0
    except Exception:
        pass
    try:
        result = subprocess.run(
            [RUN_NETWORK],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version_line = (result.stdout + result.stderr).split("\n")[0]
        info["run_network_version"] = version_line.strip()
    except Exception:
        info["run_network_version"] = "unknown"
    return info


# ---------------------------------------------------------------------------
# Model pool discovery
# ---------------------------------------------------------------------------


def load_suite(suite_path: Path) -> list[dict]:
    """Load a suite manifest JSON file."""
    with open(suite_path) as f:
        data = json.load(f)
    return data["models"]


def discover_pool_a() -> list[Path]:
    """Discover Pool A: ssys Antimony models (cross-validated S25)."""
    if not SSYS_ANT_DIR.exists():
        return []
    models = sorted(SSYS_ANT_DIR.glob("*.ant"))
    return models


def discover_pool_b(require_xval: bool = True) -> list[Path]:
    """Discover Pool B: BioModels Antimony models.

    Includes top-level ``*.ant`` plus ``review_extra/*.ant`` when
    present (manifest materialization layout).

    Args:
        require_xval: If True, only include models cross-validated in
            prior benchmark results. If False, include all discovered models.
    """
    if not BIOMODELS_ANT_DIR.exists():
        return []

    top = sorted(BIOMODELS_ANT_DIR.glob("*.ant"))
    rex = BIOMODELS_ANT_DIR / "review_extra"
    extra = sorted(rex.glob("*.ant")) if rex.is_dir() else []
    all_models = sorted({*top, *extra}, key=lambda p: p.name)

    if not require_xval or not COMMENTARY_RESULTS.exists():
        return all_models

    # Filter to cross-validated models
    with open(COMMENTARY_RESULTS) as f:
        data = json.load(f)

    xval_ids = set()
    for entry in data.get("results", []):
        if entry.get("xval"):
            xval_ids.add(entry["model"])

    return [p for p in all_models if p.stem in xval_ids]


def discover_pool_c() -> list[dict]:
    """Discover Pool C: BNG ODE models with curated t_end from suite_ode.json."""
    return load_suite(SUITE_ODE)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def timed_runs(
    run_fn,
    n_warmup: int = DEFAULT_WARMUP,
    n_runs: int = DEFAULT_RUNS,
    verbose: bool = False,
) -> dict:
    """Execute run_fn multiple times and collect timing statistics.

    Args:
        run_fn: Callable that returns a dict with at least 'wall_time' key.
            May also return 'error', 'species', 'n_steps', etc.
        n_warmup: Number of warmup runs (not counted).
        n_runs: Number of timed runs.
        verbose: Print per-run times.

    Returns:
        dict with 'median_time', 'all_times', 'min_time', 'max_time',
        and any extra keys from the first timed run.
    """
    all_times = []
    first_result = None

    for i in range(n_warmup + n_runs):
        result = run_fn()
        if "error" in result:
            return {"error": result["error"], "median_time": -1}

        label = "warmup" if i < n_warmup else "timed"
        if verbose:
            print(f"    [{label}] {result['wall_time']:.4f}s")

        if i >= n_warmup:
            all_times.append(result["wall_time"])
            if first_result is None:
                first_result = result

    return {
        "median_time": median(all_times),
        "min_time": min(all_times),
        "max_time": max(all_times),
        "all_times": all_times,
        **(first_result or {}),
    }


# ---------------------------------------------------------------------------
# BNGsim runners
# ---------------------------------------------------------------------------


def run_bngsim_ode(
    net_path: str,
    t_end: float,
    n_steps: int,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict:
    """Run BNGsim ODE on a .net file. Returns timing + trajectory.

    ``rtol``/``atol`` default to ``None`` (BNGsim's internal CVODE defaults,
    1e-8/1e-8). Callers running a cross-engine fairness comparison should pass
    explicit tolerances so every engine integrates at the same accuracy.
    """
    import bngsim

    try:
        model = bngsim.Model.from_net(net_path)
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ode")

        t0 = time.perf_counter()
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, rtol=rtol, atol=atol)
        elapsed = time.perf_counter() - t0

        species = np.asarray(result.species)
        stats = result.solver_stats

        return {
            "wall_time": elapsed,
            "n_cvode_steps": stats.get("n_steps", 0),
            "n_rhs_evals": stats.get("n_rhs_evals", 0),
            "species": species,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


def run_bngsim_ode_antimony(ant_path: str, t_end: float, n_steps: int) -> dict:
    """Run BNGsim ODE on an Antimony file. Returns timing + trajectory."""
    import bngsim

    try:
        model = bngsim.Model.from_antimony(ant_path)
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ode")

        t0 = time.perf_counter()
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1)
        elapsed = time.perf_counter() - t0

        species = np.asarray(result.species)
        species_names = list(result.species_names)

        return {
            "wall_time": elapsed,
            "species": species,
            "species_names": species_names,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


def run_bngsim_ssa(net_path: str, t_end: float, n_steps: int, seed: int) -> dict:
    """Run BNGsim SSA. Returns timing + trajectory."""
    import bngsim

    try:
        model = bngsim.Model.from_net(net_path)
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ssa")

        t0 = time.perf_counter()
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
        elapsed = time.perf_counter() - t0

        species = np.asarray(result.species)
        stats = result.solver_stats

        return {
            "wall_time": elapsed,
            "n_steps": stats.get("n_steps", 0),
            "species": species,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


def run_bngsim_ssa_sbml(xml_path: str, t_end: float, n_steps: int, seed: int) -> dict:
    """Run BNGsim SSA on an SBML file. Returns timing + trajectory + species names.

    Mirrors ``run_bngsim_ssa`` but loads via ``Model.from_sbml`` and surfaces
    ``species_names`` so callers can join trajectories by SBML id rather than
    column index (DSMTS expected-value CSVs are per-name).
    """
    import bngsim

    try:
        model = bngsim.Model.from_sbml(xml_path)
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="ssa")

        t0 = time.perf_counter()
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
        elapsed = time.perf_counter() - t0

        species = np.asarray(result.species)
        species_names = list(result.species_names)
        stats = result.solver_stats

        return {
            "wall_time": elapsed,
            "n_steps": stats.get("n_steps", 0),
            "species": species,
            "species_names": species_names,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


def run_bngsim_psa(net_path: str, t_end: float, n_steps: int, seed: int, poplevel: float) -> dict:
    """Run BNGsim PSA. Returns timing + trajectory."""
    import bngsim

    try:
        model = bngsim.Model.from_net(net_path)
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="psa", poplevel=poplevel)

        t0 = time.perf_counter()
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
        elapsed = time.perf_counter() - t0

        species = np.asarray(result.species)
        stats = result.solver_stats

        return {
            "wall_time": elapsed,
            "n_steps": stats.get("n_steps", 0),
            "species": species,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


def run_bngsim_psa_sbml(
    xml_path: str, t_end: float, n_steps: int, seed: int, poplevel: float
) -> dict:
    """Run BNGsim PSA on an SBML file. Returns timing + trajectory + species names.

    Mirrors ``run_bngsim_psa`` but loads via ``Model.from_sbml`` and surfaces
    ``species_names`` (matching ``run_bngsim_ssa_sbml``). The SBML loader's
    SSA validation gate runs at construct time (shared by ``ssa`` and ``psa``).
    """
    import bngsim

    try:
        model = bngsim.Model.from_sbml(xml_path)
        m = model.clone()
        m.reset()
        sim = bngsim.Simulator(m, method="psa", poplevel=poplevel)

        t0 = time.perf_counter()
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
        elapsed = time.perf_counter() - t0

        species = np.asarray(result.species)
        species_names = list(result.species_names)
        stats = result.solver_stats

        return {
            "wall_time": elapsed,
            "n_steps": stats.get("n_steps", 0),
            "species": species,
            "species_names": species_names,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


def run_bngsim_nfsim(
    xml_path: str,
    t_end: float,
    n_steps: int,
    seed: int,
    gml: int = 1000000,
) -> dict:
    """Run BNGsim NFsim. Returns timing + observables."""
    import bngsim

    try:
        # Minimal dummy model for NfsimSimulator
        dummy_net = Path(xml_path).parent / "_dummy.net"
        if not dummy_net.exists():
            dummy_net.write_text(
                "begin parameters\n  1 k 1\nend parameters\n"
                "begin species\n  1 A() 1\nend species\n"
                "begin reactions\nend reactions\n"
                "begin groups\n  1 A 1\nend groups\n"
            )
        model = bngsim.Model.from_net(str(dummy_net))
        sim = bngsim.Simulator(
            model,
            method="nfsim",
            xml_path=str(xml_path),
            gml=gml,
        )

        t0 = time.perf_counter()
        result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
        elapsed = time.perf_counter() - t0

        obs_names = list(result.observable_names)
        obs_data = np.asarray(result.observables)
        times = np.asarray(result.time)

        return {
            "wall_time": elapsed,
            "obs_names": obs_names,
            "obs_data": obs_data,
            "times": times,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


# ---------------------------------------------------------------------------
# run_network runners
# ---------------------------------------------------------------------------


def run_rn_ode(
    net_path: str,
    t_end: float,
    n_steps: int,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict:
    """Run run_network ODE. Returns timing + trajectory from .cdat.

    When ``rtol``/``atol`` are given they are passed to run_network as ``-r``/
    ``-a`` so the subprocess integrates at the same tolerances as the in-process
    engines (default integrator is CVODE). ``None`` preserves run_network's
    built-in defaults.
    """
    sample_time = t_end / n_steps

    with tempfile.TemporaryDirectory(prefix="rn_ode_") as tmpdir:
        prefix = os.path.join(tmpdir, "out")
        cmd = [RUN_NETWORK, "-g", net_path, "-o", prefix]
        if atol is not None:
            cmd += ["-a", f"{float(atol):.15g}"]
        if rtol is not None:
            cmd += ["-r", f"{float(rtol):.15g}"]
        cmd += [
            net_path,
            f"{sample_time:.15g}",
            str(n_steps),
        ]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=RN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"wall_time": RN_TIMEOUT, "error": "timeout"}
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            err = (proc.stdout + proc.stderr)[:300]
            return {"wall_time": elapsed, "error": err}

        cdat = prefix + ".cdat"
        if not os.path.exists(cdat):
            return {"wall_time": elapsed, "error": "No .cdat file"}

        try:
            data = np.loadtxt(cdat, comments="#")
            species = data[:, 1:]  # skip time column
            return {"wall_time": elapsed, "species": species}
        except Exception as e:
            return {"wall_time": elapsed, "error": f"Parse error: {e}"}


def run_rn_ssa(net_path: str, t_end: float, n_steps: int, seed: int) -> dict:
    """Run run_network SSA. Returns timing + step count."""
    import re

    sample_time = t_end / n_steps

    with tempfile.TemporaryDirectory(prefix="rn_ssa_") as tmpdir:
        prefix = os.path.join(tmpdir, "out")
        cmd = [
            RUN_NETWORK,
            "-p",
            "ssa",
            "-h",
            str(seed),
            "-g",
            net_path,
            "-o",
            prefix,
            net_path,
            f"{sample_time:.15g}",
            str(n_steps),
        ]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=RN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"wall_time": RN_TIMEOUT, "error": "timeout"}
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            err = (proc.stdout + proc.stderr)[:300]
            return {"wall_time": elapsed, "error": err}

        output = proc.stdout + proc.stderr
        steps = -1
        m = re.search(r"TOTAL STEPS:\s*(\d+)", output)
        if m:
            steps = int(m.group(1))

        return {
            "wall_time": elapsed,
            "n_steps": steps,
        }


def run_rn_psa(net_path: str, t_end: float, n_steps: int, seed: int, poplevel: float) -> dict:
    """Run run_network PSA. Returns timing + step count."""
    import re

    sample_time = t_end / n_steps

    with tempfile.TemporaryDirectory(prefix="rn_psa_") as tmpdir:
        prefix = os.path.join(tmpdir, "out")
        cmd = [
            RUN_NETWORK,
            "-p",
            "ssa",
            "--poplevel",
            str(int(poplevel)),
            "-h",
            str(seed),
            "-g",
            net_path,
            "-o",
            prefix,
            net_path,
            f"{sample_time:.15g}",
            str(n_steps),
        ]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=RN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"wall_time": RN_TIMEOUT, "error": "timeout"}
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            err = (proc.stdout + proc.stderr)[:300]
            return {"wall_time": elapsed, "error": err}

        output = proc.stdout + proc.stderr
        steps = -1
        m = re.search(r"TOTAL STEPS:\s*(\d+)", output)
        if m:
            steps = int(m.group(1))

        return {
            "wall_time": elapsed,
            "n_steps": steps,
        }


# ---------------------------------------------------------------------------
# libRoadRunner runner
# ---------------------------------------------------------------------------


def _ant_to_sbml(ant_path: str) -> str:
    """Convert Antimony file to SBML string via libantimony."""
    import antimony

    antimony.clearPreviousLoads()
    rc = antimony.loadAntimonyFile(str(ant_path))
    if rc < 0:
        raise RuntimeError(antimony.getLastError())
    mod_name = antimony.getModuleNames()[-1]
    sbml = antimony.getSBMLString(mod_name)
    if not sbml:
        raise RuntimeError("Empty SBML from antimony")
    return sbml


def run_roadrunner_ode(ant_path: str, t_end: float, n_steps: int) -> dict:
    """Run libRoadRunner ODE on an Antimony file. Returns timing + trajectory."""
    try:
        import roadrunner
    except ImportError:
        return {"wall_time": -1, "error": "roadrunner not installed"}

    try:
        sbml = _ant_to_sbml(ant_path)
        rr = roadrunner.RoadRunner(sbml)
        rr.integrator.absolute_tolerance = 1e-12
        rr.integrator.relative_tolerance = 1e-8

        t0 = time.perf_counter()
        result = rr.simulate(0, t_end, n_steps + 1)
        elapsed = time.perf_counter() - t0

        data = np.array(result)
        species_names = [c[1:-1] if c.startswith("[") else c for c in result.colnames[1:]]

        return {
            "wall_time": elapsed,
            "species": data[:, 1:],
            "species_names": species_names,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


def run_roadrunner_ode_sbml(
    sbml_path: str,
    t_end: float,
    n_steps: int,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict:
    """Run libRoadRunner ODE on an SBML file.

    ``rtol``/``atol`` default to ``None``, which keeps the harness's historical
    RoadRunner settings (rel-tol 1e-8, abs-tol 1e-12; note RoadRunner's own CVODE
    rel-tol default is 1e-6). Pass explicit tolerances to match the other engines
    in a cross-engine fairness comparison.
    """
    try:
        import roadrunner
    except ImportError:
        return {"wall_time": -1, "error": "roadrunner not installed"}

    try:
        rr = roadrunner.RoadRunner(str(sbml_path))
        rr.integrator.absolute_tolerance = 1e-12 if atol is None else float(atol)
        rr.integrator.relative_tolerance = 1e-8 if rtol is None else float(rtol)

        t0 = time.perf_counter()
        result = rr.simulate(0, t_end, n_steps + 1)
        elapsed = time.perf_counter() - t0

        data = np.array(result)
        species_names = [c[1:-1] if c.startswith("[") else c for c in result.colnames[1:]]

        return {
            "wall_time": elapsed,
            "species": data[:, 1:],
            "species_names": species_names,
        }
    except Exception as e:
        return {"wall_time": -1, "error": str(e)[:300]}


# ---------------------------------------------------------------------------
# Cross-validation utilities
# ---------------------------------------------------------------------------


def cross_validate_trajectories(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    near_zero_frac: float = 0.0,
) -> tuple[bool, float, str]:
    """Compare two species trajectory arrays.

    ``near_zero_frac`` (default 0.0 = strict legacy per-point relative gate)
    raises the absolute-tolerance floor to ``near_zero_frac * max(|traj_b|)`` and
    judges agreement with a mixed absolute/relative criterion
    (``|a-b| <= atol_eff + rtol*|b|``). This stops species whose magnitude is
    negligible relative to the trajectory scale (numerical noise floor) from
    dominating the relative-error metric with spurious large values — e.g. a
    species at 2e-9 vs 4e-7 in a trajectory whose meaningful species are O(100).
    Callers needing the strict gate (e.g. validation) leave it at 0.

    Returns:
        (passed, max_rel_err, detail_string)
    """
    if traj_a is None or traj_b is None:
        return False, float("inf"), "Missing trajectory"

    if traj_a.shape != traj_b.shape:
        return False, float("inf"), (f"Shape mismatch: {traj_a.shape} vs {traj_b.shape}")

    abs_diff = np.abs(traj_a - traj_b)
    abs_ref = np.abs(traj_b)
    atol_eff = atol
    if near_zero_frac > 0 and traj_b.size:
        atol_eff = max(atol, float(near_zero_frac) * float(np.max(abs_ref)))

    denom = np.maximum(abs_ref, atol_eff)
    rel_err = abs_diff / denom
    max_rel_err = float(np.max(rel_err)) if rel_err.size else 0.0

    if near_zero_frac > 0:
        # Mixed absolute/relative pass test (scale-aware floor).
        passed = bool(np.all(abs_diff <= atol_eff + rtol * abs_ref))
    else:
        passed = max_rel_err <= rtol

    if passed:
        return True, max_rel_err, f"max_rel_err={max_rel_err:.2e}"
    idx = np.unravel_index(np.argmax(rel_err), rel_err.shape)
    detail = (
        f"max_rel_err={max_rel_err:.2e} at t_idx={idx[0]}, "
        f"sp_idx={idx[1]}: A={traj_a[idx]:.6g} vs B={traj_b[idx]:.6g}"
    )
    return False, max_rel_err, detail


def cross_validate_by_name(
    names_a: list[str],
    traj_a: np.ndarray,
    names_b: list[str],
    traj_b: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-8,
    near_zero_threshold: float = 1e-8,
) -> tuple[bool, float, int, str]:
    """Cross-validate trajectories by matching species names.

    Uses near-zero masking: when both trajectories are below
    ``near_zero_threshold`` at a time point, absolute error is used
    instead of relative error to avoid inflating errors on
    effectively-zero trajectories.

    Returns:
        (passed, max_rel_err, n_matched, detail_string)
    """
    if traj_a is None or traj_b is None:
        return False, float("inf"), 0, "Missing trajectory"

    # Build name→column index maps
    map_a = {n: i for i, n in enumerate(names_a)}
    map_b = {n: i for i, n in enumerate(names_b)}

    common = sorted(set(map_a) & set(map_b))
    if not common:
        return False, float("inf"), 0, "No common species names"

    n_times = min(traj_a.shape[0], traj_b.shape[0])
    max_rel_err = 0.0
    worst = ""

    for name in common:
        a_col = traj_a[:n_times, map_a[name]]
        b_col = traj_b[:n_times, map_b[name]]

        abs_diff = np.abs(a_col - b_col)
        abs_ref = np.abs(b_col)
        denom = np.maximum(abs_ref, atol)
        raw_rel = abs_diff / denom

        # Near-zero masking: if both values are tiny, use absolute
        # error to avoid inflating errors on effectively-zero species
        both_tiny = (np.abs(a_col) < near_zero_threshold) & (abs_ref < near_zero_threshold)
        masked_rel = np.where(both_tiny, abs_diff, raw_rel)
        max_re = float(np.max(masked_rel))

        if max_re > max_rel_err:
            max_rel_err = max_re
            worst = name

    passed = max_rel_err <= rtol
    detail = (
        f"max_rel_err={max_rel_err:.2e} ({worst}), "
        f"{len(common)} matched of {len(names_a)}/{len(names_b)}"
    )
    return passed, max_rel_err, len(common), detail


def check_sanity(species: np.ndarray) -> tuple[bool, str]:
    """Check trajectory for NaN/Inf/large-negative values."""
    if np.any(np.isnan(species)):
        return False, "NaN detected"
    if np.any(np.isinf(species)):
        return False, "Inf detected"
    if np.any(species < -0.5):
        return False, f"Negative species (min={np.min(species):.3g})"
    return True, "OK"


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def ensure_results_dir():
    """Create results directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(data: dict, name: str):
    """Save results as JSON to the results directory."""
    ensure_results_dir()
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"\n✅ Results saved to {path}")
    return path


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def load_results(name: str) -> dict:
    """Load results JSON from the results directory."""
    path = RESULTS_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def markdown_header(title: str, info: dict) -> list[str]:
    """Generate standard markdown header with machine info."""
    lines = [
        f"# {title}\n",
        "## Machine Specs\n",
        f"- **Platform**: {info.get('platform', 'n/a')}",
        f"- **Processor**: {info.get('processor', 'n/a')}",
        f"- **CPU count**: {info.get('cpu_count', 'n/a')}",
        f"- **Python**: {info.get('python', 'n/a')}",
        f"- **BNGsim version**: {info.get('bngsim_version', 'n/a')}"
        + (" (working tree dirty)" if info.get("git_dirty") else "")
        + (f" @ {info['git_commit']}" if info.get("git_commit") else ""),
        f"- **AMICI version**: {info.get('amici_version', 'n/a')}",
        f"- **run_network**: {info.get('run_network_version', 'n/a')}",
        f"- **Git commit**: {info.get('git_commit', 'n/a')}",
        f"- **Date**: {info.get('date', 'n/a')}",
        "",
    ]
    return lines


def geometric_mean(values: list[float]) -> float:
    """Compute geometric mean of positive values."""
    if not values:
        return 0.0
    return float(np.exp(np.mean(np.log(values))))


def prepare_amici_runtime():
    """Import amici and its low-level SUNDIALS-binding submodule.

    Returns (amici, amici_sundials).  The benchmark script uses ``hasattr``
    ladders on both objects to support multiple AMICI API generations:
    high-level snake_case (``model.create_solver``,
    ``amici.runAmiciSimulation``) and low-level SWIG names
    (``amici_sundials.SensitivityMethod_forward``,
    ``amici_sundials.run_simulation``).  In AMICI 1.0.x the SWIG layer
    lives at ``amici.sim.sundials``.
    """
    import amici
    import amici.sim.sundials as amici_sundials

    return amici, amici_sundials


# ---------------------------------------------------------------------------
# AMICI ODE runner (AMICI >= 1.0 snake_case API)
#
# AMICI 1.0.0 (2026-02-26) renamed the whole Python API from camelCase to
# snake_case: runAmiciSimulation -> run_simulation, getSolver -> create_solver,
# getStateIds -> get_state_ids, and added Model.simulate() as a convenience
# entry point. The comparison benchmarks were written against AMICI 0.x, so
# every old call now raises AttributeError; when that was swallowed by a broad
# ``except Exception`` the AMICI column silently went empty (GH #227). These
# helpers drive the current API and FAIL LOUD on drift — a missing method
# raises AmiciApiError (NOT a RuntimeError), so it is never mistaken for a
# per-model integration failure and can propagate past the callers' error-row
# handlers to crash visibly the next time AMICI's API moves.
# ---------------------------------------------------------------------------


class AmiciApiError(Exception):
    """Installed AMICI lacks the snake_case Python API these benchmarks target.

    Deliberately NOT a ``RuntimeError`` subclass: callers catch ``RuntimeError``
    to record an honest per-model error row, but an API drift must escape that
    net and crash loudly (the GH #227 failure mode was a swallowed AttributeError
    emptying the AMICI column without a trace).
    """


def _amici_require(obj, attr, what):
    if not hasattr(obj, attr):
        raise AmiciApiError(
            f"AMICI {what} has no '{attr}': the installed AMICI Python API has "
            f"drifted from the snake_case surface (AMICI >= 1.0) these benchmarks "
            f"expect. Update the driver in harness/common.py. See GH #227."
        )


def amici_prepare_model(amici_module, *, atol=1e-12, rtol=1e-8):
    """Return a configured ``(model, solver)`` from a compiled AMICI model module.

    ``amici_module`` is the object returned by ``amici.import_model_module(...)``.
    Uses the AMICI >= 1.0 snake_case API (``get_model`` / ``create_solver`` /
    ``set_absolute_tolerance``). Fail-loud on API drift (GH #227).
    """
    _amici_require(amici_module, "get_model", "model module")
    model = amici_module.get_model()
    _amici_require(model, "create_solver", "model")
    solver = model.create_solver()
    _amici_require(solver, "set_absolute_tolerance", "solver")
    solver.set_absolute_tolerance(atol)
    solver.set_relative_tolerance(rtol)
    return model, solver


def amici_simulate(model, solver, tspan):
    """Run one AMICI CVODES simulation over ``tspan`` on a prepared model/solver.

    Returns ``(rdata, elapsed_sec)``, where ``elapsed_sec`` times ONLY the
    ``model.simulate()`` call (matching the old benchmarks, which excluded the
    get_model/get_solver setup from the timed region). Fail-loud on API drift
    (GH #227); raises ``RuntimeError`` on a non-success CVODES status so callers
    can catch it and record an honest error row without masking API drift.
    """
    _amici_require(model, "set_timepoints", "model")
    _amici_require(model, "simulate", "model")
    model.set_timepoints(np.asarray(tspan, dtype=float))
    t0 = time.perf_counter()
    rdata = model.simulate(solver=solver)
    elapsed = time.perf_counter() - t0
    if int(rdata.status) != 0:
        raise RuntimeError(f"AMICI integration failed (status {int(rdata.status)})")
    return rdata, elapsed


def amici_state_names(model):
    """State (species) ids of a compiled AMICI model via the snake_case
    ``get_state_ids`` (AMICI >= 1.0). Fail-loud on API drift (GH #227)."""
    _amici_require(model, "get_state_ids", "model")
    return list(model.get_state_ids())
