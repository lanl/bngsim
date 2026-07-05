#!/usr/bin/env python3
"""``python_ssa`` suite runner — pure-Python SSA workflow comparison.

Promoted from the SSA half of ``harness/comparison/bench_pythonic_workflows.py``
(paper Table S6). Compares BNGsim's in-process exact-SSA (Gillespie) engine
against **gillespy2** — a pure-Python SSA package — across two
model-construction paths:

  * **ModelBuilder** — the model is defined entirely in Python (BNGsim's
    ``ModelBuilder`` API / a hand-built gillespy2 ``Model``), no files.
  * **.net reader** — the model is parsed from a BNG ``.net`` file by
    BNGsim's universal ``_net_reader``; the same parsed structure is
    handed to both engines.

This is a deliberately small suite — gillespy2's pure-Python SSA is only
feasible on small networks, so the corpus is the single 4-species
``simple_system`` model (cf. the larger SSA corpus exercised by the
``ssa`` suite against ``run_network``).

Two gates per (path):

1. correctness — an ensemble of replicate trajectories is simulated by
   each engine and the two ensemble means are compared cell-by-cell with
   a two-sample *z*-test (``_netbench.zscore_gate``). Both engines run
   exact SSA, so the means must agree within stochastic error — the same
   exact-vs-exact test the ``ssa`` suite uses.
2. timing — warmup + timed-run median wall time of a single trajectory.
   Per the suite design rule, timing is only reported for a path that
   passed correctness.

Usage:
    python run.py                     # both gates
    python run.py --mode correctness  # z-test only
    python run.py --mode timing       # timing only
    python run.py --replicates 40     # larger correctness ensemble

Output (git-ignored ``results/``):
    python_ssa_results.json + python_ssa_results.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

_BENCH_ROOT = Path(__file__).resolve().parents[2]  # bngsim/benchmarks
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402
from _effort import add_effort_arg  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_WARMUP = nb.DEFAULT_WARMUP
DEFAULT_RUNS = nb.DEFAULT_RUNS
DEFAULT_REPLICATES = nb.DEFAULT_REPLICATES

# Exact-vs-exact ensemble-mean z-test tolerance. 6.0 matches the `ssa`
# suite: it must clear the extreme-value spread of the worst per-cell |z|
# across a whole trajectory.
Z_TOL = nb.DEFAULT_Z_TOL

# The single model — gillespy2's pure-Python SSA is infeasible on the
# large networks; see the module docstring.
MODEL = {
    "name": "simple_system",
    "net_file": "models/net/ode/simple_system.net",
    "species": 4,
    "reactions": 4,
    "t_end": 5.0,
    "n_steps": 200,
}


# ── Timing helper ─────────────────────────────────────────────────────────


def time_fn(fn, *, warmup, runs):
    """Time fn (warmup + runs). Returns (median_time_or_None, error_or_None)."""
    for _ in range(warmup):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            return None, str(e)[:200]
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            return None, str(e)[:200]
        times.append(time.perf_counter() - t0)
    return median(times), None


def fmt(t):
    if t is None:
        return "---"
    return "<0.001" if t < 0.001 else f"{t:.4f}"


# ── Model builders ────────────────────────────────────────────────────────


def build_simple_system_bngsim():
    """Build simple_system via BNGsim's ModelBuilder API (4 sp, 4 rxn)."""
    from bngsim._bngsim_core import ModelBuilder
    from bngsim._model import Model

    b = ModelBuilder()
    b.add_parameter("kon", 10.0)
    b.add_parameter("koff", 5.0)
    b.add_parameter("kcat", 0.7)
    b.add_parameter("dephos", 0.5)

    s0 = b.add_species("X_u", 5000.0)
    s1 = b.add_species("Xp", 0.0)
    s2 = b.add_species("Y", 500.0)
    s3 = b.add_species("XY", 0.0)

    b.add_observable("Xu_free", [(s0, 1.0)])
    b.add_observable("Xp_total", [(s1, 1.0)])

    b.add_reaction([s0, s2], [s3], "elementary", "kon")
    b.add_reaction([s1], [s0], "elementary", "dephos")
    b.add_reaction([s3], [s0, s2], "elementary", "koff")
    b.add_reaction([s3], [s1, s2], "elementary", "kcat")

    return Model(_core=b.build())


# simple_system species, in the order BNGsim's ModelBuilder adds them.
_MB_SPECIES_ORDER = ["Xu", "Xp", "Y", "XY"]


def build_simple_system_gillespy2():
    """Build simple_system as a gillespy2 Model (species in _MB_SPECIES_ORDER)."""
    import gillespy2

    m = gillespy2.Model(name="simple_system")
    m.add_parameter(gillespy2.Parameter(name="kon", expression="10"))
    m.add_parameter(gillespy2.Parameter(name="koff", expression="5"))
    m.add_parameter(gillespy2.Parameter(name="kcat", expression="0.7"))
    m.add_parameter(gillespy2.Parameter(name="dephos", expression="0.5"))

    Xu = gillespy2.Species(name="Xu", initial_value=5000)
    Xp = gillespy2.Species(name="Xp", initial_value=0)
    Y = gillespy2.Species(name="Y", initial_value=500)
    XY = gillespy2.Species(name="XY", initial_value=0)
    m.add_species([Xu, Xp, Y, XY])

    m.add_reaction(
        gillespy2.Reaction(name="R1", reactants={Xu: 1, Y: 1}, products={XY: 1}, rate="kon")
    )
    m.add_reaction(
        gillespy2.Reaction(name="R2", reactants={Xp: 1}, products={Xu: 1}, rate="dephos")
    )
    m.add_reaction(
        gillespy2.Reaction(name="R3", reactants={XY: 1}, products={Xu: 1, Y: 1}, rate="koff")
    )
    m.add_reaction(
        gillespy2.Reaction(name="R4", reactants={XY: 1}, products={Xp: 1, Y: 1}, rate="kcat")
    )
    return m, _MB_SPECIES_ORDER


def build_gillespy2_from_net(net_path):
    """Build a gillespy2 Model from a parsed .net file.

    Returns ``(model, species_order)`` — ``species_order`` is the gillespy2
    species-name list in .net index order (the same order BNGsim's
    ``from_net`` uses), so the two engines' trajectory arrays line up
    column-for-column. BNGL species names carry pattern syntax
    (``X(p~0,y!1).Y(x!1)``) that gillespy2 rejects, so each species is
    given the safe synthetic name ``S<index>``; the dynamics are
    name-agnostic.
    """
    import gillespy2
    from bngsim._net_reader import parse_net_file

    p = parse_net_file(net_path)
    m = gillespy2.Model(name="from_net")
    for pname, pval, _, _ in p["parameters"]:
        m.add_parameter(gillespy2.Parameter(name=pname, expression=str(pval)))

    species_order = []
    species_map = {}
    for idx, (_sname, ic, _) in enumerate(p["species"]):
        safe = f"S{idx}"
        sp = gillespy2.Species(name=safe, initial_value=int(round(ic)))
        m.add_species(sp)
        species_map[idx] = sp
        species_order.append(safe)

    for ri, r in enumerate(p["reactions"]):
        reactants: dict = {}
        products: dict = {}
        for idx in r["reactants"]:
            sp = species_map[idx]
            reactants[sp] = reactants.get(sp, 0) + 1
        for idx in r["products"]:
            sp = species_map[idx]
            products[sp] = products.get(sp, 0) + 1
        m.add_reaction(
            gillespy2.Reaction(
                name=f"R{ri}", reactants=reactants, products=products, rate=r["rate_law"]
            )
        )
    return m, species_order


# ── SSA trajectory runners — return an (n_t, n_sp) species array ──────────


def bngsim_mb_ssa(t_end, n_steps, seed):
    import bngsim

    sim = bngsim.Simulator(build_simple_system_bngsim(), method="ssa")
    result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
    return np.asarray(result.species, dtype=float)


def bngsim_net_ssa(net_path, t_end, n_steps, seed):
    import bngsim

    sim = bngsim.Simulator(bngsim.Model.from_net(net_path), method="ssa")
    result = sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=seed)
    return np.asarray(result.species, dtype=float)


def _gillespy2_traj(model, species_order, t_end, n_steps, seed):
    """Run one gillespy2 SSA trajectory; return an (n_t, n_sp) array."""
    import gillespy2

    model.timespan(np.linspace(0, t_end, n_steps + 1))
    res = model.run(solver=gillespy2.NumPySSASolver, seed=seed, number_of_trajectories=1)
    traj = res[0]
    return np.stack([np.asarray(traj[s], dtype=float) for s in species_order], axis=1)


def gillespy2_mb_ssa(t_end, n_steps, seed):
    model, order = build_simple_system_gillespy2()
    return _gillespy2_traj(model, order, t_end, n_steps, seed)


def gillespy2_net_ssa(net_path, t_end, n_steps, seed):
    model, order = build_gillespy2_from_net(net_path)
    return _gillespy2_traj(model, order, t_end, n_steps, seed)


# ── Per-path orchestration ────────────────────────────────────────────────


def run_path(
    path_label, bng_runner, g2_runner, t_end, n_steps, *, mode, warmup, runs, replicates, seed_base
):
    """Run one path: ensemble z-test correctness + single-trajectory timing."""
    entry: dict = {"path": path_label}

    # Correctness — N replicate trajectories per engine, two-sample z-test.
    if mode in ("correctness", "both"):
        bng_list, g2_list = [], []
        err = None
        for i in range(replicates):
            seed = seed_base + i
            try:
                bng_list.append(bng_runner(t_end, n_steps, seed))
                g2_list.append(g2_runner(t_end, n_steps, seed))
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                break
        if err is not None:
            entry["correctness_ok"] = False
            entry["error"] = err
        else:
            passed, max_z, detail = nb.zscore_gate(bng_list, g2_list, Z_TOL)
            entry["correctness_ok"] = bool(passed)
            entry["max_z"] = max_z
            entry["z_detail"] = detail
            entry["replicates"] = replicates
    else:
        entry["correctness_ok"] = True  # timing-only mode does not gate

    # Timing — single-trajectory median wall time, only if correctness passed.
    if mode in ("timing", "both") and entry["correctness_ok"]:
        t_b, e_b = time_fn(lambda: bng_runner(t_end, n_steps, 42), warmup=warmup, runs=runs)
        t_g, e_g = time_fn(lambda: g2_runner(t_end, n_steps, 42), warmup=warmup, runs=runs)
        entry["bngsim_time_s"] = t_b
        entry["gillespy2_time_s"] = t_g
        if e_b:
            entry["bngsim_error"] = e_b
        if e_g:
            entry["gillespy2_error"] = e_g
        if t_b and t_g and t_b > 0:
            entry["speedup_bngsim_vs_gillespy2"] = t_g / t_b
    return entry


def generate_markdown(payload: dict, outpath: Path) -> None:
    info = payload["machine_info"]
    lines = [
        "# `python_ssa` suite results",
        "",
        f"- mode: `{payload['mode']}`  effort: `{payload['effort']}`",
        f"- host: {info.get('platform', '?')}",
        f"- model: {payload['model']} ({payload['species']} sp, {payload['reactions']} rxn)",
        "",
        "Exact-SSA workflow comparison — BNGsim Gillespie vs gillespy2.",
        "Correctness is a two-sample ensemble-mean z-test; timing is only",
        "reported for a path that passed correctness.",
        "",
        "| Path | max\\|z\\| | correct | BNGsim (s) | gillespy2 (s) | speedup |",
        "|------|---------|---------|------------|---------------|---------|",
    ]
    for p in payload["paths"]:
        mz = p.get("max_z")
        mz_s = "—" if mz is None else f"{mz:.2f}"
        ok = "ok" if p.get("correctness_ok") else "FAIL"
        sp = p.get("speedup_bngsim_vs_gillespy2")
        sp_s = "—" if sp is None else f"{sp:.1f}x"
        lines.append(
            f"| {p['path']} | {mz_s} | {ok} | {fmt(p.get('bngsim_time_s'))} | "
            f"{fmt(p.get('gillespy2_time_s'))} | {sp_s} |"
        )
    lines.append("")
    outpath.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(
        description="python_ssa suite — pure-Python SSA workflow comparison"
    )
    ap.add_argument(
        "--mode",
        choices=("correctness", "timing", "both"),
        default="both",
        help="Which gates to run (default: both).",
    )
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
        help="Ensemble size for the correctness z-test (default: %(default)s).",
    )
    # --effort is accepted for cross-suite uniformity; this suite has a
    # single model, so every tier runs the same job.
    add_effort_arg(ap)
    args = ap.parse_args()

    net_path = str(_BENCH_ROOT / MODEL["net_file"])
    te, ns = MODEL["t_end"], MODEL["n_steps"]

    try:
        import gillespy2  # noqa: F401
    except ImportError:
        print("ERROR: gillespy2 is not installed — required by the python_ssa suite.")
        sys.exit(1)

    print("=" * 72)
    print("  python_ssa suite — pure-Python SSA workflow comparison (Table S6)")
    print(f"  mode={args.mode}  effort={args.effort}  replicates={args.replicates}")
    print(f"  model: {MODEL['name']} ({MODEL['species']} sp, {MODEL['reactions']} rxn)")
    print(f"  Protocol: {args.warmup}w + {args.runs}t timed; z-test tol={Z_TOL}")
    print("=" * 72)

    paths = []
    for label, bng_runner, g2_runner in [
        ("modelbuilder", bngsim_mb_ssa, gillespy2_mb_ssa),
        (
            "net_reader",
            lambda t, n, s: bngsim_net_ssa(net_path, t, n, s),
            lambda t, n, s: gillespy2_net_ssa(net_path, t, n, s),
        ),
    ]:
        if label == "net_reader" and not Path(net_path).exists():
            print(f"\n  [{label}] SKIP: {net_path} not found")
            continue
        res = run_path(
            label,
            bng_runner,
            g2_runner,
            te,
            ns,
            mode=args.mode,
            warmup=args.warmup,
            runs=args.runs,
            replicates=args.replicates,
            seed_base=2000,
        )
        paths.append(res)
        mz = res.get("max_z")
        mz_s = "" if mz is None else f"  max|z|={mz:.2f}"
        ok = "ok" if res.get("correctness_ok") else "FAIL"
        extra = f"  ({res['error']})" if res.get("error") else ""
        print(f"\n  [{label}] {ok}{mz_s}{extra}")
        if args.mode != "correctness":
            print(f"    BNGsim   {fmt(res.get('bngsim_time_s')):>10}")
            print(f"    gillespy2 {fmt(res.get('gillespy2_time_s')):>9}")

    payload = {
        "machine_info": nb.machine_info(),
        "mode": args.mode,
        "effort": args.effort,
        "model": MODEL["name"],
        "species": MODEL["species"],
        "reactions": MODEL["reactions"],
        "protocol": {"warmup": args.warmup, "runs": args.runs, "replicates": args.replicates},
        "z_tol": Z_TOL,
        "paths": paths,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "python_ssa_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    generate_markdown(payload, RESULTS_DIR / "python_ssa_results.md")
    print(f"\nResults: {json_path}")
    print(f"Report:  {RESULTS_DIR / 'python_ssa_results.md'}")


if __name__ == "__main__":
    main()
