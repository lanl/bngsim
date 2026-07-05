#!/usr/bin/env python3
"""Pythonic workflow comparison: 8 configurations (Table S6).

Extends bench_pure_python.py to all 8 configurations:
  {BNGsim, scipy/LSODA, Diffrax, gillespy2} × {pure Python ModelBuilder, universal .net reader}

Models: simple_system (4sp), tcr_signaling (37sp), egfr_net (356sp)
Protocol: 2w+5t median wall time.
gillespy2: SSA only, seed=42.
Diffrax: exclude egfr_net if JIT > 5min.

Usage:
    python bench_pythonic_workflows.py               # full run
    python bench_pythonic_workflows.py --quick        # skip egfr_net
    python bench_pythonic_workflows.py --model tcr    # single model

Output:
    results/bench_pythonic_workflows.json
"""

import argparse
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    BENCHMARKS_DIR,
    DEFAULT_RUNS,
    DEFAULT_WARMUP,
    add_bngsim_timeout_arg,
    format_bngsim_timeout,
    get_machine_info,
    save_results,
)

_WARMUP = DEFAULT_WARMUP
_RUNS = DEFAULT_RUNS

# ── Model definitions ─────────────────────────────────────────────────────

MODELS = [
    {
        "name": "simple_system",
        "net_file": "ode/simple_system.net",
        "species": 4,
        "reactions": 4,
        "t_end": 5.0,
        "n_steps": 200,
    },
    {
        "name": "tcr_signaling",
        "net_file": "ode/tcr_signaling.net",
        "species": 37,
        "reactions": 97,
        "t_end": 300.0,
        "n_steps": 1000,
    },
    {
        "name": "egfr_net",
        "net_file": "ode/egfr_net.net",
        "species": 356,
        "reactions": 3749,
        "t_end": 120.0,
        "n_steps": 120,
    },
]


# ── Timing helpers ────────────────────────────────────────────────────────


def time_fn(fn, nw=_WARMUP, nt=_RUNS):
    """Time fn with warmup. Returns (median_time, error_str_or_None)."""
    for _ in range(nw):
        try:
            fn()
        except Exception as e:
            return None, str(e)[:200]
    times = []
    for _ in range(nt):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:
            return None, str(e)[:200]
        times.append(time.perf_counter() - t0)
    return median(times), None


def fmt(t):
    if t is None:
        return "---"
    return "<0.001" if t < 0.001 else f"{t:.4f}"


def ratio(a, b):
    if a is None or b is None or b < 1e-12:
        return "---"
    return f"{a / b:.1f}x"


# ── Pure Python model builder (simple_system only) ────────────────────────


def build_simple_system_bngsim():
    """Build simple_system via BNGsim ModelBuilder API."""
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


def _simple_system_scipy_rhs():
    """Return (rhs, y0) for simple_system."""
    kon, koff, kcat, dephos = 10.0, 5.0, 0.7, 0.5
    y0 = np.array([5000.0, 0.0, 500.0, 0.0])

    def rhs(t, y):
        Xu, Xp, Y, XY = y
        r1 = kon * Xu * Y
        r2 = dephos * Xp
        r3 = koff * XY
        r4 = kcat * XY
        return np.array(
            [
                -r1 + r2 + r3,
                -r2 + r4,
                -r1 + r3 + r4,
                r1 - r3 - r4,
            ]
        )

    return rhs, y0


# ── Engine runners: ModelBuilder path ─────────────────────────────────────


def run_bngsim_modelbuilder_ode(t_end, n_steps, bngsim_timeout=None):
    import bngsim

    model = build_simple_system_bngsim()
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)


def run_scipy_modelbuilder_ode(t_end, n_steps):
    from scipy.integrate import solve_ivp

    rhs, y0 = _simple_system_scipy_rhs()
    t_eval = np.linspace(0, t_end, n_steps + 1)
    solve_ivp(rhs, (0, t_end), y0, method="LSODA", t_eval=t_eval, rtol=1e-8, atol=1e-8)


def run_diffrax_modelbuilder_ode(t_end, n_steps):
    import diffrax
    import jax.numpy as jnp

    kon, koff, kcat, dephos = 10.0, 5.0, 0.7, 0.5
    y0 = jnp.array([5000.0, 0.0, 500.0, 0.0])

    def rhs(t, y, args):
        Xu, Xp, Y, XY = y
        r1 = kon * Xu * Y
        r2 = dephos * Xp
        r3 = koff * XY
        r4 = kcat * XY
        return jnp.array(
            [
                -r1 + r2 + r3,
                -r2 + r4,
                -r1 + r3 + r4,
                r1 - r3 - r4,
            ]
        )

    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_end, n_steps + 1))
    diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_end,
        dt0=0.01,
        y0=y0,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )


def run_bngsim_modelbuilder_ssa(t_end, n_steps):
    import bngsim

    model = build_simple_system_bngsim()
    sim = bngsim.Simulator(model, method="ssa")
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=42)


def run_gillespy2_modelbuilder_ssa(t_end, n_steps):
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

    m.timespan(np.linspace(0, t_end, n_steps + 1))
    m.run(solver=gillespy2.NumPySSASolver, seed=42, number_of_trajectories=1)


# ── Engine runners: NET reader path ───────────────────────────────────────


def run_bngsim_netreader_ode(net_path, t_end, n_steps, bngsim_timeout=None):
    """BNGsim: _net_reader → ModelBuilder → CVODE."""
    import bngsim
    from bngsim._net_reader import build_model_from_parsed, parse_net_file

    model = build_model_from_parsed(parse_net_file(net_path))
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)


def run_scipy_netreader_ode(net_path, t_end, n_steps):
    """scipy: _net_reader → numpy RHS → LSODA."""
    from bngsim._net_reader import parse_net_file
    from scipy.integrate import solve_ivp

    p = parse_net_file(net_path)
    nsp = len(p["species"])
    y0 = np.array([ic for _, ic, _ in p["species"]])
    pv = {n: v for n, v, _, _ in p["parameters"]}
    obs_sp = p["observables"]
    func_sp = p["functions"]
    rxns = p["reactions"]
    fx = [f for _, _, f in p["species"]]

    def rhs(t, y):
        o = {on: sum(f * y[i] for i, f in ent) for on, ent in obs_sp}
        ns = {
            **pv,
            **o,
            "time": t,
            "t": t,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "sin": np.sin,
            "cos": np.cos,
            "abs": np.abs,
            "min": min,
            "max": max,
            "pow": pow,
        }
        for fn, fe in func_sp:
            try:
                ns[fn] = eval(fe, {"__builtins__": {}}, ns)
            except Exception:
                ns[fn] = 0.0
        dy = np.zeros(nsp)
        for r in rxns:
            rl = r["rate_law"]
            if r["type"] == "functional":
                rate = ns.get(rl, 0.0)
            else:
                rate = pv.get(rl, 0.0)
                for ri in r["reactants"]:
                    rate *= y[ri]
            for ri in r["reactants"]:
                dy[ri] -= rate
            for pi in r["products"]:
                dy[pi] += rate
        for i, f_val in enumerate(fx):
            if f_val:
                dy[i] = 0.0
        return dy

    t_eval = np.linspace(0, t_end, n_steps + 1)
    solve_ivp(rhs, (0, t_end), y0, method="LSODA", t_eval=t_eval, rtol=1e-8, atol=1e-8)


def run_diffrax_netreader_ode(net_path, t_end, n_steps):
    """Diffrax: _net_reader → JAX RHS → Kvaerno5."""
    import diffrax
    import jax.numpy as jnp
    from bngsim._net_reader import parse_net_file

    p = parse_net_file(net_path)
    nsp = len(p["species"])
    y0 = jnp.array([ic for _, ic, _ in p["species"]])
    pv = {n: v for n, v, _, _ in p["parameters"]}
    obs_sp = p["observables"]
    func_sp = p["functions"]
    rxns = p["reactions"]
    fx = [f for _, _, f in p["species"]]

    def rhs(t, y, args):
        y_np = y  # JAX array
        o = {on: sum(f * y_np[i] for i, f in ent) for on, ent in obs_sp}
        ns = {
            **pv,
            **o,
            "time": float(t),
            "t": float(t),
            "exp": jnp.exp,
            "log": jnp.log,
            "sqrt": jnp.sqrt,
            "sin": jnp.sin,
            "cos": jnp.cos,
            "abs": jnp.abs,
            "min": jnp.minimum,
            "max": jnp.maximum,
            "pow": jnp.power,
        }
        for fn, fe in func_sp:
            try:
                ns[fn] = eval(fe, {"__builtins__": {}}, ns)
            except Exception:
                ns[fn] = 0.0
        dy = jnp.zeros(nsp)
        for r in rxns:
            rl = r["rate_law"]
            if r["type"] == "functional":
                rate = ns.get(rl, 0.0)
            else:
                rate = pv.get(rl, 0.0)
                for ri in r["reactants"]:
                    rate = rate * y_np[ri]
            for ri in r["reactants"]:
                dy = dy.at[ri].add(-rate)
            for pi in r["products"]:
                dy = dy.at[pi].add(rate)
        for i, f_val in enumerate(fx):
            if f_val:
                dy = dy.at[i].set(0.0)
        return dy

    term = diffrax.ODETerm(rhs)
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_end, n_steps + 1))
    diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_end,
        dt0=0.01,
        y0=y0,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    )


def run_bngsim_netreader_ssa(net_path, t_end, n_steps):
    """BNGsim: from_net → SSA."""
    import bngsim

    model = bngsim.Model.from_net(net_path)
    sim = bngsim.Simulator(model, method="ssa")
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=42)


def run_gillespy2_netreader_ssa(net_path, t_end, n_steps):
    """gillespy2: parse .net → gillespy2 Model → SSA."""
    import gillespy2
    from bngsim._net_reader import parse_net_file

    p = parse_net_file(net_path)

    m = gillespy2.Model(name="from_net")

    for pname, pval, _, _ in p["parameters"]:
        m.add_parameter(gillespy2.Parameter(name=pname, expression=str(pval)))

    species_map = {}
    for idx, (sname, ic, _) in enumerate(p["species"]):
        sp = gillespy2.Species(name=sname, initial_value=int(round(ic)))
        m.add_species(sp)
        species_map[idx] = sp

    for ri, r in enumerate(p["reactions"]):
        reactants = {}
        products = {}
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

    m.timespan(np.linspace(0, t_end, n_steps + 1))
    m.run(solver=gillespy2.NumPySSASolver, seed=42, number_of_trajectories=1)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    global _WARMUP, _RUNS
    ap = argparse.ArgumentParser(description="Pythonic workflow comparison (8 configs)")
    ap.add_argument("--quick", action="store_true", help="Skip egfr_net (356sp)")
    ap.add_argument("--model", type=str, default="", help="Run only this model (substring match)")
    ap.add_argument("--warmup", type=int, default=_WARMUP)
    ap.add_argument("--runs", type=int, default=_RUNS)
    add_bngsim_timeout_arg(ap, default=None)
    args = ap.parse_args()

    _WARMUP = args.warmup
    _RUNS = args.runs

    info = get_machine_info()
    all_results = {}

    # ── Part A: ModelBuilder path (simple_system only) ────────────────
    print("=" * 72)
    print("  A) MODEL DEFINED IN PYTHON (ModelBuilder / hand-coded RHS)")
    print(f"  BNGsim ODE timeout guard: {format_bngsim_timeout(args.bngsim_timeout)}")
    print("=" * 72)

    te, ns = 5.0, 200

    configs_a = [
        (
            "BNGsim ODE",
            "bngsim_mb_ode",
            lambda timeout=args.bngsim_timeout: run_bngsim_modelbuilder_ode(
                te,
                ns,
                bngsim_timeout=timeout,
            ),
        ),
        ("scipy LSODA", "scipy_mb_ode", lambda: run_scipy_modelbuilder_ode(te, ns)),
    ]

    # Try diffrax
    try:
        import diffrax

        configs_a.append(
            ("Diffrax Kvaerno5", "diffrax_mb_ode", lambda: run_diffrax_modelbuilder_ode(te, ns))
        )
    except ImportError:
        print("  Diffrax not installed, skipping")

    # SSA configs
    configs_a.append(("BNGsim SSA", "bngsim_mb_ssa", lambda: run_bngsim_modelbuilder_ssa(te, ns)))

    try:
        import gillespy2

        configs_a.append(
            ("gillespy2 SSA", "gillespy2_mb_ssa", lambda: run_gillespy2_modelbuilder_ssa(te, ns))
        )
    except ImportError:
        print("  gillespy2 not installed, skipping")

    part_a = {"model": "simple_system", "species": 4, "reactions": 4}

    hdr = f"  {'Engine':<25} {'Time (s)':>10}"
    print(hdr)
    print("  " + "-" * 35)

    for label, key, fn in configs_a:
        t, err = time_fn(fn, nw=_WARMUP, nt=_RUNS)
        part_a[key] = t
        if err:
            part_a[f"{key}_error"] = err
        print(f"  {label:<25} {fmt(t):>10}")

    all_results["A_modelbuilder"] = part_a

    # ── Part B: NET reader path (all models) ──────────────────────────
    print()
    print("=" * 72)
    print("  B) MODEL FROM UNIVERSAL .NET READER")
    print("=" * 72)

    all_results["B_net_reader"] = []

    models_to_run = MODELS
    if args.quick:
        models_to_run = [m for m in MODELS if m["species"] < 100]
    if args.model:
        models_to_run = [m for m in MODELS if args.model.lower() in m["name"].lower()]

    hdr = (
        f"  {'Model':<20} {'Sp':>4} {'BNGsim ODE':>12} "
        f"{'scipy ODE':>12} {'Diffrax ODE':>12} "
        f"{'BNGsim SSA':>12} {'gillespy2':>12}"
    )
    print(hdr)
    print("  " + "-" * 88)

    for mdef in models_to_run:
        name = mdef["name"]
        net_path = str(BENCHMARKS_DIR / mdef["net_file"])
        te = mdef["t_end"]
        ns = mdef["n_steps"]
        nsp = mdef["species"]

        if not Path(net_path).exists():
            print(f"  SKIP {name}: not found")
            continue

        row = {"name": name, "species": nsp, "reactions": mdef["reactions"]}

        # BNGsim ODE
        t_bo, e1 = time_fn(
            lambda p=net_path, t=te, n=ns, timeout=args.bngsim_timeout: run_bngsim_netreader_ode(
                p,
                t,
                n,
                bngsim_timeout=timeout,
            ),
            nw=_WARMUP,
            nt=_RUNS,
        )
        row["bngsim_ode"] = t_bo

        # scipy ODE
        t_so, e2 = time_fn(
            lambda p=net_path, t=te, n=ns: run_scipy_netreader_ode(p, t, n), nw=_WARMUP, nt=_RUNS
        )
        row["scipy_ode"] = t_so

        # Diffrax ODE (skip for large models)
        t_do = None
        if nsp <= 100:
            try:
                import diffrax  # noqa: F401  availability check; runner imports it on call

                t_do, e3 = time_fn(
                    lambda p=net_path, t=te, n=ns: run_diffrax_netreader_ode(p, t, n),
                    nw=_WARMUP,
                    nt=_RUNS,
                )
            except ImportError:
                pass
        row["diffrax_ode"] = t_do

        # BNGsim SSA
        t_bs, e4 = time_fn(
            lambda p=net_path, t=te, n=ns: run_bngsim_netreader_ssa(p, t, n), nw=_WARMUP, nt=_RUNS
        )
        row["bngsim_ssa"] = t_bs

        # gillespy2 SSA (skip for very large models)
        t_gs = None
        if nsp <= 100:
            try:
                import gillespy2  # noqa: F401  availability check; runner imports it on call

                t_gs, e5 = time_fn(
                    lambda p=net_path, t=te, n=ns: run_gillespy2_netreader_ssa(p, t, n),
                    nw=_WARMUP,
                    nt=_RUNS,
                )
            except ImportError:
                pass
        row["gillespy2_ssa"] = t_gs

        all_results["B_net_reader"].append(row)

        print(
            f"  {name:<20} {nsp:>4} {fmt(t_bo):>12} "
            f"{fmt(t_so):>12} {fmt(t_do):>12} "
            f"{fmt(t_bs):>12} {fmt(t_gs):>12}"
        )

    # ── Summary table ─────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  SUMMARY: Speedup vs BNGsim (ODE)")
    print("=" * 72)

    for row in all_results["B_net_reader"]:
        base = row.get("bngsim_ode")
        if base is None or base <= 0:
            continue
        print(f"\n  {row['name']} ({row['species']} sp):")
        for eng, key in [("scipy", "scipy_ode"), ("Diffrax", "diffrax_ode")]:
            t = row.get(key)
            if t and t > 0:
                print(f"    {eng}: {ratio(t, base)} slower")

    # Save
    output = {
        "machine_info": info,
        "protocol": {
            "warmup": _WARMUP,
            "runs": _RUNS,
            "bngsim_timeout": args.bngsim_timeout,
        },
        "results": all_results,
    }
    save_results(output, "bench_pythonic_workflows")


if __name__ == "__main__":
    main()
