#!/usr/bin/env python3
"""Pythonic benchmarks: BNGsim vs scipy LSODA.

Two comparisons:
  A) MODEL DEFINED IN PYTHON — no files, no .net reader.
     BNGsim: ModelBuilder API. scipy: hand-coded numpy RHS.
  B) MODEL FROM .NET READER — universal _net_reader parses .net.
     BNGsim: _net_reader → ModelBuilder. scipy: _net_reader → numpy RHS.

Protocol: 2 warmup + 5 timed, median wall time.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import add_bngsim_timeout_arg, format_bngsim_timeout

N_WARMUP = 2
N_TIMED = 5
BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"


def time_fn(fn, nw=N_WARMUP, nt=N_TIMED):
    for _ in range(nw):
        try:
            fn()
        except Exception as e:
            return None, str(e)
    times = []
    for _ in range(nt):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:
            return None, str(e)
        times.append(time.perf_counter() - t0)
    return median(times), None


def fmt(t):
    if t is None:
        return "---"
    return "<0.001" if t < 0.001 else f"{t:.3f}"


def ratio(a, b):
    if a is None or b is None or b < 1e-12:
        return "---"
    return f"{a / b:.1f}x"


# ══════════════════════════════════════════════════════════════
# COMPARISON A: Model defined entirely in Python (no files)
# ══════════════════════════════════════════════════════════════
# We define the same model two ways:
#   - BNGsim: via ModelBuilder (C++ model built from Python calls)
#   - scipy: via a hand-coded numpy RHS function


def build_simple_system_bngsim():
    """simple_system: 4 species, 4 reactions, all elementary."""
    from bngsim._bngsim_core import ModelBuilder
    from bngsim._model import Model

    b = ModelBuilder()
    b.add_parameter("kon", 10.0)
    b.add_parameter("koff", 5.0)
    b.add_parameter("kcat", 0.7)
    b.add_parameter("dephos", 0.5)

    s0 = b.add_species("X_u", 5000.0)  # X(p~0,y)
    s1 = b.add_species("Xp", 0.0)  # X(p~1,y)
    s2 = b.add_species("Y", 500.0)  # Y(x)
    s3 = b.add_species("XY", 0.0)  # X(p~0,y!1).Y(x!1)

    b.add_observable("Xu_free", [(s0, 1.0)])
    b.add_observable("Xp_total", [(s1, 1.0)])

    b.add_reaction([s0, s2], [s3], "elementary", "kon")
    b.add_reaction([s1], [s0], "elementary", "dephos")
    b.add_reaction([s3], [s0, s2], "elementary", "koff")
    b.add_reaction([s3], [s1, s2], "elementary", "kcat")

    return Model(_core=b.build())


def run_simple_system_bngsim_ode(t_end, n_steps, bngsim_timeout=None):
    import bngsim

    model = build_simple_system_bngsim()
    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)


def run_simple_system_scipy(t_end, n_steps):
    kon, koff, kcat, dephos = 10.0, 5.0, 0.7, 0.5
    y0 = np.array([5000.0, 0.0, 500.0, 0.0])

    def rhs(t, y):
        Xu, Xp, Y, XY = y
        r1 = kon * Xu * Y  # Xu + Y -> XY
        r2 = dephos * Xp  # Xp -> Xu
        r3 = koff * XY  # XY -> Xu + Y
        r4 = kcat * XY  # XY -> Xp + Y
        return np.array(
            [
                -r1 + r2 + r3,  # dXu/dt
                -r2 + r4,  # dXp/dt
                -r1 + r3 + r4,  # dY/dt
                r1 - r3 - r4,  # dXY/dt
            ]
        )

    t_eval = np.linspace(0, t_end, n_steps + 1)
    solve_ivp(rhs, (0, t_end), y0, method="LSODA", t_eval=t_eval, rtol=1e-8, atol=1e-8)


def run_simple_system_bngsim_ssa(t_end, n_steps):
    import bngsim

    model = build_simple_system_bngsim()
    sim = bngsim.Simulator(model, method="ssa")
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, seed=42)


def run_simple_system_gillespy2(t_end, n_steps):
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


# ══════════════════════════════════════════════════════════════
# COMPARISON B: Model from universal .net reader (file-based)
# ══════════════════════════════════════════════════════════════


def run_bngsim_from_net_reader(net_path, t_end, n_steps, bngsim_timeout=None):
    """BNGsim: _net_reader → ModelBuilder → CVODE."""
    from bngsim._net_reader import build_model_from_parsed, parse_net_file

    model = build_model_from_parsed(parse_net_file(net_path))
    import bngsim

    sim = bngsim.Simulator(model, method="ode")
    sim.run(t_span=(0, t_end), n_points=n_steps + 1, timeout=bngsim_timeout)


def run_scipy_from_net_reader(net_path, t_end, n_steps):
    """scipy: _net_reader → numpy RHS → LSODA."""
    from bngsim._net_reader import parse_net_file

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


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="Pythonic benchmarks: BNGsim vs scipy LSODA")
    ap.add_argument("--quick", action="store_true", help="Skip the large egfr_net case")
    add_bngsim_timeout_arg(ap, default=None)
    args = ap.parse_args()

    quick = args.quick
    results = {}

    print(f"BNGsim ODE timeout guard: {format_bngsim_timeout(args.bngsim_timeout)}\n")

    # ── Comparison A: Literal Python definition ──────────────
    print("=" * 72)
    print("A) MODEL DEFINED IN PYTHON (no files, no .net reader)")
    print("   BNGsim: ModelBuilder API.  scipy: hand-coded numpy RHS.")
    print("=" * 72)

    te, ns = 5.0, 200  # simple_system: 4sp, 4rxn, t_end=5

    h = f"{'Method':<12} {'Engine':<16} {'Time (s)':>10}"
    print(h)
    print("-" * len(h))

    tb_ode, e1 = time_fn(
        lambda timeout=args.bngsim_timeout: run_simple_system_bngsim_ode(
            te,
            ns,
            bngsim_timeout=timeout,
        )
    )
    tl_ode, e2 = time_fn(lambda: run_simple_system_scipy(te, ns))
    tb_ssa, e3 = time_fn(lambda: run_simple_system_bngsim_ssa(te, ns))
    tg_ssa, e4 = time_fn(lambda: run_simple_system_gillespy2(te, ns))

    print(f"{'ODE':<12} {'BNGsim CVODE':<16} {fmt(tb_ode):>10}")
    print(f"{'ODE':<12} {'scipy LSODA':<16} {fmt(tl_ode):>10}  ({ratio(tl_ode, tb_ode)})")
    print(f"{'SSA':<12} {'BNGsim Gillespie':<16} {fmt(tb_ssa):>10}")
    print(f"{'SSA':<12} {'gillespy2':<16} {fmt(tg_ssa):>10}  ({ratio(tg_ssa, tb_ssa)})")

    results["A_literal"] = {
        "model": "simple_system",
        "species": 4,
        "reactions": 4,
        "ode_bngsim": tb_ode,
        "ode_scipy": tl_ode,
        "ssa_bngsim": tb_ssa,
        "ssa_gillespy2": tg_ssa,
    }

    for tag, err in [("BNGsim ODE", e1), ("scipy ODE", e2), ("BNGsim SSA", e3), ("gillespy2", e4)]:
        if err:
            print(f"  [{tag}] {err}", file=sys.stderr)

    # ── Comparison B: Universal .net reader ──────────────────
    print()
    print("=" * 72)
    print("B) MODEL FROM UNIVERSAL .NET READER")
    print("   BNGsim: _net_reader→ModelBuilder→CVODE.")
    print("   scipy:  _net_reader→numpy RHS→LSODA.")
    print("=" * 72)

    NET_MODELS = [
        ("tcr_signaling", 37, 97, 300, 1000),
        ("egfr_net", 356, 3749, 120, 120),
    ]

    h2 = f"{'Model':<20} {'Sp':>4} {'Rxn':>5} {'BNGsim':>10} {'scipy':>10} {'scipy/BNG':>10}"
    print(h2)
    print("-" * len(h2))

    results["B_net_reader"] = []
    for name, nsp, nrxn, te, ns in NET_MODELS:
        if quick and nsp > 40:
            continue
        np_ = str(BENCH_DIR / f"ode/{name}.net")

        tb, e1 = time_fn(
            lambda p=np_, t=te, n=ns, timeout=args.bngsim_timeout: run_bngsim_from_net_reader(
                p,
                t,
                n,
                bngsim_timeout=timeout,
            )
        )
        tl, e2 = time_fn(lambda p=np_, t=te, n=ns: run_scipy_from_net_reader(p, t, n))

        results["B_net_reader"].append(
            {
                "name": name,
                "species": nsp,
                "reactions": nrxn,
                "bngsim": tb,
                "scipy": tl,
            }
        )

        print(f"{name:<20} {nsp:>4} {nrxn:>5} {fmt(tb):>10} {fmt(tl):>10} {ratio(tl, tb):>10}")
        if e1:
            print(f"  [BNGsim] {e1}", file=sys.stderr)
        if e2:
            print(f"  [scipy] {e2}", file=sys.stderr)

    # Save
    op = Path(__file__).parent.parent / "results" / "bench_pythonic.json"
    op.parent.mkdir(exist_ok=True)
    with open(op, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {op}")


if __name__ == "__main__":
    main()
