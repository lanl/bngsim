"""GH #84 — the regime the first benchmark MISSED: large DENSE SBML models.

9 of the 12 rr_parity models with N>786 are dense (density 0.57-1.00) at
N=1025-6867 — the actual "large dense ODE model" target of #84. The first A/B
benchmark stopped at N=786 (all RHS-bound) and wrongly concluded "no win". This
measures built-in dense LU vs BLAS dgetrf where the O(N^3) factorization should
finally dominate.

Each (model, backend) runs in its own subprocess with a hard subprocess timeout
(a built-in run that's too slow gets killed and reported as >Ns instead of
blocking the LAPACK leg). One timed run (no warm-up — Accelerate init is
negligible next to a multi-second N>1000 factorization).

  RR_PARITY_DIR=.../rr_parity python bench_large_dense.py [MODEL ...]
Env: BENCH_TIMEOUT (per measurement, default 480s).
"""

import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("RR_PARITY_DIR") or os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "parity_checks", "rr_parity"
    )
)
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "480"))

# Large DENSE models (density >= 0.10 → dense path), smallest-first so partial
# results are useful even if the 5k-6k giants run long.
DEFAULT = [
    "MODEL1007060000",  # 1025, 1.00
    "MODEL1112100000",  # 1265, 0.60
    "MODEL1009150002",  # 1604, 0.85
    "MODEL1601050000",  # 2047, 0.81
    "MODEL1703150000",  # 2129, 0.57
    "MODEL1504130000",  # 5063, 1.00
]


def xml_for(mid):
    hits = glob.glob(f"{ROOT}/models/{mid}/*.xml")
    return hits[0] if hits else None


REPEATS = int(os.environ.get("BENCH_REPEATS", "3"))


def worker(mid, backend):
    os.environ["BNGSIM_LAPACK_DENSE"] = backend
    import statistics
    import warnings

    warnings.filterwarnings("ignore")
    import bngsim

    params = {
        j["model_id"]: j["params"] for j in json.load(open(f"{ROOT}/ode_jobs.json"))["jobs"]
    }[mid]
    model = bngsim.Model.from_sbml(xml_for(mid))  # load once, excluded from timing
    n = model.n_species
    dens = model._core.codegen_jacobian_plan()["density"]
    run_kw = dict(
        t_span=(params["t_start"], params["t_end"]),
        n_points=min(params["n_points"], 51),
        rtol=params["rtol"],
        atol=params["atol"],
    )
    samples = []
    facts = solves = steps = ls = None
    for _ in range(REPEATS):
        sim = bngsim.Simulator(model, method="ode", force_dense_linear_solver=True)
        t0 = time.perf_counter()
        r = sim.run(**run_kw)
        samples.append(time.perf_counter() - t0)
        s = r.solver_stats
        facts, solves, steps, ls = (
            s["n_jac_evals"],
            s["n_nonlin_iters"],
            s["n_steps"],
            s["linear_solver"],
        )
    print(
        json.dumps(
            {
                "secs": statistics.median(samples),
                "ls": ls,
                "steps": steps,
                "facts": facts,
                "solves": solves,
                "N": n,
                "density": dens,
            }
        )
    )


def measure(mid, backend):
    try:
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", mid, backend],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env={**os.environ, "RR_PARITY_DIR": ROOT},
        )
        for line in reversed(out.stdout.splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        return {"error": (out.stderr.strip().splitlines() or ["no output"])[-1][:60]}
    except subprocess.TimeoutExpired:
        return {"timeout": True}


def main():
    cands = sys.argv[1:] or DEFAULT
    print(
        f"large DENSE models — built-in dense LU vs BLAS dgetrf   timeout={TIMEOUT}s repeats={REPEATS}"
    )
    print(
        f"{'model':<18}{'N':>6}{'dens':>6}{'builtin_s':>11}{'lapack_s':>10}{'speedup':>9}"
        f"{'steps':>7}{'facts':>7}{'solves':>8}"
    )
    for mid in cands:
        if not xml_for(mid):
            print(f"{mid:<18}  -- no xml --")
            continue
        b = measure(mid, "off")
        l = measure(mid, "force")
        n = (b or l or {}).get("N", "?")
        dens = (b or l or {}).get("density", 0.0)

        def fmt(m):
            if m is None or m.get("timeout"):
                return f">{TIMEOUT}"
            if "error" in m:
                return "ERR"
            return f"{m['secs']:.2f}"

        bs, lp = fmt(b), fmt(l)
        if b and l and "secs" in b and "secs" in l and l["secs"] > 0:
            spd = f"{b['secs'] / l['secs']:.2f}x"
        elif b and b.get("timeout") and l and "secs" in l:
            spd = ">timeout"
        else:
            spd = "-"
        g = b if (isinstance(b, dict) and "facts" in b) else (l if isinstance(l, dict) else {})
        steps, facts, solves = g.get("steps", "?"), g.get("facts", "?"), g.get("solves", "?")
        dstr = f"{dens:.2f}" if isinstance(dens, float) else str(dens)
        print(
            f"{mid:<18}{str(n):>6}{dstr:>6}{bs:>11}{lp:>10}{spd:>9}"
            f"{str(steps):>7}{str(facts):>7}{str(solves):>8}",
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
