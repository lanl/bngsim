"""GH #84 end-to-end A/B: built-in dense LU vs BLAS dgetrf, per model.

Each (model, backend) measurement runs in its OWN subprocess with a hard
wall-clock timeout, because a forced-dense *built-in* factor on a large dense
model can run for many minutes — and that slow run lives inside the C++
extension (GIL held), so an in-process signal.alarm can't interrupt it. The
subprocess timeout kills it cleanly and we report ">Ns" instead of hanging.

We force the dense path (force_dense_linear_solver=True) and toggle the BLAS
factor via the BNGSIM_LAPACK_DENSE override. The worker times only sim.run()
(model load excluded) and prints one JSON line; the parent diffs the two.

  RR_PARITY_DIR=.../rr_parity python bench_lapack_gate.py [MODEL ...]
Env: BENCH_TIMEOUT (per measurement, default 90s).
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
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "90"))

DEFAULT = [
    "BIOMD0000000497",
    "BIOMD0000000496",
    "BIOMD0000000574",
    "MODEL1011090000",
    "BIOMD0000000595",
    "BIOMD0000000470",
    "MODEL9087474843",
]


def xml_for(mid):
    hits = glob.glob(f"{ROOT}/models/{mid}/*.xml")
    return hits[0] if hits else None


REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))


# ── Worker: load model ONCE, A/B both backends in-process with warm-up +
#    median-of-REPEATS timing. Same process → Accelerate is warm after the first
#    BLAS call, so the BLAS init cost is not charged to the timed region. Prints
#    one JSON line with both backends' medians. ────────────────────────────────
def worker(mid):
    import statistics
    import warnings

    warnings.filterwarnings("ignore")
    import bngsim

    params = {
        j["model_id"]: j["params"] for j in json.load(open(f"{ROOT}/ode_jobs.json"))["jobs"]
    }[mid]
    path = xml_for(mid)
    model = bngsim.Model.from_sbml(path)
    n = model.n_species
    dens = model._core.codegen_jacobian_plan()["density"]
    run_kw = dict(
        t_span=(params["t_start"], params["t_end"]),
        n_points=min(params["n_points"], 201),
        rtol=params["rtol"],
        atol=params["atol"],
    )

    def timed(backend):
        os.environ["BNGSIM_LAPACK_DENSE"] = backend
        ls = None
        samples = []
        for k in range(REPEATS + 1):  # first run is warm-up (discarded)
            sim = bngsim.Simulator(model, method="ode", force_dense_linear_solver=True)
            t0 = time.perf_counter()
            r = sim.run(**run_kw)
            dt = time.perf_counter() - t0
            ls = r.solver_stats["linear_solver"]
            if k > 0:
                samples.append(dt)
        return statistics.median(samples), ls

    b_secs, b_ls = timed("off")
    l_secs, l_ls = timed("force")
    print(
        json.dumps(
            {
                "builtin": b_secs,
                "lapack": l_secs,
                "b_ls": b_ls,
                "l_ls": l_ls,
                "N": n,
                "density": dens,
            }
        )
    )


def measure(mid):
    """Spawn a worker subprocess (load+A/B); return dict or None on timeout/error."""
    try:
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", mid],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env={**os.environ, "RR_PARITY_DIR": ROOT},
        )
        for line in reversed(out.stdout.splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        return {"error": (out.stderr.strip().splitlines() or ["no output"])[-1][:50]}
    except subprocess.TimeoutExpired:
        return {"timeout": True}


def main():
    cands = sys.argv[1:] or DEFAULT
    import bngsim

    print(
        f"HAS_LAPACK_DENSE={bool(getattr(bngsim._bngsim_core, 'HAS_LAPACK_DENSE', False))}"
        f"  timeout={TIMEOUT}s  repeats={REPEATS} (median, warm)"
    )
    print(
        f"{'model':<20}{'N':>5}{'dens':>7}{'builtin_ms':>13}{'lapack_ms':>12}{'speedup':>9}{'ls':>4}"
    )
    for mid in cands:
        if not xml_for(mid):
            print(f"{mid:<20}  -- no xml --")
            continue
        m = measure(mid)
        if m is None or m.get("timeout"):
            print(f"{mid:<20}  TIMEOUT(>{TIMEOUT}s) — built-in factor too slow to measure")
            continue
        if "error" in m:
            print(f"{mid:<20}  ERR: {m['error']}")
            continue
        spd = m["builtin"] / m["lapack"] if m["lapack"] > 0 else 0.0
        print(
            f"{mid:<20}{m['N']:>5}{m['density']:>7.3f}{m['builtin'] * 1000:>13.1f}"
            f"{m['lapack'] * 1000:>12.1f}{spd:>8.2f}x{m['l_ls']:>4}"
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
