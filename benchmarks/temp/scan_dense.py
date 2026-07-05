import os

os.environ["BNGSIM_ALLOW_STALE_CORE"] = "1"
import warnings

warnings.filterwarnings("ignore")
import glob
import json
import signal
import time

import bngsim

# Corpus root: env override, else repo-relative (this file lives at
# bngsim/dev/investigations/lu_backend/). No hardcoded absolute paths.
ROOT = os.environ.get("RR_PARITY_DIR") or os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "parity_checks", "rr_parity"
    )
)
jobs = json.load(open(f"{ROOT}/ode_jobs.json"))["jobs"]
pmap = {j["model_id"]: j["params"] for j in jobs}


class TO(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TO()))

# large-N candidates (from SBML species counts)
cands = [
    "MODEL1601250000",
    "MODEL3631586579",
    "MODEL0404023805",
    "BIOMD0000000497",
    "BIOMD0000000496",
    "MODEL9087474843",
    "MODEL2402030002",
    "BIOMD0000000579",
    "BIOMD0000000595",
    "MODEL9089538076",
    "BIOMD0000000205",
    "BIOMD0000000574",
    "BIOMD0000000470",
    "MODEL1011090000",
]


def run(f, p, fd):
    m = bngsim.Model.from_sbml(f)
    sim = bngsim.Simulator(m, method="ode", force_dense_linear_solver=fd)
    t0 = time.perf_counter()
    r = sim.run(
        t_span=(p["t_start"], p["t_end"]),
        n_points=min(p["n_points"], 201),
        rtol=p["rtol"],
        atol=p["atol"],
    )
    return m.n_species, time.perf_counter() - t0, r.solver_stats


print(
    f"{'model':<20}{'N':>5}{'sparse_ms':>11}{'dense_ms':>10}{'d/s':>7}  {'verdict':<14} setups/nni"
)
for mid in cands:
    p = pmap.get(mid)
    f = (glob.glob(f"{ROOT}/models/{mid}/*.xml") or [None])[0]
    if not p or not f:
        print(f"{mid:<20} -- skip --")
        continue
    try:
        signal.alarm(40)
        N, ws, ss = run(f, p, False)
        signal.alarm(0)
        signal.alarm(40)
        _, wd, sd = run(f, p, True)
        signal.alarm(0)
        ratio = wd / ws if ws > 0 else 0
        verdict = "DENSE(target)" if ratio <= 1.25 else "sparse"
        print(
            f"{mid:<20}{N:>5}{ws * 1000:>11.1f}{wd * 1000:>10.1f}{ratio:>7.2f}  {verdict:<14} {sd.get('n_jac_evals')}/{sd.get('n_nonlin_iters')}"
        )
    except TO:
        signal.alarm(0)
        print(f"{mid:<20}{'':>5}  TIMEOUT(>40s)")
    except Exception as e:
        signal.alarm(0)
        print(f"{mid:<20}  EXC {type(e).__name__}: {str(e)[:45]}")
