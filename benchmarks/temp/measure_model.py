import glob
import json
import os
import signal
import sys
import time

os.environ.setdefault(
    "BNGSIM_ALLOW_STALE_CORE", "1"
)  # only ssa_simulator.cpp is stale; ODE path unaffected
import warnings

warnings.filterwarnings("ignore")
import bngsim


class TO(Exception):
    pass


def _alarm(sig, frm):
    raise TO()


signal.signal(signal.SIGALRM, _alarm)

# Corpus root: env override, else repo-relative (no hardcoded absolute paths).
ROOT = os.environ.get("RR_PARITY_DIR") or os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "parity_checks", "rr_parity"
    )
)
jobs = json.load(open(f"{ROOT}/ode_jobs.json"))["jobs"]
pmap = {j["model_id"]: j["params"] for j in jobs}


def xml_for(mid):
    hits = glob.glob(f"{ROOT}/models/{mid}/*.xml")
    return hits[0] if hits else None


cands = sys.argv[1:] or [
    "BIOMD0000000205",
    "BIOMD0000000595",
    "BIOMD0000000579",
    "BIOMD0000000497",
    "MODEL0404023805",
    "MODEL1601250000",
]
print(f"{'model':<20}{'N':>5}{'wall_s':>9}{'n_steps':>9}{'GETRF':>8}{'GETRS':>8}  notes")
for mid in cands:
    p = pmap.get(mid)
    f = xml_for(mid)
    if not p or not f:
        print(f"{mid:<20}  -- no job/xml --")
        continue
    try:
        model = bngsim.Model.from_sbml(f)
        N = model.n_species
        sim = bngsim.Simulator(model, method="ode", force_dense_linear_solver=True)
        signal.alarm(90)
        t0 = time.perf_counter()
        r = sim.run(
            t_span=(p["t_start"], p["t_end"]),
            n_points=min(p["n_points"], 501),
            rtol=p["rtol"],
            atol=p["atol"],
        )
        wall = time.perf_counter() - t0
        signal.alarm(0)
        s = r.solver_stats
        getrf = s.get("n_jac_evals", 0)  # CVodeGetNumLinSolvSetups -> one GETRF each
        getrs = s.get("n_nonlin_iters", 0)  # ~one GETRS per Newton iter
        print(f"{mid:<20}{N:>5}{wall:>9.3f}{s.get('n_steps', 0):>9}{getrf:>8}{getrs:>8}")
    except Exception as e:
        print(f"{mid:<20}  EXC: {type(e).__name__}: {str(e)[:60]}")
