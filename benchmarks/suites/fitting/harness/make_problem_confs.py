"""
make_problem_confs.py
---------------------
Regenerate the per-problem ``conf/subprocess.conf`` / ``conf/bngsim.conf``
pairs for the bngsim multi-model fitting benchmark.

Each benchmark row is one fitting problem run two ways:

  * ``subprocess.conf``  -- ``bngl_backend=bionetgen`` (BNG2.pl / run_network)
  * ``bngsim.conf``      -- ``bngl_backend=bngsim``    (in-process BNGsim)

The two confs of a row are byte-identical except for ``bngl_backend`` and
``output_dir`` -- they do *identical* fitting work, so the benchmark
measures wall-clock for the same work.

Source of truth for each problem is its mmc3 ``-ss.conf`` (PyBNF iScience
2019 Data S1). Those are cluster jobs: ``parallel_count`` in the hundreds,
``max_iterations=100000`` with ``min_objective`` early-stop. This script
strips the cluster knobs and the early-stop, pins ``parallel_count`` to a
laptop-safe value, and pins a *fixed* ``max_iterations`` (the ``iters``
field below). To recalibrate the run length, edit ``iters`` and re-run
this script -- do not hand-edit the generated confs.

Scatter search is used for every problem (``fit_type=ss``).

Usage:  python harness/make_problem_confs.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
MMC3 = BENCH_DIR / "mmc3"

# Laptop cap: 4 of 6 cores (Dask keeps one for the scheduler; one left free).
PARALLEL_COUNT = 4

# A single sim that runs past this many seconds is killed and the parameter
# set penalised. Pure safety net for ODE rows (a sim is sub-second here).
WALL_TIME_SIM = 3600

# Free-parameter declaration keywords PyBNF understands.
_VAR_RE = re.compile(r"^\s*(log)?(uniform|normal)_var\s*=", re.IGNORECASE)


def _key_re(key: str) -> re.Pattern[str]:
    """Regex matching a ``key=value`` conf line, ignoring trailing comments."""
    return re.compile(rf"^\s*{key}\s*=\s*(.+?)\s*(#.*)?$", re.IGNORECASE)


# Each problem: slug (== benchmark dir), the mmc3 -ss.conf to mine, the
# model basename, the .exp/.prop data basenames, and the fixed scatter-
# search iteration count for the production run (calibrated separately).
#
# Only the mmc3 ODE problems that run clean on *both* PyBNF backends are
# here -- see harness/problems.py for the smoke matrix and the problems
# that were dropped. The Kozer EGFR row
# (problem 10) is hand-curated in kozer_egfr/conf/ and is not generated
# by this script.
PROBLEMS = [
    dict(
        slug="prob05_threestep",
        title="Problem 5 -- three-step cascade (3 params)",
        ss_conf="05-threestep/fit_ss/fit_ss.conf",
        model="m1.bngl",
        data=["d1.exp", "d1.prop"],
        iters=50,
    ),
    dict(
        slug="prob07_egg",
        title="Problem 7 -- egg-shaped curve, 10 params",
        ss_conf="07-egg/egg-ss.conf",
        model="egg.bngl",
        data=["egg.exp"],
        iters=50,
    ),
    dict(
        slug="prob24_jnk",
        title="Problem 24 -- Jnk cascade, 12 params",
        ss_conf="24-jnk/fey-ss.conf",
        model="JNKmodel_180724_bnf.bngl",
        data=["tc.exp", "dr_contr.exp", "dr_simkk4.exp", "dr_simkk7.exp", "dr_jnkinh.exp"],
        iters=50,
    ),
    dict(
        slug="prob18_mapk",
        title="Problem 18 -- MAPK scaffold, 13 params",
        ss_conf="18-mapk/mapk-ss.conf",
        model="Scaff-22_tofit.bngl",
        data=["doseresponse.exp"],
        iters=50,
    ),
]


def _scan(text: str, key: str) -> str | None:
    """Return the value of ``key=...`` from a conf, ignoring trailing comments."""
    for line in text.splitlines():
        m = _key_re(key).match(line)
        if m:
            return m.group(1).strip()
    return None


def _vars(text: str) -> list[str]:
    """Return the (uncommented) free-parameter declaration lines."""
    return [ln.strip() for ln in text.splitlines() if _VAR_RE.match(ln)]


def render(prob: dict, backend: str, smoke: bool = False) -> str:
    src = (MMC3 / prob["ss_conf"]).read_text()
    objfunc = _scan(src, "objfunc") or "sos"
    pop = _scan(src, "population_size") or "9"
    normalization = _scan(src, "normalization")
    beta = _scan(src, "beta")
    var_lines = _vars(src)
    if not var_lines:
        raise SystemExit(f"{prob['slug']}: no free-parameter vars found in {prob['ss_conf']}")

    # Paths are relative to the problem directory: run_jobs.py invokes
    # pybnf with cwd set there, and PyBNF resolves model= against cwd.
    data = ", ".join(f"data/{d}" for d in prob["data"])
    backend_label = (
        "BNG2.pl / run_network subprocess" if backend == "bionetgen" else "in-process BNGsim"
    )

    # Smoke confs exercise the full job path cheaply: one iteration, a
    # small population, 3 cores. They verify a backend works end to end
    # without doing a real fit.
    if smoke:
        pop = str(min(int(pop), 5))
        iters = 1
        parallel = 3
        out = f"output/_smoke/{backend}"
    else:
        iters = prob["iters"]
        parallel = PARALLEL_COUNT
        out = f"output/{backend}"

    lines = [
        f"# {prob['title']}{'  [SMOKE]' if smoke else ''}",
        f"# Backend: {backend_label}.",
        "# Auto-generated by harness/make_problem_confs.py -- edit the PROBLEMS",
        "# table there and re-run; do not hand-edit this file.",
        "",
        f"model=bngl/{prob['model']} : {data}",
        f"output_dir={out}",
        f"bngl_backend={backend}",
        "",
        "# Scatter search, fixed iteration budget (no min_objective early-stop)",
        "# so the subprocess and in-process rows do identical fitting work.",
        "fit_type=ss",
        f"objfunc={objfunc}",
        f"population_size={pop}",
        f"max_iterations={iters}",
        f"parallel_count={parallel}",
    ]
    if normalization:
        lines.append(f"normalization={normalization}")
    if beta:
        lines.append(f"beta={beta}")
    lines += [
        f"wall_time_sim={WALL_TIME_SIM}",
        "delete_old_files=1",
        "verbosity=1",
        "",
        "# Free parameters (bounds verbatim from the mmc3 -ss.conf).",
        *var_lines,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also write conf/{subprocess,bngsim}_smoke.conf "
        "(1 iteration, small population, 3 cores).",
    )
    args = parser.parse_args()

    for prob in PROBLEMS:
        conf_dir = BENCH_DIR / prob["slug"] / "conf"
        conf_dir.mkdir(parents=True, exist_ok=True)
        for backend, stem in (("bionetgen", "subprocess"), ("bngsim", "bngsim")):
            (conf_dir / f"{stem}.conf").write_text(render(prob, backend))
            if args.smoke:
                (conf_dir / f"{stem}_smoke.conf").write_text(render(prob, backend, smoke=True))
        extra = " (+ smoke)" if args.smoke else ""
        print(
            f"[make_confs] {prob['slug']}: subprocess.conf + bngsim.conf{extra} "
            f"(iters={prob['iters']})"
        )


if __name__ == "__main__":
    main()
