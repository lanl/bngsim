"""
consistency_check.py
--------------------
Verify that the two PyBNF backends agree on the biology before we trust
their timing comparison.

Every benchmark problem is a deterministic ODE fit run two ways: through
the BNG2.pl / run_network subprocess stack and through in-process
BNGsim. Both integrate the *same* BioNetGen-generated reaction network,
so for any fixed parameter set they must produce the same trajectory and
hence the same objective.

The check exploits PyBNF's ``random_seed`` config key. Scatter search's
parameter-set sampling depends only on the seed and the algorithm, not
on the simulator backend. So running each problem on both backends with
the *same* seed and a short iteration budget makes both explore the
identical set of parameter vectors. We then match parameter vectors
between the two runs and compare the objective each backend assigned --
they should agree to tight numerical tolerance.

This needs no separate model/network plumbing: it drives PyBNF exactly
as the benchmark does, just briefly and with a pinned seed.

Output: ``results/consistency.json`` -- one row per problem with the
worst objective disagreement seen and whether it passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import (  # noqa: E402
    BENCH_DIR,
    EFFORT_LEVELS,
    PROBLEMS,
    Problem,
    problems_at_effort,
)

RESULTS_DIR = BENCH_DIR / "results"

# Pinned seed: any fixed integer works; both backends of a problem use it.
SEED = 20260517
# Short budget -- the gate only needs a handful of shared parameter sets.
CONSISTENCY_ITERS = 3
CONSISTENCY_PARALLEL = 3
# ODE backends integrating the same network should agree to ~integration
# tolerance; 1e-3 relative is a generous gross-disagreement gate.
TOL_REL = 1e-3


@dataclass
class ConsistencyResult:
    problem: str
    label: str
    n_psets_compared: int
    max_rel_obj_diff: float
    threshold: float
    passed: bool
    notes: str = ""


# Conf keys we override for the consistency run; existing occurrences are
# dropped and replaced with the values below.
_OVERRIDE = {
    "max_iterations": str(CONSISTENCY_ITERS),
    "parallel_count": str(CONSISTENCY_PARALLEL),
    "random_seed": str(SEED),
}


def _make_consistency_conf(prob: Problem, conf_attr: str, backend: str) -> tuple[Path, Path]:
    """Write a short, seeded variant of a production conf. Returns
    (conf_path, output_dir)."""
    src: Path = getattr(prob, conf_attr)
    out_rel = f"output/_consistency/{backend}"
    kept = []
    for line in src.read_text().splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in _OVERRIDE or key == "output_dir":
            continue
        kept.append(line)
    kept.append("")
    kept.append("# --- consistency-check overrides (consistency_check.py) ---")
    kept.append(f"output_dir={out_rel}")
    for k, v in _OVERRIDE.items():
        kept.append(f"{k}={v}")
    conf_path = prob.dir / "conf" / f"_consistency_{backend}.conf"
    conf_path.write_text("\n".join(kept) + "\n")
    return conf_path, (prob.dir / out_rel)


def _parse_psets(output_dir: Path) -> dict[tuple[str, ...], float]:
    """Map each explored parameter vector to its objective.

    Reads ``Results/sorted_params_final.txt``. The parameter columns are
    byte-identical across backends (same seed), so the raw value strings
    key the dict directly.
    """
    path = output_dir / "Results" / "sorted_params_final.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path} has no data rows")
    header = [c.strip() for c in lines[0].lstrip("#").split("\t") if c.strip()]
    obj_idx = header.index("Obj")
    param_idx = [i for i, c in enumerate(header) if i not in (0, obj_idx)]
    out: dict[tuple[str, ...], float] = {}
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split("\t") if c.strip() != ""]
        if obj_idx >= len(cells):
            continue
        key = tuple(cells[i] for i in param_idx if i < len(cells))
        out[key] = float(cells[obj_idx])
    return out


def _run(conf_path: Path, prob_dir: Path, pybnf_cmd: str, bngpath: str) -> None:
    env = os.environ.copy()
    if bngpath:
        env["BNGPATH"] = bngpath
    proc = subprocess.run(
        [pybnf_cmd, "-c", str(conf_path), "-o"],
        cwd=prob_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pybnf exit {proc.returncode}: {proc.stderr.strip()[-400:]}")


def run_problem(prob: Problem, pybnf_cmd: str, bngpath: str) -> ConsistencyResult:
    res = ConsistencyResult(prob.slug, prob.label, 0, float("nan"), TOL_REL, False)
    try:
        sub_conf, sub_out = _make_consistency_conf(prob, "subprocess_conf", "subprocess")
        bng_conf, bng_out = _make_consistency_conf(prob, "bngsim_conf", "bngsim")
        _run(sub_conf, prob.dir, pybnf_cmd, bngpath)
        _run(bng_conf, prob.dir, pybnf_cmd, bngpath)
        sub = _parse_psets(sub_out)
        bng = _parse_psets(bng_out)
    except (RuntimeError, OSError, ValueError) as err:
        res.notes = str(err)
        return res

    shared = sorted(set(sub) & set(bng))
    if not shared:
        res.notes = "no shared parameter vectors between backends"
        return res

    worst = 0.0
    compared = 0
    for key in shared:
        a, b = sub[key], bng[key]
        # Skip parameter sets that failed (inf) in either backend.
        if a != a or b != b or a in (float("inf"),) or b in (float("inf"),):
            continue
        denom = max(abs(a), abs(b), 1e-30)
        worst = max(worst, abs(a - b) / denom)
        compared += 1

    res.n_psets_compared = compared
    res.max_rel_obj_diff = worst
    if compared == 0:
        res.notes = "all shared parameter sets failed in at least one backend"
        return res
    res.passed = worst <= TOL_REL
    return res


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="*", default=None, help="Check only these problems (slugs)."
    )
    parser.add_argument(
        "--effort",
        choices=list(EFFORT_LEVELS),
        default="high",
        help="Cumulative effort threshold: 'low' checks only low-effort "
        "problems, 'medium' low + medium, 'high' all (default: high).",
    )
    parser.add_argument("--pybnf-cmd", default=os.environ.get("PYBNF_CMD", "pybnf"))
    parser.add_argument("--bngpath", default=os.environ.get("BNGPATH", ""))
    parser.add_argument("--results-out", type=Path, default=RESULTS_DIR / "consistency.json")
    args = parser.parse_args()

    if not args.bngpath:
        print("[consistency] WARNING: $BNGPATH unset; subprocess runs may fail.", file=sys.stderr)

    # Cumulative effort threshold: 'low' < 'medium' < 'high' (= all).
    selected = {p.slug for p in problems_at_effort(args.effort)}
    print(
        f"[consistency] effort={args.effort}: {len(selected)} of {len(PROBLEMS)} problems",
        flush=True,
    )

    results: list[ConsistencyResult] = []
    for prob in PROBLEMS:
        if prob.slug not in selected:
            continue
        if args.only and prob.slug not in args.only:
            continue
        print(f"[consistency] {prob.slug} ...", flush=True)
        t0 = time.perf_counter()
        res = run_problem(prob, args.pybnf_cmd, args.bngpath)
        verdict = "PASS" if res.passed else "FAIL"
        print(
            f"[consistency]   -> {verdict} max_rel_obj_diff={res.max_rel_obj_diff:.2e} "
            f"({res.n_psets_compared} psets, {time.perf_counter() - t0:.0f}s) {res.notes}",
            flush=True,
        )
        results.append(res)
        args.results_out.parent.mkdir(parents=True, exist_ok=True)
        args.results_out.write_text(json.dumps([asdict(r) for r in results], indent=2))

    print(f"[consistency] wrote {args.results_out}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
