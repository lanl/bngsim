#!/usr/bin/env python3
"""Generated models through the differential checks, and each finding shrunk to
a reproducer (issue #1306).

For each seed and format, ``_generate.generate`` draws a model (a small
reaction network whose rate laws are random trees over bngsim's expression
grammar), and ``_differential.check_all`` runs it through every path that
should agree. A check that reads DIFF, or EXCEPTION where the base ran, is a
finding: ``_generate.shrink`` cuts the model down while the same check still
finds the same thing, and the smallest model it reaches is written out as a
ready reproducer, beside a JSON note of what it shows.

A model whose base does not run is no finding (its law left the domain, or a
refusal by name), and is counted apart. A seed is its model, so a finding is
reproduced from its seed alone.

    python parity_checks/differential/differential_generate.py \
        --seeds 0:200 --formats net,sbml --out runs/generated.json \
        --repro-dir runs/reproducers
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "amici_parity"))

import _differential as D  # noqa: E402
import _generate as G  # noqa: E402

DEFAULT_TIMEOUT = 600.0
DEFAULT_SHRINK_BUDGET = 300
# Generated laws can switch on the clock and on rateOf at every step (seed 39 in
# SBML crawls): a fifth of the corpus's budget keeps such a seed from a timeout.
DEFAULT_MAX_STEPS = 20_000
FINDING = ("diff", "exception")


def default_checks() -> tuple[str, ...]:
    # The locale check needs a comma-decimal locale; the generated models read
    # no file a locale could misread that the corpus does not.
    return tuple(c for c in D.CHECKS if c != "locale")


def _job(model_id: str, path: Path, spec: dict) -> D.Job:
    return D.Job(
        model_id,
        path,
        0.0,
        spec["t_end"],
        spec["n_points"],
        rtol=spec["rtol"],
        atol=spec["atol"],
        max_steps=spec["max_steps"],
    )


def _kind(v: D.Verdict) -> tuple[str, str]:
    """What a finding is, for the shrinker to keep: its status, and for an
    exception the class it raised."""
    if v.status == "exception":
        return v.status, v.note.split(":", 1)[0]
    return v.status, ""


def find(model: G.Model, spec: dict, workdir: Path, checks) -> dict[str, D.Verdict] | None:
    """Every check's verdict on ``model``; None where its base does not run."""
    path = G.write(model, workdir / f"m{time.perf_counter_ns()}")
    verdicts = D.check_all(_job(path.stem, path, spec), None, tuple(checks))
    if any(v.status == "no_base" for v in verdicts.values()):
        return None
    return verdicts


def investigate(spec: dict) -> dict:
    """One seed: its model through the checks, each finding shrunk and written
    out. In this process; :func:`_worker` runs it in a child."""
    t0 = time.perf_counter()
    res: dict = {"seed": spec["seed"], "format": spec["format"]}
    try:
        model = G.generate(spec["seed"], spec["format"], spec["depth"])
        res["size"] = G.size(model)
        with tempfile.TemporaryDirectory(prefix="differential_generate_") as d:
            work = Path(d)
            verdicts = find(model, spec, work, spec["checks"])
            if verdicts is None:
                res["status"] = "no_base"
            else:
                res["status"] = "done"
                res["rows"] = {c: v.status for c, v in verdicts.items()}
                res["findings"] = []
                for check, v in verdicts.items():
                    if v.status not in FINDING:
                        continue
                    want = _kind(v)

                    def holds(m, check=check, want=want):
                        got = find(m, spec, work, (check,))
                        return got is not None and _kind(got[check]) == want

                    # The model as found, written before shrinking, which this
                    # seed's cap also covers: a seed killed mid-shrink leaves it.
                    name = f"{spec['format']}_{spec['seed']}_{check}"
                    found = G.write(model, Path(spec["repro_dir"]) / name)
                    found.with_suffix(".json").write_text(
                        json.dumps(
                            {
                                "seed": spec["seed"],
                                "check": check,
                                "status": v.status,
                                "note": v.note[:500],
                                "size": G.size(model),
                                "shrunk_size": None,
                                "reproducer": str(found),
                            },
                            indent=1,
                        )
                    )
                    small = G.shrink(model, holds, spec["shrink_budget"])
                    final = find(small, spec, work, (check,))
                    repro = G.write(small, Path(spec["repro_dir"]) / name)
                    res["findings"].append(
                        {
                            "check": check,
                            "status": v.status,
                            "note": v.note[:500],
                            "size": G.size(model),
                            "shrunk_size": G.size(small),
                            "reproducer": str(repro),
                            "reproducer_note": final[check].note[:500] if final else None,
                        }
                    )
                    Path(repro).with_suffix(".json").write_text(
                        json.dumps({"seed": spec["seed"], **res["findings"][-1]}, indent=1)
                    )
    except Exception as e:  # noqa: BLE001 - a generator or shrinker failure is this seed's
        res["status"] = "error"
        res["why"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    res["wall_sec"] = round(time.perf_counter() - t0, 3)
    return res


def _worker(spec: dict, q) -> None:
    import logging
    import warnings

    # A generated law trips bngsim's advisories (a sensitivity declined, a
    # crossing moved) as often as not; the verdicts carry what matters.
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    q.put(investigate(spec))


def _seeds(text: str) -> list[int]:
    lo, _, hi = text.partition(":")
    return list(range(int(lo), int(hi))) if hi else [int(lo)]


def run(specs: list[dict], workers: int, timeout: float) -> list[dict]:
    """Each spec in its own process, at most ``workers`` at once, each capped
    at ``timeout`` seconds (a law the integrator crawls through is a seed's
    TIMEOUT, not the run's)."""
    ctx = mp.get_context("spawn")
    pending, running, out = list(specs), [], []
    while pending or running:
        while pending and len(running) < workers:
            spec = pending.pop(0)
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(spec, q))
            p.start()
            running.append((spec, p, q, time.monotonic()))
        time.sleep(0.05)
        for item in list(running):
            spec, p, q, t0 = item
            done = not q.empty()
            if done or not p.is_alive() or time.monotonic() - t0 > timeout:
                res = q.get() if not q.empty() else None
                if res is None:
                    # Still running at the cap is a timeout; gone without a
                    # result is a crash (a segfault in bngsim), the finding a
                    # generator is for, and not to be read as slowness.
                    alive = p.is_alive()
                    p.kill()
                    p.join(timeout=5)
                    res = {
                        "seed": spec["seed"],
                        "format": spec["format"],
                        "status": "timeout" if alive else "dead",
                    }
                    if not alive:
                        res["exitcode"] = p.exitcode
                p.join(timeout=5)
                out.append(res)
                running.remove(item)
                tag = res["status"]
                n = len(res.get("findings", []))
                print(
                    f"  {spec['format']} seed {spec['seed']}: {tag}{f', {n} finding(s)' if n else ''}",
                    flush=True,
                )
    return sorted(out, key=lambda r: (r["format"], r["seed"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", default="0:50", help="lo:hi, or one seed")
    ap.add_argument("--formats", default="net,sbml")
    ap.add_argument("--checks", default=",".join(default_checks()))
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--n-points", type=int, default=21)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--shrink-budget", type=int, default=DEFAULT_SHRINK_BUDGET)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="per run: a law the integrator crawls through fails fast, as its base's",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per seed (s)")
    ap.add_argument("--out", default=str(HERE / "runs" / "generated.json"))
    ap.add_argument("--repro-dir", default=str(HERE / "runs" / "reproducers"))
    args = ap.parse_args(argv)

    checks = [c for c in args.checks.split(",") if c]
    unknown = sorted(set(checks) - set(D.CHECKS))
    if unknown:
        ap.error(f"unknown checks {unknown}; known: {', '.join(D.CHECKS)}")
    formats = [f for f in args.formats.split(",") if f]
    bad = sorted(set(formats) - set(G.FORMATS))
    if bad:
        ap.error(f"unknown formats {bad}; known: {', '.join(G.FORMATS)}")
    Path(args.repro_dir).mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "seed": s,
            "format": f,
            "checks": checks,
            "depth": args.depth,
            "t_end": args.t_end,
            "n_points": args.n_points,
            "rtol": args.rtol,
            "atol": args.atol,
            "shrink_budget": args.shrink_budget,
            "max_steps": args.max_steps,
            "repro_dir": args.repro_dir,
        }
        for f in formats
        for s in _seeds(args.seeds)
    ]
    t0 = time.perf_counter()
    results = run(specs, args.workers, args.timeout)
    findings = [
        dict(seed=r["seed"], format=r["format"], **f)
        for r in results
        for f in r.get("findings", [])
    ]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    report = {
        "meta": {
            "seeds": args.seeds,
            "formats": formats,
            "checks": checks,
            "depth": args.depth,
            "window": [0.0, args.t_end, args.n_points],
            "tolerances": [args.rtol, args.atol],
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "counts": counts,
            "n_findings": len(findings),
        },
        "findings": findings,
        "models": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"report: {args.out}  {counts}  findings: {len(findings)}")
    for f in findings:
        print(
            f"  {f['format']} seed {f['seed']} {f['check']} {f['status']}: size {f['size']} -> {f['shrunk_size']}  {f['reproducer']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
