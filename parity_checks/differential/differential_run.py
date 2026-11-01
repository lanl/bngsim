#!/usr/bin/env python3
"""Each model through every path that should give the same answer -> a ``_core``
report (issue #1257).

The checks are ``_differential.py``'s: a sensitivity run's state against a plain
run's, the FD Jacobian against the analytic one, KLU against the dense solver,
compiled against interpreted, cc against MIR, a comma-decimal locale against C,
8x the output points against the base's grid, ``steady_state()`` (integrated
and Newton-polished) against a run to equilibrium, and four rules for bugs whose
answers are right (a sensitivity run refused, a sparsity pattern that misses
entries, analytic Jacobian entries that differences of the right-hand side
contradict (#1308), a backend other than the one asked for).
Four more transform the model file itself, where the answer is known to follow,
which the loader reads a second way (#1305, ``_transforms.py``): its species and
reactions reordered, its names renamed, each reaction split in two at half the
rate, and, where it does not read the clock, every rate and the window rescaled.
Three more reuse what every other check makes fresh (#1307): a run stopped and
continued, a second run on a Simulator or a Model a first one ran, and a batch
on threads against the same points one at a time.
One row per (model, check), its ``method`` the check:

    the path agrees with its reference .............. PASS
    it does not, past the bound ..................... DIFF
    it declined by type (a refusal by name) ......... UNSUPPORTED
    the path does not apply here (no KLU, no MIR),
      or the model does not load or run at all ....... SKIP
    it raised where the base ran .................... EXCEPTION
    it did not finish within the model's cap, or its
      compile within bngsim's own limit ............. TIMEOUT

The baseline (``nightly/baselines/differential.json``) holds tonight's known
findings, so a night alerts on what changed: a model whose paths agreed and do
not, a sensitivity run that ran and is refused, a new crash.

The corpus is the models other suites already run, each over its own window and
at its own tolerances:

* ``fd_sens/fd_sens_jobs.json``: the 53 committed ``.net`` models;
* ``rr_parity/ode_jobs.json``: the BioModels the nightly's corpus job fetched,
  those present and under ``--max-bytes``, at most ``--max-points`` samples;
* ``rr_parity/ssa_jobs.json`` (``--ssa-jobs``): rr_ssa's models, as SSA jobs at
  their own window and seed, at most ``--max-reps`` replicates (#1310);
* ``--nf-archive``: bng_nf's models from its reference archive, each entry's
  BNG-XML and recorded run, as NF jobs for NFsim against RuleMonkey (#1312);
* ``--net-dir``: every ``.net`` under a directory, and ``--model-list``: a file
  of model paths, each over ``--default-span``.

The locale check runs each model under a comma-decimal ``LC_NUMERIC`` in a child
process. Without one, every model would read "fails under de_DE", so the run
stops at once (exit 2) unless ``--no-locale`` leaves the check out; the nightly
generates ``de_DE.UTF-8`` first.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "amici_parity"))

import _amici_sens as asens  # noqa: E402
import _differential as D  # noqa: E402
from _core import JobResult, Outcome, tally, versions, write_report  # noqa: E402
from _core.versions import git_rev  # noqa: E402

FD_SENS_JOBS = HERE.parent / "fd_sens" / "fd_sens_jobs.json"
RR_JOBS = HERE.parent / "rr_parity" / "ode_jobs.json"
# The SSA arm's corpus (#1310): rr_ssa's jobs, each at its own window and seed.
SSA_JOBS = HERE.parent / "rr_parity" / "ssa_jobs.json"
DEFAULT_MAX_REPS = 20
REFERENCE_ENGINE = "bngsim_paths"
DEFAULT_TIMEOUT = 600.0
#: The share of a model's cap its checks may use before the pool kills it: the
#: rest is the margin for the run that ends past the deadline, and the reply.
CAP_SHARE = 0.9
DEFAULT_MAX_SPECIES = 200

_OUTCOME = {
    "pass": Outcome.PASS,
    "diff": Outcome.DIFF,
    "unsupported": Outcome.UNSUPPORTED,
    "skip": Outcome.SKIP,
    # the model does not load, or its plain run does not complete: no reference
    # (rr_ode, bng_ode and sbml_semantic own that); a PASS turning into one alerts
    "no_base": Outcome.SKIP,
    "exception": Outcome.EXCEPTION,
    "timeout": Outcome.TIMEOUT,
    "dead": Outcome.EXCEPTION,
}


def fd_sens_jobs(path: Path) -> list[D.Job]:
    out = []
    for j in json.loads(path.read_text())["jobs"]:
        out.append(
            D.Job(
                j["model_id"],
                REPO / j["path"],
                float(j.get("t_start", 0.0)),
                float(j["t_end"]),
                int(j["n_points"]),
                float(j.get("rtol", 1e-8)),
                float(j.get("atol", 1e-10)),
                int(j.get("max_steps", 100_000)),
            )
        )
    return out


def rr_jobs(path: Path, max_bytes: int, max_points: int) -> list[D.Job]:
    """The SBML jobs whose model is on disk and no larger than ``max_bytes``."""
    out = []
    for j in json.loads(path.read_text())["jobs"]:
        model = path.parent / j["model"]
        if not model.is_file() or (max_bytes and model.stat().st_size > max_bytes):
            continue
        p = j["params"]
        out.append(
            D.Job(
                j["model_id"],
                model,
                float(p.get("initial_time", p["t_start"])),
                float(p["t_end"]),
                min(int(p["n_points"]), max_points),
                float(p["rtol"]),
                float(p["atol"]),
            )
        )
    return out


def ssa_jobs(path: Path, max_bytes: int, max_points: int, max_reps: int) -> list[D.Job]:
    """rr_ssa's SBML jobs whose model is on disk and no larger than
    ``max_bytes``, as SSA jobs: their own window and seed, at most
    ``max_points`` samples and ``max_reps`` replicates."""
    out = []
    for j in json.loads(path.read_text())["jobs"]:
        model = path.parent / j["model"]
        if not model.is_file() or (max_bytes and model.stat().st_size > max_bytes):
            continue
        p = j["params"]
        out.append(
            D.Job(
                j["model_id"],
                model,
                float(p.get("initial_time", p["t_start"])),
                float(p["t_end"]),
                min(int(p["n_points"]), max_points),
                method="ssa",
                n_rep=min(int(p.get("n_rep", max_reps)), max_reps),
                seed=int(p.get("seed", 1)),
            )
        )
    return out


def nf_jobs(archive: Path, max_reps: int, xml_dir: Path) -> list[D.Job]:
    """bng_nf's jobs, from its reference archive (``reference_archive.py fetch
    bng_nf``): each entry holds the BNG-XML BNG2.pl wrote for the model and the
    run it was recorded for, so the NF arm runs the NF track's own models and
    windows without BNG2.pl. Written to ``xml_dir``. An entry recorded as a
    multi-segment protocol is left out: one segment is not its run."""
    import numpy as np

    out = []
    for f in sorted(Path(archive).glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
        run, text = meta.get("run") or {}, (meta.get("build") or {}).get("artifact_text")
        if meta.get("track") != "nf" or not text or run.get("mode") != "single_segment":
            continue
        xml = xml_dir / f"{f.stem}.xml"
        xml.write_text(text)
        sim = {"block_same_complex_binding": bool(meta.get("nf_bscb", True))}
        if run.get("gml") is not None:
            sim["molecule_limit"] = int(run["gml"])
        out.append(
            D.Job(
                meta["model_id"],
                xml,
                float(run["t_start"]),
                float(run["t_end"]),
                int(run["n_steps"]) + 1,
                sim=sim,
                method="nf",
                n_rep=min(int(meta.get("n_rep", max_reps)), max_reps),
                seed=int(meta.get("seed_base", 1)),
            )
        )
    return out


def _job_of(p: Path, span: tuple[float, float]) -> D.Job:
    return D.Job(str(p.relative_to(REPO) if p.is_relative_to(REPO) else p), p, *span, 21)


def net_dir_jobs(root: Path, span: tuple[float, float]) -> list[D.Job]:
    return [_job_of(p, span) for p in sorted(root.rglob("*.net"))]


def list_jobs(listing: Path, span: tuple[float, float]) -> list[D.Job]:
    """A ``.net`` or SBML file per line, each over ``span``."""
    return [_job_of(Path(x).resolve(), span) for x in listing.read_text().split() if x]


def _row(v) -> dict:
    return {"status": v.status, "value": v.value, "note": v.note, "extra": v.extra}


def _keep(spec: dict, check: str, v) -> None:
    """Each verdict as it is reached, to ``spec["partial"]``: a worker the pool
    kills at the cap (one compiling C past the deadline) leaves the runner every
    row it finished, and only the rest TIMEOUT."""
    path = spec.get("partial")
    if not path:
        return
    path = Path(path)
    rows = json.loads(path.read_text()) if path.is_file() else {}
    rows[check] = _row(v)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, default=str))
    tmp.replace(path)


def _worker(spec: dict, q) -> None:
    """One model's checks in a child process; puts one result dict on ``q``."""
    t0 = time.perf_counter()
    res: dict = {"key": spec["key"]}
    try:
        import warnings

        warnings.simplefilter("ignore")
        job = D.Job.from_dict(spec["job"])
        try:
            # A network-free model has no species list: its BNG-XML is the sessions'.
            n = 0 if job.method == "nf" else len(D.load(job.path).species_names)
        except Exception as e:  # noqa: BLE001 - a refusal at load is the loader's to judge
            res["status"] = "no_base"
            res["why"] = f"does not load: {type(e).__name__}: {str(e).splitlines()[0][:300]}"
            res["wall_sec"] = round(time.perf_counter() - t0, 3)
            q.put(res)
            return
        if n > spec["max_species"]:
            res["status"] = "over"
            res["why"] = f"{n} species, over the cap of {spec['max_species']}"
        else:
            # Short of the cap, at which the pool kills the worker and every row
            # of the model with it: past the deadline a check is its own TIMEOUT.
            deadline = time.monotonic() + spec["timeout"] * CAP_SHARE - (time.perf_counter() - t0)
            verdicts = D.check_all(
                job,
                spec["locale"],
                tuple(spec["checks"]),
                deadline,
                lambda c, v: _keep(spec, c, v),
            )
            res["rows"] = {c: _row(v) for c, v in verdicts.items()}
            res["status"] = "done"
    except Exception as e:  # noqa: BLE001 - a model that does not load is its rows' EXCEPTION
        res["status"] = "exception"
        res["why"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    res["wall_sec"] = round(time.perf_counter() - t0, 3)
    q.put(res)


def _value(v):
    return None if v is None or v != v or v in (float("inf"), float("-inf")) else float(v)


def to_results(spec: dict, r: dict, stamp: dict) -> list[JobResult]:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    rows = r.get("rows")
    if rows is None:  # the model as a whole: over the cap, raised, timed out or died
        status = {"over": "skip"}.get(r.get("status"), r.get("status", "exception"))
        note = r.get("why") or {
            "timeout": f"cap of {spec['timeout']:.0f} s",
            "dead": f"the worker died (exit code {r.get('exitcode')})",
        }.get(status, "")
        # What a worker killed at the cap, or one that died or raised, finished
        # stands (_keep): only the rest take the model's outcome.
        kept = spec.get("partial") and Path(spec["partial"])
        done = json.loads(kept.read_text()) if kept and kept.is_file() else {}
        rows = {
            c: done.get(c) or {"status": status, "value": None, "note": note, "extra": {}}
            for c in spec["checks"]
        }
    out = []
    for check, row in rows.items():
        out.append(
            JobResult(
                model_id=spec["job"]["model_id"],
                method=check,
                reference_engine=REFERENCE_ENGINE,
                outcome=_OUTCOME.get(row["status"], Outcome.EXCEPTION).value,
                metric="max_rel_err",
                value=_value(row["value"]),
                exception=row["note"] if row["status"] != "pass" else "",
                wall_sec=r.get("wall_sec"),
                timestamp=now,
                versions=stamp,
                comment=row["note"] if row["status"] == "pass" else "",
                extra={k: v for k, v in (row.get("extra") or {}).items() if v is not None},
            )
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fd-sens-jobs", default=str(FD_SENS_JOBS))
    ap.add_argument("--rr-jobs", default=str(RR_JOBS), help="'' leaves the BioModels out")
    ap.add_argument("--ssa-jobs", default=str(SSA_JOBS), help="'' leaves the SSA arm out")
    ap.add_argument("--max-reps", type=int, default=DEFAULT_MAX_REPS, help="SSA, NF replicates")
    ap.add_argument(
        "--nf-archive", default="", help="bng_nf's reference archive: the NF arm, on its models"
    )
    ap.add_argument("--net-dir", action="append", default=[], help="every .net under it")
    ap.add_argument("--model-list", action="append", default=[], help="a file of model paths")
    ap.add_argument("--default-span", type=float, nargs=2, default=(0.0, 10.0))
    ap.add_argument("--max-bytes", type=int, default=1_000_000, help="largest SBML file")
    ap.add_argument("--max-points", type=int, default=101)
    ap.add_argument("--max-species", type=int, default=DEFAULT_MAX_SPECIES)
    ap.add_argument("--out", default=str(HERE / "runs" / "report_differential.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per model (s)")
    ap.add_argument("--checks", default=",".join(D.CHECKS))
    ap.add_argument("--no-locale", action="store_true", help="leave the locale check out")
    ap.add_argument("--models", default="", help="comma-separated model_id substrings")
    args = ap.parse_args(argv)
    if args.max_reps < 1:
        ap.error(f"--max-reps must be at least 1, got {args.max_reps}")

    checks = [c for c in args.checks.split(",") if c and not (args.no_locale and c == "locale")]
    unknown = sorted(set(checks) - set(D.CHECKS))
    if unknown:
        ap.error(f"unknown checks {unknown}; known: {', '.join(D.CHECKS)}")
    locale = None
    if "locale" in checks:
        locale = D.comma_locale()
        if locale is None:
            print(
                "differential: no comma-decimal locale (tried "
                f"{', '.join(D.COMMA_LOCALES)}). Every model would read 'fails under "
                "de_DE'. Generate one (locale-gen de_DE.UTF-8) or pass --no-locale.",
                file=sys.stderr,
            )
            return 2

    jobs = fd_sens_jobs(Path(args.fd_sens_jobs)) if args.fd_sens_jobs else []
    if args.rr_jobs:
        jobs += rr_jobs(Path(args.rr_jobs), args.max_bytes, args.max_points)
    if args.ssa_jobs:
        jobs += ssa_jobs(Path(args.ssa_jobs), args.max_bytes, args.max_points, args.max_reps)
    if args.nf_archive:
        xml_dir = Path(tempfile.mkdtemp(prefix="differential-nf-"))
        atexit.register(shutil.rmtree, xml_dir, ignore_errors=True)
        jobs += nf_jobs(Path(args.nf_archive), args.max_reps, xml_dir)
    for d in args.net_dir:
        jobs += net_dir_jobs(Path(d).resolve(), tuple(args.default_span))
    for f in args.model_list:
        jobs += list_jobs(Path(f), tuple(args.default_span))
    seen, unique = set(), []
    for j in jobs:  # the first manifest to name a model keeps it, per method
        ids = {(j.method, j.model_id), (j.method, j.path.resolve())}
        if not ids & seen:
            seen |= ids
            unique.append(j)
    if args.models:
        subs = [s.strip() for s in args.models.split(",") if s.strip()]
        unique = [j for j in unique if any(s in j.model_id for s in subs)]
    # Each model's verdicts as its worker reaches them (_keep), for one the pool kills.
    kept = Path(tempfile.mkdtemp(prefix="differential-rows-"))
    atexit.register(shutil.rmtree, kept, ignore_errors=True)
    specs = [
        {
            "key": j.model_id if j.method == "ode" else f"{j.model_id}:{j.method}",
            "job": j.as_dict(),
            # A job runs the checks its method has: an SSA job the SSA arm's.
            "checks": list(D.checks_for(j, checks)),
            "locale": locale,
            "max_species": args.max_species,
            "timeout": args.timeout,
            "partial": str(kept / f"{i}.json"),
        }
        for i, j in enumerate(unique)
    ]
    print(f"differential: {len(specs)} models x {len(checks)} checks; locale {locale}")
    t0 = time.perf_counter()

    def on_done(finished, total, res):
        rows = res.get("rows") or {}
        bad = [c for c, v in rows.items() if v["status"] in ("diff", "exception")]
        print(
            f"  [{finished}/{total}] {res.get('status', '?'):9} {res['key']} {' '.join(bad)}",
            flush=True,
        )

    raw = asens.schedule(
        specs, _worker, workers=args.workers, timeout_of=lambda s: s["timeout"], on_done=on_done
    )
    stamp = versions.stamp(REFERENCE_ENGINE)
    # #1309: a model whose answer is contested, by its stable row or by a
    # DIFF whose third arms split, listed by name.
    contested = sorted(
        spec["job"]["model_id"]
        for spec, r in zip(specs, raw, strict=False)
        if D.contested(r.get("rows") or {})
    )
    results = [x for spec, r in zip(specs, raw, strict=False) for x in to_results(spec, r, stamp)]
    meta = {
        "suite": "differential",
        "reference_engine": REFERENCE_ENGINE,
        "regime": "path_agreement",
        "git_rev": git_rev(str(HERE)),
        "versions": stamp,
        "tally": tally(r.outcome for r in results),
        "contested": contested,
        "n_jobs": len(results),
        "n_models": len(specs),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "config": {
            "checks": checks,
            "locale": locale,
            "path_rtol_factor": D.PATH_RTOL_FACTOR,
            "drift_factor": D.DRIFT_FACTOR,
            "max_species": args.max_species,
            "max_points": args.max_points,
        },
        "hardware": asens.hardware_info(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(out, results, meta=meta)
    print(f"report: {out}  {meta['tally']}")
    if contested:
        print(f"contested ({len(contested)}): {', '.join(contested)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
