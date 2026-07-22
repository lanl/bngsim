#!/usr/bin/env python3
"""Emit ``_core`` GOLDEN references for the bng_parity suite (bngsim-only).

Golden references are the per-job bngsim reference data downstream consumers
(PyBNF, PyBioNetGen) regenerate through their own bridges to verify they
reproduce bngsim, *without* needing a legacy-BNG / RoadRunner / AMICI install
for comparison. This is pure bngsim output — no differ, no subprocess reference
side. (The timing+parity matrix — ``bng_ode_run.py`` / ``bng_stoch_run.py`` — is
what runs the reference engine and compares; this file only fingerprints bngsim.)

Contract (the four locked decisions)
-------------------------------------
* **D1 — artifact(s):** ``.gdat`` (named observables) is canonical and always
  golden when present; ``.scan`` (parameter_scan output) is golden alongside it
  when present; ``.cdat`` (species) is a FALLBACK used only when a job emits no
  ``.gdat`` and no ``.scan`` (so there is always something to fingerprint). A
  job emitting several files (multi-suffix / multi-phase models, e.g. an
  equilibration + main run on different time grids) is covered file-by-file:
  the job checksum spans all its files, and the fingerprint carries one
  per-file entry under ``artifacts``.
* **D2 — stochastic basis:** a single fixed ``seed=1`` trajectory (the
  manifest ``DEFAULT_SEED``), NOT an ensemble. The checksum is then a true
  byte-identity key within a pinned (version, platform, seed) cell — the
  strongest possible check for a consumer verifying its bridge on a pinned
  bngsim wheel. (bngsim's C++ re-seeds deterministically, so same model state
  + ``seed=1`` reproduces a trajectory exactly; see GH #10 seed semantics.)
* **D3 — full-trajectory subset:** only the hand-picked ``TRAJECTORY_ALLOWLIST``
  below (each entry with a one-line reason) gets its full ``(time, values,
  names)`` written under ``golden/trajectories/``; every other job gets
  checksum + fingerprint only. Allow-listed jobs that were not part of the
  consumed sweep are simply emitted without a trajectory (no error).
* **D4 — mechanic:** this lean separate script consumes an existing bngsim
  sweep ``--sweep-out`` (or runs its own bngsim-only sweep via
  ``parity_sweep.py --simulator bngsim --n-seeds 1``) and writes
  ``golden/golden.json`` + the trajectory subset.

BNG2.pl is required
-------------------
bngsim has no BNGL parser: for a BNGL/.net job it shells out to BNG2.pl to
generate the reaction network, then simulates that network in-process. So even
though golden generation never *compares* against legacy BNG, it still needs
``$BNGPATH`` (or ``$BNG2_PL``) set for network generation. No path is hardcoded.

Usage
-----
    export BNGPATH=/path/to/BioNetGen-2.9.3
    # run a fresh bngsim-only seed=1 sweep of the fast tier, then golden it:
    python parity_golden.py --include fast/ --workers 3
    # or reuse an existing parity_sweep bngsim output tree:
    python parity_golden.py --sweep-out runs/core/bngsim --no-run-sweep
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves

from _core import Golden, read_manifest, require_bng, versions, write_golden  # noqa: E402
from _core.fingerprint import CHECKSUM_SIGFIGS, _round_sig, golden_pair  # noqa: E402
from _core.versions import git_rev  # noqa: E402

SWEEP = HERE / "parity_sweep.py"
DEFAULT_SEED = 1  # manifest _meta.defaults.seed; the pinned stochastic trajectory
MAX_WORKERS_SAFE = 4  # this 6-core / 16 GB box froze before; never exceed.


def _scrub_home(p: str) -> str:
    """Replace the absolute home-dir prefix with ``~`` so provenance paths written
    into the committed golden ``_meta`` stay machine- and username-independent for
    public release. Applied to the reference paths only, never to fingerprints."""
    home = str(Path.home())
    return "~" + p[len(home) :] if p.startswith(home) else p


def _repo_relpath(p: str) -> str:
    """Repo-relative path for the committed golden ``_meta`` (machine- and
    layout-independent for public release); home-scrubbed absolute fallback for
    paths outside the repo tree (e.g. a scratch ``sweep_out``)."""
    try:
        return str(Path(p).resolve().relative_to(HERE.parents[1]))
    except ValueError:
        return _scrub_home(p)


# Hand-picked representative subset that gets a FULL committed trajectory (D3).
# Each entry: model_id -> reason. Spans single/multi-artifact, oscillatory, and
# (when those tiers are swept) ssa/nf. Entries not present in the consumed sweep
# are emitted without a trajectory rather than erroring, so the list can be
# aspirational ahead of the slow/glacial stochastic golden run.
TRAJECTORY_ALLOWLIST: dict[str, str] = {
    # --- deterministic (ode), realized by the fast tier --------------------
    "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl": "small single-action ODE; the canonical smoke trajectory",
    "fast/rulehub/Published/Hat2016/Hat_2016.bngl": "3-suffix multi-phase ODE (equil/irrad/relax) on DIFFERENT time grids — exercises the multi-artifact path",
    "fast/rulehub/Published/Pekalski2013/Pekalski_2013.bngl": "two named suffixes (ode1/ode2) -> two .gdat files of different lengths",
    "fast/rulehub/Published/PyBioNetGen/core/polynomialground/polynomial_ground.bngl": "wt/mut paired runs -> two equal-grid .gdat files",
    "fast/rulehub/Examples/biology/brusselatoroscillator/brusselator-oscillator.bngl": "stiff limit-cycle oscillator — sharp features sensitive to integrator divergence",
    "fast/rulehub/Examples/biology/e2frbcellcycleswitch/e2f-rb-cell-cycle-switch.bngl": "bistable cell-cycle switch — a long flat-then-jump trajectory",
    "fast/rulehub/Tutorials/nfkb/nfkb.bngl": "multiple None-suffix actions overwriting one .gdat (last-writer-wins) — medium network",
    "fast/rulehub/Published/Lang2024/Lang_2024.bngl": "recent published multi-action ODE model",
    "fast/rulehub/Examples/biology/sonichedgehoggradient/sonic-hedgehog-gradient.bngl": "4-action gradient model — several observables",
    # --- stochastic (ssa / nf), realized only when those tiers are swept ----
    "original/bngl_models/my_models/ssa/oscillatory_system.bngl": "small SSA oscillator at seed=1 — discrete-event trajectory",
    "original/bngl_models/my_models/ssa/gk_simple.bngl": "minimal Goldbeter-Koshland SSA at seed=1",
    "slow/rulehub/Tutorials/NativeTutorials/ABCssa/ABC_ssa.bngl": "canonical A+B->C SSA tutorial at seed=1",
    "slow/rulehub/Tutorials/NativeTutorials/CircadianOscillator/CircadianOscillator.bngl": "larger SSA circadian oscillator at seed=1",
    "glacial/rulehub/Published/Goldstein1980/blbr_heterogeneity_goldstein1980.bngl": "NFsim parameter_scan -> .scan artifact at seed=1 — exercises scan + nf paths",
}


def _load_parity_diff():
    """Import the sibling parity_diff module (for its tuned ``load_normalized``)."""
    spec = importlib.util.spec_from_file_location("parity_diff", HERE / "parity_diff.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _resolve_bngpath() -> str:
    """Resolve legacy-BNG (for BNGL network generation) via the shared resolver.

    bngsim has no BNGL parser, so a BNGL/.net job needs BNG2.pl to generate its
    network before bngsim simulates it in-process. ``_core.require_bng`` takes
    ``$BNG2_PL`` / ``$BNGPATH`` first and falls back to the BNG2.pl PyBioNetGen
    bundles (`uv sync --group parity`), so an env var is an override rather than
    the only way in; it exports ``$BNGPATH`` for the child sweep. Nothing is
    hardcoded, and a failure names every place it looked.
    """
    return str(
        require_bng(
            "golden generation never compares against legacy BNG, but bngsim still "
            "needs BNG2.pl to generate the reaction network from BNGL before it "
            "simulates in-process."
        ).root
    )


def _run_sweep(sweep_out: Path, jobs_path: Path, args) -> None:
    """Run a bngsim-only seed=1 sweep into ``sweep_out`` (D2: single fixed seed)."""
    cmd = [
        sys.executable,
        str(SWEEP),
        "--jobs",
        str(jobs_path),
        "--simulator",
        "bngsim",
        "--out",
        str(sweep_out),
        "--n-seeds",
        "1",  # D2: a single fixed seed=1 trajectory, no ensemble
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--regime",
        args.regime,
    ]
    if args.include:
        cmd += ["--include", args.include]
    if args.exclude:
        cmd += ["--exclude", args.exclude]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.models:
        cmd += ["--models", args.models]
    print(f"$ BNGPATH={os.environ['BNGPATH']} \\\n  {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(f"bngsim sweep failed (exit {proc.returncode})")


def _job_out_dir(sweep_out: Path, model_id: str, method: str) -> Path:
    """Locate a job's bngsim output dir.

    parity_sweep writes ``<sweep_out>/<relparent>/<stem>/{det|seedK}/``. A
    deterministic (ode) job lands in ``det``; a stochastic job's pinned seed=1
    run lands in ``seed1``. (When the consumed sweep was a multi-seed report run,
    ``seed1`` is still present — we use exactly that one.)

    The role is chosen from the manifest ``method``, but parity_sweep classifies
    the regime from the *model text* (``is_stochastic``), and the two can disagree
    for protocol/scan models — e.g. a ``parameter_scan`` the manifest tags ``ode``
    that the sweep nonetheless seeds and writes under ``seed1/``. So we prefer the
    method-implied role but fall back to whichever role the sweep actually
    produced, rather than silently dropping such a job from the golden set.
    """
    rel = Path(model_id)
    base = sweep_out / rel.parent / rel.stem
    primary = "det" if method != "stochastic" else f"seed{DEFAULT_SEED}"
    alt = f"seed{DEFAULT_SEED}" if primary == "det" else "det"
    primary_dir = base / primary
    if primary_dir.is_dir():
        return primary_dir
    alt_dir = base / alt
    if alt_dir.is_dir():
        return alt_dir
    return primary_dir  # neither exists; caller records out_dir_missing


def _run_status(out_dir: Path) -> str | None:
    """The sweep's recorded run status for a job ('ok'/'crash'/…), or None.

    parity_sweep writes a ``_run.log`` per unit with a ``# status=<s>`` header.
    A crashed/timed-out run can leave partial output (e.g. a multi-action model
    killed mid-write on its trailing action), and that partial trajectory is
    NOT byte-reproducible — a consumer whose timeout lands a step earlier gets a
    different file. So golden never fingerprints non-ok runs (see the crash
    guard in the main loop). None means the log is absent/unreadable (older
    sweeps): treat as ok and fall back to artifact-level checks.
    """
    log = out_dir / "_run.log"
    if not log.is_file():
        return None
    try:
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("# status="):
                return line.split("=", 1)[1].strip()
            if not line.startswith("#"):
                break  # header ends at the first non-comment line
    except OSError:
        return None
    return None


def _read_protocol_prefix(out_dir: Path) -> dict | None:
    """The per-model multi-phase replay verdict the sweep recorded, or None (GH #179).

    parity_sweep writes ``_provenance.json`` (``{engine, method, track,
    protocol_prefix}``) next to a bngsim run's artifacts. The ``protocol_prefix`` block
    carries ``dirty_carryover`` / ``replayed`` / ``segments`` / ``reinit_ics`` /
    ``replay_error`` — whether this job ran a faithful full-protocol replay (option-2),
    a single-segment direct run, or a graceful option-1 fallback. None when absent
    (a non-bngsim or older sweep, or a single-phase job that recorded no prefix).
    """
    prov = out_dir / "_provenance.json"
    if not prov.is_file():
        return None
    try:
        return json.loads(prov.read_text()).get("protocol_prefix")
    except (OSError, ValueError):
        return None


def _select_artifacts(out_dir: Path) -> tuple[list[Path], str]:
    """Per D1: ``.gdat`` + ``.scan`` if present, else ``.cdat`` fallback.

    Returns ``(files, reason)``; ``files`` empty with a reason when nothing is
    fingerprintable (job dir missing, or only non-data artifacts like ``.net``).
    """
    if not out_dir.is_dir():
        return [], "out_dir_missing"
    gdats = sorted(out_dir.glob("*.gdat"))
    scans = sorted(out_dir.glob("*.scan"))
    primary = gdats + scans
    if primary:
        return primary, ""
    cdats = sorted(out_dir.glob("*.cdat"))  # fallback only with no gdat/scan
    if cdats:
        return cdats, ""
    return [], "no_gdat_cdat_scan"


def _split(data: np.ndarray, names) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A loaded ``(n_time, 1+n_var)`` array -> (time, values, var-names)."""
    time = data[:, 0]
    values = data[:, 1:]
    if names is not None and len(names) == data.shape[1]:
        vnames = list(names[1:])
    else:  # positional fallback (header unavailable / width mismatch)
        vnames = [f"col{i}" for i in range(1, data.shape[1])]
    return time, values, vnames


def _build_from_files(files: list[Path], load_normalized):
    """Checksum/fingerprint/trajectories for a fixed file set, or (None, reason)."""
    per_file_checksum: dict[str, str] = {}
    artifacts: dict[str, dict] = {}
    trajectories: dict[str, dict] = {}
    for f in files:
        data, names, err = load_normalized(str(f))
        if err or data is None:
            return None, f"load_error:{f.name}:{err}"
        if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] == 0:
            return None, f"degenerate:{f.name}:shape={getattr(data, 'shape', None)}"
        time, values, vnames = _split(data, names)
        cs, fp = golden_pair(time, values, vnames)
        per_file_checksum[f.name] = cs
        artifacts[f.name] = fp
        # Store the trajectory rounded to the checksum's sig figs so the
        # committed reference matches exactly what the checksum hashes.
        trajectories[f.name] = {
            "time": _round_sig(np.asarray(time, float), CHECKSUM_SIGFIGS).tolist(),
            "names": vnames,
            "values": _round_sig(np.asarray(values, float), CHECKSUM_SIGFIGS).tolist(),
        }
    job_checksum = hashlib.sha256(
        "\n".join(f"{k}={per_file_checksum[k]}" for k in sorted(per_file_checksum)).encode()
    ).hexdigest()
    job_fingerprint = {
        "n_artifacts": len(artifacts),
        "kinds": sorted({Path(n).suffix.lstrip(".") for n in artifacts}),
        "artifacts": artifacts,
    }
    return (job_checksum, job_fingerprint, trajectories), None


def build_golden_for_job(out_dir: Path, load_normalized):
    """Build (checksum, fingerprint, trajectories) for one job, or (None, reason).

    ``checksum`` spans every selected file (sha256 over the sorted per-file
    checksums) so a byte divergence in *any* artifact is caught. ``fingerprint``
    nests one flat per-file fingerprint under ``artifacts`` — each directly
    comparable with ``_core.fingerprint.fingerprint_max_rel`` for the
    cross-platform fallback. ``trajectories`` maps filename -> rounded
    ``(time, names, values)`` for the allow-listed subset (caller decides whether
    to write it).

    Per D1, ``.cdat`` is the fallback when no ``.gdat``/``.scan`` is present. We
    also fall back to ``.cdat`` when the primary ``.gdat``/``.scan`` selection is
    *degenerate* (e.g. a time-only ``.gdat`` from a model with no observables):
    a network ODE/SSA run still writes a full species ``.cdat`` worth
    fingerprinting. (A network-free NFsim run writes no ``.cdat``, so such a job
    stays correctly skipped — there is genuinely nothing to fingerprint.)
    """
    files, reason = _select_artifacts(out_dir)
    if not files:
        return None, reason
    result, reason = _build_from_files(files, load_normalized)
    if result is None and reason.startswith("degenerate"):
        primary_names = {f.name for f in files}
        cdats = [c for c in sorted(out_dir.glob("*.cdat")) if c.name not in primary_names]
        if cdats:
            cdat_result, cdat_reason = _build_from_files(cdats, load_normalized)
            if cdat_result is not None:
                return cdat_result, None
    return result, reason


def _traj_relpath(model_id: str) -> str:
    """Suite-relative (under golden/) trajectory path for a model_id."""
    safe = model_id.replace("/", "__")
    if safe.endswith(".bngl"):
        safe = safe[: -len(".bngl")]
    return f"trajectories/{safe}.json"


def _downsample_traj(artifacts: dict, stride: int) -> dict:
    """Keep every ``stride``-th row of each artifact (the last row always kept)."""
    out = {}
    for fn, a in artifacts.items():
        n = len(a["time"])
        idx = list(range(0, n, stride))
        if idx and idx[-1] != n - 1:
            idx.append(n - 1)  # always preserve the final state
        out[fn] = {
            "time": [a["time"][i] for i in idx],
            "names": a["names"],
            "values": [a["values"][i] for i in idx],
        }
    return out


def _traj_payload(model_id, method, reason, artifacts, stride):
    payload = {
        "model_id": model_id,
        "method": method,
        "seed": DEFAULT_SEED if method == "stochastic" else None,
        "reason": reason,
        "sigfigs": CHECKSUM_SIGFIGS,
        "artifacts": _downsample_traj(artifacts, stride) if stride > 1 else artifacts,
    }
    if stride > 1:
        # The checksum/fingerprint in golden.json stay full-resolution; this
        # committed trajectory is a size-bounded debug reference only.
        payload["downsampled_stride"] = stride
    return json.dumps(payload, separators=(",", ":")) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs", default=str(HERE / "jobs.json"), help="_core manifest (jobs.json)")
    ap.add_argument(
        "--sweep-out",
        default="",
        help="bngsim sweep root to consume (the dir with the per-model trees + "
        "_summary.json). Defaults to <golden-dir>/../runs/golden/bngsim. Run a "
        "fresh seed=1 bngsim sweep into it unless --no-run-sweep.",
    )
    ap.add_argument(
        "--golden-dir",
        default=str(HERE / "golden"),
        help="Output dir for golden.json + trajectories/ (committed).",
    )
    ap.add_argument(
        "--no-run-sweep",
        action="store_true",
        help="Consume an existing --sweep-out instead of running a fresh sweep.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help=f"Concurrent sims for the sweep (clamped to {MAX_WORKERS_SAFE}; RAM-governed).",
    )
    ap.add_argument("--timeout", type=int, default=180, help="Per-model sweep timeout (s)")
    ap.add_argument(
        "--regime",
        choices=("all", "ode", "stochastic"),
        default="all",
        help="Restrict the sweep to deterministic (ode) or stochastic units — "
        "stage the fast ODE set apart from the slow, supervised stochastic set. "
        "(The fingerprint pass still covers every job already present in the "
        "shared --sweep-out, so golden.json accumulates across staged runs.)",
    )
    ap.add_argument("--include", default="", help="Substring path filter (subset / smoke)")
    ap.add_argument("--exclude", default="", help="Substring path filter — drop matches")
    ap.add_argument("--limit", type=int, default=0, help="Max jobs in the sweep (0=all)")
    ap.add_argument("--models", default="", help="Comma-separated basenames to restrict the sweep")
    ap.add_argument(
        "--max-traj-kb",
        type=int,
        default=950,
        help="Per-trajectory-file size cap (KB); rows are downsampled to fit "
        "(stays under the 1 MB check-added-large-files hook). golden.json keeps "
        "full-resolution checksum+fingerprint regardless.",
    )
    args = ap.parse_args()

    if args.workers > MAX_WORKERS_SAFE:
        print(f"WARN: --workers {args.workers} clamped to {MAX_WORKERS_SAFE} (memory safety).")
        args.workers = MAX_WORKERS_SAFE

    bngpath = _resolve_bngpath()
    jobs_path = Path(args.jobs).resolve()
    golden_dir = Path(args.golden_dir).resolve()
    sweep_out = (
        Path(args.sweep_out).resolve()
        if args.sweep_out
        else (HERE / "runs" / "golden" / "bngsim").resolve()
    )

    if not args.no_run_sweep:
        _run_sweep(sweep_out, jobs_path, args)
    elif not (sweep_out / "_summary.json").exists():
        sys.exit(
            f"ABORT: --no-run-sweep but no sweep found at {sweep_out} (_summary.json missing)."
        )

    # Fold the sweep's BNGsim-backend provenance + per-job engine audit into the
    # golden so it RECORDS which PyBioNetGen/bngsim produced it and which engine each
    # job ran on. Neither layer can silently substitute the legacy stack anymore: the
    # sweep drives bngsim directly (run_bngsim_job — GH #175), and since the pin moved
    # to 43b09a5 the bridge itself raises instead of quietly routing an un-inspectable
    # model to BNG2.pl under simulator='bngsim' (RuleWorld/PyBioNetGen#109, GH #4).
    # See bngsim_backend.py. Best-effort: older sweeps lack these keys.
    sweep_summary = {}
    try:
        summary_file = sweep_out / "_summary.json"
        if summary_file.exists():
            sweep_summary = json.loads(summary_file.read_text())
    except Exception:
        sweep_summary = {}

    _meta, jobs = read_manifest(jobs_path)
    pd = _load_parity_diff()

    golden: list[Golden] = []
    skipped: dict[str, str] = {}
    protocol_replay: dict[str, dict] = {}  # GH #179: per-model multi-phase replay verdict
    n_traj = 0
    traj_dir = golden_dir / "trajectories"
    for job in jobs:
        out_dir = _job_out_dir(sweep_out, job.model_id, job.method)
        status = _run_status(out_dir)
        if status is not None and status != "ok":
            # Crashed/timed-out run: any output is partial and not byte-
            # reproducible. Skip rather than fingerprint a fragile reference.
            skipped[job.model_id] = f"run_{status}"
            continue
        built, reason = build_golden_for_job(out_dir, pd.load_normalized)
        if built is None:
            # out_dir_missing just means this job wasn't in the consumed sweep
            # (filtered subset) — not a failure; record it quietly.
            skipped[job.model_id] = reason
            continue
        checksum, fingerprint, trajectories = built
        traj_rel = None
        if job.model_id in TRAJECTORY_ALLOWLIST:
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_rel = _traj_relpath(job.model_id)
            traj_path = golden_dir / traj_rel
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            reason = TRAJECTORY_ALLOWLIST[job.model_id]
            # Bound each committed trajectory below the large-file hook cap by
            # downsampling rows if needed (golden.json stays full-resolution).
            stride = 1
            text = _traj_payload(job.model_id, job.method, reason, trajectories, stride)
            cap_bytes = args.max_traj_kb * 1024
            while len(text.encode()) > cap_bytes and stride < 64:
                stride *= 2
                text = _traj_payload(job.model_id, job.method, reason, trajectories, stride)
            if len(text.encode()) > cap_bytes:
                print(
                    f"WARN: {job.model_id} trajectory still > {args.max_traj_kb}KB at stride {stride}"
                )
            elif stride > 1:
                print(
                    f"  downsampled {job.model_id} trajectory (stride {stride}) to fit {args.max_traj_kb}KB cap"
                )
            traj_path.write_text(text)
            n_traj += 1
        golden.append(
            Golden(
                model_id=job.model_id,
                method=job.method,
                checksum=checksum,
                fingerprint=fingerprint,
                trajectory=traj_rel,
            )
        )
        # GH #179: carry this job's multi-phase replay verdict into the golden _meta.
        pp = _read_protocol_prefix(out_dir)
        if pp is not None:
            protocol_replay[job.model_id] = pp

    golden.sort(key=lambda g: g.model_id)
    missing_only = sum(1 for r in skipped.values() if r == "out_dir_missing")
    real_skips = {k: v for k, v in skipped.items() if v != "out_dir_missing"}
    # Scrub the home-dir prefix off the backend's ``bionetgen_path`` (the one
    # machine-specific path carried up from the gitignored sweep _summary into the
    # committed golden _meta) — keep the rest of the backend provenance verbatim.
    backend = sweep_summary.get("bngsim_backend")
    if backend and backend.get("bionetgen_path"):
        backend = {**backend, "bionetgen_path": _scrub_home(backend["bionetgen_path"])}
    meta = {
        "suite": "bng_parity",
        "kind": "golden",
        "jobs_manifest": _repo_relpath(str(jobs_path)),
        "sweep_out": _repo_relpath(str(sweep_out)),
        "bngpath": _scrub_home(bngpath),
        "git_rev": git_rev(str(HERE)),
        "versions": versions.stamp(),  # bngsim + python + platform (no ref engine)
        # BNGsim-backend provenance + per-job engine audit, carried up from the
        # sweep's _summary.json. ``bionetgen_commit`` pins the bionetgen+bngsim that
        # produced this golden (reproduce with requirements-pybionetgen.txt). The
        # sweep now drives GENUINE bngsim directly (run_bngsim_job), so
        # ``engine_audit.by_track`` records the bngsim engine each job used and
        # ``engine_audit.unsupported`` lists any job bngsim could not run (skipped,
        # NOT silently run on the legacy stack) — see GH #175.
        "bngsim_backend": backend,
        "bionetgen_commit": (backend or {}).get("bionetgen_commit"),
        "engine_audit": sweep_summary.get("engine_audit"),
        # GH #179: per-model multi-phase replay verdict — for each dirty_carryover job,
        # whether bngsim ran a faithful full-protocol replay (``replayed=True`` + the
        # segment count, ``reinit_ics`` for a #181 seed-species re-init) or fell back to
        # the option-1 single-segment best-effort (``replayed=False`` + ``replay_error``).
        # The ``protocol_replay_summary`` rolls these up; single-phase jobs are absent.
        "protocol_replay_summary": {
            "models": len(protocol_replay),
            "dirty_carryover": sum(
                1 for v in protocol_replay.values() if v.get("dirty_carryover")
            ),
            "replayed": sum(1 for v in protocol_replay.values() if v.get("replayed")),
            "fallback": sum(
                1
                for v in protocol_replay.values()
                if v.get("dirty_carryover") and not v.get("replayed")
            ),
            "reinit_ics": sum(int(v.get("reinit_ics") or 0) for v in protocol_replay.values()),
        },
        "protocol_replay": dict(sorted(protocol_replay.items())),
        "seed": DEFAULT_SEED,
        "checksum_sigfigs": CHECKSUM_SIGFIGS,
        "n_golden": len(golden),
        "n_manifest_jobs": len(jobs),
        "n_trajectories": n_traj,
        "n_not_in_sweep": missing_only,
        "contract": (
            "D1: .gdat canonical + .scan when present; .cdat fallback only with no "
            "gdat/scan. Multi-file jobs: checksum spans all files, fingerprint nests "
            "one flat per-file entry under 'artifacts' (each fingerprint_max_rel-"
            "comparable). D2: stochastic = single fixed seed=1 trajectory (byte-"
            "identity within a pinned version/platform/seed cell). D3: full "
            "trajectories only for the TRAJECTORY_ALLOWLIST subset, rounded to "
            f"{CHECKSUM_SIGFIGS} sig figs. Consumers regenerate through their bridge "
            "on a pinned bngsim and match the checksum byte-for-byte; the per-file "
            "fingerprint is the cross-platform tolerance fallback."
        ),
    }
    if real_skips:
        meta["skipped"] = dict(sorted(real_skips.items())[:50])
        print(f"WARN: {len(real_skips)} swept job(s) produced no fingerprintable artifact.")

    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path = golden_dir / "golden.json"
    write_golden(golden_path, golden, meta=meta)
    print(f"\ngolden: {golden_path}")
    print(f"  golden records:   {len(golden)} / {len(jobs)} manifest jobs")
    print(f"  trajectories:     {n_traj} (allow-listed)")
    print(f"  not in sweep:     {missing_only}")
    print(f"  no-artifact skips: {len(real_skips)}")
    print(f"  generated:        {_dt.datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
