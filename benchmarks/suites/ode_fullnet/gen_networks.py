#!/usr/bin/env python3
"""Phase 1 of the full-network ODE timing benchmark: generate + cache the FULL
reaction network for every BNGL ODE model in the bng_parity corpus.

WHY THIS EXISTS
---------------
The parity harness (``parity_checks/bng_parity``) validates BNGsim against
BioNetGen ``run_network``. To keep those comparisons cheap it *caps* some
networks via ``overrides.REHAB`` (``max_iter=>3``), so e.g. ``fceri_fyn`` is
run as a 12-species toy instead of its full 1281-species network. Capping is
fine for *validation* (both engines run the same network), but it corrupts a
*timing* study: the large-sparse-stiff regime — exactly where BNGsim's KLU
sparse solver beats a dense solver — is thrown away.

This benchmark re-times the corpus at FULL network size. It is split in two
phases so the expensive, uncertain part (network generation) is done once and
cached:

  * Phase 1 (this script): generate each model's full ``.net`` and cache it
    under ``nets/`` (git-ignored, regenerable). Records N, reactions, and the
    ``generate_network`` directive used, in ``nets_manifest.json``.
  * Phase 2 (``run_timing.py``): load each cached ``.net`` and time BNGsim warm
    integration vs. ``run_network``, in one consistent environment.

NETGEN POLICY ("as originally formulated")
------------------------------------------
For every model we use its OWN ``generate_network(...)`` directive
(``_bng_common._model_gen_network`` — preserves the author's ``max_stoich`` /
``max_agg`` caps), falling back to a bare uncapped ``generate_network({overwrite
=>1})``. We do NOT apply the parity ``max_iter=>3`` cap. A model that shipped
without a runnable ODE protocol (a parity ``REHAB`` entry) gets the *uncapped*
injected protocol (``overrides._net_rehab("ode", max_iter=None)``) so it still
expands to a full, bounded network. This exactly reproduces the network the
Jacobian characterization used (which is why S4 has ``fceri_fyn`` at N=1281).

RESUMABLE: a model already recorded ``ok`` (with its ``.net`` present) is
skipped. ``--only-failed`` reprocesses everything not currently ``ok`` — use it
with a longer ``--timeout`` for the slow-netgen tail.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    ~/Code/bngsim/.venv/bin/python gen_networks.py --timeout 180 --workers 6
    ~/Code/bngsim/.venv/bin/python gen_networks.py --only-failed --timeout 2400 --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- Resolve the bng_parity + parity_checks modules we reuse -----------------
BNGSIM = Path(os.environ.get("BNGSIM_ROOT", Path.home() / "Code" / "bngsim"))
PARITY = BNGSIM / "parity_checks"
BNG_PARITY = PARITY / "bng_parity"
sys.path.insert(0, str(PARITY))
sys.path.insert(0, str(BNG_PARITY))
import _bng_common as bc  # noqa: E402
import overrides as OV  # noqa: E402
from _core import read_manifest  # noqa: E402

HERE = Path(__file__).resolve().parent
NETS = HERE / "nets"
MANIFEST = HERE / "nets_manifest.json"
JOBS = BNG_PARITY / "jobs.json"
MODELS_ROOT = BNG_PARITY / "models"

_manifest_lock = threading.Lock()


def resolve_bng2pl() -> str:
    """BNG2.pl from $BNGPATH / $BNG2_PL (never hardcoded beyond the canonical default)."""
    if os.environ.get("BNG2_PL"):
        p = Path(os.environ["BNG2_PL"])
    else:
        root = Path(
            os.environ.get("BNGPATH", str(Path.home() / "Simulations" / "BioNetGen-2.9.3"))
        )
        p = root if root.name == "BNG2.pl" else root / "BNG2.pl"
    if not p.exists():
        sys.exit(f"ABORT: BNG2.pl not found at {p} (set BNGPATH to the BioNetGen-2.9.3 root)")
    return str(p)


def net_counts(net_path: Path) -> tuple[int, int]:
    """(#species, #reactions) from a BNG2.pl .net by counting non-blank block lines."""
    text = Path(net_path).read_text()

    def _count(block: str) -> int:
        m = re.search(rf"begin {block}(.*?)end {block}", text, re.DOTALL | re.IGNORECASE)
        return sum(1 for ln in m.group(1).splitlines() if ln.strip()) if m else 0

    return _count("species"), _count("reactions")


def _sanitize(model_id: str) -> str:
    return model_id.replace("/", "__")


def gen_one(job, bng2_pl: str, timeout: float) -> dict:
    """Generate the full network for one ODE job; cache the .net; return a manifest row."""
    row: dict = {"model_id": job.model_id, "method": job.method}
    bngl_path = MODELS_ROOT / job.model
    try:
        bngl_text = bngl_path.read_text(errors="replace")
    except OSError as exc:
        return {**row, "status": "bad_test", "detail": f"unreadable BNGL: {exc}"}

    # NETGEN POLICY: use the model's OWN generate_network directive (author's caps
    # preserved) so the network is byte-identical to what the Jacobian
    # characterization / S4 used (e.g. fceri_fyn's own max_iter=>100 -> 1281 sp).
    # NEVER apply the parity REHAB max_agg/max_iter cap here: for fceri_fyn that cap
    # collapses 1281 -> 76. A model with no directive of its own falls back to a
    # bare uncapped generate_network({overwrite=>1}); the combinatorially explosive
    # no-directive models then time out and are resolved case-by-case.
    own_gen = bc._model_gen_network(bngl_text)
    rehab_in_parity = OV.REHAB.get(job.model_id) is not None
    gen_network = own_gen  # None -> _netgen_bngl appends the bare uncapped default
    state_prefix, prefix_info = bc.state_setup_prefix(bngl_text, track="ode")
    dirty = bool(prefix_info.get("dirty_carryover"))

    workdir = Path(tempfile.mkdtemp(prefix="fullnet_"))
    try:
        net_path, netgen_sec, err = bc.generate_network(
            bngl_text,
            bng2_pl,
            workdir,
            timeout=timeout,
            gen_network=gen_network,
            state_prefix=("" if dirty else state_prefix),
        )
        row.update(
            {
                "netgen_sec": round(netgen_sec, 3),
                "gen_network": gen_network or "generate_network({overwrite=>1})",
                "has_own_gen_network": own_gen is not None,
                "rehab_in_parity": rehab_in_parity,
                "dirty_carryover": dirty,
                "netgen_timeout_used": timeout,
            }
        )
        if net_path is None:
            row["status"] = "netgen_timeout" if "timed out" in (err or "") else "netgen_failed"
            row["detail"] = err
            return row
        nsp, nrxn = net_counts(net_path)
        dest = NETS / f"{_sanitize(job.model_id)}.net"
        # Strip the optional `reactions_text` block: it is redundant with the
        # numeric `reactions` block, and bngsim's .net loader rejects any file that
        # contains it ("stoi: no conversion" — it tries std::stoi on the species
        # names). BNG2.pl only emits it for a couple of models. Removing it lets
        # bngsim load them (verified byte-for-byte equivalent network otherwise).
        text = net_path.read_text()
        text = re.sub(
            r"begin reactions_text.*?end reactions_text\n?",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        dest.write_text(text)
        row.update({"status": "ok", "net_file": dest.name, "n_species": nsp, "n_reactions": nrxn})
        return row
    except Exception as exc:  # noqa: BLE001 - record, don't abort the sweep
        row["status"] = "error"
        row["detail"] = f"{type(exc).__name__}: {exc}"[:400]
        return row
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(man: dict) -> None:
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    tmp.replace(MANIFEST)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--timeout", type=float, default=180.0, help="Per-model netgen timeout (s).")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent netgen subprocesses.")
    ap.add_argument(
        "--only-failed", action="store_true", help="Reprocess models not currently 'ok'."
    )
    ap.add_argument("--models", default="", help="Comma-separated model_id substring filter.")
    ap.add_argument("--limit", type=int, default=0, help="Max models after filtering (0=all).")
    args = ap.parse_args()

    bng2_pl = resolve_bng2pl()
    _meta, alljobs = read_manifest(JOBS)
    jobs = [j for j in alljobs if j.method == "ode"]
    if args.models:
        subs = [s.strip() for s in args.models.split(",") if s.strip()]
        jobs = [j for j in jobs if any(s in j.model_id for s in subs)]

    man = load_manifest()
    todo = []
    for j in jobs:
        cur = man.get(j.model_id)
        cached_ok = (
            cur and cur.get("status") == "ok" and (NETS / (cur.get("net_file") or "")).exists()
        )
        if args.only_failed:
            if cached_ok:
                continue
        elif cached_ok:
            continue
        todo.append(j)
    if args.limit:
        todo = todo[: args.limit]

    print("=" * 72)
    print("  Phase 1 — full-network generation + .net cache (ode_fullnet)")
    print("=" * 72)
    print(
        f"  corpus ODE jobs: {len(jobs)}   to process: {len(todo)}   "
        f"already ok: {sum(1 for j in jobs if (man.get(j.model_id) or {}).get('status') == 'ok')}"
    )
    print(f"  timeout: {args.timeout}s   workers: {args.workers}   BNG2.pl: {bng2_pl}")
    print(f"  cache: {NETS}")
    print()
    if not todo:
        print("nothing to do.")
        return 0

    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, j, bng2_pl, args.timeout): j for j in todo}
        for fut in as_completed(futs):
            row = fut.result()
            with _manifest_lock:
                man[row["model_id"]] = row
                save_manifest(man)
            done += 1
            st = row.get("status")
            tag = "ok " if st == "ok" else st.upper()
            extra = (
                f"N={row.get('n_species')} rxn={row.get('n_reactions')}"
                if st == "ok"
                else (row.get("detail") or "")[:70]
            )
            print(
                f"  [{done}/{len(todo)}] {tag:15} {row.get('netgen_sec', '?'):>7}s "
                f"{row['model_id'].split('/')[-1]:45} {extra}"
            )

    dt = time.perf_counter() - t0
    # Summary over the WHOLE corpus (not just this pass).
    from collections import Counter

    tally = Counter((man.get(j.model_id) or {}).get("status", "missing") for j in jobs)
    print()
    print(f"  processed {done} in {dt:.0f}s. corpus status: {dict(tally)}")
    print(f"  manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
