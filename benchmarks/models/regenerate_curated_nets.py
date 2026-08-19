#!/usr/bin/env python3
"""Regenerate ``net/curated/*.net`` from the curated BNGL-Models records.

The eight BNGL models the manuscript names by title are house-curated records in
``wshlavacek/BNGL-Models``; the ``.net`` files the benchmark suites actually time
are generated from them here, once, and shared by every suite that needs them
(``suites/ssa_table5`` runs all eight, ``suites/psa`` runs three).  Before this
script existed the two suites vendored *separate* pre-generated copies that had
drifted from the records and from each other -- same three models, three
different sha256s -- so a re-run faithfully re-measured superseded networks.

``curated_nets.json`` is the manifest: for each model, the record it comes from,
the sha256 of the upstream ``.bngl``, the sha256 of the generated ``.net``, and
the species/reaction counts the record itself declares.

What generation does, and does not, do
--------------------------------------
Every action *except* ``generate_network`` is dropped, so a record's own
simulate/scan protocol never runs -- the horizons the suites use are the
manuscript's, not the records', and are declared by the suites.
``generate_network`` and its options are **kept**, and that is load-bearing: the
prion record raises ``max_iter=>150`` over BNG's 100-iteration default and caps
chain length with ``max_stoich=>{PrP=>120}``, which is the whole difference
between its 121/3843 network and the 104/2809 one an earlier vendored copy
froze.  Nothing is baked into the seed species afterwards: a ``.net`` here is
``generate_network`` on the curated model body and nothing else.

Usage::

    python regenerate_curated_nets.py            # regenerate every model
    python regenerate_curated_nets.py --only prion_aggregation
    python regenerate_curated_nets.py --check    # verify, write nothing (CI//pre-run)
    python regenerate_curated_nets.py --sync-sources   # re-copy the .bngl from the
                                                       # collection, then regenerate

``--check`` re-derives every ``.net`` into a temporary directory and diffs it
against what is committed, so a stale artifact is a failure rather than a silent
re-measurement.  It needs BNG2.pl; without one it reports what it could not check
and exits non-zero only on a real mismatch.

BNG2.pl is located via ``$BNG2_PL``, else ``$BNGPATH/BNG2.pl``, else the
canonical ``~/Simulations/BioNetGen-2.9.3`` install (same convention as
``benchmarks/_netbench.py``).  The collection is located via
``$BNGL_MODELS_ROOT``, else ``~/Code/BNGL-Models``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "curated_nets.json"

BNGPATH = Path(os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3")))
BNG2_PL = Path(os.environ.get("BNG2_PL", str(BNGPATH / "BNG2.pl")))
BNGL_MODELS_ROOT = Path(
    os.environ.get("BNGL_MODELS_ROOT", str(Path.home() / "Code" / "BNGL-Models"))
)

# Actions dropped before generating. `generate_network` is deliberately absent:
# its options decide the network (see the module docstring).
_ACTION = re.compile(
    r"^(simulate|simulate_ode|simulate_ssa|simulate_pla|simulate_nf|parameter_scan|"
    r"bifurcate|readFile|writeFile|writeModel|writeNetwork|writeXML|writeSBML|writeMfile|"
    r"writeMexfile|writeMDL|writeSSC|writeLatex|visualize|setConcentration|setParameter|"
    r"saveConcentrations|resetConcentrations|saveParameters|resetParameters|"
    r"setModelName|substanceUnits|quit)\b"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def logical_lines(text: str) -> list[str]:
    """Join BNGL backslash continuations, so an action is one line to match on.

    Stripping actions line-by-line is not enough: every simulate in these records
    is wrapped across two or three physical lines with a trailing ``\\``, and
    deleting only the first leaves ``seed=>2,print_functions=>1})`` behind as a
    top-level statement that aborts BNG2.pl.  A commented-out action stays a
    comment when joined, since every one of its physical lines carries its own
    ``#``.
    """
    out: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1].rstrip() + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def strip_actions(text: str) -> str:
    """The curated model body plus its own ``generate_network``, nothing else."""
    kept = [ln for ln in logical_lines(text) if not _ACTION.match(ln.strip())]
    body = "\n".join(kept).rstrip()
    if not re.search(r"^\s*generate_network\b", body, flags=re.M):
        body += "\n\ngenerate_network({overwrite=>1})"
    return body + "\n"


def generate(bngl: Path, out: Path, timeout: int = 1800) -> Path:
    """Run BNG2.pl on the action-stripped model and place its ``.net`` at *out*."""
    if not BNG2_PL.exists():
        raise FileNotFoundError(
            f"BNG2.pl not found at {BNG2_PL} (set $BNG2_PL or $BNGPATH to a "
            f"BioNetGen-2.9.3 install)"
        )
    with tempfile.TemporaryDirectory(prefix="curated_net_") as tmp:
        work = Path(tmp)
        (work / bngl.name).write_text(strip_actions(bngl.read_text()))
        for extra in bngl.parent.glob("*.tfun"):
            shutil.copy(extra, work / extra.name)
        proc = subprocess.run(
            ["perl", str(BNG2_PL), bngl.name],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        net = work / f"{bngl.stem}.net"
        if not net.exists():
            raise RuntimeError(
                f"BNG2.pl wrote no .net for {bngl.name}\n"
                f"{proc.stdout[-600:]}\n{proc.stderr[-400:]}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(net, out)
    return out


def count_block(net_text: str, name: str) -> int:
    n, inside = 0, False
    for raw in net_text.splitlines():
        line = raw.strip()
        if line.startswith(f"begin {name}"):
            inside = True
            continue
        if line.startswith(f"end {name}"):
            inside = False
            continue
        if inside and line and not line.startswith("#"):
            n += 1
    return n


def sync_sources(models: list[dict]) -> list[str]:
    """Re-copy each vendored ``.bngl`` from the collection; report what moved."""
    changed = []
    for m in models:
        upstream = BNGL_MODELS_ROOT / m["source"]
        if not upstream.exists():
            raise FileNotFoundError(
                f"{upstream} not found (set $BNGL_MODELS_ROOT to a BNGL-Models checkout)"
            )
        local = HERE / m["bngl"]
        before = sha256(local) if local.exists() else ""
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(upstream, local)
        after = sha256(local)
        if after != before:
            changed.append(f"{m['name']}: {before[:12] or '(absent)'} -> {after[:12]}")
        m["source_sha256"] = after
    return changed


def verify_sources(models: list[dict]) -> list[str]:
    """Vendored ``.bngl`` vs the manifest sha256, and vs the collection if present."""
    problems = []
    have_collection = BNGL_MODELS_ROOT.is_dir()
    for m in models:
        local = HERE / m["bngl"]
        if not local.exists():
            problems.append(f"{m['name']}: vendored source missing ({m['bngl']})")
            continue
        got = sha256(local)
        if got != m["source_sha256"]:
            problems.append(
                f"{m['name']}: {m['bngl']} sha256 {got[:12]} != manifest {m['source_sha256'][:12]}"
            )
        elif have_collection:
            upstream = BNGL_MODELS_ROOT / m["source"]
            if upstream.exists() and sha256(upstream) != got:
                problems.append(
                    f"{m['name']}: {m['bngl']} has drifted from {m['source']} in the collection"
                )
    return problems


def record(m: dict, net: Path) -> None:
    """Refresh the manifest fields that describe *net*."""
    text = net.read_text()
    m["net_sha256"] = sha256(net)
    m["species"] = count_block(text, "species")
    m["reactions"] = count_block(text, "reactions")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--only", help="restrict to one model name")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed .net files instead of rewriting them",
    )
    ap.add_argument(
        "--sync-sources",
        action="store_true",
        help="re-copy the .bngl files from $BNGL_MODELS_ROOT before generating",
    )
    args = ap.parse_args()

    manifest = load_manifest()
    models = manifest["models"]
    if args.only:
        models = [m for m in models if m["name"] == args.only]
        if not models:
            sys.exit(f"no model named {args.only!r} in {MANIFEST.name}")

    if args.sync_sources:
        if args.check:
            sys.exit("--sync-sources rewrites files; it cannot be combined with --check")
        for line in sync_sources(models):
            print(f"  source updated  {line}")

    problems = verify_sources(models)
    for p in problems:
        print(f"  SOURCE  {p}")

    for m in models:
        bngl = HERE / m["bngl"]
        net = HERE / m["net"]
        if args.check:
            if not net.exists():
                problems.append(f"{m['name']}: {m['net']} is missing")
                print(f"  MISSING {m['name']:24} {m['net']}")
                continue
            got = sha256(net)
            if got != m["net_sha256"]:
                problems.append(
                    f"{m['name']}: {m['net']} sha256 {got[:12]} != manifest {m['net_sha256'][:12]}"
                )
                print(f"  DRIFTED {m['name']:24} committed .net != manifest sha256")
                continue
            if not BNG2_PL.exists():
                print(f"  pinned  {m['name']:24} sha256 ok (BNG2.pl absent; not regenerated)")
                continue
            with tempfile.TemporaryDirectory(prefix="curated_check_") as tmp:
                fresh = generate(bngl, Path(tmp) / f"{m['name']}.net")
                if sha256(fresh) != got:
                    problems.append(
                        f"{m['name']}: {m['net']} does not match a fresh generation "
                        f"from {m['bngl']}"
                    )
                    print(f"  STALE   {m['name']:24} committed .net != freshly generated")
                    continue
            print(f"  ok      {m['name']:24} {m['species']:4d} sp / {m['reactions']:5d} rx")
            continue

        generate(bngl, net)
        record(m, net)
        print(
            f"  wrote   {m['name']:24} {m['species']:4d} sp / {m['reactions']:5d} rx  {m['net']}"
        )

    if not args.check:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\n-> {MANIFEST}")

    if problems:
        print("\n" + "\n".join(f"FAIL {p}" for p in problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
