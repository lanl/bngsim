#!/usr/bin/env python3
"""Recover the 2 reactions_text loader-bug models into the Jacobian
characterization (Figure S4): BaruaBCR_2012 and ComplexDegradation generate their
full network fine but bngsim's .net loader rejects the redundant `reactions_text`
block. We monkeypatch bc.generate_network to strip that block (same fix the
ode_fullnet cache uses), characterize the 2 models with the UNMODIFIED jac-char
logic, merge them into the raw characterization (replacing their failed rows),
re-run the analysis, and copy the refreshed raw + analysis into the paper's
latex/generated/ so generate_klu_figure.py picks up 585 (not 583) points.

The 7 genuinely network-free-by-design models stay excluded (no finite ODE network).

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    python recover_s4_points.py                 # characterize, merge, copy
    python recover_s4_points.py --no-copy       # leave the paper checkout alone
    python recover_s4_points.py --paper-dir DIR # copy somewhere else
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

BNGSIM = Path(os.environ.get("BNGSIM_ROOT", Path.home() / "Code" / "bngsim"))
BNG_PARITY = BNGSIM / "parity_checks" / "bng_parity"
sys.path.insert(0, str(BNGSIM / "parity_checks"))
sys.path.insert(0, str(BNG_PARITY))
import _bng_common as bc  # noqa: E402

# --- Monkeypatch: strip the reactions_text block from every generated .net -----
_orig_gen = bc.generate_network


def _gen_strip(*a, **k):
    net, sec, err = _orig_gen(*a, **k)
    if net is not None:
        t = Path(net).read_text()
        t2 = re.sub(
            r"begin reactions_text.*?end reactions_text\n?", "", t, flags=re.DOTALL | re.IGNORECASE
        )
        if t2 != t:
            Path(net).write_text(t2)
    return net, sec, err


bc.generate_network = _gen_strip

import jacobian_characterization as jc  # noqa: E402  (imports bc; patch is in place)

RUNS = BNG_PARITY / "runs"
RAW = RUNS / "jacobian_characterization.json"
ANA = RUNS / "jacobian_characterization_analysis.json"
PAPER_GEN = Path.home() / "Code" / "bngsim-paper" / "latex" / "generated"

TARGETS = [
    "slow/rulehub/Published/BaruaBCR2012/BaruaBCR_2012.bngl",
    "slow/rulehub/Tutorials/NativeTutorials/ComplexDegradation/ComplexDegradation.bngl",
]


def build_parser():
    """The CLI. This script had the longest reach of any that ignored argv.

    A bare invocation regenerates two networks through BNG2.pl, rewrites the
    characterization under ``parity_checks/bng_parity/runs/``, and then copies
    both files into a *different repository's* ``latex/generated/`` -- one of
    them committed there. It read no argv, so ``--help`` did all of that
    (GH #489). ``--paper-dir`` and ``--no-copy`` bound the last step; the
    recovery itself takes no options, since the two target models are the
    point of the script.
    """
    parser = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=PAPER_GEN,
        help=f"Where to copy the refreshed raw + analysis (default: {PAPER_GEN}).",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Merge and re-analyze only; do not touch the paper checkout.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(os.environ.get("BNGPATH", str(Path.home() / "Simulations" / "BioNetGen-2.9.3")))
    bng2_pl = str(root if root.name == "BNG2.pl" else root / "BNG2.pl")
    horizons = jc.load_horizons()

    new_rows = {}
    for mid in TARGETS:
        print(f"characterizing {mid.split('/')[-1]} ...", flush=True)
        row = jc.characterize_model(mid, horizons.get(mid, {}), bng2_pl)
        print(
            f"   status={row.get('status')} N={row.get('N')} density={row.get('density')}",
            flush=True,
        )
        new_rows[mid] = row

    doc = json.loads(RAW.read_text())
    results = doc["results"]
    by_id = {r["model_id"]: i for i, r in enumerate(results)}
    for mid, row in new_rows.items():
        if mid in by_id:
            results[by_id[mid]] = row
        else:
            results.append(row)
    doc["results"] = results
    RAW.write_text(json.dumps(doc))
    print(f"merged {len(new_rows)} rows into {RAW}", flush=True)

    analysis = jc.analyze(RAW, None, None)
    ANA.write_text(json.dumps(analysis, indent=1))
    print(f"re-analyzed -> {ANA}", flush=True)

    # Copy refreshed raw (gitignored) + analysis (committed) into the paper.
    # Absent target = skip, not crash: the merge above has already happened, and
    # this directory only exists on a machine with the paper checked out beside
    # bngsim.
    if args.no_copy:
        print("--no-copy: left the paper checkout alone")
    elif not args.paper_dir.is_dir():
        print(f"skipped copy: {args.paper_dir} is not a directory (pass --paper-dir)")
    else:
        shutil.copyfile(RAW, args.paper_dir / "jacobian_characterization.json")
        shutil.copyfile(ANA, args.paper_dir / "jacobian_characterization_analysis.json")
        print(f"copied raw + analysis to {args.paper_dir}")
        print("now run the paper's scripts/generate_klu_figure.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
