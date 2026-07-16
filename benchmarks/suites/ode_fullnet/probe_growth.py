#!/usr/bin/env python3
"""Probe network growth for the models whose full (uncapped) netgen times out.

For each model, run BNG2.pl ``generate_network`` at a ladder of ``max_iter`` caps
and record the species/reaction count at each. A network that keeps ~multiplying
every iteration is combinatorially unbounded (network-free by design); one that
plateaus has a finite full network that merely needs more time. This turns the
"case-by-case" decision into concrete numbers.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    ~/Code/bngsim/.venv/bin/python probe_growth.py --iters 1,2,3,4,5,6 --timeout 120
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

BNGSIM = Path(os.environ.get("BNGSIM_ROOT", Path.home() / "Code" / "bngsim"))
PARITY = BNGSIM / "parity_checks"
BNG_PARITY = PARITY / "bng_parity"
sys.path.insert(0, str(PARITY))
sys.path.insert(0, str(BNG_PARITY))
import _bng_common as bc  # noqa: E402

MODELS_ROOT = BNG_PARITY / "models"

# The 7 no-directive, combinatorially-large models that time out at full netgen.
DEFAULT_MODELS = [
    "original/bngl_models/my_models/nf/cleavage_mechanism_v1.bngl",
    "slow/rulehub/Published/Chattaraj2021/Chattaraj_2021.bngl",
    "slow/rulehub/Published/ChylekFceRI2014/ChylekFceRI_2014.bngl",
    "slow/rulehub/Published/ChylekTCR2014/ChylekTCR_2014.bngl",
    "slow/rulehub/Published/Massole2023/Massole_2023.bngl",
    "slow/rulehub/Tutorials/NativeTutorials/Chyleklibrary/Chylek_library.bngl",
    "slow/rulehub/Tutorials/NativeTutorials/Suderman2013/Suderman_2013.bngl",
]


def counts(net_path: Path) -> tuple[int, int]:
    text = Path(net_path).read_text()

    def c(block: str) -> int:
        m = re.search(rf"begin {block}(.*?)end {block}", text, re.DOTALL | re.IGNORECASE)
        return sum(1 for ln in m.group(1).splitlines() if ln.strip()) if m else 0

    return c("species"), c("reactions")


def resolve_bng2pl() -> str:
    root = Path(os.environ.get("BNGPATH", str(Path.home() / "Simulations" / "BioNetGen-2.9.3")))
    return str(root if root.name == "BNG2.pl" else root / "BNG2.pl")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--iters", default="1,2,3,4,5,6")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--models", default="", help="Override the default model list (comma-separated)."
    )
    args = ap.parse_args()

    bng2 = resolve_bng2pl()
    iters = [int(x) for x in args.iters.split(",")]
    models = args.models.split(",") if args.models else DEFAULT_MODELS

    for mid in models:
        bngl_text = (MODELS_ROOT / mid).read_text(errors="replace")
        print(f"\n=== {mid.split('/')[-1]} ===")
        for k in iters:
            wd = Path(tempfile.mkdtemp(prefix="probe_"))
            try:
                net, sec, err = bc.generate_network(
                    bngl_text,
                    bng2,
                    wd,
                    timeout=args.timeout,
                    gen_network=f"generate_network({{overwrite=>1,max_iter=>{k}}})",
                )
                if net is None:
                    print(f"  max_iter={k:2}: TIMEOUT/FAIL after {sec:.0f}s  {err[:60]}")
                    break
                nsp, nrxn = counts(net)
                print(f"  max_iter={k:2}: N={nsp:6}  rxn={nrxn:7}  ({sec:.1f}s)")
            finally:
                import shutil

                shutil.rmtree(wd, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
