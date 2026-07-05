#!/usr/bin/env python3
"""cBNGL BNG2.pl round-trip validator (GH #224).

For each given model: sbml_to_bngl (cBNGL) → BNG2.pl generate_network → .net →
from_net → name-aligned ODE-RHS delta vs the source. This is the cBNGL
faithfulness oracle (no in-tree cBNGL reader by design — see the
no-intree-cbngl-reader decision). Mirrors python/tests/test_sbml_to_bngl.py's
round-trip helpers so a corpus-wide sweep uses the same measure the unit tests do.

Usage:
    .venv/bin/python parity_checks/rr_parity/cbngl_bng2_validate.py MODEL1112110002 [...]
    .venv/bin/python parity_checks/rr_parity/cbngl_bng2_validate.py --accepted \
        [--out runs/cbngl_bng2.json] [--timeout 120]   # all cBNGL-accepted models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from bngsim._exceptions import ConversionError
from bngsim.convert._bng2 import find_bng2

HERE = Path(__file__).resolve().parent

# The shipped find_bng2 uses $BNGPATH / PATH; honor an explicit $BNG2_PL override as
# a last-resort fallback so the harness runs without a full $BNGPATH setup (mirrors
# the test convention).
_BNG2_ENV = os.environ.get("BNG2_PL")
_BNG2 = find_bng2() or (Path(_BNG2_ENV) if _BNG2_ENV and Path(_BNG2_ENV).is_file() else None)


def _corpus_xml(model_id: str) -> Path | None:
    hits = sorted((HERE / "models").glob(f"{model_id}/*.xml"))
    return hits[0] if hits else None


def validate_one(model_id: str, timeout: int = 120) -> dict:
    import bngsim
    from bngsim.convert import sbml_to_bngl
    from bngsim.convert._bng2 import roundtrip_rhs_delta

    xml = _corpus_xml(model_id)
    if xml is None:
        return {"model_id": model_id, "status": "missing"}
    try:
        src = bngsim.Model.from_sbml(xml)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = sbml_to_bngl(xml, strict=False)
        if rep.lossy:
            return {
                "model_id": model_id,
                "status": "refused",
                "reason": "; ".join(rep.lossy)[:120],
            }
        bngl_text = rep.output_text
    except Exception as e:
        return {
            "model_id": model_id,
            "status": "convert_error",
            "reason": f"{type(e).__name__}: {e}"[:160],
        }

    # Reuse the shipped oracle so the harness measures faithfulness exactly as the
    # production sbml_to_bngl(validate="bng2") gate does. Map its failure modes
    # (timeout / no-net / unalignable) back to the harness's status taxonomy.
    try:
        delta, n_rl = roundtrip_rhs_delta(
            src, bngl_text, stem=model_id, timeout=timeout, bng2=_BNG2
        )
    except ConversionError as e:
        msg = str(e)
        if "timed out" in msg:
            return {"model_id": model_id, "status": "bng_timeout"}
        if "produced no network" in msg:
            return {"model_id": model_id, "status": "bng_no_net", "reason": msg[-200:]}
        if "align" in msg:
            return {"model_id": model_id, "status": "species_mismatch", "reason": msg[:160]}
        return {"model_id": model_id, "status": "reload_error", "reason": msg[:160]}
    if n_rl != src.n_species:
        return {
            "model_id": model_id,
            "status": "species_mismatch",
            "reason": f"{src.n_species}->{n_rl}",
        }
    return {
        "model_id": model_id,
        "status": "faithful" if delta <= 1e-6 else "rhs_mismatch",
        "delta": delta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument(
        "--accepted",
        action="store_true",
        help="validate every cBNGL-accepted model from convert_sweep.json",
    )
    ap.add_argument("--sweep", default=str(HERE / "runs" / "convert_sweep.json"))
    ap.add_argument("--out", default=str(HERE / "runs" / "cbngl_bng2.json"))
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    if _BNG2 is None:
        print("BNG2.pl not found (set $BNGPATH)", file=sys.stderr)
        return 2

    ids = list(args.models)
    if args.accepted:
        with open(args.sweep) as fh:
            sweep = json.load(fh)
        ids += [r["model_id"] for r in sweep["rows"] if r.get("bngl") == "accepted"]

    rows = []
    from collections import Counter

    for i, mid in enumerate(ids):
        r = validate_one(mid, timeout=args.timeout)
        rows.append(r)
        d = f" delta={r['delta']:.2e}" if "delta" in r else ""
        print(
            f"[{i + 1}/{len(ids)}] {mid}: {r['status']}{d}"
            + (f"  {r.get('reason', '')}" if r.get("reason") else ""),
            file=sys.stderr,
            flush=True,
        )

    if args.accepted:
        Path(args.out).write_text(
            json.dumps(
                {"summary": dict(Counter(r["status"] for r in rows)), "rows": rows}, indent=2
            )
        )
        print(json.dumps(dict(Counter(r["status"] for r in rows)), indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
