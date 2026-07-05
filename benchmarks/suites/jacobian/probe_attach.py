#!/usr/bin/env python3
"""Report, per model, whether the analytical Functional Jacobian actually attaches.

The speedup benchmark forces FD with BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0 and
leaves it on (default) for "analytical".  But "analytical" mode silently falls
back to FD when the self-check bails (symbolic divergence, singular-at-init,
per-observable .net path, etc.).  If a model bails, the "analytical vs FD"
timing compares FD against FD and any ~1x reading is meaningless -- so the
speedup table must only include models that genuinely attach.

This probe re-runs the attach driver on a freshly loaded model and reports its
boolean verdict (True = analytical attached; False = on FD), plus the count of
functional reactions and whether the self-check fallback message fired.

Usage::  python probe_attach.py [MODEL_ID ...]   (default: the suite's TARGETS)
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODELS_DIR = ROOT / "parity_checks" / "rr_parity" / "models"
NET_DIR = HERE.parents[1] / "models" / "net"

# `.net` networks (the per-observable Functional path, GH #76 task 2):
# model id → path relative to benchmarks/models/net/.
NET_MODELS = {"egfr_net_red": "ode/egfr_net_red.net"}

DEFAULT_IDS = [
    "BIOMD0000000013",
    "MODEL9089538076",
    "BIOMD0000000595",
    "MODEL9087255381",
    "MODEL9087474843",
    "egfr_net_red",
]


def sbml_path(model_id: str) -> Path | None:
    d = MODELS_DIR / model_id
    if not d.is_dir():
        return None
    cand = d / f"{model_id}.xml"
    if cand.exists():
        return cand
    xmls = sorted(d.glob("*.xml"))
    return xmls[0] if xmls else None


def probe(model_id: str) -> dict:
    os.environ.pop("BNGSIM_ANALYTICAL_FUNCTIONAL_JAC", None)  # default ON
    os.environ["BNGSIM_JAC_DEBUG"] = "1"
    is_net = model_id in NET_MODELS
    path = (NET_DIR / NET_MODELS[model_id]) if is_net else sbml_path(model_id)
    if path is None or not path.exists():
        return {"model": model_id, "error": "missing"}
    rec = {"model": model_id}
    try:
        import bngsim
        from bngsim import _jacobian

        model = bngsim.Model.from_net(str(path)) if is_net else bngsim.Model.from_sbml(str(path))
        core = model._core
        ctx = core.functional_jacobian_context()
        rec["n_functional_rxns"] = len(ctx.get("functional_reactions") or [])
        ns = getattr(core, "n_species", None)
        rec["n_species"] = ns() if callable(ns) else ns
        # Re-run the attach driver and capture its verdict + any fallback message.
        buf = io.StringIO()
        with redirect_stderr(buf):
            attached = _jacobian.attach_functional_jacobian(core)
        rec["attached"] = bool(attached)
        err = buf.getvalue()
        rec["self_check_failed"] = "failed FD self-check" in err
        rec["debug_stderr"] = err.strip()[-400:]
    except Exception as e:  # noqa: BLE001
        import traceback

        rec["error"] = repr(e)
        rec["traceback"] = traceback.format_exc()[-800:]
    return rec


def main():
    ids = sys.argv[1:] or DEFAULT_IDS
    out = [probe(mid) for mid in ids]
    (HERE / "results" / "attach_probe.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(
            f"{r['model']}: attached={r.get('attached')} "
            f"n_func_rxns={r.get('n_functional_rxns')} ns={r.get('n_species')} "
            f"self_check_failed={r.get('self_check_failed')} err={r.get('error')}"
        )
    print("WROTE results/attach_probe.json")


if __name__ == "__main__":
    main()
