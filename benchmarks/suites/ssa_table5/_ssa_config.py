#!/usr/bin/env python3
"""Shared config for the ssa_table5 four-engine SSA timing harness.

``corpus.json`` is the single source of truth for *what* is measured -- the 14
models (8 BNGL + 6 SBML), their artifacts, horizons and output-point counts, and
their provenance.  This module reads it and adds only *how* the harness runs
them: per-model warm-rep counts, the cheap->expensive order, the coverage
authority (which of the 14x4 cells run / are N/A + why), and per-(engine, model)
artifact resolution.  Horizons and species/reaction counts used to be typed out
here as well as in the corpus, which is drift waiting to happen: the corpus is
not on the timing path, so a horizon edited in one place and not the other would
be invisible until the manuscript quoted the wrong number.

Coverage was fixed by convert_all.py (net<->SBML faithfulness) + direct SBML event
inspection; see results/converted/conversion_log.json. It is authoritative here so the
runner never forces an engine onto a model it cannot faithfully simulate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERTED = HERE / "results" / "converted"
CORPUS_JSON = HERE / "corpus.json"

# Per-run wall cap (brief: 120 s). Warm stops early once cumulative warm wall exceeds
# WARM_BUDGET_SEC (keeps a slow engine from running N full reps). The orchestrator adds
# a hard CELL_WALL_CAP backstop (SIGKILL) for a run that can't self-interrupt (RR/COPASI
# sit in a C call the per-run SIGALRM can't preempt).
PER_RUN_CAP_SEC = 120.0
WARM_BUDGET_SEC = 150.0
CELL_WALL_CAP_SEC = 300.0

ENGINES = ["bngsim", "run_network", "roadrunner", "copasi"]

# Runner policy, per model key. Everything else -- kind, artifact path, t_end,
# n_points -- comes from corpus.json (loaded below), which is the corpus SSOT.
#
#   warm  : number of warm reps (see WARM_BUDGET_SEC; 3 for the expensive models)
#   order : the cheap->expensive index (estimated total Gillespie events =
#           events/time * t_end), which is the order the orchestrator runs cells in
_POLICY: dict[str, dict] = {
    # BNGL (native .net for bngsim + run_network)
    "gene_bursts": dict(warm=10, order=1),
    "samoilov_futile_cycle": dict(warm=10, order=2),
    "gene_expression": dict(warm=10, order=5),
    "mckane_predator_prey": dict(warm=10, order=8),
    "gene_expr_3stage": dict(warm=10, order=11),
    "prion_aggregation": dict(warm=3, order=12),
    "tcr_signaling": dict(warm=3, order=13),
    "erk_activation": dict(warm=3, order=14),
    # SBML (native .xml for bngsim + roadrunner + copasi)
    "BIOMD0000000862": dict(warm=10, order=3),
    "BIOMD0000000344": dict(warm=10, order=4),
    "BIOMD0000000860": dict(warm=10, order=6),
    "BIOMD0000000478": dict(warm=10, order=7),
    "BIOMD0000000035": dict(warm=10, order=9),
    "BIOMD0000000864": dict(warm=10, order=10),
}


def _load_models() -> dict[str, dict]:
    """corpus.json -> the runner's model registry.

    ``file`` is kept exactly as the corpus writes it (relative to this directory);
    the BNGL rows point out of the suite at the shared curated artifacts under
    ``benchmarks/models/net/curated/``, which the psa suite runs too. ``n_points``
    is the number of output rows -- corpus BNGL rows declare ``n_steps``, and an
    engine wanting a step count uses ``n_points - 1``.
    """
    corpus = json.loads(CORPUS_JSON.read_text())
    models: dict[str, dict] = {}
    for entry in corpus["bngl"]:
        models[entry["name"]] = dict(
            kind="bngl",
            file=entry["file"],
            t_end=float(entry["t_end"]),
            n_points=int(entry["n_steps"]) + 1,
        )
    for entry in corpus["sbml"]:
        models[entry["id"]] = dict(
            kind="sbml",
            file=entry["file"],
            t_end=float(entry["t_end"]),
            n_points=int(entry["n_points"]),
        )
    missing = set(models) ^ set(_POLICY)
    if missing:
        raise RuntimeError(f"corpus.json and _POLICY disagree on the model set: {sorted(missing)}")
    for key, policy in _POLICY.items():
        models[key].update(policy)
    return models


MODELS: dict[str, dict] = _load_models()


# Coverage authority: (model, engine) -> "ok" or ("na"|"flag", reason).
# "na"   = engine cannot faithfully simulate this model; cell is not run.
# "flag" = engine is run (timing valid) but a correctness caveat applies.
_NA = "na"
_FLAG = "flag"
_EV_TIME_TRIG = "time-triggered event(s); RR-gillespie warns 'time not treated continuously' and won't fire them faithfully"
_EV_DROP_RN = (
    "SBML->.net conversion dropped time-triggered event(s); .net dynamics differ from source SBML"
)
_EV_STATE_344 = "state-triggered event kalive:=0 on CellDeath>=1; RR-gillespie won't fire it (COPASI does). Timing valid; trajectory faithful only while CellDeath<1"
_REPEATED_RR = (
    "repeated reactant (N + N -> ...): the converted SBML law k*N*N is not the exact "
    "propensity k*N*(N-1), and RR-gillespie fires the SBML law (GH #9). COPASI derives the "
    "combinatorial propensity itself, so its cell stands"
)


def _coverage() -> dict[str, dict[str, object]]:
    cov: dict[str, dict[str, object]] = {}
    for k, m in MODELS.items():
        c = {"bngsim": "ok", "copasi": "ok"}  # bngsim + COPASI cover all 14
        if m["kind"] == "bngl":
            c["run_network"] = "ok"  # native .net
            c["roadrunner"] = "ok"  # converted SBML; overridden below where unfaithful
        else:
            c["run_network"] = "ok"  # via converted .net; overridden below where events dropped
            c["roadrunner"] = "ok"  # native SBML; overridden below where events present
        cov[k] = c
    # run_network N/A: SBML models whose SBML->.net dropped an event
    for k in ("BIOMD0000000860", "BIOMD0000000862", "BIOMD0000000864", "BIOMD0000000344"):
        cov[k]["run_network"] = (_NA, _EV_DROP_RN)
    # roadrunner N/A: SBML models with time-triggered events
    for k in ("BIOMD0000000860", "BIOMD0000000862", "BIOMD0000000864"):
        cov[k]["roadrunner"] = (_NA, _EV_TIME_TRIG)
    # roadrunner flag: 344 state-triggered event
    cov["BIOMD0000000344"]["roadrunner"] = (_FLAG, _EV_STATE_344)
    # roadrunner N/A: BNGL model whose net->SBML carries a repeated reactant. The
    # curated Samoilov record includes the external noise driver of Expressions 7
    # and 8, whose autocatalytic step N + N -> E+ + N is second order in the same
    # species; the superseded copy had no driver and so no such reaction.
    cov["samoilov_futile_cycle"]["roadrunner"] = (_NA, _REPEATED_RR)
    return cov


COVERAGE = _coverage()


def cell_status(model_key: str, engine: str):
    """('ok'|'na'|'flag', reason)."""
    v = COVERAGE[model_key][engine]
    if v == "ok":
        return ("ok", "")
    return v  # (na|flag, reason)


def artifact_for(model_key: str, engine: str) -> tuple[str, str]:
    """(kind_for_engine, path) — the file THIS engine loads for THIS model.

    kind_for_engine is 'net' or 'sbml' (what the loader expects), which may differ from
    the model's native kind because of the cross-engine conversions:
      * bngsim      : native (.net for bngl, .xml for sbml)
      * run_network : .net   (native for bngl; converted results/converted/<id>.net for sbml)
      * roadrunner  : sbml   (native .xml for sbml; converted results/converted/<name>.xml for bngl)
      * copasi      : sbml   (native .xml for sbml; converted results/converted/<name>.xml for bngl)
    """
    m = MODELS[model_key]
    # resolved, because the corpus points the BNGL rows out of the suite at the
    # shared ../../models/net/curated/ artifacts and these paths land in the
    # results JSON.
    native = (HERE / m["file"]).resolve()
    if engine == "bngsim":
        return (("net" if m["kind"] == "bngl" else "sbml"), str(native))
    if engine == "run_network":
        if m["kind"] == "bngl":
            return ("net", str(native))
        return ("net", str(CONVERTED / f"{model_key}.net"))
    if engine in ("roadrunner", "copasi"):
        if m["kind"] == "sbml":
            return ("sbml", str(native))
        return ("sbml", str(CONVERTED / f"{model_key}.xml"))
    raise ValueError(engine)


def ordered_models() -> list[str]:
    return sorted(MODELS, key=lambda k: MODELS[k]["order"])


# BNG run_network binary. Same resolution as benchmarks/_netbench.py, so one
# BNGPATH covers every suite: $RUN_NETWORK wins, else $BNGPATH/bin/run_network,
# else the canonical ~/Simulations/BioNetGen-2.9.3 install. This used to be an
# absolute path into one developer's home directory, which no other machine has.
BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
RUN_NETWORK_BIN = os.environ.get("RUN_NETWORK", os.path.join(BNGPATH, "bin", "run_network"))
