#!/usr/bin/env python3
"""Shared config for the ssa_table5 four-engine SSA timing harness.

Models (14: 8 BNGL + 6 SBML), their resolved (t_end, n_points), the cheap->expensive
run order, per-model warm-rep counts, the coverage authority (which of the 14x4 cells
run / are N/A + why), and per-(engine, model) artifact resolution.

Coverage was fixed by convert_all.py (net<->SBML faithfulness) + direct SBML event
inspection; see results/converted/conversion_log.json. It is authoritative here so the
runner never forces an engine onto a model it cannot faithfully simulate.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERTED = HERE / "results" / "converted"

# Per-run wall cap (brief: 120 s). Warm stops early once cumulative warm wall exceeds
# WARM_BUDGET_SEC (keeps a slow engine from running N full reps). The orchestrator adds
# a hard CELL_WALL_CAP backstop (SIGKILL) for a run that can't self-interrupt (RR/COPASI
# sit in a C call the per-run SIGALRM can't preempt).
PER_RUN_CAP_SEC = 120.0
WARM_BUDGET_SEC = 150.0
CELL_WALL_CAP_SEC = 300.0

ENGINES = ["bngsim", "run_network", "roadrunner", "copasi"]

# key -> (kind, source-file relative to HERE, t_end, n_points, warm_N, order)
# n_points = number of output rows; engines wanting a step count use n_points-1.
# order is the cheap->expensive index (estimated total Gillespie events = events/time * t_end).
MODELS: dict[str, dict] = {
    # BNGL (native .net for bngsim + run_network)
    "gene_bursts": dict(
        kind="bngl",
        file="models/bngl/gene_bursts.net",
        t_end=3600.0,
        n_points=361,
        warm=10,
        order=1,
    ),
    "samoilov_futile_cycle": dict(
        kind="bngl",
        file="models/bngl/samoilov_futile_cycle.net",
        t_end=0.0018,
        n_points=181,
        warm=10,
        order=2,
    ),
    "gene_expression": dict(
        kind="bngl",
        file="models/bngl/gene_expression.net",
        t_end=60000.0,
        n_points=1001,
        warm=10,
        order=5,
    ),
    "mckane_predator_prey": dict(
        kind="bngl",
        file="models/bngl/mckane_predator_prey.net",
        t_end=1200.0,
        n_points=1201,
        warm=10,
        order=8,
    ),
    "gene_expr_3stage": dict(
        kind="bngl",
        file="models/bngl/gene_expr_3stage.net",
        t_end=200000000.0,
        n_points=10001,
        warm=10,
        order=11,
    ),
    "prion_aggregation": dict(
        kind="bngl",
        file="models/bngl/prion_aggregation.net",
        t_end=300.0,
        n_points=30001,
        warm=3,
        order=12,
    ),
    "tcr_signaling": dict(
        kind="bngl",
        file="models/bngl/tcr_signaling.net",
        t_end=10000.0,
        n_points=1001,
        warm=3,
        order=13,
    ),
    "erk_activation": dict(
        kind="bngl",
        file="models/bngl/erk_activation.net",
        t_end=8640.0,
        n_points=1001,
        warm=3,
        order=14,
    ),
    # SBML (native .xml for bngsim + roadrunner + copasi)
    "BIOMD0000000862": dict(
        kind="sbml",
        file="models/sbml/BIOMD0000000862.xml",
        t_end=28800.0,
        n_points=101,
        warm=10,
        order=3,
    ),
    "BIOMD0000000344": dict(
        kind="sbml",
        file="models/sbml/BIOMD0000000344.xml",
        t_end=10.0,
        n_points=1001,
        warm=10,
        order=4,
    ),
    "BIOMD0000000860": dict(
        kind="sbml",
        file="models/sbml/BIOMD0000000860.xml",
        t_end=14400.0,
        n_points=1001,
        warm=10,
        order=6,
    ),
    "BIOMD0000000478": dict(
        kind="sbml",
        file="models/sbml/BIOMD0000000478.xml",
        t_end=10.0,
        n_points=1001,
        warm=10,
        order=7,
    ),  # t_end placeholder (confirm from paper)
    "BIOMD0000000035": dict(
        kind="sbml",
        file="models/sbml/BIOMD0000000035.xml",
        t_end=400.0,
        n_points=1000,
        warm=10,
        order=9,
    ),
    "BIOMD0000000864": dict(
        kind="sbml",
        file="models/sbml/BIOMD0000000864.xml",
        t_end=48800.0,
        n_points=101,
        warm=10,
        order=10,
    ),
}

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


def _coverage() -> dict[str, dict[str, object]]:
    cov: dict[str, dict[str, object]] = {}
    for k, m in MODELS.items():
        c = {"bngsim": "ok", "copasi": "ok"}  # bngsim + COPASI cover all 14
        if m["kind"] == "bngl":
            c["run_network"] = "ok"  # native .net
            c["roadrunner"] = (
                "ok"  # converted SBML, all 8 faithful (no repeated reactant, delta~0)
            )
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
    native = HERE / m["file"]
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


# BNG run_network binary
RUN_NETWORK_BIN = "/Users/wish/Simulations/BioNetGen-2.9.3/bin/run_network"
