# ssa_table5 — four-engine exact-SSA timing harness

Measures per-model **exact Gillespie SSA** cost (cold + warm) across four engines —
**BNGsim** (`method="ssa"`), BNG **`run_network -p ssa`**, **RoadRunner** `gillespie`,
**COPASI** Direct method — over the 14-model corpus (`corpus.json`; 8 BNGL + 6 SBML).

## Pipeline

```
../../models/regenerate_curated_nets.py --check
                      # are the 8 BNGL .net files still what the curated BNGL-Models
                      #   records generate? run this BEFORE a final timing run
convert_all.py        # bngsim converter only: .net->SBML (RR+COPASI) & SBML->.net (run_network)
                      #   -> results/converted/*.{xml,net} + conversion_log.json  (fixes coverage)
run_ssa_timing.py     # orchestrate the 14x4 matrix; 6 isolated cell subprocesses, incremental save
                      #   -> results/ssa_timing_ballpark.json
emit_ssa_table.py     # -> results/ssa_timing_ballpark.md  (+ enrich the json with per-cell reasons)
merge_jobout.py       # re-assemble the json from results/_jobout/*.json (after re-running any cell)
```

`corpus.json` is the SSOT for the model set: artifacts, horizons and output-point
counts are declared there once and read by `_ssa_config.py`, `convert_all.py`,
`run_bngsim_activity.py` and `emit_ssa_table.py`.

Supporting modules: `_ssa_config.py` (runner policy — cheap→expensive order, warm-N,
coverage authority, per-engine artifact resolution — over the corpus it loads from
`corpus.json`) and `_ssa_cell.py` (one isolated `(engine, model)` cell: load + cold +
warm).

## Definitions

- **COLD** = wall of `load/convert + build + first run`.
- **WARM** = median wall of the next *N* runs reusing the loaded model (`reset()`/reseed
  + run), N=10 for most, **N=3 for erk/prion/tcr**; warm stops once cumulative warm wall
  exceeds `WARM_BUDGET_SEC` (150 s).
- **events/run, events/s** — BNGsim reports its own firing count (`solver_stats["n_steps"]`);
  the other engines don't, so their events use the BNGsim reference at the same
  model+horizon (`events_self=false`, flagged `†`).
- **Per-run cap** 120 s. BNGsim uses its native `run(timeout=)`; run_network uses the
  subprocess timeout; RoadRunner uses a SIGALRM guard (its `simulate` is interruptible);
  **COPASI's `process()` is NOT signal-interruptible** (a SIGALRM corrupts it into an early
  false failure), so its 120 s cap is enforced **post-hoc by wall time**, with the
  orchestrator's 300 s hard cell cap (SIGKILL of the cell subprocess) as the hang backstop.

## Run it

`$PY` is any interpreter with bngsim importable — the repo's `.venv/bin/python3`,
or whatever `python` your environment resolves to. `run_network` is located the
same way as in every other suite: `$RUN_NETWORK`, else `$BNGPATH/bin/run_network`,
else `~/Simulations/BioNetGen-2.9.3/bin/run_network`.

```bash
export BNGPATH=~/Simulations/BioNetGen-2.9.3        # wherever your BNG 2.9.3 lives
PY=python3
cd benchmarks/suites/ssa_table5
$PY ../../models/regenerate_curated_nets.py --check     # artifacts still current?
$PY convert_all.py                                      # once; caches converted artifacts
$PY run_ssa_timing.py --workers 6                       # BALLPARK (contention)
$PY emit_ssa_table.py

# FINAL numbers — clean serial re-run (no contention):
$PY run_ssa_timing.py --workers 1 && $PY emit_ssa_table.py
```

`--only m1,m2` restricts models; `--engines bngsim,copasi` restricts engines;
`--warm-override N` overrides warm-N (used for smoke tests).

> ⚠️ The committed `results/ssa_timing_ballpark.json` is a **ballpark** collected under
> 6-way concurrency to stand up the harness. Final table numbers require the serial
> (`--workers 1`) re-run.

## Known engine limitations surfaced (not harness bugs)

- **run_network** on SBML `.net`s that lost time-triggered events in conversion
  (860/862/864/344) → N/A; on a converted `.net` with a functional rate law (478 R4) →
  fails (`-p ssa` has no functional propensities). The `edgepop` observable crash noted
  here previously was a property of a superseded prion artifact; the curated
  `prion_aggregation.net` runs under `-p ssa` and under PSA (smoke-checked — re-confirm
  at the full `t_end=300` in the serial run).
- **RoadRunner** `gillespie` won't fire time-triggered events (860/862/864 N/A) and warns
  "time not treated continuously"; the 344 state-triggered event it also won't fire (run
  but flagged). It also fires the SBML law `k*N*N` where the exact propensity for a
  repeated reactant is `k*N*(N-1)` (GH #9), which takes out `samoilov_futile_cycle`
  (N/A) — the curated record includes the driver step `N + N -> E+ + N`.
- **COPASI** needs quantity unit `#` (particle number) for molecule-count models it imports
  as `mol` (converted-BNGL + native BIOMD035), else it refuses ("particle number too big").
