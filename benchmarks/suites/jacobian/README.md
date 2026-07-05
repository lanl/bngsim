# `jacobian` suite — analytical vs finite-difference Functional Jacobian

Measures the payoff of the GH #76 analytical Jacobian for Functional /
Michaelis–Menten rate laws (on by default since bngsim 0.9.20) against the
finite-difference (FD) Jacobian it replaces.

## What it does

Two gates per model, mirroring the `ode` suite's correctness + timing design:

1. **correctness** — run each model with the analytical Jacobian (default) and
   with FD forced (`BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0`), and confirm the species
   trajectories agree. They are the *same* solution: the Jacobian only changes
   CVODE's Newton / step-acceptance path, so at a fixed solver tolerance the two
   runs differ only at ~integration-tolerance level (peak-relative ~1e-8;
   collapses to ~1e-12 of peak when rtol is tightened — see
   `diagnose_divergence.py`). The gate is peak-relative (rtol 1e-5, atol
   1e-6·peak), the same convention as `ode`'s `cross_validate`; an element-wise
   relative check would false-fail on species passing through zero.
2. **timing** — min-of-N wall-clock of the `run()` integrate call for each mode
   (model load excluded), the speedup ratio, and the CVODE solver counts. Min
   (not median) is the headline because contention noise is additive; median +
   `rel_spread` are reported so a noisy run is visible.

Every `(model, mode)` runs in a **fresh subprocess** — the analytical-vs-FD
decision is made once, at model-load time, from the environment.

## Running

```
python run.py                      # both gates, full sweep, 25 reps... (use --repeats)
python run.py --mode timing --repeats 25
python run.py --mode correctness
python run.py --effort low         # cheap subset (cumulative tiers)
python run.py --quick              # one analytical run per model, smoke

python probe_attach.py             # per-model: does the analytical Jacobian attach?
python diagnose_divergence.py MODEL9089538076   # tolerance-convergence of a divergence
```

Run the timing sweep as the **sole** heavy job — the large models are
compute-bound and a contended machine inflates the spread badly.

## What the numbers say (and don't)

On the SBML functional models in `parity_checks/rr_parity`, the measured
speedup is **modest (~1.0–1.2×)**. That is an honest result, not a
disappointment, and the reason is the FD baseline:

- **Large models (ns ≥ 50)** use a sparse KLU solver whose FD Jacobian is
  *colored* finite differences (Curtis–Powell–Reid): O(n_colors) RHS evals, not
  O(ns). For the sparse metabolic models here n_colors is small, so FD is
  already cheap and the analytical scatter saves little (~1.01×).
- **Small models (ns < 50)** use the dense path with CVODE's internal O(ns) FD;
  analytical wins per Jacobian, but total time is a few ms and overhead-bound
  (~1.04×).
- The issue's premise — "FD costs O(N) extra RHS evals" — holds only for the
  dense path. For large sparse models the colored-FD baseline already removes
  the O(N).

The analytical Jacobian wins **decisively** only when a model is large **and**
dense (high chromatic number → coloring degrades to O(ns)) with an expensive
RHS — i.e. large rule-based networks. Those are exactly the `.net` per-observable
path (a Functional rate `func(observables)·∏reactants`). The C++ side previously
*rejected* these and routed the whole model to FD; the GH #76 follow-up (task 2)
now scatters their analytical product-rule derivative, so a rule-based functional
network like `egfr_net_red.net` (40 species, 16 per-observable Functional
reactions) attaches and runs ~1.15× faster per step (it was on FD before). The
larger and denser such a network, the bigger the win — colored FD on a dense
functional network degrades toward O(ns) RHS evals, each of which re-evaluates
every Functional rate law.

The feature's primary, fully-delivered value is **correctness + infrastructure**:
an exact Jacobian, validated against FD and RoadRunner across the 1597-model
BioModels SBML corpus (1186 attach, 0 wrong, 0 regressions; the rest correctly
bail to FD), on by default, with a self-check that never ships a wrong Jacobian.

## Models

Five functional SBML models spanning sizes, from
`parity_checks/rr_parity/models/<id>/`, plus one rule-based `.net` network from
`benchmarks/models/net/` that exercises the per-observable path:

| id | ns | note |
|---|---|---|
| BIOMD0000000013 | 27 | Calvin cycle, stiff, dense path |
| MODEL9089538076 | 200 | mid-size, sparse |
| BIOMD0000000595 | 218 | 1490 rxns, reaches steady state fast |
| MODEL9087255381 | 289 | large, sparse |
| MODEL9087474843 | 290 | 497 rxns, large, sparse |
| egfr_net_red (`.net`) | 40 | EGFR reduced, 123 rxns, **16 per-observable Functional reactions** — the GH #76 task-2 path (was FD before) |

All six genuinely **attach** the analytical Jacobian (verified by
`probe_attach.py`) — the ~1× readings are real analytical-vs-colored-FD
comparisons, not analytical-vs-analytical artifacts.

## Outputs

- `results/jacobian_results.json` — full record (machine info, per-model
  correctness + timing + solver counts).
- `results/jacobian_results.md` — rendered report with the interpretation above.
- `results/attach_probe.json`, `results/diag_<id>.json` — diagnostics.
- `results/_*.log` — raw run logs (gitignored scratch).
