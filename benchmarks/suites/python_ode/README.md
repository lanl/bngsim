# `python_ode` suite

Pure-Python ODE workflow comparison — BNGsim CVODE versus two
Python-ecosystem ODE engines, on models defined **entirely in Python**
(paper Table S6, ODE half).

Promoted from the ODE half of
`harness/comparison/bench_pythonic_workflows.py` (the SSA half is the
`python_ssa` suite).

## Engines

| Engine | Solver | Model source |
|--------|--------|--------------|
| BNGsim | CVODE (the cross-validation reference) | `ModelBuilder` API |
| scipy | `solve_ivp` LSODA | hand-coded numpy RHS |
| Diffrax | `Kvaerno5`, JIT-compiled JAX, adaptive `PIDController` | hand-coded JAX RHS |

Every model is hand-coded three ways from the same reaction network, so
all three engines integrate an **exact** representation — there is no
parsed-file RHS in the loop. (The predecessor drove scipy/Diffrax from a
hand-rolled `_net_reader`→RHS translation; that path was dropped because
it could not faithfully reproduce BNG's network rate-law semantics and
gave wrong trajectories for scipy *and* Diffrax on functional-rate
models — masked at the time because the predecessor ran timing-only.)

## Models

| Model | sp | Character |
|-------|----|-----------|
| `m1_ground` | 4 | linear decay chain A→B→C→D — non-stiff |
| `sir` | 3 | epidemic S→I→R — non-stiff |
| `lotka_volterra` | 2 | predator/prey oscillator — non-stiff |
| `simple_system` | 4 | phosphorylation cycle, high molecule counts — **stiff** |

The set is curated to span Diffrax's working range: three classic
non-stiff systems it handles cleanly, plus one stiff high-count system
(`simple_system`) that exhausts its adaptive-stepper budget — kept as an
honest data point, not a hidden failure.

## Gates

| Gate | Check |
|------|-------|
| correctness | Each engine's trajectory is cross-validated against BNGsim CVODE on the shared time grid (`rtol=1e-4`, additive `atol` scaled from the per-species peak). |
| timing | Warmup + timed-run median wall time; reported only for an engine that passed cross-validation. |

```sh
python run.py                     # both gates, full sweep
python run.py --mode correctness  # cross-validation only
python run.py --mode timing       # timing only
python run.py --effort low        # cheap subset (m1_ground; cumulative tiers)
python run.py --model sir         # substring filter on model name
```

`--effort` tiers: `low` = `m1_ground`; `medium` adds `sir` +
`lotka_volterra`; `high` adds `simple_system` (the stiff one Diffrax
fails — placed last so the cheap tiers skip its slow adaptive-stepper
grind). Results are written to the git-ignored `results/`
(`python_ode_results.json` + `.md`).
