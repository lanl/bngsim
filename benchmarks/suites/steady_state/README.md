# `steady_state` suite

Benchmarks the wall-clock cost of computing dose-response curves under
five distinct steady-state strategies — the KINSOL dose-response table
(paper Table S8; see the table-number note in `_dev/phase4_plan.md`).

Promoted from `harness/comparison/bench_kinsol_dose_response.py`.

## Strategies

| Column | Strategy |
|--------|----------|
| A | `run_network` long-time integration (BNG2.pl subprocess). |
| B | BNGsim CVODE long-time integration (in-process). |
| C | BNGsim KINSOL — Newton-first with simulation fallback (the policy BNGsim `method="auto"` implements internally). |
| D | BNGsim KINSOL — integration burst, then strict Newton refinement. |

For each model a dose parameter is scanned; per dose the script measures
the settling time `t_s` (strict "enter and never leave" of the 99.73%
band) and times every strategy at the integration horizon `t_end = t_s`.

## Models

The corpus is vendored in-repo at `../../models/net/ode/` — four
ODE models (`RAFi_ground`, `Scaff_22_ground`, `mwc`,
`wofsy_goldstein`), each with a designated dose-scan parameter.

## Gates

| Gate | Check |
|------|-------|
| correctness | Every engine that reports a steady state is cross-checked against the `y_ss` reference (strict Newton, or the long-horizon CVODE tail when Newton does not converge). A dose passes when all engines agree within `rtol = 1e-2` — just above the 99.73% settling band. |
| timing | Warmup + timed-run wall-clock comparison, median reported. |

Per the suite design rule, **timing is only reported for doses that
passed correctness** — a wall time is meaningless if the engines landed
on different steady states.

```sh
python run.py                     # both gates, full sweep
python run.py --mode correctness  # cross-check only
python run.py --mode timing       # timing only
python run.py --effort low        # cheap subset (cumulative tiers)
python run.py --quick 5           # limit scan points per model
python run.py --model RAFi        # substring filter on model name
```

`run_network` is located via `BNGPATH` / `RUN_NETWORK` (see the
top-level `benchmarks/README.md`). Results are written to the
git-ignored `results/` (`steady_state_results.json` + `.md`).

## Outcome

This suite settled the steady-state default. `method="newton"` lost to
`method="integration"` on all six models — 1.4–3.9×, geometric mean 2.5× —
because #27 made the two-tier solver integrate *first*, so its KINSOL polish
is work layered on top of the integration path rather than an alternative to
it. `steady_state()` now defaults to `method="integration"` (issue #28); the
`newton` columns stay in the table as the standing regression check that the
polish has not silently become the cheaper path on some model class. The suite
exercises both paths explicitly, so it is unaffected by the default itself.
