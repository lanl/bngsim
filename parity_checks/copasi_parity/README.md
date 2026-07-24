# `copasi_parity` — bngsim vs COPASI (SBML, ODE)

Validates bngsim's SBML engine against [COPASI](http://copasi.org/) on the same
1323-model curated-BioModels ODE corpus as `rr_parity`. The COPASI sibling of
`rr_parity/rr_run.py` (RoadRunner) and `amici_parity/amici_run.py` (AMICI): it
reuses `rr_parity`'s bngsim adapter (`bn_ode`), the shared `_core.differ` oracle,
the process scheduler, the SBML corpus under `rr_parity/models/`, and the job
manifest `rr_parity/ode_jobs.json` — only the reference engine changes.

```
copasi_parity/
  _copasi_common.py   the co_ode reference adapter (+ re-exports from _rr_common)
  copasi_run.py       the ODE runner: bngsim vs COPASI -> _core report
  runs/               per-run _core reports — GITIGNORED
```

## Running

```sh
cd bngsim
python parity_checks/rr_parity/materialize.py            # place the gitignored SBML tree first
.venv/bin/python parity_checks/copasi_parity/copasi_run.py --workers 8
.venv/bin/python parity_checks/copasi_parity/copasi_run.py --models BIOMD0000000012
```

Writes a `_core` report to `runs/report_ode.json` (same schema as the other two
suites; `_meta.tally` is the coverage breakdown). COPASI runs through the same
`python-copasi` build already in the venv; there is **no** per-model codegen/compile
step (contrast AMICI), so a sweep is as cheap as `rr_parity`.

## The COPASI adapter (`co_ode`)

`co_ode(xml, t_start, t_end, n_points, rtol, atol)` returns `(time, values, names,
timing)` with the same signature as `rr_ode` / `amici_ode`, so the shared comparison
path is untouched. Three COPASI-specific points, each verified against a bngsim/RR
divergence during development:

- **Locale.** Importing the COPASI SWIG module resets the process `LC_CTYPE` to
  ASCII, which then breaks bngsim's UTF-8 SBML file reads. `_import_copasi()` saves
  and restores `LC_CTYPE` around the (one-time) import so the reference engine cannot
  corrupt the test engine.
- **State selection.** COPASI's time series is keyed by SBML id and kept for *every*
  dynamic quantity, not only metabolites — Hodgkin-Huxley `V, m, h, n` and
  FitzHugh-Nagumo `v, u` live as SBML *parameters* with rate rules, so a
  metabolites-only filter would return an empty species set and manufacture a
  spurious `disjoint species` DIFF. `align_common` intersects with bngsim's states.
- **Output grid.** COPASI's uniform timecourse grid is anchored at the model's
  initial time, so it reproduces `linspace(t_start, t_end, n_points)` exactly only
  when `t_start` lands on that grid (the common `t_start == 0` case). For the ~20
  curated windows that do not start at 0 (e.g. `[100, 400]`), `co_ode` integrates on
  a finer grid and cubic-spline-interpolates onto the exact target grid — knot-exact
  where the grids coincide, so it removes the spurious time-grid mismatch without
  manufacturing a divergence.

## Comparison protocol

Identical to `rr_parity` / `amici_parity`: both engines forced to a tight shared
integration tolerance (`rtol=1e-9`, `atol=1e-12`), compared over the common
(intersection) species in concentration units via `_core.differ.deterministic_verdict`.
Failure is attributed per engine (both always run): bngsim raised + COPASI ran →
`EXCEPTION`; bngsim ran + COPASI raised → `REFERENCE_FAILED` (non-scoring); both
raised → `BAD_TEST`; wall-cap overrun → `TIMEOUT`. Only the engine-agnostic `tol`
overrides are honored; rr_parity's RoadRunner-calibrated overrides are intentionally
NOT applied, so COPASI adjudicates every model independently.
