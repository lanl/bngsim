# `copasi_parity` — bngsim vs COPASI (SBML, ODE)

Validates bngsim's SBML engine against [COPASI](http://copasi.org/) on the same
1323-model curated-BioModels ODE corpus as `rr_parity`. The COPASI sibling of
`rr_parity/rr_run.py` (RoadRunner) and `amici_parity/amici_run.py` (AMICI): it
reuses `rr_parity`'s bngsim adapter (`bn_ode`), the shared `_core.differ` oracle,
the process scheduler, the SBML corpus under `rr_parity/models/`, and the job
manifest `rr_parity/ode_jobs.json` — only the reference engine changes.

```
copasi_parity/
  _copasi_common.py      the co_ode reference adapter (+ re-exports from _rr_common)
  copasi_run.py          the ODE runner: bngsim vs COPASI -> _core report
  copasi_dispositions.py COPASI-calibrated per-row dispositions (issue #1550)
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
- **Start time.** COPASI integrates from the model's own initial time whatever
  output start it is given, while `bn_ode` starts the model's initial state at the
  time it is given. So `copasi_run` gives both engines the job's SED-ML
  `initial_time`, as `rr_run` does (GH #19), and compares `n_points` over
  `[initial_time, t_end]`. Until issue #1550 it gave them the outputStartTime
  (`t_start`), and on the 19 jobs whose window starts after the model does, bngsim
  solved the model shifted in time by the gap. Twelve of the 42 DIFFs at bngsim
  0.15.1 are among those 19, and BIOMD0000000569 turned from PASS to DIFF at 0.17.0
  once #873's differ stopped forgiving it. Started at the initial time, on 0.17.0,
  569 and 11 of the 12 PASS (BIOMD 42, 691, 769, 770, 829, 834, 878, 884, 943, 975,
  1005), as does 925, whose COPASI run on the finer offset grid had returned
  non-finite values. BIOMD0000000627 stays
  a DIFF on its own account.
- **Output grid.** COPASI's uniform timecourse grid is anchored at the model's
  initial time, so it reproduces `linspace(t_start, t_end, n_points)` exactly only
  when `t_start` lands on that grid. Every corpus job now records from the model's
  initial time, which does. For a caller that asks for a window that does not
  (e.g. `[100, 400]` step 0.3), `co_ode` integrates on a finer grid and
  cubic-spline-interpolates onto the exact target grid — knot-exact where the grids
  coincide, so it removes the spurious time-grid mismatch without manufacturing a
  divergence.

## Comparison protocol

Identical to `rr_parity` / `amici_parity`: both engines forced to a tight shared
integration tolerance (`rtol=1e-9`, `atol=1e-12`), compared over the common
(intersection) species in concentration units via `_core.differ.deterministic_verdict`,
which is given that `atol` for its movement gate (#873), as the other runners give it.
Failure is attributed per engine (both always run): bngsim raised + COPASI ran →
`EXCEPTION`; bngsim ran + COPASI raised → `REFERENCE_FAILED` (non-scoring); both
raised → `BAD_TEST`; wall-cap overrun → `TIMEOUT`. Only the engine-agnostic `tol`
overrides are honored; rr_parity's RoadRunner-calibrated overrides are intentionally
NOT applied, so COPASI adjudicates every model independently.

## COPASI-calibrated dispositions (issue #1550)

Where COPASI is the wrong engine, an `INVALID_REFERENCE` entry in
`copasi_dispositions.py` turns the row's `DIFF` into a non-scoring
`REFERENCE_FAILED` with the `invalid_result` refusal class. It is AMICI's mechanism
(`amici_dispositions.apply_disposition`), so the rules are the same: an entry cites
evidence from an oracle that is not bngsim, a row that PASSes on its own reports the
entry STALE, and a fresh `max_rel` more than a factor of 2 from the recorded one
reports it DRIFTED. bngsim and RoadRunner agree on every cell of both rows.

| model | what COPASI does | evidence beyond bngsim |
|---|---|---|
| `BIOMD0000000183` | Imports four rate laws, `compartment * species * kRT * b^(1/2)`, as mass action with `k1 = kRT`, dropping `b^(1/2)` | At one state, those four COPASI rates are RoadRunner's divided by `sqrt(b)` and the other 348 agree; a one-reaction model reproduces it |
| `BIOMD0000000285` | Does not fire `PIdeath > 0` and `p38death > 0`, which start at 0 with zero slope, even with `initialValue = false` (the cell lives, NatP 19520.1 at t = 10, as in AMICI, #1546) | RoadRunner fires both; COPASI also misses `B > 0` with B held at 0 until t = 0.5 and rising after, which every reading of the start rule fires |

Both rows read PASS at bngsim 0.15.1. 183 failed 24873 cells then, all forgiven by
the differ before #873. On 285 bngsim did not fire the events either (#1150 changed
that, by the decision on #934), and that differ forgives today's disagreement too
(10759 cells failing, 0 hard). `test_copasi_parity.py` pins both
COPASI behaviors on minimal models, where COPASI and RoadRunner are installed.
