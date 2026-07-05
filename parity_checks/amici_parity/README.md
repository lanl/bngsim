# `amici_parity` — bngsim vs AMICI (SBML ODE parity)

Cross-engine correctness + efficiency of bngsim's ODE path against
[AMICI](https://github.com/AMICI-dev/AMICI), the SBML/CVODES reference. The AMICI
sibling of `rr_parity` (bngsim vs RoadRunner): same bngsim adapter, same
`_core.differ` oracle, same SBML corpus and HTML matrix format — only the
*reference* engine changes. ODE-only by design. The AMICI reference build is
pinned to `AMICI-dev/AMICI@667b17b6b` (`v1.0.1-12-g667b17b6b`) — see `AMICI_PIN.json`.

> **Scope note.** This suite was previously a placeholder for sensitivity /
> gradient validation. It is now the **ODE trajectory + timing parity** suite
> (mirroring rr_parity's ODE matrix). Sensitivity validation can return later as a
> separate mode; the forward-sensitivity benchmark still lives at
> `benchmarks/suites/forward_sens`.

## Why AMICI is a useful second reference

rr_parity proves bngsim against one oracle (RoadRunner). AMICI is an *independent*
SBML ODE engine with a different internal design, so it adjudicates models RR
can't disambiguate. Where bngsim ≡ RoadRunner but AMICI diverges (or vice versa),
the suite localizes a real cross-engine difference that a single reference would
hide. AMICI's design also contrasts sharply with RR on the axes the matrix shows:

| | bngsim | RoadRunner | AMICI |
|---|---|---|---|
| RHS backend | ExprTk / cc / MIR | LLVM JIT | **per-model C++ (gcc/clang)** |
| Jacobian | analytical | finite-difference | **analytical (symbolic)** |
| Linear solver | Dense / KLU / LAPACK | dense (built-in LU) | **KLU (sparse)** |
| model build | ~ms–s | ~tens of ms (JIT) | **~20 s (C++ compile, cached)** |

## How it works

Each job runs **both** engines in one disposable subprocess and compares them
directly with the shared `_core.differ` protocol (combined abs+rel per-cell
tolerance + fail-fraction budget + hard ceilings), both forced to a tight shared
integration tolerance (`rtol=1e-9`, `atol=1e-12`). The verdict is derived from
per-engine status — AMICI is the existence proof:

```
both ran, within tolerance ................. PASS
both ran, metric over tolerance / non-finite DIFF
both ran, species/time grids disjoint ...... DIFF (loud, value=inf)
bngsim raised, AMICI ran ................... EXCEPTION (actionable bngsim bug)
bngsim ran, AMICI raised ................... REFERENCE_FAILED (no oracle; non-scoring)
both raised ................................ BAD_TEST (no signal; non-scoring)
wall-clock cap exceeded .................... TIMEOUT
```

The trajectory compared is AMICI's state vector `rdata.x` (named by
`rdata.state_ids`), aligned to bngsim by SBML id over the intersection species —
the same partial-overlap contract rr_parity uses for RoadRunner's floating-species
emission. Two build flags are pinned for parity:

- `compute_conservation_laws=False` — keep every species an independent state, so
  the AMICI state set matches what bngsim/RR emit (CL elimination would drop the
  eliminated species from `rdata.x` and shrink the compared set).
- `observation_model=[]` — no observable / likelihood model. We compare *state
  trajectories*, which AMICI computes identically with or without observables;
  AMICI's default would also build its parameter-estimation likelihood layer
  (`y`/`sigmay`/`Jy`/`dJydy`, the NLL and its derivatives) which forward
  simulation never uses and for which bngsim/RR build nothing comparable. Skipping
  it makes the build a true pure-ODE cost (≈halves compile on small models) and
  keeps the comparison apples-to-apples. The analytical Jacobian and state
  dynamics are unaffected.

### No AMICI source patch needed — its build self-times

Unlike RoadRunner — whose `RoadRunner(xml)` fuses parse+interpret+JIT, forcing a
C++-instrumented build to split them — AMICI **self-times every build phase** via
its `@log_execution_time` decorator. We lower AMICI's build loggers to DEBUG and
capture those records to decompose the otherwise-opaque `sbml2amici()` into a full
per-phase breakdown — no AMICI source patch, since the only C++ step (the compile)
is itself a single timed phase:

```
parse     = "loading/validating SBML"          # libSBML parse
interpret = "processing SBML *"                 # SBML → symbolic model
jac       = "computing dwdx/dxdotdw/…"          # analytic-Jacobian symbolic chain
codegen   = "generating cpp code" − jac         # C++ source emission only
compile   = "compiling cpp code"                # cmake + ninja + swig + link  ← dominant
load      = import_model_module(...)            # load the compiled .so
integrate = cold + warm model.simulate() reps   # shared cold/warm taxonomy
```

Empirically the **compile dominates** (~95% of build on small models — a ~20-30 s
floor that is the per-model C++ build, not the symbolic work), while the analytic-
Jacobian derivation is milliseconds (and grows with model size). `rdata.cpu_time`
(pure CVODES, ms) cross-checks the Python-wall integrate numbers. Headline
efficiency is the **warm** (per-integration) cost.

The report retains this full split, but the **matrix display folds `codegen` +
`compile` + `load` into a single "RHS build" row** — compilation is a *sub-step*
of building the callable RHS/Jacobian evaluator, not a peer phase, so showing them
as separate rows is misleading (and bngsim can't be split that way: it reports one
inseparable `codegen` number, and ExprTk has no compile at all). "RHS build" thus
means the same thing for every engine: generate the evaluator code and compile it
(bngsim: ~ms ExprTk / ~s native-C; AMICI: ~20 s C++). The Jacobian *derivation* is
its own row; parse and interpret stay separate too.

### Compiled-model cache

AMICI generates and compiles a bespoke C++ extension per model. These are cached
on disk in `amici_cache/` keyed by SBML-content hash (gitignored), so the first
sweep pays the cold compile and every re-run is load-only.

## Overrides — departure from rr_parity

Only the engine-agnostic **`tol`** overrides (ill-conditioned IVPs, applied to
both engines) are honored. rr_parity's `known_artifact` / `invalid_reference` /
`no_oracle_adjudicated` overrides are calibrated against *RoadRunner*; applying
them here could mask a genuine bngsim-vs-AMICI difference, so they are **not**
applied. AMICI adjudicates every model independently.

## Files

| File | Role |
|---|---|
| `amici_run.py` | the sweep runner (fork of `rr_parity/rr_run.py`); writes `runs/report_ode.json` |
| `_amici_common.py` | the AMICI reference adapter (`amici_ode`) + warmup; reuses rr_parity's `bn_ode` and engine-agnostic helpers |
| `build_amici_jobs.py` | builds the curated subset `amici_ode_jobs.json` (stratified by species count + feature coverage) |
| `generate_amici_matrix.py` | renders `runs/report_ode.json` → `runs/amici_matrix.html` (fork of rr_parity's generator) |
| `amici_ode_jobs.json` | the curated job manifest (model paths resolve under `rr_parity/`) |

The SBML corpus and the full ODE manifest are **reused from `rr_parity/`** (no
duplication); model paths resolve under `../rr_parity/`.

## Usage

All commands run from `bngsim/` (the dir with `.venv`, AMICI built into it).

**Prereq (once):** the SBML corpus is gitignored — materialize it if
`parity_checks/rr_parity/models/` is empty:

```bash
.venv/bin/python parity_checks/rr_parity/materialize.py
```

**Run + render:**

```bash
# sweep the curated 50-model subset (default manifest, committed)
.venv/bin/python parity_checks/amici_parity/amici_run.py --workers 4

# render → runs/amici_matrix.html
.venv/bin/python parity_checks/amici_parity/generate_amici_matrix.py
open parity_checks/amici_parity/runs/amici_matrix.html
```

**Full corpus** (the big job — ~1300 models, ~3–3.5 h):

```bash
.venv/bin/python parity_checks/amici_parity/amici_run.py \
    --manifest parity_checks/rr_parity/ode_jobs.json --workers 4 --timeout 600
```

**Other flags:** `--models BIOMD0000000012,BIOMD0000000010` (subset by id) ·
`--limit N` · `--out report.json`. Re-generate the curated subset with
`build_amici_jobs.py` (rarely needed; it's committed). Render is decoupled from
the run: `generate_amici_matrix.py --report <path>` points at any report.

**Know before running:**

- First run **cold-compiles each model (~20 s, C++)**; results cache to
  `amici_cache/` (gitignored) → **resumable**: kill/rerun skips what's built.
- On the full corpus, **~16 giant models (>500 species) TIMEOUT** at the
  `--timeout` cap — expected and bounded (AMICI can't compile them in reasonable
  time); everything else gets a verdict. Raise `--timeout` / drop `--workers` to 2
  to push the giants (RAM-heavy).
- **Exit code 1 when DIFFs exist** — normal, not a crash.
- Output `runs/report_ode.json` + `runs/amici_matrix.html` are gitignored
  (regenerable). Versions stamped via `_core.versions.amici_version()` (the live
  package version); the exact AMICI reference build is pinned in `AMICI_PIN.json`.
