# `amici_parity` — bngsim vs AMICI (SBML ODE + forward-sensitivity parity)

Cross-engine correctness + efficiency of bngsim's ODE path against
[AMICI](https://github.com/AMICI-dev/AMICI), the SBML/CVODES reference. The AMICI
sibling of `rr_parity` (bngsim vs RoadRunner): same bngsim adapter, same
`_core.differ` oracle, same SBML corpus and HTML matrix format — only the
*reference* engine changes. The AMICI reference build is
pinned to `AMICI-dev/AMICI@667b17b6b` (`v1.0.1-12-g667b17b6b`) — see `AMICI_PIN.json`.

The suite runs **two independent jobs** over the same corpus:

| Job | Compares | Runner | Report | Page |
|---|---|---|---|---|
| ODE | state trajectories `x(t)` | `amici_run.py` | `runs/report_ode.json` | `runs/amici_matrix.html` |
| forward sensitivity | the tensor `dx_i(t)/dp_j` | `amici_sens_run.py` | `runs/report_sens.json` | `runs/amici_sens_matrix.html` |

They share the corpus, the oracle, the verdict taxonomy and the cold/warm timing
taxonomy; they differ in the quantity under test. See
[Forward-sensitivity job](#forward-sensitivity-job-amici_sens_runpy) below. The
older, `.net`/BNGL-driven sensitivity **benchmark** (a 4-model timing table for
the paper, not a corpus sweep) still lives separately at
`benchmarks/suites/forward_sens`.

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
| `amici_sens_run.py` | the **forward-sensitivity** sweep runner; writes `runs/report_sens.json` |
| `_amici_sens.py` | the sensitivity adapters (`bn_sens`, `amici_sens`) + the cross-engine parameter alignment |
| `generate_amici_sens_matrix.py` | renders `runs/report_sens.json` → `runs/amici_sens_matrix.html` |

The SBML corpus and the full ODE manifest are **reused from `rr_parity/`** (no
duplication); model paths resolve under `../rr_parity/`.

## Usage

All commands run from `bngsim/` (the dir with `.venv`, AMICI built into it).

**Prereq (once):** install AMICI. It is a PEP 735 dependency group, so it is not
in the venv by default and `uv sync` without it will *remove* it:

```bash
uv sync --extra dev --group amici    # builds the AMICI_PIN.json commit, ~1-4 min
```

Needs swig, a C++ toolchain and a BLAS (macOS: Accelerate, nothing to install).
See CONTRIBUTING.md for the `git-lfs: command not found` case, which is a stale
`~/.gitconfig` rather than anything to do with AMICI.

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

## Forward-sensitivity job (`amici_sens_run.py`)

The second job. Where the ODE job asks *do the two engines agree on the
trajectory*, this one asks *do they agree on its derivatives with respect to the
model parameters* — the forward-sensitivity tensor `dx_i(t)/dp_j` that both
engines obtain from a coupled CVODES extended-ODE solve — and what the **warm**
per-solve cost of producing it is on each side.

It shares the corpus, the `_core.differ` oracle, the verdict taxonomy and the
cold/warm timing taxonomy with the ODE job — plus one outcome the ODE job never
emits, `UNSUPPORTED` (see [Declared refusals](#declared-refusals-are-unsupported-not-exception)).
What is genuinely new is the parameter alignment, the parameter cap, and the
method pinning.

### Parameter alignment — the one hard part

Species ids match across engines (same SBML), so `align_common` handles them with
one adjustment: AMICI renames any id that collides with a symbol in its own
generated C++ namespace — `x` *is* the state vector there — so `<species id="x">`
comes back as `amici_x`. `_amici_common.align_common` undoes that leading prefix
before intersecting (issue #321); without it, `BIOMD0000000114`/`115`/`346`/`919`
were reported as fully disjoint species sets at `value=inf` despite the engines
agreeing to `max_rel_err=0`. The strip is positional, and an ambiguous one (a
model declaring both `x` and `amici_x`) is dropped rather than guessed at.

Parameter ids do **not** match, because each engine flattens SBML *local*
(per-`kineticLaw`) parameters under its own scheme:

| | SBML reaction `J0`, local parameter `V1` |
|---|---|
| bngsim | `_lp_J0_V1` |
| AMICI | `J0_V1` |

Global parameters keep their SBML id on both sides, so the mapping is a `_lp_`
prefix strip and the shared list is the intersection, minus two exclusions:
bngsim's **compartment-size parameters** (it refuses them as sensitivity targets)
and anything AMICI reports as **fixed** rather than free (no `sx` column exists
for it). A model whose intersection is empty is `BAD_TEST` — no oracle exists —
never a vacuous pass.

Both engines then receive the identical list in the identical order, so the
parameter axis needs no alignment at comparison time. AMICI's parameter scale is
pinned to **linear**, because AMICI can report `dx/d ln(p)` and comparing that
against bngsim's `dx/dp` would differ by a factor of `p` on every cell — a
whole-tensor DIFF that looks like an engine bug.

### The parameter cap (`--param-cap`, default 20)

Forward sensitivity integrates a coupled system of size `n_species*(Np+1)`, so
cost is linear in `Np` and a 100-parameter model is 100× a 1-parameter one.
Uncapped, the big models dominate the wall clock and no two rows' timings are
comparable. The cap picks its parameters by sorting the shared ids and taking an
**evenly spaced** subset — deterministic across re-runs and across machines, and
not clustered on whichever reaction sorts first (a sorted SBML id list groups by
reaction prefix). Every row records the `Np` used *and* the pre-cap candidate
count, so the matrix shows "20 of 43" rather than silently truncating.

### Methods

One job per (model, CVODES corrector method), with **both engines pinned to the
same method** within a job so a timing pair is strictly apples-to-apples:

- `staggered` (CV_STAGGERED) — state advanced first, then the sensitivities as a
  separate linear-in-the-sensitivities solve. CVODES' and bngsim's default.
- `simultaneous` (CV_SIMULTANEOUS) — state and all sensitivity variables advanced
  as one coupled nonlinear system per step. AMICI's compiled-in default.

Running both separates the engine effect from the method effect. The AMICI
compile is shared between them (the corrector method is a solver setting, not a
codegen one), so jobs are scheduled **method-major**: the second method's pass is
load-only.

### State parity is checked too, and reported separately

A sensitivity comparison means nothing if the engines are not on the same
trajectory. Each row carries its own state verdict, kept out of the headline
metric so a sensitivity DIFF on a model whose states already disagree is never
mistaken for a sensitivity-specific bug.

### Declared refusals are `UNSUPPORTED`, not `EXCEPTION`

Forward sensitivity has constructs bngsim *declares* it cannot differentiate, and
raises on rather than answer wrongly:

- an **event** whose crossing time moves with a requested parameter in a way
  `dt*/dp` cannot be computed for, which would leave every post-jump sensitivity
  silently stale (GH #205; `BIOMD0000000342`); and
- a **rate law** codegen cannot differentiate to closed form — a non-smooth
  `min`/`max`/`abs`/`floor`, `rateOf()` inside a rate law, an unparseable
  expression — where the alternative is unreliable finite differences (GH #214).

Both are `bngsim.SensitivityUnsupportedError`, and the runner buckets them
`UNSUPPORTED` (non-scoring), not `EXCEPTION`. This is the same treatment
rr_parity gives `SsaValidationError`, for the same reason: `EXCEPTION` means
*AMICI ran and bngsim broke*, an actionable bug, and mixing documented capability
gaps into it dilutes exactly the signal the bucket exists to carry.

The match is on exception **type**, never on message prefix. These refusals carry
long prose citing GH issue numbers; a prefix match would silently demote every
one of them back to `EXCEPTION` the first time someone rewords a sentence — a
regression with no test that could catch it and no symptom but a quietly worse
number.

**What is NOT a declared refusal.** A codegen *build* failure — a compile
timeout, a `cc` error, no backend at all — stays `EXCEPTION`. The model-path
codegen entry points return the same `None` for "declined" and "failed", so the
second used to be reported as the first: `BIOMD0000000608` generated 66.6 MB of
correctly-differentiated C, blew its 600 s compile budget, and was told its rate
laws were not differentiable. Routing that into a non-scoring bucket would hide a
resource limit instead of merely mislabelling it, so the refusal now consults
`bngsim._codegen.last_codegen_error()` and only refuses as *declared* when
nothing actually failed.

### Reading a failure out of the report

A row's `exception` is capped at 400 characters, because the report is the
durable artifact and 2,646 unbounded tracebacks are not readable. The cap drops
the **middle**, not the tail, and says how much it dropped:

```
bngsim-params: UnderSpecifiedModelError: Parameters 'Ca_SR_DS_Calcium_Concentrations',
'Ca_SR_LCC_and_RyR_fluxes', ... ...[518 chars elided]... ame model). Set the parameter
value or add an initialAssignment. To restore the legacy lenient default-to-0
behavior, set BNGSIM_ALLOW_UNSET_PARAMS=1.
```

That shape is deliberate (issue #324). Several bngsim refusals enumerate model
symbols *before* naming the fault, so a head cut spent the whole budget on names
— `MODEL0848342500`, `MODEL7980735163` and `MODEL9808533471` could not be
classified from the report at all and had to be read against their source.

**Group on `exception_class`, not on `exception`.** Every row carries
`"<phase>:<ExceptionType>"` — `bngsim-params:UnderSpecifiedModelError`,
`amici-build:SBMLException`, `compare:ValueError` — which is stable across models
however long their symbol lists are, and is joined with ` || ` when both engines
raised. The phase is where in the job the raise came from: `amici-build`,
`bngsim-params`, `bngsim`, `amici`, `compare`. A bad_test row with no raise
behind it (no differentiable parameter shared with AMICI) is keyed
`shared-params:none`.

```bash
python3 -c "
import json, collections
rows = json.load(open('runs/report_sens_full.json'))['results']
print(collections.Counter(r['exception_class'] for r in rows if r.get('exception_class')))
"
```

`reference_refusal` (`feature_gap` / `compile` / `integrator` / `other`) is
decided in the worker against the **full** AMICI message, not against the capped
text, so it does not depend on where the cap happened to fall.

### Run + render

```bash
# curated 50-model subset, both methods (the fast default)
.venv/bin/python parity_checks/amici_parity/amici_sens_run.py --workers 4

# render → runs/amici_sens_matrix.html
.venv/bin/python parity_checks/amici_parity/generate_amici_sens_matrix.py
open parity_checks/amici_parity/runs/amici_sens_matrix.html
```

**Full corpus** — the reportable run, all 1323 vendored BioModels SBML:

```bash
.venv/bin/python parity_checks/amici_parity/amici_sens_run.py \
    --manifest parity_checks/rr_parity/ode_jobs.json --workers 4 --timeout 900
```

**Other flags:** `--methods staggered` (one method only) · `--param-cap N`
(`0` = uncapped) · `--models` / `--include` / `--exclude` / `--limit` ·
`--checkpoint runs/sens_ck.jsonl` (JSONL sidecar so a killed run is
reconstructable) · `--out`.

**Know before running:**

- Cold compiles are **heavier than the ODE job's** — the sensitivity build emits
  the `dxdotdp` / sensitivity-RHS C++ on top of the pure-ODE body. The cache is
  `amici_sens_cache/` (gitignored), **separate from `amici_cache/`**: the two
  extensions are not interchangeable and must never collide.
- The cache commits by **atomic rename**, so it is safe under concurrency (both
  methods of a model share a key) and kill-safe: a directory that exists is a
  complete build, so kill/rerun is resumable.
- Needs the `amici` dependency group (`uv sync --extra dev --group amici`). The
  runner puts the venv's `bin/` at the front of `PATH` itself, so cmake finds
  AMICI's pinned `swig` even when the venv was never activated — without that,
  every model fails to compile and the sweep reports a wall of
  `REFERENCE_FAILED/compile` that looks like an AMICI feature gap.
- **Exit code 1 when DIFFs exist** — normal, not a crash.
