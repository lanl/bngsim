# `bng_parity` — bngsim vs the legacy BioNetGen stack (BNGL/.net)

Validates bngsim against the legacy BNG stack (BNG2.pl → run_network / NFsim)
on the BNGL corpus the PyBioNetGen parity effort assembled: ODE, SSA, and NF.

## Layout

```
bng_parity/
  models/<tier>/<source>/<relpath>.bngl   896 vendored models (+11 companions)
  manifest.json / manifest.csv            vendor_corpus.py output (provenance + sha256)
  jobs.json                               _core manifest (896 jobs) — the SPEC
  vendor_corpus.py                        fetches the corpus from pinned GitHub commits
  overrides.py                            per-model fixtures lifted from parity_sweep
  build_jobs.py                           manifest.json + overrides.py -> jobs.json
  bng_ode_run.py / bng_stoch_run.py       parity+timing runners (vs legacy BNG) -> report_*.json
  generate_bng_matrix.py                  report_*.json -> bng_matrix_*.html
  parity_golden.py / parity_sweep.py      bngsim-only golden references (+ its sweeper)
  parity_diff.py                          .gdat/.cdat I/O helpers (load_normalized)
```

## Corpus (896 models, fetched from pinned GitHub commits)

`vendor_corpus.py` materializes the corpus from three **public** repos, each
pinned to a commit — no local-checkout, and no PyBioNetGen, dependency. It
shallow-fetches each pin (`git fetch --depth 1 origin <sha>`) into a gitignored
`.sources_cache/`, resolves every model, copies it (+ run companions) into
`models/`, and rewrites the manifest with provenance + recomputed sha256:

| `source` | n | repo | pin |
|---|---|---|---|
| `bngl_models` (tier `original`) | 377 | `wshlavacek/BNGL-Models` | `0df6cbd` |
| rulehub (fast/slow/glacial) | 339 | `RuleWorld/RuleHub` | `479d6d6` |
| rulemonkey | 130 | `richardposner/RuleMonkey` | `0f70112` |
| bngl_library | 49 | `wshlavacek/BNGL-Models` (`bngl_models/`) | `0df6cbd` |
| curated (Creamer) | 1 | `wshlavacek/BNGL-Models` (`creamer/`) | `0df6cbd` |

Bump a pin in `vendor_corpus.py`'s `PINS` (a deliberate, reviewed change) to
move the corpus to newer upstream. A developer who already has a checkout *at
the pinned commit* can point `$RULEHUB_DIR` / `$RULEMONKEY_DIR` /
`$BNGL_MODELS_DIR` at it to skip the clone; a wrong-commit checkout surfaces as
`sha changed` in the run summary.

**Derived fixtures (hosted in BNGL-Models, not referenced upstream).** Five
models were edited during the parity effort and don't exist upstream as-run, so
they live in `wshlavacek/BNGL-Models` with provenance headers naming their
origin + transform:

- `creamer/Creamer_2012.bngl` — RuleHub tutorial with no upstream simulation
  protocol; 542 params constant-folded, seed populations ×0.01, NFsim actions
  appended (`source=curated`).
- `bench_rulehub/{tcr_gen20ind9,tcr_iter28p4h2,tcr_iter9p44,example3_fit}.bngl`
  — RuleHub `Published/{Mitra2019,Thomas2016}` fit models with the
  equilibration/ligand-add `t_end` (and example3's `parameter_scan`) shortened
  so the NFsim parity run is tractable; model bodies unchanged from source.
  Kept under `source=rulehub` (`bench_rulehub/` relpath) in membership but
  resolved from BNGL-Models — see `resolve()` in `vendor_corpus.py`.

**Membership.** Which 896 models is read from the committed `manifest.json`'s
`(tier, source, relpath)` triples (the frozen result of PyBioNetGen's
`dev/parity_report_FINAL_0.9.7.json` per_model keys + the Lin2019
`ERK_model`/`prion_model`). The report itself is no longer consulted.

Companion files needed to *run a simulation* (`.tfun`, `.net`, `.species`,
`.xml`, ...) are vendored beside their model; repo cruft (READMEs, CI yaml,
source) is not — and the PyBNF/BioNetFit fitting artifacts `.conf` (job configs)
and `.exp` (fit-target data tables) are **excluded** (not simulation
dependencies: fitting parity is PyBNF's). Each model carries only its own bare
`<stem>.net`; per-`simulate` `<stem>_<suffix>.net` snapshots are dropped
(regenerated each run). Output names run through `windows_safe()`. Re-vendor:
`python vendor_corpus.py`.

568 deterministic, 328 stochastic. 104 carry `__FREE` — these are *filled*
best-fit models (real parameter values), not the unfilled PyBNF fitting
templates (which were excluded upstream and are PyBNF's parity, not ours).

## Overrides (lifted from hardcoded dicts)

`overrides.py` holds the per-model parity fixtures (t_end caps, tolerance
tightening, `frac` symbol renames, action injection, timeouts) that used to be
basename-keyed globals in `parity_sweep.py`, now each with a mandatory reason.
`build_jobs.py` resolves them onto vendored models and reports hygiene:

- **basename collisions** — 6 basenames (`BLBR.bngl`, `PushPull.bngl`, ...) map
  to 2 vendored models each; the override fans out to all. Recorded in
  `jobs.json` `_meta.override_basename_collisions`. Migrating overrides to
  relpath/sha256 keying (the plan's intent) is the disambiguation step.
- **stale keys** — 3 `frac` overrides (`V1988a_endemic_infection.bngl` etc.)
  match no vendored model: they target models added *after* the FINAL_0.9.7
  report (the `frac` fix postdates it). Left in `overrides.py` for when the
  corpus is refreshed; flagged in `_meta.override_stale_keys`.

## Running the sweep without freezing the box

`parity_sweep.py` shells each model out to a throwaway `python -c` subprocess (so a
crash/leak is isolated to one job): the **subprocess** side runs
`bionetgen.run(...)` → `BNG2.pl → run_network/NFsim`; the **bngsim** side runs
`_bng_common.run_bngsim_job` → `BNG2.pl` netgen + bngsim in-process (GH #175).
Either way a "worker" is a process *subtree*, and a single NFsim/RuleMonkey model
can hold several GB. Eight of those on 16 GB RAM overruns
memory; with a near-full disk macOS can't grow swap and the kernel jetsams
(WindowServer included) → frozen machine. Guards (all on by default):

- **Workers capped by RAM, not just the flag.** `--workers` (default **4**) is
  reduced to `min(flag, cores, available_RAM // --mem-per-worker-gb)`. On a
  16 GB box at idle this lands around 3.
- **Per-launch memory floor.** A worker waits to start its sim while free RAM is
  below `--mem-floor-gb` (default 3.0). No deadlock: parked workers consume
  nothing, so RAM recovers and they proceed.
- **Disk pre-flight.** Aborts below 8 GB free / warns below 20 GB on the output
  volume (swap headroom). The startup banner prints workers, RAM, and disk.

Override only if you know the headroom: `--workers N --mem-per-worker-gb G
--mem-floor-gb F`. Without `psutil` the governor is off — keep `--workers` low.

## Running the parity check

The live parity + timing entrypoint is the **matrix** (detailed under "ODE timing
+ parity matrix" and "Stochastic timing + parity matrix" below). The runners drive
bngsim and the legacy reference engine on each `jobs.json` model, compare via the
shared `_core.differ` oracle, and emit a `_core` report that `generate_bng_matrix.py`
renders to an HTML table:

```
export BNGPATH=/path/to/BioNetGen-2.9.3      # legacy reference (BNG2.pl + run_network + NFsim)
python bng_ode_run.py   --workers 4                  # ODE -> runs/report_ode.json
python bng_stoch_run.py --track ssa --workers 4      # SSA -> runs/report_ssa.json
python bng_stoch_run.py --track nf  --workers 4      # NF  -> runs/report_nf.json
python generate_bng_matrix.py runs/report_ode.json   # -> runs/bng_matrix_ode.html
```

- **Verdict kernel** = the shared `_core.differ` (`deterministic_verdict` /
  `ensemble_verdict`) — one source of truth across `bng_parity`, `rr_parity`, and
  `amici_parity`.
- **Adjudicated dispositions** (a DIFF that is a known reference-engine bug or a
  benign sampling artifact, *not* a bngsim defect) live in the matrix runners
  themselves: `KNOWN_DETERMINISTIC_ARTIFACTS` (ODE, `bng_ode_run.py`) and
  `NF_KNOWN_REF` / `SSA_KNOWN_DISPOSITION` (stochastic, `bng_stoch_run.py`). Each
  renders KNOWN ARTIFACT / KNOWN REF — non-scoring — with its independent-oracle
  evidence in the row comment. The matrix is the single adjudication home.
- **Reference engine** = legacy BNG from `$BNGPATH`/`$BNG2_PL` (no hardcoded
  paths). Verified against BNG 2.9.3.

> **Retired (GH #69).** The earlier `_core` sweep — `parity_core_run.py` /
> `parity_run.py` driving `parity_diff.py`'s *own* copy of the cross-engine
> comparison plus a hand-maintained KNOWN_ARTIFACT catalog — was a divergent
> duplicate of the verdict and has been removed. `parity_diff.py` now holds only its
> `.gdat`/`.cdat` I/O helpers (`load_normalized`), which `parity_golden.py` uses.
### Reproducible BNGsim-backend env (pin · bootstrap · audit)

The `--simulator bngsim` route drives **genuine bngsim directly** via
`_bng_common.run_bngsim_job` (GH #175): BNG2.pl generates the network (`.net`) /
BNG-XML (`.xml`) — the one thing bngsim can't do (no BNGL parser) — then bngsim
simulates **in-process** through the bridge's *direct* route
(`execute_bngsim_direct_job`). It does **not** call `bionetgen.run(simulator='bngsim')`:
with a stock BNG2.pl that routes a BNGL `simulate` through the BNG2.pl-owned backend
hook (`ROUTE_BNGL_BNGSIM`), and since the shipped BNG2.pl carries no such hook it
runs `run_network`/NFsim **natively** — producing BNG2.pl output mislabelled
"bngsim" (the bug GH #175 fixed; the old golden was BNG2.pl-vs-bngsim mislabelled).
Engine selection is **method-faithful** — the model gets the engine it asks for:
ode/ssa/psa run on the reaction network, `method=>"nf"` runs on **NFsim** and
`method=>"rm"` on **RuleMonkey** (both vendored into bngsim; they are swappable
network-free engines but one is never substituted for the other), and a job whose
only method is unsupported (`pla`) is **skipped** — never silently run on the legacy
stack. To make this identical and provable on every machine, three pieces:

- **Pin (`../requirements-pybionetgen.txt`).** PyBioNetGen pinned to the exact
  RuleWorld commit (`git+…RuleWorld/PyBioNetGen@5109a46`) — no local checkout, no
  PR-branch, no `PYTHONPATH`. bngsim is not on PyPI, so it ships from this repo's
  wheel (`scripts/ship_wheel.py`).
- **Bootstrap (`bootstrap_parity_env.py`).** Turnkey: `uv venv` → install the
  pinned PyBioNetGen (handling its build-time `pip install numpy` + BNG download
  via `--no-build-isolation`) → install the bngsim wheel → **verify the backend is
  live** in the new env. `--check-only` verifies an existing env. A machine that
  can't run bngsim fails *here*, loudly, not silently mid-sweep.
  ```
  python bng_parity/bootstrap_parity_env.py --venv .venv-parity --build-bngsim
  python bng_parity/bootstrap_parity_env.py --check-only --python .venv-parity/bin/python3
  ```
- **Audit (`bngsim_backend.py`).** Two layers: (1) `backend_status()` — the
  pre-flight gate (bngsim importable + version-compatible) plus provenance
  (`bionetgen_commit`, recorded in `_summary.json` and golden `_meta`); (2)
  `predict_engine()` / `audit_unit_engines()` — classify **every** job the way the
  driver runs it (`_bng_common.classify_bngsim_track`) and report which bngsim
  engine each uses. `parity_sweep` runs this on every bngsim sweep, prints the
  breakdown (`engine plan: bngsim=… (ode=… nf=… psa=… rm=… ssa=…) unsupported/skip=…`),
  records it in `_summary.json`, and with **`--strict-engine`** aborts if any job
  bngsim can't run is in the set. The **engine pre-flight** then PROVES execution:
  a probe driven through `run_bngsim_job` must write numeric output and emit the
  per-job `BNGSIM_ENGINE_OK` marker before the sweep starts. `parity_golden`
  carries the audit + provenance into golden `_meta`.

**No silent engine substitution.** `run_bngsim_job` never falls back: a job either
runs on the bngsim engine its method selects, or it raises/skips (an unsupported
`pla`, an unresolvable horizon, or a missing vendored engine on a broken build are
all hard errors — there is no quiet downgrade). The decisive regression lives in
`tests/test_bngsim_golden_engine.py`: with the bundled `run_network` moved aside,
`run_bngsim_job` still produces output (it never needs run_network) while the old
`bionetgen.run(simulator='bngsim')` BNGL route crashes — proving the genuine path
is bngsim and the old one was BNG2.pl.

The timing matrices (`bng_ode_run.py` / `bng_stoch_run.py`) drive bngsim in-process
the same way (`Model.from_net` → `Simulator`, `NfsimSession`/`RuleMonkeySession`)
and need only BNG2.pl (`$BNGPATH`) for network generation.

## ODE timing + parity matrix (`bng_ode_run.py` → `generate_bng_matrix.py`)

The sibling of `rr_parity`/`amici_parity`'s ODE matrices: a focused **timing +
correctness** track that produces the same HTML summary (row-color taxonomy,
verdict badges, three-tier timing, per-integration speedups, wins-by-species
table). Where the `_core` sweep above answers *do the trajectories agree?* by
diffing two `bionetgen.run()` subprocesses (and only captures total wall), this
track also answers *how fast does each engine integrate?* — so it **drives both
engines directly** instead of through the heavyweight `bionetgen.run()` subtree:

```
export BNGPATH=/path/to/BioNetGen-2.9.3      # BNG2.pl + bin/run_network
python bng_ode_run.py --workers 4            # all 591 ODE jobs -> runs/report_ode.json
python generate_bng_matrix.py                # -> runs/bng_matrix_ode.html
open runs/bng_matrix_ode.html
# subset / debug: --limit N · --include fast/ · --models <id,id> · --out report.json
```

- **One shared network.** BNG2.pl generates the reaction-network `.net` **once**
  per model (actions stripped, a bare `generate_network` appended) — the shared
  per-model build *prefix*, timed and attributed to neither integrator. The same
  byte-identical `.net` then feeds both engines, so the comparison is
  apples-to-apples on identical input.
- **BNGsim** integrates it in-process (`Model.from_net` → `Simulator(method="ode")
  .run`), exposing its build phases (net load · analytical-Jacobian derivation ·
  RHS codegen) and a **cold → warm** integration split (warm = the marginal cost
  with the built model reused — the fitting-loop cost).
- **run_network** (the legacy CVODE binary, `$BNGPATH/bin/run_network`)
  integrates the SAME `.net` as a direct subprocess. Each call is a fresh process
  (no warm-solver reuse), so it is measured **per-call**, plus run_network's own
  self-reported `Initialization`/`Propagation` CPU split.
- **Judgment** is the shared `_core.differ.deterministic_verdict` over the network
  species (the `.cdat` columns, identical network order). Outcome attribution is
  per-engine (the same taxonomy as the SBML suites): both ran within tol → `PASS`;
  exceeds → `DIFF`; bngsim raised, run_network ran → `EXCEPTION` (actionable bngsim
  bug); bngsim ran, run_network raised → `REFERENCE_FAILED`; netgen failed / both
  raised → `BAD_TEST`; no resolvable ODE horizon → `SKIP`; wall-cap → `TIMEOUT`.
- **Overrides:** the engine-agnostic `tol` override (ill-conditioned IVPs) is
  honored, applied identically to both engines — like `amici_parity`. A
  `rehab`/`action_inject` override is also applied: those models ship without a
  runnable protocol, so the injected fixture (its `generate_network` caps + the
  `simulate` it appends) is the authoritative source of the netgen caps and the
  horizon — without it the model has no resolvable action and is dropped as SKIP.
  The KNOWN_ARTIFACT/PASS_REF_BUG reclassifications stay in the correctness suite
  (`parity_diff.py`); this track adjudicates every model independently.
- **Horizon** (`t_end`/`n_steps`/`atol`/`rtol`) is read from the model's
  representative ODE `simulate` action (the largest-span one; an injected fixture's
  `simulate` when one is present), resolving any parameter-name/expression token
  against the `.net`'s resolved parameter table.
- Outputs (`runs/report_ode.json`, `runs/bng_matrix_ode.html`) are gitignored
  (regenerable).

## Stochastic timing + parity matrix (`bng_stoch_run.py` → `generate_bng_matrix.py`)

The stochastic sibling of the ODE track: one runner, two tracks (`--track ssa|nf`),
the same three-column HTML matrix and color taxonomy, to the stochastic cost model.
Where the ODE track diffs one deterministic trajectory, this runs an **N-replicate
ensemble** per engine and compares the **observables by name** (a network-free model
has no species, so observables-by-name is the uniform axis for both).

```
export BNGPATH=/path/to/BioNetGen-2.9.3      # BNG2.pl + bin/run_network + bin/NFsim
python bng_stoch_run.py --track ssa --workers 4   # ~107 SSA jobs -> runs/report_ssa.json
python bng_stoch_run.py --track nf  --workers 4   # ~227 NF jobs  -> runs/report_nf.json
python generate_bng_matrix.py runs/report_ssa.json   # -> runs/bng_matrix_ssa.html
python generate_bng_matrix.py runs/report_nf.json    # -> runs/bng_matrix_nf.html
# subset / debug: --limit N · --tier slow · --models <id,id> · --n-rep 10 · --out report.json
```

- **One shared artifact.** BNG2.pl produces it **once** per model (the shared build
  prefix, attributed to neither engine): for **SSA** the reaction-network `.net`
  (`generate_network`); for **NF** the BNG-XML (`writeXML` — a network-free model
  has no `.net`). A model that ships without a runnable protocol uses its
  `rehab`/`action_inject` fixture for the horizon (and, for SSA, the injected
  `generate_network` caps) — see the ODE section's **Overrides** note.
- **BNGsim** loads the artifact once and reseeds an in-process ensemble — SSA:
  `Model.from_net` → clone + `Simulator(method="ssa")` per replicate; NF: one
  `NfsimSession` → `initialize(seed)` + `simulate` per replicate. Timed as a
  per-model load (no Jacobian, no codegen on the stochastic path) + a per-replicate
  ensemble (replicate median, **cold → warm**, ensemble total, per-replicate setup).
- **The legacy binary** runs the SAME artifact as a fresh process per seed (per-call
  cost, no warm reuse): SSA → `run_network -p ssa` with its own init/propagation CPU
  split; NF → `NFsim` with its own total CPU + reactions/sec (and `-cb` auto-added
  when the model has a `Species` observable, which bngsim handles natively).
- **Judgment** is the shared `_core.differ.ensemble_verdict` over the `.gdat`
  observables (aligned by name): each ensemble is reduced to per-cell (mean, sem)
  and compared with a per-cell K=3σ frac-pass test; **PASS** iff ≥ 99% of cells
  pass. Outcome attribution is per-engine: agree → `PASS`; disagree → `DIFF`; bngsim
  raised, legacy ran → `EXCEPTION`; bngsim ran, legacy raised → `REFERENCE_FAILED`;
  a clean `validate_for_ssa` refusal → `UNSUPPORTED`; build failed / both raised →
  `BAD_TEST`; no resolvable horizon → `SKIP`; wall-cap → `TIMEOUT`.
- **NF `block_same_complex_binding`.** bngsim runs at its **correct** default
  (`bscb=True`); the legacy NFsim v1.14.3 has **no `-bscb`** flag, so ring-forming /
  multivalent models legitimately diverge — a known **reference-side** divergence
  (the correctness suite reclassifies these `PASS_REF_BUG`), surfaced here as `DIFF`.
  `--nf-bscb off` matches the legacy binary for an apples-to-apples timing run.
- **Replicate count + seed-escalation.** Defaults to the corpus' `n_rep=10`; a
  borderline DIFF on a noisy model usually clears at higher N (small-N Monte-Carlo
  noise inflates the z-gate). The runner escalates a DIFF up `ESCALATION_RUNGS`
  (10 → 30 → 100, pooling the already-run seeds) and reads the **direction of
  change** rather than a fixed `frac_pass` floor (GH #185): if agreement improves
  with more replicates it is sampling noise (keep climbing to convergence → `PASS`);
  if it stalls or worsens it is a real divergence (stop early). A fixed floor cannot
  work because noise and real divergences overlap in `frac_pass` space (the old
  `0.95` floor left converging-noise models like `m02_logistic` and
  `S1987_B1_lotka_volterra` as false DIFFs). KNOWN_ARTIFACT / PASS_REF_BUG
  reclassification is applied here too (`annotate_ssa_known` / `annotate_ref_bscb`),
  tagging the row non-scoring without changing the honest verdict.
- Outputs (`runs/report_{ssa,nf}.json`, `runs/bng_matrix_{ssa,nf}.html`) are
  gitignored (regenerable). The full SSA/NF ensembles are slow — run them supervised.

## Emitting golden references

`parity_golden.py` emits the `_core` **golden** references — pure bngsim output
(no differ, no subprocess side) that consumers regenerate through their own
bridges to verify they reproduce bngsim. See `golden/README.md` for the full
contract; in brief:

```
export BNGPATH=/path/to/BioNetGen-2.9.3      # bngsim needs BNG2.pl for BNGL netgen
python parity_golden.py --include fast/ --workers 4    # fresh seed=1 sweep -> golden/
python parity_golden.py --no-run-sweep --sweep-out runs/core/bngsim  # reuse a report sweep
```

- `.gdat` canonical (+`.scan` when present); `.cdat` fallback only with no
  `.gdat`/`.scan`. Multi-file jobs: checksum spans all files, fingerprint nests
  one flat per-file entry under `artifacts`.
- Stochastic = a single fixed **`seed=1`** trajectory (byte-identity within a
  pinned version/platform/seed cell), NOT an ensemble.
- Full trajectories only for the `TRAJECTORY_ALLOWLIST` subset; everyone else
  gets checksum + fingerprint. `golden/` is committed; `runs/` is gitignored.
- BNG2.pl is still required: bngsim has no BNGL parser, so it shells out for
  network generation before simulating in-process.

## Status / next steps

- ✅ corpus vendored, `jobs.json` conforms to `_core`, overrides lifted.
- ✅ **runners wired to `jobs.json` + `_core`**: `bng_ode_run.py` /
  `bng_stoch_run.py` run bngsim vs legacy BNG, compare via `_core.differ`, and emit
  a `_core` report (versions stamped) that `generate_bng_matrix.py` renders.
- ✅ **golden references** (`parity_golden.py` → `_core.write_golden`): per-job
  checksum (byte-identity, verified reproducible across independent runs for ODE
  *and* SSA) + per-artifact fingerprint, full trajectory for a hand-picked
  subset. Committed set covers **all 895 manifest jobs** (ODE + stochastic
  `seed=1`), regenerated under a confirmed in-process bngsim engine (the merged
  PyBioNetGen BNGsim bridge, RuleWorld/PyBioNetGen#102) at **bngsim 0.9.49** — see
  `golden/README.md` for the regeneration provenance (`_meta.bngsim`/`bngpath`).
- ⏳ migrate override keys basename → relpath/sha256 to kill the 6 fan-outs.
