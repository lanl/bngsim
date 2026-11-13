# Test corpora

RuleMonkey's correctness story rests on four independent test suites
totalling 203 BNGL models. Every check compares the RM trajectory
against a gold-standard reference:

- an NFsim ensemble, BNG2 ODE or a hand-rolled Gillespie SSA for the first
  three suites;
- BioNetGen's own reaction network, integrated by its ODE or SSA solver, for
  `bng_oracle`.

A reference with a spread is scored by a z-score threshold. A
`feature_coverage` model whose reference is a deterministic BNG2 ODE is
scored by relative error instead.

| Suite | Models | Reference | Driver | Wall time |
|---|---:|---|---|---|
| `feature_coverage` | 90 | NFsim 20-rep + BNG2 ODE | `harness/benchmark_feature_coverage.py` | ~3 min |
| `corpus` | 71 | NFsim 100-rep ensemble (with 2 SSA exceptions) | `harness/benchmark_full.py` | smoke 2 min / guard 15 min / full 3 h |
| `nfsim_basicmodels` | 29 | NFsim 100-rep ensemble | `harness/basicmodels.py` | ~5 min |
| `bng_oracle` | 13 | BNG2 network: ODE, or an SSA ensemble | `tests/bng_oracle/check.py` (ctest `bng_oracle`) | ~40 s |

Last full sweep of the three NFsim suites (2026-08-06): **182 / 182
PASS** at the canonical verdict thresholds. CI runs `feature_coverage`
on every pull request into `main` and every push to it. `bng_oracle`
runs as part of `ctest` in the release builds on Ubuntu, macOS and
Windows and in the ASan builds; it is registered only when CMake finds
`python3`.

## `feature_coverage` — the BNGL feature matrix

90 small, fast models (`tests/models/feature_coverage/*.bngl`),
designed to exercise every BNGL feature and every feature
interaction RM has historically been wrong about.  Each model is
≤ 600 molecules and runs in seconds.  Reference data is vendored
in-tree at `tests/models/feature_coverage/reference/`.

Models break down by family:

| Prefix | Count | Purpose |
|---|---:|---|
| `ft_*` | 53 | One feature per model — the breadth coverage |
| `edg_*` | 20 | "Edge cases designed to break RM" — stress probes added in 3.1.0 |
| `combo_*` | 8 | Feature combinations that have caused past regressions |
| `ss_*` | 5 | Steady-state / longer-time-scale variants |
| `nf_*` | 4 | Network-free-only constructs (NFsim ref only) |

Each model carries its own NFsim reference (or BNG2 ODE reference
where NFsim refuses the model — the `--full` mode of the harness
cross-checks both).  Per-model rationale lives in `EDG_RATIONALE.md`
inside the model directory.

```bash
python3 harness/benchmark_feature_coverage.py            # all 90, ~3 min
python3 harness/benchmark_feature_coverage.py --tier base   # 53 base ft_* models only
python3 harness/benchmark_feature_coverage.py ft_bond_wildcards combo_strict_product_plus
```

This is the suite to run when iterating on a fix.  CI runs it on
every pull request into `main` and every push to it.

## `corpus` — real-world rule-based models

71 models from the literature and from research lab benchmarks
(`tests/models/corpus/*.bngl`) — TLBR, BLBR, EGFR, T-cell receptor
signalling, Posner-Goldstein cooperativity, etc.  These are the
"would my real model run on RM?" canary; sizes range from
~100 molecules to ~10⁵.

Reference is a 100-replicate NFsim ensemble (`tests/reference/nfsim/`),
with two per-model exceptions:

| Model | Reference | Why |
|---|---|---|
| `toy_jim` | hand-rolled Gillespie SSA | NFsim over-counts `K(Y~P)` by ~12% on the disjoint transphosphorylation pattern. SSA confirmed RM is correct (`harness/ssa/toy_jim_ssa.py`). |
| `rm_tlbr_rings` | hand-rolled Gillespie SSA | NFsim under-counts on disjoint + symmetric ring-closure patterns. SSA confirmed RM is correct (`harness/ssa/rm_tlbr_rings_ssa.py`). |

See [`tests/reference/nfsim/PROVENANCE.md`](https://github.com/richardposner/RuleMonkey/blob/main/tests/reference/nfsim/PROVENANCE.md)
for the full provenance.

The driver supports three tiers, chosen by wall budget:

| Tier | Models | Reps | Wall | When to run |
|---|---:|---:|---|---|
| smoke | 8 | 2 | ~2 min | Tight-loop iteration on a fix; catastrophic-regression catch only |
| guard | 29 | 3 | ~15 min | Pre-commit gate; must stay green |
| full | 71 | 10 | ~3 h | Pre-release; declares a fix done |

```bash
python3 harness/benchmark_full.py --tier smoke             # 2 min
python3 harness/benchmark_full.py --tier guard             # 15 min
python3 harness/benchmark_full.py --tier full              # 3 h, 10 reps
python3 harness/benchmark_full.py --tier full --adaptive   # inflates reps for noisy models
python3 harness/benchmark_full.py tcr ensemble             # specific models
```

Tier membership is defined inline in the `SMOKE_MODELS` /
`GUARD_MODELS` Python lists at the top of `benchmark_full.py`.

The verdict metric is a per-model time-integrated z-score
(`tz_max`) compared against an adaptive threshold derived from a
self-split noise floor (`tests/reference/nfsim/noise_floor.tsv`).
When the calibration file is absent the harness falls back to a
flat `T = 5` threshold with a warning.

## `nfsim_basicmodels` — the inherited NFsim regression suite

29 models (`tests/models/nfsim_basicmodels/*.bngl`) derived from
NFsim's own `validate/basicModels/` regression suite — small,
purpose-built tests that NFsim used to verify specific behaviours of
its rule engine. The suite was inherited when RuleMonkey was forked
from NFsim and is preserved here as one of four test corpora.

The 29 models cover: simple binding / unbinding / phosphorylation,
multiple identical sites and dimerisation, ring closure, explicit
and implicit cross-phosphorylation, species deletion / synthesis,
mRNA monomer connectivity, function and parameter handling, and
several intramolecular and edge-case feature checks.  Per-model
names, simulation times, and flags are in
`tests/reference/basicmodels/sim_params.tsv`.

Seven upstream NFsim tests are deliberately excluded (r27 / r28 /
r31 / r33 / r34 / r35 / r36) — see
[`tests/reference/basicmodels/PROVENANCE.md`](https://github.com/richardposner/RuleMonkey/blob/main/tests/reference/basicmodels/PROVENANCE.md)
for the per-test rationale.

```bash
python3 harness/basicmodels.py                # all 29 models
python3 harness/basicmodels.py --batch 1..20  # contiguous slice
python3 harness/basicmodels.py r07 r25        # specific ids
python3 harness/basicmodels.py --reps 10      # more RM reps
```

`basicmodels.py` is a thin wrapper around `benchmark_full.py` that
points the same verdict logic at the basicmodels reference
directory.

## `bng_oracle` — BioNetGen's own answer

13 models (`tests/bng_oracle/models/*.bngl`) whose expected behaviour comes
from BNG2.pl's `generate_network()` expansion rather than from an NFsim
ensemble. A model belongs here when NFsim is wrong about it. An NFsim
reference would then be permanently red for the wrong reason, and would pull
RuleMonkey back onto the defect a fix just removed. A model that merely agrees
with NFsim belongs in the other suites.

Each model records one of two reference kinds:

- **`ode`**: the network integrated by BioNetGen's ODE solver. This is the
  default, and valid when populations are large enough for the ODE to be the
  mean of the SSA.
- **`ssa`**: a committed mean and SEM from an ensemble of BioNetGen network
  SSA runs. Used for models that run at a handful of complexes, where the ODE
  answers a different question, or where fluctuations feed back into the rate.

The models pin:

- per-complex counting of context patterns (#33, #42, #52), including the
  propensity following that count in the same event when a bond forms or
  breaks out of the event's own reach;
- whole-species deletes, and partial deletes without `DeleteMolecules`;
- the `+` and `.` between product patterns (#102, #104), and rules whose
  products can never form (#105);
- the Arrhenius rate law as BNG2's network builds it (#107, #109).

The per-model table, with what each column pins, is in the suite's
[README](https://github.com/richardposner/RuleMonkey/blob/main/tests/bng_oracle/README.md).

```bash
ctest --preset release -R bng_oracle                        # as CI runs it
python3 tests/bng_oracle/check.py --driver build/release/rm_driver
BNG2=~/path/to/BNG2.pl python3 tests/bng_oracle/generate_references.py   # regenerate
```

Scoring is a z on the difference of two means: RuleMonkey's ensemble SEM is
combined with the reference's own in quadrature, so an SSA reference is not
treated as exact. Each model failed by more than ten standard errors
against the engine before its fix.

## C++ smoke tests

A ctest suite runs as part of every build.  The ones covering the
model-configuration surface:

| Test | What it checks |
|---|---|
| `smoke_test` | Engine loads `A_plus_A.xml` and steps a short simulation without crashing |
| `set_param_test` | Parameter overrides reach the engine for Ele / MM / initial-conc / eBNGL-energy paths; derived-parameter cascades work in declaration and reverse-dependency order; BNG2 `expr=` derivations re-derive seed amounts; clearing an override un-bakes a prior run; unknown names throw; mid-session add_molecules conserves mass |
| `param_expr_gate_test` | Corpus-wide: a no-op `set_param` switches the cascade onto `<Parameter expr=>`, and no model's parameters may drift off their loaded `<Parameter value=>` when it does (issue #23) |
| `initial_amount_test` | Seed-species introspection and pin/release; pins outrank parameter derivations, reach the engine, cover literal `concentration=` and `Fixed="1"` clamp targets, and un-bake on clear |
| `save_load_test` | A save/load continuation matches the same session continued in process, sample for sample; XML mismatch is refused via the schema fingerprint; pre-V2 state files are refused |

```bash
ctest --preset release       # whole suite, ~80 s; bng_oracle is half of it
ctest --preset asan          # same tests under AddressSanitizer + UBSan
```

Both presets are wired into CI (`.github/workflows/ci.yml`).

## When to run what

- **Editing engine code or fixing a bug**: `feature_coverage`
  (always) plus `--tier smoke` for fast catastrophic-regression
  detection.
- **Before pushing a fix**: `--tier guard` (15 min) is the
  pre-commit gate.
- **Before declaring a release**: `--tier full` (3 h) plus
  `feature_coverage` plus `nfsim_basicmodels`.  This is the
  182/182 verification that backs the release notes.
- **Adding a new feature**: write a `feature_coverage/ft_*` model
  for it and a `combo_*` or `edg_*` if it interacts with an
  existing feature in a non-obvious way.
- **Fixing behaviour where NFsim is wrong**: add a `bng_oracle`
  model, with BioNetGen's network as the reference.

## Reference data layout

```
tests/
  models/
    feature_coverage/{ft,edg,combo,ss,nf}_*.bngl    # 90 BNGL sources
    corpus/*.bngl                                    # 71 BNGL sources
    nfsim_basicmodels/v*.bngl                        # 29 BNGL sources
  reference/
    nfsim/
      xml/*.xml                                      # XML for all 71 corpus models
      ensemble/{model}.{mean,std,tint}.tsv           # NFsim 100-rep aggregates
      noise_floor.tsv                                # adaptive threshold calibration
      sim_params.tsv                                 # per-model t_end, n_steps, flags
      summary.tsv                                    # per-model NFsim wall times
      PROVENANCE.md                                  # generation history + SSA exceptions
      MANIFEST.tsv                                   # SHA-256 of each vendored file above
      replicates/                                    # per-rep trajectories (gitignored, 1.9 GB)
    basicmodels/
      xml/r*.xml
      ensemble/r*.{mean,std,tint}.tsv
      sim_params.tsv
      MANIFEST.tsv
      replicates/                                    # gitignored
      PROVENANCE.md
    rm/
      summary.tsv                                    # RM wall times for the timing comparison
  bng_oracle/
    models/*.bngl                                    # 13 BNGL sources
    xml/*.xml                                        # BNG2-written XML
    reference/{model}.{mean,sem}.tsv                 # BNG2 network ODE or SSA reference
    check.py                                         # scorer (ctest bng_oracle)
    generate_references.py                           # regenerates xml/ and reference/
```

The `.tsv` aggregates are the canonical artifacts; the `replicates/`
directories are disposable scratch held locally only when actively
regenerating references.

Each NFsim reference root (`nfsim`, `basicmodels`, and
`feature_coverage/reference`) carries a `MANIFEST.tsv` of SHA-256
hashes, and the harness
verifies the tree against it before running anything — a hand-edited
reference would otherwise shift a parity verdict with nothing to say it
had.  The manifest covers the vendored files only: `replicates/` is
absent on a clean clone and several gigabytes on a machine that just
regenerated an ensemble, so hashing it would fail the gate in both
directions over files no verdict reads.  The trees are
marked `-text` in `.gitattributes` so that a checkout cannot rewrite the
bytes the hashes were taken over — without that, Git for Windows' CRLF
conversion faults every text file in every tree.  `ctest`'s
`ref_manifest_test` checks each committed manifest against its tree, and
the eol attribute alongside it, which is where a manifest that reached
outside that coverage now surfaces (issue #63).
To rewrite one after an intentional reference change, re-run its harness
with the manifest-writing flags — `--generate-refs` / `--force-refs` for
`feature_coverage`, `--no-verify-manifest --write-manifest` for
`benchmark_full` / `basicmodels`.
