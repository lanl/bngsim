# Test corpora

RuleMonkey's correctness story rests on three independent test suites
totalling 177 BNGL models. Every parity check compares the RM
trajectory against a gold-standard reference (NFsim ensemble, BNG2
ODE, or hand-rolled Gillespie SSA) using a per-model time-integrated
z-score threshold.

| Suite | Models | Reference | Driver | Wall time |
|---|---:|---|---|---|
| `feature_coverage` | 77 | NFsim 20-rep + BNG2 ODE | `harness/benchmark_feature_coverage.py` | ~3 min |
| `corpus` | 71 | NFsim 100-rep ensemble (with 2 SSA exceptions) | `harness/benchmark_full.py` | smoke 2 min / guard 15 min / full 3 h |
| `nfsim_basicmodels` | 29 | NFsim 100-rep ensemble | `harness/basicmodels.py` | ~5 min |

Current health (2026-05-02): **177 / 177 PASS** at the canonical
verdict thresholds. The CI job runs `feature_coverage` on every push
to `main`.

## `feature_coverage` — the BNGL feature matrix

77 small, fast models (`tests/models/feature_coverage/*.bngl`),
designed to exercise every BNGL feature and every feature
interaction RM has historically been wrong about.  Each model is
≤ 600 molecules and runs in seconds.  Reference data is vendored
in-tree at `tests/models/feature_coverage/reference/`.

Models break down by family:

| Prefix | Count | Purpose |
|---|---:|---|
| `ft_*` | 42 | One feature per model — the breadth coverage |
| `edg_*` | 20 | "Edge cases designed to break RM" — stress probes added in 3.1.0 |
| `combo_*` | 8 | Feature combinations that have caused past regressions |
| `ss_*` | 4 | Steady-state / longer-time-scale variants |
| `nf_*` | 3 | Network-free-only constructs (NFsim ref only) |

Each model carries its own NFsim reference (or BNG2 ODE reference
where NFsim refuses the model — the `--full` mode of the harness
cross-checks both).  Per-model rationale lives in `EDG_RATIONALE.md`
inside the model directory.

```bash
python3 harness/benchmark_feature_coverage.py            # all 77, ~3 min
python3 harness/benchmark_feature_coverage.py --tier base   # 42 base ft_* models only
python3 harness/benchmark_feature_coverage.py ft_bond_wildcards combo_strict_product_plus
```

This is the suite to run when iterating on a fix.  CI runs it on
every push.

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

See [`tests/reference/nfsim/PROVENANCE.md`](../tests/reference/nfsim/PROVENANCE.md)
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
from NFsim and is preserved here as one of three test corpora.

The 29 models cover: simple binding / unbinding / phosphorylation,
multiple identical sites and dimerisation, ring closure, explicit
and implicit cross-phosphorylation, species deletion / synthesis,
mRNA monomer connectivity, function and parameter handling, and
several intramolecular and edge-case feature checks.  Per-model
names, simulation times, and flags are in
`tests/reference/basicmodels/sim_params.tsv`.

Seven upstream NFsim tests are deliberately excluded (r27 / r28 /
r31 / r33 / r34 / r35 / r36) — see
[`tests/reference/basicmodels/PROVENANCE.md`](../tests/reference/basicmodels/PROVENANCE.md)
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

## C++ smoke tests

Three ctest tests run as part of every build:

| Test | What it checks |
|---|---|
| `smoke_test` | Engine loads `A_plus_A.xml` and steps a short simulation without crashing |
| `set_param_test` | Parameter overrides reach the engine for Ele / MM / initial-conc paths; derived-parameter cascades work; unknown names throw; mid-session add_molecules conserves mass |
| `save_load_test` | save/load round-trip preserves trajectory; XML mismatch is refused via the schema fingerprint; pre-V2 state files are refused |

```bash
ctest --preset release       # all three, ~2s
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
  177/177 verification that backs the release notes.
- **Adding a new feature**: write a `feature_coverage/ft_*` model
  for it and a `combo_*` or `edg_*` if it interacts with an
  existing feature in a non-obvious way.

## Reference data layout

```
tests/
  models/
    feature_coverage/{ft,edg,combo,ss,nf}_*.bngl    # 77 BNGL sources
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
      replicates/                                    # per-rep trajectories (gitignored, 1.9 GB)
    basicmodels/
      xml/r*.xml
      ensemble/r*.{mean,std,tint}.tsv
      sim_params.tsv
      replicates/                                    # gitignored
      PROVENANCE.md
    rm/
      summary.tsv                                    # RM wall times for the timing comparison
```

The `.tsv` aggregates are the canonical artifacts; the `replicates/`
directories are disposable scratch held locally only when actively
regenerating references.
