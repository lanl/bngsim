# BNGsim Benchmark Harness

Paper-ready benchmark harnesses for BNGsim performance validation and
cross-engine comparison. Each harness is a self-contained script that
non-developers can re-run with a single command.

**Canonical reference**: `bngsim/dev/benchmarks/benchmarking_playbook.md` (full procedure,
platform notes, data-flow documentation). This README is the quick-start.

## First-Time Setup (Fresh Clone)

BioModels benchmarks (Pool B) use a manifest-driven fetch pipeline. On a fresh
clone, you must materialize the model corpus once before running BioModels
benchmarks:

```bash
# Option 1: Pre-materialize the full pool (~1013 models, 10-30 min)
python bngsim/benchmarks/convert_sbml_to_ant.py

# Option 2: Use --ensure-pool in harness scripts (on-demand, subset-bounded)
python bngsim/harness/comparison/bench_biomodels_sbml.py --quick 20 --ensure-pool

# Option 3: Set env var for automatic materialization in all harness scripts
export BENCH_AUTO_ENSURE_POOL=1
python bngsim/harness/comparison/bench_biomodels_sbml.py --quick 20
```

Models are fetched from EBI BioModels (requires network access) and converted
to Antimony. Generated files are gitignored and stay local. See
`bngsim/benchmarks/biomodels_ant/README.txt` for details.

**Note:** The fetch/convert pipeline requires Python 3.10+ (uses union type
annotations). Harness scripts themselves work on Python 3.9 if `.ant` files
already exist.

## Quick Start (recommended)

```bash
# From repo root (direnv activates .venv automatically)
cd /path/to/repo   # your checkout root (contains bngsim/)

# 1. Validate manifest sync (should be 0 failures)
python bngsim/harness/validate_jobs.py --quick

# 2. Run all enabled jobs (env check + metadata + execution + archival)
python bngsim/harness/run_jobs.py

# 3. Or run a single job / phase
python bngsim/harness/run_jobs.py --job bench_ssa
python bngsim/harness/run_jobs.py --phase validation
python bngsim/harness/run_jobs.py --phase comparison
```

## Manual Script-by-Script Execution

### Validation (correctness — run first)

```bash
python bngsim/harness/validation/validate_ode.py       # 27 models vs run_network
python bngsim/harness/validation/validate_ssa.py       # SSA determinism + moments
python bngsim/harness/validation/validate_nf.py        # 5 NFsim models
python bngsim/harness/validation/validate_sbml.py --pool a --quick 10
```

### Comparison (performance — generates paper data)

```bash
# ── Tables S1–S4: BNGsim vs BNG engines ──────────────────────
python bngsim/harness/comparison/bench_ode_4engine.py           # Table S1
python bngsim/harness/comparison/bench_ssa_vs_runnetwork.py     # Table S2
python bngsim/harness/comparison/bench_psa_vs_runnetwork.py     # Table S3
python bngsim/harness/comparison/bench_nf_vs_nfsim.py           # Table S4

# ── Table S5: Antimony 4-comparison ──────────────────────────
python bngsim/harness/calibrate_horizons.py --pool ab           # prerequisite
python bngsim/harness/comparison/bench_ant_3engine.py           # Table S5

# ── Table S6: Pythonic workflows (8 configs) ──────────────────
python bngsim/harness/comparison/bench_pythonic_workflows.py    # Table S6

# ── Table S7: KINSOL dose-response ───────────────────────────
python bngsim/harness/comparison/bench_kinsol_dose_response.py  # Table S7

# ── Table S8: SBML Test Suite 3-engine pass rates ────────────
python bngsim/harness/sbml_test_suite/run_sbml_test_suite.py --engines all --candidates

# ── Table S9 / S10: Forward sensitivity ─────────────────────
python bngsim/harness/comparison/bench_forward_sensitivity.py   # Table S10 (legacy: S9)

# ── Large-scale BioModels (requires libroadrunner, ~1–2 hours) ─
python bngsim/harness/comparison/bench_biomodels_sbml.py        # Figure 2
python bngsim/harness/comparison/bench_biomodels_warmup.py      # Figure 4
python bngsim/harness/comparison/bench_amortization.py          # Figure 3

# ── SBML event models (requires RR + AMICI, ~1–2 hours) ──────
python bngsim/harness/comparison/bench_sbml_events.py           # Figure S2

# ── Generate paper tables ────────────────────────────────────
python bngsim/harness/comparison/generate_paper_tables.py
```

## Job Manifest

All 22 benchmark jobs are defined in `bngsim/harness/jobs.yaml`. This is the
**single source of truth** for scripts, model suites, engines, paper artifacts,
and dependencies.

```bash
# List all jobs
python bngsim/harness/run_jobs.py --list

# Validate manifest sync
python bngsim/harness/validate_jobs.py --quick
```

## Directory Structure

```
harness/
├── README.md                              # This file (quick-start)
├── jobs.yaml                              # Job manifest (22 jobs)
├── run_jobs.py                            # Orchestrated runner
├── validate_jobs.py                       # Manifest sync checker
├── common.py                              # Shared utilities (timing, runners, xval)
├── calibrate_horizons.py                  # Adaptive t_end for Antimony pools
├── convert_bngl_to_sbml.py               # BNGL → SBML via BNG2.pl
├── .gitignore                             # Ignores results/
├── validation/                            # Correctness gates
│   ├── validate_ode.py                    # BNGsim vs run_network (27 models)
│   ├── validate_ssa.py                    # SSA determinism + moments (10)
│   ├── validate_nf.py                     # BNGsim vs BNG2.pl NFsim (5)
│   └── validate_sbml.py                   # BNGsim vs libRoadRunner (~960)
├── comparison/                            # Performance benchmarks
│   ├── bench_ode_4engine.py               # Table S1: 4-engine ODE (27 models)
│   ├── bench_ssa_vs_runnetwork.py         # Table S2: SSA (10 models)
│   ├── bench_psa_vs_runnetwork.py         # Table S3: PSA (3×4 Nc)
│   ├── bench_nf_vs_nfsim.py              # Table S4: NFsim (5 models)
│   ├── bench_ant_3engine.py               # Table S5: Antimony 4-comparison (117)
│   ├── bench_pure_python.py               # Legacy Table 6 (scipy/gillespy2)
│   ├── bench_pythonic_workflows.py        # Table S6: 8-config (4 engines × 2 inputs)
│   ├── bench_kinsol_dose_response.py      # Table S7: KINSOL vs CVODE vs run_net
│   ├── bench_forward_sensitivity.py       # Table S10: serial/sharded vs AMICI
│   ├── bench_biomodels_sbml.py            # Figure 2: BioModels scatter (506)
│   ├── bench_biomodels_warmup.py          # Figure 4: 3-engine warmup (472)
│   ├── bench_amortization.py              # Figure 3: codegen amortization
│   ├── bench_sbml_events.py               # Figure S2: SBML events 4-panel (214)
│   ├── bench_1013_4engine.py              # Figure S1: BioModels 4-panel (~1013 default)
│   └── generate_paper_tables.py           # JSON → Markdown + LaTeX
├── sbml_test_suite/                       # SBML Test Suite harness
│   ├── SUITE_PIN.json                     # canonical version pin (commit 473e119d); all refs point here
│   ├── fetch_semantic_suite.py            # clone the pinned semantic corpus into a checkout
│   ├── run_sbml_test_suite.py             # Table S8: 3-engine pass rates + candidates
│   ├── sbml_test_suite_results.json       # Cached results
│   └── dsmts/                             # DSMTS SSA gate (run_dsmts.py)
│       ├── dsmts_index.json               # 39-case metadata + settings
│       ├── dsmts_manifest.json            # per-file sha256 provenance (pinned @473e119d, LGPL-2.1-or-later)
│       ├── generate_dsmts_manifest.py     # (re)build the manifest; cross-checks upstream
│       ├── build_dsmts_index.py           # (re)build the index from an upstream suite
│       ├── vendor_dsmts_cases.py          # (re)vendor cases/ from an upstream suite
│       └── cases/                         # Vendored 39 cases (~140 KB) — hermetic, no external checkout
└── results/                               # Output (git-ignored)
    └── *.json                             # Machine-readable results
```

## Paper Artifact Mapping

| Paper artifact | Script | Status |
|---|---|---|
| Table S1 (ODE 4-engine) | `bench_ode_4engine.py` | ✅ |
| Table S2 (SSA) | `bench_ssa_vs_runnetwork.py` | ✅ |
| Table S3 (PSA) | `bench_psa_vs_runnetwork.py` | ✅ |
| Table S4 (NFsim) | `bench_nf_vs_nfsim.py` | ✅ |
| Table S5 (Antimony 4-comp) | `bench_ant_3engine.py` | ✅ |
| Table S6 (Pythonic 8-config) | `bench_pythonic_workflows.py` | ✅ NEW |
| Table S7 (KINSOL dose-resp) | `bench_kinsol_dose_response.py` | ✅ NEW |
| Table S8 (SBML Test Suite) | `run_sbml_test_suite.py` | ✅ EXTENDED |
| Table S10 (Forward sensitivity) | `bench_forward_sensitivity.py` | ✅ NEW |
| Figure 2 (BioModels scatter) | `bench_biomodels_sbml.py` | ✅ |
| Figure 3 (Amortization) | `bench_amortization.py` | ✅ |
| Figure 4 (Warmup) | `bench_biomodels_warmup.py` | ✅ |
| Figure S1 (BioModels 4-panel) | `bench_1013_4engine.py` | ✅ |
| Figure S2 (SBML events) | `bench_sbml_events.py` | ✅ NEW |

## Model Pools

| Pool | Location | Count | Description |
|---|---|---|---|
| Pool A | `benchmarks/models/antimony/ssys/` | 117 | ssys Antimony test suite |
| Pool B | `benchmarks/biomodels_ant/` (+ `review_extra/`) | manifest-driven | BioModels Antimony |
| Pool C | `benchmarks/suite_ode.json` | 28 | BNG ODE models (2–3744 sp) |
| SSA | `benchmarks/suite_ssa.json` | 10 | Stochastic models |
| PSA | `benchmarks/suite_psa.json` | 3×4 | PSA Nc sweep |
| NF | `benchmarks/nf/` | 5 | Network-free BNGL models |
| Events | `benchmarks/sbml_events/` | 214 | BioModels SBML with events |
| Candidates | `~/Code/ssys/.../sbml_candidates/` | 978 | ssys-filtered BioModels |

## Protocol

All comparison harnesses follow the same timing protocol:
- **2 warmup + 5 timed runs**, median wall time reported
- **BNGsim timing**: excludes model loading (amortized in fitting loops)
- **run_network timing**: includes full subprocess overhead (real PyBNF cost)
- **Seeds**: fixed sequence for reproducibility
- **Override flags**: `--quick N`, `--runs R`, `--warmup W`

## Prerequisites

```bash
# From repo root (NOT from bngsim/)
cd /path/to/repo   # your checkout root (contains bngsim/)

# bngsim must be installed into the root .venv
uv pip install --no-build-isolation -e ./bngsim

# Verify
python -c "import bngsim; print(bngsim.__version__)"

# Cross-engine comparisons (optional)
uv pip install libroadrunner    # Tables S1/S5, Figures 2–4, S1–S2
uv pip install amici             # Tables S1/S5/S8/S9, Figures S1–S2

# External tools (resolve via env vars; never hardcode paths)
# BNG2.pl:     $BNG2_PL      (e.g. <BioNetGen>/BNG2.pl)
# run_network: $RUN_NETWORK  (e.g. <BioNetGen>/bin/run_network)

# SBML Test Suite (Table S8) — pinned in sbml_test_suite/SUITE_PIN.json
# Repo: https://github.com/sbmlteam/sbml-test-suite  @ commit 473e119d (3.4.0-36-g473e119dd)
# Check it out, then: export SBML_TEST_SUITE_DIR=<checkout>
# (or run sbml_test_suite/fetch_semantic_suite.py to clone the pinned commit)
```

## Anti-patterns

1. **Never `cd bngsim` then run `uv`** — creates `bngsim/.venv`
2. **Never run benchmarks from `bngsim/.venv`** — wrong packages
3. **Never skip metadata** — results without provenance are worthless
4. **Never mix `--quick` results with full results** in paper tables

If `bngsim/.venv` exists, delete it: `rm -rf bngsim/.venv`
