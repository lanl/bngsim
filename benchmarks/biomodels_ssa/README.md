# BioModels SSA Benchmark Suite

Automated pipeline for building an SSA (Stochastic Simulation Algorithm) benchmark
suite from the [BioModels](https://www.ebi.ac.uk/biomodels) repository. Fetches SBML
models, converts them to BNG `.net` format, and validates them for use with
[bngsim](../../README.md).

> **Distinct from `benchmarks/suites/biomodels/`.** This directory builds the SSA
> `.net` / Antimony pool consumed by the `bench_1013_4engine` harness. The SBML
> BioModels corpus for the ODE-parity suite (BNGsim vs RoadRunner/AMICI) lives at
> `benchmarks/suites/biomodels/` — different corpus, different benchmark family.

## Quick Start

```bash
# Set BioNetGen path
export BNGPATH=/path/to/BioNetGen   # your local BioNetGen install root

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
./run_pipeline.sh

# Or with full SSA ensemble cross-validation (slow)
./run_pipeline.sh --full
```

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Fetch | `step1_fetch.py` | Download SBML from BioModels API |
| 2. Filter | `step2_filter.py` | Identify SSA-suitable candidates |
| 3. Convert | `step3_convert.py` | SBML → BNGL → .net (atomizer + BNG2.pl) |
| 4. Validate | `step4_validate.py` | Multi-level validation (see below) |
| 5. Report | `step5_report.py` | Generate benchmark catalog and report |

## Validation Levels

The validation step ensures benchmark integrity through five levels:

| Level | Test | Criterion |
|-------|------|-----------|
| 1 | **BNGsim Load** | `.net` file parses without error |
| 2 | **BNGsim ODE** | ODE simulation: no NaN, no Inf, no negative species |
| 3 | **SSA Self-Consistency** | Same seed → **bit-identical** SSA trajectories |
| 4 | **ODE Cross-Validation** | BNGsim ODE ≈ libRoadRunner ODE (validates conversion) |
| 5 | **SSA Ensemble** | BNGsim vs libRoadRunner ensemble means agree (optional, `--full`) |

**Level 3 is critical**: it guarantees that bngsim's SSA is deterministically
reproducible for each benchmark model.

**Level 4 validates model conversion**: if the SBML → BNGL → .net pipeline
altered the dynamics, the ODE solutions will diverge.

**Level 5 validates stochastic dynamics**: both engines should sample from the
same distribution (even though individual trajectories differ due to different RNGs).

## Usage

### Automated Pipeline (Recommended)

```bash
./run_pipeline.sh              # Default: levels 1–4
./run_pipeline.sh --full       # All levels including SSA ensemble
./run_pipeline.sh --from convert   # Resume from conversion
./run_pipeline.sh --only fetch     # Run only fetch step
./run_pipeline.sh --force          # Force re-run all steps
./run_pipeline.sh --clean          # Delete outputs (keep downloads)
./run_pipeline.sh --clean-all      # Delete everything
./run_pipeline.sh --status         # Show pipeline status
```

### Manual Execution

```bash
# Step 1: Fetch SBML models
python step1_fetch.py --target-total 2000 --strategy sequential

# Step 2: Filter candidates
python step2_filter.py

# Step 3: Convert SBML → BNGL → .net
python step3_convert.py

# Step 4: Validate
python step4_validate.py           # Levels 1–4
python step4_validate.py --ode-only  # Levels 1–2 only (fast)
python step4_validate.py --full      # All levels (slow)

# Step 5: Generate report
python step5_report.py
```

## Directory Structure

```
benchmarks/biomodels_ssa/
├── run_pipeline.sh      # Pipeline orchestration
├── step1_fetch.py       # Download SBML
├── step2_filter.py      # Filter candidates
├── step3_convert.py     # SBML → BNGL → .net
├── step4_validate.py    # Multi-level validation
├── step5_report.py      # Generate reports
├── config.py            # Shared configuration
├── utils.py             # Shared utilities
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignore data/results
│
├── data/                # Downloads & intermediates (gitignored)
│   ├── sbml_downloads/  # Raw SBML from BioModels
│   ├── sbml_candidates/ # Filtered SBML files
│   ├── bngl_models/     # BNGL from atomizer
│   └── net_models/      # .net from BNG2.pl
│
└── results/             # Outputs (gitignored)
    ├── candidates.csv       # Filter classification
    ├── conversion_log.csv   # Conversion success/failure
    ├── validation_log.csv   # Validation results
    ├── benchmark_models.csv # Final catalog
    └── REPORT.md            # Auto-generated report
```

## Conversion Toolchain

```
BioModels (SBML)
    │
    ▼
bionetgen atomize -i model.xml -o output/
    │  (PyBioNetGen atomizer: SBML → BNGL)
    ▼
BNGL file (with generate_network action)
    │
    ▼
perl $BNGPATH/BNG2.pl model.bngl
    │  (BioNetGen: BNGL → .net)
    ▼
.net file → bngsim.Model.from_net()
```

## Dependencies

- **Python packages**: `bioservices`, `python-libsbml`, `pandas`, `tqdm`,
  `libroadrunner`, `numpy`
- **BioNetGen**: BNG2.pl (set `BNGPATH` environment variable)
- **PyBioNetGen**: `bionetgen` CLI (provides `bionetgen atomize`)
- **bngsim**: The simulation engine being benchmarked

## Configuration

Edit `config.py` for:
- BioNetGen paths (`BNGPATH`, `BNG2_PL`)
- Timeout settings (atomizer, BNG2.pl, validation)
- Complexity thresholds (max species, reactions)
- Validation tolerances (ODE cross-validation, SSA ensemble)
- Blocking features (events, delays, algebraic rules)

## Notes

- BioModels API has rate limits — fetch includes delays
- Atomizer conversion has high attrition (~50% of models may fail)
- BNG2.pl `generate_network` can hang on large models (timeout protected)
- The pipeline preserves SBML downloads across `--clean` operations
- All data files are gitignored to avoid repo bloat
- The benchmark catalog (`benchmark_models.csv`) is the key output

## Provenance

The fetch and filter scripts are adapted from the
[ssys](https://github.com/lanl/ssys) `biomodels_batch/` pipeline,
originally designed for S-system recasting benchmarks. The conversion,
validation, and reporting scripts are new, purpose-built for SSA
benchmarking with bngsim.
