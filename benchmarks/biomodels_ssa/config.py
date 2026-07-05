"""Configuration for BioModels SSA benchmark suite."""

import os

# ─── BioNetGen paths ──────────────────────────────────────────────────────────

BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
BNG2_PL = os.path.join(BNGPATH, "BNG2.pl")

# Atomizer CLI (PyBioNetGen package provides 'bionetgen atomize' command)
# Uses dedicated Python 3.11 venv because atomizer requires:
#   - Python 3.11 (uses `imp` module, removed in 3.12)
#   - setuptools<70 (for `pkg_resources`)
#   - python-libsbml==5.20.4 (5.21.0 has SWIG API incompatibility)
# Create with:
#   uv venv --python 3.11 .venv311
#   uv pip install --python .venv311/bin/python3.11 bionetgen 'setuptools<70' 'python-libsbml==5.20.4'
ATOMIZER_VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv311")
ATOMIZER_CMD = os.path.join(ATOMIZER_VENV, "bin", "bionetgen")

# ─── Fetch settings ──────────────────────────────────────────────────────────

FETCH_STRATEGIES = ["random", "sequential"]
DEFAULT_STRATEGY = "random"

# No model count limit — fetch all available ODE models
MAX_MODELS = None
START_INDEX = 0

# ─── Complexity thresholds ────────────────────────────────────────────────────

MAX_SPECIES = 100
MAX_REACTIONS = 200
MAX_PARAMETERS = 500

# ─── Timeout settings (seconds) ──────────────────────────────────────────────

FETCH_TIMEOUT = 30  # Per model download
ATOMIZE_TIMEOUT = 120  # SBML → BNGL via atomizer
GENERATE_NET_TIMEOUT = 60  # BNGL → .net via BNG2.pl
VALIDATION_TIMEOUT = 30  # Per validation run (ODE or SSA)

# ─── Parallel processing ─────────────────────────────────────────────────────

N_WORKERS = 4

# ─── Features that block conversion attempts ─────────────────────────────────

BLOCKING_FEATURES = [
    "events",
    "delays",
    "algebraic_rules",
    "sbml_l3_packages",
    "not_substance_units",  # Species not in particle counts
]

# Features that flag as challenging but don't block
WARNING_FEATURES = [
    "piecewise",
    "piecewise_heavy",
    "time_dependent",
    "sin_cos",
    "unsupported_trig",
    "negative_species",
    "non_integer_concentrations",
]

# ─── Validation settings ─────────────────────────────────────────────────────

# ODE cross-validation: max relative error between BNGsim ODE and libRoadRunner ODE
ODE_CROSS_VALIDATION_RTOL = 1e-4

# SSA self-consistency: BNGsim must produce bit-identical results with same seed
# (no tolerance — must be exact)

# SSA ensemble size for statistical cross-validation
SSA_ENSEMBLE_SIZE = 100
SSA_ENSEMBLE_SEED_BASE = 1000  # Seeds will be 1000, 1001, ..., 1099

# SSA statistical cross-validation: ensemble means must agree within this
# many standard errors (typically 2-3 for 95-99% confidence)
SSA_CROSS_VALIDATION_N_SIGMA = 3.0

# Simulation parameters for validation runs
VALIDATION_T_END = 10.0
VALIDATION_N_POINTS = 101

# ─── BioModels API settings ──────────────────────────────────────────────────

BIOMODELS_API_BASE = "https://www.ebi.ac.uk/biomodels"
BIOMODELS_REST_API = f"{BIOMODELS_API_BASE}/model/download"
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.5  # Delay between requests to avoid hammering API

# ─── Paths (relative to benchmarks/biomodels_ssa/) ───────────────────────────────

DATA_DIR = "data"
SBML_DOWNLOADS_DIR = f"{DATA_DIR}/sbml_downloads"
SBML_CANDIDATES_DIR = f"{DATA_DIR}/sbml_candidates"
BNGL_MODELS_DIR = f"{DATA_DIR}/bngl_models"
NET_MODELS_DIR = f"{DATA_DIR}/net_models"
FETCH_HISTORY = f"{DATA_DIR}/fetch_history.json"
MODEL_REGISTRY = f"{DATA_DIR}/model_registry.json"
METADATA_FILE = f"{DATA_DIR}/metadata.json"

RESULTS_DIR = "results"
CANDIDATES_CSV = f"{RESULTS_DIR}/candidates.csv"
CONVERSION_LOG_CSV = f"{RESULTS_DIR}/conversion_log.csv"
VALIDATION_LOG_CSV = f"{RESULTS_DIR}/validation_log.csv"
BENCHMARK_MODELS_CSV = f"{RESULTS_DIR}/benchmark_models.csv"
REPORT_MD = f"{RESULTS_DIR}/REPORT.md"

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_FILE = "benchmark.log"
