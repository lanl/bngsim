#!/bin/bash
#
# BioModels SSA Benchmark Pipeline
# =================================
#
# Fetches models from BioModels, filters for SSA suitability,
# converts SBML → BNGL → .net, and validates for benchmarking.
#
# Usage:
#   ./run_pipeline.sh              # Default pipeline
#   ./run_pipeline.sh --full       # Include SSA ensemble validation
#   ./run_pipeline.sh --from convert   # Start from conversion
#   ./run_pipeline.sh --only fetch     # Run only fetch step
#   ./run_pipeline.sh --force          # Force re-run all steps
#   ./run_pipeline.sh --clean          # Delete all outputs first
#   ./run_pipeline.sh --status         # Show pipeline status
#   ./run_pipeline.sh --help           # Show this help
#
set -euo pipefail

# ===========================================================================
# Configuration
# ===========================================================================

MIN_SBML_FILES=100
MIN_CANDIDATE_FILES=50
MIN_NET_FILES=10

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ===========================================================================
# Paths
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="data"
RESULTS_DIR="results"
SBML_DIR="$DATA_DIR/sbml_downloads"
CANDIDATES_DIR="$DATA_DIR/sbml_candidates"
BNGL_DIR="$DATA_DIR/bngl_models"
NET_DIR="$DATA_DIR/net_models"
CANDIDATES_CSV="$RESULTS_DIR/candidates.csv"
CONVERSION_CSV="$RESULTS_DIR/conversion_log.csv"
VALIDATION_CSV="$RESULTS_DIR/validation_log.csv"
BENCHMARK_CSV="$RESULTS_DIR/benchmark_models.csv"

# ===========================================================================
# Helpers
# ===========================================================================

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[DONE]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[SKIP]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

log_stage() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════${NC}"
    echo ""
}

count_files() {
    local dir="$1"
    local pattern="${2:-*}"
    if [[ -d "$dir" ]]; then
        find "$dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# ===========================================================================
# Stage Completion Checks
# ===========================================================================

is_fetch_complete() {
    local count
    count=$(count_files "$SBML_DIR" "*.xml")
    [[ $count -ge $MIN_SBML_FILES ]]
}

is_filter_complete() {
    [[ -f "$CANDIDATES_CSV" ]] && \
    [[ $(count_files "$CANDIDATES_DIR" "*.xml") -ge $MIN_CANDIDATE_FILES ]]
}

is_convert_complete() {
    [[ -f "$CONVERSION_CSV" ]] && \
    [[ $(count_files "$NET_DIR" "*.net") -ge $MIN_NET_FILES ]]
}

is_validate_complete() {
    [[ -f "$VALIDATION_CSV" ]]
}

is_report_complete() {
    [[ -f "$BENCHMARK_CSV" ]]
}

# ===========================================================================
# Stage Implementations
# ===========================================================================

stage_fetch() {
    log_stage "STAGE 1: Fetch SBML models from BioModels"
    log_info "Fetching ODE models from BioModels API..."
    python step1_fetch.py --target-total 2000 --strategy sequential
    local count
    count=$(count_files "$SBML_DIR" "*.xml")
    log_success "Downloaded $count SBML files"
}

stage_filter() {
    log_stage "STAGE 2: Filter candidates for SSA benchmarking"
    log_info "Applying heuristic filters..."
    python step2_filter.py
    local count
    count=$(count_files "$CANDIDATES_DIR" "*.xml")
    log_success "Filtered to $count candidate models"
}

stage_convert() {
    log_stage "STAGE 3: Convert SBML → BNGL → .net"
    log_info "Running atomizer + BNG2.pl..."
    python step3_convert.py
    local count
    count=$(count_files "$NET_DIR" "*.net")
    log_success "Generated $count .net files"
}

stage_validate() {
    log_stage "STAGE 4: Validate models"
    log_info "Running multi-level validation..."
    local validate_args=""
    if $INCLUDE_FULL; then
        validate_args="--full"
    fi
    python step4_validate.py $validate_args
    log_success "Validation complete"
}

stage_report() {
    log_stage "STAGE 5: Generate report"
    log_info "Building benchmark catalog..."
    python step5_report.py
    log_success "Report generated"
}

# ===========================================================================
# Clean
# ===========================================================================

clean_outputs() {
    log_stage "CLEANING: Removing all output files"

    local dirs=("$BNGL_DIR" "$NET_DIR" "$RESULTS_DIR")
    for dir in "${dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
            log_info "Deleted $dir"
        fi
    done

    log_success "Cleanup complete (SBML downloads preserved)"
}

clean_all() {
    log_stage "CLEANING: Removing ALL files (including downloads)"
    if [[ -d "$DATA_DIR" ]]; then
        rm -rf "$DATA_DIR"
        log_info "Deleted $DATA_DIR"
    fi
    if [[ -d "$RESULTS_DIR" ]]; then
        rm -rf "$RESULTS_DIR"
        log_info "Deleted $RESULTS_DIR"
    fi
    log_success "Full cleanup complete"
}

# ===========================================================================
# Status
# ===========================================================================

show_status() {
    log_stage "Pipeline Status"

    local stages=("fetch" "filter" "convert" "validate" "report")
    for stage in "${stages[@]}"; do
        local check_func="is_${stage}_complete"
        local status
        if $check_func 2>/dev/null; then
            status="${GREEN}✓ COMPLETE${NC}"
        else
            status="${YELLOW}○ PENDING${NC}"
        fi
        printf "  %-15s %b\n" "$stage" "$status"
    done

    echo ""
    echo "File counts:"
    echo "  SBML downloads:   $(count_files "$SBML_DIR" "*.xml")"
    echo "  Candidates:       $(count_files "$CANDIDATES_DIR" "*.xml")"
    echo "  BNGL models:      $(count_files "$BNGL_DIR" "*.bngl")"
    echo "  .net models:      $(count_files "$NET_DIR" "*.net")"
}

# ===========================================================================
# Help
# ===========================================================================

show_help() {
    cat << 'EOF'
BioModels SSA Benchmark Pipeline
=================================

Usage:
  ./run_pipeline.sh [OPTIONS]

Options:
  --full        Include SSA ensemble cross-validation (slow)
  --from STAGE  Start from a specific stage
                Stages: fetch, filter, convert, validate, report
  --only STAGE  Run only a specific stage
  --force       Force re-run stages even if complete
  --clean       Delete outputs (preserve SBML downloads)
  --clean-all   Delete everything including downloads
  --status      Show completion status
  --help        Show this help message

Examples:
  ./run_pipeline.sh                   # Default pipeline
  ./run_pipeline.sh --full            # With SSA ensemble validation
  ./run_pipeline.sh --from convert    # Start from conversion
  ./run_pipeline.sh --only fetch      # Only fetch models
  ./run_pipeline.sh --force           # Re-run everything
  ./run_pipeline.sh --clean --force   # Fresh start (keep downloads)

Stages (in order):
  1. fetch     - Download SBML from BioModels
  2. filter    - Identify SSA-suitable candidates
  3. convert   - SBML → BNGL → .net
  4. validate  - Multi-level validation
  5. report    - Generate benchmark catalog

The pipeline auto-detects completed stages and skips them
unless --force is used.
EOF
}

# ===========================================================================
# Main Runner
# ===========================================================================

run_stage() {
    local stage="$1"
    local force="${2:-false}"

    case "$stage" in
        fetch)
            if ! $force && is_fetch_complete; then
                log_warn "Fetch complete ($(count_files "$SBML_DIR" "*.xml") files)"
            else
                stage_fetch
            fi
            ;;
        filter)
            if ! $force && is_filter_complete; then
                log_warn "Filter complete ($(count_files "$CANDIDATES_DIR" "*.xml") candidates)"
            else
                stage_filter
            fi
            ;;
        convert)
            if ! $force && is_convert_complete; then
                log_warn "Convert complete ($(count_files "$NET_DIR" "*.net") .net files)"
            else
                stage_convert
            fi
            ;;
        validate)
            if ! $force && is_validate_complete; then
                log_warn "Validation complete"
            else
                stage_validate
            fi
            ;;
        report)
            stage_report
            ;;
        *)
            log_error "Unknown stage: $stage"
            exit 1
            ;;
    esac
}

# ===========================================================================
# Argument Parsing
# ===========================================================================

FORCE=false
CLEAN=false
CLEAN_ALL=false
START_FROM=""
ONLY_STAGE=""
INCLUDE_FULL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            START_FROM="$2"
            shift 2
            ;;
        --only)
            ONLY_STAGE="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --clean-all)
            CLEAN_ALL=true
            shift
            ;;
        --full)
            INCLUDE_FULL=true
            shift
            ;;
        --status)
            show_status
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ===========================================================================
# Main Execution
# ===========================================================================

log_stage "BioModels SSA Benchmark Pipeline"
echo "Working directory: $SCRIPT_DIR"
echo "Force mode: $FORCE"
echo "Full validation: $INCLUDE_FULL"
echo ""

# Check BNGPATH
if [[ -z "${BNGPATH:-}" ]]; then
    log_warn "BNGPATH not set, using default from config.py"
fi

# Clean if requested
if $CLEAN_ALL; then
    clean_all
elif $CLEAN; then
    clean_outputs
fi

# Single stage
if [[ -n "$ONLY_STAGE" ]]; then
    run_stage "$ONLY_STAGE" "$FORCE"
    log_success "Stage '$ONLY_STAGE' complete!"
    exit 0
fi

# Determine stages
STAGES=()
case "${START_FROM:-all}" in
    all|"")     STAGES=(fetch filter convert validate report) ;;
    fetch)      STAGES=(fetch filter convert validate report) ;;
    filter)     STAGES=(filter convert validate report) ;;
    convert)    STAGES=(convert validate report) ;;
    validate)   STAGES=(validate report) ;;
    report)     STAGES=(report) ;;
    *)
        log_error "Unknown stage: $START_FROM"
        exit 1
        ;;
esac

# Run pipeline
for stage in "${STAGES[@]}"; do
    run_stage "$stage" "$FORCE"
done

log_stage "Pipeline Complete!"
show_status
