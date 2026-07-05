#!/usr/bin/env python3
"""Forward sensitivity benchmark (Supplementary Table S9).

Primary timings (paper table): **BNGsim serial** CVODES forward sensitivity with
the same free-parameter list as AMICI (SBML export, name-aligned to the .net file),
in one coupled extended ODE solve (same problem size and ordering as AMICI ``sx``).
Each model is benchmarked once per CVODES corrector method (see "Methods" below);
both engines are pinned to the same method per timed pair so the comparison is
strictly apples-to-apples.

Predecessor: this script was previously named ``bench_forward_sensitivity.py`` and
also computed a Fisher Information Matrix (``Result.fisher_information(sigma=1.0)``)
as an annex per model. The FIM annex was a small side calculation (only its
shape and condition number were stored; the matrix itself was not used) so it
has been removed to keep this script focused on what it actually measures.
``Result.fisher_information(...)`` remains available in bngsim's user-facing
API for anyone wanting to do real identifiability analysis with their own
sigma; it is just not invoked from this benchmark anymore.

Reference timings: **BNGsim sharded** parallel chunked runs across a sweep of
worker counts (default 2,3,4,5,6 cores), one sweep per method. The sharded path
is a different paradigm from AMICI — it splits the sensitivity parameter list
into independent CVODES jobs and runs them in worker threads (the C++ CVODES
solve releases the GIL, so threads run concurrently on separate cores).

Also runs AMICI forward sensitivity on SBML exported from the companion BNGL,
once per method (the AMICI compile is shared across methods; only the solver's
internal corrector setting is toggled between timed runs).

Cross-engine alignment (the BNGsim ``.net`` is the source of truth):

  - **Initial conditions** are pulled from the ``.net`` and pushed into AMICI
    via ``set_initial_state(...)``. This handles ``setConcentration`` / equilibration
    actions in the ``.bngl`` (which the action-stripper removes before SBML
    conversion) so AMICI starts from the same state vector as BNGsim.
  - **Parameter values** are pulled from the ``.net`` and pushed into AMICI via
    ``set_free_parameter_by_id(...)`` / ``set_fixed_parameter_by_id(...)``.
    Defensive against future models that mutate parameters via
    ``setParameter`` / ``parameter_scan`` actions (also stripped from the SBML).
  - **Sensitivity parameter list** is reduced to AMICI's
    ``get_free_parameter_ids()``, in identical order — same Np-element vector
    on both sides.
  - After both seeders run, ``_verify_alignment(...)`` reports the count of
    diverging entries and the worst observed relative error (recorded in JSON
    as ``model['alignment_check']``; printed as one line per model). Anything
    non-zero means the engines would integrate slightly different problems.

BNGsim auto-codegen policy (auto mode, default): codegen is triggered on the
coupled forward-sensitivity system size ``n_species*(Np+1)`` (the "effective
RHS dimension") against ``S10_BNG_CODEGEN_MIN_EFFDIM`` (default 256) — mirroring
bngsim's own ``Simulator`` sensitivity auto-codegen, so the benchmark's auto
decision matches the library's. A few-parameter solve on a large network still
codegens; only a tiny coupled system stays on the interpreted in-memory
expression evaluator (where the compile cost cannot amortize). Override via
``S10_BNG_CODEGEN_MODE=always|never|auto``; thresholds are
``S10_BNG_CODEGEN_MIN_EFFDIM`` (coupled-size gate) and the legacy
``S10_BNG_CODEGEN_MIN_PARAMS`` (param-count fallback when species count is
unavailable). The summary tables
mark the models that triggered codegen with a ``cg=+`` column and list them
in the legend, so the reader can footnote them when reproducing the table.

Why the paper run uses ``S10_BNG_CODEGEN_MODE=always``: AMICI compiles its
RHS+Jacobian via CasADi → clang on every model, so the AMICI side is always
running compiled C. To keep the engines apples-to-apples for the paper
table, the BNGsim side is also run with codegen enabled across the full
suite (Np threshold bypassed). The ``always`` mode is set as the protocol
on the paper run; the JSON output records this at top-level
``protocol.bngsim_codegen_mode`` for later auditing, and Supplementary
Table S9 carries a footnote disclosing the choice.

Pre-simulation cost (BNG2.pl→SBML, AMICI compile, BNGsim setup+codegen) is
excluded from the timed medians on both sides — both engines snapshot
``time.perf_counter()`` after their respective build/load phases. The
benchmark records each pre-sim cost per model under
``model['pre_sim_ms']`` so the table can disclose what's discounted; on
this hardware AMICI compile dominates at ~25-65 s/model (~40-50% of total
wallclock), while BNGsim setup+codegen is sub-10 ms once the .so cache is
warm.

Methods (plain English — both still integrate state + sensitivities as one
coupled extended ODE in a single CVODES pass; they differ only in how each
step's nonlinear solve is structured):

  - **simultaneous** (CV_SIMULTANEOUS): state and all sensitivity variables are
    advanced together as one big coupled nonlinear system at every integration
    step. Often a touch faster per step on small / well-conditioned problems;
    the per-step solve is larger so it can struggle on stiff or large systems.
    This is **AMICI's compiled-in default**.
  - **staggered** (CV_STAGGERED): state is advanced first, then — with the new
    state in hand — sensitivities are advanced as a separate (linear-in-the-
    sensitivities) solve. Two smaller nonlinear solves per step instead of one
    big one. Often more robust for stiff or large systems. This is **CVODES'
    and BNGsim's default**.

  CVODES has a third mode (CV_STAGGERED1, one parameter at a time) that BNGsim
  does not currently expose; the benchmark does not measure it.

Models: subset of suite_ode.json that both BNGsim and AMICI can process.

Usage:
    python run.py                         # full run, both methods, sharded sweep 2..6
    python run.py --mode correctness      # alignment / cross-validation only (no sharded sweep)
    python run.py --mode timing           # timing report (default report is both)
    python run.py --effort low            # cheap subset (cumulative tiers)
    python run.py --quick                 # small models only
    python run.py --model egfr            # single model
    python run.py --no-sharded            # serial + AMICI only (faster)
    python run.py --methods simultaneous  # only the AMICI-default method
    python run.py --methods staggered     # only the BNGsim-default method
    S9_WORKER_COUNTS=2,4,6 python run.py  # custom sharded sweep

Output (git-ignored results/):
    forward_sens_results.json

Promoted from harness/comparison/bench_forward_sensitivity.py — the .net /
.bngl corpus is repointed at the vendored models/ tree and the runner
gained the locked --mode / --effort suite flags. The intricate AMICI
alignment machinery is otherwise unchanged.
"""

import argparse
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

_BENCH_ROOT = Path(__file__).resolve().parents[2]  # bngsim/benchmarks
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402
from _effort import add_effort_arg, effort_allows  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

# Paths / suite manifest. The .net corpus + suite manifest are vendored
# in-repo under models/ and _dev/; BNG2.pl is located via BNGPATH / BNG2_PL.
BENCHMARKS_DIR = _BENCH_ROOT
SUITE_ODE = _BENCH_ROOT / "_dev" / "suite_ode.json"
_BNGPATH = os.environ.get("BNGPATH", os.path.expanduser("~/Simulations/BioNetGen-2.9.3"))
BNG2_PL = os.environ.get("BNG2_PL", os.path.join(_BNGPATH, "BNG2.pl"))

get_machine_info = nb.machine_info


def load_suite(suite_path):
    """Load a suite manifest JSON file (list of model dicts)."""
    with open(suite_path) as f:
        data = json.load(f)
    return data["models"] if isinstance(data, dict) else data


def prepare_amici_runtime():
    """Import amici and its low-level SUNDIALS-binding submodule."""
    import amici
    import amici.sim.sundials as amici_sundials

    return amici, amici_sundials


def save_results(data, name):
    """Write the results JSON to the git-ignored results/ dir."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nResults: {path}")
    return path


def _find_bngl(name):
    """Locate a model's companion .bngl in the vendored models/bngl/ tree.

    Phase 3 vendored .net and .bngl into separate trees
    (``models/net/<role>/`` and ``models/bngl/<source>/``), so the
    companion .bngl is no longer beside the .net. Returns the first
    match, or None when no .bngl is vendored (BNGsim-only row).
    """
    hits = sorted((_BENCH_ROOT / "models" / "bngl").glob(f"*/{name}.bngl"))
    return hits[0] if hits else None


# Protocol
_WARMUP = 1
_RUNS = 3
SENS_MAX_STEPS = int(os.environ.get("SENS_MAX_STEPS", "1000000"))
S9_XVAL_TRAJ_RTOL = float(os.environ.get("S9_XVAL_TRAJ_RTOL", "1e-3"))
# Absolute floor for trajectory cross-validation denominators.  Without this,
# species that are genuinely ~0 (or exactly 0 after preprocessing) can produce
# astronomical relative errors from floating noise alone (e.g. ``ShcP()`` in
# ``egfr_path``), even when the two simulators agree on the biologically relevant scale.
S9_XVAL_TRAJ_ATOL = float(os.environ.get("S9_XVAL_TRAJ_ATOL", "1.0"))
S9_XVAL_SENS_RTOL = float(os.environ.get("S9_XVAL_SENS_RTOL", "1e-2"))
# Absolute floor for the symmetric relerr denominator on raw sensitivities.
# 1e-12 is below the noise floor of CVODES forward sensitivity at default
# tolerances, so cells where both engines compute "essentially zero" produce
# spurious relerr ~ 2.0 (opposite-sign float noise) and dominate the headline
# max. Default 1e-6 matches CVODES' default-tolerance forward-sensitivity
# noise floor: at zero crossings (e.g. SHP2_base_model R(...)/kkin_Y1 near
# t≈7) both engines diverge by ~1e-6 absolute purely from solver noise, so
# any tighter floor produces spurious headline relerr from noise alone.
S9_XVAL_SENS_ATOL = float(os.environ.get("S9_XVAL_SENS_ATOL", "1e-6"))
# Relative floor: any cell whose denominator is below
# (S9_XVAL_SENS_ATOL_REL * max|sx|) is treated as noise floor. This is the
# right scaling for sensitivity arrays that span many orders of magnitude
# across (state, param) pairs (e.g., near-zero IC parameters vs. dynamic
# rate constants on large-magnitude states).
S9_XVAL_SENS_ATOL_REL = float(os.environ.get("S9_XVAL_SENS_ATOL_REL", "1e-9"))
# Windowed denominator floor: a sensitivity trajectory that crosses zero
# pulls the local denominator toward its neighbouring peak rather than the
# noise floor. denom[t,sp,p] = max(|a|, |b|, S9_XVAL_SENS_ATOL_REL_WIN ×
# max|sa[t-w:t+w+1, sp, p]|, abs_floor). With WIN_RADIUS=5 and ATOL_REL_WIN
# =1e-2, a cell at noise floor in an otherwise 1e-2-scale trajectory has
# denom ~1e-4 — large enough that |diff| at CVODES default-tolerance noise
# (1e-6) doesn't dominate the headline relerr.
S9_XVAL_SENS_ATOL_REL_WIN = float(os.environ.get("S9_XVAL_SENS_ATOL_REL_WIN", "1e-2"))
S9_XVAL_SENS_WIN_RADIUS = int(os.environ.get("S9_XVAL_SENS_WIN_RADIUS", "5"))
S9_XVAL_ALLOW_ASSIGNMENT = os.environ.get("S9_XVAL_ALLOW_ASSIGNMENT", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
S9_CHUNK_SIZE = int(os.environ.get("S9_CHUNK_SIZE", "2"))
S9_WORKERS_ENV = os.environ.get("S9_WORKER_COUNTS", "2,3,4,5,6")
# Upper limit on free parameters for serial all-at-once BNGsim (avoid accidental OOM).
# Set to a very large integer to attempt serial for all models.
S9_SERIAL_MAX_PARAMS = int(os.environ.get("S9_SERIAL_MAX_PARAMS", str(10**6)))
S10_BNG_CODEGEN_MODE = os.environ.get("S10_BNG_CODEGEN_MODE", "auto").strip().lower()
S10_BNG_CODEGEN_MIN_PARAMS = int(os.environ.get("S10_BNG_CODEGEN_MIN_PARAMS", "30"))
# Auto-mode codegen trigger, expressed as the coupled forward-sensitivity system
# size n_species*(n_params+1) — mirrors bngsim's own Simulator sensitivity
# auto-codegen so the benchmark's auto decision matches the library's. (The
# legacy S10_BNG_CODEGEN_MIN_PARAMS gate is the fallback when the species count
# is unavailable.) Default 256 == the library's BNGSIM_CODEGEN_THRESHOLD.
S10_BNG_CODEGEN_MIN_EFFDIM = int(os.environ.get("S10_BNG_CODEGEN_MIN_EFFDIM", "256"))

# Target models (medium/large from suite_ode.json)
TARGET_MODELS = [
    "egfr_path",  # 18 sp, ~11 params (quick sanity)
    "tcr_signaling",  # 37 sp, ~23 params
    "Scaff_22_ground",  # 85 sp, ~27 params
    "SHP2_base_model",  # 149 sp, ~40 params
]

# --effort tier per target model (by sensitivity problem size / cost).
MODEL_EFFORT = {
    "egfr_path": "low",
    "tcr_signaling": "medium",
    "Scaff_22_ground": "high",
    "SHP2_base_model": "high",
}


def _use_bng_codegen_for_model(n_params: int, n_species: int | None = None) -> bool:
    mode = S10_BNG_CODEGEN_MODE
    if mode in {"1", "true", "yes", "on", "always"}:
        return True
    if mode in {"0", "false", "no", "off", "never"}:
        return False
    # auto mode: enable codegen on the coupled forward-sensitivity system size
    # n_species*(n_params+1), matching bngsim's Simulator auto-codegen decision —
    # a few-param solve on a large network still codegens, where the old
    # n_params-only gate would have starved it onto the interpreted path. Falls
    # back to the param-count gate when the species count is unknown.
    try:
        if n_species is not None and int(n_species) > 0:
            return int(n_species) * (int(n_params) + 1) >= int(S10_BNG_CODEGEN_MIN_EFFDIM)
    except (TypeError, ValueError):
        pass
    return int(n_params) >= int(S10_BNG_CODEGEN_MIN_PARAMS)


# ── Helpers ───────────────────────────────────────────────────────────────


def get_param_names(net_path):
    """Get all parameter names from a .net file."""
    from bngsim._bngsim_core import NetworkModel

    model = NetworkModel.from_net(str(net_path))
    return list(model.param_names)


def geometric_mean(values):
    if not values:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in values) / len(values)))


def _relerr_stats(a, b, atol=1e-12):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), atol)
    rel = np.abs(aa - bb) / denom
    return {
        "max": float(np.max(rel)),
        "p95": float(np.percentile(rel, 95)),
        "med": float(np.median(rel)),
    }


def _windowed_max_abs(sx_a, sx_b, radius):
    """Per-cell windowed peak magnitude over a half-window of ``radius``
    samples on each side of the time axis. Used as a local denominator
    scale so a zero-crossing sample inherits the neighbourhood's magnitude
    instead of being dominated by the absolute solver-noise floor."""
    if sx_a.size == 0:
        return np.zeros_like(sx_a)
    abs_max = np.maximum(np.abs(sx_a), np.abs(sx_b))
    n_t = abs_max.shape[0]
    if radius <= 0 or n_t <= 1:
        return abs_max
    out = np.empty_like(abs_max)
    for i in range(n_t):
        lo = max(0, i - radius)
        hi = min(n_t, i + radius + 1)
        out[i] = abs_max[lo:hi].max(axis=0)
    return out


def _normalize_id(name):
    s = str(name)
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if s.startswith("_ant_"):
        s = s[5:]
    if "::" in s:
        s = s.split("::", 1)[1]
    return s


def _xval_species_key(name):
    """Map BNGsim / SBML species labels onto a common key for alignment.

    BNGsim uses BNGL-style names from the .net file (often ``Species()``), while AMICI
    state ids from SBML export typically omit the empty ``()`` call syntax. Stripping
    ``()`` after ``_normalize_id`` makes the Table S9 cross-check usable.
    """
    s = _normalize_id(name)
    return s.replace("()", "")


# ── IC parameter linkage recovery (issue (A) in dev/report-residual-fwd-sens-bugs.md) ──
#
# Two complementary asymmetries break ∂y(0)/∂p between BNGsim and AMICI:
#
#   (A1) ``setConcentration`` actions in the .bngl preserve ``species → param``
#        links in the BNG2.pl-emitted .net (BNGsim sees them) but the bench's
#        action-stripper drops them when generating SBML, so AMICI loses them.
#        Fix: walk the .net's species block; for any line whose IC column is a
#        parameter name, ensure the SBML has a matching <initialAssignment>.
#
#   (A2) ``begin seed species`` parameter-named ICs (``Grb2 Grb2_tot``) survive
#        in SBML via libSBML's <initialAssignment>, but BNG2.pl substitutes the
#        post-equilibration *literal* into the .net's species block (because
#        the .bngl runs ``simulate(steady_state=>1)`` before ``writeNetwork``),
#        so BNGsim loses the link. Fix: rewrite the .net so each such species
#        line shows the parameter name, and override the parameter's value to
#        the .net's literal IC (so BNGsim's runtime IC matches the post-eq
#        value). After ``Model.from_net``, the bench restores the parameter's
#        nominal value via ``set_param`` — derived params re-evaluate, but
#        species ICs are not re-propagated, so the runtime IC stays put.


_SPECIES_LINE_RE = re.compile(r"^\s*(?P<idx>\d+)\s+(?P<sp>\S+)\s+(?P<ic>\S+)\s*$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


def _strip_empty_parens(name: str) -> str:
    """Strip a trailing empty ``()`` suffix from a species name.

    BNG's bare-pattern species (``EGF``) get rendered as ``EGF()`` in
    the .net's species block and the SBML's species name attribute,
    while a full-state species (``pMHC(p~ag,tcr)``) carries meaningful
    parens. We want the empty-suffix form to round-trip to the bare name
    but the full-state form to remain literal.
    """
    s = name.strip()
    if s.endswith("()"):
        return s[:-2]
    return s


def _parse_bngl_seed_species_param_refs(bngl_path: Path) -> dict[str, str]:
    """Return ``{species_name: param_name}`` for every entry in the .bngl's
    ``begin seed species`` block whose IC column is a parameter name (i.e.
    not a numeric literal). Species names are returned without trailing
    ``()`` so they line up with BNGsim's species naming.
    """
    refs: dict[str, str] = {}
    try:
        text = Path(bngl_path).read_text()
    except OSError:
        return refs
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        low = stripped.lower()
        if low.startswith("begin seed species") or low.startswith("begin species"):
            in_block = True
            continue
        if low.startswith("end seed species") or low.startswith("end species"):
            in_block = False
            continue
        if not in_block or not stripped:
            continue
        # Strip trailing comment
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].strip()
            if not stripped:
                continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        sp_name = _strip_empty_parens(parts[0])
        ic_tok = parts[-1]
        if _FLOAT_RE.match(ic_tok):
            continue
        # Parameter-name IC (e.g., "Grb2_tot"). Strip a leading $ marker for fixed species.
        if ic_tok.startswith("$"):
            ic_tok = ic_tok[1:]
        if not _IDENT_RE.match(ic_tok):
            continue
        refs[sp_name] = ic_tok
    return refs


def _parse_net_species_block(net_path: Path) -> list[tuple[int, str, str]]:
    """Return a list of ``(idx, species_name, ic_token)`` for the .net's
    ``begin species`` block, preserving file order. The ic_token is the raw
    string (numeric literal OR parameter name)."""
    out: list[tuple[int, str, str]] = []
    text = Path(net_path).read_text()
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("begin species"):
            in_block = True
            continue
        if stripped.startswith("end species"):
            in_block = False
            continue
        if not in_block or not stripped:
            continue
        # Strip trailing comment
        body = stripped.split("#", 1)[0].strip()
        if not body:
            continue
        m = _SPECIES_LINE_RE.match(body)
        if not m:
            continue
        idx = int(m.group("idx"))
        sp = m.group("sp")
        ic = m.group("ic")
        sp_clean = _strip_empty_parens(sp.lstrip("$"))
        out.append((idx, sp_clean, ic))
    return out


def _parse_net_param_values(net_path: Path) -> dict[str, float]:
    """Return ``{param_name: value}`` for every numeric parameter in the
    .net's ``begin parameters`` block. Skips ConstantExpression (derived)
    parameters since their value is not a stable literal."""
    values: dict[str, float] = {}
    text = Path(net_path).read_text()
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("begin parameters"):
            in_block = True
            continue
        if stripped.startswith("end parameters"):
            in_block = False
            continue
        if not in_block or not stripped or stripped.startswith("#"):
            continue
        if "ConstantExpression" in stripped:
            continue
        body = stripped.split("#", 1)[0].strip()
        if not body:
            continue
        parts = body.split()
        if len(parts) < 3:
            continue
        with contextlib.suppress(ValueError):
            values[parts[1]] = float(parts[2])
    return values


def _collect_ic_param_links(
    bngl_path: Path | None, net_path: Path
) -> tuple[dict[str, str], dict[str, float]]:
    """Compute the (species → IC parameter) link map for one model.

    Returns:
      ic_link_map: ``{species_name_no_parens: param_name}`` for every species
        whose IC should be tied to a parameter. Sources, in priority order:
          1. .net species block IC = parameter name (e.g., ``EGF EGF_tot``
             post-``setConcentration``). BNGsim already tracks these.
          2. .bngl ``begin seed species`` IC = parameter name (e.g.,
             ``Grb2 Grb2_tot``). Lost on the BNGsim side after equilibration.
      param_overrides: ``{param_name: value_to_set_back_after_load}``. Only
        populated for category-2 entries where the post-equilibration .net
        literal differs from the parameter's nominal value — those need a
        rewritten .net that pins the param to the post-eq literal so
        BNGsim's runtime IC is correct, then a ``set_param`` after load to
        restore the nominal so reaction rates evaluate at the original.
    """
    ic_link_map: dict[str, str] = {}
    param_overrides: dict[str, float] = {}

    net_species = _parse_net_species_block(net_path)
    net_params = _parse_net_param_values(net_path)

    # Category 1: .net IC column is already a parameter name.
    for _idx, sp_name, ic in net_species:
        if _FLOAT_RE.match(ic):
            continue
        # Already a parameter reference. Strip $ for fixed species.
        candidate = ic.lstrip("$")
        if _IDENT_RE.match(candidate):
            ic_link_map[sp_name] = candidate

    if bngl_path is None or not Path(bngl_path).exists():
        return ic_link_map, param_overrides

    # Category 2: .bngl seed species IC is a parameter, but the .net has a
    # literal value (BNG2.pl substituted it after equilibration).
    bngl_refs = _parse_bngl_seed_species_param_refs(Path(bngl_path))
    if not bngl_refs:
        return ic_link_map, param_overrides

    net_species_by_name = {sp: (idx, ic) for idx, sp, ic in net_species}

    for sp_name, p_name in bngl_refs.items():
        if sp_name in ic_link_map:
            continue  # already resolved via .net
        if sp_name not in net_species_by_name:
            continue
        if p_name not in net_params:
            continue
        _idx, net_ic = net_species_by_name[sp_name]
        if not _FLOAT_RE.match(net_ic):
            continue  # already param-named, but not in ic_link_map?
        try:
            net_ic_value = float(net_ic)
        except ValueError:
            continue
        ic_link_map[sp_name] = p_name
        nominal = net_params[p_name]
        # Only override the param value when the post-eq literal differs
        # from the nominal beyond fp noise. If they agree, no runtime fixup
        # is needed because BNGsim's resolved IC will already match.
        if not math.isclose(net_ic_value, nominal, rel_tol=1e-9, abs_tol=1e-12):
            param_overrides[p_name] = nominal
    return ic_link_map, param_overrides


def _rewrite_net_with_ic_links(
    net_path: Path,
    ic_link_map: dict[str, str],
    param_overrides: dict[str, float],
    out_path: Path,
    *,
    net_species: list[tuple[int, str, str]] | None = None,
) -> None:
    """Write a copy of ``net_path`` whose species block expresses each
    ``species → param`` link literally, and whose parameters block has
    ``param_overrides`` keys pinned to the .net's post-equilibration
    literal IC value (so BNGsim builds with runtime IC = post-eq value).

    The bench restores those parameters to their nominal values by
    calling ``model.set_param`` after ``Model.from_net``.
    """
    text = Path(net_path).read_text()
    if net_species is None:
        net_species = _parse_net_species_block(net_path)

    species_literal_by_name: dict[str, str] = {
        sp: ic for _idx, sp, ic in net_species if _FLOAT_RE.match(ic)
    }

    out_lines: list[str] = []
    in_species = False
    in_params = False
    for line in text.splitlines():
        stripped_full = line.strip()
        if stripped_full.startswith("begin species"):
            in_species = True
            out_lines.append(line)
            continue
        if stripped_full.startswith("end species"):
            in_species = False
            out_lines.append(line)
            continue
        if stripped_full.startswith("begin parameters"):
            in_params = True
            out_lines.append(line)
            continue
        if stripped_full.startswith("end parameters"):
            in_params = False
            out_lines.append(line)
            continue

        if in_species and stripped_full and not stripped_full.startswith("#"):
            body = stripped_full.split("#", 1)
            payload = body[0].strip()
            comment = ("  #" + body[1]) if len(body) > 1 else ""
            m = _SPECIES_LINE_RE.match(payload)
            if m:
                sp_clean = _strip_empty_parens(m.group("sp").lstrip("$"))
                if sp_clean in ic_link_map and _FLOAT_RE.match(m.group("ic")):
                    new_payload = f"   {m.group('idx')} {m.group('sp')} {ic_link_map[sp_clean]}"
                    out_lines.append(new_payload + comment)
                    continue
            out_lines.append(line)
            continue

        if in_params and stripped_full and not stripped_full.startswith("#") and param_overrides:
            body = stripped_full.split("#", 1)
            payload = body[0].strip()
            comment = ("  #" + body[1]) if len(body) > 1 else ""
            parts = payload.split(None, 2)
            if len(parts) >= 3 and parts[1] in param_overrides:
                p_name = parts[1]
                # Find the species using this parameter to look up its post-eq
                # literal — that's the value we want pinned in the rewritten .net.
                # (We deliberately use the species' literal, not the param's
                # nominal, so BNGsim's species IC equals the post-eq value.)
                target_value: float | None = None
                for sp_name, linked_p in ic_link_map.items():
                    if linked_p == p_name and sp_name in species_literal_by_name:
                        try:
                            target_value = float(species_literal_by_name[sp_name])
                        except ValueError:
                            target_value = None
                        break
                if target_value is not None:
                    new_payload = f"    {parts[0]} {p_name} {target_value:.12e}"
                    out_lines.append(new_payload + comment)
                    continue
            out_lines.append(line)
            continue

        out_lines.append(line)
    Path(out_path).write_text("\n".join(out_lines) + "\n")


def _inject_sbml_initial_assignments(sbml_path: str, ic_link_map: dict[str, str]) -> dict:
    """Inject ``<initialAssignment>`` entries into the SBML at ``sbml_path``
    for any (species, parameter) pair in ``ic_link_map`` that the SBML
    doesn't already cover. Reports counts; treats SBML id "S{idx}" mapping
    failures as soft errors (skipped).

    BNGsim writes SBML species ids in the form ``S{1-based_index}``; we
    map species *name* → SBML *id* via the species-name attribute that
    BNG2.pl emits.
    """
    import libsbml

    info = {
        "injected": [],
        "already_present": [],
        "missing_species": [],
        "missing_param": [],
        "amici_only": [],
    }
    if not ic_link_map:
        return info

    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromFile(sbml_path)
    if doc.getNumErrors() > 0:
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                info.setdefault("read_errors", []).append(err.getMessage())
    sbml_model = doc.getModel()
    if sbml_model is None:
        info["error"] = "no_model"
        return info

    name_to_sid: dict[str, str] = {}
    for i in range(sbml_model.getNumSpecies()):
        sp = sbml_model.getSpecies(i)
        # BNG2.pl writeSBML uses name="EGF()" and id="S1". Map by stripped name.
        nm = (sp.getName() or "").strip()
        if not nm:
            continue
        nm_clean = _strip_empty_parens(nm.lstrip("$"))
        name_to_sid[nm_clean] = sp.getId()

    sbml_param_ids = {
        sbml_model.getParameter(i).getId() for i in range(sbml_model.getNumParameters())
    }

    for sp_name, p_name in ic_link_map.items():
        sid = name_to_sid.get(sp_name)
        if sid is None:
            info["missing_species"].append(sp_name)
            continue
        if p_name not in sbml_param_ids:
            # AMICI requires the parameter to exist in SBML; otherwise the
            # initialAssignment would dangle. Skip rather than corrupt.
            info["missing_param"].append((sp_name, p_name))
            continue
        existing = sbml_model.getInitialAssignment(sid)
        if existing is not None:
            info["already_present"].append((sp_name, p_name))
            continue
        ia = sbml_model.createInitialAssignment()
        ia.setSymbol(sid)
        math_ast = libsbml.parseL3Formula(p_name)
        if math_ast is None:
            info.setdefault("formula_parse_errors", []).append((sp_name, p_name))
            continue
        ia.setMath(math_ast)
        info["injected"].append((sp_name, p_name))

    # Detect AMICI-only links: any species-id <initialAssignment> in the SBML
    # whose math is a single parameter reference but whose species isn't
    # in our ic_link_map (BNGsim's view). After (A1)+(A2a) this should be
    # empty for the four target models.
    sid_to_name = {
        sbml_model.getSpecies(i).getId(): sbml_model.getSpecies(i).getName()
        for i in range(sbml_model.getNumSpecies())
    }
    bng_pairs = {(sp, p) for sp, p in ic_link_map.items()}
    for i in range(sbml_model.getNumInitialAssignments()):
        ia = sbml_model.getInitialAssignment(i)
        sym = ia.getSymbol()
        if sym not in sid_to_name:
            continue  # parameter-level assignment (e.g., loop1..loop5)
        formula = libsbml.formulaToString(ia.getMath()) or ""
        formula_clean = formula.strip()
        if not _IDENT_RE.match(formula_clean):
            continue  # not a single parameter reference
        sp_clean = _strip_empty_parens((sid_to_name[sym] or "").lstrip("$"))
        if (sp_clean, formula_clean) not in bng_pairs:
            info["amici_only"].append((sp_clean, formula_clean))

    if info["injected"]:
        libsbml.writeSBMLToFile(doc, sbml_path)
    return info


def _apply_param_overrides(model, overrides: dict[str, float]) -> None:
    """Restore each parameter's nominal value after BNGsim loads a rewritten
    .net (see ``_rewrite_net_with_ic_links``). Derived params re-evaluate;
    species ICs are *not* re-propagated, so the runtime IC stays at the
    post-equilibration literal that the rewritten .net pinned."""
    for name, value in overrides.items():
        with contextlib.suppress(Exception):
            model.set_param(name, value)


def _bngsim_species_y0_vector(net_path):
    """Species amounts/concentrations at t=0 from the frozen .net (BNGsim reference).

    Several shipped BNGL files include **actions** (equilibration, ``setConcentration``,
    etc.) that establish the initial condition actually used in the companion ``.net``.
    SBML export from the BNGL model block alone can therefore disagree with the ``.net``
    (notably ``egfr_path``). For AMICI↔BNGsim paired benchmarks we therefore seed AMICI
    from the same ``.net`` IC vector BNGsim uses.
    """
    import bngsim

    mb = bngsim.Model.from_net(str(net_path))
    rb = bngsim.Simulator(mb, method="ode").run(t_span=(0.0, 1e-12), n_points=2)
    y0 = np.asarray(rb.species[0], dtype=float)
    names = list(rb.species_names)
    return y0, names


def _amici_state_id_list(model):
    if hasattr(model, "getStateIds"):
        return list(model.getStateIds())
    if hasattr(model, "get_state_ids"):
        return list(model.get_state_ids())
    if hasattr(model, "get_state_ids_solver"):
        return list(model.get_state_ids_solver())
    return []


def _amici_state_name_list(model):
    if hasattr(model, "getStateNames"):
        return list(model.getStateNames())
    if hasattr(model, "get_state_names"):
        return list(model.get_state_names())
    if hasattr(model, "get_state_names_solver"):
        return list(model.get_state_names_solver())
    return []


def seed_amici_initial_state_from_net(model, net_path):
    """Override AMICI initial states to match BNGsim's ``.net`` ICs (best-effort).

    Returns a small dict describing what happened (for JSON/debuggability).
    """
    try:
        y0, b_names = _bngsim_species_y0_vector(net_path)
    except Exception as e:
        return {"ok": False, "reason": f"bngsim_ic_probe_failed:{e}"}

    s_ids = _amici_state_id_list(model)
    s_names = _amici_state_name_list(model)
    if not s_ids or len(s_ids) != int(y0.shape[0]):
        return {
            "ok": False,
            "reason": "state_count_mismatch",
            "n_amici_states": len(s_ids),
            "n_bng_species": int(y0.shape[0]),
        }

    b_key_to_i = {_xval_species_key(n): i for i, n in enumerate(b_names)}
    x_new = np.zeros(len(s_ids), dtype=float)
    missing = []
    for j, sid in enumerate(s_ids):
        label = s_names[j] if j < len(s_names) and str(s_names[j]).strip() else sid
        key = _xval_species_key(label)
        if key in b_key_to_i:
            x_new[j] = float(y0[b_key_to_i[key]])
        else:
            missing.append(str(sid))
    if missing:
        # If we cannot map every state, do not partially rewrite AMICI's IC vector.
        return {
            "ok": False,
            "reason": "unmapped_states",
            "n_missing": len(missing),
            "missing_sample": missing[:12],
        }

    if hasattr(model, "set_initial_state"):
        model.set_initial_state(x_new)
    elif hasattr(model, "setInitialState"):
        model.setInitialState(x_new)
    else:
        return {"ok": False, "reason": "no_set_initial_state_api"}

    return {
        "ok": True,
        "n_states": len(s_ids),
        "source": "bngsim.net_probe",
    }


def _bngsim_param_value_map(net_path):
    """Return ``{param_name: value}`` for every parameter in the .net file.

    These are the post-action parameter values BNGsim actually integrates
    with — the canonical "what BNGsim runs" reference for cross-engine
    alignment (mirrors the role of ``_bngsim_species_y0_vector`` for ICs).
    """
    import bngsim

    model = bngsim.Model.from_net(str(net_path))
    return {n: float(model.get_param(n)) for n in model._core.param_names}


def _amici_get_free_parameter_value(model, pid):
    """Best-effort getter for one of AMICI's free parameter values, by id."""
    if hasattr(model, "get_free_parameter_by_id"):
        try:
            return float(model.get_free_parameter_by_id(pid))
        except Exception:
            return None
    if hasattr(model, "getParameterById"):
        try:
            return float(model.getParameterById(pid))
        except Exception:
            return None
    return None


def _amici_set_parameter_by_id(model, pid, value):
    """Try free-parameter setter first, then fixed-parameter setter.

    Returns the kind that took ('free', 'fixed') or ``None`` if neither
    accepted the id. Handles both snake_case and camelCase AMICI bindings.
    """
    # snake_case (current AMICI)
    if hasattr(model, "set_free_parameter_by_id"):
        try:
            model.set_free_parameter_by_id(pid, float(value))
            return "free"
        except Exception:
            pass
    if hasattr(model, "set_fixed_parameter_by_id"):
        try:
            model.set_fixed_parameter_by_id(pid, float(value))
            return "fixed"
        except Exception:
            pass
    # camelCase (older AMICI)
    if hasattr(model, "setParameterById"):
        try:
            model.setParameterById(pid, float(value))
            return "free"
        except Exception:
            pass
    if hasattr(model, "setFixedParameterById"):
        try:
            model.setFixedParameterById(pid, float(value))
            return "fixed"
        except Exception:
            pass
    return None


def seed_amici_parameters_from_net(model, net_path):
    """Push BNGsim ``.net`` parameter values into AMICI for every shared id.

    Defensive companion to ``seed_amici_initial_state_from_net``: aligns
    parameter values across engines so that even if BNG2.pl emits different
    numerics for ``.net`` vs SBML (e.g., a future model uses ``setParameter``
    actions, which the action-stripper drops before SBML conversion), both
    engines integrate the same problem. For models without parameter-mutating
    actions this is a no-op against bit-identical values; the explicit
    overwrite still serves as a contract that BNGsim's ``.net`` is the source
    of truth.

    Returns a status dict with counts and a small sample of any names that
    could not be resolved on the AMICI side.
    """
    bng_params = _bngsim_param_value_map(net_path)

    free_ids = []
    if hasattr(model, "get_free_parameter_ids"):
        free_ids = list(model.get_free_parameter_ids())
    elif hasattr(model, "getParameterIds"):
        free_ids = list(model.getParameterIds())

    fixed_ids = []
    if hasattr(model, "get_fixed_parameter_ids"):
        fixed_ids = list(model.get_fixed_parameter_ids())
    elif hasattr(model, "getFixedParameterIds"):
        fixed_ids = list(model.getFixedParameterIds())

    amici_ids = list(free_ids) + list(fixed_ids)

    n_set_free = 0
    n_set_fixed = 0
    unmatched_amici = []
    failed_set = []

    # Push every matching id from .net into AMICI; remember the rest.
    for pid in amici_ids:
        if pid not in bng_params:
            unmatched_amici.append(pid)
            continue
        kind = _amici_set_parameter_by_id(model, pid, bng_params[pid])
        if kind == "free":
            n_set_free += 1
        elif kind == "fixed":
            n_set_fixed += 1
        else:
            failed_set.append(pid)

    return {
        "ok": not failed_set,
        "source": "bngsim.net_probe",
        "n_amici_free": len(free_ids),
        "n_amici_fixed": len(fixed_ids),
        "n_set_free": n_set_free,
        "n_set_fixed": n_set_fixed,
        "n_unmatched_amici_ids": len(unmatched_amici),
        "unmatched_amici_sample": unmatched_amici[:12],
        "failed_set_sample": failed_set[:12],
    }


def _verify_alignment(model, net_path, *, atol=1e-12, rtol=1e-12):
    """Compare AMICI's current parameter values + initial state vs the .net.

    Run AFTER ``seed_amici_initial_state_from_net`` and
    ``seed_amici_parameters_from_net`` so this is a defense-in-depth check:
    if both seeders worked and BNG2.pl emitted consistent ``.net`` and SBML,
    every common entry has zero relative error. Anything non-zero here is a
    problem worth surfacing in JSON and the terminal.

    Returns ``{"params": {...}, "ic": {...}}`` with per-section counts of
    common/diverged entries, the worst observed relative error, and a small
    sample of divergent entries for triage.
    """

    def _relerr(a, b):
        if a == 0.0 and b == 0.0:
            return 0.0
        denom = max(abs(a), abs(b), atol)
        return abs(a - b) / denom

    # Parameters
    bng_params = _bngsim_param_value_map(net_path)
    free_ids = (
        list(model.get_free_parameter_ids())
        if hasattr(model, "get_free_parameter_ids")
        else (list(model.getParameterIds()) if hasattr(model, "getParameterIds") else [])
    )
    p_diverged = []
    p_max_re = 0.0
    p_n_common = 0
    for pid in free_ids:
        if pid not in bng_params:
            continue
        amici_val = _amici_get_free_parameter_value(model, pid)
        if amici_val is None:
            continue
        p_n_common += 1
        re = _relerr(bng_params[pid], amici_val)
        if re > p_max_re:
            p_max_re = re
        if re > rtol:
            p_diverged.append(
                {"name": pid, "net": bng_params[pid], "amici": amici_val, "relerr": re}
            )

    # Initial state (species)
    try:
        y0_net, _names = _bngsim_species_y0_vector(net_path)
    except Exception as e:
        ic_block = {
            "ok": False,
            "reason": f"bngsim_ic_probe_failed:{e}",
            "n_common": 0,
            "n_diverged": 0,
            "max_relerr": None,
            "diverged_sample": [],
        }
        return {
            "params": {
                "ok": not p_diverged,
                "n_common": p_n_common,
                "n_diverged": len(p_diverged),
                "max_relerr": p_max_re,
                "diverged_sample": p_diverged[:12],
            },
            "ic": ic_block,
        }

    s_ids = _amici_state_id_list(model)
    s_names = _amici_state_name_list(model)
    if hasattr(model, "get_initial_state"):
        amici_y0 = list(model.get_initial_state())
    elif hasattr(model, "getInitialState"):
        amici_y0 = list(model.getInitialState())
    else:
        amici_y0 = []

    ic_diverged = []
    ic_max_re = 0.0
    ic_n_common = 0
    if s_ids and len(amici_y0) == len(s_ids):
        # Map AMICI state ids to BNGsim species index using the same
        # convention as seed_amici_initial_state_from_net.
        b_key_to_i = {_xval_species_key(n): i for i, n in enumerate(_names)}
        for j, sid in enumerate(s_ids):
            label = s_names[j] if j < len(s_names) and str(s_names[j]).strip() else sid
            key = _xval_species_key(label)
            if key not in b_key_to_i:
                continue
            ic_n_common += 1
            net_val = float(y0_net[b_key_to_i[key]])
            amici_val = float(amici_y0[j])
            re = _relerr(net_val, amici_val)
            if re > ic_max_re:
                ic_max_re = re
            if re > rtol:
                ic_diverged.append(
                    {"name": str(sid), "net": net_val, "amici": amici_val, "relerr": re}
                )

    return {
        "params": {
            "ok": not p_diverged,
            "n_common": p_n_common,
            "n_diverged": len(p_diverged),
            "max_relerr": p_max_re,
            "diverged_sample": p_diverged[:12],
        },
        "ic": {
            "ok": not ic_diverged,
            "n_common": ic_n_common,
            "n_diverged": len(ic_diverged),
            "max_relerr": ic_max_re,
            "diverged_sample": ic_diverged[:12],
        },
    }


def _amici_all_parameter_ids_values(model):
    """Robust across AMICI Python binding variants (camelCase vs snake_case)."""
    if hasattr(model, "getParameterIds"):
        pids = list(model.getParameterIds())
    elif hasattr(model, "get_parameter_list"):
        plist = list(model.get_parameter_list())
        pids = []
        for p in plist:
            if hasattr(p, "get_id"):
                pids.append(p.get_id())
            elif hasattr(p, "getId"):
                pids.append(p.getId())
            else:
                pids.append(str(p))
    else:
        pids = []

    if hasattr(model, "getUnscaledParameters"):
        pvs = list(model.getUnscaledParameters())
    elif hasattr(model, "get_unscaled_parameters"):
        pvs = list(model.get_unscaled_parameters())
    elif hasattr(model, "getParameters"):
        pvs = list(model.getParameters())
    elif hasattr(model, "get_parameters"):
        pvs = list(model.get_parameters())
    else:
        pvs = []

    out = {}
    if pids and pvs and len(pids) == len(pvs):
        out = {str(a): float(b) for a, b in zip(pids, pvs, strict=False)}
    return pids, pvs, out


def align_bng_params_to_amici(bng_param_names, amici_param_ids):
    """Map AMICI free-parameter ids (SBML export order) onto BNGsim .net parameter names.

    Table S9 times the **same** forward-sensitivity problem: BNGsim uses
    ``sensitivity_params=bench_params`` where ``bench_params`` matches AMICI's
    ``get_free_parameter_ids()`` order. Parameters fixed in SBML (omitted from AMICI
    sensitivities) are intentionally excluded from both engines.
    """
    bng_list = list(bng_param_names)
    bng_set = set(bng_list)
    by_norm = {}
    for n in bng_list:
        kn = _normalize_id(n)
        if kn not in by_norm:
            by_norm[kn] = n

    bench_params = []
    match_modes = []
    unmatched = []
    for aid in amici_param_ids:
        chosen = None
        mode = None
        if aid in bng_set:
            chosen = aid
            mode = "name"
        else:
            an = _normalize_id(aid)
            if an in by_norm:
                chosen = by_norm[an]
                mode = "normalized_id"
            else:
                for bn in bng_list:
                    if _normalize_id(bn) == an:
                        chosen = bn
                        mode = "normalized_both"
                        break
        if chosen is None:
            unmatched.append(aid)
        else:
            bench_params.append(chosen)
            match_modes.append(mode)

    if unmatched:
        return {
            "ok": False,
            "bench_params": None,
            "unmatched_amici_ids": unmatched,
            "note": "No BNG .net parameter matched these AMICI free-parameter ids",
        }

    return {
        "ok": True,
        "bench_params": bench_params,
        "per_param_match": match_modes,
        "match": "mixed" if len(set(match_modes)) > 1 else match_modes[0],
    }


def _sens_xval(bng_result, amici_meta):
    """Cross-validate trajectory + forward sensitivities on common axes."""
    bng_states = list(getattr(bng_result, "species_names", []))
    bng_params = list(getattr(bng_result, "sensitivity_params", []))
    bng_x = np.asarray(getattr(bng_result, "species", np.array([])))
    bng_sx = np.asarray(getattr(bng_result, "sensitivities", np.array([])))

    # AMICI SBML exports often use internal state ids like ``S1``, ``S2``, ... while
    # still providing human-readable ``get_state_names()`` labels that match BNGL
    # ``Species()`` strings. For cross-validation we prefer the readable labels when
    # available so trajectory + sensitivity slices line up with BNGsim's ``.net``.
    am_state_ids = list(amici_meta.get("state_ids", []))
    am_state_names = list(amici_meta.get("state_names", []))
    if am_state_names and am_state_ids and len(am_state_names) == len(am_state_ids):
        am_states = am_state_names
    else:
        am_states = am_state_ids
    am_params = list(amici_meta.get("param_ids", []))
    am_x = np.asarray(amici_meta.get("x", np.array([])))
    am_sx = np.asarray(amici_meta.get("sx", np.array([])))

    if bng_x.size == 0 or bng_sx.size == 0 or am_x.size == 0 or am_sx.size == 0:
        return {
            "ok": False,
            "category": "missing_arrays",
            "error": "missing trajectory/sensitivity arrays",
        }
    if bng_sx.ndim != 3 or am_sx.ndim != 3:
        return {
            "ok": False,
            "category": "shape_mismatch",
            "error": f"unexpected sens dims bng={bng_sx.shape} am={am_sx.shape}",
        }
    nt = min(bng_x.shape[0], am_x.shape[0], bng_sx.shape[0], am_sx.shape[0])
    if nt <= 0:
        return {"ok": False, "category": "empty_time_axis", "error": "empty time axis"}

    # BNGsim's CVODES sensitivity path can occasionally report an inconsistent ``t=0``
    # species row while subsequent samples match AMICI (observed on ``egfr_path``).
    # When that pattern is strong, drop the first time sample for xval only.
    t0_drop = 0
    if nt >= 3:
        n_state_probe = int(min(bng_x.shape[1], am_x.shape[1], bng_sx.shape[1], am_sx.shape[1]))
        if n_state_probe > 0:
            bb0 = np.asarray(bng_x[0, :n_state_probe], dtype=float)
            aa0 = np.asarray(am_x[0, :n_state_probe], dtype=float)
            bb1 = np.asarray(bng_x[1, :n_state_probe], dtype=float)
            aa1 = np.asarray(am_x[1, :n_state_probe], dtype=float)
            d0 = np.maximum(np.maximum(np.abs(bb0), np.abs(aa0)), S9_XVAL_TRAJ_ATOL)
            d1 = np.maximum(np.maximum(np.abs(bb1), np.abs(aa1)), S9_XVAL_TRAJ_ATOL)
            re0 = float(np.median(np.abs(bb0 - aa0) / d0))
            re1 = float(np.median(np.abs(bb1 - aa1) / d1))
            if re0 > 10.0 * max(S9_XVAL_TRAJ_RTOL, 1e-9) and re1 <= S9_XVAL_TRAJ_RTOL:
                t0_drop = 1
                bng_x = bng_x[1:]
                am_x = am_x[1:]
                bng_sx = bng_sx[1:]
                am_sx = am_sx[1:]
                nt = min(bng_x.shape[0], am_x.shape[0], bng_sx.shape[0], am_sx.shape[0])

    bng_state_ix = {n: i for i, n in enumerate(bng_states)}
    am_state_ix = {n: i for i, n in enumerate(am_states)}
    am_state_name_ix = {n: i for i, n in enumerate(am_state_names)}
    bng_state_ix_norm = {_normalize_id(n): i for i, n in enumerate(bng_states)}
    am_state_ix_norm = {_normalize_id(n): i for i, n in enumerate(am_states)}
    am_state_name_ix_norm = {_normalize_id(n): i for i, n in enumerate(am_state_names)}
    bng_xk = [_xval_species_key(n) for n in bng_states]
    am_xk = [_xval_species_key(n) for n in am_states]
    bng_state_ix_xval = (
        {bng_xk[i]: i for i in range(len(bng_xk))} if len(set(bng_xk)) == len(bng_xk) else {}
    )
    am_state_ix_xval = (
        {am_xk[i]: i for i in range(len(am_xk))} if len(set(am_xk)) == len(am_xk) else {}
    )

    state_match = None
    common_states = []
    bs = None
    as_ = None
    tier_maps = [
        ("name", bng_state_ix, am_state_ix),
        ("state_name", bng_state_ix, am_state_name_ix),
        ("normalized_name", bng_state_ix_norm, am_state_ix_norm),
        ("normalized_state_name", bng_state_ix_norm, am_state_name_ix_norm),
    ]
    if bng_state_ix_xval and am_state_ix_xval:
        tier_maps.append(("xval_species_key", bng_state_ix_xval, am_state_ix_xval))
    for sm, b_ix, a_ix in tier_maps:
        inter = sorted(set(b_ix.keys()) & set(a_ix.keys()))
        if inter:
            state_match = sm
            common_states = inter
            bs = [b_ix[k] for k in inter]
            as_ = [a_ix[k] for k in inter]
            break

    if state_match is None:
        if not S9_XVAL_ALLOW_ASSIGNMENT:
            return {
                "ok": False,
                "category": "unalignable_state_mapping",
                "error": (
                    "no common states by name, normalized_name, or xval_species_key "
                    "(BNGL ``Species()`` vs SBML id)"
                ),
                "diagnostic": {
                    "bng_states_head": list(bng_states[:12]),
                    "am_states_head": list(am_states[:12]),
                    "n_bng_states": len(bng_states),
                    "n_am_states": len(am_states),
                },
            }
        if linear_sum_assignment is None:
            return {
                "ok": False,
                "category": "assignment_dependency_missing",
                "error": "scipy.optimize.linear_sum_assignment unavailable for assignment fallback",
            }
        n_state = min(bng_x.shape[1], am_x.shape[1], bng_sx.shape[1], am_sx.shape[1])
        if n_state <= 0:
            return {
                "ok": False,
                "category": "unalignable_state_mapping",
                "error": "no alignable states",
            }
        aa = bng_x[:nt, :n_state]
        bb = am_x[:nt, :n_state]
        rel3 = np.abs(aa[:, :, None] - bb[:, None, :]) / np.maximum(np.abs(bb[:, None, :]), 1e-12)
        cost = np.median(rel3, axis=0)
        li, ri = linear_sum_assignment(cost)
        bs = [int(i) for i in li]
        as_ = [int(i) for i in ri]
        common_states = [f"state_{i}" for i in range(len(bs))]
        state_match = "trajectory_assignment"

    bng_param_ix = {n: i for i, n in enumerate(bng_params)}
    am_param_ix = {n: i for i, n in enumerate(am_params)}
    bng_param_ix_norm = {_normalize_id(n): i for i, n in enumerate(bng_params)}
    am_param_ix_norm = {_normalize_id(n): i for i, n in enumerate(am_params)}
    common_params = sorted(set(bng_param_ix) & set(am_param_ix))
    param_match = "name"
    if not common_params:
        common_params = sorted(set(bng_param_ix_norm) & set(am_param_ix_norm))
        if common_params:
            bp = [bng_param_ix_norm[p] for p in common_params]
            ap = [am_param_ix_norm[p] for p in common_params]
            param_match = "normalized_name"
        else:
            return {
                "ok": False,
                "category": "unalignable_param_mapping",
                "error": "no common parameters by name/normalized-name",
            }
    else:
        bp = [bng_param_ix[p] for p in common_params]
        ap = [am_param_ix[p] for p in common_params]

    traj_b = bng_x[:nt][:, bs]
    traj_a = am_x[:nt][:, as_]
    traj_stats = _relerr_stats(traj_b, traj_a, atol=S9_XVAL_TRAJ_ATOL)

    sx_b = bng_sx[:nt][:, bs, :][:, :, bp]
    sx_a = am_sx[:nt][:, as_, :][:, :, ap]
    # Effective atol = max(absolute_floor, relative_floor * max|sx|). Cells
    # whose magnitude is below the CVODES noise floor (relative to the largest
    # sensitivity in the dataset) are excluded from driving the symmetric
    # relerr — otherwise opposite-sign float noise on near-zero entries gives
    # relerr ~ 2.0 and dominates the headline. See bench docstring.
    _sens_data_scale = float(
        max(np.abs(sx_b).max() if sx_b.size else 0.0, np.abs(sx_a).max() if sx_a.size else 0.0)
    )
    sens_atol_floor = max(S9_XVAL_SENS_ATOL, S9_XVAL_SENS_ATOL_REL * _sens_data_scale)
    # Windowed local floor: pulls each cell's denom up to a fraction of the
    # neighbouring trajectory's peak so a zero-crossing sample in an
    # otherwise nonzero trajectory doesn't dominate the headline relerr.
    sens_window_peak = _windowed_max_abs(sx_a, sx_b, S9_XVAL_SENS_WIN_RADIUS)
    sens_atol_eff = np.maximum(sens_atol_floor, S9_XVAL_SENS_ATOL_REL_WIN * sens_window_peak)
    sens_stats = _relerr_stats(sx_b, sx_a, atol=sens_atol_eff)
    # Dimensionless normalized sensitivities: (p/x) * dx/dp.
    # This mitigates apparent mismatches driven by unit/scale differences.
    pv_map = amici_meta.get("param_values_by_id") or {}
    pvals = []
    for p in common_params:
        try:
            pvals.append(float(pv_map.get(p, 1.0)))
        except Exception:
            pvals.append(1.0)
    pvals = np.asarray(pvals, dtype=float)
    xb = np.maximum(np.abs(traj_b), 1e-12)[:, :, None]
    xa = np.maximum(np.abs(traj_a), 1e-12)[:, :, None]
    nsx_b = (sx_b * pvals[None, None, :]) / xb
    nsx_a = (sx_a * pvals[None, None, :]) / xa
    sens_norm_stats = _relerr_stats(nsx_b, nsx_a, atol=S9_XVAL_SENS_RTOL)

    # Per-cell diagnostics: which (state, param) pairs drove the worst error.
    # Cheap to always compute (a few argpartitions over arrays already in memory)
    # and very useful for triage when a row reports sens=FAIL with max ~ 1.0.
    traj_denom = np.maximum(np.maximum(np.abs(traj_b), np.abs(traj_a)), S9_XVAL_TRAJ_ATOL)
    traj_per_pt = np.abs(traj_b - traj_a) / traj_denom  # (nt, n_states)
    state_max_re = traj_per_pt.max(axis=0) if traj_per_pt.size else np.array([])
    n_top_states = min(3, state_max_re.size)
    if n_top_states > 0:
        top_state_idx = np.argsort(-state_max_re)[:n_top_states]
        worst_traj_states = [
            {"state": str(common_states[i]), "max_relerr": float(state_max_re[i])}
            for i in top_state_idx
        ]
    else:
        worst_traj_states = []

    sens_denom = np.maximum(np.maximum(np.abs(sx_b), np.abs(sx_a)), sens_atol_eff)
    sens_per_pt = np.where(sens_denom > 0, np.abs(sx_b - sx_a) / sens_denom, 0.0)
    pair_max_re = sens_per_pt.max(axis=0) if sens_per_pt.size else np.zeros((0, 0))
    if pair_max_re.size:
        flat = pair_max_re.flatten()
        n_top_pairs = min(3, flat.size)
        flat_top_idx = np.argsort(-flat)[:n_top_pairs]
        n_p_dim = pair_max_re.shape[1]
        worst_sens_pairs = [
            {
                "state": str(common_states[i // n_p_dim]),
                "param": str(common_params[i % n_p_dim]),
                "max_relerr": float(flat[i]),
            }
            for i in flat_top_idx
        ]
    else:
        worst_sens_pairs = []

    traj_pass = traj_stats["max"] <= S9_XVAL_TRAJ_RTOL
    sens_pass = sens_stats["max"] <= S9_XVAL_SENS_RTOL
    sens_norm_med_pass = sens_norm_stats["med"] <= S9_XVAL_SENS_RTOL
    sens_norm_p95_loose_pass = sens_norm_stats["p95"] <= 1.0
    sens_norm_pass = sens_norm_med_pass and sens_norm_p95_loose_pass
    pass_ok = traj_pass and (sens_pass or sens_norm_pass)
    category = "pass" if pass_ok else "numerical_mismatch"
    terminal_zero_states = []
    if not pass_ok and traj_b.shape[0] > 0:
        b_end = np.asarray(traj_b[-1], dtype=float)
        a_end = np.asarray(traj_a[-1], dtype=float)
        for i, st in enumerate(common_states):
            if abs(a_end[i]) <= 1e-18 and abs(b_end[i]) >= 1e-3:
                terminal_zero_states.append(str(st))
        if len(terminal_zero_states) >= max(3, int(0.2 * max(1, len(common_states)))):
            category = "amici_terminal_zero_mismatch"
    if traj_pass and (not sens_pass) and sens_norm_pass:
        category = "normalized_sensitivity_pass_only"
    return {
        "ok": True,
        "xval_t0_drop_rows": int(t0_drop),
        "pass": bool(pass_ok),
        "traj_pass": bool(traj_pass),
        "sens_pass": bool(sens_pass),
        "sens_norm_pass": bool(sens_norm_pass),
        "sens_norm_med_pass": bool(sens_norm_med_pass),
        "sens_norm_p95_loose_pass": bool(sens_norm_p95_loose_pass),
        "common_states": len(common_states),
        "common_params": len(common_params),
        "state_match": state_match,
        "param_match": param_match,
        "traj_max_re": traj_stats["max"],
        "traj_p95_re": traj_stats["p95"],
        "traj_med_re": traj_stats["med"],
        "sens_max_re": sens_stats["max"],
        "sens_p95_re": sens_stats["p95"],
        "sens_med_re": sens_stats["med"],
        "sens_norm_max_re": sens_norm_stats["max"],
        "sens_norm_p95_re": sens_norm_stats["p95"],
        "sens_norm_med_re": sens_norm_stats["med"],
        "thresholds": {
            "traj_max_rtol": S9_XVAL_TRAJ_RTOL,
            "traj_atol": S9_XVAL_TRAJ_ATOL,
            "sens_max_rtol": S9_XVAL_SENS_RTOL,
            "sens_atol": S9_XVAL_SENS_ATOL,
            "sens_atol_rel": S9_XVAL_SENS_ATOL_REL,
            "sens_atol_rel_win": S9_XVAL_SENS_ATOL_REL_WIN,
            "sens_win_radius": S9_XVAL_SENS_WIN_RADIUS,
            "sens_atol_floor_global": sens_atol_floor,
            "sens_atol_effective_max": float(np.max(sens_atol_eff))
            if sens_atol_eff.size
            else None,
            "sens_data_scale": _sens_data_scale,
            "sens_norm_med_rtol": S9_XVAL_SENS_RTOL,
            "sens_norm_p95_loose_rtol": 1.0,
        },
        "terminal_zero_states": terminal_zero_states[:25],
        "worst_traj_states": worst_traj_states,
        "worst_sens_pairs": worst_sens_pairs,
        "category": category,
    }


# ── BNGsim serial (all-at-once) ──────────────────────────────────────────


def bench_bngsim_serial(
    net_path,
    t_end,
    n_points,
    all_params,
    *,
    use_codegen=False,
    n_runs=None,
    sensitivity_method="staggered",
    param_overrides=None,
):
    if n_runs is None:
        n_runs = _RUNS
    """Serial all-at-once CVODES sensitivity (simulation-only timing).

    ``sensitivity_method`` selects the CVODES corrector strategy used inside
    the single coupled state + sensitivity ODE solve. See
    ``bngsim.Simulator``'s docstring for what 'staggered' vs 'simultaneous'
    actually do; in short, 'simultaneous' matches AMICI's default and
    'staggered' matches CVODES' / BNGsim's default.
    """
    import bngsim

    # Build once, then time only repeated solve calls (AMICI parity).
    model = bngsim.Model.from_net(str(net_path))
    if param_overrides:
        _apply_param_overrides(model, param_overrides)
    sim = bngsim.Simulator(
        model,
        method="ode",
        sensitivity_params=all_params,
        sensitivity_method=sensitivity_method,
        codegen=bool(use_codegen),
        net_path=(str(net_path) if use_codegen else ""),
    )

    times = []
    for _ in range(n_runs + _WARMUP):
        # CVODES runs mutate the underlying Model species concentrations to the
        # terminal state (see CvodeSimulator write-back). Warmup iterations must
        # therefore restore the original ``.net`` initial conditions before each
        # timed solve; otherwise the first output row (t=0) can reflect stale
        # state and corrupt AMICI cross-validation.
        model.reset()
        t0 = time.perf_counter()
        result = sim.run(
            t_span=(0, t_end),
            n_points=n_points,
            max_steps=SENS_MAX_STEPS,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    timed = times[_WARMUP:]
    return median(timed), result


# ── BNGsim sharded (parallel) ────────────────────────────────────────────


def bench_bngsim_sharded(
    net_path,
    t_end,
    n_points,
    all_params,
    chunk_size,
    n_workers,
    *,
    use_codegen=False,
    n_runs=None,
    sensitivity_method="staggered",
    param_overrides=None,
):
    if n_runs is None:
        n_runs = _RUNS
    """Parallel chunked sensitivity with simulation-only timing.

    Each parameter chunk runs a fresh CVODES forward-sensitivity job in
    a worker thread. ``sensitivity_method`` is propagated to every chunk
    so all parallel jobs use the same corrector strategy ('staggered'
    vs 'simultaneous').
    """
    import bngsim

    # Build once, then time only chunked sensitivity solve path.
    model = bngsim.Model.from_net(str(net_path))
    if param_overrides:
        _apply_param_overrides(model, param_overrides)
    sim = bngsim.Simulator(
        model,
        method="ode",
        sensitivity_method=sensitivity_method,
        codegen=bool(use_codegen),
        net_path=(str(net_path) if use_codegen else ""),
    )

    times = []
    for _ in range(n_runs + _WARMUP):
        model.reset()
        t0 = time.perf_counter()
        result = sim.compute_all_sensitivities(
            t_span=(0, t_end),
            n_points=n_points,
            params=all_params,
            chunk_size=chunk_size,
            n_workers=n_workers,
            max_steps=SENS_MAX_STEPS,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    timed = times[_WARMUP:]
    return median(timed), result


def bench_bngsim_sharded_with_fallback(
    net_path,
    t_end,
    n_points,
    all_params,
    n_workers,
    primary_chunk_size,
    use_codegen=False,
    n_runs=None,
    sensitivity_method="staggered",
    param_overrides=None,
):
    if n_runs is None:
        n_runs = _RUNS
    """Run sharded sensitivities with a stability fallback.

    Primary path uses configured chunk size. If a model/chunk is numerically
    fragile, retry with smaller chunk sizes to avoid dropping the whole row.
    ``sensitivity_method`` is forwarded to every retry attempt unchanged.
    Returns (time_s, result, used_chunk_size).
    """
    fallback_sizes = []
    if primary_chunk_size > 2:
        fallback_sizes.append(2)
    fallback_sizes.append(1)
    chunk_try = [primary_chunk_size] + [c for c in fallback_sizes if c < primary_chunk_size]
    last_err = None
    for csize in chunk_try:
        try:
            t, r = bench_bngsim_sharded(
                net_path,
                t_end,
                n_points,
                all_params,
                chunk_size=csize,
                n_workers=n_workers,
                use_codegen=use_codegen,
                n_runs=n_runs,
                sensitivity_method=sensitivity_method,
                param_overrides=param_overrides,
            )
            return t, r, csize
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err is not None else RuntimeError("all chunk sizes failed")


# ── AMICI forward sensitivity ─────────────────────────────────────────────


def _convert_bngl_to_sbml(bngl_path, output_dir):
    """Convert .bngl to .xml via BNG2.pl writeSBML() robustly."""
    bngl_path = Path(bngl_path)
    stem = bngl_path.stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"sens_sbml_{stem}_") as tmpdir:
        tmp = Path(tmpdir)
        local_bngl = tmp / bngl_path.name
        local_bngl.write_text(bngl_path.read_text())
        for tfun in bngl_path.parent.glob("*.tfun"):
            with contextlib.suppress(Exception):
                (tmp / tfun.name).write_text(tfun.read_text())

        src = local_bngl.read_text()
        action_pat = re.compile(
            r"^\s*(generate_network|simulate|simulate_nf|writeXML|writeNetwork|"
            r"writeSBML|writeMDL|resetConcentrations|resetParameters|"
            r"saveConcentrations|setConcentration|setParameter|"
            r"parameter_scan|bifurcate|begin\s+actions|end\s+actions)\b.*$"
        )
        cleaned = [ln for ln in src.splitlines() if not action_pat.match(ln.strip())]
        local_bngl.write_text(
            "\n".join(cleaned).rstrip() + "\n\ngenerate_network({overwrite=>1})\nwriteSBML()\n"
        )

        try:
            subprocess.run(
                ["perl", str(BNG2_PL), str(local_bngl)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(tmp),
                check=False,
            )
        except Exception:
            return None

        candidates = [
            tmp / f"{stem}_sbml.xml",
            tmp / f"{stem}.xml",
            tmp / f"{stem}.sbml",
        ]
        sbml_src = next((p for p in candidates if p.exists()), None)
        if sbml_src is None:
            return None

        sbml_out = out_dir / f"{stem}.xml"
        sbml_out.write_text(sbml_src.read_text())
        return str(sbml_out)


def amici_setup_forward_sensitivity(sbml_path, build_dir, t_end, n_points, net_path=None):
    """Compile SBML→AMICI and configure forward sensitivities; read free-parameter ids.

    ``build_dir`` must remain on disk while the returned ``model`` is used.

    When ``net_path`` is provided, AMICI's initial state vector is overwritten to match
    the frozen BNGsim ``.net`` initial conditions (see ``seed_amici_initial_state_from_net``).
    """
    amici, amici_sundials = prepare_amici_runtime()
    model_name = Path(sbml_path).stem.replace("-", "_").replace(".", "_")
    importer = amici.SbmlImporter(str(sbml_path))
    importer.sbml2amici(model_name, str(build_dir))

    model_module = amici.import_model_module(model_name, str(build_dir))
    model = (
        model_module.getModel() if hasattr(model_module, "getModel") else model_module.get_model()
    )
    solver = model.getSolver() if hasattr(model, "getSolver") else model.create_solver()

    tspan = np.linspace(0, t_end, n_points)
    if hasattr(model, "setTimepoints"):
        model.setTimepoints(tspan)
    else:
        model.set_timepoints(tspan)

    if hasattr(solver, "setAbsoluteTolerance"):
        solver.setAbsoluteTolerance(1e-12)
        solver.setRelativeTolerance(1e-8)
        solver.setSensitivityMethod(amici.SensitivityMethod.forward)
        solver.setSensitivityOrder(amici.SensitivityOrder.first)
    else:
        solver.set_absolute_tolerance(1e-12)
        solver.set_relative_tolerance(1e-8)
        solver.set_sensitivity_method(amici_sundials.SensitivityMethod_forward)
        solver.set_sensitivity_order(amici_sundials.SensitivityOrder_first)

    np_attr = getattr(model, "np", None)
    if callable(np_attr):
        n_model_params = int(np_attr())
    elif np_attr is not None:
        n_model_params = int(np_attr)
    else:
        n_model_params = int(getattr(model, "n_parameters", 0))
    if hasattr(model, "setParameterScale"):
        model.setParameterScale([amici.ParameterScaling.none] * n_model_params)
    else:
        model.set_parameter_scale([amici_sundials.ParameterScaling_none] * n_model_params)

    if hasattr(model, "get_free_parameter_ids"):
        param_ids = list(model.get_free_parameter_ids())
    elif hasattr(model, "getParameterIds"):
        param_ids = list(model.getParameterIds())
    elif hasattr(model, "get_parameter_ids"):
        param_ids = list(model.get_parameter_ids())
    else:
        param_ids = []

    state_ids = (
        list(model.getStateIds())
        if hasattr(model, "getStateIds")
        else list(model.get_state_ids_solver())
    )
    if hasattr(model, "getStateNames"):
        state_names = list(model.getStateNames())
    elif hasattr(model, "get_state_names_solver"):
        state_names = list(model.get_state_names_solver())
    else:
        state_names = []

    ic_info = None
    param_sync_info = None
    alignment_check = None
    if net_path is not None:
        try:
            ic_info = seed_amici_initial_state_from_net(model, Path(net_path))
        except Exception as e:
            ic_info = {"ok": False, "reason": f"seed_exception:{e}"}
        try:
            param_sync_info = seed_amici_parameters_from_net(model, Path(net_path))
        except Exception as e:
            param_sync_info = {"ok": False, "reason": f"seed_exception:{e}"}
        try:
            alignment_check = _verify_alignment(model, Path(net_path))
        except Exception as e:
            alignment_check = {"ok": False, "reason": f"verify_exception:{e}"}

    return {
        "amici": amici,
        "amici_sundials": amici_sundials,
        "model": model,
        "solver": solver,
        "param_ids": param_ids,
        "state_ids": state_ids,
        "state_names": state_names,
        "n_model_params": n_model_params,
        "sens_mode": "forward",
        "ic_from_net": ic_info,
        "param_sync_from_net": param_sync_info,
        "alignment_check": alignment_check,
    }


def _set_amici_internal_sens_method(solver, amici, amici_sundials, method):
    """Pin AMICI's CVODES internal corrector method.

    ``method`` is ``"simultaneous"`` (CV_SIMULTANEOUS) or ``"staggered"``
    (CV_STAGGERED). Both still integrate state + sensitivities as one
    coupled extended ODE in a single CVODES pass; this just selects how
    each step's nonlinear solve is structured. AMICI's compiled-in default
    is ``simultaneous``; CVODES' / BNGsim's default is ``staggered``.
    See bench script docstring for plain-English notes.
    """
    if method not in ("simultaneous", "staggered"):
        raise ValueError(f"Unknown internal sensitivity method: {method!r}")

    # AMICI's Python binding is mostly snake_case in current builds, but
    # the camelCase fallbacks remain valid on older releases. Try both,
    # mirroring the rest of this file (e.g. set_sensitivity_method below).
    if hasattr(solver, "setInternalSensitivityMethod") and hasattr(
        amici, "InternalSensitivityMethod"
    ):
        enum = getattr(amici.InternalSensitivityMethod, method)
        solver.setInternalSensitivityMethod(enum)
        return method

    enum_name = f"InternalSensitivityMethod_{method}"
    enum = getattr(amici_sundials, enum_name)
    solver.set_internal_sensitivity_method(enum)
    return method


def amici_run_forward_sensitivity_timed(model, solver, amici, amici_sundials, n_runs=None):
    if n_runs is None:
        n_runs = _RUNS
    """Median wall time of forward sensitivity simulations only (compile excluded)."""
    times = []
    rdata = None
    for _ in range(n_runs + _WARMUP):
        t0 = time.perf_counter()
        if hasattr(amici, "runAmiciSimulation"):
            rdata = amici.runAmiciSimulation(model, solver)
        elif hasattr(amici_sundials, "run_simulation"):
            rdata = amici_sundials.run_simulation(model, solver)
        elif hasattr(model, "simulate"):
            try:
                rdata = model.simulate(solver)
            except TypeError:
                rdata = model.simulate()
        else:
            raise RuntimeError("No AMICI simulation entrypoint found")
        times.append(time.perf_counter() - t0)
    timed = times[_WARMUP:]
    return median(timed), rdata


def amici_pack_sensitivity_metadata(rdata, model, param_ids, state_ids):
    """Build sx/x arrays and shape metadata for cross-validation."""
    sx_raw = np.asarray(rdata.sx)
    x = np.asarray(rdata.x)
    if sx_raw.size == 0 or not np.isfinite(sx_raw).all():
        raise RuntimeError("AMICI forward sensitivity array empty or non-finite")
    sx_arr = sx_raw
    if sx_arr.ndim == 3 and len(state_ids) > 0 and len(param_ids) > 0:
        if sx_arr.shape[1] == len(param_ids) and sx_arr.shape[2] == len(state_ids):
            sx_arr = np.transpose(sx_arr, (0, 2, 1))
        elif sx_arr.shape[1] == len(state_ids) and sx_arr.shape[2] == len(param_ids):
            pass
        elif sx_arr.shape[1] == len(param_ids):
            sx_arr = np.transpose(sx_arr, (0, 2, 1))
    if sx_arr.ndim == 3:
        n_sens = int(sx_arr.shape[2])
        if len(param_ids) > n_sens:
            param_ids = param_ids[:n_sens]
        elif len(param_ids) < n_sens:
            param_ids = param_ids + [f"p{i}" for i in range(len(param_ids), n_sens)]
    _all_pids, _all_pvs, param_values_by_id = _amici_all_parameter_ids_values(model)
    return {
        "sx": sx_arr,
        "x": x,
        "state_ids": state_ids,
        "state_names": (
            list(model.getStateNames())
            if hasattr(model, "getStateNames")
            else (
                list(model.get_state_names_solver())
                if hasattr(model, "get_state_names_solver")
                else []
            )
        ),
        "param_ids": param_ids,
        "param_values_by_id": param_values_by_id,
        "sens_mode": "forward",
        "sx_shape": list(sx_arr.shape),
    }


def bench_amici_sensitivity(sbml_path, t_end, n_points, n_runs=_RUNS, net_path=None):
    """AMICI forward sensitivity (standalone; compiles a private build dir)."""
    with tempfile.TemporaryDirectory(prefix="amici_sens_") as tmpdir:
        bundle = amici_setup_forward_sensitivity(
            sbml_path, tmpdir, t_end, n_points, net_path=net_path
        )
        t_med, rdata = amici_run_forward_sensitivity_timed(
            bundle["model"],
            bundle["solver"],
            bundle["amici"],
            bundle["amici_sundials"],
            n_runs=n_runs,
        )
        meta = amici_pack_sensitivity_metadata(
            rdata,
            bundle["model"],
            list(bundle["param_ids"]),
            list(bundle["state_ids"]),
        )
        n_sens = len(meta["param_ids"])
        return t_med, n_sens, meta


def _per_method_bucket(model_result, method):
    """Return (creating if needed) the per-method results dict inside ``model_result``."""
    pm = model_result.setdefault("per_method", {})
    return pm.setdefault(method, {"sharded": {}})


def run_bngsim_serial_for_method(
    net_path,
    t_end,
    n_pts,
    bench_params,
    n_params,
    model_result,
    method,
    *,
    use_codegen=False,
    param_overrides=None,
):
    """BNGsim serial CVODES forward sensitivity for one corrector method.

    Writes timings/errors into ``model_result["per_method"][method]`` and
    returns the BNGsim ``Result`` (or ``None`` on skip/fail) for downstream
    cross-validation against the matching AMICI run.
    """
    bucket = _per_method_bucket(model_result, method)
    res_serial = None

    try:
        if n_params <= S9_SERIAL_MAX_PARAMS:
            print(
                f"  BNGsim serial ({method}): in progress (CVODES forward sensitivities for "
                f"{n_params} parameters; no per-step logs; "
                f"codegen={'on' if use_codegen else 'off'})",
                flush=True,
            )
            t_serial, res_serial = bench_bngsim_serial(
                net_path,
                t_end,
                n_pts,
                bench_params,
                use_codegen=use_codegen,
                sensitivity_method=method,
                param_overrides=param_overrides,
            )
            bucket["bngsim_serial_ms"] = t_serial * 1000
            print(f"  BNGsim serial ({method}):    {t_serial * 1000:>10.1f} ms")
        else:
            bucket["bngsim_serial_ms"] = None
            bucket["bngsim_serial_skip_reason"] = f"Np={n_params}>{S9_SERIAL_MAX_PARAMS}"
            print(f"  BNGsim serial ({method}):    SKIP (Np={n_params}>{S9_SERIAL_MAX_PARAMS})")
    except Exception as e:
        bucket["bngsim_serial_ms"] = None
        bucket["bngsim_serial_error"] = str(e)[:200]
        res_serial = None
        print(f"  BNGsim serial ({method}):    FAIL ({str(e)[:80]})")

    return res_serial


def run_amici_for_method(bundle, method, model_result):
    """Time AMICI forward sensitivity with the specified internal corrector.

    Pins ``solver.set_internal_sensitivity_method(...)`` to ``method``,
    times the simulation block, packs the sx/x metadata for cross-validation,
    and writes timings into ``model_result["per_method"][method]``.

    Returns the packed AMICI metadata dict, or ``None`` on failure.
    """
    bucket = _per_method_bucket(model_result, method)
    try:
        _set_amici_internal_sens_method(
            bundle["solver"],
            bundle["amici"],
            bundle["amici_sundials"],
            method,
        )
    except Exception as e:
        bucket["amici_ms"] = None
        bucket["amici_error"] = f"set_internal_sensitivity_method({method}) failed: {e}"[:200]
        print(f"  AMICI fwd sens ({method}):  FAIL ({str(e)[:80]})")
        return None

    try:
        print(
            f"  AMICI fwd sens ({method}): in progress (simulation-only timing; "
            "compile already done; pinned internal method)",
            flush=True,
        )
        t_amici, rdata = amici_run_forward_sensitivity_timed(
            bundle["model"],
            bundle["solver"],
            bundle["amici"],
            bundle["amici_sundials"],
            n_runs=_RUNS,
        )
        amici_meta = amici_pack_sensitivity_metadata(
            rdata,
            bundle["model"],
            list(bundle["param_ids"]),
            list(bundle["state_ids"]),
        )
        bucket["amici_ms"] = t_amici * 1000
        bucket["amici_n_params"] = len(bundle["param_ids"])
        bucket["amici_sens_mode"] = amici_meta.get("sens_mode", "forward")
        bucket["amici_internal_method"] = method
        bucket["amici_sx_shape"] = amici_meta.get("sx_shape")
        print(
            f"  AMICI fwd sens ({method}):  {t_amici * 1000:>10.1f} ms "
            f"({len(bundle['param_ids'])} sens. params)"
        )
        return amici_meta
    except Exception as e:
        bucket["amici_ms"] = None
        bucket["amici_error"] = str(e)[:200]
        print(f"  AMICI fwd sens ({method}):  FAIL ({str(e)[:80]})")
        return None


def xval_for_method(bng_result, amici_meta, method, model_result):
    """Cross-validate one BNGsim ↔ AMICI pair (same internal method) and print."""
    bucket = _per_method_bucket(model_result, method)
    if bng_result is None or amici_meta is None:
        bucket["xval"] = {
            "ok": False,
            "category": "missing_results",
            "error": "BNGsim or AMICI result unavailable for this method",
        }
        return

    xval = _sens_xval(bng_result, amici_meta)
    bucket["xval"] = xval

    if not xval.get("ok"):
        print(
            f"  XVAL ({method}):     SKIP "
            f"({xval.get('category', 'unknown')}: {xval.get('error', 'unknown')})"
        )
        return

    ok_txt = "PASS" if xval.get("pass") else "FAIL"
    traj_txt = "PASS" if xval.get("traj_pass") else "FAIL"
    sens_txt = "PASS" if xval.get("sens_pass") else "FAIL"
    sensn_txt = "PASS" if xval.get("sens_norm_pass") else "FAIL"
    cat = xval.get("category")
    cat_note = f" category={cat}" if cat and cat != "pass" else ""
    print(
        f"  XVAL ({method}):     "
        f"{ok_txt}  states={xval['common_states']} "
        f"params={xval['common_params']} "
        f"match=({xval['state_match']},{xval['param_match']}) "
        f"traj={traj_txt}[max={xval['traj_max_re']:.2e}, "
        f"p95={xval['traj_p95_re']:.2e}, med={xval['traj_med_re']:.2e}] "
        f"sens={sens_txt}[max={xval['sens_max_re']:.2e}, "
        f"p95={xval['sens_p95_re']:.2e}, med={xval['sens_med_re']:.2e}] "
        f"sens_norm={sensn_txt}[max={xval['sens_norm_max_re']:.2e}, "
        f"p95={xval['sens_norm_p95_re']:.2e}, med={xval['sens_norm_med_re']:.2e}]"
        f"{cat_note}"
    )
    if not xval.get("sens_pass", True) and xval.get("worst_sens_pairs"):
        p0 = xval["worst_sens_pairs"][0]
        t0 = (xval.get("worst_traj_states") or [{}])[0]
        print(
            f"  XVAL ({method}) diag: worst sens pair "
            f"state={p0['state']!r} param={p0['param']!r} "
            f"max_relerr={p0['max_relerr']:.2e}"
            + (
                f"; worst traj state={t0.get('state')!r} max_relerr={t0.get('max_relerr', 0):.2e}"
                if t0
                else ""
            )
        )


def run_bngsim_sharded_for_method(
    net_path,
    t_end,
    n_pts,
    bench_params,
    worker_counts,
    args,
    model_result,
    method,
    *,
    use_codegen=False,
    param_overrides=None,
):
    """BNGsim sharded CVODES forward sensitivity sweep for one corrector method.

    Loops over ``worker_counts`` and writes per-nw timings into
    ``model_result["per_method"][method]["sharded"][str(nw)]``. Returns the
    first successful chunked ``Result`` (used as a fallback xval source if
    the serial run was skipped/failed for this method).
    """
    bucket = _per_method_bucket(model_result, method)
    bucket.setdefault("sharded", {})
    res_sharded_first = None

    if args.no_sharded:
        print(f"  BNGsim sharded ({method}): SKIP (--no-sharded)")
        return res_sharded_first

    for nw in worker_counts:
        try:
            print(
                f"  BNGsim sharded ({method}) nw={nw}: in progress...",
                flush=True,
            )
            t_shard, res_shard, used_chunk_size = bench_bngsim_sharded_with_fallback(
                net_path,
                t_end,
                n_pts,
                bench_params,
                n_workers=nw,
                primary_chunk_size=S9_CHUNK_SIZE,
                use_codegen=use_codegen,
                sensitivity_method=method,
                param_overrides=param_overrides,
            )
            if res_sharded_first is None:
                res_sharded_first = res_shard

            cpu_time = nw * t_shard

            bucket["sharded"][str(nw)] = {
                "wall_ms": t_shard * 1000,
                "cpu_ms": cpu_time * 1000,
                "chunk_size": used_chunk_size,
            }
            note = (
                ""
                if used_chunk_size == S9_CHUNK_SIZE
                else f" (fallback chunk_size={used_chunk_size})"
            )
            print(
                f"  BNGsim ({method}) nw={nw:<3d}: "
                f"wall={t_shard * 1000:>8.1f} ms  "
                f"cpu≈{cpu_time * 1000:>8.1f} ms{note}"
            )
        except Exception as e:
            bucket["sharded"][str(nw)] = {
                "wall_ms": None,
                "cpu_ms": None,
                "error": str(e)[:200],
            }
            print(f"  BNGsim ({method}) nw={nw}: FAIL ({str(e)[:80]})")

    return res_sharded_first


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    global _WARMUP, _RUNS

    ap = argparse.ArgumentParser(
        description="Forward sensitivity benchmark (Supplementary Table S9)"
    )
    ap.add_argument("--quick", action="store_true", help="Small models only (≤37 sp)")
    ap.add_argument("--model", type=str, default="", help="Run only this model")
    ap.add_argument(
        "--no-sharded",
        action="store_true",
        help="Skip parallel chunked BNGsim (serial + AMICI only)",
    )
    ap.add_argument("--warmup", type=int, default=_WARMUP)
    ap.add_argument("--runs", type=int, default=_RUNS)
    ap.add_argument(
        "--methods",
        type=str,
        default="simultaneous,staggered",
        help=(
            "Comma-separated CVODES corrector methods to run "
            "(any subset of 'simultaneous,staggered'). Both engines are "
            "pinned to the same method per pair for apples-to-apples timing. "
            "Default runs both."
        ),
    )
    ap.add_argument(
        "--mode",
        choices=("correctness", "timing", "both"),
        default="both",
        help=(
            "Which gates to run (default: both). 'correctness' runs the "
            "BNGsim<->AMICI alignment + trajectory/sensitivity "
            "cross-validation and skips the sharded timing sweep (which is "
            "pure timing); 'timing'/'both' run the full timed comparison."
        ),
    )
    add_effort_arg(ap)
    args = ap.parse_args()

    # The sharded sweep is a pure-timing reference (no correctness output),
    # so correctness mode skips it.
    if args.mode == "correctness":
        args.no_sharded = True

    _WARMUP = args.warmup
    _RUNS = args.runs

    methods = []
    for tok in str(args.methods).split(","):
        tok = tok.strip().lower()
        if tok in ("simultaneous", "staggered") and tok not in methods:
            methods.append(tok)
    if not methods:
        ap.error(
            "No valid methods selected. --methods must list one or both of "
            "'simultaneous' / 'staggered'."
        )

    info = get_machine_info()
    suite = load_suite(SUITE_ODE)
    cpu_count = os.cpu_count() or 1
    raw_workers = []
    for tok in str(S9_WORKERS_ENV).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            w = int(tok)
            if w > 0:
                raw_workers.append(w)
        except Exception:
            continue
    worker_counts = [w for w in raw_workers if w <= cpu_count]
    if not worker_counts:
        worker_counts = [min(6, cpu_count)]

    print("=" * 100)
    print("  Forward Sensitivity Benchmark (Table S9)")
    print("=" * 100)
    print(f"  CPU count:         {cpu_count}")
    print(f"  Worker counts:     {worker_counts}")
    print(f"  Chunk size:        {S9_CHUNK_SIZE}")
    print(f"  Serial max params: {S9_SERIAL_MAX_PARAMS}")
    print(f"  Mode / effort:     {args.mode} / {args.effort}")
    print(f"  Sharded pass:      {not args.no_sharded}")
    print(f"  Methods:           {methods}")
    print(f"  Protocol:          {_WARMUP}w + {_RUNS}t, median")
    print()
    print("  CVODES corrector methods (both still solve state + sensitivities")
    print("  as one coupled extended ODE in a single CVODES pass; they differ")
    print("  only in how each step's nonlinear solve is structured):")
    print("    simultaneous (CV_SIMULTANEOUS) — state and all sensitivities")
    print("                                     solved together as one big")
    print("                                     coupled nonlinear system per")
    print("                                     step. AMICI's default.")
    print("    staggered    (CV_STAGGERED)    — state advanced first, then")
    print("                                     sensitivities advanced as a")
    print("                                     separate solve. Two smaller")
    print("                                     solves per step. CVODES' /")
    print("                                     BNGsim's default.")
    print()

    all_results = []

    for entry in suite:
        name = entry["name"]

        if args.model:
            if args.model.lower() not in name.lower():
                continue
        elif name not in TARGET_MODELS:
            continue

        if args.quick and entry.get("species", 0) > 40:
            continue

        # Cumulative --effort tier filter (explicit --model bypasses it).
        if not args.model and not effort_allows(args.effort, MODEL_EFFORT.get(name, "high")):
            continue

        net_path = BENCHMARKS_DIR / entry["net_file"]
        t_end = entry["t_end"]
        n_pts = entry.get("n_steps", 200)

        if not net_path.exists():
            print(f"  SKIP {name}: {net_path} not found")
            continue

        all_params_net = get_param_names(net_path)
        n_params_net = len(all_params_net)
        n_sp = entry.get("species", "?")
        bngl_path = _find_bngl(name)

        model_result = {
            "name": name,
            "n_species": n_sp,
            "n_params_net": n_params_net,
            "methods_run": list(methods),
        }

        if bngl_path is not None and bngl_path.exists():
            with tempfile.TemporaryDirectory(prefix="s10_amici_") as work:
                wd = Path(work)
                pre_sim_ms: dict[str, float | None] = {
                    "bng2pl_writeSBML": None,
                    "amici_compile": None,
                    "bngsim_setup_codegen": None,
                }
                model_result["pre_sim_ms"] = pre_sim_ms
                _t0 = time.perf_counter()
                sbml_path = _convert_bngl_to_sbml(bngl_path, wd)
                pre_sim_ms["bng2pl_writeSBML"] = (time.perf_counter() - _t0) * 1000
                if not sbml_path:
                    model_result["sbml_error"] = "SBML conversion failed"
                    model_result["n_params"] = n_params_net
                    model_result["n_chunks"] = math.ceil(n_params_net / max(1, S9_CHUNK_SIZE))
                    print(f"\n{'─' * 80}")
                    print(f"  {name} ({n_sp} sp, {n_params_net} net params)")
                    print(f"{'─' * 80}")
                    print("  SKIP AMICI/BNG paired run: SBML conversion failed")
                    all_results.append(model_result)
                    continue

                # ── (A) IC parameter linkage recovery ──
                # Compute a single (species → IC parameter) link map from the
                # .bngl + .net, then apply it on both sides:
                #   • AMICI: inject <initialAssignment> into the SBML for any
                #     missing entry (recovers `setConcentration`-style links
                #     dropped by the action stripper).
                #   • BNGsim: rewrite the .net so each species line points at
                #     its IC parameter and the parameter's value matches the
                #     post-equilibration literal; restore the param's nominal
                #     after Model.from_net via param_overrides.
                ic_link_map, param_overrides = _collect_ic_param_links(bngl_path, net_path)
                model_result["ic_link_map"] = dict(ic_link_map)
                model_result["ic_link_param_overrides"] = dict(param_overrides)
                bngsim_net_path = net_path
                if ic_link_map:
                    inj_info = _inject_sbml_initial_assignments(sbml_path, ic_link_map)
                    amici_only = inj_info.get("amici_only", [])
                    model_result["sbml_ic_link_injection"] = {
                        "n_injected": len(inj_info.get("injected", [])),
                        "n_already_present": len(inj_info.get("already_present", [])),
                        "n_missing_species": len(inj_info.get("missing_species", [])),
                        "n_missing_param": len(inj_info.get("missing_param", [])),
                        "n_amici_only": len(amici_only),
                        "injected_sample": inj_info.get("injected", [])[:8],
                        "amici_only_sample": amici_only[:8],
                    }
                    model_result["ic_link_mismatches"] = {
                        "amici_only": amici_only,  # links AMICI has but BNGsim doesn't
                    }
                    rewritten = wd / f"{net_path.stem}.ic_linked.net"
                    _rewrite_net_with_ic_links(net_path, ic_link_map, param_overrides, rewritten)
                    if rewritten.exists():
                        bngsim_net_path = rewritten

                try:
                    _t0 = time.perf_counter()
                    bundle = amici_setup_forward_sensitivity(
                        sbml_path,
                        wd / "amici_build",
                        t_end,
                        n_pts,
                        net_path=net_path,
                    )
                    pre_sim_ms["amici_compile"] = (time.perf_counter() - _t0) * 1000
                except Exception as e:
                    model_result["amici_compile_error"] = str(e)[:200]
                    model_result["n_params"] = n_params_net
                    model_result["n_chunks"] = math.ceil(n_params_net / max(1, S9_CHUNK_SIZE))
                    print(f"\n{'─' * 80}")
                    print(f"  {name} ({n_sp} sp, {n_params_net} net params)")
                    print(f"{'─' * 80}")
                    print(f"  SKIP AMICI compile: {str(e)[:120]}")
                    all_results.append(model_result)
                    continue

                model_result["amici_ic_from_net"] = bundle.get("ic_from_net")
                model_result["amici_param_sync_from_net"] = bundle.get("param_sync_from_net")
                model_result["alignment_check"] = bundle.get("alignment_check")

                align = align_bng_params_to_amici(all_params_net, bundle["param_ids"])
                model_result["param_alignment"] = align
                if not align.get("ok"):
                    model_result["n_params"] = n_params_net
                    model_result["n_chunks"] = math.ceil(n_params_net / max(1, S9_CHUNK_SIZE))
                    print(f"\n{'─' * 80}")
                    print(f"  {name} ({n_sp} sp, {n_params_net} net params)")
                    print(f"{'─' * 80}")
                    print(
                        "  SKIP paired benchmark: could not map AMICI free parameters "
                        f"to .net names: {align.get('unmatched_amici_ids', align)}"
                    )
                    all_results.append(model_result)
                    continue

                bench_params = align["bench_params"]
                n_params = len(bench_params)
                n_chunks = math.ceil(n_params / max(1, S9_CHUNK_SIZE))
                use_codegen = _use_bng_codegen_for_model(
                    n_params, n_species=n_sp if isinstance(n_sp, int) else None
                )
                model_result["n_params"] = n_params
                model_result["n_chunks"] = n_chunks
                model_result["bngsim_codegen_enabled"] = bool(use_codegen)
                model_result["bngsim_codegen_mode"] = S10_BNG_CODEGEN_MODE
                model_result["bench_amici_apples_to_apples"] = True
                model_result["amici_n_model_params"] = bundle["n_model_params"]
                model_result["amici_n_sensitivity_params"] = len(bundle["param_ids"])

                # Derived-parameter audit on the BNGsim side: how many
                # bench_params are flagged is_expression=True in the .net
                # (i.e. derived ConstantExpressions like _rateLaw{N})?
                # AMICI's free-parameter list is the source of truth for
                # what's an independent variable in the model, so we
                # don't drop these — but if any survive into AMICI's
                # free list and into bench_params, we want it visible
                # so a future model with derived knobs in AMICI's free
                # set can be flagged in the JSON / terminal rather than
                # silently sneaking through.
                try:
                    import bngsim as _bngsim_audit

                    _audit_model = _bngsim_audit.Model.from_net(str(net_path))
                    _is_expr_by_name = dict(
                        zip(
                            _audit_model._core.param_names,
                            list(_audit_model._core.param_is_expression),
                            strict=False,
                        )
                    )
                    derived_in_bench = [n for n in bench_params if _is_expr_by_name.get(n, False)]
                    align["n_derived_in_bench"] = len(derived_in_bench)
                    align["derived_in_bench_sample"] = derived_in_bench[:12]
                except Exception as e:
                    align["n_derived_in_bench"] = None
                    align["derived_in_bench_error"] = str(e)[:200]

                print(f"\n{'─' * 80}")
                print(
                    f"  {name} ({n_sp} sp, Np={n_params} sensitivity params aligned "
                    f"to AMICI free params; {n_params_net} params in .net)"
                )
                print(f"{'─' * 80}")

                # Defensive alignment line: report whether AMICI's loaded
                # state matches the .net source of truth on both parameters
                # and species ICs. Anything non-zero here means the two
                # engines would integrate slightly different problems.
                _ic = (model_result.get("alignment_check") or {}).get("ic") or {}
                _pa = (model_result.get("alignment_check") or {}).get("params") or {}
                _ic_ok = bool(_ic.get("ok"))
                _pa_ok = bool(_pa.get("ok"))
                _ic_max = _ic.get("max_relerr")
                _pa_max = _pa.get("max_relerr")
                _ic_max_s = f"{_ic_max:.1e}" if isinstance(_ic_max, (int, float)) else "n/a"
                _pa_max_s = f"{_pa_max:.1e}" if isinstance(_pa_max, (int, float)) else "n/a"
                _badge = "OK" if (_ic_ok and _pa_ok) else "WARN"
                print(
                    f"  Alignment ({_badge}):  "
                    f"ICs n={_ic.get('n_common', 0)} max_relerr={_ic_max_s}; "
                    f"params n={_pa.get('n_common', 0)} max_relerr={_pa_max_s} "
                    f"(.net is the source of truth; values pushed into AMICI defensively)"
                )
                if not _ic_ok and _ic.get("diverged_sample"):
                    d0 = _ic["diverged_sample"][0]
                    print(
                        f"  Alignment WARN ic:   first diverged state {d0['name']!r}: "
                        f"net={d0['net']!r} amici={d0['amici']!r} relerr={d0['relerr']:.2e}"
                    )
                if not _pa_ok and _pa.get("diverged_sample"):
                    d0 = _pa["diverged_sample"][0]
                    print(
                        f"  Alignment WARN par:  first diverged param {d0['name']!r}: "
                        f"net={d0['net']!r} amici={d0['amici']!r} relerr={d0['relerr']:.2e}"
                    )

                # Derived-parameter audit line: BNGsim's .net flags some
                # parameters as is_expression=True (derived ConstantExpressions
                # whose values track other parameters, e.g. _rateLaw{N}). For a
                # fair AMICI<->BNGsim comparison we let AMICI's
                # get_free_parameter_ids() decide what's an independent
                # variable; if any of those happen to be derived in BNGsim,
                # we flag it here so the issue is visible and recorded in
                # JSON rather than silent.
                _n_derived = align.get("n_derived_in_bench")
                if _n_derived is None:
                    print(
                        f"  Independence:        WARN derived-param audit failed "
                        f"({align.get('derived_in_bench_error', 'unknown error')})"
                    )
                elif _n_derived == 0:
                    print(
                        f"  Independence:        OK ({n_params} bench params are all primary "
                        f"in BNGsim — none flagged is_expression=True)"
                    )
                else:
                    sample = align.get("derived_in_bench_sample") or []
                    print(
                        f"  Independence (WARN): {_n_derived}/{n_params} bench params are "
                        f"is_expression=True in BNGsim (derived ConstantExpressions whose "
                        f"values track other parameters); they appear in AMICI's free list "
                        f"but their sensitivities are mathematically less meaningful."
                    )
                    print(f"  Independence WARN:   first derived: {sample[:6]}")

                # One-shot BNGsim setup+codegen timing. Captures the cost of
                # ``Model.from_net`` + ``Simulator(...)`` construction (with
                # codegen .so resolution if codegen=on). The benchmark itself
                # excludes this from the timed medians because it lives
                # outside the ``time.perf_counter()`` block in
                # ``bench_bngsim_serial``; this number lets the supplementary
                # table disclose what's discounted on the BNGsim side.
                try:
                    import bngsim as _bngsim_setup_audit

                    _t0 = time.perf_counter()
                    _audit_m = _bngsim_setup_audit.Model.from_net(str(bngsim_net_path))
                    if param_overrides:
                        _apply_param_overrides(_audit_m, param_overrides)
                    _audit_s = _bngsim_setup_audit.Simulator(
                        _audit_m,
                        method="ode",
                        sensitivity_params=bench_params,
                        sensitivity_method="staggered",
                        codegen=bool(use_codegen),
                        net_path=(str(bngsim_net_path) if use_codegen else ""),
                    )
                    pre_sim_ms["bngsim_setup_codegen"] = (time.perf_counter() - _t0) * 1000
                    del _audit_s, _audit_m, _bngsim_setup_audit
                except Exception as _e:
                    pre_sim_ms["bngsim_setup_codegen"] = None
                    pre_sim_ms.setdefault("bngsim_setup_error", str(_e)[:200])

                def _fmt_pre(v):
                    return f"{v:.1f}ms" if isinstance(v, (int, float)) else "n/a"

                print(
                    f"  Pre-sim cost (excluded from timed medians): "
                    f"BNG2.pl→SBML={_fmt_pre(pre_sim_ms.get('bng2pl_writeSBML'))}; "
                    f"AMICI compile={_fmt_pre(pre_sim_ms.get('amici_compile'))}; "
                    f"BNGsim setup+codegen={_fmt_pre(pre_sim_ms.get('bngsim_setup_codegen'))}"
                )

                # Run each method as a paired AMICI ↔ BNGsim block, with the
                # internal corrector pinned identically on both engines.
                for method in methods:
                    print(f"\n  ── Method: {method} ─────────────")
                    res_serial = run_bngsim_serial_for_method(
                        bngsim_net_path,
                        t_end,
                        n_pts,
                        bench_params,
                        n_params,
                        model_result,
                        method,
                        use_codegen=use_codegen,
                        param_overrides=param_overrides,
                    )
                    res_sharded_first = run_bngsim_sharded_for_method(
                        bngsim_net_path,
                        t_end,
                        n_pts,
                        bench_params,
                        worker_counts,
                        args,
                        model_result,
                        method,
                        use_codegen=use_codegen,
                        param_overrides=param_overrides,
                    )
                    amici_meta = run_amici_for_method(bundle, method, model_result)
                    bng_for_xval = res_serial if res_serial is not None else res_sharded_first
                    xval_for_method(bng_for_xval, amici_meta, method, model_result)
        else:
            bench_params = all_params_net
            n_params = n_params_net
            n_chunks = math.ceil(n_params / max(1, S9_CHUNK_SIZE))
            use_codegen = _use_bng_codegen_for_model(
                n_params, n_species=n_sp if isinstance(n_sp, int) else None
            )
            model_result["n_params"] = n_params
            model_result["n_chunks"] = n_chunks
            model_result["bngsim_codegen_enabled"] = bool(use_codegen)
            model_result["bngsim_codegen_mode"] = S10_BNG_CODEGEN_MODE
            model_result["bench_amici_apples_to_apples"] = False
            print(f"\n{'─' * 80}")
            print(
                f"  {name} ({n_sp} sp, {n_params} params [no .bngl — BNGsim-only row]) "
                f"{n_chunks} chunks)"
            )
            print(f"{'─' * 80}")

            for method in methods:
                print(f"\n  ── Method: {method} ─────────────")
                _ = run_bngsim_serial_for_method(
                    net_path,
                    t_end,
                    n_pts,
                    bench_params,
                    n_params,
                    model_result,
                    method,
                    use_codegen=use_codegen,
                )
                _ = run_bngsim_sharded_for_method(
                    net_path,
                    t_end,
                    n_pts,
                    bench_params,
                    worker_counts,
                    args,
                    model_result,
                    method,
                    use_codegen=use_codegen,
                )
                bucket = _per_method_bucket(model_result, method)
                bucket["amici_ms"] = None
                bucket["amici_error"] = "No .bngl file for SBML conversion"
                print(f"  AMICI fwd sens ({method}):  SKIP (no .bngl)")

        all_results.append(model_result)

    # ── Summary tables ────────────────────────────────────────────────
    # Four tables: serial-sim, serial-stag, sharded-sim, sharded-stag.
    # Each method only prints if it was actually run (--methods filter).
    print(f"\n{'=' * 100}")
    print("  SUMMARY")
    print(f"{'=' * 100}")

    def _fmt_ms(x):
        return f"{x:>8.1f}ms" if isinstance(x, (int, float)) and x is not None else f"{'---':>10}"

    def _traj_tag(xv):
        if not isinstance(xv, dict) or not xv.get("ok"):
            return "—"
        return "PASS" if xv.get("traj_pass") else "FAIL"

    def _sens_tag(xv):
        """sens column: PASS on raw; 'norm' on magnitude-normalized only; FAIL otherwise."""
        if not isinstance(xv, dict) or not xv.get("ok"):
            return "—"
        if xv.get("sens_pass"):
            return "PASS"
        if xv.get("sens_norm_pass"):
            return "norm"
        return "FAIL"

    def _cg_tag(r):
        return "+" if bool(r.get("bngsim_codegen_enabled")) else "-"

    cg_models = [r["name"] for r in all_results if r.get("bngsim_codegen_enabled")]

    for method in methods:
        title = (
            "Serial CVODES forward sensitivity — both engines pinned to "
            f"{method} ({'CV_SIMULTANEOUS' if method == 'simultaneous' else 'CV_STAGGERED'})"
        )
        print(f"\n  {title}")
        print("  " + "-" * len(title))
        hdr = (
            f"  {'Model':<22} {'Sp':>5} {'Np':>4} {'cg':>3} "
            f"{'AMICI':>10}  {'BNGsim serial':>14}  {'traj':>5} {'sens':>5}"
        )
        print(hdr)
        for r in all_results:
            pm = (r.get("per_method") or {}).get(method) or {}
            amici_ms = pm.get("amici_ms")
            bng_ms = pm.get("bngsim_serial_ms")
            xv = pm.get("xval")
            row = (
                f"  {r['name']:<22} {r.get('n_species', '?'):>5} "
                f"{r.get('n_params', '?'):>4} {_cg_tag(r):>3} "
                f"{_fmt_ms(amici_ms):>10}  "
                f"{_fmt_ms(bng_ms):>14}  "
                f"{_traj_tag(xv):>5} {_sens_tag(xv):>5}"
            )
            print(row)

    if not args.no_sharded:
        for method in methods:
            title = (
                f"BNGsim sharded — {method} corrector "
                f"({'CV_SIMULTANEOUS' if method == 'simultaneous' else 'CV_STAGGERED'})"
            )
            print(f"\n  {title}")
            print("  " + "-" * len(title))
            hdr = f"  {'Model':<22} {'Sp':>5} {'Np':>4} {'cg':>3}"
            for nw in worker_counts:
                hdr += f"  {'nw=' + str(nw):>10}"
            print(hdr)
            for r in all_results:
                pm = (r.get("per_method") or {}).get(method) or {}
                shmap = pm.get("sharded") or {}
                row = (
                    f"  {r['name']:<22} {r.get('n_species', '?'):>5} "
                    f"{r.get('n_params', '?'):>4} {_cg_tag(r):>3}"
                )
                for nw in worker_counts:
                    sh = shmap.get(str(nw), {})
                    w = sh.get("wall_ms")
                    row += f"  {_fmt_ms(w)}"
                print(row)

    # Footnote / legend explaining the cg, traj, sens columns.
    print()
    print("  Legend:")
    print("    cg   = '+' when BNGsim used a generated C ODE/Jacobian .so for this model;")
    print("           '-' when BNGsim used the interpreted in-memory expression evaluator.")
    print(
        f"           Auto policy: codegen enabled when the coupled system size "
        f"n_species*(Np+1) >= {S10_BNG_CODEGEN_MIN_EFFDIM} "
        f"(override via S10_BNG_CODEGEN_MODE = always|never|auto)."
    )
    if cg_models:
        print(f"           Models with codegen on this run: {cg_models}")
    else:
        print("           No models triggered codegen on this run.")
    print(
        f"    traj = PASS/FAIL on BNGsim<->AMICI species trajectory cross-validation "
        f"(rtol={S9_XVAL_TRAJ_RTOL}, atol={S9_XVAL_TRAJ_ATOL})."
    )
    print("    sens = PASS on raw forward sensitivities; 'norm' when only the")
    print("           parameter/state-magnitude-normalized check passes (typically")
    print("           when raw sensitivities are dominated by float noise on")
    print("           near-zero entries); FAIL otherwise.")

    # Save
    output = {
        "machine_info": info,
        "mode": args.mode,
        "effort": args.effort,
        "protocol": {
            "warmup": _WARMUP,
            "runs": _RUNS,
            "primary_comparison": (
                "bngsim_cvodes_forward_serial_vs_amici_forward, paired per "
                "internal corrector method (simultaneous and staggered)"
            ),
            "methods_run": list(methods),
            "chunk_size": S9_CHUNK_SIZE,
            "worker_counts": worker_counts,
            "serial_max_params": S9_SERIAL_MAX_PARAMS,
            "no_sharded_run": bool(args.no_sharded),
            "amici_sensitivity_method": "forward",
            "timing_policy": (
                "simulation-only on preloaded models (compile excluded for AMICI). "
                "BNGsim serial/sharded integrate sensitivities only for parameters that "
                "match AMICI free-parameter ids from the SBML export (same count and order "
                "as AMICI sx columns); fixed/constant SBML parameters are excluded from both sides. "
                "Both engines are pinned to the same internal CVODES corrector method per pair "
                "(set on AMICI via solver.set_internal_sensitivity_method(...) and on BNGsim via "
                "Simulator(sensitivity_method=...))."
            ),
            "alignment_policy": (
                "BNGsim's .net is the cross-engine source of truth. Before timed runs, AMICI's "
                "initial state is overwritten from the .net (seed_amici_initial_state_from_net) "
                "and AMICI's parameter values are overwritten from the .net "
                "(seed_amici_parameters_from_net). _verify_alignment then re-reads AMICI's "
                "loaded values and records, per model, the count of diverging parameters and ICs "
                "plus the worst observed relative error (model.alignment_check). For the four "
                "target models BNG2.pl emits bit-identical .net and SBML parameter values, so "
                "this is a no-op against zero divergence; the explicit overwrite makes the "
                "comparison robust against future models that use parameter-mutating actions "
                "(setParameter / parameter_scan / etc.) which the BNGL action stripper removes "
                "before SBML export."
            ),
            "bngsim_codegen_footnote_policy": (
                "Models marked cg=+ in the summary table used BNGsim's generated C "
                "ODE/Jacobian .so during timed runs; cg=- used the interpreted in-memory "
                "expression evaluator. Auto threshold: Np >= "
                f"{S10_BNG_CODEGEN_MIN_PARAMS}."
            ),
            "methods": {
                "simultaneous": (
                    "CV_SIMULTANEOUS — at every CVODES step, the state ODE and all "
                    "parameter-sensitivity ODEs are solved together as one big coupled "
                    "nonlinear system. AMICI's compiled-in default."
                ),
                "staggered": (
                    "CV_STAGGERED — at every CVODES step, advance the state first; then, "
                    "with the new state in hand, advance the sensitivity ODEs as a separate "
                    "(linear-in-the-sensitivities) solve. Two smaller nonlinear solves per "
                    "step instead of one big one. CVODES' / BNGsim's default."
                ),
                "note": (
                    "Both modes integrate state + sensitivities as one coupled extended ODE "
                    "in a single CVODES pass; they differ only in how each step's nonlinear "
                    "solve is structured. CVODES has a third mode (CV_STAGGERED1, one "
                    "parameter at a time) that BNGsim does not currently expose."
                ),
            },
            "bngsim_codegen_policy": (
                f"{S10_BNG_CODEGEN_MODE} (auto threshold: Np >= {S10_BNG_CODEGEN_MIN_PARAMS})"
            ),
            "bngsim_codegen_mode": S10_BNG_CODEGEN_MODE,
            "bngsim_codegen_rationale": (
                "AMICI compiles its RHS+Jacobian via CasADi → clang on every model "
                "(no interpreted fallback). The paper run uses "
                "S10_BNG_CODEGEN_MODE=always so BNGsim is also running compiled C "
                "for all models, keeping the engines apples-to-apples (same compiled-"
                "code class on both sides). The interpreted ExprTk path remains the "
                "BNGsim default for ad-hoc use; it can be selected with "
                "S10_BNG_CODEGEN_MODE=never. On Scaff_22_ground with simultaneous "
                "corrector the interpreted path has been observed to wedge "
                "(no per-step progress in the log over tens of minutes); codegen "
                "sidesteps that and lets the four-model paper run complete."
            ),
            "pre_sim_timing_policy": (
                "Pre-simulation cost (BNG2.pl→SBML, AMICI compile, BNGsim "
                "setup+codegen) is excluded from the timed medians on both sides "
                "and reported separately under model.pre_sim_ms."
            ),
            "seed_policy": "deterministic ODE sensitivity (no stochastic seed)",
            "xval_species_alignment": (
                "After exact and normalized-name matching, align BNGL ``Species()`` labels to SBML "
                "state ids via _xval_species_key (strip empty ``()`` from normalized names). "
                "Set S9_XVAL_ALLOW_ASSIGNMENT=1 to fall back to trajectory-based assignment."
            ),
        },
        "models": all_results,
    }
    save_results(output, "forward_sens_results")


if __name__ == "__main__":
    main()
