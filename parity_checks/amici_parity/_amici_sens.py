"""Forward-sensitivity adapters for the bngsim-vs-AMICI parity suite.

The sensitivity sibling of :mod:`_amici_common`. That module compares *state
trajectories* ``x(t)``; this one compares the **forward-sensitivity tensor**
``dx_i(t)/dp_j`` that both engines obtain from a coupled CVODES/CVODES extended
ODE solve. Everything engine-agnostic (the scheduler, the cold/warm integration
taxonomy, the shared tolerances) is imported from :mod:`_amici_common`, which in
turn takes it from ``rr_parity._rr_common`` — so all three suites share ONE
timing taxonomy and ONE comparison protocol. ``align_common`` also comes from
:mod:`_amici_common`, but wrapped rather than verbatim (see below).

What this module adds:

  * :func:`shared_sensitivity_params` — the cross-engine parameter alignment.
  * :func:`bn_sens`    — bngsim's forward sensitivities (+ timing).
  * :func:`amici_sens` — AMICI's forward sensitivities (+ timing).
  * :func:`sens_verdict` — the oracle over the (n_t, n_species, n_param) tensor.

Parameter alignment — the one genuinely new problem
---------------------------------------------------
Both engines read the same SBML, so their *species* ids match, and
``align_common`` handles them with only one adjustment: AMICI renames an id that
collides with its own generated C++ symbols (``x`` IS the state vector there), so
``_amici_common.align_common`` undoes that ``amici_`` prefix first (issue #321).
Their *parameter* ids do NOT match, because each engine flattens SBML **local**
(per-reaction ``kineticLaw``) parameters under its own naming scheme:

    SBML   reaction "J0", local parameter "V1"
    bngsim ``_lp_J0_V1``      (``_lp_`` + reaction id + ``_`` + parameter id)
    AMICI  ``J0_V1``          (reaction id + ``_`` + parameter id)

Global parameters keep their SBML id verbatim on both sides. So the mapping is a
``_lp_`` prefix strip on the bngsim side, after which the two id sets are
directly comparable and the shared list is their intersection. Two exclusions:

  * **compartment-size parameters** (``Model.compartment_size_params``) — bngsim
    refuses these as sensitivity targets (they are structural, not kinetic), so
    they can never be part of a shared list;
  * anything AMICI reports as *fixed* rather than free — ``get_free_parameter_ids()``
    is the authoritative AMICI-side set, and a fixed parameter has no ``sx`` column.

A model whose intersection is EMPTY yields no comparison at all; the runner
classifies it BAD_TEST rather than passing it vacuously.

Why the parameter list is bounded, and by WHAT
----------------------------------------------
Forward sensitivity integrates a coupled system of size ``n_species*(Np+1)``, so
a job's cost is set by that **product**, not by ``Np`` alone. The bound is
therefore a budget on the product (:data:`DEFAULT_PARAM_BUDGET`, via
:func:`budget_cap`), not a flat ceiling on ``Np``: a flat cap spends the same 20
columns on a 3-species toy and on a 1604-species model, over-paying at one end
and under-sampling at the other. Whatever the resulting cap, :func:`select_params`
spreads its pick evenly over the sorted id list — deterministic across re-runs,
and not concentrated on whichever reaction happens to sort first.

Uncapped is deliberately not the default. ``MODEL1009150002`` (1604 species,
7304 parameters) has a measured warm ``simultaneous`` solve of 14.4 s at
``Np=20``; at full ``Np`` that extrapolates to roughly 87 minutes, so removing
the bound converts three currently-PASSing models into TIMEOUTs. Memory is not
the constraint (~0.6 GB worst case) — wall-clock is.

Both the ``Np`` used and the pre-cap candidate count are recorded per model, so
the report discloses what was dropped instead of silently truncating, and no
timing number is read without the ``Np`` that produced it.

Parameter scale
---------------
AMICI can report sensitivities with respect to *transformed* parameters
(``ParameterScaling_ln`` / ``log10``), which would silently compare
``dx/d ln p`` against bngsim's ``dx/dp`` and differ by a factor of ``p``. The
scale is pinned to ``none`` explicitly here rather than trusted to remain the
import default.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _amici_common as ac  # noqa: E402

# Engine-agnostic surface, re-exported so the runner imports only this module.
align_common = ac.align_common
is_declared_refusal = ac.is_declared_refusal
schedule = ac.schedule
hardware_info = ac.hardware_info
load_and_filter = ac.load_and_filter
model_path = ac.model_path
sundials_version = ac.sundials_version
ensure_build_path = ac.ensure_build_path
_integrate_stats = ac._integrate_stats
_warm_rep_count = ac._warm_rep_count
DEFAULT_RTOL = ac.DEFAULT_RTOL
DEFAULT_ATOL = ac.DEFAULT_ATOL

# Max parameters per model. See "Why a parameter cap" above.
DEFAULT_PARAM_CAP = 20

# --------------------------------------------------------------------------- #
# The coupled-state budget (issue #331)
# --------------------------------------------------------------------------- #
# Forward sensitivity integrates a system of size ``n_species*(Np+1)``, so the
# cost of a job is set by that PRODUCT, not by ``Np`` alone. A flat parameter cap
# is therefore the wrong shape: it spends the same 20 columns on a 3-species toy
# (free) and on a 1604-species model (brutal), while leaving 441 small models
# sampled at 20 when they could be compared in full.
#
# Budgeting the product fixes both ends at once. Measured over the corpus at
# 20,000 states: 441 models get MORE parameters than the flat cap of 20, 525 are
# unchanged, and exactly ONE gets fewer — ``MODEL1009150002`` (1604 species),
# which drops 20 -> 12. That model is why an uncapped run is not an option: at
# its full 7304 parameters one warm ``simultaneous`` solve extrapolates from a
# measured 14.4 s to roughly 87 MINUTES, so uncapping would convert three
# currently-PASSing models into TIMEOUTs — trading real signal for none.
#
# Memory is NOT the constraint (the worst case is ~0.6 GB of CVODES Nordsieck
# history); wall-clock is.
DEFAULT_PARAM_BUDGET = 20_000


def budget_cap(n_species: int, budget: int, hard_cap: int = 0) -> int:
    """Parameters affordable for this model, from the coupled-state budget.

    ``budget`` is the ceiling on ``n_species * Np``; ``0`` disables it. ``hard_cap``
    is an optional additional ceiling (``--param-cap``), ``0`` for none. At least
    one parameter is always allowed — a model is never silently reduced to no
    comparison at all, which would masquerade as BAD_TEST.
    """
    cap = 0 if not budget else max(1, budget // max(int(n_species), 1))
    if hard_cap:
        cap = min(cap, hard_cap) if cap else hard_cap
    return cap


# The two CVODES forward-sensitivity corrector methods both engines expose.
# staggered    (CV_STAGGERED)   — state solved first, then the sensitivities as a
#                                 separate linear-in-the-sensitivities solve.
#                                 CVODES' and bngsim's default.
# simultaneous (CV_SIMULTANEOUS)— state and all sensitivity variables advanced as
#                                 one big coupled nonlinear system per step.
#                                 AMICI's compiled-in default.
# Both engines are pinned to the SAME method per job so a timing pair is strictly
# apples-to-apples; running both methods separates the engine effect from the
# method effect.
SENS_METHODS = ("staggered", "simultaneous")

# The bngsim prefix for a flattened SBML local (per-kineticLaw) parameter.
BN_LOCAL_PARAM_PREFIX = "_lp_"

# Compiled-model cache for the SENSITIVITY build. Deliberately a DIFFERENT
# directory and a different flag string from _amici_common.AMICI_CACHE: the
# sensitivity build emits a large extra body of C++ (dxdotdp and the sx right-hand
# side), so its extension is not interchangeable with the pure-ODE one and must
# never collide with it in the cache.
AMICI_SENS_CACHE = _HERE / "amici_sens_cache"
_SENS_BUILD_FLAGS = "sens=1;cl=0;obs=0;v1"


# --------------------------------------------------------------------------- #
# Parameter alignment
# --------------------------------------------------------------------------- #
def bn_param_to_sbml_id(name: str) -> str:
    """Map a bngsim parameter name back to its SBML id.

    bngsim flattens a reaction-local ``kineticLaw`` parameter to
    ``_lp_<reaction>_<param>``; AMICI names the same quantity ``<reaction>_<param>``.
    Stripping the prefix puts both engines in AMICI's namespace. A global
    parameter carries no prefix and passes through unchanged.
    """
    if name.startswith(BN_LOCAL_PARAM_PREFIX):
        return name[len(BN_LOCAL_PARAM_PREFIX) :]
    return name


def select_params(ids: list[str], cap: int) -> list[str]:
    """Deterministically pick at most ``cap`` ids, spread evenly over ``sorted(ids)``.

    Sorting first makes the choice independent of either engine's internal
    ordering, so a re-run — or a run on a machine where AMICI orders its free
    parameters differently — selects the identical set. Spreading (rather than
    taking the first ``cap``) avoids concentrating the whole sample on one
    reaction's locals, since a sorted SBML id list clusters by reaction prefix
    (``J0_K1, J0_Ki, J0_V1, J0_n, J1_KK2, …``).
    """
    ordered = sorted(ids)
    if cap <= 0 or len(ordered) <= cap:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, cap)
    return [ordered[int(round(i))] for i in idx]


def shared_sensitivity_params(
    bn_param_names: list[str],
    bn_compartment_size_params,
    amici_free_ids: list[str],
    cap: int = DEFAULT_PARAM_CAP,
) -> tuple[list[str], dict[str, str], int]:
    """The cross-engine sensitivity parameter list.

    Returns ``(shared_ids, bn_name_by_id, n_candidates)`` where ``shared_ids`` are
    SBML ids (AMICI's namespace) capped by :func:`select_params`,
    ``bn_name_by_id`` maps each back to the name bngsim wants, and
    ``n_candidates`` is how many were shared *before* the cap — so the report can
    disclose what the cap dropped rather than silently truncating.

    Excludes bngsim's compartment-size parameters (it refuses them as sensitivity
    targets) and anything not in AMICI's free-parameter set (a fixed parameter has
    no ``sx`` column). ``shared_ids`` may be empty; that is the caller's BAD_TEST.
    """
    excluded = set(bn_compartment_size_params or ())
    bn_by_id: dict[str, str] = {}
    for name in bn_param_names:
        if name in excluded:
            continue
        sbml_id = bn_param_to_sbml_id(name)
        # First writer wins: if a global parameter and a stripped local name ever
        # collide, the comparison would be ambiguous, so keep it out entirely.
        if sbml_id in bn_by_id:
            bn_by_id[sbml_id] = ""  # poisoned — dropped below
        else:
            bn_by_id[sbml_id] = name
    candidates = [p for p in amici_free_ids if bn_by_id.get(p)]
    chosen = select_params(candidates, cap)
    return chosen, {p: bn_by_id[p] for p in chosen}, len(candidates)


# --------------------------------------------------------------------------- #
# bngsim side
# --------------------------------------------------------------------------- #
def bn_sens(
    xml: str,
    t_start: float,
    t_end: float,
    n_points: int,
    rtol: float,
    atol: float,
    sens_params: list[str],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """One bngsim forward-sensitivity run.

    Returns ``(time, x[n_t, n_sp], sx[n_t, n_sp, n_p], species_names, timing)``.
    ``sens_params`` are bngsim-side parameter NAMES (``bn_name_by_id`` values from
    :func:`shared_sensitivity_params`), in the shared order; the returned ``sx``
    parameter axis follows that order.

    Timing mirrors :func:`_amici_common.amici_ode`'s schema so the two engines'
    columns line up: ``io`` / ``parse`` / ``interpret`` / ``jac_derive`` /
    ``codegen`` for setup, then the shared cold+warm integration stats. The
    Simulator construction is timed separately as ``sens_setup_sec`` because that
    is where the sensitivity RHS is prepared (bngsim wires sensitivities at
    construction, not at ``run()``).

    The COLD solve's tensor is what the verdict uses; warm reps are timing-only
    and never change it (``model.reset()`` restores the IC between reps, since
    bngsim's ``run()`` continues from state).
    """
    import bngsim
    import bngsim._sbml_loader as sbml_loader

    t0 = time.perf_counter()
    if Path(xml).exists():
        xml_string = Path(xml).read_text()
        io_sec = time.perf_counter() - t0
    else:
        xml_string = xml
        io_sec = 0.0

    model = sbml_loader.load_sbml_string(xml_string)

    t1 = time.perf_counter()
    sim = bngsim.Simulator(
        model,
        method="ode",
        sensitivity_params=list(sens_params),
        sensitivity_method=method,
    )
    sens_setup_sec = time.perf_counter() - t1

    t1 = time.perf_counter()
    r = sim.run(t_span=(t_start, t_end), n_points=n_points, rtol=rtol, atol=atol)
    cold_sec = time.perf_counter() - t1

    warm: list[float] = []
    for _ in range(_warm_rep_count(cold_sec)):
        try:
            model.reset()
            t1 = time.perf_counter()
            sim.run(t_span=(t_start, t_end), n_points=n_points, rtol=rtol, atol=atol)
            warm.append(time.perf_counter() - t1)
        except Exception:
            break
    integ = _integrate_stats(cold_sec, warm)

    stats = r.solver_stats if hasattr(r, "solver_stats") else {}
    ls_code = stats.get("linear_solver", 0)
    timing = {
        "io_sec": round(io_sec, 6),
        "parse_sec": round(sim.last_libsbml_parse_sec, 6),
        "interpret_sec": round(sim.last_interpret_sec, 6),
        "jac_derive_sec": round(sim.last_jacobian_sec, 6),
        "codegen_sec": round(sim.last_codegen_sec, 6),
        "sens_setup_sec": round(sens_setup_sec, 6),
        **integ,
        "config": {
            "codegen": sim.codegen_backend,
            "jacobian": sim.jacobian_strategy,
            "linear_solver": ac._rc.LINEAR_SOLVER_NAMES.get(ls_code, f"kind_{ls_code}"),
            "sens_method": method,
        },
    }

    sx = np.asarray(r.sensitivities, dtype=float)  # (n_t, n_species, n_param)
    return (
        np.asarray(r.time, dtype=float),
        np.asarray(r.species, dtype=float),
        sx,
        list(r.species_names),
        timing,
    )


# --------------------------------------------------------------------------- #
# AMICI side
# --------------------------------------------------------------------------- #
def _build_sens_model(sbml_str: str):
    """Build-or-load the sensitivity-enabled AMICI extension. ``(model, timing, cached)``.

    Same phase capture and taxonomy as :func:`_amici_common._build_model`, but
    with ``generate_sensitivity_code=True``, which emits the ``dxdotdp`` /
    sensitivity-RHS C++ on top of the pure-ODE body. That makes the compile
    materially larger than the ODE suite's, which is why this cache is keyed and
    stored separately.

    **Concurrency.** Unlike the ODE job — one job per model, so two workers never
    build the same key — this job runs one job per (model, method), and both
    methods share a key *on purpose* so the expensive compile is paid once. Two
    workers can therefore reach the same cache key at the same time. Building
    directly into the final directory let them stomp each other: cmake configuring
    a tree another process was writing, and a failed builder's cleanup deleting a
    good build out from under a live one (observed as ``FileNotFoundError`` on the
    cache dir and spurious ``CalledProcessError`` compile failures, i.e. a batch of
    REFERENCE_FAILED rows that were an artifact of the harness, not of AMICI).

    So this builds into a **process-unique staging directory** and then atomically
    renames it into place. The rename is the commit point: a directory that exists
    under ``AMICI_SENS_CACHE`` is always a complete build, never a half-written
    one. If another worker won the race, the rename finds the target already there
    — we discard our copy and load theirs. This also makes the cache kill-safe,
    which is what the suite's "resumable" claim depends on: a run killed mid-
    compile leaves a stray staging dir, never a corrupt cache entry.

    ``amici_sens_run.py`` additionally orders jobs method-major so the two methods
    of one model rarely race at all; this makes it *correct* when they do.
    """
    import amici

    ensure_build_path()

    key = hashlib.sha256((sbml_str + _SENS_BUILD_FLAGS).encode()).hexdigest()[:16]
    name = f"amicisens_{key}"
    mdir = AMICI_SENS_CACHE / name

    zero = {
        "parse_sec": 0.0,
        "interpret_sec": 0.0,
        "jac_derive_sec": 0.0,
        "codegen_sec": 0.0,
        "compile_sec": 0.0,
    }

    def _load(d: Path):
        t0 = time.perf_counter()
        mod = amici.import_model_module(name, str(d))
        return mod.get_model(), round(time.perf_counter() - t0, 6)

    if mdir.exists():
        # No rmtree-on-failure here (the ODE path's self-heal): with the atomic
        # rename below, a present directory is by construction complete, so a load
        # failure is not corruption we should "fix" — and deleting it could pull
        # the extension out from under a concurrent worker mid-import.
        model, load_sec = _load(mdir)
        return model, {**zero, "load_sec": load_sec}, True

    AMICI_SENS_CACHE.mkdir(parents=True, exist_ok=True)
    staging = AMICI_SENS_CACHE / f".building_{name}_{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)

    cap = ac._PhaseCapture()
    base_logger = logging.getLogger("amici")
    base_logger.addHandler(cap)
    saved = {nm: logging.getLogger(nm).level for nm in ac._AMICI_BUILD_LOGGERS}
    for nm in ac._AMICI_BUILD_LOGGERS:
        logging.getLogger(nm).setLevel(logging.DEBUG)
    try:
        with ac._silence_fds():
            importer = amici.SbmlImporter(sbml_str, from_file=False)
            importer.sbml2amici(
                name,
                str(staging),
                verbose=logging.DEBUG,
                generate_sensitivity_code=True,
                compute_conservation_laws=False,
                observation_model=[],
            )
        try:
            staging.rename(mdir)
            loaded_from = mdir
        except OSError:
            # Another worker committed this key first. Theirs is byte-equivalent
            # (same content hash, same flags), so use it and drop ours.
            loaded_from = mdir if mdir.exists() else staging
        model, load_sec = _load(loaded_from)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        base_logger.removeHandler(cap)
        for nm, lvl in saved.items():
            logging.getLogger(nm).setLevel(lvl)

    b = ac._bucket_phases(cap.phases)
    return (
        model,
        {
            "parse_sec": round(b["parse"], 6),
            "interpret_sec": round(b["interpret"], 6),
            "jac_derive_sec": round(b["jac"], 6),
            "codegen_sec": round(b["codegen"], 6),
            "compile_sec": round(b["compile"], 6),
            "load_sec": round(load_sec, 6),
        },
        False,
    )


def amici_free_parameter_ids(xml: str) -> tuple[list[str], object]:
    """``(free_parameter_ids, model)`` for ``xml`` — builds/loads the extension.

    Split out from :func:`amici_sens` because the shared parameter list must be
    negotiated (against bngsim's names) *before* either engine runs, and the AMICI
    id list is only knowable from a built model.
    """
    sbml_str = Path(xml).read_text() if Path(xml).exists() else xml
    model, timing, cached = _build_sens_model(sbml_str)
    return list(model.get_free_parameter_ids()), (model, timing, cached)


def amici_sens(
    built,
    t_start: float,
    t_end: float,
    n_points: int,
    rtol: float,
    atol: float,
    sens_ids: list[str],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """One AMICI CVODES forward-sensitivity run over the pre-built model.

    ``built`` is the tuple :func:`amici_free_parameter_ids` returned (model +
    build timing + cache flag), so the build is paid once per model and shared
    across the sensitivity methods rather than recompiled per method.

    Returns ``(time, x[n_t, n_sp], sx[n_t, n_sp, n_p], state_ids, timing)``.
    NOTE the transpose: AMICI's ``rdata.sx`` is ``(n_t, n_param, n_species)`` while
    bngsim's ``Result.sensitivities`` is ``(n_t, n_species, n_param)``; this
    function returns bngsim's layout so the comparison never has to think about it.

    ``set_parameter_list`` restricts the solve to the shared ids, so AMICI
    integrates exactly the same ``n_species*(Np+1)`` coupled system as bngsim —
    both a fairness requirement for the timing and a large cost saving on
    many-parameter models.
    """
    import amici.sim.sundials as ss

    model, build_timing, cached = built

    free_ids = list(model.get_free_parameter_ids())
    # Pin the parameter scale to LINEAR. AMICI can report d x / d(ln p) or
    # d x / d(log10 p); either would differ from bngsim's d x / d p by a factor of
    # p (or p*ln10) and produce a whole-tensor DIFF that looks like an engine bug.
    model.set_parameter_scale([ss.ParameterScaling_none] * len(free_ids))
    model.set_parameter_list([free_ids.index(p) for p in sens_ids])

    solver = model.create_solver()
    solver.set_relative_tolerance(rtol)
    solver.set_absolute_tolerance(atol)
    solver.set_sensitivity_order(ss.SensitivityOrder_first)
    solver.set_sensitivity_method(ss.SensitivityMethod_forward)
    solver.set_internal_sensitivity_method(
        ss.InternalSensitivityMethod_staggered
        if method == "staggered"
        else ss.InternalSensitivityMethod_simultaneous
    )
    try:
        ls_code = int(solver.get_linear_solver())
    except Exception:
        ls_code = -1
    linear_solver = ac._linear_solver_names().get(ls_code, f"kind_{ls_code}")

    # Anchor the initial condition at t_start — see the same call in
    # _amici_common.amici_ode. Sensitivities inherit the problem doubly: an
    # unrequested [0, t_start] prelude moves the state the sensitivities are
    # linearized about, so dx/dp is computed at the wrong operating point.
    model.set_t0(t_start)
    ts = np.linspace(t_start, t_end, n_points)
    model.set_timepoints(ts)

    t1 = time.perf_counter()
    rdata = model.simulate(solver=solver)
    cold_sec = time.perf_counter() - t1
    if int(rdata.status) != 0:
        try:
            msg = ss.simulation_status_to_str(int(rdata.status))
        except Exception:
            msg = str(rdata.status)
        raise RuntimeError(f"AMICI sensitivity integration failed (status {rdata.status}: {msg})")

    warm: list[float] = []
    for _ in range(_warm_rep_count(cold_sec)):
        try:
            t1 = time.perf_counter()
            rd = model.simulate(solver=solver)
            if int(rd.status) != 0:
                break
            warm.append(time.perf_counter() - t1)
        except Exception:
            break
    integ = _integrate_stats(cold_sec, warm)

    build_total = sum(build_timing[k] for k in build_timing)
    timing = {
        "io_sec": 0.0,
        "parse_sec": build_timing["parse_sec"],
        "interpret_sec": build_timing["interpret_sec"],
        "jac_derive_sec": build_timing["jac_derive_sec"],
        "codegen_sec": build_timing["codegen_sec"],
        "compile_sec": build_timing["compile_sec"],
        "load_sec": build_timing["load_sec"],
        **integ,
        "integrate_cpu_ms": round(float(rdata.cpu_time), 4),
        "parse_interpret_codegen_sec": round(build_total, 6),
        "config": {
            "codegen": "C++ (compiled, +sensitivity)",
            "jacobian": "analytical (symbolic)",
            "linear_solver": linear_solver,
            "cached": cached,
            "sens_method": method,
        },
    }

    sx = np.transpose(np.asarray(rdata.sx, dtype=float), (0, 2, 1))  # -> (n_t, n_sp, n_p)
    return (
        np.asarray(ts, dtype=float),
        np.asarray(rdata.x, dtype=float),
        sx,
        list(rdata.state_ids),
        timing,
    )


# --------------------------------------------------------------------------- #
# The oracle
# --------------------------------------------------------------------------- #
def flatten_tensor(sx: np.ndarray) -> np.ndarray:
    """``(n_t, n_sp, n_p)`` -> ``(n_t, n_sp*n_p)`` for the 2-D differ protocol.

    Each ``(species, parameter)`` pair becomes one column, which is precisely the
    granularity ``_core.differ`` wants: its per-column peak-over-time terms
    (``ABS_TOL_COL``) and its significance gate (``SIGNIF_FLOOR``) then judge each
    sensitivity coefficient against *its own* dynamic range. That matters far more
    here than for trajectories, because sensitivity magnitudes span many orders
    across parameters within one model (``dx/dK`` vs ``dx/dn``), and a single
    tensor-wide relative bar would either drown the small columns in the large
    ones or flag every near-zero cell.
    """
    n_t = sx.shape[0]
    return sx.reshape(n_t, -1)


# How many multiples of the solver's resolvable sensitivity magnitude a cell must
# exceed before a disagreement there is treated as signal. See
# :func:`sensitivity_noise_mask` for the derivation; 100 is chosen empirically as
# the smallest round factor that clears the observed reference-side noise (AMICI
# reporting ~1e-11 where the true derivative is exactly 0) while leaving a genuine
# large-magnitude divergence — BIOMD0000000457's parameter_12, |sx| ~ 1e15 — fully
# intact. Local error control does not bound global error, so some headroom over
# the raw 1x floor is required; the value is deliberately reported per run rather
# than hidden, via the report's ``noise_floor`` block.
SENS_NOISE_FACTOR = 100.0


def sensitivity_noise_mask(
    bn_sx: np.ndarray,
    am_sx: np.ndarray,
    param_values,
    atol: float,
    factor: float = SENS_NOISE_FACTOR,
) -> np.ndarray:
    """Cells where a disagreement is below what either solver could resolve.

    CVODES does not error-control the sensitivity variables on the same absolute
    scale as the states: with the standard scaling it applies ``atol_S_j =
    atol / |p_j|`` to ``s_ij = dx_i/dp_j``, which is the statement that the
    *product* ``s_ij * p_j`` — a quantity in the units of ``x_i`` — is resolvable
    only down to ``atol``. A disagreement smaller than that is two engines
    reporting their own noise, not a difference in the model.

    This matters far more for sensitivities than for trajectories, and in a way
    ``differ``'s scale-relative terms cannot express. A sensitivity that is
    *identically zero* is common and correct — a parameter simply does not
    influence a species — and the engines disagree about it in the worst possible
    way for a relative metric: one returns exact ``0.0`` and the other returns its
    own integration noise, so ``|a-b| / max(|a|,|b|)`` saturates at exactly 1.0 no
    matter how tiny both numbers are. Observed on BIOMD0000000569, where finite
    differences on bngsim's own trajectories confirm the true derivative is 0,
    bngsim returns 0, and AMICI returns ~1e-11..1e-14. Nothing keyed to the
    tensor's own peak can fix that, because the saturated ratio is scale-free.

    So the floor is absolute and physically derived: forgive a cell when
    ``|bn - am| <= factor * atol / |p_j|``. Returns a boolean mask shaped like the
    inputs, suitable for ``differ``'s ``forgive_mask``. Note that ``differ`` never
    forgives a one-sided non-finite cell regardless of this mask, so a NaN column
    (the GH #310 signature) still fails — the floor silences noise, not blow-ups.

    A parameter whose value is 0 has no meaningful ``atol/|p|`` scaling; those
    columns fall back to the raw ``atol``, which is the conservative choice (it
    forgives less than any positive ``|p| < 1`` would).
    """
    p = np.abs(np.asarray(param_values, dtype=float))
    floor = np.where(p > 0, factor * atol / np.where(p > 0, p, 1.0), factor * atol)
    return np.abs(np.asarray(bn_sx) - np.asarray(am_sx)) <= floor[np.newaxis, np.newaxis, :]


def sens_resolution_floors(param_values, atol, factor: float = SENS_NOISE_FACTOR):
    """Per-parameter resolvable magnitude ``factor*atol/|p_j|``, or ``None``.

    Shares the formula with :func:`sensitivity_noise_mask` rather than restating
    it, so the two cannot drift into disagreeing about what "resolvable" means.
    Returns ``None`` when the caller has no parameter values — exactly when the
    mask is also inactive — so a report field of ``None`` honestly reads "not
    assessed" rather than a fabricated 0.

    Returned PER COLUMN, never reduced to one scalar. Reducing with ``max`` would
    let a single tiny-valued parameter inflate the threshold for the whole tensor
    and mark a live model degenerate: ``BIOMD0000000002`` has real dynamics
    (state span 1.9) and a genuine ``max|sx| = 3.7e-5``, but one small parameter
    puts the largest column floor at ``8.3e-5``, which a global max would declare
    "entirely unresolvable". That is the same global-reduction mistake as the
    transversality noise floor in issue #322, one module over.
    """
    if param_values is None or atol is None:
        return None
    p = np.abs(np.asarray(param_values, dtype=float))
    if p.size == 0:
        return None
    return np.where(p > 0, factor * atol / np.where(p > 0, p, 1.0), factor * atol)


# The mask's generous multiplier is deliberately NOT reused for the degeneracy
# test: the two questions carry opposite risk.
#
#   forgiving a cell   — being generous avoids false DIFFs, so SENS_NOISE_FACTOR
#                        is 100x the raw resolution on purpose.
#   claiming vacuity   — being generous INVENTS vacuity, marking real signal
#                        unresolvable and quietly discounting a genuine pass.
#
# Measured on the corpus: at factor=100, 5 of 19 flagged models (BIOMD0000000335,
# 500, 501, 827, MODEL1201140005) have columns that clear their raw floor and are
# not degenerate at all. So the claim is made against the RAW resolution.
DEGENERACY_FACTOR = 1.0


def resolvable_param_columns(bn_sx: np.ndarray, param_values, atol) -> int | None:
    """How many parameter columns carry a sensitivity above their OWN raw floor.

    ``0`` means every column is beneath what the solver can resolve, i.e. the
    comparison established nothing regardless of how well the engines agreed —
    the vacuous-pass condition of issue #328. ``None`` when the floor is not
    assessable.

    Judged at ``DEGENERACY_FACTOR`` (the raw ``atol/|p_j|``), not at the mask's
    ``SENS_NOISE_FACTOR`` — see above for why the asymmetry is deliberate.
    """
    floors = sens_resolution_floors(param_values, atol, factor=DEGENERACY_FACTOR)
    if floors is None or bn_sx.size == 0:
        return None
    peaks = np.nanmax(np.abs(bn_sx), axis=(0, 1))  # peak over (time, species) per param
    return int(np.count_nonzero(peaks > floors))


def sens_verdict(
    bn_sx: np.ndarray,
    am_sx: np.ndarray,
    *,
    param_values=None,
    atol: float | None = None,
) -> dict:
    """Verdict for two aligned sensitivity tensors, via the shared differ protocol.

    Deliberately reuses ``_core.differ.deterministic_verdict`` on the flattened
    tensor rather than defining a second oracle, so a tolerance change lands in
    every suite at once. The one sensitivity-specific addition is the
    solver-resolution floor of :func:`sensitivity_noise_mask`, passed through
    ``differ``'s existing ``forgive_mask`` hook rather than by weakening any shared
    constant — ``differ``'s own terms are all relative to the compared data's
    scale, and the zero-derivative case above is scale-free.

    ``param_values`` (in the same order as the tensor's parameter axis) and
    ``atol`` (the integration tolerance both engines ran at) enable the floor.
    Omitting either reproduces the original scale-only behavior, so the function
    stays usable from a context that does not know the parameter values.

    The returned dict is ``differ``'s, plus ``n_noise_forgiven`` — the count of
    cells the floor silenced, recorded so a run can never quietly forgive its way
    to a PASS without that being visible in the report.
    """
    from _core import differ

    mask = None
    if param_values is not None and atol is not None:
        mask = flatten_tensor(sensitivity_noise_mask(bn_sx, am_sx, param_values, atol))
    v = differ.deterministic_verdict(
        flatten_tensor(bn_sx), flatten_tensor(am_sx), forgive_mask=mask
    )
    v["n_noise_forgiven"] = int(mask.sum()) if mask is not None else 0
    return v


def measure_warmup() -> dict:
    """Per-process warmup for both engines — see :func:`_amici_common.measure_warmup`."""
    return ac.measure_warmup()


def set_amici_quiet() -> None:
    ac.set_amici_quiet()


def classify_reference_refusal(exc: str) -> str:
    """Map an AMICI sensitivity failure to a refusal subclass.

    Delegates to the ODE classifier — the failure modes (feature gap / compile /
    integrator) are the same, and the sensitivity build simply has more C++ that
    can fail to compile.
    """
    return ac.classify_reference_refusal(exc)
