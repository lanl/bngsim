#!/usr/bin/env python3
"""Jacobian-based characterization of the bng_parity ODE test set.

For each ODE model in the corpus we compute two intrinsic, model-level quantities
that predict which linear-algebra regime an implicit stiff solve falls into:

  * **Structural Jacobian density** = nnz(J) / N^2, N = number of ODEs (species).
    nnz is the STRUCTURAL sparsity pattern (union of |J_ij| > 0 over several random
    strictly-positive states), so accidental numeric zeros at one state don't
    deflate it. This is what a sparse linear solver actually sees.

  * **Stiffness ratio** = max|Re lambda| / min_nonzero|Re lambda| over the eigenvalues
    of the REDUCED Jacobian (the conserved-moiety modes removed via BNGsim's own
    ``conservation_laws``). Evaluated at several points along the trajectory;
    we report the MAX over time. Oscillatory models (a weakly-damped complex mode
    would otherwise dominate the denominator) are DETECTED and moved to their own
    category rather than reported as spuriously stiff.

The intent (see the paper supplement) is to partition the ODE test set into the
~O(N) regime (non-stiff and/or non-dense -> explicit or sparse solve) and the
~O(N^3) regime (stiff AND dense -> dense LU dominates). The partition + the
cost~N regression that validates it live in ``--analyze`` mode, which joins this
report with ``runs/report_ode.json`` (per-model N and warm integration cost).

Alongside the Jacobian quantities each row also carries the two model-composition
counts the paper's representative-models table needs (issue #42):
``n_seed_nonzero`` (seed species whose RESOLVED initial value is nonzero) and
``n_independent_parameters`` (numeric-valued independent parameters), the latter
with the ``excluded_parameters`` names + reasons that produced it, so the count is
reviewable rather than a hidden judgment. See :func:`seed_and_parameter_census`.

The trajectory half runs on the parity gate's own horizon and tolerances, read from
``runs/report_ode.json`` (or the report ``--horizons`` names): without that report the
run refuses to start, and a model the gate has no row for gets its density only, never
a made-up horizon (issue #509). Every row records the ``run`` block it was integrated
on. Trajectory states are ``Result.state``, the full integrator state (issue #508); a
sample whose RHS or Jacobian is non-finite is skipped and counted rather than voiding
the model (issue #510); and the Jacobian is ``Model.jacobian``'s — closed-form when it
covers every reaction, the difference quotient of ``Model.rhs`` otherwise — with
``jacobian_method`` saying which (issue #512).

Environment: run with the bngsim editable checkout venv, e.g.
    BNGPATH=/path/to/BioNetGen-2.9.3 \
      ~/Code/bngsim/.venv/bin/python jacobian_characterization.py --limit 5

Reads the vendored corpus (``models/<model_id>``) + ``jobs.json`` read-only; the
only write is the output report JSON. Network generation shells out to BNG2.pl
exactly as the parity runner does (``_bng_common.generate_network``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _bng_common as bc  # noqa: E402  (sibling module; path injected above)

# ---- tunables (documented; overridable via CLI) ----------------------------
DENSITY_SAMPLES = 5  # random states unioned for the structural pattern
NONZERO_REL_TOL = 1e-9  # |Re lambda| below this * max is treated as zero/marginal
OSC_DAMPING_CUT = 0.01  # complex mode with |Re|/|Im| < this is "oscillatory"
OSC_NEARZERO_BAND = 1e-3  # ...and only if its |Re| sits in the near-zero band
FULL_GRID_MAX_N = 300  # N <= this: evaluate at every trajectory point
DENSE_TIME_SAMPLES = 64  # N >  this: this many log-spaced trajectory points (was 3)
EIG_MAX_N = 5000  # N > this: skip eigen-work (dense eig too costly); density only
DEFAULT_TIMEOUT = 240.0
RNG_SEED = 12345

# ---- parameter-census policy (see seed_and_parameter_census) ---------------
# Unit-conversion constants: symbols a model carries so molecule counts and
# concentrations can be interconverted. They are physical constants / unit
# bookkeeping, not model parameters, so the paper's caption excludes them. Both
# name sets are matched CASE-SENSITIVELY and kept deliberately narrow — a name
# not listed is counted, and every exclusion is emitted per model in
# ``excluded_parameters`` so the count stays auditable.
#
# Avogadro's number, in whatever unit scaling a model works in (6.022e23 /mol,
# 6.02e8 /um^3, 6.022e5 /uM at 1 pL, ...). The magnitude guard is what keeps the
# short spellings safe: ``nA 5`` in ``race.bngl`` is an Erlang step count, and
# ``Na`` in an electrophysiology model would be sodium, not Avogadro.
AVOGADRO_PARAM_NAMES = frozenset({"NA", "Na", "N_A", "NAv", "NaV", "Nav", "NA_V", "Avogadro"})
AVOGADRO_MIN_VALUE = 1e5
# Reference volumes and geometric factors carried purely to convert counts to
# concentrations (a cell volume, a membrane-adjacent layer thickness). No
# magnitude guard: a volume is any magnitude, so these rest on the name alone.
UNIT_GEOMETRY_PARAM_NAMES = frozenset(
    {"V_ref", "Vref", "V_cell", "Vcell", "vol_ref", "h_mem", "hmem"}
)
# BNG2.pl's own synthetic parameters (``_rateLaw{N}``, emitted when a BNGL rate
# law is a compound expression such as ``chi*kon``) live in the reserved
# leading-underscore namespace and are not model parameters at all.
SYNTHETIC_PARAM_RE = re.compile(r"^_")


def _is_unit_conversion(name: str, value_expr: str) -> bool:
    """Whether a PRIMARY (numeric-valued) parameter is unit-conversion bookkeeping."""
    if name in UNIT_GEOMETRY_PARAM_NAMES:
        return True
    if name not in AVOGADRO_PARAM_NAMES:
        return False
    try:
        return abs(float(value_expr)) >= AVOGADRO_MIN_VALUE
    except ValueError:  # not a literal -> already classified as derived
        return False


# ---------------------------------------------------------------------------
# model enumeration + horizon (from the existing parity artifacts)
# ---------------------------------------------------------------------------
def load_ode_jobs() -> list[dict]:
    """ODE jobs from jobs.json (one per vendored model, method == 'ode')."""
    jobs = json.loads((HERE / "jobs.json").read_text())["jobs"]
    return [j for j in jobs if j.get("method") == "ode"]


DEFAULT_HORIZONS = "runs/report_ode.json"  # the parity gate's report, relative to HERE


def load_horizons(path: Path | None = None) -> dict[str, dict]:
    """model_id -> {t_start,t_end,n_steps,rtol,atol,n_species,cost_sec,outcome,...} from the
    parity gate's report (``runs/report_ode.json``, or ``path``).

    Raises ``FileNotFoundError`` when the report is absent. The harness characterizes the
    models the gate passed on the runs it passed them on, and without the report it has
    no horizon or tolerances to run them on; ``runs/`` is gitignored, so a fresh checkout
    has none — run ``bng_ode_run.py`` first, or pass ``--horizons`` (issue #509). The
    gate writes the tolerances it resolved (its shared default, or the model's ``tol``
    override) into ``timing.spec``, so these are the gate's, not the BNGL's.
    """
    out: dict[str, dict] = {}
    rp = Path(path) if path is not None else HERE / DEFAULT_HORIZONS
    if not rp.exists():
        raise FileNotFoundError(
            f"parity gate report not found: {rp} — run bng_ode_run.py first, or pass "
            "--horizons <report_ode.json> (issue #509)"
        )
    for r in json.loads(rp.read_text()).get("results", []):
        t = r.get("timing") or {}
        spec = t.get("spec") or {}
        bs = t.get("bngsim") or {}
        out[r["model_id"]] = {
            "t_start": spec.get("t_start", 0.0),
            "t_end": spec.get("t_end"),
            "n_steps": spec.get("n_steps"),
            "rtol": spec.get("rtol"),
            "atol": spec.get("atol"),
            "n_species": spec.get("n_species"),
            "cost_sec": bs.get("integrate_warm_median_sec"),
            "outcome": r.get("outcome"),
            "linear_solver": (bs.get("config") or {}).get("linear_solver"),
            "source": str(rp),
        }
    return out


def gate_run_settings(horizon: dict | None) -> dict | None:
    """The horizon and tolerances the gate ran this model on — the ``run`` block every row
    records so a report can be checked against the parity report it describes.

    ``None`` when the gate has no row for the model: the caller then skips the trajectory
    instead of integrating on a default ``t_end`` nothing in the output would admit to
    (issue #509). A report row from before the gate recorded its tolerances falls back to
    the gate's shared defaults and says so in ``tolerance_source``.
    """
    if not horizon or horizon.get("t_end") is None:
        return None
    rtol, atol = horizon.get("rtol"), horizon.get("atol")
    tol_src = "gate_report"
    if rtol is None or atol is None:
        rtol = bc.DEFAULT_RTOL if rtol is None else rtol
        atol = bc.DEFAULT_ATOL if atol is None else atol
        tol_src = "gate_default"
    return {
        "t_start": float(horizon.get("t_start") or 0.0),
        "t_end": float(horizon["t_end"]),
        "n_steps": int(horizon.get("n_steps") or bc.DEFAULT_N_STEPS),
        "rtol": float(rtol),
        "atol": float(atol),
        "horizon_source": horizon.get("source") or "gate_report",
        "tolerance_source": tol_src,
    }


# ---------------------------------------------------------------------------
# per-model characterization
# ---------------------------------------------------------------------------
def _prop(obj, name):
    v = getattr(obj, name)
    return v() if callable(v) else v


def _link_matrix(cl: dict, n: int) -> tuple[list[int], np.ndarray]:
    """Build (independent indices, link matrix L: n x n_ind) from conservation laws.

    x = L @ x_ind + const, so the reduced Jacobian is J[ind, :] @ L and its
    eigenvalues are exactly the nonzero eigenvalues of the full Jacobian.
    """
    ind = list(cl["independent"])
    dep = list(cl["dependent"])
    if not ind:  # no conservation laws (n_laws==0): the whole system is independent.
        return list(range(n)), np.eye(n)  # else the reduced Jacobian is empty -> false degenerate
    C = np.asarray(cl["coefficients"], float) if cl.get("coefficients") else np.zeros((0, n))
    n_ind = len(ind)
    L = np.zeros((n, n_ind))
    for b, s in enumerate(ind):
        L[s, b] = 1.0
    for k, d in enumerate(dep):
        cdd = C[k, d]
        if cdd == 0:
            continue
        for b, s in enumerate(ind):
            L[d, b] = -C[k, s] / cdd
    return ind, L


def _time_indices(times: np.ndarray, n: int, k: int = DENSE_TIME_SAMPLES) -> list[int]:
    """Trajectory sample indices for the stiffness sweep.

    For N <= FULL_GRID_MAX_N we evaluate at every output time (the eig is cheap).
    For N > FULL_GRID_MAX_N the full output grid would mean one dense eig per
    output time (~1000 at N~1600), so instead we take up to ``k`` (=DENSE_TIME_SAMPLES)
    log-spaced-in-time points spanning [first t>0, t_final], always pinning index 0
    (the initial time) and the final index. Log spacing concentrates samples where
    fast stiff transients live (early times) while still reaching the final state.
    3 points (the old behavior) under-samples the trajectory max/median badly; ~64
    resolves them well while keeping the eigen-work bounded. Snapping log-spaced
    target times to nearest grid points on a coarse early grid may yield slightly
    fewer than ``k`` distinct indices, which is fine.
    """
    T = np.asarray(times, float)
    m = len(T)
    if n <= FULL_GRID_MAX_N or m <= k:
        return list(range(m))
    pos = T[T > 0]
    t_lo = pos.min() if pos.size else (T[1] if m > 1 else T[0])
    t_hi = T[-1]
    targets = np.geomspace(max(t_lo, 1e-300), max(t_hi, t_lo), num=k)
    picks = {0, m - 1}
    picks.update(int(np.argmin(np.abs(T - tt))) for tt in targets)
    return sorted(picks)


def _classify_eigs(eigs: np.ndarray) -> dict:
    """One time point: max|Re|, min-nonzero|Re|, ratio, and oscillatory flag."""
    re = np.abs(eigs.real)
    im = np.abs(eigs.imag)
    mag = np.abs(eigs)  # sqrt(Re^2 + Im^2)
    max_re = float(re.max()) if re.size else 0.0
    if max_re == 0.0:
        return {"ratio": float("inf"), "max_re": 0.0, "min_re": 0.0, "oscillatory": False}
    floor = NONZERO_REL_TOL * max_re
    # Oscillatory: a *genuine* (magnitude above the zero floor -> not a numerical/
    # conservation-residual zero) weakly-damped complex mode in the near-zero |Re| band.
    # The magnitude floor is essential: without it, machine-zero eigenvalues with a tiny
    # imaginary part (|Re|~1e-13, |Im|~1e-7) are mislabeled oscillatory (e.g. fceri_fyn).
    band = OSC_NEARZERO_BAND * max_re
    osc = bool(
        np.any(
            (im > 0)
            & (mag > floor)
            & (re < band)
            & (re < OSC_DAMPING_CUT * np.maximum(im, 1e-300))
        )
    )
    nz = re[re > floor]
    ratio = float(max_re / nz.min()) if nz.size else float("inf")
    return {
        "ratio": ratio,
        "max_re": max_re,
        "min_re": (float(nz.min()) if nz.size else 0.0),
        "oscillatory": osc,
    }


# ``jacobian_method`` spellings per ``JacobianMatrix.source``.
JACOBIAN_METHOD = {"analytical": "native_analytical", "finite-difference": "finite_difference"}


def jacobian_evaluators(m):
    """``(jac_at, rhs_at, fields)`` for a loaded model (issue #512).

    ``jac_at(y, t)`` is ``Model.jacobian``: the closed form when it covers every reaction,
    the difference quotient of ``Model.rhs`` otherwise — never the partial assembly the
    private ``_dense_analytical_jacobian`` hook returns when ``analytical_jacobian_complete``
    is False, which characterized a Lorenz attractor as a zero Jacobian. ``fields`` carries
    the flag and, next to it, ``jacobian_method``: which source the rows were computed from.
    """
    complete = bool(m.prepare_analytical_jacobian())
    source = m.jacobian(np.asarray(m.get_state(), float)).source

    def jac_at(y, t=0.0):
        return np.asarray(m.jacobian(y, t=t), float)

    def rhs_at(y, t=0.0):
        return np.asarray(m.rhs(y, t=t), float)

    fields = {
        "analytical_jacobian_complete": complete,
        "jacobian_method": JACOBIAN_METHOD.get(source, source),
    }
    return jac_at, rhs_at, fields


def density_pattern(jac_at, n: int, samples: int = DENSITY_SAMPLES, seed: int = RNG_SEED):
    """Structural pattern: the union of |J| > 0 over ``samples`` random strictly-positive
    states. Both the closed form and the difference quotient return EXACT zeros where a
    derivative is structurally zero, so |J| > 0 is the pattern from either source."""
    rng = np.random.default_rng(seed)
    pat = np.zeros((n, n), bool)
    for _ in range(samples):
        pat |= np.abs(jac_at(rng.uniform(0.1, 2.0, n))) > 0
    return pat


class NoFiniteSample(RuntimeError):
    """Every sampled trajectory state had a non-finite RHS or Jacobian (issue #510)."""


def sweep_trajectory(jac_at, rhs_at, X, T, idxs, ind, L, atol: float) -> dict:
    """The eigenvalue sweep over the sampled trajectory states, tolerant of a degenerate
    sample (issue #510).

    An output-grid state is an interpolant and can sit slightly negative where the
    integrator's internal states were not; a rate law with a fractional power or a log
    is undefined there, so the RHS or the Jacobian is non-finite while the trajectory
    itself is fine. A negative entry smaller in magnitude than ``atol`` — zero to the
    solver — is clamped to zero before the evaluation, and a sample whose RHS or Jacobian
    is still non-finite is skipped and counted instead of raising out of the whole model
    (one such sample used to void a model's stiffness). Raises :class:`NoFiniteSample`
    when no sample survives.

    Returns ``per_time`` (one :func:`_classify_eigs` record per surviving sample),
    ``pattern`` (the union of |J| > 0 over them) and the ``n_used`` / ``n_skipped`` /
    ``n_clamped`` counts.
    """
    pat = None
    per_time: list[dict] = []
    n_skipped = n_clamped = 0
    for i in idxs:
        y = np.array(X[i], float)
        below = (y < 0) & (np.abs(y) < atol)
        if below.any():
            y[below] = 0.0
            n_clamped += 1
        t = float(T[i])
        f = np.asarray(rhs_at(y, t), float)
        J = np.asarray(jac_at(y, t), float) if np.isfinite(f).all() else None
        if J is None or not np.isfinite(J).all():
            n_skipped += 1
            continue
        nz = np.abs(J) > 0
        pat = nz if pat is None else (pat | nz)
        eigs = np.linalg.eigvals(J[np.ix_(ind, range(J.shape[0]))] @ L)
        per_time.append({"t": t, **_classify_eigs(eigs)})
    if not per_time:
        raise NoFiniteSample(
            f"all {len(idxs)} sampled trajectory states had a non-finite RHS or Jacobian"
        )
    return {
        "per_time": per_time,
        "pattern": pat,
        "n_used": len(per_time),
        "n_skipped": n_skipped,
        "n_clamped": n_clamped,
    }


def trajectory_fields(
    res,
    cl: dict,
    n: int,
    jac_at,
    rhs_at,
    pat,
    atol: float,
    dense_time_samples: int = DENSE_TIME_SAMPLES,
) -> dict:
    """The stiffness sweep over one solve, as row fields (issues #508, #510).

    States come from ``Result.state`` — the full integrator state, which on an SBML model
    with an event-promoted parameter or compartment is wider than ``Result.species`` and
    is the only vector the evaluators accept (a ``species`` row used to be read past its
    end). ``pat``, the random-state density pattern, is extended in place with the
    sampled states' nonzeros, and the density is re-read from it.
    """
    X = np.asarray(res.state, float)
    T = np.asarray(res.time, float)
    if X.shape[1] != n:
        raise ValueError(f"Result.state has {X.shape[1]} columns; the model has {n} state entries")
    ind, L = _link_matrix(cl, n)
    idxs = _time_indices(T, n, dense_time_samples)
    sw = sweep_trajectory(jac_at, rhs_at, X, T, idxs, ind, L, atol)
    pat |= sw["pattern"]
    finite = [p["ratio"] for p in sw["per_time"] if np.isfinite(p["ratio"])]
    return {
        "n_reported_species": len(res.species_names),
        "nnz": int(pat.sum()),
        "density": int(pat.sum()) / (n * n),
        "stiffness_ratio_max": float(max(finite)) if finite else float("inf"),
        # median over finite per-time ratios = the "sustained" stiffness (vs the peak).
        "stiffness_ratio_median": float(np.median(finite)) if finite else float("inf"),
        "n_time_points": sw["n_used"],
        "n_time_points_skipped": sw["n_skipped"],
        "n_time_points_clamped": sw["n_clamped"],
        "oscillatory": bool(any(p["oscillatory"] for p in sw["per_time"])),
        "per_time": sw["per_time"],
    }


def _parse_net_parameters(net_text: str) -> list[tuple[str, str, bool]]:
    """``(name, value_expr, is_derived)`` per ``begin parameters`` line of a .net file.

    The block is ``<index> <name> <value-or-expression>  # Constant|ConstantExpression``.
    A parameter is DERIVED when its value column is not a numeric literal (``R_dim
    Rtot/2``) — the version-independent reading of the caption's "numeric-valued" — or
    when BNG2.pl tags the line ``ConstantExpression``, whichever fires first.

    The .net block is the authority here rather than ``Model.param_names``: the loaded
    model's parameter table also carries BNGL global functions (``obs_X() a+b*c``), which
    are expressions over the model's state, not declared parameters.
    """
    out: list[tuple[str, str, bool]] = []
    in_block = False
    for raw in net_text.splitlines():
        s = raw.strip()
        if s.startswith("begin parameters"):
            in_block = True
            continue
        if in_block and s.startswith("end parameters"):
            break
        if not in_block or not s or s.startswith("#"):
            continue
        body, _, comment = s.partition("#")
        fields = body.split()
        if len(fields) < 3:
            continue
        name, expr = fields[1], " ".join(fields[2:])
        try:
            float(expr)
            numeric = True
        except ValueError:
            numeric = False
        out.append((name, expr, (not numeric) or "ConstantExpression" in comment))
    return out


def seed_and_parameter_census(m, net_text: str) -> dict:
    """Model-composition counts for the paper's representative-models table (issue #42).

    ``n_seed_nonzero`` — species whose RESOLVED initial value is nonzero. Read off the
    LOADED network instead of counting ``seed species`` lines, so parameter-valued ICs
    are already evaluated (BNG substituted ``Rtot`` / ``ic_wt_cell__X`` / ...): seed
    lines whose parameter resolves to 0 are correctly not counted, and every species BNG
    did not seed is exactly 0.0. Must be called BEFORE the trajectory run, while the
    state is still the initial condition. When a state-setup prefix was replayed into
    netgen (``dirty_carryover`` / ``setConcentration`` actions) this is the state the
    benchmarked run actually starts from, which is the intended reading.

    ``n_independent_parameters`` — primary, numeric-valued model parameters that are not
    unit-conversion constants, i.e. the caption's "numeric-valued independent parameters,
    including those that set initial amounts; unit-conversion constants and parameters
    derived from others are excluded". A bare count is only meaningful together with the
    policy that produced it, so every excluded name is emitted with its reason:

      ``derived``          the value is an expression over other parameters (``R_dim
                           Rtot/2``, Blinov's ``loop1..loop5``), so it is not an
                           independent knob — see :func:`_parse_net_parameters`.
      ``unit_conversion``  the parameter is unit bookkeeping rather than model content
                           — Avogadro's number or a reference volume / layer thickness;
                           see :func:`_is_unit_conversion`.

    ``_rateLaw{N}`` (BNG's own synthetic compound-rate-law parameters) are dropped up
    front and appear in neither ``n_parameters`` nor ``excluded_parameters``, so
    ``n_independent_parameters == n_parameters - len(excluded_parameters)`` holds.
    """
    n_declared = 0
    excluded: list[str] = []
    reasons: dict[str, str] = {}
    for name, expr, derived in _parse_net_parameters(net_text):
        if SYNTHETIC_PARAM_RE.match(name):
            continue
        n_declared += 1
        if derived:
            reason = "derived"
        elif _is_unit_conversion(name, expr):
            reason = "unit_conversion"
        else:
            continue
        excluded.append(name)
        reasons[name] = reason
    return {
        "n_seed_nonzero": int(np.count_nonzero(np.asarray(m._core.get_state(), float))),
        "n_parameters": n_declared,
        "n_independent_parameters": n_declared - len(excluded),
        "excluded_parameters": excluded,
        "excluded_parameter_reasons": reasons,
    }


def characterize_model(
    model_id: str,
    horizon: dict,
    bng2_pl: str,
    timeout: float = DEFAULT_TIMEOUT,
    dense_time_samples: int = DENSE_TIME_SAMPLES,
) -> dict:
    """Full characterization of one ODE model. Never raises: errors -> status field.

    ``horizon`` is the model's entry from :func:`load_horizons`. The composition census
    and the density need no trajectory and are always computed; the stiffness sweep
    integrates on the gate's horizon and tolerances (the row's ``run`` block) and is
    skipped — status ``ok_density_only`` — when the gate has no row for the model,
    rather than run on a default horizon nothing in the report would admit to (issue
    #509). A sweep that fails keeps the density under ``ok_no_stiffness``.
    """
    from bngsim import Model, Simulator

    row: dict = {"model_id": model_id, "status": "ok"}
    bngl_path = HERE / "models" / model_id
    if not bngl_path.exists():
        return {**row, "status": "no_bngl"}
    bngl_text = bngl_path.read_text(errors="replace")
    run = gate_run_settings(horizon)
    row["gate_outcome"] = (horizon or {}).get("outcome")
    row["run"] = run

    # network generation — same prefix the parity runner uses (single-phase state).
    gen_network = bc._model_gen_network(bngl_text)
    try:
        state_prefix, info = bc.state_setup_prefix(bngl_text, track="ode")
        dirty = bool(info.get("dirty_carryover"))
    except Exception:
        state_prefix, dirty = "", False
    workdir = Path(tempfile.mkdtemp(prefix="bng_jac_"))
    try:
        net_path, netgen_sec, netgen_err = bc.generate_network(
            bngl_text,
            bng2_pl,
            workdir,
            timeout=timeout,
            gen_network=gen_network,
            state_prefix=("" if dirty else state_prefix),
        )
        if net_path is None:
            return {**row, "status": "netgen_failed", "detail": netgen_err}

        net_text = Path(net_path).read_text(errors="replace")
        m = Model.from_net(str(net_path))
        core = m._core
        n = int(_prop(m, "n_species"))
        row["N"] = n
        row["n_reactions"] = int(_prop(m, "n_reactions"))
        cl = core.conservation_laws
        row["n_conservation_laws"] = int(cl["n_laws"])
        row["rank"] = n - int(cl["n_laws"])
        row["dirty_carryover"] = dirty
        # Composition counts must be read while the state is still the IC (before run()).
        row.update(seed_and_parameter_census(m, net_text))

        # The Jacobian the model actually carries (issue #512) and the structural density
        # from it: exact zeros where structurally zero, so |J|>0 is the pattern.
        jac_at, rhs_at, jac_fields = jacobian_evaluators(m)
        row.update(jac_fields)
        pat = density_pattern(jac_at, n)
        row["nnz"] = int(pat.sum())
        row["density"] = row["nnz"] / (n * n)
    except Exception as exc:
        return {
            **row,
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-1500:],
        }
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    if n > EIG_MAX_N:
        return {
            **row,
            "status": "ok_density_only",
            "detail": f"N={n} > EIG_MAX_N={EIG_MAX_N}; stiffness skipped",
        }
    if run is None:
        return {
            **row,
            "status": "ok_density_only",
            "detail": "no horizon for this model in the gate report; trajectory skipped (issue #509)",
        }

    # Trajectory + stiffness on the gate's run, in its own try: a sweep that fails must
    # not lose the census and density already in the row.
    try:
        res = Simulator(m, method="ode").run(
            t_span=(run["t_start"], run["t_end"]),
            n_points=run["n_steps"] + 1,
            rtol=run["rtol"],
            atol=run["atol"],
        )
        row.update(
            trajectory_fields(res, cl, n, jac_at, rhs_at, pat, run["atol"], dense_time_samples)
        )
        row["category"] = (
            "oscillatory" if row["oscillatory"] else "pending"
        )  # stiff/nonstiff set in --analyze
        return row
    except Exception as exc:
        row["status"] = "ok_no_stiffness"
        row["detail"] = f"{type(exc).__name__}: {exc}"[:300]
        row["trace"] = traceback.format_exc()[-1200:]
        return row


# ---------------------------------------------------------------------------
# analysis: partition into O(N) vs O(N^3) regimes + validate by cost~N regression
# ---------------------------------------------------------------------------
def _loglog_fit(N, cost):
    """Slope/intercept/R^2 of log10(cost) ~ log10(N)."""
    N = np.asarray(N, float)
    cost = np.asarray(cost, float)
    ok = (N > 0) & (cost > 0)
    x = np.log10(N[ok])
    y = np.log10(cost[ok])
    if x.size < 3 or np.ptp(x) == 0:
        return {"slope": None, "intercept": None, "r2": None, "n": int(x.size)}
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": int(x.size)}


def analyze(
    char_path: Path,
    dense_threshold: float | None,
    stiff_threshold: float | None,
    horizons_path: Path | None = None,
) -> dict:
    """Reclassify, partition by solver-relevant regime (sparse/dense stiff), regress cost~N.

    A row characterized from a partial analytical Jacobian — ``jacobian_method``
    ``native_analytical`` with ``analytical_jacobian_complete`` False, which only a report
    from before issue #512 can carry — is filed as ``incomplete_jacobian`` and kept out of
    every census: its density and spectrum describe a matrix with terms missing, and a
    Lorenz attractor came out "degenerate" that way. A row without a stiffness sweep
    (``ok_density_only``) is ``no_stiffness``, not degenerate: degenerate means a zero
    density, or a zero spectrum the sweep actually measured.
    """
    char = json.loads(Path(char_path).read_text())["results"]
    try:
        horizons = load_horizons(horizons_path)
    except FileNotFoundError as exc:
        print(f"[jac] {exc}; the cost~N regression will have no costs", file=sys.stderr)
        horizons = {}

    def maxre(r):
        return max([p.get("max_re", 0) for p in (r.get("per_time") or [])], default=0.0)

    pts = []
    incomplete: list[str] = []
    for r in char:
        if not str(r.get("status", "")).startswith("ok"):
            continue
        N = r.get("N")
        dens = r.get("density")
        if N is None or dens is None:
            continue
        if (
            r.get("jacobian_method") == "native_analytical"
            and r.get("analytical_jacobian_complete") is False
        ):
            incomplete.append(r["model_id"])
            continue
        h = horizons.get(r["model_id"], {})
        swept = bool(r.get("per_time"))
        degenerate = (dens == 0) or (swept and maxre(r) == 0)  # zero-Jacobian / trivial
        pts.append(
            {
                "model_id": r["model_id"],
                "N": N,
                "density": dens,
                "stiffness": r.get("stiffness_ratio_max"),
                "degenerate": degenerate,
                "oscillatory": bool(r.get("oscillatory")) and not degenerate,
                "swept": swept,
                "jacobian_method": r.get("jacobian_method"),
                "cost_sec": h.get("cost_sec"),
                "linear_solver": h.get("linear_solver"),
            }
        )

    deg = [p for p in pts if p["degenerate"]]
    osc = [p for p in pts if p["oscillatory"]]
    no_sweep = [p for p in pts if not p["swept"] and not p["degenerate"]]
    methods: dict[str, int] = {}
    for p in pts:
        methods[str(p["jacobian_method"])] = methods.get(str(p["jacobian_method"]), 0) + 1
    live = [
        p
        for p in pts
        if not p["degenerate"]
        and not p["oscillatory"]
        and p["stiffness"] is not None
        and np.isfinite(p["stiffness"])
    ]

    dvals = np.array([p["density"] for p in live])
    svals = np.array([p["stiffness"] for p in live])
    Nvals = np.array([p["N"] for p in live], float)

    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else None

    corr = float(np.corrcoef(np.log10(Nvals), dvals)[0, 1]) if len(live) > 2 else None

    dth = dense_threshold if dense_threshold is not None else float(np.median(dvals))
    sth = stiff_threshold if stiff_threshold is not None else 1e3

    # Regime is model x SOLVER: among stiff models (where the linear solve dominates),
    # sparse -> a sparse-aware solver (KLU) stays ~O(N) while dense-only tools pay O(N^3);
    # dense -> everyone pays O(N^3). Non-stiff -> explicit-ish, O(N) for all.
    for p in live:
        p["cls"] = (
            "nonstiff"
            if p["stiffness"] < sth
            else ("sparse_stiff" if p["density"] < dth else "dense_stiff")
        )

    def fit(group):
        c = [p for p in group if p.get("cost_sec")]
        return _loglog_fit([p["N"] for p in c], [p["cost_sec"] for p in c])

    classes = {}
    for c in ("sparse_stiff", "dense_stiff", "nonstiff"):
        g = [p for p in live if p["cls"] == c]
        Ns = np.array([p["N"] for p in g])
        solv = {}
        for p in g:
            solv[p["linear_solver"]] = solv.get(p["linear_solver"], 0) + 1
        classes[c] = {
            "n": len(g),
            "N_min": int(Ns.min()) if g else None,
            "N_med": int(np.median(Ns)) if g else None,
            "N_max": int(Ns.max()) if g else None,
            "solvers": solv,
            "cost_vs_N": fit(g),
        }

    ladder = sorted([p for p in live if p["cls"] == "sparse_stiff"], key=lambda p: -p["N"])
    ladder_rows = [
        {
            "model_id": p["model_id"],
            "N": p["N"],
            "density": round(p["density"], 3),
            "stiffness": p["stiffness"],
            "solver": p["linear_solver"],
        }
        for p in ladder
    ]

    print("\n===== Jacobian regime analysis (reframed: solver x N) =====")
    print(
        f"ok {len(pts)} -> degenerate {len(deg)}, genuine oscillatory {len(osc)}, "
        f"no stiffness sweep {len(no_sweep)}, live {len(live)} "
        f"| jacobian_method {methods} | incomplete analytical Jacobian (excluded) {len(incomplete)}"
    )
    corr_txt = "n/a" if corr is None else f"{corr:+.2f}"
    print(f"corr(log10 N, density) = {corr_txt}  (negative => big networks are sparse)")
    print(
        f"density median {np.median(dvals):.3f} | thresholds: dense>= {dth:.3f}, stiff>= {sth:g}"
    )
    for c in ("sparse_stiff", "dense_stiff", "nonstiff"):
        cc = classes[c]
        f = cc["cost_vs_N"]
        sl = "n/a" if f["slope"] is None else f"N^{f['slope']:.2f} (R^2={f['r2']:.2f})"
        print(
            f"  {c:13s} n={cc['n']:3d}  N[min/med/max]={cc['N_min']}/{cc['N_med']}/{cc['N_max']}"
            f"  cost~{sl}  solvers={cc['solvers']}"
        )
    print("\n  sparse-stiff ladder (BNGsim-KLU-advantage candidates), top by N:")
    for row in ladder_rows[:15]:
        print(
            f"    N={row['N']:4d} dens={row['density']:.3f} stiff={row['stiffness']:.1g} "
            f"solv={row['solver']}  {row['model_id'].split('/')[-1]}"
        )

    summary = {
        "counts": {
            "ok": len(pts),
            "degenerate": len(deg),
            "oscillatory": len(osc),
            "no_stiffness": len(no_sweep),
            "live": len(live),
            "incomplete_jacobian": len(incomplete),
        },
        "jacobian_method": methods,
        "corr_logN_density": corr,
        "thresholds": {"dense>=": dth, "stiff>=": sth},
        "density_pctiles": {q: pct(dvals, q) for q in (10, 25, 50, 75, 90)},
        "stiffness_log10_pctiles": {q: pct(np.log10(svals), q) for q in (10, 25, 50, 75, 90)},
        "classes": classes,
    }
    return {
        "summary": summary,
        "sparse_stiff_ladder": ladder_rows,
        "groups": {
            c: [p["model_id"] for p in live if p["cls"] == c]
            for c in ("sparse_stiff", "dense_stiff", "nonstiff")
        },
        "degenerate": [p["model_id"] for p in deg],
        "oscillatory": [p["model_id"] for p in osc],
        "no_stiffness": [p["model_id"] for p in no_sweep],
        "incomplete_jacobian": incomplete,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "jacobian_characterization.json")
    ap.add_argument("--limit", type=int, default=None, help="characterize only the first N models")
    ap.add_argument("--model", type=str, default=None, help="substring filter on model_id")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument(
        "--dense-time-samples",
        type=int,
        default=DENSE_TIME_SAMPLES,
        help="for N > FULL_GRID_MAX_N: number of log-spaced trajectory "
        "points at which stiffness is evaluated (default: %(default)s)",
    )
    ap.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="optional: skip models whose report n_species exceeds this "
        "(the native analytical Jacobian handles large N fine; default: no skip)",
    )
    ap.add_argument(
        "--analyze",
        nargs="?",
        const=True,
        default=False,
        help="analysis mode: partition + cost~N regression from a "
        "characterization JSON (default: --out path)",
    )
    ap.add_argument("--dense-threshold", type=float, default=None)
    ap.add_argument("--stiff-threshold", type=float, default=None)
    ap.add_argument(
        "--horizons",
        type=Path,
        default=None,
        help="the parity gate's report (bng_ode_run.py's runs/report_ode.json) each model's "
        f"horizon and tolerances are taken from (default: {DEFAULT_HORIZONS}); the run "
        "refuses to start without one (issue #509)",
    )
    args = ap.parse_args()

    if args.analyze:
        path = args.out if args.analyze is True else Path(args.analyze)
        res = analyze(path, args.dense_threshold, args.stiff_threshold, args.horizons)
        ap_out = path.with_name(path.stem + "_analysis.json")
        ap_out.write_text(json.dumps(res, indent=1))
        print(f"[jac] wrote {ap_out}")
        return 0

    try:
        horizons = load_horizons(args.horizons)
    except FileNotFoundError as exc:
        print(f"[jac] refusing to run: {exc}", file=sys.stderr)
        return 2
    bng2_pl = bc.resolve_bng2_pl(os.environ.get("BNGPATH") or os.environ.get("BNG2_PL"))
    jobs = load_ode_jobs()
    ids = [j["model_id"] for j in jobs]
    if args.model:
        ids = [i for i in ids if args.model in i]
    if args.limit:
        ids = ids[: args.limit]

    print(f"[jac] {len(ids)} ODE models | bng2_pl={bng2_pl}", flush=True)
    rows = []
    t0 = time.perf_counter()
    for k, mid in enumerate(ids, 1):
        h = horizons.get(mid, {})
        if args.max_n is not None and (h.get("n_species") or 0) > args.max_n:
            rows.append({"model_id": mid, "status": "skipped_too_large", "N": h.get("n_species")})
            print(
                f"[{k:3d}/{len(ids)}] skipped_too_large  N={h.get('n_species')} {mid}", flush=True
            )
            continue
        r = characterize_model(
            mid, h, bng2_pl, timeout=args.timeout, dense_time_samples=args.dense_time_samples
        )
        rows.append(r)
        tag = r.get("status")
        extra = ""
        if r.get("N") is not None:
            extra = (
                f"N={r['N']} dens={r.get('density', float('nan')):.3f} "
                f"stiff[max/med]={r.get('stiffness_ratio_max', float('nan')):.3g}/"
                f"{r.get('stiffness_ratio_median', float('nan')):.3g} "
                f"npts={r.get('n_time_points', '-')}"
                f"{'/skip' + str(r['n_time_points_skipped']) if r.get('n_time_points_skipped') else ''} "
                f"ic0={r.get('n_seed_nonzero', '-')} par={r.get('n_independent_parameters', '-')} "
                f"{'OSC ' if r.get('oscillatory') else ''}"
            )
        print(f"[{k:3d}/{len(ids)}] {tag:16s} {extra}{mid}", flush=True)

    out = {
        "_meta": {
            "generator": "jacobian_characterization.py",
            "bngsim_version": __import__("bngsim").__version__,
            "n_models": len(rows),
            "horizons": str(args.horizons or (HERE / DEFAULT_HORIZONS)),
            "n_models_with_horizon": sum(1 for i in ids if gate_run_settings(horizons.get(i))),
            "params": {
                "DENSITY_SAMPLES": DENSITY_SAMPLES,
                "NONZERO_REL_TOL": NONZERO_REL_TOL,
                "OSC_DAMPING_CUT": OSC_DAMPING_CUT,
                "OSC_NEARZERO_BAND": OSC_NEARZERO_BAND,
                "FULL_GRID_MAX_N": FULL_GRID_MAX_N,
                "DENSE_TIME_SAMPLES": args.dense_time_samples,
                "EIG_MAX_N": EIG_MAX_N,
                "AVOGADRO_PARAM_NAMES": sorted(AVOGADRO_PARAM_NAMES),
                "AVOGADRO_MIN_VALUE": AVOGADRO_MIN_VALUE,
                "UNIT_GEOMETRY_PARAM_NAMES": sorted(UNIT_GEOMETRY_PARAM_NAMES),
            },
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    ok = sum(1 for r in rows if str(r.get("status")).startswith("ok"))
    print(
        f"[jac] wrote {args.out}  ({ok}/{len(rows)} characterized, "
        f"{out['_meta']['elapsed_sec']}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
