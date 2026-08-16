#!/usr/bin/env python3
"""Run the amici_parity FORWARD-SENSITIVITY sweep (bngsim vs AMICI) -> a ``_core`` report.

The sensitivity sibling of ``amici_run.py``. That job compares state trajectories
``x(t)``; this one compares the forward-sensitivity tensor ``dx_i(t)/dp_j`` that
both engines get from a coupled CVODES extended-ODE solve, and times the warm
per-solve cost of producing it. Same corpus, same ``_core.differ`` oracle, same
verdict taxonomy, same report shape — only the quantity under test changes.

    both ran, tensors within tolerance .......... PASS
    both ran, metric over tolerance / non-finite  DIFF
    both ran, species/time grids disjoint ....... DIFF (loud, value=inf)
    no shared differentiable parameter ........... BAD_TEST (nothing to compare)
    bngsim DECLARED it cannot differentiate ..... UNSUPPORTED (clean refusal; non-scoring)
    bngsim raised, AMICI ran .................... EXCEPTION (actionable bngsim bug)
    bngsim ran, AMICI raised .................... REFERENCE_FAILED (no oracle; non-scoring)
    both raised .................................. BAD_TEST (no signal; non-scoring)
    wall-clock cap exceeded ...................... TIMEOUT

Two of those verdicts can also be reached by an AUTHORED disposition rather than
by the engines (issue #380, ``amici_dispositions.py``): a DIFF attributed with
third-oracle evidence to a defect in AMICI's own answer becomes REFERENCE_FAILED
with ``reference_refusal=invalid_result``, and a DIFF that is an artifact of
comparing at a cell no oracle can resolve becomes PASS. Both record their reason
on the row, apply only while the row is genuinely a DIFF, and are flagged STALE
the moment it agrees on its own.

UNSUPPORTED is the forward-sensitivity peer of rr_parity's ``SsaValidationError``
row: bngsim inspected the model, found a construct whose derivative it cannot
produce (a non-differentiable event crossing time, a rate law codegen cannot
differentiate to closed form) and declined rather than return wrong numbers. It
is recognized BY TYPE — ``bngsim.SensitivityUnsupportedError`` — never by message
text, so rewording a refusal cannot silently sink it back into EXCEPTION.

One job per (model, sensitivity method). Both engines are pinned to the SAME
CVODES corrector method within a job — ``staggered`` (CVODES'/bngsim's default)
and ``simultaneous`` (AMICI's compiled-in default) — so a timing pair isolates the
engine difference, and running both separates the engine effect from the method
effect. The AMICI compile is shared across the two methods via the on-disk cache:
the first job for a model pays it, the second is load-only.

Cost warning: the sensitivity build emits the ``dxdotdp`` / sensitivity-RHS C++ on
top of the pure-ODE body, so cold compiles are materially heavier than
``amici_run.py``'s, and the solve integrates ``n_species*(Np+1)`` states instead
of ``n_species``. Its cache (``amici_sens_cache/``) is separate from the ODE one
and shares nothing with it. Expect the first full-corpus pass to be long; re-runs
are load-only and resumable (kill/rerun skips what is already built).

STATE PARITY IS CHECKED TOO, and separately: a sensitivity comparison is
meaningless if the two engines are not on the same trajectory. The state verdict
is reported in its own field rather than folded into the headline metric, so a
sensitivity DIFF on a model whose states already disagree is never mistaken for a
sensitivity-specific bug.

Usage:
    cd bngsim && .venv/bin/python parity_checks/amici_parity/amici_sens_run.py --workers 4
    # full corpus (the reportable run)
    .venv/bin/python parity_checks/amici_parity/amici_sens_run.py \\
        --manifest parity_checks/rr_parity/ode_jobs.json --workers 4 --timeout 900

Output: runs/report_sens.json (a _core report; runs/ is gitignored).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RR_PARITY = HERE.parent / "rr_parity"  # the SBML corpus + the full ODE manifest
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import _amici_sens as asens  # noqa: E402
import amici_dispositions  # noqa: E402
from _core import (  # noqa: E402
    FAILING,
    JobResult,
    Outcome,
    capture_exception,
    differ,
    read_manifest,
    tally,
    versions,
    write_report,
)
from _core.versions import git_rev  # noqa: E402

LOCAL_JOBS = HERE / "amici_ode_jobs.json"
RR_ODE_JOBS = RR_PARITY / "ode_jobs.json"
# Higher than the ODE job's 180s: the sensitivity build is a bigger compile and the
# coupled solve carries (Np+1)x the states.
DEFAULT_SENS_TIMEOUT = 600.0


SENS_REGIME = "sens"


def _job_tol(job) -> dict | None:
    """The engine-agnostic ``tol`` override (ill-conditioned IVP), or None. As in
    ``amici_run.py``, the RoadRunner-calibrated overrides are NOT applied here —
    an AMICI-calibrated disposition lives in ``amici_dispositions.py`` instead."""
    for o in job.overrides:
        if o.field == "tol":
            return {"rtol": float(o.value["rtol"]), "atol": float(o.value["atol"])}
    return None


DEFAULT_RTOL = asens.DEFAULT_RTOL
DEFAULT_ATOL = asens.DEFAULT_ATOL


def _compare(bn, am, param_values=None, atol=None) -> dict:
    """Compare one (state, sensitivity) pair. Returns a result-fragment dict.

    ``bn``/``am`` are ``(t, x, sx, species_names)``. The parameter axis needs no
    alignment — both engines were handed the identical shared id list in the
    identical order — but the species axis does, since neither engine's column
    order is assumed (and AMICI's state set can be a subset).

    ``param_values`` / ``atol`` enable the solver-resolution noise floor (see
    ``_amici_sens.sensitivity_noise_mask``): a sensitivity that is identically
    zero makes the relative metric saturate at 1.0 against the other engine's
    integration noise, which nothing scale-relative can forgive.
    """
    bn_t, bn_x, bn_sx, bn_names = bn
    am_t, am_x, am_sx, am_names = am

    if bn_t.shape != am_t.shape or not np.allclose(bn_t, am_t, rtol=0, atol=1e-9):
        return {
            "status": "diff",
            "value": float("inf"),
            "comment": f"time grid mismatch (bn n={bn_t.shape}, amici n={am_t.shape})",
        }
    align = asens.align_common(bn_names, am_names)
    if align is None:
        return {
            "status": "diff",
            "value": float("inf"),
            "comment": f"disjoint species sets: bn={bn_names[:4]} amici={am_names[:4]}",
        }
    bn_idx, am_idx, common = align

    # State parity first — it qualifies the sensitivity verdict below.
    sv = differ.deterministic_verdict(bn_x[:, bn_idx], am_x[:, am_idx])

    # Sensitivity parity: the headline.
    v = asens.sens_verdict(
        bn_sx[:, bn_idx, :], am_sx[:, am_idx, :], param_values=param_values, atol=atol
    )

    n_p = bn_sx.shape[2]

    # Degeneracy witnesses (issue #328). A job can be reported PASS when the
    # WHOLE sensitivity tensor lies below the magnitude either solver can
    # resolve: nothing was meaningfully compared, but the row is indistinguishable
    # from a real pass. differ's significance gate cannot catch it — that gate is
    # relative to the file-wide peak, and when everything is tiny there is no
    # larger peak to be judged against.
    #
    # These make the census computable from the report instead of from a corpus
    # re-run. Note the OBVIOUS proxy does not work: n_below_noise_floor/n_cells
    # hits 100% for excellent agreement too, because the mask keys on |bn-am|,
    # not on magnitude — MODEL7909395757 has 3636/3636 cells inside the floor at
    # max|sx| = 0.6. Magnitude has to be recorded directly, and PER COLUMN: a
    # single tiny-valued parameter would otherwise inflate a global floor and
    # mark a live model degenerate (see sens_resolution_floors).
    sx_used = bn_sx[:, bn_idx, :]
    sx_peak = float(np.nanmax(np.abs(sx_used))) if sx_used.size else 0.0
    x_used = bn_x[:, bn_idx]
    state_span = float(np.nanmax(np.nanmax(x_used, axis=0) - np.nanmin(x_used, axis=0)))
    n_resolvable = asens.resolvable_param_columns(sx_used, param_values, atol)

    # Issue #316 — the comment reports what the floor RESCUED, not how much of
    # the tensor sits inside it. The old "noise N" was the latter under a name
    # that read as the former, so a row could say "noise 4820" while the floor
    # had changed nothing about the verdict.
    comment = (
        f"{len(common)} sp x {n_p} par; fail {v['n_fail']}/{v['n_cells']} "
        f"(hard {v['n_hard_fail']}, soft {v['n_soft_fail']}, "
        f"forgiven {v['budget_forgiven']}, noise-rescued {v['n_noise_rescued']}"
        f"{' DECISIVE' if v['noise_decisive'] else ''}); "
        f"state {'ok' if sv['passed'] else 'DIFF'}"
    )
    if n_resolvable == 0:
        comment += "; DEGENERATE: no parameter column resolvable"
    return {
        "status": "pass" if v["passed"] else "diff",
        "value": v["max_rel"],
        "comment": comment,
        "state_passed": bool(sv["passed"]),
        "state_max_rel": float(sv["max_rel"]),
        "n_common_species": len(common),
        "n_params": int(n_p),
        # issue #316 — three numbers, because the audit question needs three.
        # How much of the tensor is under solver resolution (data); how much of
        # the verdict rested on the floor (verdict); and whether the floor was
        # what made this row a PASS (decision).
        "n_below_noise_floor": int(v["n_below_noise_floor"]),
        "n_noise_rescued": int(v["n_noise_rescued"]),
        "noise_decisive": bool(v["noise_decisive"]),
        # issue #328 — the two numbers a vacuous-pass census needs.
        "max_abs_sx": sx_peak,
        "n_resolvable_params": n_resolvable,
        "state_span": state_span,
    }


def make_specs(
    jobs,
    methods: list[str],
    *,
    rtol: float,
    atol: float,
    timeout: float | None,
    param_cap: int,
    param_budget: int,
    config_env: dict,
    max_steps: int = asens.DEFAULT_SENS_MAX_STEPS,
) -> tuple[list[dict], int]:
    """Expand the manifest jobs into worker specs. ``(specs, n_tol_overridden)``.

    The iteration order is **method-major** — every model under method A, then
    every model under method B — and that is load-bearing, not cosmetic. Both
    methods of a model deliberately share one compiled AMICI extension (the
    compile is the dominant cost and the corrector method is a solver setting, not
    a codegen one). Emitting a model's two methods adjacent would put them in
    flight simultaneously under any worker count > 1, so several workers would
    race to build the same cache key; the build commits by atomic rename so the
    result is still *correct*, but every loser throws away a full C++ compile.
    Method-major means the second method's jobs start only once the first pass has
    populated the cache, making them load-only.

    ``max_steps`` rides on the spec rather than being read from a module constant
    inside the worker, so the number a row was solved under is a property of the
    job the report can carry — and so a test can pin that both engines got the
    same one (issue #339).

    Extracted from ``main()`` so this ordering property is directly testable
    without running a sweep.
    """
    specs: list[dict] = []
    n_tol_ov = 0
    for meth in methods:
        for j in jobs:
            params = dict(j.params)
            tol_ov = _job_tol(j)
            if tol_ov:
                params["rtol"], params["atol"] = tol_ov["rtol"], tol_ov["atol"]
                if meth == methods[0]:  # count each model once, not once per method
                    n_tol_ov += 1
            else:
                params["rtol"], params["atol"] = rtol, atol
            cap = timeout if timeout is not None else DEFAULT_SENS_TIMEOUT
            specs.append(
                {
                    "key": f"{j.model_id}:sens:{meth}",
                    "model_id": j.model_id,
                    "method": f"sens/{meth}",
                    "sens_method": meth,
                    "metric": "max_rel_err",
                    "tol": differ.REL_TOL,
                    "xml": str(asens.model_path(RR_PARITY, j)),
                    "params": params,
                    "cap": float(cap),
                    "param_cap": int(param_cap),
                    "param_budget": int(param_budget),
                    "max_steps": int(max_steps),
                    "config_env": dict(config_env),
                }
            )
    return specs, n_tol_ov


def _classify_failure(
    bn_exc: str,
    am_exc: str,
    bn_unsupported: bool = False,
    bn_cls: str = "",
    am_cls: str = "",
) -> tuple[str, str, str]:
    """Attribute a failed job to an engine. ``(status, exception_text, exception_class)``.

    ``bn_unsupported`` marks a DECLARED bngsim refusal (a
    ``SensitivityUnsupportedError``: the model carries a construct bngsim states
    it cannot differentiate). It wins over every other bucket, including the
    both-raised BAD_TEST: the declaration is a fact about bngsim and this model,
    and stays true whatever AMICI did with it. Both buckets are non-scoring, so
    the choice costs no signal and gains a named one — UNSUPPORTED rows are
    countable as "bngsim's declared sensitivity gap", which BAD_TEST ("nothing
    to compare") is not. The AMICI text is kept in the message either way.

    ``bn_cls`` / ``am_cls`` are the ``CapturedException.cls`` keys for the same
    two raises (issue #324). The class tracks the text exactly — whichever sides
    the text keeps, the class keeps, joined the same way — so a census can group
    on it without re-parsing a capped message. Optional so a caller that has no
    classes still gets the old two thirds of the answer, as ``""``.
    """
    if bn_unsupported:
        if am_exc:
            return "unsupported", f"{bn_exc} || {am_exc}", _join_cls(bn_cls, am_cls)
        return "unsupported", bn_exc, bn_cls
    if bn_exc and am_exc:
        return "bad_test", f"{am_exc} || {bn_exc}", _join_cls(am_cls, bn_cls)
    if am_exc:
        return "reference_failed", am_exc, am_cls
    return "exception", bn_exc, bn_cls


def _join_cls(first: str, second: str) -> str:
    """Join two grouping keys the way their texts are joined, skipping blanks."""
    return " || ".join(c for c in (first, second) if c)


# Shared with amici_run.py — see `_amici_common.is_declared_refusal` for why the
# match is by exception TYPE and never by message text. Kept under this name so
# the module reads the same as before it was hoisted.
_is_declared_refusal = asens.is_declared_refusal


def _worker(spec: dict, q) -> None:
    """Run BOTH engines' forward sensitivities for one (model, method) job."""
    warmup = asens.measure_warmup()
    asens.set_amici_quiet()
    for _k, _v in spec.get("config_env", {}).items():
        os.environ[_k] = _v

    p = spec["params"]
    xml = spec["xml"]
    method = spec["sens_method"]
    res = {k: spec[k] for k in ("key", "model_id", "method")}
    res.update(
        {
            "metric": spec["metric"],
            "tol": spec["tol"],
            "value": None,
            "comment": "",
            "exception": "",
            "sens_method": method,
        }
    )

    # --- AMICI build FIRST: its free-parameter ids define the shared list, so the
    # bngsim side cannot even be configured until the reference model exists. A
    # build failure is therefore REFERENCE_FAILED with bngsim untested. ---
    built = None
    am_exc = ""
    am_cls = ""
    am_full = ""
    am_wall = 0.0
    try:
        t0 = time.perf_counter()
        am_ids, built = asens.amici_free_parameter_ids(xml)
        am_wall = time.perf_counter() - t0
    except Exception as exc:
        cap = capture_exception("amici-build", exc)
        res["status"] = "reference_failed"
        res["exception"] = cap.text
        res["exception_class"] = cap.cls
        # Issue #324 — classify from the FULL message, here, rather than from the
        # capped one in the parent. The keyword that names an AMICI refusal
        # subclass is not guaranteed to sit inside the head budget.
        res["reference_refusal"] = asens.classify_reference_refusal(cap.full)
        res["wall_sec"] = round(time.perf_counter() - t0, 3)
        res["timing"] = {"warmup": warmup}
        q.put(res)
        return

    # --- Negotiate the shared, capped parameter list. ---
    try:
        import bngsim._sbml_loader as sbml_loader

        _src = Path(xml).read_text() if Path(xml).exists() else xml
        _m = sbml_loader.load_sbml_string(_src)
        # The cap is derived PER MODEL from the coupled-state budget (issue #331):
        # cost scales with n_species*(Np+1), so a flat Np ceiling over-spends on
        # big models and under-samples small ones. n_species is only knowable
        # here, once the model is loaded.
        eff_cap = asens.budget_cap(len(_m.species_names), spec["param_budget"], spec["param_cap"])
        # Issue #329 — a parameter name that is also a FUNCTION name is a slot the
        # engine rewrites from that function's expression before every derivative
        # evaluation (an <assignmentRule> target, in SBML terms), so bngsim refuses
        # the column rather than answering the identically-zero one it used to.
        # Same set the Simulator refuses (`Simulator._function_backed_params`),
        # spelled the same way so the harness cannot ask for a column the library
        # would reject nor drop one it would answer.
        _fn_backed = _m._internal_param_names() & set(_m.function_names)
        shared_ids, bn_by_id, n_cand = asens.shared_sensitivity_params(
            list(_m.param_names),
            _m.compartment_size_params,
            am_ids,
            cap=eff_cap,
            bn_function_backed_params=_fn_backed,
        )
        res["param_cap_effective"] = eff_cap
        res["n_species_model"] = len(_m.species_names)
        # Parameter VALUES, in the shared order — the noise floor needs |p_j| to
        # convert the solver's state-space atol into a resolvable magnitude for
        # s_ij = dx_i/dp_j. Read here, while the model is loaded for the name
        # negotiation, rather than reloading the SBML a third time.
        param_values = [float(_m.get_param(bn_by_id[i])) for i in shared_ids]
        del _m
    except Exception as exc:
        # A DECLARED refusal here is almost always UnderSpecifiedModelError: the
        # model reads a symbol it never defines, so the load itself refuses
        # (issue #323). AMICI accepts the same models by defaulting the symbol to
        # 0, which is why this shows up as bngsim-only. It is a documented
        # refusal, not a bug — UNSUPPORTED, not EXCEPTION.
        #
        # This site is issue #324's own reproducer: #323's message enumerates the
        # under-specified parameters BEFORE saying what is wrong with them, so on
        # a model with a long list the head cut kept only names.
        cap = capture_exception("bngsim-params", exc)
        res["status"] = "unsupported" if _is_declared_refusal(exc) else "exception"
        res["exception"] = cap.text
        res["exception_class"] = cap.cls
        res["wall_sec"] = round(am_wall, 3)
        q.put(res)
        return

    if not shared_ids:
        # Nothing differentiable in common — no oracle exists for this model. Not a
        # bngsim bug and not an AMICI refusal, so neither EXCEPTION nor
        # REFERENCE_FAILED would be honest; BAD_TEST is the non-scoring bucket.
        res["status"] = "bad_test"
        res["exception"] = (
            f"no shared differentiable parameter (amici free={len(am_ids)}, "
            f"bngsim-eligible shared={n_cand})"
        )
        # Not a raise, but it is a row a census wants to group, and the counts in
        # the text above make it un-groupable without a key of its own (#324).
        res["exception_class"] = "shared-params:none"
        res["wall_sec"] = round(am_wall, 3)
        res["timing"] = {"warmup": warmup}
        q.put(res)
        return

    res["n_params"] = len(shared_ids)
    res["n_param_candidates"] = n_cand
    # Issue #339 — read ONCE and handed to both engines below, so no future edit
    # can raise the budget for one and not the other. Recorded on the row for the
    # same reason `n_params` is: a timing number is uninterpretable without it,
    # and a REFERENCE_FAILED row needs to say what budget it failed under.
    max_steps = int(spec.get("max_steps", asens.DEFAULT_SENS_MAX_STEPS))
    res["max_steps"] = max_steps

    # --- bngsim side ---
    bn = None
    bn_exc = ""
    bn_cls = ""
    bn_unsupported = False
    bn_timing = None
    bn_wall = 0.0
    try:
        t0 = time.perf_counter()
        out = asens.bn_sens(
            xml,
            p["t_start"],
            p["t_end"],
            p["n_points"],
            p["rtol"],
            p["atol"],
            [bn_by_id[i] for i in shared_ids],
            method,
            max_steps=max_steps,
        )
        bn_wall = time.perf_counter() - t0
        bn, bn_timing = out[:4], out[4]
    except Exception as exc:
        bn_unsupported = _is_declared_refusal(exc)
        cap = capture_exception("bngsim", exc)
        bn_exc, bn_cls = cap.text, cap.cls

    # --- AMICI side (over the model built above) ---
    am = None
    am_timing = None
    try:
        t0 = time.perf_counter()
        out = asens.amici_sens(
            built,
            p["t_start"],
            p["t_end"],
            p["n_points"],
            p["rtol"],
            p["atol"],
            shared_ids,
            method,
            max_steps=max_steps,
        )
        am_wall += time.perf_counter() - t0
        am, am_timing = out[:4], out[4]
    except Exception as exc:
        cap = capture_exception("amici", exc)
        am_exc, am_cls, am_full = cap.text, cap.cls, cap.full

    res["wall_sec"] = round(bn_wall + am_wall, 3)
    if am is not None:
        res["am_finite"] = bool(np.isfinite(am[2]).all())

    timing = {}
    if bn_timing:
        timing["bngsim"] = bn_timing
    if am_timing:
        timing["amici"] = am_timing
    if warmup:
        timing["warmup"] = warmup
    if timing:
        res["timing"] = timing

    if bn_exc or am_exc:
        res["status"], res["exception"], res["exception_class"] = _classify_failure(
            bn_exc, am_exc, bn_unsupported, bn_cls, am_cls
        )
        if res["status"] == "reference_failed":
            # Issue #324 — the full AMICI text, not the capped one the parent
            # would otherwise re-read out of the row.
            res["reference_refusal"] = asens.classify_reference_refusal(am_full)
        q.put(res)
        return

    try:
        res.update(_compare(bn, am, param_values=param_values, atol=p["atol"]))
    except Exception as exc:
        cap = capture_exception("compare", exc)
        res["status"] = "exception"
        res["exception"] = cap.text
        res["exception_class"] = cap.cls
    q.put(res)


_OUTCOME = {
    "pass": Outcome.PASS,
    "diff": Outcome.DIFF,
    "exception": Outcome.EXCEPTION,
    "unsupported": Outcome.UNSUPPORTED,
    "reference_failed": Outcome.REFERENCE_FAILED,
    "bad_test": Outcome.BAD_TEST,
    "timeout": Outcome.TIMEOUT,
    "dead": Outcome.EXCEPTION,
}


def _make_progress(checkpoint_path: Path | None = None):
    def _progress(finished: int, total: int, res: dict) -> None:
        if checkpoint_path is not None:
            try:
                with open(checkpoint_path, "a") as _ck:
                    _ck.write(json.dumps(res, default=str) + "\n")
            except Exception:
                pass
        st = res.get("status", "?")
        tag = {
            "pass": "PASS",
            "diff": "DIFF",
            "exception": "ERR ",
            "unsupported": "UNSUP",
            "reference_failed": "REFFAIL",
            "bad_test": "BADTEST",
            "timeout": "SLOW",
            "dead": "DEAD",
        }.get(st, st)
        extra = ""
        if st in ("pass", "diff") and res.get("value") is not None:
            extra = f" {res['metric']}={res['value']:.3g} (Np={res.get('n_params', '?')})"
        elif st in ("exception", "dead", "unsupported", "reference_failed", "bad_test"):
            extra = f" {(res.get('exception') or 'died')[:64]}"
        elif st == "timeout":
            extra = f" >{res.get('cap')}s"
        print(
            f"  [{finished}/{total}] {tag} {res['model_id']} "
            f"({res.get('sens_method', '?')}){extra}",
            flush=True,
        )

    return _progress


_BNGSIM_CONFIG_ENV_VARS = (
    "BNGSIM_CODEGEN_JIT",
    "BNGSIM_LAPACK_DENSE",
    "BNGSIM_ANALYTICAL_FUNCTIONAL_JAC",
    "BNGSIM_NO_CODEGEN",
)

_CONFIG_COMBOS: dict[str, dict] = {
    "auto": {"env": {}},
    "mir": {"env": {"BNGSIM_CODEGEN_JIT": "mir"}},
    "fd-jac": {"env": {"BNGSIM_ANALYTICAL_FUNCTIONAL_JAC": "0"}},
}


def _bngsim_config_meta(args) -> dict:
    spec = _CONFIG_COMBOS[args.config]
    env = dict.fromkeys(_BNGSIM_CONFIG_ENV_VARS)
    env.update(spec["env"])
    return {
        "combo": args.config,
        "bngsim_method": "ode+forward-sensitivity",
        "codegen_threshold": int(os.environ.get("BNGSIM_CODEGEN_THRESHOLD", "256")),
        "env": env,
        "rtol": args.rtol,
        "atol": args.atol,
        "param_cap": args.param_cap,
        "param_budget": args.param_budget,
        "max_steps": args.max_steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="", help="_core report path (default runs/report_sens.json)")
    ap.add_argument(
        "--manifest",
        default="",
        help="Job manifest (default: amici_ode_jobs.json if present, else "
        "rr_parity/ode_jobs.json, the full 1323-model corpus).",
    )
    ap.add_argument("--config", choices=sorted(_CONFIG_COMBOS), default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--methods",
        default=",".join(asens.SENS_METHODS),
        help="Comma-separated CVODES corrector methods, both engines pinned per job "
        f"(default: {','.join(asens.SENS_METHODS)}).",
    )
    ap.add_argument(
        "--param-budget",
        type=int,
        default=asens.DEFAULT_PARAM_BUDGET,
        help="Ceiling on the COUPLED SYSTEM SIZE n_species*Np, from which each "
        "model's parameter count is derived (issue #331). Cost scales with that "
        "product, not with Np alone, so a flat cap over-spends on big models and "
        f"under-samples small ones (default {asens.DEFAULT_PARAM_BUDGET}, "
        "0 = no budget).",
    )
    ap.add_argument(
        "--param-cap",
        type=int,
        default=0,
        help="Optional ADDITIONAL hard ceiling on parameters per model, applied "
        "on top of --param-budget (default 0 = none; the budget governs). Set to "
        f"{asens.DEFAULT_PARAM_CAP} to reproduce the pre-#331 flat cap.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=asens.DEFAULT_SENS_MAX_STEPS,
        help="Per-solve integrator step budget, applied SYMMETRICALLY to both "
        f"engines (default {asens.DEFAULT_SENS_MAX_STEPS}, issue #339). Both "
        "default to 10,000 on their own, which #331's higher Np exhausted on 8 "
        "models — 6 of them recover here and 2 fail at any budget (a NaN in "
        "AMICI's sensitivity RHS). The per-job --timeout remains the real bound "
        "on a runaway.",
    )
    ap.add_argument("--checkpoint", default="")
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Per-job wall-clock cap (s); default {DEFAULT_SENS_TIMEOUT}s.",
    )
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    ap.add_argument("--limit", type=int, default=0, help="Max models after filtering (0=all).")
    ap.add_argument("--models", default="", help="Comma-separated model_id filter.")
    ap.add_argument("--include", default="")
    ap.add_argument("--exclude", default="")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in asens.SENS_METHODS]
    if bad:
        sys.exit(f"unknown sensitivity method(s) {bad}; choose from {asens.SENS_METHODS}")

    if args.manifest:
        manifest = Path(args.manifest).resolve()
    elif LOCAL_JOBS.exists():
        manifest = LOCAL_JOBS
    else:
        manifest = RR_ODE_JOBS
    if not manifest.exists():
        sys.exit(f"missing manifest {manifest}; run build_amici_jobs.py.")
    _meta, rjobs = read_manifest(manifest)
    jobs = asens.load_and_filter(rjobs, args, suite_dir=RR_PARITY)
    if not jobs:
        sys.exit("no jobs after filtering.")

    missing = [j.model_id for j in jobs if not asens.model_path(RR_PARITY, j).exists()]
    if missing:
        sys.exit(
            f"{len(missing)} job(s) have no vendored SBML (e.g. {missing[:3]}). "
            "Run `python rr_parity/materialize.py` to place the gitignored model tree."
        )

    combo_spec = _CONFIG_COMBOS[args.config]
    specs, n_tol_ov = make_specs(
        jobs,
        methods,
        rtol=args.rtol,
        atol=args.atol,
        timeout=args.timeout,
        param_cap=args.param_cap,
        param_budget=args.param_budget,
        config_env=dict(combo_spec["env"]),
        max_steps=args.max_steps,
    )

    ver = versions.stamp("amici")
    ver["sundials"] = asens.sundials_version()
    print("=" * 72)
    print("  bngsim vs AMICI — SBML forward-sensitivity parity (amici_parity)")
    print("=" * 72)
    print(
        f"  models: {len(jobs)}   methods: {','.join(methods)}   "
        f"jobs: {len(specs)}   workers: {args.workers}"
    )
    print(f"  bngsim {ver['bngsim']}   amici {ver['amici']}   manifest {manifest.name}")
    print(
        f"  param budget: {args.param_budget or 'none'} coupled states"
        f"   extra cap: {args.param_cap or 'none'}   "
        f"tol rtol={args.rtol:g} atol={args.atol:g}   "
        f"max_steps {args.max_steps} (both engines)"
    )
    print()

    t0 = time.perf_counter()
    raw = asens.schedule(
        specs,
        _worker,
        workers=args.workers,
        timeout_of=lambda s: s["cap"],
        on_done=_make_progress(Path(args.checkpoint).resolve() if args.checkpoint else None),
    )
    elapsed = time.perf_counter() - t0

    results = []
    now = _dt.datetime.now().isoformat(timespec="seconds")
    n_state_diff = 0
    # AMICI-calibrated dispositions (issue #380): a row whose DIFF was attributed
    # to a defect in AMICI's own answer, or to a cell no oracle can resolve. Read
    # from the module rather than from the manifest, because the manifest is shared
    # with rr_parity and an AMICI-calibrated entry must not leak into that suite.
    dispositions = amici_dispositions.dispositions_for_regime(SENS_REGIME)
    disposition_states = Counter()
    for r in raw:
        outcome = _OUTCOME.get(r.get("status"), Outcome.EXCEPTION)
        comment = r.get("comment", "")
        if r.get("status") == "timeout":
            comment = f"killed at {r.get('cap')}s wall cap"
        elif r.get("status") == "dead":
            comment = f"worker died (exit={r.get('exitcode')})"
        refusal = None
        if outcome == Outcome.REFERENCE_FAILED:
            # Issue #324 — prefer the worker's verdict, which saw the full AMICI
            # message. Falling back to the stored text keeps a resumed run whose
            # checkpoint predates this field classifying as it always did.
            refusal = r.get("reference_refusal") or asens.classify_reference_refusal(
                r.get("exception", "")
            )
            comment = (
                f"{comment} | amici refusal={refusal}" if comment else f"amici refusal={refusal}"
            )
        # Applied AFTER the auto-derived refusal, so an INVALID_REFERENCE row —
        # which has no AMICI exception to classify — sets its own
        # ``invalid_result`` class instead of being handed a default one, and so a
        # row the disposition does not apply to keeps whatever the exception said.
        disp = dispositions.get(r["model_id"])
        if disp:
            outcome, comment, disp_refusal, state = amici_dispositions.apply_disposition(
                disp, outcome, r.get("value"), comment
            )
            disposition_states[f"{disp['kind']}:{state}"] += 1
            if disp_refusal:
                refusal = disp_refusal
        if r.get("state_passed") is False:
            n_state_diff += 1
        wall = r.get("wall_sec") or 0.0
        results.append(
            JobResult(
                model_id=r["model_id"],
                method=r["method"],
                reference_engine="amici",
                outcome=str(outcome),
                metric=r.get("metric"),
                value=r.get("value"),
                tol=r.get("tol"),
                exception=r.get("exception", ""),
                exception_class=r.get("exception_class"),
                wall_sec=round(wall, 3) if wall else None,
                timestamp=now,
                versions=ver,
                comment=comment,
                reference_refusal=refusal,
                timing=r.get("timing"),
                extra={
                    k: r[k]
                    for k in (
                        "sens_method",
                        "n_params",
                        "n_param_candidates",
                        "n_common_species",
                        "state_passed",
                        "state_max_rel",
                        # issue #316 — what the solver-resolution floor is doing:
                        # how much of the tensor sits under it, how much of the
                        # verdict it rescued, and whether it decided the row.
                        "n_below_noise_floor",
                        "n_noise_rescued",
                        "noise_decisive",
                        # issue #328 — degeneracy witnesses, so a vacuous-pass
                        # census is a query over the report rather than a re-run.
                        "param_cap_effective",
                        "n_species_model",
                        # issue #339 — the step budget the row was solved under,
                        # the same number on both sides.
                        "max_steps",
                        "max_abs_sx",
                        "n_resolvable_params",
                        "state_span",
                    )
                    if k in r
                },
            )
        )

    results.sort(key=lambda x: (x.model_id, x.method))
    counts = tally(r.outcome for r in results)
    refusal_breakdown = dict(
        Counter(r.reference_refusal for r in results if r.reference_refusal).most_common()
    )
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        suffix = "" if args.config == "auto" else f"__{args.config}"
        out_path = HERE / "runs" / f"report_sens{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "suite": "amici_parity",
        "reference_engine": "amici",
        "regime": "forward_sensitivity",
        "git_rev": git_rev(str(HERE)),
        "versions": ver,
        "tally": counts,
        "n_jobs": len(results),
        "n_models": len(jobs),
        "elapsed_sec": round(elapsed, 2),
        "hardware": asens.hardware_info(),
        "concurrency": {"workers": args.workers, "mode": "process-parallel"},
        "config": _bngsim_config_meta(args),
        "sens_methods": methods,
        "param_cap": args.param_cap,
        "param_budget": args.param_budget,
        "max_steps": args.max_steps,
        "state_parity": {
            "n_state_diff": n_state_diff,
            "note": (
                "Jobs whose STATE trajectories already disagreed. The headline "
                "metric is the sensitivity tensor; this counts the rows where the "
                "underlying trajectories diverged too, so a sensitivity DIFF there "
                "is not attributable to the sensitivity machinery."
            ),
        },
        "integration_tol": {"rtol": args.rtol, "atol": args.atol, "applied_to": "both engines"},
        "reference_refusal_breakdown": {
            "counts": refusal_breakdown,
            "note": (
                "Sub-classification of REFERENCE_FAILED (bngsim ran, AMICI refused), "
                "auto-derived from the AMICI exception: feature_gap / compile / "
                "integrator / other. The one class that is NOT auto-derived is "
                f"'{amici_dispositions.INVALID_RESULT_REFUSAL}' — AMICI answered and "
                "the answer is what is unusable, so there is no exception to read; it "
                "comes from an authored amici_dispositions entry."
            ),
        },
        "overrides": {
            "tol_overridden_jobs": n_tol_ov,
            "note": (
                "Only engine-agnostic tol overrides are honored, as in amici_run.py; "
                "the RoadRunner-calibrated overrides are NOT applied. AMICI-calibrated "
                "dispositions are separate — see 'dispositions' below."
            ),
        },
        "dispositions": {
            "counts": dict(disposition_states.most_common()),
            "note": (
                "AMICI-calibrated per-row dispositions (issue #380), from "
                "amici_dispositions.py. 'invalid_reference:applied' — AMICI ran but "
                "its answer is defective, so the DIFF became a non-scoring "
                "REFERENCE_FAILED/"
                f"{amici_dispositions.INVALID_RESULT_REFUSAL}; 'comparison_artifact:"
                "applied' — neither engine is wrong and the DIFF became a PASS with "
                "the reason recorded. ':stale' means the row agrees on its own now "
                "(prune the entry); ':inapplicable' means the row produced no "
                "comparison this run (it raised, timed out or had no oracle), so the "
                "entry was neither used nor falsified. Every applied row carries its "
                "reason in the row comment."
            ),
        },
        "oracle_basis": (
            "Cross-engine NUMERIC tolerance on the FORWARD-SENSITIVITY tensor "
            "dx_i(t)/dp_j, via the shared _core.differ deterministic_verdict applied "
            "to the tensor flattened to (n_time, n_species*n_param) — one column per "
            "(species, parameter) pair, so differ's per-column peak terms and "
            "significance gate judge each coefficient against its own dynamic range. "
            "Both engines are pinned to the same CVODES corrector method, the same "
            f"integration tolerance (rtol={args.rtol:g}, atol={args.atol:g}), and the "
            "SAME shared parameter list in the same order (SBML ids; bngsim's _lp_ "
            "local-parameter prefix stripped, compartment-size parameters and "
            "AMICI-fixed parameters excluded, capped at "
            f"{args.param_cap or 'no limit'} per model). AMICI's parameter scale is "
            "pinned to linear so both report dx/dp rather than dx/dln(p). State "
            "trajectories are compared separately and reported per row. Failure is "
            "attributed per engine: bngsim raised + AMICI ran -> EXCEPTION; bngsim "
            "ran + AMICI raised -> REFERENCE_FAILED (non-scoring); both raised, or "
            "no shared differentiable parameter -> BAD_TEST. A bngsim "
            "SensitivityUnsupportedError — a DECLARED refusal to differentiate a "
            "construct (non-differentiable event crossing time, or a rate law "
            "codegen cannot close-form differentiate) — is UNSUPPORTED, matched by "
            "exception TYPE and not by message text, and is non-scoring: it is a "
            "documented capability gap, not an actionable bug. A DIFF may finally "
            "be re-dispositioned by an AUTHORED amici_dispositions entry, which "
            "requires third-oracle evidence naming which engine is wrong: an "
            "AMICI-side defect makes the row REFERENCE_FAILED (no oracle, "
            "non-scoring), and a cell no oracle can resolve makes it PASS. Both "
            "record the evidence in the row's comment."
        ),
    }
    write_report(out_path, results, meta=meta)

    print()
    print("=" * 72)
    print(
        "  "
        + "  ".join(f"{k}: {v}" for k, v in counts.items() if v)
        + f"   elapsed {elapsed:.1f}s"
    )
    if n_state_diff:
        print(f"  note: {n_state_diff} job(s) also had a STATE trajectory DIFF")
    if refusal_breakdown:
        print(
            "  REFERENCE_FAILED by refusal: "
            + ", ".join(f"{c}={n}" for c, n in refusal_breakdown.items())
        )
    if disposition_states:
        print(
            "  dispositions: " + ", ".join(f"{c}={n}" for c, n in disposition_states.most_common())
        )
    print(f"  report: {out_path}")
    print("=" * 72)
    n_fail = sum(counts.get(o.value, 0) for o in FAILING)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
