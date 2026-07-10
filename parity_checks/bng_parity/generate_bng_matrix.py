#!/usr/bin/env python3
"""Render ``runs/report_ode.json`` (from ``bng_ode_run.py``) to a single-page HTML
parity + timing matrix — the bng sibling of ``rr_parity``/``amici_parity``'s
matrices, same color taxonomy, verdict badges, and three-tier timing layout.

Two engines side by side: **BNGsim** (in-process ``.net`` ODE) vs **run_network**
(the legacy CVODE binary), both integrating the SAME BNG2.pl-generated network.
Network generation (BNG2.pl) is the shared per-model build prefix, shown once in
the model cell and attributed to neither integrator. The headline timing story is
BNGsim's warm (reused-model) marginal cost vs run_network's per-call cost — what a
fitting/MCMC loop actually pays per evaluation on each stack.

    python generate_bng_matrix.py [report.json] [out.html]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Mirror of ``bng_stoch_run.DEFAULT_N_REP`` — the base stochastic ensemble size a
# default run uses. Kept as a local constant (not imported) so this renderer stays
# lightweight and stack-free; a stochastic row is only "non-default" when its n_rep
# differs from this. Keep in sync with bng_stoch_run.py.
_DEFAULT_N_REP = 10


# --------------------------------------------------------------------------- #
# Taxonomy (identical to the SBML matrices)
# --------------------------------------------------------------------------- #
def classify_row(outcome: str, subclass: str | None = None) -> tuple[str, str]:
    # GH #69: a DIFF tagged ``subclass="known_artifact"`` is a documented
    # comparison artifact (integrator phase-wander / staircase knife-edge /
    # reference-side tolerance tail) within its recorded magnitude bound — it
    # renders KNOWN ARTIFACT, non-scoring, not a bngsim bug.
    if outcome == "DIFF" and subclass == "known_artifact":
        return "status-known", "KNOWN ARTIFACT"
    if outcome == "PASS":
        return "status-passed", "PASSED"
    if outcome in ("DIFF", "TIMEOUT"):
        return "status-failed", "FAILED"
    if outcome in ("REFERENCE_FAILED", "EXCEPTION"):
        return "status-triaged", "TRIAGED"
    return "status-refused", "REFUSED"  # BAD_TEST / SKIP


def verdict_badge(outcome: str, engine: str) -> tuple[str, str]:
    """(text, css) for the verdict badge inside one engine's cell.

    engine is "bngsim" or "run_network". The legacy run_network is the existence
    proof, so EXCEPTION (bngsim raised) shows run_network PASS, and
    REFERENCE_FAILED (run_network raised) shows bngsim PASS.
    """
    if outcome == "PASS":
        return "PASS", "verdict-pass"
    if outcome == "DIFF":
        return "DIFF", "verdict-diff"
    if outcome == "TIMEOUT":
        return "TIMEOUT", "verdict-diff"
    if outcome == "EXCEPTION":
        return (
            ("EXCEPTION", "verdict-exception") if engine == "bngsim" else ("PASS", "verdict-pass")
        )
    if outcome == "REFERENCE_FAILED":
        return ("PASS", "verdict-pass") if engine == "bngsim" else ("REF-FAIL", "verdict-reffail")
    if outcome == "SKIP":
        return "SKIP", "verdict-refused"
    return "REFUSED", "verdict-refused"  # BAD_TEST


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def fmt_t(sec: float | None) -> str:
    if sec is None:
        return "—"
    if sec == 0:
        return "0"
    ms = sec * 1000
    if ms < 1:
        return f"{ms:.2f} ms"
    if ms < 10:
        return f"{ms:.1f} ms"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def _esc(s) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _tcell(label: str, value: str) -> str:
    return f'<div class="tcell"><div class="tk">{label}</div><div class="tv">{value}</div></div>'


def _fmt_sec(sec) -> str:
    if sec is None:
        return "—"
    try:
        return f"{float(sec):g}s"
    except (TypeError, ValueError):
        return str(sec)


def _runtime_settings_html(meta: dict, results: list[dict]) -> str:
    """Visible provenance for non-default runtime settings and per-model overrides."""
    runtime = meta.get("runtime") or {}
    overrides = meta.get("overrides") or {}
    counts = overrides.get("counts_by_field") or {}

    global_timeout = runtime.get("global_timeout_sec")
    timeout_source = runtime.get("global_timeout_source")
    default_timeout = runtime.get("default_timeout_sec")
    top_bits = []
    if global_timeout is not None:
        top_bits.append(
            f"global timeout={_fmt_sec(global_timeout)}"
            + (f" ({_esc(timeout_source)})" if timeout_source else "")
        )
    if default_timeout is not None:
        top_bits.append(f"default timeout={_fmt_sec(default_timeout)}")
    if runtime.get("subprocess_timeout_rule"):
        top_bits.append(f"subprocess cap: {_esc(runtime.get('subprocess_timeout_rule'))}")
    if runtime.get("rep_timeout_rule"):
        top_bits.append(f"replicate cap: {_esc(runtime.get('rep_timeout_rule'))}")
    itol = meta.get("integration_tol") or {}
    if itol.get("rtol") is not None and itol.get("atol") is not None:
        top_bits.append(f"tol=rtol={itol.get('rtol'):g}, atol={itol.get('atol'):g}")

    count_bits = []
    for field in sorted(counts):
        count_bits.append(f"{_esc(field)}={counts[field]}")
    for key, label in (
        ("tol_overridden_jobs", "tol-overridden jobs"),
        ("timeout_overridden_jobs", "timeout-overridden jobs"),
    ):
        val = runtime.get(key)
        if val:
            count_bits.append(f"{label}={val}")

    row_items = []
    for r in results:
        settings = (r.get("timing") or {}).get("settings") or {}
        row_specific = (
            settings.get("timeout_source") == "job_override"
            or settings.get("tol_source") == "job_override"
            or settings.get("rep_timeout_source") == "cli"
            or (settings.get("n_rep") is not None and settings.get("n_rep") != _DEFAULT_N_REP)
            or (settings.get("seed_base") is not None and settings.get("seed_base") != 1)
            or settings.get("nf_block_same_complex_binding") is False
        )
        if not row_specific:
            continue
        bits = []
        if settings.get("timeout_source") != "default":
            bits.append(
                f"timeout={_fmt_sec(settings.get('timeout_sec'))} "
                f"({_esc(settings.get('timeout_source'))})"
            )
        if settings.get("rep_timeout_source") == "cli":
            bits.append(f"rep timeout={_fmt_sec(settings.get('rep_timeout_sec'))} (cli)")
        if settings.get("tol_source") and settings.get("tol_source") != "default":
            bits.append(
                "tol="
                f"rtol={settings.get('rtol'):g}, atol={settings.get('atol'):g} "
                f"({_esc(settings.get('tol_source'))})"
            )
        if settings.get("n_rep") is not None and settings.get("n_rep") != _DEFAULT_N_REP:
            bits.append(f"n_rep={settings.get('n_rep')}")
        if settings.get("seed_base") is not None and settings.get("seed_base") != 1:
            bits.append(f"seed_base={settings.get('seed_base')}")
        if settings.get("nf_block_same_complex_binding") is False:
            bits.append("NF block_same_complex_binding=off")
        reason = settings.get("timeout_reason")
        if reason:
            bits.append(f"reason: {_esc(reason)}")
        if bits:
            row_items.append(
                f"<li><code>{_esc(r.get('model_id', ''))}</code>: {'; '.join(bits)}</li>"
            )

    override_items = []
    for ov in overrides.get("jobs") or []:
        val = ov.get("value")
        if isinstance(val, (dict, list)):
            val = json.dumps(val, sort_keys=True)
        override_items.append(
            f"<li><code>{_esc(ov.get('model_id', ''))}</code>: "
            f"{_esc(ov.get('field', ''))}={_esc(val)} — {_esc(ov.get('reason', ''))}</li>"
        )

    summary = " · ".join(top_bits + count_bits) or "default runtime settings"
    row_html = (
        "<h4>Rows using non-default resolved settings</h4><ul>" + "".join(row_items) + "</ul>"
        if row_items
        else "<p>No row used non-default resolved runtime settings.</p>"
    )
    override_html = (
        "<h4>Manifest overrides selected for this run</h4><ul>" + "".join(override_items) + "</ul>"
        if override_items
        else "<p>No manifest overrides selected for this run.</p>"
    )
    note = overrides.get("note") or ""
    note_html = f"<p>{_esc(note)}</p>" if note else ""
    return f"""
        <details class="catdefs" open style="margin:6px 0 2px 0;">
            <summary style="cursor:pointer;font-weight:600;color:#495057;font-size:0.9em;">Runtime settings &amp; overrides — {summary}</summary>
            <div style="font-size:0.85em;color:#495057;line-height:1.65;margin:6px 0;">
                {note_html}
                {row_html}
                {override_html}
            </div>
        </details>
    """


# --------------------------------------------------------------------------- #
# Engine-cell timing tiers
# --------------------------------------------------------------------------- #
def _bn_total_sec(t: dict | None) -> float | None:
    """BNGsim per-model build subtotal (load + jac + codegen) — the one-time cost
    a fresh Simulator pays before the first solve (excludes the shared netgen)."""
    if not t:
        return None
    return (t.get("load_sec") or 0) + (t.get("jac_derive_sec") or 0) + (t.get("codegen_sec") or 0)


def render_bngsim_cell(outcome: str, timing: dict) -> str:
    """Config + verdict + the BNGsim timing tiers (build · integration)."""
    bn = timing.get("bngsim") or {}
    cfg = bn.get("config", {})
    badge_text, badge_css = verdict_badge(outcome, "bngsim")
    has = bool(bn)
    mode = (timing.get("spec") or {}).get("mode")

    parts = [
        '<div class="engine-cell">',
        '<div class="config-section"><div class="config-label">ODE solver</div>'
        '<div class="config-value">CVODE — BDF, Newton</div></div>',
        f'<div class="config-section"><div class="config-label">RHS backend</div>'
        f'<div class="config-value">{_esc(_rhs_label(cfg.get("codegen")))}</div></div>',
        f'<div class="config-section"><div class="config-label">Jacobian</div>'
        f'<div class="config-value">{_esc(_jac_label(cfg.get("jacobian")))}</div></div>',
        f'<div class="config-section"><div class="config-label">Linear solver</div>'
        f'<div class="config-value">{_esc(cfg.get("linear_solver", "—"))}</div></div>',
        f'<div class="verdict {badge_css}">{badge_text}</div>',
    ]
    if not has:
        _msg = {
            "BAD_TEST": "the model failed before bngsim could run",
            "SKIP": "no runnable ODE horizon was resolved",
            "TIMEOUT": "the job exceeded the wall clock before timing was recorded",
            "REFERENCE_FAILED": "bngsim ran, but this older report did not record detailed bngsim timing",
        }.get(outcome, "bngsim timing was not recorded in this report")
        if mode == "multi_segment":
            _msg = "full-protocol trajectory was compared; detailed bngsim timing is unavailable in this older report"
        parts.append(f'<div class="error-msg">{_esc(_msg)}</div>')
        parts.append("</div>")
        return "".join(parts)

    if bn.get("multi_segment"):
        parts.append(
            '<div class="ttier"><div class="ttier-head">② FULL PROTOCOL '
            "<span>multi-phase replay; coarse wall timing, not a warm single-segment solve</span></div>"
            '<div class="ttier-grid two">'
            + _tcell("Replay wall", fmt_t(bn.get("integrate_sec")))
            + _tcell("Segments", "protocol")
            + "</div></div>"
        )
        parts.append("</div>")
        return "".join(parts)

    subtotal = _bn_total_sec(bn)
    parts.append(
        '<div class="ttier"><div class="ttier-head">② Per-model build '
        "<span>once per model · excludes shared netgen</span></div>"
        '<div class="ttier-grid bng">'
        + _tcell("Net load", fmt_t(bn.get("load_sec")))
        + _tcell("Jacobian", fmt_t(bn.get("jac_derive_sec")))
        + _tcell("Codegen", fmt_t(bn.get("codegen_sec")))
        + _tcell("Subtotal", fmt_t(subtotal))
        + "</div></div>"
    )
    warm = bn.get("integrate_warm_min_sec") or bn.get("integrate_sec")
    # ③ ENGINE — the harness-stripped per-integration inner loop: bngsim's warm solve
    # (model reused: CVODE on the JIT'd/compiled RHS + analytical Jacobian). This is
    # the apples-to-apples comparison against run_network's self-reported propagation CPU.
    parts.append(
        '<div class="ttier"><div class="ttier-head">③ ENGINE · per integration '
        "<span>warm solve, model reused · is our inner loop competitive?</span></div>"
        '<div class="ttier-grid two">'
        + _tcell("Warm (marginal)", fmt_t(warm))
        + _tcell("Cold (first solve)", fmt_t(bn.get("integrate_cold_sec")))
        + "</div></div>"
    )
    # ④ WORKFLOW — the fitting-loop cost: bngsim builds the model ONCE then reuses the
    # warm solver every evaluation, so the per-iteration cost stays at the warm solve
    # while the one-time build amortizes away (run_network re-pays its full per-call
    # wall each iteration — see its ④).
    parts.append(
        '<div class="ttier"><div class="ttier-head">④ WORKFLOW · fitting loop '
        "<span>build once + reuse · per-evaluation cost in a fit/MCMC loop</span></div>"
        '<div class="ttier-grid two">'
        + _tcell("Per-iteration (warm)", fmt_t(warm))
        + _tcell("One-time build", fmt_t(subtotal))
        + "</div></div>"
    )
    parts.append("</div>")
    return "".join(parts)


def render_run_network_cell(outcome: str, timing: dict) -> str:
    """Config + verdict + the run_network timing tiers (per-call only — each
    invocation is a fresh process, no warm-solver reuse)."""
    rn = timing.get("run_network") or {}
    cfg = rn.get("config", {})
    badge_text, badge_css = verdict_badge(outcome, "run_network")
    has = bool(rn)
    mode = (timing.get("spec") or {}).get("mode")

    parts = [
        '<div class="engine-cell">',
        '<div class="config-section"><div class="config-label">ODE solver</div>'
        '<div class="config-value">CVODE — BDF, Newton</div></div>',
        '<div class="config-section"><div class="config-label">RHS backend</div>'
        '<div class="config-value">Compiled C (run_network binary)</div></div>',
        '<div class="config-section"><div class="config-label">Jacobian</div>'
        '<div class="config-value">Finite-difference (CVODE)</div></div>',
        '<div class="config-section"><div class="config-label">Linear solver</div>'
        f'<div class="config-value">{_esc(cfg.get("linear_solver", "—"))}</div></div>',
        f'<div class="verdict {badge_css}">{badge_text}</div>',
    ]
    if not has:
        _msg = {
            "BAD_TEST": "the model failed before run_network could run",
            "SKIP": "no runnable ODE horizon was resolved",
            "TIMEOUT": "the job exceeded the wall clock before timing was recorded",
            "REFERENCE_FAILED": "run_network/reference did not produce a trajectory",
            "EXCEPTION": "run_network ran; bngsim failed before comparison",
        }.get(outcome, "run_network timing was not recorded in this report")
        if mode == "multi_segment" and outcome != "REFERENCE_FAILED":
            _msg = "full native protocol was compared; detailed run_network timing is unavailable in this older report"
        parts.append(f'<div class="error-msg">{_esc(_msg)}</div>')
        parts.append("</div>")
        return "".join(parts)

    if rn.get("multi_segment"):
        parts.append(
            '<div class="ttier"><div class="ttier-head">② FULL PROTOCOL '
            "<span>native BNG2.pl protocol; coarse wall timing, no inner CPU split</span></div>"
            '<div class="ttier-grid two">'
            + _tcell("Protocol wall", fmt_t(rn.get("integrate_sec")))
            + _tcell("Calls", str(rn.get("n_calls") or 1))
            + "</div></div>"
        )
        parts.append("</div>")
        return "".join(parts)

    parts.append(
        '<div class="ttier"><div class="ttier-head">② Per-model build '
        "<span>none — reads the .net at each call</span></div>"
        '<div class="ttier-body"><div class="tcell"><div class="tk">Model build</div>'
        '<div class="tv">— (per-call)</div></div></div></div>'
    )
    # ③ ENGINE — run_network's OWN self-reported propagation CPU: the harness-stripped
    # CVODE inner loop, excluding the per-spawn init the per-call wall pays each time.
    # Both engines are CVODE, so this isolates the RHS/Jacobian backend.
    parts.append(
        '<div class="ttier"><div class="ttier-head">③ ENGINE · per integration '
        "<span>self-reported propagation CPU · the solve on its own</span></div>"
        '<div class="ttier-body"><div class="tcell"><div class="tk">Propagation CPU</div>'
        f'<div class="tv">{fmt_t(rn.get("propagation_cpu_sec"))}</div></div></div></div>'
    )
    # ④ WORKFLOW — the fitting-loop cost: a fresh process EVERY evaluation (spawn +
    # re-read .net + build + solve), so the per-call wall is paid in full each iteration.
    # The Init CPU is exactly the per-spawn re-init that bngsim amortizes away.
    parts.append(
        '<div class="ttier"><div class="ttier-head">④ WORKFLOW · fitting loop '
        "<span>fresh process each call · spawn + read .net + solve, ×N</span></div>"
        '<div class="ttier-grid bng">'
        + _tcell("Per-call (wall)", fmt_t(rn.get("integrate_sec")))
        + _tcell("Init CPU / call", fmt_t(rn.get("init_cpu_sec")))
        + _tcell("First call", fmt_t(rn.get("integrate_cold_sec")))
        + "</div></div>"
    )
    parts.append("</div>")
    return "".join(parts)


def _rhs_label(raw) -> str:
    return {
        "exprtk": "ExprTk bytecode (interpreted)",
        "cc": "Native C (cc-compiled .so)",
        "mir": "MIR JIT (in-process)",
    }.get(raw, raw or "—")


def _jac_label(raw) -> str:
    return {
        "analytical": "Analytical (SymPy, GH#76)",
        "fd": "Finite-difference",
        "jax": "JAX autodiff",
    }.get(raw, raw or "—")


# --------------------------------------------------------------------------- #
# Speedup / cost aggregation
# --------------------------------------------------------------------------- #
def _bn_warm(t):
    bn = (t or {}).get("bngsim") or {}
    return bn.get("integrate_warm_min_sec") or bn.get("integrate_sec")


def _bn_cold(t):
    return ((t or {}).get("bngsim") or {}).get("integrate_cold_sec")


def _rn_call(t):
    """run_network WORKFLOW cost — the full per-call wall (a fresh process every
    evaluation: spawn + re-read .net + build + solve)."""
    return ((t or {}).get("run_network") or {}).get("integrate_sec")


def _rn_engine(t):
    """run_network ENGINE cost — its OWN self-reported propagation CPU (the CVODE
    inner loop on its own, excluding the per-spawn init the per-call wall pays)."""
    return ((t or {}).get("run_network") or {}).get("propagation_cpu_sec")


def _geomean_ratio(pairs: list[tuple[float, float]]) -> tuple[float | None, int]:
    """Geometric mean of run_network/bngsim cost ratios over (bn, rn) pairs with
    both > 0. >1 ⇒ bngsim faster. Returns (geomean, n_pairs)."""
    logs = [math.log(rn / bn) for bn, rn in pairs if bn and rn and bn > 0 and rn > 0]
    if not logs:
        return None, 0
    return math.exp(sum(logs) / len(logs)), len(logs)


def _ratio_badge(leg_sec, bn_sec, kind: str, legacy_label: str, sub: str) -> str:
    """One ENGINE or WORKFLOW speedup badge (shared by the ODE and stochastic matrices).
    ratio = legacy ÷ bngsim, so >1 ⇒ BNGsim faster and <1 ⇒ the legacy engine faster
    (shown honestly — never an unqualified "N× faster"). ``kind`` is ``"engine"`` or
    ``"workflow"`` (sets the colored accent); ``sub`` is the per-track caption."""
    if not (bn_sec and leg_sec and bn_sec > 0 and leg_sec > 0):
        return ""
    ratio = leg_sec / bn_sec
    if ratio >= 1:
        txt, cls = f"BNGsim {ratio:.1f}× faster", "spd-fast"
    else:
        txt, cls = f"{_esc(legacy_label)} {1 / ratio:.1f}× faster", "spd-slow"
    return f'<div class="speedup {cls} spd-{kind}">{txt} <span>{sub}</span></div>'


def _ode_engine_badge(t: dict) -> str:
    """ODE ENGINE badge — bngsim warm solve vs run_network's self-reported propagation CPU."""
    return _ratio_badge(
        _rn_engine(t), _bn_warm(t), "engine", "run_network", "ENGINE · per-integration inner loop"
    )


def _ode_workflow_badge(t: dict) -> str:
    """ODE WORKFLOW badge — bngsim warm (per-fit-iteration, model reused) vs run_network's
    full per-call wall (re-spawn + re-read .net every evaluation)."""
    return _ratio_badge(
        _rn_call(t),
        _bn_warm(t),
        "workflow",
        "run_network",
        "WORKFLOW · fitting loop (amortization)",
    )


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header h1 { margin: 0 0 10px 0; color: #333; }
        .metadata { color: #666; font-size: 0.9em; line-height: 1.8; }
        .metadata strong { color: #333; }
        .hwcaveat { margin-top: 6px; padding: 7px 9px; background: #fff3cd; border-radius: 4px; color: #856404; font-size: 0.95em; line-height: 1.5; }
        .path-info { font-family: 'Courier New', monospace; background: #f8f9fa; padding: 2px 6px; border-radius: 3px; color: #495057; font-size: 0.85em; }
        .summary { display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap; }
        .summary-item { padding: 10px 15px; border-radius: 4px; font-weight: 500; }
        .summary-passed { background: #d4edda; color: #155724; }
        .summary-failed { background: #f8d7da; color: #721c24; }
        .summary-triaged { background: #fff3cd; color: #856404; }
        .summary-refused { background: #e2e3e5; color: #383d41; }
        .summary-known { background: #d1ecf1; color: #0c5460; }
        .rowtag { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 0.72em; font-weight: 700; letter-spacing: 0.02em; vertical-align: middle; margin-right: 6px; }
        .coststats { font-size: 0.92em; color: #333; background: #eef5fb; border: 1px solid #cfe2f3; border-radius: 6px; padding: 12px 14px; margin-top: 16px; line-height: 1.7; }
        .coststats b { color: #0b5394; }
        .plots { margin-top: 16px; padding-top: 14px; border-top: 1px solid #e3e6ea; }
        .plotctl { font-size: 0.85em; color: #555; margin-bottom: 8px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
        .plotctl .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; border: 1.5px solid #888; vertical-align: middle; }
        .plotctl .dot.bn { background: #0072B2; border-color: #0072B2; }
        .plotctl .dot.rn { background: #D55E00; border-color: #D55E00; border-radius: 0; }
        .plottitle { font-size: 0.85em; color: #444; font-weight: 600; margin-bottom: 2px; }
        .costplot { width: 100%; max-width: 940px; display: block; background: #fff; border: 1px solid #eee; border-radius: 4px; }
        .winstab { max-width: 620px; margin-top: 12px; }
        .winstab table { width: 100%; border-collapse: collapse; font-size: 0.85em; box-shadow: none; border-radius: 0; }
        .winstab th { position: static; background: #f1f3f5; color: #333; padding: 4px 8px; font-weight: 600; border-bottom: 1px solid #dee2e6; }
        .winstab td { padding: 3px 8px; border-bottom: 1px solid #f1f3f5; font-family: 'Courier New', monospace; }
        .winstab tr.wintot td { border-top: 2px solid #ced4da; font-weight: 700; background: #f8f9fb; }
        .sortbar { background: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 0.9em; color: #333; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
        .sortbar select { font-size: 0.95em; padding: 4px 6px; border-radius: 4px; border: 1px solid #ced4da; }
        .sortbar .sortnote { color: #868e96; font-size: 0.88em; }
        table { width: 100%; border-collapse: collapse; table-layout: fixed; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }
        th { background: #2c3e50; color: white; padding: 12px; text-align: left; font-weight: 600; position: sticky; top: 0; z-index: 10; }
        th.col-model { width: 26%; }
        th.col-engine { width: 37%; }
        td { padding: 14px; border-bottom: 4px solid #adb5bd; vertical-align: top; overflow-wrap: anywhere; }
        tr:hover { background: #f8f9fa; }
        .model-info { font-family: 'Courier New', monospace; font-size: 1.0em; font-weight: 600; color: #2c3e50; padding-bottom: 8px; margin-bottom: 8px; border-bottom: 1px solid #dee2e6; word-break: break-all; }
        .model-metadata { font-size: 0.92em; color: #6c757d; line-height: 1.6; padding-bottom: 8px; border-bottom: 1px solid #dee2e6; }
        .model-netgen { font-size: 0.9em; color: #0b5394; background: #eef5fb; border-radius: 4px; padding: 5px 8px; margin: 8px 0; }
        .model-netgen span { color: #6c757d; }
        .model-features { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
        .feature-badge { font-size: 0.82em; padding: 3px 7px; border-radius: 3px; background: #e9ecef; color: #495057; font-weight: 500; }
        .row-comment { margin-top: 8px; padding-top: 8px; border-top: 1px solid #dee2e6; font-size: 0.86em; color: #495057; line-height: 1.55; word-break: break-word; }
        .row-detail { margin-top: 5px; font-size: 0.76em; color: #adb5bd; line-height: 1.4; font-family: 'Courier New', monospace; word-break: break-word; }
        .engine-cell { border-left: 4px solid; padding-left: 12px; }
        .status-passed { background: linear-gradient(to right, #28a745 4px, #e8f5e9 4px); border-left-color: #28a745; }
        .status-failed { background: linear-gradient(to right, #dc3545 4px, #ffebee 4px); border-left-color: #dc3545; }
        .status-triaged { background: linear-gradient(to right, #ffc107 4px, #fffbea 4px); border-left-color: #ffc107; }
        .status-refused { background: linear-gradient(to right, #6c757d 4px, #f8f9fa 4px); border-left-color: #6c757d; }
        .status-known { background: linear-gradient(to right, #17a2b8 4px, #e7f6f8 4px); border-left-color: #17a2b8; }
        .config-section { margin-bottom: 8px; }
        .config-label { font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.5px; color: #868e96; margin-bottom: 1px; }
        .config-value { font-size: 0.96em; color: #212529; font-weight: 600; }
        .ttier { margin-top: 10px; padding-top: 8px; border-top: 1px solid #e3e6ea; }
        .ttier:first-of-type { border-top: 1px solid #ced4da; }
        .ttier-head { font-size: 0.80em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; color: #495057; margin-bottom: 5px; }
        .ttier-head span { text-transform: none; letter-spacing: 0; font-weight: 400; color: #868e96; font-size: 0.92em; }
        .ttier-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
        .ttier-grid.two { grid-template-columns: repeat(2, 1fr); }
        .ttier-body { display: flex; }
        .tcell { font-size: 0.90em; }
        .tk { color: #868e96; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 2px; }
        .tv { font-weight: 600; color: #212529; font-family: 'Courier New', monospace; font-size: 0.96em; }
        .verdict { display: inline-block; padding: 5px 12px; border-radius: 3px; font-size: 0.88em; font-weight: 600; margin-top: 8px; }
        .verdict-pass { background: #28a745; color: white; }
        .verdict-diff { background: #dc3545; color: white; }
        .verdict-exception { background: #ffc107; color: #000; }
        .verdict-reffail { background: #ffc107; color: #000; }
        .verdict-refused { background: #6c757d; color: white; }
        .verdict-known { background: #ffc107; color: #000; }
        .speedup { display: inline-block; margin-top: 8px; margin-left: 8px; padding: 5px 10px; border-radius: 3px; font-size: 0.85em; font-weight: 600; }
        .speedup span { font-weight: 400; opacity: 0.85; font-size: 0.9em; }
        .spd-fast { background: #d4edda; color: #155724; }
        .spd-slow { background: #f8d7da; color: #721c24; }
        /* ENGINE vs WORKFLOW badges stack as separate lines with a colored accent. */
        .spd-engine, .spd-workflow { display: block; margin-left: 8px; margin-right: 8px; }
        .spd-engine { border-left: 3px solid #6c757d; }
        .spd-workflow { border-left: 3px solid #0072B2; }
        .metric { font-size: 0.86em; color: #495057; margin-top: 5px; font-family: 'Courier New', monospace; }
        .metric.alert { color: #dc3545; font-weight: 600; }
        .error-msg { color: #721c24; background: #f8d7da; padding: 7px 9px; border-radius: 3px; font-size: 0.85em; margin-top: 6px; font-family: 'Courier New', monospace; line-height: 1.4; }
        .legend { background: white; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .legend h3 { margin: 0 0 12px 0; color: #333; }
        .legend-items { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
        .legend-item { display: flex; align-items: center; gap: 10px; }
        .legend-color { width: 40px; height: 30px; border-radius: 4px; border: 1px solid #dee2e6; flex-shrink: 0; }
        .legend-desc { font-size: 0.9em; color: #495057; }
        .legend p { font-size: 0.9em; color: #495057; line-height: 1.6; }
"""

_SORT_JS = """
    <script>
    function doSort() {
        var key = document.getElementById('sortsel').value;
        var dir = document.querySelector('input[name=sortdir]:checked');
        var desc = dir && dir.value === 'desc';
        var tb = document.querySelector('#matrix tbody');
        var rows = Array.prototype.slice.call(tb.querySelectorAll(':scope > tr'));
        var numeric = (key !== 'id');
        rows.sort(function (a, b) {
            var c;
            if (numeric) { c = parseFloat(a.getAttribute('data-' + key)) - parseFloat(b.getAttribute('data-' + key)); }
            else { var av = a.getAttribute('data-id'), bv = b.getAttribute('data-id'); c = (av < bv) ? -1 : (av > bv ? 1 : 0); }
            if (isNaN(c)) c = 0;
            if (!c) c = parseInt(a.getAttribute('data-order')) - parseInt(b.getAttribute('data-order'));
            return desc ? -c : c;
        });
        var frag = document.createDocumentFragment();
        rows.forEach(function (r) { frag.appendChild(r); });
        tb.appendChild(frag);
    }
    </script>
"""


def _plot_js(xlabel: str) -> str:
    """The cost scatter (bngsim vs legacy per-model cost). Each model contributes a
    BNGsim circle and a legacy square at the same x; the FASTER engine is drawn
    FILLED (a timing win), the slower OPEN (a loss). y-scale toggles log/linear via a
    radio (defaults to log if absent). The x-axis is labeled with ``xlabel``."""
    lab = json.dumps(xlabel)
    return (
        """
    <script>
    var PLOT_XLABEL = """
        + lab
        + """;
    function drawPlot() {
        var c = document.getElementById('cv'); if (!c) return;
        var dpr = window.devicePixelRatio || 1;
        var W = c.clientWidth, H = 360;
        c.width = W * dpr; c.height = H * dpr;
        var g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, W, H);
        var pad = {l: 64, r: 14, t: 14, b: 48};
        var bn = PLOT.bn, rn = PLOT.rn, n = bn.length;
        var all = bn.concat(rn).filter(function (v) { return v > 0; });
        if (!all.length) return;
        var sEl = document.querySelector('input[name="plotscale"]:checked');
        var logScale = !sEl || sEl.value === 'log';
        var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
        var ylo, yhi;
        if (logScale) { ylo = Math.floor(Math.log10(lo)); yhi = Math.ceil(Math.log10(hi)); if (yhi <= ylo) yhi = ylo + 1; }
        else { ylo = 0; yhi = hi * 1.05; }
        function X(i) { return pad.l + (n <= 1 ? 0.5 : i / (n - 1)) * (W - pad.l - pad.r); }
        function Y(v) { var t = logScale ? (Math.log10(v) - ylo) / (yhi - ylo) : (v - ylo) / (yhi - ylo); return pad.t + (1 - t) * (H - pad.t - pad.b); }
        g.strokeStyle = '#eee'; g.fillStyle = '#999'; g.font = '10px sans-serif'; g.textAlign = 'right'; g.textBaseline = 'middle';
        function ylab(v) { return v >= 1 ? (Math.round(v * 100) / 100) + ' s' : (Math.round(v * 1e5) / 100) + ' ms'; }
        if (logScale) {
            for (var e = Math.ceil(ylo); e <= Math.floor(yhi); e++) {
                var yv = Math.pow(10, e), yy = Y(yv);
                g.beginPath(); g.moveTo(pad.l, yy); g.lineTo(W - pad.r, yy); g.stroke();
                g.fillText(ylab(yv), pad.l - 6, yy);
            }
        } else {
            for (var k = 0; k <= 5; k++) { var v = ylo + (yhi - ylo) * k / 5, yk = Y(v); g.beginPath(); g.moveTo(pad.l, yk); g.lineTo(W - pad.r, yk); g.stroke(); g.fillText(ylab(v), pad.l - 6, yk); }
        }
        // Each model: fill the FASTER engine (win), outline the slower (loss).
        for (var i = 0; i < n; i++) {
            var b = bn[i], r = rn[i], both = b > 0 && r > 0;
            var bFill = both ? (b < r) : (b > 0), rFill = both ? (r < b) : (r > 0);
            if (b > 0) { var x = X(i), y = Y(b); g.strokeStyle = '#0072B2'; g.fillStyle = '#0072B2'; g.lineWidth = 1.2; g.beginPath(); g.arc(x, y, 3, 0, 6.29); if (bFill) g.fill(); else g.stroke(); }
            if (r > 0) { var x2 = X(i), y2 = Y(r); g.strokeStyle = '#D55E00'; g.fillStyle = '#D55E00'; g.lineWidth = 1.2; if (rFill) g.fillRect(x2 - 2.6, y2 - 2.6, 5.2, 5.2); else g.strokeRect(x2 - 2.4, y2 - 2.4, 4.8, 4.8); }
        }
        g.fillStyle = '#555'; g.font = '11px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
        g.fillText(PLOT_XLABEL, pad.l + (W - pad.l - pad.r) / 2, H - 8);
        g.save(); g.translate(14, pad.t + (H - pad.t - pad.b) / 2); g.rotate(-Math.PI / 2);
        g.fillText('ensemble wall (' + (logScale ? 'log' : 'linear') + ')', 0, 0); g.restore();
    }
    window.addEventListener('load', drawPlot);
    window.addEventListener('resize', drawPlot);
    </script>
"""
    )


_PLOT_JS = _plot_js("models, sorted by BNGsim ensemble cost (low → high)")


def _wins_table(results) -> tuple[str, str]:
    """(coststats_html, wins_table_html) — the ENGINE-vs-WORKFLOW split for ODE. For
    each model an ENGINE pair (bngsim warm solve vs run_network's self-reported
    propagation CPU — both CVODE, isolating the RHS/Jacobian backend) and a WORKFLOW
    pair (bngsim warm, the per-fit-iteration cost with the model reused, vs run_network's
    full per-call wall) are binned by species count and reduced to two separate
    geomeans, so a large amortization win is never read as a faster integrator."""
    eng_pairs = []  # (bn_warm, rn_propagation, n_species) — harness-stripped solve
    wf_pairs = []  # (bn_warm, rn_call, n_species) — per-fit-iteration cost
    for r in results:
        t = r.get("timing") or {}
        nsp = ((t.get("spec") or {}).get("n_species")) or 0
        bw = _bn_warm(t)
        rn_e, rn_c = _rn_engine(t), _rn_call(t)
        if bw and rn_e and bw > 0 and rn_e > 0:
            eng_pairs.append((bw, rn_e, nsp))
        if bw and rn_c and bw > 0 and rn_c > 0:
            wf_pairs.append((bw, rn_c, nsp))
    eng_geo, eng_n = _geomean_ratio([(b, r) for b, r, _ in eng_pairs])
    wf_geo, wf_n = _geomean_ratio([(b, r) for b, r, _ in wf_pairs])
    if eng_geo is None and wf_geo is None:
        return ('<div class="coststats">No model ran on both engines yet.</div>', "")

    def _dir(geo):
        if geo is None:
            return "—"
        return (
            f"<b>BNGsim {geo:.1f}× faster</b>"
            if geo >= 1
            else f"<b>run_network {1 / geo:.1f}× faster</b>"
        )

    bn_eng_wins = sum(1 for b, r, _ in eng_pairs if b < r)
    rn_eng_wins = sum(1 for b, r, _ in eng_pairs if r < b)
    bn_wf_wins = sum(1 for b, r, _ in wf_pairs if b < r)
    rn_wf_wins = sum(1 for b, r, _ in wf_pairs if r < b)
    coststats = (
        '<div class="coststats">'
        f"<b>ENGINE</b> (per-integration inner loop, harness-stripped): over <b>{eng_n}</b> "
        f"model(s) the geomean ratio (run_network propagation CPU ÷ BNGsim warm solve) is "
        f"{_dir(eng_geo)} — BNGsim wins <b>{bn_eng_wins}</b>, run_network <b>{rn_eng_wins}</b>. "
        "Both engines are CVODE, so this isolates the RHS/Jacobian backend.<br>"
        f"<b>WORKFLOW</b> (fitting-loop cost — what you actually pay per evaluation): over "
        f"<b>{wf_n}</b> model(s) the geomean ratio (run_network per-call wall ÷ BNGsim warm) is "
        f"{_dir(wf_geo)} — BNGsim wins <b>{bn_wf_wins}</b>, run_network <b>{rn_wf_wins}</b>. "
        "BNGsim builds the model once and reuses the warm solver across evaluations; run_network "
        "spawns a fresh process and re-reads the .net every call — the WORKFLOW win is that "
        "amortization, NOT a faster integrator.</div>"
    )

    bins = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 150), (151, 10**9)]

    def _label(lo, hi):
        if hi >= 10**9:
            return f"≥{lo}"
        if lo == hi:
            return str(lo)
        return f"{lo}–{hi}"

    def _gtxt(g):
        if g is None:
            return "—"
        return f"{g:.1f}×" if g >= 1 else f"{1 / g:.1f}× (rn)"

    rows = []
    for lo, hi in bins:
        e_sub = [(b, r) for b, r, nsp in eng_pairs if lo <= nsp <= hi]
        w_sub = [(b, r) for b, r, nsp in wf_pairs if lo <= nsp <= hi]
        if not e_sub and not w_sub:
            continue
        eg, _ = _geomean_ratio(e_sub)
        wg, _ = _geomean_ratio(w_sub)
        rows.append(
            f"<tr><td>{_label(lo, hi)}</td><td>{len(w_sub)}</td>"
            f"<td>{_gtxt(eg)}</td><td>{_gtxt(wg)}</td></tr>"
        )
    eg_all, _ = _geomean_ratio([(b, r) for b, r, _ in eng_pairs])
    wg_all, _ = _geomean_ratio([(b, r) for b, r, _ in wf_pairs])
    rows.append(
        f'<tr class="wintot"><td>Total</td><td>{len(wf_pairs)}</td>'
        f"<td>{_gtxt(eg_all)}</td><td>{_gtxt(wg_all)}</td></tr>"
    )
    table = (
        '<div class="winstab"><div class="plottitle">ENGINE vs WORKFLOW geomean by species count '
        "(ratio = run_network ÷ BNGsim; &gt;1 ⇒ BNGsim faster, &ldquo;(rn)&rdquo; ⇒ run_network faster)"
        "</div><table><thead><tr><th>species</th><th>n</th>"
        "<th>ENGINE geomean</th><th>WORKFLOW geomean</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    return coststats, table


def _legend() -> str:
    return """    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-items">
            <div class="legend-item"><div class="legend-color" style="background:#28a745"></div><div class="legend-desc"><b>PASSED</b> — both engines ran and agreed within the oracle tolerance.</div></div>
            <div class="legend-item"><div class="legend-color" style="background:#dc3545"></div><div class="legend-desc"><b>FAILED</b> — DIFF (exceeds tolerance) or TIMEOUT.</div></div>
            <div class="legend-item"><div class="legend-color" style="background:#17a2b8"></div><div class="legend-desc"><b>KNOWN ARTIFACT</b> — a DIFF that is a documented comparison artifact within a recorded magnitude bound (integrator phase-wander, staircase knife-edge sampled on a discontinuity, or a reference-side tolerance tail), not a bngsim defect (GH #69). Non-scoring — excluded from FAILED.</div></div>
            <div class="legend-item"><div class="legend-color" style="background:#ffc107"></div><div class="legend-desc"><b>TRIAGED</b> — EXCEPTION (bngsim raised, run_network ran: actionable bngsim bug) or REFERENCE_FAILED (bngsim ran, run_network raised: no oracle).</div></div>
            <div class="legend-item"><div class="legend-color" style="background:#6c757d"></div><div class="legend-desc"><b>REFUSED</b> — BAD_TEST (netgen failed / both raised) or SKIP (no resolvable ODE horizon).</div></div>
        </div>
        <h3 style="margin-top:18px">Timing tiers — ENGINE vs WORKFLOW</h3>
        <p>Both engines are <b>CVODE</b> (BDF/Newton), so a single blended "speedup" conflates two different costs. The matrix reports them separately:<br>
        <b>① Per-process startup</b> (header) — one-time per worker: BNGsim's <code>import bngsim</code> + SymPy warm; run_network's process-spawn proxy. Amortized to ~0 in a fitting loop.<br>
        <b>② Per-model build</b> — BNGsim: <code>.net</code> load + analytical-Jacobian derivation + RHS codegen (one-time per model). run_network has none — it re-reads the <code>.net</code> on every call.<br>
        <b>③ ENGINE · per integration</b> — the harness-stripped inner loop: <i>is our inner loop competitive?</i> BNGsim's <b>warm</b> solve (model reused) vs run_network's OWN self-reported <b>propagation CPU</b>. Both are CVODE, so this isolates the RHS/Jacobian backend (BNGsim's analytical Jacobian + JIT'd/compiled RHS vs run_network's finite-difference Jacobian + compiled-C RHS). <b>Cold</b> = BNGsim's first solve incl. one-time CVODE setup.<br>
        <b>④ WORKFLOW · fitting loop</b> — what you actually pay per evaluation: <i>is driving it cheaper?</i> BNGsim builds the model once and reuses the warm solver, so the per-iteration cost stays at the warm solve; run_network spawns a fresh process and re-reads the <code>.net</code> EVERY call, so its <b>per-call wall</b> (incl. the <b>Init CPU</b> re-init) is paid in full each iteration. The WORKFLOW win is that <b>amortization</b> — a real capability win (a CLI cannot persist a session), but <b>NOT</b> a faster integrator. run_network has no warm mode (every call is cold).<br>
        <b>BNG2.pl netgen</b> (model cell) is the <b>shared build prefix</b> — the same network feeds both integrators, so it is attributed to neither.</p>
        <h3 style="margin-top:18px">Oracle</h3>
        <p>Cross-engine numeric tolerance via the shared <code>_core.differ</code> protocol over the network species (<code>.cdat</code> columns, identical network order): combined abs+rel per-cell tolerance with a fail-fraction budget gated by hard ceilings. <code>max_rel_err</code> is the post-budget worst remaining relative error.</p>
    </div>
"""


def generate_html(report_path: Path, output_path: Path) -> None:
    data = json.loads(Path(report_path).read_text())
    meta = data.get("_meta", {})
    results = data.get("results", [])
    ver = meta.get("versions", {})
    tally = meta.get("tally", {})
    hw = meta.get("hardware", {})
    conc = meta.get("concurrency", {})
    itol = meta.get("integration_tol", {})

    # Sort by species count (timing.spec.n_species), then model_id.
    def _nsp(r):
        return ((r.get("timing") or {}).get("spec") or {}).get("n_species") or 0

    results = sorted(results, key=lambda r: (_nsp(r), r.get("model_id", "")))

    # Warmup aggregate (per-process startup tier, shown once).
    bn_warm = [(r.get("timing") or {}).get("warmup", {}).get("bngsim_sec") for r in results]
    bn_warm = [x for x in bn_warm if x]
    warm_txt = (
        f"BNGsim startup (import + SymPy warm): median {fmt_t(sorted(bn_warm)[len(bn_warm) // 2])} "
        f"over {len(bn_warm)} workers"
        if bn_warm
        else "—"
    )

    coststats, wins = _wins_table(results)
    runtime_html = _runtime_settings_html(meta, results)

    # Plot data: per-integration warm (bngsim) and per-call (run_network), in the
    # sorted (species-count) order so the scatter reads left→right by complexity.
    bn_series = [round(_bn_warm(r.get("timing")) or 0.0, 8) for r in results]
    rn_series = [round(_rn_call(r.get("timing")) or 0.0, 8) for r in results]
    plot_json = json.dumps({"bn": bn_series, "rn": rn_series})

    # GH #69: split documented comparison artifacts out of the DIFF FAILED bucket
    # — they are non-scoring (a known integrator/reference effect, not a bngsim bug).
    known_n = sum(
        1 for r in results if r.get("outcome") == "DIFF" and r.get("subclass") == "known_artifact"
    )
    pass_n = tally.get("PASS", 0)
    fail_n = tally.get("DIFF", 0) + tally.get("TIMEOUT", 0) - known_n
    triaged_n = tally.get("EXCEPTION", 0) + tally.get("REFERENCE_FAILED", 0)
    refused_n = tally.get("BAD_TEST", 0) + tally.get("SKIP", 0)

    rows_html = []
    for order, r in enumerate(results):
        outcome = r.get("outcome", "")
        row_cls, _cat = classify_row(outcome, r.get("subclass"))
        timing = r.get("timing") or {}
        spec = timing.get("spec") or {}
        netgen = (timing.get("netgen") or {}).get("netgen_sec")
        model_id = r.get("model_id", "")
        nsp = spec.get("n_species") or 0
        val = r.get("value")
        metric_html = ""
        if val is not None and math.isfinite(val):
            alert = " alert" if (r.get("tol") and val > r.get("tol")) else ""
            metric_html = f'<div class="metric{alert}">largest relative gap: {val:.2e}</div>'
        comment = r.get("comment") or ""
        exc = r.get("exception") or ""
        comment_html = ""
        if comment or exc:
            comment_html = (
                f'<div class="row-comment">{_esc(comment)}</div>' if comment else ""
            ) + (f'<div class="row-detail">{_esc(exc)}</div>' if exc else "")

        # feature/horizon tags
        feats = []
        if spec:
            feats.append(f"t_end={spec.get('t_end')}")
            feats.append(f"{spec.get('n_steps')} steps")
        tier = "/".join(model_id.split("/")[:2]) if "/" in model_id else ""
        feat_html = "".join(f'<span class="feature-badge">{_esc(f)}</span>' for f in feats)
        netgen_html = (
            f'<div class="model-netgen">BNG2.pl netgen <span>(shared prefix)</span>: '
            f"<b>{fmt_t(netgen)}</b></div>"
            if netgen is not None
            else ""
        )
        model_cell = (
            f'<div class="model-info">{_esc(model_id)}</div>'
            f'<div class="model-metadata">{nsp} species · {_esc(tier)}</div>'
            f"{netgen_html}"
            f"{metric_html}"
            f"{_ode_engine_badge(timing)}"
            f"{_ode_workflow_badge(timing)}"
            f'<div class="model-features">{feat_html}</div>'
            f"{comment_html}"
        )

        bn_cell = render_bngsim_cell(outcome, timing)
        rn_cell = render_run_network_cell(outcome, timing)
        # data-* for client-side sorting (WORKFLOW = warm vs per-call wall; ENGINE =
        # warm vs run_network's self-reported propagation CPU).
        bnw = _bn_warm(timing) or 0
        rnc = _rn_call(timing) or 0
        rne = _rn_engine(timing) or 0
        wf_spd = (rnc / bnw) if (bnw and rnc) else 0
        eng_spd = (rne / bnw) if (bnw and rne) else 0
        rows_html.append(
            f'<tr class="{row_cls}" data-order="{order}" data-id="{_esc(model_id)}" '
            f'data-nsp="{nsp}" data-bnwarm="{bnw}" data-rncall="{rnc}" '
            f'data-wfspeedup="{wf_spd}" data-engspeedup="{eng_spd}">'
            f"<td>{model_cell}</td><td>{bn_cell}</td><td>{rn_cell}</td></tr>"
        )

    hwline = (
        f"{_esc(hw.get('cpu', '?'))} · {hw.get('physical_cores', '?')} cores · "
        f"timings measured under {conc.get('workers', '?')}-way process parallelism "
        "(absolute costs inflate under contention; ratios are robust)."
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>bng_parity ODE — BNGsim vs run_network</title>
<style>{_CSS}</style>
</head><body>
    <div class="header">
        <h1>bng_parity — BNGsim vs the legacy BNG2.pl/run_network stack (ODE)</h1>
        <div class="metadata">
            <strong>bngsim</strong> {_esc(ver.get("bngsim"))} &nbsp;·&nbsp;
            <strong>BioNetGen</strong> {_esc(ver.get("bng"))} &nbsp;·&nbsp;
            <strong>SUNDIALS</strong> {_esc(ver.get("sundials"))} &nbsp;·&nbsp;
            git {_esc(meta.get("git_rev"))} &nbsp;·&nbsp; {_esc(meta.get("generated"))}<br>
            run_network: <span class="path-info">{_esc(meta.get("run_network"))}</span><br>
            ODE tolerance (both engines): rtol={itol.get("rtol")}, atol={itol.get("atol")} &nbsp;·&nbsp;
            oracle: <code>_core.differ.deterministic_verdict</code> &nbsp;·&nbsp; {len(results)} ODE jobs<br>
            {_esc(warm_txt)}
        </div>
        <div class="hwcaveat">{hwline}</div>
        <div class="summary">
            <div class="summary-item summary-passed">✓ PASSED: {pass_n}</div>
            <div class="summary-item summary-failed">✗ FAILED: {fail_n}</div>
            {f'<div class="summary-item summary-known">◇ KNOWN ARTIFACT: {known_n}</div>' if known_n else ""}
            <div class="summary-item summary-triaged">⚑ TRIAGED: {triaged_n}</div>
            <div class="summary-item summary-refused">— REFUSED/SKIP: {refused_n}</div>
        </div>
        {runtime_html}
        {coststats}
        <div class="plots">
            <div class="plotctl">
                <span class="dot bn"></span> BNGsim (warm, per-evaluation)
                <span class="dot rn"></span> run_network (per-call wall)
                <span class="sortnote">log scale; lower is faster · WORKFLOW view (ENGINE story is in the per-row badges)</span>
            </div>
            <div class="plottitle">WORKFLOW: per-evaluation cost vs model complexity</div>
            <canvas id="cv" class="costplot"></canvas>
            {wins}
        </div>
    </div>
    <div class="sortbar">
        <label>Sort by:
        <select id="sortsel" onchange="doSort()">
            <option value="nsp">species count</option>
            <option value="id">model id</option>
            <option value="wfspeedup">WORKFLOW speedup</option>
            <option value="engspeedup">ENGINE speedup</option>
            <option value="bnwarm">BNGsim warm time</option>
            <option value="rncall">run_network per-call time</option>
        </select></label>
        <span class="sortdir">
            <label><input type="radio" name="sortdir" value="asc" checked onchange="doSort()"> asc</label>
            <label><input type="radio" name="sortdir" value="desc" onchange="doSort()"> desc</label>
        </span>
        <span class="sortnote">ties fall back to the default species-count order</span>
    </div>
    <table id="matrix">
        <thead><tr>
            <th class="col-model">Model</th>
            <th class="col-engine">BNGsim (in-process .net ODE)</th>
            <th class="col-engine">run_network (legacy CVODE binary)</th>
        </tr></thead>
        <tbody>
        {"".join(rows_html)}
        </tbody>
    </table>
    {_legend()}
    <script>var PLOT = {plot_json};</script>
    {_PLOT_JS}
    {_SORT_JS}
</body></html>
"""
    Path(output_path).write_text(html)


# =========================================================================== #
# Stochastic (SSA / NF) matrix — the same three-column BNGsim-vs-legacy layout
# and color taxonomy as the ODE matrix, to the stochastic cost model: a per-model
# load (no Jacobian/codegen) + a per-replicate ENSEMBLE (independent reseeded
# trajectories — no warm-reuse of one solve). The ensemble oracle replaces the
# deterministic verdict; the legacy reference is run_network (SSA) or NFsim (NF).
# =========================================================================== #
_REGIME = {
    "ssa": {
        "track": "SSA",
        "bn_method": "Gillespie SSA (exact)",
        # Fallback only — the live label comes from timing.config.rhs, which reports the
        # backend the engine actually used (cc / mir / interpreted; GH #190).
        "bn_rhs": "ExprTk propensities (interpreted)",
        "build": "BNG2.pl netgen",
        "build_key": "netgen_sec",
        "bn_load_label": "Net load",
        "complexity": "events/replicate",
    },
    "nf": {
        "track": "NF (network-free)",
        "bn_method": "NFsim-style network-free",
        "bn_rhs": "rule-based (no network)",
        "build": "BNG2.pl writeXML",
        "build_key": "writexml_sec",
        "bn_load_label": "Session build",
        "complexity": "observables",
    },
}


def stoch_verdict_badge(outcome: str, engine: str, subclass: str | None = None) -> tuple[str, str]:
    """(text, css) for a verdict badge in one engine's stochastic cell.

    The legacy stack is the existence proof, so EXCEPTION/UNSUPPORTED (bngsim side)
    shows legacy PASS, and REFERENCE_FAILED (legacy raised) shows bngsim PASS.
    ``engine`` is ``"bngsim"`` or ``"legacy"``.

    A DIFF with a documented non-scoring disposition is NOT a bngsim fault, so it does
    not show a red DIFF on the bngsim cell (GH #183/#69): ``ref_bscb`` (the legacy
    reference is the buggy oracle, bngsim confirmed correct) shows bngsim PASS · legacy
    KNOWN REF; ``known_artifact`` (a benign sampling artifact both engines share) shows
    KNOWN on both.
    """
    if outcome == "DIFF" and subclass == "ref_bscb":
        return ("PASS", "verdict-pass") if engine == "bngsim" else ("KNOWN REF", "verdict-known")
    if outcome == "DIFF" and subclass == "known_artifact":
        return "KNOWN", "verdict-known"
    if outcome == "PASS":
        return "PASS", "verdict-pass"
    if outcome == "DIFF":
        return "DIFF", "verdict-diff"
    if outcome == "TIMEOUT":
        # Coverage gap (too slow to finish the ensemble in the wall), not a
        # correctness failure — gray, consistent with the gray TIMEOUT row.
        return "TIMEOUT", "verdict-refused"
    if outcome == "EXCEPTION":
        return (
            ("EXCEPTION", "verdict-exception") if engine == "bngsim" else ("PASS", "verdict-pass")
        )
    if outcome == "UNSUPPORTED":
        return (
            ("UNSUPPORTED", "verdict-refused") if engine == "bngsim" else ("PASS", "verdict-pass")
        )
    if outcome == "REFERENCE_FAILED":
        return ("PASS", "verdict-pass") if engine == "bngsim" else ("REF-FAIL", "verdict-reffail")
    if outcome == "SKIP":
        return "SKIP", "verdict-refused"
    return "REFUSED", "verdict-refused"  # BAD_TEST


# ── The two explicitly-labeled comparisons (the heart of the ENGINE/WORKFLOW
# split). Both engines run the SAME algorithm class, so a single blended "speedup"
# conflates two different costs: the per-trajectory INNER LOOP (ENGINE) and the
# N-replicate ensemble wall (WORKFLOW). The CLIs re-spawn + re-read the artifact on
# every replicate, so the ensemble multiplies their per-call re-init by N while
# bngsim amortizes it to ~0 — a real capability win, but a WORKFLOW one, NOT engine
# speed. SSA v21 is the canonical case: run_network's propagation CPU (0.01s) BEATS
# bngsim's warm trajectory (0.021s) at ENGINE, yet bngsim wins ~12× at WORKFLOW. ──


def _bn_engine(t):
    """BNGsim ENGINE per-trajectory cost — the warm single-trajectory wall (the
    model load and per-replicate setup are stripped: the in-process inner loop
    on its own). Falls back to the all-replicate median if there is no warm rep."""
    bn = (t or {}).get("bngsim") or {}
    return bn.get("rep_warm_median_sec") or bn.get("rep_median_sec")


def _leg_engine(t, regime):
    """Legacy ENGINE per-trajectory cost — the engine's OWN self-reported inner-loop
    CPU (``run_network`` propagation_cpu for SSA; NFsim total_cpu for NF). This is the
    harness-stripped number: it excludes the per-spawn re-init that the per-call wall
    pays on every replicate, so it exposes how fast the legacy inner loop really is."""
    leg = (t or {}).get("legacy") or {}
    return leg.get("propagation_cpu_sec") if regime == "ssa" else leg.get("total_cpu_sec")


def _bn_workflow(t):
    """BNGsim WORKFLOW cost — the N-replicate in-process ensemble wall (build the
    model once, then reseed N trajectories: the real fitting/sampling cost)."""
    return ((t or {}).get("bngsim") or {}).get("ensemble_sec")


def _leg_workflow(t):
    """Legacy WORKFLOW cost — the N-replicate ensemble wall (N fresh spawns, each
    re-reading the artifact: no in-process amortization is possible for a CLI)."""
    return ((t or {}).get("legacy") or {}).get("ensemble_sec")


def _stoch_engine_badge(t: dict, regime: str, legacy_label: str) -> str:
    return _ratio_badge(
        _leg_engine(t, regime),
        _bn_engine(t),
        "engine",
        legacy_label,
        "ENGINE · per-trajectory inner loop",
    )


def _stoch_workflow_badge(t: dict, legacy_label: str) -> str:
    return _ratio_badge(
        _leg_workflow(t),
        _bn_workflow(t),
        "workflow",
        legacy_label,
        "WORKFLOW · N-rep ensemble (amortization)",
    )


def render_bngsim_stoch_cell(
    outcome: str, timing: dict, regime: str, subclass: str | None = None
) -> str:
    """Config + verdict + BNGsim stochastic timing tiers (load · ensemble)."""
    rg = _REGIME[regime]
    bn = timing.get("bngsim") or {}
    cfg = bn.get("config", {})
    badge_text, badge_css = stoch_verdict_badge(outcome, "bngsim", subclass)
    has = bool(bn) and bn.get("rep_median_sec") is not None

    parts = [
        '<div class="engine-cell">',
        f'<div class="config-section"><div class="config-label">Method</div>'
        f'<div class="config-value">{_esc(cfg.get("method", rg["bn_method"]))}</div></div>',
        f'<div class="config-section"><div class="config-label">Propensities</div>'
        f'<div class="config-value">{_esc(cfg.get("rhs", rg["bn_rhs"]))}</div></div>',
        f'<div class="verdict {badge_css}">{badge_text}</div>',
    ]
    if regime == "nf" and "block_same_complex_binding" in cfg:
        bscb = "on" if cfg.get("block_same_complex_binding") else "off"
        parts.insert(
            3,
            f'<div class="config-section"><div class="config-label">block_same_complex_binding</div>'
            f'<div class="config-value">{bscb}</div></div>',
        )
    if not has:
        _msg = {
            "TIMEOUT": "bngsim's ensemble did not finish within the per-model wall (TIMEOUT — a coverage gap, not a failure)",
            "UNSUPPORTED": "bngsim cleanly declined this model (documented limitation)",
            "BAD_TEST": "the model failed to build — neither engine ran",
        }.get(outcome, "bngsim recorded no ensemble timing for this run")
        parts.append(f'<div class="error-msg">{_esc(_msg)}</div>')
        parts.append("</div>")
        return "".join(parts)

    n = bn.get("n_rep", 0) or 0
    cold, warm = bn.get("rep_cold_sec"), bn.get("rep_warm_median_sec")
    cw = f"{fmt_t(cold)} → {fmt_t(warm)}" if warm is not None else fmt_t(cold)
    parts.append(
        '<div class="ttier"><div class="ttier-head">② Per-model build '
        "<span>once per model · excludes the shared build prefix · no Jacobian; eligible "
        "mass-action models compile a propensity vector on rep 1 (see Propensities)</span></div>"
        '<div class="ttier-grid bng">'
        + _tcell(rg["bn_load_label"], fmt_t(bn.get("load_sec")))
        + "</div></div>"
    )
    # ③ ENGINE — the harness-stripped per-trajectory inner loop: bngsim's warm
    # single-trajectory wall + the stochastic-activity throughput (events for SSA,
    # reaction firings for NF via the globalEventCounter). This is the apples-to-apples
    # comparison against the legacy's self-reported propagation/total CPU.
    eng_cells = _tcell("Warm / trajectory", fmt_t(warm))
    if regime == "ssa" and bn.get("events_per_rep") is not None:
        eng_cells += _tcell("Events / rep", f"{bn.get('events_per_rep'):,.0f}")
    elif regime == "nf" and bn.get("reactions_per_rep") is not None:
        rps = bn.get("reactions_per_sec")
        eng_cells += _tcell("Reactions / rep", f"{bn.get('reactions_per_rep'):,.0f}")
        eng_cells += _tcell("Reactions / s", f"{rps:,.0f}/s" if rps else "—")
    parts.append(
        '<div class="ttier"><div class="ttier-head">③ ENGINE · per trajectory '
        "<span>harness-stripped inner loop · is our inner loop competitive?</span></div>"
        '<div class="ttier-grid bng">' + eng_cells + "</div></div>"
    )
    # ④ WORKFLOW — what you actually pay: the N-replicate ensemble wall. bngsim builds
    # the model ONCE and reseeds in-process, so the per-call re-init the CLIs repeat is
    # amortized to ~0 here (a real, legitimate win the legacy binaries cannot match).
    wf_cells = (
        _tcell(f"Ensemble ({n})", fmt_t(bn.get("ensemble_sec")))
        + _tcell("Per-replicate (median)", fmt_t(bn.get("rep_median_sec")))
        + _tcell("Cold → warm", cw)
    )
    parts.append(
        '<div class="ttier"><div class="ttier-head">④ WORKFLOW · N-rep ensemble '
        "<span>build-once + reseed · is driving it cheaper?</span></div>"
        '<div class="ttier-grid bng">' + wf_cells + "</div></div>"
    )
    parts.append("</div>")
    return "".join(parts)


def render_legacy_stoch_cell(
    outcome: str, timing: dict, regime: str, legacy_label: str, subclass: str | None = None
) -> str:
    """Config + verdict + the legacy timing tiers (per-call only — a fresh process
    each replicate, no warm reuse) for run_network (SSA) / NFsim (NF)."""
    leg = timing.get("legacy") or {}
    cfg = leg.get("config", {})
    badge_text, badge_css = stoch_verdict_badge(outcome, "legacy")
    has = bool(leg) and leg.get("rep_median_sec") is not None

    parts = [
        '<div class="engine-cell">',
        f'<div class="config-section"><div class="config-label">Method</div>'
        f'<div class="config-value">{_esc(cfg.get("method", "—"))}</div></div>',
        f'<div class="config-section"><div class="config-label">Backend</div>'
        f'<div class="config-value">{_esc(cfg.get("codegen", "—"))}</div></div>',
        f'<div class="verdict {badge_css}">{badge_text}</div>',
    ]
    if not has:
        _msg = {
            "REFERENCE_FAILED": f"{legacy_label} errored — no reference ensemble to compare (a reference-side failure; bngsim ran fine, so the row is unscored)",
            "TIMEOUT": f"{legacy_label} did not finish the ensemble within the wall (TIMEOUT)",
            "BAD_TEST": "the model failed to build — neither engine ran",
        }.get(outcome, f"{legacy_label} recorded no ensemble timing for this run")
        parts.append(f'<div class="error-msg">{_esc(_msg)}</div>')
        parts.append("</div>")
        return "".join(parts)

    parts.append(
        '<div class="ttier"><div class="ttier-head">② Per-model build '
        "<span>none — re-reads the artifact at each call</span></div>"
        '<div class="ttier-body"><div class="tcell"><div class="tk">Model build</div>'
        '<div class="tv">— (per-call)</div></div></div></div>'
    )
    # ③ ENGINE — the legacy engine's OWN self-reported inner-loop CPU. The
    # harness-stripped number: it excludes the per-spawn re-init (run_network's Init
    # CPU / NFsim's process startup), so it shows how fast the legacy inner loop truly
    # is. On SSA v21 this is FASTER than bngsim's warm trajectory — the engine the
    # WORKFLOW ratio alone would hide.
    if regime == "ssa":
        eng_cells = _tcell("Propagation CPU", fmt_t(leg.get("propagation_cpu_sec")))
    else:
        rps = leg.get("reactions_per_sec")
        rps_txt = f"{rps:,.0f}/s" if isinstance(rps, (int, float)) and rps else "—"
        eng_cells = _tcell("Total CPU", fmt_t(leg.get("total_cpu_sec"))) + _tcell(
            "Reactions / s", rps_txt
        )
    parts.append(
        '<div class="ttier"><div class="ttier-head">③ ENGINE · per trajectory '
        "<span>engine's own self-reported CPU · the inner loop on its own</span></div>"
        '<div class="ttier-grid bng">' + eng_cells + "</div></div>"
    )
    # ④ WORKFLOW — what you actually pay: a fresh process PER replicate. Every call
    # re-spawns and re-reads the artifact, so the ensemble wall is N × (re-init + sim);
    # the re-init (run_network's Init CPU) is exactly the part bngsim amortizes away.
    n_leg = leg.get("n_calls") or leg.get("n_rep") or 0
    wf_cells = _tcell(f"Ensemble ({n_leg})", fmt_t(leg.get("ensemble_sec"))) + _tcell(
        "Per-call (median)", fmt_t(leg.get("rep_median_sec"))
    )
    if regime == "ssa":
        wf_cells += _tcell("Init CPU / call", fmt_t(leg.get("init_cpu_sec")))
        # Make the per-call wall decomposition explicit: wall ≈ Propagation CPU
        # (tier ③) + Init CPU (re-read/re-init, above) + Spawn+I/O (process fork+exec
        # and .gdat write/read-back). The last term — the cost of being a CLI with no
        # warm mode, paid on EVERY replicate — is otherwise only inferable as a residual.
        rep_med = leg.get("rep_median_sec")
        init_cpu = leg.get("init_cpu_sec")
        prop_cpu = leg.get("propagation_cpu_sec")
        if all(isinstance(x, (int, float)) for x in (rep_med, init_cpu, prop_cpu)):
            wf_cells += _tcell("Spawn+I/O / call", fmt_t(max(0.0, rep_med - init_cpu - prop_cpu)))
    wf_cells += _tcell("First call", fmt_t(leg.get("rep_cold_sec")))
    parts.append(
        '<div class="ttier"><div class="ttier-head">④ WORKFLOW · N-rep ensemble '
        "<span>fresh process each seed · per-call wall = propagation CPU + init CPU + "
        "spawn/I/O, ×N</span></div>"
        '<div class="ttier-grid bng">' + wf_cells + "</div></div>"
    )
    parts.append("</div>")
    return "".join(parts)


def _stoch_wins_table(results, regime: str, legacy_label: str) -> tuple[str, str]:
    """(coststats_html, wins_table_html) — the ENGINE-vs-WORKFLOW split. For each
    model both an ENGINE pair (bngsim warm trajectory vs the legacy's self-reported
    inner-loop CPU) and a WORKFLOW pair (the two N-replicate ensemble walls) are
    collected, binned by observable count, and reduced to two SEPARATE geomeans, so a
    big WORKFLOW amortization number is never read as engine speed."""
    eng_pairs = []  # (bn_engine, leg_engine, n_obs) — harness-stripped inner loop
    wf_pairs = []  # (bn_workflow, leg_workflow, n_obs) — N-rep ensemble wall
    # PASS rows only (GH #190, aligned with rr_parity): a speed comparison is only fair
    # where both engines ran the SAME ensemble to completion AND agreed. A DIFF/TIMEOUT/
    # REF-FAIL row is not apples-to-apples and would bias the geomean.
    n_total = len(results)
    results = [r for r in results if r.get("outcome") == "PASS"]
    n_pass = len(results)
    for r in results:
        t = r.get("timing") or {}
        nobs = ((t.get("spec") or {}).get("n_obs")) or 0
        be, le = _bn_engine(t), _leg_engine(t, regime)
        if be and le and be > 0 and le > 0:
            eng_pairs.append((be, le, nobs))
        bw, lw = _bn_workflow(t), _leg_workflow(t)
        if bw and lw and bw > 0 and lw > 0:
            wf_pairs.append((bw, lw, nobs))
    eng_geo, eng_n = _geomean_ratio([(b, lg) for b, lg, _ in eng_pairs])
    wf_geo, wf_n = _geomean_ratio([(b, lg) for b, lg, _ in wf_pairs])
    if eng_geo is None and wf_geo is None:
        return ('<div class="coststats">No model ran on both engines yet.</div>', "")

    def _dir(geo):
        if geo is None:
            return "—"
        return (
            f"<b>BNGsim {geo:.1f}× faster</b>"
            if geo >= 1
            else f"<b>{_esc(legacy_label)} {1 / geo:.1f}× faster</b>"
        )

    bn_eng_wins = sum(1 for b, lg, _ in eng_pairs if b < lg)
    leg_eng_wins = sum(1 for b, lg, _ in eng_pairs if lg < b)
    bn_wf_wins = sum(1 for b, lg, _ in wf_pairs if b < lg)
    leg_wf_wins = sum(1 for b, lg, _ in wf_pairs if lg < b)
    coststats = (
        '<div class="coststats">'
        f'<div style="color:#6c757d;margin-bottom:6px;">Speed compared over <b>{n_pass} '
        f"PASS model(s)</b> of {n_total} (both engines ran the same 100-replicate ensemble and "
        "agreed) — a non-PASS row is not an apples-to-apples timing comparison.</div>"
        f"<b>ENGINE</b> (per-trajectory inner loop, harness-stripped): over <b>{eng_n}</b> "
        f"model(s) the geomean ratio (legacy self-reported CPU ÷ BNGsim warm trajectory) is "
        f"{_dir(eng_geo)} — BNGsim wins <b>{bn_eng_wins}</b>, {_esc(legacy_label)} <b>{leg_eng_wins}</b>. "
        "This is the real engine-speed question, and the legacy inner loop can win it.<br>"
        f"<b>WORKFLOW</b> (N-replicate ensemble wall — what you actually pay): over <b>{wf_n}</b> "
        f"model(s) the geomean ratio (legacy ensemble ÷ BNGsim ensemble) is {_dir(wf_geo)} — "
        f"BNGsim wins <b>{bn_wf_wins}</b>, {_esc(legacy_label)} <b>{leg_wf_wins}</b>. BNGsim loads the "
        f"model once and reseeds an in-process ensemble; {_esc(legacy_label)} spawns a fresh process "
        "and re-reads the artifact on every replicate, so its per-call re-init is paid N times — the "
        "WORKFLOW win is that amortization, NOT a faster engine.</div>"
    )
    bins = [(0, 1), (2, 3), (4, 6), (7, 12), (13, 25), (26, 60), (61, 10**9)]

    def _label(lo, hi):
        if hi >= 10**9:
            return f"≥{lo}"
        return str(lo) if lo == hi else f"{lo}–{hi}"

    def _gtxt(g):
        if g is None:
            return "—"
        return f"{g:.1f}×" if g >= 1 else f"{1 / g:.1f}× (leg)"

    rows = []
    for lo, hi in bins:
        e_sub = [(b, lg) for b, lg, nobs in eng_pairs if lo <= nobs <= hi]
        w_sub = [(b, lg) for b, lg, nobs in wf_pairs if lo <= nobs <= hi]
        if not e_sub and not w_sub:
            continue
        eg, _ = _geomean_ratio(e_sub)
        wg, _ = _geomean_ratio(w_sub)
        rows.append(
            f"<tr><td>{_label(lo, hi)}</td><td>{len(w_sub)}</td>"
            f"<td>{_gtxt(eg)}</td><td>{_gtxt(wg)}</td></tr>"
        )
    eg_all, _ = _geomean_ratio([(b, lg) for b, lg, _ in eng_pairs])
    wg_all, _ = _geomean_ratio([(b, lg) for b, lg, _ in wf_pairs])
    rows.append(
        f'<tr class="wintot"><td>Total</td><td>{len(wf_pairs)}</td>'
        f"<td>{_gtxt(eg_all)}</td><td>{_gtxt(wg_all)}</td></tr>"
    )
    table = (
        '<div class="winstab"><div class="plottitle">ENGINE vs WORKFLOW geomean by observable count '
        f"(ratio = {_esc(legacy_label)} ÷ BNGsim; &gt;1 ⇒ BNGsim faster, &ldquo;(leg)&rdquo; ⇒ legacy faster)"
        "</div><table><thead><tr><th>observables</th><th>n</th>"
        "<th>ENGINE geomean</th><th>WORKFLOW geomean</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )
    return coststats, table


def _stoch_legend(regime: str, legacy_label: str) -> str:
    rg = _REGIME[regime]
    if regime == "ssa":
        oracle_extra = (
            "over the network species' OBSERVABLES (the <code>.gdat</code> groups), "
            "aligned by name."
        )
        algo_class = "Gillespie SSA"
        bn_path = (
            "loads the <code>.net</code> once, builds the Simulator once, then "
            "<code>reset()</code> + reseeds N Gillespie trajectories in-process"
        )
        leg_path = (
            "<code>run_network -p ssa</code> spawns a fresh process per seed (read .net + "
            "SSA), with its own self-reported init/propagation CPU split"
        )
        eng_self = "<code>run_network</code>'s self-reported <b>propagation CPU</b>"
        eng_through = "events/rep"
        bscb_note = ""
    else:
        oracle_extra = (
            "over the OBSERVABLES (the <code>.gdat</code>; a network-free model has no "
            "species), aligned by name."
        )
        algo_class = "NFsim network-free simulation"
        bn_path = (
            "builds one <code>NfsimSession</code> from the BNG-XML, then reseeds N "
            "network-free trajectories"
        )
        leg_path = (
            "<code>NFsim</code> spawns a fresh process per seed (with <code>-cb</code> when the "
            "model has a Species observable), reporting its own total CPU + reactions/sec"
        )
        eng_self = "<code>NFsim</code>'s self-reported <b>total CPU</b> + reactions/s"
        eng_through = "reaction firings/sec (the <code>globalEventCounter</code>)"
        bscb_note = (
            "<p><b>NF block_same_complex_binding.</b> BNGsim runs at its <b>correct</b> default "
            "(<code>bscb=True</code>); the legacy NFsim v1.14.3 has <b>no -bscb flag</b> and "
            "applies same-complex binding, so on ring-forming / multivalent models the two "
            "legitimately diverge — a <b>known reference-side</b> divergence (the correctness "
            "suite reclassifies these as PASS_REF_BUG), surfaced here as DIFF.</p>"
        )
    return f"""    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-items">
            <div class="legend-item"><div class="legend-color" style="background:#28a745"></div><div class="legend-desc"><b>PASSED</b> — both engines' ensembles agreed within the K-sigma frac-pass gate.</div></div>
            <div class="legend-item"><div class="legend-color" style="background:#dc3545"></div><div class="legend-desc"><b>FAILED</b> — a DIFF: the two ensembles disagree past the gate and it is NOT a known reference artifact. The only "look at bngsim" state. (TIMEOUT is a coverage gap, not a failure — see REFUSED.)</div></div>
            <div class="legend-item"><div class="legend-color" style="background:#ffc107"></div><div class="legend-desc"><b>TRIAGED</b> — a difference explained AWAY from bngsim: a DIFF pinned on the reference (<b>KNOWN REF</b> — the {_esc(legacy_label)} oracle is buggy, confirmed independently) or a benign shared sampling artifact (<b>KNOWN ARTIFACT</b>); EXCEPTION (bngsim raised while {_esc(legacy_label)} ran → actionable bngsim bug); or REFERENCE_FAILED (bngsim ran, {_esc(legacy_label)} raised → no oracle). None scores against bngsim.</div></div>
            <div class="legend-item"><div class="legend-color" style="background:#6c757d"></div><div class="legend-desc"><b>REFUSED</b> — no scored comparison, and NOT a bngsim fault: <b>TIMEOUT</b> (too slow to finish the 100-replicate ensemble within the per-model wall — usually the {_esc(legacy_label)} reference, one subprocess per seed; bngsim itself stays cheap), UNSUPPORTED (bngsim cleanly refused, e.g. validate_for_ssa), BAD_TEST (build failed / both raised), or SKIP (no resolvable horizon).</div></div>
        </div>
        <h3 style="margin-top:18px">Timing tiers — ENGINE vs WORKFLOW</h3>
        <p>Both engines run the <b>same algorithm</b> ({algo_class}), so a single blended "speedup" conflates two different costs. The matrix reports them separately:<br>
        <b>① Per-process startup</b> (header) — one-time per worker: BNGsim's <code>import bngsim</code> (this path imports <b>no SymPy</b>); {_esc(legacy_label)}'s process-spawn proxy. Amortized to ~0 over a run.<br>
        <b>② Per-model build</b> — BNGsim: the {rg["bn_load_label"].lower()} (no Jacobian); eligible mass-action models also compile a structure-specialized propensity vector on replicate 1 (the real backend is shown under <b>Propensities</b>: cc-compiled / interpreted). The Simulator is built ONCE and reseeded per replicate (no per-rep clone/construct). {_esc(legacy_label)} has no build step — it re-reads the artifact on every call.<br>
        <b>③ ENGINE · per trajectory</b> — the harness-stripped inner loop: <i>is our inner loop competitive?</i> BNGsim's <b>warm trajectory</b> (load + setup stripped) plus its stochastic-activity throughput ({eng_through}) vs {eng_self}. This is the real engine-speed question — and the legacy inner loop can sometimes <b>win</b> it.<br>
        <b>④ WORKFLOW · N-rep ensemble</b> — what you actually pay: <i>is driving it cheaper?</i> BNGsim {bn_path}; {leg_path}, so the legacy per-call re-init (its <b>Init CPU</b>) is paid N times. A large WORKFLOW speedup is that <b>amortization</b> — a real, legitimate capability win (the CLIs cannot persist a session), but <b>NOT</b> a faster engine. The legacy "warm"≈"cold": every call re-spawns, so it has no real warm mode (only the BNGsim side shows a meaningful <b>Cold→warm</b> = replicate 1's one-time page-fault/CPU ramp vs the median of the rest).<br>
        <b>{rg["build"]}</b> (model cell) is the <b>shared build prefix</b> — the same artifact feeds both engines, so it is attributed to neither.</p>
        <h3 style="margin-top:18px">Oracle</h3>
        <p>Cross-engine STOCHASTIC tolerance via the shared <code>_core.differ.ensemble_verdict</code> {oracle_extra} Each model runs an N-replicate ensemble per engine on an identical seed schedule; the two ensembles are reduced to per-cell (mean, sem) and compared with a per-cell K-sigma test where <b>K = 3·√(N/10)</b> (effect-size-preserving — a larger replicate count sharpens the statistical precision without mechanically tightening the meaningful-difference threshold; so 100 reps does not manufacture DIFFs that 10 reps would have passed) plus a relative-agreement escape hatch and a near-zero skip. <b>PASS</b> iff ≥ 99% of cells pass; <code>frac_pass</code> is the realized fraction, <code>max_z</code> the worst per-cell z.</p>
        {bscb_note}
    </div>
"""


def generate_stoch_html(report_path: Path, output_path: Path) -> None:
    """Render a stochastic (SSA/NF) report (``runs/report_ssa.json`` /
    ``runs/report_nf.json`` from ``bng_stoch_run.py``) to the BNGsim-vs-legacy
    matrix — the stochastic sibling of :func:`generate_html`, same taxonomy and
    layout, with the ensemble verdict and per-replicate timing tiers."""
    data = json.loads(Path(report_path).read_text())
    meta = data.get("_meta", {})
    results = data.get("results", [])
    ver = meta.get("versions", {})
    tally_ = meta.get("tally", {})
    hw = meta.get("hardware", {})
    conc = meta.get("concurrency", {})
    regime = meta.get("regime", "ssa")
    rg = _REGIME[regime]
    legacy_label = meta.get("reference_label", "legacy")
    ens = meta.get("ensemble", {})

    def _nobs(r):
        return ((r.get("timing") or {}).get("spec") or {}).get("n_obs") or 0

    def _activity(r):
        bn = (r.get("timing") or {}).get("bngsim") or {}
        return bn.get("events_per_rep") if bn.get("events_per_rep") is not None else _nobs(r)

    # Sort by the complexity axis (events/rep for SSA, observables for NF).
    results = sorted(results, key=lambda r: (_activity(r) or 0, r.get("model_id", "")))

    # Warmup aggregate (per-process startup tier, shown once).
    bn_warm = [(r.get("timing") or {}).get("warmup", {}).get("bngsim_sec") for r in results]
    bn_warm = [x for x in bn_warm if x]
    warm_txt = (
        f"BNGsim startup (import bngsim): median {fmt_t(sorted(bn_warm)[len(bn_warm) // 2])} "
        f"over {len(bn_warm)} workers"
        if bn_warm
        else "—"
    )

    coststats, wins = _stoch_wins_table(results, regime, legacy_label)
    runtime_html = _runtime_settings_html(meta, results)

    # The scatter plots the WORKFLOW view (the N-rep ensemble wall, what you pay) —
    # the ENGINE story lives in the per-row badges + the wins table's ENGINE column.
    # PASS rows only (consistent with the wins table + rr_parity): a fair speed
    # scatter needs both engines to have run the SAME ensemble to completion and
    # agreed. Pair each model's (BNGsim, legacy) ensemble wall and sort by BNGsim
    # cost (low→high) so the x-ordering matches the axis label.
    _wf_pairs = [
        (_bn_workflow(r.get("timing")) or 0.0, _leg_workflow(r.get("timing")) or 0.0)
        for r in results
        if r.get("outcome") == "PASS"
    ]
    _wf_pairs.sort(key=lambda p: (p[0] <= 0, p[0]))
    bn_series = [round(a, 8) for a, _ in _wf_pairs]
    leg_series = [round(b, 8) for _, b in _wf_pairs]
    plot_json = json.dumps({"bn": bn_series, "rn": leg_series})

    # Four buckets, identical to the rr_parity SSA matrix (GH #190): PASSED (green),
    # FAILED (red, real bngsim-fault DIFF), TRIAGED (yellow, a DIFF explained away from
    # bngsim — reference-side bug / shared artifact / EXCEPTION / REFERENCE_FAILED),
    # REFUSED (gray coverage gap — TIMEOUT / UNSUPPORTED / BAD_TEST / SKIP). There is no
    # separate "KNOWN" bucket: a not-bngsim-fault DIFF folds into TRIAGED.
    known_n = sum(
        1
        for r in results
        if r.get("outcome") == "DIFF" and r.get("subclass") in ("ref_bscb", "known_artifact")
    )
    pass_n = tally_.get("PASS", 0)
    fail_n = tally_.get("DIFF", 0) - known_n  # real bngsim-fault DIFFs only
    triaged_n = tally_.get("EXCEPTION", 0) + tally_.get("REFERENCE_FAILED", 0) + known_n
    refused_n = (
        tally_.get("BAD_TEST", 0)
        + tally_.get("SKIP", 0)
        + tally_.get("UNSUPPORTED", 0)
        + tally_.get("TIMEOUT", 0)
    )
    _ssa_total = max(len(results), 1)

    def _pct(c):
        return 100.0 * c / _ssa_total

    rows_html = []
    for order, r in enumerate(results):
        outcome = r.get("outcome", "")
        row_cls, _cat = classify_row_stoch(outcome, r.get("subclass"))
        timing = r.get("timing") or {}
        spec = timing.get("spec") or {}
        build = (timing.get("build") or {}).get(rg["build_key"])
        model_id = r.get("model_id", "")
        nobs = spec.get("n_obs") or 0
        activity = _activity(r)
        val = r.get("value")
        metric_html = ""
        if val is not None and math.isfinite(val):
            alert = " alert" if (r.get("tol") and val < r.get("tol")) else ""
            metric_html = f'<div class="metric{alert}">points in agreement: {val * 100:.1f}%</div>'
        comment = r.get("comment") or ""
        exc = r.get("exception") or ""
        comment_html = ""
        if comment or exc:
            comment_html = (
                f'<div class="row-comment">{_esc(comment)}</div>' if comment else ""
            ) + (f'<div class="row-detail">{_esc(exc)}</div>' if exc else "")

        feats = []
        if spec:
            feats.append(f"t_end={spec.get('t_end')}")
            feats.append(f"{spec.get('n_steps')} steps")
            feats.append(f"{spec.get('n_rep')} reps")
        tier = "/".join(model_id.split("/")[:2]) if "/" in model_id else ""
        feat_html = "".join(f'<span class="feature-badge">{_esc(f)}</span>' for f in feats)
        build_html = (
            f'<div class="model-netgen">{rg["build"]} <span>(shared prefix)</span>: '
            f"<b>{fmt_t(build)}</b></div>"
            if build is not None
            else ""
        )
        bn_t = timing.get("bngsim") or {}
        activity_html = ""
        if regime == "ssa" and bn_t.get("events_per_rep") is not None:
            activity_html = (
                f'<div class="model-netgen">stochastic activity: '
                f"<b>{bn_t.get('events_per_rep'):,.0f}</b> events/rep "
                f"<span>({bn_t.get('events_per_time', 0):,.0f}/time-unit)</span></div>"
            )
        elif regime == "nf" and bn_t.get("reactions_per_rep") is not None:
            activity_html = (
                f'<div class="model-netgen">stochastic activity: '
                f"<b>{bn_t.get('reactions_per_rep'):,.0f}</b> reactions/rep "
                f"<span>({bn_t.get('reactions_per_time', 0):,.0f}/time-unit)</span></div>"
            )
        # Explicit per-row disposition tag (PASSED / FAILED / KNOWN … / TIMEOUT /
        # TRIAGED / REFUSED), colored to match the row + the summary chips + the
        # legend — so the category is named on every row, not just implied by color.
        _tagcls = {
            "status-passed": "summary-passed",
            "status-failed": "summary-failed",
            "status-known": "summary-known",
            "status-triaged": "summary-triaged",
            "status-refused": "summary-refused",
        }.get(row_cls, "summary-refused")
        rowtag = f'<span class="rowtag {_tagcls}">{_esc(_cat)}</span>'
        model_cell = (
            f'<div class="model-info">{rowtag} {_esc(model_id)}</div>'
            f'<div class="model-metadata">{nobs} observables · {_esc(tier)}</div>'
            f"{build_html}"
            f"{activity_html}"
            f"{metric_html}"
            f"{_stoch_engine_badge(timing, regime, legacy_label)}"
            f"{_stoch_workflow_badge(timing, legacy_label)}"
            f'<div class="model-features">{feat_html}</div>'
            f"{comment_html}"
        )

        bn_cell = render_bngsim_stoch_cell(outcome, timing, regime, r.get("subclass"))
        leg_cell = render_legacy_stoch_cell(
            outcome, timing, regime, legacy_label, r.get("subclass")
        )
        bn_wf = _bn_workflow(timing) or 0
        leg_wf = _leg_workflow(timing) or 0
        wf_spd = (leg_wf / bn_wf) if (bn_wf and leg_wf) else 0
        bn_eng = _bn_engine(timing) or 0
        leg_eng = _leg_engine(timing, regime) or 0
        eng_spd = (leg_eng / bn_eng) if (bn_eng and leg_eng) else 0
        rows_html.append(
            f'<tr class="{row_cls}" data-order="{order}" data-id="{_esc(model_id)}" '
            f'data-nsp="{activity or 0}" data-nobs="{nobs}" data-bnens="{bn_wf}" '
            f'data-legens="{leg_wf}" data-wfspeedup="{wf_spd}" data-engspeedup="{eng_spd}">'
            f"<td>{model_cell}</td><td>{bn_cell}</td><td>{leg_cell}</td></tr>"
        )

    hwline = (
        f"{_esc(hw.get('cpu', '?'))} · {hw.get('physical_cores', '?')} cores · "
        f"timings measured under {conc.get('workers', '?')}-way process parallelism "
        "(absolute costs inflate under contention; ratios are robust)."
    )
    k_sigma = ens.get("k_sigma", 3.0)
    pass_frac = ens.get("pass_frac", 0.99)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>bng_parity {rg["track"]} — BNGsim vs {_esc(legacy_label)}</title>
<style>{_CSS}</style>
</head><body>
    <div class="header">
        <h1>bng_parity — BNGsim vs the legacy BNG stack ({rg["track"]})</h1>
        <div class="metadata">
            <strong>bngsim</strong> {_esc(ver.get("bngsim"))} &nbsp;·&nbsp;
            <strong>BioNetGen</strong> {_esc(ver.get("bng"))} &nbsp;·&nbsp;
            git {_esc(meta.get("git_rev"))} &nbsp;·&nbsp; {_esc(meta.get("generated"))}<br>
            {_esc(legacy_label)}: <span class="path-info">{_esc(meta.get("legacy_bin"))}</span><br>
            ensemble oracle: <code>_core.differ.ensemble_verdict</code>
            (K={_esc(k_sigma)}σ = 3·√(N/10), frac≥{_esc(pass_frac)}) &nbsp;·&nbsp; {len(results)} {rg["track"]} jobs<br>
            {_esc(warm_txt)}
        </div>
        <div class="hwcaveat">{hwline}</div>
        <div class="summary">
            <div class="summary-item summary-passed">✓ {pass_n} PASSED ({_pct(pass_n):.1f}%)</div>
            <div class="summary-item summary-failed">✗ {fail_n} FAILED ({_pct(fail_n):.1f}%)</div>
            <div class="summary-item summary-triaged">⚠ {triaged_n} TRIAGED ({_pct(triaged_n):.1f}%)</div>
            <div class="summary-item summary-refused">⊘ {refused_n} REFUSED ({_pct(refused_n):.1f}%)</div>
        </div>
        {runtime_html}
        <details class="catdefs" open style="margin:6px 0 2px 0;">
            <summary style="cursor:pointer;font-weight:600;color:#495057;font-size:0.9em;">How is each of the four categories assigned?</summary>
            <div style="font-size:0.85em;color:#495057;line-height:1.65;margin:6px 0;">
                <div style="margin-bottom:5px;"><span style="color:#1e7e34;font-weight:700;">PASSED</span> — both engines ran the full N-replicate ensemble and their per-cell means agreed within the K-σ frac-pass gate (≥99% of cells).</div>
                <div style="margin-bottom:5px;"><span style="color:#dc3545;font-weight:700;">FAILED</span> — a DIFF the gate pins on bngsim (ensembles disagree past K-σ and it is not a known reference artifact). The only "look at bngsim" state.</div>
                <div style="margin-bottom:5px;"><span style="color:#856404;font-weight:700;">TRIAGED</span> — a difference explained AWAY from bngsim: a DIFF pinned on the reference (KNOWN REF — the {_esc(legacy_label)} oracle is the buggy one, confirmed independently) or a benign shared sampling artifact (KNOWN ARTIFACT); EXCEPTION (bngsim raised while {_esc(legacy_label)} ran → actionable bngsim bug); or REFERENCE_FAILED ({_esc(legacy_label)} raised while bngsim ran → no oracle). None scores against bngsim.</div>
                <div><span style="color:#383d41;font-weight:700;">REFUSED</span> — no scored comparison, and NOT a bngsim fault: TIMEOUT (too slow to finish 100 reps in the wall — usually the per-seed-subprocess {_esc(legacy_label)} reference), UNSUPPORTED (bngsim cleanly declined), BAD_TEST (both raised), or SKIP (no horizon).</div>
            </div>
        </details>
        {coststats}
        <div class="plots">
            <div class="plotctl">
                <strong>y-scale:</strong>
                <label><input type="radio" name="plotscale" value="log" checked onchange="drawPlot()"> log</label>
                <label><input type="radio" name="plotscale" value="linear" onchange="drawPlot()"> linear</label>
                &nbsp;&nbsp;<span class="dot bn"></span> BNGsim (circle)
                <span class="dot rn"></span> {_esc(legacy_label)} (square)
                &nbsp;·&nbsp; <strong>filled = faster (timing win)</strong>, open = slower
                <span class="sortnote">WORKFLOW view: the N-replicate ensemble wall (what you actually pay), lower is faster. The harness-stripped ENGINE comparison is in the per-row badges + the wins table.</span>
            </div>
            <div class="plottitle">WORKFLOW: N-replicate ensemble wall — BNGsim vs {_esc(legacy_label)} (one circle + one square per model)</div>
            <canvas id="cv" class="costplot"></canvas>
            {wins}
        </div>
    </div>
    <div class="sortbar">
        <label>Sort by:
        <select id="sortsel" onchange="doSort()">
            <option value="nsp">{rg["complexity"]}</option>
            <option value="nobs">observable count</option>
            <option value="id">model id</option>
            <option value="wfspeedup">WORKFLOW speedup</option>
            <option value="engspeedup">ENGINE speedup</option>
            <option value="bnens">BNGsim ensemble time</option>
            <option value="legens">{_esc(legacy_label)} ensemble time</option>
        </select></label>
        <span class="sortdir">
            <label><input type="radio" name="sortdir" value="asc" checked onchange="doSort()"> asc</label>
            <label><input type="radio" name="sortdir" value="desc" onchange="doSort()"> desc</label>
        </span>
        <span class="sortnote">ties fall back to the default complexity order</span>
    </div>
    <table id="matrix">
        <thead><tr>
            <th class="col-model">Model</th>
            <th class="col-engine">BNGsim (in-process {rg["track"]})</th>
            <th class="col-engine">{_esc(legacy_label)} (legacy binary)</th>
        </tr></thead>
        <tbody>
        {"".join(rows_html)}
        </tbody>
    </table>
    {_stoch_legend(regime, legacy_label)}
    <script>var PLOT = {plot_json};</script>
    {_plot_js("models, sorted by BNGsim ensemble cost (low → high)")}
    {_SORT_JS}
</body></html>
"""
    Path(output_path).write_text(html)


def classify_row_stoch(outcome: str, subclass: str | None = None) -> tuple[str, str]:
    """Row color class for a stochastic outcome (UNSUPPORTED joins the gray bucket).

    A DIFF carrying a known non-scoring disposition renders distinctly instead of FAILED
    (GH #183/#69): ``ref_bscb`` -> KNOWN REF (the legacy reference engine is the buggy
    oracle; bngsim confirmed by an independent oracle), ``known_artifact`` -> KNOWN
    ARTIFACT (a benign sampling artifact both engines agree on). Neither is a bngsim bug.
    """
    # A DIFF that is NOT a bngsim fault (buggy reference oracle, or a benign shared
    # sampling artifact) is TRIAGED (yellow) — the same bucket rr_parity puts its
    # diff_not_bngsim / rr_known / RR-LIMIT cases in. Kept with a descriptive tag.
    if outcome == "DIFF" and subclass == "ref_bscb":
        return "status-triaged", "KNOWN REF"
    if outcome == "DIFF" and subclass == "known_artifact":
        return "status-triaged", "KNOWN ARTIFACT"
    # Scored against the INDEPENDENT .net Gillespie SECOND oracle (the legacy reference
    # crashed — e.g. edgepop). Distinct badge so it is not blended with a run_network
    # verdict, but it still counts (PASS = bngsim confirmed by an independent engine;
    # DIFF = bngsim genuinely disagrees with the .net oracle, actionable).
    if subclass == "oracle_net_gillespie":
        if outcome == "PASS":
            return "status-passed", "PASS (net oracle)"
        if outcome in ("DIFF", "TIMEOUT"):
            return "status-failed", "DIFF (net oracle)"
    if outcome == "PASS":
        return "status-passed", "PASSED"
    if outcome == "DIFF":
        return "status-failed", "FAILED"
    # TIMEOUT is a COVERAGE GAP (the model was too slow to finish 100 reps in the
    # wall budget — most often the BNG2.pl reference, one subprocess per seed), NOT a
    # correctness failure. Gray, like SKIP/UNSUPPORTED — never red (it does not mean
    # "look at bngsim").
    if outcome == "TIMEOUT":
        return "status-refused", "TIMEOUT"
    if outcome in ("REFERENCE_FAILED", "EXCEPTION"):
        return "status-triaged", "TRIAGED"
    return "status-refused", "REFUSED"  # BAD_TEST / SKIP / UNSUPPORTED


def main() -> int:
    args = sys.argv[1:]
    report = Path(args[0]) if args else HERE / "runs" / "report_ode.json"
    if not report.exists():
        sys.exit(f"missing report: {report} (run bng_ode_run.py / bng_stoch_run.py first)")
    # Dispatch on the report's regime: ode -> the deterministic matrix; ssa/nf ->
    # the stochastic matrix. The default out name follows the regime.
    regime = json.loads(report.read_text()).get("_meta", {}).get("regime", "ode")
    default_out = report.parent / f"bng_matrix_{regime}.html"
    out = Path(args[1]) if len(args) > 1 else default_out
    if regime in ("ssa", "nf"):
        generate_stoch_html(report, out)
    else:
        generate_html(report, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
