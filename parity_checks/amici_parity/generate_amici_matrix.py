#!/usr/bin/env python3
"""Generate HTML parity matrix from report_ode.json.

Renders the bngsim-vs-AMICI ODE sweep results as a filterable HTML table.
Color scheme:
  - Green: PASSED
  - Red: FAILED (DIFF or TIMEOUT)
  - Yellow: TRIAGED (REFERENCE_FAILED or EXCEPTION)
  - Gray: REFUSED (BAD_TEST - both engines rejected)

Both engine columns have the same color per row. Within each cell, verdict badges
show specific outcomes (PASS, DIFF, EXCEPTION, etc.).

Usage:
    python generate_matrix.py [--report runs/report_ode.json] [--out runs/parity_matrix.html]
"""

from __future__ import annotations

import argparse
import json
import re
import sys as _sys
from datetime import datetime
from html import escape as _escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
# This generator lives in amici_parity/, but the SBML corpus (models/), the
# model→path manifest (ode_jobs.json), and _sbml_features.py all live in the
# sibling rr_parity/. Put parity_checks/ (for _core) AND rr_parity/ on the path
# before importing the shared feature extractor; model paths resolve under RR_PARITY.
RR_PARITY = HERE.parent / "rr_parity"
_sys.path.insert(0, str(HERE.parent))
_sys.path.insert(0, str(RR_PARITY))

# Make the shared parity protocol importable so the DIFF-tolerance legend reads its
# constants LIVE from the source of truth (parity_checks/_core/differ.py) rather than
# restating them as annotation that could drift. Falls back to None if unavailable.
import sys as _sys  # noqa: E402

from _sbml_features import extract_sbml_features  # noqa: E402

_sys.path.insert(0, str(HERE.parent))
try:
    from _core import differ as _differ
except Exception:  # pragma: no cover - legend just falls back to documented literals
    _differ = None


def classify_row(outcome: str) -> tuple[str, str]:
    """Return (css_class, category_label) for the entire row.

    PASSED (green) = PASS
    FAILED (red) = DIFF or TIMEOUT
    TRIAGED (yellow) = REFERENCE_FAILED or EXCEPTION
    REFUSED (gray) = BAD_TEST
    """
    if outcome == "PASS":
        return "status-passed", "PASSED"
    elif outcome in ("DIFF", "TIMEOUT"):
        return "status-failed", "FAILED"
    elif outcome in ("REFERENCE_FAILED", "EXCEPTION"):
        return "status-triaged", "TRIAGED"
    elif outcome == "BAD_TEST":
        return "status-refused", "REFUSED"
    else:
        return "status-refused", "REFUSED"


def get_verdict_badge(outcome: str, engine: str, succeeded: bool) -> tuple[str, str]:
    """Return (badge_text, badge_color) for verdict badge within a cell.

    engine: 'bngsim' or 'amici'
    succeeded: did this engine succeed?
    """
    if outcome == "PASS":
        return "PASS", "verdict-pass"
    elif outcome == "DIFF":
        return "DIFF", "verdict-diff"
    elif outcome == "TIMEOUT":
        return "TIMEOUT", "verdict-diff"
    elif outcome == "EXCEPTION":
        # Default for the common case: bngsim raised, AMICI ran. The caller
        # refines this per exception_kind() — a worker crash (unattributable) and
        # a compare-step raise must NOT show bngsim EXCEPTION + AMICI PASS.
        if engine == "bngsim":
            return "EXCEPTION", "verdict-exception"
        else:
            return "PASS", "verdict-pass"
    elif outcome == "REFERENCE_FAILED":
        # bngsim succeeded, AMICI failed
        if engine == "bngsim":
            return "PASS", "verdict-pass"
        else:
            return "REF-FAIL", "verdict-reffail"
    elif outcome == "BAD_TEST":
        return "REFUSED", "verdict-refused"
    else:
        return outcome, "verdict-refused"


def exception_kind(result: dict) -> str:
    """Sub-classify an EXCEPTION row by what actually failed, from the recorded
    ``exception`` (rr_run formats each source with a distinct prefix).

    EXCEPTION lumps several different things together; only the first is an
    actionable bngsim defect with AMICI as a proven oracle:

    - ``"bngsim"`` — bngsim raised while AMICI ran (``exception`` is
      ``bngsim: …``): an actionable bngsim bug.
    - ``"compare"`` — both engines ran but the comparison step raised
      (``exception`` is ``compare: …``): a harness/data issue, not an engine
      fault — AMICI did produce a trajectory.
    - ``"dead"`` — the worker died with no per-engine status, so the scheduler
      synthesized the result with an **empty** ``exception`` (the comment is
      ``worker died (exit=…)``). That empty string is the definitive crash
      signature. AMICI is documented to segfault on pathological SBML, so a
      crash is **unattributable**: either engine may have been the one that died.
    - ``"other"`` — a non-empty exception with none of the known prefixes. The
      current runner never emits this (a AMICI raise is REFERENCE_FAILED,
      not EXCEPTION), but a legacy report predating per-engine attribution can
      carry a ``amici: …`` EXCEPTION; render it without blaming bngsim.

    Only meaningful when ``outcome == "EXCEPTION"``.
    """
    exc = result.get("exception") or ""
    if not exc:
        return "dead"
    if exc.startswith("bngsim:"):
        return "bngsim"
    if exc.startswith("compare:"):
        return "compare"
    return "other"


def fmt_ms(sec: float | None) -> str:
    """Format seconds as milliseconds with appropriate precision."""
    if sec is None or sec == 0:
        return "—"
    ms = sec * 1000
    if ms < 1:
        return f"{ms:.2f}ms"
    elif ms < 10:
        return f"{ms:.1f}ms"
    elif ms < 1000:
        return f"{int(ms)}ms"
    else:
        return f"{ms / 1000:.2f}s"


def parse_phase_timing(timing: dict, *, combined_key: str, combined_folds_codegen: bool) -> dict:
    """Resolve per-phase seconds from one engine's timing dict, applying the
    combined-phase fallback.

    When the fine-grained ``parse_sec``/``interpret_sec`` (and, for an engine
    whose combined phase also folds in codegen, ``codegen_sec``) are all absent
    or zero, the engine reported a single inseparable combined phase under
    ``combined_key`` — it cannot split them without an instrumented build. Show
    that combined value in ``parse`` and mark the folded sub-phases ``None`` (the
    renderer prints "—"). Returns ``{io, parse, interpret, jac, codegen, compile,
    load, integrate, total}``; ``total`` treats ``None`` sub-phases as 0.

    Phase notes: ``jac`` = the analytical-Jacobian symbolic derivation (BOTH
    engines now report it — bngsim via SymPy/GH#76, AMICI via its dxdot/dx symbolic
    chain captured from its build logs); ``compile`` = AMICI's separate C++
    compilation step (cmake/ninja/swig/link — its dominant pre-sim cost; ``None``
    for bngsim, which folds any native ``cc`` compile into ``codegen``); ``load`` =
    AMICI's compiled-extension import (``None`` for bngsim).
    """
    io = timing.get("io_sec", 0)
    parse = timing.get("parse_sec", 0)
    interpret = timing.get("interpret_sec", 0)
    codegen = timing.get("codegen_sec", 0)
    # Analytical-Jacobian symbolic derivation — a per-model phase BOTH engines
    # report (bngsim: SymPy, GH#76; AMICI: the dxdot/dx chain). ``None`` only for a
    # pre-instrumentation report → the renderer prints "—". Folded into the total.
    jac = timing.get("jac_derive_sec")
    take_combined = (
        parse == 0 and interpret == 0 and (codegen == 0 if combined_folds_codegen else True)
    )
    if take_combined:
        parse = timing.get(combined_key, 0)
        interpret = None
        if combined_folds_codegen:
            codegen = None
    integrate = timing.get("integrate_sec", 0)
    # AMICI-only extra build tiers (None for bngsim → renderer prints "—").
    compile_ = timing.get("compile_sec")
    load = timing.get("load_sec")
    total = (
        io
        + (parse or 0)
        + (interpret or 0)
        + (jac or 0)
        + (codegen or 0)
        + (compile_ or 0)
        + (load or 0)
        + integrate
    )
    return {
        "io": io,
        "parse": parse,
        "interpret": interpret,
        "jac": jac,
        "codegen": codegen,
        "compile": compile_,
        "load": load,
        "integrate": integrate,
        "total": total,
    }


# --------------------------------------------------------------------------- #
# Display normalizers (T-review 2026-06-14): ONE canonical label per concept,
# used identically in both engine columns so the same solver/backend never
# appears under two names, and an engine's architectural defaults (AMICI is
# always FD + CVODE dense) are shown even on a failed run rather than a wrong
# guess. bngsim's config is run-dependent, so an absent bngsim config shows "—".
# --------------------------------------------------------------------------- #
def load_manifest_paths(suite_dir: Path) -> dict:
    """Map ``model_id -> real suite-relative SBML path`` from ``ode_jobs.json`` —
    the SAME manifest the runner resolves models through. ~Half the corpus does NOT
    use the ``<id>_url.xml`` naming (the vendored file keeps its SED-ML-referenced
    name), so the old path GUESS silently dropped species/reaction/citation
    metadata for ~50% of rows. Reading the manifest is the truth; returns {} (and
    the renderer falls back to the guess) only if the manifest is unreadable."""
    try:
        data = json.loads((suite_dir / "ode_jobs.json").read_text())
        return {j["model_id"]: j["model"] for j in data.get("jobs", []) if j.get("model")}
    except (OSError, ValueError, KeyError):
        return {}


def resolve_sbml_path(suite_dir: Path, model_id: str, manifest_paths: dict) -> Path | None:
    """The real SBML path for a model: the manifest path first, then the legacy
    ``models/<id>/<id>_url.xml`` guess. ``None`` if neither exists."""
    rel = manifest_paths.get(model_id)
    if rel:
        p = (suite_dir / rel).resolve()
        if p.exists():
            return p
    guess = suite_dir / "models" / model_id / f"{model_id}_url.xml"
    return guess if guess.exists() else None


def jacobian_display(raw: str | None, *, engine: str) -> str:
    """Human-readable Jacobian strategy. AMICI ALWAYS derives an analytical
    (symbolic) Jacobian — a fixed property of its generated model — so an AMICI cell
    with no recorded config (a failed/refused run) still shows analytical. bngsim's
    strategy is run-dependent (analytical / FD fallback), so an absent bngsim config
    shows "—" rather than a guess. The actual scheme is spelled out either way."""
    if raw:
        key = raw.strip().lower()
        if key.startswith("fd") or "difference" in key:
            return "Finite-difference approximation (difference quotient)"
        if key.startswith("analytic"):
            return "Analytical (symbolic, SymPy)"
        if key == "jax":
            return "JAX automatic differentiation"
        return raw
    # AMICI ALWAYS derives an analytic (symbolic) Jacobian at codegen — a fixed
    # property of its generated model — so a configless (failed/refused) AMICI cell
    # still shows analytic, never a wrong FD default. bngsim's strategy is
    # run-dependent, so an absent bngsim config shows "—".
    return "Analytical (symbolic, SymPy)" if engine == "amici" else "—"


def linear_solver_display(raw: str | None, config: dict, *, engine: str) -> str:
    """Canonical linear-solver name, identical across columns for the same solver
    (items 10/11). All three are LU factorizations; the distinction is purely the
    implementation, so the names say so:
        kind 0 "Dense"        → CVODE dense (SUNDIALS built-in LU)
        kind 2 "LAPACK-dense" → LAPACK dense (dgetrf LU)   [+ GH#132 dgetrf count]
        kind 1 "KLU"          → KLU (sparse LU)
    AMICI records ``"Dense (built-in LU)"`` for the SAME kind-0 solver bngsim
    records as ``"Dense"`` — both normalize to one string so they no longer look
    like two different dense solvers."""
    if raw is None:
        # AMICI defaults to KLU (sparse LU) — show it even on a configless cell.
        return "KLU (sparse LU)" if engine == "amici" else "—"
    key = raw.strip().lower()
    if key.startswith("lapack"):
        if "dense_blas_factorizations" in config:
            n_blas = config.get("dense_blas_factorizations", 0)
            n_fac = config.get("n_factorizations", 0)
            return (
                f"LAPACK dense (dgetrf LU; {n_blas}/{n_fac} factorizations took dgetrf)"
                if n_blas
                else "LAPACK dense (dgetrf LU; gate not crossed — identical to built-in dense)"
            )
        return "LAPACK dense (dgetrf LU)"
    if key.startswith("klu"):
        return "KLU (sparse LU)"
    return "CVODE dense (SUNDIALS built-in LU)"


def rhs_backend_display(raw: str | None, *, engine: str) -> str:
    """How the ODE right-hand side is evaluated — the row the matrix mislabeled
    "Codegen" (item 5). ExprTk is a bytecode interpreter, NOT codegen; ``cc`` /
    MIR / LLVM are the actual compiled backends. AMICI generates per-model C++ and
    compiles it to a shared library (gcc/clang)."""
    if engine == "amici":
        return raw or "Per-model C++ (gcc/clang-compiled .so)"
    if not raw:
        return "—"
    return {
        "exprtk": "ExprTk bytecode (interpreted — no codegen)",
        "cc": "Native C (cc-compiled .so)",
        "mir": "MIR JIT (compiled in-process)",
    }.get(raw.strip().lower(), raw)


def cache_display(config: dict, *, engine: str) -> str:
    """Plain-English compiled-artifact cache state (item 3). This is the codegen
    cache and is ORTHOGONAL to the cold/warm integration split: the RHS backend is
    resolved ONCE here, before any solve, whereas cold-vs-warm is CVODE's one-time
    solver setup on the first integration. An engine that produced no config
    (crashed/refused) shows "—"."""
    if not config:
        return "—"
    cached = config.get("cached")
    if engine == "amici":
        # AMICI's "cache" is the on-disk compiled C++ extension (amici_cache/),
        # reused across runs by SBML-content hash; a miss pays the ~20s codegen.
        return (
            "Reused compiled C++ extension from disk cache"
            if cached
            else "Compiled fresh this run (per-model C++ codegen, ~20s)"
        )
    if cached is None:
        return "No compiled artifact (ExprTk bytecode VM)"
    return "Reused cached .so from disk" if cached else "Compiled fresh this run"


def ode_solver_display(meta: dict, *, engine: str) -> str:
    """The CVODE configuration needed to reproduce a solve (item 9): both engines
    integrate with CVODE using the BDF (stiff) formula and a Newton nonlinear
    solver, at the shared rtol/atol forced on both. The SUNDIALS version is
    bngsim's vendored build (read from ``sundials_config.h``); AMICI bundles
    its own CVODE, whose version it does not separately expose."""
    cfg = meta.get("config", {})
    rtol, atol = cfg.get("rtol"), cfg.get("atol")
    tol = f"rtol={rtol:g}, atol={atol:g}" if rtol is not None and atol is not None else "—"
    if engine == "amici":
        rrv = meta.get("versions", {}).get("amici", "?")
        build = f"bundled with AMICI {rrv}"
    else:
        sv = meta.get("versions", {}).get("sundials")
        build = f"SUNDIALS {sv}" if sv else "SUNDIALS (version unrecorded)"
    return f"CVODE — BDF formula, Newton iteration · {build} · {tol}"


def humanize_comment(result: dict) -> str:
    """Translate the runner's terse, dev-oriented ``comment`` into plain English,
    and SUPPRESS the all-passed cell-fail breakdown on a clean PASS — it says
    nothing the green badge doesn't (item 4). Returns "" when there is nothing
    worth showing. Unknown shapes pass through verbatim so nothing is silently
    lost. The refusal/timeout/crash parts are intentionally dropped here because
    they are already spelled out in the per-engine footnote on the same row."""
    raw = (result.get("comment") or "").strip()
    if not raw:
        return ""
    outcome = result.get("outcome")
    out: list[str] = []
    for part in (p.strip() for p in raw.split(" | ")):
        if not part:
            continue
        m = re.match(
            r"(\d+) sp; fail (\d+)/(\d+) \(hard (\d+), soft (\d+), forgiven (\d+)\)", part
        )
        if m:
            n_sp, n_fail, n_cells, n_hard, _n_soft, n_forg = (int(x) for x in m.groups())
            if n_fail == 0 and outcome == "PASS":
                continue  # clean pass — say nothing
            seg = (
                f"{n_fail:,} of {n_cells:,} compared points "
                f"(over {n_sp} shared species) exceeded tolerance"
            )
            extra = []
            if n_hard:
                extra.append(f"{n_hard:,} beyond the hard ceiling")
            if n_forg:
                extra.append(f"{n_forg:,} forgiven by the fail-fraction budget")
            out.append(seg + (f" ({'; '.join(extra)})" if extra else "") + ".")
            continue
        m = re.match(r"known artifact([^:]*): (.+)", part)
        if m:
            out.append(f"Known artifact{m.group(1)} — reclassified to PASS: {m.group(2)}")
            continue
        m = re.match(r"adjudicated (\w+)([^:]*): (.+)", part)
        if m:
            out.append(f"Adjudicated {m.group(1)}{m.group(2)}: {m.group(3)}")
            continue
        # Refusal / timeout / crash — already in the per-engine footnote; drop here.
        if re.match(r"reference refusal=\w+|killed at [\d.]+s wall cap|worker died", part):
            continue
        out.append(part)
    return " ".join(out)


_SORT_SENTINEL = 1e18


def _data_num(v) -> str:
    """A numeric ``data-*`` sort value: the number itself, or a large sentinel for
    a missing/unparsed value so it sorts LAST in the client-side sort."""
    return f"{v:.9g}" if isinstance(v, (int, float)) else f"{_SORT_SENTINEL:.9g}"


def _tcell(label: str, value: str) -> str:
    """One label-over-value cell in a per-model / per-integration timing tier."""
    return f'<div class="tcell"><div class="tk">{label}</div><div class="tv">{value}</div></div>'


def render_timing_tiers(timing: dict, *, engine: str) -> str:
    """The three cost tiers as three labeled lines, the structure requested in the
    2026-06-14 review:

      1. Per-process startup — the once-per-run warmup (≈ constant across models),
         surfaced on every row so it is findable without hunting the header.
      2. Per-model build — the one-time cost to turn SBML into a ready model,
         broken into I/O · Parse · Interpret · Jacobian · RHS build, with a subtotal.
         "RHS build" = generate the RHS/Jacobian evaluator code AND compile it to a
         callable form — ONE phase, identically defined for both engines (compilation
         is part of building the evaluator, not a separate step). bngsim: ExprTk
         bytecode (~0, interpreted) / native-C (cc) compile / MIR JIT. AMICI: C++
         codegen + compilation — dominated by the ~20-30 s compile.
      3. Per-integration — the bottom line: WARM (the marginal cost a reused
         Simulator pays) and TOTAL, COLD (the first solve = one-time CVODE setup
         PLUS the solve, i.e. the full cost of one integration from scratch).
    """
    warmup = timing.get("warmup") or {}
    if engine == "bngsim":
        eng = timing.get("bngsim") or {}
        phases = parse_phase_timing(
            eng, combined_key="parse_interpret_sec", combined_folds_codegen=False
        )
        proc_label, proc_val = "SymPy import", warmup.get("bngsim_sec")
    else:
        eng = timing.get("amici") or {}
        phases = parse_phase_timing(
            eng, combined_key="parse_interpret_codegen_sec", combined_folds_codegen=True
        )
        proc_label, proc_val = "Library import", warmup.get("amici_sec")

    build = (
        phases["io"]
        + (phases["parse"] or 0)
        + (phases["interpret"] or 0)
        + (phases["jac"] or 0)
        + (phases["codegen"] or 0)
        + (phases["compile"] or 0)
        + (phases["load"] or 0)
    )
    interp = fmt_ms(phases["interpret"]) if phases["interpret"] is not None else "—"
    jac = fmt_ms(phases["jac"]) if phases["jac"] is not None else "—"
    # RHS build = codegen (emit the evaluator code) + compile (build it to a callable
    # form) + load, folded into ONE phase so the row means the same thing for both
    # engines. Compilation is a SUB-STEP of building the RHS evaluator, not a peer
    # phase: bngsim reports it as a single inseparable codegen number (and ExprTk has
    # no compile at all); AMICI's separable emit/compile/load are summed here.
    rhs_has = phases["codegen"] is not None or phases["compile"] is not None
    rhs_total = (phases["codegen"] or 0) + (phases["compile"] or 0) + (phases["load"] or 0)
    rhs = fmt_ms(rhs_total) if rhs_has else "—"

    cold = eng.get("integrate_cold_sec")
    warm_min = eng.get("integrate_warm_min_sec")
    warm_max = eng.get("integrate_warm_max_sec")
    warm_med = eng.get("integrate_warm_median_sec")
    n_warm = eng.get("integrate_n_warm", 0)
    if warm_min is not None and n_warm:
        spread = (
            fmt_ms(warm_min) if warm_min == warm_max else f"{fmt_ms(warm_min)}–{fmt_ms(warm_max)}"
        )
        warm_val = f"median {fmt_ms(warm_med)} · range {spread} (n={n_warm})"
    else:
        warm_val = "— (no warm reps)"

    per_model = "".join(
        [
            _tcell("I/O", fmt_ms(phases["io"])),
            _tcell("Parse (libSBML)", fmt_ms(phases["parse"])),
            _tcell("Interpret", interp),
            _tcell("Jacobian", jac),
            _tcell("RHS build", rhs),
            _tcell("Build subtotal", fmt_ms(build)),
        ]
    )
    per_integ = "".join(
        [
            _tcell("Warm (marginal, reused)", warm_val),
            _tcell("Total, cold (first solve)", fmt_ms(cold)),
        ]
    )
    return f"""<div class="ttier">
                        <div class="ttier-head">① Per-process startup <span>once per run · shared by every model · excluded from per-model totals</span></div>
                        <div class="ttier-body">{_tcell(proc_label, fmt_ms(proc_val))}</div>
                    </div>
                    <div class="ttier">
                        <div class="ttier-head">② Per-model build <span>once per model</span></div>
                        <div class="ttier-grid">{per_model}</div>
                    </div>
                    <div class="ttier">
                        <div class="ttier-head">③ Per-integration <span>the bottom line — what fitting / MCMC pays per evaluation</span></div>
                        <div class="ttier-grid two">{per_integ}</div>
                    </div>"""


def timeout_cap_sec(result: dict, default: float = 1500.0) -> float:
    """The wall-clock cap (s) a TIMEOUT row was killed at, parsed from its comment
    ("killed at 1500.0s wall cap"). The cap is the one hard fact a timeout DOES give
    us: a lower bound on the reference engine's cost. Falls back to ``default``."""
    m = re.search(r"killed at ([\d.]+)s", result.get("comment", "") or "")
    return float(m.group(1)) if m else default


def load_rr_timing_index() -> dict:
    """Borrow BNGsim's per-model timing from the rr_parity full ODE report for rows
    THIS sweep could not record itself — TIMEOUT rows, where the shared subprocess was
    killed at the wall cap (during AMICI's C++ compile) BEFORE it returned BNGsim's
    result. BNGsim runs these models; AMICI is the laggard (the suite README: >500-
    species models can't compile in reasonable time). BNGsim and AMICI run through the
    identical ``bn_ode`` adapter and timing schema, so rr_parity's ``bngsim`` sub-dict
    grafts straight into ``render_timing_tiers`` — the cell is labeled with this
    provenance so it is never mistaken for a number measured in this run.

    Returns ``(index, src_version)`` where ``index`` is
    ``{model_id: {"bngsim": dict|None, "warmup": dict, "rr_outcome": str}}`` for every
    model rr_parity scored (``bngsim`` is ``None`` when rr_parity also lacked a clean
    BNGsim run, e.g. it reclassified the row BAD_TEST/TIMEOUT) and ``src_version`` is the
    bngsim version that rr_parity run was measured at (for the provenance label — it can
    differ from this sweep's). ``({}, "")`` if the rr_parity report is absent — the
    feature then degrades to the cap lower bound only.
    """
    import json

    p = HERE.parent / "rr_parity" / "runs" / "report_ode.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}, ""
    res = d.get("results", d) if isinstance(d, dict) else d
    src_version = (
        (d.get("_meta", {}) if isinstance(d, dict) else {}).get("versions", {}).get("bngsim", "")
    )
    out: dict = {}
    for r in res:
        t = r.get("timing") or {}
        out[r["model_id"]] = {
            "bngsim": t.get("bngsim"),
            "warmup": t.get("warmup") or {},
            "rr_outcome": r.get("outcome", "?"),
        }
    return out, src_version


def render_amici_timeout_tiers(cap: float) -> str:
    """AMICI timing cell for a TIMEOUT row. The job was killed at the wall cap before
    AMICI finished, so no phase number was recorded — but the cap is a hard LOWER BOUND
    on AMICI's cost (it provably ran ≥ cap). For these giant models the unfinished phase
    is the C++ codegen/compile, so the bound lands on the build; per-integration is never
    reached. Mirrors ``render_timing_tiers`` structure so the column stays aligned."""
    lb = f"&gt; {cap:.0f}s"
    per_model = "".join(
        [
            _tcell("I/O", "—"),
            _tcell("Parse (libSBML)", "—"),
            _tcell("Interpret", "—"),
            _tcell("Jacobian", "—"),
            _tcell("RHS build", lb),
            _tcell("Build subtotal", lb),
        ]
    )
    per_integ = "".join(
        [
            _tcell("Warm (marginal, reused)", "not reached"),
            _tcell("Total, cold (first solve)", "not reached"),
        ]
    )
    return f"""<div class="ttier">
                        <div class="ttier-head">① Per-process startup <span>once per run · shared by every model · excluded from per-model totals</span></div>
                        <div class="ttier-body">{_tcell("Library import", "—")}</div>
                    </div>
                    <div class="ttier">
                        <div class="ttier-head">② Per-model build <span>once per model · killed at the {cap:.0f}s wall cap</span></div>
                        <div class="ttier-grid">{per_model}</div>
                    </div>
                    <div class="ttier">
                        <div class="ttier-head">③ Per-integration <span>the bottom line — what fitting / MCMC pays per evaluation</span></div>
                        <div class="ttier-grid two">{per_integ}</div>
                    </div>"""


def read_corpus_partition() -> dict | None:
    """Read the committed stage-1 keep/drop partition (the biomodels suite's
    manifest.csv) so the corpus-provenance note is documented from the source of
    truth, not restated by hand. Returns ``{total, keep, drops:{reason:count},
    libsbml}`` or ``None`` if the manifest isn't where expected."""
    import csv as _csv

    path = HERE.parents[1] / "benchmarks" / "suites" / "biomodels" / "manifest.csv"
    try:
        lines = path.read_text().splitlines()
        comment = next((line for line in lines if line.startswith("#")), "")
        m = re.search(r"libsbml ([\d.]+)", comment)
        libsbml = m.group(1) if m else ""
        rows = list(_csv.DictReader([line for line in lines if not line.startswith("#")]))
        keep = sum(1 for r in rows if r.get("verdict") == "keep")
        drops: dict[str, int] = {}
        for r in rows:
            if r.get("verdict") == "drop":
                reason = r.get("reason") or "?"
                drops[reason] = drops.get(reason, 0) + 1
        return {"total": len(rows), "keep": keep, "drops": drops, "libsbml": libsbml}
    except (OSError, ValueError):
        return None


def format_run_config(meta: dict) -> dict:
    """Extract the bngsim method combo for the HTML header from ``_meta["config"]``
    (T2.1). Back-compat: a report written before the config block existed has no
    ``"config"`` key, so every lookup is a ``.get`` with a sensible default and
    nothing raises. Returns ``{combo, tol, overrides}`` ready to interpolate.
    """
    config = meta.get("config", {})
    combo = config.get("combo", "auto")
    rtol, atol = config.get("rtol"), config.get("atol")
    tol = f"rtol={rtol:g}, atol={atol:g}" if rtol is not None and atol is not None else "—"
    active = {k: v for k, v in (config.get("env") or {}).items() if v is not None}
    if config.get("force_dense_linear_solver"):
        active["force_dense_linear_solver"] = True
    overrides = (
        ", ".join(f"{k}={v}" for k, v in active.items()) if active else "none (engine auto)"
    )
    return {"combo": combo, "tol": tol, "overrides": overrides}


def _warmup_summary_html(results: list) -> str:
    """Header line for the per-process warmup cost, aggregated over all jobs.

    Each rr_parity job is its own subprocess, so each carries one warmup sample
    (``timing.warmup`` — bngsim SymPy import, AMICI ``.so`` import). Reports
    mean ± std (and median) per engine over every job that recorded one. Returns
    "" when no job carries warmup (e.g. a pre-instrumentation report). Warmup is a
    once-per-process cost amortized to ~0 in fitting/MCMC — shown here, never inside
    a per-model Load/Total.
    """
    import statistics as _st

    def _samples(key):
        out = []
        for r in results:
            w = (r.get("timing") or {}).get("warmup") or {}
            v = w.get(key)
            if v is not None:
                out.append(float(v))
        return out

    bn, rr = _samples("bngsim_sec"), _samples("amici_sec")
    if not bn and not rr:
        return ""

    def _stat(xs):
        if not xs:
            return "—"
        sd = _st.stdev(xs) if len(xs) > 1 else 0.0
        return (
            f"{_st.mean(xs) * 1000:.0f} ± {sd * 1000:.0f} ms "
            f"(median {_st.median(xs) * 1000:.0f} ms, n={len(xs)})"
        )

    sources = {
        (r.get("timing") or {}).get("warmup", {}).get("amici_source")
        for r in results
        if (r.get("timing") or {}).get("warmup")
    }
    rr_src = next((s for s in sources if s), "")
    rr_tag = f" (measured from AMICI's {_escape(rr_src)})" if rr_src else ""
    return (
        '<div style="margin-top: 12px; line-height: 1.6;"><strong>Per-process startup cost.</strong> '
        "Before any model is loaded, each engine pays a one-time cost to import its "
        "libraries. This happens once per run, is nearly the same for every model, and "
        "is never counted in a model's per-model or per-integration time below — a fitting "
        "or MCMC run amortizes it to almost nothing. Averaged over every job in this sweep: "
        f"BNGsim's SymPy import took {_stat(bn)}, and AMICI's library import took "
        f"{_stat(rr)}{rr_tag}.</div>"
    )


def bngsim_total_sec(result: dict) -> float | None:
    """Total bngsim wall (io + parse_interpret + codegen + integrate) for one
    result row, or ``None`` if it carries no bngsim timing (a failed/refused row
    has nothing to compare). Reuses the same phase resolution the matrix renders.
    """
    bn = (result.get("timing") or {}).get("bngsim")
    if not bn:
        return None
    return parse_phase_timing(
        bn, combined_key="parse_interpret_sec", combined_folds_codegen=False
    )["total"]


def combo_from_report(meta: dict, report_path: Path) -> str:
    """The combo a report belongs to: the recorded _meta config combo (T2.1), or,
    for a report written before that block, parsed from the report filename
    (report_ode.json → auto, report_ode__<combo>.json → <combo>)."""
    combo = (meta.get("config") or {}).get("combo")
    if combo:
        return combo
    stem = report_path.stem  # e.g. "report_ode__mir"
    return stem.split("__", 1)[1] if "__" in stem else "auto"


def aggregate_combo_speedups(combos: dict) -> dict:
    """Per-combo aggregate bngsim-time speedup vs the ``auto`` baseline.

    ``combos`` maps combo name → ``{"totals": {model_id: bngsim_total_sec}, ...}``.
    For each non-auto combo, the speedup is ``sum(auto) / sum(combo)`` over the
    models BOTH ran with timing (so a model only one combo could time never skews
    the ratio) — >1 means the combo is faster. Returns combo → ``{speedup,
    n_common, auto_sec, combo_sec}`` (speedup None when there is no auto baseline
    or no shared timed model).
    """
    auto = combos.get("auto")
    out: dict[str, dict] = {}
    for combo, info in combos.items():
        if combo == "auto" or auto is None:
            out[combo] = {"speedup": None, "n_common": 0, "auto_sec": None, "combo_sec": None}
            continue
        common = set(info["totals"]) & set(auto["totals"])
        if not common:
            out[combo] = {"speedup": None, "n_common": 0, "auto_sec": None, "combo_sec": None}
            continue
        combo_sec = sum(info["totals"][m] for m in common)
        auto_sec = sum(auto["totals"][m] for m in common)
        speedup = (auto_sec / combo_sec) if combo_sec > 0 else None
        out[combo] = {
            "speedup": speedup,
            "n_common": len(common),
            "auto_sec": auto_sec,
            "combo_sec": combo_sec,
        }
    return out


def generate_html(report_path: Path, output_path: Path) -> None:
    """Generate HTML matrix from report JSON."""
    report = json.loads(report_path.read_text())
    meta = report["_meta"]
    results = report["results"]

    # Resolve each model's REAL SBML path from the job manifest (item 6 — not the
    # old <id>_url.xml guess, which missed ~half the corpus) and extract its
    # structural features ONCE here, so the complexity sort and the per-row
    # rendering share exactly the same counts.
    # The corpus + manifest live under rr_parity/ (RR_PARITY), not this suite dir.
    manifest_paths = load_manifest_paths(RR_PARITY)
    features_by_id: dict[str, dict] = {}
    for r in results:
        mid = r["model_id"]
        p = resolve_sbml_path(RR_PARITY, mid, manifest_paths)
        features_by_id[mid] = extract_sbml_features(p) if p else {}

    # Sort smallest → largest by structural complexity (item 1): species, then
    # reactions, then parameters, then compartments, then model_id. model_id is
    # unique, so this is a deterministic total order. A model whose SBML could not
    # be read (no counts) sorts last via a +inf sentinel, then by id.
    def _n(feat, key):
        v = feat.get(key)
        return v if isinstance(v, int) else float("inf")

    def sort_key(r):
        f = features_by_id[r["model_id"]]
        return (
            _n(f, "n_species"),
            _n(f, "n_reactions"),
            _n(f, "n_parameters"),
            _n(f, "n_compartments"),
            r["model_id"],
        )

    results_sorted = sorted(results, key=sort_key)

    tally = meta["tally"]
    n_jobs = meta["n_jobs"]
    elapsed = meta["elapsed_sec"]
    versions = meta["versions"]

    # Defensive summary counts: a report predating the all-outcomes tally seeding
    # (or any hand-written one) can omit an outcome key, and n_jobs could be 0 on
    # an empty report — so read with .get and guard the percentage divisor rather
    # than KeyError/ZeroDivisionError mid-render.
    n_pass = tally.get("PASS", 0)
    n_failed = tally.get("DIFF", 0) + tally.get("TIMEOUT", 0)
    n_triaged = tally.get("REFERENCE_FAILED", 0) + tally.get("EXCEPTION", 0)
    n_refused = tally.get("BAD_TEST", 0)
    pct = (lambda c: 100 * c / n_jobs) if n_jobs else (lambda c: 0.0)

    # Per-process warmup (the third cost type) — aggregated over every job, since
    # each job runs in its own subprocess (one warmup sample apiece). Shown once in
    # the header, never folded into any per-model Load.
    warmup_html = _warmup_summary_html(results)

    # T2.2: the bngsim method combo this sweep ran under (back-compat safe).
    run_config = format_run_config(meta)
    combo = run_config["combo"]
    combo_tol = run_config["tol"]
    combo_overrides = run_config["overrides"]

    # Hardware + concurrency context (so the timing numbers are interpretable): the
    # CPU/OS the sweep ran on, and that per-integration costs were collected under
    # N-way process concurrency (not a quiescent serial machine).
    hw = meta.get("hardware") or {}
    conc = meta.get("concurrency") or {}
    workers = conc.get("workers")
    hardware_html = ""
    if hw or workers:
        cores = []
        if hw.get("physical_cores"):
            cores.append(f"{hw['physical_cores']} physical")
        if hw.get("logical_cores"):
            cores.append(f"{hw['logical_cores']} logical")
        cores_s = (" / ".join(cores) + " cores") if cores else ""
        cpu = hw.get("cpu", "unknown")
        plat = hw.get("platform", versions.get("platform", "?"))
        hardware_html = (
            f"<div><strong>Hardware:</strong> {cpu}"
            f"{' · ' + cores_s if cores_s else ''} · {plat}</div>"
        )
        if workers:
            on_cores = f" on {hw['logical_cores']} cores" if hw.get("logical_cores") else ""
            hardware_html += (
                '<div class="hwcaveat"><strong>Timing context:</strong> per-integration costs '
                f"were measured with <strong>{workers} worker jobs running concurrently</strong>"
                f"{on_cores} (process-parallel, for throughput), so absolute costs include CPU/memory "
                "contention and are <strong>not</strong> quiescent single-process times. The "
                "BNGsim-vs-AMICI comparison is unaffected — both engines run back-to-back in the "
                "same job under the same ambient load.</div>"
            )

    # Corpus provenance — where the SBML comes from and how it was filtered, read
    # from the committed stage-1 manifest so the counts can't drift.
    part = read_corpus_partition()
    provenance_html = ""
    if part:
        drop_str = ", ".join(
            f"{n} {reason.replace('_', ' ')}"
            for reason, n in sorted(part["drops"].items(), key=lambda x: -x[1])
        )
        libsbml = f"libSBML {part['libsbml']} " if part["libsbml"] else ""
        provenance_html = f"""<details class="provenance">
            <summary>Corpus &amp; provenance — how these {part["keep"]} models were selected</summary>
            <div class="prov-body">
                <p><strong>Source.</strong> SBML is downloaded from the <strong>EBI BioModels</strong> database — curated <code>BIOMD*</code> and non-curated <code>MODEL*</code> ODE models — via its REST API (<code>fetch.py</code>).</p>
                <p><strong>Filter.</strong> A stage-1 {libsbml}pass partitions the {part["total"]} downloaded models into the <strong>{part["keep"]} structurally ODE-simulable</strong> models shown here and <strong>{part["total"] - part["keep"]} dropped</strong> ({drop_str}). The criterion is <em>un-simulability</em>, not SBML-validity severity: <em>no model</em> = libSBML read no model element (unreadable / non-SBML); <em>zero dynamics</em> = no reactions and no rate rules; <em>undefined symbol</em> = a kinetic law or trigger references an undeclared symbol (AMICI rejects these too); <em>missing kinetic law</em> = a reaction has no rate. Events, <strong>event delays</strong>, algebraic rules, and function definitions are <strong>kept</strong> (tagged), never dropped.</p>
                <p><strong>Protocol.</strong> SBML carries no simulation horizon, so each model's time grid and per-model tolerances come from SED-ML (the sys-bio/temp-biomodels mirror), tiered by provenance: a curated figure-reproducing SED-ML &gt; a generic template &gt; an invented placeholder for the uncurated <code>MODEL*</code> ids. The parity check then forces one shared rtol/atol on both engines.</p>
                <p><strong>Nothing is dropped silently.</strong> A kept model that one engine ultimately cannot load or integrate is reported <em>loudly</em> as REFERENCE_FAILED or BAD_TEST in the table below — never quietly removed.</p>
            </div>
        </details>"""

    # Big-picture cost plots (per-integration cost vs model index, in the table's
    # complexity sort order). Built from the same warm-median / cold figures the
    # cells show. Two charts: warm (marginal) and cold (first solve), each BNGsim
    # vs AMICI, log-scaled y.
    n_models = len(results_sorted)
    bn_warm = [
        ((r.get("timing") or {}).get("bngsim") or {}).get("integrate_warm_median_sec")
        for r in results_sorted
    ]
    rr_warm = [
        ((r.get("timing") or {}).get("amici") or {}).get("integrate_warm_median_sec")
        for r in results_sorted
    ]
    bn_cold = [
        ((r.get("timing") or {}).get("bngsim") or {}).get("integrate_cold_sec")
        for r in results_sorted
    ]
    rr_cold = [
        ((r.get("timing") or {}).get("amici") or {}).get("integrate_cold_sec")
        for r in results_sorted
    ]

    # Per-model BUILD SUBTOTAL (tier ②: SBML → ready RHS/Jacobian evaluator) — the
    # SAME number the cells' "Build subtotal" shows, via the SAME parse_phase_timing
    # resolution (so plot ≡ cells). This is bngsim's headline build advantage: ExprTk
    # / native-C codegen in ms–s vs AMICI's ~20-30 s per-model C++ compile.
    def _build_subtotal(eng_timing, *, engine):
        if not eng_timing:
            return None
        if engine == "bngsim":
            ph = parse_phase_timing(
                eng_timing, combined_key="parse_interpret_sec", combined_folds_codegen=False
            )
        else:
            ph = parse_phase_timing(
                eng_timing,
                combined_key="parse_interpret_codegen_sec",
                combined_folds_codegen=True,
            )
        b = (
            ph["io"]
            + (ph["parse"] or 0)
            + (ph["interpret"] or 0)
            + (ph["jac"] or 0)
            + (ph["codegen"] or 0)
            + (ph["compile"] or 0)
            + (ph["load"] or 0)
        )
        return b if b > 0 else None

    bn_build = [
        _build_subtotal((r.get("timing") or {}).get("bngsim"), engine="bngsim")
        for r in results_sorted
    ]
    rr_build = [
        _build_subtotal((r.get("timing") or {}).get("amici"), engine="amici")
        for r in results_sorted
    ]

    # Aggregate cost comparison. For a ratio of paired costs the GEOMETRIC mean is
    # the right average (it's scale-symmetric: 2x faster and 2x slower cancel, and
    # one outlier can't dominate the way it does an arithmetic mean). Computed over
    # models where BOTH engines produced a positive cost.
    import math as _math

    def _pairs(bn_list, rr_list):
        return [
            (a, b)
            for a, b in zip(bn_list, rr_list, strict=False)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0
        ]

    def _geomean_ratio(bn_list, rr_list):
        """Geometric mean of BNGsim/AMICI cost ratios (<1 ⇒ BNGsim faster)."""
        ps = _pairs(bn_list, rr_list)
        if not ps:
            return None, 0
        return _math.exp(sum(_math.log(a / b) for a, b in ps) / len(ps)), len(ps)

    def _ratio_phrase(geo):
        if geo is None:
            return "—"
        if geo < 1:
            return f"{geo:.2f} → BNGsim {1 / geo:.2f}× faster"
        return f"{geo:.2f} → BNGsim {geo:.2f}× slower"

    warm_geo, warm_geo_n = _geomean_ratio(bn_warm, rr_warm)
    cold_geo, cold_geo_n = _geomean_ratio(bn_cold, rr_cold)
    build_geo, build_geo_n = _geomean_ratio(bn_build, rr_build)

    def _build_totals():
        """Sum build subtotals over models where BOTH engines built (shared
        denominator), regardless of the cross-engine verdict — build is a property
        of each engine's own compile, not of their agreement."""
        bn_s = rr_s = 0.0
        nn = 0
        for a, b in zip(bn_build, rr_build, strict=False):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0:
                bn_s += a
                rr_s += b
                nn += 1
        return bn_s, rr_s, nn

    build_bn_tot, build_rr_tot, build_tot_n = _build_totals()
    species_seq = [features_by_id.get(r["model_id"], {}).get("n_species") for r in results_sorted]

    # Aggregate wall time to evaluate every green (PASS) model once, summed over the
    # models where BOTH engines have a cost (so the two totals share a denominator).
    def _total_cost(metric):
        bn_s = rr_s = 0.0
        nn = 0
        for r in results_sorted:
            if r["outcome"] != "PASS":
                continue
            t = r.get("timing") or {}
            a = (t.get("bngsim") or {}).get(metric)
            b = (t.get("amici") or {}).get(metric)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0:
                bn_s += a
                rr_s += b
                nn += 1
        return bn_s, rr_s, nn

    warm_bn_tot, warm_rr_tot, warm_tot_n = _total_cost("integrate_warm_median_sec")
    cold_bn_tot, cold_rr_tot, cold_tot_n = _total_cost("integrate_cold_sec")

    def _tot_phrase(bn_s, rr_s):
        if bn_s <= 0 or rr_s <= 0:
            return "—"
        faster = "BNGsim" if bn_s < rr_s else "AMICI"
        ratio = max(bn_s, rr_s) / min(bn_s, rr_s)
        return (
            f"BNGsim <strong>{fmt_ms(bn_s)}</strong> vs AMICI <strong>{fmt_ms(rr_s)}</strong> "
            f"— {faster} {ratio:.2f}× faster overall"
        )

    # Wins binned by SPECIES COUNT (upper-inclusive edges; a final >1000 bin), not
    # by equal model-count — equal-count bins can't subdivide the high-species tail
    # (only ~15 models exceed 500 species, so they'd lump into one bucket). These
    # same edges drive the plot x-axis ticks below, so the two line up.
    SPECIES_EDGES = [2, 5, 10, 20, 35, 51, 100, 250, 500, 1000]

    def _species_ranges():
        ranges, prev = [], None
        for e in SPECIES_EDGES:
            ranges.append((prev, e))
            prev = e
        ranges.append((prev, None))  # the open-ended >last-edge bin
        return ranges

    def _bin_label(lo, hi):
        if hi is None:
            return f"&gt;{lo}"
        if lo is None:
            return f"≤{hi}"
        return f"{lo + 1}–{hi}"

    def _wins_rows(bn_list, rr_list):
        """Per species-count bin: (label, n compared, BNGsim wins, AMICI wins)
        by which engine had the lower cost."""
        rows = []
        for lo, hi in _species_ranges():
            seg = [
                (bn_list[i], rr_list[i])
                for i, s in enumerate(species_seq)
                if isinstance(s, int)
                and (lo is None or s > lo)
                and (hi is None or s <= hi)
                and isinstance(bn_list[i], (int, float))
                and isinstance(rr_list[i], (int, float))
                and bn_list[i] > 0
                and rr_list[i] > 0
            ]
            bn_w = sum(1 for a, b in seg if a < b)
            rr_w = sum(1 for a, b in seg if b < a)
            rows.append((_bin_label(lo, hi), len(seg), bn_w, rr_w))
        return rows

    def _win_row_html(label, nseg, bn_w, rr_w, *, total=False):
        bn_pct = (100 * bn_w / nseg) if nseg else 0
        weight = "700" if total else "600"
        cls = ' class="wintot"' if total else ""
        return (
            f"<tr{cls}>"
            f"<td>{label}</td><td>{nseg}</td>"
            f'<td style="color:#0072B2;font-weight:{weight};">{bn_w}</td>'
            f'<td style="color:#D55E00;font-weight:{weight};">{rr_w}</td>'
            f'<td><div class="winbar"><span style="width:{bn_pct:.0f}%;"></span></div></td>'
            "</tr>"
        )

    def _wins_table_html(title, rows):
        body = "".join(_win_row_html(*r) for r in rows)
        body += _win_row_html(
            "all models",
            sum(r[1] for r in rows),
            sum(r[2] for r in rows),
            sum(r[3] for r in rows),
            total=True,
        )
        return (
            f'<div class="winstab"><div class="plottitle">{title}</div>'
            "<table><thead><tr><th>Species</th><th>n</th>"
            '<th style="color:#0072B2;">BNGsim wins</th><th style="color:#D55E00;">AMICI wins</th>'
            "<th>BNGsim win fraction</th></tr></thead><tbody>"
            f"{body}</tbody></table></div>"
        )

    # Plot x-axis ticks: the model index at the right edge of each species bin,
    # labeled by species count, plus the largest model — so the plot's x-axis lines
    # up with the wins-table bins. (The plot's x is still model index / complexity
    # order; these just label it by species at the bin boundaries.)
    def _plot_xticks():
        out = {}
        for e in SPECIES_EDGES:
            idx = None
            for i, s in enumerate(species_seq):
                if isinstance(s, int) and s <= e:
                    idx = i
            if idx is not None:
                out[idx] = str(e)
        valid = [i for i, s in enumerate(species_seq) if isinstance(s, int)]
        if valid:
            out[valid[-1]] = str(species_seq[valid[-1]])
        return sorted([k, v] for k, v in out.items())

    plot_xticks = _plot_xticks()

    wins_html = (
        '<div class="winsgrid">'
        + _wins_table_html("Warm-cost wins by species bin", _wins_rows(bn_warm, rr_warm))
        + _wins_table_html("Cold-cost wins by species bin", _wins_rows(bn_cold, rr_cold))
        + "</div>"
    )
    stats_html = (
        '<div class="coststats">'
        f"<strong>Geometric-mean cost ratio (BNGsim : AMICI)</strong> — "
        f"warm: <strong>{_ratio_phrase(warm_geo)}</strong> (n={warm_geo_n}) &nbsp;·&nbsp; "
        f"cold: <strong>{_ratio_phrase(cold_geo)}</strong> (n={cold_geo_n})"
        '<div style="margin-top:7px;padding-top:7px;border-top:1px solid #e3e6ea;">'
        "<strong>Total wall time to evaluate every green (PASS) model once</strong>"
        f'<div style="margin-top:3px;">warm — reused Simulators (n={warm_tot_n}): {_tot_phrase(warm_bn_tot, warm_rr_tot)}</div>'
        f"<div>cold — fresh Simulators (n={cold_tot_n}): {_tot_phrase(cold_bn_tot, cold_rr_tot)}</div>"
        "</div>"
        '<div style="font-size:0.85em;color:#868e96;margin-top:6px;">Why warm <em>n</em> can be '
        "below cold <em>n</em>: every model does 1 cold solve plus <strong>up to 5</strong> warm reps, "
        "but the warm count is scaled back toward 0 as a model's cold solve approaches the time budget "
        "(so a slow / stiff solve never multiplies wall time). A few slow models do 0 warm reps — they "
        "have a cold cost but no warm median — so they drop out of the warm aggregates. It is not a parse error.</div>"
        "</div>"
    )

    # Per-model BUILD cost section (tier ②) — its own block, since build is a one-time
    # per-model cost (SBML → ready evaluator), categorically different from the per-
    # integration tier ③ above. This is where the engines diverge most: bngsim's ms–s
    # codegen vs AMICI's ~20-30 s per-model C++ compile.
    build_stats_html = (
        '<div class="coststats">'
        "<strong>Geometric-mean build-subtotal ratio (BNGsim : AMICI)</strong> — "
        f"<strong>{_ratio_phrase(build_geo)}</strong> (n={build_geo_n} models both engines built)"
        '<div style="margin-top:7px;padding-top:7px;border-top:1px solid #e3e6ea;">'
        "<strong>Total wall time to build every model once</strong> (n="
        f"{build_tot_n}, both engines built): {_tot_phrase(build_bn_tot, build_rr_tot)}</div>"
        '<div style="font-size:0.85em;color:#868e96;margin-top:6px;">Build subtotal = '
        "I/O + Parse + Interpret + Jacobian + RHS build (the same per-model figure each "
        "cell shows). TIMEOUT giants are absent here — AMICI never finished their compile, "
        "so there is no paired build number; the table rows show AMICI &gt; the wall cap.</div>"
        "</div>"
    )
    build_wins_html = (
        '<div class="winsgrid">'
        + _wins_table_html("Build-subtotal wins by species bin", _wins_rows(bn_build, rr_build))
        + "</div>"
    )
    build_section_html = f"""<div class="plots" style="margin-top:18px;">
        <h2 style="margin:0 0 6px 0;font-size:1.05em;color:#333;">Per-model build cost across the corpus <span style="font-weight:400;font-size:0.85em;color:#777;">(one-time: SBML → ready RHS/Jacobian evaluator)</span></h2>
        {build_stats_html}
        <div class="plotwrap"><div class="plottitle">Build subtotal — BNGsim codegen (ExprTk / native-C, ms–s) vs AMICI per-model C++ compile (~20-30 s)</div><canvas id="cv_build" class="costplot"></canvas></div>
        {build_wins_html}
    </div>"""

    def _round(xs):
        return [round(x, 7) if isinstance(x, (int, float)) else None for x in xs]

    plot_data_json = json.dumps(
        {
            "n": n_models,
            "bw": _round(bn_warm),
            "rw": _round(rr_warm),
            "bc": _round(bn_cold),
            "rc": _round(rr_cold),
            "bb": _round(bn_build),
            "rb": _round(rr_build),
            # x-axis ticks: [model_index, species_label] at each species-bin edge,
            # so the plot's x-axis lines up with the wins-table bins below.
            "xt": plot_xticks,
        },
        separators=(",", ":"),
    )
    # Canvas scatter (drawn client-side): one point per engine per model, filled =
    # the faster engine for that model (a "win"), open = the slower. Log/linear
    # toggle. Only models where both engines have a cost are plotted. Canvas (not
    # 2600 SVG nodes) so it stays fast and redraws instantly on toggle/resize.
    plots_html = f"""<div class="plots">
        <h2 style="margin:0 0 6px 0;font-size:1.05em;color:#333;">Per-integration cost across the corpus</h2>
        <div class="plotctl">
            <strong>y-scale:</strong>
            <label><input type="radio" name="plotscale" value="log" checked onchange="drawPlots()"> log</label>
            <label><input type="radio" name="plotscale" value="linear" onchange="drawPlots()"> linear (robust, capped at 99th pct)</label>
            <span class="plotkey"><span class="dot fill" style="background:#0072B2;border-color:#0072B2;"></span>/<span class="dot" style="border-color:#0072B2;"></span> BNGsim (circle) &nbsp; <span class="dot sq fill" style="background:#D55E00;border-color:#D55E00;"></span>/<span class="dot sq" style="border-color:#D55E00;"></span> AMICI (square) &nbsp;·&nbsp; <strong>filled = faster (win)</strong>, open = slower &nbsp;·&nbsp; x = species count (models in complexity order, n={n_models})</span>
        </div>
        {stats_html}
        <div class="plotwrap"><div class="plottitle">Warm (marginal, reused) — median of the warm solves</div><canvas id="cv_warm" class="costplot"></canvas></div>
        <div class="plotwrap"><div class="plottitle">Total, cold (first solve) — incl. one-time CVODE setup</div><canvas id="cv_cold" class="costplot"></canvas></div>
        {wins_html}
        <script>
        var PLOT_DATA = {plot_data_json};
        function _scatter(canvasId, bn, rr, logScale) {{
            var cv = document.getElementById(canvasId);
            if (!cv) return;
            var cssW = cv.clientWidth || 940, cssH = 210;
            var dpr = window.devicePixelRatio || 1;
            cv.width = cssW * dpr; cv.height = cssH * dpr; cv.style.height = cssH + 'px';
            var ctx = cv.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, cssW, cssH);
            var padL = 60, padR = 12, padT = 10, padB = 34, pw = cssW - padL - padR, ph = cssH - padT - padB;
            var n = PLOT_DATA.n, vals = [];
            for (var i = 0; i < n; i++) {{ if (bn[i] > 0) vals.push(bn[i]); if (rr[i] > 0) vals.push(rr[i]); }}
            if (!vals.length) return;
            var ymin = Math.min.apply(null, vals), ymax = Math.max.apply(null, vals);
            var lo, hi, cap = null;
            if (logScale) {{ lo = Math.floor(Math.log10(ymin)); hi = Math.ceil(Math.log10(ymax)); if (hi <= lo) hi = lo + 1; }}
            else {{
                // Robust linear: cap the axis at the 99th percentile of the data so a
                // few slow models don't flatten everything else; points above the cap
                // are clamped to the top edge and counted (see annotation below).
                var sv = vals.slice().sort(function (a, b) {{ return a - b; }});
                cap = sv[Math.min(sv.length - 1, Math.floor(sv.length * 0.99))];
                if (!(cap > 0)) cap = ymax;
                lo = 0; hi = cap * 1.02;
            }}
            function fx(i) {{ return padL + (n > 1 ? pw * i / (n - 1) : pw / 2); }}
            function fy(v) {{ var t = logScale ? (Math.log10(Math.max(v, 1e-12)) - lo) / (hi - lo) : (v - lo) / (hi - lo); return padT + ph * (1 - t); }}
            ctx.font = '9px -apple-system,sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            if (logScale) {{
                for (var e = lo; e <= hi; e++) {{ var gy = fy(Math.pow(10, e)); ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(cssW - padR, gy); ctx.stroke(); ctx.fillStyle = '#999'; var lab = e >= 0 ? (Math.pow(10, e) + 's') : (Math.round(Math.pow(10, e + 3) * 1000) / 1000 + 'ms'); ctx.fillText(lab, padL - 5, gy); }}
            }} else {{
                for (var k = 0; k <= 5; k++) {{ var v = lo + (hi - lo) * k / 5, gy2 = fy(v); ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(padL, gy2); ctx.lineTo(cssW - padR, gy2); ctx.stroke(); ctx.fillStyle = '#999'; ctx.fillText((v * 1000).toFixed(v < 0.01 ? 2 : 0) + 'ms', padL - 5, gy2); }}
            }}
            var nOver = 0;
            // Shape encodes the engine (circle = BNGsim, square = AMICI) and
            // fill encodes the winner — so the chart is readable without relying on
            // color (color-blind safe; the colors are the Okabe-Ito palette too).
            function pt(x, v, color, filled, square) {{
                var y = fy(v), rad = 2.2;
                if (y < padT) {{ y = padT; nOver++; }}
                ctx.globalAlpha = filled ? 0.72 : 0.6;
                if (square) {{
                    if (filled) {{ ctx.fillStyle = color; ctx.fillRect(x - rad, y - rad, 2 * rad, 2 * rad); }}
                    else {{ ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.strokeRect(x - rad + 0.5, y - rad + 0.5, 2 * rad - 1, 2 * rad - 1); }}
                }} else {{
                    ctx.beginPath(); ctx.arc(x, y, rad, 0, 6.2832);
                    if (filled) {{ ctx.fillStyle = color; ctx.fill(); }}
                    else {{ ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke(); }}
                }}
                ctx.globalAlpha = 1;
            }}
            for (var i = 0; i < n; i++) {{
                var b = bn[i], r = rr[i];
                if (!(b > 0) || !(r > 0)) continue;
                var bWin = b < r;
                pt(fx(i), b, '#0072B2', bWin, false);
                pt(fx(i), r, '#D55E00', !bWin, true);
            }}
            if (!logScale && nOver > 0) {{
                ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillStyle = '#c0392b';
                ctx.font = '10px -apple-system,sans-serif';
                ctx.fillText('\\u25b2 ' + nOver + ' point(s) above ' + (cap * 1000).toFixed(cap < 0.01 ? 2 : 0) + 'ms cap (clamped)', cssW - padR, padT + 2);
            }}
            // x-axis ticks at the SPECIES-BIN edges (same bins as the wins tables),
            // labeled by species count. Tick marks are drawn at every edge; labels
            // are drawn only when they won't collide (the high-species edges crowd
            // the right since few models are that large), but the last is forced.
            var xt = PLOT_DATA.xt || [];
            ctx.font = '9px -apple-system,sans-serif'; ctx.textBaseline = 'top';
            var lastLabelX = -1e9;
            for (var t = 0; t < xt.length; t++) {{
                var idx = xt[t][0], lab = xt[t][1], x = fx(idx);
                ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke();
                ctx.strokeStyle = '#bbb'; ctx.beginPath(); ctx.moveTo(x, padT + ph); ctx.lineTo(x, padT + ph + 3); ctx.stroke();
                if (x - lastLabelX > 24 || t === xt.length - 1) {{
                    ctx.fillStyle = '#888'; ctx.textAlign = (t === xt.length - 1 ? 'right' : 'center');
                    ctx.fillText(lab, x, padT + ph + 4);
                    lastLabelX = x;
                }}
            }}
            ctx.textAlign = 'center'; ctx.fillStyle = '#666';
            ctx.fillText('species count  (models ordered by complexity, n=' + n + ')', padL + pw / 2, padT + ph + 16);
        }}
        function drawPlots() {{
            var sel = document.querySelector('input[name=plotscale]:checked');
            var logScale = !sel || sel.value === 'log';
            _scatter('cv_warm', PLOT_DATA.bw, PLOT_DATA.rw, logScale);
            _scatter('cv_cold', PLOT_DATA.bc, PLOT_DATA.rc, logScale);
            _scatter('cv_build', PLOT_DATA.bb, PLOT_DATA.rb, logScale);
        }}
        window.addEventListener('load', drawPlots);
        window.addEventListener('resize', drawPlots);
        </script>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BNGsim vs AMICI — ODE Parity Matrix ({datetime.now().strftime("%Y-%m-%d")})</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9em;
            line-height: 1.8;
        }}
        .metadata strong {{
            color: #333;
        }}
        .provenance {{
            margin-top: 8px;
            background: #f8f9fb;
            border: 1px solid #e3e6ea;
            border-radius: 4px;
            padding: 6px 10px;
        }}
        .provenance summary {{
            cursor: pointer;
            font-weight: 600;
            color: #495057;
        }}
        .provenance .prov-body {{
            margin-top: 6px;
            font-size: 0.95em;
            line-height: 1.55;
        }}
        .provenance .prov-body p {{ margin: 6px 0; }}
        .hwcaveat {{
            margin-top: 6px;
            padding: 7px 9px;
            background: #fff3cd;
            border-radius: 4px;
            color: #856404;
            font-size: 0.95em;
            line-height: 1.5;
        }}
        .path-info {{
            font-family: 'Courier New', monospace;
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            color: #495057;
            font-size: 0.85em;
        }}
        .summary {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .summary-item {{
            padding: 10px 15px;
            border-radius: 4px;
            font-weight: 500;
        }}
        .summary-passed {{ background: #d4edda; color: #155724; }}
        .summary-failed {{ background: #f8d7da; color: #721c24; }}
        .summary-triaged {{ background: #fff3cd; color: #856404; }}
        .summary-refused {{ background: #e2e3e5; color: #383d41; }}

        .plots {{
            margin-top: 18px;
            padding-top: 14px;
            border-top: 1px solid #e3e6ea;
        }}
        .plotctl {{
            font-size: 0.85em;
            color: #555;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .plotctl label {{ cursor: pointer; }}
        .plotctl .plotkey {{ color: #777; }}
        .plotctl .dot {{
            display: inline-block;
            width: 9px; height: 9px;
            border-radius: 50%;
            border: 1.5px solid #888;
            background: transparent;
            vertical-align: middle;
        }}
        .plotctl .dot.fill {{ background: #888; }}
        .plotctl .dot.sq {{ border-radius: 0; }}
        .plotwrap {{ margin-bottom: 10px; }}
        .plottitle {{ font-size: 0.85em; color: #444; font-weight: 600; margin-bottom: 2px; }}
        .costplot {{
            width: 100%;
            max-width: 940px;
            display: block;
            background: #fff;
            border: 1px solid #eee;
            border-radius: 4px;
        }}
        .coststats {{
            font-size: 0.9em;
            color: #444;
            background: #f8f9fb;
            border: 1px solid #e3e6ea;
            border-radius: 4px;
            padding: 8px 10px;
            margin-bottom: 10px;
        }}
        .winsgrid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 14px;
            margin-top: 6px;
            max-width: 940px;
        }}
        .winstab table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
            font-size: 0.82em;
            box-shadow: none;
            border-radius: 0;
        }}
        .winstab th {{
            position: static;
            background: #f1f3f5;
            color: #333;
            padding: 4px 8px;
            font-weight: 600;
            border-bottom: 1px solid #dee2e6;
        }}
        .winstab td {{
            padding: 3px 8px;
            border-bottom: 1px solid #f1f3f5;
            font-family: 'Courier New', monospace;
            overflow-wrap: normal;
        }}
        .winbar {{
            background: #D55E00;
            border-radius: 2px;
            height: 9px;
            width: 60px;
            overflow: hidden;
        }}
        .winbar span {{ display: block; height: 100%; background: #0072B2; }}
        .winstab tr.wintot td {{
            border-top: 2px solid #ced4da;
            font-weight: 700;
            background: #f8f9fb;
        }}
        .sortbar {{
            background: white;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 0.9em;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .sortbar select {{
            font-size: 0.95em;
            padding: 4px 6px;
            border-radius: 4px;
            border: 1px solid #ced4da;
        }}
        .sortbar .sortnote {{ color: #868e96; font-size: 0.88em; }}
        .sortbar .sortdir label {{ cursor: pointer; margin-right: 4px; }}
        .sortbar select:disabled, .sortbar input:disabled {{ opacity: 0.5; cursor: progress; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            /* Fixed layout: the browser sizes columns from the declared widths
               instead of measuring all 1300+ rows, so re-sorting reflows fast. */
            table-layout: fixed;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        th.col-model {{ width: 25%; }}
        th.col-engine {{ width: 37.5%; }}

        td {{
            padding: 14px;
            border-bottom: 4px solid #adb5bd;
            vertical-align: top;
            overflow-wrap: anywhere;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}

        .model-info {{
            font-family: 'Courier New', monospace;
            font-size: 1.05em;
            font-weight: 600;
            color: #2c3e50;
            padding-bottom: 8px;
            margin-bottom: 8px;
            border-bottom: 1px solid #dee2e6;
        }}
        .model-metadata {{
            font-size: 0.95em;
            color: #6c757d;
            line-height: 1.6;
            padding-bottom: 8px;
            border-bottom: 1px solid #dee2e6;
        }}
        .model-citation {{
            font-size: 0.95em;
            color: #495057;
            font-style: italic;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid #dee2e6;
        }}
        .model-features {{
            display: flex;
            gap: 6px;
            margin-top: 8px;
            flex-wrap: wrap;
        }}
        .feature-badge {{
            font-size: 0.82em;
            padding: 3px 7px;
            border-radius: 3px;
            background: #e9ecef;
            color: #495057;
            font-weight: 500;
        }}
        .feature-badge.notable {{
            background: #fff3cd;
            color: #856404;
        }}
        .row-comment {{
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #dee2e6;
            font-size: 0.82em;
            color: #6c757d;
            line-height: 1.5;
            font-family: 'Courier New', monospace;
            word-break: break-word;
        }}

        .engine-cell {{
            border-left: 4px solid;
            padding-left: 12px;
        }}

        .status-passed {{
            background: linear-gradient(to right, #28a745 4px, #e8f5e9 4px);
            border-left-color: #28a745;
        }}
        .status-failed {{
            background: linear-gradient(to right, #dc3545 4px, #ffebee 4px);
            border-left-color: #dc3545;
        }}
        .status-triaged {{
            background: linear-gradient(to right, #ffc107 4px, #fffbea 4px);
            border-left-color: #ffc107;
        }}
        .status-refused {{
            background: linear-gradient(to right, #6c757d 4px, #f8f9fa 4px);
            border-left-color: #6c757d;
        }}

        .config-section {{
            margin-bottom: 8px;
        }}
        .config-label {{
            font-size: 0.78em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #868e96;
            margin-bottom: 1px;
        }}
        .config-value {{
            font-size: 0.98em;
            color: #212529;
            font-weight: 600;
        }}
        .config-detail {{
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 1px;
        }}

        .timing {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
            padding-top: 10px;
            padding-bottom: 10px;
            border-top: 1px solid #dee2e6;
            border-bottom: 1px solid #dee2e6;
        }}
        .timing-item {{
            font-size: 0.90em;
        }}
        .timing-label {{
            color: #868e96;
            font-size: 0.78em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }}
        .timing-value {{
            font-weight: 600;
            color: #212529;
            font-family: 'Courier New', monospace;
            font-size: 1.0em;
        }}

        /* Three-tier timing block (per-process · per-model · per-integration) */
        .ttier {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid #e3e6ea;
        }}
        .ttier:first-of-type {{
            border-top: 1px solid #ced4da;
        }}
        .ttier-head {{
            font-size: 0.80em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            color: #495057;
            margin-bottom: 5px;
        }}
        .ttier-head span {{
            text-transform: none;
            letter-spacing: 0;
            font-weight: 400;
            color: #868e96;
            font-size: 0.92em;
        }}
        .ttier-grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 8px;
        }}
        .ttier-grid.two {{
            grid-template-columns: 1fr 1fr;
        }}
        .ttier-body {{
            display: flex;
        }}
        .tcell {{
            font-size: 0.90em;
        }}
        .tk {{
            color: #868e96;
            font-size: 0.78em;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 2px;
        }}
        .tv {{
            font-weight: 600;
            color: #212529;
            font-family: 'Courier New', monospace;
            font-size: 0.98em;
        }}
        /* Per-integration tier — the bottom line — gets a subtle highlight. */
        .ttier:last-of-type {{
            background: #f1f3f9;
            border-radius: 4px;
            padding: 8px 8px 9px;
            margin-left: -8px;
            margin-right: -8px;
        }}

        .verdict {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 3px;
            font-size: 0.88em;
            font-weight: 600;
            margin-top: 8px;
        }}
        .verdict-pass {{ background: #28a745; color: white; }}
        .verdict-diff {{ background: #dc3545; color: white; }}
        .verdict-exception {{ background: #ffc107; color: #000; }}
        .verdict-reffail {{ background: #ffc107; color: #000; }}
        .verdict-refused {{ background: #6c757d; color: white; }}

        .metric {{
            font-size: 0.88em;
            color: #495057;
            margin-top: 5px;
            font-family: 'Courier New', monospace;
        }}
        .metric.alert {{
            color: #dc3545;
            font-weight: 600;
        }}

        .footnote {{
            font-size: 0.88em;
            margin-top: 8px;
            padding: 7px 9px;
            border-radius: 3px;
            line-height: 1.5;
            color: #856404;
            background: #fff3cd;
        }}

        .error-msg {{
            color: #721c24;
            background: #f8d7da;
            padding: 7px 9px;
            border-radius: 3px;
            font-size: 0.85em;
            margin-top: 6px;
            font-family: 'Courier New', monospace;
            line-height: 1.4;
        }}

        .legend {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend h3 {{
            margin: 0 0 15px 0;
            color: #333;
        }}
        .legend-items {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-color {{
            width: 40px;
            height: 30px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            flex-shrink: 0;
        }}
        .legend-desc {{
            font-size: 0.9em;
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BNGsim vs AMICI — ODE Parity Matrix</h1>
        <div class="metadata">
            <div><strong>Suite:</strong> amici_parity (ODE regime)</div>
            <div><strong>Swept:</strong> {meta.get("generated", "N/A")} &nbsp;·&nbsp; <strong>Rendered:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
            <div><strong>Versions:</strong> BNGsim {versions.get("bngsim", "N/A")} · AMICI {versions.get("amici", "not installed")} · CVODE/SUNDIALS {versions.get("sundials", "n/a")}</div>
            {hardware_html}
            <div><strong>BNGsim method combo:</strong> {combo} &nbsp;·&nbsp; <strong>integration tolerance:</strong> {combo_tol} &nbsp;·&nbsp; <strong>overrides:</strong> {combo_overrides}</div>
            <div><strong>Jobs:</strong> {n_jobs} models &nbsp;·&nbsp; <strong>sweep wall time:</strong> {elapsed:.1f}s</div>
            <div><strong>Git revision:</strong> <span class="path-info">{meta.get("git_rev", "N/A")[:8]}</span></div>
            <div style="margin-top: 8px;">
                <div><strong>SBML source:</strong> <span class="path-info">$BIOMODELS_SBML_DIR (EBI BioModels REST API, fetched by benchmarks/suites/biomodels/fetch.py)</span>; materialized into <span class="path-info">models/</span></div>
                <div><strong>SED-ML source:</strong> <span class="path-info">$BIOMODELS_SEDML_DIR (github.com/sys-bio/temp-biomodels, final/)</span>; materialized into <span class="path-info">models/</span></div>
            </div>
            {provenance_html}
            {warmup_html}
        </div>
        <div class="summary">
            <div class="summary-item summary-passed">✓ {n_pass} PASSED ({pct(n_pass):.1f}%)</div>
            <div class="summary-item summary-failed">✗ {n_failed} FAILED ({pct(n_failed):.1f}%)</div>
            <div class="summary-item summary-triaged">⚠ {n_triaged} TRIAGED ({pct(n_triaged):.1f}%)</div>
            <div class="summary-item summary-refused">⊘ {n_refused} REFUSED ({pct(n_refused):.1f}%)</div>
        </div>
        {plots_html}
        {build_section_html}
    </div>

    <div class="sortbar">
        <strong>Sort rows by:</strong>
        <select id="sortsel" onchange="requestSort()">
            <option value="order">Model complexity — species → reactions → parameters → compartments (default)</option>
            <option value="id">Model ID</option>
            <option value="status">Status (failures first)</option>
            <option value="bnwarm">BNGsim warm per-integration cost (green rows only)</option>
            <option value="rrwarm">AMICI warm per-integration cost (green rows only)</option>
            <option value="bncold">BNGsim cold (first-solve) cost (green rows only)</option>
            <option value="rrcold">AMICI cold (first-solve) cost (green rows only)</option>
        </select>
        <span class="sortdir"><strong>Order:</strong>
            <label><input type="radio" name="sortdir" value="asc" checked onchange="requestSort()"> ascending</label>
            <label><input type="radio" name="sortdir" value="desc" onchange="requestSort()"> descending</label>
        </span>
        <span id="sortstatus" class="sortnote">ties fall back to the default activity order; missing values sort last</span>
    </div>

    <table id="matrix">
        <thead>
            <tr>
                <th class="col-model">Model</th>
                <th class="col-engine">BNGsim</th>
                <th class="col-engine">AMICI</th>
            </tr>
        </thead>
        <tbody>
"""

    # For TIMEOUT rows the shared subprocess was killed at the wall cap before it
    # could record BNGsim's result, so borrow BNGsim's per-model timing from the
    # rr_parity ODE sweep (same bn_ode adapter) — BNGsim runs these; AMICI is the
    # laggard. Kept LOCAL to row rendering (never merged into `results`) so it cannot
    # leak into the aggregate speed plots / geomean, which must stay this-run-only.
    rr_index, rr_src_version = load_rr_timing_index()

    # Table rows
    for row_index, result in enumerate(results_sorted):
        model_id = result["model_id"]
        outcome = result["outcome"]
        value = result.get("value")
        tol = result.get("tol", 1e-4)
        exception = result.get("exception", "")
        timing = result.get("timing") or {}
        bn_timing = timing.get("bngsim", {})
        rr_timing = timing.get("amici", {})

        # Row-level classification (both columns same color)
        row_css_class, row_category = classify_row(outcome)
        # EXCEPTION conflates three cases (see exception_kind); only a bngsim
        # raise is an actionable bngsim defect. A worker death (segfault) is
        # unattributable — recolor it gray (REFUSED) so it is not presented as a
        # triageable bngsim bug with AMICI as a clean oracle.
        exc_kind = exception_kind(result) if outcome == "EXCEPTION" else None
        if exc_kind == "dead":
            row_css_class = "status-refused"

        # Model column with SBML metadata — features precomputed above from the
        # manifest-resolved SBML path (item 6).
        sbml_features = features_by_id.get(model_id, {})

        n_species = sbml_features.get("n_species", "?")
        n_reactions = sbml_features.get("n_reactions", "?")
        n_parameters = sbml_features.get("n_parameters", "?")
        n_compartments = sbml_features.get("n_compartments", "?")
        citation = sbml_features.get("citation", "Unknown")
        features = sbml_features.get("features", [])

        # Feature badges
        feature_badges = ""
        for feat in features:
            # Mark certain features as notable
            notable = (
                "notable" if any(x in feat for x in ["event", "rule", "large", "variable"]) else ""
            )
            feature_badges += f'<span class="feature-badge {notable}">{feat}</span>'

        # Row-level comment — the reclassification reason (known-artifact DIFF→
        # PASS, invalid-reference, no-oracle adjudication), the per-cell
        # fail-fraction breakdown, the reference-refusal class, or a "worker
        # died (exit=…)" / "killed at Ns wall cap" note. rr_run records all of
        # this "loud, never silent", but the ODE matrix previously rendered it
        # only in the SSA table — so a reclassified green PASS gave no hint it
        # had been a DIFF. Surface it here (escaped — comments/exceptions can
        # carry < > & from species names and error text).
        note = humanize_comment(result)
        comment_html = f'<div class="row-comment">{_escape(note)}</div>' if note else ""

        # Explain a genuinely species-free SBML (state lives in parameters driven by
        # rate rules) so a reader unfamiliar with SBML does not read "0 species" as an
        # error. Only for a real parse that found 0 species — not a "?" failed parse.
        zero_species_note = (
            '<div class="row-comment">This SBML defines <strong>no species</strong>: '
            "its state variables are SBML <em>parameters</em> evolved by rate rules "
            "(a valid, if unusual, ODE model — e.g. a system written directly in "
            "concentrations/parameters rather than reacting species).</div>"
            if n_species == 0
            else ""
        )

        model_html = f"""                <td>
                    <div class="model-info">{model_id}</div>
                    <div class="model-citation">{citation}</div>
                    <div class="model-metadata">
                        <strong>{n_species}</strong> species · <strong>{n_reactions}</strong> reactions · <strong>{n_parameters}</strong> parameters · <strong>{n_compartments}</strong> compartment{"s" if n_compartments != 1 else ""}
                    </div>
                    <div class="model-features">{feature_badges}</div>
                    {zero_species_note}
                    {comment_html}
                </td>"""

        # BNGsim column
        bn_badge, bn_badge_color = get_verdict_badge(outcome, "bngsim", outcome != "EXCEPTION")
        # A worker death is unattributable, and a compare-step raise means bngsim
        # did NOT raise — neither should wear the bngsim "EXCEPTION" badge.
        if exc_kind == "dead":
            bn_badge, bn_badge_color = "DIED", "verdict-refused"
        elif exc_kind == "compare":
            bn_badge, bn_badge_color = "COMPARE-ERR", "verdict-exception"

        # Config rows via the shared display normalizers (one canonical label per
        # concept; RR architectural defaults shown even on a failed run; bngsim
        # run-dependent fields show "—" when no config was produced).
        bn_config = bn_timing.get("config", {})
        bn_solver = ode_solver_display(meta, engine="bngsim")
        bn_rhs = rhs_backend_display(bn_config.get("codegen"), engine="bngsim")
        bn_jacobian = jacobian_display(bn_config.get("jacobian"), engine="bngsim")
        bn_linear_solver = linear_solver_display(
            bn_config.get("linear_solver"), bn_config, engine="bngsim"
        )
        bn_cached = cache_display(bn_config, engine="bngsim")

        # Agreement metric, spelled out for a reader without the differ context:
        # the largest relative difference between the two engines' trajectories
        # (over the species both report), after the per-cell tolerance + fail-budget
        # protocol; it must stay below `tol` for the row to PASS.
        metric_html = ""
        if value is not None and outcome in ("PASS", "DIFF"):
            metric_class = "alert" if outcome == "DIFF" else ""
            verdict_word = "exceeds" if outcome == "DIFF" else "within"
            metric_html = (
                f'<div class="metric {metric_class}">'
                f"Largest BNGsim-vs-AMICI relative difference: <strong>{value:.2e}</strong> "
                f"({verdict_word} the {tol:g} agreement tolerance)</div>"
            )

        # Relative warm per-integration speed: BNGsim median ÷ AMICI median
        # over the n warm solves. <1 ⇒ BNGsim is faster. Shown only when both
        # engines recorded a warm median (a clean PASS/DIFF row).
        ratio_html = ""
        bn_wmed = bn_timing.get("integrate_warm_median_sec")
        rr_wmed = rr_timing.get("integrate_warm_median_sec")
        if bn_wmed and rr_wmed:
            ratio = bn_wmed / rr_wmed
            faster = ratio < 1.0
            phrase = f"{1 / ratio:.2f}× faster than" if faster else f"{ratio:.2f}× slower than"
            color = "#28a745" if faster else "#dc3545"
            ratio_html = (
                f'<div class="metric">Warm per-integration cost: BNGsim is '
                f'<strong style="color:{color}">{phrase}</strong> AMICI '
                f"({fmt_ms(bn_wmed)} vs {fmt_ms(rr_wmed)}, medians of n={bn_timing.get('integrate_n_warm', 0)}"
                f"/{rr_timing.get('integrate_n_warm', 0)} warm solves)</div>"
            )

        # Footnote for bngsim — EXCEPTION is three cases; only a bngsim raise is
        # an actionable bngsim bug. A crash is unattributable (don't blame
        # bngsim / don't credit AMICI); a compare raise means both ran.
        bn_footnote = ""
        if outcome == "EXCEPTION":
            if exc_kind == "bngsim":
                bn_footnote = '<div class="footnote"><strong>Actionable BNGsim bug:</strong> AMICI ran this model</div>'
            elif exc_kind == "compare":
                bn_footnote = '<div class="footnote">Both engines ran; the comparison step raised — a harness/data issue, not an engine fault</div>'
            elif exc_kind == "dead":
                bn_footnote = '<div class="footnote">Worker crashed with no per-engine status (often a AMICI segfault on pathological SBML); <strong>which engine failed is unattributable</strong></div>'
            else:  # other — a non-prefixed raise (legacy reports); don't blame bngsim
                bn_footnote = '<div class="footnote">An engine raised an exception</div>'
            if exception:
                bn_footnote += f'<div class="error-msg">{_escape(exception)}</div>'
        elif outcome == "BAD_TEST":
            bn_footnote = '<div class="footnote">Both engines rejected this model</div>'

        # Timing tiers — default to THIS run's captured numbers. A TIMEOUT row captured
        # nothing (the shared subprocess was killed at the wall cap during AMICI's C++
        # compile, before BNGsim's result was returned), so render the facts we DO know
        # instead of blanks: AMICI ran ≥ the cap (a hard lower bound, shown as "> cap s"),
        # and BNGsim runs these — its real per-model timing borrowed from the rr_parity
        # ODE sweep (same bn_ode adapter), clearly attributed.
        bn_tiers_html = render_timing_tiers(timing, engine="bngsim")
        rr_tiers_html = render_timing_tiers(timing, engine="amici")
        if outcome == "TIMEOUT":
            cap = timeout_cap_sec(result)
            rr_tiers_html = render_amici_timeout_tiers(cap)
            borrowed = rr_index.get(model_id) or {}
            bn_borrow = borrowed.get("bngsim")
            if bn_borrow:
                bn_tiers_html = render_timing_tiers(
                    {"bngsim": bn_borrow, "warmup": borrowed.get("warmup") or {}},
                    engine="bngsim",
                )
                bn_badge, bn_badge_color = "RAN", "verdict-pass"
                src = f" at bngsim {rr_src_version}" if rr_src_version else ""
                bn_footnote = (
                    '<div class="footnote"><strong>BNGsim runs this model.</strong> '
                    f"AMICI exceeded the {cap:.0f}s wall cap during its C++ compile, so the "
                    "shared job was killed before it could record BNGsim's result; the "
                    f"timing above is BNGsim's own run from the rr_parity ODE sweep (measured{src}, "
                    "identical bn_ode adapter).</div>"
                )
            else:
                rr_out = borrowed.get("rr_outcome")
                bn_badge, bn_badge_color = "NO DATA", "verdict-refused"
                extra = f" — rr_parity scored it {rr_out}" if rr_out and rr_out != "?" else ""
                bn_footnote = (
                    f'<div class="footnote">AMICI exceeded the {cap:.0f}s wall cap, so this '
                    "job recorded no BNGsim result, and rr_parity has no clean BNGsim "
                    f"timing for this model either{extra}.</div>"
                )

        bn_html = f"""                <td class="engine-cell {row_css_class}">
                    <div class="config-section">
                        <div class="config-label">ODE Solver</div>
                        <div class="config-value">{bn_solver}</div>
                    </div>
                    <div class="config-section">
                        <div class="config-label">RHS backend</div>
                        <div class="config-value">{bn_rhs}</div>
                    </div>
                    <div class="config-section">
                        <div class="config-label">Jacobian</div>
                        <div class="config-value">{bn_jacobian}</div>
                    </div>
                    <div class="config-section">
                        <div class="config-label">Linear solver</div>
                        <div class="config-value">{bn_linear_solver}</div>
                    </div>
                    <div class="config-section">
                        <div class="config-label">Codegen cache</div>
                        <div class="config-value">{bn_cached}</div>
                    </div>
                    {bn_tiers_html}
                    <div class="verdict {bn_badge_color}">{bn_badge}</div>
                    {metric_html}
                    {ratio_html}
                    {bn_footnote}
                </td>"""

        # AMICI column
        rr_badge, rr_badge_color = get_verdict_badge(
            outcome, "amici", outcome != "REFERENCE_FAILED"
        )
        # On an unattributable worker crash, AMICI may have been the one to
        # die — don't show it as a clean PASS. (A compare raise keeps PASS: RR
        # did produce a trajectory; only the comparison step failed.)
        if exc_kind == "dead":
            rr_badge, rr_badge_color = "DIED", "verdict-refused"

        # Config rows — AMICI's fixed architecture (analytical symbolic Jacobian,
        # KLU sparse solver, per-model C++ compile): the linear solver is read from
        # AMICI's solver accessor; jacobian/codegen are its known design, shown even
        # on a failed/refused run.
        rr_config = rr_timing.get("config", {})
        rr_solver = ode_solver_display(meta, engine="amici")
        rr_rhs = rhs_backend_display(rr_config.get("codegen"), engine="amici")
        rr_jacobian = jacobian_display(rr_config.get("jacobian"), engine="amici")
        rr_linear_solver = linear_solver_display(
            rr_config.get("linear_solver"), rr_config, engine="amici"
        )
        rr_cached = cache_display(rr_config, engine="amici")

        # AMICI footnote
        rr_footnote = ""
        if outcome == "REFERENCE_FAILED":
            refusal = result.get("reference_refusal", "unknown")
            refusal_display = refusal.replace("_", " ")
            settled = (
                " (settled)"
                if refusal
                in ["adjudicated_confirm", "adjudicated_uncoverable", "overstrict_missing_value"]
                else ""
            )
            rr_footnote = (
                f'<div class="footnote"><strong>Refusal:</strong> {refusal_display}{settled}</div>'
            )
            if exception and "amici" in exception.lower():
                rr_footnote += f'<div class="error-msg">{_escape(exception)}</div>'
        elif outcome == "EXCEPTION":
            if exc_kind == "bngsim":
                rr_footnote = '<div class="footnote">AMICI ran this model; BNGsim raised</div>'
            elif exc_kind == "compare":
                rr_footnote = '<div class="footnote">AMICI ran; the comparison step raised</div>'
            elif exc_kind == "dead":
                rr_footnote = '<div class="footnote">Worker crashed — which engine failed is unattributable</div>'
            # else: "other" (legacy non-prefixed raise) — leave RR footnote empty
        elif outcome == "BAD_TEST":
            rr_footnote = '<div class="footnote">Both engines rejected this model</div>'
        elif outcome == "TIMEOUT":
            cap = timeout_cap_sec(result)
            rr_footnote = (
                f'<div class="error-msg">AMICI exceeded the {cap:.0f}s wall cap — killed '
                "during C++ codegen/compile of this large model. The cap is a hard lower "
                f"bound (AMICI ran ≥ {cap:.0f}s, shown as &gt; {cap:.0f}s above), not a "
                "measured cost.</div>"
            )

        rr_html = f"""                <td class="engine-cell {row_css_class}">
                    <div class="config-section">
                        <div class="config-label">ODE Solver</div>
                        <div class="config-value">{rr_solver}</div>
                    </div>
                    <div class="config-section">
                        <div class="config-label">RHS backend</div>
                        <div class="config-value">{rr_rhs}</div>
                    </div>
                    <div class="config-section">
                        <div class="config-label">Jacobian</div>
                        <div class="config-value">{rr_jacobian}</div>
                    </div>
                    <div class="config-section">
                        <div class="config-label">Linear solver</div>
                        <div class="config-value">{rr_linear_solver}</div>
                    </div>
                    <div class="config-section">
                        <div class="config-label">Codegen cache</div>
                        <div class="config-value">{rr_cached}</div>
                    </div>
                    {rr_tiers_html}
                    <div class="verdict {rr_badge_color}">{rr_badge}</div>
                    {rr_footnote}
                </td>"""

        # data-* attributes drive the client-side sort. Numeric keys use a large
        # sentinel (_data_num) for missing/unparsed values so they always sort last;
        # the JS falls back to data-order (the complexity order) to break ties.
        statusrank = {
            "DIFF": 0,
            "TIMEOUT": 1,
            "EXCEPTION": 2,
            "REFERENCE_FAILED": 3,
            "BAD_TEST": 4,
            "PASS": 5,
        }.get(outcome, 6)
        data_attrs = (
            f' data-order="{row_index}" data-id="{_escape(model_id)}" data-status="{statusrank}"'
            f' data-species="{_data_num(sbml_features.get("n_species"))}"'
            f' data-reactions="{_data_num(sbml_features.get("n_reactions"))}"'
            f' data-params="{_data_num(sbml_features.get("n_parameters"))}"'
            f' data-comp="{_data_num(sbml_features.get("n_compartments"))}"'
            f' data-bnwarm="{_data_num(bn_timing.get("integrate_warm_median_sec"))}"'
            f' data-rrwarm="{_data_num(rr_timing.get("integrate_warm_median_sec"))}"'
            f' data-bncold="{_data_num(bn_timing.get("integrate_cold_sec"))}"'
            f' data-rrcold="{_data_num(rr_timing.get("integrate_cold_sec"))}"'
        )

        html += f"""            <tr{data_attrs}>
{model_html}
{bn_html}
{rr_html}
            </tr>
"""

    # Method-selection, DIFF-protocol, and provenance legend sections. Constants
    # are read live: the codegen threshold from the report's _meta["config"] (what
    # the sweep actually ran under) and the agreement tolerances from the differ
    # module (the parity protocol's source of truth) — not restated by hand.
    cg_threshold = meta.get("config", {}).get("codegen_threshold", 256)
    d_rel = getattr(_differ, "REL_TOL", 1e-4)
    d_abs_col = getattr(_differ, "ABS_TOL_COL", 1e-6)
    d_abs_file = getattr(_differ, "ABS_TOL_FILE", 1e-9)
    d_budget = getattr(_differ, "FAIL_FRAC_BUDGET", 5e-3)
    d_hard_rel = getattr(_differ, "HARD_REL_CEILING", 0.05)
    d_hard_abs = getattr(_differ, "HARD_ABS_CEILING_FILE", 1e-2)
    method_diff_provenance_html = f"""
        <h3 style="margin-top: 25px;">How BNGsim chooses its methods (this sweep ran <code>auto</code>)</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div style="margin-bottom: 6px; color: #868e96;">In <strong>auto</strong> mode BNGsim picks each method per model from the model's structure; the per-cell rows above report what it <em>actually</em> chose for that model. A user can override any of these.</div>
            <div><strong>RHS backend:</strong> <em>ExprTk</em> bytecode (a fast interpreter, no compilation) by default; auto switches to compiled <em>native C</em> (cc) at or above <strong>{cg_threshold} species</strong>. The in-process <em>MIR JIT</em> backend is <strong>opt-in only</strong> (<code>BNGSIM_CODEGEN_JIT=mir</code>) and auto never selects it.</div>
            <div><strong>Jacobian:</strong> the analytical (symbolic) Jacobian by default, derived lazily on the first solve (GH #145); BNGsim falls back to a finite-difference Jacobian when a symbolic one can't be built. <strong>AMICI also uses an analytical (symbolic) Jacobian</strong> — it differentiates the dxdot/dx chain at code-generation time (its defining strength), so both engines here are analytical-Jacobian.</div>
            <div><strong>Linear solver:</strong> the CVODE dense (SUNDIALS built-in LU) solver by default; a sparse <em>KLU</em> solver for sufficiently sparse Jacobians; the <em>LAPACK dense</em> (dgetrf) factorization is opt-in (<code>BNGSIM_LAPACK_DENSE=1</code>) and, once enabled, engages <em>automatically</em> — the solver runs the built-in dense LU for the first K (≈5) factorizations and switches to BLAS dgetrf only once a run refactors more than K times (i.e. proves factorization-bound), for systems ≥256×256. This is an adaptive factorization-<em>count</em> gate, not a model-size gate (GH #132): no static N/density threshold reliably separated wins from regressions, so the decision is made at runtime from the factorization count.</div>
            <div><strong>Forcing a method:</strong> any of the above can be pinned — e.g. <code>BNGSIM_CODEGEN_JIT=mir</code> (MIR JIT, which auto won't pick), <code>BNGSIM_LAPACK_DENSE=1</code> (LAPACK dense), <code>BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0</code> (force finite-difference Jacobian), or a forced dense solver. AMICI's configuration is fixed (CVODE · analytical symbolic Jacobian · KLU sparse solver · per-model C++ compile) and is not tuned here.</div>
        </div>
        <h3 style="margin-top: 25px;">How agreement (PASS vs DIFF) is decided</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div>The two engines are compared cell-by-cell over the shared species at every output time. A cell agrees when the absolute difference between the BNGsim and AMICI values satisfies a combined tolerance: <code>|a−b| ≤ {d_abs_file:g}·(file&nbsp;peak) + {d_abs_col:g}·(species&nbsp;peak) + {d_rel:g}·max(|a|,|b|)</code> — an absolute floor for tiny values plus a relative term for large ones.</div>
            <div>A few failing cells are tolerated: <strong>soft</strong> failures are forgiven up to a budget of <strong>{d_budget:.1%}</strong> of all cells. But a cell whose relative difference exceeds <strong>{d_hard_rel:.0%}</strong> in a magnitude-carrying column, or whose absolute difference exceeds <strong>{d_hard_abs:g}·(file peak)</strong>, is a <strong>hard</strong> failure and is never forgiven.</div>
            <div><strong>PASS</strong> = no hard failures and the soft-failure fraction is within budget. <strong>DIFF</strong> = otherwise. The number shown in each row (“Largest BNGsim-vs-AMICI relative difference”) is the worst remaining relative difference after the budget forgives soft failures — exactly <code>0</code> on a clean PASS.</div>
            <div style="color:#868e96;">(Protocol source of truth: <span class="path-info">parity_checks/_core/differ.py</span>; integration tolerances rtol/atol are forced identically on both engines and shown in the header.)</div>
        </div>
        <h3 style="margin-top: 25px;">Provenance — measured, not annotated</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div>Wherever possible, every value is read from what <em>actually executed</em>, not from hand-written notes that could rot: the per-cell <strong>RHS backend, Jacobian strategy, linear solver, and cache state</strong> come straight from each engine's own runtime accessors; <strong>timings</strong> are measured at each phase boundary; <strong>species/reaction/parameter/compartment counts</strong> are parsed from the actual SBML file; the <strong>SUNDIALS version</strong> is read from the build's <code>sundials_config.h</code>; and the agreement constants above are imported live from <code>differ.py</code>.</div>
            <div>The only annotated constants are properties no engine exposes via a query API: that CVODE uses the <strong>BDF</strong> formula with a <strong>Newton</strong> nonlinear solver (a compile-time choice in both engines), and that AMICI derives an <strong>analytical (symbolic) Jacobian</strong> and generates <strong>per-model C++</strong> — established from AMICI's documented design and confirmed in its build logs. AMICI's <strong>linear solver</strong> (read from its solver accessor) and <strong>every build-phase timing</strong> (parse / interpret / Jacobian / codegen / compile, captured from AMICI's own <code>log_execution_time</code> phase records) ARE measured at runtime, not annotated.</div>
        </div>"""

    # Footer with legend
    html += """        </tbody>
    </table>

    <div class="legend">
        <h3>Row Color Legend</h3>
        <div class="legend-items">
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(to right, #28a745 4px, #e8f5e9 4px);"></div>
                <div class="legend-desc"><strong>Green (PASSED):</strong> Both engines succeeded and agreed within tolerance</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(to right, #dc3545 4px, #ffebee 4px);"></div>
                <div class="legend-desc"><strong>Red (FAILED):</strong> DIFF (tolerance exceeded) or TIMEOUT (undetermined)</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(to right, #ffc107 4px, #fffbea 4px);"></div>
                <div class="legend-desc"><strong>Yellow (TRIAGED):</strong> REFERENCE_FAILED (AMICI refused) or EXCEPTION (BNGsim bug, requires triage)</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(to right, #6c757d 4px, #f8f9fa 4px);"></div>
                <div class="legend-desc"><strong>Gray (REFUSED):</strong> BAD_TEST (both engines rejected the model) or a worker crash (unattributable — either engine may have died)</div>
            </div>
        </div>
        <h3 style="margin-top: 25px;">Verdict Badges (within cells)</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div><strong>PASS:</strong> Engine succeeded</div>
            <div><strong>DIFF:</strong> Engine succeeded but result outside tolerance</div>
            <div><strong>EXCEPTION:</strong> BNGsim raised while AMICI ran (actionable BNGsim bug)</div>
            <div><strong>COMPARE-ERR:</strong> Both engines ran but the comparison step raised (harness/data issue)</div>
            <div><strong>DIED:</strong> Worker crashed (e.g. a segfault) with no per-engine status — which engine failed is unattributable</div>
            <div><strong>REF-FAIL:</strong> AMICI refused to load model</div>
            <div><strong>TIMEOUT:</strong> Wall-clock cap exceeded</div>
            <div><strong>REFUSED:</strong> Engine rejected model</div>
        </div>
        <h3 style="margin-top: 25px;">Timing: three cost tiers</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div style="margin-bottom: 6px; color: #868e96;">Each engine cell reports timing as three labeled lines, one per cost tier. A fitting or MCMC run pays tier ① once for the whole run, tier ② once per model, and tier ③ on <em>every</em> objective-function evaluation — so tier ③ is the bottom line.</div>
            <div><strong>① Per-process startup</strong> (once per run): one-time engine initialization — BNGsim's SymPy import, AMICI's library import (its ~48&nbsp;MB statically-linked binary; CVODE/SUNDIALS init is ~0.5&nbsp;ms at first model construction, not at import). Nearly constant across models and amortized to ~0 in fitting/MCMC, so it is never added into the per-model or per-integration numbers.</div>
            <div><strong>② Per-model build</strong> (once per model): the cost to turn one SBML file into a ready-to-solve model.
                <span style="display:block; margin-left:14px;">— <strong>I/O:</strong> reading the SBML file from disk.</span>
                <span style="display:block; margin-left:14px;">— <strong>Parse (libSBML):</strong> the libSBML XML parse — the shared C++ parser both engines use, so this should be comparable between them.</span>
                <span style="display:block; margin-left:14px;">— <strong>Interpret:</strong> converting the parsed document into the engine's internal model (BNGsim: Python loader; AMICI: C++) — different implementations, so this differs.</span>
                <span style="display:block; margin-left:14px;">— <strong>Jacobian:</strong> symbolic derivation of the analytical Jacobian. BNGsim uses SymPy (GH #76), deferred to the first solve as of GH #145, so on a freshly loaded model it can be ~0 here and appear instead inside the cold integration. <strong>AMICI also derives an analytical Jacobian</strong> — its dxdot/dx symbolic chain, captured here from AMICI's build logs — so both engines show a value (it grows with model size; milliseconds for small models).</span>
                <span style="display:block; margin-left:14px;">— <strong>RHS build:</strong> turning the symbolic RHS/Jacobian into the callable evaluator named in the "RHS backend" row — <em>generate the code AND compile it</em>, as one phase (compilation is part of building the evaluator, not a separate step). BNGsim's ExprTk bytecode is ~0.2&nbsp;ms (interpreted, nothing to compile); its native-C (cc) backend generates C and compiles it (~seconds), reported as one inseparable number. <strong>AMICI</strong> generates per-model C++ and compiles it (cmake-configure + ninja + SWIG wrap + static link against SUNDIALS/KLU) — a ~20–30&nbsp;s floor <strong>dominated by the compile</strong>, and AMICI's defining pre-simulation cost. The Jacobian <em>derivation</em> (above) is separate; this row is the evaluator-code build.</span>
                <span style="display:block; margin-left:14px;">— <strong>Build subtotal:</strong> the sum of the tier-② phases — the one-time per-model cost.</span></div>
            <div><strong>③ Per-integration</strong> (the bottom line): CVODE numerical integration only.
                <span style="display:block; margin-left:14px;">— <strong>Warm (marginal, reused):</strong> the minimum over repeated solves of an already-set-up Simulator (initial conditions reset each time). This is the marginal cost a Simulator that is reused across evaluations would pay — the best case for fitting/MCMC.</span>
                <span style="display:block; margin-left:14px;">— <strong>Total, cold (first solve):</strong> the very first solve, which includes one-time CVODE setup (workspace + linear-solver allocation, first Jacobian, RHS page-fault, CPU ramp) <em>plus</em> the solve — i.e. the full cost of one integration from scratch, as paid by any caller that builds a fresh Simulator per evaluation rather than reusing one.</span></div>
            <div style="margin-top: 10px; padding: 8px; background: #e7f1ff; border-radius: 4px;"><strong>The cold/warm split is NOT the codegen cache.</strong> The RHS backend is compiled/loaded exactly once (the "Codegen cache" row says whether a compiled artifact was reused), <em>before</em> any integration. The cold→warm difference is CVODE's own first-call setup, paid on the first solve and amortized on later solves of the same Simulator — independent of codegen caching.</div>
            <div style="margin-top: 8px; padding: 8px; background: #f1f3f5; border-radius: 4px;"><strong>Linear solvers</strong> are all LU factorizations; they differ only in implementation. <em>CVODE dense (SUNDIALS built-in LU)</em> is SUNDIALS' own dense factorization (BNGsim uses it by default). <em>LAPACK dense (dgetrf LU)</em> is the same dense LU done by LAPACK's <code>dgetrf</code> (BNGsim, opt-in; once enabled it engages automatically once a run refactors more than K≈5 times, for N≥256 — an adaptive factorization-count gate, GH #132, not a model-size gate). <em>KLU (sparse LU)</em> is a sparse LU for sparse Jacobians — <strong>AMICI's default</strong>, and BNGsim's choice for sufficiently sparse Jacobians.</div>
            <div style="margin-top: 8px; padding: 8px; background: #fff3cd; border-radius: 4px;"><strong>Note:</strong> Each BNGsim phase is timed at its own boundary inside the loader (no subtraction). AMICI's Parse · Interpret · Jacobian · RHS build (codegen + compile) are captured from AMICI's own <code>log_execution_time</code> phase records (it self-times every build step) — no AMICI source patch is needed, since the only C++ step, the compile, is itself a single timed phase. On a cache hit the build is skipped and these read "—"/0 (only the extension load is paid).</div>
            <div style="margin-top: 8px; padding: 8px; background: #e7f5ee; border-radius: 4px;"><strong>Fair-comparison build config (AMICI <code>observation_model=[]</code>).</strong> AMICI is built with no observable / measurement model, because this suite compares <em>state trajectories</em> (<code>rdata.x</code>), which AMICI computes identically with or without observables. AMICI's default would also generate its parameter-estimation likelihood layer (<code>y</code>, <code>sigmay</code>, <code>Jy</code>, <code>dJydy</code>, …) — the negative-log-likelihood and its derivatives — which forward simulation never uses and for which BNGsim builds nothing comparable in its simulate path. Skipping it (a documented AMICI option) makes the build a true pure-ODE cost and the comparison apples-to-apples; on small models it roughly halves the compile. The analytical Jacobian, RHS, and state dynamics are unaffected.</div>
        </div>"""
    html += method_diff_provenance_html
    html += """
    </div>
    <script>
    var COST_KEYS = { bnwarm: 1, rrwarm: 1, bncold: 1, rrcold: 1 };
    var _sorting = false;

    function _setControlsDisabled(d) {
        document.getElementById('sortsel').disabled = d;
        var radios = document.querySelectorAll('input[name=sortdir]');
        for (var i = 0; i < radios.length; i++) radios[i].disabled = d;
    }

    // Re-sorting a ~9 MB table reflows the whole DOM, which can take a moment, so
    // we lock the controls and show a working indicator while it runs. The actual
    // work is deferred with setTimeout so the browser paints the indicator first.
    function requestSort() {
        if (_sorting) return;
        _sorting = true;
        _setControlsDisabled(true);
        var status = document.getElementById('sortstatus');
        status.textContent = '⏳ sorting…';
        setTimeout(function () {
            try { doSort(); } finally {
                _sorting = false;
                _setControlsDisabled(false);
            }
        }, 30);
    }

    function doSort() {
        var key = document.getElementById('sortsel').value;
        var dirEl = document.querySelector('input[name=sortdir]:checked');
        var desc = dirEl && dirEl.value === 'desc';
        var costSort = !!COST_KEYS[key];
        var tb = document.querySelector('#matrix tbody');
        var rows = Array.prototype.slice.call(tb.querySelectorAll(':scope > tr'));

        // Cost sorts are only meaningful for green (PASS, data-status=5) rows —
        // everything else has no/sentinel cost. Hide the rest (also speeds reflow).
        var shown = 0, hidden = 0;
        rows.forEach(function (r) {
            var show = !costSort || r.getAttribute('data-status') === '5';
            r.style.display = show ? '' : 'none';
            if (show) shown++; else hidden++;
        });

        var numeric = (key !== 'id');
        rows.sort(function (a, b) {
            var c;
            if (numeric) {
                c = parseFloat(a.getAttribute('data-' + key)) - parseFloat(b.getAttribute('data-' + key));
            } else {
                var av = a.getAttribute('data-id'), bv = b.getAttribute('data-id');
                c = (av < bv) ? -1 : (av > bv ? 1 : 0);
            }
            if (!c) { c = parseInt(a.getAttribute('data-order')) - parseInt(b.getAttribute('data-order')); }
            return desc ? -c : c;
        });
        var frag = document.createDocumentFragment();
        rows.forEach(function (r) { frag.appendChild(r); });
        tb.appendChild(frag);

        document.getElementById('sortstatus').textContent = costSort
            ? ('showing ' + shown + ' green (PASS) rows; ' + hidden + ' non-PASS rows hidden for the cost sort')
            : 'ties fall back to the default activity order; missing values sort last';
    }
    </script>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"Generated: {output_path}")
    print(
        f"  {n_jobs} models | {tally.get('PASS', 0)} PASS | {tally.get('DIFF', 0)} DIFF | "
        f"{tally.get('EXCEPTION', 0)} EXCEPTION"
    )
    print(
        f"  {tally.get('TIMEOUT', 0)} TIMEOUT | {tally.get('REFERENCE_FAILED', 0)} REF_FAIL | "
        f"{tally.get('BAD_TEST', 0)} BAD_TEST"
    )


def _ssa_row_class(outcome: str, subclass: str | None) -> tuple[str, str]:
    """(css_class, badge) for an SSA result row. Honest SSA policy (GH #108):
    a real regression (DIFF/bngsim_suspect or ode_level, EXCEPTION) is red; a
    DIFF attributed away from bngsim (diff_not_bngsim / rr_known) is yellow; a
    too_slow TIMEOUT is a known exact-SSA coverage gap (gray, not a failure);
    PASS is green."""
    if outcome == "PASS":
        return "status-passed", "PASS"
    if outcome == "DIFF":
        green = subclass in (
            "diff_not_bngsim",
            "rr_known",
            "partial_horizon",
            "attribution_incomplete",
        )
        label = {
            "partial_horizon": "PARTIAL",
            "attribution_incomplete": "DIFF (fault?)",
        }.get(subclass, f"DIFF·{subclass}")
        return ("status-triaged", label) if green else ("status-failed", "DIFF")
    if outcome == "EXCEPTION":
        return "status-failed", "EXCEPTION"
    if outcome == "REFERENCE_FAILED":
        return "status-triaged", "REF-FAIL"
    if outcome in ("TIMEOUT", "SKIP", "UNSUPPORTED", "BAD_TEST"):
        return "status-refused", outcome
    return "status-refused", outcome


def _ssa_geomean(ratios: list[float]) -> float | None:
    """Geometric mean of a list of positive ratios, or None if none are valid.
    Geomean (not arithmetic) is the right average for speedup ratios — it is
    symmetric under inversion, so "2× faster" and "2× slower" cancel."""
    import math

    vals = [r for r in ratios if isinstance(r, (int, float)) and r > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _ssa_engine_samples(results: list, engine: str, key: str) -> list[float]:
    """All non-null ``timing[engine][key]`` values across the report's rows."""
    out = []
    for r in results:
        eng = (r.get("timing") or {}).get(engine) or {}
        v = eng.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _ssa_warmup_summary_html(results: list) -> str:
    """Per-process startup line for the SSA matrix, aggregated over all jobs.

    The SSA taxonomy differs from the ODE one: bngsim's SSA warmup is the
    ``_bngsim_core`` extension load (it derives NO Jacobian and imports NO SymPy),
    and AMICI's is its LLVM/JIT engine init (gillespie still JITs). Each job
    is its own subprocess → one warmup sample per row. Returns "" when no row
    carries warmup (a pre-instrumentation report)."""
    import statistics as _st

    def _w(key):
        out = []
        for r in results:
            w = (r.get("timing") or {}).get("warmup") or {}
            v = w.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    bn, rr = _w("bngsim_sec"), _w("amici_sec")
    if not bn and not rr:
        return ""

    def _stat(xs):
        if not xs:
            return "—"
        sd = _st.stdev(xs) if len(xs) > 1 else 0.0
        return (
            f"{_st.mean(xs) * 1000:.0f} ± {sd * 1000:.0f} ms "
            f"(median {_st.median(xs) * 1000:.0f} ms, n={len(xs)})"
        )

    sources = {
        (r.get("timing") or {}).get("warmup", {}).get("amici_source")
        for r in results
        if (r.get("timing") or {}).get("warmup")
    }
    rr_src = next((s for s in sources if s), "")
    rr_tag = f" (from AMICI's {_escape(rr_src)})" if rr_src else ""
    return (
        '<div class="tnote"><strong>① Per-process startup.</strong> Once per run, before any '
        "model is loaded, each engine imports its libraries — counted here, never in a model's "
        "per-model or per-replicate time below (a fitting/MCMC run amortizes it to ~0). Averaged "
        f"over every job: bngsim's <code>_bngsim_core</code> import took {_stat(bn)}, and "
        f"AMICI's library import took {_stat(rr)}{rr_tag}.</div>"
    )


def _resolve_ssa_sbml(model_id: str, config: dict) -> Path | None:
    """Best-effort path to an SSA model's SBML, so the renderer can parse the same
    species/reaction/parameter/compartment counts + feature tags the ODE matrix
    shows. SSA models are biomodels (``<biomodels_sbml_dir>/<id>.xml``, the dir is
    recorded in the report config) or BNGL-derived roundtrip SBMLs
    (``$SBML_ROUNDTRIP_DIR`` / the benchmarks default). Returns the first that
    exists, else None (the cell falls back to "?")."""
    import os

    cands: list[Path] = []
    for d in (config.get("biomodels_sbml_dir"), os.environ.get("BIOMODELS_SBML_DIR")):
        if d:
            cands.append(Path(d) / f"{model_id}.xml")
    rdir = os.environ.get("SBML_ROUNDTRIP_DIR")
    if rdir:
        cands.append(Path(rdir) / f"{model_id}.xml")
    cands.append(HERE.parents[1] / "benchmarks" / "suites" / "sbml_roundtrip" / f"{model_id}.xml")
    for c in cands:
        try:
            if c.exists():
                return c
        except OSError:
            continue
    return None


def render_ssa_timing_tiers(timing: dict, *, engine: str) -> str:
    """The SSA three-tier timing block for one engine — the SSA analog of the ODE
    ``render_timing_tiers``, to the SSA taxonomy:

      ① Per-process startup — the once-per-run library import (BNGsim's
         ``_bngsim_core`` extension load — the SSA path imports no SymPy and derives
         no Jacobian; AMICI's library import).
      ② Per-model load — BNGsim is parse+interpret only (no Jacobian, no codegen);
         AMICI adds codegen+JIT (which dominates).
      ③ Per-replicate ensemble — the bottom line: N independent reseeded Gillespie
         trajectories (model loaded once, cloned/reset per replicate), reported as
         the replicate median + rep-1 COLD vs warm-rest + ensemble total. BNGsim
         also shows its per-replicate clone/construct setup, which AMICI (one
         reused compiled model) does not pay.
    """
    warmup = timing.get("warmup") or {}
    eng = timing.get(engine) or {}
    if engine == "bngsim":
        proc_label, proc_val = "_bngsim_core import", warmup.get("bngsim_sec")
        load_cells = "".join(
            [
                _tcell("Parse (libSBML)", fmt_ms(eng.get("parse_sec"))),
                _tcell("Interpret", fmt_ms(eng.get("interpret_sec"))),
                _tcell("Build subtotal", fmt_ms(eng.get("load_sec"))),
            ]
        )
    else:
        proc_label, proc_val = "Library import", warmup.get("amici_sec")
        load_cells = "".join(
            [
                _tcell("Parse (libSBML)", fmt_ms(eng.get("parse_sec"))),
                _tcell("Interpret", fmt_ms(eng.get("interpret_sec"))),
                _tcell("Codegen + JIT", fmt_ms(eng.get("codegen_sec"))),
                _tcell("Build subtotal", fmt_ms(eng.get("load_sec"))),
            ]
        )

    if not eng:
        rep_cells = _tcell("Per-replicate", "—")
    else:
        n = eng.get("n_rep", 0) or 0
        cold, warm = eng.get("rep_cold_sec"), eng.get("rep_warm_median_sec")
        cw = f"{fmt_ms(cold)} → {fmt_ms(warm)}" if warm is not None else fmt_ms(cold)
        cells = [
            _tcell("Per-replicate (median)", fmt_ms(eng.get("rep_median_sec"))),
            _tcell("Cold → warm", cw),
            _tcell(f"Ensemble ({n} reps)", fmt_ms(eng.get("ensemble_sec"))),
        ]
        if engine == "bngsim":
            cells.append(_tcell("Setup / rep", fmt_ms(eng.get("setup_per_rep_sec"))))
        rep_cells = "".join(cells)

    return f"""<div class="ttier">
                        <div class="ttier-head">① Per-process startup <span>once per run · shared by every model · excluded from the per-model totals</span></div>
                        <div class="ttier-body">{_tcell(proc_label, fmt_ms(proc_val))}</div>
                    </div>
                    <div class="ttier">
                        <div class="ttier-head">② Per-model load <span>once per model — SBML → ready-to-simulate</span></div>
                        <div class="ttier-grid ssa">{load_cells}</div>
                    </div>
                    <div class="ttier">
                        <div class="ttier-head">③ Per-replicate ensemble <span>the bottom line — one Gillespie trajectory, ×N reseeded replicates</span></div>
                        <div class="ttier-grid ssa">{rep_cells}</div>
                    </div>"""


def _ssa_verdict_badges(outcome: str, subclass: str | None) -> tuple[str, str, str, str]:
    """Per-engine verdict badges ``(bn_badge, bn_color, rr_badge, rr_color)`` for an
    SSA row, encoding the screen's attribution: a DIFF the oracle pinned on
    AMICI (``diff_not_bngsim`` / ``rr_known``) shows BNGsim PASS · AMICI
    DIFF (BNGsim tracked its own ODE; AMICI's gillespie diverged), whereas a
    ``bngsim_suspect`` / ``ode_level`` DIFF flags both. A ``amici:``-prefixed
    refusal is REFERENCE_FAILED (BNGsim ran, AMICI couldn't)."""
    if outcome == "PASS":
        return "PASS", "verdict-pass", "PASS", "verdict-pass"
    if outcome == "DIFF":
        if subclass == "partial_horizon":
            # Both engines were compared, but only over an achievable sub-window —
            # a provisional verdict, neither a clean pass nor a confirmed fault.
            return "PARTIAL", "verdict-reffail", "PARTIAL", "verdict-reffail"
        if subclass == "attribution_incomplete":
            # SSA ensembles diverged but the fault-assigning ODE oracle didn't finish.
            return "DIFF (fault?)", "verdict-reffail", "DIFF (fault?)", "verdict-reffail"
        if subclass in ("diff_not_bngsim", "rr_known"):
            return "PASS", "verdict-pass", "DIFF", "verdict-diff"
        return "DIFF", "verdict-diff", "DIFF", "verdict-diff"
    if outcome == "EXCEPTION":
        return "EXCEPTION", "verdict-exception", "—", "verdict-refused"
    if outcome == "REFERENCE_FAILED":
        return "PASS", "verdict-pass", "REF-FAIL", "verdict-reffail"
    if outcome == "TIMEOUT":
        return "TOO SLOW", "verdict-refused", "TOO SLOW", "verdict-refused"
    if outcome == "BAD_TEST":
        return "REFUSED", "verdict-refused", "REFUSED", "verdict-refused"
    if outcome == "SKIP":
        return "SKIP", "verdict-refused", "SKIP", "verdict-refused"
    return outcome, "verdict-refused", outcome, "verdict-refused"


# Shared visual language with the ODE matrix — a plain (non-f) string so the CSS
# braces need no escaping. Kept structurally identical to generate_html's inline
# styles (same header/summary/plots/sortbar/table/engine-cell/legend classes) so
# the two matrices look like siblings, with one SSA addition (.ttier-grid.ssa).
_SSA_MATRIX_CSS = """
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 20px; background: #f5f5f5; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header h1 { margin: 0 0 10px 0; color: #333; }
        .metadata { color: #666; font-size: 0.9em; line-height: 1.8; }
        .metadata strong { color: #333; }
        .hwcaveat { margin-top: 6px; padding: 7px 9px; background: #fff3cd; border-radius: 4px;
                    color: #856404; font-size: 0.95em; line-height: 1.5; }
        .path-info { font-family: 'Courier New', monospace; background: #f8f9fa; padding: 2px 6px;
                     border-radius: 3px; color: #495057; font-size: 0.85em; }
        .summary { display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap; }
        .summary-item { padding: 10px 15px; border-radius: 4px; font-weight: 500; }
        .summary-passed { background: #d4edda; color: #155724; }
        .summary-failed { background: #f8d7da; color: #721c24; }
        .summary-triaged { background: #fff3cd; color: #856404; }
        .summary-refused { background: #e2e3e5; color: #383d41; }
        .plots { margin-top: 18px; padding-top: 14px; border-top: 1px solid #e3e6ea; }
        .plotctl { font-size: 0.85em; color: #555; margin-bottom: 8px; display: flex;
                   align-items: center; gap: 8px; flex-wrap: wrap; }
        .plotctl label { cursor: pointer; }
        .plotctl .plotkey { color: #777; }
        .plotctl .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
                        border: 1.5px solid #888; background: transparent; vertical-align: middle; }
        .plotctl .dot.fill { background: #888; }
        .plotctl .dot.sq { border-radius: 0; }
        .plotwrap { margin-bottom: 10px; }
        .plottitle { font-size: 0.85em; color: #444; font-weight: 600; margin-bottom: 2px; }
        .costplot { width: 100%; max-width: 940px; display: block; background: #fff;
                    border: 1px solid #eee; border-radius: 4px; }
        .coststats { font-size: 0.9em; color: #444; background: #f8f9fb; border: 1px solid #e3e6ea;
                     border-radius: 4px; padding: 8px 10px; margin-bottom: 10px; }
        .winsgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
                    gap: 14px; margin-top: 6px; max-width: 940px; }
        .winstab table { width: 100%; border-collapse: collapse; table-layout: auto; font-size: 0.82em;
                         box-shadow: none; border-radius: 0; }
        .winstab th { position: static; background: #f1f3f5; color: #333; padding: 4px 8px;
                      font-weight: 600; border-bottom: 1px solid #dee2e6; }
        .winstab td { padding: 3px 8px; border-bottom: 1px solid #f1f3f5;
                      font-family: 'Courier New', monospace; overflow-wrap: normal; }
        .winbar { background: #D55E00; border-radius: 2px; height: 9px; width: 60px; overflow: hidden; }
        .winbar span { display: block; height: 100%; background: #0072B2; }
        .winstab tr.wintot td { border-top: 2px solid #ced4da; font-weight: 700; background: #f8f9fb; }
        .sortbar { background: white; padding: 12px 16px; border-radius: 8px; margin-bottom: 14px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-size: 0.9em; color: #333; display: flex;
                   align-items: center; gap: 10px; flex-wrap: wrap; }
        .sortbar select { font-size: 0.95em; padding: 4px 6px; border-radius: 4px; border: 1px solid #ced4da; }
        .sortbar .sortnote { color: #868e96; font-size: 0.88em; }
        .sortbar .sortdir label { cursor: pointer; margin-right: 4px; }
        .sortbar select:disabled, .sortbar input:disabled { opacity: 0.5; cursor: progress; }
        table { width: 100%; border-collapse: collapse; table-layout: fixed; background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }
        th { background: #2c3e50; color: white; padding: 12px; text-align: left; font-weight: 600;
             position: sticky; top: 0; z-index: 10; }
        th.col-model { width: 25%; }
        th.col-engine { width: 37.5%; }
        td { padding: 14px; border-bottom: 4px solid #adb5bd; vertical-align: top; overflow-wrap: anywhere; }
        tr:hover { background: #f8f9fa; }
        .model-info { font-family: 'Courier New', monospace; font-size: 1.05em; font-weight: 600;
                      color: #2c3e50; padding-bottom: 8px; margin-bottom: 8px; border-bottom: 1px solid #dee2e6; }
        .model-metadata { font-size: 0.95em; color: #6c757d; line-height: 1.6; padding-bottom: 8px;
                          border-bottom: 1px solid #dee2e6; }
        .model-activity { font-size: 0.9em; color: #0b5394; background: #eef5fb; border-radius: 4px;
                          padding: 5px 8px; margin-bottom: 8px; }
        .model-activity span { color: #6c757d; font-size: 0.9em; }
        .model-citation { font-size: 0.95em; color: #495057; font-style: italic; margin-bottom: 8px;
                          padding-bottom: 8px; border-bottom: 1px solid #dee2e6; }
        .model-features { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
        .feature-badge { font-size: 0.82em; padding: 3px 7px; border-radius: 3px; background: #e9ecef;
                         color: #495057; font-weight: 500; }
        .feature-badge.notable { background: #fff3cd; color: #856404; }
        .row-comment { margin-top: 8px; padding-top: 8px; border-top: 1px solid #dee2e6; font-size: 0.82em;
                       color: #6c757d; line-height: 1.5; font-family: 'Courier New', monospace; word-break: break-word; }
        .engine-cell { border-left: 4px solid; padding-left: 12px; }
        .status-passed { background: linear-gradient(to right, #28a745 4px, #e8f5e9 4px); border-left-color: #28a745; }
        .status-failed { background: linear-gradient(to right, #dc3545 4px, #ffebee 4px); border-left-color: #dc3545; }
        .status-triaged { background: linear-gradient(to right, #ffc107 4px, #fffbea 4px); border-left-color: #ffc107; }
        .status-refused { background: linear-gradient(to right, #6c757d 4px, #f8f9fa 4px); border-left-color: #6c757d; }
        .config-section { margin-bottom: 8px; }
        .config-label { font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.5px; color: #868e96; margin-bottom: 1px; }
        .config-value { font-size: 0.98em; color: #212529; font-weight: 600; }
        .ttier { margin-top: 10px; padding-top: 8px; border-top: 1px solid #e3e6ea; }
        .ttier:first-of-type { border-top: 1px solid #ced4da; }
        .ttier-head { font-size: 0.80em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px;
                      color: #495057; margin-bottom: 5px; }
        .ttier-head span { text-transform: none; letter-spacing: 0; font-weight: 400; color: #868e96; font-size: 0.92em; }
        .ttier-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
        .ttier-grid.ssa { grid-template-columns: repeat(4, 1fr); }
        .ttier-body { display: flex; }
        .tcell { font-size: 0.90em; }
        .tk { color: #868e96; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 2px; }
        .tv { font-weight: 600; color: #212529; font-family: 'Courier New', monospace; font-size: 0.98em; }
        .ttier:last-of-type { background: #f1f3f9; border-radius: 4px; padding: 8px 8px 9px; margin-left: -8px; margin-right: -8px; }
        .verdict { display: inline-block; padding: 5px 12px; border-radius: 3px; font-size: 0.88em; font-weight: 600; margin-top: 8px; }
        .verdict-pass { background: #28a745; color: white; }
        .verdict-diff { background: #dc3545; color: white; }
        .verdict-exception { background: #ffc107; color: #000; }
        .verdict-reffail { background: #ffc107; color: #000; }
        .verdict-refused { background: #6c757d; color: white; }
        .metric { font-size: 0.88em; color: #495057; margin-top: 5px; font-family: 'Courier New', monospace; }
        .metric.alert { color: #dc3545; font-weight: 600; }
        .footnote { font-size: 0.88em; margin-top: 8px; padding: 7px 9px; border-radius: 3px; line-height: 1.5;
                    color: #856404; background: #fff3cd; }
        .error-msg { color: #721c24; background: #f8d7da; padding: 7px 9px; border-radius: 3px; font-size: 0.85em;
                     margin-top: 6px; font-family: 'Courier New', monospace; line-height: 1.4; }
        .legend { background: white; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .legend h3 { margin: 0 0 15px 0; color: #333; }
        .legend-items { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
        .legend-item { display: flex; align-items: center; gap: 10px; }
        .legend-color { width: 40px; height: 30px; border-radius: 4px; border: 1px solid #dee2e6; flex-shrink: 0; }
        .legend-desc { font-size: 0.9em; color: #495057; }
"""

# Row-sort script — same UX as the ODE matrix; the cost keys are the SSA ones
# (per-replicate / load), and rows are not hidden for a cost sort (SSA timing is
# present on any row both engines ran, not just PASS).
_SSA_SORT_JS = """    <script>
    var _sorting = false;
    function _setControlsDisabled(d) {
        document.getElementById('sortsel').disabled = d;
        var radios = document.querySelectorAll('input[name=sortdir]');
        for (var i = 0; i < radios.length; i++) radios[i].disabled = d;
    }
    function requestSort() {
        if (_sorting) return;
        _sorting = true; _setControlsDisabled(true);
        document.getElementById('sortstatus').textContent = '⏳ sorting…';
        setTimeout(function () { try { doSort(); } finally { _sorting = false; _setControlsDisabled(false); } }, 30);
    }
    function doSort() {
        var key = document.getElementById('sortsel').value;
        var dirEl = document.querySelector('input[name=sortdir]:checked');
        var desc = dirEl && dirEl.value === 'desc';
        var tb = document.querySelector('#matrix tbody');
        var rows = Array.prototype.slice.call(tb.querySelectorAll(':scope > tr'));
        var numeric = (key !== 'id');
        rows.sort(function (a, b) {
            var c;
            if (numeric) { c = parseFloat(a.getAttribute('data-' + key)) - parseFloat(b.getAttribute('data-' + key)); }
            else { var av = a.getAttribute('data-id'), bv = b.getAttribute('data-id'); c = (av < bv) ? -1 : (av > bv ? 1 : 0); }
            if (!c) { c = parseInt(a.getAttribute('data-order')) - parseInt(b.getAttribute('data-order')); }
            return desc ? -c : c;
        });
        var frag = document.createDocumentFragment();
        rows.forEach(function (r) { frag.appendChild(r); });
        tb.appendChild(frag);
        document.getElementById('sortstatus').textContent = 'ties fall back to the default activity order; missing values sort last';
    }
    </script>
"""


def _ssa_legend_html(z_tol, min_count, n_rep) -> str:
    """Bottom-of-page legend, mirroring the ODE matrix's footer: row colors, verdict
    badges, the three timing tiers (SSA), how PASS-vs-DIFF is decided (the z-gate),
    and the measured-not-annotated provenance note."""
    return f"""    <div class="legend">
        <h3>Row Color Legend</h3>
        <div class="legend-items">
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(to right, #28a745 4px, #e8f5e9 4px);"></div>
                <div class="legend-desc"><strong>Green (PASSED):</strong> Both engines' ensembles agreed within the z-gate</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(to right, #dc3545 4px, #ffebee 4px);"></div>
                <div class="legend-desc"><strong>Red (FAILED):</strong> a divergence the screen pins on BNGsim (its SSA mean strays from its own ODE), or BNGsim raised (EXCEPTION) — the only states that mean "look at BNGsim"</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(to right, #ffc107 4px, #fffbea 4px);"></div>
                <div class="legend-desc"><strong>Yellow (TRIAGED):</strong> a divergence attributed to AMICI's gillespie (BNGsim tracks its own ODE, AMICI does not, and the two ODEs agree) or a filed AMICI issue, or AMICI refused the model (REF-FAIL) — none reflect on BNGsim</div>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: linear-gradient(to right, #6c757d 4px, #f8f9fa 4px);"></div>
                <div class="legend-desc"><strong>Gray (REFUSED):</strong> TOO SLOW (exact SSA intractable within the wall cap — a coverage gap, not a failure), both engines rejected the model (BAD_TEST), or the SBML was missing (SKIP)</div>
            </div>
        </div>
        <h3 style="margin-top: 25px;">Verdict Badges (within cells)</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div><strong>PASS:</strong> this engine's ensemble agreed (or, on a AMICI-side divergence, BNGsim tracked its own ODE)</div>
            <div><strong>DIFF:</strong> this engine's ensemble diverged beyond the z-gate</div>
            <div><strong>EXCEPTION:</strong> BNGsim raised while AMICI ran (actionable BNGsim issue)</div>
            <div><strong>REF-FAIL:</strong> AMICI refused to load/run the model; BNGsim ran it</div>
            <div><strong>TOO SLOW:</strong> exact SSA exceeded the per-model wall cap (coverage gap)</div>
            <div><strong>REFUSED:</strong> both engines rejected the model</div>
        </div>
        <h3 style="margin-top: 25px;">How agreement (PASS vs DIFF) is decided</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div>Each model is run as an ensemble of <strong>{n_rep} independent Gillespie trajectories per engine</strong> on a shared seed schedule. At every output time and every shared species, the two ensembles' means are compared with a two-sample <strong>z-score</strong> = |mean<sub>BNGsim</sub> − mean<sub>AMICI</sub>| ÷ combined standard error. A cell agrees when its z-score stays ≤ <strong>{z_tol}</strong>.</div>
            <div>Cells where <em>both</em> engines sit at or below <strong>{min_count} molecules</strong> are skipped: at a handful of molecules the replicate variance is ~0 and a one-molecule Monte-Carlo difference reads as a spurious many-sigma fail (the guard is two-sided, so it never masks a one-sided low-count divergence).</div>
            <div><strong>PASS</strong> = no cell exceeds the gate. <strong>DIFF</strong> = at least one does — and is then attributed against a third oracle, <em>each engine's own deterministic ODE</em>: if BNGsim's SSA mean tracks BNGsim's ODE while AMICI's does not (and the two ODEs agree), the divergence is AMICI-side (yellow), not a BNGsim defect (red). The number shown per row ("Largest mean z-score") is the worst surviving cell.</div>
            <div style="color:#868e96;">(Protocol source of truth: <span class="path-info">parity_checks/rr_parity/ssa_screen.py</span>; the screen also runs the third-oracle attribution and the low-count floor.)</div>
        </div>
        <h3 style="margin-top: 25px;">Timing: three cost tiers</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div style="margin-bottom: 6px; color: #868e96;">Each engine cell reports timing as three labeled tiers. An ensemble-sampling or fitting run pays tier ① once for the whole run, tier ② once per model, and tier ③ on <em>every</em> trajectory — so tier ③ is the bottom line.</div>
            <div><strong>① Per-process startup</strong> (once per run): one-time library import — BNGsim's <code>_bngsim_core</code> extension load (the SSA path imports <strong>no SymPy</strong> and derives no Jacobian), AMICI's library import. Amortized to ~0 over a run, so it is never added into the per-model or per-replicate numbers.</div>
            <div><strong>② Per-model load</strong> (once per model): SBML → ready-to-simulate.
                <span style="display:block; margin-left:14px;">— <strong>Parse (libSBML):</strong> the shared C++ XML parse — comparable between engines.</span>
                <span style="display:block; margin-left:14px;">— <strong>Interpret:</strong> building the engine's internal model (BNGsim: Python; AMICI: C++).</span>
                <span style="display:block; margin-left:14px;">— <strong>Codegen + JIT</strong> (AMICI only): LLVM IR-generation + JIT compilation, which dominates AMICI's load. BNGsim evaluates propensities with an ExprTk interpreter and has <strong>no codegen</strong> on the SSA path.</span>
                <span style="display:block; margin-left:14px;">— <strong>Build subtotal:</strong> the one-time per-model cost.</span></div>
            <div><strong>③ Per-replicate ensemble</strong> (the bottom line): the marginal cost of one Gillespie trajectory.
                <span style="display:block; margin-left:14px;">— <strong>Per-replicate (median):</strong> the median single-trajectory wall over the {n_rep} replicates.</span>
                <span style="display:block; margin-left:14px;">— <strong>Cold → warm:</strong> the first replicate (cold — one-time page-fault / CPU ramp) vs the median of the rest (warm). Unlike the ODE matrix there is no warm <em>reuse</em> of a single solve: each replicate is an independent reseeded trajectory.</span>
                <span style="display:block; margin-left:14px;">— <strong>Ensemble (N reps):</strong> total simulation wall over all replicates.</span>
                <span style="display:block; margin-left:14px;">— <strong>Setup / rep</strong> (BNGsim only): the clone + Simulator construction BNGsim pays per replicate; AMICI reuses one compiled model, so it has no analog. Shown separately so it never contaminates the per-trajectory comparison.</span></div>
        </div>
        <h3 style="margin-top: 25px;">Provenance — measured, not annotated</h3>
        <div style="font-size: 0.9em; color: #495057; line-height: 1.8;">
            <div>Every value is read from what <em>actually executed</em>: the <strong>species/reaction/parameter/compartment counts and feature tags</strong> are parsed from the SBML file; the <strong>timings</strong> are measured at each phase boundary (BNGsim's via its public <code>Model</code>/<code>Simulator</code> accessors, AMICI's via its instrumented <code>getLoadTimings()</code> and <code>__warmup_sec__</code>); the <strong>versions</strong> come from the live packages. The timing is measurement-only — it never changes a verdict.</div>
        </div>
    </div>
"""


def generate_ssa_html(report_path: Path, output_path: Path) -> None:
    """Render an SSA cross-engine screen ``_core`` report (runs/ssa_report.json) as
    a matrix that DELIBERATELY MIRRORS the ODE matrix (``generate_html``) — same
    BNGsim-vs-AMICI three-column layout, same green/yellow/red/gray color
    convention, same model cell (species/reaction/parameter/compartment counts +
    feature tags), same top-of-page cost plots + wins tables + summary boxes, same
    bottom-of-page legend — so a reader moving between the two tables is never
    disoriented. Only the *content* differs where the regime does: the agreement
    metric is a mean z-score (not a relative trajectory difference), and the timing
    taxonomy is SSA's (per-process startup / per-model load / per-replicate
    ensemble), since BNGsim's SSA path derives no Jacobian and emits no codegen and
    simulation is an ensemble of N independent reseeded Gillespie trajectories
    rather than one repeated solve.
    """

    report = json.loads(report_path.read_text())
    meta = report["_meta"]
    results = report["results"]
    config = meta.get("config", {})
    versions = meta.get("versions", {})
    machine = meta.get("machine", {})

    # Per-model structural features (species/reactions/parameters/compartments +
    # feature tags + citation), parsed from the same SBML the screen ran — exactly
    # as the ODE matrix does, so the two model cells are identical.
    features_by_id: dict[str, dict] = {}
    for r in results:
        p = _resolve_ssa_sbml(r["model_id"], config)
        features_by_id[r["model_id"]] = extract_sbml_features(p) if p else {}

    def _n(feat, key):
        v = feat.get(key)
        return v if isinstance(v, int) else float("inf")

    def _events_of(r):
        # Mean reaction events per unit simulated time (BNGsim ensemble) — the
        # stochastic-activity axis the matrix is ordered and plotted on, since it
        # drives per-trajectory cost far more than the species count. None when
        # BNGsim never ran the model (its timing/events are absent).
        v = ((r.get("timing") or {}).get("bngsim") or {}).get("events_per_time")
        return v if isinstance(v, (int, float)) else None

    def sort_key(r):
        # Primary order: stochastic activity (reaction events / unit time). Models
        # BNGsim never ran (no events) sort last via +inf, then by structural
        # complexity, then id — a deterministic total order.
        ev = _events_of(r)
        f = features_by_id[r["model_id"]]
        return (
            ev if ev is not None else float("inf"),
            _n(f, "n_species"),
            _n(f, "n_reactions"),
            _n(f, "n_parameters"),
            _n(f, "n_compartments"),
            r["model_id"],
        )

    results_sorted = sorted(results, key=sort_key)
    n_models = len(results_sorted)

    # Summary buckets, by the honest SSA row policy (_ssa_row_class): green PASS,
    # red scoring-DIFF/EXCEPTION, yellow attributed-away-DIFF/REFERENCE_FAILED, gray
    # TOO-SLOW/BAD_TEST/SKIP — the SSA analog of the ODE PASSED/FAILED/TRIAGED/REFUSED.
    buckets = {"status-passed": 0, "status-failed": 0, "status-triaged": 0, "status-refused": 0}
    for r in results:
        css, _ = _ssa_row_class(r["outcome"], r.get("subclass"))
        buckets[css] = buckets.get(css, 0) + 1
    n_pass, n_failed = buckets["status-passed"], buckets["status-failed"]
    n_triaged, n_refused = buckets["status-triaged"], buckets["status-refused"]
    pct = (lambda c: 100 * c / n_models) if n_models else (lambda c: 0.0)

    # Run config, in plain English (no internal knob names like "effort").
    n_rep = config.get("n_replicates", "?")
    z_tol = config.get("mean_z_tol", "?")
    min_count = config.get("min_mean_count", "?")
    t_end = config.get("biomodels_t_end", "?")
    n_steps = config.get("biomodels_n_steps", "?")
    seed_base = config.get("seed_base", "?")
    wall_cap = config.get("per_model_timeout_sec", "?")
    jobs = config.get("jobs")

    warmup_html = _ssa_warmup_summary_html(results)

    # Hardware + concurrency caveat (same wording shape as the ODE matrix).
    cpu = machine.get("processor") or versions.get("platform", "?")
    plat = machine.get("platform", versions.get("platform", "?"))
    cores = machine.get("cpu_count")
    hardware_html = (
        f"<div><strong>Hardware:</strong> {_escape(str(cpu))}"
        f"{f' · {cores} logical cores' if cores else ''} · {_escape(str(plat))}</div>"
    )
    if jobs:
        hardware_html += (
            '<div class="hwcaveat"><strong>Timing context:</strong> per-replicate and load costs '
            f"were measured with <strong>{jobs} model jobs running concurrently</strong> (process-"
            "parallel, for throughput), so absolute costs include CPU/memory contention and are "
            "<strong>not</strong> quiescent single-process times. The BNGsim-vs-AMICI "
            "comparison is unaffected — both engines run back-to-back in the same job under the same "
            "ambient load.</div>"
        )

    # ---- top-of-page cost plots, wins tables, geomean stats ----------------
    # Two cost dimensions (the SSA analog of the ODE warm/cold pair): per-replicate
    # simulation (one Gillespie trajectory — the bottom line for ensemble sampling)
    # and per-model load (SBML → ready model — where BNGsim's no-JIT path is far
    # cheaper than AMICI's). Same canvas-scatter + wins-by-species-bin machinery.
    def _series(engine, key):
        return [((r.get("timing") or {}).get(engine) or {}).get(key) for r in results_sorted]

    bn_rep, rr_rep = _series("bngsim", "rep_median_sec"), _series("amici", "rep_median_sec")
    bn_load, rr_load = _series("bngsim", "load_sec"), _series("amici", "load_sec")
    species_seq = [features_by_id.get(r["model_id"], {}).get("n_species") for r in results_sorted]
    event_seq = [_events_of(r) for r in results_sorted]
    has_timing = any(isinstance(x, (int, float)) for x in bn_rep + rr_rep)
    has_events = any(isinstance(x, (int, float)) for x in event_seq)

    # The plot x-axis / wins-bin / sort variable: stochastic ACTIVITY (reaction
    # events per unit time) when the report carries it — that, not the species
    # count, is what drives per-trajectory cost — else the structural species count
    # (older reports). One code path; only the edges + axis label differ.
    if has_events:
        axis_seq = event_seq
        AXIS_EDGES = [0.1, 1, 10, 100, 1000, 10000]
        _axis_is_int = False
    else:
        axis_seq = species_seq
        AXIS_EDGES = [2, 5, 10, 20, 35, 51, 100, 250, 500, 1000]
        _axis_is_int = True

    import math as _math

    def _pairs(a_list, b_list):
        return [
            (a, b)
            for a, b in zip(a_list, b_list, strict=False)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0
        ]

    def _geomean_ratio(a_list, b_list):
        ps = _pairs(a_list, b_list)
        if not ps:
            return None, 0
        return _math.exp(sum(_math.log(a / b) for a, b in ps) / len(ps)), len(ps)

    def _ratio_phrase(geo):
        if geo is None:
            return "—"
        if geo < 1:
            return f"{geo:.2f} → BNGsim {1 / geo:.2f}× faster"
        return f"{geo:.2f} → BNGsim {geo:.2f}× slower"

    rep_geo, rep_geo_n = _geomean_ratio(bn_rep, rr_rep)
    load_geo, load_geo_n = _geomean_ratio(bn_load, rr_load)

    def _axis_ranges():
        ranges, prev = [], None
        for e in AXIS_EDGES:
            ranges.append((prev, e))
            prev = e
        ranges.append((prev, None))
        return ranges

    def _fmt_axis(v):
        if not isinstance(v, (int, float)):
            return "?"
        if _axis_is_int:
            return str(int(v))
        if v >= 1:
            return str(int(v)) if float(v).is_integer() else f"{v:g}"
        return f"{v:g}"

    def _bin_label(lo, hi):
        if hi is None:
            return f"&gt;{_fmt_axis(lo)}"
        if lo is None:
            return f"≤{_fmt_axis(hi)}"
        if _axis_is_int:
            return f"{int(lo) + 1}–{int(hi)}"
        return f"{_fmt_axis(lo)}–{_fmt_axis(hi)}"

    def _wins_rows(a_list, b_list):
        rows_ = []
        for lo, hi in _axis_ranges():
            seg = [
                (a_list[i], b_list[i])
                for i, s in enumerate(axis_seq)
                if isinstance(s, (int, float))
                and (lo is None or s > lo)
                and (hi is None or s <= hi)
                and isinstance(a_list[i], (int, float))
                and isinstance(b_list[i], (int, float))
                and a_list[i] > 0
                and b_list[i] > 0
            ]
            rows_.append(
                (
                    _bin_label(lo, hi),
                    len(seg),
                    sum(1 for a, b in seg if a < b),
                    sum(1 for a, b in seg if b < a),
                )
            )
        return rows_

    def _win_row_html(label, nseg, bn_w, rr_w, *, total=False):
        bn_pct = (100 * bn_w / nseg) if nseg else 0
        weight = "700" if total else "600"
        cls = ' class="wintot"' if total else ""
        return (
            f"<tr{cls}><td>{label}</td><td>{nseg}</td>"
            f'<td style="color:#0072B2;font-weight:{weight};">{bn_w}</td>'
            f'<td style="color:#D55E00;font-weight:{weight};">{rr_w}</td>'
            f'<td><div class="winbar"><span style="width:{bn_pct:.0f}%;"></span></div></td></tr>'
        )

    def _wins_table_html(title, rows_):
        body = "".join(_win_row_html(*r) for r in rows_)
        body += _win_row_html(
            "all models",
            sum(r[1] for r in rows_),
            sum(r[2] for r in rows_),
            sum(r[3] for r in rows_),
            total=True,
        )
        axis_th = "Events/time" if has_events else "Species"
        return (
            f'<div class="winstab"><div class="plottitle">{title}</div>'
            f"<table><thead><tr><th>{axis_th}</th><th>n</th>"
            '<th style="color:#0072B2;">BNGsim wins</th><th style="color:#D55E00;">AMICI wins</th>'
            "<th>BNGsim win fraction</th></tr></thead><tbody>"
            f"{body}</tbody></table></div>"
        )

    def _plot_xticks():
        out = {}
        for e in AXIS_EDGES:
            idx = None
            for i, s in enumerate(axis_seq):
                if isinstance(s, (int, float)) and s <= e:
                    idx = i
            if idx is not None:
                out[idx] = _fmt_axis(e)
        valid = [i for i, s in enumerate(axis_seq) if isinstance(s, (int, float))]
        if valid:
            out[valid[-1]] = _fmt_axis(axis_seq[valid[-1]])
        return sorted([k, v] for k, v in out.items())

    def _round(xs):
        return [round(x, 9) if isinstance(x, (int, float)) else None for x in xs]

    plots_html = ""
    if has_timing:
        bin_word = "reaction-rate" if has_events else "species"
        wins_html = (
            '<div class="winsgrid">'
            + _wins_table_html(
                f"Per-replicate-cost wins by {bin_word} bin", _wins_rows(bn_rep, rr_rep)
            )
            + _wins_table_html(f"Load-cost wins by {bin_word} bin", _wins_rows(bn_load, rr_load))
            + "</div>"
        )
        stats_html = (
            '<div class="coststats">'
            "<strong>Geometric-mean cost ratio (BNGsim : AMICI)</strong> — "
            f"per-replicate simulation: <strong>{_ratio_phrase(rep_geo)}</strong> (n={rep_geo_n}) "
            f"&nbsp;·&nbsp; per-model load: <strong>{_ratio_phrase(load_geo)}</strong> (n={load_geo_n})"
            '<div style="font-size:0.85em;color:#868e96;margin-top:6px;">Per-replicate simulation is '
            "the marginal cost of one Gillespie trajectory — the bottom line for ensemble sampling. "
            "Per-model load is paid once per model (BNGsim parses+interprets only; AMICI also "
            "JIT-compiles, which dominates its load). &lt;1 ⇒ BNGsim faster. "
            + (
                "Models are ordered left-to-right by <strong>stochastic activity</strong> — mean "
                "reaction events fired per unit simulated time over the ensemble — which drives "
                "per-trajectory cost far more than the species count does."
                if has_events
                else "Models are ordered by structural complexity (species count)."
            )
            + "</div></div>"
        )
        x_full = "reaction events / unit time" if has_events else "species count"
        order_word = "stochastic activity" if has_events else "complexity"
        plot_data_json = json.dumps(
            {
                "n": n_models,
                "br": _round(bn_rep),
                "rr": _round(rr_rep),
                "bl": _round(bn_load),
                "rl": _round(rr_load),
                "xt": _plot_xticks(),
                "xlabel": f"{x_full}  (models ordered by {order_word}, n={n_models})",
            },
            separators=(",", ":"),
        )
        plots_html = f"""<div class="plots">
        <h2 style="margin:0 0 6px 0;font-size:1.05em;color:#333;">Cost across the corpus</h2>
        <div class="plotctl">
            <strong>y-scale:</strong>
            <label><input type="radio" name="plotscale" value="log" checked onchange="drawPlots()"> log</label>
            <label><input type="radio" name="plotscale" value="linear" onchange="drawPlots()"> linear (robust, capped at 99th pct)</label>
            <span class="plotkey"><span class="dot fill" style="background:#0072B2;border-color:#0072B2;"></span>/<span class="dot" style="border-color:#0072B2;"></span> BNGsim (circle) &nbsp; <span class="dot sq fill" style="background:#D55E00;border-color:#D55E00;"></span>/<span class="dot sq" style="border-color:#D55E00;"></span> AMICI (square) &nbsp;·&nbsp; <strong>filled = faster (win)</strong>, open = slower &nbsp;·&nbsp; x = {x_full} (models in {order_word} order, n={n_models})</span>
        </div>
        {stats_html}
        <div class="plotwrap"><div class="plottitle">Per-replicate simulation — one Gillespie trajectory (median over the {n_rep} replicates)</div><canvas id="cv_rep" class="costplot"></canvas></div>
        <div class="plotwrap"><div class="plottitle">Per-model load — SBML → ready-to-simulate (BNGsim parse+interpret · AMICI +codegen/JIT)</div><canvas id="cv_load" class="costplot"></canvas></div>
        {wins_html}
        <script>
        var PLOT_DATA = {plot_data_json};
        function _scatter(canvasId, bn, rr, logScale) {{
            var cv = document.getElementById(canvasId);
            if (!cv) return;
            var cssW = cv.clientWidth || 940, cssH = 210;
            var dpr = window.devicePixelRatio || 1;
            cv.width = cssW * dpr; cv.height = cssH * dpr; cv.style.height = cssH + 'px';
            var ctx = cv.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, cssW, cssH);
            var padL = 60, padR = 12, padT = 10, padB = 34, pw = cssW - padL - padR, ph = cssH - padT - padB;
            var n = PLOT_DATA.n, vals = [];
            for (var i = 0; i < n; i++) {{ if (bn[i] > 0) vals.push(bn[i]); if (rr[i] > 0) vals.push(rr[i]); }}
            if (!vals.length) return;
            var ymin = Math.min.apply(null, vals), ymax = Math.max.apply(null, vals);
            var lo, hi, cap = null;
            if (logScale) {{ lo = Math.floor(Math.log10(ymin)); hi = Math.ceil(Math.log10(ymax)); if (hi <= lo) hi = lo + 1; }}
            else {{
                var sv = vals.slice().sort(function (a, b) {{ return a - b; }});
                cap = sv[Math.min(sv.length - 1, Math.floor(sv.length * 0.99))];
                if (!(cap > 0)) cap = ymax;
                lo = 0; hi = cap * 1.02;
            }}
            function fx(i) {{ return padL + (n > 1 ? pw * i / (n - 1) : pw / 2); }}
            function fy(v) {{ var t = logScale ? (Math.log10(Math.max(v, 1e-12)) - lo) / (hi - lo) : (v - lo) / (hi - lo); return padT + ph * (1 - t); }}
            ctx.font = '9px -apple-system,sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
            if (logScale) {{
                for (var e = lo; e <= hi; e++) {{ var gy = fy(Math.pow(10, e)); ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(cssW - padR, gy); ctx.stroke(); ctx.fillStyle = '#999'; var lab = e >= 0 ? (Math.pow(10, e) + 's') : (Math.round(Math.pow(10, e + 3) * 1000) / 1000 + 'ms'); ctx.fillText(lab, padL - 5, gy); }}
            }} else {{
                for (var k = 0; k <= 5; k++) {{ var v = lo + (hi - lo) * k / 5, gy2 = fy(v); ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(padL, gy2); ctx.lineTo(cssW - padR, gy2); ctx.stroke(); ctx.fillStyle = '#999'; ctx.fillText((v * 1000).toFixed(v < 0.01 ? 2 : 0) + 'ms', padL - 5, gy2); }}
            }}
            var nOver = 0;
            function pt(x, v, color, filled, square) {{
                var y = fy(v), rad = 2.2;
                if (y < padT) {{ y = padT; nOver++; }}
                ctx.globalAlpha = filled ? 0.72 : 0.6;
                if (square) {{
                    if (filled) {{ ctx.fillStyle = color; ctx.fillRect(x - rad, y - rad, 2 * rad, 2 * rad); }}
                    else {{ ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.strokeRect(x - rad + 0.5, y - rad + 0.5, 2 * rad - 1, 2 * rad - 1); }}
                }} else {{
                    ctx.beginPath(); ctx.arc(x, y, rad, 0, 6.2832);
                    if (filled) {{ ctx.fillStyle = color; ctx.fill(); }}
                    else {{ ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke(); }}
                }}
                ctx.globalAlpha = 1;
            }}
            for (var i = 0; i < n; i++) {{
                var b = bn[i], r = rr[i];
                if (!(b > 0) || !(r > 0)) continue;
                var bWin = b < r;
                pt(fx(i), b, '#0072B2', bWin, false);
                pt(fx(i), r, '#D55E00', !bWin, true);
            }}
            if (!logScale && nOver > 0) {{
                ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillStyle = '#c0392b';
                ctx.font = '10px -apple-system,sans-serif';
                ctx.fillText('\\u25b2 ' + nOver + ' point(s) above ' + (cap * 1000).toFixed(cap < 0.01 ? 2 : 0) + 'ms cap (clamped)', cssW - padR, padT + 2);
            }}
            var xt = PLOT_DATA.xt || [];
            ctx.font = '9px -apple-system,sans-serif'; ctx.textBaseline = 'top';
            var lastLabelX = -1e9;
            for (var t = 0; t < xt.length; t++) {{
                var idx = xt[t][0], lab = xt[t][1], x = fx(idx);
                ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); ctx.stroke();
                ctx.strokeStyle = '#bbb'; ctx.beginPath(); ctx.moveTo(x, padT + ph); ctx.lineTo(x, padT + ph + 3); ctx.stroke();
                if (x - lastLabelX > 24 || t === xt.length - 1) {{
                    ctx.fillStyle = '#888'; ctx.textAlign = (t === xt.length - 1 ? 'right' : 'center');
                    ctx.fillText(lab, x, padT + ph + 4);
                    lastLabelX = x;
                }}
            }}
            ctx.textAlign = 'center'; ctx.fillStyle = '#666';
            ctx.fillText(PLOT_DATA.xlabel || ('species count, n=' + n), padL + pw / 2, padT + ph + 16);
        }}
        function drawPlots() {{
            var sel = document.querySelector('input[name=plotscale]:checked');
            var logScale = !sel || sel.value === 'log';
            _scatter('cv_rep', PLOT_DATA.br, PLOT_DATA.rr, logScale);
            _scatter('cv_load', PLOT_DATA.bl, PLOT_DATA.rl, logScale);
        }}
        window.addEventListener('load', drawPlots);
        window.addEventListener('resize', drawPlots);
        </script>
    </div>"""

    # ---- table rows --------------------------------------------------------
    table_rows = ""
    for row_index, r in enumerate(results_sorted):
        model_id = r["model_id"]
        outcome = r["outcome"]
        subclass = r.get("subclass")
        value, tol = r.get("value"), r.get("tol")
        timing = r.get("timing") or {}
        bn_t, rr_t = timing.get("bngsim", {}), timing.get("amici", {})
        row_css, _ = _ssa_row_class(outcome, subclass)

        f = features_by_id.get(model_id, {})
        n_sp = f.get("n_species", "?")
        n_rx = f.get("n_reactions", "?")
        n_pa = f.get("n_parameters", "?")
        n_co = f.get("n_compartments", "?")
        citation = f.get("citation", "Unknown")
        feature_badges = "".join(
            f'<span class="feature-badge {"notable" if any(x in feat for x in ["event", "rule", "large", "variable"]) else ""}">{_escape(feat)}</span>'
            for feat in f.get("features", [])
        )
        note = (r.get("comment") or "").strip()
        comment_html = f'<div class="row-comment">{_escape(note)}</div>' if note else ""

        # Stochastic activity (the sort/plot axis), surfaced prominently at the top
        # of the left model cell: mean reaction events fired per unit simulated time
        # over the BNGsim ensemble — the variable that drives per-trajectory cost.
        ev = _events_of(r)
        evr = bn_t.get("events_per_rep")
        if isinstance(ev, (int, float)):
            rate_s = (
                f"{ev:,.0f}"
                if ev >= 100
                else f"{ev:.1f}"
                if ev >= 1
                else (f"{ev:.2g}" if ev > 0 else "0")
            )
            if bn_t.get("events_probe"):
                # Recovered for a timed-out model from a short-window probe (it never
                # finished the full ensemble) — flag it as an estimate, not a mean.
                qual = "bounded-window probe — too active to finish the full run"
            else:
                per_rep_s = f"{evr:,.0f}" if isinstance(evr, (int, float)) else "?"
                qual = f"{per_rep_s} per trajectory, mean of {n_rep}"
            activity_html = (
                f'<div class="model-activity"><strong>≈ {rate_s}</strong> reaction events / time unit'
                f" <span>({qual})</span></div>"
            )
        elif outcome == "TIMEOUT":
            # Timed out AND too active for even the smallest probe window to finish
            # — the activity is real but beyond what the probe can measure. It is
            # still the most active end of the corpus, so it sorts last (rightmost).
            activity_html = (
                '<div class="model-activity"><strong>extreme activity</strong> '
                "<span>(exceeds the probe's measurable range — never completes)</span></div>"
            )
        else:
            activity_html = ""

        model_html = f"""                <td>
                    <div class="model-info">{model_id}</div>
                    {activity_html}
                    <div class="model-citation">{_escape(str(citation))}</div>
                    <div class="model-metadata">
                        <strong>{n_sp}</strong> species · <strong>{n_rx}</strong> reactions · <strong>{n_pa}</strong> parameters · <strong>{n_co}</strong> compartment{"s" if n_co != 1 else ""}
                    </div>
                    <div class="model-features">{feature_badges}</div>
                    {comment_html}
                </td>"""

        bn_badge, bn_color, rr_badge, rr_color = _ssa_verdict_badges(outcome, subclass)

        # Agreement metric — the SSA analog of the ODE relative-difference line.
        metric_html = ""
        if value is not None and tol is not None and outcome in ("PASS", "DIFF"):
            alert = "alert" if outcome == "DIFF" else ""
            word = "exceeds" if outcome == "DIFF" else "within"
            metric_html = (
                f'<div class="metric {alert}">Largest BNGsim-vs-AMICI mean z-score: '
                f"<strong>{value:.2f}</strong> ({word} the {tol:g} agreement gate)</div>"
            )

        # Per-replicate speed ratio (BNGsim vs AMICI), same shape as the ODE row.
        ratio_html = ""
        bm, rm = bn_t.get("rep_median_sec"), rr_t.get("rep_median_sec")
        if bm and rm:
            ratio = bm / rm
            faster = ratio < 1.0
            phrase = f"{1 / ratio:.2f}× faster than" if faster else f"{ratio:.2f}× slower than"
            color = "#28a745" if faster else "#dc3545"
            ratio_html = (
                '<div class="metric">Per-replicate cost: BNGsim is '
                f'<strong style="color:{color}">{phrase}</strong> AMICI '
                f"({fmt_ms(bm)} vs {fmt_ms(rm)}, medians of n={bn_t.get('n_rep', 0)}/{rr_t.get('n_rep', 0)} replicates)</div>"
            )

        bn_foot = rr_foot = ""
        if outcome == "EXCEPTION":
            bn_foot = '<div class="footnote"><strong>Actionable BNGsim issue:</strong> BNGsim raised on this model</div>'
            if r.get("exception"):
                bn_foot += f'<div class="error-msg">{_escape(r["exception"][:400])}</div>'
        elif outcome == "REFERENCE_FAILED":
            rr_foot = (
                '<div class="footnote"><strong>AMICI refused</strong> this model (its '
                "gillespie solver lacks a feature BNGsim supports — e.g. rate rules); BNGsim ran it.</div>"
            )
            if r.get("exception"):
                rr_foot += f'<div class="error-msg">{_escape(r["exception"][:400])}</div>'
        elif outcome == "TIMEOUT":
            bn_foot = rr_foot = (
                '<div class="footnote">Exact SSA could not finish within the per-model wall cap — a '
                "known coverage gap for intractable models, not a failure.</div>"
            )
        elif outcome == "BAD_TEST":
            bn_foot = rr_foot = '<div class="footnote">Both engines rejected this model.</div>'
        elif outcome == "DIFF" and subclass in ("diff_not_bngsim", "rr_known"):
            rr_foot = (
                '<div class="footnote">The divergence is on <strong>AMICI\'s</strong> side '
                "(BNGsim's SSA mean tracks its own ODE; AMICI's does not, and the two ODEs agree) "
                "— not a BNGsim defect.</div>"
            )
        elif outcome == "DIFF" and subclass == "partial_horizon":
            bn_foot = rr_foot = (
                '<div class="footnote"><strong>Partial-horizon adjudication.</strong> The full '
                "horizon is exact-SSA-intractable, so both engines were compared over the largest "
                "window they could simulate — a <strong>provisional</strong> verdict, not a full "
                "pass/fail. See the activity rate and the comment for the window.</div>"
            )
        elif outcome == "DIFF" and subclass == "attribution_incomplete":
            bn_foot = rr_foot = (
                '<div class="footnote">The SSA ensembles <strong>diverged</strong> over the full '
                "horizon, but the deterministic-ODE oracle that assigns fault did not finish — so "
                "this is <strong>provisional</strong> (fault undetermined), not a confirmed BNGsim "
                "defect.</div>"
            )

        bn_rhs = "ExprTk bytecode (interpreted — no codegen)"
        rr_cache = cache_display(rr_t.get("config", {}), engine="amici")
        bn_html = f"""                <td class="engine-cell {row_css}">
                    <div class="config-section"><div class="config-label">Method</div><div class="config-value">Gillespie SSA (exact, direct method)</div></div>
                    <div class="config-section"><div class="config-label">Propensity evaluation</div><div class="config-value">{bn_rhs}</div></div>
                    <div class="config-section"><div class="config-label">Compiled-model cache</div><div class="config-value">None — ExprTk interpreter (no codegen)</div></div>
                    {render_ssa_timing_tiers(timing, engine="bngsim")}
                    <div class="verdict {bn_color}">{bn_badge}</div>
                    {metric_html}
                    {ratio_html}
                    {bn_foot}
                </td>"""
        rr_html = f"""                <td class="engine-cell {row_css}">
                    <div class="config-section"><div class="config-label">Method</div><div class="config-value">Gillespie SSA (exact)</div></div>
                    <div class="config-section"><div class="config-label">Propensity evaluation</div><div class="config-value">LLVM IR → JIT (compiled)</div></div>
                    <div class="config-section"><div class="config-label">Compiled-model cache</div><div class="config-value">{rr_cache}</div></div>
                    {render_ssa_timing_tiers(timing, engine="amici")}
                    <div class="verdict {rr_color}">{rr_badge}</div>
                    {rr_foot}
                </td>"""

        # "Failures first" sorts by the ROW COLOR (the honest SSA category), not the
        # raw outcome — otherwise a red and a yellow DIFF (same outcome, different
        # subclass) interleave and the yellow rows don't cluster. red → yellow →
        # gray → green.
        statusrank = {
            "status-failed": 0,  # red: scoring DIFF / EXCEPTION
            "status-triaged": 1,  # yellow: attributed-away DIFF / REF-FAIL
            "status-refused": 2,  # gray: TOO SLOW / BAD_TEST / SKIP
            "status-passed": 3,  # green
        }.get(row_css, 4)
        data_attrs = (
            f' data-order="{row_index}" data-id="{_escape(model_id)}" data-status="{statusrank}"'
            f' data-events="{_data_num(ev)}"'
            f' data-species="{_data_num(f.get("n_species"))}"'
            f' data-reactions="{_data_num(f.get("n_reactions"))}"'
            f' data-params="{_data_num(f.get("n_parameters"))}"'
            f' data-comp="{_data_num(f.get("n_compartments"))}"'
            f' data-bnrep="{_data_num(bn_t.get("rep_median_sec"))}"'
            f' data-rrrep="{_data_num(rr_t.get("rep_median_sec"))}"'
            f' data-bnload="{_data_num(bn_t.get("load_sec"))}"'
            f' data-rrload="{_data_num(rr_t.get("load_sec"))}"'
        )
        table_rows += f"""            <tr{data_attrs}>
{model_html}
{bn_html}
{rr_html}
            </tr>
"""

    generated = meta.get("generated", datetime.now().strftime("%Y-%m-%d %H:%M"))
    rr_ver = versions.get("amici", "not installed")
    bn_ver = versions.get("bngsim", "N/A")
    libsbml_ver = versions.get("libsbml")
    libsbml_bit = f" · libSBML {libsbml_ver}" if libsbml_ver else ""

    config_html = (
        f"<div><strong>How each model is checked:</strong> run {n_rep} independent Gillespie "
        f"trajectories per engine (shared seeds from {seed_base}) over t=0→{t_end} sampled at "
        f"{n_steps} points, then compare the two ensemble means cell-by-cell.</div>"
        f"<div><strong>Agreement gate:</strong> a model PASSes when every (species, time-point) "
        f"cell's mean z-score between the engines stays ≤ <strong>{z_tol}</strong>; cells where both "
        f"engines sit at ≤ {min_count} molecules are skipped (at a few molecules Monte-Carlo noise "
        f"swamps the signal). A surviving divergence is attributed against each engine's own ODE.</div>"
        f"<div><strong>Per-model wall cap:</strong> {wall_cap}s — a model that can't finish exact "
        f"SSA in this budget is marked TOO SLOW (a coverage gap, shown gray, not counted as a "
        f"failure).</div>"
    )

    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"    <title>BNGsim vs AMICI — SSA Parity Matrix ({datetime.now().strftime('%Y-%m-%d')})</title>\n"
        f"    <style>{_SSA_MATRIX_CSS}</style>\n</head>\n<body>\n"
        '    <div class="header">\n'
        "        <h1>BNGsim vs AMICI — SSA Parity Matrix</h1>\n"
        '        <div class="metadata">\n'
        "            <div><strong>Suite:</strong> rr_parity (SSA regime — stochastic, Gillespie)</div>\n"
        f"            <div><strong>Swept:</strong> {generated} &nbsp;·&nbsp; <strong>Rendered:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>\n"
        f"            <div><strong>Versions:</strong> BNGsim {bn_ver} · AMICI {rr_ver}{libsbml_bit}</div>\n"
        f"            {hardware_html}\n"
        f"            {config_html}\n"
        f"            <div><strong>Models:</strong> {n_models} &nbsp;·&nbsp; <strong>Git revision:</strong> "
        f'<span class="path-info">{(meta.get("git_commit") or meta.get("git_rev") or "N/A")[:8]}</span></div>\n'
        f"            {warmup_html}\n"
        "        </div>\n"
        '        <div class="summary">\n'
        f'            <div class="summary-item summary-passed">✓ {n_pass} PASSED ({pct(n_pass):.1f}%)</div>\n'
        f'            <div class="summary-item summary-failed">✗ {n_failed} FAILED ({pct(n_failed):.1f}%)</div>\n'
        f'            <div class="summary-item summary-triaged">⚠ {n_triaged} TRIAGED ({pct(n_triaged):.1f}%)</div>\n'
        f'            <div class="summary-item summary-refused">⊘ {n_refused} REFUSED ({pct(n_refused):.1f}%)</div>\n'
        "        </div>\n"
        f"        {plots_html}\n"
        "    </div>\n"
        '    <div class="sortbar">\n'
        "        <strong>Sort rows by:</strong>\n"
        '        <select id="sortsel" onchange="requestSort()">\n'
        '            <option value="order">Stochastic activity — reaction events / unit time (default)</option>\n'
        '            <option value="events">Reaction events / unit time</option>\n'
        '            <option value="species">Species count</option>\n'
        '            <option value="id">Model ID</option>\n'
        '            <option value="status">Status (failures first)</option>\n'
        '            <option value="bnrep">BNGsim per-replicate cost</option>\n'
        '            <option value="rrrep">AMICI per-replicate cost</option>\n'
        '            <option value="bnload">BNGsim load cost</option>\n'
        '            <option value="rrload">AMICI load cost</option>\n'
        "        </select>\n"
        '        <span class="sortdir"><strong>Order:</strong>\n'
        '            <label><input type="radio" name="sortdir" value="asc" checked onchange="requestSort()"> ascending</label>\n'
        '            <label><input type="radio" name="sortdir" value="desc" onchange="requestSort()"> descending</label>\n'
        "        </span>\n"
        '        <span id="sortstatus" class="sortnote">ties fall back to the default activity order; missing values sort last</span>\n'
        "    </div>\n"
        '    <table id="matrix">\n        <thead>\n            <tr>\n'
        '                <th class="col-model">Model</th>\n'
        '                <th class="col-engine">BNGsim</th>\n'
        '                <th class="col-engine">AMICI</th>\n'
        "            </tr>\n        </thead>\n        <tbody>\n"
        + table_rows
        + "        </tbody>\n    </table>\n"
        + _ssa_legend_html(z_tol, min_count, n_rep)
        + _SSA_SORT_JS
        + "</body>\n</html>\n"
    )
    output_path.write_text(html)
    tally = meta.get("outcome_tally", {})
    tally_str = " · ".join(f"{k} {v}" for k, v in tally.items() if v)
    print(f"Generated SSA matrix: {output_path}")
    print(
        f"  {n_models} models | {tally_str} | regressions: {meta.get('n_not_expected', 0)}"
        f"{' | timing rendered' if has_timing else ' | (no timing in report)'}"
    )


def generate_index(runs_dir: Path, index_path: Path) -> int:
    """Render every ``runs_dir/report_ode*.json`` to its own per-combo matrix HTML
    and write a top-level ``index.html`` linking them with each combo's aggregate
    bngsim-time speedup vs the auto baseline (T3.2). Returns the combo count.
    """
    report_paths = sorted(runs_dir.glob("report_ode*.json"))
    if not report_paths:
        print(f"No report_ode*.json found in {runs_dir}")
        return 0

    combos: dict[str, dict] = {}
    for rp in report_paths:
        data = json.loads(rp.read_text())
        meta = data.get("_meta", {})
        results = data.get("results", [])
        combo = combo_from_report(meta, rp)
        html_name = "amici_matrix.html" if combo == "auto" else f"amici_matrix__{combo}.html"
        generate_html(rp, runs_dir / html_name)
        totals = {r["model_id"]: bngsim_total_sec(r) for r in results}
        tally = meta.get("tally", {})
        combos[combo] = {
            "meta": meta,
            "html": html_name,
            "totals": {k: v for k, v in totals.items() if v is not None},
            "pass": tally.get("PASS", 0),
            "n": meta.get("n_jobs", len(results)),
        }

    speedups = aggregate_combo_speedups(combos)
    # auto first, then the rest alphabetically.
    order = sorted(combos, key=lambda c: (c != "auto", c))

    rows = ""
    for combo in order:
        info = combos[combo]
        sp = speedups[combo]
        link = info["html"]
        if combo == "auto":
            speed_cell = "<em>baseline</em>"
        elif sp["speedup"] is None:
            speed_cell = "—"
        else:
            faster = sp["speedup"] >= 1.0
            tag = "faster" if faster else "slower"
            color = "#28a745" if faster else "#dc3545"
            speed_cell = (
                f'<span style="color:{color};font-weight:600;">{sp["speedup"]:.2f}× {tag}</span>'
                f' <span style="color:#888;">(n={sp["n_common"]})</span>'
            )
        rows += (
            "            <tr>"
            f'<td><a href="{link}">{combo}</a></td>'
            f"<td>{info['n']}</td>"
            f"<td>{info['pass']}</td>"
            f"<td>{speed_cell}</td>"
            "</tr>\n"
        )

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>rr_parity — method-combo index</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin: 30px; background: #f5f5f5; color: #333; }}
        h1 {{ margin-bottom: 4px; }}
        .sub {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                 border-radius: 6px; overflow: hidden; }}
        th, td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #343a40; color: white; }}
        a {{ color: #0366d6; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>rr_parity — method-combo index</h1>
    <div class="sub">Generated {generated} · speedup = aggregate bngsim wall vs the
        <strong>auto</strong> baseline over the models both combos timed (&gt;1 = faster).</div>
    <table>
        <thead><tr><th>Combo</th><th>Models</th><th>PASS</th><th>Speedup vs auto</th></tr></thead>
        <tbody>
{rows}        </tbody>
    </table>
</body>
</html>
"""
    index_path.write_text(index_html)
    print(f"Generated index: {index_path}  ({len(combos)} combo report(s))")
    for combo in order:
        sp = speedups[combo]
        s = "baseline" if combo == "auto" else (f"{sp['speedup']:.2f}x" if sp["speedup"] else "—")
        print(f"  {combo:14s} {s}")
    return len(combos)


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML parity matrix from report_ode.json"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=HERE / "runs" / "report_ode.json",
        help="Path to report JSON (default: runs/report_ode.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "runs" / "amici_matrix.html",
        help="Output HTML path (default: runs/amici_matrix.html)",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help=(
            "Render every runs/report_ode*.json to its own per-combo HTML plus a "
            "top-level index.html comparing each combo to the auto baseline (T3.2). "
            "Scans the directory of --report; writes index.html beside it."
        ),
    )
    parser.add_argument(
        "--regime",
        choices=["auto", "ode", "ssa"],
        default="auto",
        help=(
            "Which matrix to render. 'auto' (default) detects SSA reports by their "
            "outcome_tally/method=ssa meta and renders the SSA matrix (T4.1); "
            "everything else renders the ODE matrix."
        ),
    )
    args = parser.parse_args()

    if args.index:
        runs_dir = args.report.parent
        n = generate_index(runs_dir, runs_dir / "index.html")
        return 0 if n else 1

    if not args.report.exists():
        print(f"Error: Report file not found: {args.report}")
        return 1

    regime = args.regime
    if regime == "auto":
        meta = json.loads(args.report.read_text()).get("_meta", {})
        # The SSA _core report (ssa_attribution.payload_to_report) carries
        # outcome_tally + method='ssa'; the ODE report (rr_run) carries regime='ode'.
        regime = "ssa" if (meta.get("method") == "ssa" or "outcome_tally" in meta) else "ode"

    if regime == "ssa":
        generate_ssa_html(args.report, args.out)
    else:
        generate_html(args.report, args.out)
    return 0


if __name__ == "__main__":
    exit(main())
