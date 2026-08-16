#!/usr/bin/env python3
"""Render ``runs/report_sens.json`` -> ``runs/amici_sens_matrix.html``.

The forward-sensitivity companion to ``generate_amici_matrix.py``. It is a small
purpose-built renderer rather than a fork of that 3.5k-line generator, because
the sensitivity page answers different questions and its rows are a different
shape (one per model x CVODES corrector method, each carrying a parameter count
without which no timing number means anything).

What the page shows, and why each column earns its place:

  verdict       PASS/DIFF/... on the SENSITIVITY tensor — the headline.
  state         Whether the underlying STATE trajectories agreed. A sensitivity
                DIFF on a row whose states already diverged is not evidence about
                the sensitivity machinery, and the page must not let the two be
                confused.
  Np            Parameters actually differentiated, and the candidate count the
                cap was applied to. The coupled system is n_species*(Np+1), so a
                warm time is uninterpretable without it, and "20 of 43" discloses
                the truncation instead of hiding it.
  warm solve    The headline efficiency number: marginal per-solve cost of the
                coupled state+sensitivity integration, for each engine, plus the
                ratio. Warm (not cold) because that is what a fitting or MCMC
                loop actually pays per evaluation.
  build         The one-time pre-simulation cost, split parse / interpret /
                Jacobian derivation / RHS build. AMICI's sensitivity C++ compile
                dominates and is shown separately from the solve so it is never
                mistaken for per-iteration cost.

Usage:
    python generate_amici_sens_matrix.py [--report runs/report_sens.json]
                                         [--out runs/amici_sens_matrix.html]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys as _sys
from datetime import datetime
from html import escape as _escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(HERE.parent))
_sys.path.insert(0, str(HERE))

# ``INVALID_RESULT_REFUSAL`` from the disposition module rather than a literal, so
# the badge below and the class the runner writes can never drift apart. The other
# import reuses the ODE generator's small formatting helpers rather than restating
# them, so the two pages format a millisecond, a row colour and a verdict badge
# identically. Both are side-effect free (everything is behind main()).
from amici_dispositions import INVALID_RESULT_REFUSAL  # noqa: E402
from generate_amici_matrix import (  # noqa: E402
    classify_row,
    fmt_ms,
)

try:
    from _core import differ as _differ
except Exception:  # pragma: no cover — the legend falls back to documented values
    _differ = None


def is_degenerate_pass(outcome: str, extra: dict) -> bool:
    """A PASS that established nothing — issue #328's vacuous pass.

    ``n_resolvable_params == 0`` means every parameter column of the sensitivity
    tensor sat below the magnitude the solver can resolve (``atol/|p_j|``), so the
    comparison measured integration noise against zero: the two engines "agreed"
    about a tensor neither could resolve. ``None`` (the floor was not assessable)
    is not a degeneracy claim, and ``differ``'s own significance gate cannot catch
    this — that gate is relative to the file-wide peak, and when the whole tensor
    is tiny there is no larger peak to judge against.

    Only a PASS can be vacuous. A DIFF failed on something real — the noise floor
    never forgives a one-sided non-finite cell, so a NaN column still DIFFs — so a
    zero-resolvable DIFF established a genuine disagreement, not nothing.
    """
    return outcome == "PASS" and extra.get("n_resolvable_params") == 0


def sens_row_class(outcome: str, extra: dict, refusal: str | None = None) -> tuple[str, str]:
    """``(row css class, verdict badge)`` for one sensitivity row.

    Mirrors the SSA matrix's vacuous-pass treatment (GH #190): a PASS that
    resolved no parameter column is rendered as a gray coverage gap with a
    distinct ``NO SIGNAL`` badge, not a green ``PASS``. The report and the tally
    still record it as PASS — this is issue #328's "keep PASS, flag it": the
    outcome is unchanged, only how a reader sees the row, so a vacuous row can no
    longer be read as evidence of agreement.

    ``refusal`` gets the same treatment for the opposite direction (issue #380).
    A REFERENCE_FAILED row normally means AMICI could not produce an answer at
    all; ``invalid_result`` means it produced one and the answer is the problem —
    a materially different fact for a reader deciding whether a model is worth
    re-triaging, and one the shared gray badge would flatten. Same outcome, same
    tally, distinct badge.
    """
    if is_degenerate_pass(outcome, extra):
        return "status-refused", "NO SIGNAL"
    cls, _cat = classify_row(outcome)
    if outcome == "REFERENCE_FAILED" and refusal == INVALID_RESULT_REFUSAL:
        return cls, "REF INVALID"
    return cls, outcome


def _fmt_ratio(bn: float | None, am: float | None) -> tuple[str, str]:
    """``(text, css_class)`` for the bngsim-vs-AMICI warm-solve ratio.

    Expressed as "bngsim is Nx faster/slower", the direction a reader of this
    suite cares about. Returns an em dash when either side has no warm sample
    (a solve too slow for the warm budget), never a fabricated 1.0.
    """
    if not bn or not am:
        return "—", ""
    r = am / bn
    if r >= 1:
        return f"{r:.2f}x faster", "ratio-good"
    return f"{1 / r:.2f}x slower", "ratio-bad"


def _engine_warm(timing: dict, engine: str) -> float | None:
    t = (timing or {}).get(engine) or {}
    return t.get("integrate_warm_min_sec") or t.get("integrate_sec")


def _engine_build(timing: dict, engine: str) -> float:
    t = (timing or {}).get(engine) or {}
    return sum(
        float(t.get(k) or 0.0)
        for k in (
            "parse_sec",
            "interpret_sec",
            "jac_derive_sec",
            "codegen_sec",
            "compile_sec",
            "load_sec",
            "sens_setup_sec",
        )
    )


def _phase_cells(timing: dict, engine: str) -> str:
    t = (timing or {}).get(engine) or {}
    # "RHS build" folds codegen+compile+load for the same reason the ODE matrix
    # does: compiling is a sub-step of producing the callable evaluator, not a
    # peer phase, and bngsim cannot be split that way at all.
    rhs = sum(
        float(t.get(k) or 0.0)
        for k in ("codegen_sec", "compile_sec", "load_sec", "sens_setup_sec")
    )
    parts = [
        ("parse", t.get("parse_sec")),
        ("interpret", t.get("interpret_sec")),
        ("jac", t.get("jac_derive_sec")),
        ("RHS build", rhs),
    ]
    return "".join(
        f'<div class="phase"><span class="pl">{lbl}</span>'
        f'<span class="pv">{fmt_ms(val)}</span></div>'
        for lbl, val in parts
    )


def _summarize(results: list, key) -> str:
    vals = [v for v in (key(r) for r in results) if v]
    if not vals:
        return "—"
    return f"median {fmt_ms(statistics.median(vals))} (n={len(vals)})"


def _speedup_summary(results: list) -> str:
    """Geometric mean of the warm-solve ratio over rows where both engines
    produced a warm sample. Geometric, not arithmetic: these are ratios, and an
    arithmetic mean of speedups is dominated by whichever direction happens to
    produce large numbers."""
    ratios = []
    for r in results:
        bn = _engine_warm(r.get("timing") or {}, "bngsim")
        am = _engine_warm(r.get("timing") or {}, "amici")
        if bn and am:
            ratios.append(am / bn)
    if not ratios:
        return "—"
    g = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    direction = "faster" if g >= 1 else "slower"
    val = g if g >= 1 else 1 / g
    return f"bngsim {val:.2f}x {direction} (geomean over {len(ratios)} rows)"


CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e3e3e3;--head:#f6f7f9;
      --pass:#e7f6ec;--fail:#fdecea;--triage:#fff8e1;--refuse:#f1f1f1;
      --good:#137333;--bad:#c5221f;}
*{box-sizing:border-box}
body{margin:0;padding:24px;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",
     Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);margin-bottom:18px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.card{border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:150px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.card .v{font-size:18px;font-weight:600;margin-top:2px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);
      vertical-align:top;white-space:nowrap}
th{background:var(--head);position:sticky;top:0;font-size:11px;text-transform:uppercase;
   letter-spacing:.04em;color:var(--muted)}
tr.status-passed{background:var(--pass)}
tr.status-failed{background:var(--fail)}
tr.status-triaged{background:var(--triage)}
tr.status-refused{background:var(--refuse)}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;
       font-weight:600;background:#fff;border:1px solid var(--line)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.ratio-good{color:var(--good);font-weight:600}
.ratio-bad{color:var(--bad);font-weight:600}
.phase{display:flex;justify-content:space-between;gap:10px;font-size:11px;color:var(--muted)}
.phase .pv{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--fg)}
.note{color:var(--muted);font-size:12px;max-width:70ch;white-space:normal}
.legend{margin-top:22px;border-top:1px solid var(--line);padding-top:14px}
.legend h2{font-size:14px;margin:0 0 8px}
.legend p{margin:0 0 8px;max-width:80ch;color:#333}
.state-ok{color:var(--good)}
.state-bad{color:var(--bad);font-weight:600}
"""


def generate_html(report_path: Path, output_path: Path) -> None:
    data = json.loads(report_path.read_text())
    meta = data.get("_meta", data.get("meta", {}))
    results = data.get("results", [])

    tallies = meta.get("tally", {})
    ver = meta.get("versions", {})
    cap = meta.get("param_cap")
    methods = meta.get("sens_methods", [])
    n_state_diff = (meta.get("state_parity") or {}).get("n_state_diff", 0)

    rows = []
    for r in sorted(results, key=lambda x: (x.get("model_id", ""), x.get("method", ""))):
        extra = r.get("extra") or {}
        cls, badge = sens_row_class(r.get("outcome", ""), extra, r.get("reference_refusal"))
        timing = r.get("timing") or {}
        bn_warm = _engine_warm(timing, "bngsim")
        am_warm = _engine_warm(timing, "amici")
        ratio_txt, ratio_cls = _fmt_ratio(bn_warm, am_warm)

        np_used = extra.get("n_params")
        np_cand = extra.get("n_param_candidates")
        if np_used is None:
            np_txt = "—"
        elif np_cand and np_cand > np_used:
            np_txt = f"{np_used} <span class='note'>of {np_cand}</span>"
        else:
            np_txt = str(np_used)

        state = extra.get("state_passed")
        if state is None:
            state_txt = "—"
        elif state:
            state_txt = "<span class='state-ok'>ok</span>"
        else:
            state_txt = "<span class='state-bad'>DIFF</span>"

        val = r.get("value")
        val_txt = "—" if val is None else (f"{val:.3g}" if val == val else "nan")

        # Issue #324 — lead a failure detail with its grouping key. The message
        # behind it is capped and its variable part can be an arbitrarily long
        # symbol enumeration, so the key is what makes a column of these
        # scannable ("everything that says bngsim-params:ModelError").
        detail = _escape(r.get("comment") or r.get("exception") or "")
        exc_cls = r.get("exception_class")
        if exc_cls and not r.get("comment"):
            detail = f"[{_escape(exc_cls)}] {detail}"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td class='mono'>{_escape(r.get('model_id', ''))}</td>"
            f"<td>{_escape(extra.get('sens_method') or r.get('method', ''))}</td>"
            f"<td><span class='badge'>{_escape(badge)}</span></td>"
            f"<td>{state_txt}</td>"
            f"<td class='mono'>{val_txt}</td>"
            f"<td>{np_txt}</td>"
            f"<td class='mono'>{fmt_ms(bn_warm)}</td>"
            f"<td class='mono'>{fmt_ms(am_warm)}</td>"
            f"<td class='{ratio_cls}'>{ratio_txt}</td>"
            f"<td>{_phase_cells(timing, 'bngsim')}</td>"
            f"<td>{_phase_cells(timing, 'amici')}</td>"
            f"<td class='note'>{detail}</td>"
            f"</tr>"
        )

    rel_tol = getattr(_differ, "REL_TOL", 1e-4)
    ceil = getattr(_differ, "HARD_REL_CEILING", 0.05)
    budget = getattr(_differ, "FAIL_FRAC_BUDGET", 5e-3)

    # Issue #328 — how many of those PASS rows established nothing (every parameter
    # column below the solver's resolvable floor). A subset of the pass tally, not
    # a separate outcome, so the reader sees "N passed, of which M compared nothing".
    n_no_signal = sum(
        1 for r in results if is_degenerate_pass(r.get("outcome", ""), r.get("extra") or {})
    )
    # Issue #380 — the REFERENCE_FAILED rows where AMICI answered and the answer is
    # what is unusable, as opposed to the ones where it could not answer at all.
    n_ref_invalid = sum(1 for r in results if r.get("reference_refusal") == INVALID_RESULT_REFUSAL)

    cards = [
        ("models", meta.get("n_models", "—")),
        ("jobs", meta.get("n_jobs", len(results))),
        ("pass", tallies.get("PASS", 0)),
        ("no signal", n_no_signal),
        ("diff", tallies.get("DIFF", 0)),
        ("unsupported", tallies.get("UNSUPPORTED", 0)),
        ("ref failed", tallies.get("REFERENCE_FAILED", 0)),
        ("ref invalid", n_ref_invalid),
        ("bad test", tallies.get("BAD_TEST", 0)),
        ("timeout", tallies.get("TIMEOUT", 0)),
        ("param cap", cap if cap else "none"),
    ]
    cards_html = "".join(
        f"<div class='card'><div class='k'>{_escape(str(k))}</div>"
        f"<div class='v'>{_escape(str(v))}</div></div>"
        for k, v in cards
    )

    hw = meta.get("hardware", {})
    workers = (meta.get("concurrency") or {}).get("workers")
    tol = meta.get("integration_tol", {})

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>bngsim vs AMICI — forward-sensitivity parity</title>
<style>{CSS}</style></head><body>
<h1>bngsim vs AMICI — forward-sensitivity parity</h1>
<div class="sub">
  bngsim {_escape(str(ver.get("bngsim", "?")))} ·
  AMICI {_escape(str(ver.get("amici", "?")))} ·
  SUNDIALS {_escape(str(ver.get("sundials", "?")))} ·
  methods {_escape(", ".join(methods)) or "?"} ·
  rtol {tol.get("rtol", "?")} atol {tol.get("atol", "?")} ·
  generated {datetime.now().isoformat(timespec="seconds")}
</div>
<div class="cards">{cards_html}</div>
<div class="cards">
  <div class="card"><div class="k">warm solve — bngsim</div><div class="v">
    {_summarize(results, lambda r: _engine_warm(r.get("timing") or {}, "bngsim"))}</div></div>
  <div class="card"><div class="k">warm solve — AMICI</div><div class="v">
    {_summarize(results, lambda r: _engine_warm(r.get("timing") or {}, "amici"))}</div></div>
  <div class="card"><div class="k">warm speedup</div><div class="v">
    {_speedup_summary(results)}</div></div>
  <div class="card"><div class="k">build — AMICI</div><div class="v">
    {_summarize(results, lambda r: _engine_build(r.get("timing") or {}, "amici"))}</div></div>
</div>

<div class="wrap"><table>
<thead><tr>
  <th>model</th><th>method</th><th>verdict</th><th>state</th><th>max rel err</th>
  <th>Np</th><th>bngsim warm</th><th>AMICI warm</th><th>ratio</th>
  <th>bngsim build</th><th>AMICI build</th><th>detail</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>

<div class="legend">
<h2>Reading this page</h2>
<p><b>Verdict</b> is on the forward-sensitivity tensor <span class="mono">dx_i(t)/dp_j</span>,
compared with the shared <span class="mono">_core.differ</span> protocol applied to the tensor
flattened to one column per (species, parameter) pair — per-cell tolerance
<span class="mono">rel {rel_tol:g}</span>, hard ceiling <span class="mono">{ceil:g}</span>,
fail-fraction budget <span class="mono">{budget:g}</span>. Judging each coefficient against its
own column peak matters more here than for trajectories: sensitivity magnitudes span many orders
across the parameters of a single model.</p>
<p><b>State</b> is a separate verdict on the underlying trajectories. A sensitivity DIFF on a row
where state is also DIFF says nothing about the sensitivity machinery — the two engines were not
on the same trajectory to begin with. {n_state_diff} row(s) in this run are in that category.</p>
<p><b>NO SIGNAL</b> is a PASS that established nothing (issue #328): every parameter column of the
sensitivity tensor sat below the magnitude either solver can resolve (<span class="mono">atol/|p_j|</span>),
so the two engines "agreed" about a tensor neither could resolve. {n_no_signal} row(s) here are in
that category. Such a row stays a PASS in the tally — the comparison did not fail — but it is
rendered as a gray coverage gap rather than a green agreement, so a vacuous row is not read as
evidence. The <span class="mono">differ</span> significance gate cannot catch this on its own: that
gate is relative to the file-wide peak, and when the whole tensor is tiny there is no larger peak to
judge against. The per-row <span class="mono">extra</span> fields disambiguate the cause — a near-zero
<span class="mono">state_span</span> means the model does essentially nothing over the horizon, while
a moving state with an all-zero <span class="mono">max_abs_sx</span> is worth a second look.</p>
<p><b>Np</b> is how many parameters were actually differentiated; "20 of 43" means the cap
dropped the rest. The coupled system solved is <span class="mono">n_species*(Np+1)</span>, so a
warm time cannot be compared across rows with different Np. Both engines are handed the identical
parameter list, in the identical order, with AMICI's parameter scale pinned to linear so both
report <span class="mono">dx/dp</span> rather than <span class="mono">dx/dln(p)</span>.</p>
<p><b>Warm solve</b> is the marginal per-solve cost — what a fitting or MCMC loop pays per
evaluation — not the first solve. <b>Build</b> is the one-time pre-simulation cost; AMICI's
sensitivity C++ compile dominates it and is cached on disk, so it is paid once per model per
corpus, never per iteration.</p>
<p><b>Non-scoring rows.</b> None of these count toward the failing tally
(<span class="mono">DIFF / EXCEPTION / TIMEOUT</span> are the scoring outcomes).
UNSUPPORTED means bngsim <i>declared</i> it cannot differentiate this model — a
<span class="mono">SensitivityUnsupportedError</span> raised because an event's crossing time
moves in a way it cannot compute, or because a rate law does not differentiate to closed form.
It is matched by exception <b>type</b>, never by message text, and is a documented capability gap
rather than a bug, so bucketing it with EXCEPTION would only dilute that signal.
REFERENCE_FAILED means AMICI could not produce an oracle (bngsim is untested, not vindicated).
BAD_TEST means either both engines failed, or the model has no parameter both engines can
differentiate — there is nothing to compare, which is not a verdict about bngsim.</p>
<p><b>REF INVALID</b> is a REFERENCE_FAILED reached by an authored disposition rather than by an
AMICI crash (issue #380): AMICI <i>ran</i> and returned finite numbers, but a defect in its own
handling of the model makes them unusable as an oracle, established against an independent third
engine with the perturbation applied to the SBML text. {n_ref_invalid} row(s) here are in that
category, each carrying its evidence in the detail column. It is REFERENCE_FAILED and not PASS
because the engines did not agree — naming the gap is the point — and not BAD_TEST because bngsim
ran the model fine. A DIFF that <i>neither</i> engine is at fault for (a cell at a discontinuity
that no finite-difference oracle can resolve) is instead reclassified to PASS, again with the
reason in the detail column. Both hold only while the row is genuinely a DIFF: one that agrees on
its own is flagged STALE for pruning, so a disposition can never quietly outlive its evidence.</p>
<p class="note">Timings collected under {workers or "?"}-way process concurrency on
{_escape(str(hw.get("cpu", "unknown")))}
({hw.get("physical_cores", "?")} physical cores) — comparable within this page, not against a
single-process benchmark.</p>
</div>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(HERE / "runs" / "report_sens.json"))
    ap.add_argument("--out", default=str(HERE / "runs" / "amici_sens_matrix.html"))
    args = ap.parse_args()
    report = Path(args.report).resolve()
    if not report.exists():
        raise SystemExit(f"no report at {report}; run amici_sens_run.py first.")
    out = Path(args.out).resolve()
    generate_html(report, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
