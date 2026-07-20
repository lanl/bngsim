#!/usr/bin/env python3
"""Render the ssa_table5 four-engine SSA timing results (results/ssa_timing_ballpark.json)
to a markdown report + enrich the JSON with normalized per-cell reasons.

Outputs:
  * results/ssa_timing_ballpark.md   — main per-cell table, per-model speedup-vs-bngsim,
                                        14x4 coverage matrix, legend + methodology
  * rewrites results/ssa_timing_ballpark.json with a `reason` on every cell + `legend`
"""

from __future__ import annotations

import json
from pathlib import Path

import _ssa_config as C

HERE = Path(__file__).resolve().parent
JPATH = HERE / "results" / "ssa_timing_ballpark.json"
MPATH = HERE / "results" / "ssa_timing_ballpark.md"

ENGINE_LABEL = {
    "bngsim": "BNGsim",
    "run_network": "run_network",
    "roadrunner": "RoadRunner",
    "copasi": "COPASI",
}

# Post-hoc faithfulness overrides discovered by inspecting the trajectories (not a pre-run
# coverage gate — these cells RAN, but produced an unfaithful trajectory). COPASI's Direct
# method fires a time-triggered event only if the system is active around the trigger time;
# 860 and 864 are INERT until their t=3600 event seeds all activity, so COPASI computes an
# infinite time-to-next-reaction and jumps PAST the trigger — the event never fires and the
# trajectory is degenerate (all species 0 at t_end; the ~130 µs "run" is timing nothing).
# bngsim fires them (TF1=500 / Signal=1); 862 is active from t=0 so COPASI catches its event.
UNFAITHFUL = {
    (
        "BIOMD0000000860",
        "copasi",
    ): "COPASI Direct method did NOT fire the t=3600 event (TF1:=500): "
    "system inert until the event, SSA jumps past the trigger -> degenerate all-zero trajectory "
    "(bngsim fires it: TF1=500). Wall is timing an inert run, not the model.",
    (
        "BIOMD0000000864",
        "copasi",
    ): "COPASI Direct method did NOT fire the t=3600 event (Signal:=1): "
    "inert-until-event, SSA jumps past the trigger -> degenerate all-zero trajectory "
    "(bngsim fires it: Signal=1, miR=325). Wall is timing an inert run, not the model.",
}


def normalize_reason(r: dict) -> str:
    """One-line human reason for a cell's outcome."""
    st = r.get("status")
    key = (r["model"], r["engine"])
    if key in UNFAITHFUL and st == "ok":
        return "N/A (ran but incorrect) — " + UNFAITHFUL[key]
    if st == "N/A":
        return r.get("na_reason", "")
    if st == "ok":
        bits = []
        if r.get("faithfulness_flag"):
            bits.append("FLAG: " + r["faithfulness_flag"])
        for n in r.get("notes", []) or []:
            bits.append(n)
        return " | ".join(bits)
    err = (r.get("error") or "").lower()
    if "functional" in err:
        return "run_network -p ssa cannot evaluate a functional (non-mass-action) rate law in the converted .net"
    if "edgepop" in err:
        return "run_network 'edgepop' observable crash (known legacy BNG reference-side bug)"
    if (
        "overran" in err
        or "exceeded 120" in err
        or "cold run exceeded" in err
        or "per-run cap" in err
    ):
        return "too_slow: a single exact-SSA run exceeded the 120 s per-run cap"
    if "process failed" in err:
        return "COPASI Direct method aborted at runtime"
    return r.get("error", "")[:160]


def display_status(r: dict) -> str:
    # A cell that RAN but produced an incorrect (degenerate) trajectory is N/A —
    # same bucket as an engine that cannot simulate the model at all (RoadRunner's
    # event-blindness). It ran (raw status "ok") but the result is not usable.
    if (r["model"], r["engine"]) in UNFAITHFUL and r.get("status") == "ok":
        return "N/A"
    st = r.get("status")
    return {
        "ok": "ok",
        "N/A": "N/A",
        "too_slow": "too_slow",
        "sim_fail": "fail",
        "load_error": "fail",
        "missing": "missing",
    }.get(st, st)


def fnum(x, nd=4):
    if isinstance(x, (int, float)):
        if x == 0:
            return "0"
        if abs(x) >= 100:
            return f"{x:,.1f}"
        return f"{x:.{nd}g}"
    return "—"


def eng_events_run(r):
    """events/run to display: self for BNGsim, the shared reference (†) for the others."""
    if r.get("events_self") and r.get("events_warm_median") is not None:
        return f"{r['events_warm_median']:,}"
    ref = r.get("events_ref_used")
    return f"{ref:,}†" if ref else "—"


def main():
    d = json.loads(JPATH.read_text())
    by = {(r["model"], r["engine"]): r for r in d["results"]}
    # enrich + persist reason
    for r in d["results"]:
        r["reason"] = normalize_reason(r)
        r["display_status"] = display_status(r)
        if (r["model"], r["engine"]) in UNFAITHFUL:
            r["unfaithful"] = True
            r["unfaithful_reason"] = UNFAITHFUL[(r["model"], r["engine"])]
            r["events_per_sec"] = (
                None  # the reference-based rate is meaningless for a degenerate run
            )

    lines = []
    A = lines.append
    A("# arXiv Table 5 — exact-SSA cost across four engines (BALLPARK)\n")
    A(
        f"**Suite:** `ssa_table5` · **BNGsim** {d.get('bngsim_version')} · "
        f"**engines:** BNGsim (`method=ssa`), BNG `run_network -p ssa`, RoadRunner `gillespie`, "
        f"COPASI Direct method — all exact Gillespie SSA.\n"
    )
    hw = d.get("hardware", {})
    A(
        f"**Hardware:** {hw.get('cpu', '?')} ({hw.get('physical_cores', '?')} physical cores) · "
        f"Python {hw.get('python', '?')}\n"
    )
    A(
        "> ⚠️ **BALLPARK / SETUP RUN.** Wall times were collected under "
        f"**{d.get('workers')}-way concurrency** (mild contention) with a "
        f"**{int(d.get('per_run_cap_sec', 120))} s per-run cap**. The harness is validated end-to-end; "
        "**final numbers require a clean serial re-run (`--workers 1`)**. Cold = load/convert + build + "
        "first run; Warm = median of the next N runs reusing the loaded model "
        f"(warm truncated once cumulative warm wall > {int(d.get('warm_budget_sec', 150))} s).\n"
    )

    # ---- main per-cell table ----
    A("## Per-cell timing (model × engine)\n")
    A(
        "`events/run` is the engine's own Gillespie firing count where it reports one (BNGsim); "
        "for run_network / RoadRunner / COPASI it is the shared BNGsim reference at the same "
        "model+horizon (†), and `events/s` = that count ÷ warm-median wall.\n"
    )
    A(
        "| model | sp·rx | engine | conversion | status | cold (s) | warm (s) | N | events/run | events/s |"
    )
    A("|---|--:|---|---|:--:|--:|--:|--:|--:|--:|")
    for k in C.ordered_models():
        sprx = _sprx(k)
        for i, eng in enumerate(C.ENGINES):
            r = by.get((k, eng), {})
            stat = display_status(r)
            badge = {"ok": "✅", "N/A": "▫️", "too_slow": "🐌", "fail": "❌"}.get(stat, stat)
            mlabel = f"`{k}`" if i == 0 else ""
            splabel = sprx if i == 0 else ""
            # An N/A cell has no usable timing — blank cold/warm/N/events-per-sec (and
            # the conversion label) so a degenerate run's wall is never shown as a time.
            na = stat == "N/A"
            conv = "—" if na else (r.get("conversion") or "—")
            cold = "—" if na else fnum(r.get("cold_sec"))
            warm = "—" if na else fnum(r.get("warm_median_sec"))
            nwarm = "—" if na else (r.get("n_warm", 0) or "—")
            eps = "—" if na else fnum(r.get("events_per_sec"), 4)
            A(
                f"| {mlabel} | {splabel} | {ENGINE_LABEL[eng]} | {conv} | {badge} "
                f"| {cold} | {warm} | {nwarm} | {eng_events_run(r)} | {eps} |"
            )
    A("")

    # ---- speedup vs bngsim ----
    A("## Speed-up vs BNGsim (warm median; >1 = faster than BNGsim)\n")
    A("| model | BNGsim warm (s) | run_network | RoadRunner | COPASI |")
    A("|---|--:|--:|--:|--:|")
    for k in C.ordered_models():
        sp = d["speedup"][k]
        bt = sp["bngsim_warm_median_sec"]
        cells = []
        for eng in ("run_network", "roadrunner", "copasi"):
            e = sp["engines"][eng]
            s = e["speedup_vs_bngsim"]
            stt = e["status"]
            if (k, eng) in UNFAITHFUL:
                cells.append("▫️ N/A")
            elif stt == "ok" and s is not None:
                cells.append(f"{e['warm_median_sec']:.4g} s ({s:.2f}×)")
            else:
                cells.append(
                    {
                        "N/A": "▫️ N/A",
                        "too_slow": "🐌 slow",
                        "sim_fail": "❌ fail",
                        "load_error": "❌ fail",
                        "missing": "—",
                    }.get(stt, stt)
                )
        A(f"| `{k}` | {fnum(bt)} | {cells[0]} | {cells[1]} | {cells[2]} |")
    A("")

    # ---- coverage matrix ----
    A("## Coverage matrix (14 × 4)\n")
    A(
        "✅ ran · 🐌 too_slow (>120 s/run) · ❌ engine limitation · "
        "▫️ N/A (cannot faithfully simulate, or ran but produced an incorrect trajectory)\n"
    )
    A("| model | BNGsim | run_network | RoadRunner | COPASI |")
    A("|---|:--:|:--:|:--:|:--:|")
    sym = {"ok": "✅", "too_slow": "🐌", "fail": "❌", "N/A": "▫️", "missing": "·"}
    for k in C.ordered_models():
        row = [
            sym.get(display_status(by.get((k, e), {"model": k, "engine": e})), "?")
            for e in C.ENGINES
        ]
        A(f"| `{k}` | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    A("")
    # tally from DISPLAY status (unfaithful pulled out of ok)
    dcnt = {}
    for k in C.ordered_models():
        for e in C.ENGINES:
            dcnt[display_status(by.get((k, e), {"model": k, "engine": e}))] = (
                dcnt.get(display_status(by.get((k, e), {"model": k, "engine": e})), 0) + 1
            )
    A(
        f"**Tally:** {dcnt.get('ok', 0)} ran · {dcnt.get('too_slow', 0)} too_slow · "
        f"{dcnt.get('fail', 0)} engine-limited · {dcnt.get('N/A', 0)} N/A (of 56 cells).\n"
    )

    # ---- notes / reasons for every non-clean cell ----
    A("## Why each non-✅ cell is not a clean timing\n")
    A("| model | engine | outcome | reason |")
    A("|---|---|:--:|---|")
    for k in C.ordered_models():
        for eng in C.ENGINES:
            r = by.get((k, eng), {})
            ds = display_status(r)
            if ds == "missing":
                continue
            if ds == "ok":
                if r.get("faithfulness_flag"):  # 344 RoadRunner
                    A(f"| `{k}` | {ENGINE_LABEL[eng]} | ✅(flag) | {r['reason']} |")
                continue
            badge = {"N/A": "▫️", "too_slow": "🐌", "fail": "❌", "unfaithful": "⚠️"}.get(ds, ds)
            A(f"| `{k}` | {ENGINE_LABEL[eng]} | {badge} | {normalize_reason(r)} |")
    A("")
    A("---")
    A("### Methodology notes\n")
    A(
        "- **Conversions** use BNGsim's own converter only (never BNG2.pl): `.net`→SBML feeds "
        "RoadRunner+COPASI; SBML→`.net` feeds run_network. Every `ConversionReport` is logged in "
        "`results/converted/conversion_log.json` (all faithful: `max_rhs_delta` ≲ 1e-16, no repeated "
        "reactants)."
    )
    A(
        "- **COPASI molecule-count fix:** models COPASI imports with quantity unit `mol` (all "
        "converted-BNGL SBML + native BIOMD035) are forced to `#` (particle number) so amount 1 ≡ 1 "
        "particle — otherwise COPASI multiplies by Avogadro and refuses the stochastic init "
        "('particle number too big'). Native BIOMD 860/862/864/344 import as `#`, 478 as dimensionless."
    )
    A(
        "- **run_network events:** each SSA replicate is a fresh process (no warm-solver reuse); "
        "'warm' here is repeated fresh-process calls, so its cold/warm gap is small."
    )
    A(
        "- **events/s reference (†):** run_network / RoadRunner / COPASI do not expose a firing "
        "count, so their `events/s` uses the BNGsim reference event count for the same "
        "model+horizon (a matched but independent stochastic realization)."
    )
    A(
        "- **Time-triggered events & inert-until-event models (KEY FINDING):** only **BNGsim** "
        "fires the t=3600 events in the miRNA-OA models (860/862/864) faithfully. RoadRunner-gillespie "
        "warns and won't fire them (**N/A**). **COPASI's Direct method fires them only when the system is "
        "active around the trigger** — so it is faithful on **862** (baseline activity from t=0) but on "
        "**860 and 864** (inert until the event) its SSA jumps past the trigger and returns a degenerate "
        "all-zero trajectory in ~130 µs. That is an incorrect result, so COPASI 860/864 are reported "
        "**N/A** (same as RoadRunner), not as a timing. This corrects the corpus's optimistic "
        "'COPASI=yes' for 860/864: for exact SSA, **BNGsim is the only engine that covers the "
        "inert-until-event models**."
    )
    A(
        "- Sub-millisecond cells (e.g. COPASI on the tiniest networks) are timer-resolution noisy; "
        "the serial re-run with the full warm-N settles them."
    )

    MPATH.write_text("\n".join(lines) + "\n")

    # persist enriched JSON + legend
    d["legend"] = {
        "status": {
            "ok": "ran (cold+warm captured)",
            "too_slow": ">120s per single run",
            "sim_fail": "engine limitation (functional rate law / edgepop / runtime abort)",
            "N/A": "coverage authority: engine cannot faithfully simulate this model",
        },
        "events_run_dagger": "for run_network/RoadRunner/COPASI, events/run and events/s use the "
        "BNGsim reference firing count at the same model+horizon (events_self=false).",
        "speedup": "warm-median BNGsim / warm-median engine; >1 = faster than BNGsim.",
    }
    JPATH.write_text(json.dumps(d, indent=2, default=str))
    print(f"wrote {MPATH}\nwrote {JPATH} (enriched)")


def _sprx(k):
    """species·reactions label from the corpus."""
    corpus = json.loads((HERE / "corpus.json").read_text())
    for m in corpus["bngl"]:
        if m["name"] == k:
            return f"{m['species']}·{m['reactions']}"
    for m in corpus["sbml"]:
        if m["id"] == k:
            return f"{m['species']}·{m['reactions']}"
    return ""


if __name__ == "__main__":
    main()
