#!/usr/bin/env python3
"""Nightly parity regression check: a fresh run's verdicts against a baseline (GH #702).

Every parity runner (``rr_run.py``, ``bng_ode_run.py``, ``bng_stoch_run.py``,
``amici_sens_run.py``) exits 1 whenever any model is DIFF / EXCEPTION / TIMEOUT,
and a full corpus always has some, so a nightly job cannot use the exit code as
its verdict. The SBML-suite runner exits 0 no matter what. This tool makes the
REPORT the verdict:

``extract``
    report -> a normalized per-case verdict file: outcome, subclass, for ODE rows
    the bngsim solver work counters (``solver_stats``), and bngsim-side wall
    seconds, the sum of ``BNGSIM_PHASES``, which the file records. A baseline is an
    extracted verdict file from a reviewed run, recorded on the same runner class
    the nightly uses.

``diff``
    fresh verdicts vs a baseline. Alerts (exit 1) on:

    * **regressed** -- a case that was good (PASS / pass) is anything else now,
      including missing from the run. Lost coverage (PASS -> TIMEOUT / SKIP /
      REFERENCE_FAILED) counts, because a model that silently stops being compared
      is how a regression hides.
    * **new crash** -- a case that was already failing now crashes (EXCEPTION /
      load_fail / sim_fail / error) where it did not before.
    * **efficiency** -- on a case good in both runs, a bngsim work counter
      (steps, RHS evaluations, Jacobian evaluations) grew by more than
      ``--work-ratio`` and ``--work-min-delta``; or a counter's total over all
      such cases grew by more than ``--work-total-ratio``; or bngsim's total wall
      seconds grew by more than ``--wall-ratio`` on the same CPU model, or
      ``--wall-ratio-any-cpu`` across models, and by more than
      ``--wall-min-delta`` seconds (coarse: hosted runners mix CPU generations;
      precise timing is benchmarks/perf_ab.py on a fixed machine).
    * **wall phases** -- the two files' wall seconds sum different phases, so the
      wall check cannot run until the baseline is re-extracted
      (``--phases-may-differ`` reports it instead).
    * **incomplete** -- the run compared fewer than ``--min-fraction`` of the
      baseline's cases, so a green result would mean "compared too little".

    Improvements, new cases and changes between two failing states are reported
    but never alert.

``summarize``
    several diff JSONs -> one Markdown body (the tracking-issue comment).

``gate``
    fresh verdicts vs a baseline, exactly: exit 1 on ANY change of a case's
    outcome, in either direction. The per-pull-request SBML semantic-suite check
    (GH #700, ``ci/sbml_suite_gate.py``) uses it on the same baseline the nightly
    reads, so a pull request that changes an outcome records it in that baseline.

Stdlib only, so the alerting job can run it without building bngsim.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = 1

KINDS = ("core", "sbml-suite", "dsmts")

# What counts as good / as a crash, per report kind.
GOOD = {"core": {"PASS"}, "sbml-suite": {"pass"}, "dsmts": {"pass"}}
CRASH = {
    "core": {"EXCEPTION"},
    "sbml-suite": {"load_fail", "sim_fail", "harness_error"},
    "dsmts": {"error"},
}
# Outcomes that are an answer, and a wrong one, as opposed to a refusal. A case
# that moves from a refusal to one of these traded a loud failure for a silent one.
WRONG_ANSWER = {"sbml-suite": {"value_mismatch", "var_missing", "shape_mismatch"}}

# bngsim ``solver_stats`` counters kept per case. The first three are compared;
# the rest are kept so a regression report can say *why* the work grew.
WORK_KEYS = (
    "n_steps",
    "n_rhs_evals",
    "n_jac_evals",
    "n_nonlin_iters",
    "n_err_test_fails",
    "n_nonlin_conv_fails",
)
WORK_ALERT_KEYS = ("n_steps", "n_rhs_evals", "n_jac_evals")
# A per-case increase must clear this absolute floor too, so a 3 -> 5 Jacobian
# count on a trivial model does not alert.
WORK_MIN_DELTA = {"n_steps": 25, "n_rhs_evals": 50, "n_jac_evals": 10}
# And the wall total must grow by this many seconds too, so a fixed cost on a
# total that is near zero does not alert as a blowup: before issue #901 the
# stochastic legs' ``bngsim_sec`` was only the load phase, 0.011 s over bng_nf's
# 231 cases, and the millisecond XML scan #885 added to each session's
# construction read as x56. With the ensemble counted, bng_nf totals about 4100 s
# and bng_ssa about 840 s, and the ODE and sensitivity legs 79-913 s, so a real
# doubling clears it on every leg.
WALL_MIN_DELTA_SEC = 30.0

# What a verdict file extracted before issue #901 summed, which is what one that
# does not say is read as: comparing its bngsim_sec with one that counts the
# ensemble compared a load phase with a simulation (0.212 s -> 533 s, x2515).
# Spelled out rather than derived from BNGSIM_PHASES, so that a phase added later
# is not read as one an old file summed.
LEGACY_BNGSIM_PHASES = (
    "io_sec",
    "parse_sec",
    "interpret_sec",
    "load_sec",
    "jac_derive_sec",
    "codegen_sec",
    "integrate_cold_sec",
)
# bngsim-side phases summed into ``bngsim_sec`` (whichever the runner records).
# A stochastic run's simulation (bng_stoch_run.py) is ``ensemble_sec``, the runs
# of the base ensemble's ``n_rep`` replicates, plus on bng_nf ``setup_sec``, each
# replicate's ``NfsimSession.initialize``, which rebuilds the NFsim system: 452 s
# of bng_nf's 4126 s, and most of the time on some models. The runner records
# only that setup's mean, ``setup_per_rep_sec``, so ``setup_sec`` is it times
# ``n_rep``.
# Without the two, bng_nf summed only the session's construction, 0.012 s beside
# 4004.5 s of NFsim, and bng_ssa its parse and load, so a 2x slower ensemble
# could not alert (issue #901). The other per-replicate statistics
# (``rep_*_sec``) are views of ``ensemble_sec`` and are not summed. The
# replicates a borderline verdict adds at rungs 30 and 90 are not timed, so a
# case sums the same 10-replicate workload whether or not it escalated that
# night. An ODE row records neither.
BNGSIM_PHASES = LEGACY_BNGSIM_PHASES + ("ensemble_sec", "setup_sec")

NOTE_LIMIT = 200


def _note(text) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= NOTE_LIMIT else text[: NOTE_LIMIT - 3] + "..."


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def _core_work(row: dict) -> dict | None:
    bn = (row.get("timing") or {}).get("bngsim") or {}
    work = bn.get("work")
    if not isinstance(work, dict):
        return None
    out = {k: int(work[k]) for k in WORK_KEYS if isinstance(work.get(k), (int, float))}
    return out or None


def _core_phase_secs(row: dict) -> dict[str, float]:
    """The BNGSIM_PHASES a ``_core`` row records a time for, in seconds."""
    bn = (row.get("timing") or {}).get("bngsim") or {}
    out = {k: float(bn[k]) for k in BNGSIM_PHASES if isinstance(bn.get(k), (int, float))}
    per_rep, n_rep = bn.get("setup_per_rep_sec"), bn.get("n_rep")
    if "setup_sec" not in out and isinstance(per_rep, (int, float)) and isinstance(n_rep, int):
        out["setup_sec"] = float(per_rep) * n_rep
    return out


def _core_bngsim_sec(row: dict) -> float | None:
    secs = _core_phase_secs(row)
    return round(sum(secs.values()), 6) if secs else None


def _core_phases_seen(report: dict) -> set[str]:
    """The BNGSIM_PHASES some row of a ``_core`` report records a time for."""
    return {k for row in report.get("results", []) for k in _core_phase_secs(row)}


def _core_backend(row: dict) -> str | None:
    """The RHS backend bngsim actually ran ("exprtk" / "cc" / "mir").

    A multi-segment protocol replay records it as ``backend`` (its ``config.codegen``
    names the replay mode); a single run records it as ``config.codegen``.
    """
    bn = (row.get("timing") or {}).get("bngsim") or {}
    if bn.get("backend"):
        return bn["backend"]
    cfg = bn.get("config") or {}
    return cfg.get("codegen")


def extract_core(report: dict) -> dict[str, dict]:
    """``_core`` report (``{"_meta":..., "results": [...]}``) -> cases."""
    cases: dict[str, dict] = {}
    for row in report.get("results", []):
        key = f"{row['model_id']}|{row.get('method', '')}"
        if key in cases:
            raise ValueError(f"duplicate case {key!r} in report")
        rec: dict = {"outcome": row["outcome"], "subclass": row.get("subclass")}
        if row["outcome"] not in GOOD["core"]:
            rec["note"] = _note(row.get("exception") or row.get("comment"))
        work = _core_work(row)
        if work:
            rec["work"] = work
        sec = _core_bngsim_sec(row)
        if sec is not None:
            rec["bngsim_sec"] = sec
        backend = _core_backend(row)
        if backend:
            rec["backend"] = backend
        cases[key] = rec
    return cases


# Where the runner found the suite, up to and including ``cases/semantic/``.
_SUITE_PREFIX = re.compile(r"[^\s'\"]*?[/\\]cases[/\\]semantic[/\\]")


def extract_sbml_suite(payload: dict) -> dict[str, dict]:
    """``benchmarks/suites/sbml_test_suite/run.py`` payload -> cases.

    Out-of-scope cases (``skipped``: non-TimeCourse, missing files) are dropped:
    they never ran on either side, so they carry no signal.

    A message that names a case file by its absolute path differs per machine
    (the runner's bngsim errors name the file alone since GH #700; an older
    report's load failures and a harness_error's text do not), so the note keeps
    only the part under ``cases/semantic``.
    """
    cases: dict[str, dict] = {}
    for row in payload.get("cases", []):
        status = row.get("status", "unknown")
        if status == "skipped":
            continue
        rec: dict = {"outcome": status, "subclass": None}
        if status not in GOOD["sbml-suite"]:
            rec["note"] = _note(_SUITE_PREFIX.sub("", str(row.get("error") or "")))
        cases[str(row["case"])] = rec
    return cases


def extract_dsmts(payload: dict) -> dict[str, dict]:
    """``harness/run_dsmts.py`` payload (``cases`` keyed by id) -> cases."""
    cases: dict[str, dict] = {}
    for cid, row in sorted((payload.get("cases") or {}).items()):
        status = row.get("status", "unknown")
        rec: dict = {"outcome": status, "subclass": None}
        if status not in GOOD["dsmts"]:
            rec["note"] = _note(row.get("error") or row.get("reason"))
        cases[str(cid)] = rec
    return cases


EXTRACTORS = {"core": extract_core, "sbml-suite": extract_sbml_suite, "dsmts": extract_dsmts}


def _source_meta(kind: str, payload: dict) -> dict:
    if kind == "core":
        meta = payload.get("_meta") or {}
        keep = ("suite", "regime", "git_rev", "versions", "generated", "elapsed_sec", "config")
        out = {k: meta[k] for k in keep if k in meta}
        cpu = (meta.get("hardware") or {}).get("cpu")
        if cpu:
            out["cpu"] = cpu  # wall time is only comparable on the same CPU model
        return out
    if kind == "sbml-suite":
        return {k: payload[k] for k in ("n_cases", "n_in_scope", "engines") if k in payload}
    return {
        "summary": {k: v for k, v in (payload.get("summary") or {}).items() if k != "by_test_tag"}
    }


def tally(cases: dict[str, dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in cases.values():
        out[rec["outcome"]] = out.get(rec["outcome"], 0) + 1
    return dict(sorted(out.items()))


def extract(kind: str, report_paths, suite: str, provenance: dict | None = None) -> dict:
    """One or more reports of one suite -> a verdict file.

    Several reports are merged case by case (a sweep split into passes, e.g. the
    giant models run on their own); a case in two of them is refused.
    """
    if isinstance(report_paths, (str, Path)):
        report_paths = [report_paths]
    cases: dict[str, dict] = {}
    metas = []
    phases_seen: set[str] = set()
    for path in report_paths:
        payload = json.loads(Path(path).read_text())
        part = EXTRACTORS[kind](payload)
        if kind == "core":
            phases_seen |= _core_phases_seen(payload)
        dup = sorted(set(part) & set(cases))
        if dup:
            raise ValueError(f"case(s) in more than one report: {dup[:5]}")
        cases.update(part)
        metas.append(_source_meta(kind, payload))
    source = dict(metas[0]) if metas else {}
    if len(metas) > 1:
        source["parts"] = metas
    out = {
        "schema": SCHEMA,
        "suite": suite,
        "kind": kind,
        "source": {**source, **(provenance or {})},
        "n_cases": len(cases),
        "tally": tally(cases),
        "cases": dict(sorted(cases.items())),
    }
    if kind == "core":
        # Which phases each case's bngsim_sec sums, and which of them a case
        # recorded, so that a diff compares two files' wall time only where each
        # counted what the other timed (issue #901).
        out["bngsim_phases"] = list(BNGSIM_PHASES)
        out["bngsim_phases_seen"] = sorted(phases_seen)
    return out


def _wall_comparable(base: dict, fresh: dict) -> bool:
    """Whether two verdict files' bngsim_sec measure the same thing: every phase
    either recorded a time for is one the other summed. A file that does not say
    summed LEGACY_BNGSIM_PHASES, and is taken to have timed all of them. So an ODE
    leg, which records no ensemble, compares across #901, and the first night
    that counts a stochastic ensemble does not compare it with a load phase."""

    def counted(obj: dict) -> set[str]:
        return set(obj.get("bngsim_phases") or LEGACY_BNGSIM_PHASES)

    def seen(obj: dict) -> set[str]:
        recorded = obj.get("bngsim_phases_seen")
        return set(recorded) if recorded is not None else counted(obj)

    return seen(base) <= counted(fresh) and seen(fresh) <= counted(base)


def dump_verdicts(obj: dict, path: Path) -> None:
    """Write a verdict file with one case per line, so a re-baseline diff reads per model."""
    head = {k: v for k, v in obj.items() if k != "cases"}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, sort_keys=True)},")
    lines.append('  "cases": {')
    items = list(obj["cases"].items())
    for i, (cid, rec) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"    {json.dumps(cid)}: {json.dumps(rec, sort_keys=True)}{comma}")
    lines.append("  }")
    lines.append("}")
    Path(path).write_text("\n".join(lines) + "\n")


def load_verdicts(path: Path) -> dict:
    obj = json.loads(Path(path).read_text())
    if obj.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported verdict schema {obj.get('schema')!r}")
    if obj.get("kind") not in KINDS:
        raise ValueError(f"{path}: unknown kind {obj.get('kind')!r}")
    return obj


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def rebaselined(then: dict, now: dict) -> dict[str, dict]:
    """The cases the committed baseline re-recorded between two of its versions:
    case id -> its record in ``now``, where ``then`` held another verdict or none."""
    old = then["cases"]
    return {
        cid: rec
        for cid, rec in now["cases"].items()
        if cid not in old or _verdict(old[cid]) != _verdict(rec)
    }


def _verdict(rec: dict) -> tuple:
    return rec["outcome"], rec.get("subclass")


def _label(rec: dict | None) -> str:
    if rec is None:
        return "MISSING"
    sub = rec.get("subclass")
    return f"{rec['outcome']} ({sub})" if sub else rec["outcome"]


def diff(
    base: dict,
    fresh: dict,
    *,
    work_ratio: float = 1.25,
    work_total_ratio: float = 1.05,
    wall_ratio: float = 2.0,
    wall_ratio_any_cpu: float = 4.0,
    wall_min_delta: float = WALL_MIN_DELTA_SEC,
    min_fraction: float = 0.9,
    expect_backend: str | None = None,
    flaky: dict[str, str] | None = None,
    timeout_flips_info: bool = False,
    phases_may_differ: bool = False,
    rebaselined: dict[str, dict] | None = None,
) -> dict:
    """Compare two verdict files of the same kind. Returns a JSON-able result.

    ``flaky`` maps case ids whose verdict or work measurably depends on the runner
    to the reason; any change of theirs is reported, never alerted: an outcome
    change under ``changed``, a per-case work increase under ``efficiency`` with
    the reason beside it (GH #1153). Their work still counts in the totals.
    ``timeout_flips_info`` does the same for every transition into or out of
    TIMEOUT: right for comparing two single nights, which may land on runner CPUs
    of different speed, and wrong for the committed baseline, where PASS ->
    TIMEOUT is the efficiency regression the check exists to catch.

    Two files whose ``bngsim_sec`` sum different phases are not compared on wall
    time. Against the committed baseline that alerts, since the check would stay
    off until someone re-extracted it; ``phases_may_differ`` only reports it,
    which is right for two single nights across a change to BNGSIM_PHASES.

    ``rebaselined`` maps the cases the committed baseline re-recorded since
    ``base`` was run (:func:`rebaselined`) to their new records. Between two
    nights, a change into the verdict the baseline now records was reviewed when
    it was recorded, so it is reported, not alerted: a pull request that makes a
    check stricter re-records the rows it moves, and they would otherwise alert
    on its own run and on the first night after it merges. A fix that regresses
    before anyone re-baselined it is not among them, and still alerts.
    """
    flaky = flaky or {}
    rebaselined = rebaselined or {}
    if base["kind"] != fresh["kind"]:
        raise ValueError(f"kind mismatch: baseline {base['kind']} vs fresh {fresh['kind']}")
    kind = base["kind"]
    good, crash = GOOD[kind], CRASH[kind]
    b, f = base["cases"], fresh["cases"]

    regressed, new_crash, improved, changed, new, dropped = [], [], [], [], [], []
    for cid in sorted(set(b) | set(f)):
        br, fr = b.get(cid), f.get(cid)
        if br is None:
            new.append({"case": cid, "after": _label(fr), "note": fr.get("note", "")})
            continue
        bo = br["outcome"]
        if fr is None:
            item = {"case": cid, "before": _label(br), "after": "MISSING", "note": ""}
            (regressed if bo in good else dropped).append(item)
            continue
        fo = fr["outcome"]
        item = {"case": cid, "before": _label(br), "after": _label(fr), "note": fr.get("note", "")}
        soft = None
        if cid in flaky:
            soft = f"flaky: {flaky[cid]}"
        elif cid in rebaselined and _verdict(rebaselined[cid]) == _verdict(fr):
            soft = "re-baselined since that night"
        elif timeout_flips_info and "TIMEOUT" in (bo, fo) and bo != fo:
            soft = "timeout flip between two nights (runner speed)"
        if soft and (bo, br.get("subclass")) != (fo, fr.get("subclass")):
            changed.append({**item, "note": f"[{soft}] {item['note']}".strip()})
        elif soft:
            pass
        elif bo in good and fo not in good:
            regressed.append(item)
        elif bo not in good and fo in good:
            improved.append(item)
        elif bo not in good and fo in crash and bo != fo:
            new_crash.append(item)
        elif (bo, br.get("subclass")) != (fo, fr.get("subclass")):
            changed.append(item)

    # Efficiency: only where both runs are good and both carry counters.
    both_good = [c for c in b if c in f and b[c]["outcome"] in good and f[c]["outcome"] in good]
    work_cases = [c for c in both_good if b[c].get("work") and f[c].get("work")]
    work_items = []
    totals = {}
    for key in WORK_ALERT_KEYS:
        tb = tf = 0
        for c in work_cases:
            vb, vf = b[c]["work"].get(key), f[c]["work"].get(key)
            if vb is None or vf is None:
                continue
            tb += vb
            tf += vf
            if vf > vb * work_ratio and vf - vb >= WORK_MIN_DELTA[key]:
                item = {
                    "case": c,
                    "counter": key,
                    "before": vb,
                    "after": vf,
                    "ratio": round(vf / vb, 3) if vb else None,
                }
                if c in flaky:
                    item["flaky"] = True
                    item["note"] = f"flaky: {flaky[c]}"
                work_items.append(item)
        totals[key] = {
            "before": tb,
            "after": tf,
            "ratio": round(tf / tb, 4) if tb else None,
            "alert": bool(tb) and tf > tb * work_total_ratio,
        }
    work_items.sort(key=lambda d: (-(d["ratio"] or 0), d["case"], d["counter"]))

    wall_cases = [
        c
        for c in both_good
        if b[c].get("bngsim_sec") is not None and f[c].get("bngsim_sec") is not None
    ]
    wb = sum(b[c]["bngsim_sec"] for c in wall_cases)
    wf = sum(f[c]["bngsim_sec"] for c in wall_cases)
    # Hosted runners come in several CPU models, and compiled code in particular
    # runs ~1.8x slower on the older ones (measured: the codegen arm on an EPYC
    # 7763 vs an EPYC 9V45), so a wall ratio across CPU models mostly measures the
    # hardware. It alerts at wall_ratio only on the same CPU model, and across
    # models only past wall_ratio_any_cpu. The work counters are the precise signal.
    cpu_b, cpu_f = base.get("source", {}).get("cpu"), fresh.get("source", {}).get("cpu")
    same_cpu = bool(cpu_b) and cpu_b == cpu_f
    limit = wall_ratio if same_cpu else wall_ratio_any_cpu
    # Two files whose bngsim_sec sums different phases measure different things:
    # the first night after #901 compared bng_ssa's load phase with its ensemble.
    same_phases = _wall_comparable(base, fresh)
    phases_alert = bool(wall_cases) and not same_phases and not phases_may_differ
    wall = {
        "n_cases": len(wall_cases),
        "before": round(wb, 3),
        "after": round(wf, 3),
        "ratio": round(wf / wb, 3) if wb else None,
        "cpu_baseline": cpu_b,
        "cpu_fresh": cpu_f,
        "same_cpu": same_cpu,
        "limit": limit,
        "min_delta": wall_min_delta,
        "same_phases": same_phases,
        "phases_alert": phases_alert,
        "alert": same_phases and wb > 0 and wf > wb * limit and wf - wb > wall_min_delta,
    }

    # A run meant to exercise a specific backend must have run it: exprtk for the
    # interpreter arm, cc for the compiled one.
    backend_miss = []
    if expect_backend:
        for c, rec in sorted(f.items()):
            be = rec.get("backend")
            if be is not None and be != expect_backend and rec["outcome"] in good:
                backend_miss.append({"case": c, "after": be, "note": f"expected {expect_backend}"})

    n_base, n_fresh = len(b), len(f)
    incomplete = n_fresh == 0 or (n_base > 0 and n_fresh < min_fraction * n_base)

    alerts = {
        "regressed": len(regressed),
        "new_crash": len(new_crash),
        "efficiency_cases": sum(1 for x in work_items if not x.get("flaky")),
        "efficiency_totals": sum(1 for t in totals.values() if t["alert"]),
        "wall": int(wall["alert"]),
        "wall_phases": int(phases_alert),
        "backend": len(backend_miss),
        "incomplete": int(incomplete),
    }
    return {
        "suite": fresh.get("suite") or base.get("suite"),
        "kind": kind,
        "n_baseline": n_base,
        "n_fresh": n_fresh,
        "tally_baseline": base.get("tally") or tally(b),
        "tally_fresh": fresh.get("tally") or tally(f),
        "alerts": alerts,
        "n_alerts": sum(alerts.values()),
        "regressed": regressed,
        "new_crash": new_crash,
        "efficiency": work_items,
        "work_totals": totals,
        "wall": wall,
        "backend_miss": backend_miss,
        "improved": improved,
        "changed": changed,
        "new": new,
        "dropped": dropped,
        "thresholds": {
            "work_ratio": work_ratio,
            "work_total_ratio": work_total_ratio,
            "wall_ratio": wall_ratio,
            "wall_ratio_any_cpu": wall_ratio_any_cpu,
            "wall_min_delta": wall_min_delta,
            "min_fraction": min_fraction,
            "work_min_delta": WORK_MIN_DELTA,
        },
    }


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
# What a pull request must fix, then what it must record in the baseline.
GATE_FIX = (
    ("harness_error", "The harness could not score these cases"),
    ("regressed", "Regressed: good in the baseline, not now"),
    ("now_wrong", "Refused in the baseline, a wrong answer now"),
)
GATE_RECORD = (
    ("improved", "Improved: failing in the baseline, good now"),
    ("changed", "Changed between two failing outcomes"),
    ("new", "New cases (not in the baseline)"),
    ("dropped", "Failing cases no longer in the run"),
)


def gate(base: dict, fresh: dict) -> dict:
    """Hold a fresh run to the baseline case for case (the per-PR gate, GH #700).

    ``diff`` decides what a nightly should wake someone for, so it lets
    improvements, churn between failing states and new cases through. A gate on
    a pull request can be exact instead: the run is deterministic and takes
    seconds, so ANY change of a case's outcome (or subclass) fails it. Each
    change is classified by what the pull request has to do about it:

    * fix it (``GATE_FIX``): a new ``harness_error``, a good case that is no
      longer good (or missing), or a refusal that became a wrong answer;
    * record it in the baseline in the same pull request (``GATE_RECORD``): an
      improvement, a change between two failing outcomes, a new case, or a
      failing case that left the run.

    Notes (the failure text) are not compared, so rewording a refusal does not
    fail the gate. A run with no cases fails it whatever the baseline holds.
    """
    if base["kind"] != fresh["kind"]:
        raise ValueError(f"kind mismatch: baseline {base['kind']} vs fresh {fresh['kind']}")
    kind = base["kind"]
    good, wrong = GOOD[kind], WRONG_ANSWER.get(kind, set())
    b, f = base["cases"], fresh["cases"]
    out: dict[str, list] = {name: [] for name, _ in GATE_FIX + GATE_RECORD}
    for cid in sorted(set(b) | set(f)):
        br, fr = b.get(cid), f.get(cid)
        item = {
            "case": cid,
            "before": _label(br),
            "after": _label(fr),
            "note": (fr or {}).get("note", ""),
        }
        if br is None:
            # A case the harness cannot score is to fix even when it is new: a
            # baseline that recorded it would match it on every later run.
            out["harness_error" if fr["outcome"] == "harness_error" else "new"].append(item)
            continue
        if fr is None:
            out["regressed" if br["outcome"] in good else "dropped"].append(item)
            continue
        bo, fo = br["outcome"], fr["outcome"]
        if (bo, br.get("subclass")) == (fo, fr.get("subclass")):
            continue
        if fo == "harness_error":
            out["harness_error"].append(item)
        elif bo in good and fo not in good:
            out["regressed"].append(item)
        elif bo not in good and fo in good:
            out["improved"].append(item)
        elif fo in wrong and bo not in wrong:
            out["now_wrong"].append(item)
        else:
            out["changed"].append(item)
    n_fix = sum(len(out[name]) for name, _ in GATE_FIX)
    n_record = sum(len(out[name]) for name, _ in GATE_RECORD)
    return {
        "suite": fresh.get("suite") or base.get("suite"),
        "kind": kind,
        "n_baseline": len(b),
        "n_fresh": len(f),
        "tally_baseline": base.get("tally") or tally(b),
        "tally_fresh": fresh.get("tally") or tally(f),
        "empty": not f,
        "n_fix": n_fix,
        "n_record": n_record,
        "ok": bool(f) and n_fix == 0 and n_record == 0,
        **out,
    }


def render_gate_md(result: dict, limit: int = 60) -> str:
    """The gate's verdict as Markdown: the tallies, then every change by class."""
    verdict = (
        "matches the baseline"
        if result["ok"]
        else "**no cases ran**"
        if result["empty"]
        else f"**{result['n_fix']} to fix, {result['n_record']} to record**"
    )
    lines = [
        f"### {result['suite']} — per-PR gate",
        "",
        f"{verdict} · {result['n_fresh']} cases run, {result['n_baseline']} in the baseline",
        "",
        "| outcome | baseline | this run |",
        "|---|---|---|",
    ]
    for o in sorted(set(result["tally_baseline"]) | set(result["tally_fresh"])):
        lines.append(
            f"| {o} | {result['tally_baseline'].get(o, 0)} | {result['tally_fresh'].get(o, 0)} |"
        )
    lines.append("")
    for name, title in GATE_FIX + GATE_RECORD:
        rows = result[name]
        if rows:
            lines += [f"**{title}: {len(rows)}**", ""]
            lines += _table(rows, ["case", "before", "after", "note"], limit) + [""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _table(rows: list[dict], cols: list[str], limit: int) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows[:limit]:
        cells = [str(r.get(c, "")).replace("|", "\\|") for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        out.append(
            f"| ... {len(rows) - limit} more (see the run's artifacts) |" + " |" * (len(cols) - 1)
        )
    return out


def render_md(result: dict, label: str = "", limit: int = 40) -> str:
    s = result["suite"]
    n = result["n_alerts"]
    head = f"### {s}" + (f" — {label}" if label else "")
    status = f"**{n} alert(s)**" if n else "no regressions"
    lines = [
        head,
        "",
        f"{status} · {result['n_fresh']} cases tonight, {result['n_baseline']} in baseline",
        "",
    ]

    outcomes = sorted(set(result["tally_baseline"]) | set(result["tally_fresh"]))
    lines.append("| outcome | baseline | tonight |")
    lines.append("|---|---|---|")
    for o in outcomes:
        lines.append(
            f"| {o} | {result['tally_baseline'].get(o, 0)} | {result['tally_fresh'].get(o, 0)} |"
        )
    lines.append("")

    if result["alerts"]["incomplete"]:
        lines += [
            f"**Incomplete run:** {result['n_fresh']} cases against {result['n_baseline']} in the "
            "baseline. The comparison covers too little to call green.",
            "",
        ]
    if result["regressed"]:
        lines += [f"**Regressed (good before, not now): {len(result['regressed'])}**", ""]
        lines += _table(result["regressed"], ["case", "before", "after", "note"], limit) + [""]
    if result["new_crash"]:
        lines += [f"**New crashes: {len(result['new_crash'])}**", ""]
        lines += _table(result["new_crash"], ["case", "before", "after", "note"], limit) + [""]
    if result["backend_miss"]:
        lines += [f"**Did not run the expected backend: {len(result['backend_miss'])}**", ""]
        lines += _table(result["backend_miss"], ["case", "after", "note"], limit) + [""]
    if result["efficiency"]:
        n_flaky = sum(1 for x in result["efficiency"] if x.get("flaky"))
        n_alert = len(result["efficiency"]) - n_flaky
        head = f"{n_alert} counter increase(s) above x{result['thresholds']['work_ratio']}"
        if n_flaky:
            head += f", and {n_flaky} on flaky cases, reported and not alerted"
        lines += [f"**Efficiency: {head}**" if n_alert else f"Efficiency: {head}", ""]
        cols = ["case", "counter", "before", "after", "ratio"]
        if any("note" in x for x in result["efficiency"]):
            cols.append("note")
        lines += _table(result["efficiency"], cols, limit)
        lines.append("")
    tot = result["work_totals"]
    if any(t["before"] for t in tot.values()):
        lines.append("Total bngsim work over cases good in both runs:")
        lines.append("")
        lines.append("| counter | baseline | tonight | ratio | alert |")
        lines.append("|---|---|---|---|---|")
        for k, t in tot.items():
            lines.append(
                f"| {k} | {t['before']} | {t['after']} | {t['ratio']} | {'YES' if t['alert'] else ''} |"
            )
        lines.append("")
    w = result["wall"]
    if w["n_cases"] and not w.get("same_phases", True):
        tail = (
            " — **ALERT**: re-extract the baseline, so that the wall check runs again"
            if w.get("phases_alert")
            else ", and compare again once both are extracted with the same ones"
        )
        lines += [
            f"bngsim wall over {w['n_cases']} cases: {w['before']} s -> {w['after']} s, "
            f"not compared: the two files time different phases (issue #901){tail}.",
            "",
        ]
    elif w["n_cases"]:
        flag = " — **ALERT**" if w["alert"] else ""
        where = (
            f"same CPU ({w.get('cpu_fresh')})"
            if w.get("same_cpu")
            else f"CPU {w.get('cpu_baseline')} -> {w.get('cpu_fresh')}"
        )
        lines += [
            f"bngsim wall over {w['n_cases']} cases: {w['before']} s -> {w['after']} s "
            f"(x{w['ratio']}; {where}; alerts above x{w.get('limit')} and "
            f"+{w.get('min_delta')} s){flag}",
            "",
        ]
    for name, title in (
        ("improved", "Improved (failing before, good now) — re-baseline to lock these in"),
        ("changed", "Changed between failing states"),
        ("new", "New cases (not in baseline)"),
        ("dropped", "Failing cases no longer in the run"),
    ):
        rows = result[name]
        if rows:
            lines += [f"<details><summary>{title}: {len(rows)}</summary>", ""]
            cols = (
                ["case", "after", "note"] if name == "new" else ["case", "before", "after", "note"]
            )
            lines += _table(rows, cols, limit) + ["", "</details>", ""]
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_extract(args) -> int:
    prov = json.loads(args.provenance) if args.provenance else None
    obj = extract(args.kind, [Path(r) for r in args.report], args.suite, prov)
    dump_verdicts(obj, Path(args.out))
    print(f"{args.suite}: {obj['n_cases']} cases {obj['tally']} -> {args.out}")
    return 0


def _rebaselined_from(then: str, now: str) -> dict[str, dict]:
    """:func:`rebaselined` between two committed baseline files; nothing when
    either is not given or not there (the run it describes predates the suite)."""
    if not (then and now and Path(then).exists() and Path(now).exists()):
        return {}
    return rebaselined(load_verdicts(Path(then)), load_verdicts(Path(now)))


def _cmd_diff(args) -> int:
    fresh = load_verdicts(Path(args.fresh))
    base_path = Path(args.baseline)
    if not base_path.exists():
        msg = (
            f"### {fresh['suite']}" + (f" — {args.label}" if args.label else "") + "\n\n"
            f"No baseline at `{args.baseline}`: nothing to compare against. "
            f"Tonight: {fresh['n_cases']} cases, {fresh['tally']}.\n"
        )
        if args.md_out:
            Path(args.md_out).write_text(msg)
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(
                    {
                        "suite": fresh["suite"],
                        "label": args.label,
                        "no_baseline": True,
                        "n_alerts": 0,
                    }
                )
            )
        print(msg)
        return 0 if args.allow_missing_baseline else 2
    base = load_verdicts(base_path)
    flaky = {}
    if args.flaky and Path(args.flaky).exists():
        flaky = json.loads(Path(args.flaky).read_text()).get(fresh["suite"], {})
    result = diff(
        base,
        fresh,
        work_ratio=args.work_ratio,
        work_total_ratio=args.work_total_ratio,
        wall_ratio=args.wall_ratio,
        wall_ratio_any_cpu=args.wall_ratio_any_cpu,
        wall_min_delta=args.wall_min_delta,
        min_fraction=args.min_fraction,
        expect_backend=args.expect_backend or None,
        flaky=flaky,
        timeout_flips_info=args.timeout_flips_info,
        phases_may_differ=args.phases_may_differ,
        rebaselined=_rebaselined_from(args.baseline_then, args.baseline_now),
    )
    result["label"] = args.label
    md = render_md(result, args.label)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=1))
    if args.md_out:
        Path(args.md_out).write_text(md)
    print(md)
    return 1 if result["n_alerts"] else 0


def _cmd_gate(args) -> int:
    result = gate(load_verdicts(Path(args.baseline)), load_verdicts(Path(args.fresh)))
    md = render_gate_md(result)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=1))
    if args.md_out:
        Path(args.md_out).write_text(md)
    print(md)
    return 0 if result["ok"] else 1


BODY_LIMIT = 60_000  # a GitHub issue comment caps at 65,536 characters


def _cmd_summarize(args) -> int:
    """Diff JSONs (+ the workflow's job results) -> the tracking-issue body."""
    needs = json.loads(args.needs_json) if args.needs_json else {}
    failed_jobs = {
        j: v.get("result")
        for j, v in needs.items()
        if v.get("result") not in ("success", "skipped")
    }
    parts, n = [], 0
    for p in args.diffs:
        try:
            r = json.loads(Path(p).read_text())
        except (OSError, ValueError) as exc:
            parts.append(f"### {p}\n\nCould not read diff: {exc}\n")
            n += 1
            continue
        if r.get("no_baseline"):
            continue
        if r.get("no_report"):
            parts.append(f"### {r.get('suite')}\n\n**No report**: the run did not finish.\n")
            n += 1
            continue
        n += r.get("n_alerts", 0)
        if r.get("n_alerts"):
            parts.append(render_md(r, r.get("label", "")))

    head = []
    if args.run_url:
        sha = f" on `{args.sha[:12]}`" if args.sha else ""
        head.append(f"Nightly parity run {args.run_url}{sha}.")
        head.append("")
    if needs:
        head += ["| job | result |", "|---|---|"]
        head += [f"| {j} | {v.get('result')} |" for j, v in sorted(needs.items())]
        head.append("")
    if failed_jobs:
        head.append(
            "A job that did not succeed either found a regression (below) or could not run; "
            "a job that could not run means that suite was not checked tonight."
        )
        head.append("")
    body = "\n".join(head) + ("\n".join(parts) if parts else "No regressions.\n")
    if len(body) > BODY_LIMIT:
        body = body[:BODY_LIMIT] + "\n\n... truncated; the full reports are the run's artifacts.\n"
    if args.out:
        Path(args.out).write_text(body)
    print(body)
    return 1 if (n or failed_jobs) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="report -> normalized verdict file")
    e.add_argument("--kind", choices=KINDS, required=True)
    e.add_argument("--suite", required=True, help="suite name recorded in the file (e.g. rr_ode)")
    e.add_argument("--report", required=True, action="append", help="repeat to merge passes")
    e.add_argument("--out", required=True)
    e.add_argument("--provenance", default="", help="JSON object merged into 'source'")
    e.set_defaults(fn=_cmd_extract)

    d = sub.add_parser("diff", help="fresh verdicts vs a baseline; exit 1 on any alert")
    d.add_argument("--baseline", required=True)
    d.add_argument("--fresh", required=True)
    d.add_argument("--label", default="")
    d.add_argument("--json-out", default="")
    d.add_argument("--md-out", default="")
    d.add_argument("--work-ratio", type=float, default=1.25)
    d.add_argument("--work-total-ratio", type=float, default=1.05)
    d.add_argument("--wall-ratio", type=float, default=2.0, help="wall alert, same CPU model")
    d.add_argument(
        "--wall-ratio-any-cpu", type=float, default=4.0, help="wall alert, across CPU models"
    )
    d.add_argument(
        "--wall-min-delta",
        type=float,
        default=WALL_MIN_DELTA_SEC,
        help="wall alert: seconds the total must also grow by",
    )
    d.add_argument("--min-fraction", type=float, default=0.9)
    d.add_argument(
        "--expect-backend",
        default="",
        help="the backend every good row must have run, e.g. exprtk (interpreter arm) "
        "or cc (compiled arm)",
    )
    d.add_argument(
        "--flaky",
        default="",
        help="JSON file {suite: {case: reason}}: changes of these cases, in outcome "
        "or in work, never alert",
    )
    d.add_argument(
        "--timeout-flips-info",
        action="store_true",
        help="report (do not alert on) transitions into or out of TIMEOUT",
    )
    d.add_argument(
        "--phases-may-differ",
        action="store_true",
        help="report (do not alert on) the two files' bngsim_sec summing different phases",
    )
    d.add_argument(
        "--baseline-then",
        default="",
        help="the committed baseline as of the --baseline run's commit; with "
        "--baseline-now, a change into a verdict the baseline re-recorded since "
        "then is reported, not alerted",
    )
    d.add_argument("--baseline-now", default="", help="the committed baseline now")
    d.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="exit 0 (not 2) when the baseline file does not exist",
    )
    d.set_defaults(fn=_cmd_diff)

    s = sub.add_parser("summarize", help="diff JSONs -> one Markdown body; exit 1 if any alert")
    s.add_argument("diffs", nargs="*")
    s.add_argument("--out", default="")
    s.add_argument("--needs-json", default="", help="the workflow's toJSON(needs)")
    s.add_argument("--run-url", default="")
    s.add_argument("--sha", default="")
    s.set_defaults(fn=_cmd_summarize)

    g = sub.add_parser("gate", help="fresh verdicts vs a baseline; exit 1 on any outcome change")
    g.add_argument("--baseline", required=True)
    g.add_argument("--fresh", required=True)
    g.add_argument("--json-out", default="")
    g.add_argument("--md-out", default="")
    g.set_defaults(fn=_cmd_gate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
