#!/usr/bin/env python3
"""Sweep the BNGL corpus through bionetgen.run() with an explicit simulator.

For each .bngl, classifies the model as deterministic (only ode/cvode actions)
or stochastic (any nf/ssa/pla/psa action). Deterministic models are run once
per side. Stochastic models are patched to inject seeds 1..N (default N=10) and
run once per seed per side, into per-seed output directories so the diff
script can compute ensemble means and stds.

Designed to be called twice: once with --simulator subprocess, once with
--simulator bngsim, into different --out roots. The diff script then
ensemble-compares the two roots.

Inherits the gotcha-fixes from seeded_sweep.py:
  * cwd=tempfile.gettempdir() so the venv install isn't shadowed by a
    PyBioNetGen source dir.
  * regex `\\g<1>` not `\\1` for backref-then-digits substitution.
  * TEND_OVERRIDES (shorter t_end) and TIMEOUT_OVERRIDES (longer budget)
    for slow models.
  * Spaces-in-filenames safe (Path-based, no bash globbing).
  * Doesn't depend on bionetgen.bngmodel() — uses regex over raw text so
    parser-rejecting BNGL like prion2_YTLedits.bngl can still be classified.
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import read_manifest` resolves
sys.path.insert(0, str(HERE))  # so the sibling `bngsim_backend` resolves

try:
    import psutil
except ImportError:  # memory governor degrades to a no-op without psutil
    psutil = None

OUTPUT_EXTENSIONS = {".gdat", ".cdat", ".net", ".scan", ".xml", ".species"}
DETERMINISTIC_METHODS = {"ode", "cvode"}

# --- resource governor ------------------------------------------------------
# Each "worker" here is not one process: run_one shells out to
# `python -c bionetgen.run(...)`, which itself spawns Perl BNG2.pl ->
# run_network / NFsim. So 8 workers can mean 24+ live processes, and a single
# NFsim model can hold multiple GB. On a 16 GB / 6-core box that overruns RAM;
# macOS then leans on dynamic swap, and if the disk is near-full swap can't
# grow, so the kernel jetsams (WindowServer included) and the machine freezes.
# These guards bound concurrency by MEMORY, not just by the --workers number,
# and refuse to start a sweep that would leave no swap headroom.
MEM_PER_WORKER_GB = 2.5  # planning estimate of peak RSS per worker subtree
MEM_FLOOR_GB = 3.0  # don't launch a new sim while free RAM is below this
DISK_ABORT_GB = 8.0  # refuse to start below this much free disk (swap room)
DISK_WARN_GB = 20.0  # warn below this


def _available_gb():
    """Available (not just free) system RAM in GiB, or None without psutil."""
    if psutil is None:
        return None
    return psutil.virtual_memory().available / 2**30


def _await_memory(floor_gb, max_wait_s=180.0):
    """Block until available RAM >= floor_gb, or max_wait_s elapses.

    Called inside each worker right before it launches its heavy subprocess, so
    that under memory pressure new sims queue instead of piling on. No deadlock
    risk: if every worker is parked here, none are consuming, so RAM recovers
    and they proceed. Jitter keeps workers from unblocking in lockstep. Returns
    seconds waited (for the log).
    """
    if not floor_gb or psutil is None:
        return 0.0
    start = time.monotonic()
    while True:
        avail = _available_gb()
        if avail is None or avail >= floor_gb:
            return time.monotonic() - start
        if time.monotonic() - start > max_wait_s:
            return time.monotonic() - start  # degrade rather than hang forever
        time.sleep(0.5 + random.random())


def plan_workers(requested, mem_per_worker_gb):
    """Cap requested workers by core count and an available-RAM budget.

    Returns (effective_workers, reason) where reason explains any reduction.
    """
    cores = os.cpu_count() or 1
    capped = min(requested, cores)
    reason = "" if capped == requested else f"capped to {cores} cores"
    avail = _available_gb()
    if avail is not None and mem_per_worker_gb > 0:
        mem_cap = max(1, int(avail // mem_per_worker_gb))
        if mem_cap < capped:
            reason = f"capped to {mem_cap} by RAM budget ({avail:.1f} GB avail / {mem_per_worker_gb} GB per worker)"
            capped = mem_cap
    return capped, reason


# Per-model t_end CAP (seconds of simulated time), keyed by .bngl filename.
# CAP semantics: every active simulate/scan action whose t_end EXCEEDS the cap
# is reduced to the cap; actions already at or below it are left untouched.
# (This is a change from the older "set every t_end to X" behaviour, which is
# wrong for multi-phase models — capping a 25-unit equilibration to 1 must not
# also *stretch* a 1.5-unit main run. For single-phase models cap == set.)
# Applied identically to BOTH stacks so the parity check stays apples-to-apples;
# both engines integrate the same shortened horizon, so the ensemble still
# agrees. These are runtime fixtures (we only need bngsim and BNG2.pl to match
# on whatever window we ask for), not statements about the models' biology.
TEND_OVERRIDES = {
    "AD 3 State FREE Expanding nfs.bngl": 100,  # default 650 -> 180s timeout
    # B6: bngsim codegen falls back to interpreted ODE RHS for this model
    # ("Starred arguments in lambda not supported"), and the default
    # t_end=730 over 10 NF seeds blows the 180s budget. Cap to keep the
    # parity check exercising both deterministic and stochastic segments
    # within the timeout. The cap is a sweep-side workaround for a
    # documented bngsim codegen limitation; not a model-correctness change.
    "scaling_example.bngl": 50,
    # --- candidate-corpus glacial tier -------------------------------------
    # These 28 models are stochastic (nf/ssa) and time-integration-bound, so
    # wall scales ~linearly with t_end. Single-run wall was 12-65 s; tagged
    # "glacial" only because the DIFF runs 10 seeds (projected = wall x 10 >
    # 2 min). Caps target ~6-10 s/run (~60-100 s DIFF -> "slow" tier). Comment
    # = original t_end(s) and measured single-run wall. Two basenames cover two
    # models each (noted). Values were measured-and-tuned, not just estimated.
    "jobs_tofit_gen48ind13.bngl": 100,  # nf 1000,  ~65s
    "jobs_tofit_gen34ind26.bngl": 100,  # nf 1000,  ~65s
    "jobs_tofit_iter44p17.bngl": 100,  # nf 1000,  ~63s
    "fceri_ji.bngl": 60,  # nf 500,   ~61s
    "example2_fit.bngl": 16,  # nf 126,   ~61s
    "v21.bngl": 8,  # ode+ssa 100, ~56s (ssa-bound); cap 15 still ~14s
    "tcr_iter28p4h2.bngl": 2.0,  # equil 15.6 + main 0.94, ~59s
    "tcr_iter9p44.bngl": 2.0,  # equil 15.6 + main 0.94, ~51s
    "egfr_nf_iter5p12h10.bngl": 9,  # nf 60,    ~51s
    "e6.bngl": 25,  # nf 200,   ~60s
    "e5.bngl": 30,  # nf 200,   ~46s
    "e4.bngl": 40,  # nf 200,   ~35s
    "e3.bngl": 60,  # nf 200,   ~25s
    # e2 NOT capped: it's a noisy model that sits right on the ensemble-test
    # threshold — capped to 80 it marginally DIFFs (4/303 cells), but at full
    # t_end=200 (~20s) it PASSES. Run it full-horizon rather than paper over
    # the noise with extra seed replicates.
    "e1.bngl": 110,  # nf 200,   ~14s
    "egfr_net.bngl": 30,  # nf 120,   ~33s
    "bench_blbr_rings_posner1995.bngl": 900,  # nf 3000,  ~27s
    "PushPull.bngl": 1200,  # nf 4000,  ~25s
    "tcr_gen20ind9.bngl": 1.0,  # rulehub: equil 25.2 + main 1.5 (~61s); rulemonkey: main 1.5 (~12s)
    "bench_blbr_dembo1978_monovalent_inhibitor.bngl": 1200,  # nf 3000,  ~19s
    "receptor_nf_iter91p28.bngl": 60,  # equil 600 + main 60, ~19s
    "example3_fit.bngl": 1800,  # nf 5000; rulehub bench (~17s) + rulemonkey (~21s)
    "tlbr_yang2008.bngl": 1800,  # nf 3000,  ~13s
    "BLBR.bngl": 5,  # nf 10 / n_steps 1000, ~16s
    "example6_ground_truth.bngl": 60,  # equil 600 + main 60, ~15s
    "blbr_heterogeneity_goldstein1980.bngl": 5,  # nf parameter_scan, t_end 10 (also n_scan_pts cut below), ~28s
}

# Per-model n_scan_pts override, keyed by .bngl filename. Reduces the number of
# parameter_scan sample points for scan-bound stochastic models whose cost is
# dominated by the point count, not by t_end. Applied to both stacks; each scan
# point is still ensemble-compared cell-by-cell in the .scan, so fewer points
# is a smaller-but-still-real parity check.
NSCANPTS_OVERRIDES = {
    "blbr_heterogeneity_goldstein1980.bngl": 6,  # was 18
}

# Per-model ACTION INJECTION, keyed by .bngl filename. Some corpus models carry
# no run action (only generate_network, or none), so they emit no comparable
# .gdat/.cdat/.scan and can't be parity-diffed. We append a simulate action at
# sweep time (before stochastic classification, so the injected method decides
# the regime) to turn them into real fixtures, applied identically to BOTH
# stacks. The source .bngl stays pristine; the action lives here, mirroring the
# seed/t_end/tol/rename injections. The appended text is also subject to the
# t_end cap and seed injection below, so a capped horizon still applies.
ACTION_INJECT = {
    # generate_network-only EGFR tutorial: add an ODE run so it produces a
    # .gdat (9 observables). Network gen + ODE integrate in ~2 s on both
    # stacks; the screen's ~25 s was parallel-worker contention, not real cost.
    "Rule_based_egfr_tutorial.bngl": 'simulate({method=>"ode",t_end=>20,n_steps=>100})',
}

# Per-model ODE solver tolerance overrides, keyed by .bngl filename. Some
# models are ill-conditioned initial-value problems whose trajectory is not
# resolved at BNG2.pl's and bngsim's shared default atol/rtol=1e-8: BOTH
# CVODE configurations give a (different) wrong answer there — one merely
# stays bounded while the other can run negative — so the default-tolerance
# DIFF is a property of the IVP, not of either engine. Tightening to 1e-12
# makes the two integrators converge to the same physical solution (verified
# for eco_coevolution_host_parasite: they then agree to ~9 sig figs). We
# inject the tighter tolerance into BOTH stacks so the parity check exercises
# the model at a tolerance where it is actually well-posed. A non-negativity
# clamp is deliberately NOT the fix — some models carry legitimately-negative
# observables. Applies to ode/cvode actions; BNG ignores atol/rtol on
# nf/ssa actions, so injecting unconditionally is harmless.
TOL_OVERRIDES = {
    "eco_coevolution_host_parasite.bngl": {"atol": "1e-12", "rtol": "1e-12"},
}

# SYMBOL_RENAMES (retired). The old sweep-time whole-word rename is gone — its two
# classes of problem are now handled by one mechanism each, so the on-disk model always
# matches what runs (no silent runtime substitution):
#   * typo'd tolerance args (atoll/abstol/reltol) -> patched at import by
#     vendor_corpus.CORPUS_REPAIRS (the vendored file is corrected);
#   * reserved-word names (frac) -> the engine copes under the hood (bngsim #63/#64
#     ODE/NFsim remap; cf. _bng_common._resolve_token for Python keywords). The 3 `frac`
#     models are no longer in the corpus anyway.

# Per-model timeout overrides (seconds), keyed by .bngl filename. The default
# --timeout is a parity budget tuned for fast models. This dict previously
# carried 600 s entries for two 200-point parameter_scans
# (harmonicOscillator, ATG_update_mTORC1_assembly_more_complete_scheme): the
# backend-hook route delegated each scan point as a separate atomic job, so
# 200 points overran the budget. The in-process parameter_scan driver now
# runs both in ~2-3 s, so the overrides are no longer needed.
TIMEOUT_OVERRIDES = {
    # NFsim model, t_end=1000 over 1000 steps: ~82 s/seed on an idle machine
    # but it overran the 180 s parity budget under 2-worker contention during
    # the slow-tier sweep (timeout is a contention artifact, not a real
    # failure — the model runs fine). 300 s covers it with margin.
    "mlnr.bngl": 300,
    # Lin2019 glacial models: each runs 4 simulate actions (ODE + exact/SSA +
    # std_scaling/SSA + partial_scaling/SSA). ~292 s/377 s in isolation but
    # ~505 s/602 s under load (~1.6-1.7x runtime variance); the 602 s run
    # crashed against an earlier 600 s budget. 1200 s gives robust margin so
    # the run completes rather than a boundary-race crash that truncates the
    # trailing action's .gdat. Horizon left intact (runtime is acceptable).
    "ERK_model.bngl": 1200,
    "prion_model.bngl": 1200,
}


# --- override resolution -----------------------------------------------------
# Two sources resolve to one normalized per-model dict so the unit-building loop
# is mode-agnostic:
#   * legacy  — the basename-keyed module dicts above (TEND_OVERRIDES, ...).
#   * --jobs  — each Job.overrides list from the _core manifest, keyed by
#               model_id (relpath under models/), which kills the basename
#               collisions the legacy dicts fan out on.
# Normalized keys: t_end, n_scan_pts, action_inject, tol, timeout (each None when
# absent). The Override.field vocabulary in jobs.json
# (t_end_cap/n_scan_pts/action_inject/tol/timeout) maps 1:1.
_OVERRIDE_FIELD_MAP = {
    "t_end_cap": "t_end",
    "n_scan_pts": "n_scan_pts",
    "action_inject": "action_inject",
    "rehab": "rehab",
    "tol": "tol",
    "timeout": "timeout",
}


def _empty_overrides():
    return {
        "t_end": None,
        "n_scan_pts": None,
        "action_inject": None,
        "rehab": None,
        "tol": None,
        "timeout": None,
    }


def legacy_overrides_for(name):
    """Resolve the normalized override dict from the basename-keyed dicts.

    `rehab` is intentionally absent here: it is keyed by model_id (not basename)
    and only travels via the _core manifest (--jobs), so it is resolved in
    overrides_from_job, never in the legacy basename walk.
    """
    return {
        "t_end": TEND_OVERRIDES.get(name),
        "n_scan_pts": NSCANPTS_OVERRIDES.get(name),
        "action_inject": ACTION_INJECT.get(name),
        "rehab": None,
        "tol": TOL_OVERRIDES.get(name),
        "timeout": TIMEOUT_OVERRIDES.get(name),
    }


def overrides_from_job(job):
    """Resolve the normalized override dict from a _core Job's overrides list.

    Applied identically to both engines, exactly like the legacy dicts — these
    are runtime fixtures (shorten a horizon, tighten a tolerance, rename a
    reserved symbol), each carrying a mandatory reason in the manifest.
    """
    out = _empty_overrides()
    if job is None:
        return out
    for ov in job.overrides:
        key = _OVERRIDE_FIELD_MAP.get(ov.field)
        if key is not None:
            out[key] = ov.value
    return out


def parse_simulate_methods(text):
    """Return list of (suffix_or_None, method) for each simulate-style action.

    Strips comments first. Handles both simulate({method=>X,...}) and
    simulate_<method>({...}).
    """
    text = re.sub(r"#.*", "", text)
    out = []
    for blob in re.findall(r"simulate\s*\(\s*\{([^}]*)\}", text, re.DOTALL):
        method_m = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
        suffix_m = re.search(r"suffix\s*=>\s*['\"]?([^'\",}\s]+)['\"]?", blob)
        method = (method_m.group(1) if method_m else "ode").lower()
        suffix = suffix_m.group(1) if suffix_m else None
        out.append((suffix, method))
    for m_method, blob in re.findall(
        r"simulate_(\w+)\s*\(\s*\{([^}]*)\}",
        text,
        re.DOTALL,
    ):
        suffix_m = re.search(r"suffix\s*=>\s*['\"]?([^'\",}\s]+)['\"]?", blob)
        suffix = suffix_m.group(1) if suffix_m else None
        out.append((suffix, m_method.lower()))
    return out


def parse_workflow_methods(text):
    """Return the method of each active parameter_scan/bifurcate action.

    parameter_scan/bifurcate run a per-point simulation whose method is
    given by the action's own ``method=>`` arg (default ode).
    """
    text = re.sub(r"#.*", "", text)
    out = []
    for blob in re.findall(
        r"(?:parameter_scan|bifurcate)\s*\(\s*\{([^}]*)\}",
        text,
        re.DOTALL,
    ):
        method_m = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
        out.append((method_m.group(1) if method_m else "ode").lower())
    return out


def is_stochastic(text):
    methods = [m for _, m in parse_simulate_methods(text)]
    methods += parse_workflow_methods(text)
    return any(m not in DETERMINISTIC_METHODS for m in methods)


TEND_RE = re.compile(r"(t_end\s*=>\s*)([0-9eE.+\-*/() ]+)")
NSCANPTS_RE = re.compile(r"(n_scan_pts\s*=>\s*)(\d+)")
SEED_INJECT_RE = re.compile(r"((?:simulate(?:_\w+)?|parameter_scan|bifurcate)\s*\(\s*\{)")


def _cap_tend(line, cap):
    """Reduce any t_end in `line` that EXCEEDS `cap` to `cap`; leave smaller ones.

    Cap (not set) semantics so a multi-phase model — e.g. a long equilibration
    + a short main run — has only its over-budget phase shortened. The matched
    value is restricted by TEND_RE to digits/operators/exponent chars, so the
    arithmetic eval (which resolves expressions like ``10*60``) can't execute
    anything; an unparseable value is treated as over-budget and capped.
    """
    if cap is None:
        return line

    def repl(m):
        raw = m.group(2).strip()
        try:
            val = eval(raw, {"__builtins__": {}}, {})
        except Exception:
            val = None
        if val is None or val > cap:
            return f"{m.group(1)}{cap}"
        return m.group(0)

    return TEND_RE.sub(repl, line)


def _set_nscanpts(line, n):
    """Override parameter_scan n_scan_pts (cuts scan-bound runtime). No-op if None."""
    if n is None:
        return line
    return NSCANPTS_RE.sub(rf"\g<1>{n}", line)


def patch_bngl(text, seed, tend_override=None, nscanpts_override=None):
    """Inject seed=>K into every active simulate/scan action; optional t_end cap.

    parameter_scan/bifurcate get the seed too — BNG2.pl forwards the
    action's params (including ``seed``) to each per-point simulation, so
    an ``ssa`` scan needs the seed on the scan action, not a ``simulate``.
    t_end is capped (see _cap_tend) and n_scan_pts optionally reduced; both
    applied identically on each stack.
    """
    out_lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out_lines.append(line)
            continue
        new_line = SEED_INJECT_RE.sub(rf"\g<1>seed=>{seed},", line)
        new_line = _cap_tend(new_line, tend_override)
        new_line = _set_nscanpts(new_line, nscanpts_override)
        out_lines.append(new_line)
    return "".join(out_lines)


def patch_bngl_tend_only(text, tend_override, nscanpts_override=None):
    """For deterministic models we only need the t_end cap (no seed)."""
    if tend_override is None and nscanpts_override is None:
        return text
    out_lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out_lines.append(line)
            continue
        out_lines.append(_set_nscanpts(_cap_tend(line, tend_override), nscanpts_override))
    return "".join(out_lines)


def inject_tol(text, tol_override):
    """Set atol/rtol on every active simulate-style action's parameter block.

    Mirrors the seed injection: inserts ``atol=>X,rtol=>Y,`` right after the
    action's opening ``{``. Any pre-existing atol/rtol tokens in the block are
    stripped first so the override wins (no duplicate keys). Comment lines are
    left untouched. Applied identically to both stacks.
    """
    if not tol_override:
        return text
    atol = tol_override["atol"]
    rtol = tol_override["rtol"]
    action_open = re.compile(r"((?:simulate(?:_\w+)?|parameter_scan|bifurcate)\s*\(\s*\{)")
    strip_re = re.compile(r"[ar]tol\s*=>\s*[-+0-9.eE]+\s*,?\s*")
    out_lines = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("#") or not action_open.search(line):
            out_lines.append(line)
            continue
        new_line = strip_re.sub("", line)
        new_line = action_open.sub(rf"\g<1>atol=>{atol},rtol=>{rtol},", new_line)
        # Tidy any comma artifacts from stripping a pre-existing token.
        new_line = re.sub(r"\{\s*,", "{", new_line)
        new_line = re.sub(r",(\s*\})", r"\1", new_line)
        new_line = re.sub(r",\s*,", ",", new_line)
        out_lines.append(new_line)
    return "".join(out_lines)


# Action statements stripped before a REHAB protocol is appended. We drop the
# begin/end actions block AND any bare top-level action line, so a dud model's
# leftover output/setup commands (notably a writeMfile()/writeSBML() that aborts
# when no network exists, or an uncapped native generate_network that explodes)
# can't run ahead of the injected protocol.
_ACTION_LINE_RE = re.compile(
    r"^(generate_network|simulate\w*|parameter_scan|bifurcate|readFile|setOption|"
    r"writeSBML|writeMfile|writeModel|writeXML|writeFile|writeNET|visualize|"
    r"setConcentration|saveConcentrations|saveParameters|resetConcentrations|"
    r"resetParameters|setParameter|quit)\b"
)


def strip_actions(text):
    """Remove any begin/end actions block and stray top-level action statements.

    Used only on REHAB models, which are then given a fresh actions block. Lines
    inside model blocks (parameters, rules, observables) are untouched — the
    action keywords above don't occur there.
    """
    text = re.sub(r"begin actions.*?end actions", "", text, flags=re.S | re.I)
    return "\n".join(ln for ln in text.splitlines() if not _ACTION_LINE_RE.match(ln.strip()))


def run_one(simulator, bngl_path, run_path, out_dir, timeout, mem_floor_gb=0.0):
    """Run a (possibly patched) .bngl through bionetgen.run() in a subprocess.

    `bngl_path` is the original (just for logging / summary identity).
    `run_path` is what bionetgen.run() actually loads (patched copy or original).
    Blocks on the memory governor before launching so a burst of memory-heavy
    NFsim models can't co-resident-overrun RAM and freeze the machine.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_run.log"
    # `timeout` is already the per-model effective budget (main resolves the
    # per-model timeout override for both legacy and --jobs modes before
    # dispatch), so no basename lookup here.
    #
    # The two sides take DIFFERENT engine paths (GH #175):
    #   * subprocess — the legacy BNG2.pl reference: bionetgen.run runs the model's
    #     own actions through run_network / NFsim. This is what we compare against.
    #   * bngsim — drive GENUINE bngsim via _bng_common.run_bngsim_job: BNG2.pl
    #     generates the .net/.xml, then bngsim simulates IN-PROCESS (the bridge's
    #     direct route). NOT bionetgen.run(simulator='bngsim'), which with a stock
    #     BNG2.pl routes a BNGL simulate back to run_network/NFsim — producing
    #     BNG2.pl output mislabelled "bngsim" (the bug this fixes).
    # Both run in a throwaway subprocess so a crash/segfault/leak in either engine
    # is isolated to one job, not the pool worker.
    if simulator == "bngsim":
        inner = (
            "import os, sys, json\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import _bng_common as C\n"
            "bng2 = C.resolve_bng2_pl(os.environ.get('BNGPATH') or os.environ.get('BNG2_PL'))\n"
            f"info = C.run_bngsim_job({str(run_path)!r}, {str(out_dir)!r}, bng2, timeout={timeout})\n"
            # GH #179: persist the per-model replay verdict (protocol_prefix.replayed /
            # segments / replay_error / reinit_ics) alongside the artifacts so the golden
            # generator can carry it into golden.json _meta (provenance for which jobs ran
            # a faithful multi-phase replay vs an option-1 best-effort fallback).
            "pp = info.get('protocol_prefix')\n"
            "if pp is not None:\n"
            f"    p = os.path.join({str(out_dir)!r}, '_provenance.json')\n"
            "    open(p, 'w').write(json.dumps({'engine': info['engine'], 'method': "
            "info['method'], 'track': info['track'], 'protocol_prefix': pp}))\n"
            "print('BNGSIM_ENGINE_OK', info['engine'], info['method'], info['track'], info['artifacts'])\n"
        )
    else:
        inner = (
            "import sys, bionetgen\n"
            f"bionetgen.run({str(run_path)!r}, out={str(out_dir)!r}, "
            f"timeout={timeout}, suppress=True, simulator={simulator!r})\n"
        )
    mem_wait = _await_memory(mem_floor_gb)
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", inner],
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            cwd=tempfile.gettempdir(),
        )
        elapsed = time.monotonic() - start
        artifacts = (
            sorted(
                f.name for f in out_dir.iterdir() if f.is_file() and f.suffix in OUTPUT_EXTENSIONS
            )
            if out_dir.exists()
            else []
        )
        if proc.returncode == 0:
            log_path.write_text(
                f"# {bngl_path}\n# run={run_path}\n# simulator={simulator}\n"
                f"# status=ok\n# wall_seconds={elapsed:.2f}\n"
                f"# mem_wait_seconds={mem_wait:.1f}\n"
                f"# artifacts={artifacts}\n\n"
                f"--- STDOUT ---\n{proc.stdout}\n\n--- STDERR ---\n{proc.stderr}\n"
            )
            return {
                "bngl": str(bngl_path),
                "run": str(run_path),
                "category": Path(bngl_path).parent.name,
                "status": "ok",
                "wall_seconds": elapsed,
                "artifacts": artifacts,
                "out_dir": str(out_dir),
                "error": "",
            }
        log_path.write_text(
            f"# {bngl_path}\n# run={run_path}\n# simulator={simulator}\n"
            f"# status=crash\n# returncode={proc.returncode}\n"
            f"# wall_seconds={elapsed:.2f}\n\n"
            f"--- STDOUT ---\n{proc.stdout}\n\n--- STDERR ---\n{proc.stderr}\n"
        )
        return {
            "bngl": str(bngl_path),
            "run": str(run_path),
            "category": Path(bngl_path).parent.name,
            "status": "crash",
            "wall_seconds": elapsed,
            "artifacts": artifacts,
            "out_dir": str(out_dir),
            "error": (proc.stderr.strip().splitlines() or [""])[-1][:500],
        }
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        log_path.write_text(
            f"# {bngl_path}\n# run={run_path}\n# simulator={simulator}\n"
            f"# status=timeout\n# wall_seconds={elapsed:.2f}\n\n"
            f"--- STDOUT ---\n{e.stdout or ''}\n\n--- STDERR ---\n{e.stderr or ''}\n"
        )
        return {
            "bngl": str(bngl_path),
            "run": str(run_path),
            "category": Path(bngl_path).parent.name,
            "status": "timeout",
            "wall_seconds": elapsed,
            "artifacts": [],
            "out_dir": str(out_dir),
            "error": f"timed out after {timeout}s",
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        log_path.write_text(
            f"# {bngl_path}\n# run={run_path}\n# simulator={simulator}\n"
            f"# status=error\n# wall_seconds={elapsed:.2f}\n\n"
            f"--- TRACEBACK ---\n{traceback.format_exc()}\n"
        )
        return {
            "bngl": str(bngl_path),
            "run": str(run_path),
            "category": Path(bngl_path).parent.name,
            "status": "error",
            "wall_seconds": elapsed,
            "artifacts": [],
            "out_dir": str(out_dir),
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }


# Trivial model (a single ODE simulate) the engine pre-flight drives through the
# genuine-bngsim path (run_bngsim_job) to prove bngsim actually simulates.
_ENGINE_PROBE_BNGL = """\
begin model
begin parameters
 k 1.0
end parameters
begin species
 A() 100
 B() 0
end species
begin reaction rules
 A() -> B() k
end reaction rules
begin observables
 Molecules Atot A()
end observables
end model
generate_network({overwrite=>1})
simulate({method=>"ode",t_end=>1,n_steps=>5,print_CDAT=>1})
"""


def _assert_bngsim_engine_live(timeout):
    """Pre-flight: PROVE bngsim actually simulates before sweeping (GH #175).

    The bngsim side drives ``_bng_common.run_bngsim_job``: BNG2.pl generates the
    network/XML, then bngsim simulates IN-PROCESS through the bridge's direct route
    (``execute_bngsim_direct_job``). It does NOT call
    ``bionetgen.run(simulator='bngsim')``, which with a stock BNG2.pl routes a BNGL
    ``simulate`` back to run_network/NFsim (BNG2.pl) — the very mislabelling this
    guard now defends against. The guard proves EXECUTION, not a routing prediction:

      1. ``bngsim_backend.backend_status()`` — bngsim is importable AND version-
         compatible (what ``run_bngsim_job``'s direct route needs), plus the
         bionetgen commit/version provenance.
      2. a trivial ODE probe driven through ``run_bngsim_job`` completes, writes
         real numeric output (.gdat/.cdat), AND emits the per-job
         ``BNGSIM_ENGINE_OK`` marker — proving bngsim's in-process direct route ran
         (``run_bngsim_job`` never invokes run_network/NFsim, so the marker cannot
         appear from a legacy fallback).

    The DECISIVE no-trust proof — run the probe with ``run_network`` moved aside,
    where the genuine path still succeeds while the old bridge BNGL path crashes —
    lives in tests/test_bngsim_golden_engine.py (it mutates a controlled env, which
    is unsafe to do under a live sweep). Abort loudly if any signal fails.
    """
    import bngsim_backend

    status = bngsim_backend.backend_status()
    where = status["bionetgen_path"] or f"<bionetgen not importable: {status['reason']}>"
    version = status["bngsim_version"]
    if not status["available"]:
        sys.exit(
            "ABORT: --simulator bngsim, but bngsim is UNAVAILABLE in the active env "
            "(BNGSIM_AVAILABLE is False"
            + (f"; reason: {status['reason']}" if status["reason"] else "")
            + "). run_bngsim_job's in-process direct route cannot run, so there is no "
            f"genuine bngsim to sweep.\n  active bionetgen: {where}\n"
            f"  bionetgen commit: {status['bionetgen_commit']}\n"
            f"  bngsim required:  >= {status['min_bngsim_version']}  (found: {version})\n"
            "  fix: install a PyBioNetGen carrying the merged bngsim integration "
            "(RuleWorld/PyBioNetGen#102) into the SAME env as an importable, "
            "version-compatible bngsim — see bootstrap_parity_env.py, then retry."
        )

    probe_dir = Path(tempfile.mkdtemp(prefix="bngsim_engine_probe_"))
    try:
        bngl = probe_dir / "engine_probe.bngl"
        bngl.write_text(_ENGINE_PROBE_BNGL)
        out_dir = probe_dir / "out"
        result = run_one("bngsim", bngl, bngl, out_dir, min(timeout, 120))
        numeric = (
            sorted(p.name for p in out_dir.glob("*") if p.suffix in {".gdat", ".cdat", ".scan"})
            if out_dir.exists()
            else []
        )
        log_text = ""
        log_file = out_dir / "_run.log"
        if log_file.is_file():
            log_text = log_file.read_text(errors="replace")
        engine_proven = "BNGSIM_ENGINE_OK" in log_text
        if result["status"] != "ok" or not numeric or not engine_proven:
            sys.exit(
                "ABORT: --simulator bngsim, but the genuine-bngsim probe did NOT prove "
                f"execution (status={result['status']}; artifacts={result.get('artifacts')}; "
                f"engine_marker={'present' if engine_proven else 'MISSING'}). A real bngsim "
                "simulate did not complete — refusing to sweep rather than emit an "
                "empty/wrong/mislabelled golden.\n"
                f"  active bionetgen: {where}\n"
                f"  bngsim:           {version}\n"
                f"  probe _run.log:   {log_file}"
            )
        print(
            f"bngsim engine pre-flight: OK (genuine in-process bngsim {version}, "
            f"bionetgen {status['bionetgen_commit']}; probe wrote {', '.join(numeric)})"
        )
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def _audit_engine_routes(units, strict):
    """Classify every unit's engine up front (the way the sweep actually runs it).

    The pre-flight proves bngsim is wired in; this proves what each individual job
    will do. Because the sweep drives bngsim DIRECTLY (``run_bngsim_job``), every
    job whose method bngsim supports (ode/ssa/psa/nf/rm) genuinely runs on bngsim —
    there is no bridge-routing pass that could silently fall back to BNG2.pl. The
    only non-bngsim outcome is a job whose ONLY method is unsupported (``pla``),
    which the sweep SKIPS (never run on the legacy stack and mislabelled — GH #175).
    Returns a JSON-able audit dict (recorded in _summary.json). With ``strict`` any
    unsupported/unknown job is a hard abort (golden uses this).
    """
    import bngsim_backend

    audit = bngsim_backend.audit_unit_engines(units)
    counts = audit["counts"]
    by_track = ", ".join(f"{k}={v}" for k, v in sorted(audit["by_track"].items()))
    print(
        "engine plan:    bngsim={bngsim} ({by_track})  unsupported/skip={unsupported}  "
        "unknown={unknown}".format(by_track=by_track or "-", **counts)
    )
    unsupported = audit["unsupported"]
    unknown = audit["unknown"]
    if unsupported or unknown:
        print(
            f"  !! {len(unsupported)} job(s) bngsim cannot run + {len(unknown)} unclassifiable "
            "— these will NOT produce a bngsim reference:"
        )
        for unit, reason in (unsupported + unknown)[:50]:
            print(f"     {unit.get('bngl')}  [{reason}]")
        if strict:
            sys.exit(
                f"ABORT (--strict-engine): {len(unsupported)} job(s) declare only methods "
                f"bngsim cannot run and {len(unknown)} could not be classified. A bngsim "
                "golden/parity run must produce a genuine bngsim reference for every job. "
                "Exclude the listed models (e.g. pla-only) or fix the unclassifiable ones."
            )
    # Slim, JSON-able record for _summary.json (model path + the classifier reason).
    return {
        "counts": counts,
        "by_track": audit["by_track"],
        "unsupported": sorted([u.get("bngl"), reason] for u, reason in unsupported),
        "unknown": sorted([u.get("bngl"), reason] for u, reason in unknown),
        "strict": bool(strict),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="",
        help="Directory tree with .bngl files (legacy mode). In --jobs mode this "
        "is the models/ base the manifest's model_id paths are relative to; "
        "defaults to <jobs.json dir>/models.",
    )
    ap.add_argument(
        "--jobs",
        default="",
        help="A _core manifest (jobs.json). When given, the sweep runs exactly the "
        "manifest's jobs (selected by model_id) and applies each Job.overrides "
        "instead of the basename-keyed override dicts.",
    )
    ap.add_argument("--out", required=True, help="Output root for patched copies + artifacts")
    ap.add_argument(
        "--simulator",
        required=True,
        choices=("subprocess", "bngsim"),
        help="Simulator to pass to bionetgen.run()",
    )
    ap.add_argument("--n-seeds", type=int, default=10, help="Seeds 1..N for stochastic models")
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max concurrent sims (further capped by core count and the RAM budget; "
        "each worker spawns a python->BNG2.pl->run_network/NFsim subtree, not one process)",
    )
    ap.add_argument(
        "--mem-per-worker-gb",
        type=float,
        default=MEM_PER_WORKER_GB,
        help="Planning estimate of peak RAM per worker; caps --workers to fit available RAM",
    )
    ap.add_argument(
        "--mem-floor-gb",
        type=float,
        default=MEM_FLOOR_GB,
        help="A worker waits to launch its sim while free RAM is below this (0 disables)",
    )
    ap.add_argument("--timeout", type=int, default=180, help="Per-model timeout (s)")
    ap.add_argument(
        "--regime",
        choices=("all", "ode", "stochastic"),
        default="all",
        help="Restrict to deterministic (ode) or stochastic units. Stages the "
        "fast, safe ODE set apart from the slow, supervision-heavy stochastic set.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max .bngl files (0=all)")
    ap.add_argument(
        "--strict-engine",
        action="store_true",
        help="With --simulator bngsim, ABORT if any non-pla job would route to the "
        "legacy BNG2.pl subprocess (or error) instead of bngsim. The engine audit "
        "always runs and reports; this makes a surprise fallback fatal (golden uses it).",
    )
    ap.add_argument("--include", default="", help="Substring filter on file path (debugging)")
    ap.add_argument("--exclude", default="", help="Substring filter — drop matching file paths")
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated model basenames (.bngl optional); "
        "restrict the sweep to exactly these — for selective "
        "high-seed re-runs of models flagged DIFF",
    )
    args = ap.parse_args()

    # Mode: --jobs drives off the _core manifest (select by model_id, apply
    # Job.overrides); legacy walks --root and uses the basename override dicts.
    jobs_by_relpath = None
    if args.jobs:
        from _core import read_manifest

        jobs_path = Path(args.jobs).resolve()
        _meta, jobs = read_manifest(jobs_path)
        root = Path(args.root).resolve() if args.root else jobs_path.parent / "models"
        jobs_by_relpath = {j.model_id: j for j in jobs}
    else:
        if not args.root:
            ap.error("--root is required unless --jobs is given")
        root = Path(args.root).resolve()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    patch_root = out_root / "_patched"
    patch_root.mkdir(parents=True, exist_ok=True)

    # Disk pre-flight: macOS swap is dynamic and shares this volume, so a
    # near-full disk means the kernel can't page out under memory pressure and
    # jetsams/freezes instead. Refuse to start without swap headroom.
    free_disk_gb = shutil.disk_usage(out_root).free / 2**30
    if free_disk_gb < DISK_ABORT_GB:
        sys.exit(
            f"ABORT: only {free_disk_gb:.1f} GB free on {out_root}; need >= "
            f"{DISK_ABORT_GB} GB so the OS keeps swap headroom (else the box can "
            f"freeze under memory pressure). Free up disk and retry."
        )
    if free_disk_gb < DISK_WARN_GB:
        print(f"WARN: low disk: {free_disk_gb:.1f} GB free on {out_root} (< {DISK_WARN_GB} GB).")

    # Engine pre-flight: refuse to run a "bngsim" sweep when bngsim is not wired
    # into the active PyBioNetGen bridge (wrong/old PyBioNetGen, or bngsim missing
    # from its env). See the function docstring for the merged-contract signals.
    if args.simulator == "bngsim":
        _assert_bngsim_engine_live(args.timeout)

    n_missing = 0
    if jobs_by_relpath is not None:
        # Manifest order; each model_id is a relpath under models/.
        candidate_files = []
        for model_id in jobs_by_relpath:
            p = root / model_id
            if p.exists():
                candidate_files.append(p)
            else:
                n_missing += 1
        if n_missing:
            print(f"WARN: {n_missing} manifest model(s) not found under {root}")
    else:
        candidate_files = sorted(root.rglob("*.bngl"))
    if args.include:
        candidate_files = [f for f in candidate_files if args.include in str(f)]
    if args.exclude:
        candidate_files = [f for f in candidate_files if args.exclude not in str(f)]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        wanted |= {m + ".bngl" for m in set(wanted) if not m.endswith(".bngl")}
        candidate_files = [f for f in candidate_files if f.name in wanted]

    # For each model, decide regime + emit (run_path, out_dir, role) units.
    # role is "deterministic" or seed_K.
    units = []  # list of dicts: bngl, run, out_dir, role, regime
    n_det = 0
    n_stoch = 0
    n_unreadable = 0
    for src in candidate_files:
        try:
            text = src.read_text(errors="replace")
        except Exception:
            n_unreadable += 1
            continue
        rel = src.relative_to(root)
        if jobs_by_relpath is not None:
            ovr = overrides_from_job(jobs_by_relpath.get(str(rel)))
        else:
            ovr = legacy_overrides_for(src.name)
        tend_override = ovr["t_end"]
        nscanpts_override = ovr["n_scan_pts"]
        tol_override = ovr["tol"]
        unit_timeout = ovr["timeout"] or args.timeout
        # Rehabilitate a dud model (overrides.REHAB, keyed by model_id): strip
        # its existing actions — so a stray writeMfile()/writeSBML() can't abort
        # the run, and an uncapped native generate_network can't explode — then
        # append the short rehab protocol. Done BEFORE the stochastic check so
        # the injected method picks the regime (like action_inject).
        rehab = ovr["rehab"]
        if rehab:
            text = strip_actions(text).rstrip() + "\nbegin actions\n" + rehab + "\nend actions\n"
        # Inject a run action for models that ship without one, BEFORE the
        # stochastic check so the injected method picks the regime. Appended
        # to the (renamed) source; the seed/t_end patchers then treat it like
        # any native action.
        action_inject = ovr["action_inject"]
        if action_inject:
            text = text.rstrip() + "\n" + action_inject + "\n"
        if is_stochastic(text):
            n_stoch += 1
            for seed in range(1, args.n_seeds + 1):
                patched_dir = patch_root / rel.parent / f"{rel.stem}__seed{seed}"
                patched_dir.mkdir(parents=True, exist_ok=True)
                patched_path = patched_dir / rel.name
                patched_path.write_text(
                    inject_tol(
                        patch_bngl(text, seed, tend_override, nscanpts_override), tol_override
                    )
                )
                out_dir = out_root / rel.parent / rel.stem / f"seed{seed}"
                units.append(
                    {
                        "bngl": str(src),
                        "run": str(patched_path),
                        "out_dir": str(out_dir),
                        "role": f"seed{seed}",
                        "regime": "stochastic",
                        "timeout": unit_timeout,
                    }
                )
        else:
            n_det += 1
            # Write a patched copy if any override or rename applies. Note
            # `action_inject`/`rehab`: a model classified deterministic *because
            # of* an injected ODE action must run the patched copy that carries
            # that action — otherwise it falls back to the raw source and
            # produces only the network (.net), no trajectory.
            if (
                tend_override is not None
                or nscanpts_override is not None
                or tol_override is not None
                or action_inject
                or rehab
            ):
                patched_dir = patch_root / rel.parent
                patched_dir.mkdir(parents=True, exist_ok=True)
                patched_path = patched_dir / rel.name
                patched_path.write_text(
                    inject_tol(
                        patch_bngl_tend_only(text, tend_override, nscanpts_override), tol_override
                    )
                )
                run_path = patched_path
            else:
                run_path = src
            out_dir = out_root / rel.parent / rel.stem / "det"
            units.append(
                {
                    "bngl": str(src),
                    "run": str(run_path),
                    "out_dir": str(out_dir),
                    "role": "det",
                    "regime": "deterministic",
                    "timeout": unit_timeout,
                }
            )

    # Regime filter (deterministic/ode vs stochastic). Lets a golden/parity run
    # stage the fast, safe ODE set separately from the slow, supervision-heavy
    # stochastic (NFsim/SSA) set without entangling them across tiers.
    if args.regime != "all":
        want = "deterministic" if args.regime == "ode" else "stochastic"
        units = [u for u in units if u["regime"] == want]

    if args.limit:
        units = units[: args.limit]

    # Engine audit (bngsim sweeps only): classify each unit's route so a fallback
    # to the legacy stack is never unnoticed. Runs on the EXACT units that will be
    # swept (post regime/limit filtering). Recorded in _summary.json.
    engine_audit = None
    backend_prov = None
    if args.simulator == "bngsim":
        import bngsim_backend

        backend_prov = bngsim_backend.backend_status()
        engine_audit = _audit_engine_routes(units, strict=args.strict_engine)

    if args.jobs:
        print(f"jobs manifest: {Path(args.jobs).resolve()} ({len(jobs_by_relpath)} jobs)")
    print(f"sweep root:    {root}")
    print(f"sweep out:     {out_root}")
    print(f"simulator:     {args.simulator}")
    print(f"n_seeds:       {args.n_seeds}")
    print(f"deterministic: {n_det}")
    print(f"stochastic:    {n_stoch}  (× {args.n_seeds} seeds = {n_stoch * args.n_seeds} runs)")
    print(f"unreadable:    {n_unreadable}")
    print(f"total runs:    {len(units)}")
    eff_workers, plan_reason = plan_workers(args.workers, args.mem_per_worker_gb)
    note = f" ({plan_reason})" if plan_reason else ""
    print(f"workers:       {eff_workers} of {args.workers} requested{note}")
    avail = _available_gb()
    print(
        f"memory:        {avail:.1f} GB avail, floor {args.mem_floor_gb} GB/launch"
        if avail is not None
        else "memory:        psutil unavailable — governor OFF (keep --workers low)"
    )
    print(f"disk:          {free_disk_gb:.1f} GB free on out volume")
    print(f"per-model timeout: {args.timeout}s")
    print(f"python:        {sys.executable}")

    probe = subprocess.run(
        [sys.executable, "-c", "import bionetgen; print(bionetgen.__file__)"],
        capture_output=True,
        text=True,
        cwd=tempfile.gettempdir(),
    )
    bionetgen_path = (probe.stdout.strip().splitlines() or ["<unresolved>"])[-1]
    print(f"bionetgen:     {bionetgen_path}")
    bngsim_path = None
    if args.simulator == "bngsim":
        probe2 = subprocess.run(
            [
                sys.executable,
                "-c",
                "import bngsim; print(bngsim.__file__, getattr(bngsim, '__version__', '?'))",
            ],
            capture_output=True,
            text=True,
            cwd=tempfile.gettempdir(),
        )
        bngsim_path = (probe2.stdout.strip().splitlines() or [""])[-1]
        print(f"bngsim:        {bngsim_path}")

    summary = []
    started_at = time.time()
    with cf.ProcessPoolExecutor(max_workers=eff_workers) as pool:
        futs = {
            pool.submit(
                run_one,
                args.simulator,
                u["bngl"],
                u["run"],
                u["out_dir"],
                u["timeout"],
                args.mem_floor_gb,
            ): u
            for u in units
        }
        for done, fut in enumerate(cf.as_completed(futs), start=1):
            u = futs[fut]
            res = fut.result()
            res["role"] = u["role"]
            res["regime"] = u["regime"]
            summary.append(res)
            if done % 50 == 0 or done == len(units):
                print(
                    f"[{done}/{len(units)}] {res['status']:8s} "
                    f"{res['wall_seconds']:6.1f}s  {u['role']:8s} {res['bngl']}"
                )

    elapsed_total = time.time() - started_at
    by_status = {}
    for r in summary:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1

    summary_path = out_root / "_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "out": str(out_root),
                "python": sys.executable,
                "bionetgen_path": bionetgen_path,
                "bngsim_path": bngsim_path,
                # bngsim-backend provenance + the per-unit engine audit, so a
                # "bngsim" sweep records WHICH PyBioNetGen/bngsim it used and that
                # every job actually routed to bngsim (no unnoticed legacy fallback).
                "bngsim_backend": backend_prov,
                "engine_audit": engine_audit,
                "simulator": args.simulator,
                "n_seeds": args.n_seeds,
                "workers_requested": args.workers,
                "workers_effective": eff_workers,
                "mem_per_worker_gb": args.mem_per_worker_gb,
                "mem_floor_gb": args.mem_floor_gb,
                "free_disk_gb_at_start": round(free_disk_gb, 1),
                "jobs_manifest": str(Path(args.jobs).resolve()) if args.jobs else None,
                # In legacy mode the basename dicts below were the override
                # source; in --jobs mode overrides come per-model from the
                # manifest's Job.overrides (recorded there, not snapshot here).
                "tend_overrides": None if args.jobs else TEND_OVERRIDES,
                "tol_overrides": None if args.jobs else TOL_OVERRIDES,
                "timeout_overrides": None if args.jobs else TIMEOUT_OVERRIDES,
                "n_deterministic_models": n_det,
                "n_stochastic_models": n_stoch,
                "n_units": len(units),
                "elapsed_total_seconds": elapsed_total,
                "by_status": by_status,
                "results": sorted(summary, key=lambda r: (r["bngl"], r.get("role", ""))),
            },
            indent=2,
        )
    )

    print(f"\nDone in {elapsed_total / 60:.1f}m. By status: {by_status}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
