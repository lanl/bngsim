"""Per-model parity overrides, lifted out of parity_sweep.py's hardcoded dicts.

In PyBioNetGen these lived as basename-keyed module globals (TEND_OVERRIDES,
TOL_OVERRIDES, ...) with the rationale buried in inline comments. The plan
moves them into the manifest as first-class ``Override`` records — each with a
mandatory ``reason`` — so the spec is self-documenting and every deviation from
defaults is auditable.

Each override is applied IDENTICALLY to both engines (bngsim and the legacy BNG
stack), so the parity check stays apples-to-apples: these are runtime fixtures
(shorten a horizon, tighten a tolerance, rename a reserved symbol), never
statements about a model's biology.

Keying: authored by basename (how the source dicts were written). build_jobs.py
resolves each onto the vendored model(s) it matches by relpath, and warns on a
basename that hits more than one vendored model so collisions surface instead of
silently fanning out.
"""

from __future__ import annotations

# t_end CAP (simulated-time units): any active simulate/scan action whose t_end
# EXCEEDS the cap is reduced to it (multi-phase models keep short phases). Each
# value's reason captures why (most: keep a stochastic ensemble inside the
# parity wall-clock budget).
TEND_CAP = {
    "AD 3 State FREE Expanding nfs.bngl": (100, "default 650 over NF seeds blows the 180s budget"),
    "scaling_example.bngl": (
        50,
        "bngsim codegen falls back to interpreted RHS; t_end=730 × 10 NF seeds overruns",
    ),
    "jobs_tofit_gen48ind13.bngl": (100, "nf t_end=1000, ~65s/run × seeds"),
    "jobs_tofit_gen34ind26.bngl": (100, "nf t_end=1000, ~65s/run × seeds"),
    "jobs_tofit_iter44p17.bngl": (100, "nf t_end=1000, ~63s/run × seeds"),
    "fceri_ji.bngl": (60, "nf t_end=500, ~61s/run × seeds"),
    "example2_fit.bngl": (16, "nf t_end=126, ~61s/run × seeds"),
    "v21.bngl": (8, "ode+ssa t_end=100, ssa-bound ~56s/run × seeds"),
    "tcr_iter28p4h2.bngl": (2.0, "equil 15.6 + main 0.94, ~59s/run × seeds"),
    "tcr_iter9p44.bngl": (2.0, "equil 15.6 + main 0.94, ~51s/run × seeds"),
    "egfr_nf_iter5p12h10.bngl": (9, "nf t_end=60, ~51s/run × seeds"),
    "e6.bngl": (25, "nf t_end=200, ~60s/run × seeds"),
    "e5.bngl": (30, "nf t_end=200, ~46s/run × seeds"),
    "e4.bngl": (40, "nf t_end=200, ~35s/run × seeds"),
    "e3.bngl": (60, "nf t_end=200, ~25s/run × seeds"),
    "e1.bngl": (110, "nf t_end=200, ~14s/run × seeds"),
    "egfr_net.bngl": (30, "nf t_end=120, ~33s/run × seeds"),
    "bench_blbr_rings_posner1995.bngl": (900, "nf t_end=3000, ~27s/run × seeds"),
    "PushPull.bngl": (1200, "nf t_end=4000, ~25s/run × seeds"),
    "tcr_gen20ind9.bngl": (1.0, "equil 25.2 + main 1.5, ~61s/run × seeds"),
    "bench_blbr_dembo1978_monovalent_inhibitor.bngl": (1200, "nf t_end=3000, ~19s/run × seeds"),
    "receptor_nf_iter91p28.bngl": (60, "equil 600 + main 60, ~19s/run × seeds"),
    "example3_fit.bngl": (1800, "nf t_end=5000, rulehub+rulemonkey benches ~17-21s/run × seeds"),
    "tlbr_yang2008.bngl": (1800, "nf t_end=3000, ~13s/run × seeds"),
    "BLBR.bngl": (5, "nf t_end=10 / n_steps 1000, ~16s/run × seeds"),
    "example6_ground_truth.bngl": (60, "equil 600 + main 60, ~15s/run × seeds"),
    "blbr_heterogeneity_goldstein1980.bngl": (
        5,
        "nf parameter_scan t_end=10 (also n_scan_pts cut), ~28s/run",
    ),
}

# parameter_scan n_scan_pts reduction (scan-bound stochastic cost is dominated
# by point count, not t_end). Each scan point is still ensemble-compared.
NSCANPTS = {
    "blbr_heterogeneity_goldstein1980.bngl": (
        6,
        "was 18; scan-bound runtime cut, still cell-compared",
    ),
}

# Append a run action to models that ship without one (so they emit comparable
# output). The injected method decides det/stochastic regime.
ACTION_INJECT = {
    "Rule_based_egfr_tutorial.bngl": (
        'simulate({method=>"ode",t_end=>20,n_steps=>100})',
        "generate_network-only EGFR tutorial; add an ODE run so it produces a .gdat",
    ),
}

# ODE solver tolerance overrides for ill-conditioned IVPs where BOTH CVODE
# configs give a (different) wrong answer at the shared 1e-8 default — the DIFF
# is a property of the IVP, not either engine. Tightening makes them converge.
TOL = {
    "eco_coevolution_host_parasite.bngl": (
        {"atol": "1e-12", "rtol": "1e-12"},
        "ill-conditioned IVP; both engines diverge at 1e-8, agree to ~9 sig figs at 1e-12",
    ),
}

# SYMBOL_RENAME (retired) — the old sweep-time whole-word rename for two classes of
# problem model. Both are now handled by a single, non-confusing mechanism each, so the
# file-on-disk always matches what runs (no silent runtime substitution):
#   * Typo'd action args (atoll/abstol/reltol) — genuine typos that BNG2.pl silently
#     ignored, bypassing the modeler's intended tolerance. Now PATCHED AT IMPORT via
#     vendor_corpus.CORPUS_REPAIRS (the on-disk model is corrected).
#   * Reserved-word collisions (frac, NFsim/ExprTk) — a VALID model with a hostile name.
#     The engine should cope under the hood (cf. the Python-keyword cope in
#     _bng_common._resolve_token); bngsim's #63/#64 remap covers the ODE/NFsim paths.
#     The 3 `frac` models are no longer in the corpus anyway (their entries were stale).

# Per-model timeout extension (seconds) for models whose overrun is a worker-
# contention artifact, not a real failure.
TIMEOUT = {
    "mlnr.bngl": (
        300,
        "NFsim t_end=1000/1000 steps, ~82s/seed; 180s budget overrun under 2-worker contention",
    ),
    "ERK_model.bngl": (
        1200,
        "Lin2019: 4 simulate actions (ODE+exact+std_scaling+partial_scaling SSA) on a large "
        "network. ~292s isolated but ~505s under load (~1.7x runtime variance); 1200s gives "
        "robust margin so the run completes (vs a boundary-race crash that truncates output).",
    ),
    "prion_model.bngl": (
        1200,
        "Lin2019: PrP=>120 network, 4 simulate actions at n_steps=30000. ~377s isolated but "
        "crashed at 602s against a 600s budget under load (~1.6x variance); 1200s gives 2x "
        "margin over the worst observation so the run completes cleanly.",
    ),
}


def overrides_for(basename: str) -> list[dict]:
    """All overrides authored for a model basename, as {field, value, reason} dicts."""
    out = []
    if basename in TEND_CAP:
        v, why = TEND_CAP[basename]
        out.append({"field": "t_end_cap", "value": v, "reason": why})
    if basename in NSCANPTS:
        v, why = NSCANPTS[basename]
        out.append({"field": "n_scan_pts", "value": v, "reason": why})
    if basename in ACTION_INJECT:
        v, why = ACTION_INJECT[basename]
        out.append({"field": "action_inject", "value": v, "reason": why})
    if basename in TOL:
        v, why = TOL[basename]
        out.append({"field": "tol", "value": v, "reason": why})
    if basename in TIMEOUT:
        v, why = TIMEOUT[basename]
        out.append({"field": "timeout", "value": v, "reason": why})
    return out


# ── Rehabilitation, keyed by MODEL_ID (relpath under models/) ────────────────
# Some corpus models ship without a runnable simulation protocol: no actions
# block, every action commented out, generate_network-only, or (for a couple)
# observables-less. They emit no fingerprintable artifact, so they can't be a
# golden/parity fixture. We *rehabilitate* them by replacing their actions with
# a short protocol that exercises bngsim and yields a .gdat (or a .cdat for the
# observable-less ones, via the D1 fallback). We are checking engine reliability,
# not reproducing biology, so a tiny horizon and a capped network are fine.
#
# Keyed by model_id (NOT basename): several dud basenames (e.g. PushPull) also
# name a *non*-dud model elsewhere in the corpus, and the rehab must not touch
# those. The runner strips the model's existing actions (so a stray writeMfile()
# can't abort the run) and appends these; the injected method decides the regime.
#
# Protocols (all verified to run in a few seconds, except Chattaraj which needs
# the tightest network cap):
#   _NF   network-free NFsim models that define observables -> seed=1 nf run.
#   _ODE  capped generate_network + a short ODE run with print_CDAT=>1 (so an
#         observables-less model still yields a .cdat for the fallback).
#   _ODE1 same, with max_iter=>1 for a combinatorially-explosive network.
#   _SSA  same as _ODE but a seed=1 SSA run — used for models that live in the
#         ssa/ directory so the assigned method matches their intended regime.
#   _SSA_FULL  SSA on a COMPLETE network (no max_iter cap; max_agg=>4 still bounds
#         it). Needed when a capped network would be incomplete: run_network
#         on-the-fly-expands an incomplete network during SSA while bngsim keeps
#         the fixed network, so they diverge unless the network is complete.
# The directory (my_models/{nf,ode,ssa}/) is the model's *intended* method; we
# match it where the engine allows. One unavoidable exception runs as _ODE despite
# a stochastic-method directory: cleavage_mechanism_v1 (nf/, but defines no
# observables, so NFsim yields only a time column — the capped-ODE+.cdat route is
# the only fingerprintable output).
_NF = 'simulate({method=>"nf",t_end=>10,n_steps=>20})'


def _net_rehab(method: str = "ode", max_iter: int | None = 3) -> str:
    # max_iter=None omits the iteration cap entirely, so generate_network runs to
    # completion (bounded only by max_agg=>4). A complete network matters for SSA:
    # run_network on-the-fly-EXPANDS a capped/incomplete network during the SSA run
    # (revealing species incrementally — also why its .cdat goes ragged), whereas
    # bngsim simulates the fixed network as generated. Feeding both a *complete*
    # network makes them agree. (ODE never on-the-fly-expands, so the cap is moot
    # there.)
    cap = f"max_iter=>{max_iter}," if max_iter is not None else ""
    return (
        f"generate_network({{overwrite=>1,{cap}max_agg=>4}})\n"
        f'simulate({{method=>"{method}",t_start=>0,t_end=>1,n_steps=>20,print_CDAT=>1}})'
    )


_ODE = _net_rehab("ode")
_ODE1 = _net_rehab("ode", 1)
_SSA = _net_rehab("ssa")
# Complete-network SSA (no max_iter cap): see _net_rehab + agentCellSystem note.
_SSA_FULL = _net_rehab("ssa", max_iter=None)

REHAB = {
    "original/bngl_models/my_models/nf/actin_branch_forFitToData.bngl": _NF,
    "original/bngl_models/my_models/nf/checkring.bngl": _NF,
    # cleavage defines no observables, so an NFsim run is time-only; a capped
    # network + ODE gives a .cdat to fingerprint via the D1 fallback instead.
    "original/bngl_models/my_models/nf/cleavage_mechanism_v1.bngl": _ODE,
    "original/bngl_models/my_models/nf/e1_1.bngl": _NF,
    "original/bngl_models/my_models/nf/e2_1.bngl": _NF,
    "original/bngl_models/my_models/nf/e3_1.bngl": _NF,
    "original/bngl_models/my_models/nf/e4_1.bngl": _NF,
    "original/bngl_models/my_models/nf/e5_1.bngl": _NF,
    "original/bngl_models/my_models/nf/e6_1.bngl": _NF,
    "original/bngl_models/my_models/ode/Models_b.bngl": _ODE,
    "original/bngl_models/my_models/ode/Models_f.bngl": _ODE,
    "original/bngl_models/my_models/ode/Models_n.bngl": _ODE,
    "original/bngl_models/my_models/ssa/PushPull.bngl": _SSA,
    "original/bngl_models/my_models/ssa/PushPull_1.bngl": _SSA,
    # agentCellSystem (ssa/): runs SSA on a COMPLETE network (_SSA_FULL). With the
    # generic max_iter=>3 cap its network is incomplete (25 sp); run_network then
    # on-the-fly-expands it during SSA (→ 41 sp, ragged .cdat) while bngsim keeps
    # the 25-sp fixed network — the two diverge. max_agg=>4 bounds the full network
    # to 57 sp (converges by iter 8, <0.1s), so both engines agree and the .cdat is
    # dense. See RuleWorld/PyBioNetGen#103 for the separate ragged-.cdat loader bug.
    "original/bngl_models/my_models/ssa/agentCellSystem.bngl": _SSA_FULL,
    "slow/rulehub/Published/BaruaBCR2012/BaruaBCR_2012.bngl": _ODE,
    "slow/rulehub/Published/Chattaraj2021/Chattaraj_2021.bngl": _ODE1,
    "slow/rulehub/Published/ChylekFceRI2014/ChylekFceRI_2014.bngl": _ODE,
    "slow/rulehub/Published/ChylekTCR2014/ChylekTCR_2014.bngl": _ODE,
    "slow/rulehub/Published/Kocieniewski2012/Kocieniewski_2012.bngl": _ODE,
    "slow/rulehub/Published/Massole2023/Massole_2023.bngl": _ODE,
    "slow/rulehub/Published/Mukhopadhyay2013/Mukhopadhyay_2013.bngl": _ODE,
    "slow/rulehub/Published/PyBioNetGen/tests/SimpleGenOnly/Simple_GenOnly.bngl": _ODE,
    "slow/rulehub/Published/Zhang2021/Zhang_2021.bngl": _ODE,
    "slow/rulehub/Published/fcerifyn/fceri_fyn.bngl": _ODE,
    "slow/rulehub/Tutorials/NativeTutorials/Chyleklibrary/Chylek_library.bngl": _ODE,
    "slow/rulehub/Tutorials/NativeTutorials/ComplexDegradation/ComplexDegradation.bngl": _ODE,
    "slow/rulehub/Tutorials/NativeTutorials/Suderman2013/Suderman_2013.bngl": _ODE,
    "slow/rulehub/Tutorials/NativeTutorials/visualize/visualize.bngl": _ODE,
    "slow/rulemonkey/tests/cpp/negative_rate_clamp_model.bngl": _ODE,
    "slow/rulemonkey/tests/models/nfsim_basicmodels/v29.bngl": _ODE,
    "slow/rulemonkey/tests/models/nfsim_basicmodels/v30.bngl": _ODE,
    "slow/rulemonkey/tests/models/nfsim_basicmodels/v32.bngl": _ODE,
}


# All basenames any override touches — build_jobs.py uses this to detect a
# basename that matches no vendored model (a stale override) and the inverse.
# (REHAB is keyed by model_id, not basename, so it is checked separately.)
ALL_KEYS = set(TEND_CAP) | set(NSCANPTS) | set(ACTION_INJECT) | set(TOL) | set(TIMEOUT)
