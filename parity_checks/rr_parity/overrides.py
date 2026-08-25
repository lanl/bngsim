"""Per-model parity overrides for the rr_parity (bngsim-vs-RoadRunner) suite.

The SBML analog of ``bng_parity/overrides.py``. Three kinds of authored
deviation-from-default, each keyed by the suite's job key ``"model_id:method"``
(a model can be both an ODE and an SSA job, and an override usually scopes to
one regime — BIOMD19's ODE divergence is unrelated to its SSA behaviour, a
sub-particle SSA artifact says nothing about the ODE run):

  KNOWN_ARTIFACT  — an accepted/genuine cross-engine divergence that is NOT a
                    bngsim defect: either bngsim is the correct engine (RR has
                    the bug) or the divergence is an unavoidable property of the
                    comparison (a degenerate model the ensemble-mean test can't
                    judge cleanly). The runner reclassifies a DIFF on this key to
                    ``Outcome.PASS`` and records the reason in
                    ``JobResult.comment`` — never a silent pass. Each entry MUST
                    cite evidence: which engine is correct and how we know.

  INVALID_REFERENCE — the reference engine (RoadRunner) *runs* (does not raise)
                    but its trajectory is unusable as an oracle — it emits
                    non-finite output (NaN/Inf) over the horizon, so there is no
                    valid reference to compare against. The auto-derived taxonomy
                    keys only on raised-vs-not, so such a job lands in EXCEPTION
                    (mislabeling it an actionable bngsim bug) or, once bngsim can
                    integrate the model, in DIFF (scoring the NaN against bngsim
                    as a divergence); this narrow human disposition reclassifies
                    it to ``Outcome.BAD_TEST`` (no parity signal). The runner
                    applies it only while the premise holds — RR's output is
                    non-finite, and on a DIFF that non-finite output accounts for
                    the whole divergence (#485) — and otherwise flags the entry
                    STALE (recovery stays visible; it can never silently mask a
                    real bngsim bug). ODE-only in practice. Each entry MUST cite
                    evidence (the RR finite fraction, and what settles bngsim's
                    own trajectory: its failure mode, or an independent oracle).
                    The bar is the same as KNOWN_ARTIFACT: this is *not* a place
                    to bury a real bngsim defect — a finite RR reference makes it
                    stale at once, and so does a failing cell in any column RR
                    kept finite.

  NO_ORACLE_ADJUDICATED — a REFERENCE_FAILED row (bngsim ran, RoadRunner refused)
                    whose correctness blind spot was closed by an INDEPENDENT
                    oracle that does not share the reference's failure mode (#117).
                    The outcome stays REFERENCE_FAILED (non-scoring); this records
                    the verdict (confirm / uncoverable) + evidence and flips the
                    row's auto-derived ``reference_refusal`` to a SETTLED class so
                    a future sweep does not re-triage it. Self-stales the moment
                    the row stops being REFERENCE_FAILED. Each entry MUST cite the
                    oracle evidence (see dev/notes/issue117_adjudication.md).

  TOL_OVERRIDES   — a per-model integration tolerance (rtol/atol) applied
                    IDENTICALLY to both engines, for an ill-conditioned IVP where
                    the two engines diverge from *each other* at the shared
                    default tol but converge when it is tightened (the DIFF is a
                    property of the IVP, not either engine). ODE-only; SSA
                    ignores rtol/atol. Mirrors bng_parity's ``TOL``.

The bar for KNOWN_ARTIFACT is deliberately high: marking a real bngsim bug as
PASS-with-reason would hide a regression. A divergence we cannot yet attribute
to a specific engine stays a DIFF (honestly flagged) and goes on the triage
list, NOT here. (Example handled this way: BIOMD0000000019 ODE — all-hOSU=true
across V≠1 compartments with cross-compartment bimolecular reactions; the
v0.9.10 hOSU/V≠1 fix only covered the single-compartment Elementary path, so
this is a *suspected bngsim loader bug*, not an accepted artifact. See
``dev/notes/rr_parity_triage.md``.)

build_ode_jobs.py / build_ssa_jobs.py bake ``overrides_for(model_id, method)``
into each ``Job.overrides`` (first-class ``Override`` records, so the committed
manifest is self-documenting); rr_run.py / rr_golden.py read them back from
``Job.overrides``. This module is the single source of truth — the manifests are
a generated cache of it (regenerate after editing here).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Accepted / genuine divergences — reclassified DIFF -> PASS with the reason.
# Keyed "model_id:method". `issue` is an optional tracker reference.
# --------------------------------------------------------------------------- #
KNOWN_ARTIFACT: dict[str, dict] = {
    # ── ODE artifacts attributed during the 2026-05-29 rr_parity DIFF triage ──
    # (28-REAL ledger, dev/notes/rr_parity_diff_resolution.md §🅰). Each was
    # third-oracle attributed to a non-bngsim cause; none masks a bngsim defect.
    "BIOMD0000000342:ode": {
        "issue": None,
        "reason": (
            "RoadRunner bug (bngsim correct). RR loads init([TGF_beta_ex])=0 "
            "despite the SBML carrying initialConcentration=0.05, and no "
            "initialAssignment / rule / firing event overrides it (the only "
            "event's trigger needs stimulation_type==2, but type=1). bngsim "
            "honors the literal 0.05, so the whole TGF-beta receptor cascade "
            "activates (worst LRC_endo, relmax 1.0); RR leaves it frozen. "
            "bngsim matches the literal SBML and the biological intent."
        ),
    },
    "MODEL2012040002:ode": {
        "issue": None,
        "reason": (
            "Invalid SBML (bngsim spec-compliant). Species mw8449...0055ce is "
            "constant=true yet appears as a reactant — libsbml validation error "
            "20611. Per spec a constant species cannot change, so bngsim freezes "
            "it at 1000 (correct); RR ignores constant= and depletes it to ~155 "
            "(relmax 0.845). Corpus-quality defect, not a bngsim bug."
        ),
    },
    "BIOMD0000000373:ode": {
        "issue": None,
        "reason": (
            "Long-horizon oscillator phase drift (no engine attributable). "
            "reldiv 0.0 at t=5000; only at the full t=7e5 horizon do the curves "
            "separate, and there the bn/rr amplitude ranges match — only the "
            "phase differs (worst 'n', relmax 0.987). A property of a sensitive "
            "limit-cycle IVP, not a loader divergence."
        ),
    },
    "BIOMD0000000376:ode": {
        "issue": None,
        "reason": (
            "Long-horizon oscillator phase drift (no engine attributable). "
            "reldiv 0.0 at t=5000; at full t=6e5 the amplitude ranges match and "
            "only the phase differs (worst 'n', relmax 0.830). Sensitive "
            "limit-cycle IVP, same class as BIOMD0000000373."
        ),
    },
    "BIOMD0000000682:ode": {
        "issue": None,
        "reason": (
            "Oscillator phase drift (no engine attributable). Beta-cell bursting "
            "(HH-style); reldiv 0.0 at t=5000, and at full t=6e5 the min/max are "
            "*identical* [0.00011, 0.1139] between engines — pure phase offset "
            "(worst 'n', relmax 0.755). Not a loader divergence."
        ),
    },
    "MODEL2310250001:ode": {
        "issue": None,
        "reason": (
            "Unstable IVP / integrator behaviour (no engine attributable). BOTH "
            "engines blow up to ~-5e16 and agree within 6% (relmax 0.061, barely "
            "over the 0.05 ceiling, worst _6PG_0). This is integrator behaviour "
            "on an exploding solution, not a loader defect."
        ),
    },
    "BIOMD0000000879:ode": {
        "issue": 72,
        "reason": (
            "RoadRunner grid/tol-dependent on a narrow-pulse forcing (bngsim "
            "correct, oracle-confirmed). Rodrigues2019 chemoimmunotherapy of "
            "CLL: the chemo schedule is a piecewise-in-time assignment rule with "
            "7 infusions each only 0.125 t-units wide (t=0,21,42,63,84,105,126) — "
            "narrow RHS discontinuities an adaptive integrator can step over at a "
            "coarse output grid. bngsim now registers each time threshold as a "
            "root (GH #72) so it resolves every pulse and is grid-independent. "
            "RoadRunner's result at the dt=2 sweep grid instead diverges and "
            "converges to bngsim's when its grid is refined or tol tightened "
            "(OBSERVED behavior — RR source not inspected, so no mechanism is "
            "asserted). Worst cell N at t=42: bngsim 5.118e9 vs RR 1.18e10 (reld "
            "0.334, invariant across rtol 1e-9..1e-11 — NOT a tolerance "
            "artifact). Third-oracle attribution: (1) a segmented SciPy LSODA "
            "integration with the pulse edges as exact breakpoints gives "
            "N(2000)~2.49 / N(42)~5.18e9, the immune-controlled branch bngsim "
            "reproduces; (2) RoadRunner ON A FINE GRID (dt=0.1) matches bngsim to "
            "4+ sig figs (N(42)=5.118e9 both) — i.e. RR's coarse-grid value is "
            "the outlier. bngsim's pre-#72 escape to N~9.6e11 (carrying capacity "
            "k=1e12) was the real defect, now fixed. Tolerance override cannot "
            "help (the RR divergence is grid-, not purely tol-, sensitive). See "
            "dev/notes/rr_parity_triage.md."
        ),
    },
    "MODEL7908934508:ode": {
        "issue": 72,
        "reason": (
            "RoadRunner tol/grid-dependent on a narrow stimulus pulse (bngsim "
            "correct). Reisert2003 olfactory CNG/Ca-CaM model; `kp_act` is a "
            "piecewise-in-time stimulus = 5.5 during (0.1,0.2) and (4.1,4.2), "
            "else 1.6e-5 — two 0.1-wide activation pulses, far narrower than the "
            "invented-horizon dt=1 output grid. With the #72 discontinuity roots "
            "bngsim resolves both pulses at any tol/grid (tol-stable: "
            "CaM4(t=1)=6.05e-6, identical at rtol 1e-9 and 1e-10). RoadRunner at "
            "the dt=2/sweep tol instead returns CaM4(t=1)=2e-17 (worst reld 1.0 "
            "on the near-zero CNG channel) but recovers the SAME 6.05e-6 when its "
            "tol is tightened to 1e-10/1e-16 OR the grid is refined — i.e. RR's "
            "coarse/loose value is the outlier and bngsim matches "
            "RR-at-higher-resolution (OBSERVED; RR source not inspected). Pre-#72 "
            "bngsim also did not resolve the pulses and so PASSed by matching "
            "RR's coarse result; resolving them is the correctness improvement. "
            "TOL_OVERRIDE not used (the DIFF is RR grid-, not purely tol-, "
            "sensitive). See dev/notes/rr_parity_triage.md."
        ),
    },
    "BIOMD0000000627:ode": {
        "issue": 482,
        "reason": (
            "RoadRunner steps over a smooth pre-stimulus ramp (bngsim correct, "
            "oracle-confirmed). Jolivet2015 neuro-glial metabolism: the stimulus is "
            "gated by `is_stimulated = piecewise(0, (time<=200) or (time>=t_0+t_end), "
            "1)`, but the blood-flow multiplier is NOT gated. `f_CBF_dyn` is a "
            "logistic centred at `t_0+t_1-3` = 199, so perfusion (`F_in = "
            "F_0*f_CBF_dyn`) rises 1.0000004 -> 1.4157864 over t=196..200 — one "
            "second BEFORE the switch. The pre-200 trajectory is therefore NOT a "
            "steady state, and species_24 (CO2) is washed out along that ramp. "
            "bngsim resolves it because the #305 crossing stop lands the step on "
            "t=200, which pulls the step size down through the ramp approaching it; "
            "the ramp itself is smooth, so it registers no #72 root and earns no #88 "
            "bound of its own (the model has no floor/ceiling/modulo, so "
            "_periodic_disc_max_step is None — #274 is not involved). RoadRunner at "
            "the sweep grid steps clean over the ramp and reports the flat "
            "pre-stimulus baseline 2.2084924 right through t=200 (worst species_24 at "
            "t=199.6: bngsim 1.9434505 vs RR 2.2084924, reld 0.136). RR's value is "
            "the outlier and gets WORSE as tol tightens: at rtol/atol 1e-11/1e-14 and "
            "1e-12/1e-16 it stays flat through t=200.4 too, then missing part of the "
            "stimulus response as well. Tol-stable wrongness is the signature of a "
            "feature never sampled, and a denser output grid cannot help because "
            "CV_NORMAL interpolates output points (OBSERVED behavior — RR source not "
            "inspected, so no mechanism is asserted). Third-oracle attribution — "
            "three independent converged references, all matching bngsim to 7 digits "
            "at t=197.6/198.8/199.6/200.0/200.4 "
            "(2.2081597/2.1457629/1.9434505/1.9173758/1.9130018): (1) RR with "
            "maximum_time_step=0.05; (2) RR with maximum_time_step=0.01, agreeing "
            "with (1) to 7 digits and so converged; (3) RR with NO step bound at all, "
            "simply restarted at t=190 so its own step control has to resolve the "
            "window. Conversely, disabling the #305 crossing stop in bngsim "
            "reproduces RR's default numbers bit for bit — which is what bngsim <= "
            "0.12.2 did, and why this reads as a 0.13.0 regression when it is the "
            "0.13.0 fix. TOL_OVERRIDE cannot help (tightening tol moves RR the wrong "
            "way). See GH #482."
        ),
    },
    "BIOMD0000000338:ode": {
        "issue": None,
        "reason": (
            "Sharp ill-conditioned transient (no engine attributable). Wajima2009 "
            "coagulation cascade with a t=0 dilution event (compartment_1 x3). The "
            "XIIa activation curve has a narrow spike whose HEIGHT is "
            "engine-independent (peak ~74.2 in every engine/tol: bngsim 74.12, "
            "RoadRunner 74.22 at rtol 1e-10, 74.24 at rtol 5e-11) but whose TIMING "
            "is below the conditioning limit -- it shifts from t=0.170 to t=0.182 "
            "WITHIN RoadRunner alone as rtol tightens 1e-10->5e-11, and within "
            "bngsim it moves with the output grid (bngsim peaks at t=0.121). The "
            "differ compares pointwise, so a ~0.05 t-unit phase shift of a sharp "
            "spike registers as max_rel_err 0.986 (worst XIIa, sweep grid); the "
            "post-transient trajectory (t>=0.3) agrees across engines. Neither "
            "engine integrates tight enough to pin the phase (the transient is "
            "stiff: bngsim hits the CVODE step cap / times out below ~1e-10, RR "
            "raises CV_TOO_MUCH_WORK at the sweep tol), so no high-accuracy "
            "reference can attribute the 'true' timing. Same class as the "
            "oscillator phase-drift entries (BIOMD373/376/682, "
            "MODEL0406553884/0912940495). NOT a loader or Jacobian defect: the #74 "
            "compartment-resize fix is correct (Pk=450/3 post-event, "
            "regression-tested) and the #76 analytical Functional Jacobian matches "
            "a finite-difference Jacobian entrywise to FD precision at physical "
            "states (a default-on speedup, not a wrong result -- jacobian='fd' and "
            "'analytical' both spike). See dev/notes/rr_parity_triage.md."
        ),
    },
    "BIOMD0000000339:ode": {
        "issue": None,
        "reason": (
            "Sharp ill-conditioned transient (no engine attributable) -- the SAME "
            "class as BIOMD0000000338, of which this is the Wajima2009 coagulation-"
            "cascade sibling (same t=0 dilution event, same worst species XIIa). The "
            "XIIa activation spike has an engine-INDEPENDENT height (peak ~74.2: "
            "bngsim 74.20, RoadRunner 74.14 @rtol1e-9) but tol/grid-DEPENDENT timing "
            "(bngsim-analytical t=0.16, bngsim-fd t=0.115, RR t=0.15), and the post-"
            "transient values agree to the digit across engines (t=1.0: bngsim "
            "0.2242 = RR 0.2242; t=2.0: bngsim 0.2246 = RR 0.2246). The differ "
            "compares pointwise, so the phase shift of a sharp spike on the coarse "
            "sweep grid reads as max_rel_err 0.957 (worst XIIa). Unlike #338, no "
            "Jacobian strategy happens to phase-align with RR on that grid "
            "(jacobian=fd and =analytical both DIFF ~0.96), but the divergence is "
            "the same artifact, not a distinct defect -- the equal peak height and "
            "the digit-identical post-transient tail rule out a real divergence. "
            "Both engines are stiff (RR raises CV_TOO_MUCH_WORK at rtol1e-10), so no "
            "high-accuracy reference can pin the phase. NOT a loader/Jacobian defect. "
            "Repro dev/investigations/peak_339.py; see the #338 entry + "
            "dev/notes/rr_parity_triage.md."
        ),
    },
    # NOTE: MODEL1708310001:ode is the #88 root fix — the loader now bounds the
    # integrator step below the periodic floor()/modulo dose window, so bngsim is
    # tol-stable at the segmented oracle (953.07). The residual RR-only step-over
    # at the sweep tol is a TOL_OVERRIDES entry (both engines agree at 1e-11/1e-16).
    #
    # NOTE: the BIOMD0000000002:ssa KNOWN_ARTIFACT entry (sub-particle degenerate
    # Edelstein AChR, bngsim->0 vs RR fractional) was removed in #23: the model is
    # no longer in the SSA-admitted set (ssa_candidates.json), so it builds no SSA
    # job and the override keyed nothing (stale_keys flagged it). Its exclusion
    # rationale now lives in the validation pipeline; see git history for the text.
}

# --------------------------------------------------------------------------- #
# Reference engine ran but produced no usable trajectory (non-finite output) ->
# reclassified to BAD_TEST. Keyed "model_id:method". `issue` is optional.
# Applied only while the premise holds (RR non-finite, and on a DIFF only where
# the non-finite columns account for every surviving failure); a finite RR
# reference, or a failure in a column RR kept finite, makes the entry STALE.
# --------------------------------------------------------------------------- #
INVALID_REFERENCE: dict[str, dict] = {
    "MODEL2002070001:ode": {
        "issue": 485,
        "reason": (
            "No valid reference: RoadRunner runs without raising but its "
            "trajectory is non-finite over the horizon (101/707 = 14.3% finite "
            "cells, re-verified 2026-08-25 against RR 2.9.2), so it is not a "
            'usable oracle. Both compartments declare size="NaN"; every '
            "species is hasOnlySubstanceUnits=true and no <math> block in the "
            "file names either compartment, so the IVP is well posed in amounts "
            "and the size is inert — but RR carries species as concentrations, "
            "so x1..x6 (the species in those compartments) are NaN from t=0 "
            "while x7, the one rate-rule variable that is not a species, stays "
            "finite and agrees with bngsim cell for cell. The NaN reaches RR's "
            "state through the compartment, not through the dynamics. bngsim "
            "used to fail here too (CVODE flag=-4, error test failed repeatedly "
            "/ |h|=hmin, at t~=0.0134); since the #353 unit-volume substitution "
            "for a non-finite size it integrates the model (707/707 finite) and "
            "AMICI 1.0.1 confirms that trajectory independently — 0/707 cells "
            "failing, max_rel_err 0.0 at the sweep tolerance. So the half of "
            'this entry\'s original premise that read "bngsim also failed" is '
            "gone, and the half that matters is not: there is still no reference "
            "trajectory to compare against, which is BAD_TEST (no parity signal) "
            "rather than the DIFF the taxonomy would record — a DIFF would score "
            "RR's NaN as a bngsim divergence. NOT reclassified to PASS, which "
            "would credit RR with validating a model it answered NaN on. If RR "
            "ever returns a finite trajectory here, or any surviving failure "
            "lands in a column RR kept finite, the override goes stale and the "
            "natural verdict — including any real bngsim bug — resurfaces."
        ),
    },
}

# --------------------------------------------------------------------------- #
# Per-model integration tolerance (applied to BOTH engines). ODE-only.
#   value: (rtol, atol). Each with a mandatory reason.
# --------------------------------------------------------------------------- #
TOL_OVERRIDES: dict[str, dict] = {
    # ── Tolerance artifacts: both engines diverge at the shared sweep tol
    # (1e-9/1e-12) but CONVERGE when tightened to 1e-10/1e-16 (the divergence is
    # a property of the ill-conditioned IVP at the loose default, not either
    # engine). The peak-relative gate sees a significant column past the ceiling
    # at the sweep tol; tightening collapses it to zero real columns -> PASS.
    # Verified per model: n_real(sweep) > 0, n_real(1e-10/1e-16) == 0. The
    # documented `eco_coevolution_host_parasite` analog from bng_parity.
    "BIOMD0000000374:ode": {
        "rtol": 1e-10,
        "atol": 1e-16,
        "reason": (
            "Tolerance artifact (both engines converge when tightened). At the "
            "shared sweep tol the calcium oscillator's 'n'/'V_membrane' diverge "
            "~9% peak-relative (worst reldiv 0.093); at 1e-10/1e-16 (applied to "
            "BOTH engines) the divergence collapses below the gate (zero real "
            "columns) -> PASS. Property of the stiff limit-cycle IVP, not a "
            "loader divergence."
        ),
    },
    "BIOMD0000000876:ode": {
        "rtol": 1e-10,
        "atol": 1e-16,
        "reason": (
            "Tolerance artifact (both engines converge when tightened). "
            "'V_Virus' diverges ~13% peak-relative at the sweep tol (worst "
            "reldiv 0.132); at 1e-10/1e-16 (both engines) zero real columns "
            "remain -> PASS. Ill-conditioned IVP at the loose default."
        ),
    },
    "BIOMD0000000951:ode": {
        "rtol": 1e-10,
        "atol": 1e-16,
        "reason": (
            "Tolerance artifact (both engines converge when tightened). "
            "Coagulation cascade: 'mIIa' + 6 other columns diverge up to ~8% "
            "peak-relative at the sweep tol (worst reldiv 0.083); at 1e-10/1e-16 "
            "(both engines) zero real columns remain -> PASS. Stiff IVP."
        ),
    },
    # MODEL0913003363:ode — RETIRED 2026-08-25 as a stale key (GH #482 follow-up).
    # The model is `drop,no_model` in the stage-1 corpus filter
    # (benchmarks/suites/biomodels/manifest.csv): its fetched file is zero bytes
    # (sha256 = the empty string's, e3b0c442...b855), one of 305 corpus entries in
    # that state. build_ode_jobs.py therefore builds no job for it and the key
    # could never match — `stale_keys` has flagged it since the initial public
    # release. Recorded here rather than deleted outright, so the disposition is
    # not re-derived if the corpus ever fetches the model successfully:
    # tolerance artifact, both engines converge when tightened. 'm' + 3 other
    # columns diverged sharply at the sweep tol (worst reldiv 0.951, an early
    # transient); at rtol/atol 1e-10/1e-16 on BOTH engines zero real columns
    # remained -> PASS. Ill-conditioned transient at the loose default.
    "BIOMD0000000827:ode": {
        "rtol": 1e-10,
        "atol": 1e-16,
        "reason": (
            "Tiny-magnitude solution under-resolved by the sweep atol (not a "
            "bngsim defect). Ito2019 gefitinib-resistance model: every species "
            "stays at ~1e-12 (max 7.4e-12, min 3.7e-17), so the shared sweep "
            "atol=1e-12 is coarser than the solution itself and bngsim's CVODE "
            "raises CV_CONV_FAILURE at t=10 (analytical AND fd Jacobian both fail "
            "-- not a #76 issue). RoadRunner tolerates the coarse atol and "
            "integrates; at a scale-appropriate tol (1e-10/1e-16 applied to BOTH "
            "engines) bngsim integrates too and the two agree to max_rel=0 (10 "
            "species). A per-model tolerance disposition, mirroring the other "
            "tolerance-artifact entries; a scale-aware default atol would obviate "
            "it (separate possible enhancement)."
        ),
    },
    "MODEL1708310001:ode": {
        "rtol": 1e-11,
        "atol": 1e-16,
        "reason": (
            "Periodic floor()/modulo dosing — RoadRunner resolves the dose "
            "pulses only at a tighter tol (#88). Claret2009 colorectal-cancer "
            "OS: a single exponentially-growing tumour size y under a periodic "
            "chemo schedule (nested piecewise over floor()-based cycle/day "
            "arithmetic); the dose windows are 0.0625 days wide. The exact "
            "segmented answer is y(100)=953.07 (closed-form per-segment "
            "integration, confirmed by fine LSODA). The bngsim loader now bounds "
            "the integrator step below the dose-window width (the auto "
            "max_step_size for a periodic floor/modulo schedule), so bngsim is "
            "tol-STABLE at 953.07 for either Jacobian at EVERY tol (1e-9..1e-12). "
            "RoadRunner steps over the same pulses at the sweep tol (y(100)=1570 "
            "at 1e-9, 1534.8 at 1e-10) and only resolves them — at the same "
            "fixed grid — once tightened to 1e-11 (953.07). At 1e-11/1e-16 "
            "applied to BOTH engines the full trajectory agrees to max_rel=0 "
            "(0/101 columns) -> PASS. A tolerance disposition, not a bngsim "
            "defect (bngsim is the tol-stable, oracle-exact engine here)."
        ),
    },
    # ── Ill-conditioned IVP held non-PASS through the peak-relative gate ──
    "BIOMD0000000375:ode": {
        "rtol": 1e-11,
        "atol": 1e-16,
        "reason": (
            "Stiff / ill-conditioned IVP — held non-PASS, not an accepted "
            "divergence. At the shared sweep tol (1e-9/1e-12) both engines run "
            "but bngsim hits the CVODE step cap (CV_TOO_MUCH_WORK, unconverged) "
            "and the residual is a 1.58% peak-relative disagreement on a "
            "significant column — below the differ's peak-relative gate, so a "
            "bare gate would silently PASS an unconverged solve. Tightening to "
            "1e-11/1e-16 (applied to BOTH engines) honestly surfaces the "
            "ill-conditioning: RoadRunner raises (cannot integrate it that "
            "tight) -> EXCEPTION, the disposition the 28-REAL ledger records for "
            "this model ('honest EXCEPTION; not a loader bug'). Not a "
            "KNOWN_ARTIFACT — no engine is attributed as correct."
        ),
    },
}


# --------------------------------------------------------------------------- #
# NO_ORACLE adjudications — a REFERENCE_FAILED row settled by an INDEPENDENT
# hand-rolled oracle, NOT by RoadRunner.
# --------------------------------------------------------------------------- #
# REFERENCE_FAILED (bngsim ran, RR raised) is auto-derived and non-scoring, and
# its `reference_refusal` sub-classes (integrator / feature_gap / recursive /
# other) are "oracle-unverified, triage-worthy" — a correctness blind spot
# (#117). When that blind spot is closed by an independent oracle that does NOT
# share the reference's failure mode (a scale-appropriate scipy stiff solver, or
# a model-specific scipy-literal integrator — see
# `dev/investigations/oracle_117.py` + `dev/notes/issue117_adjudication.md`),
# the verdict is recorded HERE so a future sweep marks the row *settled* instead
# of re-triaging it. This does NOT change the auto-derived outcome (still
# REFERENCE_FAILED, non-scoring) — it sets `reference_refusal` to
# `adjudicated_<verdict>` (a SETTLED class) and records the evidence in the
# comment, never a silent pass.
#
# verdict ∈ {"confirm", "uncoverable"}:
#   confirm     — the independent oracle reproduces bngsim within the suite's
#                 deterministic_verdict (bngsim is correct where verifiable).
#   uncoverable — no ground truth exists (malformed model / irreducibly
#                 ill-conditioned); recorded so it is not re-investigated.
#
# Self-staling: the disposition holds ONLY while the row is genuinely
# REFERENCE_FAILED. If RoadRunner later integrates the model (the row becomes
# PASS/DIFF) or bngsim's status changes (EXCEPTION/BAD_TEST), rr_run flags the
# entry STALE in the comment — the recovery (or a newly-surfaced bngsim defect)
# can never be silently masked. Each entry MUST cite its oracle evidence.
NO_ORACLE_ADJUDICATED: dict[str, dict] = {
    "BIOMD0000000250:ode": {
        "verdict": "confirm",
        "issue": "GH #117",
        "reason": (
            "RR CVODE CV_CONV_FAILURE (stiff) + COPASI failed. Independent scipy "
            "BDF reproduces all 47 species to max_rel=0 (worst per-species "
            "peak-rels ~5e-3 sit on species at peak ~1e-30, i.e. numerical zero "
            "below the significance floor). bngsim confirmed."
        ),
    },
    # MODEL1006230083:ode — RETIRED 2026-08-25 as a stale key (GH #482 follow-up),
    # same cause as MODEL6963432821 below and MODEL0913003363 in TOL_OVERRIDES:
    # `drop,no_model` in the stage-1 corpus filter, so no job is built and the key
    # can never match. The GH #117 adjudication it carried, kept for the record:
    # RR CVODE CV_TOO_MUCH_WORK + COPASI failed; pure rate-rule ODE system on 5
    # parameters (no reactions); independent scipy LSODA/BDF reproduce to
    # max_rel=0 (peak-rel ~1e-8); verdict confirm — bngsim confirmed.
    "MODEL1112050001:ode": {
        "verdict": "confirm",
        "issue": "GH #117",
        "reason": (
            "Liu2009_GlucoseMobilization, 28 rate-rule species; bngsim integrates "
            "to ±1e15 but is CORRECT: bngsim, RoadRunner and scipy all agree "
            "max_rel=0 for t<=3e-7. Stiffness wall at t~3.36e-7 (same SUNDIALS "
            "CVODE as RR, t+h=t step underflow; bngsim continues, RR aborts "
            "CV_TOO_MUCH_WORK). Not a pole: scipy BDF leaps it (max_rel=0 to "
            "t=1.0) and reproduces the blow-up scale to t=100 (bngsim 6.79e15 vs "
            "BDF 6.77e15). Past t~10 the system is exponentially ill-conditioned "
            "(~e^{135t}) so two correct solvers diverge on fine structure while "
            "agreeing on scale — intrinsic to the model on the invented t_end=100, "
            "not a bngsim defect. bngsim confirmed where verifiable."
        ),
    },
    "MODEL2001200002:ode": {
        "verdict": "confirm",
        "issue": "GH #117",
        "reason": (
            "RR CVODE CV_CONV_FAILURE + COPASI failed. 7-species reaction net with "
            "pow(L/T,1.36) Hill terms; species underflow (1e-33..1e-41) makes a "
            "stiff solver's trial step overshoot negative -> pow(neg,1.36)=NaN. "
            "With species floored at 1e-100 in the RHS (60 orders below the "
            "smallest physical value), all three independent solvers (Radau, BDF, "
            "LSODA) agree with bngsim to max_rel=0 (peak-rel ~2e-7). bngsim "
            "confirmed."
        ),
    },
    # MODEL6963432821:ode — RETIRED 2026-08-25 as a stale key (GH #482 follow-up),
    # same cause as MODEL1006230083 above: `drop,no_model` in the stage-1 corpus
    # filter, so no job is built and the key can never match. The GH #117
    # adjudication it carried, kept for the record: RR CVODE CV_TOO_MUCH_WORK +
    # COPASI failed; pure rate-rule ODE system on 7 parameters; independent scipy
    # LSODA/BDF reproduce to max_rel=0 (peak-rel ~4e-9); verdict confirm — bngsim
    # confirmed.
    "MODEL9811206584:ode": {
        "verdict": "confirm",
        "issue": "GH #117",
        "reason": (
            "RR CVODE CV_TOO_MUCH_WORK + COPASI failed. Pure rate-rule ODE system "
            "on 4 parameters. Independent scipy BDF reproduces to max_rel=0 "
            "(peak-rel ~1e-9). bngsim confirmed."
        ),
    },
    # MODEL2205030001:ode — RETIRED 2026-06-09 when GH #119 landed. The model is
    # malformed (all 23 initial assignments reference undefined symbols; 16 Kd_*
    # constants dropped on export). It was adjudicated UNCOVERABLE while bngsim
    # fail-opened (silently dropping the IAs and running) — a REFERENCE_FAILED row
    # settled here pending #119. With #119 fixed, bngsim now REFUSES the model at
    # load (undefined-symbol guard), so RR refusal + bngsim refusal makes the row
    # auto-classify as BAD_TEST (both engines correctly decline a malformed model),
    # self-documented by the engines' exception strings. No override is needed for
    # that natural classification, so the entry is removed rather than left to stale.
}


def overrides_for(model_id: str, method: str) -> list[dict]:
    """All overrides authored for ``model_id:method`` as {field, value, reason} dicts.

    ``field="tol"`` -> ``value={"rtol":..,"atol":..}`` (both engines);
    ``field="known_artifact"`` -> ``value={"issue":..}`` (reason in ``reason``);
    ``field="invalid_reference"`` -> ``value={"issue":..}`` (reason in ``reason``);
    ``field="no_oracle_adjudicated"`` -> ``value={"issue":..,"verdict":..}`` (reason
    in ``reason``) — annotates a REFERENCE_FAILED row settled by an independent
    oracle (verdict confirm/uncoverable), marking its refusal class SETTLED.
    The order is tol-first so a tol override is applied before the run and a
    post-run reclassification (known_artifact -> PASS, invalid_reference ->
    BAD_TEST) after it. The reclassifiers never co-occur with each other or with
    tol today, but the ordering keeps the contract unambiguous.
    """
    key = f"{model_id}:{method}"
    out: list[dict] = []
    if key in TOL_OVERRIDES:
        d = TOL_OVERRIDES[key]
        out.append(
            {
                "field": "tol",
                "value": {"rtol": d["rtol"], "atol": d["atol"]},
                "reason": d["reason"],
            }
        )
    if key in KNOWN_ARTIFACT:
        d = KNOWN_ARTIFACT[key]
        out.append(
            {"field": "known_artifact", "value": {"issue": d.get("issue")}, "reason": d["reason"]}
        )
    if key in INVALID_REFERENCE:
        d = INVALID_REFERENCE[key]
        out.append(
            {
                "field": "invalid_reference",
                "value": {"issue": d.get("issue")},
                "reason": d["reason"],
            }
        )
    if key in NO_ORACLE_ADJUDICATED:
        d = NO_ORACLE_ADJUDICATED[key]
        out.append(
            {
                "field": "no_oracle_adjudicated",
                "value": {"issue": d.get("issue"), "verdict": d["verdict"]},
                "reason": d["reason"],
            }
        )
    return out


# Every key any override touches — build scripts use this to detect a stale
# entry (a key matching no vendored job).
ALL_KEYS = (
    set(KNOWN_ARTIFACT) | set(INVALID_REFERENCE) | set(TOL_OVERRIDES) | set(NO_ORACLE_ADJUDICATED)
)


def stale_keys(model_ids: set[str], method: str) -> list[str]:
    """Override keys for ``method`` whose model_id is NOT in ``model_ids``.

    A non-empty result means an override names a model that the manifest for
    this regime doesn't build — a stale entry to fix (mirrors bng_parity's
    build-time stale-override detection).
    """
    return sorted(
        k for k in ALL_KEYS if k.endswith(f":{method}") and k.rsplit(":", 1)[0] not in model_ids
    )
