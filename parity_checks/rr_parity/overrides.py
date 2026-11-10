"""Per-model parity overrides for the rr_parity (bngsim-vs-RoadRunner) suite.

The SBML analog of ``bng_parity/overrides.py``. Five kinds of authored
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
                    oracle that does not share the reference's failure mode.
                    The outcome stays REFERENCE_FAILED (non-scoring); this records
                    the verdict (confirm / uncoverable) + evidence and flips the
                    row's auto-derived ``reference_refusal`` to a SETTLED class so
                    a future sweep does not re-triage it. Self-stales the moment
                    the row stops being REFERENCE_FAILED. Each entry MUST cite
                    oracle evidence that runs from this repository: a ``confirm``
                    entry is re-derived by ``scipy_oracle.py`` in
                    ``parity_checks/tests/test_no_oracle_adjudications.py``
                    (#1559), and an ``uncoverable`` one by its own test.

  NO_GROUND_TRUTH — both engines ran, and over the job's horizon the model has no
                    solution that a double-precision integrator can follow: no
                    tolerance makes either engine agree with ITSELF, so neither
                    answer is an oracle and their agreement or disagreement says
                    nothing about either. A DIFF or a PASS on this key is
                    reclassified to ``Outcome.BAD_TEST`` (non-scoring), with the
                    reason in ``JobResult.comment`` under the label "no ground
                    truth". Not a KNOWN_ARTIFACT, which scores PASS and so claims
                    bngsim was checked; not a DIFF, which claims it diverged from
                    a valid oracle. A DIFF whose metric is not finite (a grid or
                    species mismatch, or non-finite output: what a loader
                    regression looks like) keeps its scoring DIFF, and any other
                    outcome is left as it is (#1565's rule, from the AMICI suite's
                    NO_GROUND_TRUTH). Each entry MUST cite the convergence study
                    that shows it, run from this repository by
                    ``parity_checks/tests/test_rr_known_artifact_evidence.py``.

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
it was held as a *suspected bngsim loader bug*, not an accepted artifact. It
passes on its own on the nightly now. The triage note this paragraph used to
cite was never in this repository.)

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
    # That triage's ledger and notes were never committed to this repository, so
    # the entries it left are re-derived here from evidence that runs from it.
    # BIOMD0000000879 and MODEL7908934508 survive, each now settled by RoadRunner
    # run so that it cannot step over a pulse (a bounded step, and for 879 a
    # restart at every pulse edge). BIOMD0000000373, 376, 682 and MODEL2310250001
    # moved to NO_GROUND_TRUTH on 2026-11-10: no tolerance makes either engine
    # agree with itself over their horizons, so their PASS checked nothing.
    # parity_checks/tests/test_rr_known_artifact_evidence.py runs all six.
    #
    # BIOMD0000000342:ode and MODEL2012040002:ode — RETIRED 2026-11-02 (GH #1288),
    # stale since libRoadRunner 2.11.0 (#1349) and the reference-data-v4 record
    # (#1443): RoadRunner now gets both right on its own (max_rel_err 1.0 -> 0 and
    # 0.85 -> 0), and the nightly flags both "STALE known-artifact entry: passes
    # on its own now" (run 41499451003). What they recorded: on 2.9.2 RR loaded
    # init([TGF_beta_ex]) = 0 in BIOMD342 despite initialConcentration=0.05 (bngsim
    # honored the literal SBML), and ignored constant=true on MODEL2012040002's
    # species mw8449...0055ce, depleting it to ~155 where bngsim held it at 1000.
    "BIOMD0000000879:ode": {
        "issue": 72,
        "reason": (
            "RoadRunner steps over narrow dose pulses (bngsim correct, "
            "oracle-confirmed). Rodrigues2019 chemoimmunotherapy of CLL: "
            "Chemotherapy_Input, a piecewise-in-time assignment rule, is 8640 during "
            "seven infusions 0.125 wide (t = 0, 21, 42, 63, 84, 105, 126) and 0 "
            "otherwise. bngsim registers each threshold as a root (GH #72) and "
            "resolves every pulse: N(42) = 5.11805e9 at every rtol from 1e-9 to "
            "1e-12. RoadRunner as rr_run runs it gives N(42) = 1.1795e10 at the "
            "same tolerances (max_rel_err 0.334); a wrong value that tightening "
            "does not move is a feature never sampled. Oracle: RoadRunner restarted "
            "at every pulse edge, so no pulse falls inside one integration, with "
            "maximum_time_step 0.05 and 0.01 at the suite's tolerance. Each "
            "segment ends one ulp short of its edge: `time >= 21` already holds AT "
            "21, and a bounded RoadRunner that must end a step there stalls "
            "(CV_TOO_MUCH_WORK). The two references agree (err/movement 5.0e-7), "
            "bngsim passes against both (1.2e-6 and 7.9e-7), and RoadRunner's own "
            "run fails against them. parity_checks/tests/"
            "test_rr_known_artifact_evidence.py runs this. The entry's earlier "
            "evidence was a triage note never committed to this repository."
        ),
    },
    "MODEL7908934508:ode": {
        "issue": 72,
        "reason": (
            "RoadRunner steps over narrow stimulus pulses (bngsim correct, "
            "oracle-confirmed). Reidl2006 calcium oscillation in olfactory cilia: "
            "kp_act, a piecewise-in-time assignment rule, is 5.5 during (0.1, 0.2) "
            "and (4.1, 4.2) and 1.6e-5 otherwise, on an invented dt = 1 output "
            "grid. bngsim registers the thresholds as roots (GH #72) and resolves "
            "both pulses. RoadRunner as rr_run runs it steps over them: CaM4(1) = "
            "2.0e-17 where the solution is 6.05e-6 (max_rel_err 1.0). Oracle: "
            "RoadRunner with maximum_time_step 0.05 and 0.01, each under half a "
            "pulse, at the suite's tolerance. The two agree (err/movement 7.2e-5), "
            "bngsim passes against both (1.1e-3 and 9.8e-4; CaM4(1) = 6.0487e-6 "
            "against 6.0492e-6), and RoadRunner's own run fails against them. At "
            "rtol 1e-12, atol 1e-18 the bounds 0.01, 0.002 and 0.0005 agree to "
            "1e-5 (CaM4(1) = 6.04919e-6), and bngsim at that tolerance matches. "
            "parity_checks/tests/test_rr_known_artifact_evidence.py runs this. The "
            "entry's earlier evidence was a triage note never committed to this "
            "repository."
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
    # BIOMD0000000338 and 339 — GH #1304. Until #1485 (PR #1535) bngsim's dense
    # solver left the same exact 0 by rounding, and these two entries described a
    # spike every engine shared as "no engine attributable". bngsim no longer
    # spikes on any route; the entries now say who leaves the 0 and how we know.
    "BIOMD0000000338:ode": {
        "issue": 1304,
        "reason": (
            "RoadRunner leaves an exact solution by rounding (bngsim correct). "
            "Wajima2009 aPTT test. XIIa, K and CA start at 0; each of the six "
            "reactions that change them is proportional to one of the three, and "
            "no rule, initial assignment or event sets them. So XIIa = K = CA = 0 "
            "for all t is the exact solution. It is unstable: the XIIa <-> K loop "
            "grows any departure. bngsim keeps it exactly, on KLU and (since GH "
            "#1485) on the dense solver, with either Jacobian: Fg(2) = 310.989 at "
            "rr_run's rtol 1e-9, atol 1e-12. RoadRunner's rounding tips XIIa off 0, "
            "upward or downward depending on its build, tolerance and output grid. "
            "Upward, XIIa spikes and the cascade clots: at rr_run's tolerances, RR "
            "2.9.2 on macOS arm64 peaks at XIIa = 72.8 (t = 0.14) and ends at "
            "Fg(2) = 0.7365, and the row is this DIFF, max_rel_err 1.0. Downward, "
            "XIIa goes below 0, RoadRunner's step collapses, and the row is "
            "REFERENCE_FAILED, CV_TOO_MUCH_WORK: so with RR 2.11.0 at the same "
            "tolerances, on the Linux nightly (run 42676740345) and on macOS "
            "arm64, where a run on the job's grid stops at t = 0.097 with XIIa = "
            "-2.0e-7 and K = -7.0e-8 (which way XIIa goes on Linux is inferred "
            "from that). Oracle: RoadRunner on the same SBML with XIIa, K and CA "
            "made constant at 0, which it cannot leave, matches bngsim's run of "
            "the model as written on all 55 species, KLU and dense alike (PASS, "
            "error/movement 9.3e-9 with RR 2.9.2 and 9.6e-9 with 2.11.0, at rtol "
            "1e-9, atol 1e-12). python/tests/test_coagulation_exact_zero.py runs "
            "that oracle."
        ),
    },
    "BIOMD0000000339:ode": {
        "issue": 1304,
        "reason": (
            "RoadRunner leaves an exact solution by rounding (bngsim correct), as "
            "on BIOMD0000000338. Wajima2009 PT test: TF = 300 starts the extrinsic "
            "path, and XIIa = K = CA = 0 is again an exact, unstable solution (the "
            "same six reactions, each proportional to one of the three). bngsim "
            "keeps it exactly on both linear solvers and either Jacobian: Fg(2) = "
            "0.611371 at every rtol from 1e-6 to 1e-12. At rr_run's rtol 1e-9, "
            "atol 1e-12, RR 2.9.2 on macOS arm64 goes downward: XIIa passes r4's "
            "pole at -1 (-150 when it stops), the step collapses, and the row is "
            "REFERENCE_FAILED, CV_TOO_MUCH_WORK. RR 2.11.0 goes upward and "
            "completes: on macOS arm64 XIIa peaks at 71.4 (t = 0.10) and Fg(2) = "
            "0.611052, and on the Linux nightly the row is this DIFF, max_rel_err "
            "1.0 (run 42676740345; which way XIIa goes there is inferred from the "
            "arm64 run). Oracle: RoadRunner on the SBML with XIIa, K and CA made "
            "constant at 0 matches bngsim's run of the model as written on all 55 "
            "species, KLU and dense alike (PASS, error/movement 8.4e-9 with RR "
            "2.9.2 and 7.7e-9 with 2.11.0, at rtol 1e-9, atol 1e-12). "
            "python/tests/test_coagulation_exact_zero.py runs that oracle."
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
    #
    # NOTE: BIOMD0000000570, 814, 816, 817, 829, 928, 994, 995, 996 and 1059 (ode)
    # were known artifacts for a while after #1015 (#1021): RoadRunner skipped
    # their `time == c` and open-window events between output samples, which
    # bngsim fires. rr_run now runs RoadRunner through those instants
    # (_rr_common.event_instants), so it fires them too and the comparison is
    # real again; all 10 pass on their own. See git history for the entries.
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
        "rtol": 1e-12,
        "atol": 1e-16,
        "reason": (
            "Tolerance artifact (both engines converge when tightened). At the "
            "shared sweep tol the calcium oscillator's 'n'/'V_membrane' diverge "
            "~9% peak-relative (worst reldiv 0.093). At 1e-10/1e-16 that is gone "
            "from the magnitude gates, but one spike of 'n' (t=36240) still differs "
            "by 1.2e-2 of its range, past the #873 movement ceiling: RoadRunner is "
            "1.0e-2 off its own 1e-12 run there, bngsim 1.4e-3. At 1e-12/1e-16 (both "
            "engines) the worst cell is 2.2e-4 of the range -> PASS. Property of the "
            "stiff limit-cycle IVP, not a loader divergence."
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
# other) are "oracle-unverified, triage-worthy" — a correctness blind spot.
# When that blind spot is closed by an independent oracle that does NOT share the
# reference's failure mode (`scipy_oracle.py`: the right-hand side built from the
# SBML text with libsbml and integrated by scipy, which shares no code with
# bngsim or SUNDIALS), the verdict is recorded HERE so a future sweep marks the row *settled* instead
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
# can never be silently masked. Each entry MUST cite oracle evidence that runs
# from this repository: `parity_checks/tests/test_no_oracle_adjudications.py`
# re-derives every `confirm` entry with `scipy_oracle.py` and fails on a confirm
# entry it has no oracle run for (#1559).
NO_ORACLE_ADJUDICATED: dict[str, dict] = {
    # BIOMD0000000250:ode and MODEL2001200002:ode were confirmed first under
    # "GH #117", an unrelated issue here, on evidence files that were never in
    # this repository. #1559 re-derived both with scipy_oracle.py; the reasons
    # below are what test_no_oracle_adjudications.py checks.
    "BIOMD0000000250:ode": {
        "verdict": "confirm",
        "issue": "GH #1559",
        "reason": (
            "RoadRunner (CV_CONV_FAILURE at t~1.25) and COPASI both fail. "
            "Nakakuki2010, 47 species and 78 reactions, Hill exponents 3.97 and "
            "3.58 on ppERK_n and FOSn_2, 36 species starting at 0. scipy on a "
            "right-hand side built from the SBML text (scipy_oracle.py), which "
            "equals RoadRunner's own rates at t=0 and at a random state: Radau and "
            "BDF at rtol 1e-11, atol 1e-25 agree (err/movement 1.1e-11), and "
            "bngsim at the suite tolerance passes against both on all 47047 cells "
            "(err/movement 2.9e-9). Every species below the suite's atol has "
            "df_i/dx_i <= 0, so none is held on a repelling equilibrium (#1542). "
            "bngsim confirmed."
        ),
    },
    # MODEL1006230083:ode — RETIRED 2026-08-25 as a stale key (GH #482 follow-up),
    # same cause as MODEL6963432821 below and MODEL0913003363 in TOL_OVERRIDES:
    # `drop,no_model` in the stage-1 corpus filter, so no job is built and the key
    # can never match. The adjudication it carried, kept for the record (it
    # cited "GH #117", an unrelated issue here, and evidence files that were never
    # in this repository, #1559):
    # RR CVODE CV_TOO_MUCH_WORK + COPASI failed; pure rate-rule ODE system on 5
    # parameters (no reactions); independent scipy LSODA/BDF reproduce to
    # max_rel=0 (peak-rel ~1e-8); verdict confirm — bngsim confirmed.
    # MODEL1112050001:ode — re-adjudicated by GH #1542 from "confirm". The earlier
    # entry cited "GH #117", which on lanl/bngsim is an unrelated issue. Its reason
    # said scipy BDF "leaps" the stiffness wall near t = 3.36e-7 and took that as
    # proof the wall is not a pole. The model does overflow there (x13/x14, below),
    # so a run that continues past it is not following the solution. The AMICI
    # suite records the same finding as a NO_GROUND_TRUTH disposition.
    "MODEL1112050001:ode": {
        "verdict": "uncoverable",
        "issue": "GH #1542",
        "reason": (
            "Liu2009_GlucoseMobilization, 28 rate-rule species, has no solution in "
            "double precision on the invented t_end=100. x8's rate rule drains it at "
            "the constant k6*PTP*k8 (k8 = 7.06e12; it reads like a typo for x8), so "
            "x4+x5 = -3.42e8*t^2 early, and x9' = k22*PTP*x10 - k7*x9*(x4+x5)/IRp "
            "(IRp = 8.97e-13) grows x9 as 1e-12*exp(5.29e20*t^3), which scipy DOP853 "
            "on the SBML text reproduces at t=1e-7. k9 reads x12/PI3K with PI3K = "
            "5e-15 and x12 follows -x9, so x13 and x14 reach -/+2.6e10 by "
            "t=3.363e-7 (scipy BDF; bngsim at atol 1e-24 agrees) and grow "
            "super-exponentially: RoadRunner's CV_TOO_MUCH_WORK at t~3.36e-7 is "
            "that. At the sweep's atol of 1e-12, x9 starts at the tolerance, so "
            "bngsim (like AMICI) holds it on its repelling equilibrium and runs on "
            "to t=100 along a trajectory that is not a solution. The earlier "
            "adjudication confirmed bngsim on that trajectory; nothing past "
            "t~3.36e-7 can be confirmed."
        ),
    },
    "MODEL2001200002:ode": {
        "verdict": "confirm",
        "issue": "GH #1559",
        "reason": (
            "RoadRunner (CV_CONV_FAILURE at t~53) and COPASI both fail. "
            "Mpekris2017, 7 species, D = 1.43*T*(L/T)^1.36/(2.73 + (L/T)^1.36): T, "
            "Csc, I and L decay to 1e-34..1e-42, where a trial step that goes "
            "negative makes (L/T)^1.36 NaN. scipy in log coordinates "
            "(scipy_oracle.py; x = exp(u) cannot go negative, so no floor or clamp "
            "changes the model), on a right-hand side that equals RoadRunner's own "
            "rates: explicit DOP853 and BDF agree (err/movement 1.2e-8), and bngsim "
            "at the suite tolerance passes against both on all 707 cells "
            "(err/movement 1.7e-8). bngsim's L is below 0 from t=23 on (78 of 101 "
            "rows, down to -8.5e-10), where the solution is positive and below "
            "1.7e-15: 8.5e-10 of L's peak of 1. Every species below the suite's "
            "atol has df_i/dx_i <= -0.76, so none is held on a repelling "
            "equilibrium (#1542). bngsim confirmed. The earlier evidence floored "
            "species at 1e-100 in the right-hand side, which changes the model."
        ),
    },
    # MODEL6963432821:ode — RETIRED 2026-08-25 as a stale key (GH #482 follow-up),
    # same cause as MODEL1006230083 above: `drop,no_model` in the stage-1 corpus
    # filter, so no job is built and the key can never match. The adjudication it
    # carried, kept for the record (cited as "GH #117", like the one above): RR
    # CVODE CV_TOO_MUCH_WORK + COPASI failed; pure rate-rule ODE system on 7
    # parameters; independent scipy LSODA/BDF reproduce to max_rel=0 (peak-rel
    # ~4e-9); verdict confirm — bngsim confirmed.
    # MODEL9811206584:ode — RETIRED 2026-11-02 (GH #1288). libRoadRunner 2.11.0
    # (#1349) integrates the model, so the row is no longer REFERENCE_FAILED: it
    # PASSes against RR itself, and the nightly flags the entry "STALE
    # no-oracle-adjudication: natural=PASS" (run 41499451003, replaying
    # reference-data-v4, #1443). The adjudication it carried (cited as "GH #117",
    # like the ones above): RR CVODE CV_TOO_MUCH_WORK + COPASI failed on 2.9.2;
    # pure rate-rule ODE system on 4 parameters; independent scipy BDF reproduces
    # to max_rel=0 (peak-rel ~1e-9); verdict confirm — bngsim confirmed.
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


# --------------------------------------------------------------------------- #
# No ground truth over the job's horizon -> a DIFF or PASS becomes BAD_TEST.
# Keyed "model_id:method". `issue` is optional.
# --------------------------------------------------------------------------- #
# Both engines ran, and no tolerance makes either one agree with itself, so
# neither trajectory is an oracle: a PASS would claim bngsim was checked, and a
# DIFF that it diverged from a valid reference. Mirrors the AMICI suite's
# NO_GROUND_TRUTH (amici_dispositions.py), including #1565's rule that a DIFF
# whose metric is not finite keeps scoring. Each entry MUST cite a convergence
# study that runs from this repository: parity_checks/tests/
# test_rr_known_artifact_evidence.py runs one for every key here.
#
# All four were KNOWN_ARTIFACT entries until 2026-11-10, scored PASS with "no
# engine attributable", on a triage note never committed to this repository.
NO_GROUND_TRUTH: dict[str, dict] = {
    "BIOMD0000000373:ode": {
        "issue": None,
        "reason": (
            "Bertram2004 pancreatic beta cell (model B), 7 rate rules, bursting "
            "with glycolytic oscillations, over the SED-ML figure's t_end = 7e5. "
            "Neither engine agrees with itself as the tolerance tightens: run at "
            "rtol 1e-12, atol 1e-18 and at 1e-13, 1e-20, bngsim's two runs DIFF "
            "(err/movement 0.79) and so do RoadRunner's (0.82), as far apart as "
            "the engines are from each other. Each tenfold tightening from 1e-9 "
            "moves the time where successive runs first part by 1% of a range "
            "only from 1.26e4 to 2.94e4, so no tolerance a double can hold pins "
            "the phase out to 7e5. Over the first 1.19e4 the engines agree at the "
            "suite's tolerance, and every species' min and max agree within 0.9% "
            "of its range: the amplitudes are pinned, the phase is not. Formerly "
            "a KNOWN_ARTIFACT, 'long-horizon oscillator phase drift'. parity_checks/"
            "tests/test_rr_known_artifact_evidence.py runs this study."
        ),
    },
    "BIOMD0000000376:ode": {
        "issue": None,
        "reason": (
            "Bertram2007 islet cell oscillations, 11 rate rules, over the SED-ML "
            "figure's t_end = 6e5. Neither engine agrees with itself as the "
            "tolerance tightens: at rtol 1e-12, atol 1e-18 against 1e-13, 1e-20, "
            "bngsim's two runs DIFF (err/movement 0.63) and so do RoadRunner's "
            "(0.63). Tightening from 1e-9 to 1e-13 moves the time where "
            "successive runs first part by 1% of a range only from 7.2e3 to about "
            "1e5. Over the first 6.6e3 the engines agree at the suite's "
            "tolerance, and every species' min and max agree within 1.7% of its "
            "range: the amplitudes are pinned, the phase out to 6e5 is not. "
            "Formerly a KNOWN_ARTIFACT, 'long-horizon oscillator phase drift'. "
            "parity_checks/tests/test_rr_known_artifact_evidence.py runs this study."
        ),
    },
    "BIOMD0000000682:ode": {
        "issue": None,
        "reason": (
            "Wierschem2004 electrical bursting in pancreatic islets, 5 rate rules, "
            "over the SED-ML figure's t_end = 6e5. Neither engine agrees with "
            "itself as the tolerance tightens: at rtol 1e-12, atol 1e-18 against "
            "1e-13, 1e-20, bngsim's two runs DIFF (err/movement 0.76) and so do "
            "RoadRunner's (0.89). Successive runs first part by 1% of a range "
            "near t = 2.2e5 at every rtol from 1e-9 to 1e-13. Up to t = 2.18e5 "
            "the engines agree at the suite's tolerance, and every species' min "
            "and max agree within 0.5% of its range: the amplitudes are pinned, "
            "the phase past 2.2e5 is not. Formerly a KNOWN_ARTIFACT, 'oscillator "
            "phase drift'. parity_checks/tests/test_rr_known_artifact_evidence.py "
            "runs this study."
        ),
    },
    "MODEL2310250001:ode": {
        "issue": None,
        "reason": (
            "Millard2023 gluconolactone-gluconate bypass in E. coli, over an "
            "invented t_end = 100. Every rate is isotopic_irreversible(v, S, "
            "Stot) = v*S/Stot, and uptake is zero-order, so while biomass X grows "
            "exponentially (to 2.3e16) the extracellular glucose and gluconate "
            "totals are driven through 0 between t = 8 and 9 and on to -4.9e17 "
            "and -7.3e32. S/Stot is singular where Stot crosses 0, and past that "
            "the gluconate isotopologue split is repelled from its source's at a "
            "rate that grows as 1/|Stot|: it has no unique solution. So GNT_e_0/1 "
            "and _6PG_0/1, which it feeds, converge at no tolerance in either "
            "engine (at rtol 1e-12 against 1e-13 they move by 0.23 to 11 times "
            "their range; GNT_e_0(100) is -6.5e31 in bngsim at 1e-9, +4.9e31 at "
            "1e-11), "
            "and GAP_0/1 and PYR_0/1 downstream by up to 10%. The other 17 "
            "columns, the totals among them, converge in both engines and agree "
            "(within 2.1e-5 of their range), and every column agrees through t = "
            "8. Formerly a KNOWN_ARTIFACT, 'both engines blow up to ~-5e16 and "
            "agree within 6%' (max_rel_err is 0.34 now). parity_checks/tests/"
            "test_rr_known_artifact_evidence.py runs this study."
        ),
    },
}


def overrides_for(model_id: str, method: str) -> list[dict]:
    """All overrides authored for ``model_id:method`` as {field, value, reason} dicts.

    ``field="tol"`` -> ``value={"rtol":..,"atol":..}`` (both engines);
    ``field="known_artifact"`` -> ``value={"issue":..}`` (reason in ``reason``);
    ``field="invalid_reference"`` -> ``value={"issue":..}`` (reason in ``reason``);
    ``field="no_ground_truth"`` -> ``value={"issue":..}`` (reason in ``reason``) —
    a DIFF or PASS with no ground truth over the horizon, reclassified BAD_TEST;
    ``field="no_oracle_adjudicated"`` -> ``value={"issue":..,"verdict":..}`` (reason
    in ``reason``) — annotates a REFERENCE_FAILED row settled by an independent
    oracle (verdict confirm/uncoverable), marking its refusal class SETTLED.
    The order is tol-first so a tol override is applied before the run and a
    post-run reclassification (known_artifact -> PASS, invalid_reference /
    no_ground_truth -> BAD_TEST) after it. A key in more than one reclassifier
    raises ValueError: a row has one disposition. None co-occurs with tol today,
    but the ordering keeps the contract unambiguous.
    """
    key = f"{model_id}:{method}"
    reclassifiers = [
        name
        for name, src in (
            ("KNOWN_ARTIFACT", KNOWN_ARTIFACT),
            ("INVALID_REFERENCE", INVALID_REFERENCE),
            ("NO_GROUND_TRUTH", NO_GROUND_TRUTH),
            ("NO_ORACLE_ADJUDICATED", NO_ORACLE_ADJUDICATED),
        )
        if key in src
    ]
    if len(reclassifiers) > 1:
        raise ValueError(
            f"{key} is authored in {' and '.join(reclassifiers)}; a row has one "
            "disposition (bngsim checked and right, no usable reference, no ground "
            "truth, or settled by an independent oracle)"
        )
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
    if key in NO_GROUND_TRUTH:
        d = NO_GROUND_TRUTH[key]
        out.append(
            {"field": "no_ground_truth", "value": {"issue": d.get("issue")}, "reason": d["reason"]}
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
    set(KNOWN_ARTIFACT)
    | set(INVALID_REFERENCE)
    | set(TOL_OVERRIDES)
    | set(NO_ORACLE_ADJUDICATED)
    | set(NO_GROUND_TRUTH)
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
