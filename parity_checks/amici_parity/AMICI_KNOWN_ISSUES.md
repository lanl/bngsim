# AMICI known issues on the bngsim parity suite

Findings from the **full** `amici_parity` ODE sweep — bngsim vs AMICI over the
complete 1323-model SBML corpus (`amici_run.py --manifest ../rr_parity/ode_jobs.json`)
— plus Classes 3 and 4, which come from the forward-sensitivity sweep over the same
corpus (`amici_sens_run.py`).
**Every AMICI problem below is AMICI-side; bngsim is not the outlier on a single
one.** On all 42 of the 44 ODE divergences where an independent third oracle is available,
**RoadRunner agrees with bngsim exactly** and AMICI is the odd engine out (the other 2
are inconclusive only because RoadRunner *also* failed those models). Class 4 is the
narrower claim its own section states: AMICI is not a usable oracle there, which is not
by itself a bngsim confirmation.

- **Environment:** AMICI pinned to `AMICI-dev/AMICI@667b17b6b` (`v1.0.1-12-g667b17b6b`,
  BSD-3; built from source into `bngsim/.venv`, needs `swig`) — see `AMICI_PIN.json` for the
  full pin. bngsim 0.9.51, RoadRunner 2.9.2. Verdict kernel = the shared
  `_core.differ.deterministic_verdict` (rtol=1e-9, atol=1e-12).
- **Result (1323 models):** **1050 PASS**, 44 DIFF, 165 REFERENCE_FAILED, 8 EXCEPTION,
  42 BAD_TEST, 14 TIMEOUT. So AMICI fails-to-run or diverges on ~20% of the corpus;
  bngsim runs and matches RoadRunner throughout.

## Class 1 — AMICI cannot run the model (173: REFERENCE_FAILED + EXCEPTION)

AMICI's per-model codegen/compile or its integrator fails, so there is no AMICI
trajectory to compare. bngsim runs these fine. Root causes:

| count | root cause |
|---|---|
| 34 | **Integrator bailout** — `AMICI integration failed (status < 0)` (CVODE gives up; stiffness / step failure) |
| 30 | **Unsupported math `floor()`** in a rate law — AMICI can't codegen it |
| 30 | **C++ model compile failed** — incl. the **event-codegen bug** `deltax.cpp: undeclared identifier 'D'` on event-bearing models (e.g. `BIOMD675/1028/1029`) |
| 19 | **SBML document failed to load** in AMICI's importer |
| 13 | **Unsupported math `ceiling()`** |
| 9 | **SBML package/extension** unsupported (comp / fbc / …) |
| 7 | **Event: non-persistent trigger** unsupported |
| 4 | AMICI internal: `'And' object has no attribute 'evalf'` (piecewise/logical codegen) |
| 4 / 3 / 1 | AMICI internal: **local-symbol collision** (`pi` / `avogadro` / `time` already reserved) |
| 3 | **Event: execution delays** unsupported |
| 2 | `StoichiometryMath` unsupported |
| 14 | other |

So **events are a modest slice** (~10 explicit + some of the `deltax` compiles); the
dominant failures are **integrator bailouts, unsupported `floor`/`ceiling` math, model
compile failures, and SBML-load failures**, plus a handful of AMICI internal bugs.

## Class 2 — AMICI integration outliers (44 DIFF; 42 confirmed AMICI-side)

Both engines ran but disagree, and **RoadRunner sides with bngsim** (max_rel_err = 0):

- **4** — AMICI trajectory went non-finite (`val = inf`; e.g. `BIOMD114/115/346/919`).
- **36** — large divergence (≥ 0.5 relative; AMICI grossly off — e.g. `BIOMD943`, `BIOMD125`).
- **4** — mild divergence.
- **42 of 44** confirmed AMICI-outlier (bngsim == RoadRunner exactly). The remaining 2
  (`BIOMD339`, `MODEL2003200002`) are inconclusive: RoadRunner refused them too, so there
  is no third oracle — but bngsim ran cleanly.

## Class 3 — AMICI's sensitivity RHS goes non-finite (2, forward-sensitivity sweep)

Found while establishing issue #339's step budget. Both models were reported
`AMICI_TOO_MUCH_WORK` / `AMICI_FIRST_SRHSFUNC_ERR` and read as step-budget
exhaustion; probing them at **10,000, 100,000 and 1,000,000** steps shows the budget
is not the cause. AMICI's own generated sensitivity right-hand side returns `NaN`,
so the corrector cannot converge and burns whatever budget it is given:

| model | Np | AMICI diagnostic |
|---|---|---|
| `MODEL0911120000` | 33 | `[AMICI:NaN] AMICI encountered a NaN value for sxdot[7] at t=27.574128` — then `TOO_MUCH_WORK` at every budget (0.2 s → 3.1 s → 32.6 s as the budget rises) |
| `MODEL1701170001` | 135 | `NaN value for sxdot[0]` on the **first** RHS call → `AMICI_FIRST_SRHSFUNC_ERR`, immediately, at every budget |

bngsim solves both at the same `Np` and the same tolerances, so these stay
`REFERENCE_FAILED` — no oracle, not a bngsim defect. The distinction matters for
triage: a `TOO_MUCH_WORK` row is *usually* a budget that can be raised (6 of the 8
models in #339 were), and these two are the counterexample that shows the status
code alone does not settle it.

## Class 4 — AMICI answers, and the answer is unusable (1, forward-sensitivity sweep)

Classes 1–3 are AMICI *failing to produce* a result. This is the harder kind: AMICI
runs, returns finite numbers, and those numbers are wrong against the SBML. It is
triaged out of #325 and recorded as an `INVALID_REFERENCE` entry in
`amici_dispositions.py` (issue #380), so the row is a non-scoring
`REFERENCE_FAILED`/`invalid_result` instead of a bngsim `DIFF`. Worth reporting upstream
to AMICI-dev.

Measured with **RoadRunner as the independent third engine** and the perturbation applied
to the **SBML text**, so no engine can discard the write.

### `MODEL2105110001` — AMICI computes no switch-time (saltation) term

`recruit_neu_t_switch` is a kineticLaw-local parameter inside
`piecewise(n1*CYT, t < t_switch, 0)` — a clock switch at t = 10. RR's FD, taken away from
the crossing to avoid the half-step artifact, converges to bngsim as h shrinks:

| h | RoadRunner FD | bngsim | AMICI |
|---|---|---|---|
| 0.5 | −5.2584 | −6.5085 | **0** |
| 0.1 | −6.71147 | −6.78896 | **0** |

and on `depletion_neu_t_switch` (same construct): RR −0.941533, bngsim −0.947201, AMICI
−15.4723.

**Systematic, not a one-model artifact.** It recurs on every model where a parameter sets
a piecewise or clock switch time — exactly the class bngsim built #48 / #56 / #358 / #375
for — so without the disposition it keeps re-filling the triage queue with rows where
bngsim is right.

### The other two #325 rows are not in this class

`BIOMD0000000117` is recorded as a `COMPARISON_ARTIFACT` instead, because **neither engine
is wrong**: bngsim and AMICI agree to 7 significant figures (`tstim`: −40.13189 vs
−40.13190; `v0`: 181.9811 vs 181.9811) and RR's FD converges to both as h shrinks (−11.04
at h=0.04 → −37.93 at h=0.004). What survives is 1 hard cell out of 22022, on a stimulus
discontinuity no finite-difference oracle can resolve.

`MODEL1607210000` has **no entry at all**: #383 landed first and the row is no longer a
DIFF. #380 measured it at HEAD `d5c4323` as `max_rel=1` with 2849 hard-failing cells and
attributed that to AMICI being inert in 30 of its own 31 free parameters. What #383 fixed
was the *other* side of the same construct — bngsim could not keep those
`<initialAssignment>`s symbolic (the **#313** freeze warning fired on this model, naming
`Saci1181KO`, `v12_k1`, `v1a_v`, `v3_k1`, `v9_k1`) and carried no initial-condition term
for them. Seeded, the row is **PASS at `max_rel=0` on both corrector methods**, 0 of 31310
cells failing.

One thread is left open rather than buried. Re-measured on post-#383 main, a 1% SBML-text
perturbation of those five parameters moves **bngsim's** trajectory by exactly `0` over
the manifest horizon (t 0→100, 101 points) — and AMICI's by exactly `0` too, which is
*why* they now agree. #380 reports RoadRunner moving 0.069 / 0.0098 / 0.089 / 0.089 /
0.089 for the same writes. Two engines that share no code agreeing on exact zero usually
points at the model rather than at a shared bug, but that is a third-engine adjudication,
not a disposition, and it is filed separately.

## Bottom line

AMICI passes ~79% of the SBML ODE corpus. Where it fails it is an **AMICI**
limitation (codegen gaps for discrete math and events, integrator bailouts, internal
bugs) or an **AMICI** integration error — never a bngsim defect. Across both reference
engines (RoadRunner and AMICI) on the full SBML corpus, **bngsim is never the outlier.**

## Reproduce

```bash
export BNGPATH=/path/to/BioNetGen-2.9.3
python amici_run.py --manifest ../rr_parity/ode_jobs.json --workers 8 \
    --out runs/report_ode_full.json
# a specific AMICI compile failure (verbose shows the real C++/SBML error):
python -c "from amici import SbmlImporter; \
  SbmlImporter('../rr_parity/models/BIOMD0000000675/BIOMD0000000675.xml').sbml2amici('m','/tmp/m',verbose=True)"
```
