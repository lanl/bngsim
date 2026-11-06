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
by itself a bngsim confirmation. Class 6 is not an AMICI problem either: it is a
model with no solution for either engine to be checked against.

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

## Class 4 — AMICI answers, and the answer is unusable (2 models, 3 rows)

Classes 1–3 are AMICI *failing to produce* a result. This is the harder kind: AMICI
runs, returns finite numbers, and those numbers are wrong against the SBML. Each is
recorded as an `INVALID_REFERENCE` entry in `amici_dispositions.py` (issue #380), so the
row is a non-scoring `REFERENCE_FAILED`/`invalid_result` instead of a bngsim `DIFF`. Both
are worth reporting upstream to AMICI-dev.

The oracle is never bngsim. Usually that means **RoadRunner as the independent third
engine**, with the perturbation applied to the **SBML text** so no engine can discard the
write. Where a model reduces exactly, the **closed form** serves instead, and is stronger:
there is no third implementation left to be wrong.

### `MODEL0910846879` — AMICI holds an `<assignmentRule>` piecewise at 0

Authored on **both** regimes (`:ode` and `:sens`), because the defect is in the state
trajectory and the sensitivity row is downstream of it — a tensor cannot be compared when
the `x(t)` it is differentiated about is already 92% apart.

This model has no reactions and one state, so it reduces exactly. Every assignment rule is
over constants, giving `TVZ = 9.341479411e-4`, and the lone rateRule
`dTVD/dt = (TVZ + DR − TVD)/TVDDL` integrates to
`TVD(t) = TVZ + (TVD0 − TVZ)·exp(−t/30)` with `TVD0 = 9.80838e-4`.

| t | closed form | bngsim | AMICI |
|---|---|---|---|
| 0 | 9.80838e-4 | 9.80838e-4 | 9.80838e-4 |
| 50 | 9.429665541e-4 | 9.429665538e-4 | **1.852563686e-4** |
| 100 | 9.35813562e-4 | 9.358135621e-4 | **3.499040825e-5** |

bngsim matches to 1.9e-9 (integration tolerance). AMICI's `3.499040825e-5` is
`TVD0·exp(−100/30)` to 10 significant figures — the trajectory for `TVZ = 0` exactly.

**AMICI's own expression vector is the second half of the evidence.** It computes
`TVZ1 = 9.3414794e-4` correctly at every time point, then reports
`TVZ = piecewise(0, TVZ1 < 0, TVZ1)` as `0` at every time point — the `otherwise` branch of
a condition that is false throughout. The control is in the same model:
`AHTH = piecewise(0, AHTH1 < 0, AHTH1)` is structurally identical and AMICI gets it right
(`9.5283044e-4`). So it is not the piecewise shape as such. What separates the two is that
`TVZ`'s condition reads `TVZ1`, which is itself a sum over another piecewise (`AHTH`),
while `AHTH`'s condition reads a plain expression.

**Reported upstream: [AMICI-dev/AMICI#3233](https://github.com/AMICI-dev/AMICI/issues/3233).**
Minimized there to four assignment rules and one state, with the mechanism pinned: a
piecewise inside another piecewise's *condition* makes the outer one take the wrong
branch. Changing the inner piecewise's first-piece value flips the outer result, even
though that value cannot change the inner value (its own condition is false either way),
which is what shows the outer condition is what sees the wrong thing. Replacing the inner
operand with a literal also fixes it, so the operand has to be a rule-defined parameter.

Surfaced by issue #382, which moved this model off CVODES' difference quotient and so put
a comparison in front of the row. The DIFF itself is older and is identical before and
after that change (`max_rel` 0.918 ode / 1.0 sens either way).

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

The exact zero those two engines agree on is **the model's answer, not a shared gap**
(issue #387, which this section used to leave open pending a third engine). Re-run on a
venv that has RoadRunner, the 1% SBML-text perturbation moves RoadRunner by exactly `0`
too — all three engines agree, and #380's RoadRunner column (0.069 / 0.0098 / 0.089 /
0.089 / 0.089) does not reproduce. Nor is it only those five parameters: **all 32** free
parameters (27 global, 5 kineticLaw-local) move both engines by `0`, bar `1.11e-16` — one
ulp — on `Saci1181KO` / `v7b_k1` / `v17_k1`.

**The model cannot move.** Each of its nine `<initialAssignment>`s sets a species'
first-order decay constant to that species' *total* production flux at t=0 over its own
initial concentration:

```
k4   = v3_k1·saci1181 / Saci1181
k2   = (v1a_v + v1b_k1·Saci1181^h/(K^h + Saci1181^h)) / saci1181
k11b = v12_k1·Saci1181KO·Saci1181·AbfR_P / AbfR
```

so t=0 is an exact fixed point *by construction, for any values of the production
parameters* — and a fixed point stays one for all t. Scaling a production flux scales the
matched decay constant identically. For `k11b` the cancelling pair is visible in the
reactions: `v11b` (AbfR→AbfR_P at `k11b·AbfR`) against `v12` (AbfR_P→AbfR at
`v12_k1·Saci1181KO·Saci1181·AbfR_P`), and `k11b` is *defined* as the ratio that makes the
two identical.

**The control**, because a sweep of zeros on its own proves only that nothing was
measured: `Starvation` ships at `0` — which is also why a multiplicative perturbation *of*
it reads zero, and why the five kineticLaw-local parameters do, all of them living in the
Starvation-gated `v1c` / `v5b` / `v11`. At `Starvation=1` the model comes alive
(trajectory drift 213.9, the two engines agreeing to 1.4e-6 relative) and those same five
writes move **both** engines by identical amounts: 0.205 / 0.276 / 0.987 / 5.99 / 4.93.

So #383 moved bngsim off a wrong non-zero column onto a *right* zero one. #380's
RoadRunner column is most likely a **runtime** write (`rr[pid] = …`) rather than a text
edit: RoadRunner does not re-evaluate `<initialAssignment>` on a runtime write, so `k11b`
keeps its base value while `v12`'s flux takes the perturbed one, the cancellation breaks,
and the trajectory moves. A 2% runtime write reproduces all five reported numbers to
within 5% and in the right proportions. That is the mechanism, not a confirmed
reconstruction — the exact recipe was not recovered, and nothing above depends on it.

## Class 5 — events AMICI does not fire (issue #1542)

Five rows that passed against bngsim 0.15.1 read `DIFF` against 0.17.0. bngsim had
been skipping the same events AMICI skips, so the two agreed. bngsim began firing
them (#1015, #1150). RoadRunner fires them too: its CVODE on BIOMD0000000285, and,
on the other four, RoadRunner run through the event instants, as `rr_run.py` has
done since #1024. On all five, bngsim and RoadRunner agree with 0 failing cells, and
AMICI is the engine out.

| model | trigger AMICI misses | remedy | row now |
|---|---|---|---|
| BIOMD0000000816, 817 | `time == 2`, `time == 4` (virus doses) | run AMICI through the instants | PASS, 0/6006 |
| BIOMD0000000928 | `time == 20` (tenfold dilution) | run AMICI through the instants | PASS, 0/1544 |
| BIOMD0000001059 | `time == 2000` (SMAC release) | run AMICI through the instants | PASS, 0/10010 |
| BIOMD0000000285 | `PIdeath > 0`, `p38death > 0`, both 0 at the start | `INVALID_REFERENCE` disposition | REFERENCE_FAILED |

**`time == c` between output samples.** AMICI rewrites `==` as `>=` and `<=`, whose
trigger function `min(t − c, c − t)` touches 0 at `c` without changing sign. CVODES
reports it only when a step ends exactly on `c`, which AMICI arranges only at an
output time (or at another event's fixed trigger time). RoadRunner misses it the
same way on a grid with no sample at `c`, and the remedy is the same (#1024):
`amici_run.py` now gives AMICI the instants as extra output times and compares the
job grid. See the README's
[event-instants section](README.md#amici-runs-through-the-event-instants-issue-1542),
which also lists the shapes the instants cannot help.

**A trigger whose residual is exactly 0 at the start.** AMICI reads it as true at
t = 0, for `>` as well as `>=`. BIOMD0000000285 is SBML L2V4, whose triggers read
`initialValue = true`, so to AMICI its death triggers are true before t = 0 and at
it, and they never fire. The cell lives (NatP 19520.1 at t = 10). bngsim (since
#1150, the decision on #934) and RoadRunner read the trigger as false at t = 0 and
true just after, so both events fire and set `kalive = 0`, and the cell stays at its
initial state (NatP 19500). RoadRunner fails 11463/27027 cells against AMICI, the
same as bngsim does, and 0 against bngsim.

That reading of the SBML start rule is a choice the project made, but AMICI's answer
does not come from the other reading. With `initialValue = true`, as here, AMICI
also misses a crossing that every reading fires: `B > 0` with `B` held at 0 until
t = 0.5 and rising after. AMICI integrates `B` right (`B(1) = 0.5`) and never
fires; bngsim and RoadRunner fire just after 0.5. With `initialValue = false` AMICI
fires, at t = 0 itself, three triggers that are never true: `B > 0` with `B(0) = 0`
falling, `B > 0` with `B` held at 0, and `B < 0` with `B(0) = 0` rising. RoadRunner
and bngsim fire none of them. An output time cannot help, because the trigger reads a
species, so the row is an `INVALID_REFERENCE` entry in `amici_dispositions.py`.
While it stands, the AMICI row checks nothing on this model; rr_ode is the check.

**Not in this class: MODEL1112050001**, the sixth model in #1542. It has no events;
see Class 6.

## Class 6 — a model with no solution to compare against (MODEL1112050001, issue #1542)

The sixth model in #1542 has no events. Against AMICI it read `PASS` on bngsim 0.15.1,
and on x86-64 it still does (0/2929 cells, #1546). On arm64 it has read `DIFF` since
#1076: 676/2929 cells, 240 hard. Those are #1542's M1 Max numbers, measured again on
an M4 Max at `e3b66384`. Neither verdict says anything about bngsim. Over the job's
horizon the model has no solution in double precision, and both engines return a
trajectory that is not one. The row is a `NO_GROUND_TRUTH` entry in
`amici_dispositions.py`, which makes it a non-scoring `BAD_TEST` whichever way it
falls; the matrix page draws it as `NO TRUTH`. rr_parity adjudicates the model
`uncoverable` too.

**The closed form.** The rate rule `x8' = k19·x5 − k20·x8 − k6·PTP·k8` drains `x8` at
a constant rate. `k8` is a 7.06e12 rate constant, and the term reads like a typo for
`x8`. Early on, that makes `x4 + x5 = −3.42e8·t²` (a run reads −3.420e-6 at t = 1e-7).
Then `x9' = k22·PTP·x10 − k7·x9·(x4+x5)/IRp`, with `IRp = 8.97e-13`, grows `x9` at
the rate `a = k7·|x4+x5|/IRp = 1.59e21·t²`. The model conserves `x9 + x10 + x12` and
`x11 + x12`, and keeps `x11 ≥ 0`, so `x9' ≥ (a − k22)·x9 + k22·9e-13`. To leading
order, `x9 = 1e-12·exp(5.29e20·t³)`:

| t | the law | measured |
|---|---|---|
| 1e-7 | 1.697e-12 | 1.697e-12: scipy's DOP853 (explicit Runge-Kutta, not SUNDIALS) on a right-hand side built from the SBML text, rtol 1e-12, atol 1e-80 |
| 3.36e-7 | 5.115e-4 | 5.115e-4: bngsim at rtol 1e-10, atol 1e-30 |
| 1.1e-6 | passes the largest double | |
| 1 | ln(x9) ≈ 5e20 | |

**What both engines return instead.** `x9` starts at 1e-12, which is the suite's atol,
so neither engine's error test sees it. Each holds `x9` on its repelling equilibrium,
`−k22·x10/a`, instead of following it. At t = 1 that is −1.609e-32 in bngsim and
−2.649e-34 in AMICI, each equal to its own equilibrium to 4 digits. Both then run on
to t = 100. A tighter tolerance (atol 1e-14, or rtol 1e-10) makes both engines try to
follow `x9`, and both fail between t = 3.3e-7 and 5.6e-7: bngsim's step collapses
until t + h == t, and AMICI reports `AMICI_TOO_MUCH_WORK`. So no engine has an answer
here past about t = 1e-6, and the AMICI row compares two trajectories that are not
solutions.

**Why the verdict flips.** On that trajectory, once `x6` drives `x2` negative,
`x1' = x1·(k1·|x2| − b3) + b1·vp·h1p/v` is unstable too, and rounding decides whether
a step holds `x1` the same way.

| at t = 1 | `x1` | `x2` |
|---|---|---|
| AMICI | 7.483e8 | −5.011e-8 |
| bngsim, atol 1.01e-12 | 7.483e8 | −5.011e-8 |
| bngsim, atol 1e-12, arm64 since #1076 | 3.528e-25 | −7.518e8 |

`x3`, `x4`, `x5` and `x7` differ in the same way, and those cells are the `DIFF`. At
rtol 1e-9 / atol 1e-12 exactly, bngsim at main on arm64 holds `x1`. Moving atol by 1
to 20 ulps either way (40 points) lets `x1` grow every time. Over 120 draws, with
rtol and atol each scaled by an independent uniform factor in
[1 − 1e-3, 1 + 1e-3]:

| engine | `x1` grows | `x1` held | fails |
|---|---|---|---|
| bngsim, main, arm64 | 119 | 0 | 1 |
| AMICI 1.0.1 | 117 | 0 | 3 |

At the manifest's own rtol 1e-6 / atol 1e-9 the outcome is less settled. A ±10% scan of
rtol (41 points) gives bngsim 24 grows, 11 held and 6 failed.

**What moved it.** A first-parent bisect over `3183494c..0b7b3424` stops at #1076
(`35e9bb1a`). #1076 compiles ExprTk with `-ffp-contract=off`, so that on arm64,
`a*b + c` is no longer fused into a multiply-add and rounds as it does on x86-64. Its
parent, `2a4d72e0`, lets `x1` grow. Removing only that flag puts #1076 back to `x1`
growing, and main (`e3b66384`) as well. #1076 also restored the exact sign rule in
`cvode_simulator.cpp`'s #928 handler. With that change kept and the flag removed,
`x1` grows, so the handler is not the cause. #1076 changed rounding on purpose, and
its fix stands. Any later rounding change (in ExprTk, SUNDIALS or the compiler) can
flip this row either way, on either platform, and the disposition makes that moot.

## Bottom line

AMICI passes ~79% of the SBML ODE corpus. Where it fails it is an **AMICI**
limitation (codegen gaps for discrete math and events, integrator bailouts, internal
bugs) or an **AMICI** integration error — never a bngsim defect. Across both reference
engines (RoadRunner and AMICI) on the full SBML corpus, **bngsim is never the outlier.**
Class 6's model has no solution for either engine to be measured against.

## Reproduce

```bash
export BNGPATH=/path/to/BioNetGen-2.9.3
python amici_run.py --manifest ../rr_parity/ode_jobs.json --workers 8 \
    --out runs/report_ode_full.json
# a specific AMICI compile failure (verbose shows the real C++/SBML error):
python -c "from amici import SbmlImporter; \
  SbmlImporter('../rr_parity/models/BIOMD0000000675/BIOMD0000000675.xml').sbml2amici('m','/tmp/m',verbose=True)"
```
