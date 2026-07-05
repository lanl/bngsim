# Vendored SBML benchmark models

Standalone SBML models vendored for benchmarking, kept separate from the
BNGL-derived `.net`/`.bngl` trees and the download-driven `biomodels` corpus.

## `Smith2013_BIOMD0000000474_petab.xml`

**Smith2013 — "Computational modelling of the regulation of Insulin signalling
by oxidative stress"**, Graham R. Smith & Daryl P. Shanley, *BMC Systems Biology*
**7**:41 (2013), [doi:10.1186/1752-0509-7-41](https://doi.org/10.1186/1752-0509-7-41).
133 species, 367 reactions; insulin signalling + oxidative stress + FOXO feedback.

- **Source:** the PEtab Benchmark Collection
  (`Benchmarking-Initiative/Benchmark-Models-PEtab`,
  `Benchmark-Models/Smith_BMCSystBiol2013/model_Smith_BMCSystBiol2013.xml`),
  BSD-3-Clause. This is the **PEtab/AMICI re-implementation**, NOT the raw
  BioModels file.
- **Why it's here:** it is the exact problem behind AMICI's benchmark
  `t_fwd≈50s / t_adj≈5s` *gradient-solve* timings (forward vs adjoint; these are
  solve times, not codegen/compile time), so it is the apples-to-apples reference
  for any bngsim↔AMICI sensitivity comparison, and — because its events are all
  fixed-time — the natural **Phase-1 validation target for GH #212**
  (forward-sensitivity-through-events). As of GH #212 (0.9.69) bngsim propagates
  forward sensitivities through these fixed-time events: the coupled state+
  sensitivity solve runs through the t=15 stimulation-end event with finite
  `dx/dp` (the jump math is validated to ~1e-6 vs central differences on
  synthetic models). **KNOWN GAP — GH #214:** a *full*-horizon sensitivity run to
  t=3000 currently **fails** at the t=2880 insulin-restimulation event (CVODE
  `flag=-3`). This is a **units / error-control artifact**, NOT a dynamics or
  jump-math bug, and the root cause is verified: bngsim and AMICI integrate the
  **same IVP** (the 107 shared real-state trajectories agree to ≤5e-7), but bngsim
  works in **concentrations** while AMICI works in **amounts**, so with Smith's
  sub-picoliter compartments bngsim's states and sensitivities are ~1/V
  (≈1e10–1e13×) larger — reaching ~1e18 vs AMICI's ~1e10. The CVODES error test
  then cannot hold the step on those inflated sensitivity variables across the
  large jump. Confirmed both ways: `CVodeSetSensErrCon(false)` (sensitivities out
  of the error norm) lets bngsim reach t=3000 with finite output, and AMICI — in
  amount units — completes the same forward-sensitivity run in 4.3 s (status=0,
  `sx` finite). Fix direction: integrate in amounts / non-dimensionalize the
  coupled solve. Plain ODE crosses all three events cleanly in <1s. Repro:
  `../../suites/forward_sens/smith_event_sens.py`.

### How this differs from the BioModels entry (BIOMD0000000474 / MODEL1212210000)

Same model ID, curated as BIOMD0000000474. The reaction **network is identical**
(133 sp / 367 rxn / 29 rules) across encodings; the differences are:

1. **Parameter representation (standard SBML flattening, semantics-preserving).**
   BioModels: 81 global + 307 local (kineticLaw) params + 366 rate-law
   `functionDefinition`s. PEtab/AMICI: **391 global, 0 function definitions** —
   the 366 rate-law lambdas are inlined and the 307 local params promoted to
   global (names like `R100_ksynth`, `R101_ktr`), plus 3 estimation params
   (`t_ins`, `indicator_jnk`, `indicator_foxo`). This is PEtab/AMICI tooling
   (`flattenSBML` / `convertLocalParameters`), not a model change.
2. **Stimulation protocol (events), re-parameterized for estimation.** All events
   in every encoding are fixed-time (`gt/geq(time, const)`), persistent, no delay
   — i.e. already AMICI-handleable; PEtab did **not** simplify event types for
   AMICI. PEtab (3 events) parameterizes the first-stimulation time as `t_ins` and
   uses a 2880–2895 second pulse; the BioModels author "eventsForFigure3A" file
   (4 events, vendored under `parity_checks/rr_parity/models/BIOMD0000000474/`)
   uses a fixed 0–15 + 2785–2800 double pulse; the curated base
   (`BIOMD0000000474_url.xml`) has 0 events (autonomous).
3. **One documented dynamics discrepancy.** Per the PEtab README, the
   re-implementation differs at Figure 2H (SOD2/InR transcription rates off by
   ~0.8/0.4, "source unclear, likely undocumented changes to the model"). A
   fidelity wart in the dynamics, unrelated to events/AMICI.

The BioModels curated/author encodings are already in the test set
(`biomodels` manifest, `rr_parity/ode_jobs.json`); this file adds the
PEtab/AMICI encoding specifically for AMICI-parity sensitivity work.
