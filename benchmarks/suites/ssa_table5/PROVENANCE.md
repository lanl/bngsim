# Table 5 — provenance concerns

Rule for inclusion: **published** model that in the **published study** was
**simulated with exact SSA**, at the **real (figure) horizon**. Below, every
model I am not fully confident about, so you can spot-check.

## Munsky (Fig 9) vs Shahrezaei-2008 — do they overlap?

**No — they are distinct models, kept as two separate rows.** They share the
two-state ("telegraph") promoter motif, so they're the same *family*, but the
structure and readout differ:

| | `gene_expr_3stage` (Shahrezaei) | `gene_expression` (Munsky Fig 9) |
|---|---|---|
| source | Shahrezaei & Swain 2008, PNAS 105:17256 | Munsky, Neuert, van Oudenaarden 2012, Science 336:183 (BNGL from Chylek et al. 2015, Phys Biol 12:045007, Fig 9) |
| genes | **one** gene | **four** genes (1 constitutive + 3 regulated) |
| cascade | DNA on/off → **mRNA → Protein** (3-stage) | promoter on/off → **mRNA only** |
| point | analytic protein/mRNA distributions | 3 genes at **equal mean**, **different** transcript distributions |
| size | 6 sp / 6 rx | 10 sp / 14 rx |

So there is **thematic** overlap (both are "stochastic gene expression"), but no
structural redundancy — one is a single-gene protein-producing 3-stage model,
the other a multi-gene mRNA-level telegraph comparison. I verified this directly
in the two `.net` files. If you want maximal topical spread in Table 5 they can
both stay (different sizes, different readouts); if you'd rather not have two
gene-expression rows, drop one — your call. I left both in.

## BNGL — items to confirm

- **`vilar_circadian` (Vilar 2002)** — ✅ confirmed originally-SSA. New addition;
  network regenerated with BNG 2.9.3 (9 sp / 16 rx). No concern.
- **`tcr_signaling` (Lipniacki 2008)** — ✅ confirmed. Horizon resolved: net
  regenerated from RuleHub Lin-2019 TCR_model.bngl at the Lin-2019 exact-SSA
  horizon **t_end=10000**.
- **`erk_activation` (Kochańczyk 2017)** — ⚠️ two things:
  1. Confirm the reference and that the source study used SSA.
  2. **Exact SSA is expensive**: populations reach ~3×10⁶, so exact SSA is
     ~6×10⁸ events/replicate (~77 s/rep measured in `bng_parity`). It is the slow
     anchor of the table. (This is exactly why it was `ssa_skip`/PSA-only before.)
     It's feasible — just budget for it.
- **`prion_aggregation` (Rubenstein 2007)** — ⚠️ confirm the Rubenstein-2007
  reference and SSA origin. Horizon resolved: net regenerated from RuleHub
  Lin-2019 prion_model.bngl at **t_end=300, n_steps=30000**.
- **`gene_expr_3stage` (Shahrezaei 2008)** — ✅ confirmed.
- **`gene_expression` (Munsky 2012)** — ✅ confirmed SSA. Nuance only: the BNGL is
  an illustrative re-implementation from the Chylek 2015 review of the Munsky
  Science-review model — didactic, not a data-fitted primary study. Both used SSA.

## SBML — the premise "DIFF only because of long simulation time" is not accurate

You said we could take all the SBML files I found and that they're DIFF only for
runtime. Reading the parity subclasses, that's not quite the situation — and it
splits three ways:

1. **Clean PASS, but SSA provenance doubtful** — `Cui2008` (966) and
   `Ouzounoglou2014` (559). These are **not** DIFF; they pass SSA parity cleanly
   at a real figure horizon. The problem is the opposite: their source studies
   look like **ODE** models (a zinc-homeostasis transcription model and an
   α-synuclein/neuronal-homeostasis model). They cleared the SSA-*compatibility*
   screen (mass-action, integer-ish ICs), which is **not** the same as having been
   simulated with SSA in the paper. **Please verify these two before we use them.**
   If they weren't SSA in the source study, they don't meet your rule.

2. **DIFF because RoadRunner won't fire time-events (not runtime)** — the
   `Proctor2017` trio `860`/`862`/`864`. Subclass `rr_time_event`: these models
   have time-triggered events, and RoadRunner's gillespie silently does not fire
   them (it warns and freezes at the initial condition). So the DIFF is a
   **RoadRunner capability gap**, not simulation cost — and it means **RoadRunner
   cannot produce a reference trajectory** for these rows. If Table 5's SBML
   columns are BNGsim vs RoadRunner-gillespie vs COPASI, the **RR column is N/A**
   for these three unless you anchor them on COPASI instead. Provenance itself is
   strong (Proctor group = Gillespie/COPASI stochastic).

3. **DIFF genuinely because of long simulation time** — `Proctor2011` (344) and
   `Smith2013` (474). Subclass `partial_horizon`: exact SSA is too slow to reach
   t_end, so only a partial window was simulable in the screen. This matches your
   "long simulation time" description. Strong SSA provenance. Note `Smith2013`
   **also** carries time-triggered events, so it has the RR gap of case (2) on top
   of being the largest SBML network (133 sp / 367 rx).

### Resolution (2026-07-18b)

- **Dropped** `Cui2008` (966) and `Ouzounoglou2014` (559): ODE-origin, fail the
  "SSA in the source study" rule. Files removed.
- **Kept** the **Proctor/Smith** models (344, 860, 862, 864, 474): strong
  originally-SSA provenance.
- **Engines:** the SBML rows are compared across **three exact-SSA engines** —
  bngsim-SSA, RoadRunner-gillespie, COPASI-SSA. Every engine is run on every
  model; a cell an engine can't produce is marked **N/A** with a footnote (not
  dropped, not pre-selected). Expected N/A: RoadRunner on 860/862/864 (won't fire
  time-triggered events); verify RR on 474 (also has events). Proctor2011 (344) has
  no events, so all three run it. `344`/`474` are run to their full horizon.

## New additions (systematic scans, 18c/18d)

**SBML — 3 non-Proctor originally-SSA models** (found by scanning every model's
`<notes>` for stochastic-method language; provenance is quoted *in the model*):
- `Besozzi2012` (478) — ✅ notes: *"defined according to the stochastic formulation
  of chemical kinetics [Gillespie 1977]… performing stochastic simulations."*
  No events; all three engines run it. Confirm exact citation (PMID 22818197).
- `Karapetyan2016` (586 ATC / 587 RTC) — ✅ notes: *"we use stochastic simulation to
  show that multiple binding sites… mitigat[e] the binary noise."* 9 assignment
  rules on a 10-species model — **verify exact-SSA handling of the rules**. Two
  circuit variants; prune to one if redundant. Confirm citation (PMID 26764732).

**BNGL — 3 originally-SSA models** (corpus header-scan; each cites its paper):
- `mckane_predator_prey` — ✅ McKane & Newman 2005 PRL. **Demographic** (ecological)
  stochasticity, not biochemical — keep or prune by scope.
- `gene_bursts` — ✅ Lin & Doering 2016 PRE. Source runs an ODE equilibration
  (t_end=360000) before the SSA run — replicate if matching the paper's IC.
- `samoilov_futile_cycle` — ✅ Samoilov, Plyasunov & Arkin 2005 PNAS. BNGL is a
  from_antimony conversion — verify parameters match the paper.

Two BNGL models were deliberately **not** added: `ExampleModel5_v2` (also Munsky
2012 — redundant with `gene_expression`) and `Lipniacki2006` (the encoding is the
deterministic *limit*, not the stochastic model).

## Diagnostics & fixes (18e — after the first bngsim activity run)

- **Removed `Smith2013` (474).** Under exact SSA its `ROS` species is driven
  negative (9,945×). Root cause: the SBML is a *concentration-unit* ODE model
  (`hasOnlySubstanceUnits="false"`, custom `function_XX` rate laws with
  compartment-volume factors) — exact discrete SSA on it is ill-posed. bngsim is
  correct (literal rate-law evaluation, matching CVODE; it does not floor at zero).
  This is a model-encoding problem, not a bngsim bug, and it undercuts 474's
  exact-SSA provenance, so it was dropped.
- **Fixed `gene_bursts`.** It first measured 0 events — an artifact of building the
  `.net` from `generate_network` alone, which drops the model's ODE-equilibration
  protocol (seed ICs mRNA=0/Protein=0 are near-inert; basal transcription only, so
  seed=1 drew 0 by an e⁻² chance). Fixed by baking the ODE-equilibrated steady
  state (mRNA=0, Protein=467) into the `.net`; it now measures ~0.13–0.33
  events/rep. Not a bngsim bug and not misspecification — a corpus-construction gap.
- **SBML audit → molecule-count rule.** Checked all remaining models for the 474
  signature. 860/862/864/344/478 are molecule-count encodings (`initialAmount`,
  `hasOnlySubstanceUnits=true`) and are clean. `Karapetyan` 586/587 are
  concentration-unit (`initialConcentration`, `hasOnlySubstanceUnits=false`, unit
  volume): they threw no negativity, but exact SSA on a concentration model is
  ill-posed regardless — the "amounts" aren't molecule counts, so the noise level is
  an artifact of the unit interpretation. **REMOVED 586/587.** Rule going forward:
  **SBML rows must be molecule-count encodings** (`initialAmount` /
  `hasOnlySubstanceUnits=true`); concentration-unit SBML is an ODE encoding.

- **Added Vilar2002 SBML (BIOMD35).** Verified molecule-count (`hasOnlySubstanceUnits
  =true`, DA=DR=1), runs clean under SSA (no negativity), non-Proctor. It's the
  cross-format twin of the BNGL `vilar_circadian` row — same published model, SBML
  encoding. Caveat: the two encodings are not bit-identical (events/time ~2743 SBML
  vs ~1515 BNGL at t_end=400) — a curated rate-constant difference to reconcile.

- **De-duplicated Vilar: removed the BNGL encoding, kept the SBML.** A parameter
  diff showed the two encodings are structurally identical (same 10 species / 16
  reactions / ICs) and share 14 of 15 rate constants; they differ in exactly one —
  repressor degradation `delta_R`: BNGL 0.05 (written `0.2/4` in the RuleHub source)
  vs SBML 0.2. The BNGL author quartered the published value, so the SBML (0.2) is
  the faithful Vilar-2002 encoding. That single 4× difference is the whole 2,743 vs
  1,515 events/time gap. Vilar is now SBML-only (BIOMD35).

Final set: **8 BNGL + 6 SBML** (non-Proctor SBML: Besozzi/478, Vilar/035). BNGsim
exact-SSA activity measured (`results/bngsim_activity.json`); RoadRunner/COPASI
columns pending.

## Re-pointed at the curated records (2026-08-19, lanl/bngsim#423)

Every BNGL row now comes from its `wshlavacek/BNGL-Models` record @`6783b6e9`,
generated by `../../models/regenerate_curated_nets.py` into the shared
`../../models/net/curated/` that `suites/psa` runs too. The `.net` files above
predated `wshlavacek/bngsim-paper#6`'s re-pointing of rows B07–B14, so a re-run
would have re-measured superseded models. **SBML rows are untouched** — they are
not BNGL-Models records.

| row | record | was | now |
|---|---|---|---|
| B07 `gene_bursts` | `bursty_autoregulated_gene_expression_lin2016` | 2/4 | 2/4 |
| B08 `mckane_predator_prey` | `demographic_noise_predator_prey_cycles_mckane2005` | 3/5 | 3/5 |
| B09 `gene_expr_3stage` | `three_stage_stochastic_gene_expression_shahrezaei2008` | 6/6 | **4/6** |
| B10 `samoilov_futile_cycle` | `noise_induced_bistable_futile_cycle_samoilov2005` | 6/6 | **7/10** |
| B11 `gene_expression` | `two_state_gene_expression_noise_munsky2012` | 10/14 | 10/14 |
| B12 `erk_activation` | `mapk_relaxation_oscillations_kochanczyk2017` | 34/65 | 34/65 |
| B13 `tcr_signaling` | `tcr_signaling_bistability_lipniacki2008` | 37/97 | 37/97 |
| B14 `prion_aggregation` | `prion_nucleated_polymerization_rubenstein2007_benchmark` | 104/2809 | **121/3843** |

Three earlier notes on this page are now superseded and are kept only as the
record of what the old artifacts were:

- **The `gene_bursts` IC bake becomes a declared relaxation.** The 18e fix baked
  an ODE-equilibrated steady state (mRNA=0, Protein=467) by hand into the `.net`.
  The same state is now *derived*: `curated_nets.json` declares a `relax` step
  for this model, `regenerate_curated_nets.py` runs it against the curated record
  before writing the artifact, and the result is rounded to whole molecules.
  **The measurement does not move** — this `.net` and the 18e one both give
  median 923 / mean 930 events at `t_end=3600` over 200 seeds, none of them zero.

  This row is the one place the "curated model body at the table's horizon" rule
  needs a qualifier, and it is worth stating plainly. B07's horizon and its
  initial state were a *matched pair* in the superseded actions block —
  `simulate ode t_end=360000` immediately followed by `simulate ssa t_end=3600` —
  and the manuscript kept the horizon. Keeping half the pair is what makes the
  row arbitrary: from the bare `0/0` seeds at `t_end=3600` the model sits in its
  basal regime and measures median 95 events with **25 of 200 replicates firing
  nothing**, which times process overhead rather than the model. Every source for
  this model relaxes first; the record does too, just to a shorter 3.6×10⁴ s.

  The relaxation horizon is not a free parameter, which is what keeps this from
  being hand-tuning by another name: 3.6×10⁵, 3.6×10⁶ and 3.6×10⁷ s all give
  mRNA=0.389150 / Protein=466.979857 to every digit BNG prints, so the seed is
  the ODE **steady state**. The record's own 3.6×10⁴ s is *not* converged
  (Protein=111.7, still climbing toward 467) — which is why the relaxation
  follows the superseded block's 3.6×10⁵ rather than the record's own value.

  The seeds are rounded because a fractional molecule count is ill-posed for a
  discrete solver and the engines disagree about it: bngsim rounds (and warns),
  `run_network` rounds, but RoadRunner's gillespie takes 0.389 literally and
  walks mRNA to −0.61 over the horizon — the same signature that got
  Smith2013/474 dropped above. Rounded in the artifact, all three start from
  (0, 467).
- **`erk_activation` is a relabelling, not a re-measurement.** The record and the
  superseded copy generate the same 34/65 network with identical seed species and
  identical effective rate constants — the old copy carried a symbolic system-size
  parameter `lambda=1` with derived `_rateLaw*` expressions. Measured activity is
  identical to the event at `t_end=86.4`. The record has *no* active exact-SSA
  action (its only active action is an ODE run at `t_end=86400` for Fig. 4B, and
  its sole stochastic line is a commented-out partial-scaling block); the 8640
  horizon comes from Lin et al. 2019 and the manuscript says so.
- **`samoilov_futile_cycle` loses its RoadRunner cell.** The record is the primary
  file, with the external noise driver of Expressions 7 and 8 — whose
  `N + N -> E+ + N` step is second order in the same species. The converted SBML
  law `k*N*N` is not the exact propensity `k*N*(N-1)`, and RR-gillespie fires the
  SBML law (GH #9), so that cell is **N/A**; COPASI derives the combinatorial
  propensity itself and its cell stands. The superseded copy had no driver. The
  record's `_unordered_pair` variant does not rescue the cell: it writes the same
  `5,5 -> 3,5` reaction with the rate constant halved, so the converted law is
  still `k*N*N`.

`prion_aggregation`'s event count barely moves (~605 k at `t_end=10`, old and new
alike) because the 17 added species are chains of length 104–120 with zero initial
condition — the cost moves through *per-event* work instead, ~20 % at `t_end=10`,
and the full-horizon `t_end=300` run now exceeds `run_bngsim_activity.py`'s 60 s
cap (72 s) where the 104/2809 copy fit inside it.

## B10's horizon comes from the record too (2026-08-20, lanl/bngsim#425)

The re-pointing above changed B10's model and left its horizon alone, so the row
ran a 7/10 model at `t_end=0.0018` — a value chosen for the 6/6 artifact that had
just been replaced. The record's own protocol runs to `t_end=10` sampled every
0.005 s, reproducing Fig. 3A. The row now runs that horizon, so the model and the
horizon come from the same file.

**Where 0.0018 came from.** It is not from Samoilov et al. (2005). The superseded
artifact was converted from `../../models/antimony/ssys/Samoilov2005.ant`, one of
the 117 hand-written Antimony models that make up a different corpus in this
repository, and `t_end=0.0018` with `n_steps=180` is that file's own `@SIM`
annotation. That file describes itself as "the underlying mass-action ODE system
(no external noise)" and exists there as a **stiffness pathology case** for
S-system recasting — its header records that trajectory integration of the recast
fails. The horizon was picked to exercise a numerical experiment on a
deterministic model, and the superseded `.net` even inherited that file's
`EPS_INIT` perturbation, seeding `X`, `C1` and `C2` at 1 molecule where the paper
has 0. So B10 was never running a horizon anyone had chosen for a stochastic
simulation of this system.

This is the one B-row where adopting the record's horizon was the right call
rather than keeping the manuscript's, and it is worth saying why. `gene_bursts`
and `gene_expression` carry horizons the manuscript chose *for the model it is
running*; B10's was inherited from a model that is no longer in the corpus.

Two measurements settled it, both on the committed `.net`:

- **The model has not started at 0.0018 s.** The trajectory is still in the
  burn-in from the Fig. 3 initial condition: `X*` has fallen only from 2000 to
  about 1615 molecules, where the record's own protocol note puts the operating
  band at 110–286. Over 30 seeds the trajectory first enters that band at a
  median 0.0167 s — about nine times later than the row was stopping. The
  noise-induced switching the model is published for had not begun.
- **The cell measured setup, not simulation.** Interleaved 195 times against a
  run of the same model with the horizon set so it fires zero events, the
  `t_end=0.0018` cell costs 425 µs against a 386 µs zero-event floor, so **91 %
  of what it timed was per-run fixed overhead**; for `run_network` it is 15.8 ms
  against 14.1 ms, **89 %**. Table 5 is a cost table, so a cell that is 9–11 %
  simulation is a bad row on its own terms.

Cost was not the reason for the short horizon and is not a reason to keep it. At
`t_end=10` a replicate fires a median 1.36×10⁷ events over 30 seeds, for 0.46 s of
bngsim wall and 1.70 s of `run_network` wall, both far inside the harness's 120 s
per-run cap and in the range of the other rows. The activity figure in the README
table is the harness's own convention, one replicate at seed 1: 1,348,230 events
per second of simulated time. COPASI is not measured here, but its throughput on
the other converted-BNGL rows (4×10⁶–2.3×10⁷ events/s) puts it in the same range
of a few seconds.

**B10's cost moves by about four orders of magnitude**, so unlike B07 this one
does move a published number and the manuscript reports the new one. The
alternative — re-pointing B10 at the record's `_no_driver` variant, which is the
6/6 model `0.0018` was chosen for — was not taken: that file is the paper's
control rather than this study's model, and `wshlavacek/bngsim-paper#6` chose the
primary file deliberately.

The coverage does not change with the horizon: the RoadRunner cell is N/A because
of the repeated reactant, at any `t_end`.
