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

Final set: **8 BNGL + 6 SBML** (non-Proctor SBML: Besozzi/478, Vilar/035); all files
vendored under `models/`. BNGsim exact-SSA activity measured
(`results/bngsim_activity.json`); RoadRunner/COPASI columns pending.
