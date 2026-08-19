# `ssa_table5` — exact-SSA corpus for arXiv Table 5

Model set for arXiv Table 5 (per-model exact Gillespie-SSA cost across engines).
Every model is published **and** was simulated with exact SSA **in its source
study**, at the published time horizon. A *corpus* (models + horizons +
provenance), not a runner. Current size: **8 BNGL + 6 SBML**.

`corpus.json` is the SSOT — artifacts, horizons and output-point counts are
declared there once and read by `_ssa_config.py`, `convert_all.py`,
`run_bngsim_activity.py` and `emit_ssa_table.py`.

`events/time` = BNGsim exact-SSA activity (Gillespie events per unit simulated
time), 1 replicate, from `results/bngsim_activity.json` (`run_bngsim_activity.py`).

## BNGL set — BNGsim-SSA vs `run_network`

Every BNGL row is generated from its curated `BNGL-Models` record — see
`../../models/curated_nets.json` for the record and both sha256s, and
`../../models/regenerate_curated_nets.py --check` to confirm the committed
`.net` is still what that record generates. The three Lin-2019 systems are the
same artifacts `suites/psa` runs.

| model | sp | rx | t_end | events/time | reference | prov |
|---|--:|--:|--:|--:|---|:--:|
| `samoilov_futile_cycle` | 7 | 10 | 0.0018 | 762,778 | Samoilov, Plyasunov & Arkin 2005, PNAS | ✅ |
| `prion_aggregation` | 121 | 3843 | 300 | 105,156 | Rubenstein 2007 / Lin 2019 | ⚠️ |
| `erk_activation` | 34 | 65 | 8 640 | 68,993\* | Kochańczyk 2017 / Lin 2019 — **slow anchor** | ⚠️ |
| `tcr_signaling` | 37 | 97 | 10 000 | 12,545 | Lipniacki 2008 / Lin 2019 | ✅ |
| `mckane_predator_prey` | 3 | 5 | 1 200 | 216 | McKane & Newman 2005, PRL (demographic) | ✅ |
| `gene_expression` | 10 | 14 | 60 000 | 0.48 | Munsky 2012, Science (Fig 2B) | ✅ |
| `gene_expr_3stage` | 4 | 6 | 2×10⁸ | 0.033 | Shahrezaei & Swain 2008, PNAS | ✅ |
| `gene_bursts` | 2 | 4 | 3 600 | ~0.023† | Lin & Doering 2016, Phys Rev E | ✅ |

\* 596M events, ~156 s/replicate at the full horizon — it exceeds
`run_bngsim_activity.py`'s 60 s cap, which records it as `partial` on a reduced
horizon; the number here is a full-horizon measurement.
† `gene_bursts` seed species are the record's own (mRNA=0, Protein=0), so at the
manuscript's `t_end=3600` — one cell cycle — the model sits in its basal regime:
median ~84 events over 10 seeds, and a replicate can draw 0 (seed 1 does). The
record's own protocol relaxes deterministically for 36 000 s and then runs SSA
for 3.6×10⁶ s; the manuscript keeps 3600. A superseded vendored copy had an
ODE-equilibrated `Protein=467` baked into its `.net` and measured ~0.25/time;
nothing is baked in now.

### What moved when the rows were re-pointed at the curated records

| model | was | now | why |
|---|---|---|---|
| `samoilov_futile_cycle` | 6 sp / 6 rx | **7 / 10** | the record is the primary file, external noise driver (Expressions 7–8) included; the `_no_driver` variant is the paper's control, not this study's model |
| `gene_expr_3stage` | 6 / 6 | **4 / 6** | the superseded copy carried a `Src()` marker and a `$Null()` sink the record does not; dynamics unchanged |
| `prion_aggregation` | 104 / 2809 | **121 / 3843** | the record's `generate_network` raises `max_iter=>150`, so chains reach its own `max_stoich=>{PrP=>120}` cap; the 17 added species are zero-population chain tails, so the event count is unchanged and only per-event cost rises (~20 % at `t_end=10`) |
| `tcr_signaling` | 37 / 97 | 37 / 97 | same network, but the record starts from the paper's primed state rather than a clean one → ~3 % more events |
| `erk_activation` | 34 / 65 | 34 / 65 | pure relabelling — same seeds, same effective rate constants, activity identical to the event |
| `gene_expression`, `mckane_predator_prey` | — | unchanged | identical network *and* identical measured activity |

**Horizons did not move.** Several records declare a different `t_end` for their
own protocol (`gene_bursts` 3.6×10⁶, `gene_expression` 10⁸,
`samoilov_futile_cycle` 10); Tables 5 and 7 keep the horizons above and the
manuscript documents the divergence.

**One coverage cell moved with them.** BNGL rows reach RoadRunner and COPASI
through `convert_all.py`'s `.net`→SBML conversion, and the curated Samoilov
record's driver contains `N + N -> E+ + N` — second order in the same species.
The SBML law `k*N*N` is not the exact propensity `k*N*(N-1)` and RR-gillespie
fires the SBML law (GH #9), so **`samoilov_futile_cycle`/RoadRunner is N/A**;
COPASI derives the combinatorial propensity itself and its cell stands. The
superseded copy had no driver and so no such reaction. `convert_all.py` now
checks every verdict it computes against `_ssa_config.COVERAGE` and exits
non-zero if they disagree.

## SBML set — BNGsim-SSA vs RoadRunner-SSA vs COPASI-SSA (all exact)

Every engine is run on every model; a cell an engine can't produce is **N/A** + footnote.

| id | study | sp | rx | events/time | bngsim | RR | COPASI | prov |
|---|---|--:|--:|--:|:--:|:--:|:--:|:--:|
| `…478` | Besozzi 2012 Ras/cAMP/PKA (yeast) | 33 | 39 | 11,947 | ✓ | ✓ | ✓ | ✅ |
| `…035` | Vilar 2002 circadian oscillator | 10 | 16 | 2,743 | ✓ | ✓ | ✓ | ✅ |
| `…344` | Proctor 2011 proteostasis | 54 | 80 | 2,281 | ✓ | ✓ | ✓ | ⚠️ |
| `…864` | Proctor 2017 miRNA-OA negFB | 7 | 9 | 30.7 | ✓ | N/A¹ | ✓ | ⚠️ |
| `…860` | Proctor 2017 miRNA-OA posFFL | 4 | 5 | 3.1 | ✓ | N/A¹ | ✓ | ⚠️ |
| `…862` | Proctor 2017 miRNA-OA posFB | 9 | 11 | 0.53 | ✓ | N/A¹ | ✓ | ⚠️ |

¹ RoadRunner-gillespie won't fire time-triggered events → N/A.

**SBML inclusion rule: molecule-count encodings only** (`initialAmount` /
`hasOnlySubstanceUnits=true`). Concentration-unit SBML is an ODE encoding — exact
SSA on it is ill-posed. **Removed for this reason:** `Smith2013` (474, `ROS` driven
negative) and `Karapetyan2016` (586/587, `initialConcentration`). The remaining SBML
models all pass.

Non-Proctor SBML: **Besozzi (478)** and **Vilar (035)**. Vilar was previously also in
the BNGL set; that BNGL encoding was **removed** because it differed in one rate
constant — repressor degradation `delta_R = 0.05` (written `0.2/4` in the RuleHub
source) vs the published Vilar-2002 value `0.2` used here. All 14 other rate
constants and the 16-reaction network are identical; this SBML is the faithful
encoding (and the δR difference is why the two measured 2,743 vs 1,515 events/time).

**Provenance:** ✅ confirmed originally-SSA · ⚠️ probable, Bill to confirm ref / SSA
origin. Details in `PROVENANCE.md`.
