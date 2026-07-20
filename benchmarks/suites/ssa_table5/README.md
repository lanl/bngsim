# `ssa_table5` — exact-SSA corpus for arXiv Table 5

Model set for arXiv Table 5 (per-model exact Gillespie-SSA cost across engines).
Every model is published **and** was simulated with exact SSA **in its source
study**, at the published time horizon. A *corpus* (models + horizons +
provenance), not a runner. Current size: **8 BNGL + 6 SBML**.

`events/time` = BNGsim exact-SSA activity (Gillespie events per unit simulated
time), 1 replicate, from `results/bngsim_activity.json` (`run_bngsim_activity.py`).

## BNGL set — BNGsim-SSA vs `run_network`

| model | sp | rx | t_end | events/time | reference | prov |
|---|--:|--:|--:|--:|---|:--:|
| `samoilov_futile_cycle` | 6 | 6 | 0.0018 | 833,333 | Samoilov, Plyasunov & Arkin 2005, PNAS | ✅ |
| `erk_activation` | 34 | 65 | 8 640 | ~106,000\* | Kochańczyk 2017 / Lin 2019 — **slow anchor** | ⚠️ |
| `prion_aggregation` | 104 | 2809 | 300 | 105,166 | Rubenstein 2007 / Lin 2019 | ⚠️ |
| `tcr_signaling` | 37 | 97 | 10 000 | 13,445 | Lipniacki 2008 / Lin 2019 | ✅ |
| `mckane_predator_prey` | 3 | 5 | 1 200 | 216 | McKane & Newman 2005, PRL (demographic) | ✅ |
| `gene_expression` | 10 | 14 | 60 000 | 0.48 | Munsky 2012, Science (Fig 9) | ✅ |
| `gene_bursts` | 2 | 4 | 3 600 | ~0.25† | Lin & Doering 2016, Phys Rev E | ✅ |
| `gene_expr_3stage` | 6 | 6 | 2×10⁸ | 0.033 | Shahrezaei & Swain 2008, PNAS | ✅ |

\* erk exceeds the 60 s cap at full horizon; activity measured on a 10×-reduced
horizon (rate is representative; full-horizon wall ~130 s).
† `gene_bursts` seed species are the **ODE-equilibrated steady state** (mRNA=0,
Protein=467) baked into the `.net` — from bare zero ICs the model is near-inert
(basal transcription only), so it measured 0; equilibrated it is ~0.13–0.33/rep.

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
