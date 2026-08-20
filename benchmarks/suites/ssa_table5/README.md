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
| `samoilov_futile_cycle` | 7 | 10 | 10 | 1,348,230 | Samoilov, Plyasunov & Arkin 2005, PNAS | ✅ |
| `prion_aggregation` | 121 | 3843 | 300 | 105,156 | Rubenstein 2007 / Lin 2019 | ⚠️ |
| `erk_activation` | 34 | 65 | 8 640 | 68,993\* | Kochańczyk 2017 / Lin 2019 — **slow anchor** | ⚠️ |
| `tcr_signaling` | 37 | 97 | 10 000 | 12,545 | Lipniacki 2008 / Lin 2019 | ✅ |
| `mckane_predator_prey` | 3 | 5 | 1 200 | 216 | McKane & Newman 2005, PRL (demographic) | ✅ |
| `gene_expression` | 10 | 14 | 60 000 | 0.48 | Munsky 2012, Science (Fig 2B) | ✅ |
| `gene_expr_3stage` | 4 | 6 | 2×10⁸ | 0.033 | Shahrezaei & Swain 2008, PNAS | ✅ |
| `gene_bursts` | 2 | 4 | 3 600 | 0.26† | Lin & Doering 2016, Phys Rev E | ✅ |

\* 596M events, ~156 s/replicate at the full horizon — it exceeds
`run_bngsim_activity.py`'s 60 s cap, which records it as `partial` on a reduced
horizon; the number here is a full-horizon measurement.
† `gene_bursts` seeds from the ODE steady state (mRNA=0, Protein=467), produced
by the `relax` step its manifest entry declares and rounded to whole molecules.
Its protocol — in the record and in the superseded source alike — relaxes
deterministically and carries that state into the SSA run; it is never simulated
from the bare `0/0` seeds. The relaxation horizon is converged, so the seed is a
property of the model rather than a free parameter: 3.6×10⁵, 3.6×10⁶ and
3.6×10⁷ s all give the same state, while the record's own 3.6×10⁴ s stops at
`Protein=111.7`, still climbing. **This row's measurement does not move** —
this `.net` and the superseded baked copy both give median 923 / mean 930 events
at `t_end=3600` over 200 seeds, none of them zero. Without the relaxation it
would be median 95, with 25 of 200 replicates firing nothing.

### What moved when the rows were re-pointed at the curated records

| model | was | now | why |
|---|---|---|---|
| `samoilov_futile_cycle` | 6 sp / 6 rx | **7 / 10** | the record is the primary file, external noise driver (Expressions 7–8) included; the `_no_driver` variant is the paper's control, not this study's model. Its horizon moved too — see below |
| `gene_expr_3stage` | 6 / 6 | **4 / 6** | the superseded copy carried a `Src()` marker and a `$Null()` sink the record does not; dynamics unchanged |
| `prion_aggregation` | 104 / 2809 | **121 / 3843** | the record's `generate_network` raises `max_iter=>150`, so chains reach its own `max_stoich=>{PrP=>120}` cap; the 17 added species are zero-population chain tails, so the event count is unchanged and only per-event cost rises (~20 % at `t_end=10`) |
| `tcr_signaling` | 37 / 97 | 37 / 97 | same network, but the record starts from the paper's primed state rather than a clean one → ~3 % more events |
| `erk_activation` | 34 / 65 | 34 / 65 | pure relabelling — same seeds, same effective rate constants, activity identical to the event |
| `gene_bursts` | 2 / 4 | 2 / 4 | same network; its `Protein=467` seed is now *derived* — a declared, converged ODE relaxation of the record — instead of hand-baked into the `.net`, and measures identically (median 923 over 200 seeds, both artifacts) |
| `gene_expression`, `mckane_predator_prey` | — | unchanged | identical network *and* identical measured activity |

### `samoilov_futile_cycle`'s horizon is now the record's own

Settled in [#425](https://github.com/lanl/bngsim/issues/425). `t_end=0.0018` was
chosen for the superseded 6/6 artifact, and once the 7/10 record replaced it that
value had no source. The row now runs the record's own `t_end=10` with
`n_steps=2000`, the Fig. 3A protocol, so the horizon and the model come from the
same place.

`0.0018` is not a number from Samoilov et al. (2005). It is the `@SIM` annotation
of `../../models/antimony/ssys/Samoilov2005.ant`, the file the superseded artifact
was converted from — a **deterministic** ODE encoding ("no external noise") that
lives in a different corpus as a stiffness pathology case for S-system recasting.
Two measurements on the curated record decided the replacement:

- **The model had not started.** At `t_end=0.0018` the trajectory is still in the
  burn-in from the Fig. 3 initial condition — `X*` has fallen only from 2000 to
  about 1600 molecules (median 1606 over 30 seeds), where the published operating
  band is 110–286. Over 30
  seeds it first reaches that band at a median 0.0167 s, roughly nine times later
  than the row was stopping.
- **The cell measured almost no simulation.** Interleaved against a run of the
  same model that fires zero events, **91 % of the bngsim wall and 89 % of the
  `run_network` wall was per-run fixed overhead** — 425 µs against a 386 µs floor,
  and 15.8 ms against 14.1 ms. That is a poor timing cell whatever one thinks
  about the modelling.

Cost is not a reason to keep the short horizon. At `t_end=10` a replicate fires a
median 1.36×10⁷ events for **0.46 s** of bngsim wall and **1.70 s** of
`run_network` wall, both far inside the harness's 120 s per-run cap and in the
range of the other rows. **B10's cost moves by about four orders of magnitude and
the manuscript reports the new number.**

The `_no_driver` variant is the model `0.0018` was chosen for, but it is the
paper's control rather than this study's model, and `wshlavacek/bngsim-paper#6`
chose the primary file deliberately. The RoadRunner cell stays N/A either way, and
the record's `_unordered_pair` variant does not rescue it: it writes the same
`5,5 -> 3,5` reaction with the rate constant halved, so the converted SBML law is
still `k·N·N`.

**Every other horizon is unchanged**, and every one of them is now stated against
its record. Each BNGL row in `corpus.json` carries `record_horizon`, the `t_end`
of the record's own active exact-SSA action, and
`python/tests/test_curated_benchmark_corpus.py` checks it against the record and
fails if a row runs something different without a caveat naming the record's
value. Where they stand:

| model | Table 5 | record's own | |
|---|--:|--:|---|
| `mckane_predator_prey` | 1 200 | 1 200 | same |
| `samoilov_futile_cycle` | 10 | 10 | same, as of #425 |
| `gene_bursts` | 3 600 | 3.6×10⁶ | one cell cycle, the manuscript's |
| `gene_expression` | 6×10⁴ | 10⁸ | the manuscript's |
| `tcr_signaling` | 10 000 | 10 800 | Lin 2019's exact-SSA horizon |
| `gene_expr_3stage` | 2×10⁸ | 2.1×10⁸ | the record discards its first 10⁷ s as burn-in, so both measure the same 2×10⁸ s window |
| `prion_aggregation` | 300 | 10 | Lin 2019's Fig. 7 horizon, 0 to 300 days ([#429](https://github.com/lanl/bngsim/issues/429)). The record's ten-day run is its own, and its protocol note miscredits it to Lin 2019 |
| `erk_activation` | 8 640 | — | the record declares no exact-SSA action at all; 8 640 comes from Lin 2019 |

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
