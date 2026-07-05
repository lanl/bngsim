# RuleMonkey vs NFsim — wall-time comparison

Single-machine, sequential, end-to-end subprocess wall time.  3 reps per engine per model; median reported.  Both engines invoked from a fresh process per rep (no warm caches).  Same XML, same t_end, same n_steps, same seed sequence.

**This is an efficiency report, not a correctness report.**  Models where NFsim runs but produces incorrect observables (e.g. `ft_tfun` due to NFsim's TFUN handler returning zero rate) are still included in the timing table; correctness is covered separately by `feature_coverage`, `benchmark_full`, and `basicmodels` suites.

Generated: `2026-05-02 12:51:38 MDT` (`benchmark_rm_vs_nfsim_timing.py`)

## Summary

- Models scored: **169 / 177** (both engines completed all 3 reps)
- NFsim N/A: **8** (RM ran; NFsim refused / errored)
- **Median speedup (NFsim wall ÷ RM wall): 0.86×**
- Geometric mean speedup: 0.77×
- RM faster than NFsim on **68 / 169** models; NFsim faster on 101

## Caveats

- Wall time, not CPU time.  Includes process startup, XML load, and result write — same for both engines, so the ratio is fair, but absolute numbers shift across machines.
- Speedup depends on workload: long-horizon, high-event-rate models show RM's biggest wins; short-horizon trivial models are near parity or NFsim-favored due to RM's slightly heavier per-process startup.
- Single-replicate timings can be noisy.  See min/max columns for spread; the median column is what to quote.
- N/A in the NFsim column means NFsim refused or errored on at least one of the 3 reps.  The reason column gives the captured diagnostic.

## Per-model results

Sorted by RM median wall time (most expensive first).

| # | Suite | Model | t_end | n_steps | RM median | RM min/max | NFsim median | NFsim min/max | Speedup | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | corpus | `t3` | 100 | 100 | 45.2 s | 44.4 s / 45.7 s | 21.1 s | 20.4 s / 25.0 s | 0.47× |  |
| 2 | corpus | `rm_tlbr_rings` | 1000 | 100 | 31.4 s | 30.0 s / 32.6 s | N/A | — | — | NFsim produced no output |
| 3 | corpus | `fceri_ji` | 500 | 500 | 30.9 s | 30.8 s / 31.9 s | 7892 ms | 7788 ms / 8226 ms | 0.26× |  |
| 4 | corpus | `machine` | 14 | 100 | 22.9 s | 22.9 s / 23.1 s | 10.0 s | 10.0 s / 10.1 s | 0.44× |  |
| 5 | corpus | `tcr` | 60 | 12 | 21.3 s | 21.0 s / 21.5 s | 10.9 s | 10.7 s / 11.1 s | 0.51× |  |
| 6 | corpus | `AN` | 100 | 100 | 18.6 s | 18.2 s / 18.6 s | 2172 ms | 2168 ms / 2189 ms | 0.12× |  |
| 7 | corpus | `ensemble` | 15 | 100 | 17.1 s | 17.0 s / 17.2 s | 7790 ms | 7785 ms / 7871 ms | 0.46× |  |
| 8 | corpus | `example2_fit` | 126 | 20 | 16.5 s | 16.1 s / 17.3 s | N/A | — | — | NFsim produced no output |
| 9 | corpus | `example4_fit` | 13 | 100 | 16.3 s | 15.8 s / 16.5 s | 9712 ms | 9168 ms / 9725 ms | 0.60× |  |
| 10 | corpus | `e9` | 200 | 100 | 16.3 s | 16.2 s / 16.5 s | 13.5 s | 13.1 s / 13.6 s | 0.83× |  |
| 11 | corpus | `e8` | 200 | 100 | 14.9 s | 14.7 s / 14.9 s | 11.7 s | 11.7 s / 11.8 s | 0.79× |  |
| 12 | corpus | `egfr_nf_iter5p12h10` | 60 | 20 | 13.9 s | 13.7 s / 14.4 s | N/A | — | — | NFsim produced no output |
| 13 | corpus | `e7` | 200 | 100 | 13.6 s | 13.6 s / 13.7 s | 10.4 s | 10.2 s / 10.7 s | 0.76× |  |
| 14 | corpus | `bench_blbr_rings_posner1995` | 3000 | 300 | 13.2 s | 10.0 s / 14.3 s | 5969 ms | 5908 ms / 5993 ms | 0.45× |  |
| 15 | corpus | `e6` | 200 | 100 | 12.2 s | 12.2 s / 12.6 s | 8765 ms | 8582 ms / 9259 ms | 0.72× |  |
| 16 | corpus | `blbr_rings_posner1995` | 3000 | 300 | 11.8 s | 11.7 s / 12.3 s | 5636 ms | 5603 ms / 6146 ms | 0.48× |  |
| 17 | corpus | `e5` | 200 | 100 | 11.2 s | 10.9 s / 11.3 s | 7555 ms | 7413 ms / 7740 ms | 0.67× |  |
| 18 | corpus | `PushPull` | 4000 | 40 | 10.9 s | 10.9 s / 11.0 s | 4012 ms | 3937 ms / 4167 ms | 0.37× |  |
| 19 | corpus | `e4` | 200 | 100 | 9493 ms | 9424 ms / 9672 ms | 6351 ms | 6095 ms / 6353 ms | 0.67× |  |
| 20 | corpus | `ANx` | 100 | 100 | 9446 ms | 9246 ms / 9449 ms | 1291 ms | 1275 ms / 1324 ms | 0.14× |  |
| 21 | corpus | `e3` | 200 | 100 | 8456 ms | 8342 ms / 8670 ms | 4955 ms | 4850 ms / 5015 ms | 0.59× |  |
| 22 | corpus | `egfr_net` | 120 | 120 | 7710 ms | 7683 ms / 7802 ms | 4443 ms | 4443 ms / 4613 ms | 0.58× |  |
| 23 | corpus | `e2` | 200 | 100 | 7385 ms | 7142 ms / 7541 ms | 3594 ms | 3445 ms / 3682 ms | 0.49× |  |
| 24 | corpus | `lat` | 10 | 100 | 7191 ms | 7126 ms / 8748 ms | 3018 ms | 2957 ms / 3025 ms | 0.42× |  |
| 25 | corpus | `e1` | 200 | 100 | 5961 ms | 5920 ms / 6008 ms | 2117 ms | 2101 ms / 2152 ms | 0.36× |  |
| 26 | nfsim_basicmodels | `r02` | 100 | 100 | 4512 ms | 4485 ms / 4519 ms | 1130 ms | 1115 ms / 1143 ms | 0.25× |  |
| 27 | corpus | `bench_blbr_dembo1978_monovalent_inhibitor` | 3000 | 300 | 4283 ms | 4183 ms / 4345 ms | N/A | — | — | NFsim produced no output |
| 28 | corpus | `pltr` | 1000 | 1000 | 3632 ms | 3418 ms / 3638 ms | 79.5 s | 79.4 s / 84.1 s | 21.89× |  |
| 29 | corpus | `mlnr` | 1000 | 1000 | 3334 ms | 3299 ms / 3351 ms | 79.9 s | 77.8 s / 82.5 s | 23.97× |  |
| 30 | nfsim_basicmodels | `r01` | 100 | 100 | 3286 ms | 3247 ms / 3303 ms | 1344 ms | 1333 ms / 1375 ms | 0.41× |  |
| 31 | nfsim_basicmodels | `r20` | 100 | 100 | 3262 ms | 3247 ms / 3336 ms | 1415 ms | 1396 ms / 1418 ms | 0.43× |  |
| 32 | corpus | `oscSystem` | 200 | 200 | 3007 ms | 2887 ms / 3101 ms | 401 ms | 381 ms / 414 ms | 0.13× |  |
| 33 | nfsim_basicmodels | `r07` | 20 | 100 | 2838 ms | 2824 ms / 2843 ms | 884 ms | 877 ms / 891 ms | 0.31× |  |
| 34 | corpus | `testcase2a` | 1000 | 1000 | 2699 ms | 2672 ms / 3014 ms | 30.9 s | 30.0 s / 31.1 s | 11.44× |  |
| 35 | corpus | `tlbr` | 200 | 200 | 2642 ms | 2624 ms / 2650 ms | 2337 ms | 2332 ms / 2369 ms | 0.88× |  |
| 36 | nfsim_basicmodels | `r06` | 20 | 100 | 2610 ms | 2604 ms / 2636 ms | 503 ms | 477 ms / 507 ms | 0.19× |  |
| 37 | corpus | `rm_tlbr` | 1000 | 1000 | 2580 ms | 2572 ms / 2618 ms | 30.2 s | 30.0 s / 30.4 s | 11.69× |  |
| 38 | corpus | `simple_system` | 100 | 50 | 2124 ms | 2094 ms / 2149 ms | 896 ms | 869 ms / 917 ms | 0.42× |  |
| 39 | nfsim_basicmodels | `r08` | 20 | 100 | 2111 ms | 2099 ms / 2124 ms | 489 ms | 484 ms / 492 ms | 0.23× |  |
| 40 | corpus | `simple_nfsim` | 100 | 50 | 2111 ms | 2103 ms / 2129 ms | 908 ms | 827 ms / 926 ms | 0.43× |  |
| 41 | corpus | `nfsim_aggregation_gelation` | 50 | 200 | 2104 ms | 2093 ms / 2130 ms | 790 ms | 781 ms / 794 ms | 0.38× |  |
| 42 | corpus | `tcr_gen27ind33` | 0.94 | 12 | 2004 ms | 1952 ms / 2049 ms | 819 ms | 808 ms / 867 ms | 0.41× |  |
| 43 | corpus | `tcr_iter28p4h2` | 0.94 | 12 | 1985 ms | 1943 ms / 1987 ms | 823 ms | 822 ms / 834 ms | 0.41× |  |
| 44 | nfsim_basicmodels | `r10` | 100 | 100 | 1896 ms | 1892 ms / 1909 ms | 341 ms | 339 ms / 342 ms | 0.18× |  |
| 45 | corpus | `tcr_iter9p44` | 0.94 | 12 | 1863 ms | 1824 ms / 1868 ms | 790 ms | 761 ms / 942 ms | 0.42× |  |
| 46 | corpus | `tcr_gen20ind9` | 1.5 | 12 | 1823 ms | 1818 ms / 1826 ms | 788 ms | 774 ms / 801 ms | 0.43× |  |
| 47 | nfsim_basicmodels | `r04` | 50 | 100 | 1769 ms | 1762 ms / 1792 ms | 748 ms | 741 ms / 753 ms | 0.42× |  |
| 48 | corpus | `bench_blbr_cooperativity_posner2004` | 10 | 10 | 1280 ms | 1252 ms / 1286 ms | 704 ms | 699 ms / 711 ms | 0.55× |  |
| 49 | corpus | `bench_blbr_cooperativity_posner2004_rings` | 10 | 10 | 1174 ms | 1161 ms / 1209 ms | 624 ms | 617 ms / 643 ms | 0.53× |  |
| 50 | corpus | `stiff` | 10 | 100 | 978 ms | 970 ms / 995 ms | 471 ms | 471 ms / 480 ms | 0.48× |  |
| 51 | corpus | `example3_fit` | 5000 | 10 | 870 ms | 825 ms / 876 ms | 2173 ms | 2170 ms / 2316 ms | 2.50× |  |
| 52 | corpus | `BLBR` | 10 | 100 | 704 ms | 693 ms / 706 ms | 620 ms | 611 ms / 624 ms | 0.88× |  |
| 53 | corpus | `isingspin_localfcn` | 5000 | 100 | 603 ms | 599 ms / 615 ms | 162 ms | 159 ms / 164 ms | 0.27× |  |
| 54 | nfsim_basicmodels | `r03` | 50 | 100 | 557 ms | 541 ms / 560 ms | 240 ms | 237 ms / 242 ms | 0.43× |  |
| 55 | corpus | `rm_blbr` | 10 | 100 | 479 ms | 474 ms / 480 ms | 241 ms | 240 ms / 250 ms | 0.50× |  |
| 56 | nfsim_basicmodels | `r18` | 10 | 100 | 434 ms | 428 ms / 465 ms | 284 ms | 275 ms / 300 ms | 0.65× |  |
| 57 | nfsim_basicmodels | `r12` | 10 | 100 | 433 ms | 422 ms / 441 ms | 157 ms | 156 ms / 163 ms | 0.36× |  |
| 58 | corpus | `bench_blbr_heterogeneity_goldstein1980` | 10 | 10 | 418 ms | 416 ms / 429 ms | 107 ms | 106 ms / 108 ms | 0.26× |  |
| 59 | corpus | `CaMKII_holo` | 10 | 100 | 413 ms | 406 ms / 436 ms | 563 ms | 562 ms / 570 ms | 1.36× |  |
| 60 | feature_coverage | `ss_long_polymer` | 500 | 100 | 376 ms | 373 ms / 377 ms | 160 ms | 159 ms / 161 ms | 0.43× |  |
| 61 | nfsim_basicmodels | `r09` | 100 | 100 | 355 ms | 351 ms / 357 ms | 63.4 ms | 62.1 ms / 64.9 ms | 0.18× |  |
| 62 | corpus | `st_multi_2` | 10 | 100 | 343 ms | 342 ms / 375 ms | 151 ms | 141 ms / 227 ms | 0.44× |  |
| 63 | corpus | `toy_jim` | 100 | 100 | 336 ms | 335 ms / 338 ms | 135 ms | 135 ms / 140 ms | 0.40× |  |
| 64 | feature_coverage | `ft_multimol_sym_obs` | 20 | 100 | 332 ms | 326 ms / 337 ms | 72.7 ms | 71.5 ms / 97.2 ms | 0.22× |  |
| 65 | feature_coverage | `ss_branching_aggregate` | 400 | 100 | 270 ms | 268 ms / 273 ms | 154 ms | 153 ms / 154 ms | 0.57× |  |
| 66 | corpus | `ANx_noActivity` | 100 | 100 | 255 ms | 251 ms / 268 ms | 130 ms | 116 ms / 138 ms | 0.51× |  |
| 67 | corpus | `poly` | 100 | 100 | 225 ms | 218 ms / 233 ms | 122 ms | 120 ms / 122 ms | 0.54× |  |
| 68 | corpus | `bench_tlbr_yang2008` | 3000 | 300 | 202 ms | 202 ms / 205 ms | 188 ms | 188 ms / 194 ms | 0.93× |  |
| 69 | corpus | `bench_blbr_rings_posner1995_no_rings` | 50 | 50 | 197 ms | 183 ms / 198 ms | 155 ms | 152 ms / 157 ms | 0.79× |  |
| 70 | corpus | `bench_blbr_dembo1978` | 40 | 40 | 181 ms | 178 ms / 186 ms | 148 ms | 147 ms / 188 ms | 0.82× |  |
| 71 | corpus | `A_plus_B_rings` | 10 | 10000 | 166 ms | 164 ms / 167 ms | 165 ms | 164 ms / 167 ms | 0.99× |  |
| 72 | corpus | `bench_tlbr_solution_macken1982` | 500 | 250 | 157 ms | 153 ms / 159 ms | 116 ms | 113 ms / 117 ms | 0.74× |  |
| 73 | feature_coverage | `ss_symmetric_homopoly` | 300 | 100 | 150 ms | 149 ms / 151 ms | 125 ms | 125 ms / 128 ms | 0.83× |  |
| 74 | feature_coverage | `ft_stiff_system` | 200 | 100 | 147 ms | 146 ms / 148 ms | 45.3 ms | 44.2 ms / 45.3 ms | 0.31× |  |
| 75 | feature_coverage | `ft_multisite_phospho` | 200 | 100 | 143 ms | 142 ms / 144 ms | 69.4 ms | 68.9 ms / 70.2 ms | 0.49× |  |
| 76 | nfsim_basicmodels | `r11` | 1 | 100 | 133 ms | 131 ms / 135 ms | 52.0 ms | 51.6 ms / 52.4 ms | 0.39× |  |
| 77 | feature_coverage | `ss_tlbr_rings` | 400 | 100 | 112 ms | 110 ms / 112 ms | 134 ms | 132 ms / 135 ms | 1.20× |  |
| 78 | corpus | `A_plus_A_mixed_1` | 30 | 100 | 111 ms | 110 ms / 112 ms | 44.8 ms | 43.7 ms / 45.1 ms | 0.40× |  |
| 79 | nfsim_basicmodels | `r13` | 30 | 100 | 105 ms | 98.8 ms / 106 ms | 23.7 ms | 23.4 ms / 23.8 ms | 0.23× |  |
| 80 | corpus | `st_multi_1` | 10 | 100 | 99.9 ms | 99.4 ms / 104 ms | 44.1 ms | 43.2 ms / 45.9 ms | 0.44× |  |
| 81 | nfsim_basicmodels | `r14` | 30 | 100 | 73.3 ms | 72.0 ms / 75.5 ms | 27.3 ms | 27.3 ms / 29.0 ms | 0.37× |  |
| 82 | feature_coverage | `ft_push_pull` | 200 | 100 | 72.4 ms | 71.5 ms / 73.9 ms | 28.8 ms | 28.6 ms / 30.0 ms | 0.40× |  |
| 83 | nfsim_basicmodels | `r21` | 100 | 100 | 69.7 ms | 69.5 ms / 72.8 ms | 86.1 ms | 86.1 ms / 86.1 ms | 1.24× |  |
| 84 | feature_coverage | `ft_receptor_dimerization` | 100 | 100 | 56.5 ms | 56.4 ms / 57.8 ms | 29.0 ms | 27.3 ms / 31.5 ms | 0.51× |  |
| 85 | nfsim_basicmodels | `r16` | 5 | 100 | 44.7 ms | 44.6 ms / 46.2 ms | 20.2 ms | 19.7 ms / 20.6 ms | 0.45× |  |
| 86 | feature_coverage | `ft_cooperative_binding` | 100 | 100 | 44.3 ms | 41.0 ms / 44.4 ms | 16.5 ms | 16.4 ms / 16.5 ms | 0.37× |  |
| 87 | feature_coverage | `combo_strict_product_plus` | 100 | 100 | 37.8 ms | 36.8 ms / 39.1 ms | 20.3 ms | 19.3 ms / 22.0 ms | 0.54× |  |
| 88 | corpus | `receptor_nf_iter36p0h3` | 60 | 60 | 32.8 ms | 31.7 ms / 33.6 ms | 20.2 ms | 20.1 ms / 20.5 ms | 0.61× |  |
| 89 | corpus | `Tutorial_Example` | 5 | 50 | 30.7 ms | 30.3 ms / 32.2 ms | 18.5 ms | 18.3 ms / 18.6 ms | 0.60× |  |
| 90 | nfsim_basicmodels | `r15` | 1 | 100 | 27.7 ms | 27.1 ms / 28.0 ms | 15.7 ms | 15.0 ms / 15.7 ms | 0.57× |  |
| 91 | corpus | `example6_ground_truth` | 60 | 60 | 26.9 ms | 26.8 ms / 28.6 ms | 17.3 ms | 16.4 ms / 17.5 ms | 0.64× |  |
| 92 | corpus | `A_plus_A` | 10 | 100 | 22.0 ms | 21.2 ms / 22.7 ms | 14.4 ms | 13.5 ms / 14.4 ms | 0.66× |  |
| 93 | feature_coverage | `nf_branching_aggregate` | 200 | 100 | 21.5 ms | 20.7 ms / 23.9 ms | 17.6 ms | 17.3 ms / 18.2 ms | 0.82× |  |
| 94 | nfsim_basicmodels | `r05` | 1 | 100 | 21.2 ms | 20.2 ms / 22.5 ms | 15.7 ms | 15.5 ms / 15.8 ms | 0.74× |  |
| 95 | feature_coverage | `ft_continue` | 100 | 100 | 19.6 ms | 19.3 ms / 20.6 ms | 12.7 ms | 12.5 ms / 13.1 ms | 0.65× |  |
| 96 | feature_coverage | `ft_blbr` | 100 | 100 | 16.2 ms | 15.7 ms / 17.8 ms | 14.4 ms | 14.1 ms / 14.6 ms | 0.89× |  |
| 97 | feature_coverage | `ft_tlbr` | 100 | 100 | 15.5 ms | 15.3 ms / 18.1 ms | 15.0 ms | 14.6 ms / 15.3 ms | 0.97× |  |
| 98 | corpus | `nfsim_coarse_graining` | 400 | 200 | 15.5 ms | 15.0 ms / 16.4 ms | 14.7 ms | 14.4 ms / 15.0 ms | 0.95× |  |
| 99 | feature_coverage | `nf_large_multivalent` | 100 | 100 | 15.2 ms | 13.7 ms / 15.5 ms | 13.5 ms | 13.4 ms / 13.9 ms | 0.89× |  |
| 100 | feature_coverage | `ft_multimol_pattern_sym_nonreacting` | 10 | 40 | 15.1 ms | 14.8 ms / 15.8 ms | 11.7 ms | 11.4 ms / 12.3 ms | 0.78× |  |
| 101 | corpus | `nfsim_dynamic_compartments` | 20 | 100 | 13.9 ms | 13.2 ms / 21.6 ms | 13.2 ms | 10.2 ms / 14.0 ms | 0.95× |  |
| 102 | feature_coverage | `ft_include_reactants` | 100 | 100 | 13.7 ms | 13.2 ms / 13.9 ms | 10.1 ms | 9.9 ms / 10.9 ms | 0.73× |  |
| 103 | feature_coverage | `combo_localfcn_multisite` | 200 | 100 | 13.0 ms | 12.5 ms / 14.3 ms | 11.6 ms | 11.5 ms / 11.7 ms | 0.90× |  |
| 104 | feature_coverage | `ft_perturbation` | 100 | 100 | 12.8 ms | 11.7 ms / 14.5 ms | 11.0 ms | 10.6 ms / 11.0 ms | 0.86× |  |
| 105 | feature_coverage | `ft_exclude_products` | 200 | 50 | 12.5 ms | 11.4 ms / 12.8 ms | 10.0 ms | 9.7 ms / 11.6 ms | 0.80× |  |
| 106 | corpus | `st` | 10 | 100 | 12.2 ms | 12.1 ms / 13.9 ms | 11.0 ms | 10.6 ms / 11.3 ms | 0.90× |  |
| 107 | feature_coverage | `nf_linear_polymer` | 200 | 100 | 12.1 ms | 11.8 ms / 13.9 ms | 10.9 ms | 10.5 ms / 12.7 ms | 0.90× |  |
| 108 | feature_coverage | `combo_addbond_connected` | 200 | 100 | 12.0 ms | 11.7 ms / 12.7 ms | 10.2 ms | 10.1 ms / 11.2 ms | 0.85× |  |
| 109 | feature_coverage | `combo_symmetric_rings` | 100 | 100 | 12.0 ms | 11.9 ms / 14.3 ms | 13.6 ms | 13.4 ms / 14.7 ms | 1.13× |  |
| 110 | feature_coverage | `combo_synth_degrade_equilibrium` | 500 | 100 | 11.7 ms | 10.4 ms / 13.3 ms | 10.5 ms | 10.5 ms / 11.3 ms | 0.89× |  |
| 111 | feature_coverage | `ft_multi_site_binding` | 100 | 100 | 11.5 ms | 11.2 ms / 12.5 ms | 11.7 ms | 11.6 ms / 11.9 ms | 1.02× |  |
| 112 | feature_coverage | `combo_shorthand_embed` | 100 | 100 | 11.3 ms | 10.6 ms / 11.6 ms | 11.2 ms | 11.0 ms / 11.5 ms | 0.99× |  |
| 113 | feature_coverage | `edg_multi_pattern_obs` | 30 | 60 | 11.0 ms | 10.8 ms / 11.5 ms | 9.5 ms | 9.3 ms / 9.8 ms | 0.86× |  |
| 114 | feature_coverage | `ft_signaling_cascade` | 100 | 100 | 11.0 ms | 10.3 ms / 11.0 ms | 11.3 ms | 10.8 ms / 11.5 ms | 1.03× |  |
| 115 | corpus | `basicTLBR` | 20 | 100 | 10.9 ms | 10.5 ms / 11.3 ms | 14.9 ms | 14.8 ms / 15.0 ms | 1.37× |  |
| 116 | feature_coverage | `edg_branched_polymer` | 30 | 60 | 10.5 ms | 9.0 ms / 11.0 ms | 14.3 ms | 14.2 ms / 16.0 ms | 1.37× |  |
| 117 | nfsim_basicmodels | `r19` | 10 | 100 | 10.3 ms | 9.6 ms / 11.6 ms | 17.1 ms | 17.0 ms / 17.5 ms | 1.67× |  |
| 118 | nfsim_basicmodels | `r26` | 100 | 100 | 10.1 ms | 8.8 ms / 10.4 ms | 11.0 ms | 10.9 ms / 11.8 ms | 1.10× |  |
| 119 | nfsim_basicmodels | `r24` | 100 | 100 | 10.0 ms | 8.8 ms / 10.2 ms | 11.1 ms | 10.9 ms / 11.6 ms | 1.11× |  |
| 120 | feature_coverage | `combo_multimol_unimol` | 100 | 100 | 9.9 ms | 9.8 ms / 11.1 ms | 10.7 ms | 10.6 ms / 10.7 ms | 1.07× |  |
| 121 | feature_coverage | `ft_energy_patterns` | 500 | 100 | 9.9 ms | 9.6 ms / 10.3 ms | 11.0 ms | 10.6 ms / 11.4 ms | 1.11× |  |
| 122 | feature_coverage | `ft_multimol_unimol_unbind_sym` | 2 | 40 | 9.7 ms | 8.8 ms / 12.0 ms | 8.5 ms | 8.4 ms / 8.8 ms | 0.87× |  |
| 123 | feature_coverage | `ft_population_map` | 60 | 60 | 9.7 ms | 9.1 ms / 10.2 ms | 10.2 ms | 9.5 ms / 11.2 ms | 1.05× |  |
| 124 | feature_coverage | `ft_exclude_reactants` | 100 | 100 | 9.6 ms | 9.0 ms / 10.3 ms | 9.8 ms | 9.5 ms / 10.1 ms | 1.03× |  |
| 125 | corpus | `nfsim_ring_closure_polymer` | 50 | 200 | 9.4 ms | 8.9 ms / 9.7 ms | 10.6 ms | 10.3 ms / 10.8 ms | 1.13× |  |
| 126 | feature_coverage | `ft_complex_seed` | 50 | 50 | 9.3 ms | 9.1 ms / 10.1 ms | 9.2 ms | 8.8 ms / 9.6 ms | 0.99× |  |
| 127 | feature_coverage | `edg_pattern_local_fcn` | 40 | 80 | 9.3 ms | 9.2 ms / 10.2 ms | 9.8 ms | 9.5 ms / 10.1 ms | 1.05× |  |
| 128 | feature_coverage | `ft_ring_closure` | 100 | 100 | 9.1 ms | 9.1 ms / 9.5 ms | 10.2 ms | 9.6 ms / 10.2 ms | 1.11× |  |
| 129 | feature_coverage | `ft_nested_functions` | 100 | 100 | 9.1 ms | 8.9 ms / 10.2 ms | N/A | — | — | NFsim produced no output |
| 130 | feature_coverage | `ft_bond_wildcards` | 100 | 100 | 9.0 ms | 9.0 ms / 10.1 ms | 10.0 ms | 9.8 ms / 10.3 ms | 1.12× |  |
| 131 | feature_coverage | `ft_species_vs_molecules` | 50 | 100 | 8.8 ms | 8.5 ms / 9.0 ms | 9.5 ms | 9.5 ms / 9.6 ms | 1.07× |  |
| 132 | nfsim_basicmodels | `r25` | 100 | 100 | 8.7 ms | 8.4 ms / 10.6 ms | 11.2 ms | 11.0 ms / 11.4 ms | 1.28× |  |
| 133 | nfsim_basicmodels | `r30` | 100 | 100 | 8.7 ms | 7.6 ms / 10.1 ms | 8.9 ms | 8.8 ms / 9.1 ms | 1.02× |  |
| 134 | feature_coverage | `ft_catalytic_unbinding` | 100 | 100 | 8.4 ms | 7.9 ms / 8.8 ms | 10.0 ms | 9.6 ms / 11.0 ms | 1.20× |  |
| 135 | feature_coverage | `edg_state_wildcard_set` | 40 | 80 | 8.3 ms | 7.8 ms / 10.3 ms | 9.7 ms | 9.4 ms / 10.1 ms | 1.16× |  |
| 136 | feature_coverage | `combo_exclude_with_complex` | 100 | 100 | 8.3 ms | 8.3 ms / 11.3 ms | 9.7 ms | 9.5 ms / 10.0 ms | 1.17× |  |
| 137 | feature_coverage | `edg_three_mol_pattern` | 30 | 60 | 8.2 ms | 8.1 ms / 9.7 ms | 9.9 ms | 9.5 ms / 10.6 ms | 1.20× |  |
| 138 | feature_coverage | `ft_competitive_binding` | 100 | 100 | 8.1 ms | 7.9 ms / 8.9 ms | 10.0 ms | 9.9 ms / 10.3 ms | 1.24× |  |
| 139 | feature_coverage | `edg_homotrimer_binding` | 50 | 100 | 8.1 ms | 7.4 ms / 8.9 ms | 9.7 ms | 9.7 ms / 10.0 ms | 1.20× |  |
| 140 | feature_coverage | `ft_multi_product` | 200 | 100 | 8.0 ms | 7.3 ms / 9.0 ms | 9.3 ms | 8.9 ms / 9.7 ms | 1.16× |  |
| 141 | feature_coverage | `ft_receptor_heterogeneity` | 100 | 100 | 8.0 ms | 7.7 ms / 10.7 ms | 9.9 ms | 9.7 ms / 11.4 ms | 1.23× |  |
| 142 | feature_coverage | `edg_oscillator` | 40 | 200 | 8.0 ms | 7.8 ms / 8.7 ms | 10.9 ms | 10.8 ms / 11.4 ms | 1.37× |  |
| 143 | feature_coverage | `ft_multi_op_rule` | 100 | 100 | 7.8 ms | 7.4 ms / 8.1 ms | 9.0 ms | 9.0 ms / 9.4 ms | 1.15× |  |
| 144 | nfsim_basicmodels | `r29` | 100 | 100 | 7.8 ms | 7.4 ms / 10.0 ms | 8.9 ms | 8.7 ms / 9.1 ms | 1.14× |  |
| 145 | feature_coverage | `ft_multistate` | 300 | 100 | 7.7 ms | 7.1 ms / 9.1 ms | 9.8 ms | 9.6 ms / 10.1 ms | 1.27× |  |
| 146 | feature_coverage | `ft_clamped_species_strict` | 50 | 50 | 7.7 ms | 7.2 ms / 9.3 ms | 7.9 ms | 7.8 ms / 8.0 ms | 1.03× |  |
| 147 | feature_coverage | `edg_state_increment_chain` | 30 | 60 | 7.7 ms | 7.6 ms / 10.6 ms | 8.9 ms | 8.9 ms / 9.0 ms | 1.17× |  |
| 148 | nfsim_basicmodels | `r22` | 100 | 100 | 7.6 ms | 7.4 ms / 9.9 ms | 9.4 ms | 9.2 ms / 10.2 ms | 1.23× |  |
| 149 | corpus | `gene_expr_func` | 1000 | 100 | 7.5 ms | 5.9 ms / 7.5 ms | 8.4 ms | 8.2 ms / 8.9 ms | 1.11× |  |
| 150 | feature_coverage | `ft_local_functions` | 100 | 100 | 7.4 ms | 7.4 ms / 9.2 ms | 9.5 ms | 9.4 ms / 9.6 ms | 1.27× |  |
| 151 | nfsim_basicmodels | `r23` | 100 | 100 | 7.2 ms | 7.2 ms / 10.7 ms | 9.8 ms | 9.4 ms / 10.5 ms | 1.35× |  |
| 152 | feature_coverage | `ft_tfun` | 40 | 80 | 7.2 ms | 5.9 ms / 7.5 ms | N/A | — | — | NFsim refused XML |
| 153 | feature_coverage | `ft_state_wildcards` | 200 | 100 | 7.1 ms | 6.8 ms / 7.6 ms | 10.2 ms | 8.9 ms / 10.3 ms | 1.45× |  |
| 154 | feature_coverage | `edg_synth_bonded_complex` | 40 | 80 | 7.1 ms | 6.5 ms / 7.7 ms | 8.8 ms | 8.3 ms / 9.2 ms | 1.24× |  |
| 155 | feature_coverage | `ft_match_once` | 10 | 50 | 7.0 ms | 6.5 ms / 7.2 ms | 8.4 ms | 8.3 ms / 9.0 ms | 1.19× |  |
| 156 | feature_coverage | `edg_compound_op_swap` | 30 | 60 | 7.0 ms | 6.2 ms / 7.4 ms | 8.8 ms | 8.6 ms / 9.0 ms | 1.25× |  |
| 157 | feature_coverage | `edg_fixed_competition` | 40 | 80 | 6.9 ms | 6.2 ms / 7.8 ms | 9.0 ms | 8.7 ms / 9.4 ms | 1.30× |  |
| 158 | feature_coverage | `ft_synthesis_degradation` | 500 | 100 | 6.9 ms | 6.6 ms / 8.0 ms | 8.6 ms | 8.5 ms / 8.9 ms | 1.25× |  |
| 159 | feature_coverage | `edg_double_state_change` | 30 | 60 | 6.9 ms | 6.4 ms / 7.0 ms | 8.5 ms | 8.2 ms / 8.6 ms | 1.23× |  |
| 160 | feature_coverage | `edg_ring_break_constraint` | 10 | 40 | 6.8 ms | 6.3 ms / 7.6 ms | 8.0 ms | 7.7 ms / 8.5 ms | 1.17× |  |
| 161 | feature_coverage | `edg_dynamic_rate_zero_obs` | 40 | 80 | 6.7 ms | 5.9 ms / 9.6 ms | 8.4 ms | 8.4 ms / 8.9 ms | 1.24× |  |
| 162 | feature_coverage | `ft_delete_molecules` | 200 | 100 | 6.7 ms | 6.3 ms / 7.6 ms | 9.2 ms | 9.2 ms / 9.5 ms | 1.39× |  |
| 163 | feature_coverage | `edg_self_dimerize` | 30 | 60 | 6.6 ms | 6.4 ms / 7.8 ms | 8.3 ms | 8.3 ms / 8.5 ms | 1.25× |  |
| 164 | corpus | `A_plus_A_mixed_2` | 30 | 100 | 6.6 ms | 6.4 ms / 8.0 ms | 9.3 ms | 9.0 ms / 9.9 ms | 1.41× |  |
| 165 | feature_coverage | `ft_functional_rate` | 300 | 100 | 6.6 ms | 6.1 ms / 7.3 ms | 8.6 ms | 8.3 ms / 9.1 ms | 1.32× |  |
| 166 | feature_coverage | `ft_total_rate` | 100 | 100 | 6.6 ms | 6.0 ms / 6.7 ms | 8.4 ms | 8.3 ms / 8.6 ms | 1.27× |  |
| 167 | feature_coverage | `ft_clamped_species` | 100 | 100 | 6.5 ms | 5.9 ms / 6.7 ms | 8.0 ms | 8.0 ms / 8.6 ms | 1.23× |  |
| 168 | feature_coverage | `edg_synth_bind_existing` | 30 | 60 | 6.4 ms | 6.3 ms / 8.4 ms | 8.4 ms | 8.0 ms / 8.6 ms | 1.32× |  |
| 169 | feature_coverage | `ft_mm_ratelaw` | 30 | 60 | 6.3 ms | 6.0 ms / 7.3 ms | 8.2 ms | 8.2 ms / 8.7 ms | 1.30× |  |
| 170 | feature_coverage | `ft_conditional_rate` | 500 | 100 | 6.3 ms | 6.2 ms / 7.9 ms | 8.4 ms | 8.4 ms / 9.1 ms | 1.34× |  |
| 171 | feature_coverage | `edg_seeded_ring` | 20 | 40 | 6.3 ms | 6.1 ms / 7.1 ms | 9.0 ms | 8.6 ms / 10.3 ms | 1.43× |  |
| 172 | nfsim_basicmodels | `r32` | 5 | 200 | 6.3 ms | 6.0 ms / 8.0 ms | 8.8 ms | 8.7 ms / 9.2 ms | 1.41× |  |
| 173 | feature_coverage | `edg_zero_rate_rule` | 40 | 80 | 6.2 ms | 6.0 ms / 6.7 ms | 8.3 ms | 8.0 ms / 8.3 ms | 1.32× |  |
| 174 | nfsim_basicmodels | `r17` | 5 | 50 | 6.2 ms | 6.0 ms / 7.8 ms | 8.4 ms | 8.2 ms / 8.9 ms | 1.35× |  |
| 175 | feature_coverage | `edg_deep_param_chain` | 40 | 80 | 6.1 ms | 5.8 ms / 7.6 ms | N/A | — | — | NFsim produced no output |
| 176 | corpus | `nfsim_hybrid_particle_field` | 10 | 50 | 6.0 ms | 5.8 ms / 6.7 ms | 8.0 ms | 7.9 ms / 8.7 ms | 1.33× |  |
| 177 | feature_coverage | `edg_time_dependent_rate` | 50 | 100 | 5.9 ms | 5.7 ms / 6.8 ms | N/A | — | — | NFsim produced no output |
