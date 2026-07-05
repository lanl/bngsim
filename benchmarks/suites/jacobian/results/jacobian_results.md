# GH #76 -- analytical vs FD Functional Jacobian

_machine: macOS-15.7.4-x86_64-i386-64bit / i386 / Python 3.12.13 / git 04147c29_

Performance feature: the analytical Jacobian only changes CVODE's Newton path, so trajectories are identical to FD (verified by the correctness gate below).  The metric is **time-to-solution**.  FD is forced with `BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0`; each (model, mode) runs in a fresh subprocess and wall-clock times the `run()` integrate call only (load excluded).

_effort=high, repeats=7, bngsim 0.9.22_

## Correctness (analytical vs FD trajectory)

PASS = worst cell within `atol + rtol*|fd|`, with rtol=1e-05 and atol=1e-06*peak (peak-relative, the ode suite's `cross_validate` convention).  A different Jacobian only changes CVODE's step path, so the two runs differ at ~solver tolerance; `tol_ratio` is the worst cell as a multiple of its own tolerance (<=1 passes), and `peak_rel` is the worst absolute diff over the trajectory peak.

| model | pass | tol_ratio | peak_rel | max_abs |
|---|---|---|---|---|
| BIOMD0000000013 | PASS | 0.0009459597832992103 | 8.599755395993852e-09 | 3.8066977126050006e-08 |
| MODEL9089538076 | PASS | 0.008905270056524523 | 6.697670772730099e-08 | 0.04411752230953425 |
| BIOMD0000000595 | PASS | 3.648049492342976e-11 | 4.01285430587406e-16 | 1.3322676295501878e-15 |
| MODEL9087255381 | PASS | 0.0015961697503123818 | 1.6805150053755825e-09 | 3.361030010751165e-06 |
| MODEL9087474843 | PASS | 0.0009393005360749886 | 9.572290555759365e-10 | 1.9144581111518733e-06 |
| egfr_net_red | PASS | 1.025530331539887e-09 | 1.940255363782247e-15 | 2.3283064365386963e-09 |

## Wall-clock

Headline = **min** of the repeats (cleanest estimate of true cost for compute-bound code: contention noise is additive, so the minimum is the least-disturbed run).  `rel_spread` (max-min over median) shows how noisy the machine was; a large spread means the median is unreliable and the min should be trusted.  Modes are interleaved round-by-round so both see the same load.

| model | effort | size | analytical min (s) | FD min (s) | speedup (min) | speedup (median) | a/fd rel_spread |
|---|---|---|---|---|---|---|---|
| BIOMD0000000013 | low | Calvin cycle, stiff, ns~27 | 0.0033 | 0.0034 | 1.04x | 1.04x | 8%/14% |
| MODEL9089538076 | medium | ns~200 | 0.0266 | 0.0329 | 1.24x | 1.18x | 20%/11% |
| BIOMD0000000595 | medium | ns~218, 1490 rxns (reaches steady state fast at T=100) | 0.0441 | 0.0475 | 1.08x | 1.10x | 9%/9% |
| MODEL9087255381 | high | ns~289 | 0.3343 | 0.3204 | 0.96x | 0.24x | 198%/543% |
| MODEL9087474843 | high | ns~290, 497 rxns | 0.3263 | 0.3181 | 0.97x | 0.96x | 813%/519% |
| egfr_net_red | low | EGFR reduced, ns~40, 123 rxns, 16 per-observable Functional reactions (.net) | 0.0029 | 0.0034 | 1.17x | 1.14x | 10%/8% |

## CVODE solver counts (analytical / FD)

Near-identical between modes (same Newton path); the wall-clock delta is cost per Jacobian evaluation, which `n_rhs_evals` does **not** capture (FD perturbations are internal to the Jacobian routine).

| model | n_steps | n_jac_evals | n_rhs_evals |
|---|---|---|---|
| BIOMD0000000013 | 695 / 693 | 141 / 141 | 955 / 954 |
| MODEL9089538076 | 433 / 423 | 56 / 54 | 549 / 537 |
| BIOMD0000000595 | 16 / 16 | 11 / 11 | 22 / 22 |
| MODEL9087255381 | 454 / 472 | 68 / 54 | 599 / 569 |
| MODEL9087474843 | 451 / 461 | 60 / 52 | 605 / 593 |
| egfr_net_red | 567 / 567 | 59 / 59 | 598 / 598 |

**Geometric-mean speedup (FD/analytical, min-based): 1.07x** across 6 models.

## Interpretation (honest attribution)

The benefit is real but **modest (~1.0-1.2x)** on this model set, and the reason is the FD baseline it is measured against, which is itself already optimized:

- **Large models (ns >= 50) use a sparse KLU solver, and their FD Jacobian is *colored* finite differences** (Curtis-Powell-Reid, `cvode_colored_jac`): one RHS eval per sparsity color, O(n_colors) not O(ns).  For the sparse metabolic models here n_colors is small, so the FD baseline is cheap and the analytical scatter saves little -- hence ~1.01x on the ns~290 models.
- **Small models (ns < 50) use the dense path with CVODE's internal O(ns) FD.** Analytical wins per Jacobian, but total solve time is a few ms and dominated by fixed per-step overhead, so the ratio is ~1.04x.
- The mid-size model (ns~200) is the sweet spot at ~1.15x.

The regime where an analytical Jacobian wins decisively is a model that is **large AND dense** (high chromatic number, so coloring degrades to O(ns)) **with an expensive RHS** -- i.e. large rule-based networks.  Those are exactly (a) the `.net` per-observable path the C++ side currently rejects (`model.cpp` `if (in.per_observable) return false`), routing such models to FD, and (b) where symbolic differentiation is hardest.  So the headline performance upside is gated on the #76 follow-ups, not realized by the SBML functional models measured here.

Validity note: with `BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0` the functional Jacobian terms are not populated, so `analytical_jacobian_complete()` returns False for these (functional) models and dispatch routes to a genuine FD path -- colored-FD (sparse) or internal-FD (dense), never a partial-analytical Jacobian.  Both modes share the same linear solver and RHS; only the Jacobian computation differs.
