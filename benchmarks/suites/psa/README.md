# `psa` suite

Benchmarks BNGsim's in-process PSA engine against BNG2.pl's
`run_network` — emits the PSA correctness + timing table.

PSA — the **partial-scaling approximation** (Lin, Feng & Hlavacek 2019,
*J Chem Phys* **150**: 244101) — trades exactness for speed on models
whose populations make exact SSA infeasible. Each model is swept over a
list of `Nc` population levels.

## Models

The corpus is vendored in-repo at `../../models/net/psa/` — 3 models
(`tcr_signaling`, `erk_activation`, `prion_aggregation`), each swept
over `Nc ∈ {10, 30, 100, 300}`. Pre-generated `.net` networks, all from
BNG 2.9.3.

## Gates

`run.py` applies two gates per `(model, Nc)`:

| Gate | Check |
|------|-------|
| correctness | BNGsim PSA and `run_network` PSA each simulate a replicate ensemble; the two ensemble means are compared with a **standardized-mean-difference** statistic (the mean gap in units of the process σ). Pass when the 99.9th-percentile cell agrees within 6 σ. |
| timing | Warmup + timed-run wall-clock comparison, median reported. |

Per the suite design rule, **timing is reported only for a `(model, Nc)`
that passed correctness**.

### Why not a *z*-test

The `ssa` suite compares two *exact* engines with a *z*-test — they must
agree in distribution, so a vanishing mean gap is the right expectation.
PSA is different: the two engines run **different** scaling algorithms
(BNGsim partial-scaling vs `run_network` heterogeneous adaptive
scaling), so they carry a real, bounded method bias by construction. A
*z*-test divides by the standard error of the mean, whose shrinkage with
the replicate count would eventually flag that legitimate bias as a
failure. The effect-size statistic divides by the *process* σ instead —
it is replicate-count-stable, tolerates the method bias, and still
catches a gross discrepancy (a wrong network or wrong rates drive the
percentile far past tolerance).

This is therefore a cross-engine **consistency** check. Exact SSA is
infeasible for these models — that is the motivation for PSA — so there
is no per-`Nc` ground truth; the approximation accuracy as a function of
`Nc` is characterized in Lin et al. 2019, not here.

```sh
python run.py                     # both gates, all models x Nc
python run.py --mode correctness  # correctness gate only
python run.py --mode timing       # timing gate only
python run.py --effort low        # cheap subset (cumulative tiers)
python run.py --replicates 40     # larger correctness ensemble
```

`run_network` is located via `BNGPATH` / `RUN_NETWORK` (see the
top-level `benchmarks/README.md`). Results are written to the
git-ignored `results/` (`psa_results.json` + `psa_results.md`).
