# `ssa` suite

Benchmarks BNGsim's in-process exact-SSA engine against BNG2.pl's
`run_network` — emits the SSA correctness + timing table.

## Models

The corpus is vendored in-repo at `../../models/net/ssa/` — 12
stochastic models from 2 to 3744 species (pre-generated `.net`
networks, all from BNG 2.9.3). `erk_activation` is registered but
skipped: populations up to 3×10⁶ make exact SSA O(billions of events)
per replicate — it is exercised by the `psa` suite instead.

## Gates

`run.py` applies two gates per model:

| Gate | Check |
|------|-------|
| correctness | An ensemble of replicate trajectories is simulated by each engine; the two ensemble means are compared cell-by-cell with a two-sample *z*-test. Both engines run exact SSA on the same `.net`, so the means must agree within stochastic error. Pass when `max|z|` clears the tolerance (6.0 — it must exceed the extreme-value spread of the per-cell maximum). |
| timing | Warmup + timed-run wall-clock comparison, median reported. |

Per the suite design rule, **timing is only reported for a model that
passed correctness** — a timing number is meaningless if the trajectory
is wrong.

```sh
python run.py                     # both gates, full 12-model sweep
python run.py --mode correctness  # correctness gate only
python run.py --mode timing       # timing gate only
python run.py --effort low        # cheap subset (cumulative tiers)
python run.py --replicates 40     # larger correctness ensemble
```

`run_network` is located via `BNGPATH` / `RUN_NETWORK` (see the
top-level `benchmarks/README.md`). Results are written to the
git-ignored `results/` (`ssa_results.json` + `ssa_results.md`).
