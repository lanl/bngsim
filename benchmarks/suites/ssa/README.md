# `ssa` suite

Benchmarks BNGsim's in-process exact-SSA engine against BNG2.pl's
`run_network` — emits the SSA correctness + timing table.

## Models

12 stochastic models from 2 to 3744 species, pre-generated `.net`
networks, all from BNG 2.9.3. `erk_activation` is registered but
skipped: populations up to 3×10⁶ make exact SSA O(billions of events)
per replicate — it is exercised by the `psa` suite instead.

They come from two places, resolved per model by `run._resolve_artifacts`:

- **five** — `gene_expression`, `gene_expr_3stage`, `tcr_signaling`,
  `erk_activation`, `prion_aggregation` — are models the manuscript
  names, so they are generated from their curated `BNGL-Models` records
  into `../../models/net/curated/` and **shared with `suites/ssa_table5`
  and `suites/psa`**, which run the same networks under different
  protocols. Species/reaction counts come from
  `../../models/curated_nets.json`, not from a hand-typed copy.
- **the other seven** stay at `../../models/net/ssa/`.

This suite used to vendor its own copies of all twelve. For those five
the copies were byte-identical to a copy in another suite, which is
precisely how three separate `tcr_signaling` networks came to drift
apart from each other and from the record.

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
