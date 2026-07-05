# BioModels 4-Engine Benchmark Instructions

## Scope and naming

The canonical script is:

```bash
python bngsim/harness/comparison/bench_1013_4engine.py
```

- Default run scope is the manifest union (`all_required_ids`, currently ~1013 models).
- Output/checkpoint stem is `bench_1013_4engine`.

## Pool materialization policy

Pool ensure is intentionally opt-in in harnesses.

- Enable ensure in harness: `--ensure-pool` (or `BENCH_AUTO_ENSURE_POOL=1`)
- Force skip ensure: `--skip-pool-ensure` (or `BENCH_SKIP_POOL_ENSURE=1`)
- Restrict to cached SBML only while ensuring: `--pool-no-fetch`

`--quick N` is applied to discovery and to ensure subset when ensure is enabled, so quick runs do
not pull the full corpus.

## Where Antimony files live

Default pool directory:

```bash
bngsim/benchmarks/biomodels_ant/
```

`BENCH_ANT_DIR` can point to an alternate directory. Manifest-aware consumers also support:

- top-level `*.ant`
- `review_extra/*.ant`

## Typical commands

```bash
# Smoke test, first 10 of active slice, ensure enabled
python bench_1013_4engine.py --quick 10 --ensure-pool

# Full ~1013 run (if pool already materialized)
python bench_1013_4engine.py

# Full ~1013 run with on-demand materialization
python bench_1013_4engine.py --ensure-pool

# Resume interrupted run
python bench_1013_4engine.py --resume
```

## Outputs

Primary JSON:

```bash
bngsim/harness/results/bench_1013_4engine.json
```

Phase checkpoints:

```bash
bngsim/harness/results/bench_1013_4engine_phase1.json
bngsim/harness/results/bench_1013_4engine_phase2.json
bngsim/harness/results/bench_1013_4engine_phase3.json
```

## Plot generation

```bash
python bngsim/harness/comparison/plot_1013_4panel.py \
  --json bngsim/harness/results/bench_1013_4engine.json
```

The plotting script now defaults to `bench_1013_4engine.json`.
