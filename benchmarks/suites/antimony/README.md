# `antimony` suite

Benchmarks BNGsim's Antimony loader against the 117 hand-crafted
Antimony models — emits the cross-engine Antimony figure / table.

## Models

The corpus is vendored in-repo at `../../models/antimony/ssys/` (117
`.ant` files, from the `ssys` project's `test_models1-4`). Because it
is vendored, the runner uses a fixed path — no env-var override, unlike
suites that point at external corpora.

## Runner

`run.py` applies three gates per model:

| Gate | Check |
|------|-------|
| G1 | loads in BNGsim (`Model.from_antimony`) |
| G2 | ODE simulation produces no NaN/Inf |
| G3 | cross-validates vs libRoadRunner (max rel err < 1e-3) |

```sh
python run.py                # full corpus
python run.py --limit 10     # first 10 models (smoke test)
python run.py --model m01_exp_decay
```

Per-model results are written to `results/antimony_sweep_results.json`
(git-ignored — regenerated per machine).
