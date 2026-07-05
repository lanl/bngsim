# `nf` suite

Benchmarks BNGsim's in-process `NfsimSimulator` against BNG2.pl's
standalone NFsim — emits the network-free (NFsim) correctness table.

## Models

The corpus is vendored in-repo at `../../models/bngl/nf/` — 8 core
models plus 2 experimental canaries. Because it is vendored, the runner
uses a fixed path (no env-var override, unlike the external corpora).

## Runner

`run.py`, for each model:

1. generates XML via BNG2.pl `writeXML()`,
2. runs BNG2.pl `simulate({method=>"nf",...})` → reference `.gdat`,
3. runs BNGsim `NfsimSimulator` in-process,
4. compares — same seed should give an exact match.

```sh
python run.py                                  # 8 core models
BNGSIM_NF_INCLUDE_EXPERIMENTAL=1 python run.py  # + 2 canaries (EXPERIMENTAL.md)
```

`BNG2.pl` is located via `BNGPATH` / `BNG2_PL` (see the top-level
`benchmarks/README.md`). Other knobs: `BNGSIM_NF_V1143_COMPAT`,
`BNGSIM_NF_CONNECTIVITY`. Results are written to the git-ignored
`results/nf_sweep_results.json`.
