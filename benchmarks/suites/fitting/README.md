# PyBioNetFit multi-model fitting benchmark for the bngsim paper

This folder holds the PyBioNetFit fitting benchmark that populates the
fitting-benchmark table of the bngsim paper
(`bngsim/dev/paper/latex/main.tex`). It measures how PyBNF performs with
the legacy subprocess stack (BNG2.pl / run\_network) versus the
in-process `bngsim` backend, across several models.

## What is being compared

Several BioNetGen ODE fitting problems, each fit **twice** with identical
scatter-search settings and a fixed iteration budget:

- **subprocess** — `bngl_backend=bionetgen`: PyBNF shells out to
  BNG2.pl / run\_network for every simulation.
- **bngsim** — `bngl_backend=bngsim`: PyBNF runs the simulation
  in-process through the `bngsim` extension.

The two confs of a problem differ only in `bngl_backend` and
`output_dir`, so each problem yields one table row: subprocess vs.
in-process wall-clock for the *same* fitting work, plus the objective
each backend reached.

Optimization only — scatter search (`fit_type=ss`). No MCMC sampling.

## The problems

Drawn from the PyBNF iScience-2019 example set (Data S1, `mmc3/`). The
benchmark is **ODE-only**; the problems below are the BioNetGen-ODE
problems of that set's Table 2 that run clean on *both* PyBNF backends.

| Dir                | Problem                  | Free params | Effort |
|--------------------|--------------------------|-------------|--------|
| `prob05_threestep` | P5 three-step cascade    | 3  | low    |
| `kozer_egfr`       | P10 Kozer EGFR           | 9  | high   |
| `prob07_egg`       | P7 egg-shaped curve      | 10 | low    |
| `prob24_jnk`       | P24 Jnk cascade          | 12 | medium |
| `prob18_mapk`      | P18 MAPK scaffold        | 13 | medium |

The **effort** column tiers each problem by cost so the benchmark can be
run as a cheap subset — see *Effort tiers* below.

### Problems that were considered and dropped

Smoke testing found five other BioNetGen-ODE problems that hit PyBNF /
bngsim-bridge integration bugs, not benchmark-design issues:

- **prob19/20 (RAFi)** — the in-process backend crashes on
  `parameter_scan` + `steady_state=>1` (internal#45).
- **prob15/06 (IGF1R, degranulation)** — the bngsim bridge rejects their
  interleaved `parameter_scan` / `setConcentration` action blocks
  (internal#46).
- **prob13 (receptor)** — PyBNF v1.3.0's `find_t_length` parser fails on
  the model, both backends (lanl/PyBNF issue #390).

SSA and network-free probe rows were also dropped — this is an ODE-only
benchmark. The network-free half of the original single-model Kozer
benchmark is preserved, runnable, under `_stash/kozer_networkfree/`.

## Effort tiers

The full benchmark is a multi-hour sweep. To run only a cheap subset,
`run_jobs.py` and `consistency_check.py` take `--effort {low,medium,high}`
with **cumulative** semantics:

| `--effort` | Problems run                                              |
|------------|-----------------------------------------------------------|
| `low`      | `prob05_threestep`, `prob07_egg`                          |
| `medium`   | the above **+** `prob24_jnk`, `prob18_mapk`               |
| `high`     | the above **+** `kozer_egfr` — **all problems** (default) |

The default is `high`, so omitting the flag runs the whole benchmark
exactly as before. `--effort low` is the quick check; `--effort medium`
adds the mid-cost models but still skips the expensive Kozer EGFR row.

Each problem's tier lives in the `effort` field of `harness/problems.py`.
The tiers were assigned from a short iters=3 calibration — combined
subprocess + in-process wall clock per problem:

| Problem            | Combined wall clock | Tier   |
|--------------------|---------------------|--------|
| `prob05_threestep` | 15.6 s              | low    |
| `prob07_egg`       | 31.7 s              | low    |
| `prob18_mapk`      | 116.9 s             | medium |
| `prob24_jnk`       | 142.9 s             | medium |
| `kozer_egfr`       | not calibrated      | high   |

`kozer_egfr` builds a 913-species / ~12k-reaction network (~8 min of
network generation per job); it is too expensive to calibrate and is
tiered `high` by fiat. `harness/calibrate.py` regenerates
`results/_calib_runs.json` (it skips `kozer_egfr` unless
`--include-skipped` is passed).

## Layout

```
fitting/
├── README.md                 # this file
├── harness/
│   ├── problems.py            # the benchmark problem registry (one place)
│   ├── make_problem_confs.py  # regenerate the per-problem conf pairs
│   ├── calibrate.py           # iters=3 timing run -> results/_calib_runs.json
│   ├── run_jobs.py            # run every job, time it, capture objectives
│   └── consistency_check.py   # verify the two backends agree on the biology
├── emit.py                    # render the LaTeX table fragment from results/
├── <problem dirs>/
│   ├── bngl/                  # model file(s)
│   ├── data/                  # .exp / .prop data file(s)
│   └── conf/                  # subprocess.conf + bngsim.conf (+ _smoke)
├── results/                   # results JSON, _calib_runs.json
├── _stash/kozer_networkfree/  # stashed network-free Kozer jobs (not on the hot path)
└── mmc3/                      # PyBNF iScience-2019 Data S1 (gitignored)
```

The per-problem confs are generated by `harness/make_problem_confs.py`
from the `mmc3/` `-ss.conf` templates (the `kozer_egfr` confs are
hand-curated and not generated). To change the run length, edit the
`iters` field in that script's `PROBLEMS` table and re-run it.

## How to run

`bngsim` must be importable in the same environment as PyBioNetFit, and
BNG2.pl must be reachable. Export `BNGPATH`:

```
export BNGPATH=~/Simulations/BioNetGen-2.9.3
export PYBNF_CMD=~/Code/PyBNF/.venv/bin/pybnf
```

1. **Consistency check.** Verifies the two backends agree on the biology
   before the timings are trusted. Both backends are run briefly with a
   pinned `random_seed`, so scatter search explores the identical
   parameter sets; the objectives must then agree to tight tolerance.

   ```
   python harness/consistency_check.py
   ```

   Results: `results/consistency.json`. Takes the same
   `--effort {low,medium,high}` subset flag as `run_jobs.py`.

2. **Run the benchmark.**

   ```
   python harness/run_jobs.py
   ```

   Results stream to `results/fitting_runs.json` after every job, so an
   interrupted run is resumable: `--only <slug>` re-runs specific
   problems, `--backend <subprocess|bngsim>` one backend, and
   `--effort {low,medium,high}` runs only a cost-bounded subset (default
   `high` = the full sweep — see *Effort tiers* above). For a quick
   check: `python harness/run_jobs.py --effort low`.

3. **Render the table.**

   ```
   python emit.py
   ```

   Writes `bngsim/dev/paper/latex/generated/fitting.tex`, which
   `main.tex` pulls in with `\input{generated/fitting.tex}`.

## Cores

Every conf pins `parallel_count=4` — 4 of the 6 cores on the target
laptop, leaving headroom for the Dask scheduler. `run_jobs.py` runs the
problems serially, so total concurrency stays at 4.

## Smoke confs

`make_problem_confs.py --smoke` also writes `conf/{subprocess,bngsim}_smoke.conf`
(one iteration, small population, 3 cores) for re-verifying a backend
path cheaply. They are not part of the benchmark; `run_jobs.py` ignores
them.
