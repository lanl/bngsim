# BNGsim Showcase

Two runnable showcases used in the manuscript:

- **Antimony cross-engine comparison** — BNGsim (ExprTK) vs. libRoadRunner vs.
  AMICI on the hand-crafted Antimony corpus (Fig 2). Also feeds the `antimony`
  benchmark suite's summary table.
- **Minimal Python ODE-fit demo** — the manuscript code box: universal `.net`
  parser + BNGsim ODE sensitivities + SciPy trust-region-reflective fitting.

See the sections below. (The EGFR PyBNF scatter-search fit and the Becker
multistart/profile notebooks now live in the PyBNF repository.)

## Reproducible Setup

Use the showcase setup helper to install benchmark dependencies in one step.

From repo root:

```bash
uv run --active python bngsim/benchmarks/suites/showcase/setup_showcase_env.py
```

Defaults:

- Installs `bngsim` editable into the active uv environment.
- Installs pinned benchmark tool versions from
  `bngsim/benchmarks/suites/showcase/requirements_showcase_pinned.txt`.
- If you need the Antimony memory-fix build before it is on PyPI, install the
  wheel from:
  `https://github.com/sys-bio/antimony/actions/runs/22922160697/`

Options:

- `--mode latest` installs latest package releases instead of pinned versions.
- `--remove-dask` removes dask/distributed (only for AMICI-only workflows).
- `--skip-bngsim-reinstall` skips reinstalling editable `bngsim`.

## Antimony Cross-Engine Showcase

This showcase compares BNGsim (ExprTK mode) against libRoadRunner and AMICI
on the hand-crafted Antimony corpus in
`bngsim/benchmarks/models/antimony/ssys`.

Key behavior:

- Time horizons come from per-model `@SIM` comment tags (`T_START`, `T_END`, `N_STEPS`).
- Each model is trajectory-checked before timing.
- Timing points are included in the figure only when consistency passes
  (`max_rel_err <= consistency_rtol`, default `1e-3`).
- Solver tolerances are applied consistently across all three engines
  (defaults: `solver_rtol=1e-8`, `solver_atol=1e-8`).
- Consistency uses mixed normalization
  `|a-b| / max(|a|, |b|, consistency_atol)` with default
  `consistency_atol=1e-8` for stable near-zero behavior.
- The generated figure is single-panel and suitable for main-text use.

### Run

From repo root:

```bash
uv run --active python bngsim/benchmarks/suites/showcase/run_ant_exprtk_3engine.py
```

Useful flags:

- `--quick N` limits to first `N` Antimony files for smoke runs.
- `--skip-amici` runs BNGsim ExprTK vs libRoadRunner only.
- `--figure-path bngsim/dev/paper/fig_ant_exprtk_rr_amici.png` chooses figure output path.
- `--update-main-tex` inserts/updates an auto-generated figure block in
  `bngsim/dev/paper/latex/main.tex`.

Outputs:

- JSON: `bngsim/benchmarks/suites/showcase/results/<run_name>/results.json`
- Figure: `bngsim/dev/paper/fig_ant_exprtk_rr_amici.png` (and `.pdf`)
- LaTeX snippet: `bngsim/dev/paper/latex/generated/ant_exprtk_rr_amici_figure.tex`

## Minimal Python ODE Fit Demo

This runnable script mirrors the manuscript code box: universal `.net` parser
path + BNGsim ODE sensitivities + SciPy trust-region reflective fitting.

From repo root:

```bash
uv run --active python bngsim/benchmarks/suites/showcase/run_ode_trf_fit_from_net.py
```

Useful flags:

- `--net path/to/model.net` to fit another network.
- `--params k1,k2` to choose a parameter subset.
- `--data observed.npy` to use external data instead of synthetic data.
- `--outdir bngsim/benchmarks/suites/showcase/results/<run_name>` for output control.
