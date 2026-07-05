# BNGsim Benchmark Suites

This directory holds the benchmark suites that validate BNGsim against
BioNetGen's standalone engines (`run_network`, NFsim) and other reference
simulators. Each suite lives in `suites/<name>/` with its own runner and
`README.md`; see the Directory Structure section below.

## Quick Start — One-command regen via `run_all.py`

```bash
# Regen every paper artifact from every suite (walk-away-friendly):
python bngsim/benchmarks/run_all.py

# Cheap smoke (low-effort tiers only, takes minutes not hours):
python bngsim/benchmarks/run_all.py --effort low

# Just refresh generated/*.tex from already-collected results:
python bngsim/benchmarks/run_all.py --emit-only

# Dry-run -- print the planned commands, don't execute:
python bngsim/benchmarks/run_all.py --dry-run

# Print the suite registry (deps, effort-aware, BNG2.pl-shelling):
python bngsim/benchmarks/run_all.py --list
```

`run_all.py` is the **paper-regen orchestrator** added in Phase 6. It
runs every `suites/<name>/run.py` then every `suites/<name>/emit.py`,
respecting cross-suite ordering (`showcase` before `antimony`,
`python_ssa` before `python_ode`), and writes a per-run summary at
`results/run_all_<UTC>/{summary.json,summary.md}` plus per-suite
`{run,emit}.{stdout,stderr}.log`. See `_dev/phase6_plan.md` for the
locked design.

### Sharding a regen across machines

The orchestrator is designed for a *walk-away, maybe-shard* workflow:

- **Continue-on-failure by default.** A flake on one suite does not
  kill the rest. Final exit code is non-zero if any suite failed.
  Use `--halt-on-failure` to opt out.
- **Skip with reason on missing engines.** Each suite has a precheck
  (e.g., `AMICI not importable`, `SBML_TEST_SUITE_DIR not found`). A
  failing precheck reports `skipped: <reason>` in the summary; the
  sweep continues. Right default when sharding -- Machine A may have
  AMICI, Machine B may not.
- **Selection flags.** `--only ode,ssa,psa,nf,fitting` on Machine A
  and `--only biomodels,sbml_test_suite,forward_sens` on Machine B.
  Each writes to the shared `bngsim/dev/paper/latex/generated/<name>.tex`
  paths, so collecting the work is just `git commit` or `rsync`.
- **Serial by default.** Memory note `feedback_scope_process_kills`:
  concurrent heavy benchmarks have bitten us. `--parallel N` is
  opt-in; suites that shell out to BNG2.pl are gated by a single
  semaphore even under `--parallel >1`.

### Audience filtering: `paper_role` + `--audience`

The 10 table-rendering `emit.py` files (`ode`, `ssa`, `psa`, `nf`,
`antimony`, `fitting`, `forward_sens`, `python_ode`, `steady_state`,
`sbml_test_suite`) accept a `--audience {main,supp,all}` flag
(default `all`). Output filename:

| Audience | Filter | Output file |
|---|---|---|
| `all` (default) | drop only `paper_role=skip` rows | `generated/<suite>.tex` |
| `main` | only `paper_role=main` rows | `generated/<suite>_main.tex` |
| `supp` | `paper_role in {main, supp}` rows | `generated/<suite>_supp.tex` |

A row's `paper_role` is set in the suite's MODELS registry (or
PROBLEMS, CORE_MODELS, TARGET_MODELS, etc. -- whichever the suite
uses). Missing or `None` defaults to `supp`, so adding the field is
backward-compatible and no row is currently tagged. `paper_role=skip`
runs the row as part of the benchmark but never renders it into the
paper. `run_all.py` always invokes the default audience -- re-rendering
the `_main.tex` / `_supp.tex` variants is a manual `emit.py --audience`
call.

This is the *refinement-friendliness* hook for future paper iteration
(e.g., a reviewer asking "move X rows from supp to main") -- the
mechanism is plumbed; the curation pass happens later, separately.

## Quick Start — Per-suite runners

```bash
# Activate the PyBNF virtual environment
source .venv/bin/activate

# ── ODE suite — correctness + timing (BNGsim vs run_network) ─────
python bngsim/benchmarks/suites/ode/run.py    # 42 models, ~15 min

# ── SSA suite — correctness + timing (BNGsim vs run_network) ─────
python bngsim/benchmarks/suites/ssa/run.py    # 12 models
python bngsim/benchmarks/suites/ssa/run.py --effort low   # cheap subset

# ── PSA suite — correctness + timing (BNGsim vs run_network) ─────
python bngsim/benchmarks/suites/psa/run.py    # 3 models x Nc sweep

# ── NFsim Correctness Sweep (BNGsim vs BNG2.pl NFsim) ────────────
python bngsim/benchmarks/suites/nf/run.py           # 8 supported standalone-NFsim models, ~1 min
BNGSIM_NF_V1143_COMPAT=1 \
python bngsim/benchmarks/suites/nf/run.py           # same suite, legacy NFsim v1.14.3 seed-path mode
BNGSIM_NF_CONNECTIVITY=off \
python bngsim/benchmarks/suites/nf/run.py           # force wrapper connectivity=false for parity A/B
BNG2_PL=/path/to/newer/BNG2.pl \
BNGSIM_NF_INCLUDE_EXPERIMENTAL=1 \
python bngsim/benchmarks/suites/nf/run.py           # +2 opt-in external-NFsim canaries (not baseline tests)
```

Each suite writes its results into its own git-ignored `results/`
directory (`<suite>/results/`), as a machine-readable `.json` plus a
formatted `.md` report.

## Environment Variables

Benchmark scripts resolve tool and model-corpus paths from environment
variables, falling back to the canonical local layout. Set these to run on a
machine where things live elsewhere; no script edit is needed.

**BioNetGen 2.9.3 tools** — set `BNGPATH` to the install root and BNG2.pl,
`run_network`, and `NFsim` are derived from it. A per-tool variable overrides
just that binary.

| Variable | Purpose | Default |
|----------|---------|---------|
| `BNGPATH` | BioNetGen 2.9.3 install root | `~/Simulations/BioNetGen-2.9.3` |
| `BNG2_PL` | BNG2.pl override | `$BNGPATH/BNG2.pl` |
| `RUN_NETWORK` | `run_network` binary override | `$BNGPATH/bin/run_network` |
| `NFSIM` | `NFsim` binary override | `$BNGPATH/bin/NFsim` |

**Model corpora** — roots scanned for benchmark BNGL models.

| Variable | Purpose | Default |
|----------|---------|---------|
| `BNG_MODELS2` | BNG2 distributed `Models2/` | `~/Simulations/bionetgen/bng2/Models2` |
| `PYBNF_EXAMPLES` | PyBNF `examples/` tree | `~/Code/PyBNF/examples` |
| `RULEHUB_DIR` | RuleHub clone | `/tmp/RuleHub` |
| `RULEBENDER_WS` | RuleBender workspace | `~/Simulations/RuleBender-workspace` |
| `RULEBENDER_OLD_WS` | archived RuleBender workspace (`_dev/triage_bngl.py` only) | `~/Simulations/OLD_SAVE/RuleBender-workspace copy` |

AMICI, libRoadRunner, and libSBML are imported as Python packages — they are
resolved by the active virtual environment and need no path variable.

**`run_all.py` precheck overrides.** The orchestrator looks for
optional dependencies in this order: env var (if set) → canonical
install path → import (for Python packages). Set `BENCH_PYTHON` to
override the default `.venv-biomodels/bin/python` interpreter; set
`SBML_TEST_SUITE_DIR` to point at a checkout of
`https://github.com/sbmlteam/sbml-test-suite` (the orchestrator
checks `~/Code/sbml-test-suite` if unset). Missing optional engines
result in the affected suite being **skipped** with a printed reason,
not in a hard failure.

## Directory Structure

> **Reorg complete (Phase 6).** Benchmark suites live in
> `suites/<name>/` (each its own runner + README), with models
> vendored into `models/`. `bngsim/benchmarks/run_all.py` is the
> orchestrator that drives every suite end-to-end.

- `run_all.py` — orchestrator: runs every suite's `run.py` + `emit.py`,
  honors deps, writes per-run logs and a summary
- `_emit.py`, `_effort.py`, `_netbench.py` — shared helpers

- `models/bngl/<src>/`, `models/net/<src>/` — canonical de-duplicated
  model store; `<src>` is provenance or suite role (see `models/README.md`)
- `suites/ode/` — ODE correctness + timing vs `run_network` (42 models)
- `suites/ssa/` — exact-SSA correctness + timing vs `run_network` (12 models)
- `suites/psa/` — PSA correctness + timing vs `run_network` (3 models × Nc)
- `suites/nf/` — NFsim correctness vs BNG2.pl NFsim (8 core + 2 experimental)
- `suites/antimony/` — Antimony loader vs libRoadRunner (117 models)
- `suites/biomodels/` — BioModels assessment (3-engine SBML coverage + accuracy)
- `suites/fitting/` — PyBioNetFit multi-model fitting: subprocess vs in-process BNGsim
- `suites/showcase/` — paper Fig. 1 showcase runs (EGFR 4-way fit, Antimony ExprTk 3-engine, ODE TRF fit)
- `suites/sbml_roundtrip/` — BNG SBML-writer round-trip of the SSA corpus + loader smoke test
- `suites/steady_state/` — KINSOL dose-response correctness + timing (5 steady-state strategies, 4 models)
- `suites/sbml_test_suite/` — SBML semantic-suite compatibility (BNGsim/RR/AMICI; external corpus via `SBML_TEST_SUITE_DIR`)
- `suites/forward_sens/` — CVODES forward-sensitivity vs AMICI (BNGsim serial + sharded, 4 models)
- `suites/python_ode/` — pure-Python ODE workflow comparison (BNGsim CVODE / scipy LSODA / Diffrax)
- `suites/python_ssa/` — pure-Python SSA workflow comparison (BNGsim Gillespie vs gillespy2)
- `bngxml/` — BioNetGen-to-XML fixtures, not standalone-NFsim runtime tests (inline `tfun`)
- `_dev/` — non-paper dev scripts (Jacobian/codegen/sensitivity benchmarks, diagnostics)

### NFsim Benchmark Classification

- `models/bngl/nf/` holds both the core models (run end-to-end through
  standalone NFsim) and the experimental canaries; `suites/nf/run.py`
  runs the core set by default and the canaries opt-in. See
  `suites/nf/EXPERIMENTAL.md`.
- `t_dor2` is an experimental canary because standalone NFsim currently rejects that two-reactant DOR case.
- `bngxml/` contains BNGL fixtures that exercise BioNetGen XML emission only.
- Current BNGsim NFsim default uses `connectivity=false`.
- As of the March 18, 2026 validation on commit `0e67edb`, `connectivity=true`
  is correctness-clean on the supported compat sweep and on same-XML off/on
  checks across all 8 `nf/` models.
- Clean-selector timing is not a general win for `connectivity=true`
  (geometric-mean on/off ratio on the suite: `1.1834x`, i.e. `+18.34%`).
- Keep `BNGSIM_NF_CONNECTIVITY=on` for targeted experiments or
  model-specific workloads; any wrapper-default flip should be a separate
  follow-up change.

---

# SSA/PSA Stochastic Benchmark Models

This section describes the SSA/PSA models. All models use integer molecule
counts and mass-action propensities, making them suitable for stochastic simulation.

Each model is provided as both a `.bngl` file (BioNetGen Language source) and a
pre-generated `.net` file (reaction network) for direct loading by BNGsim.

## Model Catalog

### Small Models (1–10 species)

| Model | File | Species | Reactions | Max Pop | Behavior |
|-------|------|---------|-----------|---------|----------|
| Gene expression (Hill) | `gene_expression_hill` | 2 | 4 | ~1000 | Positive autoregulation, bimodal distribution |
| Flagellar motor | `flagellar_motor` | 4 | 4 | 2200 | CheY-driven motor switching via global functions |
| Simple system | `simple_system` | 4 | 4 | 5000 | Binding + phosphorylation/dephosphorylation |
| Gene expression (3-stage) | `gene_expr_3stage` | 6 | 6 | ~100 | Bursty promoter switching, analytically tractable |
| Oscillatory system | `oscillatory_system` | 6 | 8 | ~50 | Negative-feedback oscillations, functional rates |
| Gene expression (constitutive + regulated) | `gene_expression` | 10 | 14 | ~25 | Two-state gene model, multiple distribution shapes |

### Medium Models (10–50 species)

| Model | File | Species | Reactions | Max Pop | Behavior |
|-------|------|---------|-----------|---------|----------|
| ERK activation | `erk_activation` | 34 | 65 | 3×10⁶ | Oscillatory/bistable ERK pulses; PSA recommended |
| TCR signaling | `tcr_signaling` | 37 | 97 | 3×10⁵ | Bistable stochastic switching |

### Large Models (50+ species)

| Model | File | Species | Reactions | Max Pop | Behavior |
|-------|------|---------|-----------|---------|----------|
| Prion aggregation | `prion_aggregation` | 104 | 2809 | 7500 | Nucleated polymerization, stochastic seeding |
| EGFR signaling | `egfr_net` | 356 | 3749 | 12000 | EGF-stimulated receptor phosphorylation cascade |
| Multisite phosphorylation | `multisite_phos` | 1026 | 7680 | 3000 | 5-site substrate, ultrasensitive switch |
| FcεRI signaling | `fceri_gamma` | 3744 | 58276 | 6000 | Receptor aggregation, transphosphorylation |

## References

- **gene_expression**: Munsky B, Neuert G, van Oudenaarden A (2012) Science 336: 183–187.
- **gene_expression_hill**: Lin YT, Doering CR (2016) Phys Rev E 93: 022409.
- **gene_expr_3stage**: Shahrezaei V, Swain PS (2008) PNAS 105: 17256–17261.
- **simple_system**: NFsim distribution example model.
- **flagellar_motor**: NFsim distribution (motor switching model).
- **oscillatory_system**: NFsim distribution (negative-feedback oscillator).
- **multisite_phos**: NFsim distribution (multisite phosphorylation, 5 sites).
- **egfr_net**: Blinov ML et al. (2006) Biosystems 83: 136–151.
- **erk_activation**: Kochańczyk M et al. (2017) Sci Rep 7: 38244.
- **tcr_signaling**: Lipniacki T, Hat B, Faeder JR, Hlavacek WS (2008) J Theor Biol 254: 110–122.
- **prion_aggregation**: Rubenstein R et al. (2007) Biophys Chem 125: 360–367.
- **fceri_gamma**: Goldstein B, Faeder JR, Bhatt S (2004) Biophys J 87: 2234–2244.
- **PSA benchmark (TCR, Prion, ERK)**: Lin YT, Feng S, Hlavacek WS (2019) J Chem Phys 150: 244101.

## Sources

Models were collected from:
- PyBNF examples (`examples/fceri_gamma/`)
- BioNetGen test suite (`Models2/` in the BNG2 distribution)
- RuleBender workspace (`YTL_PRE_geneModel/`)
- RuleHub PSA benchmark models (Lin et al. 2019)
- NFsim distribution (`models/` directory), modernized to current BNGL conventions

### Modernization of NFsim models

The following models were modernized from the NFsim distribution:
- Added `begin model`/`end model` wrapper
- `begin molecule type` → `begin molecule types` (plural)
- `simulate_nf(...)` / `writeXML()` → `generate_network({overwrite=>1})` + `simulate({method=>"ssa",...})`
- Added `begin molecule types` block where missing (egfr_net)
- Renamed `_Na` → `NA` to avoid conflict with BNGsim built-in constants
- Uncommented CheY reactions in flagellar_motor for dynamic behavior

### Selection criteria

1. BNGL model with `generate_network` + `simulate({method=>"ssa",...})` in actions block
2. Integer molecule counts (not continuous concentrations)
3. Mass-action kinetics (functional rates allowed if observable-based)
4. Network generation completes in <60 s
5. No external dependencies (Sat/Hill rate laws are rejected; NFsim-only models excluded)

## Usage with BNGsim

```python
import bngsim

# Load a pre-generated .net file
model = bngsim.Model("tcr_signaling.net")

# ODE simulation
sim_ode = bngsim.Simulator(model, method="ode")
result_ode = sim_ode.run(t_end=300, n_steps=1000)

# SSA simulation (exact stochastic)
sim_ssa = bngsim.Simulator(model, method="ssa")
result_ssa = sim_ssa.run(t_end=300, n_steps=1000, seed=42)

# PSA simulation (approximate, faster for large populations)
sim_psa = bngsim.Simulator(model, method="psa", poplevel=300)
result_psa = sim_psa.run(t_end=300, n_steps=1000, seed=42)
```

## .net File Compatibility

- **All .net files must be generated by BNG 2.9.3** (or later). Files from BNG 2.3.1
  contain `scaledUpSpecies`/`lambda` artifacts in the `begin reactions` block that cause
  `run_network 3.0` to crash with parse errors.
- **run_network requires `-g netfile`** for models with functional rate laws. This flag
  registers observable groups before function evaluation begins.
- **To regenerate a .net file**: `perl /path/to/BNG2.pl model.bngl` — this runs
  `generate_network` from the actions block and produces a fresh `.net` file.

## Benchmarking

The SSA and PSA models are exercised by `suites/ssa/` and `suites/psa/`
— each runs a correctness gate (BNGsim vs `run_network` ensemble
comparison) and a timing comparison. See those suites' own `README.md`
for the gate details and CLI flags.

```bash
python suites/ssa/run.py     # SSA correctness + timing
python suites/psa/run.py     # PSA correctness + timing
```

## Notes

- **ERK activation**: Exact SSA is impractical due to populations up to 3×10⁶
  (billions of events needed). Use PSA with `poplevel` for efficient simulation.
- **FcεRI signaling**: Very large network (58K reactions). Network generation takes
  ~60 s via BNG2.pl. Useful for stress-testing simulator performance.
- **Prion aggregation**: Network truncated at `max_stoich=>{PrP=>120}`. Chain lengths
  beyond 120 are not populated under the default parameter settings.
- **Multisite phosphorylation**: 5 independent phospho sites on substrate S, each
  enzymatically phosphorylated/dephosphorylated. Combinatorial explosion: 1026 species
  from 2^5=32 phospho states × enzyme-binding combinations.
- **Oscillatory system / Flagellar motor**: These models use global functions
  (observable-dependent rates). Tests BNGsim's functional rate law support in SSA.
