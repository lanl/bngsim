# `ode` suite

Benchmarks BNGsim's ODE engine against BNG2.pl's `run_network` — emits
the ODE benchmark table.

## Models

A curated list of **42** published rule-based models, each generating a
reaction network that is integrated as an ODE system. The corpus is
vendored in-repo:

| `src` | Location | Models |
|-------|----------|--------|
| `models2` | `../../models/bngl/models2/` | 20 — BNG2 distribution |
| `pybnf` | `../../models/bngl/pybnf/` | 2 — PyBNF examples |
| `rulehub` | `../../models/bngl/rulehub/` | 7 — RuleHub published |
| `ode_bench` | `../../models/bngl/ode/` | 13 — 5 RuleBender + 8 SSA-bench |

`simple_nfsim` was dropped — it is a network-free (NFsim) model that
`BNG2.pl` cannot expand into a reaction network, so it has no ODE form.
The van der Pol `oscillator` model was likewise dropped from the
historical 44-model list — the RuleBender workspace no longer carries it
and no copy survives in-repo.

## Gates

`run.py` applies two gates per model. BNG2.pl first generates the `.net`
network, then:

| Gate | Check |
|------|-------|
| correctness | BNGsim and `run_network` each integrate the ODE system; the trajectories are cross-validated (relative tolerance ~1e-5). Both engines integrate the *same* network, so the solution is deterministic and a direct trajectory comparison is the right test. |
| timing | Warmup + timed-run wall-clock comparison, median reported. |

Per the suite design rule, **timing is reported only for a model that
passed cross-validation**.

```sh
python run.py                      # both gates, full 43-model sweep
python run.py --mode correctness   # cross-validation only
python run.py --mode timing        # timing only
python run.py --quick              # BNGsim load+run smoke, no run_network
python run.py --effort low         # cheap subset (cumulative tiers)
```

`BNG2.pl` / `run_network` are located via `BNGPATH` / `BNG2_PL` /
`RUN_NETWORK` (see the top-level `benchmarks/README.md`). Results are
written to the git-ignored `results/` (`ode_results.json` +
`ode_results.md`).
