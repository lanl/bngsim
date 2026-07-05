# `forward_sens` suite

CVODES forward-sensitivity benchmark — BNGsim (serial + sharded) versus
AMICI, both pinned per CVODES corrector method for an apples-to-apples
comparison (paper Table S9).

Promoted from `harness/comparison/bench_forward_sensitivity.py`. The
intricate BNGsim↔AMICI alignment machinery (SBML export, initial-
condition / parameter seeding from the `.net`, name-aligned free-
parameter list, `_verify_alignment`) is carried over unchanged; only the
model paths and the suite flags are new.

## Models

Four medium/large models from `_dev/suite_ode.json`, vendored in-repo:
`egfr_path`, `tcr_signaling`, `Scaff_22_ground`, `SHP2_base_model`. Each
needs both its `.net` (`models/net/ode/`) and its companion `.bngl`
(`models/bngl/<source>/`, located by name search since Phase 3 split the
`.net` and `.bngl` trees) — the `.bngl` is exported to SBML for the
AMICI side.

## Gates

| Gate | Check |
|------|-------|
| correctness | BNGsim and AMICI are aligned to the same problem (the `.net` is the source of truth), then their species trajectories and forward sensitivities are cross-validated. The per-model `traj` / `sens` tags are the correctness result. |
| timing | Median wall time of the coupled state+sensitivity CVODES solve — BNGsim serial, BNGsim sharded (a worker-count sweep), and AMICI. |

```sh
python run.py                       # both gates, full 4-model sweep
python run.py --mode correctness    # alignment + cross-validation only (skips the sharded sweep)
python run.py --mode timing         # timing report
python run.py --effort low          # cheap subset (egfr_path only; cumulative tiers)
python run.py --quick               # small models only (<=37 sp)
python run.py --model egfr           # single model
python run.py --no-sharded           # serial + AMICI only (faster)
python run.py --methods staggered    # one CVODES corrector method only
```

## Observable output-sensitivity validation (`output_sens.py`)

`run.py` validates the **species** forward-sensitivity tensor
(`dx_species/dp`). `output_sens.py` is its companion for the **output**-
sensitivity layer (GH #196–#207): `Result.output_sensitivities()` for
**observables** (`d obs_j/dp = Σ_i c_ji·dx_i/dp`, GH #197) and **expressions**
(BNGL functions — the GH #198 chain rule
`d f/dp = ∂f/∂p + Σ_j (∂f/∂obs_j)·d obs_j/dp`). It reuses `run.py`'s alignment
machinery (`.net`→SBML export, IC/parameter seeding, relerr kernel); the only
new piece is the AMICI side — BNG observables and functions land in the SBML as
parameters with `assignmentRule`s, so `amici.assignment_rules_to_observables`
turns the chosen ones into AMICI observables and `rdata.sy` is the AD reference
for `d output/dp`.

One kind per model: a function-bearing model validates `expression:` selectors
(registering the functions; their observable references are exercised
transitively, since converting the observables themselves would dangle the
function rules and break AMICI codegen); an observable-only model validates
`observable:` selectors. `expr_demo` (a small purpose-built model with the
functions `satB`, `ratio`) covers the expression path; the four signaling
models cover observables.

Two oracles per model: AMICI `sy` (primary, gated on max/p95) and a
finite-difference guard on BNGsim's own output trajectories (gated on the
median — independent of AMICI's observable plumbing).

```sh
python output_sens.py                 # all models
python output_sens.py --model expr    # one model (expression path)
python output_sens.py --no-fd         # skip the finite-difference guard
python output_sens.py --effort low    # cheap subset
```

Results go to the git-ignored `results/output_sens_results.json`.

## Timing/correctness gate knobs

The sharded sweep is a pure-timing reference, so `--mode correctness`
implies `--no-sharded`. Codegen is governed by the `S10_BNG_CODEGEN_MODE`
/ `S10_BNG_CODEGEN_MIN_PARAMS` env knobs (carried over from the
predecessor; the paper run uses `always`); the worker-count sweep is set
via `S9_WORKER_COUNTS`.

`BNG2.pl` is located via `BNGPATH` / `BNG2_PL`. Results are written to
the git-ignored `results/` (`forward_sens_results.json`).
