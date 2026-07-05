# `sbml_test_suite` suite

Runs the official **SBML semantic test suite** (1824 cases) against up to
**four** engines — BNGsim, libRoadRunner, AMICI, COPASI — and emits a
feature-by-feature pass/fail/skip compatibility table (paper Table S8).

## Fair by construction (GH #225)

The whole point of this suite is an **apples-to-apples** engine comparison, so
every engine is graded through **one shared path** and nothing about the grading
favours one engine over another:

| Shared (identical for every engine) | Where |
|-------------------------------------|-------|
| solver tolerances (`atol=1e-12, rtol=1e-8`) | `_grading.SOLVER_ATOL/RTOL` |
| variable classification (species / compartment / parameter, from libSBML) | `_grading.read_sbml_entities` |
| resolution ladder (species→compartment/parameter) | `_grading.resolve` |
| amount conversion (`amount = conc(t)·vol(t)`, via the species' own compartment trajectory) | `_grading.grade` |
| comparison + shape + `var_missing` semantics | `_grading.grade` |
| denominator (in-scope = `TimeCourse` with settings/SBML/CSV present) | `run.run_case` |

The only per-engine code (`_engines.py`) is the irreducible "load the model,
integrate, read value X out of this engine's native output" primitive — an
object exposing `species_conc(id)` / `entity_value(id)`. Because the resolution,
conversion, comparison and denominator all live in the shared kernel, it is
structurally impossible for one engine to be graded more generously than
another. A "bngsim wins" result is therefore an engine result, not an accounting
artifact.

How each engine exposes a value (all are concentrations; amounts are derived
uniformly in the kernel as `conc(t)·vol(t)`):

| Engine | species concentration | compartment / parameter |
|--------|-----------------------|-------------------------|
| BNGsim | species/observable column (undoing `_ant_`/`_obs_` renaming) | observable / parameter |
| libRoadRunner | `[id]` selection | bare `id` selection |
| AMICI | `MeasurementChannel` observable `formula=id` (`rdata.y`) | same |
| COPASI | time-series `getConcentrationData` by SBML id (constant/FIXED entities from the model object) | same |

Differences in the resulting table are **engine** differences, not harness ones —
e.g. `stoichiometryMath` cases (00068–00070) where this COPASI build imports the
stoichiometry as 1, or `AlgebraicRule` cases (00039–00040) bngsim/RR reject. The
deliverable is a trustworthy number, not a favourable one.

## Corpus — external, not vendored

Unlike every other suite, `sbml_test_suite` **vendors no models**. The SBML Test
Suite is a large external repository; the runner reads it from
`SBML_TEST_SUITE_DIR` and never modifies or copies it.

```sh
export SBML_TEST_SUITE_DIR=~/Code/sbml-test-suite/cases/semantic
```

Get a checkout from <https://github.com/sbmlteam/sbml-test-suite>. If the
variable is unset the runner falls back to that default path.

## Engines / dependencies

`bngsim` is the test subject; the references are `roadrunner`, `amici`, and
`COPASI` (the `python-copasi` bindings — **no CopasiSE CLI is needed**, the
in-process bindings are used the same way as RoadRunner/AMICI). An engine whose
library is absent has every case reported `skipped` and renders as `TBD` in the
table — never a misleading `0%`. AMICI compiles per-model C++ (~25–30 s cold);
compiled extensions are cached under the git-ignored `results/amici_cache/`, so
re-runs are load-only.

## Gates

| Gate | Check |
|------|-------|
| correctness | Each case is simulated and its trajectory compared to the suite's reference CSV within the per-case tolerances. The pass/fail/skip counts and the per-feature-tag breakdown are the compatibility result. |
| timing | Per-case wall time, totalled per engine over the cases that engine got right (a wall time is meaningless for a case it failed). |

```sh
python run.py                       # correctness + timing, all cases, BNGsim
python run.py --engines all         # BNGsim + libRoadRunner + AMICI + COPASI
python run.py --engines bngsim,rr   # subset
python run.py --engines bngsim,copasi
python run.py --mode correctness    # compatibility table only
python run.py --mode timing         # per-engine wall-time only
python run.py --effort low          # cheap representative subset
python run.py --quick 50            # first 50 cases (after the other filters)
python run.py --case 00001          # single case
python run.py --tag-prefix Event    # SBML L3 event subset
```

The `--effort` tiers are deterministic strided samples so a cheap run still
spans the whole feature range: `low` = case number divisible by 20 (~5%),
`medium` = divisible by 4 (~25%, a superset of `low`), `high` = all 1824 cases.

Results are written to the git-ignored `results/`
(`sbml_test_suite_results.json` + `.md`).
