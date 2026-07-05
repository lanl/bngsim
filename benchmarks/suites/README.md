# `suites/` — paper-data benchmark suites

Each subdirectory is **one benchmark suite** that produces **one paper
artifact** (a table or figure). A suite contains its runner, a job manifest
referencing models in `../models/`, any expected-result / `.exp` data, and
emits its `.tex` / figure into `../../dev/paper/latex/generated/` so the
paper `\input{}`s it (explicit provenance).

Where applicable a suite runner does **both** a correctness check and a
timing comparison — timing is only reported for jobs whose results are
verified correct.

The `run_all.py` orchestrator (repo `benchmarks/`) drives every suite here.

## Suites

`status`: **built** = runner present and validated; **planned** = not
yet migrated from the top-level `benchmarks/` scripts.

| Suite | Paper artifact | Status |
|-------|----------------|--------|
| `biomodels/` | BioModels figure (Fig S1) + SBML-events figure (Fig S2) | built |
| `antimony/` | Cross-engine Antimony figure + table | built |
| `showcase/` | Fig. 1 — the runnable code snippet | planned |
| `fitting/` | Main fitting table + full supplementary table | planned |
| `ode/` | ODE benchmark table | planned |
| `ssa/` | SSA benchmark table | planned |
| `psa/` | PSA benchmark table | planned |
| `nf/` | Network-free (NFsim) table | planned |
| `python_ode/` | Pythonic-ODE-workflow table | planned |
| `python_ssa/` | Pythonic-SSA-workflow table | planned |
| `steady_state/` | KINSOL steady-state table | planned |
| `sbml_test_suite/` | SBML Test Suite 3-engine table | planned |
| `forward_sens/` | Forward-sensitivity + FIM table | planned |
| `sbml_roundtrip/` | SBML round-trip table | planned |

The BioModels events subset is **not** a separate suite — the
`biomodels/` runner splits its output on the manifest `events` tag, so
one runner emits both Fig S1 and Fig S2.
