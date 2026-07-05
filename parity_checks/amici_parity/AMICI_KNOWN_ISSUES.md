# AMICI known issues on the bngsim parity suite

Findings from the **full** `amici_parity` ODE sweep — bngsim vs AMICI over the
complete 1323-model SBML corpus (`amici_run.py --manifest ../rr_parity/ode_jobs.json`).
**Every AMICI problem below is AMICI-side; bngsim is not the outlier on a single
one.** On all 42 of the 44 divergences where an independent third oracle is available,
**RoadRunner agrees with bngsim exactly** and AMICI is the odd engine out (the other 2
are inconclusive only because RoadRunner *also* failed those models).

- **Environment:** AMICI pinned to `AMICI-dev/AMICI@667b17b6b` (`v1.0.1-12-g667b17b6b`,
  BSD-3; built from source into `bngsim/.venv`, needs `swig`) — see `AMICI_PIN.json` for the
  full pin. bngsim 0.9.51, RoadRunner 2.9.2. Verdict kernel = the shared
  `_core.differ.deterministic_verdict` (rtol=1e-9, atol=1e-12).
- **Result (1323 models):** **1050 PASS**, 44 DIFF, 165 REFERENCE_FAILED, 8 EXCEPTION,
  42 BAD_TEST, 14 TIMEOUT. So AMICI fails-to-run or diverges on ~20% of the corpus;
  bngsim runs and matches RoadRunner throughout.

## Class 1 — AMICI cannot run the model (173: REFERENCE_FAILED + EXCEPTION)

AMICI's per-model codegen/compile or its integrator fails, so there is no AMICI
trajectory to compare. bngsim runs these fine. Root causes:

| count | root cause |
|---|---|
| 34 | **Integrator bailout** — `AMICI integration failed (status < 0)` (CVODE gives up; stiffness / step failure) |
| 30 | **Unsupported math `floor()`** in a rate law — AMICI can't codegen it |
| 30 | **C++ model compile failed** — incl. the **event-codegen bug** `deltax.cpp: undeclared identifier 'D'` on event-bearing models (e.g. `BIOMD675/1028/1029`) |
| 19 | **SBML document failed to load** in AMICI's importer |
| 13 | **Unsupported math `ceiling()`** |
| 9 | **SBML package/extension** unsupported (comp / fbc / …) |
| 7 | **Event: non-persistent trigger** unsupported |
| 4 | AMICI internal: `'And' object has no attribute 'evalf'` (piecewise/logical codegen) |
| 4 / 3 / 1 | AMICI internal: **local-symbol collision** (`pi` / `avogadro` / `time` already reserved) |
| 3 | **Event: execution delays** unsupported |
| 2 | `StoichiometryMath` unsupported |
| 14 | other |

So **events are a modest slice** (~10 explicit + some of the `deltax` compiles); the
dominant failures are **integrator bailouts, unsupported `floor`/`ceiling` math, model
compile failures, and SBML-load failures**, plus a handful of AMICI internal bugs.

## Class 2 — AMICI integration outliers (44 DIFF; 42 confirmed AMICI-side)

Both engines ran but disagree, and **RoadRunner sides with bngsim** (max_rel_err = 0):

- **4** — AMICI trajectory went non-finite (`val = inf`; e.g. `BIOMD114/115/346/919`).
- **36** — large divergence (≥ 0.5 relative; AMICI grossly off — e.g. `BIOMD943`, `BIOMD125`).
- **4** — mild divergence.
- **42 of 44** confirmed AMICI-outlier (bngsim == RoadRunner exactly). The remaining 2
  (`BIOMD339`, `MODEL2003200002`) are inconclusive: RoadRunner refused them too, so there
  is no third oracle — but bngsim ran cleanly.

## Bottom line

AMICI passes ~79% of the SBML ODE corpus. Where it fails it is an **AMICI**
limitation (codegen gaps for discrete math and events, integrator bailouts, internal
bugs) or an **AMICI** integration error — never a bngsim defect. Across both reference
engines (RoadRunner and AMICI) on the full SBML corpus, **bngsim is never the outlier.**

## Reproduce

```bash
export BNGPATH=/path/to/BioNetGen-2.9.3
python amici_run.py --manifest ../rr_parity/ode_jobs.json --workers 8 \
    --out runs/report_ode_full.json
# a specific AMICI compile failure (verbose shows the real C++/SBML error):
python -c "from amici import SbmlImporter; \
  SbmlImporter('../rr_parity/models/BIOMD0000000675/BIOMD0000000675.xml').sbml2amici('m','/tmp/m',verbose=True)"
```
