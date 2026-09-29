# `nightly` — the nightly parity regression check (GH #702)

`.github/workflows/nightly-parity.yml` runs the expensive parity sweeps every
night against independent references, and this directory decides what counts as
a regression. Nothing here blocks a PR.

| suite (baseline file) | reference | corpus |
|---|---|---|
| `sbml_semantic` | SBML Test Suite expected results | 1,824 semantic cases, pinned checkout |
| `dsmts` | DSMTS analytic mean / sd | 39 stochastic cases, committed |
| `rr_ode` | libRoadRunner | the 1,006 of 1,323 BioModels SBML a hosted runner can fetch, pinned and cached |
| `bng_ode` | BNG2.pl + run_network | 592 committed BNGL ODE jobs, ExprTk interpreter |
| `bng_ode_codegen` | BNG2.pl + run_network | the same jobs on the compiled C path (`--codegen`) |
| `bng_ssa`, `bng_nf` | run_network SSA, NFsim | 110 SSA + 235 NF committed BNGL jobs |
| `amici_sens` | AMICI forward sensitivities | 40 of the 50 curated BioModels, staggered + simultaneous |

**The BioModels gap.** 317 of the 1,323 BioModels come only from EBI's BioModels REST
API, which answers HTTP 403 to GitHub-hosted runners (the same URLs work from a
workstation), so CI runs the 992 temp-biomodels models plus the committed SBML
overrides. `present_models.py` hands each runner the models that are on disk. 300 of
the 317 carry no recorded license (17 are CC0), so re-hosting them for CI needs a
license review first.

**Giant models.** The dozen SBML files over 1 MB (up to 786 species) run in a
separate `rr-ode` pass, two at a time on the interpreted RHS: their automatic C
compile alone outlasts the per-job cap on a 4-core runner, and four at once exhausted
its memory. `amici-sens` leaves out BIOMD0000000496/497, which time out on both
engines even on a workstation; BIOMD0000000608, whose 66.7 MB generated sensitivity
RHS outlasts bngsim's 600 s compile budget; and the 7 curated models that come only
from EBI. The workflow file records the measurement behind each.

## Why a baseline, not the exit code

Every parity runner exits 1 whenever any model is DIFF / EXCEPTION / TIMEOUT, and a
full corpus always has some; the SBML runner exits 0 whatever it finds. So the
report is the verdict. `verdicts.py extract` turns a report into a per-case verdict
file, and `verdicts.py diff` compares it with a baseline. Known failures live in the
baseline and do not re-alert every night.

A night alerts on:

- **regressed**: a case that was good is anything else now, including missing from
  the run, TIMEOUT, SKIP or REFERENCE_FAILED. A model that silently stops being
  compared is how a regression hides.
- **new crash**: a failing case now crashes (EXCEPTION / load_fail / sim_fail / error).
- **efficiency**: on a case good both times, bngsim's solver work (steps, RHS
  evaluations, Jacobian evaluations, from `Result.solver_stats`) grew past x1.25
  (with an absolute floor), or a counter's corpus total grew past x1.05, or bngsim's
  total wall time grew past x2 on the same runner CPU model (x4 across models) and
  by more than 30 s. The floor is for the stochastic legs, whose bngsim time is only
  the model load and totals well under a second, so a fixed cost of a few
  milliseconds per model reads there as a x50 blowup. The
  counters are deterministic for a given model, build and platform -- two runs on
  different runner hardware matched exactly -- so they catch the algorithmic
  slowdowns (a Jacobian falling back to finite differences, step-size control, an
  event forcing restarts) that runner wall-clock noise would hide. Wall time is not:
  hosted runners mix CPU generations, and the compiled arm measured 1.8x slower on
  an EPYC 7763 than on an EPYC 9V45. Constant-factor slowdowns need a timing A/B on
  a fixed machine (`benchmarks/perf_ab.py`); the wall check here only catches a
  blowup.
- **backend**: on either BNG ODE arm, a good row that did not run that arm's
  backend: ExprTk on the interpreter arm, compiled C on the compiled arm. Without
  it, a model the interpreter arm compiled by itself shows up only as wall time
  (#872).
- **incomplete**: the run compared fewer than 90 % of the baseline's cases.

Improvements, new cases and churn between two failing states are listed in the
run summary but never alert.

Each suite is compared twice: against the committed baseline, and against the
previous nightly's verdicts (the last scheduled or manual run on `main`). The second
catches a fix that regresses before anyone re-baselined it. In that night-to-night
comparison a flip into or out of TIMEOUT is reported, not alerted: two single nights
can land on runner CPUs of different speed, and a timeout verdict depends on the wall
clock. Against the committed baseline PASS -> TIMEOUT still alerts.

`flaky.json` lists cases whose verdict measurably flips on identical code (with the
runs that show it); their changes are reported and never alert, in both comparisons.
Add one only with that evidence.

## When it alerts

The report job opens one issue titled **Nightly parity: regressions detected**, or
comments on it if it is open, with the regressed cases per suite and a link to the
run. A job that could not run at all alerts too, since that suite went unchecked.
Close the issue when the night is clean again.

Each suite's artifact `nightly-<suite>` holds `report.json` (the runner's full
report), `verdicts.json`, and `diff-baseline.{json,md}` / `diff-prev.{json,md}`.

## Re-baselining

Re-baseline after a reviewed change in outcomes: a fix that turns failing models
good (the summary lists them under *Improved*), a deliberate change in solver work,
or a corpus change.

```bash
gh run download <run-id> -n nightly-rr_ode -D /tmp/rr
cp /tmp/rr/verdicts.json parity_checks/nightly/baselines/rr_ode.json
```

Open a PR with the new file. Its per-model diff is the review, one case per line.
A baseline-only PR does not re-run the sweeps. Record baselines from a CI run, not a
local one: timeouts and solver work are only comparable on the same runner class.

## Running it locally

```bash
python parity_checks/rr_parity/rr_run.py --workers 4 --out /tmp/rr.json
python parity_checks/nightly/verdicts.py extract --kind core --suite rr_ode --report /tmp/rr.json --out /tmp/v.json
python parity_checks/nightly/verdicts.py diff --baseline parity_checks/nightly/baselines/rr_ode.json --fresh /tmp/v.json
```

A local run against a CI baseline differs in platform, so expect small work-counter
and timeout differences; the per-case outcomes are what carry over.
