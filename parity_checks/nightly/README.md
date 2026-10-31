# `nightly` — the nightly parity regression check (GH #702)

`.github/workflows/nightly-parity.yml` runs the expensive parity sweeps every
night against independent references, and this directory decides what counts as
a regression. Nothing here blocks a PR.

| suite (baseline file) | reference | corpus |
|---|---|---|
| `sbml_semantic` | SBML Test Suite expected results | 1,824 semantic cases, pinned checkout |
| `dsmts` | DSMTS analytic mean / sd | 39 stochastic cases, committed |
| `rr_ode` | libRoadRunner | all 1,323 BioModels SBML, pinned, sha256-checked and cached |
| `bng_ode` | BNG2.pl + run_network | 592 committed BNGL ODE jobs, ExprTk interpreter |
| `bng_ode_codegen` | BNG2.pl + run_network | the same jobs on the compiled C path (`--codegen`) |
| `bng_ssa`, `bng_nf` | run_network SSA, NFsim | 110 SSA + 235 NF committed BNGL jobs |
| `amici_sens` | AMICI forward sensitivities | 48 of the 50 curated BioModels, staggered + simultaneous |
| `rr_ssa` | libRoadRunner gillespie (`rr_parity/ssa_screen.py`) | the 269 of the 306 SSA-screen BioModels candidates the corpus holds (#806) |
| `fd_sens` | central differences of bngsim's own trajectories (`fd_sens/fd_sens_run.py`), and a tolerance sweep: an analytic column that still moves between rtol 1e-10 and 1e-12 where the trajectory settles is a DIFF (#1257) | 53 committed `.net` models, staggered + simultaneous, 8 of them narrow and wide pulses over the clock; the only sensitivity oracle a `.net` model has (#804) |
| `differential` | bngsim's own paths, each against another (`differential/differential_run.py`): sensitivity vs plain run, FD vs analytic Jacobian, KLU vs dense, compiled vs interpreted, cc vs MIR, `de_DE.UTF-8` vs C; and a refused sensitivity run, a sparsity pattern that misses entries, a backend other than the one asked for | `fd_sens`'s 53 models and the BioModels under 1 MB, one row per (model, check) (#1257) |
| `differential_nf` | NFsim (`nf_reject`) against RuleMonkey (`nf_exact`), in distribution: each observable's ensemble mean, by the SSA suites' ensemble verdict, a DIFF escalated as `bng_nf` escalates one | `bng_nf`'s 234 single-segment models, from its pinned reference archive (#1312) |

**The BioModels corpus.** 992 of the 1,323 BioModels come from sys-bio/temp-biomodels
on GitHub. The other 331 come only from BioModels itself, whose CloudFront refuses
every request from a GitHub-hosted runner (issue #807), so CI reads bngsim's own copy
of them: one release asset, fetched by `rr_parity/fetch_biomodels_mirror.py`, which
checks every file against the biomodels suite's provenance manifest. BioModels
releases its whole dataset under CC0 (https://www.biomodels.org/termsofuse), which
is what allows the copy. `present_models.py` hands each runner the models that are on
disk, so a local run with part of the corpus still works.

**Giant models.** The 23 SBML files over 1 MB (up to 6,047 species) run in a
separate `rr-ode` pass, two at a time on the interpreted RHS: their automatic C
compile alone outlasts the per-job cap on a 4-core runner, and four at once exhausted
its memory. `amici-sens` leaves out BIOMD0000000496/497, which time out on both
engines even on a workstation. The workflow file records the measurement behind each.

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
- **new crash**: a failing case now crashes (EXCEPTION / load_fail / sim_fail /
  harness_error / error).
- **efficiency**: on a case good both times, bngsim's solver work (steps, RHS
  evaluations, Jacobian evaluations, from `Result.solver_stats`) grew past x1.25
  (with an absolute floor), or a counter's corpus total grew past x1.05, or bngsim's
  total wall time grew past x2 on the same runner CPU model (x4 across models) and
  by more than 30 s. bngsim's wall time is the sum of `BNGSIM_PHASES` in
  `verdicts.py`: an ODE row's load, Jacobian, compile and first solve, and a
  stochastic row's load and its base ensemble (`ensemble_sec`, plus on bng_nf each
  replicate's `NfsimSession.initialize`; the replicates an escalation adds are not
  timed). Before #901 the stochastic legs summed the
  load alone, well under a second, and a fixed cost of a few milliseconds per model
  read there as a x50 blowup; the 30 s floor is from then. A verdict file records
  the phases it sums, and two that sum different ones are not compared on wall
  time: against the committed baseline that alerts until the baseline is
  re-extracted, and night to night it is only reported. The
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
catches a fix that regresses before anyone re-baselined it. A row the committed
baseline re-recorded since that night's commit, which tonight matches, is reported
there and not alerted: it was reviewed when it was recorded, as a pull request that
makes a check stricter records the rows it moves. In that night-to-night
comparison a flip into or out of TIMEOUT is reported, not alerted: two single nights
can land on runner CPUs of different speed, and a timeout verdict depends on the wall
clock. Against the committed baseline PASS -> TIMEOUT still alerts.

`flaky.json` lists cases whose verdict or solver work measurably changes on
identical code (with the runs that show it); their changes are reported and never
alert, in both comparisons. A flaky case's work still counts in the totals. Add one
only with that evidence.

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

**`sbml_semantic` is the exception.** Its verdicts carry outcomes only, no timing or
work counters, and a local run reproduces CI's case for case. Since GH #700 the
per-PR gate `ci/sbml_suite_gate.py` holds every pull request to this baseline exactly,
so the pull request that changes an outcome records it, with
`uv run --no-sync python ci/sbml_suite_gate.py --update-baseline`. An sbml_semantic
change in a nightly then means two pull requests that each passed the gate changed
an outcome together, or something reached `main` without the gate (a manual
`python-tests` run with `pytest_args` skips it).

## The reference archive (issue #805)

Five suites need a reference engine: `rr_ode` and `rr_ssa` (libRoadRunner),
`bng_ode`, `bng_ssa` and `bng_nf` (BNG2.pl, run_network, NFsim), and `amici_sens`
(AMICI). Running them every night meant provisioning BNG2.pl and building AMICI
from source each time, spending half the compute on the oracle, and letting the
oracle itself drift under the comparison (a BNG2.pl regression went unnoticed for
five months, #661). So most nights replay a recorded reference instead:

| when | mode | reference side |
|---|---|---|
| Monday to Saturday | `replay` | the published archive; no reference engine runs, and neither BNG2.pl nor AMICI is provisioned |
| Sunday | `live` | every engine runs, as before, so an oracle-side change surfaces within a week |
| on demand | `record` | live, and the run publishes a new archive |

A manual run picks its mode (`gh workflow run nightly-parity.yml -f mode=live`); a
pull request replays once a manifest is committed. `sbml`, `dsmts` and `fd_sens`
need no engine and run the same in every mode.

**What a replay compares against.** The archive is one release asset per suite,
`reference-<suite>.tar.gz` on a `reference-data-vN` release (not a bngsim
release: `release.yml` publishes `v*` tags only). `reference_archive.json` here
pins each asset's URL and sha256. Each job's `reference_archive.py prepare` step
downloads its asset, checks the sha256, and unpacks one `.npz` per job for the
runner's `--reference-archive` flag; a missing or mismatched asset fails the job,
and the night alerts. Inside, an entry holds what the engine returned (or the
error it raised) and the job identity it is valid for: the model file's sha256,
the grid, the tolerances and so on, per runner. An entry that no longer matches its
job (a model, grid or tolerance changed) is refused, never compared: the row is
REFERENCE_FAILED with `reference_refusal: archive_stale`, which alerts wherever the
case was good. A replayed row reports the engine version the entry recorded.

**Refreshing it.** Re-record after anything that changes the reference side: an
engine pin (BioNetGen, AMICI, libRoadRunner), the corpus, a forced tolerance, or a
harness change to how a runner calls its engine. A week of `archive_stale` alerts
after such a change is the reminder.

```bash
gh workflow run nightly-parity.yml -f mode=record
```

That run is a live night that also records every suite. Each suite packs its
record (`reference_archive.py pack`) before comparing, and refuses if any PASS row
has no recorded reference, since that row would then alert every night. A TIMEOUT
row with no entry is listed in the publish summary: a replay that needs the entry
refuses it as `archive_stale`, which alerts the night after any live night that
passes it (#1290). To keep that list short, `rr_run.py` records RoadRunner's side
first, before bngsim starts, and kills a job only at twice the cap, up to 600 s
(the giant pass's cap, which it does not extend); the row is still TIMEOUT past
the cap itself, as on a live night. `ssa_screen.py`, which saves its entry after
every RoadRunner call, likewise takes a model's row at its cap as a live run does
and kills the worker only at twice the cap. The
publish job then creates the next `reference-data-vN` release with all six assets,
or none if any suite failed to pack (its tag is on the default branch's head; the
manifest names the recorded commit), and writes the manifest into the run summary
and the `reference-manifest` artifact. Nothing replays from the new release until
that file is committed:

```bash
gh run download <run-id> -n reference-manifest -D /tmp/ref
cp /tmp/ref/reference_archive.json parity_checks/nightly/reference_archive.json
```

The pull request that commits it runs the nightly in replay mode against the new
assets, which is the check that they reproduce the live verdicts.

**Licensing.** An asset holds outputs computed from the models, so it goes only
where the models' licenses allow a copy. `pack` looks every entry's model up in
the corpus provenance manifests (`benchmarks/suites/biomodels/biomodels_provenance.json`,
`bng_parity/manifest.json`) and packs it only under CC0-1.0, CC-BY-4.0 or MIT; any
other license, `unknown` included, leaves the entry out, lists it, and leaves that
row `archive_stale` until someone reviews it. Today every BioModels record is CC0
and every BNG model MIT or CC-BY-4.0, so nothing is left out. An asset with any
non-CC0 entry carries bngsim's `NOTICE` (the CC-BY attribution and the MIT text)
and RuleMonkey's license, and its `entries.json` lists each entry's model, source
and license.

## Running it locally

```bash
python parity_checks/rr_parity/rr_run.py --workers 4 --out /tmp/rr.json
python parity_checks/nightly/verdicts.py extract --kind core --suite rr_ode --report /tmp/rr.json --out /tmp/v.json
python parity_checks/nightly/verdicts.py diff --baseline parity_checks/nightly/baselines/rr_ode.json --fresh /tmp/v.json
```

A local run against a CI baseline differs in platform, so expect small work-counter
and timeout differences; the per-case outcomes are what carry over.

## Which legs a pull request runs

A pull request that touches the workflow, `parity_checks/nightly/` or a leg's own
code runs the nightly, but only the legs its changed files feed (#1387). The plan
job lists the PR's files and hands them to `pr_legs.py`. Its `RULES` map each
directory to the legs that import from it, or read its data: `differential` runs
fd_sens's models, for example. A runner, and a module only it imports, feeds
only its own leg. The corpus job runs only where one of those legs reads the
BioModels corpus. The PR's other files (its changelog fragment, `python/`, its
tests) run no leg here, just as a PR with only those runs none. Neither does
documentation, nor `bng_parity/golden/` (the golden gate's data).

- **Changes to shared code run every leg:** the workflow and the nightly scripts.
- **`parity_checks/_core` and `rr_parity/_rr_common.py`**, which every parity
  runner imports, run every leg but `sbml`.
- **A triggering path no rule names runs every leg** too.
- **Scheduled and dispatched runs** run every leg.

`parity_checks/tests/test_pr_legs.py` measures each runner's imports again, file by
file and including imports inside functions, so a runner that comes to import
another file fails there until `RULES` says that file feeds its leg. The data a
job reads (a jobs list, a pin, a model) shows in no import: the test's cases pin
those by hand, so a leg that comes to read another directory's data needs a rule
and a case.
