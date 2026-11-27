# `discontinuity`: rate-law switches against exact references

Phase 0 of [#1781](https://github.com/lanl/bngsim/issues/1781) (this skeleton:
[#1912](https://github.com/lanl/bngsim/issues/1912)). Each case is a model whose
rate laws switch (`if()`, `max`, `min`, `abs`, pulses, clocks, state thresholds).
The suite sweeps each one over its parameters, tolerances, output grids and
sensitivity columns, and judges every run against a reference that does not use
bngsim's switch machinery. #1781 rebuilds the integrator's crossing handling; this
suite is the safety net for that work, and it shows the effect of each change.

## Running it

The PR slice, a few points per case, runs on every pull request as part of
`python/tests`:

```bash
uv run --no-sync python -m pytest -p no:cacheprovider python/tests/test_discontinuity_slice.py -q
```

The full sweep is the nightly's `discontinuity` leg. Run it by hand like this:

```bash
uv run --no-sync python parity_checks/discontinuity/discontinuity_run.py --workers 4 --out runs/discontinuity.json
```

Add `--cases a,b` to run some cases only. The report is a `_core` report, and the
nightly diffs it against `parity_checks/nightly/baselines/discontinuity.json` as it
diffs every other leg (`parity_checks/nightly/README.md`). The suite's own tests
are `parity_checks/tests/test_discontinuity_*.py`.

## A case

A case is a `Case` in `_cases/<family>.py` (`_case.py` documents each field):

* **the model text**, as a `.net` or SBML file. Every swept value is a model
  parameter, so one compile serves the whole case;
* **its switches**, each declared by what its condition reads (`time` or
  `state`) and whether the rate law `jump`s or `bend`s there. #1922's classifier
  will be checked against these declarations;
* **its axes**: rows of parameter values, the tolerances (1e-6 to 1e-12 by
  default), the output grids (`Uniform`, `Aligned` with a sample exactly at each
  crossing, and `Offset` with samples a hair either side of each), and the
  sensitivity column sets;
* **its reference** (below), and **a spec**: the model written out by hand in
  sympy, with an explicit 0/1 branch variable for each switch. The slice checks
  the spec against the model text through bngsim's own right-hand side, at states
  away from every surface. A wrongly written case then fails as a case written
  wrongly, not as an engine bug;
* **its slice**: at most 8 points, which between them take every tolerance, grid
  and column set the case sweeps;
* **its known failures**: a point that fails on main for a filed bug names the
  issue (`Known`). In the slice that point is a strict expected failure, and in
  the nightly baseline it must be a failing row. When the bug is fixed, both turn
  red until the mark is removed.

Every key is `<case>[<row>]|rtol=…,grid=…,sens=…`, the key the nightly's verdict
files use. For each row, grid and column set there is also a
`<case>[<row>]|converge,grid=…,sens=…` row.

## References

No reference module imports bngsim; a test checks this in a clean interpreter.
bngsim must never import one either. In particular, #1781's new engine must not
reuse `_branchwise.saltation`, or the reference would no longer be a second
opinion.

| kind | module | what it is |
|---|---|---|
| closed form | `_closed_forms.py` | formulas written by hand. `clock_exposure` is lifted from `python/tests/test_clock_carried_crossing.py`'s `_exact` and extended to every time and column; a test holds the two together |
| branch by branch | `_branchwise.py` | the spec integrated one smooth branch at a time by mpmath's Taylor method at 30 digits, with sensitivity columns. Each crossing is located on its branch's own solution, and the columns jump by the saltation formula. It refuses rather than guesses at a start on a surface, a grazing crossing, a slide, coincident surfaces, or a crossing count other than the case declares |
| frozen | `_frozen.py` | enclosures recorded offline by validated integrators. Ariadne certifies the crossing structure, CAPD encloses the columns, and both enclose the species, whose bands must overlap. Both are GPL-3.0, so they are never a dependency: their output is committed as data, tied to its case by a fingerprint. The recorders come with the first cases that need them (#1915) |
| none | `_reference.py` | no reference exists yet (some slides). The case checks only invariants and an expected refusal, and says why |

At a sample taken exactly at a crossing time the columns jump, so a reference
gives both limits there. Until the contract fixes which side a run owes (#1916),
the judge accepts either side, and each row records which one bngsim gave.

## The judge

`_judge.py`. For each species and each sensitivity column it takes the largest
distance between the run and the reference over the window, in units of
`atol + rtol·peak` (for a column, `atol/|p|`), and compares that with the case's
`k_state` and `k_sens` (default 1e3, `bngsim._check.PATH_RTOL_FACTOR`). Columns
the run cannot resolve are named but not judged; the rule is `_check.resolved`.
The convergence row fails when the error stops falling as rtol tightens: the
tightest run's ratio must stay within 10, or within the loosest run's ratio times
the square root of the tolerance range.

## What the seed found

`clock_step[a=2.5,b=0.5]` at rtol 1e-12 with sensitivity columns, on the
aligned and offset grids: the row asked for at t = 25, the crossing, came back
at t = 24.999999999999996 (macOS arm64). That is #1799 (a sensitivity run
records an output row up to 1e-12 short of the time asked for), and both points
are marked against it.
