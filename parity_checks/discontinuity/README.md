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

* **the model text**, as a `.net`, SBML or Antimony file. Every swept value is
  a model parameter, so one compile serves the whole case;
* **its switches**, each declared by what its condition reads (`time` or
  `state`) and whether the rate law `jump`s or `bend`s there. #1922's classifier
  will be checked against these declarations;
* **its axes**: rows of parameter values, the tolerances (1e-6 to 1e-12 by
  default), the output grids, the sensitivity column sets, and the engines. The
  grids are:
  * `Uniform`, `n_points`. Only its row count and ends are checked;
  * `Sampled`, the same times as `sample_times`, each checked to the bit;
  * `Aligned`, with a sample exactly at each crossing;
  * `Offset`, with samples a hair either side of each crossing;
  * `SteadyState(method)`, which runs `steady_state()` instead of a time course.

  The engines are `ode` (the default), `ode-compiled` and `ode-interpreted`
  (codegen forced on or off), and `diffrax` (`run_diffrax`: plain runs on
  `Uniform` grids). A key names its engine only when it is not `ode`;
* **its reference** (below), and **a spec**: the model written out by hand in
  sympy, with an explicit 0/1 branch variable for each switch. The slice checks
  the spec against the model text through bngsim's own right-hand side, at states
  away from every surface (on the rows `spec_rows` names, all of them by
  default). A wrongly written case then fails as a case written wrongly, not as
  an engine bug. A model with events has no spec, since the spec holds no
  resets; its passing points stand in;
* **its slice**: at most 8 points, which between them take every tolerance, grid
  and column set the case sweeps;
* **points left out** (`exclude`), each with its reason: only a point whose
  outcome rides on the step sequence. That is a point that passes on some runs
  and fails on others as atol moves in its last digits (by 1 part in 1e12 to 2.5
  in 1e11), or one that passed every such run but came within 2 times of its
  allowance on one, from a median well below it. No mark can hold such a point,
  and it would flap from night to night. #1914's family cases take theirs from
  `_cases/_flips.py`, which says how they were measured;
* **its known failures**: a point that fails on main for a filed bug names the
  issue (`Known`), and the narrowest type it fails with, as a class or as the
  name of one of bngsim's (`"SimulationError"`, `"SimulationTimeout"` for a
  stall that the case's `timeout` cuts off). In the slice that point is a strict
  expected failure, and in the nightly baseline it must be a failing row. When
  the bug is fixed, both turn red until the mark is removed.

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
| closed form | `_closed_forms.py` | formulas written by hand. `clock_exposure` is lifted from `python/tests/test_clock_carried_crossing.py`'s `_exact` and extended to every time and column; a test holds the two together. `pulse_exposure` generalizes it to a condition on `time()`, on a clock started at `T0`, or on #1776's residual, to more shapes (a window, a Gaussian) and to the width's column, and a test holds it to `clock_exposure` on every row they share. An event is written in as a jump of the species it resets, read through `past()`, so both limits at its time are there. An indicator on time alone is read exactly (`past(..., ulps=0)`), so instants a double apart are two instants; one on the state reads `past()`'s blur of a few ulps, since a run locates such a crossing only to its own accuracy. Each column is checked against a central difference of its own closed form's species, and each family case's closed form against its spec integrated branch by branch |
| limit | `_reference.py` | a steady state written by hand (`Limit`), for `SteadyState` grids. The run must also report `converged=True` |
| branch by branch | `_branchwise.py` | the spec integrated one smooth branch at a time by mpmath's Taylor method at 30 digits, with sensitivity columns. Each crossing is located on its branch's own solution, and the columns jump by the saltation formula. An event (`Reset`) sets the state at its trigger's root; every branch is then read afresh, and the columns jump by the reset's own formula (`reset_saltation`, #1915). It refuses rather than guesses at a start on a surface or on a true trigger, a grazing crossing, a slide, coincident surfaces, a reset that lands on a surface, fires another event at once or leaves its own state trigger true, or a crossing count other than the case declares |
| frozen | `_frozen.py` | enclosures recorded offline by validated integrators. Ariadne certifies the crossing structure, CAPD encloses the columns, and both enclose the species, whose bands must overlap. Both are GPL-3.0, so they are never a dependency: their output is committed as data, tied to its case by a fingerprint. No case needs them yet: every one has a closed form or a branch-by-branch reference. The recorders come with the first case that does (#2006) |
| none | `_reference.py` | no reference exists yet (some slides). The case checks only invariants and an expected refusal, and says why |

**A refusal owed.** Where a column has no derivative (the answer jumps as a
parameter crosses a value, or has a kink there), no number is right. The
reference then says the point owes a refusal (`Owed`, naming bngsim's class by
its name). Such a point passes when the run raises it, and is DIFF when the run
returns.

**A refusal permitted.** Where a column has a value but the run may decline it,
the reference says so (`Permitted`, #1915). Today that is an instant an ulp from
a tie at which the column has none: whether two instants a double apart are kept
apart is the contract's to say (#1916). Such a point passes when the run raises
that refusal, and its row's subclass then reads `refused:<class>`, so the
nightly sees a run that stops refusing, or starts. An answer is judged like any
other, and a wrong one fails. Neither kind of point is in a convergence group.

At a sample taken exactly at a crossing time the columns jump, so a reference
gives both limits there. Until the contract fixes which side a run owes (#1916),
the judge accepts either side, and each row records which one bngsim gave.

## The judge

`_judge.py`. For each species and each sensitivity column it takes the largest
distance between the run and the reference over the window, in units of
`atol + rtol·peak` (for a column, `atol/|p|`), and compares that with the case's
`k_state` and `k_sens` (default 1e3, `bngsim._check.PATH_RTOL_FACTOR`). The
peak is the largest over the samples. A case may ask for the largest over the
whole window instead (`Judge(peak="window")`): both limits at every crossing,
and a dense scan. A closed form may also name stretches where a column peaks away from
every crossing (`ClosedForm.peaks`, a smooth pulse's middle), which the scan
covers densely: a Gaussian 1e-3 wide lies between two of its 2001 uniform times. A narrow pulse's column runs 1e4 to 1e5 times its end value
inside the pulse, and CVODES holds it to rtol of that running size there, so a
grid with no sample inside the pulse would judge it against its end value. A column
that neither the run nor the reference resolves is named but not judged; the
rule is `_check.resolved`. The convergence row fails when, for any two
tolerances, the tighter run's ratio is past 30 and past the looser one's ratio
times the square root of their ratio: an error that stops falling, as #1227's
did. In the seed cases every healthy row sits at under half of that allowance.
A group holding a marked point has no convergence row, since its marks already
say it is broken, and the fix that clears them moves the row too. A case whose
judge is not the default (another `k`, the window peak, the convergence rows
off) says why in `Judge.why`.

## What the seed found

Three bugs, each marked against its issue. They show on macOS arm64 and on the
nightly's Linux x86_64 alike, at the same ratios:

* **#1984**: `clock_step[a=1.0,b=-1.0]` at rtol 1e-10 on `Aligned(2)`, with rows
  at t = 0, 25 (the crossing) and 30 only. From the crossing on, `dX/dk_clock`
  and `dX/dc` come back exactly 0: the step's jump is lost, and nothing warns.
* **#1985**: every `clock_step` column point at rtol 1e-12 is 93 to 286 times
  its tolerance, where rtol 1e-6 to 1e-10 hold every column within 31 times. An
  output row at or before the crossing does it. The case states `k_sens = 50`
  to catch it, and turns its convergence rows off: from rtol 1e-10 to 1e-12 the
  error grows 4 to 46 times across the groups, which no convergence rule could
  split without flapping.
* **#1799**: `clock_step[a=2.5,b=0.5]` at rtol 1e-12. The row asked for at
  t = 25, the crossing, comes back at t = 24.999999999999996. A sensitivity run
  records a row up to 1e-12 short of the time asked for.

## #1913's cases: the open bugs in #1781's area

Each bug that reproduced on main (f73ed887, 2026-11-27) is a case of its own,
whose failing points are marked against it. The cases keep points that pass,
off the bug's row or grid, unless the note says why none can.

| case | bug | what fails |
|---|---|---|
| `event_touch_tie` | #1618 | a touch event on the instant of an event whose time moves: dX/dtau has no value there, and the run returns one |
| `event_unlisted_value` | #1626 | a window on a species an event sets to a value not listed before the run is stepped over on 2 rows |
| `event_double_apart` | #1627 | two event stops a double apart are merged: an equality never fires, an instant runs before a switch |
| `table_index_function` | #1657 | a step tfun indexed by `h() = time()` gets no stops; A(50) = 0 for 10 on every grid |
| `clock_triangle_weak`, `clock_step_weak`, `clock_sawtooth_weak`, `clock_sawtooth_literal` | #1779 | a crossing the clock carries far faster than its species: stalls, time-outs and wrong columns |
| `steady_counter_window` | #1794 | `steady_state()` steps over a window on a unit-rate counter clock |
| `diffrax_gaussian` | #1795 | `run_diffrax` steps over a narrow Gaussian pulse: X(100) = 0 for 100 |
| `steady_slow_threshold` | #1796 | `steady_state()` stalls one ulp short of a slowly approached threshold |
| `table_step_switch_time`, `table_kink_switch_time` | #1797 | a switch time on a step tfun's breakpoint takes the table's jump into its column, or returns neither side of a kink |
| `parked_zero_window` | #1798 | `if(Bobs<=0&&Bobs>=0, kb, 0)` on a state parked at 0 loses the exit time's columns |
| `event_row_short` | #1799 | on #765's model, a sensitivity run records the row at 3e-11 from 2.95e-11 |
| `cube_switch` | #1802, #1989 | compiled, `Aobs^3 > c` is never taken; both engines take it 7e-13 of A late |
| `event_trigger_step` | #1804 | a trigger that steps at its own fire instant: dB/dT0 = -(T0+1)/T0 for -1 |
| `periodic_window` | #1805 | refused for every parameter, though no parameter moves the window |
| `counter_onset_sbml` | #1807 | the SBML loader's extra root breaks a counter-clock pulse's onset column |
| `late_switch` | #1836 | a switch late in a long run, where the doubles are coarse |
| `slow_threshold_run` | #1991, #1989 | a slow approach refused as at rest when a row falls in its band; the switch taken late |

Two bugs found while writing these were filed: #1989 (a slowly crossed state
switch is taken about 7e-13 of its species past the threshold, at any rtol)
and #1991 (that slow approach refused as come to rest). Two of the 18 have no
case:
* #1496 no longer reproduces (e137346d, #1495, casts the JAX translator's
  comparisons to doubles);
* #1777's residue cannot be told apart from how a narrow pulse's column is
  conditioned: a corner-free Gaussian of the same width comes out 5 to 10 times
  further off.

## #1914's families: pulses, steps, sawtooths and bends

Each family is one shape of rate law on each spelling of its condition, built by
`_cases/_families.py`: on `time()` itself, on a counter clock `Tobs` (from `T0`
at `k_clock`), and, for the sawtooth, on #1776's residual `a*time() - b*Tobs`.
Every model keeps the clock T, read or not: its exact value, `T0 + k_clock*t`,
says which time the state a run reports belongs to. Each pulse's crossing sits at
t = 25, a point of `Uniform(31)`, and a row at c for t = 25.37 puts it between
every uniform point. Each case sweeps rtol 1e-6 to 1e-12, the grids `u4`, `u31`,
`aligned2`, `aligned31` and `offset31`, and a plain run beside one with the
columns of its law's own parameters (for #1776's residual, kx, k_clock, c and w:
a and b, which its rows move, have none). Its rows move the width (0.003, 0.1 and
4), the clock rate (0.5, 2 and 8), where the counter stands across the pulse (50,
or 0.05), and whether the crossing sits on a grid point. The Gaussians, the
triangles' controls, take three of the triangles' rows each; having no crossing,
their `aligned31` and `offset31` grids ask for the same rows.

| family | cases | law |
|---|---|---|
| pulse | `pulse_time`, `pulse_clock` | a rectangle of area kx, `if(u>c&&u<c+w, kx/w, 0)` |
| pulse | `gauss_time`, `gauss_clock` | a Gaussian of the triangle's area and s = w/4: no corner, no switch, the triangles' control |
| step | `step_time`, `step_clock` | `if(u>c, k1, k0)`, up and down |
| step | `step_state` | `X -> Y` at `if(Xobs>th, k1, k2)`: a species that switches its own rate |
| sawtooth | `sawtooth_time`, `sawtooth_clock`, `sawtooth_coupled` | a jump up at c and a line back to 0 by c + w; `sawtooth_coupled` is #1776's own model and its healthy couplings |
| bend | `triangle_time`, `triangle_clock` | `kx*(1/w)*max(0,1-abs(u-c)/(w/2))`: corners of max and abs (#1285, #1456, #1777) |
| bend | `min_state`, `abs_state` | `min(vmax, k*Xobs)` and `kx*abs(Xobs-th)` on a decaying X |

**The judges.** The narrow pulses take the window peak, and k = 5e3 with their
convergence rows off. At rtol 1e-12 their passing points reach 2.15e3 times their
tolerance over 263 runs (below), and 2.33e3 on another set of draws, and the
corner-free Gaussian on the clock, which has no switch, 2.13e3; k is 2.1 times
the worst, and the marks judged by it lie 1.5 times above it at their lowest
(once in 263 runs) and 2.2 times otherwise.
The two clock-carried sawtooths take k = 1e4: #1998 puts their rows on the jump
at a fixed 3.0e3 to 7.2e3, and the highest passing point, past the jump at
k_clock = 8, varies from 6.2e3 to 7.8e3. `step_state` takes the window peak, and
an `Offset` hair of 1e-4 of t*: a state switch's crossing is located only to the
run's own accuracy. `min_state` and `abs_state` turn their convergence rows off.
Each says why in its `Judge.why`.

**What the sweep found.** Every point that fails on every run is marked. Every
point of these cases was run 263 times, at the committed atol and at
`atol_ratio*(1 + d)` for 262 values of d up to 2.5e-11 in size, each of which
moves the step sequence by an ulp or so (`_cases/_flips.py` lists them). The 64
points whose outcome changed are left out with their counts, and so are four that
passed every run but came within 2 times of the allowance on one: most are
#1799's short row, which shows on some step sequences only. That list is what
those runs found, not a bound. A point outside it can still change outcome on
another step sequence, at about 1 run in 250 or rarer for these kinds, so after a
change that moves the step sequence a single new row of these kinds in the
nightly's diff is that and not a regression.

* **#1995**: a sensitivity run makes no progress at a time switch's stop once
  the rate jumps by 30 there at rtol 1e-12 (`step_time`), and sooner on narrow
  pulses: 0.003 wide from rtol 1e-10 (`pulse_time`, `sawtooth_time`).
* **#1996**: a plain run of a narrow jump pulse on a counter clock fails at rtol
  1e-11 and 1e-12 just past its leading edge (`pulse_clock`, `sawtooth_clock`,
  `sawtooth_coupled`), where the same pulse on `time()` is right.
* **#1997**: the w column of a narrow rectangle on a counter clock ends 1.1e-3
  off where it is 0, the same from rtol 1e-8 to 1e-12 (`pulse_clock`).
* **#1998**: a sensitivity run moves the whole state along the flow at each
  crossing a counter clock carries, and keeps the time. A row on the crossing
  holds the moved state (`triangle_clock`'s rows on its crossings at 7.7e3 to
  2.5e5 times their tolerance, 1.2e4 to 6.2e4 with the counter at 0.05: #1777's
  lead; `step_clock`'s row on a step to 30), and past a jump the species stay off
  (`sawtooth_clock`).
* **#1799**: a sensitivity run records a row on a stop up to 1e-12 from the time
  asked, short or, at the triangle's foot on `time()`, an ulp late
  (`triangle_time`, both Gaussians, `step_state`, `sawtooth_coupled`).
* **#1999 (new)**: on some step sequences `abs_state`'s kd column at its corner
  comes back about 1e-7 of itself off at rtol 1e-10, up to 2.9e3 times its
  tolerance. Those points are left out, and its convergence rows are off.
* **#2000 (new)**: on one run of 263, `pulse_clock`'s w column loses its jump at
  the trailing edge, -1010.9 for 0 from there to t = 30; it may be #1984's kind.
  That point is left out.

`min_state` fails nowhere.

## #1915's coincident surfaces and events

Each case sweeps rtol 1e-6 to 1e-12, the grids `u4`, `u31`, `aligned2`,
`aligned31` and `offset31` (a hair of 1e-4 of t* for the cases whose crossings
are on a species: `coincident_state`, `event_state_reset`,
`event_cross_switch`), and a plain run beside one with the columns of its law's
parameters (for the ties, the rate columns alone as well). Every model keeps a witness clock.
The Antimony cases' rows reach the model through `set_param`, which keeps a
double's last bit: Antimony's text keeps 15 digits, so a literal
25.000000000000004 reads back as 25.0.

| case | law | rows |
|---|---|---|
| `coincident_sum` | `k0 + d1*[t>c1] + d2*[t>c2]`: each switch adds its own term, so every column has its value at the tie | c2 = c1, a double after and before, 1e-9 and 0.5 apart, the tie between grid points, #1627's 0.3 and 0.1*3 |
| `coincident_and` | `k0 + d1*[t>c1 && t>c2]`: at the tie the c columns owe a refusal; an ulp from it a refusal is permitted | as `coincident_sum` |
| `coincident_state` | `k0 + d1*[X>th1] + d2*[X>th2]` on a decaying X: at the tie the th columns owe a refusal | th2 = th1, a double either way, 1e-4 and well apart, the tie early |
| `switch_event` | `piecewise(k1, time > c, k0)`, and an event at te that sets X = a*X + d: at te = c the c and te columns owe a refusal | te = c, a double after and before, 0.5 either way, a = 0 (set) and 0.5, #1627's doubles either way |
| `event_state_reset` | an event that sets X = a*X + d each time X falls through th | 1, 3, 5, 6 and 16 firings |
| `event_cross_switch` | `piecewise(k1, X > th, k2)`, and an event at tau that sets X = xr across the switch's surface, with no root | set above, below, before the natural crossing, late |
| `event_clock_reset` | a counter T, a window on it, and an event that sets T back to 0 | the pulse moved, cut short, before the reset, both, wide, on a faster clock |

Every closed form is checked against central differences of its own species,
and against its spec integrated branch by branch, resets included, on 16 rows
whose instants differ (a double apart is two instants at 30 digits).

**The judges.** The state cases take the window peak, as `step_state` does. The
convergence rows are off in all but `coincident_sum` and `coincident_and`. At
rtol 1e-12 a healthy run's own global error is tens of times rtol in these
models (`event_state_reset`'s X(5) is 1.3e-10 off before any event fires), so
their rows sat on the rule's threshold and changed outcome from run to run.
`event_clock_reset` takes the window peak too: its pulse is 0.05 wide. k is the
default everywhere: outside the marks and the flips, no point exceeds 404 times
its tolerance on any run, and no mark falls under 2.7e3.
Each case says why in its `Judge.why`.

**What the sweep found.** Every point, exclusions included, was run 480 times:
at the committed atol and at `atol_ratio*(1 + d)` for 239 values of d on a grid
and 240 at random, |d| up to 2.5e-11. The 161 marks fail on all 480, and every
other point passes on all 480, apart from the 69 left out in `_cases/_flips.py`
with their counts. 45 of those are #1799's short row, 9 #2005's misplaced
crossing and 8 #1997's residue; 7 passed every run but came within 2 times of
the allowance on one, 2 of them only on a review's draws. As for #1914, that
list is what the runs found, not a bound: the random set found 13 points the
grid had not, and a review's 100 more draws 5 more. Across a whole sweep of
these cases, about 1 run in 25 shows one such point.

* **#2003 (new, silent wrong)**: two rate-law switches on `time()` a double
  apart. In a sensitivity run, the row on one instant carries the other's
  column jump (`coincident_sum`: dX/dc2 = 0.2463 for 0 at t = 25). That reads
  the two instants as two, as #1627 does; if the contract (#1916) makes
  instants an ulp apart one, the row is right and the marks come out.
* **#2004 (new)**: an event a double after a row or a stop is refused with
  #950's refusal, though the columns it refuses have values (`switch_event`).
* **#2005 (new, silent wrong)**: on some step sequences a run at rtol 1e-12
  locates a crossing on the state about 1e-8 off in time, and the species near
  it come back off: a counter's pulse loses that much of its exposure, more
  often after an event resets the counter (`event_clock_reset`), and two
  thresholds' species come back 6e-9 off (`coincident_state`). Those points are
  left out as flips, with their counts.
* **#1646**: with a row on each of two instants a double apart, the sensitivity
  run fails with `CV_TOO_CLOSE` (`coincident_sum`, `switch_event`).
* **#1799**: rows an ulp off the time asked, on a stop or on a crossing the
  clock carries (`coincident_and`, `coincident_sum`, `event_clock_reset`).
* **#1997**: past the clock pulse, dX/dw keeps a residue of 1.3e-7 to 2.2e-7
  where it is 0 at rtol 1e-12, in the Antimony spelling as in the `.net` one
  (`event_clock_reset`).

The ties behave as the contract would want: every tie that owes a refusal gets
one, by name. The additive pair, the rate columns at every tie, and an event
that moves the state across a surface come back right on every run.
`event_state_reset` and `event_cross_switch` fail nowhere, and
`coincident_state` only on its flips.
