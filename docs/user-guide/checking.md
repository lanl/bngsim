---
summary: "Check a result before relying on it: what Result.check() runs, how to read pass, diff, skip, unsupported, no_base, timeout, exception and info, species the run does not resolve, when a result has been checked against another engine, and what a check of a stochastic run does not judge."
---

# Check a result before you rely on it

`result.check()` makes the run again with one setting changed and says whether
the answer moved. Each rerun starts from the state the run started from, on a
fresh copy of the model, so a check leaves your model alone. A disagreement is
reported, not raised.

This page shows a check that passes, one that fails, checks that cannot run,
and what each part of the report means. The examples use a small model that
ships with BNGsim: A and B bind to make C.

```python
import bngsim

model = bngsim.examples.load("binding")  # A + B <-> C, at rates kf and kr
sim = bngsim.Simulator(model)
result = sim.run(t_span=(0, 10), n_points=11)

report = result.check()
print(report["ok"])  # True
print([(c["name"], c["status"]) for c in report["checks"]])  # [('tolerance', 'pass'), ('grid', 'pass')]
print(report)  # the same report as plain text
```

An ODE run gets two checks by default:

- **`tolerance`** runs the model again at a tenth of the run's `rtol` and
  `atol`. It fails where any species moves by more than 0.1% of its peak. A
  run that moves is not converged at its own tolerance.
- **`grid`** runs it again at 8 times the output points. The output values
  are read off the solver's own steps by interpolation, so a run should not
  change when the output points move.

Each check compares the species one at a time, each on its own scale. A
species at 1e-3 beside one at 1e3 is judged as carefully as the large one.

## Species the output points miss

A species whose output points miss part of its trajectory has a small peak
there: an oscillation sampled at its zero crossings, or the first point after a
fast transient between two points. Its move at a tighter tolerance is then a
large share of that small peak even when the run is right. So a species that
moves by more than 0.1% of its peak at the points is judged again, each point
against the largest value the species reaches beside it, read from the `grid`
check's run at 8 times the points. The report's `n_between` counts the species
judged this way, and `next` suggests more output points to see those values.
A spike between two points excuses only those two. A spike in every interval,
with the points in its quiet phase, excuses them all, and a species that is
zero at every one of the 8 times as many points is not seen.

`report["next"]` says what to do next, in a sentence. The report is a plain
dict, ready for `json.dumps`. Keep it with `result.record` when you report the
numbers.

## What each status means

| status | meaning |
|---|---|
| `pass` | The answer did not move past the check's bound. |
| `diff` | It did. Read the check's `note`, and `report["next"]`. |
| `skip` | The check could not run on this result. Its `note` says why. |
| `unsupported` | The check does not apply to this kind of run or model. |
| `no_base` | No species is converged at the run's tolerance, so there is nothing to judge a rerun against. |
| `timeout` | The `budget=` you gave ran out before the check finished. |
| `exception` | The changed run raised. Its `note` has the error. |
| `info` | The check reports numbers and judges nothing: it neither passes nor fails. Only `replicates`, a stochastic run's check, gives it. |

`report["ok"]` is True when at least one check passed, none is `diff`,
`timeout` or `exception`, and the run is converged. **A `skip` or an
`unsupported` check does not make a report not ok.** So `ok` says that nothing
that ran moved the answer. It does not say that every check ran. Read the
statuses. An `info` check is no pass either, so a report whose only check is
`info`, as a stochastic run's is, is not ok.

## A check that fails

A run at a loose tolerance moves when the tolerance is tightened:

```python
loose = sim.run(t_span=(0, 10), n_points=11, rtol=1e-2, start="initial")
report = loose.check()
print(report["ok"])  # False
print([(c["name"], c["status"]) for c in report["checks"]])  # [('tolerance', 'diff'), ('grid', 'no_base')]
print(report["next"])  # rerun at a tighter rtol and atol until it does not move

better = loose.rerun(rtol=1e-6)
print(better.check()["ok"])  # True
```

The `grid` check here is `no_base`: no species is converged at `rtol=1e-2`, so
a rerun on another grid has nothing to be compared with. Fix the tolerance
first, then read the other checks.

A `diff` between two BNGsim paths, such as `jacobian` or `backend` below, is a
bug. Report it with `json.dumps(report)`, `result.record` and the model file.

## Species the run does not resolve

The solver holds a species near zero to its `atol`, not to `rtol`. A species
that stays within 100 times its `atol` of zero is not resolved by the run: its
value is mostly solver noise. The report names it and judges it only against
the run's largest species, so its noise is not counted as a failure.

```python
model.set_param("kf", 1e-12)  # so little binding that C stays near zero
tiny = sim.run(t_span=(0, 10), n_points=11, start="initial")
report = tiny.check()
print(report["ok"])  # True
print(report["not_resolved"])  # ['A(b!1).B(a!1)']
model.set_param("kf", 0.01)
```

The check passed, but it says nothing about C's value. To report C, rerun with
an `atol` well below C's size, for example `tiny.rerun(atol=1e-15)`, and check
that run. `report["next"]` says so too.

## More checks

`result.check("all")` adds four checks to the default two. A list of names,
such as `result.check(["tolerance", "jacobian"])`, runs only those.

- **`jacobian`**: the run again with a difference-quotient Jacobian.
- **`linear_solver`**: the run again on the other linear solver, dense for a
  run on KLU and KLU for a dense one.
- **`backend`**: compiled code against the interpreted right-hand side.
- **`second_engine`**: the run again in libRoadRunner.

```python
report = result.check("all")
statuses = {c["name"]: c["status"] for c in report["checks"]}
print(statuses["second_engine"])  # unsupported
```

The first three compare two BNGsim paths. The fourth is the only one that
uses another simulator. It applies only where, among other things:

- the model was loaded from an SBML or Antimony file (this one is a `.net`
  file, so the check is `unsupported`), and no edit to it is recorded;
- libRoadRunner is installed (`pip install bngsim[roadrunner]`) and can run
  the model;
- the run started from the file's initial state, at the file's parameters;
- the run is a plain run over a `t_span`, not one at `sample_times=` or one
  stopped at a steady state;
- the file has not changed since it was loaded.

Otherwise the check is `skip` or `unsupported`, and its note says why. Here a
parameter changed after loading:

```python
sbml_model = bngsim.examples.load("binding", format="sbml")
sbml_model.set_param("kf", 0.02)
changed = bngsim.Simulator(sbml_model).run(t_span=(0, 10), n_points=11)
report = changed.check(["second_engine"])
print([(c["name"], c["status"]) for c in report["checks"]])  # [('second_engine', 'skip')]
print(report["ok"])  # False
```

The check's note gives the reason: here, that `kf` is 0.02 where the file has
0.01. Where libRoadRunner is not installed, it says that instead. A report
whose only check could not run is not ok, since no check passed.

## Sensitivities

A run that computed forward sensitivities also gets a `sensitivity` check by
default: its sensitivities at a tenth and a hundredth of the tolerance.

```python
sens = bngsim.Simulator(model, sensitivity_params=["kf", "kr"])
s = sens.run(t_span=(0, 10), n_points=11, start="initial")
report = s.check()
print([(c["name"], c["status"]) for c in report["checks"]])  # [('tolerance', 'pass'), ('grid', 'pass'), ('sensitivity', 'pass')]
```

`s.check("all")` adds two more:

- **`sensitivity_fd`**: the sensitivities against central difference
  quotients of runs at a tighter tolerance.
- **`sensitivity_second_engine`**: the sensitivities against AMICI's, where
  AMICI is installed. It applies to a model `second_engine` could run that has
  no event and no switch in its math. AMICI at a tenth of the run's tolerance
  is the reference. A series that AMICI itself does not converge there is
  judged against AMICI at a hundredth instead (counted in `n_judged_deeper`).
  A series AMICI does not converge at a hundredth either is not judged, and is
  named in `no_reference`. A `pass` covers only the series it judged.

## Stochastic runs

An SSA or PSA run gets one check, `replicates`. It runs the model again at 10
other seeds and reports, for each observable at the last time, the mean over
them (`mean_last`), their standard deviation (`sd_last`), the mean's standard
error (`se_last`), and `z`, how far this run is from that mean. A network-free
run (`method="nf"` or `"rm"`) gets none: there `replicates` is `unsupported`.
Either way the report is not ok.

```python
model.reset()
ssa = bngsim.Simulator(model, method="ssa")
one = ssa.run(t_span=(0, 10), n_points=11, seed=1)
report = one.check()
print([(c["name"], c["status"]) for c in report["checks"]])  # [('replicates', 'info')]
print(report["ok"])  # False
print(report["next"])  # The replicates check judges nothing, so this run is not checked ...
```

**`replicates` judges nothing.** Its status is `info`, neither a pass nor a
failure, so `ok` is False and `bngsim check` exits 6: the run is not checked.
`z` is in standard deviations of one draw about the mean of the others,
`sd_last·sqrt(1 + 1/10)`, since this run is a draw like each rerun. A correct
run's `z` is then about a t-statistic on 9 degrees of freedom, past 2.3 in
about one run in twenty, so a large `z` among several observables is expected
now and then, and is no finding either. One trajectory is one draw. Report a
mean over many replicates, with its standard error, the number of replicates
and the seed. To check a stochastic answer, compare that mean with a closed
form or with another simulator, allowing for the standard errors.

## From the command line

`bngsim check` makes the run and then the checks, and prints the same report.
It exits 6 when the report is not ok:

```bash
bngsim check model.xml --t-end 100 --checks all --json
```

## Before you call a result checked

Whether you are a person or an agent reporting a result:

1. **Read every status, not only `ok`.** A `skip` or `unsupported` check did
   not run. Say which checks ran.
2. **A `diff` is a finding.** Follow `report["next"]`. Do not report the number
   as resolved.
3. **Species in `not_resolved` have no checked value.** Rerun with a smaller
   `atol` before you report them.
4. **Say "checked against libRoadRunner" only when `second_engine` is `pass`**,
   and "checked against AMICI" only when `sensitivity_second_engine` is `pass`.
   Even then, AMICI's pass covers only the series it judged: name the ones in
   its `no_reference`. The other checks compare BNGsim with itself.
5. **A stochastic run's check is no check of its answer.** Its report is never
   ok. Report means over replicates.
6. **Keep the evidence.** Save `json.dumps(report)` beside `result.record`.

```python
def checked_against_libroadrunner(report: dict) -> bool:
    """True only where libRoadRunner ran the model and agreed."""
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    return report["ok"] and statuses.get("second_engine") == "pass"


print(checked_against_libroadrunner(result.check("all")))  # False
```

The answer here is False because the model is a `.net` file, which
libRoadRunner does not read.
