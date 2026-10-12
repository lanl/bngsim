# Table functions (TFUN)

Table functions interpolate a quantity from tabular data, which is how a model
carries a time-varying input it cannot write down: experimental forcing, a
measured case count, a drug dosing schedule.

## Three ways to create one

**1. In a `.net` file**, parsed automatically:

```
begin functions
    1 cumNcases()  tfun('case_data.tfun')           # time-indexed (default)
    2 response()   tfun('dose.tfun', drug_conc)     # parameter-indexed
    3 drive()      tfun('dose.tfun', time, method=>"step")
    4 inline()     tfun([0,1,2], [0,10,20], time, method=>"linear")
end functions
```

**2. From a file in Python:**

```python
# Time-indexed: piecewise-linear interpolation
model.add_table_function("cumNcases", file="case_data.tfun")

# Parameter-indexed (e.g. dose-response)
model.add_table_function("response", file="dose_response.tfun", index="drug_conc")

# Step interpolation (piecewise-constant)
model.add_table_function(
    "dose_step", file="dose_response.tfun", index="time", method="step"
)
```

**3. From in-memory arrays:**

```python
model.add_table_function(
    "drive", times=[0, 1, 2, 5], values=[0, 0, 1, 5], method="linear"
)
```

Introspect what a model carries:

```python
print(model.n_table_functions)      # 1
print(model.table_function_names)   # ['cumNcases']
```

## The `.tfun` file format

GDAT-style, two whitespace-separated columns, with a required `#`-prefixed
header:

```
# time  cumNcases
0  0
1  0
2  1
3  1
4  2
5  5
```

- The first line must be a `#`-prefixed header. It names the columns and is
  otherwise ignored.
- Column 1 is the index. It must be **strictly** increasing — a repeated value
  raises, not just a decreasing one.
- Column 2 is the function value, any real number.
- At least two data rows are required.
- Columns may be separated by spaces or tabs.

**Interpolation** is `linear` (the default) or `step`, chosen with `method`.
**Extrapolation** is constant: the first value is held below the range and the
last value above it.

## The index variable

By default a table function is indexed by simulation time. The second argument
to `tfun()` chooses something else:

| Call | Indexed by |
|---|---|
| `tfun('file.tfun')` | `time` (default) |
| `tfun('file.tfun', drug_conc)` | the parameter `drug_conc` |
| `tfun('file.tfun', A_tot)` | the observable `A_tot` |
| `tfun('file.tfun', time, method=>"step")` | `time`, piecewise-constant |
| `tfun([0,1,2], [0,10,20], time)` | `time`, with inline data and no file |

The index is evaluated at each step, and the table function returns the
interpolated value there.

### Breakpoints and the integrator

A table over time is constant (`step`) or straight (`linear`) between its
breakpoints, so over a segment that begins and ends at the same value an
adaptive integrator sees nothing to resolve and can step across it. bngsim
stops the ODE integration on the breakpoints a step could span unseen. A table
breaks at a breakpoint where a `step` table's value jumps or a `linear` table's
slope turns. A feature that leaves and comes back has breaks that differ: a
pulse is a jump up and a jump down. So a breakpoint is a stop where its two
neighbours do not account for its break (it is not about their mean), or where
the breakpoint before or after it has no break to speak of (under a quarter of
its own): the foot of a feature on a flat or straight stretch, and the ends of
the table. In a `step` table a jump next to one of the other sign is a stop
too. Along a stretch of
breaks that do account for each other, where the table only goes on rising,
falling or bending smoothly, there is a stop each time they have added up to a
twentieth of the range of the table's values. The integrator restarts at a
jump of a `step` table and at a bend of a `linear` table that is such a
feature; at a stop along a smooth stretch of a `linear` table it only ends a
step. So a dosing pulse such as

```
F() tfun([0,10,12,50], [0,5,0,0], time, method=>"step")
```

is integrated whatever its width, whatever the output grid, and however small
it is beside the largest value in the table (maintenance doses of 5 after a
loading dose of 1000 are each stopped at). A table of many samples of a smooth
signal breaks a little at every breakpoint, each break the mean of its
neighbours, and is stopped at its ends, where it changes curvature, and a few
times along the way: its bends are the curvature of what was sampled, which
the integrator's error control follows. The number of stops does not grow with
the number of samples (a few dozen for five periods of a sinusoid; 0.5 ms a
run at 10,000 points or 100,000, where it was 0.3).
A table of noisy data breaks unlike its neighbours at every breakpoint and is
stopped, with a restart, at each.

A rate law that switches on a table over time, `if(abs(G - 2) < 0.01, k, 0)`
with `G` such a table, is a condition on the clock, and is stopped at where it
changes like any other (see "Rate laws that switch on time" in the
[solver guide](solvers.md)).

A table indexed
by an observable that is a unit-rate counter (the way a BNGL model often spells
time) is stopped at where the counter reaches each breakpoint. A table indexed
by anything else that moves with the state gets no stops: its breakpoints are
crossed at times nothing knows before the run. If such a table has a narrow
segment, pass `max_step` no larger than half the time the index takes to cross
it. The stochastic engines end a step on each breakpoint of a table over time
as well.

## Header and index canonicalization

The column-1 header, the column-2 header, and the index name passed to `tfun()`
are all normalized before validation, so one `.tfun` file works across the
spellings BNG accepts. This matches BioNetGen's `TfunReader.pm`.

- The time index matches case-insensitively: `time`, `Time`, `T`, `TIME` and
  `t()` all canonicalize to the model's time variable.
- A trailing `()` is stripped from both header columns and from the index
  argument, whatever the index kind. So a header of `# drug_conc()  response()`
  is accepted against `tfun('file.tfun', drug_conc)` targeting the `drug_conc`
  parameter, and `# Time  cumNcases()` is accepted on a time-indexed table.

## `tfun(...)` inside a larger expression

Supported on both the `.net` interpreter and the codegen path:

```
begin functions
    1 f_complex() (tfun('drive.tfun', time) + 5) / k_scale
    2 f_combo()    tfun([0,1,2], [10,20,40], time) / 10 + offset
end functions
```

The loader extracts each embedded `tfun(...)` call into a synthetic anonymous
table function — visible as `<bng_func>__tfun<k>` in `table_function_names` —
and rewrites the call site so the wrapping arithmetic survives into ExprTk
evaluation. The codegen path emits a `tfun_eval(tf_id, idx, ctx)` callback
nested inside the translated wrapper math.

Multiple `tfun(...)` calls per function body work, each getting its own
synthetic name and `tf_id`. That is a strict extension of BioNetGen's own
parser, which stores only one `tfunData` per expression.

## Table functions and the Jacobian

A table indexed by `time` or by a parameter does not depend on the state, so a
rate law that reads one keeps the analytical Jacobian: the table's current value
enters `∂f/∂y` as a coefficient, the way `time()` does, and is never itself
differentiated. `sim.jacobian_strategy` reports `"analytical"` for such a model.

- With codegen, the compiled Jacobian reads a table that is a whole function
  body (`f() tfun(...)`) from the value the function recomputation already
  holds. A table embedded in arithmetic (`f() 2*tfun(...)`) leaves the compiled
  Jacobian to the interpreted analytical one, which is equally exact.
- A table indexed by an **observable** depends on the state through its index,
  and a rate law that reads one uses the finite-difference Jacobian.
- Sensitivities treat a parameter-indexed table as depending on its parameter.

## How it works

BNGsim parses `tfun()` syntax directly in `net_file_loader.cpp`. No change to
BNG2.pl is required, so `tfun()` can be added to a `.net` file by hand or built
through the Python API.

## NFsim XML TFUN format

- The canonical placeholder in `<Expression>` is `__TFUN_VAL__`.
- File-backed:
  `<Function type="TFUN" file="..." ctrName="..." method="linear|step">`
- Inline:
  `<Function type="TFUN" mode="inline" ctrName="..." xData="..." yData="..." method="linear|step">`

Validation rules:

- `xData` / `yData` CSV values are whitespace-trimmed.
- Scientific notation is accepted, for example `1e-3` or `2.5E+2`.
- `xData` and `yData` lengths must match.
- `xData` must be strictly increasing.
