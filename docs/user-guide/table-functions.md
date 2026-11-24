---
summary: "BioNetGen table functions (TFUN): declaring one, the .tfun file format, its index variable, and giving a table new data."
---

# Table functions (TFUN)

A table function interpolates a value from a table of data. Use one for an
input the model cannot write as a formula: an experimental forcing, a measured
case count, a dosing schedule.

## Declaring one

Declare a table function in the model's `functions` block. BNGsim reads `tfun`
in a `.net` file directly:

```
begin functions
    1 cumNcases()  tfun('case_data.tfun')           # indexed by time (the default)
    2 response()   tfun('dose.tfun', drug_conc)     # indexed by a parameter
    3 drive()      tfun('schedule.tfun', time, method=>"step")
    4 inline()     tfun([0,1,2], [0,10,20], time, method=>"linear")
    5 scaled()     2*tfun('signal.tfun', time)      # a table inside a function
end functions
```

A table that is a function's whole body takes the function's name
(`cumNcases`, `drive`). A table inside a function's arithmetic is named
`<function>__tfun<k>` (`scaled__tfun0`), counting from 0 along the line.

To try the examples on this page, write the tables that block reads, and a
`.net` file that holds it, with a reaction for each table to drive:

```python
from pathlib import Path

import bngsim

Path("case_data.tfun").write_text("# time  cumNcases\n0 0\n1 0\n2 1\n3 1\n4 2\n5 5\n", encoding="utf-8")
Path("dose.tfun").write_text("# drug_conc  response\n0 0\n1 0.5\n10 0.9\n", encoding="utf-8")
Path("schedule.tfun").write_text("# time  drive\n0 0\n24 5\n48 0\n", encoding="utf-8")
Path("signal.tfun").write_text("# time  scaled\n0 1\n10 2\n", encoding="utf-8")
Path("model.net").write_text("""\
begin parameters
    1 drug_conc  1.0
    2 kdeg       0.1
end parameters
begin species
    1 A() 0
end species
begin functions
    1 cumNcases()  tfun('case_data.tfun')
    2 response()   tfun('dose.tfun', drug_conc)
    3 drive()      tfun('schedule.tfun', time, method=>"step")
    4 inline()     tfun([0,1,2], [0,10,20], time, method=>"linear")
    5 scaled()     2*tfun('signal.tfun', time)
end functions
begin reactions
    1 0 1 cumNcases
    2 0 1 response
    3 0 1 drive
    4 0 1 inline
    5 0 1 scaled
    6 1 0 kdeg
end reactions
begin groups
    1 A 1
end groups
""", encoding="utf-8")
model = bngsim.Model.from_net("model.net")
```

```python
print(model.n_table_functions)      # 5
print(model.table_function_names)   # ['cumNcases', 'response', 'drive', 'inline', 'scaled__tfun0']
```

No change to BNG2.pl is needed, so you can add `tfun()` to a `.net` file by
hand or build it through the Python API.

## Giving a table new data

`add_table_function`, under the name of one of the model's tables, replaces that
table's data in place:

```python
# From a .tfun file of the same form as the model's own
Path("case_data_2021.tfun").write_text("# time  cumNcases\n0 0\n1 2\n2 3\n3 7\n", encoding="utf-8")
model.add_table_function("cumNcases", file="case_data_2021.tfun")

# From arrays, read as a step from now on
model.add_table_function(
    "drive", times=[0, 24, 25, 48, 49], values=[0, 5, 0, 5, 0], method="step"
)

# A parameter-indexed table names its index again
Path("dose_response_v2.tfun").write_text("# drug_conc  response\n0 0\n1 0.6\n10 0.95\n", encoding="utf-8")
model.add_table_function("response", file="dose_response_v2.tfun", index="drug_conc")

# A table inside a function, by its own name
model.add_table_function("scaled__tfun0", times=[0, 10], values=[1, 3])
```

Every engine reads the new data from then on: the interpreter, compiled code,
the SSA, the steady-state solvers, and a `Simulator` made before the call. The
breakpoints a run stops on move with it. The table keeps its interpolation
unless `method` names another (a `.tfun` file carries none). A clone keeps the
data it was cloned with.

Watch out:

- The new data must use the same index as the old: `time`, or the same
  parameter, observable or function. `Time`, `t` and `time()` all name the
  clock, and `drug_conc()` names `drug_conc`.
- A file's value column may be headed with the table's name or with its
  function's name.
- Bad data, a file that cannot be read, or a different index raises
  `ModelError` and leaves the table as it was.
- A function that is not a table cannot take a table's data:
  `add_table_function` raises `ModelError` for its name.
- Under a name the model does not have, `add_table_function` adds a table that
  nothing reads, and warns. A loaded model's expressions are fixed, so the new
  table changes no run. To drive a model from data, declare the `tfun` in the
  model and give it data here.

## The `.tfun` file format

Two whitespace-separated columns under a `#` header, as in a `.gdat` file:

```
# time  cumNcases
0  0
1  0
2  1
3  1
4  2
5  5
```

- The first line must be a `#` header. It names the two columns: the index,
  then the table (see [Header and index spellings](#header-and-index-spellings)).
- Column 1 is the index. It must be strictly increasing; a repeated value
  raises.
- Column 2 is the value, any real number.
- At least two data rows.
- Spaces or tabs between columns.

Interpolation is `linear` (the default) or `step`, set with `method`. Outside
the table's range, the first or last value is held.

### Header and index spellings

The headers and the index name are normalized before they are checked, as in
BioNetGen's `TfunReader.pm`, so one file works with any spelling BNG accepts:

- The time index matches case-insensitively: `time`, `Time`, `T`, `TIME` and
  `t()` all name the model's time.
- A trailing `()` is dropped from both headers and from the index. So
  `# drug_conc()  response()` matches `tfun('file.tfun', drug_conc)`, and
  `# Time  cumNcases()` matches a time-indexed table.

## The index

A table is indexed by time unless the second argument to `tfun()` names
something else:

| Call | Indexed by |
|---|---|
| `tfun('file.tfun')` | `time` (default) |
| `tfun('file.tfun', drug_conc)` | the parameter `drug_conc` |
| `tfun('file.tfun', A_tot)` | the observable `A_tot` |
| `tfun('file.tfun', time, method=>"step")` | `time`, piecewise-constant |
| `tfun([0,1,2], [0,10,20], time)` | `time`, with inline data and no file |
| `tfun('file.tfun', h)` | the function `h` |

The index may be written as a call, `A_tot()` or `time()`, as BioNetGen writes
it into a `.net` file. The table returns its interpolated value at the index's
current value.

## `tfun(...)` inside a larger expression

A `tfun` call can sit inside arithmetic, and a function can hold more than one.
This works in the interpreter and in compiled code:

```
begin functions
    1 f_complex() (tfun('drive.tfun', time) + 5) / k_scale
    2 f_combo()    tfun([0,1,2], [10,20,40], time) / 10 + offset
end functions
```

Each embedded call becomes its own table, listed in `table_function_names` as
`<function>__tfun<k>`. BioNetGen's own parser keeps only one table per
expression.

## Tables over time and the ODE solver

Between breakpoints, a table over time is flat (`step`) or straight
(`linear`). An adaptive solver could step right over a short pulse and never
see it. So BNGsim stops the integration at the breakpoints a step could miss:

- at each feature: a breakpoint whose break its neighbors do not account
  for, such as the edges of a pulse or the ends of the table;
- along a stretch that only rises, falls or bends smoothly, a stop each time
  the breaks have added up to a twentieth of the table's range.

In a `step` table every stop restarts the solver. In a `linear` table a feature
restarts it, and a stop along a smooth stretch only ends a step. So a finely
sampled rising staircase is stopped at its features and about once per
twentieth of its range, not at every jump.

So a dosing pulse such as

```
F() tfun([0,10,12,50], [0,5,0,0], time, method=>"step")
```

is integrated whatever its width and whatever the output grid, even when it is
small next to the table's largest value (maintenance doses of 5 after a loading
dose of 1000).

A finely sampled smooth signal gets only a few stops, and their number does not
grow with the number of samples. Noisy data gets a stop, with a restart, at
every breakpoint.

A rate law that switches on a table over time, such as
`if(abs(G - 2) < 0.01, k, 0)` with `G` a table, is stopped at where it switches
(see "Rate laws that switch on time" in the [solver guide](solvers.md)).

A table indexed by an observable that counts time at unit rate, as BNGL models
often write time, is stopped at where the counter reaches each breakpoint. A
table indexed by anything else that moves with the state gets no stops,
because nothing knows in advance when its breakpoints are crossed. If such a
table has a narrow segment, pass `max_step` no larger than half the time the
index takes to cross it.

The stochastic engines also end a step at each breakpoint of a table over time.

## Table functions and the Jacobian

A table indexed by `time` or by a parameter does not depend on the state. A
rate law that reads one keeps the analytical Jacobian: the table's current
value enters as a coefficient, as `time()` does. `sim.jacobian_strategy` reports
`"analytical"` for such a model.

- In compiled code, a table that is a function's whole body is read from the
  value the function already holds. A table inside arithmetic
  (`f() 2*tfun(...)`) uses the interpreted analytical Jacobian, which is just as
  exact.
- A table indexed by an observable depends on the state, and a rate law that
  reads one uses a finite-difference Jacobian.
- Sensitivities treat a parameter-indexed table as depending on its parameter.

## NFsim XML

In NFsim XML, a table function is written as:

- File-backed:
  `<Function type="TFUN" file="..." ctrName="..." method="linear|step">`
- Inline:
  `<Function type="TFUN" mode="inline" ctrName="..." xData="..." yData="..." method="linear|step">`
- The placeholder for its value in `<Expression>` is `__TFUN_VAL__`.

Rules:

- `xData` and `yData` are comma-separated; whitespace around values is trimmed.
- Scientific notation such as `1e-3` or `2.5E+2` is accepted.
- `xData` and `yData` must have the same length.
- `xData` must be strictly increasing.

`method="nf"` reads a `step` table over time at each of its x values, and
integrates a rate law that reads a `linear` one between them. See
[Rate laws that read `time()`](network-free.md#rate-laws-that-read-time).
