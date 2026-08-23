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
