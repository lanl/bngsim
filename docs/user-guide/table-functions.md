# Table functions (TFUN)

## Table functions (TFUN)

```python
# Time-indexed: load tabular data for piecewise-linear interpolation
model.add_table_function("cumNcases", file="case_data.tfun")

# Parameter-indexed (e.g., dose-response)
model.add_table_function("response", file="dose_response.tfun", index="drug_conc")

# Step interpolation (piecewise-constant)
model.add_table_function(
    "dose_step", file="dose_response.tfun", index="time", method="step"
)

# From in-memory arrays
model.add_table_function(
    "drive", times=[0, 1, 2, 5], values=[0, 0, 1, 5], method="linear"
)

# Introspect
print(model.n_table_functions)      # 1
print(model.table_function_names)   # ['cumNcases']
```

Table functions can also be defined in `.net` files using the `tfun()` syntax:

```
begin functions
    1 cumNcases()  tfun('case_data.tfun')           # time-indexed (default)
    2 response()   tfun('dose.tfun', drug_conc)     # parameter-indexed
    3 drive()      tfun('dose.tfun', time, method=>"step")
    4 inline()     tfun([0,1,2], [0,10,20], time, method=>"linear")
end functions
```

`.tfun` file format (GDAT-style, `#`-prefixed header required):
```
# time  cumNcases
0  0
1  0
2  1
3  1
4  2
5  5
```

- Column 1: index values (must be monotonically increasing)
- Column 2: function values
- Interpolation: `linear` (default) or `step`
- Extrapolation: constant (hold first/last value beyond endpoints)
- Minimum 2 data rows required

## Table Functions (TFUN) — Detailed Reference

Table functions provide piecewise-linear interpolation from tabular data,
useful for time-varying inputs (e.g., experimental forcing, drug dosing).

**Three ways to create table functions:**

1. **In a `.net` file** (parsed automatically by BNGsim):
```
begin functions
    1 cumNcases()  tfun('case_data.tfun')            # time-indexed
    2 response()   tfun('dose.tfun', drug_conc)      # parameter-indexed
end functions
```

2. **From a file in Python:**
```python
model.add_table_function("cumNcases", file="case_data.tfun")
model.add_table_function("response", file="dose.tfun", index="drug_conc")
```

3. **From in-memory arrays:**
```python
model.add_table_function("drive",
    times=[0, 1, 2, 5, 10],
    values=[0, 0, 1, 5, 5])
```

**`.tfun` file format** (GDAT-style, two columns, `#`-prefixed header):
```
# time  cumNcases
0  0
1  0
2  1
3  1
4  2
5  5
6  10
7  20
```

Requirements:
- First line must be a `#`-prefixed header (column names; ignored by parser)
- Column 1: index values — must be **monotonically increasing** (strictly)
- Column 2: function values (any real numbers)
- Minimum 2 data rows
- Whitespace-separated columns (spaces or tabs)

**Interpolation**: Linear between data points.
**Extrapolation**: Constant — holds first value below range, last value above range.

**Index variable**: By default, the table function is indexed by simulation time.
Specify a different index with the second argument:
- `tfun('file.tfun')` — indexed by `time` (default)
- `tfun('file.tfun', drug_conc)` — indexed by parameter `drug_conc`
- `tfun('file.tfun', A_tot)` — indexed by observable `A_tot`
- `tfun('file.tfun', time, method=>"step")` — piecewise-constant interpolation
- `tfun([0,1,2], [0,10,20], time)` — inline data (no external file)

The index variable is evaluated at each time step, and the table function returns
the interpolated value at that index.

**Header / index canonicalization** (matches BioNetGen's `TfunReader.pm`):
the `.tfun` column-1 header, the column-2 header, and the index name passed
to `tfun()` are normalized before validation so a single `.tfun` file works
across BNG-acceptable spellings.

- The time index matches case-insensitively: `time`, `Time`, `T`, `TIME`,
  and `t()` all canonicalize to the model's time variable.
- A trailing `()` is stripped from both `.tfun` header columns and from
  the index argument, regardless of index kind. So
  `# drug_conc()  response()` is accepted against a
  `tfun('file.tfun', drug_conc)` call that targets the `drug_conc`
  parameter, and a header of `# Time  cumNcases()` is accepted on a
  time-indexed tfun.

**Wrapper-form `tfun(...)` inside a larger expression** is supported on
both the `.net` interpreter and the codegen paths:

```
begin functions
    1 f_complex() (tfun('drive.tfun', time) + 5) / k_scale
    2 f_combo()    tfun([0,1,2], [10,20,40], time) / 10 + offset
end functions
```

The loader extracts each embedded `tfun(...)` call into a synthetic
anonymous table function (visible as `<bng_func>__tfun<k>` in
`table_function_names`) and rewrites the call site so the wrapping
arithmetic survives untouched into ExprTk evaluation. The codegen path
emits a `tfun_eval(tf_id, idx, ctx)` callback nested inside the
translated wrapper math. Multiple `tfun(...)` calls per function body
are supported (each gets its own synthetic name and `tf_id`); this is a
strict extension of BioNetGen's own parser, which only stores one
`tfunData` per expression.

**How it works**: BNGsim parses `tfun()` syntax directly in `net_file_loader.cpp`.
No changes to BNG2.pl are required — users can add `tfun()` to .net files or use
the Python API.

**NFsim XML TFUN format**:
- Canonical placeholder in `<Expression>` is `__TFUN_VAL__`.
- File-backed TFUN:
  - `<Function type="TFUN" file="..." ctrName="..." method="linear|step">`
- Inline TFUN:
  - `<Function type="TFUN" mode="inline" ctrName="..." xData="..." yData="..." method="linear|step">`
- Validation rules:
  - `xData`/`yData` CSV values are whitespace-trimmed
  - scientific notation is accepted (e.g., `1e-3`, `2.5E+2`)
  - `xData` and `yData` lengths must match
  - `xData` must be strictly increasing
