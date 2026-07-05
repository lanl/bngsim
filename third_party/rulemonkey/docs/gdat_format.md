# `.gdat` output format

`rm_driver` writes its trajectory output to `stdout` in `.gdat`
format.  This document is the authoritative reference for that
format.  CPU time goes to `stderr`.

The default column layout is `time` followed by every observable —
identical to what NFsim emits.  Passing `--print-functions` to
`rm_driver` appends one trailing column per *global function* (the
BNGL `begin functions` entries with no local, per-molecule arguments).
This opt-in mirrors BNGL's `print_functions=>1`; NFsim gates the
equivalent output behind `turnOnGlobalFuncOut()`.  Without the flag —
or for a model with no global functions — the file is observables-only.

## File shape

One header line followed by N data lines.  Every line is tab-separated.

```
# time<TAB>obs_1<TAB>...<TAB>obs_K[<TAB>fn_1<TAB>...<TAB>fn_F]
t_0<TAB>v_1,0<TAB>...<TAB>v_K,0[<TAB>g_1,0<TAB>...<TAB>g_F,0]
t_1<TAB>v_1,1<TAB>...<TAB>v_K,1[<TAB>g_1,1<TAB>...<TAB>g_F,1]
...
t_N<TAB>v_1,N<TAB>...<TAB>v_K,N[<TAB>g_1,N<TAB>...<TAB>g_F,N]
```

The bracketed `fn_*` / `g_*` columns are present only when
`--print-functions` is passed.

- The header line begins with a literal `#` followed by a single
  space, then `time`, then a tab-separated list of observable names
  in the order they were declared in the BNGL `begin observables`
  block.  With `--print-functions`, the global-function names follow,
  in `begin functions` declaration order.
- Observable and function names are emitted verbatim from the XML's
  `<Observable name="...">` / `<Function id="...">` attributes.  RM
  never renames or re-encodes them.
- Global functions are evaluated at every sample time from the same
  pool state as the observables — they are exposed values RM already
  computes for rate-law evaluation, not a separate pass.
- Every data line has exactly `1 + K` tab-separated fields by default,
  or `1 + K + F` with `--print-functions`, where `K` is the number of
  observables and `F` the number of global functions.
- For a `rm_driver model.xml T N` invocation, the file has
  **`N + 1` data lines** — samples at `t_start`, `t_start + dt`, …,
  `t_end`, where `dt = (t_end − t_start) / N`.

The in-process C++ API (`rulemonkey::Result`) exposes
`function_names` / `function_data` **unconditionally** — the
`--print-functions` opt-in governs only `rm_driver`'s text output, not
the embedding surface.

## Numeric precision

Time, observable, and global-function values are emitted via the C++
`<<` operator with `std::cout`'s default precision: **6 significant
figures**.
This matches NFsim's default and is sufficient for the parity
benchmarks in this repo.  External consumers who need more digits
should:

- Round their reference data to six sig figs at compare time, or
- Patch `cpp/cli/rm_driver.cpp` to set `std::cout << std::setprecision(N)`
  for the desired precision (the `setprecision(17)` round-trip
  setting is appropriate for bit-exact regeneration).

The in-process C++ API (`rulemonkey::Result`) carries full
double-precision values; the precision loss is purely a property of
the CLI driver's text output.

## Standard error

```
ERROR: <message>          (only if loading or running failed)
WARN:  <unsupported feature> (during XML parse, if features at
                              Severity::Warn are present)
ERROR: <unsupported feature> (during XML parse, if features at
                              Severity::Error are present)
<cpu_seconds>             (last line, on success)
```

A successful run terminates with exit code 0 and prints a single
floating-point seconds value as the last `stderr` line.  If
unsupported BNGL features were detected, those are reported on
preceding `stderr` lines (Warn-severity always; Error-severity if
present, exit 2 unless `--ignore-unsupported`).

## Parsing recipe

```python
# Python — handles header normalisation and float conversion
def parse_gdat(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    headers = lines[0].lstrip("#").strip().split("\t")  # ['time', 'A', ...]
    rows = [list(map(float, l.split("\t"))) for l in lines[1:]]
    return headers, rows

headers, rows = parse_gdat("traj.gdat")
times = [r[0] for r in rows]
A = [r[headers.index("A")] for r in rows]
```

Two header-parsing notes:

- The `#` may be followed by a single space (RM emits `# time...`),
  or by no space (some NFsim builds emit `#time...`); `lstrip("#").strip()`
  handles both.
- The first column is always called `time` (lowercase) regardless of
  whether the BNGL declared an observable called `time`.

## Edge cases

- **Zero observables**: a model with no `begin observables` block
  produces a header of just `# time\n` (plus any global-function
  columns, if `--print-functions` is set) and rows with a single time
  field per line.  Parsing the columns by tab still works.
- **No function columns**: without `--print-functions` — or for a
  model with no global functions (no `begin functions` block, or only
  local functions) — the file carries no trailing function columns and
  is observables-only.  Index columns by header name (see the parsing
  recipe) rather than position so a consumer is agnostic to whether
  function columns are present.
- **Absorbing state**: when the system reaches zero total propensity
  (no rule can fire), RM fills the remaining sample points by
  repeating the last computed state, so the row count is unchanged.
- **`n_steps == 0`**: not supported on the CLI (rejected with an
  actionable error; pass `n_steps >= 1`).  The in-process engine
  treats `n_points == 0` as a two-endpoint sampling — the
  `step_to(time)` API uses this internally and discards the
  resulting trajectory, so the *caller* sees no output but the
  endpoints are still computed.
- **Explicit output times**: the in-process `TimeSpec::sample_times`
  field (issue #16) overrides the uniform `n_points` grid with an
  explicit sorted list of instants — `Result::time` then echoes that
  list and there is one output row per requested time.  The CLI always
  emits the uniform grid; explicit times are an embedder-only feature
  (see `model_semantics.md`).

## See also

- [`quickstart.md`](quickstart.md) — full command-line walkthrough.
- [`include/rulemonkey/types.hpp`](../include/rulemonkey/types.hpp)
  — the in-process `Result` struct that holds full-precision values.
