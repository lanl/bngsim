---
summary: "The bngsim command (also python -m bngsim): doctor, guide, examples, inspect, run, check, convert and cache; one JSON object on standard output with --json, errors as JSON on standard error, and an exit code for each outcome."
---

# The `bngsim` command

`bngsim` inspects and runs a model, and this install of BNGsim, from a shell. It is also
`python -m bngsim`, which works wherever BNGsim is installed, even where the
console script is not on `PATH`.

Each command is a thin layer over a Python call. The command adds no behavior
of its own, so what it prints is what the call returns.

| command | what it does | Python call behind it |
|---|---|---|
| `bngsim doctor` | what this install can do, what is missing, and the fix for each; with the Python, the platform, and the C compiler codegen would use and whether it builds a test library | `bngsim.capabilities()` |
| `bngsim guide` | prints the agent guide of this version | `bngsim.agent_guide()` |
| `bngsim skill` | prints the guide as a `SKILL.md` file | `bngsim.agent_guide(skill=True)` |
| `bngsim examples [NAME]` | lists the example models that ship with BNGsim, with their formats and files | `bngsim.examples` |
| `bngsim inspect MODEL [--limit N\|all]` | what the model is, and which methods it admits and why not. Runs nothing. | `bngsim.Model.load(MODEL).describe()` |
| `bngsim run MODEL --t-end T [...]` | one run from the model's initial state: its record and a digest of its values, the arrays to `--out` | `bngsim.Simulator(bngsim.Model.load(MODEL), ...).run(...)`, `Result.record`, `Result.digest()` |
| `bngsim check MODEL --t-end T [...]` | the run, then the run made again with one setting changed; exits 6 when the check is not ok | the same, then `Result.check()` |
| `bngsim convert TOOL ...` | the converters: `sbml2net`, `net2sbml`, `sbml2bngl`, `validate`, `omex` | `bngsim-sbml2net`, `bngsim-net2sbml`, `bngsim-sbml2bngl`, `bngsim-validate-conversion`, `bngsim-omex` |
| `bngsim cache ...` | the codegen artifact cache | `bngsim-cache` |

Run `bngsim doctor` first in a new environment, and paste its output into a bug
report.

`bngsim inspect` on a `.bngl` file runs BNG2.pl to generate the network, as
`Model.load` does, and describes that network.

## Running a model

`bngsim run` makes one run and prints what it did, never the arrays:

```bash
bngsim run model.xml --t-end 100 --n-points 101 --json
bngsim run model.xml --sample-times 0,1,5,10 --set kf=2 --set A=50 --out run.npz
bngsim run model.net --method ssa --t-end 10 --replicates 20 --seed 1
bngsim check model.xml --t-end 100 --checks all --json
```

- **What it prints:** with `--json`, `{"model", "record", "digest", "out"}`. `record` is `Result.record` without the start vector, `digest` is `Result.digest()` (`--species` and `--max-columns` pass through), and `out` is the absolute path `--out` wrote, or `null`. `check` adds `"check"`, the `Result.check()` report. Without `--json`, the digest's text form, and for `check` the report's.
- **The run:** `--method` (`ode`, `ssa`, `psa`, `nf` or `rm`, the last two with `--xml`), `--t-start`, `--t-end` and `--n-points`, or `--sample-times`. `--seed`, `--rtol` and `--atol` (a number, `auto` or `tracking`). `--steady-state` stops at a steady state.
- **The model:** `--set NAME=VALUE` sets a parameter, or a species' initial amount, before the run; it is repeatable, and a name the model does not have is a usage error that names the close ones. Each command loads the model fresh, so each run starts from its initial state.
- **Replicates:** `bngsim run --replicates N` makes N stochastic runs (`run_replicates`) and digests them as a stack: statistics across runs at the last time. It takes no `--sample-times`, tolerances or `--steady-state`, as `run_replicates` takes none.
- **The arrays:** `--out FILE` writes them: `.npz` with NumPy alone, `.h5` with h5py (which `Result.load` reads back), or `.csv`/`.tsv` (one kind of column, `--out-kind observables` or `species`; one run only).
- **A time limit, always:** every run has a wall-clock limit, `--timeout` seconds, 600 by default; past it, exit 5. `check`'s reruns share `--budget` seconds (600 by default); a check that does not fit is reported as `timeout`, and the check is not ok.
- **A `.bngl` file** can mean one run of the network BNG2.pl generates, or the file's own actions. `bngsim run` makes the first, and asks for `--network` to say so; the second is `bngsim.run_bngl(path)` in Python.
- **Not commands yet:** `run_batch`, `parameter_scan`, `run_until` and the `steady_state` family. Use Python for those.

## Output a program can read

- With `--json`, a command prints exactly one JSON object on standard output,
  and nothing else goes there. Anything else written to standard output while
  the command works, by Python or by compiled code, goes to standard error. A
  number JSON cannot hold (NaN, an infinity) is written as the string `"nan"`,
  `"inf"` or `"-inf"`; a value JSON has no form for (a path) is written as its
  text. On Windows, compiled code that writes through a C runtime other than
  Python's is not redirected; that is not tested there.
- `--json` can come before or after the command: `bngsim --json doctor` and
  `bngsim doctor --json` are the same.
- An error prints nothing on standard output. With `--json`, it is one JSON
  object on standard error, the same as `BngsimError.to_dict()`:

  ```json
  {
    "type": "MissingFileError",
    "code": "file.not_found",
    "message": "SBML file not found: /data/model.xml",
    "details": {"path": "/data/model.xml"},
    "remedies": [],
    "docs_url": "https://bngsim.readthedocs.io/en/latest/reference/error-codes.html#code-file-not-found",
    "notices": []
  }
  ```

  `notices` holds what a run that failed had noted before it raised. The codes
  are listed in [Error and notice codes](error-codes.md). A command line the
  command cannot take is a `UsageError` with the code `usage.invalid_argument`,
  and its `details["usage"]` is the usage line. Without `--json`, an error is a
  line of text on standard error, after the traceback when the error is not one
  BNGsim raises on purpose.

`bngsim convert` and `bngsim cache` hand their arguments to the existing
commands, which keep their own output and exit codes; their help and usage
name them as `bngsim convert TOOL` and `bngsim cache`. Give their options after
`convert TOOL` or `cache`. A wrong tool name, or an option
before `convert` or `cache`, is a usage error like any other (JSON with
`--json`); an error the tool itself reports is its own text.

## Exit codes

The exit code says what happened, so a caller can tell a refusal from a crash
without reading any text.

| code | meaning |
|---|---|
| 0 | success |
| 1 | error: the command was well formed and failed (a file that does not exist, a model that cannot be read, a package that is missing) |
| 2 | usage error: the command line is wrong (an unknown command or option, a missing or invalid argument) |
| 3 | refusal: the model holds a construct BNGsim cannot simulate faithfully. The model, not the command, has to change. |
| 4 | failed run: a run started and the solver gave up |
| 5 | timeout: a run took longer than its wall-clock limit |
| 6 | check not ok: `bngsim check` finished, and the run's answer moved when a setting changed, a changed run raised or ran out of time, or the run is not converged at its own tolerance. The report is on standard output, as for success. |
