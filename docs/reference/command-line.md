---
summary: "The bngsim command (also python -m bngsim): doctor, guide, examples, inspect, run, check, run-until, run-batch, parameter-scan, the steady-state calls, run-bngl, convert and cache; one JSON object on standard output with --json, errors as JSON on standard error, and an exit code for each outcome."
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
| `bngsim run-until MODEL --t-end T --stop-when EXPR` | a run to `T`, cut at the first output point where a condition holds: a condition met exits 0 with the run it cut | `Simulator.add_stop_condition(EXPR)`, `Simulator.run_until(T)` |
| `bngsim run-batch MODEL --params ROWS --t-end T` | a run per parameter set, digested as a stack | `Simulator.run_batch(params=ROWS)` |
| `bngsim parameter-scan MODEL --parameter NAME ...` | a run per value of one parameter, digested as a stack | `Simulator.parameter_scan(NAME, ...)` |
| `bngsim steady-state MODEL` | the steady state the model reaches: the solver's verdict and the values there | `Simulator.steady_state()` |
| `bngsim steady-state-batch MODEL --params ROWS` | a steady state per parameter set: how many converged, statistics of the rest | `Simulator.steady_state_batch(ROWS)` |
| `bngsim steady-states MODEL` | the model's distinct steady states from many starts, with each one's stability | `Simulator.steady_states()` |
| `bngsim run-bngl FILE.bngl` | the file's own actions, as BNG2.pl runs them: a digest of each run | `bngsim.run_bngl(FILE)` |
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
bngsim check model.xml --t-end 100 --sensitivity-params kf,kr --json
```

- **What it prints:** with `--json`, `{"model", "record", "digest", "out", "set"}`, and `"seeds"` after `--replicates`. `record` is `Result.record` without the start vector, `digest` is `Result.digest()` (`--species` and `--max-columns` pass through; for a run with sensitivities, `Result.digest(sensitivities=True)`), `out` is the absolute path `--out` wrote, or `null`, `set` is what `--set` set, and `seeds` each replicate's seed. `check` adds `"check"`, the `Result.check()` report. Without `--json`, the digest's text form, and for `check` the report's.
- **The run:** `--method` (`ode`, `ssa`, `psa` with `--poplevel N`, `nf` or `rm` with `--xml FILE`), `--t-start`, `--t-end` and `--n-points`, or `--sample-times` (increasing; in place of the other three). `--seed`; `--rtol` and `--atol` (a number, `auto` or `tracking`) for an ODE run. `--steady-state` stops at a steady state.
- **Sensitivities:** `--sensitivity-params NAME,...` and `--sensitivity-ic SPECIES,...` make an ODE run compute forward sensitivities: they are `Simulator`'s `sensitivity_params` and `sensitivity_ic`. Each is repeatable, and the names add up; a comma inside parentheses is part of a name, so `--sensitivity-ic "R(l,d)"` names one species. The digest then ranks, for each column it lists, the parameters and initial amounts that move it most at the last time, five of each, and the record lists them as `sensitivity_params` and `sensitivity_ic`. A name the model does not have is a usage error, with its close matches. A species given as a parameter, a parameter as a species, and an observable of one species are pointed at the option that takes them, by the model's own spelling (`'X' is the species 'X()'`). As in Python, a sensitivity run needs a C compiler. `run` and `check` take these options; the other run commands do not.
- **The model:** `--set NAME=VALUE` sets a parameter (`Model.set_param`) or a species' starting concentration (`Model.set_concentration`: the amount over the compartment's size, which for a species declared in amounts in a compartment of size 2 is half its amount). It is repeatable, the value must be a finite number, and a name the model does not have, or a parameter defined by an expression, is a usage error. A species set this way becomes the model's initial state, so the run starts from it, and the record says `started_from: "initial"`.
- **An option the run would not use is refused, not dropped:** `--rtol` or `--atol` with a stochastic method, `--seed` with `ode`, `--xml` with a method other than `nf` or `rm`, `--poplevel` with one other than `psa`, `--network` with a file that is not `.bngl`, `--out-kind` with an `--out` that is not `.csv` or `.tsv`, `--sample-times` beside `--t-start`, `--t-end` or `--n-points`, `--sensitivity-params` or `--sensitivity-ic` with a stochastic method, a name given twice to them, and a `.csv` or `.tsv` `--out` beside either (it holds no sensitivities; `.npz` and `.h5` do) are usage errors (exit 2).
- **The checks:** `bngsim check --checks` takes `all` or names separated by commas: `tolerance`, `grid`, `jacobian`, `linear_solver`, `backend` and `second_engine` for an ODE run, `replicates` for a stochastic one, and `sensitivity` and `sensitivity_fd` for a run with sensitivities (`bngsim check --help` lists them). Left out, it runs `tolerance` and `grid` for an ODE run, `replicates` for a stochastic one, and `sensitivity` too for a run with sensitivities, as `Result.check()` does. `all` adds both sensitivity checks to a run with sensitivities.
- **Replicates:** `bngsim run --replicates N` makes N stochastic runs (`run_replicates`) and digests them as a stack: statistics across runs at the last time. `--replicates 1` is one run. It takes no `--sample-times` or `--steady-state`, as `run_replicates` takes none.
- **The arrays:** `--out FILE` writes them, replacing any file of that name: `.npz` with NumPy alone, `.h5` or `.hdf5` with h5py (which `Result.load` reads back), or `.csv`/`.tsv` (one kind of column, `--out-kind observables` or `species`; one run only). Any other name, a directory, a missing directory, or `.h5` without h5py is refused before the run.
- **A time limit, always:** every run has a wall-clock limit, `--timeout` seconds, 600 by default; past it, exit 5. `--replicates N` shares it: each of the N runs may take `--timeout`/N, and the record's `timeout` says so. Loading the model is not counted (a `.bngl` file's BNG2.pl has its own limit). `check`'s reruns share `--budget` seconds (600 by default), on top of the run's; a check that does not fit is reported as `timeout`, and the check is not ok.
- **A `.bngl` file** can mean the network BNG2.pl generates, or the file's own actions. `bngsim run` and every other command that runs a model read the first, and ask for `--network` to say so; `bngsim run-bngl` runs the second.

## The other run calls

Each is named for its Python call, with `-` for `_` (`bngsim run_batch` is answered with the right name), and takes the options of `bngsim run` that its call can use, spelled as the call's keywords are: any other option is refused, and named, with exit 2. `--set`, `--json`, `--max-columns` and the exit codes are `run`'s.

```bash
bngsim run-until model.xml --t-end 1000 --stop-when "A_tot < 10" --json
bngsim run-batch model.xml --t-end 100 --params '[{"kf": 1}, {"kf": 2}]' --out rows.npz
bngsim parameter-scan model.xml --t-end 100 --parameter kf --par-min 0.1 --par-max 10 --n-scan-pts 9 --log-scale
bngsim steady-state model.xml --method newton --species --json
bngsim steady-state-batch model.xml --params rows.csv --json
bngsim steady-states model.xml --n-starts 200 --seed 1 --json
bngsim run-bngl model.bngl --out runs/ --json
```

- **`run-until`** registers each `--stop-when EXPR` (repeatable, checked in order; the expression reads the observables, a species as its name with `(` and `)` written `_`, and `t`) and runs from the model's initial state to `--t-end`. `run_until` integrates the whole span, then cuts the run at the first output point where a condition holds, so `--n-points` sets how finely the cut falls, and the whole span counts against `--timeout`. A condition met is the answer asked for, not an error: the command exits 0 and prints the cut run, as the `StopConditionMet` carries it, with `"stop_condition"` naming it (`null` when none held). It takes `--n-points` (default one per unit of time), `--seed`, `--rtol` and `--atol`, as `run_until` does, and no `--t-start`, `--sample-times` or `--steady-state`. Its `--method` is `ode`, `ssa` or `psa`, as for `run-batch` and `parameter-scan`: those calls run no network-free engine.
- **`run-batch`** runs once per row of `--params`, each from the initial state, and **`parameter-scan`** once per value of `--parameter` (`--par-scan-vals 1,2,5`, or all of `--par-min`, `--par-max` and `--n-scan-pts`, with `--log-scale` to space them geometrically; not both; `--no-reset-conc` starts each value where the last ended). Both print what `run --replicates` prints: the record and digest of the rows stacked (`Result.squeeze`), with `"seeds"` for a stochastic method, and for a scan `"parameter"` and `"values"`. `--out` takes `.npz` or `.h5`, one file for every row. They take no `--sample-times` or `--steady-state`: rows a steady state cuts at their own times do not stack, and `steady-state-batch` is the steady state per parameter set. A species `--set` before a scan is the state each value starts from, so a parameter that seeds another species still doses it; a scan of a parameter that seeds a species the `--set` holds is refused, since no value would move it.
- **`--params ROWS`** is JSON, a list of `{name: value}` objects, or a `.json`, `.csv` or `.tsv` file: a header of names, then a row of numbers per set. A name the model does not have is a usage error, as in `--set`, and so is a name given twice in one row. A species `--set` becomes the initial state every row starts from, so it is refused beside a row whose parameters set a species' starting value: that row's dose would be dropped.
- **`steady-state`** prints `{"model", "call", "parameters", "result", "out", "set"}`: `call` is the call's name and the keywords given, `parameters` the model's at the call, and `result` the solver's verdict (`converged`, `criterion`, `method_used`, `residual`, `newton_distance`, `continued_run_distance`, `root_stability`, the largest real part of the eigenvalues, the steps) and the values at the state returned: the observables, and with `--species` the species, the largest `--max-columns` of each (default 20), listed in the model's order. It takes `--method integration` or `newton` (the call's `method`, not the simulation method: a steady state is of the ODEs), `--criterion`, `--tol`, `--max-time`, `--max-steps`, `--rtol` and `--atol`. **It exits 0 whether or not the solve converged**: read `converged`.
- **`steady-state-batch`** solves once per row of `--params` (`--n-workers` threads), and `result` gives how many rows converged, the first unconverged rows, and per column statistics across the rows that converged. **`steady-states`** searches from `--n-starts` starts (`--seed`, `--upper`, `--decades`, `--dedupe-rtol`, `--tol`, `--n-workers`), and `result` gives each root's stability, multiplicity and residual, and the value at every root of the species that differ most between them.
- **`--out`** of the steady-state commands is a `.npz` file: `species_names`, `concentrations` (a row per parameter set for a batch, with the sets as `param_names` and `params`) or `states` (a row per root), `observable_names` and `observables`, and the rest of each result's arrays.
- **`run-bngl FILE.bngl`** runs the file's actions as `bngsim.run_bngl` does (`--seed`, `--rtol`, `--atol` a number, `--allow-lossy`; `--seed` is refused for a file with no stochastic run, and `--rtol` and `--atol` for one with no ODE run; an action it cannot run as BNG2.pl would is refused before anything runs, exit 3, and an error an action meets while it runs, such as a species the model lacks, exits 1), and prints `{"model", "runs", "out"}`: each run's `index`, `kind`, `label` (its suffix, or `null`), `method` and `digest`, a scan's `"values"` with a digest of its stack, and a bifurcation's sweep back as `"backward"`. `--out DIR` writes each run's arrays to `DIR/N.npz`, or `N-SUFFIX.npz` for a run with a suffix.
- **A time limit, always:** `run-batch` and `parameter-scan` share `--timeout` among their rows, as `--replicates` does: each row may take `--timeout` over the number of rows, and the record's `timeout` says so. `run-until`, the steady-state commands and `run-bngl`'s actions take no limit of their own in Python, so the command keeps it: past `--timeout` (600 s by default) it reports a `run.timeout` error, exit 5, and the process ends there, with the call stopped where it was. That holds for a program that calls `bngsim._cli.main()` itself, too: the process ends, and no `atexit` handler runs. Loading the model is not counted, and BNG2.pl, for `run-bngl`, has its own limit.

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

  `notices` holds what a run that failed had noted before it raised. With
  `--json`, whatever else was written to standard error while the command
  worked (Python's warnings and log, the solver's own lines) is held, and
  written to standard error after a success, or put in the error object as
  `log` after a failure, so the error is the only thing there. An interrupt
  or an exit writes it out before the process stops; a process killed from
  outside, or one that crashes in compiled code, loses it. The codes
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
| 6 | check not ok: `bngsim check` finished, and the run's answer moved when a setting changed, a changed run raised or ran out of time, the run is not converged at its own tolerance, or no check asked for applies to the run. The report is on standard output, as for success. |
