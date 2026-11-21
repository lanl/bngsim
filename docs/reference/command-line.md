---
summary: "The bngsim command (also python -m bngsim): doctor, guide, examples, inspect, convert and cache; one JSON object on standard output with --json, errors as JSON on standard error, and an exit code for each outcome."
---

# The `bngsim` command

`bngsim` inspects a model and this install of BNGsim from a shell. It is also
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
| `bngsim convert TOOL ...` | the converters: `sbml2net`, `net2sbml`, `sbml2bngl`, `validate`, `omex` | `bngsim-sbml2net`, `bngsim-net2sbml`, `bngsim-sbml2bngl`, `bngsim-validate-conversion`, `bngsim-omex` |
| `bngsim cache ...` | the codegen artifact cache | `bngsim-cache` |

Run `bngsim doctor` first in a new environment, and paste its output into a bug
report.

`bngsim inspect` on a `.bngl` file runs BNG2.pl to generate the network, as
`Model.load` does, and describes that network.

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
    "docs_url": "https://bngsim.readthedocs.io/en/latest/reference/error-codes.html#code-file-not-found"
  }
  ```

  The codes are listed in [Error codes](error-codes.md). A command line the
  command cannot take is a `UsageError` with the code `usage.invalid_argument`,
  and its `details["usage"]` is the usage line. Without `--json`, an error is a
  line of text on standard error, after the traceback when the error is not one
  BNGsim raises on purpose.

`bngsim convert` and `bngsim cache` hand their arguments to the existing
commands, which keep their own output, exit codes and help, so their help
names them by their own names (`bngsim-sbml2net`, `bngsim-cache`). Give their
options after `convert TOOL` or `cache`. A wrong tool name, or an option
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

Codes 4 and 5 are for the commands that run a model, which are still to come
(issue #1708).
