---
summary: "What stays the same from one BNGsim release to the next while it is at 0.x: call names, error and notice codes, the versioned JSON forms (Result.record, Result.check, Result.digest and the steady-state forms), capabilities() keys, the command line's JSON and exit codes, and saved result files; and how a change to any of them is announced."
---

# Compatibility between releases

BNGsim is at version 0.x, so it makes no promise about everything. It does
make the promises on this page, about what a program or an agent reads from
it. Anything not listed here can change in any release. The
[changelog](https://github.com/lanl/bngsim/blob/HEAD/CHANGELOG.md) says when
something does.

## Calls

The calls in `bngsim.__all__`, and the documented methods, properties and
keywords of `Model`, `Simulator` and `Result`, keep their names.

A call or keyword that is to be renamed or removed is first deprecated. It
keeps working, warns `DeprecationWarning`, and is listed under "Deprecated" in
the changelog, for at least one minor release. When it is removed, it is listed
under "Removed", and the old name raises an error that names the call to use.
No public call has been removed so far.

An option that a bundled engine deprecates, such as RuleMonkey's
`block_same_complex_binding=False`, is listed under "Deprecated" too, but the
warning is the engine's own: RuleMonkey prints one line to standard error.

## Error and notice codes

Every error carries a `code`, and every run notice on `result.notices` carries
one too. A code stays the same across releases, where the message may change,
and no code is both an error's and a notice's. Branch on the code, not on the
message. The codes are listed in [Error and notice codes](error-codes.md).

## The JSON forms

Each of these is a plain dict, ready for `json.dumps`, and carries a version
number of its own:

| form | version key |
|---|---|
| `Result.record` | `schema_version` |
| `Result.check()` | `check_version` |
| `Result.digest()` | `digest_version` |
| `Model.describe()`, and `bngsim inspect --json` | `schema_version` (its own, not the record's) |
| the steady-state forms of the command line's `--json` | `form_version` |

The same rule holds for each:

- **A key added is no change.** A reader should ignore keys it does not know.
- **A key removed or renamed, or a change to what a key holds, raises the
  version.** The changelog says what changed. So a reader that checks the
  version it was written for can tell a form it may misread.

Tests pin the top-level keys of `check()` and `digest()`, and the keys every
check in a report has.

## `capabilities()`

`bngsim.capabilities()["features"]` holds the same keys in every build and on
every platform. A feature an install lacks is `False`, with the reason in
`["missing"]`. A key, once added, stays. A feature BNGsim no longer has reads
`False`.

## The command line

With `--json`, each command except `bngsim convert` and `bngsim cache`, which
keep their tools' own output, prints one JSON object on standard output. Its
top-level keys have no version of their own. A key may be added in any
release. One is removed or renamed only with an entry in the changelog. The
`record`, `digest`, `check` and steady-state forms inside carry their own
versions. An error is one JSON object on standard error,
`BngsimError.to_dict()`, with its `code`. The exit codes of those commands
keep their meanings. See [The bngsim command](command-line.md).

## Saved results

`Result.load` reads an HDF5 file that `Result.save` wrote in an earlier
release. A file from a release before volumes were recorded loads as one that
does not say what the compartment sizes were. A file written by a later
release is not promised to load in an earlier one.

`Result.save` also writes `.npz`, which `Result.load` does not read: read it
with `numpy.load`. It holds plain arrays and the run's `record` as a JSON
string, so read its record's `schema_version`.
