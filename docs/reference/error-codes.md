---
summary: "The code every BNGsim error carries, what each means and what to do about it, and the remedies a program can act on."
# Generated from python/bngsim/_error_codes.py by scripts/error_codes_page.py;
# edit that module and run the script.
---

# Error codes

Every BNGsim error carries a `code`, a string that stays the same across releases where the message may change. A program should branch on the code, not on the message.

```python
try:
    model.set_param("kff", 1.0)
except bngsim.BngsimError as e:
    e.code       # 'param.not_found'
    e.details    # {'name': 'kff', 'close_matches': ['kf'], 'is_a': None}
    e.remedies   # [{'kind': 'use_name', 'name': 'kf'}]
    e.to_dict()  # type, code, message, details, remedies and docs_url
```

- `details` says what the error is about: the name, the path, the constructs.
- `remedies` lists fixes a program can act on, each a dict with a `kind`. It is empty where BNGsim can name no fix, as for a model it cannot simulate as written.
- `docs_url` points at the code's entry below.

An error with no code of its own carries its class's generic code, such as `model.error` or `run.failed`. An error of a generic class raised from a BNGsim error with a specific code, as a loader wraps its parser's, carries that error's code, details and remedies.

An error of use is a BNGsim class that is also the built-in it replaces, so an `except` written for the built-in still catches it: `UsageError` is a `ValueError`, `DependencyError` an `ImportError`, and `MissingFileError` a `FileNotFoundError`.

## Codes

(code-error)=
### `error`

Raised as `BngsimError`. An error with no more specific code.

Read the message.

(code-model-error)=
### `model.error`

Raised as `ModelError`. A model could not be loaded or used, and the error has no more specific code.

Read the message: it names the file, the construct or the call.

(code-run-failed)=
### `run.failed`

Raised as `SimulationError`. A run failed, and the error has no more specific code.

Read the message: it names what failed and where.

(code-param-error)=
### `param.error`

Raised as `ParameterError`. A parameter could not be read or set, and the error has no more specific code.

Read the message.

(code-conversion-error)=
### `conversion.error`

Raised as `ConversionError`. A conversion between formats cannot be done without changing the model.

Read the message: it names what the target format cannot hold.

(code-coupling-conservation)=
### `coupling.conservation`

Raised as `ConservationError`. A conserved total drifted past its tolerance across a coupling boundary.

Shorten the exchange interval or tighten the tolerances.

(code-usage-invalid-argument)=
### `usage.invalid_argument`

Raised as `UsageError`. An argument to a call is not one it can take.

Read the message: it names the argument and what it takes.

(code-file-not-found)=
### `file.not_found`

Raised as `MissingFileError`. The file named does not exist.

Check the path. ``details['path']`` is the one BNGsim looked for.

(code-model-unknown-format)=
### `model.unknown_format`

Raised as `ModelError`. ``Model.load`` cannot tell a file's format from its suffix.

Rename the file to a suffix in ``details['expected']``, or call the loader for its format, such as ``Model.from_sbml``.

(code-model-unsupported-construct)=
### `model.unsupported_construct`

Raised as `ModelError`. The model holds a construct BNGsim cannot simulate correctly, such as an algebraic rule or a delay.

The model cannot be simulated as written. ``details['constructs']`` lists each construct and where it is. Change the model, or report the refusal.

(code-model-unset-symbol)=
### `model.unset_symbol`

Raised as `UnderSpecifiedModelError`. The model reads a symbol it never gives a value.

Give the symbol a value in the model, or report the refusal.

(code-param-not-found)=
### `param.not_found`

Raised as `ParameterError`. The model has no parameter by that name.

Use a name from ``Model.param_names``. ``details['close_matches']`` lists near ones, and ``details['is_a']`` says what the name is when it names a species, an observable or a function instead.

(code-species-not-found)=
### `species.not_found`

Raised as `ModelError`. The model has no species by that name.

Use a name from ``Model.species_names``. ``details['close_matches']`` lists near ones.

(code-usage-missing-argument)=
### `usage.missing_argument`

Raised as `UsageError`. A call needs an argument it was not given.

Pass the argument ``details['argument']`` names.

(code-ssa-unsupported-construct)=
### `ssa.unsupported_construct`

Raised as `SsaValidationError`. The model holds a construct a stochastic method cannot simulate.

Run it with ``method='ode'``. ``details['issues']`` lists each construct.

(code-codegen-declined)=
### `codegen.declined`

Raised as `CodegenDeclinedError`. ``codegen=True`` on a model codegen declines.

Pass ``codegen=None`` to run it on the interpreter.

(code-solver-sparse-jacobian-unavailable)=
### `solver.sparse_jacobian_unavailable`

Raised as `SparseJacobianUnavailableError`. ``force_sparse_linear_solver=True`` on a model with no sparse Jacobian.

Leave out ``force_sparse_linear_solver`` to use the dense solver.

(code-sens-unsupported)=
### `sens.unsupported`

Raised as `SensitivityUnsupportedError`. BNGsim cannot compute a requested sensitivity correctly for this model.

Drop the sensitivity, change the model, or use a derivative-free method.

(code-sens-refused-during-run)=
### `sens.refused_during_run`

Raised as `SensitivityRefusedDuringRunError`. The solver reached, partway through a run, a construct whose derivative BNGsim cannot compute.

Read the message: it names the construct and what to change.

(code-sens-refused-for-parameters)=
### `sens.refused_for_parameters`

Raised as `SensitivityRefusedForParametersError`. The solver refused a sensitivity partway through a run, for the parameters requested.

Drop the parameters the message names; a smaller set may run.

(code-env-missing-dependency)=
### `env.missing_dependency`

Raised as `DependencyError`. A call needs an optional package that is not installed.

Install it. The remedy gives the ``pip install`` command.

(code-env-broken-dependency)=
### `env.broken_dependency`

Raised as `DependencyError`. A package a call needs is installed but cannot be imported.

Reinstall it for this Python. The message gives the import error.

(code-env-no-compiler)=
### `env.no_compiler`

Raised as `CodegenToolchainError`. A call needs a C compiler, and there is none.

Install a C compiler, or set ``CC`` to one.

(code-env-broken-toolchain)=
### `env.broken_toolchain`

Raised as `CodegenToolchainError`. The C compiler is there, but it cannot build a library for any model.

Fix the toolchain. The message names the compiler, the linker and the SDK.

(code-solver-too-much-work)=
### `solver.too_much_work`

Raised as `SimulationError`. The ODE solver took ``max_steps`` steps without reaching the next output time.

Read the message. Steps that still advance are retried, so this one did not.

(code-solver-too-much-accuracy)=
### `solver.too_much_accuracy`

Raised as `SimulationError`. The tolerances asked for more accuracy than double precision holds.

Loosen ``rtol`` or ``atol``.

(code-solver-stalled)=
### `solver.stalled`

Raised as `SimulationError`. The ODE solver's step size collapsed, so a whole batch of steps advanced nothing, typically at a jump in a rate law.

Loosen ``rtol`` or ``atol``, or move the jump onto an event. A larger ``max_steps`` does not help.

(code-solver-error-test-failure)=
### `solver.error_test_failure`

Raised as `SimulationError`. The ODE solver failed its error test repeatedly at one time, often at a jump or a singularity.

Read the message: it names the time and, where it can, the species and the rate law. Check the model there.

(code-solver-convergence-failure)=
### `solver.convergence_failure`

Raised as `SimulationError`. The ODE solver's Newton iteration failed to converge repeatedly at one time.

Read the message: it names the time and, where it can, the species and the rate law. Check the model there.

(code-solver-failed)=
### `solver.failed`

Raised as `SimulationError`. The ODE solver failed in another way. ``details['flag']`` is SUNDIALS' name for it.

Read the message.

(code-run-timeout)=
### `run.timeout`

Raised as `SimulationTimeout`. A run took longer than its wall-clock budget.

Pass a larger ``timeout``. ``details`` has the budget and the time taken.

(code-run-stop-condition)=
### `run.stop_condition`

Raised as `StopConditionMet`. A stop condition was met. The error carries the result up to that point.

Read ``e.result``. This is how a stop condition reports, not a failure.

## Remedies

| kind | keys | meaning |
|---|---|---|
| `install_package` | `package`, `command` | Install a package. ``command`` is the command to run. |
| `install_compiler` | none | Install a C compiler, or set ``CC`` to one. |
| `pass_argument` | `argument` | Call again with this argument. ``value`` is present where one value is the fix. |
| `use_name` | `name` | Use this name instead. |
| `use_call` | `call`, `name` | Use this call instead, as for a species named where a parameter was asked for. |
| `use_method` | `method` | Run with this method instead. |
| `loosen_tolerance` | none | Loosen ``rtol`` or ``atol``. |
