---
summary: "The code every BNGsim error and run notice carries, what each means and what to do about it, and the remedies a program can act on."
# Generated from python/bngsim/_error_codes.py by scripts/error_codes_page.py;
# edit that module and run the script.
---

# Error and notice codes

Every BNGsim error carries a `code`, a string that stays the same across releases where the message may change. A program should branch on the code, not on the message.

```python
try:
    model.set_param("kff", 1.0)
except bngsim.BngsimError as e:
    e.code       # 'param.not_found'
    e.details    # {'name': 'kff', 'close_matches': ['kf'], 'is_a': None}
    e.remedies   # [{'kind': 'use_name', 'name': 'kf'}]
    e.to_dict()  # type, code, message, details, remedies, docs_url, notices
```

- `details` says what the error is about: the name, the path, the constructs.
- `remedies` lists fixes a program can act on, each a dict with a `kind`. It is empty where BNGsim can name no fix, as for a model it cannot simulate as written.
- `docs_url` points at the code's entry below.
- `notices` lists what the failed run had noted before it raised, such as a start from the last run's end state, each as on a Result; see {ref}`Notice codes <notice-codes>`.

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

(code-model-invalid)=
### `model.invalid`

Raised as `InvalidModelError`. A model, or a part of one given to a builder, is not well formed: a name defined twice, a reaction with no rate, a reference to something not defined.

Fix the model as the message says.

(code-model-unreadable-file)=
### `model.unreadable_file`

Raised as `ModelError`. A model file is not what its format says, or its reader stops on a fatal error.

Check the file is the format its name says; the message quotes the reader.

(code-param-not-found)=
### `param.not_found`

Raised as `ParameterError` or `UsageError`. The model has no parameter by that name.

Use a name from ``Model.param_names``. ``details['close_matches']`` lists near ones, and ``details['is_a']`` says what the name is when it names a species, an observable or a function instead.

(code-species-not-found)=
### `species.not_found`

Raised as `ModelError`, `NotFoundError` or `UsageError`. The model has no species by that name.

Use a name from ``Model.species_names``. ``details['close_matches']`` lists near ones.

(code-output-not-found)=
### `output.not_found`

Raised as `NotFoundError` or `UsageError`. A name that is none of the result's species, observables or functions.

Use a name `details["close_matches"]` gives, or one of the result's `species_names` or `observable_names`.

(code-name-not-found)=
### `name.not_found`

Raised as `NotFoundError` or `UsageError`. A name or key a lookup does not find, of a kind with no code of its own.

Read the message: it lists the names there are.

(code-name-ambiguous)=
### `name.ambiguous`

Raised as `NotFoundError`. A name matches more than one entry, so a lookup by it cannot choose.

Index by position, as the message says.

(code-usage-missing-argument)=
### `usage.missing_argument`

Raised as `UsageError`. A call needs an argument it was not given.

Pass the argument ``details['argument']`` names.

(code-usage-conflicting-arguments)=
### `usage.conflicting_arguments`

Raised as `UsageError`. Two arguments a call takes one or the other of, or that cannot go together, were both given.

Read the message for the pair, and drop one.

(code-usage-wrong-type)=
### `usage.wrong_type`

Raised as `ArgumentTypeError`. An argument is of a type the call does not take.

Read the message: it names the type the call takes.

(code-usage-unknown-method)=
### `usage.unknown_method`

Raised as `UsageError`. A `method=` that BNGsim does not know.

Pass one the message lists.

(code-usage-method-unsupported)=
### `usage.method_unsupported`

Raised as `UsageError`. A call or option the simulator's method does not offer, such as sensitivities on a stochastic method.

Build the Simulator with the method the remedy names, or leave the option out.

(code-usage-not-supported)=
### `usage.not_supported`

Raised as `NotSupportedError` or `UsageError`. A call or option this object does not offer, as written.

Read the message for what does.

(code-env-feature-not-built)=
### `env.feature_not_built`

Raised as `BngsimError` or `UsageError`. A method needs a part of BNGsim this install was built without, such as NFsim or RuleMonkey.

Install a build that has it; `bngsim.capabilities()` lists what this one has.

(code-ssa-unsupported-construct)=
### `ssa.unsupported_construct`

Raised as `SsaValidationError`. The model holds a construct a stochastic method cannot simulate.

Run it with ``method='ode'``. ``details['issues']`` lists each construct.

(code-codegen-declined)=
### `codegen.declined`

Raised as `CodegenDeclinedError`. ``codegen=True`` on a model codegen declines.

Pass ``codegen=None`` to run it on the interpreter.

(code-codegen-build-failed)=
### `codegen.build_failed`

Raised as `CodegenBuildError`. Building this model's compiled right-hand side failed or ran out of time; the toolchain builds other code.

Pass `codegen=False` to run on the interpreted right-hand side, or give the build more time or cores as the message says.

(code-solver-sparse-jacobian-unavailable)=
### `solver.sparse_jacobian_unavailable`

Raised as `SparseJacobianUnavailableError`. ``force_sparse_linear_solver=True`` on a model with no sparse Jacobian.

Leave out ``force_sparse_linear_solver`` to use the dense solver.

(code-sens-unsupported)=
### `sens.unsupported`

Raised as `SensitivityUnsupportedError`. BNGsim cannot compute a requested sensitivity correctly for this model.

Drop the sensitivity, change the model, or use a derivative-free method.

(code-sens-not-computed)=
### `sens.not_computed`

Raised as `UsageError`. A call reads sensitivities, and the result has none: its run was not asked for them.

Run again with `sensitivity_params=` on the Simulator, or with `compute_all_sensitivities()`.

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

(notice-codes)=
## Notice codes

A run's warnings and fallbacks are kept on `result.notices`, and on the error of a run that fails, each a dict with a `category`, a `code` and a `message`. A notice's code is stable across releases as an error's is, and no code is both. An error's code names the decision a caller makes; a notice's names the part of the run that noted it, so a compiler a run could not use is the notice `codegen.toolchain`, and the error `env.no_compiler` or `env.broken_toolchain` where a call cannot go on without one.

| code | meaning | what to do |
|---|---|---|
| `atol.below_scale` | Species stayed below their absolute tolerance throughout while reaching a share of the largest; their error was not controlled. | Pass a smaller `atol`, or one per species, for the species the message names, and run again. |
| `codegen.cache_key_unavailable` | The structural codegen cache key could not be made; the generated source was hashed instead. | Nothing to fix: the cache still works, keyed on the generated source. |
| `codegen.failed` | Building the compiled right-hand side failed; the run used the interpreted one. | The numbers stand; only speed is lost. The message gives the compiler's error. |
| `codegen.skipped` | The compiled right-hand side was not built, because an earlier build of this model failed; the run used the interpreted one. | The numbers stand; only speed is lost. The message gives when and why the build failed; pass `codegen=True`, or raise `BNGSIM_CODEGEN_TIMEOUT` or `BNGSIM_CODEGEN_JOBS`, to try it again. |
| `codegen.ssa_propensities_interpreted` | The compiled SSA propensities could not be built; the run evaluated them interpreted. | The numbers stand; only speed is lost. The message says why. |
| `codegen.toolchain` | The C toolchain could not build a codegen library, or builds one again. | Where the message says the toolchain cannot build, install or repair the C compiler it names, as for the errors `env.no_compiler` and `env.broken_toolchain`; until then a run that can go on without a compiled library does. Where it says codegen resumes, nothing is left to fix. |
| `config.ignored_setting` | An environment setting had a value bngsim cannot read; a default was used. | Fix or unset the environment variable the message names. |
| `equality.detection_failed` | Finding the rate-law equalities a state can leave failed; the run watches fewer of them. | Check the result where the model's rate laws switch, at a tighter `rtol`. |
| `jacobian.difference_quotient` | The analytical Jacobian is not available for this model; the run used a difference quotient. | Nothing to fix: the numbers stand, and a large model runs slower. The message says why the analytical Jacobian was not made. |
| `jacobian.difference_quotient_kept` | An earlier run on this Simulator failed with the analytical Jacobian and was retried; this run used the difference quotient from the start. | Nothing to fix: the numbers stand. A new Simulator tries the analytical Jacobian again. |
| `jacobian.difference_quotient_retry` | The run failed with the analytical Jacobian and was run again from its start with a difference quotient, whose result it returned. | Nothing to fix: the result is the retry's and stands. Pass `jacobian='fd'` to skip the analytical attempt. |
| `jax.zero_gradients` | Model functions may give JAX zero gradients at discontinuities. | Nothing to fix: the JAX Jacobian may read zero across a discontinuity, and the solver adapts its step there. |
| `linear_solver.dense_without_klu` | A large model ran on the dense linear solver because this install has no KLU. | The numbers stand. An install with KLU, which `bngsim.capabilities()` reports, runs a large model faster. |
| `protocol.steady_state_not_reached` | A protocol's run asked to stop at a steady state reached none by its end. | Lengthen that run's time window, or do not read its end state as steady. |
| `rulemonkey.unsupported` | RuleMonkey reported a construct it does not support as written. | Read the message for the construct; the model may need another method to run as written. |
| `scan.dose_dropped` | A parameter scan cannot move the species the scanned parameter seeds: each point starts them at the state the scan started from, not at their dose. | Set each seeded species in `on_point`, as the message shows. For a scan that is not a sensitivity scan, `model.reset()` before it doses from the initial conditions instead. |
| `sens.chain_rule_dropped` | A derived parameter's expression could not be differentiated; sensitivities through it read zero along that path. | Read sensitivities through that parameter as missing, not zero; check them by finite differences. |
| `sens.chunk_split` | A group of sensitivity columns failed together and each was solved on its own. | Nothing to fix: the columns stand, at more cost. |
| `sens.detection_failed` | Finding the switch, event or state crossings a sensitivity depends on failed; a gradient may miss the jump at one. | Check gradients across the model's switches and events by finite differences. |
| `sens.nan_rows` | Rows of the sensitivity tensor are NaN or infinite where the chain rule they need is not available. | Leave those rows out: they mark derivatives BNGsim cannot give. |
| `sens.one_sided` | Sensitivity columns are one-sided derivatives at a switch that shares its instant with another. | Read those columns as one-sided derivatives at that instant. |
| `sens.output_budget` | Deriving expression output sensitivities ran past its time budget; the functions not reached refuse when selected. | Raise or disable the budget with `BNGSIM_SENS_DERIV_BUDGET_S`, as the message says, or select only the expression outputs derived in time. |
| `sens.parameters_skipped` | Parameters were left out of a sensitivity request because their columns could not mean what they say. | The message names the parameters left out and why; the result has no sensitivities for them. |
| `sens.rhs_difference_quotient` | The analytic sensitivity right-hand side was declined; CVODES' difference quotient carried the sensitivity columns. | Nothing to fix: the sensitivities stand, computed more slowly. |
| `smooth_pulse.event_reset` | A smooth bump on a variable an event assigns is placed from the variable's start and rate. | Check the pulse's placement where the event assigns the variable. |
| `ssa.negative_count` | A species count went negative in a stochastic run. | Fix the rate laws that let a count go negative; the trajectory past it is not a stochastic process. |
| `ssa.reverse_fire` | A reaction fired in reverse because its rate law evaluated negative. | Nothing to fix if the negative rate is meant: the reaction runs in reverse so the mean tracks the ODE. Otherwise fix the rate law so it cannot go negative. |
| `ssa.rounded_counts` | Molecule counts that were not whole numbers were rounded at the start. | Give whole-number initial counts where the rounding matters. |
| `start.continues_last_run` | The run started from the state the last run left, with the clock set back. | Call `model.reset()` before the run to start from the initial conditions. |
| `start.rows_from_initial` | The rows started from the initial conditions, not the model's current state, which differs from them. | Call `model.reset()` first where the initial conditions are the start you want; the message says how to start from the current state instead. |
| `stop_condition.non_finite` | A stop condition read NaN or infinity at a sample. | Check the condition's expression for a division by zero or an overflow. |
| `stops.detection_failed` | Finding where a rate law switches on the clock, a crossing or the state failed; the integrator approaches it without a stop or a root. | Check the result where the model's rate laws switch, at a tighter `rtol`. |
| `switch.noisy_trigger` | An event trigger crosses its threshold within the rounding of its own evaluation. | Move the threshold off the extremum, or write the trigger so it does not cancel there. |
| `switch.over_budget` | A condition crosses more often than the stops one run places. | Shorten the window; past the stops placed, the integrator approaches each crossing without one. |
| `switch.scan_failed` | The scan of a time condition failed; no stop is placed for it. | Bound the step with `max_step` so the switch is not stepped over, and report the failure: it is a defect in BNGsim. |
| `switch.unread` | A condition on the clock could not be read where the scan asked. | Check the result where that condition switches, at a tighter `rtol`. |

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
