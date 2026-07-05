# API reference



## `bngsim.Model`

Factory methods:
- **`Model.from_net(path)`** — Load from BNG `.net` file
- **`Model.from_antimony(path)`** — Load from Antimony `.ant` file (requires `antimony`, `python-libsbml`)
- **`Model.from_antimony_string(text)`** — Load from Antimony string
- **`Model.from_sbml(path)`** — Load from SBML `.xml` file (requires `python-libsbml`)
- **`Model.from_sbml_string(text)`** — Load from SBML XML string

Properties:
- `n_species`, `n_reactions`, `n_observables`, `n_parameters`, `n_functions`, `n_events`
- `species_names`, `param_names`, `observable_names`

Methods:
- **`set_param(name, value)`** — Set a parameter value
- **`get_param(name)`** — Get a parameter value
- **`set_params(dict)`** — Set multiple parameters (atomic)
- **`reset()`** — Reset species to initial concentrations
- **`clone()`** — Deep copy for parallel workers
- **`save_concentrations()`** — Snapshot current state as new initial conditions
- **`set_concentration(name, value)`** — Set a single species concentration
- **`get_concentration(name)`** — Get a single species concentration
- **`add_table_function(name, *, file, times, values, index)`** — Add a piecewise-linear table function
- `n_table_functions`, `table_function_names` — TFUN introspection

## `bngsim.Simulator`

Constructor:
- **`Simulator(model, method="ode", *, jacobian, codegen, net_path, sensitivity_params, poplevel, xml_path, gml)`**

  `method`: `"ode"`, `"ssa"`, `"psa"`, `"nf"`, `"nf_reject"`, `"nfsim"`,
  `"nf_exact"`, `"rulemonkey"`, `"rm"` (and aliases)

  ODE-specific kwargs: `jacobian` (`"auto"`, `"analytical"`, `"fd"`),
  `codegen` (bool), `net_path` (BioNetGen `.net` path only),
  `sensitivity_params` (list[str])

  For SBML and Antimony models, use `codegen=True` without `net_path`.
  `net_path` is not a generic model path and should not point to SBML XML.

  PSA: `poplevel` (float, required). Accepts `.net`, SBML, and Antimony
  models (same dispatch as SSA, sharing the `validate_for_ssa` gate).
  NFsim/RuleMonkey: `xml_path` (str, required), `gml` (int).

Simulation:
- **`run(t_span, n_points, *, seed, rtol, atol, max_steps, timeout, steady_state, steady_state_tol)`** → `Result`
- **`run_batch(t_span, n_points, *, params, seed, num_processors, squeeze, timeout, steady_state, steady_state_tol)`** → `list[Result]` or `Result`
- **`compute_all_sensitivities(t_span, n_points, *, params, chunk_size, n_workers, rtol, atol, max_steps)`** → `Result` with full sensitivity tensor

`timeout` is a wall-clock budget in seconds (or `None` to disable, the default).
When the budget is exceeded, `run` / `run_batch` raise
`bngsim.SimulationTimeout` (a typed `BngsimError`) carrying `timeout`
(the configured limit) and `elapsed` (actual wall-clock time at the trip).
Supported on every backend (ODE, SSA, PSA, NFsim, RuleMonkey). NFsim and
RuleMonkey poll the budget at a coarser granularity than ODE/SSA/PSA
(between `stepTo()` output points for NFsim; every ~1024 SSA events for
RuleMonkey, via its upstream cancellation hook), so the surfaced elapsed
time may overshoot the budget by one such interval.

Interactive:
- **`run_until(t, *, n_points, seed)`** → `Result`
- **`intervene(params)`** — Change parameters mid-simulation
- **`snapshot()`** → `dict` — Save state
- **`restore(snapshot)`** — Restore state

Stop conditions:
- **`add_stop_condition(condition, *, label)`** — `str` expression or `callable`
- **`clear_stop_conditions()`**

Steady-state (`method` ∈ `"newton"` (default), `"integration"`, `"kinsol"` alias):
- **`steady_state(*, tol, max_time, method, rtol, atol, max_steps, sensitivity_params)`** → `SteadyStateResult`
- **`steady_state_batch(params, *, tol, max_time, method, rtol, atol, max_steps, n_workers)`** → `list[SteadyStateResult]`

Configuration:
- **`set_tolerances(rtol, atol)`** — ODE solver tolerances
- **`set_max_steps(max_steps)`** — Max internal solver steps

Properties:
- `method`, `model`, `current_time`

## `bngsim.NfsimSession`

Stateful NFsim session API for multi-action workflows where parameters or live
particle counts are changed between simulation segments.

Constructor:
- **`NfsimSession(xml_path, *, molecule_limit)`** — Create an NFsim session from
  a BNGL-generated NFsim XML file

Session control:
- **`initialize(seed)`** — Initialize the live NFsim system
- **`simulate(t_start, t_end, *, n_points, timeout)`** → `Result` —
  `timeout` is a wall-clock budget in seconds (`None` disables, the
  default). Checked between `stepTo()` output points; on overrun raises
  `bngsim.SimulationTimeout` and the session must be destroyed before
  reuse.
- **`destroy()`** — Release the live NFsim session

Parameters:
- **`set_param(name, value)`** — Set a parameter before initialization
- **`get_parameter(name)`** — Evaluate a parameter in the live session

Live counts:
- **`get_molecule_count(molecule_type)`** — Count all live molecules of a type
- **`add_molecules(molecule_type, count)`** — Add default unbound molecules
- **`get_species_count(pattern)`** — Count an exact single-molecule BNGL species
- **`add_species(pattern, count)`** — Add exact single-molecule species instances
- **`remove_species(pattern, count)`** — Remove exact single-molecule species instances
- **`set_species_count(pattern, count)`** — Set the exact species count by adding
  or removing instances

Species count mutation currently supports exact, unbound, single-molecule BNGL
patterns such as `"X(p~0,y)"`, `"L(r)"`, and `"TNF()"`. Patterns must list every
component and specify every stateful component state. Multi-molecule complex
patterns fail with a clear `SimulationError`.

## `bngsim.RuleMonkeySession`

Stateful RuleMonkey session API for exact network-free workflows where
parameters are configured before initialization and live particle counts can be
changed between simulation segments.

Constructor:
- **`RuleMonkeySession(xml_path, *, molecule_limit, block_same_complex_binding)`** —
  Create a RuleMonkey session from a BNGL-generated NFsim XML file

Session control:
- **`initialize(seed)`** — Initialize the live RuleMonkey system
- **`simulate(t_start, t_end, *, n_points, timeout)`** → `Result` —
  `timeout` is a wall-clock budget in seconds (`None` disables, the
  default). Polled by upstream every ~1024 SSA events; on overrun raises
  `bngsim.SimulationTimeout` and the session must be destroyed before
  reuse.
- **`step_to(time, *, timeout)`** — Advance without recording output;
  honors the same `timeout` semantics as `simulate(...)`.
- **`destroy()`** — Release the live RuleMonkey session

Parameters and counts:
- **`set_param(name, value)`** — Set a parameter before initialization
- **`clear_param_overrides()`** — Clear parameter overrides before initialization
- **`get_parameter(name)`** — Evaluate a parameter from XML plus overrides
- **`get_molecule_count(molecule_type)`** — Count all live molecules of a type
- **`add_molecules(molecule_type, count)`** — Add default unbound molecules
- **`get_observable_names()`** — Return observable names from the XML
- **`get_observable_values()`** — Return current observable values

## `bngsim.Result`

Properties:
- **`time`** — `ndarray (n_times,)`
- **`species`** — `ndarray (n_times, n_species)` with named access
- **`observables`** — `ndarray (n_times, n_obs)` with named access (e.g. `result.observables["A_tot"]`)
- **`expressions`** — `ndarray (n_times, n_expr)` with named access
- **`sensitivities`** — `ndarray (n_times, n_species, n_params)` forward sensitivity tensor
- **`has_sensitivities`** — `bool`
- **`sensitivity_params`** — `list[str]` parameter names for sensitivity
- **`dataframe`** — `pandas.DataFrame` (requires pandas)
- **`xr`** — AMICI-style xarray accessor; `result.xr.species`, `.observables`, `.expressions`, `.sensitivities`, `.sensitivities_ic` return labeled `xarray.DataArray`s with `time`/`state`/`observable`/`expression`/`parameter`/`ic_state` coords (requires xarray)
- **`solver_stats`** — `dict` with solver diagnostics
- **`custom_attrs`** — `dict` for user metadata
- `n_times`, `n_species`, `n_observables`, `n_expressions`
- `species_names`, `observable_names`, `expression_names`

Methods:
- **`resolve_outputs(selectors)`** → `list[dict]` — resolve typed output selectors (`species:`/`observable:`/`expression:`, aliases `state:`→`species:`, `function:`→`expression:`, with `expression:foo()`→`expression:foo`) to per-output metadata `{"selector","kind","name","index","column_label"}`. Bare names resolve only if unique across all three kinds; ambiguous/unresolved selectors raise with the candidates listed.
- **`outputs(selectors)`** → `ndarray (n_times, n_outputs)` — value columns for the named selectors (one column per selector, in order)
- **`fisher_information(sigma)`** → `ndarray (n_params, n_params)` — Fisher Information Matrix from sensitivity data
- **`gradient(loss_fn)`** → `ndarray (n_params,)` — parameter gradient ∇_p L from sensitivity tensor and loss function
- **`to_gdat(path)`** — Export observables as BNG `.gdat`
- **`to_cdat(path)`** — Export species as BNG `.cdat`
- **`to_csv(path, *, kind="observables"|"species", sep=",", include_time=True, header=True)`** — Plain delimited-text export (CSV / TSV); no `#` prefix; SBML/RoadRunner-friendly
- **`to_xarray()`** → `xarray.Dataset` bundling species/observables/expressions/sensitivities/sensitivities_ic with shared `time` coord (requires xarray); `custom_attrs` + `seed` propagate to `ds.attrs`
- **`save(path)`** — Save to HDF5 (requires h5py)
- **`Result.load(path)`** — Load from HDF5 (class method)
- **`Result.squeeze(results)`** — Stack list into 3D batch result

## Exceptions

All inherit from `bngsim.BngsimError` (which inherits `RuntimeError`):

- **`ModelError`** — `.net` parse failures, invalid model state
- **`SimulationError`** — Solver failures (convergence, NaN)
- **`SimulationTimeout`** — Wall-clock budget exceeded; `.timeout` and `.elapsed` attributes
- **`ParameterError`** — Unknown parameter name, invalid value
- **`StopConditionMet`** — Stop condition triggered; `.result` has partial data

## Universal `.net` reader

- **`bngsim.parse_net_file(path)`** → `dict` — Parse a `.net` file into an engine-agnostic Python dict with keys: `parameters`, `species`, `observables`, `functions`, `reactions`. Pure Python — no C++ extension needed for parsing.
- **`bngsim.build_model_from_parsed(parsed)`** → `Model` — Build a BNGsim `Model` from the dict returned by `parse_net_file()`. Routes through `ModelBuilder` for full optimization (analytical Jacobian, conservation laws, etc.).

## Utility functions

- **`bngsim.reserved_names()`** → `dict` with `"constants"` and `"functions"` lists
- **`bngsim.configure_logging(level)`** — Enable log output

## Generated API (autodoc)

The curated summaries above cover the common surface. The full, always-in-sync
reference below is generated directly from the package docstrings.

```{eval-rst}
.. automodule:: bngsim
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
```
