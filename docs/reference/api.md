---
summary: "Every public class, method and exception, summarized, then the full reference generated from the docstrings."
---

# API reference



## `bngsim.Model`

Factory methods:
- **`Model.load(path)`** — Load any supported file, dispatching on its suffix
  (`.ant`, `.xml`/`.sbml`, `.net`, `.bngl`); raises `ModelError` for anything else
- **`Model.from_net(path)`** — Load from BNG `.net` file
- **`Model.from_bngl(path, *, protocol=False, bng2_pl=None, net_out=None, timeout=600, cache=True)`** — Load from BNGL `.bngl` by running `BNG2.pl generate_network` (requires `bngsim[bngl]`, or `$BNGPATH`/`$BNG2_PL`, plus `perl`). The file's experiment is **not** run: its actions before the first `generate_network` (or, without one, its first run) are applied, except file writes, and `generate_network` carries that call's own `max_iter`/`max_agg`/`max_stoich`, or BNG2.pl's defaults when a run with a network method (ode, ssa, pla, psa), or another action that generates the network, comes before it. `protocol=True` returns `(model, ProtocolSpec)` so the actions are recovered rather than discarded, and `bngsim.run_protocol` runs them. Generated networks are cached under `~/.cache/bngsim/networks` (`$BNGSIM_BNGL_CACHE_DIR`), keyed on the flattened model text and the BNG2.pl that produced it
- **`Model.from_antimony(path)`** — Load from Antimony `.ant` file (requires `antimony`, `python-libsbml`)
- **`Model.from_antimony_string(text)`** — Load from Antimony string
- **`Model.from_sbml(path)`** — Load from SBML `.xml` file (requires `python-libsbml`)
- **`Model.from_sbml_string(text)`** — Load from SBML XML string
- **`Model.from_biomodels(model_id, *, version=None, cache=True, refresh=False)`** — Load a BioModels entry by id, such as `"BIOMD0000000010"`. The file is checked against BioModels' published sha256 and cached in `~/.cache/bngsim/biomodels` (or `$BNGSIM_BIOMODELS_CACHE_DIR`). `refresh=True` fetches again.

Properties:
- `n_species`, `n_reactions`, `n_observables`, `n_parameters`, `n_functions`, `n_events`
- `species_names`, `param_names`, `observable_names`, `function_names`
- `analytical_jacobian_status` — `"complete"`, `"pending"` (functional rate laws not derived yet), or `"declined: <reason>"` when the model runs on the finite-difference Jacobian: the rate law and what stopped its derivation, the derivation budget, the C++ attach gate's verdict, or the environment switch. The gate's verdict names the entry it refused, with the probe state and both values — `∂f[B()]/∂[A()] is non-finite (inf) at probe 0 (the seed state) while the RHS is finite`, or `the finite-difference self-check found a trustworthy mismatch at probe 0 (the seed state) in ∂f[LX()]/∂[LX()]: analytical 0, finite-difference -10` (issue #534). The same reason is logged at INFO on the `bngsim` logger when it happens (issue #506)
- `analytical_jacobian_decline` — the same decline as a dict a census can group by: `None` while the status is `"complete"` or `"pending"`, otherwise `kind` (a stable category from `bngsim._jacobian.DECLINE_KINDS`, such as `budget`, `unresolved_symbol` or the C++ gate's `fd_mismatch`) and `reason` (the status text after `declined: `). The gate's kinds also carry the fields their verdict sets (of `rxn_idx`, `per_observable`, `target_idx`, `row`, `col`, `probe`, `analytical`, `finite_difference`, `n_nonfinite` and `detail`; a field a kind does not set is absent) and the species `row` and `col` name as `row_species` / `col_species`. Group by `kind`, not by the text, whose wording changes (issue #538)
- `primary_param_names` — the model's independent **knobs**, and the list to hand an optimizer or sampler, less `frozen_params` where a model has any. `param_names` minus two classes, one per flag: `param_is_expression` (a derived `ConstantExpression`, recomputed from the primaries underneath it) and `param_is_internal` (a slot BNGsim synthesized — an SBML `_V0_<comp>`, or a function's backing slot — which `set_param` refuses to write). A constant written as arithmetic (`gamma 1/7`) is **not** in either class and is an ordinary knob (issue #227)
- `frozen_params` — the parameters that a number the model was built with was read from. For SBML (compartment sizes, `speciesReference` ids and a kinetic law's own parameters among them): a compartment size set by an `initialAssignment` or a constant assignment rule, the initial value of a species converted by the size of a compartment that is a state, a stoichiometry, a `conversionFactor`, an initial value that could not be kept symbolic, each followed down through what it reads. For a compartmental BNGL model: the volumes BNG2.pl wrote into the rate constants (each parameter a `unit_conversion=` comment names, and what a derived parameter or a function among them reads). `set_param` refuses to change one and a sensitivity column for one is refused (issues #695, #696, #711). Empty for most models

Methods:
- **`set_param(name, value)`** — Set a parameter value
- **`get_param(name)`** — Get a parameter value
- **`set_params(dict)`** — Set multiple parameters (atomic)
- **`reset()`** — Reset species to initial concentrations (= `restore_concentrations()` with no label)
- **`describe(*, limit=50)`** → `dict` — What the model is, in one call, as a JSON-serializable dict whose `str()` is a plain-text report: `source` (format, path, sha256), `counts`, `parameters`, `compartments` and each species' `compartment` and `quantity` (SBML and Antimony only; absent for `.net` and BNGL), `other_states` (a size or parameter a rule or event moves), `species_values`, `observables`, `functions`, `reactions` (for SBML, the file's, with ids, stoichiometry and kinetic laws), `events`, `time_dependence`, `jacobian`, `next_run_starts_from`, and `methods`: for `ode`, `ssa`, `psa`, `nf`, `rm` and `steady_state`, a `status` (`"ok"`, `"refused"`, `"needs_input"`, `"unavailable"`) with `reasons`, each a `code` and the message the run would raise; `forward_sensitivities` is `"refused"`, `"unavailable"` or `"undecided"`, never `"ok"`. Lists past `limit` entries are cut and `omitted` counts what each lost, and an expression keeps 200 characters; `limit=None` keeps all. Runs nothing, compiles nothing, changes nothing (issue #1705). See [What a loaded model is](../user-guide/loading-models.md)
- **`clone()`** — Deep copy for parallel workers, and the only way to get an independent `Model`. All of a model's state lives behind one handle into the compiled extension, so `copy.copy(model)` would share it — a write through the "copy" would change the original, which would then simulate differently with nothing raised. Both `copy.copy` and `copy.deepcopy` therefore refuse and name this method (issue #643). Same for a `Simulator`: build a second one over `model.clone()`. A `Result` does still shallow-copy, having no setter to write through
- **`save_concentrations(label=None)`** — Snapshot current concentrations. No label rebases the default slot (`reset()` returns here); a `label` stores a named snapshot (BNG `saveConcentrations("name")`), and multiple named states coexist.
- **`restore_concentrations(label=None)`** — Restore a snapshot; no label restores the default slot, a `label` restores that named state (BNG `resetConcentrations("name")`)
- **`has_saved_concentrations(label=None)`** — Whether a named snapshot (or, with no label, any) exists
- `saved_concentration_labels` — Sorted names of saved named snapshots
- **`set_concentration(name, value)`** — Set a single species concentration. A fixed species can be named with its `$` or without it (`"$X(a)"` or `"X(a)"`, `"@c::$X()"` or `"@c::X()"`), as BNG2.pl's `setConcentration` takes either: the model's `species_names` drop the `$` the `.net` writes
- **`get_concentration(name)`** — Get a single species concentration, named the same way
- **`pure_sink_species()`** → `list[str]` — write-only accumulator species: a product of ≥1 reaction, a reactant of none, read by no other species' derivative, and not `$`-fixed. These put a structural floor under `||f(y)||_2/n` (issue #74)
- **`is_pure_sink()`** → `ndarray (n_species,) of bool` — the array form, so `mask=~model.is_pure_sink()`
- **`declare_ic_sensitivity({species: {param: d_ic_d_param}})`** — Declare `∂x_k(0)/∂θ` for an initial condition your own code assigns (issue #111). A declared species' row is taken as **fully specified**: parameters not named get `0.0`, so it *replaces* the model's parameter-graph row rather than adding to it
- **`effective_ic_sensitivity(params=None)`** → `dict[str, dict[str, float]]` — the reader paired with the writer above: `{species: {param: ∂x(0)/∂θ}}`, the initial-condition seeding a run from this state would use, after the #113 retirement and the #111 overlay. Answers from model structure alone — **no integration, no simulation** — so a fitting frontend can build its gradient routing once at setup. Keys are the ids `sensitivity_params=` accepts (a #147-lowered compound `<initialAssignment>` reports the original symbols, never the synthetic `_ic_<species>` carrier). A **present** entry valued `0.0` means *seeded, coefficient zero at this state*; an **absent** entry means *no seeding path* — the two are different answers. Needed because `output_sensitivities(axis="parameter")` is a total derivative that already contains this matrix times the `ic` axis (GH #155)
- **`add_table_function(name, *, file, times, values, index, method)`** — Give a table function the model declares new data, in place, over the same index and keeping its method unless `method` is given; every engine reads it from then on. A plain function's name raises `ModelError`; a new name adds a table nothing reads, with a warning (issue #772)
- `n_table_functions`, `table_function_names` — TFUN introspection

Evaluators (issue #523; see [Model evaluators](../user-guide/evaluators.md)). Each takes a state `y` of length `n_species` in `species_names` order, reads the live parameters, and leaves the stored concentrations alone:
- **`rhs(y, t=0.0)`** → `ndarray (n_species,)` — `dy/dt` at `(t, y)`: the interpreted RHS the CVODE callback evaluates, with its observable/function refresh run at `y`
- **`jacobian(y, t=0.0, *, sparse=False)`** → `JacobianMatrix` or `scipy.sparse.csc_array` — `∂f/∂y`, `J[i, j] = ∂f_i/∂x_j`; closed form when it covers every reaction, else the steady-state solver's difference quotient, and `.source` says which (`"analytical"` / `"finite-difference"`). A difference quotient whose step decides an entry (it moves by more than 1e-4 of itself, 1e-10 of its column's largest at least, when the column is taken again at the species' own step or at half the first) raises `SimulationError` (issue #1002). `sparse=True` uses the model's structural pattern
- **`propensities(y, t=0.0)`** → `ndarray (n_reactions,)` — every reaction's SSA propensity in the SSA volume convention (falling factorial for a repeated reactant, times the compartment volume), after the SSA loop's own observable refresh at `y`
- **`stoichiometry_matrix(*, sparse=False)`** → `ndarray (n_species, n_reactions)` or `csc_array` — net coefficients, 0-based; a `$`-fixed species has a zero row. The matrix `conservation_laws` was row-reduced from, so `L @ S == 0` (except in an SBML model with a reaction between compartments of different size, whose laws carry the sizes: `L @ rhs(y) == 0`)

## `bngsim.JacobianMatrix`

The dense return of `Model.jacobian`: a float64 `ndarray` subclass with one extra attribute, **`source`** — `"analytical"` or `"finite-difference"`. Behaves as an ordinary array otherwise.

## `bngsim.Simulator`

Constructor:
- **`Simulator(model, method="ode", *, jacobian, codegen, net_path, sensitivity_params, poplevel, xml_path, gml)`**

  `method`: `"ode"`, `"ssa"`, `"psa"`, `"nf"`, `"nf_reject"`, `"nfsim"`,
  `"nf_exact"`, `"rulemonkey"`, `"rm"` (and aliases)

  ODE-specific kwargs: `jacobian` (`"auto"`, `"analytical"`, `"fd"`,
  `"jax"`), `codegen` (bool or `None`), `sensitivity_params` (list[str])

  `codegen=True` compiles the model as built, whatever it was loaded from;
  `None` (the default) compiles at or above `BNGSIM_CODEGEN_THRESHOLD` species
  (256). `jacobian="jax"` differentiates a JAX RHS built from the model; a
  model it does not implement (a compartment size that is a parameter, as in
  every SBML model, amount-valued species, rate rules, `rateOf`, table
  functions) is refused with the construct named. `net_path` is deprecated and
  ignored.

  PSA: `poplevel` (float, required). Accepts `.net`, SBML, and Antimony
  models (same dispatch as SSA, sharing the `validate_for_ssa` gate).
  NFsim/RuleMonkey: `xml_path` (str, required), `gml` (int).

Simulation:
- **`run(t_span, n_points, *, seed, rtol, atol, max_steps, timeout, steady_state, steady_state_tol, carry_sensitivities, start)`** → `Result` — starts from the model's live state, where the last run ended. `start="initial"` resets the model first; `start="live"` continues with no warning; the default warns (`LiveStartWarning`) where the run starts before the time the last run ended at, from exactly the state it left (issue #1701)
- **`run_batch(t_span, n_points, *, params, seed, num_processors, squeeze, timeout, steady_state, steady_state_tol)`** → `list[Result]` or `Result` — clones + resets each point to the `.net` seed
- **`parameter_scan(parameter, par_scan_vals=None, *, par_min, par_max, n_scan_pts, log_scale, t_span, n_points, reset_conc=True, reset_to=None, on_point=None, seed, squeeze, …)`** → `list[Result]` or `Result` — BNG `parameter_scan`. With `reset_conc=True`, each point resets to the state *at scan invocation* (or to the named snapshot `reset_to`), not the seed — so a pre-equilibrate → intervene → scan protocol carries its post-intervention state. `on_point(model, value)` applies coupled `setConcentration` overrides that track the scanned value.
- **`bifurcate(parameter, par_scan_vals=None, *, …)`** → `list[Result]` or `Result` — continuation scan (`reset_conc=0`): each point continues from the previous point's end-state
- **`compute_all_sensitivities(t_span, n_points, *, params, chunk_size, n_workers, rtol, atol, max_steps)`** → `Result` with full sensitivity tensor

`timeout` is a wall-clock budget in seconds (or `None` to disable, the default).
When the budget is exceeded, `run` / `run_batch` raise
`bngsim.SimulationTimeout` (a typed `BngsimError`) carrying `timeout`
(the configured limit) and `elapsed` (actual wall-clock time at the trip).
Supported on every backend (ODE, SSA, PSA, NFsim, RuleMonkey). ODE reads
the budget at each output point and, inside the solver, every 256
right-hand-side evaluations. NFsim and RuleMonkey poll it at a coarser
granularity than ODE/SSA/PSA (between `stepTo()` output points for NFsim;
every ~1024 SSA events for RuleMonkey, via its upstream cancellation hook),
so the surfaced elapsed time may overshoot the budget by one such interval.

Interactive:
- **`run_until(t, *, n_points, seed)`** → `Result`
- **`intervene(params)`** — Change parameters mid-simulation
- **`save_concentrations(label=None)`** / **`restore_concentrations(label=None)`** — Snapshot / restore the model's concentrations by name (delegates to `Model`); pair with `parameter_scan(reset_to=label)` for a preincubate → save → scan protocol
- **`snapshot()`** → `dict` — Save state
- **`restore(snapshot)`** — Restore state
- **`get_state()`** → `ndarray` / **`set_state(state, *, time=None)`** — Bulk species vector; `time=` also sets `current_time`, so a step can be redone from a saved state and time without `restore`'s backend rebuild ([rolling a step back](../user-guide/simulation.md#rolling-a-step-back))
- **`set_time(t)`** — Set `current_time` (finite, else `ValueError`); the next `run_until` integrates from there

Stop conditions:
- **`add_stop_condition(condition, *, label)`** — `str` expression or `callable`. `run` raises `StopConditionMet` with the result cut at the first sample a condition holds at; `run_batch`, `run_replicates` and `parameter_scan` return each row that meets one cut there, with `Result.stop_condition` set, and refuse `squeeze=True` while one is registered (issue #1369). An expression reads the observables, a species by its name with `(` and `)` written as `_`, and `t`/`time`; a name it cannot read, `^` or a tuple is refused with `ValueError` where it is added (issue #1509)
- **`clear_stop_conditions()`**

Steady-state (`method` ∈ `"integration"` (default), `"newton"`, `"kinsol"` alias):
- **`steady_state(*, tol, max_time, method, rtol, atol, max_steps, sensitivity_params, mask)`** → `SteadyStateResult`
- **`steady_state_batch(params=None, *, states, tol, max_time, method, rtol, atol, max_steps, n_workers, mask, newton_burst)`** → `list[SteadyStateResult]`. Without `states`, every entry starts from the model's initial conditions (`Model.reset`), as `run_batch`'s points do, and the batch warns where the current state differs from them (issue #1521). `states` starts each entry from a state of its own, paired row by row with `params`; `newton_burst=False` with `method="newton"` solves from that state with no integration burst and reports an unstable root instead of rejecting it (issue #524)
- **`steady_states(*, n_starts, seed, upper, decades, dedupe_rtol, tol, n_workers, mask)`** → `SteadyStates`: every steady state multistart Newton finds from starts sampled on the conservation manifold, each with its stability (issue #524). See the user guide's steady-state page
- Both raise `SimulationError` for a model with an event, or one that reads the time (issue #710): the solvers read the right-hand side at `t = 0` and fire no event. `run(steady_state=True)` integrates such a model to the end of its span and does not stop early.

  `mask` (issue #74): which species enter the convergence norm — a boolean array of length `n_species`, or the species names to keep. Default `None` tests every species (BNG2.pl parity). It also restricts the KINSOL unknown set and the `dY_ss/dp` system to the same subspace; excluded species get a NaN `dY_ss/dp` row. See `Model.is_pure_sink()`.

Configuration:
- **`set_tolerances(rtol, atol)`** — ODE solver tolerances
- **`set_max_steps(max_steps)`** — Max internal solver steps
- **`auto_atol(*, rtol=None, floor=None)`** → `ndarray` — derive a per-species absolute tolerance from the model's **current** state, `rtol * max(|y_i|, floor)` (issue #196). For a tolerance that is a constant of the model rather than of the point being integrated, use `bngsim.derive_atol` below.

Every `atol` above takes a float, `bngsim.ExactAtol(value)` (exact,
`CVodeSStolerances`), a sequence of
`n_species` values ordered like `model.species_names` (per-species,
`CVodeSVtolerances`), or `"auto"` (derive one — see `auto_atol`). `run`,
`run_batch`, `parameter_scan` and `compute_all_sensitivities` read a float as a
per-species ceiling, `min(atol, rtol * max(|x_i(0)|, floor))` (issue #1420; the
[solver guide](../user-guide/solvers.md#the-default-tolerances-and-models-in-small-units)
defines the floor), except where sensitivities run on CVODES' difference quotient,
which keeps the scalar; `steady_state` holds a float you pass exactly. A wrong-length
vector raises `ValueError` rather than broadcasting. Multi-point calls resolve it
once, so every point of a batch or scan shares one tolerance. See the
[solver guide](../user-guide/solvers.md#per-species-absolute-tolerance).

Properties:
- `method`, `model`, `current_time`
- `has_analytic_sens_rhs` (bool) and `sens_rhs_decline_reason` (`str | None`,
  issue #438) — whether a forward-sensitivity run on this Simulator uses BNGsim's
  analytic `∂f/∂p` or CVODES' internal difference quotient, and why not when it
  does not. A property of the (build, model) pair rather than of the build, so no
  `capabilities()` key can answer it. Ask a Simulator built for a sensitivity run:
  a plain one carries no analytic `∂f/∂p` because nobody asked for one, which is
  not a decline. See the
  [PyBNF guide](../user-guide/pybnf.md#ask-each-model-whether-its-gradient-is-analytic).

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
- **`get_molecule_count(mol_type)`** — Count all live molecules of a type
- **`add_molecules(mol_type, count)`** — Add default unbound molecules
- **`get_species_count(pattern)`** — Count an exact single-molecule BNGL species
- **`add_species(pattern, count)`** — Add exact single-molecule species instances
- **`remove_species(pattern, count)`** — Remove exact single-molecule species instances
- **`set_species_count(pattern, count)`** — Set the exact species count by adding
  or removing instances

Species count mutation currently supports exact, unbound, single-molecule BNGL
patterns such as `"X(p~0,y)"`, `"L(r)"`, and `"TNF()"`. Patterns must list every
component and specify every stateful component state. Multi-molecule complex
patterns fail with a clear `SimulationError`.

Saved states (in-process):
- **`save_concentrations(label=None)`** — Snapshot the live molecular state
  (counts, component states, bonds). An optional `label` names it (BNG
  `saveConcentrations("name")`).
- **`restore_concentrations(label=None)`** — Rewind to the snapshot; a `label`
  must match the currently-held one.
- **`has_saved_concentrations(label=None)`** — Whether a (named) snapshot is held.

  The NFsim backend keeps a **single** in-session snapshot, so each save
  overwrites the previous one; the label lets a mismatched restore fail loudly.
  The network-based `Model` supports true multi-slot named states.

## `bngsim.RuleMonkeySession`

Stateful RuleMonkey session API for exact network-free workflows where
parameters are configured before initialization and live particle counts can be
changed between simulation segments.

Constructor:
- **`RuleMonkeySession(xml_path, *, molecule_limit, block_same_complex_binding)`** —
  Create a RuleMonkey session from a BNGL-generated NFsim XML file

Session control:
- **`initialize(seed, t_start=None)`** — Initialize the live RuleMonkey system
  with the seed state at `t_start` on the model's clock (`current_time` reads
  it). Left `None`, the clock starts at 0, and for a model that reads the clock
  the first `simulate` moves the seed state to its own start if nothing has
  moved the session first (issue #1022)
- **`simulate(t_start, t_end, *, n_points, timeout)`** → `Result` —
  `timeout` is a wall-clock budget in seconds (`None` disables, the
  default). Polled by upstream every ~1024 SSA events; on overrun raises
  `bngsim.SimulationTimeout` and the session must be destroyed before
  reuse. A segment that starts at `current_time` runs on the session clock,
  which is model time. One that starts elsewhere continues from
  `current_time` and is labeled with the times asked for; for a model that
  reads the clock (`time()` or a table function indexed by time) that raises
  `SimulationError`. `solver_stats["n_steps"]` counts the events since
  `initialize`, as an `NfsimSession`'s does.
- **`step_to(time, *, timeout)`** — Advance to `time` on the model's clock
  without recording output; honors the same `timeout` semantics as
  `simulate(...)`.
- **`destroy()`** — Release the live RuleMonkey session

Every time a session takes or reports (`current_time`, the clock read through
`evaluate`, a `save_state` snapshot) is model time. A model that reads the
clock in its functions has its events placed on time elapsed since the
session's start, which keeps a late start's waiting times (issue #1063).

Parameters and counts:
- **`set_param(name, value)`** — Set a parameter before initialization
- **`clear_param_overrides()`** — Clear parameter overrides before initialization
- **`get_parameter(name)`** — Evaluate a parameter from XML plus overrides
- **`get_molecule_count(mol_type)`** — Count all live molecules of a type
- **`add_molecules(mol_type, count)`** — Add default unbound molecules
- **`get_observable_names()`** — Return observable names from the XML
- **`get_observable_values()`** — Return current observable values

## `bngsim.Result`

Properties:
- **`time`** — `ndarray (n_times,)`
- **`species`** — `ndarray (n_times, n_species)` with named access (e.g. `result.species["A()"]`, `result.species[-1, "A()"]`; an array numpy derives from it carries no names)
- **`state`** — `ndarray (n_times, n_state)`: the full integrator state, unprojected and unremapped, in `Model.species_names` order — the rows `Model.rhs` / `Model.jacobian` take. Wider than `species` on an SBML model whose event assigns a parameter or compartment; the stored rather than the reported value where a species column is remapped after the solve. Solve results only (issue #508)
- **`state_names`** — `list[str]` column names of `state` (= the producing model's `species_names`)
- **`observables`** — `ndarray (n_times, n_obs)` with named access (e.g. `result.observables["A_tot"]`)
- **`expressions`** — `ndarray (n_times, n_expr)` with named access
- **`sensitivities`** — `ndarray (n_times, n_species, n_params)` forward sensitivity tensor
- **`has_sensitivities`** — `bool`
- **`sensitivity_params`** — `list[str]` parameter names for sensitivity
- **`one_sided_sensitivities`** — `list[tuple[float, dict[str, str]]]` — one `(t, {parameter: "increasing" | "decreasing"})` entry for each switch time at which the run sits on a kink, with the direction each listed column is the derivative for; empty for nearly every run (GH #1822)
- **`ic_sensitivity_seed`** — `dict[str, dict[str, float]] | None` — the per-run record of the same matrix `Model.effective_ic_sensitivity()` reports: what CVODES was actually seeded with *for this run*. Use the `Model` accessor to build routing at setup; use this to see what a particular run used, which is the only correct answer for a batch/scan over a nonlinear derived IC (`Rtot = R0*scale`), where the coefficients move per point. `{}` = nothing seeds; `None` = not recorded (no parameter sensitivities, non-ODE, an older HDF5 file, a squeezed batch whose rows disagreed, or a `carry_sensitivities=True` phase, where the engine seeds from the prior phase's `dx/dθ` and discards these rows — GH #210/#81)
- **`dataframe`** — `pandas.DataFrame` (requires pandas)
- **`xr`** — AMICI-style xarray accessor; `result.xr.species`, `.observables`, `.expressions`, `.sensitivities`, `.sensitivities_ic` return labeled `xarray.DataArray`s with `time`/`state`/`observable`/`expression`/`parameter`/`ic_state` coords (requires xarray)
- **`solver_stats`** — `dict` with solver diagnostics
- **`stop_condition`** — `str | None`: the label (else the expression) of the stop condition the result was cut at, `None` where none was met (issue #1369)
- **`started_from`** — `"initial" | "live" | None`: whether the run started from the model's initial conditions (the state `Model.reset()` restores) or from another state, such as the end of an earlier run; `None` where nothing recorded it (an older HDF5 file, a stack whose rows disagree) (issue #1701)
- **`start_state_sha256`** — `str | None`: the SHA-256 of the state vector the run started from, as little-endian float64 bytes in `Model.species_names` order, with `-0.0` read as `0.0` and every NaN as one NaN; two runs with the same digest started from the same values (issue #1701)
- **`record`** — `dict`, JSON-serializable: what the numbers are and how the run went (issue #1703). `columns`: each species and observable column's `quantity` (`"concentration"`, an amount over its compartment's size; `"amount"`; `"count"`, molecules; `None` where the model does not say, as for a `.net` model under ODE) and `compartment`, and each compartment's `size` for the run and SBML `units`; `time`: the time unit; `run`: method, the times asked for, `start`, `started_from` and the `start_state` vector, the `parameters` at the start, `seed`, `rtol`, `atol` as given and as the solver was given it (`scalar`, `per_species`, `tracking_decades`), `max_steps`, PSA's `poplevel`, a network-free run's `xml` file and hash, the stop conditions, `wall_time_s`; `backend`: `rhs` (`"interpreted"`, `"cc"`, `"mir"`), `jacobian` as used and whether the analytical attempt was retried on a difference quotient, `linear_solver` by name and as requested, `sensitivity_rhs`, SSA `propensities`; `run` also holds `parameters_pinned` (derived parameters a `set_param` pinned to a number), `start_carry`, the event state the run started from (pending delayed executions, trigger truth, whether the state is SSA counts), `declared_ic_sensitivity`, `strict_ssa`, `reaction_stats` and a rerun's `rerun_of`; `backend.rhs_requested` is the Simulator's `codegen=`; `model`: the source file's format, path and sha256, `edits`, the calls that changed the model after its load (`add_table_function`), and `tables_sha256`, a hash of its table data; `software`: version and build commit; `rows_differ`: a stack's paths whose `None` means its rows differ; `notices`. Recorded by `run` and the calls made of it (`run_until`, `parameter_scan`, `bifurcate`, a protocol's simulate actions), `run_batch` and `run_replicates`; `None` parts elsewhere (`compute_all_sensitivities`, which has notices, a network-free session, an older HDF5 file). `summary()` carries it without the start vector, `to_xarray()` as a JSON attribute, `save` keeps it all
- **`notices`** — `list[dict]`: each warning, fallback and solver complaint of the call that made the result, `{category, code, message}`, recorded whatever a warnings filter or logging level let through; `code` is stable across releases (`"jacobian.difference_quotient_retry"`, `"atol.below_scale"`, ...); {ref}`notice-codes` lists each with what to do about it. A batch's notice about every row is on each row; one about some rows, on those. Not what the `Simulator` said when built, nor SUNDIALS' own lines on standard error (issue #1703)
- **`custom_attrs`** — `dict` for user metadata
- `n_times`, `n_species`, `n_observables`, `n_expressions`
- `species_names`, `observable_names`, `expression_names`

Methods:
- **`rerun(**changes)`** → `Result` — make the run this result records again (issue #1703): the model built from its recorded source (the file, whose sha256 must still match, or the text a string load kept), the parameters, start state and event state the run started from, the method, seed, sample times, tolerances as applied and the code that ran, on a fresh model; the model the run used is neither read nor changed. An ODE run comes back bit for bit, a stochastic one too on the same build. `changes` are `run()`/`Simulator()` keywords (`t_span`, `n_points`, `sample_times`, `seed`, `rtol`, `atol`, `max_steps`, `max_step`, `timeout`, `steady_state`, `steady_state_tol`, `method`, `poplevel`, `jacobian`, `codegen`, and `linear_solver`: `"dense"`, `"klu"` or `"auto"`, an ODE run's, set through the Simulator's force flags: `"klu"` runs dense on a build without KLU or with a JAX Jacobian, with a `rerun.different_backend` notice, and raises `SparseJacobianUnavailableError` for a model with no sparse Jacobian for KLU), each changing that setting and keeping the rest; the new record's `run.rerun_of` names the record repeated (`record_sha256`) and what `changed`. Refuses with a code where the record cannot promise the same run: `rerun.not_recorded`, `rerun.unsupported` (a stack, a network-free run, stop conditions, carried or declared sensitivities, a model changed after its load), `rerun.model_changed` (the file's hash, or the rebuilt model's table data, parameters or species), `rerun.model_missing`; a model loaded from a string keeps its text for this, and `save` writes it into the file; another build or other code runs, with a `rerun.different_build` or `rerun.different_backend` notice
- **`check(checks=None, *, budget=None, replicates=10)`** → `dict` — make the run again with one setting changed and say whether the answer moved (issue #1707); each check is a `rerun()` from the run's own start on a fresh model, and a disagreement is reported, not raised. `checks=None` runs `tolerance` and `grid` for an ODE run (`replicates` for a stochastic one); `"all"` adds `jacobian` (difference quotient), `linear_solver` (dense for a run on KLU, KLU for a dense one; `skip` with the reason where KLU cannot run the model), `backend` (compiled against interpreted) and `second_engine` (libRoadRunner, where installed: `bngsim[roadrunner]`, for an SBML or Antimony model run from the file's initial state at its parameters, from a file unchanged since the load). A run that computed forward sensitivities also gets `sensitivity` by default, and under `"all"` `sensitivity_fd` (difference quotients) and `sensitivity_second_engine` (the sensitivities against AMICI's, where AMICI is installed, for a model `second_engine` could run with no event and no switch in its math: AMICI at a tenth of the run's tolerance is the reference, and its move from the run's tolerance is each series' drift where it is the larger; a series it moves past 1e-3 of its peak on is judged against AMICI at a hundredth instead (`n_judged_deeper`), and not judged where AMICI moves past that there too (`no_reference`); a diff is the reading's only where it holds as large a decade tighter with both engines converged; issue #1923). Every comparison is species by species on each species' own scale, past the atol the run applied to it: `tolerance` fails when a species moves by more than 1e-3 of its peak at `rtol/10`, `atol/10` (a species past that on its peak at the run's points is judged again, each point on the larger of that peak and the largest value the species reaches beside it on the grid check's run at 8x the points, so one whose points fall on its zeros, or the first point after a fast transient the points miss, is judged on the scale it moves on, and a spike between two points excuses only those two; the report's `n_between` counts such columns and `next` says to rerun with more points; a spike in every interval excuses every point, and a column zero at every one of the 8x points is not seen; issue #1926); a path fails when a converged species differs by more than `max(min(1e3·(atol_j + rtol·peak_j), max(0.01·peak_j, atol_j)), 100·atol_j, 10·d_j)`, `d_j` its own move at the tighter tolerance, or half its move at twice the tolerance where that run is finite and converged; a species not converged at the run's tolerance, or within 100 times its atol of zero where that atol exceeds `rtol·peak_j`, at the run's points and between them on the grid check's run (not resolved by the run, and named in `not_resolved`; one near zero only at its points, and past 100 atol between them, is resolved, and at each point a difference counts only past 100 atol, over the larger of 100 atol and what it reaches beside that point, issue #1974; where the budget runs out before that run, `tolerance` is a `timeout` that names such columns as not judged), is judged on the run's whole scale alone (the allowance of a species at the largest peak and atol: noise near zero passes, a departure as large as the run's species does not; issue #1887), and a run not converged at its own tolerance (`converged: false`) is never `ok` (the differential harness's factors, issue #1257). A steady-state run compares the state it stopped at. `replicates` reports, for each observable at the last time, the reruns' mean, standard deviation and standard error, and `z`, the run's distance from the mean in one draw's standard deviations (`sd·sqrt(1 + 1/n)`), and judges nothing: its status is `info`, neither a pass nor a failure, so a stochastic run's report is never `ok`. JSON-ready; `str()` is a plain-text form. Keys: `check_version`, `ok`, `method`, `stochastic`, `rtol`, `atol`, `drift`, `bound`, `converged`, `n_not_resolved`, `not_resolved`, `n_between`, `checks` (each `name`, `status` of `pass`/`diff`/`skip`/`unsupported`/`exception`/`no_base`/`timeout`/`info`, `value`, `worst_column`, `changed`, `note`), `n_pass`, `n_diff`, `n_info` (checks that judged nothing), `next`. `budget=` bounds the reruns' wall time; a check that does not fit is `timeout` and the report is not `ok`. Refuses a stack
- **`resolve_outputs(selectors)`** → `list[dict]` — resolve typed output selectors (`species:`/`observable:`/`expression:`, aliases `state:`→`species:`, `function:`→`expression:`, with `expression:foo()`→`expression:foo`) to per-output metadata `{"selector","kind","name","index","column_label"}`. Bare names resolve only if unique across all three kinds; ambiguous/unresolved selectors raise with the candidates listed.
- **`outputs(selectors)`** → `ndarray (n_times, n_outputs)` — value columns for the named selectors (one column per selector, in order)
- **`fisher_information(sigma)`** → `ndarray (n_params, n_params)` — Fisher Information Matrix from sensitivity data
- **`gradient(loss_fn)`** → `ndarray (n_params,)` — parameter gradient ∇_p L from sensitivity tensor and loss function
- **`to_gdat(path)`** — Export observables as BNG `.gdat`
- **`to_cdat(path)`** — Export species as BNG `.cdat`
- **`to_csv(path, *, kind="observables"|"species", sep=",", include_time=True, header=True)`** — Plain delimited-text export (CSV / TSV); no `#` prefix; SBML/RoadRunner-friendly
- **`to_xarray()`** → `xarray.Dataset` bundling species/observables/expressions/sensitivities/sensitivities_ic with shared `time` coord (requires xarray); `custom_attrs` + `seed` propagate to `ds.attrs`
- **`digest(*, species=False, max_columns=20, sensitivities=False, max_parameters=5)`** → `dict` — a small, deterministic description of the values, for a caller that cannot look at a plot (issue #1704); JSON-ready (every number finite or `None`), and `str()` of it is a plain-text form, one line per column. The run: `digest_version`, `method`, `stochastic`, `n_runs`, `t_start`, `t_end`, `n_points`, the sensitivity counts, `n_notices` and the first 20 `notices`, `kinds`, and the run's `rtol` (at least 1e-10) and `atol` ceiling (its scalar, or the largest per-species value where the caller gave a vector, `"auto"` or a tracking atol). Per kind (`observables`, `species` on request, or species alone where there are no observables) `n_columns`, `n_left_out` and `columns`, at most `max_columns` of them: past the bound, those with a non-finite value, then the largest ranges, in column order. A column of one deterministic run: `first`, `last`, `min`/`t_min`, `max`/`t_max`, `monotone`, `n_interior_extrema`, `tail_from` (the last sample at or before the start of the last tenth of the window), `tail_change` (the largest distance from the last value from `tail_from` on, over the range) and `settled`; of one stochastic run (or a method or tolerances not recorded) `first`, `last`, `min`, `max` alone; of a stack, statistics across runs at the last time (`last_mean`, `last_std`, `last_min`, `last_q05` … `last_q95`, `last_max`, `last_n_finite`) with `min`, `max`; always `non_finite`, `t_non_finite`, `negative`, `constant`. Turns, `settled`, `negative` and `constant` are judged by the steady-state `criterion="scaled"` tolerance, `rtol·|y| + min(atol, rtol·largest)`, at the run's own `rtol` (at least 1e-10) and `atol` (a species' own as applied; an observable what the run's rule gives a species of its size, `min(atol, rtol·max(its largest, 1e-6·largest))`); a turn is an extreme reached and left by more than it. Numbers and counts only: no word such as "oscillates". With `sensitivities=True` (issue #1883) each listed column of one run gains `sensitivity`: its last value `y`, `resolved` (whether `|y|` exceeds the column's absolute tolerance, so the run told it from 0), and up to `max_parameters` `parameters` and `initial_amounts`, ranked by `|dy_dlnp| = |p·dY/dp|` (within one column the order of the scaled sensitivity), each with `name`, `dy_dp`, `dy_dlnp` (`p` the parameter's value as the run started, or the amount the run started from, before any event at the first time), `dlny_dlnp` (`(p/y)·dY/dp`, `None` unless `resolved`) and `one_sided` (the side at the last time, `"mixed"` where two kinks took opposite sides), with counts of those left out and of those not finite (an assignment-rule row refused); the run block gains `sensitivity` with the last time `t`, a `reason` where none is read (no sensitivities, a stack), the parameters and initial amounts not ranked because their value is 0 or not recorded (a hidden parameter, or a result with no record of its parameters), and `one_sided`, each list at most 20 with its count.
- **`save(path)`** — Save to HDF5 (requires h5py); a path ending `.npz` is written with NumPy alone (issue #1704), for `numpy.load(path, allow_pickle=False)`: `time`, `species`, `observables`, `expressions`, the name arrays, each sensitivity block held with `sensitivity_params`/`sensitivity_ic_species`, and `record` as a JSON string. A `.csv`/`.tsv` path is refused with `to_csv` as the remedy
- **`Result.load(path)`** — Load from HDF5 (class method); refuses a `.npz` path, naming `numpy.load`
- **`Result.squeeze(results)`** — Stack list into 3D batch result; refuses a list holding a row that met a stop condition, or a row whose time points, species/observable/expression names, sensitivity parameters or IC species, reaction labels or present blocks differ from the first row's (issue #1513)

## `bngsim.SteadyStateResult`

Returned by `Simulator.steady_state()`.

Properties:
- **`concentrations`** — `ndarray (n_species,)` species values at steady state
- **`species_names`** — `list[str]`
- **`residual`**, **`method_used`**, **`converged`**, **`n_steps`**, **`n_rhs_evals`** — solve diagnostics
- **`n_residual_species`**, **`excluded_species`** — how many species entered `||f||_2/n`, and the 0-based indices `mask=` excluded (issue #74)
- **`unconverged_pure_sinks`** — `list[str]`; on a *failed* solve, the write-only accumulators that were in the test and are carrying flux — i.e. the residual has a structural floor, not a slow tail. Empty otherwise
- **`sensitivity`** — `ndarray (n_species, n_params)` species `dY_ss/dp` (or `None` if no `sensitivity_params`)
- **`sensitivity_params`** — `list[str]`
- **`observable_names`**, **`expression_names`** — `list[str]` labeling the output-sensitivity rows (populated on a sensitivity run; `expression_names` filters the auto-generated `_rateLawN` functions, matching `Result`)
- **`sensitivities_observables`** — `ndarray (n_observables, n_params)` exact `d(observable)/dp`
- **`sensitivities_expressions`** — `ndarray (n_expressions, n_params)` `d(function)/dp` total derivative (state chain + explicit `∂func/∂p`)
- **`rhs_backend`**, **`sens_jacobian_source`**, **`sens_dfdp_source`**, **`sens_output_source`** — which numerical path each piece took. `sens_output_source` (issues #75, #1002) is `"codegen"` / `"partial"` / `"none"` for the `d(function)/dp` block: the compiled `bngsim_codegen_output_sens` chain rule answered every function, some of them, or none. A function it does not answer is a NaN row, refused by name by `output_sensitivities`; it is not differenced
- **`sens_jacobian_rcond`** — `min|U_jj|/max|U_jj|` from the LU of the (reduced) Jacobian that was inverted. It depends on the units of the species and decides nothing; the measures below do
- **`sens_root_determinant_ratio`**, **`sens_root_pivot_share`**, **`sens_root_condition`**, **`sens_root_column_shift`**, **`sens_root_state_shift`**, **`sens_root_hold_shift`**, **`sens_root_growth_rate`**, **`sens_root_relaxation`** — whether the steady state is an isolated root of the system `dY_ss/dp` was solved on, one the system rests at, and the returned state on it (issue #995): what the determinant keeps of itself when the state is corrected by one Newton step, the least a pivot is of the terms it was computed from, the componentwise condition number (the Perron root of `|A⁻¹|·|A|`), the largest move of a column solved again a Newton step on, against the larger of itself and `1e-3/|p|`, and of a species under that step (where a column moves by more than 0.01 of itself, or the state does, the state is stepped on, up to ten times, and **`sens_root_newton_steps`** counts the steps: `concentrations`, `residual` and the sensitivities are then those of the stepped state), the same where a run ends that is taken on for `max_time` from a millionth beside the returned state, the largest real part among the eigenvalues (up to 512 unknowns; above that, among those of the species the state has at a zero), and the most of a column that a run of `max_time` would leave unestablished. `steady_state` raises `SimulationError` outside 0.6 to 1.67, below 1e-13, above 1e13 (1e-6 and 1e6 where `sens_jacobian_source` is `"finite-difference"`), above 0.01 (either, after the steps), above 0.01, where the real part times `max_time` is above 0.01 (**`sens_root_spectral_radius`** is the largest eigenvalue in size), and above 0.01; and where the run that is taken on fails or uses its `max_steps` steps short of `max_time`. **`sens_root_hold_drift`**, **`sens_root_hold_steps`** and **`sens_root_hold_time`** describe that run, and **`sens_species_scale`** is what an entry for each species is small against: its own concentration, or for a species at a zero where it has been and what it is coupled to. The **`sens_root_*_species`** and **`sens_root_*_param`** attributes name the species and the parameter each is read at (`None` where none)
- **`sens_mask_held_species`**, **`sens_mask_reader_species`** — a species `mask=` left out that the solve held, and a kept species whose rate reads it; `steady_state` raises where there is one
- **`root_stability`**, **`n_unstable_roots_rejected`** — the issue #78 linear-stability verdict on a Newton root (`"stable"` / `"undetermined"` / `"unstable"`, `""` for integration), and how many candidate roots the certificate discarded
- **`eigenvalues`** — `ndarray (n_unknowns,) complex128`; the spectrum the verdict was read off (issue #523): the Jacobian restricted to the species the polish solved for, sorted by descending real part, a conjugate pair adjacent with `+imag` first. Empty when no spectrum was computed — an integration result, or a root the certificate declined on (more than 512 unknowns)

Methods:
- **`resolve_outputs(selectors)`** → `list[dict]` — same selector grammar as `Result.resolve_outputs`
- **`output_sensitivities(selectors, *, axis="parameter")`** → `ndarray (n_selectors, n_params)` — steady-state `∂(named output)/∂p` for `species:`/`observable:`/`expression:` selectors, mirroring `Result.output_sensitivities` (no time axis). `axis="ic"` raises: the initial-condition axis is not computed (a parameter that sets an initial amount carries its share of a conserved total in its own column, issue #704)
- **`__getitem__(name)`** → `float` — steady-state concentration by species name
- **`to_dict()`** → `dict[str, float]`

## Running a BNGL protocol (issue #1366)

- **`bngsim.run_bngl(path, *, seed=None, rtol=None, atol=None, allow_lossy=False, bng2_pl=None, net_out=None, timeout=600, cache=True)`** → `ProtocolResult` — load a `.bngl` with `Model.from_bngl`, parse its actions, and run them, as BNG2.pl runs the file
- **`bngsim.run_protocol(model, protocol, *, seed=None, rtol=None, atol=None, allow_lossy=False)`** → `ProtocolResult` — run a `ProtocolSpec`'s steps on a clone of `model`, from the spec's `generation_step`. ODE, SSA and PSA runs; NFsim and PLA runs, `stop_if`/`max_sim_steps`, and an action the protocol cannot hold are refused before anything runs (`ConversionError`)
- **`ProtocolResult`** — a sequence of `ProtocolRun`, one per experiment, indexed by position or by `suffix`; `.labels`, `.model` (the model as the protocol left it), `.protocol`
- **`ProtocolRun`** — `.experiment`, `.kind` (`"simulate"`, `"scan"` or `"bifurcate"`), `.label`, `.method`; `.result` for a `simulate`; `.scan_values`/`.scan_results` for a scan or a bifurcation's upward sweep, and `.backward_values`/`.backward_results` for its sweep back down

## Exceptions

All inherit from `bngsim.BngsimError` (which inherits `RuntimeError`). Each carries a stable `code`, `details`, `remedies`, `docs_url`, the `notices` the failed run had collected, and `to_dict()`; {doc}`error-codes` lists the codes:

- **`ModelError`** — `.net` parse failures, invalid model state
- **`SimulationError`** — Solver failures (convergence, NaN)
  - **`SparseJacobianUnavailableError`** — `force_sparse_linear_solver=True` on a model that cannot supply a sparse Jacobian for KLU: its analytical Jacobian is unavailable (or suppressed by `jacobian="fd"`) and its sparsity pattern has no structural nonzero to color. A refusal by name, not a solver failure: drop the flag to use the dense solver. A `SimulationError`, as it was raised before it had a type, from `run()` and `run_batch()` alike.
- **`SimulationTimeout`** — Wall-clock budget exceeded; `.timeout` and `.elapsed` attributes
- **`ParameterError`** — Unknown parameter name, invalid value
- **`SensitivityUnsupportedError`** — Forward sensitivity declined for this model: an event whose crossing time moves in a way `dt*/dp` cannot be computed for, or a rate law that does not differentiate to closed form. A declared capability gap, not a failure — catch it to fall back on a derivative-free optimizer. Also inherits `ValueError` (what both sites raised before the class existed), so `except ValueError` still catches it. A missing compiler/JIT is *not* this exception; that is a `CodegenToolchainError` or a `CodegenBuildError`.
  - **`SensitivityRefusedDuringRunError`** — the solver's refusal partway through a sensitivity run, of a construct whose derivative it cannot produce, by a condition no choice of requested parameters changes: an event that fires as its trigger leaves its threshold with no first-order motion, a compartment that jumps twice at one instant, a delay a `run_batch` row made positive. Also a `SimulationError`, as those were raised before they had a type. A refusal whose condition can turn on a requested column (a crossing or fire it moves) is its subclass below; the one exception is #1279's refusal of a held equality the state leaves, which had this type first and turns on a column where a restart lands on the stop.
    - **`SensitivityRefusedForParametersError`** — a refusal partway through a run whose condition can turn on the parameters requested: a crossing or a fire one moves, two times one moves apart, an edge one moves the run onto. The message names it and, where it can, the parameters to drop. `compute_all_sensitivities` retries a chunk refused this way one column at a time and raises this type, naming the columns refused alone and the ones that ran, when some are refused alone too. Before it had a type it was a plain `SimulationError`, as a solver failure is; it still is one.
- **`CodegenDeclinedError`** — `codegen=True` on a model codegen declines, such as one whose functions read each other: a statement about the model, which runs on the interpreted RHS with `codegen=None` or `codegen=False`. A codegen *build* failure (a compiler error, a timeout) says nothing about the model and is a `CodegenBuildError`, still a `RuntimeError`. The deprecated `bngsim.prepare_codegen` raises it for a decline too.
- **`CodegenToolchainError`** — the C toolchain cannot build a codegen library for any model: a compile failed and the same command fails on a trivial C file too, or the compiler is not there. The message names the compiler, the linker and, on macOS, the SDK, and suggests an installed SDK that links when one does. While the compiler and its environment are unchanged the verdict holds, re-probed every 5 minutes, and `codegen=True` and forward sensitivities raise it without compiling, and past the cache lookup without generating the model's source, so a model codegen would have declined gets this error too; a library already in the cache is still used. Under `BNGSIM_CODEGEN_STRICT=1` an automatic build raises it too (issue #1544).
- **`StopConditionMet`** — Stop condition triggered; `.result` has partial data
- **`UsageError`** — an argument a call cannot take. Also a `ValueError`
- **`ArgumentTypeError`** — an argument of a type a call does not take. Also a `TypeError`
- **`NotFoundError`** — a name or key a lookup does not find, such as a result column. Also a `KeyError`, and prints as one
- **`NotSupportedError`** — a call or option an object does not offer as written. Also a `NotImplementedError`
- **`InvalidModelError`** — a model, or a part given to a `ModelBuilder`, that is not well formed. A `ModelError`, and also a `ValueError`
- **`CodegenBuildError`** — this model's compiled right-hand side failed to build or timed out; it still runs on the interpreted one (`codegen=False`)
- **`DependencyError`** — an optional package missing or broken; its remedy gives the `pip install` command. Also an `ImportError`
- **`MissingFileError`** — the file a call names does not exist. Also a `FileNotFoundError`

Each of these was raised as the built-in it extends before it had a type, so an `except` written for the built-in still catches it.

Warnings (`UserWarning` subclasses; filter or escalate them with `warnings`):

- **`LiveStartWarning`** — a call is about to start from exactly the state the
  last run left, which is not the model's initial conditions, before the time
  that run ended at: a second `run(t_span=(0, 5))`, which continues the first.
  Raised before anything integrates, by `run()` (pass `start="initial"` or
  `start="live"` to say which start is meant), by `run_until` from a clock
  behind that time (`set_time` moves it), and once per call by
  `run_replicates`, `parameter_scan` (unless `reset_to` names the start or the scan carries an equilibration's `dx/dθ`),
  `bifurcate` and `run_protocol`'s first run, where `Model.reset()` or
  `Model.save_concentrations()` says it (issue #1701). A scan that also raises
  `ScanDoseWarning` names only `Model.reset()`: a saved state does not dose.

- **`ScanDoseWarning`** — `parameter_scan` over a parameter that seeds a
  species' initial condition cannot move that species: each point starts from
  the state at the call (or `reset_to`), and after a run, in a sensitivity scan
  or from a baseline `save_concentrations()` made, a parameter write leaves it
  where it is. Names the species, once per call, before the first point;
  `Model.reset()` first, or an `on_point` hook that sets them, doses them. A
  scan with a hook is not warned about (refs #1643).

- **`OneSidedSensitivityWarning`** — a sensitivity column is a one-sided
  derivative: two rate-law switches share an instant, a rate law composes them,
  and one cannot be moved alone by any requested parameter (`sigma = t0 +
  t_delta` at `t_delta = 0`). Names each parameter and the direction its column
  is for; `Result.one_sided_sensitivities` lists the same (GH #1822).

- **`RoundoffStepWarning`** — an ODE run's solver took steps too small to move
  its clock (`t + h == t` in double precision), which move the state and not
  the time, so the rows after them can be wrong with no solver error. Names how
  many, the first row after one and the spacing of the doubles there;
  `solver_stats["n_roundoff_steps"]` holds the count. A run can take them and
  be right: `Result.check()` says whether the answer moves at a tighter
  tolerance (issue #1837).

- **`RuleMonkeyWarning`** — `method="rm"` or a `RuleMonkeySession` loaded a model
  for which RuleMonkey reports a warning-level diagnostic: how it reads a rule
  or rate law, such as a rule whose products can never be what it writes, which
  it never fires as BioNetGen generates no reaction for it (issue #1022)

## `bngsim.ModelBuilder`

Builds a model in Python. Each species also gets an observable of the same name.

- **`add_parameter(name, value)`**
- **`add_species(name, initial=0.0, *, fixed=False)`**: a `fixed` species keeps its amount.
- **`add_observable(name, species)`**: a list of species names, or a dict of name to weight.
- **`add_function(name, expression)`**: reads parameters, species, observables, other functions, and `time()`.
- **`add_reaction(reactants, products, *, rate=None, law=None)`**: `rate` is a mass-action rate constant (a parameter, a function, or an expression), multiplied by the reactants. `law` is the whole rate. Give one.
- **`add_event(trigger, assignments, *, delay=0.0, name=None)`**: when `trigger` becomes true, set species to new values, such as `{"A": "A + 100"}`.
- **`build()`** → `Model`.

## `bngsim.examples`

Small models that ship with BNGsim, each with its exact solution.

- **`names()`**, **`describe(name)`**, **`formats(name)`**
- **`path(name, format="net")`** → the model's file. Formats: `net`, `bngl`, `bng-xml`, `sbml`, `antimony`, `python`.
- **`load(name, format=None)`** → `Model`.
- **`exact(name, t, **params)`** → each observable at times `t`, as an array.
- **`steady_states(name, **params)`** → each steady state, with `"stable"`.

## `.net` files as a dict

- **`bngsim.parse_net_file(path)`** → `dict` — The components of a `.net` file as a plain Python dict with keys `parameters`, `species`, `species_ic_params`, `observables`, `functions`, `reactions` and `net_file_dir`. It is the C++ loader's own reading of the file, the one `Model.from_net` builds, with the values the built model holds, so it needs the compiled extension, raises `ValueError` where `Model.from_net` refuses the file (`FileNotFoundError` or `IsADirectoryError` for a path that is not a file), and reports the numbers `Model.from_net` puts in the same slots. Deprecated `Sat`/`Hill` rate laws come back rewritten as explicit functions, with the loader's warning.
- **`bngsim.build_model_from_parsed(parsed)`** → `Model` — Build a BNGsim `Model` from the dict returned by `parse_net_file()`, or one of the same shape. It makes the `ModelBuilder` calls the loader makes, so an unmodified dict builds the model `Model.from_net` loads.

## Absolute tolerance (issues #196, #212, #213, #1420)

- **`bngsim.AUTO`** → `"auto"` — the token every `atol=` accepts. Also the capability probe for the whole per-species feature: `hasattr(bngsim, "AUTO")`. The version string is not one — the checkout that first carried #196 still declared `0.12.2`.
- **`bngsim.derive_atol(state, rtol, *, floor=None)`** → `ndarray` — `rtol * max(|y_i|, floor)` over **a state you supply**, ordered like `model.species_names`. `floor` defaults to the smallest strictly positive entry of `state`, but not above `1.0` (also the floor if every entry is zero).
- **`bngsim.normalize_atol_vector(atol, n_species, species_names=None, *, where="atol")`** → `list[float]` — the length/position/finiteness check `run()` applies, callable at the point a vector is built rather than at the first run.

- **`bngsim.TRACKING`** → `"tracking"` — the token for the over-time mode of issue #213; sugar for `TrackingAtol()`.
- **`bngsim.TrackingAtol(decades=12.0, ceiling="auto")`** — an `atol=` value that re-evaluates the vector against the state being integrated (`CVodeWFtolerances`): `atol_i(y) = clamp(rtol*|y_i|, ceiling_i * 10**-decades, ceiling_i)`. A vector fixes the tolerance per species; this fixes it per instant, which is what a species that starts at order one and **decays** to something tiny needs. At `decades=0` it is the ceiling vector exactly, and it is never looser than that vector. `hasattr(bngsim, "TrackingAtol")` is the probe. Costs steps and can make a stiff model unintegrable — see the [solver guide](../user-guide/solvers.md#tracking-absolute-tolerance) for the corpus numbers.
- **`bngsim.ExactAtol(value)`** — an `atol=` value that holds every species to `value` exactly (`CVodeSStolerances`). A float `atol`, the default `1e-8` or one you pass, is read as a ceiling, `min(atol, rtol * max(|x_i|, floor))` per species (issue #1420); this is the opt-out, and how a scalar was read before. Accepted by `set_tolerances` and `steady_state` too, and carried by an `EvaluationSpec` as `{"exact": value}`; `bngsim.convert.read_sedml(..., exact_atol=True)` reads a SED-ML file's absolute tolerance this way. See the [solver guide](../user-guide/solvers.md#a-scalar-atol-you-pass-and-holding-it-exactly).

`derive_atol` and `Simulator.auto_atol` run the same rule; they differ only in
which state they read. `auto_atol` (and `atol="auto"`) reads the model's **live**
state, which is right for a one-off run and wrong for a fit that moves initial
conditions — there the tolerance would become a function of the fit point and
put a step in the objective. `derive_atol` reads whatever state you hand it, so
the vector can be derived once from the nominal state and held constant. See the
[solver guide](../user-guide/solvers.md#which-state-the-tolerance-comes-from).

## Codegen artifact cache (issue #205)

The compiled-artifact cache is content-addressed and **nothing prunes it
automatically**, so these are the supported way to see how large it has grown and
to bound it. Each mirrors one `bngsim-cache` verb and takes `dry_run=True`; all four
accept a `cache_dir` positional to operate on a directory other than the configured
one. See the [codegen guide](../user-guide/codegen.md#managing-the-artifact-cache).

- **`bngsim.codegen_cache_info(cache_dir=None)`** → `CacheInfo` — path, entry count, `total_bytes`, `by_kind`, `by_key`, `live` / `orphaned` (artifacts under this install's codegen key and under any other, issue #363), `other_arch` (this install's source built for another architecture, which is neither, issue #1486), `built_span` / `used_span`, and `atime_is_live` (whether this filesystem records access times, i.e. whether prune's LRU order is real recency or build order). Reads directory metadata only — the key comes off the artifact's *name*. `.to_dict()` is the JSON form.
- **`bngsim.clean_codegen_cache(cache_dir=None, *, min_age=3600.0, dry_run=False)`** → `CacheSweep` — remove leaked compile partials only (`bngsim_shard_*` directories, stray `.c`, temp libraries). No compiled artifact is touched, so no cache hit is lost.
- **`bngsim.prune_codegen_cache(cache_dir=None, *, orphaned=False, keep_keys=(), older_than=None, max_size=None, min_age=3600.0, dry_run=False)`** → `CacheSweep` — bound the cache by orphan status (`orphaned=True` drops every artifact built under another codegen key, the targeted sweep after an emitter edit, but leaves `other_arch` to the install that can load it), age (`older_than="30d"`) and/or total size (`max_size="2G"`), evicting least-recently-used artifacts. At least one bound is required; the partial sweep runs first, so the size cap covers the whole directory. `keep_keys` spares other installs sharing the directory (`None` in it spares the pre-#363 names) and is only accepted with `orphaned=True`.
- **`bngsim.clear_codegen_cache(cache_dir=None, *, min_age=0.0, dry_run=False)`** → `CacheSweep` — remove every artifact and partial BNGsim owns.

Beside an artifact whose analytic `∂f/∂p` was declined, BNGsim writes a small
decline note (`rhs_<key>_<hash>.sens.json`, issue #438) so the reason survives a
warm cache. Beside an artifact built with the output-sensitivity code it writes a
second note (`rhs_<key>_<hash>.outputs.json`, issue #1004): which functions the
artifact holds a derivative for, and why not for the others. A process that is
handed the artifact from the cache reads it, so `output_sensitivities([...])`
raises the reason the build recorded, and the map is not worked out again. (The
note answers for a model whose tables are the builder's: one given a table
function after it ran keeps its artifact and is analyzed for itself, beside
what the note says the build has no derivative for.) Both
are BNGsim's own files: counted under one kind, removed by `clear`, and removed by
`prune` exactly when the artifact they describe is removed.

Entries BNGsim did not write are classified `foreign`, reported, and **never**
removed — by `clear` as much as by `clean` — because `BNGSIM_CODEGEN_CACHE_DIR` is a
user-supplied path. Nothing modified within `min_age` is removed either, so a sweep
cannot delete the scratch files of a compile that is still running.

`bngsim.cache` carries the rest: the `CacheEntry` / `CacheInfo` / `CacheSweep` types,
the `KIND_*` taxonomy, `classify()` and `artifact_key()` (the kind and the codegen key
an entry's name identifies it by), and the `parse_duration` / `parse_size` value
parsers the CLI uses.

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
```
