# Embedding (C++ API)

RuleMonkey is a library first. `rm_driver` and `rm_scan` are thin command-line
wrappers around it, and BNGsim drives it in process. The public surface is two
headers and one library target:

- [`include/rulemonkey/simulator.hpp`](https://github.com/richardposner/RuleMonkey/blob/main/include/rulemonkey/simulator.hpp),
  the `RuleMonkeySimulator` class;
- [`include/rulemonkey/types.hpp`](https://github.com/richardposner/RuleMonkey/blob/main/include/rulemonkey/types.hpp),
  the structs it takes and returns;
- the CMake target `rulemonkey`, also available as `RuleMonkey::rulemonkey`, the
  name an installed package exports.

The comments in the two headers are the detailed reference for preconditions,
throws and guarantees. This page is a guide to them, by task. It also states
some throws the headers leave implicit, such as calling a session method with
no session. What a model *means* once loaded, including which BNGL constructs
run and which are refused, is in
[BNGL coverage and semantics](model_semantics.md).

## Building against RuleMonkey

RuleMonkey needs C++17. From another CMake project, either build it alongside
your own code:

```cmake
add_subdirectory(path/to/RuleMonkey)
add_executable(my_app my_app.cpp)
target_link_libraries(my_app PRIVATE rulemonkey)
```

or install it and find it:

```cmake
find_package(RuleMonkey CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE RuleMonkey::rulemonkey)
```

These options control the build. "Top level" means RuleMonkey is the project
being configured, not a subdirectory of yours:

| Option | Default | What it does |
|---|---|---|
| `RULEMONKEY_BUILD_CLI` | on at top level | builds `rm_driver` and `rm_scan` |
| `RULEMONKEY_BUILD_TESTS` | on at top level | builds the C++ tests |
| `RULEMONKEY_BUILD_EXAMPLES` | off | builds `examples/embed.cpp` as `rm_embed_example` |
| `RULEMONKEY_INSTALL` | on | generates the `install()` rules `find_package` needs |
| `RULEMONKEY_WARNINGS_AS_ERRORS` | on at top level | treats compiler warnings as errors |

[`examples/embed.cpp`](https://github.com/richardposner/RuleMonkey/blob/main/examples/embed.cpp)
is a minimal program that compiles against the library: it loads a model,
prints any unsupported-feature diagnostics, runs once and prints the final
observable values.

## Loading a model

```cpp
#include <rulemonkey/simulator.hpp>

rulemonkey::RuleMonkeySimulator sim("model.xml");
```

The constructor parses and validates the XML immediately, and throws
`std::runtime_error` if it cannot. The XML is what BNG2.pl's `writeXML()` (or
`simulate({method=>"nf"})`) writes; see [Quickstart](quickstart.md#2-generate-xml-from-bngl).

Check the load diagnostics before running. `unsupported_features()` returns
one entry per construct RuleMonkey refuses or wants you to look at. Each entry
has a `severity`:

- `Warn`: RuleMonkey runs the model, but the entry flags one of three things:
    - an approximation;
    - a construct BioNetGen accepts but that is almost surely a mistake;
    - a shape that runs exactly but may be slow, such as an Arrhenius rule
      whose rate RuleMonkey has to bound.
- `Error`: RuleMonkey cannot give BNGL's meaning at all.

`rm_driver` refuses a model with an `Error` unless it is given
`--ignore-unsupported`. An embedder makes that call itself. A fuller version
is under [Embedder integration pattern](model_semantics.md#embedder-integration-pattern).

```cpp
for (const auto& f : sim.unsupported_features())
  if (f.severity == rulemonkey::Severity::Error)
    throw std::runtime_error(f.feature);
```

The model's names are available at once, in XML declaration order:
`observable_names()`, `function_names()` (global functions only) and
`parameter_names()`.

A simulator owns its parsed model and configuration. It cannot be copied but
can be moved. Do not call `run()` on one instance from several threads at
once: it re-applies overrides to the shared parsed model. For replicates in
parallel, use `run_ensemble()`.

## One-shot runs

```cpp
rulemonkey::TimeSpec ts{/*t_start=*/0.0, /*t_end=*/10.0, /*n_points=*/100};
rulemonkey::Result r = sim.run(ts, /*seed=*/42);

for (std::size_t t = 0; t < r.n_times(); ++t) {
  // r.time[t]
  // r.observable_data[obs_idx][t]
  // r.function_data[fn_idx][t]   (BNGL global functions)
}
```

- **Sampling.** Output is sampled at `t_start + i*(t_end - t_start)/n_points`
  for `i = 0..n_points`, so the first row is the seed state. To record at
  other instants, for example an experiment's time points, set
  `ts.sample_times` to an ascending list. Sampling never draws from the RNG,
  so it doesn't change the trajectory.
- **`sample_times` and `t_end`.** `t_end` still bounds the run, and a sample
  at or past it records the final state. So set `t_end` to at least the last
  sample time: a `TimeSpec` left at the default `t_end = 0` records the seed
  state at every sample, with no error. An unsorted list, or a NaN in it,
  throws.
- **Clock.** `t_start` places the seed state on the model's clock, and a rate
  law that reads `time()` reads that clock, as BioNetGen's network `simulate`
  does.
- **More detail.** BNGL coverage and semantics covers both at length:
  [Where a run starts](model_semantics.md#where-a-run-starts-t_start) and
  [Recording at explicit output times](model_semantics.md#recording-at-explicit-output-times).
- **Reproducibility.** Each call starts from the seed species with a fresh RNG.
  The same XML, overrides, molecule limit, `TimeSpec` and seed give the same
  `Result` within one build.

## Parameters and seed amounts

The setters below take effect at the next `run()`, `initialize()` or
`load_state()`, and throw while a session is active: call `destroy_session()`
first. The getters work at any time.

- **`set_param(name, value)`** overrides a declared parameter. The override
  follows the model's own derivations, the `<Parameter expr=>` chains BNG2
  writes, so it reaches every rate constant and every seed amount the parameter
  feeds. A dose scan can reuse one loaded simulator.
- **`get_parameter(name)`** reports the value after that cascade, at once.
- **`clear_param_overrides()`** undoes every parameter override. Seed-amount
  pins stay.
- **`initial_species()`** lists the seed species with the amount the next run
  will use.
- **`set_initial_amount(key, amount)`** pins one seed amount outright, for an
  amount no parameter drives. `key` is the species' name exactly as
  `initial_species()` reports it (its `<Species name=>`), or its XML id; it is
  matched as a string, not parsed as a pattern. A pin outranks parameter
  overrides until `clear_initial_amount_overrides()`. Amounts are truncated
  toward zero when the molecules are made, as in NFsim.
- **`set_molecule_limit(limit)`** stops a run once an event takes the molecule
  count above the limit. That event is kept, the run holds its state for the
  remaining samples, and a session's clock stays at that event.

```cpp
for (double dose : {1.0, 7.0, 68.0}) {
  sim.set_param("AT_nM", dose);            // seed counts follow the derivation
  auto r = sim.run(ts, /*seed=*/42);
}
```

## Replicate ensembles

`run_ensemble()` runs replicates on worker threads and returns the mean and
sample standard deviation at each time point:

```cpp
rulemonkey::EnsembleSpec es;
es.n_reps = 100;
es.n_threads = 0;        // 0: one per hardware thread; 1: the calling thread only
auto ens = sim.run_ensemble(ts, es, /*base_seed=*/42);
// ens.observable_mean[obs_idx][t], ens.observable_sd[obs_idx][t]
// ens.function_mean / ens.function_sd for global functions
```

- **Seeds.** Replicate `i` is seeded `base_seed + i` and is bit-identical to
  `run(ts, base_seed + i)`.
- **Thread count.** The statistics are folded in replicate order, so the result
  is the same for any `n_threads`.
- **Trajectories.** `es.keep_trajectories = true` also returns every
  replicate's `Result`.
- **Failures.** If a replicate throws, the lowest-numbered failing replicate's
  exception is rethrown, whatever the thread count.
- **Cancellation.** `run_ensemble()` also takes a cancel callback. Worker
  threads call it one at a time under a lock, so it need not be thread-safe. A
  `false` stops every replicate and throws `Cancelled`, or a replicate's own
  exception if one had already failed.

`rm_driver --reps` is the same call.

## Parameter scans

`parameter_scan(spec, seed)` and `bifurcate(spec, seed)` are BioNetGen's
`parameter_scan` and `bifurcate` actions. They return the observables and global
functions at the end of each point's run. A `ScanSpec` names the parameter and
either explicit `values` or a `par_min`/`par_max`/`n_points` range (with
`log_scale` for geometric spacing). `per_point` is the `TimeSpec` each point
runs.

- **Fresh points.** With `reset_conc = true`, the default, every point starts
  from the seed species with the same seed, as in BioNetGen. The points differ
  only by the swept parameter.
- **Carry-over.** With `reset_conc = false`, the sweep is one continuous
  trajectory: each point continues from the last point's pool, clock and
  random stream, so a rate law that reads `time()` sees a cumulative clock.
- **Bifurcation.** `bifurcate()` always carries over. It sweeps up, then back
  down, and returns both branches on the same ascending parameter axis, so a
  bistable model shows its hysteresis.
- **Preconditions.** Both throw while a session is active. They also throw
  if `per_point.n_points` is below 1, so a default `TimeSpec` won't do.

`rm_scan` is the command-line form; [`.scan` parameter sweeps](scan_format.md)
gives its output format.

## Live sessions

A session keeps the molecule pool between calls, so a run can be paused,
inspected, changed and continued:

```cpp
sim.initialize(/*seed=*/42);                     // seed state at t = 0
auto seg1 = sim.simulate(0.0, 100.0, 100);       // sample a segment
sim.add_molecules("L", 500);                     // perturb
auto seg2 = sim.simulate(100.0, 200.0, 100);
sim.step_to(500.0);                              // advance without sampling
sim.save_state("checkpoint.bin");
sim.destroy_session();
```

- **Lifecycle.**
    - `initialize(seed, t_start)` starts a session, and `destroy_session()` ends
      it.
    - `has_session()` and `current_time()` report where it is.
    - Either `simulate` overload must start at `current_time()`. Pass it as
      `t_start`. With the `simulate(TimeSpec)` overload and `sample_times`,
      the first sample time is the segment start.
    - Session calls throw `std::runtime_error` when no session is active.
- **Reading the pool.** All of these work between calls:
    - `get_observable_values()` and `get_function_values()`, parallel to
      `observable_names()` and `function_names()`;
    - `get_molecule_count(type)`;
    - `get_species_count(pattern)`, where `pattern` is a fully specified BNGL
      species such as `"A(b!1).B(a!1)"`;
    - `evaluate_expression(expr, extra)`, which evaluates any BNGL expression
      over parameters, observables, functions and the clock. `extra` lets you
      try "what if X were Y" without changing the session.
- **Changing the pool.**
    - `add_molecules(type, n)` adds unbound molecules of one type.
    - `add_species(pattern, n)`, `remove_species(pattern, n)` and
      `set_species_count(pattern, n)` change one fully specified species.
- **Species.**
    - `enumerate_species()` lists every live species with its count, by canonical
      BNGL string.
    - `write_species_file(path)` writes the same list as a BNG-format `.species`
      file ([`.species` enumeration](species_format.md)).
    - `species_count(label)` takes a label RuleMonkey itself emitted. Use
      `get_species_count` for a hand-written pattern.
    - `total_complex_count()` is the number of live complexes.
- **Checkpoints.**
    - `save_state(path)` writes the pool, the RNG and the clock.
      `load_state(path)` restores them into a new session.
    - The file carries a fingerprint of the molecule types, and a load into a
      model whose types differ is refused. Rules and observables are not
      fingerprinted.
    - Parameter values are not in the file. `load_state()` applies the loading
      simulator's current overrides and molecule limit.
    - The RNG state round-trips only between builds made with the same C++
      standard library.
    - The header spells out both limits.
- **Continuation.** A session continued across several calls, or across
  `save_state`/`load_state`, is an exact simulation. It is not the same
  realization as one `run()` from the same seed, because each call ends on a
  waiting-time draw that it spends. See
  [BNGL coverage and semantics](model_semantics.md#functional-surface).

## Cancelling a long run

`run()`, `simulate()` and `step_to()` take an optional
`rulemonkey::CancelCallback`, a `std::function<bool()>`. The event loop calls it
about every 1024 events, and a `false` return throws `rulemonkey::Cancelled`
(a `std::runtime_error`). No partial `Result` comes back, and a live session is
left at its last completed event, ready to inspect, continue or destroy.
`run_ensemble()` takes one too (see [Replicate ensembles](#replicate-ensembles)).
An empty callback, the default, costs nothing per event.

```cpp
auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
try {
  auto r = sim.run(ts, /*seed=*/42,
                   [&] { return std::chrono::steady_clock::now() < deadline; });
} catch (const rulemonkey::Cancelled&) {
  // over budget
}
```

## Messages on stderr

A few conditions are reported as one-line warnings on stderr rather than
thrown, because the run can go on. Examples:

- **A negative rate.** A rate law that evaluates negative is clamped to zero.
  The warning prints once per rule in each `run()` or session, and once per
  rule for a whole `run_ensemble()`, so a fitting loop sees it on every run.
  [Troubleshooting](troubleshooting.md#function-rate-law-goes-negative) covers
  it.
- **Unsettled parameters.** A parameter whose derivations do not settle, which
  usually means a dependency cycle.

## Deprecated

`set_block_same_complex_binding(false)` (the CLI's `-no-bscb`) is deprecated
and prints a warning the first time it is called in a process. RuleMonkey 4.0
removes the method, and RuleMonkey then always applies BNGL's `+`. A call with
`true` only sets the default and can be deleted now. See
[BNGL coverage and semantics](model_semantics.md#strict-bngl-semantics).
