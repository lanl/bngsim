<p align="left">
  <img src="docs/rulemonkey-macaque-logo/rulemonkey-readme-card.svg" alt="RuleMonkey" width="65%">
</p>

# RuleMonkey

A network-free stochastic simulator for rule-based biochemical models written
in [BNGL](https://bionetgen.org/). RuleMonkey reads a BioNetGen-generated XML
model and runs an exact discrete-event simulation, producing observable
trajectories in `.gdat` format.

This is the **RuleMonkey 3.x** cleanroom C++17 rewrite, designed to be
embedded as an in-process simulation kernel.  The engine exposes a
small C++ API (`include/rulemonkey/simulator.hpp`) that host
applications — most notably the BNGsim simulation engine — can drive
without spawning subprocesses or going through the file system.

The legacy C implementation (RuleMonkey 2.0.25), introduced in
Colvin J, Monine MI, Gutenkunst RN, Hlavacek WS, Von Hoff DD, Posner RG.
*RuleMonkey: software for stochastic simulation of rule-based models.*
BMC Bioinformatics 11:404 (2010), is preserved in the parent fork's
git history.

## Building

Requires CMake ≥ 3.20, a C++17 compiler, and Ninja.

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

The `rm_driver` and `rm_scan` executables are built at
`build/release/rm_driver` and `build/release/rm_scan`.

When RuleMonkey is vendored with `add_subdirectory`, tests and the
command-line tools default off, while the `RuleMonkey::rulemonkey`
target remains available for `target_link_libraries`.

## Usage

`rm_driver` runs a single trajectory:

```bash
build/release/rm_driver <model.xml> <t_end> <n_steps> [seed] [-no-bscb] \
    [--ignore-unsupported] [--save-state <path>] [--load-state <path>] \
    [--t-start <time>] [--print-functions]
```

Output is tab-separated `.gdat` on stdout (header line beginning with `#`,
then time + observable columns). CPU time is printed to stderr.

`rm_scan` sweeps one parameter and reports the endpoint response at each
value — the RuleMonkey equivalent of BioNetGen's `parameter_scan` and
`bifurcate` actions:

```bash
build/release/rm_scan <model.xml> --param <name> \
    (--values v1,v2,... | --min <m> --max <x> --n-points <n> [--log]) \
    --t-end <t> [--t-start <t>] [--n-steps <k>] [--seed <s>] \
    [--bifurcate] [--reset-conc 0|1] [--print-functions] [-no-bscb] \
    [--ignore-unsupported]
```

Output is tab-separated `.scan` on stdout (one row per swept value).
See [`docs/scan_format.md`](docs/scan_format.md) for the format and the
BNG-semantics mapping.

RuleMonkey defaults to strict BNGL semantics (`block_same_complex_binding`,
equivalent to NFsim's `-bscb` flag). Pass `-no-bscb` for compatibility with
NFsim runs that disabled this check.

## Behavior differences vs NFsim

Two intentional differences are worth knowing about when comparing RM
and NFsim runs of the same BNGL model:

1. **Strict BNGL semantics on by default.** `block_same_complex_binding`
   is enabled (equivalent to NFsim's `-bscb`). Pass `-no-bscb` to match
   an NFsim run that disabled the check. See
   [`docs/model_semantics.md`](docs/model_semantics.md) § "Strict BNGL
   semantics" for the precise LHS-`+` / RHS-`+` rules and
   [`docs/troubleshooting.md`](docs/troubleshooting.md) § "Differs from
   NFsim by a small but reproducible amount" for the practical
   guidance.
2. **Negative function-based propensities are clamped, not refused.**
   If a function rate law evaluates negative at some point in the
   trajectory (a common authoring slip is `prod() = k*(Obs - setpoint)`
   with no sign guard, which can flip sign when `Obs < setpoint`), RM
   treats that reaction as propensity 0 for the affected step and
   emits one `WARN: rule '…' propensity clamped to 0 …` line to stderr
   the first time this happens to each rule; further clamps on the same
   rule are silent so an oscillator around the zero crossing doesn't
   spam the log. NFsim refuses the model and exits with a "negative
   propensity" error. Both behaviors are defensible — the underlying
   BNGL is mis-authored for SSA, where a single reaction whose rate can
   flip sign should be split into two reactions each guarded by
   `if(expr > 0, expr, 0)` (or the symmetric form for the other
   direction). See [`docs/model_semantics.md`](docs/model_semantics.md)
   § "Rate laws / Function" for the engine-side rule and
   [`docs/troubleshooting.md`](docs/troubleshooting.md) § "Function
   rate law goes negative" for porting guidance.

## Layout

```
cpp/rulemonkey/      Core engine sources
cpp/cli/             rm_driver / rm_scan command-line tools
include/rulemonkey/  Public C++ API headers
tests/
  cpp/               C++ unit tests (smoke, set_param, save/load, homodimer rate,
                     public API, expr_eval, table_function, error paths,
                     parameter_scan/bifurcate)
  models/            BNGL test corpora (feature_coverage, real-world, basicModels)
  reference/         Gold-standard reference trajectories from NFsim and earlier RM
harness/             Python benchmarking + validation drivers
```




Reference trajectories under `tests/reference/nfsim/` are 100-replicate ensemble
means and standard deviations regenerated 2026-04-10 from a known NFsim build,
**with two exceptions** (`toy_jim`, `rm_tlbr_rings`) where NFsim was found to
produce incorrect output and the reference was regenerated from a hand-rolled
Gillespie SSA. See [`tests/reference/nfsim/PROVENANCE.md`](tests/reference/nfsim/PROVENANCE.md)
for details and the regeneration scripts under `harness/ssa/`.

## Embedding (C++ API)

RuleMonkey is designed to be embedded directly into a host application.
The public surface is two headers — `include/rulemonkey/simulator.hpp`
and `include/rulemonkey/types.hpp` — and one library target,
`rulemonkey` (built by this project's CMake).

Stateless one-shot simulation:

```cpp
#include <rulemonkey/simulator.hpp>

rulemonkey::RuleMonkeySimulator sim("model.xml");
sim.set_block_same_complex_binding(true);
sim.set_param("kf", 0.5);

rulemonkey::TimeSpec ts{ /*t_start=*/0.0, /*t_end=*/10.0, /*n_points=*/101 };
auto result = sim.run(ts, /*seed=*/42);

for (size_t t = 0; t < result.n_times(); ++t) {
  // result.time[t], result.observable_data[obs_idx][t]
  // result.function_data[fn_idx][t] — BNGL global-function values
}
```

Stateful session for multi-segment workflows (initialize → simulate →
mutate → simulate → save state):

```cpp
sim.initialize(/*seed=*/42);
auto seg1 = sim.simulate(0.0, 100.0, 101);
sim.add_molecules("L", 500);                     // mid-run perturbation
auto seg2 = sim.simulate(100.0, 200.0, 101);
sim.save_state("checkpoint.bin");
sim.destroy_session();
```

Cooperative cancellation for long runs — `run`, `simulate`, and
`step_to` each take an optional `rulemonkey::CancelCallback`.  The SSA
loop polls it every ~1024 events; returning `false` raises
`rulemonkey::Cancelled` (a `std::runtime_error` subclass) at a safe
between-event point, leaving any active session at the last completed
event's time so the caller can inspect, resume, or `destroy_session()`:

```cpp
auto t0 = std::chrono::steady_clock::now();
auto budget = std::chrono::seconds(30);
try {
  auto r = sim.run(ts, /*seed=*/42,
                   [&]() { return std::chrono::steady_clock::now() - t0 < budget; });
} catch (const rulemonkey::Cancelled&) {
  // wall-clock budget exceeded — no partial Result; embedder decides what to do
}
```

Wall-clock budgets, signal handlers, and `request_cancel`-style atomic
flags all compose on top of this single callback shape.  An empty
(default-constructed) callback disables polling — there is no
per-event cost in the common case.

See `include/rulemonkey/simulator.hpp` for the complete contract,
`include/rulemonkey/types.hpp` for the result/time-spec/feature-warning
structs, and `examples/embed.cpp` for a minimal compilable example
(its doc-comment header carries `find_package` / `add_subdirectory`
CMake snippets for both consumption modes).

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — shortest path from a
  BNGL model to RuleMonkey trajectories.
- [`docs/model_semantics.md`](docs/model_semantics.md) — reference for
  "will my BNGL model run on RM?".  Lists every supported BNGL
  construct, every Tier-0 refusal (compartments, Arrhenius / Sat /
  Hill / FunctionProduct rate laws, population types, multi-molecule
  fixed species), and the best-effort warnings the engine emits at
  load time.
- [`docs/gdat_format.md`](docs/gdat_format.md) — the `.gdat` output
  format spec, including precision and parsing recipes.
- [`docs/troubleshooting.md`](docs/troubleshooting.md) — FAQ covering
  unsupported features, `set_param` gotchas, save/load schema
  fingerprints, and the `-bscb` default difference vs NFsim.
- [`docs/test_corpora.md`](docs/test_corpora.md) — the three BNGL
  parity suites (`feature_coverage`, `corpus`, `nfsim_basicmodels`),
  what each catches, and how to run them.  Separate from the C++
  unit tests in `tests/cpp/`, which run via `ctest --preset release`
  and pin down internal API contracts (set_param, save/load,
  homodimer rate vs CME, public-surface coverage).
- [`docs/timing_comparison.md`](docs/timing_comparison.md) — a
  173-model RM-vs-NFsim wall-time comparison with a "where does
  each engine win?" breakdown.
- [`docs/internals.md`](docs/internals.md) — engine-internals reading
  guide for contributors about to modify `cpp/rulemonkey/engine.cpp`:
  SSA loop, pattern matching layers, complex tracking, the
  2-mol/1-bond fast path, `fire_rule`'s OpType switch, and the five
  `select_reactants` paths.

## License

MIT. See [LICENSE](LICENSE).
