# Quickstart

The shortest path from a BNGL model to RuleMonkey trajectories.

## 1. Build

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release         # smoke + set_param + save/load tests, ~2s
```

Requires CMake ≥ 3.25, a C++17 compiler, and Ninja. The driver lands at
`build/release/rm_driver`.

## 2. Generate XML from BNGL

RuleMonkey reads BioNetGen-XML, not raw BNGL.  Use BNG2.pl's
`writeXML()` action to emit it:

```bash
# my_model.bngl ends with: writeXML();
BNG2.pl my_model.bngl
# produces my_model.xml in the cwd
```

(BNG2.pl ships with [BioNetGen](https://bionetgen.org/).  This repo
does not vendor it; the `BNG2` env var lets the harness scripts find
your local install.)

## 3. Run

```bash
build/release/rm_driver my_model.xml <t_end> <n_steps> [seed]
```

- `t_end` — final simulation time, in BNGL time units.
- `n_steps` — number of sampling intervals.  The output has
  `n_steps + 1` rows (sample at `t_start`, `t_start + dt`, …,
  `t_end`).
- `seed` — optional uint64 RNG seed (default 42).

`stdout` is `.gdat`-format trajectory data;  `stderr` is the wall-
clock CPU time in seconds.  See [`gdat_format.md`](gdat_format.md)
for the output spec.

```bash
# example: 10 time units, 100 sample intervals, seed 7
build/release/rm_driver my_model.xml 10 100 7 > traj.gdat
```

### Strict BNGL semantics (default)

`rm_driver` enables `block_same_complex_binding` by default, matching
NFsim's `-bscb` flag.  Pass `-no-bscb` for parity with NFsim runs that
omitted `-bscb`.  See [`model_semantics.md`](model_semantics.md).

### State save / restore

Continuation across runs (e.g. equilibrate, then perturb, then
continue) goes through `--save-state` / `--load-state`; the file
includes a schema fingerprint that is verified at load time, so a
state file cannot accidentally be loaded against the wrong XML.

```bash
build/release/rm_driver model.xml 100 100 --save-state ckpt.bin
build/release/rm_driver model.xml 200 100 --load-state ckpt.bin --t-start 100
```

## 4. Read the output

Every row begins with the time column, followed by the observables in
the order declared in the BNGL `begin observables` block.  Passing
`--print-functions` to `rm_driver` appends one column per global
function (the `begin functions` entries with no local arguments) —
opt-in, matching BNGL's `print_functions=>1`; the default file is
observables-only.  Index columns by header name so a reader is
agnostic to whether function columns are present.

```python
# minimal Python reader
import csv
with open("traj.gdat") as f:
    headers = f.readline().lstrip("#").strip().split("\t")  # ['time', 'A', 'B', ...]
    rows = [list(map(float, line.split("\t"))) for line in f if line.strip()]
```

For richer parsing, see [`gdat_format.md`](gdat_format.md).

## 5. Embed instead of shelling out

If you're driving RuleMonkey from C++ (for example, from BNGsim), use
the in-process API instead — no subprocess, no XML serialisation
back through the file system:

```cpp
#include <rulemonkey/simulator.hpp>

rulemonkey::RuleMonkeySimulator sim("my_model.xml");
sim.set_param("kf", 0.5);                    // optional override
auto r = sim.run({0.0, 10.0, 100}, /*seed=*/7);
// r.observable_data[obs_idx][t_idx] is the value
// r.function_data[fn_idx][t_idx]   is a global-function value
```

`Result` also carries `function_names` / `function_data` — the BNGL
`begin functions` entries that have no local (per-molecule) arguments,
sampled at the same time points as the observables.  These are the
derived quantities models commonly use as their measured outputs (e.g.
`Clusters() = monomer + dimer + …`).  The matching live-session
accessors are `function_names()` and `get_function_values()`, parallel
to `observable_names()` / `get_observable_values()`.

### Parameter scans on one loaded model

`set_param` overrides propagate through the model's own derivations, so
a dose scan reuses a single loaded simulator instead of re-emitting XML
per point.  BNG2 records both the collapsed number and the symbolic
derivation for every parameter —

```xml
<Parameter id="AT_nM" value="1"         expr="1"/>
<Parameter id="LT"    value="1806.6422" expr="((AT_nM*1e-9)*NA)*V_sim"/>
<Species    id="S1"   concentration="LT" name="L(r1,r2)"/>
```

— and RuleMonkey follows the `expr` chain when you override a parameter
the chain depends on.  Overriding `AT_nM` therefore re-derives `LT` and
reseeds `L(r1,r2)` at the new dose:

```cpp
rulemonkey::RuleMonkeySimulator sim("blbr.xml");
for (double dose : {1.0, 7.0, 68.0}) {
  sim.set_param("AT_nM", dose);              // LT and the L seed count follow
  auto r = sim.run({0.0, 5000.0, 100}, /*seed=*/7);
}
```

A parameter the override does not reach keeps its loaded `value`
bit-for-bit, so runs stay numerically identical outside the override's
dependency cone.  `get_parameter()` reflects the cascade immediately,
without an intervening run.

When the amount you want to change has no parameter behind it — a bare
`<Species concentration="1000">` — set it directly:

```cpp
for (const auto& s : sim.initial_species())
  std::printf("%-16s %-10s %g\n", s.name.c_str(), s.concentration_expr.c_str(), s.amount);

sim.set_initial_amount("L(r1,r2)", 5000);    // molecules, truncated toward zero
sim.clear_initial_amount_overrides();        // back to the concentration= expression
```

A direct pin outranks whatever the `concentration` expression derives,
including a later `set_param`, and a `Fixed="1"` species' clamp target
follows it.  Both kinds of override apply at the next `run()` /
`initialize()` and are rejected mid-session — call `destroy_session()`
first, then re-`initialize()`.

If you need to stop a long-running call from outside (wall-clock
timeout, GUI cancel button, signal handler), pass an optional
`rulemonkey::CancelCallback` to `run` / `simulate` / `step_to`.  The
SSA loop polls it every ~1024 events; returning `false` throws
`rulemonkey::Cancelled` (a `std::runtime_error` subclass) at a safe
between-event point — the simulator instance stays usable and any
active session is left at the last completed event's time, so the
caller can inspect, resume, or `destroy_session()`:

```cpp
auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
try {
  auto r = sim.run({0.0, 10.0, 100}, /*seed=*/7,
                   [&]() { return std::chrono::steady_clock::now() < deadline; });
} catch (const rulemonkey::Cancelled&) {
  // budget exceeded; no partial Result is returned
}
```

An empty (default-constructed) callback disables polling, so callers
that don't want cancellation pay no per-event overhead.

See [`examples/embed.cpp`](../examples/embed.cpp) for a complete
compilable example, and [`include/rulemonkey/simulator.hpp`](../include/rulemonkey/simulator.hpp)
for the full contract.

## What's next

- [`model_semantics.md`](model_semantics.md) — what BNGL features RM
  supports, and the Tier-0 refusals.
- [`gdat_format.md`](gdat_format.md) — output format spec.
- [`troubleshooting.md`](troubleshooting.md) — common errors and
  fixes.
- [`test_corpora.md`](test_corpora.md) — the three test suites and
  how to run them.
- [`timing_comparison.md`](timing_comparison.md) — RM vs NFsim wall
  time on 173 models.
