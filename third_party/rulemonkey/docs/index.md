# RuleMonkey

A network-free stochastic simulator for rule-based biochemical models written
in [BNGL](https://bionetgen.org/). RuleMonkey reads a BioNetGen-generated XML
model and runs an exact discrete-event simulation, producing observable
trajectories in `.gdat` format.

This is the **RuleMonkey 3.x** cleanroom C++17 rewrite, designed to be embedded
as an in-process simulation kernel. The engine exposes a small C++ API
(`include/rulemonkey/simulator.hpp`) that host applications, most notably the
BNGsim simulation engine, can drive without spawning subprocesses or going
through the file system.

## Start here

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)**

    Build the simulator, turn a `.bngl` file into XML, run it, read the output.

- **[BNGL coverage and semantics](model_semantics.md)**

    What RM supports, what it refuses, and how each construct is interpreted.
    The reference for "will my model run, and will it mean what I think".

- **[Troubleshooting and FAQ](troubleshooting.md)**

    Trajectory disagrees with BioNetGen, a rule never fires, a run is slower
    than expected.

- **[Output formats](gdat_format.md)**

    Column layout and conventions for `.gdat`, `.species` and `.scan`.

</div>

## Where the numbers come from

RuleMonkey is written against **BNG2.pl** as its reference. Where a construct
is interpreted, the interpretation is measured against BNG2's own network
expansion (ODE and SSA) rather than argued from first principles, and where RM
cannot reproduce BNG2 it says so at load time rather than quietly producing a
different trajectory. The
[BNGL coverage and semantics](model_semantics.md) page records those
decisions alongside the measurements behind them, including the cases where
the pinned NFsim release disagrees with both.

## Building

Requires CMake 3.20 or newer, a C++17 compiler, and Ninja.

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Linux (GCC/Clang), macOS (AppleClang) and Windows (MSVC) are built and tested
in CI. Full build notes, embedding instructions and the CLI reference are in
the [repository README](https://github.com/richardposner/RuleMonkey#readme).

## Citing

The legacy C implementation (RuleMonkey 2.0.25) is described in Colvin J,
Monine MI, Gutenkunst RN, Hlavacek WS, Von Hoff DD, Posner RG. *RuleMonkey:
software for stochastic simulation of rule-based models.* BMC Bioinformatics
11:404 (2010). See
[`CITATION.cff`](https://github.com/richardposner/RuleMonkey/blob/main/CITATION.cff)
for the current release.
