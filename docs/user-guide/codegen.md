---
summary: "Compiling a model's right-hand side and Jacobian to C: when BNGsim does it, the compiler it needs, the cache and how to prune it, and compiling large models on clusters."
---

# Code-generated ODE RHS

For ODE runs, BNGsim can compile a model's rate laws to native C, which runs
faster than its interpreter on large models. It compiles the model it built,
whatever it was loaded from: `.net`, BNGL, SBML, Antimony or `ModelBuilder`.

## Turning it on

```python
import bngsim

# The bundled A + B <-> C model; or from_bngl, from_sbml, ... on your own
model = bngsim.Model.from_net(bngsim.examples.path("binding"))

sim = bngsim.Simulator(model, method="ode", codegen=True)
result = sim.run(t_span=(0, 100), n_points=101)
```

- `codegen=None`, the default, compiles a model with 256 species or more
  (`BNGSIM_CODEGEN_THRESHOLD`), where native code beats the interpreter.
- `codegen=True` compiles, or raises: `bngsim.CodegenDeclinedError` if codegen
  declines the model, and `RuntimeError` or `bngsim.CodegenToolchainError` if
  the build fails.
- `codegen=False` never compiles, and ignores code an earlier `Simulator` on the
  same model compiled.
- A forward-sensitivity run always compiles, because its sensitivity equations
  are generated code, so it refuses `codegen=False`.

`Simulator.codegen_backend` says which engine an ODE run used. For an SSA run,
`result.ssa_diagnostics["propensity_backend"]` says the same.

What to watch out for:

- **You need a C compiler:** `cc` on macOS and Linux, `cl` or `gcc` on Windows,
  or `$CC`. Without one, `codegen=True` and forward sensitivities raise
  `bngsim.CodegenToolchainError`, and large models fall back to the interpreter.
  On macOS and Linux `CC` is a command line, split as the shell splits it, so it
  can carry arguments (`CC="ccache cc"`): quote a path with a space in it,
  unless `CC` is that path and nothing else. On Windows it is one path.
- **clang stops at brackets nested more than 256 deep, and BNGsim raises that
  limit (`-fbracket-depth`) for a source that needs it,** such as a long
  `if()` chain written through its true branches. A chain through the false
  branches, the usual way to write a table of values, is written flat. GCC has
  no such limit, and every other build's command is unchanged.
- **On a cluster, compute nodes often have no compiler on `PATH`,** even when
  the login node does. Load one in the batch job (`module load gcc`), or turn
  codegen off with `BNGSIM_NO_CODEGEN=1`.
- **Compiled libraries are cached** in `~/.cache/bngsim/codegen/`, and nothing
  prunes the cache for you. See
  [Managing the artifact cache](#managing-the-artifact-cache).

`bngsim.prepare_codegen(net_path)` is deprecated: it loads the file and compiles
the model. `Simulator(net_path=...)` is deprecated and ignored.

PyBNF's `BngsimModel` turns codegen on by default. Set `BNGSIM_NO_CODEGEN=1`
to turn it off.

## The cache

A compiled library is keyed on the model's structure and the codegen version,
so it is rebuilt when the model or the generated code changes. A parameter
value change almost never causes a rebuild. Each compile writes to a temporary
file and then renames it into the cache, so parallel workers compiling the same
model never load a partial library.

Set `BNGSIM_CODEGEN_CACHE_DIR` to move the cache: to fast node-local scratch, or
to a directory of libraries built ahead of time on a login node so worker jobs
never compile (see [Scheduler-free cluster
evaluation](pybnf.md#scheduler-free-cluster-evaluation)). It is read once at
import, so `export` it before starting Python.

One cache can serve two architectures: a native and a Rosetta Python on an
Apple Silicon Mac, or x86-64 and aarch64 nodes sharing a home directory. Each
builds and loads only its own libraries. On macOS every compile asks for the
Python's own architecture (`-arch arm64` or `-arch x86_64`; the
`BNGSIM_SSA_PROP_CC=1` ablation does not), because Apple's
`cc` otherwise builds for the hardware it runs on: an x86_64 Python under
Rosetta got arm64 libraries it could not load.

### Managing the artifact cache

BNGsim never deletes from the cache as a side effect of using it, since that
could remove a library another process is about to load. So the cache grows
until you prune it. It grows fastest on a machine that follows BNGsim
development: any change to the code generator makes every library on the
machine stale at once.

`bngsim-cache` prunes it:

```bash
bngsim-cache info                          # path, entries, size, dates, live vs orphaned
bngsim-cache clean                         # leftovers of interrupted compiles only
bngsim-cache prune --orphaned              # every library under another codegen key
bngsim-cache prune --older-than 30d        # libraries unused for 30 days
bngsim-cache prune --max-size 2G           # ...or until the directory fits
bngsim-cache clear --yes                   # everything BNGsim wrote
```

`pip install bngsim` puts `bngsim-cache` on your `PATH`. If it isn't there,
which is usual in a source checkout without an activated virtualenv, run the
module or the launcher:

```bash
python -m bngsim.cache info
```

```bash
.venv/bin/bngsim-cache info
```

The Python you run it with decides which libraries count as live, so run it
from the environment whose cache you mean to prune.

The same commands work in Python:

```python
import bngsim

info = bngsim.codegen_cache_info()
print(info.total_bytes, info.by_kind)

bngsim.prune_codegen_cache(max_size="2G")
```

What each command does:

- `info` breaks the cache down by kind (model RHS, SSA propensity, and the
  leftovers of interrupted compiles: `bngsim_shard_*` directories and stray `.c`
  files) and by codegen key.
- `clean` removes only those leftovers. No library is touched.
- `prune` removes libraries by orphan status, age or total size, least recently
  used first. It runs `clean` first, so `--max-size` bounds the whole directory.
- `clear` empties the cache, so every model recompiles on its next run.

Every command that deletes takes `--dry-run`, and all of them follow two rules:

- **Only files BNGsim wrote are removed.** Anything else in the directory is
  reported as `foreign` and left alone, so it is safe to point
  `BNGSIM_CODEGEN_CACHE_DIR` at shared scratch.
- **Nothing recently used is removed.** Entries used or written within
  `--min-age` (default `1h`, over the 600 s default compile timeout) are kept.
  On Linux and macOS, so is anything a running compile is writing. Raise
  `--min-age` if you build very large models with the timeout lifted.

"Least recently used" is the later of a file's access and modification times.
Some file systems don't record access times (a `noatime` Linux mount), and on
macOS a plain read doesn't update them, though loading the library does. Where
access times aren't recorded, the order falls back to build time. `info` says
which one you are getting.

### Which libraries are orphaned

A library is named `rhs_<key>_<hash><suffix>`. The key is the codegen version,
a digest of the code generator's source, and the architecture (`arm64`,
`x86-64`, `aarch64`). `info` counts libraries per key:

```console
$ bngsim-cache info
codegen cache: /home/you/.cache/bngsim/codegen
  entries:   14,377
  size:      2.0 GiB
  ...
  key:       43+317a5b34d5dc9959+arm64
  live:      412 artifact(s), 61.0 MiB
  orphaned:  13,901 artifact(s), 1.9 GiB

  codegen key                              entries         size
  -------------------------------------- --------- ------------
  42+9f0c1de2ab3c4d5e                        9,001      1.2 GiB
  43+317a5b34d5dc9959+arm64 (live)             412     61.0 MiB
  -                                          4,900    700.0 MiB
```

`prune --orphaned` removes every library whose key is not this install's, which
no run here can load again. That targets better than `--older-than`, which keeps
recent orphans and removes idle live libraries.

Each virtualenv has its own key, so one venv's orphans can be another's live
libraries. Spare another install's key with `--keep-key`, which you can repeat:

```bash
bngsim-cache prune --orphaned --keep-key 42+9f0c1de2ab3c4d5e
```

A library whose key differs only in the architecture is the other
architecture's build of the same source. `info` counts it as `other arch`, and
`prune --orphaned` leaves it alone. Prune from that architecture's Python to
sweep it.

Libraries with no key at all are listed under `-`. They count as orphaned, and
`--keep-key -` spares them.

## When the compiler fails

If a compile fails, BNGsim tries the same command on a trivial C file. If that
fails too, the toolchain is broken, not the model, and BNGsim warns once per
process, naming the compiler, the linker and, on macOS, the SDK. It suggests an
installed SDK that links, if it finds one.

Then, while the compiler and its environment (`CC`, `PATH`, `SDKROOT`,
`DEVELOPER_DIR`, ...) stay the same:

- no library is compiled, and no source is generated past the cache lookup;
- models at or above the threshold run on the interpreter;
- `codegen=True` and forward sensitivities raise `bngsim.CodegenToolchainError`;
- libraries already in the cache are still used.

Changing one of those variables in the process (`os.environ["SDKROOT"] = ...`)
gets a fresh try. BNGsim also tries again after 5 minutes, so a toolchain that
was broken for a while (a full disk) is used again once it builds. A retry that
fails, or times out, keeps the verdict for another 5 minutes. A missing
compiler is handled the same way.

One known case: conda-forge's ld64-956.6 with the macOS 27 SDK cannot read the
SDK's `libSystem.tbd` and links nothing.
`SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk` links.

A compiler that builds for another architecture than the Python's is a broken
toolchain too. Homebrew's GCC on an Apple Silicon Mac, for one, ignores
`-arch x86_64` with a warning and builds arm64. Its library is not put in the
cache, and the message names both architectures, quotes the warning and says
what to set `CC` to.

On Linux and macOS a compile does not outlive the Python process that started
it. If the process dies, even by `SIGKILL`, the compiler is killed and its
partial files are removed.

**Strict mode.** A benchmark or a fit that must not run on the interpreter by
accident sets `BNGSIM_CODEGEN_STRICT=1`. A failed automatic build then raises
instead of falling back: the ODE RHS at or above the threshold, or the SSA
propensity library. The SSA library is built for every mass-action model run
with exact SSA, so strict mode raises on its failure even for a model the SSA
engine could run without it (one with events, or more than 64 reactions). A
model codegen declines still runs on the interpreter, since that is not a
failed build. But once the toolchain is known to be broken, every model is
refused before its source is generated, and strict mode raises for those too.

## Large models

### Settings

| Variable | Default | Effect |
| --- | --- | --- |
| `BNGSIM_CODEGEN_THRESHOLD` | `256` | Species count at which `codegen=None` compiles a model. |
| `BNGSIM_NO_CODEGEN` | unset | Set it, and `codegen=None` never compiles a model. |
| `BNGSIM_CODEGEN_CACHE_DIR` | `~/.cache/bngsim/codegen` | The cache directory. Read once at import. |
| `BNGSIM_CODEGEN_STRICT` | off | `1` makes a failed automatic build raise instead of falling back. Empty, `0`, `false`, `no` and `off` are off. A decline still falls back. |
| `BNGSIM_CODEGEN_TIMEOUT` | `600` | Seconds before the compiler is killed. A timeout raises a `RuntimeError` naming this variable. `0` means no limit. |
| `BNGSIM_CODEGEN_OPT` | by size | Optimization level: `0` to `3`, or `high`, `low`, `none`. Sources over about 1 MB default to `low` (`-O1`), and unchunked sources over about 8 MB to `none` (`-O0`). Overrides the chunking level below. |
| `BNGSIM_CODEGEN_CHUNK` | `2000` | Reaction count at which the generated code is split into many small functions. `off` or `0` turns it off, `on` always splits, an integer sets the threshold. |
| `BNGSIM_CODEGEN_CHUNK_SIZE` | `256` | Reactions per small function. Smaller compiles a little faster and runs a little slower. |
| `BNGSIM_CODEGEN_JOBS` | `auto` | Parallel compilers for a split source. `auto` uses the CPUs the process is allocated, not the node's core count. `1` compiles serially. An integer caps the pool. |
| `BNGSIM_CODEGEN_MEM_PER_JOB` | `512` | Estimated peak RAM per parallel compiler, in MB. The number of compilers is capped at available RAM (cgroup limits included) divided by this. |

At `-O0` each temporary gets its own stack slot, so a large function's stack
frame grows with it. A gcc or clang library with a function whose frame is over
4 MiB (512 KiB on Windows, where a thread's stack is 1 MiB) is refused as a
failed build, since calling it could overflow the stack and crash the process.

### Splitting and parallel compiles

A C compiler slows down sharply on one huge function. So from
`BNGSIM_CODEGEN_CHUNK` reactions on, BNGsim splits the generated code (the RHS,
the analytical Jacobian, the sensitivity equations and the output evaluator)
into many small functions. This keeps compile time close to linear and lets the
code compile at `-O2` at any size. The split keeps the reaction order, so the
result is bit-identical to the unsplit code.

The small functions compile as separate files in parallel, then link into one
library, like `make -j`. The pool is sized from the CPUs the process may run on
(`os.sched_getaffinity`, `SLURM_CPUS_PER_TASK`), not `os.cpu_count()`, and
bounded by memory. The library is byte-identical whatever the number of
compilers, so parallelism never changes the cache key. With one CPU, or
`BNGSIM_CODEGEN_JOBS=1`, it compiles serially.

On a laptop every core is available. On a cluster, the speedup needs a batch
job that requests several CPUs per task. With `--cpus-per-task=1` there is
nothing to parallelize.

### Example: a Slurm job for a large model

Request several CPUs per task, enough memory for that many compilers, and a
compiler on the compute node. `--mem` divided by `BNGSIM_CODEGEN_MEM_PER_JOB` is
about the most compilers that will run.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=bngsim-codegen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # codegen compiles across these 16 cores
#SBATCH --mem=32G                 # at least cpus-per-task x RAM per compiler
#SBATCH --time=02:00:00

# A C compiler must be on PATH on the compute node, not only the login node.
module load gcc                   # site-specific

# `auto` already reads SLURM_CPUS_PER_TASK; setting it makes that explicit.
export BNGSIM_CODEGEN_JOBS="${SLURM_CPUS_PER_TASK:-auto}"

# Optional: RAM per compiler, in MB. 32G / 2048 MB caps the pool near 16.
export BNGSIM_CODEGEN_MEM_PER_JOB=2048

# Keep BLAS and OpenMP from oversubscribing the same cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun python fit_large_model.py    # builds Simulator(..., codegen=True)
```

With `--cpus-per-task=1` the same script still runs; it just compiles serially.
