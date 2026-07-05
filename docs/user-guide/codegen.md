# Code-generated ODE RHS

## Code-generated ODE RHS

For ODE simulations, BNGsim can compile model rate laws into native C code,
which is then loaded via `dlopen()`. This eliminates ExprTk bytecode
interpretation overhead for large models. The public API has two codegen
routes:

**BioNetGen `.net` models** use the `.net` codegen path. If the model was
loaded with `Model.from_net(...)`, BNGsim already remembers the `.net` path:

```python
net_model = bngsim.Model.from_net("model.net")

sim = bngsim.Simulator(
    net_model,
    method="ode",
    codegen=True,
)
result = sim.run(t_span=(0, 100), n_points=101)
```

You may still pass `net_path="model.net"` explicitly for `.net` models, but
`net_path` means exactly this: a BioNetGen `.net` file.

**SBML and Antimony models** use model-based codegen. Do not pass the SBML
XML file as `net_path`; just enable codegen on the loaded model:

```python
sbml_model = bngsim.Model.from_sbml("model.xml")

sim = bngsim.Simulator(
    sbml_model,
    method="ode",
    codegen=True,
)
result = sim.run(t_span=(0, 100), n_points=101)
```

Passing an SBML XML file as `net_path` is rejected because the `.net` parser
cannot interpret SBML. This prevents accidental compilation of an empty RHS.

Codegen is **enabled by default** in PyBNF's `BngsimModel` (set
`BNGSIM_NO_CODEGEN=1` to disable). Compiled `.so` files are cached in
`~/.cache/bngsim/codegen/` by SHA-256 hash — recompilation only happens when
the `.net` content or model-generated code changes. Compilation builds to a
process-unique temp file and `os.replace()`s it into the cache, so concurrent
Dask workers compiling the same model never observe a partial `.so`. Set
`BNGSIM_CODEGEN_CACHE_DIR` to relocate the cache — point it at fast node-local
scratch, or at a directory of artifacts pre-warmed on a login node so worker
jobs never compile (see [Scheduler-free cluster evaluation](pybnf.md#scheduler-free-cluster-evaluation)).

> **HPC / cluster note.** The codegen path shells out to a C compiler (`cc`)
> at `Simulator` construction. On many HPC systems compute nodes have **no
> compiler on `PATH` by default** even when the login node does — codegen then
> stalls or fails on the compute node despite working interactively. Ensure a
> compiler is available inside the batch/allocation environment (e.g.
> `module load gcc`) before running, or disable codegen with
> `BNGSIM_NO_CODEGEN=1`.

A few env vars tune the compile step for large reaction networks, whose flat
RHS source can be several MB:

| Variable | Default | Effect |
| --- | --- | --- |
| `BNGSIM_CODEGEN_CACHE_DIR` | `~/.cache/bngsim/codegen` | Directory for the content-addressed compiled-artifact cache. Redirect it to node-local scratch, or to a read-only dir of pre-warmed artifacts, so cluster jobs reuse one `.so` instead of recompiling. Read once at import — `export` it before launching `python`. |
| `BNGSIM_CODEGEN_TIMEOUT` | `600` | Seconds before the `cc` invocation is killed (a timeout raises a `RuntimeError` naming this var). `0` disables the limit. |
| `BNGSIM_CODEGEN_OPT` | size-based | Optimization level: an integer `0`–`3`, or `high`/`low`. Sources over ~1 MB default to `low` (`-O1`) since `-O3` costs minutes for negligible runtime gain on a single flat function. Overrides the chunking opt level below. |
| `BNGSIM_CODEGEN_CHUNK` | `2000` | Reaction count at/above which the RHS (and analytical sensitivity) body is split into many small `noinline` helper functions instead of one giant function — see below. `off`/`0` disables; `on` forces chunking at any size; an integer sets the threshold. |
| `BNGSIM_CODEGEN_CHUNK_SIZE` | `256` | Reactions per chunked helper function. Smaller blocks compile a little faster at a slight call-overhead cost. |
| `BNGSIM_CODEGEN_JOBS` | allocation-aware | Parallel compiler processes for a chunked source — see below. `auto` (default) sizes the pool from the CPUs the process is actually allocated (`sched_getaffinity` / `SLURM_CPUS_PER_TASK`), never the node's core count; `1` keeps the serial single-`cc` compile; a positive integer caps the pool. |
| `BNGSIM_CODEGEN_MEM_PER_JOB` | `512` | Estimated peak RAM per parallel compiler, in **MB**. The job count is capped at `available_RAM / this` (honoring cgroup limits) so parallel compiles never oversubscribe memory and OOM a node. Raise it on a RAM-tight node, lower it if you know the compiles are small. |

**Large-model chunking.** A flat code-generated RHS over *N* reactions is one
enormous basic block, and the C optimizer's per-function passes are superlinear
in function size — so without chunking a ~100k-reaction model can take *hours* to
compile (a synthetic mass-action RHS scales ≈ O(N^2.5) at `-O1`). At/above
`BNGSIM_CODEGEN_CHUNK` reactions BNGsim splits the body into small `noinline`
blocks, which keeps compile time roughly linear and lets the source compile at
`-O2` at any size (≈ minutes for 100k reactions). The split preserves reaction
order, so the chunked `.so` is **bit-identical** to the flat one; below the
threshold the emitted C is byte-identical to prior versions.

Chunking covers not just the RHS but the whole code-generated translation unit:
the analytical Jacobian's per-reaction scatter, the output evaluator, and the
observable/function computation each of them recomputes are all split into
`noinline` blocks too. Otherwise these would pile up in the single non-sharded
*driver* function and become the serial compile wall at genome scale even after
the RHS itself was chunked — a 113k-reaction / 18k-function model's driver was
~38 MB of C and timed out the compile budget; sharding it shrinks the driver
~10× so no single translation unit dominates.

**Parallel shard compilation.** A chunked source is still one `.c` file compiled
by a single serial `cc`, which dominates `Simulator` construction at genome
scale (a 113k-reaction model spent ~52 min almost entirely in that one compile).
Because the `noinline` blocks are already independent functions, BNGsim compiles
them as **separate translation units in parallel** (`cc -c` × N) and links the
`.o` files into the `.so` — the classic `make -j`. Wall-clock drops to roughly
*(slowest unit + link)* wherever multiple cores are available (the same
113k-reaction model with the compiled sparse Jacobian now builds in well under a
minute on 16 cores).

The pool is **allocation-aware**: it is sized from the CPUs the process may
actually run on (`os.sched_getaffinity` / `SLURM_CPUS_PER_TASK`), never
`os.cpu_count()`, so a job confined to a small slice of a shared node does not
spawn one compiler per *node* core. It is also **memory-bounded** (see
`BNGSIM_CODEGEN_MEM_PER_JOB`) so N parallel compilers cannot OOM the allocation.
The source partition is independent of the job count, so the linked `.so` is
**byte-identical regardless of how many compilers run**, and the SHA-256 codegen
cache keys on source content only — parallelism never changes the cached
artifact. A 1-core allocation (or `BNGSIM_CODEGEN_JOBS=1`) takes the unchanged
serial path, so there is no regression where there are no spare cores.

> **The speed-up requires cores.** On a laptop every core is available, so this
> is automatic. On HPC it engages **only if the batch allocation requests
> them** — with `--cpus-per-task=1` there is nothing to parallelize across.
> Request several CPUs per task (and enough memory for them) to unlock it.

### Example: Slurm batch script for a large-model fit

Request multiple CPUs per task so codegen shards across them, and make sure a
compiler is on `PATH` inside the allocation (compute nodes often lack one by
default). Memory should cover the parallel compilers — `--mem` ÷
`BNGSIM_CODEGEN_MEM_PER_JOB` is roughly the most compilers that will run.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=bngsim-codegen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # ← codegen shards across these 16 cores
#SBATCH --mem=32G                 # ≥ cpus-per-task × per-compiler peak RAM
#SBATCH --time=02:00:00

# A C compiler must be reachable on the COMPUTE node (not just the login node).
module load gcc                   # site-specific; provides `cc`/`gcc` on PATH

# Size the codegen pool from the Slurm allocation. `auto` already reads
# SLURM_CPUS_PER_TASK / the cgroup cpuset; setting it explicitly is equivalent
# and self-documenting.
export BNGSIM_CODEGEN_JOBS="${SLURM_CPUS_PER_TASK:-auto}"

# Optional: tune the per-compiler RAM estimate that bounds the pool. With
# --mem=32G and 2 GB/compiler the pool is capped near 16 (= cpus-per-task here).
export BNGSIM_CODEGEN_MEM_PER_JOB=2048   # MB

# Keep BLAS/OpenMP from oversubscribing the same cores at run time.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun python fit_large_model.py    # builds Simulator(..., codegen=True) inside
```

With `--cpus-per-task=1` the same script still runs correctly — codegen just
falls back to the serial compile (no speed-up, no error). Bump `--cpus-per-task`
(and `--mem` to match) to shard the compile across more cores.
