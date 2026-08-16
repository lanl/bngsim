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

### Managing the artifact cache

The cache is content-addressed and **nothing prunes it automatically** — bngsim
never deletes from this directory as a side effect of being used, because the
failure mode (evicting the artifact another process is about to `dlopen`) is the
exact class of bug the cache exists to avoid. So it grows until you bound it.

It grows faster than "a cache grows" suggests. The key folds a digest of the
codegen emitters' own source, so *any* edit to `_codegen.py` / `_jacobian.py` /
`_saturable_jacobian.py` / `_switch_sensitivity.py` — a comment included — orphans
every artifact on the machine at once. That is the right trade (the alternative,
under-invalidation, is a silently wrong gradient), but on a machine that tracks
bngsim development it means a fresh corpus of dead artifacts per emitter edit: six
weeks of ordinary work left 2.0 GB across 14,377 entries on one developer box.

`bngsim-cache` has four verbs:

```bash
bngsim-cache info                          # path, entries, size, dates, live vs orphaned
bngsim-cache clean                         # leaked compile partials only
bngsim-cache prune --orphaned              # every artifact under an older codegen key
bngsim-cache prune --older-than 30d        # evict artifacts unused for 30 days
bngsim-cache prune --max-size 2G           # ...or until the directory fits
bngsim-cache clear --yes                   # everything bngsim owns
```

**Getting at the command.** `pip install bngsim` puts `bngsim-cache` on your `PATH`.
If it is not there — a source checkout whose virtualenv is not activated is the usual
reason — either use the launcher directly or run the module, which needs no console
script and no reinstall:

```bash
python -m bngsim.cache info
```

```bash
.venv/bin/bngsim-cache info
```

Whichever Python you invoke it with is the one whose codegen key counts as *live*
below, so run it from the environment whose cache you mean to sweep.

`info` reports a breakdown by artifact kind — model RHS, SSA propensity, the
source-hash fallback key, plus the debris of interrupted compiles (`bngsim_shard_*`
scratch directories and stray `.c`, which nothing else cleans up) — and a second
breakdown by codegen key.

`clean` is safe by construction: it removes only that debris, so no compiled
artifact is touched and no cache hit is lost. `prune` bounds the cache by orphan
status, age and/or total size, evicting least-recently-used artifacts first; it runs
`clean`'s sweep before evicting anything, so `--max-size` is a bound on the whole
directory. `clear` empties it, which means every model recompiles on its next run.

Every mutating verb takes `--dry-run`, and two guarantees hold across all of them:

- **Only files bngsim wrote are ever removed.** `BNGSIM_CODEGEN_CACHE_DIR` is a
  path you choose, and pointing it at shared scratch is normal, so anything
  unrecognized is reported as `foreign` and left alone — by `clear` as much as by
  `clean`.
- **Nothing recently touched is removed.** A compile in flight writes its `.c` and
  its shard directory into this very directory, so every verb holds off on entries
  used or written within `--min-age` (default `1h`, comfortably over the 600 s default
  `BNGSIM_CODEGEN_TIMEOUT`), and on POSIX also holds a partial whose compile is
  still running. Raise it if you build genome-scale models with the timeout lifted.

"Least-recently-used" is the newer of a file's access and modification time.
Whether access times move at all is a property of the mount — a `noatime` Linux
mount never updates them, and on macOS APFS a plain `read()` leaves `atime` alone
while `dlopen` advances it, which is the only access that matters here. Where they
are not recorded the order degrades to build time (a FIFO — still bounded, just
less well targeted), and `info` says which one you are getting.

The same four verbs are available in-process, for a notebook or a fitting harness
that wants to bound its own cache without shelling out:

```python
import bngsim

info = bngsim.codegen_cache_info()
print(info.total_bytes, info.by_kind)

bngsim.prune_codegen_cache(max_size="2G")
```

### Which artifacts are dead

An artifact is named `rhs_<key>_<hash><suffix>`, where `<key>` is the codegen cache
key described above — the `_CODEGEN_VERSION` constant and a digest of the emitters'
source. The key is mixed into `<hash>` as well, so it is what decides validity;
carrying it beside the hash is what makes the dead corpus *countable*:

```console
$ bngsim-cache info
codegen cache: /home/you/.cache/bngsim/codegen
  entries:   14,377
  size:      2.0 GiB
  ...
  key:       28+317a5b34d5dc9959
  live:      412 artifact(s), 61.0 MiB
  orphaned:  13,901 artifact(s), 1.9 GiB

  codegen key                  entries         size
  -------------------------- --------- ------------
  27+9f0c1de2ab3c4d5e             9,001      1.2 GiB
  28+317a5b34d5dc9959 (live)        412     61.0 MiB
  -                               4,900    700.0 MiB
```

`prune --orphaned` removes exactly that second number: every artifact whose key is
not this install's, which is precisely the set no run here can load again. It is far
better targeted than `--older-than`, which keeps orphans that happen to be recent and
throws away live artifacts that are merely idle.

The per-key table is what makes a shared or pre-warmed artifact directory auditable —
which bngsim's artifacts are in it, and how much each is holding. That matters before
pruning one: a venv per project is ordinary and each has its own key, so one venv's
orphans are another's live artifacts. Spare a sibling install's key with `--keep-key`,
repeatable:

```bash
bngsim-cache prune --orphaned --keep-key 27+9f0c1de2ab3c4d5e
```

Artifacts written before this naming landed (issue #363) carry no key at all. They
count as orphaned — no keyed lookup will ever reach one again — and `info` lists them
under `-`, which is also what spares them: `--keep-key -`. Landing the scheme was
therefore a one-time full invalidation: every cache on every machine recompiles once.

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
