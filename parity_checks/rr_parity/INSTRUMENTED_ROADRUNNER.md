# Instrumented RoadRunner — per-phase load timing for `rr_parity` (GH #135)

Stock RoadRunner fuses parse + interpret + LLVM-IR-gen + JIT into one opaque
`RoadRunner(xml)` call with no public API to split them, so an honest per-phase
load breakdown for the reference engine needs a **source-patched build**. This
doc is the durable home for that build — what it adds, how to build/install it,
and how `rr_parity` consumes it (extracted from the GH #135 thread so it survives
the issue being closed).

See the README's *"Instrumented RoadRunner"* section for the short version; this
is the full reference.

## Where it lives

- **Fork/branch:** `github.com/wshlavacek/roadrunner`, branch
  **`timing-instrumentation`** (cut fresh off `develop`).
- **Commits:** `b11ca753d` (C++ `getLoadTimings()`) + `e5332d90c` (pure-Python
  `__warmup_sec__`).
- **Not** the older `rr-instrumented` branch — that was a placeholder; this is the
  built-and-tested deliverable.

> **Build status: verified on Intel macOS 15 (x86_64) AND on arm64 (Apple
> silicon, macOS 26).** The arm64 build is now reproducible — see
> `build_instrumented_roadrunner.sh` next to this doc (3-stage automation;
> override `RR_BUILD_ROOT` / `BNGSIM_VENV`) and the three arm64-specific fixes
> below. On a host without the instrumented build the sweep
> falls back to stock 2.9.2 (RoadRunner load lumped).
>
> **arm64 gotchas (not needed on Intel):**
> 1. **LLVM artifact:** use `llvm-13.x-macos-14-arm64-Release.zip`. The zip has
>    **no top-level folder** (it unpacks `bin/ include/ lib/` directly) — extract
>    into a dedicated prefix dir.
> 2. **Execute bit:** the LLVM zip stores binaries without `+x`, so
>    `roadrunner`'s `FindLLVM.cmake` can't run `llvm-config` ("LLVM_FLAGS does not
>    exist"). `chmod -R +x <llvm_prefix>/bin` after unzip.
> 3. **Cross-arch default:** `roadrunner/CMakeLists.txt` FORCE-sets
>    `CMAKE_OSX_ARCHITECTURES=x86_64` ("compile for intel chips") but guards on
>    `NOT CMAKE_OSX_ARCHITECTURES`, so pass **`-DCMAKE_OSX_ARCHITECTURES=arm64`**
>    on the roadrunner configure — otherwise it cross-targets x86_64 and fails to
>    link the (arm64) deps (expat/etc.).
>
> Also note the `libroadrunner-deps` submodules live under `third_party/` (e.g.
> `git submodule update --init third_party/zlib third_party/libsbml …`), and
> never init `third_party/llvm-13.x` (the multi-GB monorepo — use the prebuilt).

## What you get

### 1. `RoadRunner.getLoadTimings()` — `dict[str, float]` (seconds)

Repopulated on every `load()` / construction; pure measurement, never affects
model state; zero before the first successful load.

| key | what it measures | phase |
|---|---|---|
| `read_sec` | fetch SBML text (URI/file read; ~0 when a string is passed) | File I/O |
| `parse_sec` | libSBML parse (XML string → `SBMLDocument`) | Parse |
| `interpret_sec` | `SBMLDocument` → internal model symbols (no LLVM yet) | Interpret |
| `codegen_sec` | LLVM IR generation | Codegen (IR) |
| `jit_sec` | LLVM JIT compile/optimize/materialize | Codegen (JIT) |
| `model_cache_hit` | `1.0` if served from RR's in-process model cache (codegen+JIT skipped), else `0.0` | caching |

**Matrix mapping:** RoadRunner's "codegen" analog (vs bngsim's ExprTk/CMake/MIR)
= `codegen_sec + jit_sec`. `jit_sec` dominates; `codegen_sec` is the IR build.
SWIG shape is `std::unordered_map<std::string,double>` → dict via the existing
`roadrunner.i` typemap (no new template).

### 2. `roadrunner.__warmup_sec__` — `float` (seconds), set once at import

Times the `import roadrunner` cost (the `dlopen` of the extension + its C++ static
initializers). Read once per worker process as the per-process line. Pure-Python
(`wrappers/Python/roadrunner/__init__.py.in`) — no C++ rebuild needed for this
half; a rebuilt wheel carries it.

## How `rr_parity` consumes it (already wired)

`_rr_common.py` already calls both and **degrades gracefully** when they're
absent:

- `rr_ode` / `ssa_screen` call `rr.getLoadTimings()` when present; on stock 2.9.2
  the RR load collapses into a single lumped `parse_interpret_codegen_sec`
  (matrix renders "—" for the sub-phases) and the per-phase `*_sec` keys read 0.
- Warmup prefers `roadrunner.__warmup_sec__`; without it, falls back to an
  in-house import-proxy estimate (`timing.warmup.roadrunner_source ==
  "import-proxy"` flags this).
- **bngsim's full load split is unaffected by any of this** — only the RoadRunner
  load decomposition depends on the instrumented build. The parity verdict and
  bngsim timing are always complete.

Reference wiring (what `_rr_common.py` does):

```python
import roadrunner, time
result.timing["warmup_sec"] = roadrunner.__warmup_sec__   # per-process, set at import

r = roadrunner.RoadRunner(sbml_string)
result.timing.update(r.getLoadTimings())                  # per-model load split

for _ in range(3):                                        # warm the integrator first...
    r.reset(); r.simulate(t0, tend, npoints)
t = time.perf_counter()
r.reset(); r.simulate(t0, tend, npoints)
result.timing["integrate_sec"] = time.perf_counter() - t  # ...then time the warm marginal cost
```

## Building it (CI / another machine)

3-stage build; CMake hard-fails without a prebuilt deps tree (no auto-download).

1. **LLVM 13 (prebuilt).** Download the matching `llvm-13.x-<os>-<arch>-Release.zip`
   from the `sys-bio/llvm-13.x` releases → `LLVM_INSTALL_PREFIX`. Use a compiler
   matching the artifact's lineage (**AppleClang on macOS** — a Homebrew-clang /
   libc++ mismatch will break the link).
2. **libroadrunner-deps.** Clone, then init submodules **non-recursively** —
   `git submodule update --init` on the 12 top-level paths (`zlib nleq1 nleq2
   bzip2 poco expat libsbml rr-libstruct sundials NuML libSEDML SBMLNetwork`),
   **not** `--recurse-submodules` (recursion pulls the multi-GB LLVM monorepo —
   `BUILD_LLVM=OFF`, never needed — and 3+ redundant full copies of libSBML).
   Build with `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (CMake 4.x + old subprojects)
   → `RR_DEPENDENCIES_INSTALL_PREFIX`.
3. **roadrunner.** Configure with
   `-DBUILD_PYTHON=ON -DLLVM_INSTALL_PREFIX=… -DRR_DEPENDENCIES_INSTALL_PREFIX=…
   -DPython_ROOT_DIR=<venv> -DSWIG_EXECUTABLE=<swig>` + the policy flag;
   `--target install`. The Python package (with compiled `_roadrunner.so`) lands
   at `install-Release/site-packages/roadrunner`.

## Installing it into the sweep venv (shadow the wheel)

Copy the built `roadrunner/` package over the stock one in the sweep venv, e.g.
`bngsim/.venv/lib/pythonX.Y/site-packages/roadrunner` — leave the
`libroadrunner-2.9.2.dist-info` in place so pip still considers it installed.
`import roadrunner` then exposes `getLoadTimings()` / `__warmup_sec__`; the version
still reports `2.9.2` (the base is unchanged). Back up the stock package first so
you can restore by swapping the directory back. The next sweep picks up the split
with **no code change**.

## Caching semantics

RoadRunner's model cache is **in-process**, keyed by SBML MD5; cached resources
live as long as a `RoadRunner` holding them is alive.

- The fresh-process-per-model worker isolation ⇒ `model_cache_hit == 0.0` (cold
  path) for every measured load — exactly what a fair comparison wants. The flag
  also lets you *detect* an accidental warm load.
- On a cache hit, `parse_sec` is **still nonzero** — parse runs in `load()`
  *before* the cache is consulted; the cache only skips interpret/codegen/jit. So
  a warm load ≈ read + parse + a few µs of model-data alloc.
- There is also an on-disk LLVM object cache (`rrObjectCache`); `model_cache_hit`
  flags the **in-process model cache** only.

## The warmup cost model (three tiers)

"Warmup" is not one number — it splits into three tiers that live in different
places and recur at different rates:

| tier | what | magnitude (Intel mac) | measured by | recurs |
|---|---|---|---|---|
| per-process | `import roadrunner` (dlopen + static init) | ~94 ms quiet (~120 ms under load) | `roadrunner.__warmup_sec__` | once per worker |
| per-model | JIT-compile one model | ~25 ms trivial, grows with size | `getLoadTimings()` → `codegen_sec + jit_sec` | once per distinct model |
| per-integration | one `simulate()` | ~78 µs warm marginal | your Python timer around `simulate()` | every solve |

Notes:
- In this build LLVM / libSBML / SUNDIALS are **statically baked into one ~48 MB
  `_roadrunner.so`** (`otool -L` shows only `libSystem` + `libc++`), so the per-
  process cost is one monolith's dlopen + static initializers, not several
  libraries loading. LLVM target init + solver registration happen at first
  `RoadRunner()` construction (~0.5 ms, negligible), not at import.
- `__warmup_sec__` is load-dependent (a real dlopen) — **average it across the
  sweep**, don't trust a single process.
- **Warm the integrator before timing the per-integration cost.** The first
  `simulate()` pays one-time CVODE setup (~124 µs, +159% over the ~78 µs warm
  marginal). A 1-integration-per-worker sweep would otherwise report the *cold*
  sim, while production is dominated by the *warm* marginal cost. (Load timings
  stay cold — fresh-process isolation is still correct for the *load* comparison.)

## RoadRunner method-detection facts (hardcode these — the engine can't vary them)

The instrumentation work confirmed two facts that the matrix derives rather than
queries, because RoadRunner's CVODE integrator is a single fixed configuration:

- **Jacobian = `fd` (difference-quotient).** RoadRunner passes `nullptr` to
  `CVodeSetJacFn` (`source/CVODEIntegrator.cpp:745`) → CVODE's internal
  difference-quotient Jacobian. LLVM generates the model **RHS**, not the
  integrator Jacobian; `getFullJacobian()` exists for analysis but the integrator
  never uses it. So on the bngsim `analytical`/`fd`/`jax` axis RoadRunner is
  permanently `fd`.
- **Linear solver = Dense (built-in SUNDIALS LU).** Hardcoded
  `SUNLinSol_Dense(...)` (`source/CVODEIntegrator.cpp:733`) — no KLU/band/LAPACK
  option. Equivalent to bngsim `LinearSolverKind` 0 = Dense, *not* dgetrf/LAPACK
  (2) and *not* KLU (1). On the linear-solver axis RoadRunner is permanently
  `force-dense` (built-in LU).

**Implication:** the Dense/KLU/LAPACK × analytical/FD solver-method matrix has
**no RoadRunner counterpart** — RoadRunner is one fixed point (LLVM-JIT codegen,
FD Jacobian, dense LU). Frame the bngsim combo matrix against *one* RoadRunner
baseline column, not a RoadRunner matrix of the same shape. The one RoadRunner
axis that is genuinely sweepable *and* measurable is the codegen/JIT engine
(**MCJIT vs LLJIT** ± opt level, via `LLVM_BACKEND` / `LLJIT_OPTIMIZATION_LEVEL`),
which shows up directly in `getLoadTimings()` `codegen_sec`/`jit_sec` — worth at
most a 2-config RoadRunner strip.

## Verification (no behavior change)

- Fresh load: parse/interpret/codegen/jit all `> 0`, `model_cache_hit == 0`;
  `sum(phases)` = 97–99% of `RoadRunner(xml)` wall (remainder is solver
  registration + default selection lists). Second in-process load of the same
  SBML: `model_cache_hit == 1`, `codegen == jit == 0`.
- A/B simulation fingerprints (jana_wolf, sbml-test-suite 00223, EcoliCore) are
  **bit-identical** to stock 2.9.2 — the patch only wraps `std::chrono` timers
  around unchanged calls (no hot-path cost, mirroring the bngsim no-hot-path gate).
- `__warmup_sec__` matches an external `perf_counter()` around `import roadrunner`
  to within ~0.5 ms across fresh processes.

Example — jana_wolf glycolysis (149 KB):
```
fresh:  read=75µs  parse=34.8ms  interpret=0.20ms  codegen=1.7ms  jit=72ms  cache_hit=0  (sum/wall 98.8%)
warm:   parse=35.8ms  codegen=0  jit=0  cache_hit=1
```
