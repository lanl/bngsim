"""bngsim._codegen — Code-generated ODE RHS for BNGsim.

Generates a C source file implementing the CVODE RHS callback as native
compiled code. Parameters are read from a runtime array (NOT baked as
compile-time literals), so the .so is compiled ONCE per model structure
and reused for all parameter evaluations in a PyBNF fitting run.

Architecture (AMICI/libRoadRunner pattern):
  1. generate_rhs_c(net_path) -> str: Parse .net, emit C source
  2. compile_rhs(c_source, model_hash) -> Path: cc -O3 -shared -fPIC
  3. Cache compiled .so by model hash in ~/.cache/bngsim/codegen/
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import logging
import os
import platform
import re
import signal
import struct
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    # Import-time cycle: _switch_sensitivity imports this module. Only the
    # annotation needs the name, and `from __future__ import annotations` keeps
    # it a string at runtime, so the real import stays function-local (GH #68).
    from bngsim._switch_sensitivity import SwitchConditionScope

logger = logging.getLogger("bngsim")


# Cache directory for compiled .so files.
#
# Content-addressed by model hash, so the same compiled artifact is reused across
# processes and evaluations (the HPC scheduler-free contract, GH #203). Cluster
# jobs override the location via ``BNGSIM_CODEGEN_CACHE_DIR`` — e.g. point it at
# fast node-local scratch, or at a read-only directory of artifacts pre-warmed on
# a login node so worker jobs never compile. Resolved once at import (cluster jobs
# ``export`` it before launching ``python``); tests monkeypatch the module
# attribute directly. The compile path stays concurrency-safe regardless of
# location: each build writes a process-unique temp file in this directory and
# ``os.replace()``s it into the cache atomically (same filesystem), so concurrent
# jobs racing on the same model can never observe a half-written .so.
def _default_cache_dir() -> Path:
    """Resolve the codegen cache directory, honoring ``BNGSIM_CODEGEN_CACHE_DIR``."""
    env = os.environ.get("BNGSIM_CODEGEN_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "bngsim" / "codegen"


CACHE_DIR = _default_cache_dir()

# Codegen backends that JIT the RHS C source in-process instead of compiling a
# .so with `cc` + dlopen. Currently only the vendored MIR micro-JIT (GH #78).
# Lives here (not in _simulator) so every auto-codegen entry point — the
# Simulator, the SBML-loader threshold path, the sensitivity workflow — can
# share one selector without importing the simulator.
_CODEGEN_JIT_BACKENDS: frozenset[str] = frozenset({"mir"})


def _codegen_jit_backend() -> str:
    """Return the selected in-process codegen JIT backend, or '' for the default
    `cc` + dlopen path. Set ``BNGSIM_CODEGEN_JIT=mir`` to JIT the codegen RHS
    in-process via the vendored MIR micro-JIT (GH #78, prototype). An unknown
    value is ignored (falls back to the default backend) with a warning.
    """
    raw = os.environ.get("BNGSIM_CODEGEN_JIT", "").strip().lower()
    if not raw:
        return ""
    if raw in _CODEGEN_JIT_BACKENDS:
        return raw
    logger.warning(
        "Ignoring unknown BNGSIM_CODEGEN_JIT=%r; expected one of %s. Using the "
        "default cc+dlopen codegen backend.",
        raw,
        sorted(_CODEGEN_JIT_BACKENDS),
    )
    return ""


# Default seconds before the cc invocation is killed. Large reaction networks
# emit multi-MB flat RHS sources that take minutes to compile at -O3, so the
# old 60 s ceiling silently aborted codegen on big models (Issue #37).
# Override with BNGSIM_CODEGEN_TIMEOUT.
_DEFAULT_CODEGEN_TIMEOUT = 600

# C sources larger than this compile at the "low" optimization level instead of
# "high": the RHS is one flat arithmetic function, so -O3 costs minutes for
# negligible runtime gain. Override the chosen level with BNGSIM_CODEGEN_OPT
# (an integer level 0-3, or the words "high"/"low"/"none").
_CODEGEN_BIG_SOURCE_BYTES = 1_000_000

# C sources larger than this compile at -O0 (no optimization) instead of -O1.
# -O1's compile time on a single multi-MB flat arithmetic function is superlinear
# and effectively unbounded: fceri_gamma's 23.6 MB combined RHS exceeds even the
# 600 s BNGSIM_CODEGEN_TIMEOUT at -O1, so codegen times out and the model
# silently falls back to the (slower) ExprTk-bytecode RHS — and, because the
# compile never completes, nothing is cached, so every load re-spends the full
# timeout. -O0 compile time is ~linear in source size, so it degrades gracefully
# and still yields a native RHS that beats the bytecode fallback. Measured on a
# 6-core Intel Mac (Apple clang 17): 23.6 MB → -O0 11.4 s vs -O1 >600 s. The
# threshold sits well above Issue #37's Kozer-EGFR repro (4.6 MB, which compiles
# fine at -O1) so only models that currently fail change behavior — no regression
# to the band where -O1 already completes. Override with BNGSIM_CODEGEN_OPT.
_CODEGEN_HUGE_SOURCE_BYTES = 8_000_000

# Per-process counter feeding unique temp filenames for atomic .so installs.
_compile_counter = itertools.count()

# Bump when generate_combined_c output changes for unchanged .net input
# (e.g., when the dfdp/jac_vec/rhs C-emit logic itself changes). The cache
# hash mixes this in so stale .so files are not silently reused.
# v4: CodegenUserData gained tfun_ctx + tfun_eval fields; .net function
# bodies of the form tfun(...) now emit a callback into the C++ runtime
# instead of compile-failing on the undeclared tfun symbol.
# v6: wrapper-form tfun(...) (BNGL functions like `(tfun('drive') + 5)/k`)
# previously emitted invalid C; codegen now extracts every embedded tfun
# call and emits the tfun_eval callback while preserving wrapper math.
# v7: single-pass identifier substitution in _translate_expr / _translate_expr_to_c
# (Issue #25). Whitespace around translated tokens may differ from v6 output;
# bumping the version invalidates v6-vintage cached .so files.
# v9: integer literals in translated expressions are float-ified (``1`` → ``1.0``)
# so C honors ExprTk's double-division semantics — ``(1/2)`` is 0.5, not an
# integer-divided 0. Fixes rate laws with rational constants under codegen
# (MODEL1112100000 Sigma sigmoid froze every Wus_* species at ≥256 species).
# v10: GH #75 amount_valued species. Observable sums and Elementary/Functional
# species factors fold in ∏ V_c^mult for amount_valued (hOSU) reactants
# (mirrors update_observables / compute_species_factor_ode); sens RHS carries
# the same per-reaction amount_factor. Output is byte-identical outside the
# hOSU-V≠1 set, but the generator logic changed, so invalidate v9 cached .so.
# v12: GH #106 rateOf csymbol. The RHS emitter declares current_derivs[] and
# runs a two-pass probe (compute ydot, publish to current_derivs, recompute)
# for models that reference rate_of__<species>; rate_of__ tokens resolve to
# current_derivs[idx]. Byte-identical for non-rateOf models, but the generator
# gained a branch — invalidate v11 cached .so.
# v13: ExprTk max/min now emit C fmax/fmin (previously emitted verbatim, which
# fails to compile — math.h has no max/min). Byte-identical for models that use
# neither; invalidate v12 cached .so for any model that does.
# v14: GH #136 — (a) the combined model-based source gained a
# bngsim_codegen_outputs function (compiled observable/expression evaluation for
# the per-output-row recording path); (b) _emit_function_lines now emits the
# func[] block in topological (dependency) order so a forward-referenced
# assignment rule no longer reads an uninitialised slot. (b) is byte-identical
# for models already in dependency order (the whole real corpus — stable sort),
# but the RHS/Jacobian generator logic changed, so invalidate v13 cached .so.
# v15: Tier-1 large-model chunking — at/above _chunk_threshold() reactions the
# RHS and sensitivity jac_vec bodies are split into NOINLINE helper functions
# (see dev/reaction_rhs_chunking_plan.md). Below the threshold the emitted C is
# byte-identical to v14; the bump is REQUIRED for the .net path, whose cache key
# (compute_model_hash) is content+version, not source — without it a large .net
# model would silently reuse its stale flat .so instead of the chunked one.
# v16: .net parser now preserves the full multi-token reaction rate-law field
# and recognizes BNG's whitespace Michaelis-Menten form ("MM kcat Km"). Without
# the bump, cached v15 .so files for MM .net models could silently keep the
# previously-generated zero-rate RHS.
# v17: GH #160 — chunked NOINLINE blocks gained external linkage (was `static`),
# file-scope prototypes, and shard sentinel comments so compile_rhs can split a
# chunked source into independent translation units and compile them in parallel.
# Non-chunked models are byte-identical (the chunking path is untouched below the
# threshold); chunked sources change (markers + protos + linkage), so v16 chunked
# .so files are invalidated. The cache key is unchanged otherwise.
# v20: GH #198 — the combined .net source gained a bngsim_codegen_output_sens
# function (compiled observable + expression output-sensitivity evaluator). It is
# appended after bngsim_codegen_outputs on the emit_outputs path, so any model
# with observables/functions emits new source; invalidate v19 cached .so.
# v21: lanl/bngsim #5 — every codegen entry point (rhs/jac/jac_sparse/outputs/
# sens_rhs/output_sens) is now tagged BNGSIM_EXPORT so it is visible from the
# built library on Windows (an MSVC/MinGW DLL exports nothing by default, so the
# C++ loader's GetProcAddress failed). The prelude + one macro token per entry
# point change the emitted source on every platform, so invalidate v20 cached .so.
# v22: bngsim._jacobian now accepts ExprTk's symbolic logical operators
# (``&&``/``||``/``&``/``|``) and rewrites all logical forms precedence-safely,
# so a rate law whose condition combines comparisons — the overwhelmingly common
# hand-written .net spelling, ``if(((t>=sigma)&&(t<tau1)),lambda0,0)`` — now
# differentiates instead of falling back. generate_jacobian_from_model therefore
# emits bngsim_codegen_jac for models that previously got none, and the .net
# cache key is content+version (not source), so v21 .so files for those models
# must be invalidated or the stale Jacobian-less .so would be reused.
# v23: lanl/bngsim #56 — the derived-parameter chain rule now handles compound
# conditions (and ``^`` / ``not()``), so the emitted sensitivity RHS gains
# ∂p_d/∂primary terms it previously zeroed; and a derived rate constant that
# cannot be differentiated now declines the analytic sens RHS outright rather
# than emitting one with a hole. Both directions change the emitted source, and
# the .net cache key is content+version (not source), so a v22 .so would keep
# serving the pre-fix numbers — the exact silent-inertness issue #51 documents
# for #41/#43. Invalidate v22.
# v24: lanl/bngsim #68 — a Functional rate law whose ``if()`` conditions are all
# recognized clock thresholds now gets the analytic sensitivity RHS instead of
# declining to CVODES' difference quotient, so eight corpus models emit a
# bngsim_codegen_sens_rhs (with a ternary in both ∂f/∂p and J·v) where v23
# emitted none. Invalidate v23.
# v25: lanl/bngsim #55 — Michaelis–Menten reactions get the analytic sensitivity
# RHS (closed-form ∂rate/∂kcat and ∂rate/∂Km, plus their J·v from the same
# builder the analytical Jacobian uses), so an MM model emits a
# bngsim_codegen_sens_rhs where v24 declined to CVODES' difference quotient.
# Invalidate v24.
# v26: lanl/bngsim #89 — every emitted Michaelis–Menten site changes. The free
# substrate is now the stable quadratic root (the conjugate form for δ < 0), and
# the Jacobian / ∂f/∂p partials are the subtraction-free quotients rather than
# the chain rule through ∂sFree/∂E,S,Km. This moves MM *trajectories*, not only
# gradients, so a cached v25 .so would keep serving pre-fix numbers.
# Invalidate v25.
# v27: lanl/bngsim #93 — every emitted Michaelis–Menten site changes again. The
# free substrate is no longer clamped to 0 and the derivative emitters no longer
# guard on `sFree > 0`; both now guard the rate's own denominator, `Km + sFree`.
# A cached v26 .so would keep serving an RHS whose emitted Jacobian contradicts
# it wherever a species goes negative, and a zero ∂rate/∂S at S = 0.
# Invalidate v26.
# v28: lanl/bngsim #177 — the sensitivity source gains bngsim_dfdp_term_scale and
# the exported bngsim_codegen_sens_term_scale, reporting Σ|term| per row of ∂f/∂p
# alongside the signed sum. Purely additive: bngsim_dfdp, bngsim_jac_vec and
# bngsim_codegen_sens_rhs are byte-identical to v27 (pinned by
# test_sens_term_scale.py::test_the_signed_half_is_byte_identical). But a .net
# model's cache key is content+version, not source, so without the bump a v27 .so
# would be reused and the new symbol would simply be absent — the solver would
# silently keep the unfloored tolerance, which is exactly the silent-inertness
# shape issue #51 documents.
_CODEGEN_VERSION = "28"


# Modules whose *source* determines the emitted C. ``_codegen`` holds the
# emitters; ``_jacobian`` is the symbolic core feeding the Jacobian / sensitivity
# emitters; ``_saturable_jacobian`` is the saturable-rate-law branch _jacobian
# delegates to; ``_switch_sensitivity`` owns the clock-threshold recognizer that
# decides whether a conditional Functional rate law is emitted at all (issue
# #68), so an edit there changes which models get a sensitivity RHS — exactly the
# kind of silent inertness this digest exists to prevent. A change to any of them
# can change the generated source.
_CODEGEN_SOURCE_MODULES = (
    "_codegen",
    "_jacobian",
    "_saturable_jacobian",
    "_switch_sensitivity",
)


def _compute_codegen_source_digest(src_dir: Path | None = None) -> str:
    """Digest of the codegen modules' own source, for the cache key (issue #51).

    ``src_dir`` defaults to this module's directory and exists so the digest's
    react-to-an-edit behavior can be tested without editing the live package.

    The ``.net`` path keys its compiled ``.so`` on the model content plus
    ``_CODEGEN_VERSION`` rather than on the generated C, because hashing the C
    would mean a full source-gen on every cache probe. That made the constant
    load-bearing: a change that altered the emitted sensitivity RHS *without*
    bumping it was invisible to any machine with a warm cache, which kept
    loading the stale ``.so`` and returning the pre-change numbers. #41 and #43
    both shipped that way — on a warm cache they appeared to do nothing at all.

    Folding this digest in makes the omission harmless: editing an emitter
    changes the key whether or not anyone remembers the constant. It is computed
    once at import from three file reads (~350 KB, well under a millisecond) and
    costs nothing per probe, so the fast path stays fast.

    Deliberately conservative in two directions. It hashes source text, so a
    comment-only edit also invalidates — over-invalidation costs one recompile,
    while under-invalidation is a silently wrong gradient. And it covers the
    Python emitters only: a C++ change that alters ``codegen_data()`` is not
    caught here, so ``_CODEGEN_VERSION`` is still the escape hatch for that (and
    for deliberately invalidating a release's caches).

    Returns ``""`` when the sources cannot be read (a ``.pyc``-only or zipped
    install), which degrades the key to ``_CODEGEN_VERSION`` alone — the
    pre-issue-#51 behavior, never something weaker.
    """
    h = hashlib.sha256()
    here = src_dir if src_dir is not None else Path(__file__).resolve().parent
    for name in _CODEGEN_SOURCE_MODULES:
        try:
            data = (here / f"{name}.py").read_bytes()
        except OSError:
            return ""
        h.update(name.encode())
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
    return h.hexdigest()[:16]


_CODEGEN_SOURCE_DIGEST = _compute_codegen_source_digest()

# The token every codegen cache key mixes in. Use this, never ``_CODEGEN_VERSION``
# alone, anywhere a cached artifact's validity is decided (issue #51).
_CODEGEN_CACHE_KEY = f"{_CODEGEN_VERSION}+{_CODEGEN_SOURCE_DIGEST}"

# Accessor-token prefix for the SBML rateOf csymbol (GH #106). MUST match
# _RATEOF_PREFIX in bngsim/_sbml_loader.py and register_rateof_accessors() in
# src/model_impl.hpp — a rate_of__<species> token resolves to current_derivs[i].
_RATEOF_PREFIX = "rate_of__"


# ─── Tier-1 large-model function chunking ───────────────────────────────────
#
# A flat RHS (or sensitivity jac_vec) over N reactions is one enormous basic
# block. The optimizer's per-function / per-basic-block passes are superlinear
# in function size, so at -O1/-O2 a ~100k-reaction model can take many HOURS to
# compile (measured: a synthetic mass-action RHS scales ~O(N^2.5) at -O1 — 95 s
# at 20k reactions, 521 s at 40k). The existing >8 MB → -O0 fallback dodges the
# compile cliff but then ships an UNoptimized RHS the integrator calls millions
# of times. Splitting the reaction body into many small NOINLINE helper
# functions caps basic-block size, making compile time ~linear and letting the
# chunked source compile at -O2 (measured: 40k reactions in ~58 s at -O2 vs the
# flat 521 s at -O1). See dev/reaction_rhs_chunking_plan.md.
#
# Gated by reaction count: below the threshold the emitted C is byte-identical to
# the pre-chunking output, so every model currently in the parity suites is
# untouched (no cache churn, no numerical drift). The split preserves item order
# and calls blocks in order, so accumulation into a shared output array (ydot /
# Jv_out) keeps the flat body's exact floating-point order ⇒ byte-identical
# results above the gate too.

# Sentinel on the first lines of a chunked C source so compile_rhs targets -O2
# regardless of source size (the size-based -O0/-O1 tiers exist only to tame the
# flat giant function, which chunking removes).
_CHUNK_MARKER = "/* bngsim-codegen: chunked */"

# Default reaction count at/above which RHS + sensitivity bodies are chunked.
_DEFAULT_CHUNK_THRESHOLD = 2000

# Default reactions per NOINLINE block. 256 measured ~1.3x faster to compile than
# 1024 on a 40k-reaction synthetic at -O2; smaller blocks compile a little faster
# at the cost of slightly more call overhead.
_DEFAULT_CHUNK_SIZE = 256

# ─── Tier-2 parallel shard compilation (GH #160) ────────────────────────────
#
# A chunked source is one .c translation unit compiled by a single serial `cc`.
# For genome-scale models (>100k reactions) that one compile is the dominant cost
# of Simulator construction (tens of minutes). The NOINLINE blocks are
# already independent functions, so compile_rhs can split them into separate
# translation units, compile them with `cc -c` in parallel, and link the .o's
# into the .so — `make -j` for codegen. Wall-clock drops to ≈ (slowest unit +
# link) wherever multiple cores are available.
#
# Sentinel comments wrap each NOINLINE block so a chunked source can be split
# into units. They are plain C comments, so a source carrying them still
# compiles as a single TU — the serial path (1-core allocation, or
# BNGSIM_CODEGEN_JOBS=1) is unchanged.
_SHARD_BLOCK_OPEN = "/*__BNGSIM_SHARD_BLOCK__*/"
_SHARD_BLOCK_CLOSE = "/*__BNGSIM_SHARD_BLOCK_END__*/"

# NOINLINE blocks per shard translation unit. FIXED (independent of the job
# count) so the source partition — and therefore the linked .so — is identical no
# matter how many compilers run; the job count only sets concurrency. With the
# default 256-reaction blocks this is ~2k reactions of flat arithmetic per unit:
# big enough that compile time dominates process-spawn overhead, small enough to
# spread across cores.
_SHARD_UNIT_BLOCKS = 8

# Estimated peak RSS of one `cc -c` of a shard unit, for the memory cap. A unit
# is a handful of NOINLINE blocks — far smaller than the old whole-model flat
# compile that OOM-killed a 32 GB node — so this is deliberately conservative: we
# would rather under-subscribe RAM than OOM a shared node. Override (MB) with
# BNGSIM_CODEGEN_MEM_PER_JOB.
_DEFAULT_SHARD_MEM_MB = 512

# Portable codegen prelude, emitted once per translation unit. Each macro is
# #ifndef-guarded so the RHS + sensitivity + jacobian + output sources
# concatenated into one shared library don't redefine it:
#   BNGSIM_NOINLINE — keep chunked reaction blocks out of the giant RHS.
#   BNGSIM_EXPORT   — mark the entry points the C++ loader resolves by name so
#                     they are visible from the built library. On Windows an
#                     MSVC/MinGW DLL exports nothing unless the symbol is tagged
#                     __declspec(dllexport), so GetProcAddress("bngsim_codegen_rhs")
#                     failed (lanl/bngsim #5). Unix ELF/Mach-O export global
#                     symbols by default, so there it expands to nothing.
_CODEGEN_PRELUDE_LINES = (
    "#ifndef BNGSIM_NOINLINE",
    "#if defined(__GNUC__) || defined(__clang__)",
    "#define BNGSIM_NOINLINE __attribute__((noinline))",
    "#elif defined(_MSC_VER)",
    "#define BNGSIM_NOINLINE __declspec(noinline)",
    "#else",
    "#define BNGSIM_NOINLINE",
    "#endif",
    "#endif",
    "#ifndef BNGSIM_EXPORT",
    "#if defined(_WIN32)",
    "#define BNGSIM_EXPORT __declspec(dllexport)",
    "#else",
    "#define BNGSIM_EXPORT",
    "#endif",
    "#endif",
)


def _chunk_threshold() -> int | None:
    """Reaction count at/above which to chunk, or None to disable chunking.

    ``BNGSIM_CODEGEN_CHUNK`` overrides: ``off``/``0``/``none``/``false`` disables;
    ``on``/``true`` forces chunking for any reaction count (threshold 1); a
    positive integer sets the threshold.
    """
    raw = os.environ.get("BNGSIM_CODEGEN_CHUNK", "").strip().lower()
    if not raw:
        return _DEFAULT_CHUNK_THRESHOLD
    if raw in ("off", "0", "none", "false"):
        return None
    if raw in ("on", "true"):
        return 1
    try:
        n = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid BNGSIM_CODEGEN_CHUNK=%r; using default threshold %d",
            raw,
            _DEFAULT_CHUNK_THRESHOLD,
        )
        return _DEFAULT_CHUNK_THRESHOLD
    return None if n <= 0 else n


def _chunk_block_size() -> int:
    """Reactions per NOINLINE block (``BNGSIM_CODEGEN_CHUNK_SIZE`` override)."""
    raw = os.environ.get("BNGSIM_CODEGEN_CHUNK_SIZE", "").strip()
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except ValueError:
            pass
        logger.warning(
            "Ignoring invalid BNGSIM_CODEGEN_CHUNK_SIZE=%r; using %d",
            raw,
            _DEFAULT_CHUNK_SIZE,
        )
    return _DEFAULT_CHUNK_SIZE


def _should_chunk(n_reactions: int) -> bool:
    """Whether a body over ``n_reactions`` reactions should be split into blocks."""
    thr = _chunk_threshold()
    return thr is not None and n_reactions >= thr


def _emit_chunked_blocks(
    item_line_groups: list[list[str]],
    *,
    fn_prefix: str,
    signature_params: str,
    call_args: str,
    block_size: int,
    preamble: tuple[str, ...] = (),
) -> tuple[list[str], list[str], list[str]]:
    """Split per-item C line-groups into NOINLINE helper functions.

    ``item_line_groups[i]`` is the list of C lines for one reaction (already
    indented; comments/blank lines included verbatim). Returns
    ``(block_defs, call_lines, proto_lines)``: ``block_defs`` are the complete
    helper definitions (emit at file scope, before the calling function);
    ``call_lines`` are the in-order calls (emit where the inline body used to be);
    ``proto_lines`` are forward declarations (emit before the calling function).

    The helpers have **external linkage** (not ``static``) and each is wrapped in
    ``_SHARD_BLOCK_OPEN``/``_SHARD_BLOCK_CLOSE`` sentinel comments so compile_rhs
    can lift the blocks into separate translation units and compile them in
    parallel (GH #160). The prototypes let the driver TU call the blocks once
    their definitions live in other units. A source carrying the sentinels still
    compiles as one TU (they are comments), so the serial path is unaffected.

    Items keep their original order and blocks are called in order, so any
    accumulation into a shared output array preserves the flat body's exact
    arithmetic order — the chunked RHS/sens is byte-identical to the flat one.
    """
    block_defs: list[str] = []
    call_lines: list[str] = []
    proto_lines: list[str] = []
    n_blocks = (len(item_line_groups) + block_size - 1) // block_size
    width = max(3, len(str(max(n_blocks - 1, 0))))
    for bi, start in enumerate(range(0, len(item_line_groups), block_size)):
        name = f"{fn_prefix}_{bi:0{width}d}"
        proto_lines.append(f"void {name}({signature_params});")
        block_defs.append(_SHARD_BLOCK_OPEN)
        block_defs.append(f"BNGSIM_NOINLINE void {name}({signature_params}) {{")
        for ln in preamble:
            block_defs.append(f"    {ln}")
        for grp in item_line_groups[start : start + block_size]:
            block_defs.extend(grp)
        block_defs.append("}")
        block_defs.append(_SHARD_BLOCK_CLOSE)
        call_lines.append(f"    {name}({call_args});")
    return block_defs, call_lines, proto_lines


# Shard-block signatures for the obs[] / func[] computation lifted off the driver
# (GH #165). An ``obs[i]`` is a linear combination of species, so an obs block
# needs only ``y``; a ``func[i]`` body may read params, species, observables,
# earlier functions, and dispatch a table function through ``data->tfun_eval``, so
# a func block also takes ``t``/``p``/``obs``/the user_data (cast to the RHS
# ``CodegenUserData`` typedef in the block preamble). func blocks are called in
# the same topological order the flat body uses, sharing the ``func`` array, so a
# later block reads earlier blocks' slots.
_OBS_BLK_SIG = "const double* y, double* obs"
_OBS_BLK_ARGS = "y, obs"
_FUNC_BLK_SIG = (
    "double t, const double* y, const double* p, const double* obs, double* func, void* user_data"
)
_FUNC_BLK_ARGS = "t, y, p, obs, func, user_data"
_FUNC_BLK_PREAMBLE = ("CodegenUserData* data = (CodegenUserData*)user_data;",)

# The same func-block split for the *sensitivity* RHS (GH #65), minus the
# ``user_data`` parameter. ``bngsim_codegen_sens_rhs`` is handed a
# ``CodegenSensUserData``, which carries no ``tfun_ctx``/``tfun_eval`` — so a
# whole-body table function (the one construct ``_emit_function_lines`` emits as
# ``data->tfun_eval(...)``) is unreachable from here. Rather than smuggle a
# second user-data shape through the ABI, ``_emit_sens_rhs_body`` declines a
# model whose df/dp needs a tfun-backed function value, mirroring
# ``_analyze_output_sens``'s embedded-tfun decline; the emitted blocks therefore
# never reference ``data``, and the parameter would only ever be dead weight.
_SENS_FUNC_BLK_SIG = "double t, const double* y, const double* p, const double* obs, double* func"
_SENS_FUNC_BLK_ARGS = "t, y, p, obs, func"
_SENS_FUNC_BLK_ARGS_NO_OBS = "t, y, p, NULL, func"


def _shard_value_lines(
    value_lines: list[str],
    *,
    chunk: bool,
    fn_prefix: str,
    signature_params: str,
    call_args: str,
    preamble: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """Split an ``obs[]`` / ``func[]`` computation into a driver part + file-scope
    NOINLINE shard blocks (GH #165).

    ``value_lines`` is the ``_emit_observable_lines`` / ``_emit_function_lines``
    output: the array declaration first, then order-preserving ``X[i] = …;``
    assignments. Returns ``(in_func, file_scope)`` — the lines to emit *inside* the
    calling function (the declaration plus, chunked, calls to the fill blocks, or,
    flat, the inline assignments) and the lines to emit at *file scope* before that
    function (chunked: block prototypes + definitions; flat: empty).

    These large straight-line basic blocks are otherwise the serial driver wall at
    genome scale (each of the RHS, Jacobian, and outputs evaluators recomputes
    them). Lifting them into NOINLINE blocks lets the parallel shard compile
    (GH #160) split them across cores. The flat path is byte-identical to the
    inline emit, so every model below the chunk threshold is untouched.
    """
    if not value_lines:
        return [], []
    decl, assigns = value_lines[0], value_lines[1:]
    if not (chunk and assigns):
        return list(value_lines), []
    block_defs, call_lines, proto_lines = _emit_chunked_blocks(
        [[a] for a in assigns],
        fn_prefix=fn_prefix,
        signature_params=signature_params,
        call_args=call_args,
        block_size=_chunk_block_size(),
        preamble=preamble,
    )
    return [decl, *call_lines], [*proto_lines, "", *block_defs]


# ─── .net file parser (lightweight, Python-only) ────────────────────────


def _parse_net_file(net_path: str) -> dict:
    """Parse a .net file into a dict of model metadata for code generation.

    Returns dict with keys: parameters, species, reactions, observables,
    functions, fixed_species.
    """
    with open(net_path, encoding="utf-8") as f:
        content = f.read()

    result: dict[str, list] = {
        "parameters": [],  # [(index, name, expr_or_value, is_const)]
        "species": [],  # [(index, name, init_conc, is_fixed)]
        "reactions": [],  # [(index, reactants, products, rate_law, comment)]
        "observables": [],  # [(index, name, entries)]  entries=[(factor, sp_idx)]
        "functions": [],  # [(index, name, expression)]
    }

    section = None
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("begin "):
            section = line.split()[1]
            continue
        if line.startswith("end "):
            section = None
            continue

        if section == "parameters":
            result["parameters"].append(_parse_parameter_line(line))
        elif section == "species":
            result["species"].append(_parse_species_line(line))
        elif section == "reactions":
            result["reactions"].append(_parse_reaction_line(line))
        elif section == "groups":
            result["observables"].append(_parse_group_line(line))
        elif section == "functions":
            result["functions"].append(_parse_function_line(line))

    return result


def _validate_net_model_for_codegen(model: dict, net_path: str) -> None:
    """Reject inputs that did not parse as a usable BioNetGen .net model."""
    n_species = len(model.get("species", []))
    n_reactions = len(model.get("reactions", []))
    if n_species > 0 and n_reactions > 0:
        return

    raise ValueError(
        "codegen net_path must point to a BioNetGen .net file with non-empty "
        f"species and reactions sections; parsed {net_path!r} as "
        f"{n_species} species and {n_reactions} reactions. For SBML or "
        "Antimony models, load the model first and use Simulator(..., "
        "codegen=True) without passing the SBML/XML file as net_path."
    )


def _parse_parameter_line(line: str) -> tuple:
    """Parse: '1 kf 0.001  # Constant' -> (1, 'kf', '0.001', True)"""
    # Remove trailing comment
    comment_idx = line.find("#")
    is_const = True
    if comment_idx >= 0:
        comment = line[comment_idx + 1 :].strip()
        is_const = "ConstantExpression" not in comment
        line = line[:comment_idx].strip()
    parts = line.split(None, 2)
    idx = int(parts[0])
    name = parts[1]
    expr = parts[2] if len(parts) > 2 else "0"
    return (idx, name, expr, is_const)


def _parse_species_line(line: str) -> tuple:
    """Parse: '1 A() 100' -> (1, 'A()', '100', False)
    '$' marks a fixed (boundary) species. For cBNGL models BNG2.pl writes
    the marker after the `@compartment::` prefix (e.g. `@CP::$Sink()`); both
    forms are recognized and the `$` is stripped from the stored name.
    """
    parts = line.split()
    idx = int(parts[0])
    name, is_fixed = _strip_fixed_marker(parts[1])
    conc = parts[2] if len(parts) > 2 else "0"
    return (idx, name, conc, is_fixed)


def _strip_fixed_marker(name: str) -> tuple[str, bool]:
    """Return (clean_name, is_fixed). The clamp `$` may sit at position 0
    (`$Sink()`) or right after an `@<compartment>::` prefix (`@CP::$Sink()`).
    """
    if name.startswith("$"):
        return name[1:], True
    if name.startswith("@"):
        sep = name.find("::")
        if sep != -1 and sep + 2 < len(name) and name[sep + 2] == "$":
            return name[: sep + 2] + name[sep + 3 :], True
    return name, False


def _parse_reaction_line(line: str) -> tuple:
    """Parse: '1 1,2 3 kf #_R1' -> (1, [1,2], [3], 'kf', '_R1')"""
    comment = ""
    comment_idx = line.find("#")
    if comment_idx >= 0:
        comment = line[comment_idx + 1 :].strip()
        line = line[:comment_idx].strip()

    parts = line.split()
    idx = int(parts[0])

    # Parse reactants
    reactant_str = parts[1]
    reactants = [int(x) for x in reactant_str.split(",")]

    # Parse products
    product_str = parts[2]
    products = [int(x) for x in product_str.split(",")]

    # Rate law: everything after products. BNG emits multi-token forms such as
    # ``MM kcat Km``; truncating to parts[3] turns them into an unknown
    # elementary parameter and silently emits a zero rate.
    rate_law = " ".join(parts[3:]) if len(parts) > 3 else ""

    return (idx, reactants, products, rate_law, comment)


def _parse_group_line(line: str) -> tuple:
    """Parse: '1 A_tot  1,2*3,5' -> (1, 'A_tot', [(1.0, 1), (2.0, 3), (1.0, 5)])"""
    parts = line.split()
    idx = int(parts[0])
    name = parts[1]
    entries = []
    if len(parts) > 2:
        for token in parts[2].split(","):
            if "*" in token:
                factor_str, sp_str = token.split("*", 1)
                entries.append((float(factor_str), int(sp_str)))
            else:
                entries.append((1.0, int(token)))
    return (idx, name, entries)


def _parse_function_line(line: str) -> tuple:
    """Parse: '1 sat3() k3/(K4+G)' -> (1, 'sat3', 'k3/(K4+G)')"""
    parts = line.split(None, 2)
    idx = int(parts[0])
    # Remove () from function name
    name = parts[1].rstrip("()")
    expr = parts[2] if len(parts) > 2 else "0"
    return (idx, name, expr)


# ─── tfun body recognition ───────────────────────────────────────────────


_TIME_INDEX_NAMES = {"time", "t", "time()", "t()"}


def _recognize_tfun_body(expr: str) -> dict | None:
    """Recognize a BNGL function body that is exactly ``tfun(...)``.

    Returns a dict with keys ``index_name``, ``method``, ``filename`` (or
    None for inline mode), and a list of ``referenced_files`` (resolved
    later against the .net directory by the caller). Returns None if the
    body is not a whole-function tfun call.

    Used in two places: (1) standalone, to classify whole-body tfun
    functions in the codegen loop; (2) by ``_extract_tfun_calls`` to parse
    each individual ``tfun(...)`` substring once located inside a larger
    expression.
    """
    s = expr.strip()
    m = re.match(r"^tfun\s*\((.*)\)\s*$", s, re.DOTALL)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return None

    method = "linear"
    method_match = re.search(r"\bmethod\s*=>\s*['\"]([^'\"]+)['\"]", inner)
    if method_match:
        method = method_match.group(1)
        # Strip the method=>"..." segment (and its leading comma if any)
        # so the remaining tokens are the positional args.
        inner = (inner[: method_match.start()] + inner[method_match.end() :]).strip()
        inner = inner.rstrip(",").strip()

    # Inline mode: tfun([xs], [ys], index)
    if inner.startswith("["):
        # Skip past the two bracket arrays to extract the index name.
        idx_name = "time"
        depth = 0
        i = 0
        bracket_count = 0
        while i < len(inner):
            ch = inner[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    bracket_count += 1
                    if bracket_count == 2:
                        # Index name follows — skip comma+whitespace.
                        j = i + 1
                        while j < len(inner) and inner[j] in ", \t":
                            j += 1
                        idx_token = inner[j:].strip().rstrip(",").strip()
                        if idx_token:
                            idx_name = idx_token
                        break
            i += 1
        return {
            "filename": None,
            "index_name": idx_name,
            "method": method,
            "is_inline": True,
        }

    # File-based mode: tfun('file', [index])
    fn_match = re.match(r"['\"]([^'\"]+)['\"]\s*(?:,\s*(.*))?$", inner)
    if not fn_match:
        return None
    filename = fn_match.group(1)
    rest = (fn_match.group(2) or "").strip().rstrip(",").strip()
    idx_name = rest if rest else "time"
    return {
        "filename": filename,
        "index_name": idx_name,
        "method": method,
        "is_inline": False,
    }


_TFUN_PLACEHOLDER_FMT = "__BNGSIM_TFUN_PH_{idx}__"
_TFUN_PLACEHOLDER_RE = re.compile(r"__BNGSIM_TFUN_PH_(\d+)__")


def _find_close_paren_strict(expr: str, open_pos: int) -> int:
    """Return the index of the ')' that matches '(' at ``expr[open_pos]``,
    or -1 if the parens are unbalanced.

    Distinct from the legacy ``_find_matching_paren`` helper used by
    ``_replace_if_calls`` etc. — that one returns ``len(expr) - 1`` on
    unbalanced input so the caller can keep slicing. tfun extraction needs
    to fail loudly instead, hence the separate function.
    """
    depth = 1
    i = open_pos + 1
    while i < len(expr):
        c = expr[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _extract_tfun_calls(expr: str) -> tuple[str, list[dict]]:
    """Locate every ``tfun(...)`` call inside ``expr`` and replace each with a
    unique placeholder identifier.

    Returns ``(rewritten_expr, calls)``. Each ``calls[k]`` is the dict returned
    by ``_recognize_tfun_body`` for the k-th tfun substring (left-to-right
    order); the rewritten expression contains ``__BNGSIM_TFUN_PH_<k>__`` in
    place of each call so the surrounding arithmetic can be translated to C
    normally before the placeholders are substituted with ``tfun_eval``
    callbacks.

    Whole-word matching: only treat ``tfun`` as the table-function name when
    it is not part of a longer identifier (e.g., ``mytfun`` is ignored).
    """
    calls: list[dict] = []
    out_parts: list[str] = []
    cursor = 0
    pattern = re.compile(r"\btfun\s*\(")
    while True:
        m = pattern.search(expr, cursor)
        if m is None:
            out_parts.append(expr[cursor:])
            break
        out_parts.append(expr[cursor : m.start()])
        open_paren = m.end() - 1  # position of '('
        close_paren = _find_close_paren_strict(expr, open_paren)
        if close_paren < 0:
            raise ValueError(f"unbalanced parentheses in tfun call: {expr!r}")
        call_substr = expr[m.start() : close_paren + 1]
        tspec = _recognize_tfun_body(call_substr)
        if tspec is None:
            raise ValueError(f"failed to parse tfun call: {call_substr!r}")
        placeholder = _TFUN_PLACEHOLDER_FMT.format(idx=len(calls))
        out_parts.append(placeholder)
        calls.append(tspec)
        cursor = close_paren + 1
    return "".join(out_parts), calls


def _classify_tfun_index(
    index_name: str, param_idx: dict, obs_idx: dict, *, use_arrays: bool = False
) -> tuple[str, str]:
    """Resolve a tfun index name to a C expression.

    Returns ``(kind, c_expr)`` where ``kind`` is ``"time"``,
    ``"parameter"``, or ``"observable"``, and ``c_expr`` is the C
    snippet the codegen emits as the second argument of ``tfun_eval``.
    Raises ValueError if the index name doesn't resolve.

    ``use_arrays`` selects an observable index's reference form: the default
    ``obs_<name>`` local (flat .net RHS) or the ``obs[idx]`` array slot used when
    the function computation is sharded into NOINLINE blocks (GH #165), where the
    named locals are not in scope.
    """
    if index_name in _TIME_INDEX_NAMES:
        return ("time", "t")
    if index_name in param_idx:
        return ("parameter", f"p[{param_idx[index_name]}]")
    if index_name in obs_idx:
        ref = f"obs[{obs_idx[index_name]}]" if use_arrays else f"obs_{_safe_c_name(index_name)}"
        return ("observable", ref)
    raise ValueError(f"tfun index '{index_name}' is not time, a parameter, or an observable")


# ─── Rate law classification ─────────────────────────────────────────


def _classify_rate_law(rate_law: str, func_names: set):
    """Classify a rate law string.

    Returns: ('elementary', param_name, stat_factor) or
             ('functional', func_name, stat_factor) or
             ('mm', kcat_name, km_name, stat_factor)

    A rate law is MM if it matches the ``MM ...`` form, Functional if its core
    is a known function name (``func_names``), otherwise Elementary — so only
    ``func_names`` is consulted. Callers must NOT pass the parameter-name set:
    classification never reads it, and building ``set(param_idx)`` per reaction
    is accidentally O(n_reactions × n_params) on genome-scale models (GH #161).
    """
    # Check for stat_factor prefix: "2*kf" or "0.5*kf"
    stat_factor = 1.0
    core = rate_law.strip()
    m = re.match(r"^(\d+(?:\.\d*)?)\*(.+)$", core)
    if m:
        stat_factor = float(m.group(1))
        core = m.group(2).strip()

    # Check if it's MM. BNG .net files use whitespace form ("MM kcat Km");
    # accept the parenthesized form too because older tests and synthetic probes
    # used it.
    mm = re.match(r"^MM\((\w+),\s*(\w+)\)$", core) or re.match(r"^MM\s+(\w+)\s+(\w+)$", core)
    if mm:
        return ("mm", mm.group(1), mm.group(2), stat_factor)

    # Check if it's a function reference
    if core in func_names:
        return ("functional", core, stat_factor)

    # Otherwise it's elementary (parameter reference)
    return ("elementary", core, stat_factor)


# Python keywords (and a couple of always-reserved identifiers) that, when
# used as a BNGL parameter name, break sympy's ``parse_expr`` tokenizer.
# ``lambda`` is the canonical case (TokenError on "lambda *..."); the others
# raise plain SyntaxError. We alias all of them up-front so the Jacobian path
# can produce analytic chain-rule contributions for keyword-named primaries.
# Issue #27.
_PY_KEYWORD_PARAM_NAMES = frozenset(
    {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
)


def _alias_keyword_param(name: str) -> str:
    """Stable alias used when a BNGL primary parameter is named with a Python
    keyword. The alias is whole-word-substituted into the expression before
    ``parse_expr`` and round-tripped back to ``p[idx]`` after differentiation.

    **Which set of names to alias** (GH #108). There are two conventions, and the
    rule separating them is which printer the caller emits through:

    * ``_alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n`` — Python
      keywords only. Correct wherever the derivative is printed by
      :func:`bngsim._jacobian.sympy_to_c`, whose ``resolve`` callback maps every
      symbol to a C reference itself and so never lets sympy print a name at all.
      Aliasing is then needed only to get *into* sympy, which is what
      ``parse_expr``'s tokenizer refuses.
    * :func:`_sympy_symbol_alias_map` — Python keywords **and** C reserved words.
      Required only where ``sp.ccode`` does the printing, because it renames a
      symbol whose name is a C reserved word (``const`` → ``const_``) and the
      name-keyed round trip back to ``p[idx]`` then misses it.

    ``sp.ccode`` is reached from exactly one place in the package —
    :func:`_direct_derived_partials`, which prepares its expression through
    :func:`_prepare_derived_expr` and therefore the wide map. So the narrow sites
    are correct rather than lucky, and ``TestIssue108TheAliasingRule`` asserts
    both halves: that every ``ccode`` call site prepares through the wide map,
    and that a model whose parameter is named ``const``/``restrict``/``int``
    emits byte-identical C to its ordinary-named twin on every emission path.
    """
    return f"_BNG_KW_{name}"


# Identifiers that sympy's C code printer *renames on output*: a symbol whose
# name is in ``CodePrinter.reserved_words`` is printed with the printer's
# ``reserved_word_suffix`` ("_") appended, so ``Symbol("const")`` comes back out
# of ``sp.ccode`` as ``const_``. Nothing downstream declares ``const_``, so the
# name-keyed round-trip below never rewrites it to ``p[idx]`` and the emitted
# sensitivity RHS fails to compile with "use of undeclared identifier". A BNGL
# parameter named ``const`` is real (``ode/pulses_demo_fixed.bngl``), so alias
# these the same way Python keywords are aliased — the alias is not reserved, so
# it survives ccode verbatim and round-trips by name.
#
# This is sympy's own list (1.14 ``C89CodePrinter.reserved_words``, unchanged in
# the C99/C11 subclasses) plus the C99/C11 keywords sympy does not list. Listing
# a name sympy would *not* have mangled is harmless — it just takes the alias
# path — so the superset is the safe direction. ``test_derived_param_jacobian``
# asserts this stays a superset of whatever sympy's table actually holds.
_C_RESERVED_PARAM_NAMES = frozenset(
    {
        # sympy C89CodePrinter.reserved_words
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "entry",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        # C99 / C11 keywords absent from sympy's table
        "_Alignas",
        "_Alignof",
        "_Atomic",
        "_Bool",
        "_Complex",
        "_Generic",
        "_Imaginary",
        "_Noreturn",
        "_Static_assert",
        "_Thread_local",
    }
)


def _sympy_symbol_alias_map(referenced: list[str]) -> dict[str, str] | None:
    """Map each referenced primary parameter name to the sympy symbol name that
    stands in for it while differentiating.

    A name is aliased when leaving it as-is would break the parse-differentiate-
    print round trip:

    * **Python keywords** (issue #27) — ``parse_expr`` cannot tokenize them.
    * **C reserved words** — ``sp.ccode`` renames the symbol on output
      (``const`` → ``const_``), so the printed partial no longer carries a name
      the caller can map back to ``p[idx]``.

    Every other name maps to itself, so expressions over ordinary parameter
    names differentiate and print exactly as before.

    Returns ``None`` if two distinct parameters would share a symbol name (a
    model with both ``const`` and ``_BNG_KW_const``): the round trip could not
    tell them apart, and silently merging them would corrupt the chain rule.
    """
    out = {
        p: (
            _alias_keyword_param(p)
            if p in _PY_KEYWORD_PARAM_NAMES or p in _C_RESERVED_PARAM_NAMES
            else p
        )
        for p in referenced
    }
    if len(set(out.values())) != len(out):
        return None
    return out


def _substitute_symbols_once(c_str: str, replacements: dict[str, str]) -> str:
    """Whole-word-replace every key of ``replacements`` in ``c_str`` in a
    **single** left-to-right pass.

    Applying the replacements one regex at a time is wrong: each rewrite injects
    ``p[idx]`` into the string, and a later pattern can match *inside* text an
    earlier one just wrote. A model with a parameter literally named ``p``
    (``ode/localfunc_2.bngl``) hits this — ``k`` → ``p[0]`` first, then ``\\bp\\b``
    matches the ``p`` of ``p[0]`` and yields ``p[1][0]``, which fails to compile
    with "subscripted value is not an array". One alternation pass never
    revisits substituted text, so injected array references are inert.

    Longest names first so a name that is a prefix of another still loses to the
    longer match (belt-and-suspenders over the ``\\b`` anchors).
    """
    if not replacements:
        return c_str
    pattern = "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True))
    return re.sub(rf"\b(?:{pattern})\b", lambda m: replacements[m.group(0)], c_str)


def _split_top_level_commas(s: str) -> list[str]:
    """Split ``s`` on top-level (paren-depth-zero) commas. No comma → one part."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                # Unbalanced — bail out and let the caller deal with it.
                return [s]
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def _translate_bngl_if_to_piecewise(expr: str) -> str:
    """Rewrite every BNGL ``if(c, t, f)`` substring to sympy
    ``Piecewise((t, c), (f, True))``, recursively, with balanced-paren and
    top-level-comma parsing. Whole-word match on ``if`` so identifiers like
    ``stiff`` are left alone."""
    pattern = re.compile(r"(?<![A-Za-z0-9_])if\s*\(")
    out_parts: list[str] = []
    cursor = 0
    while True:
        m = pattern.search(expr, cursor)
        if m is None:
            out_parts.append(expr[cursor:])
            break
        out_parts.append(expr[cursor : m.start()])
        open_paren = m.end() - 1
        close_paren = _find_close_paren_strict(expr, open_paren)
        if close_paren < 0:
            # Malformed — leave the rest alone; parse_expr will raise.
            out_parts.append(expr[m.start() :])
            break
        inner = expr[open_paren + 1 : close_paren]
        args = _split_top_level_commas(inner)
        if len(args) != 3:
            # Not the BNGL if(c, t, f) shape — emit unchanged and move on.
            out_parts.append(expr[m.start() : close_paren + 1])
            cursor = close_paren + 1
            continue
        c_raw, t_raw, f_raw = (a.strip() for a in args)
        # Recurse so nested ifs are translated too.
        c = _translate_bngl_if_to_piecewise(c_raw)
        t = _translate_bngl_if_to_piecewise(t_raw)
        f = _translate_bngl_if_to_piecewise(f_raw)
        out_parts.append(f"Piecewise(({t}, {c}), ({f}, True))")
        cursor = close_paren + 1
    return "".join(out_parts)


# ─── ExprTk logical operators → sympy And / Or call form ──────────────────
#
# Lives here, next to the ``if()`` → Piecewise rewriter, because both the
# derived-parameter chain rule below and ``_jacobian``'s symbolic core need it
# (issues #53 and #56) and ``_jacobian`` already imports from this module.

# ExprTk spells logical AND / OR three ways each: the word form and the doubled
# and single symbolic forms. Hand-written BNGL ``.net`` conditions overwhelmingly
# use ``&&`` / ``||`` (e.g. ``if(((t>=sigma)&&(t<tau1)),lambda0,0)``). Ordered
# loosest-binding level first, so ``a && b || c`` splits at ``||`` first — ExprTk
# binds ``and`` tighter than ``or``, as does BNGL.
_LOGICAL_LEVELS: tuple[tuple[str, tuple[tuple[str, bool], ...]], ...] = (
    ("Or", (("||", False), ("|", False), ("or", True))),
    ("And", (("&&", False), ("&", False), ("and", True))),
)


_COMMA_TOKEN: tuple[tuple[str, bool], ...] = ((",", False),)


def _depth0_token_spans(s: str, tokens: tuple[tuple[str, bool], ...]) -> list[tuple[int, int]]:
    """Spans of every ``tokens`` occurrence at paren depth 0 in ``s``.

    ``tokens`` is ``(text, is_word)``; a word token additionally requires
    identifier boundaries so ``land`` / ``orbit`` are not split. Longer
    spellings are listed first so ``&&`` is never mistaken for two ``&``.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "(":
            depth += 1
            i += 1
            continue
        if c == ")":
            depth -= 1
            if depth < 0:
                return []  # unbalanced — leave the string alone
            i += 1
            continue
        matched = 0
        if depth == 0:
            for tok, is_word in tokens:
                if not s.startswith(tok, i):
                    continue
                if is_word:
                    before = s[i - 1] if i else ""
                    after = s[i + len(tok)] if i + len(tok) < n else ""
                    if (before.isalnum() or before == "_") or (after.isalnum() or after == "_"):
                        continue
                matched = len(tok)
                break
        if matched:
            spans.append((i, i + matched))
            i += matched
        else:
            i += 1
    return spans if depth == 0 else []


# Cheap pre-filter: a substring with no logical token at all is returned as-is,
# which keeps the rewrite off the hot path *and* stops it descending into deeply
# nested logical-free expressions (e.g. the 354-deep nested-if daily lookup table
# in the mallela2024 COVID model).
_LOGICAL_PRESENT_RE = re.compile(r"[&|]|\band\b|\bor\b")

# Nesting budget for the rewrite. A logical nested deeper than this is
# pathological; exhausting the budget returns the substring untouched, so
# parse_expr raises and the caller falls back — the pre-fix behavior, never a
# wrong derivative. Each level costs ~3 Python frames.
_LOGICAL_REWRITE_BUDGET = 100


def _rewrite_logicals(expr: str) -> str:
    """Rewrite ExprTk logical AND / OR into sympy ``And(...)`` / ``Or(...)`` calls.

    A direct ``&&`` → ``&`` substitution is **not** correct. Python binds ``&``
    tighter than a comparison, so ``a >= b & c < d`` reassociates to
    ``a >= (b & c) < d`` and ``parse_expr`` then raises on the chained
    comparison. Rewriting to the call form preserves ExprTk's precedence
    (comparisons bind tighter than logicals), which is what BNGL means — and it
    holds whether or not the author parenthesized each operand.
    """
    if not _LOGICAL_PRESENT_RE.search(expr):
        return expr
    if expr.count("(") != expr.count(")"):
        return expr  # malformed; parse_expr will raise and the caller falls back
    return _rewrite_logicals_checked(expr, _LOGICAL_REWRITE_BUDGET)


def _split_depth0(s: str, tokens: tuple[tuple[str, bool], ...]) -> list[str] | None:
    """``s`` split at every depth-0 ``tokens`` occurrence, or ``None`` if none."""
    spans = _depth0_token_spans(s, tokens)
    if not spans:
        return None
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(s[cursor:start])
        cursor = end
    parts.append(s[cursor:])
    return [p.strip() for p in parts]


def _rewrite_logicals_checked(s: str, budget: int) -> str:
    if budget <= 0 or not _LOGICAL_PRESENT_RE.search(s):
        return s
    # A depth-0 comma is an argument separator (``Piecewise((v, cond), …)``,
    # ``max(a, b)``), never a logical operand boundary — recurse per argument
    # first so a logical inside one argument cannot swallow the comma.
    args = _split_depth0(s, _COMMA_TOKEN)
    if args is not None:
        return ", ".join(_rewrite_logicals_checked(a, budget - 1) for a in args)
    for fn, tokens in _LOGICAL_LEVELS:
        operands = _split_depth0(s, tokens)
        if operands is None:
            continue
        return (
            f"{fn}(" + ", ".join(_rewrite_logicals_checked(p, budget - 1) for p in operands) + ")"
        )
    return _rewrite_logicals_in_groups(s, budget)


def _rewrite_logicals_in_groups(s: str, budget: int) -> str:
    """Recurse into each depth-0 ``(...)`` group of a string that has no depth-0
    logical operator, so nested conditions are rewritten too."""
    out: list[str] = []
    depth = 0
    start = -1
    for i, c in enumerate(s):
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
            continue
        if c == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                inner = s[start + 1 : i]
                out.append("(" + _rewrite_logicals_checked(inner, budget - 1) + ")")
                start = -1
            continue
        if depth == 0:
            out.append(c)
    return "".join(out)


# Names bound to sympy classes/functions in the derived-parameter local dict.
# A primary parameter sharing one of these names would be shadowed by the class
# and silently differentiate to zero, so the chain rule refuses the expression
# instead (see ``_preprocess_derived_expr``'s callers).
_DERIVED_RESERVED_NAMES = frozenset({"Piecewise", "And", "Or", "Not", "True", "False"})


def _preprocess_derived_expr(expr: str) -> str:
    """Rewrite a derived-parameter (ConstantExpression) string into a form
    ``sympy.parse_expr`` can tokenize.

    The same pipeline ``_jacobian._preprocess_exprtk`` runs for rate laws, minus
    the ``time()`` placeholder (a ConstantExpression is evaluated once, off the
    integration clock): ``if(c,t,f)`` → ``Piecewise``, ``^`` → ``**``,
    ``not(x)`` → ``Not(x)``, and logical AND / OR → sympy ``And``/``Or`` call
    form. Anything this pass leaves untranslated makes ``parse_expr`` raise, and
    the caller then drops the chain rule for that parameter (issue #56).
    """
    s = _translate_bngl_if_to_piecewise(expr)
    s = s.replace("^", "**")
    s = re.sub(r"\bnot\s*\(", "Not(", s)
    return _rewrite_logicals(s)


def _warn_sens_rhs_refused(name: str, expr: str, reason: str) -> None:
    """Report that the analytic sensitivity RHS was declined because a derived
    rate constant could not be differentiated (issue #56).

    This is the *good* outcome — the run falls back to CVODES' internal
    difference quotient and the gradient stays correct, just slower — but it is
    worth saying out loud, because the alternative the caller is avoiding is a
    sensitivity column of exact zeros that looks like a converged answer.
    """
    logger.warning(
        "Forward sensitivity: the derived rate constant %s = %r could not be "
        "differentiated (%s), so the analytic sensitivity RHS is declined for "
        "this model and CVODES' internal difference quotient is used instead "
        "(correct, but slower).",
        name,
        expr,
        reason,
    )


def _warn_chain_rule_dropped(expr: str, referenced: list[str], reason: str) -> None:
    """Report a derived-parameter expression whose chain rule could not be
    differentiated even though it *does* reference primary parameters.

    Used on the initial-condition seeding path, where — unlike the sensitivity
    RHS — there is nothing to fall back to: the seed ∂x_i(0)/∂primary is either
    computed here or left at zero. Issue #56: the caller reads a missing partial
    as a real zero, indistinguishable from a primary that genuinely does not
    appear, so this warning is the only signal that separates the two cases.

    It is deliberately not an exception. The seeding scan covers every
    parameter-referenced initial condition in the model, most of which have
    nothing to do with the requested sensitivity parameters, and raising would
    refuse models that simulate correctly today.
    """
    logger.warning(
        "Forward sensitivity: could not differentiate the derived parameter "
        "expression %r (%s). The chain rule through it is dropped, so "
        "sensitivities w.r.t. %s will read as exactly zero along this path "
        "even if they are not.",
        expr,
        reason,
        ", ".join(sorted(referenced)),
    )


def _inline_derived_param_refs(
    expr: str,
    derived_exprs: dict[str, str],
    max_passes: int = 64,
) -> str:
    """Recursively inline references to *derived* (ConstantExpression)
    parameters in ``expr`` until only primary parameters remain.

    ``derived_exprs`` maps each derived parameter name to its defining
    expression string. A derived parameter whose expression references another
    derived parameter — e.g. a detailed-balance constraint ``a2prime =
    f(a1prime)`` where ``a1prime = kcr`` — is flattened here so the caller sees
    an expression in primary parameters only (issue #41).

    **Do not flatten in order to differentiate** (GH #99). The substitution is
    exponential in the depth of the DAG: in ``ode/synthesis_v3`` a 43-character
    derived parameter flattens to 20 KB, its dependent to 40 KB, and a single
    ``sp.diff`` on the result never returns. The forward-sensitivity chain rule
    — which is what #41 added this for — now walks the DAG instead
    (:func:`_derived_param_jacobian_dag` and its numeric twin), reaching the
    same nested parameters from the expressions as written. What remains here is
    the *textual* use: deciding whether a switch-time threshold ultimately
    mentions a parameter at all, where nothing is differentiated and the
    flattened string is only scanned.

    Each substitution is whole-word (``\\b``-anchored) and parenthesized to
    preserve operator precedence. The per-pass scan first checks which derived
    names are actually present, so the overwhelmingly common single-level case
    (an expression already in primaries) returns after one tokenize with the
    string untouched — byte-identical to the pre-#41 output. A bounded pass
    count guards against reference cycles in an ill-formed .net: if a derived
    name still remains after ``max_passes``, the string is returned as-is and
    the caller's free-symbol check falls back to the no-analytic-Jacobian path.
    """
    if not derived_exprs:
        return expr
    s = expr
    for _ in range(max_passes):
        present = {t for t in re.findall(r"[A-Za-z_]\w*", s)} & derived_exprs.keys()
        if not present:
            break
        # Longest names first so an outer name is never partially rewritten by a
        # shorter one sharing a prefix (belt-and-suspenders over the \b anchors).
        for name in sorted(present, key=len, reverse=True):
            s = re.sub(rf"\b{re.escape(name)}\b", f"({derived_exprs[name]})", s)
    return s


def _check_derivation_deadline(deadline: float | None) -> None:
    """Raise :class:`_DerivationBudgetExceeded` if the build-time symbolic
    derivation has run past ``deadline`` (GH #90).

    ``deadline`` is a ``time.perf_counter()`` stamp, or ``None`` for unbounded —
    which is what every caller outside the sensitivity build passes, so this is a
    no-op for them. One spelling for every check site so the sensitivity budget
    cannot drift from the Jacobian's, whose per-observable check
    (``_jacobian.differentiate_rate_law``) this mirrors.
    """
    if deadline is not None and time.perf_counter() > deadline:
        from bngsim._jacobian import _DerivationBudgetExceeded

        raise _DerivationBudgetExceeded


def _sens_derivation_deadline(n_species: int) -> float | None:
    """Absolute ``time.perf_counter()`` deadline for one sensitivity-RHS build, or
    ``None`` when the budget is disabled (GH #90).

    Resolved once per :func:`generate_sens_from_model` / :func:`generate_sens_rhs_c`
    call and threaded down, so every ``sp.diff`` on the ∂f/∂p path shares a single
    wall-clock bound instead of each site getting its own.
    """
    from bngsim._jacobian import _sens_derivation_budget_s

    budget = _sens_derivation_budget_s(n_species=n_species)
    return None if budget is None else time.perf_counter() + budget


def _sens_budget_cache_tag() -> str:
    """Cache-key fragment for an explicitly overridden sensitivity derivation
    budget (GH #90), or ``""`` when the env var is unset.

    The budget decides whether a model gets an analytic sensitivity RHS at all, so
    it belongs in the key of any cache that is not content-addressed on the
    generated source — the ``.net`` path's in-process memo and its on-disk
    ``model_hash``, which both key on the .net's *content*. Without it a build made
    under a deliberately tight budget would be served back to one made without it,
    the same trap ``functional_sens_rhs_enabled`` already sidesteps (GH #67).
    Empty when unset, so the default key — and every ``.so`` already cached — is
    byte-identical to before.

    This does not make a *default*-budget expiry cache-safe: the budget is
    wall-clock, so a model that derives near the limit can emit on one run and
    decline on the next, and whichever came first is what the .net path cached.
    Raising the budget is the fix, and doing so lands in a fresh namespace.
    """
    from bngsim._jacobian import _SENS_BUDGET_ENV

    raw = os.environ.get(_SENS_BUDGET_ENV)
    return "" if raw is None else f":sens_budget={raw.strip().lower()}"


def _sens_budget_decline_reason(n_species: int, progress: str) -> str:
    """The decline reason for a build-time ∂f/∂p derivation that ran past its
    budget (GH #90), phrased for :func:`_warn_functional_sens_rhs_refused`.

    A budget expiry is reported through that same channel as every other decline
    rather than one of its own, and names both how far the derivation got and the
    override — the alternative to declining is a build that appears to hang, which
    is the whole point of the budget.
    """
    from bngsim._jacobian import _SENS_BUDGET_ENV, _sens_derivation_budget_s

    budget = _sens_derivation_budget_s(n_species=n_species)
    return (
        f"the build-time ∂f/∂p derivation exceeded its {budget:g}s budget "
        f"({progress}); set {_SENS_BUDGET_ENV} to raise or disable it (seconds, or "
        "inf/none/0 for unbounded)"
    )


def _output_sens_derivation_steps(
    relevant: set[int],
    n_syms: list[int],
    derived_exprs: dict[str, str],
    param_names: set[str],
) -> int:
    """How much sympy the #198 analysis is about to run, in steps (GH #97).

    One step per expression parsed, plus one per (expression, symbol it directly
    references) pair — over both phases the analysis runs: the derived-parameter
    chain rule (``∂p_d/∂primary``) and the per-function partials (``∂f/∂s``).
    ``n_syms`` is the per-function referenced-symbol count the reference-graph pass
    already tokenized for, so the function half costs nothing beyond a set
    intersection there and — the point — needs no sympy to know how much sympy is
    coming.

    The parse term is what makes this a good predictor rather than a rough one.
    Counting only the pairs charges nothing for ``_exprtk_to_sympy`` and
    ``sympy_to_c``, which run per *expression*, so a corpus model whose functions
    each read one symbol looked twice as expensive per unit as one whose functions
    read ten. Over the BioModels models above the knee, adding it takes the
    measured spread from 1.2-11.0 ms to 0.9-5.5 ms per unit — the difference
    between ~4x and ~9x headroom under the same slope.

    An upper bound, not an identity: sympy's ``free_symbols`` can be a subset (a
    symbol that cancels out of the parsed expression takes no ``diff``). A budget
    wants the bound in that direction — over-counting buys time, and the quantity
    is a *size* proxy, not an accounting.

    The derived half became a tight bound with GH #99. It counts one parse per
    derived expression and one ``diff`` per symbol that expression *names*, which
    is what the DAG walk now does; before, the phase differentiated each derived
    parameter's whole DAG flattened into one expression, so the count bore no
    fixed relation to the work — and on ``ode/synthesis_v3`` the work was
    unbounded.
    """
    n = len(relevant) + len(derived_exprs)
    for i in relevant:
        n += n_syms[i]
    for expr in derived_exprs.values():
        n += len(set(re.findall(r"[A-Za-z_]\w*", expr)) & param_names)
    return n


def _output_sens_derivation_deadline(n_steps: int = 0) -> float | None:
    """Absolute ``time.perf_counter()`` deadline for one #198 output-sensitivity
    analysis, or ``None`` when the budget is disabled (GH #97).

    Resolved once per :func:`_analyze_output_sens` — which is memoized on the
    model, so once per model — and threaded down, so every ``sp.diff`` on the
    ``d func/dθ`` path shares a single wall-clock bound. ``n_steps`` is
    :func:`_output_sens_derivation_steps`' count of the sympy to come.

    Deliberately a **separate** deadline from :func:`_sens_derivation_deadline`'s,
    even though both read ``BNGSIM_SENS_DERIV_BUDGET_S``. The two phases run on the
    same build, so one deadline would let a slow ∂f/∂p starve this one — and that
    is measured, not hypothetical: ``BIOMD0000000497`` spends 8.1 s in ∂f/∂p and
    19.2 s here, so a shared 20 s deadline would leave this phase 11.9 s for 19.2 s
    of work and cut a model that today derives in full. Sharing would have been a
    behaviour change on real corpus models, which is exactly what #90 avoided.
    """
    from bngsim._jacobian import _output_sens_derivation_budget_s

    budget = _output_sens_derivation_budget_s(n_steps=n_steps)
    return None if budget is None else time.perf_counter() + budget


def _output_sens_budget_reason(progress: str, n_steps: int = 0) -> str:
    """The per-function ``unsupported`` reason for a #198 output-sensitivity
    derivation that ran past its budget (GH #97).

    Carried on the function's ``func_info`` exactly like an undifferentiable
    construct's reason: the emitted C gets a NaN sentinel and the ``Result`` raises
    this string when a selector asks for that function's sensitivity. It names how
    far the analysis got and the override, because unlike a construct reason this
    one is about the clock, and re-running with a bigger budget is the fix.
    """
    from bngsim._jacobian import _SENS_BUDGET_ENV, _output_sens_derivation_budget_s

    budget = _output_sens_derivation_budget_s(n_steps=n_steps)
    # Finite by construction (an unbounded budget never expires); phrased without
    # the number rather than crashing if the env var moved mid-analysis.
    limit = "its budget" if budget is None else f"its {budget:g}s budget"
    return (
        f"the build-time d(function)/dθ derivation exceeded {limit} ({progress}); "
        f"set {_SENS_BUDGET_ENV} to raise or disable it (seconds, or inf/none/0 "
        "for unbounded)"
    )


def _compute_derived_param_jacobian(
    expr: str,
    primary_param_names: set,
    param_idx: dict,
    derived_exprs: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """Compute ∂(expr)/∂primary as a C source string for each primary that
    appears in ``expr``.

    Used to chain-rule through a derived (ConstantExpression) rate-constant
    parameter ``p_d = expr`` where ``expr`` is an arbitrary arithmetic
    expression in primary parameter names. Returns ``None`` if sympy is
    unavailable or the expression cannot be parsed; the caller then treats
    ``p_d`` as an independent rate constant (``∂p_d/∂primary = 0``).

    ``derived_exprs`` (optional) maps every derived parameter name to its
    defining expression. When supplied, a nested derived reference in ``expr``
    is chain-ruled through rather than rejected (issue #41), so ``p_d = f(p_e)``
    with ``p_e`` itself derived still yields the full chain rule — see
    :func:`_derived_param_jacobian_dag` for how the DAG is walked (GH #99).
    Omitting it (or passing ``None``) preserves the pre-#41 behavior of
    rejecting any non-primary free symbol.

    Two preprocessing passes (issues #27, #56) widen the set of expressions that
    yield an analytic Jacobian instead of the silent zero-contribution fallback
    — the ExprTk-to-sympy surface rewrite and the parameter-name aliasing. Both
    live in :func:`_prepare_derived_expr`, which is where they are described and
    where the numeric twin picks them up from too.

    A ``None`` here is indistinguishable downstream from a genuine zero, so
    callers that cannot afford that ambiguity use
    :func:`_derived_param_jacobian_checked` instead, which also reports *why*
    (issue #56).

    Returns
    -------
    dict[str, str] or None
        ``{primary_name: c_expr_for_partial}`` covering every primary whose
        partial derivative is non-zero. Primary names appearing in the C
        expression have already been rewritten as ``p[idx]``.
    """
    return _derived_param_jacobian_checked(expr, primary_param_names, param_idx, derived_exprs)[0]


def _derived_param_jacobian_checked(
    expr: str,
    primary_param_names: set,
    param_idx: dict,
    derived_exprs: dict[str, str] | None = None,
    deadline: float | None = None,
    cache: dict[str, tuple[dict[str, str] | None, str | None]] | None = None,
    name: str | None = None,
) -> tuple[dict[str, str] | None, str | None]:
    """:func:`_compute_derived_param_jacobian`, plus the reason it gave up.

    Returns ``(jacobian, failure_reason)``. ``failure_reason`` is ``None`` when
    the Jacobian was computed *and* when the expression legitimately has no
    partial to compute (it references no primary at all, e.g. ``_rateLaw1 = 2``);
    it is a human-readable string only when a real chain-rule contribution was
    lost. Issue #56: those two outcomes are both ``jacobian is None``, and the
    sensitivity RHS reads that as ``∂p_d/∂primary = 0`` — so a caller that emits
    an analytic sensitivity RHS must refuse the whole RHS on a failure (falling
    back to CVODES' correct-but-slower internal difference quotient) rather than
    ship a gradient component that is confidently, exactly wrong.

    ``deadline`` (GH #90) is a ``time.perf_counter()`` stamp bounding the caller's
    whole build-time derivation. It is checked on entry to every expression the
    DAG walk visits (parsing one is itself unbounded work) and again before each
    ``sp.diff``, so a single pathological expression overshoots by at most one
    partial. Expiry raises :class:`bngsim._jacobian._DerivationBudgetExceeded`
    rather than returning a reason: it is a property of the *build*, not of this
    expression, and it must unwind past the per-expression caches to decline the
    whole model. Callers that pass no deadline (the default) never see it.

    ``cache`` (GH #99) is an optional ``{derived_name: (jacobian, reason)}`` map
    shared across a caller's loop over many derived parameters, so a DAG node is
    differentiated once for the whole build rather than once per parameter that
    reaches it. Pass one from any loop; omit it and the walk still memoizes
    within this single call. ``name`` is the derived parameter ``expr`` defines,
    when it is one: it puts the top-level result in that same cache, and starts
    the cycle guard's stack, so a loop over every derived parameter costs one
    derivation per DAG node rather than two.
    """
    s = expr.strip()
    if not s:
        return None, None
    _check_derivation_deadline(deadline)
    try:
        import sympy  # noqa: F401
        from sympy.parsing.sympy_parser import parse_expr  # noqa: F401
    except ImportError:
        # No sympy at all is an environment fact, not a property of this
        # expression — every derived parameter is affected equally and the
        # caller's own sympy import has already failed.
        return None, None

    memo = {} if cache is None else cache
    if name is not None and name in memo:
        return memo[name]
    result = _derived_param_jacobian_dag(
        s,
        set(primary_param_names),
        param_idx,
        derived_exprs or {},
        deadline,
        memo,
        () if name is None else (name,),
    )
    if name is not None:
        memo[name] = result
    return result


def _derived_param_jacobian_dag(
    expr: str,
    primary_names: set[str],
    param_idx: dict,
    derived_exprs: dict[str, str],
    deadline: float | None,
    cache: dict[str, tuple[dict[str, str] | None, str | None]],
    stack: tuple[str, ...],
) -> tuple[dict[str, str] | None, str | None]:
    """``∂expr/∂primary`` for every primary ``expr`` reaches, by walking the
    derived-parameter DAG instead of flattening it (GH #99).

    Issue #41 reached a *nested* derived parameter by textually inlining it —
    and everything it depends on — before handing one expression to sympy. That
    is exponential in the depth of the DAG: in
    ``ode/synthesis_v3`` a 43-character derived parameter inlines to 20 KB, its
    dependent to 40 KB, and because the nesting lands in an *exponent* the single
    ``sp.diff`` on the result never returns (and, being one uninterruptible sympy
    call, is immune to #97's wall-clock budget).

    So take the derivative where it is small and compose:

        ∂p_d/∂primary = (∂p_d/∂primary)_direct
                        + Σ_k (∂p_d/∂s_k)·(∂s_k/∂primary)

    over the derived parameters ``s_k`` that ``p_d`` names *directly*. Each
    factor is a partial of an as-written expression, so the sympy stays on
    43-character inputs (8.0 ms for the whole of ``Fh``, against a derivation
    that does not finish); ``∂p_d/∂s_k`` prints ``s_k`` as ``p[idx]``, which the
    runtime already holds because the emitted value code reads a derived
    parameter from the same slot. This is the shape ``_functional_dfdp_terms``
    has always used for a rate law that names a derived parameter — this
    function was the one place that still flattened.

    ``cache`` memoizes each derived parameter's completed table, so a diamond in
    the DAG is differentiated once. ``stack`` is the chain of derived names
    currently being expanded: re-entering one is a reference cycle in an
    ill-formed ``.net`` (what ``_inline_derived_param_refs``' ``max_passes``
    bound guarded against), reported as a reason rather than recursed into.

    Keys are sorted, and a single-level expression takes no composition at all,
    so a derived parameter already written in primaries emits byte-identical C
    to the pre-#99 flattening path.
    """
    direct, reason = _direct_derived_partials(
        expr, primary_names, derived_exprs.keys(), param_idx, deadline
    )
    if reason is not None or not direct:
        return None, reason

    terms: dict[str, list[str]] = {}
    for name, d_c in direct.items():
        if name not in derived_exprs:
            terms.setdefault(name, []).append(d_c)
            continue
        if name in stack:
            return None, f"reference cycle through the derived parameter {name!r}"
        hit = cache.get(name)
        if hit is None:
            hit = _derived_param_jacobian_dag(
                derived_exprs[name],
                primary_names,
                param_idx,
                derived_exprs,
                deadline,
                cache,
                stack + (name,),
            )
            cache[name] = hit
        sub, why = hit
        if why is not None:
            # A lost sub-Jacobian is a lost chain rule for every primary that
            # reaches it, and the caller reads a missing partial as a hard zero
            # (issue #56) — so it fails the whole expression, not just this term.
            return None, f"through the derived parameter {name!r}: {why}"
        for prim, sub_c in (sub or {}).items():
            terms.setdefault(prim, []).append(f"({d_c})*({sub_c})")

    out = {prim: " + ".join(t) for prim, t in sorted(terms.items())}
    return (out or None), None


class _PreparedDerivedExpr(NamedTuple):
    """A derived-parameter expression parsed and ready to differentiate."""

    referenced: list[str]
    """Parameter names the expression names directly, sorted — its
    differentiation variables."""
    sym_name_of: dict[str, str]
    """Parameter name → the sympy symbol name standing in for it."""
    sym_map: dict
    """Sympy symbol name → the ``sp.Symbol`` bound for it."""
    sym_expr: object
    """The parsed sympy expression."""


_WORD_RUN_RE = re.compile(r"\w+")
_HAS_NON_WORD_RE = re.compile(r"\W")


def _names_referenced_in(text: str, names) -> list[str]:
    """Sorted members of ``names`` that occur in ``text`` as whole words.

    Equivalent to ``sorted(n for n in names if re.search(rf"\\b{re.escape(n)}\\b",
    text))``, but linear in ``len(text)`` rather than a fresh regex per candidate
    name — which is what it costs at scale, because a model's parameter count
    outruns ``re``'s internal 512-pattern cache and every search then *recompiles*
    its pattern. Smith_BMCSystBiol2013 (922 parameters, 89 derived expressions)
    compiled ~82,000 throwaway patterns here per pass, ~0.9 s of a 1.9 s
    ``Simulator(...)`` construction (GH #165).

    The equivalence: a name made only of word characters matches ``\\bname\\b``
    exactly when it is one of ``text``'s maximal ``\\w+`` runs — the ``\\b``
    anchors say precisely that the characters either side are not word characters
    — so one tokenizing pass answers the question for every such name at once. A
    name carrying a non-word character (no SBML or BNGL identifier does, but
    nothing here guarantees it) is not a maximal run and cannot be found that
    way, so those names keep the per-name search.
    """
    tokens = set(_WORD_RUN_RE.findall(text))
    referenced = {n for n in names if n in tokens}
    exotic = [n for n in names if n not in referenced and _HAS_NON_WORD_RE.search(n)]
    referenced.update(n for n in exotic if re.search(rf"\b{re.escape(n)}\b", text))
    return sorted(referenced)


def _prepare_derived_expr(
    expr: str,
    primary_names: set[str],
    derived_names,
    allow_no_reference: bool = False,
) -> tuple[_PreparedDerivedExpr | None, str | None]:
    """Parse a derived-parameter expression into sympy, ready for ``sp.diff``
    w.r.t. every parameter it names directly.

    The single preparation sequence shared by the C-emitting chain rule
    (:func:`_direct_derived_partials`), its numeric twin
    (:func:`_direct_derived_partials_numeric`), and the threshold *evaluation*
    that :func:`_derived_expr_value_numeric` backs — which differ only in what
    they do with the parsed expression. Sharing it is the point: this pipeline
    has picked up fixes one site at a time before (#27's keyword aliasing, #56's
    logical rewrite, #105's alias map on the threshold value), and a fix that
    lands on one twin and not the other reads downstream as a hard zero (GH #108).

    Two preprocessing passes (issues #27, #56) widen the set of expressions that
    yield an analytic Jacobian instead of the silent zero-contribution fallback:

    1. The ExprTk surface syntax is rewritten for sympy by
       :func:`_preprocess_derived_expr`: BNGL ``if(c, t, f)`` becomes
       ``Piecewise((t, c), (f, True))`` so sympy differentiates the conditional
       analytically (the boundary delta is sympy's standard Piecewise
       convention), ``^`` becomes ``**``, and logical operators become sympy
       ``And``/``Or``/``Not`` calls. Without the logical rewrite a *compound*
       condition — ``if((sel>=1)&&(sel<10), kA, kB)``, in any of ExprTk's six
       spellings — failed to parse and its whole chain rule was silently zeroed
       (issue #56).
    2. Parameter names that would not survive the parse-differentiate-print
       round trip are aliased to safe placeholders — Python keywords (e.g.
       ``lambda`` in ``ode/scaling_example.bngl``, which ``parse_expr`` cannot
       tokenize) and C reserved words (which ``sp.ccode`` renames on output).

    ``derived_names`` are differentiation variables here just like the primaries
    (GH #99): the caller chain-rules through them rather than inlining them, so
    a nested derived parameter is an ordinary symbol rather than an unresolved
    one.

    Returns ``(prepared, None)``, ``(None, None)`` when the expression names no
    parameter at all (a genuine zero, nothing to differentiate), or
    ``(None, reason)`` when a real partial was lost.

    ``allow_no_reference`` prepares that no-parameter expression instead of
    declining it, for the caller that wants its *value* rather than its
    derivative: ``time >= 2*3600`` has no partial worth reporting but does have a
    crossing time. The differentiating callers keep the default, so an expression
    with nothing to differentiate still returns before the parse — which is what
    keeps an unparseable constant a genuine zero for them rather than a new
    failure reason.
    """
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    # Pass 1: ExprTk → sympy surface syntax (if→Piecewise, ^→**, logicals →
    # And/Or call form). Applied to the raw string so whole-word matching of
    # ``if`` sees the source as written.
    s_pre = _preprocess_derived_expr(expr)

    diff_names = primary_names | set(derived_names)
    referenced = _names_referenced_in(s_pre, diff_names)
    if not referenced and not allow_no_reference:
        return None, None  # names no parameter — a genuine zero, not a failure

    # A parameter named like one of the sympy classes we bind below would be
    # shadowed by the class and differentiate to a silent zero. Refuse instead.
    if not _DERIVED_RESERVED_NAMES.isdisjoint(referenced):
        kind = (
            "a derived"
            if not _DERIVED_RESERVED_NAMES.isdisjoint(set(referenced) - primary_names)
            else "a primary"
        )
        return None, f"{kind} parameter shadows a sympy name"

    # Pass 2. Sort by length descending so e.g. an ``if_thresh`` param is not
    # partially matched by the alias of ``if``.
    sym_name_of = _sympy_symbol_alias_map(referenced)
    if sym_name_of is None:
        return None, "two parameters collide on the same sympy alias"
    s_aliased = s_pre
    for p_name in sorted(referenced, key=len, reverse=True):
        if sym_name_of[p_name] != p_name:
            s_aliased = re.sub(
                rf"\b{re.escape(p_name)}\b",
                sym_name_of[p_name],
                s_aliased,
            )

    # Bind every parameter's sympy symbol so sympy never reaches for built-in
    # constants or functions of the same name (e.g., ``E``, ``S``). Also bind
    # ``Piecewise`` so the if-translation in pass 1 resolves to sympy's class.
    sym_map: dict[str, sp.Symbol] = {sym_name_of[p]: sp.Symbol(sym_name_of[p]) for p in referenced}
    local_dict: dict = dict(sym_map)
    local_dict.update(Piecewise=sp.Piecewise, And=sp.And, Or=sp.Or, Not=sp.Not)

    try:
        sym_expr = parse_expr(s_aliased, local_dict=local_dict, evaluate=True)
    except Exception as exc:
        # Anything still unparseable (malformed BNGL, unsupported call, etc.).
        return None, f"{type(exc).__name__}: {exc}"

    # Reject if the expression introduced any free symbol that is neither a
    # primary nor a derived parameter (a species name reached through a
    # threshold expression, say — out of scope for this chain rule).
    allowed_sym_names = {sym_name_of[p] for p in referenced}
    free = {str(sym) for sym in sym_expr.free_symbols}
    if not free.issubset(allowed_sym_names):
        return None, f"unresolved symbol(s) {sorted(free - allowed_sym_names)}"

    return _PreparedDerivedExpr(referenced, sym_name_of, sym_map, sym_expr), None


def _direct_derived_partials(
    expr: str,
    primary_names: set[str],
    derived_names,
    param_idx: dict,
    deadline: float | None,
) -> tuple[dict[str, str] | None, str | None]:
    """``∂expr/∂s`` as a C source string for every parameter ``expr`` names
    **directly** — primary or derived alike, with no inlining.

    The one sympy round trip in the DAG walk above: parse
    (:func:`_prepare_derived_expr`), differentiate w.r.t. each referenced name,
    print as C, and rewrite the (possibly aliased) symbols back to ``p[idx]``. A
    derived name is an ordinary differentiation variable here, and prints as the
    ``p[idx]`` slot the runtime already keeps its value in;
    :func:`_derived_param_jacobian_dag` supplies its ``∂s_k/∂primary``.

    Returns ``({name: c_expr}, None)`` over the names with a non-zero partial,
    ``(None, None)`` when the expression names no parameter at all (a genuine
    zero), or ``(None, reason)`` when a real partial was lost.
    """
    import sympy as sp

    _check_derivation_deadline(deadline)

    prep, reason = _prepare_derived_expr(expr, primary_names, derived_names)
    if prep is None:
        return None, reason

    # For round-tripping the ccode output: map each (possibly-aliased) sympy
    # symbol name back to ``p[idx]`` using the ORIGINAL parameter's index.
    # Applied as one pass (see :func:`_substitute_symbols_once`) so the ``p[idx]``
    # text a rewrite injects is never itself rewritten by a parameter named ``p``.
    cref_of_sym: dict[str, str] = {
        prep.sym_name_of[p]: f"p[{param_idx[p]}]" for p in prep.referenced if p in param_idx
    }

    result: dict[str, str] = {}
    for p_name in prep.referenced:
        _check_derivation_deadline(deadline)
        deriv = sp.diff(prep.sym_expr, prep.sym_map[prep.sym_name_of[p_name]])
        if deriv == 0:
            continue
        try:
            c_str = sp.ccode(deriv)
        except Exception as exc:
            # A derivative sympy cannot render as C (an un-inlined user function
            # call, erf, DiracDelta, ...). Refuse the whole expression rather
            # than emit a partial chain rule — and never let it escape, since an
            # exception here aborts the entire codegen build.
            return None, f"not expressible in C ({exc})"
        result[p_name] = _substitute_symbols_once(c_str, cref_of_sym)
    return (result or None), None


def _derived_expr_partials_numeric(
    expr: str,
    primary_param_names: set,
    param_idx: dict,
    param_values: list,
    derived_exprs: dict[str, str],
    warn_on_failure: bool = True,
) -> dict[str, float]:
    """Numeric ∂(expr)/∂primary at the nominal parameter values, for every
    primary ``expr`` reaches through the derived-parameter DAG.

    The IC counterpart of :func:`_compute_derived_param_jacobian` (issue #43):
    a species initial condition set by a derived (ConstantExpression) parameter
    ``d = expr`` seeds ∂x_i(0)/∂primary with the *numeric* partial ∂expr/∂primary
    evaluated at the current parameter point — a constant (``Rtot = R0`` → 1,
    ``Rtot = 2*R0`` → 2, ``Rtot = R0*scale`` → nominal ``scale``). Shares the
    parse preparation (:func:`_prepare_derived_expr`) and the DAG walk
    (:func:`_derived_param_jacobian_dag`'s numeric twin below) with the
    rate-constant chain rule, but multiplies floats instead of concatenating C.

    GH #99: like that twin, it chain-rules *through* a nested derived parameter
    rather than inlining it. Inlining is what made this path hang, not just the
    C-emitting one — ``ode/synthesis_v3`` reaches a 40 KB flattened expression
    through a species initial condition, so **every** parameter-sensitivity run
    on that model hung here, before any codegen.

    Returns ``{primary_name: float_partial}`` over the primaries with a non-zero
    partial, or ``{}`` when sympy is unavailable, nothing parses, or no primary
    appears (the caller then leaves that species's derived IC unseeded — the
    pre-#43 behavior — while direct-parameter ICs stay correct). A ``{}`` that
    follows a *failure* rather than a genuine absence is logged as a warning,
    since the caller cannot tell the two apart (issue #56). Callers for which an
    empty result is a *supported* outcome rather than a lost gradient — the
    switch-time scan, which probes candidate thresholds and expects
    non-parameter ones to come back empty — pass ``warn_on_failure=False``.
    """
    s = expr.strip()
    if not s:
        return {}
    try:
        import sympy  # noqa: F401
        from sympy.parsing.sympy_parser import parse_expr  # noqa: F401
    except ImportError:
        return {}

    primaries = set(primary_param_names)
    derived_exprs = derived_exprs or {}
    out, reason = _derived_expr_partials_numeric_dag(
        s, primaries, param_idx, param_values, derived_exprs, warn_on_failure, {}, ()
    )
    if reason is not None:
        # Name the primaries whose chain rule was lost, which after #99 is the
        # set the expression *reaches* rather than the set it spells out.
        reachable = _reachable_primary_names(s, primaries, derived_exprs)
        if warn_on_failure and reachable:
            _warn_chain_rule_dropped(expr, reachable, reason)
        return {}
    return out or {}


def _reachable_primary_names(
    expr: str, primary_names: set[str], derived_exprs: dict[str, str]
) -> list[str]:
    """Every primary parameter ``expr`` reaches through the derived-parameter
    DAG, by a token closure — no parsing, no expression built (GH #99).

    Used only to say *what was lost* when the chain rule fails. Walking the DAG
    node by node keeps this linear in the graph; the flattened text it replaces
    is exponential in the graph's depth, which is the bug this whole path exists
    to avoid, and building a 40 KB string to phrase a warning would reintroduce
    it in the one place nobody would look.
    """
    seen: set[str] = set()
    out: set[str] = set()
    stack = [expr]
    while stack:
        toks = set(re.findall(r"[A-Za-z_]\w*", stack.pop()))
        out |= toks & primary_names
        for name in sorted(toks & derived_exprs.keys()):
            if name not in seen:
                seen.add(name)
                stack.append(derived_exprs[name])
    return sorted(out)


def _derived_expr_partials_numeric_dag(
    expr: str,
    primary_names: set[str],
    param_idx: dict,
    param_values: list,
    derived_exprs: dict[str, str],
    warn_on_failure: bool,
    cache: dict[str, tuple[dict[str, float] | None, str | None]],
    stack: tuple[str, ...],
) -> tuple[dict[str, float] | None, str | None]:
    """:func:`_derived_param_jacobian_dag` with floats: the same chain rule

        ∂expr/∂primary = (∂expr/∂primary)_direct
                         + Σ_k (∂expr/∂s_k)·(∂s_k/∂primary)

    walked over the derived-parameter DAG rather than flattened into it, with
    every factor evaluated at the current parameter point. Same memo, same cycle
    guard, same all-or-nothing failure: a lost sub-partial is a lost chain rule
    for every primary that reaches it, and the caller reads a missing partial as
    a hard zero.
    """
    direct, reason = _direct_derived_partials_numeric(
        expr, primary_names, derived_exprs.keys(), param_idx, param_values, warn_on_failure
    )
    if reason is not None or not direct:
        return None, reason

    out: dict[str, float] = {}
    for name, d_val in direct.items():
        if name not in derived_exprs:
            out[name] = out.get(name, 0.0) + d_val
            continue
        if name in stack:
            return None, f"reference cycle through the derived parameter {name!r}"
        hit = cache.get(name)
        if hit is None:
            hit = _derived_expr_partials_numeric_dag(
                derived_exprs[name],
                primary_names,
                param_idx,
                param_values,
                derived_exprs,
                warn_on_failure,
                cache,
                stack + (name,),
            )
            cache[name] = hit
        sub, why = hit
        if why is not None:
            return None, f"through the derived parameter {name!r}: {why}"
        for prim, sub_val in (sub or {}).items():
            out[prim] = out.get(prim, 0.0) + d_val * sub_val

    # Terms that cancel to exactly zero are dropped here rather than per-term:
    # the caller's contract is the primaries with a non-zero partial, and after
    # composition only the sum knows.
    return ({k: v for k, v in sorted(out.items()) if v != 0.0} or None), None


def _direct_derived_partials_numeric(
    expr: str,
    primary_names: set[str],
    derived_names,
    param_idx: dict,
    param_values: list,
    warn_on_failure: bool,
) -> tuple[dict[str, float] | None, str | None]:
    """``∂expr/∂s`` as a float at the current parameter point, for every
    parameter ``expr`` names **directly** — the numeric twin of
    :func:`_direct_derived_partials`.

    A derived name is an ordinary differentiation variable and substitutes its
    own current value, which the caller's ``param_values`` already carries
    (``core.get_param`` evaluates a ConstantExpression).

    A partial that is not numeric at this parameter point is dropped with a
    warning while the rest are kept: the other primaries' seeds are still valid,
    and issue #56's point is only that a missing one must not pass silently for
    a real zero.
    """
    import sympy as sp

    prep, reason = _prepare_derived_expr(expr, primary_names, derived_names)
    if prep is None:
        return None, reason

    subs = {
        prep.sym_map[prep.sym_name_of[p]]: param_values[param_idx[p]]
        for p in prep.referenced
        if p in param_idx
    }
    out: dict[str, float] = {}
    for p_name in prep.referenced:
        deriv = sp.diff(prep.sym_expr, prep.sym_map[prep.sym_name_of[p_name]])
        if deriv == 0:
            continue
        try:
            val = float(deriv.subs(subs).evalf())
        except (TypeError, ValueError) as exc:
            if warn_on_failure:
                _warn_chain_rule_dropped(expr, [p_name], f"{type(exc).__name__}: {exc}")
            continue
        if val != 0.0:
            out[p_name] = val
    return (out or None), None


def _derived_expr_value_numeric(
    expr: str,
    primary_names: set[str],
    derived_names,
    param_idx: dict,
    param_values,
) -> float | None:
    """The *value* of a derived-parameter expression at the current parameter
    point, through the same preparation as its partials.

    The third caller of :func:`_prepare_derived_expr`, and the reason #108 asked
    for one: :func:`bngsim._switch_sensitivity._evaluate_threshold` wants
    ``t*`` where :func:`_derived_expr_partials_numeric` wants ``∂t*/∂p``, and
    when the two do not agree on which expressions they can read, the caller
    gets partials with no value (or the reverse) and drops the crossing
    entirely. That is exactly how issue #105 failed, one alias map apart.

    A referenced *derived* parameter substitutes its own current value — the
    same move :func:`_direct_derived_partials_numeric` makes, and for the same
    reason (``core.get_param`` evaluates a ConstantExpression, and the engine has
    copied an assignment rule's value into its parameter before the caller's
    scope is built). Substituting it is what lets this share the DAG-walking
    twin's view of the model instead of textually inlining the expression it
    stands for: the flattening is exponential in the depth of the DAG (GH #99),
    and on ``ode/synthesis_v3`` a two-name threshold reached 61 KB and 1.2 s of
    sympy here while its partials, walking the same DAG, took 55 ms.

    Returns ``None`` — the caller's "this threshold is not a constant over the
    model's parameters" — when sympy is unavailable, when the preparation
    declines the expression, or when what it parsed does not reduce to a float
    at this parameter point.
    """
    try:
        import sympy  # noqa: F401
    except ImportError:  # pragma: no cover - sympy is a hard dep of codegen
        return None

    prep, _reason = _prepare_derived_expr(
        expr, primary_names, derived_names, allow_no_reference=True
    )
    if prep is None:
        return None
    subs = {
        prep.sym_map[prep.sym_name_of[p]]: param_values[param_idx[p]]
        for p in prep.referenced
        if p in param_idx
    }
    try:
        # `sympify` of a sympy expression is the identity; it is here so the
        # `.subs` below type-checks against `_PreparedDerivedExpr.sym_expr`,
        # which the other two callers only ever hand to `sp.diff`.
        return float(sympy.sympify(prep.sym_expr).subs(subs).evalf())
    except Exception:
        # A symbol with no value, a value sympy will not reduce to a float (a
        # leftover species name, an unevaluated Piecewise), or anything else
        # substitution raises on. Broad on purpose: this is the *only* caller
        # that has no fallback — the switch-time detector reads `None` as "not a
        # constant" and refuses the crossing, while an escaping exception would
        # take down a whole sensitivity run over one unreadable threshold.
        return None


def compute_ic_param_sens_seed(core) -> list[tuple[int, int, float]]:
    """Forward-sensitivity initial-condition seeds for parameter-referenced
    species initial conditions (issue #43).

    A species whose IC is a parameter reference (``R() R0`` or ``R() Rtot`` with
    ``Rtot = R0``) contributes ∂x_i(0)/∂p to the sensitivity seed yS_i(0). The
    C++ seeding cannot compute the chain-rule partial for a *derived* IC, so it
    is computed here from the model's parameter graph and injected via
    ``SolverOptions.set_ic_param_sens``.

    ``core`` is the C++ ``NetworkModel``. Returns
    ``[(species_idx0, primary_param_idx0, ∂IC/∂primary), ...]``:

      * a **direct** primary IC (``R() R0``) contributes coefficient 1 on that
        primary — identical to the legacy identity seeding;
      * a **derived** IC (``R() Rtot``, ``Rtot = f(primaries)``) contributes one
        entry per primary with a non-zero ∂f/∂primary, chained through nested
        derived parameters and evaluated at the current parameter values.

    Returns ``[]`` when no species IC is a parameter reference and no compartment
    size reaches one (the overwhelming majority of models — two cheap C++ vector
    fetches, no sympy import). A derived IC whose expression cannot be
    differentiated is simply omitted, leaving that species unseeded (pre-#43
    behavior) without disturbing the others.

    Issue #170 stage 3 adds the **storage** axis. bngsim stores amount/V_c, so a
    species whose declared IC is an amount has a stored initial condition that
    moves when its compartment size is written — ``∂x(0)/∂V = −x(0)/V``, which
    :meth:`NetworkModel.compartment_ic_sens_seeds` computes next to the two
    ``refresh_*`` functions that apply the same convention to the value. The same
    divide also scales every *parameter* column of such a species: the parameter
    names an amount, and ``species_ic_param_ref_divisors`` reports the ``1/V_c``
    the parameter graph here cannot see.
    """
    refs = list(core.species_ic_param_refs)  # [(species_idx0, param_idx0)]
    # (#170 stage 3) [(species_idx0, volume_param_idx0, ∂x(0)/∂V)]
    vol_seeds = [tuple(t) for t in getattr(core, "compartment_ic_sens_seeds", ())]
    if not refs:
        return [(int(a), int(b), float(c)) for a, b, c in vol_seeds]
    # Parallel to `refs`: the 1/V_c between "the parameter" and "the stored
    # value". 1.0 for every .net model and every hOSU=false SBML species, so the
    # loop below is arithmetically unchanged there. Degrades to "no divide"
    # against a core built before the accessor existed.
    ref_divisors = list(getattr(core, "species_ic_param_ref_divisors", ())) or [1.0] * len(refs)

    names = list(core.param_names)
    is_expr = list(core.param_is_expression)
    exprs = list(core.param_expressions)
    values = [core.get_param(n) for n in names]
    param_idx = {n: i for i, n in enumerate(names)}
    primary_names = {names[i] for i in range(len(names)) if not is_expr[i]}
    derived_exprs = {names[i]: exprs[i] for i in range(len(names)) if is_expr[i] and exprs[i]}

    seeds: list[tuple[int, int, float]] = []
    for (species_idx0, param_idx0), divisor in zip(refs, ref_divisors, strict=True):
        vdiv = 1.0 / divisor if divisor not in (0.0, 1.0) else 1.0
        pname = names[param_idx0]
        if is_expr[param_idx0] and derived_exprs.get(pname):
            partials = _derived_expr_partials_numeric(
                derived_exprs[pname], primary_names, param_idx, values, derived_exprs
            )
            # An unparseable derived expression yields no partials, so this
            # species is simply left unseeded (pre-#43 behavior) rather than
            # mis-seeded on the derived index.
            #
            # Issue #155: a primary the expression *reaches* keeps a row even
            # when its partial is zero at this parameter point — ∂(a*R0)/∂R0 is
            # `a`, which vanishes at a = 0 without the seeding path ceasing to
            # exist. The DAG walk drops numeric zeros (it is shared with the
            # switch-time scan, where an empty result means "not a parameter
            # threshold"), so structural presence is recovered here from the
            # token closure. A zero row seeds nothing either way; it exists so
            # `Model.effective_ic_sensitivity` can tell "seeded, zero here" from
            # "no seeding path at all", which a consumer composing its own chain
            # rule must not confuse.
            if partials:
                for prim_name in _reachable_primary_names(
                    derived_exprs[pname], primary_names, derived_exprs
                ):
                    seeds.append(
                        (
                            species_idx0,
                            param_idx[prim_name],
                            float(partials.get(prim_name, 0.0)) * vdiv,
                        )
                    )
        else:
            # Direct primary IC: seed coefficient 1 on the exact named
            # parameter, matching the legacy C++ identity seeding — except where
            # the stored value is that parameter over a volume (issue #170 stage
            # 3), which is 1.0 for every model the legacy seeding covers.
            seeds.append((species_idx0, param_idx0, vdiv))
    # The storage half. Kept separate rather than merged into a row above: a
    # species can have both (an <initialAssignment> that reads the compartment
    # gives ∂A/∂V through the parameter graph *and* the explicit −A/V² here), and
    # the C++ seeding accumulates with `+=` precisely so the two arrive as two
    # rows on the same (species, parameter) cell.
    seeds.extend((int(a), int(b), float(c)) for a, b, c in vol_seeds)
    return seeds


# ─── C code generation ───────────────────────────────────────────────


def generate_rhs_c(net_path: str) -> str:
    """Generate a C source file implementing the CVODE RHS callback.

    The generated code reads parameters from a runtime array via user_data,
    NOT baked as compile-time literals. This allows the .so to be compiled
    once and reused for all parameter evaluations.

    Parameters
    ----------
    net_path : str
        Path to the .net file.

    Returns
    -------
    str
        Complete C source code.
    """
    model = _parse_net_file(net_path)
    _validate_net_model_for_codegen(model, net_path)
    params = model["parameters"]
    species = model["species"]
    reactions = model["reactions"]
    observables = model["observables"]
    functions = model["functions"]

    n_sp = len(species)
    n_params = len(params)
    n_obs = len(observables)
    n_func = len(functions)

    # Build name->index maps (0-based)
    param_idx = {name: i for i, (_, name, _, _) in enumerate(params)}
    func_names = {name for _, name, _ in functions}
    func_idx = {name: i for i, (_, name, _) in enumerate(functions)}
    obs_idx = {name: i for i, (_, name, _) in enumerate(observables)}

    # Identify fixed species (0-based indices)
    fixed_sp = set()
    for _, _, _, _is_fixed in species:
        pass
    fixed_sp = {sp[0] - 1 for sp in species if sp[3]}

    # ── Build per-reaction rate + scatter lines (one group per reaction) ────
    # See generate_rhs_from_model for the Tier-1 chunking rationale. When
    # chunking, Functional rates reference func[idx] (the packed array passed to
    # each block) instead of the func_<name> locals, which live inside
    # bngsim_codegen_rhs and are invisible to the file-scope blocks.
    chunk = _should_chunk(len(reactions))
    block_size = _chunk_block_size()
    rxn_groups: list[list[str]] = []
    for _, reactants, products, rate_law, _comment in reactions:
        grp: list[str] = []
        g = grp.append
        kind = _classify_rate_law(rate_law, func_names)
        rate_expr: str | None = None
        if kind[0] == "elementary":
            _, pname, sf = kind
            rate_expr = _rate_elementary(pname, sf, reactants, param_idx, func_idx)
        elif kind[0] == "functional":
            _, fname, sf = kind
            rate_expr = _rate_functional(fname, sf, reactants, func_idx, use_array=chunk)
        elif kind[0] == "mm":
            # A braced block, not an expression: the stable free-substrate root
            # is a branch on delta's sign (GH #89), and inlining it would repeat
            # the sqrt four times.
            _, kcat, km, sf = kind
            if len(reactants) >= 2:
                for ln in _mm_rate_lines(
                    f"p[{param_idx[kcat]}]" if kcat in param_idx else "0.0",
                    f"p[{param_idx[km]}]" if km in param_idx else "0.0",
                    sf,
                    reactants[0] - 1,
                    reactants[1] - 1,
                ):
                    g(ln)
            else:
                rate_expr = "0.0"
        else:
            rate_expr = "0.0"
        if rate_expr is not None:
            g(f"    rate = {rate_expr};")
        # Subtract from reactants (index 0 = null reactant, skip)
        for ri in reactants:
            if ri > 0:
                g(f"    ydot[{ri - 1}] -= rate;")
        # Add to products (index 0 = null/degradation product, skip)
        for pi in products:
            if pi > 0:
                g(f"    ydot[{pi - 1}] += rate;")
        g("")
        rxn_groups.append(grp)

    rxn_needs_func = any("func[" in ln for grp in rxn_groups for ln in grp)
    if rxn_needs_func:
        _rxn_sig = "const double* y, const double* p, const double* func, double* ydot"
        _rxn_args = "y, p, func, ydot"
    else:
        _rxn_sig = "const double* y, const double* p, double* ydot"
        _rxn_args = "y, p, ydot"
    rxn_block_defs: list[str] = []
    rxn_call_lines: list[str] = []
    rxn_block_protos: list[str] = []
    if chunk:
        rxn_block_defs, rxn_call_lines, rxn_block_protos = _emit_chunked_blocks(
            rxn_groups,
            fn_prefix="rxn_blk",
            signature_params=_rxn_sig,
            call_args=_rxn_args,
            block_size=block_size,
            preamble=("double rate;",),
        )

    # ── Observable + function computation (GH #165 chunking) ────────────────
    # Flat: ``obs_<name>`` / ``func_<name>`` locals (byte-identical to pre-#165).
    # Chunked: ``obs[idx]`` / ``func[idx]`` arrays filled by NOINLINE shard blocks,
    # so this large basic block (a genome-scale model has ~18k of each) is split
    # off the serial driver into parallel translation units instead of being the
    # compile wall. The Functional reaction blocks already read ``func[idx]``, so
    # the chunked form drops the separate "pack func_<name> into func[]" step.
    obs_value_lines: list[str] = []
    if observables:
        if chunk:
            obs_value_lines.append("    double obs[N_OBS];")
        for _i, (_, name, entries) in enumerate(observables):
            if not entries:
                rhs_expr = "0.0"
            else:
                terms = []
                for factor, sp_i in entries:
                    sp0 = sp_i - 1  # 0-based
                    if factor == 1.0:
                        terms.append(f"y[{sp0}]")
                    elif factor == int(factor):
                        terms.append(f"{int(factor)}*y[{sp0}]")
                    else:
                        terms.append(f"{factor}*y[{sp0}]")
                rhs_expr = " + ".join(terms)
            if chunk:
                obs_value_lines.append(f"    obs[{_i}] = {rhs_expr};")
            else:
                obs_value_lines.append(f"    double obs_{_safe_c_name(name)} = {rhs_expr};")

    func_value_lines: list[str] = []
    if functions:
        if chunk:
            func_value_lines.append("    double func[N_FUNC];")
        # Built once and shared across every function body — see _build_ident_lookup
        # (rebuilding it per body was the second GH #161 quadratic). The reference
        # form (named locals vs obs[]/func[] arrays) follows the chunk decision.
        ident_lookup = _build_ident_lookup(param_idx, obs_idx, functions, use_arrays=chunk)
        tf_id = 0
        for _i, (_, name, expr) in enumerate(functions):
            rewritten, tfun_calls = _extract_tfun_calls(expr)
            if not tfun_calls:
                c_expr = _translate_expr(expr, ident_lookup)
            else:
                c_expr = _translate_expr(rewritten, ident_lookup)
                for k, tspec in enumerate(tfun_calls):
                    _, idx_c_expr = _classify_tfun_index(
                        tspec["index_name"], param_idx, obs_idx, use_arrays=chunk
                    )
                    placeholder = _TFUN_PLACEHOLDER_FMT.format(idx=k)
                    callback = f"data->tfun_eval({tf_id}, {idx_c_expr}, data->tfun_ctx)"
                    c_expr = c_expr.replace(placeholder, callback)
                    tf_id += 1
            if chunk:
                func_value_lines.append(f"    func[{_i}] = {c_expr};")
            else:
                func_value_lines.append(f"    double func_{_safe_c_name(name)} = {c_expr};")

    rhs_obs_in, rhs_obs_fs = _shard_value_lines(
        obs_value_lines,
        chunk=chunk,
        fn_prefix="rhs_obs_blk",
        signature_params=_OBS_BLK_SIG,
        call_args=_OBS_BLK_ARGS,
    )
    rhs_func_in, rhs_func_fs = _shard_value_lines(
        func_value_lines,
        chunk=chunk,
        fn_prefix="rhs_func_blk",
        signature_params=_FUNC_BLK_SIG,
        call_args=_FUNC_BLK_ARGS,
        preamble=_FUNC_BLK_PREAMBLE,
    )

    lines: list[str] = []
    _emit = lines.append

    # ── Header ────────────────────────────────────────────────────────
    _emit("/* Auto-generated by bngsim._codegen - DO NOT EDIT */")
    if chunk:
        _emit(_CHUNK_MARKER)
    _emit("/* Code-generated ODE RHS for CVODE */")
    _emit("")
    _emit("#include <math.h>")
    _emit("#include <stdlib.h>")
    _emit("#include <string.h>")
    _emit("")
    _emit("#ifndef M_PI")
    _emit("#define M_PI 3.14159265358979323846")
    _emit("#endif")
    _emit("#ifndef M_E")
    _emit("#define M_E 2.71828182845904523536")
    _emit("#endif")
    _emit("")
    _emit("/* User data struct passed via CVODE user_data pointer.")
    _emit("   Must match the layout set up by the C++ CvodeSimulator. */")
    _emit("typedef double (*TfunEvalFn)(int tf_id, double x, void* ctx);")
    _emit("typedef struct {")
    _emit("    double* param_values;   /* runtime parameter array */")
    _emit("    void* tfun_ctx;         /* opaque context for tfun callback */")
    _emit("    TfunEvalFn tfun_eval;   /* table-function dispatch (may be NULL) */")
    _emit("} CodegenUserData;")
    _emit("")

    # ── Dimensions as macros ──────────────────────────────────────────
    _emit(f"#define N_SPECIES {n_sp}")
    _emit(f"#define N_PARAMS  {n_params}")
    _emit(f"#define N_OBS     {n_obs}")
    _emit(f"#define N_FUNC    {n_func}")
    _emit("")

    # No per-parameter ``#define P_<name> <idx>`` macros are emitted: the rate-law
    # emitters reference parameters numerically (``p[idx]``), so the macros were
    # never used — yet, sitting in the source prefix before the first shard block,
    # they were duplicated into every parallel shard unit (~3.4 MB × ~175 units ≈
    # 600 MB of dead scratch on a genome-scale build). Dropped (GH #165 follow-up).

    # ── Tier-1 chunking: NOINLINE reaction blocks at file scope ───────────
    # Prototypes precede the definitions so the driver TU can call the blocks
    # after compile_rhs lifts their bodies into separate units (GH #160).
    # BNGSIM_EXPORT (below) tags the entry points so they are visible from the
    # built library on Windows and must be defined for every model, chunked or
    # not; BNGSIM_NOINLINE is used only by the chunked blocks (lanl/bngsim #5).
    for ln in _CODEGEN_PRELUDE_LINES:
        _emit(ln)
    _emit("")
    if chunk:
        for ln in (
            *rxn_block_protos,
            "",
            *rxn_block_defs,
            *rhs_obs_fs,
            *rhs_func_fs,
        ):
            _emit(ln)
        _emit("")

    # ── RHS function ──────────────────────────────────────────────────
    _emit(
        "BNGSIM_EXPORT int bngsim_codegen_rhs(double t, double* y, double* ydot, "
        "void* user_data) {"
    )
    _emit("    CodegenUserData* data = (CodegenUserData*)user_data;")
    _emit("    double* p = data->param_values;")
    _emit("")

    # Zero derivatives
    _emit("    /* Zero derivatives */")
    _emit("    memset(ydot, 0, N_SPECIES * sizeof(double));")
    _emit("")

    # Compute observables (flat: obs_<name> locals; chunked: obs[] array filled by
    # the rhs_obs_blk_* shard blocks built above — see the obs/func construction).
    if observables:
        _emit("    /* Compute observables */")
        for ln in rhs_obs_in:
            _emit(ln)
        _emit("")

    # Evaluate functions (in dependency order — same as .net file order). Flat:
    # func_<name> locals; chunked: func[] array filled by the rhs_func_blk_* shard
    # blocks (which also packs func[] for the Functional reaction blocks, so no
    # separate pack step is needed). tfun(...) calls dispatch through the runtime
    # callback; the tf_id ordering matches the runtime table_functions vector.
    if functions:
        _emit("    /* Evaluate functions (dependency order from .net) */")
        for ln in rhs_func_in:
            _emit(ln)
        _emit("")

    # Reactions. Chunked: call the file-scope NOINLINE blocks built above.
    # Flat (below threshold): splice the per-reaction groups inline — byte-
    # identical to the pre-chunking output.
    _emit("    /* Compute reaction rates and accumulate derivatives */")
    if chunk:
        lines.extend(rxn_call_lines)
        _emit("")
    else:
        _emit("    double rate;")
        _emit("")
        for grp in rxn_groups:
            lines.extend(grp)

    # Zero derivatives for fixed species
    if fixed_sp:
        _emit("    /* Zero derivatives for fixed species */")
        for si in sorted(fixed_sp):
            _emit(f"    ydot[{si}] = 0.0;")
        _emit("")

    _emit("    return 0;")
    _emit("}")

    return "\n".join(lines) + "\n"


def _safe_c_name(name: str) -> str:
    """Convert a BNG name to a safe C identifier."""
    # Replace non-alphanumeric chars with underscore
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _build_ident_lookup(
    param_idx: dict,
    obs_idx: dict,
    functions: list,
    *,
    use_arrays: bool = False,
) -> dict[str, tuple[str, bool]]:
    """Build the identifier → (C-reference, eats-empty-call) table that
    ``_translate_expr`` rewrites function bodies against.

    Built ONCE per ``generate_rhs_c`` and reused for every function body.
    Building it per call is O(n_functions × (n_params + n_obs + n_funcs)) —
    the second GH #161 quadratic: a genome-scale model has ~132k params +
    ~18k observables + ~18k functions, and rebuilding all ~170k entries (plus
    a ``_safe_c_name`` regex per observable and function) for each of ~18k
    function bodies dominated source generation. Insertion order sets
    precedence (later wins): params, then observables, then functions — so a
    name reused as both an observable and a function resolves to the function,
    matching ``_translate_expr_to_c``.

    ``use_arrays`` selects the observable / function reference form: the default
    ``obs_<name>`` / ``func_<name>`` locals (flat .net RHS) or the ``obs[idx]`` /
    ``func[idx]`` array slots used when the obs/func computation is sharded into
    NOINLINE blocks (GH #165) — there the named locals are not in scope, so the
    blocks read and write the passed arrays instead.

    Every model name eats a trailing ``()`` (issue #28 — see
    ``_BUILTIN_IDENT_MAP``): each resolves to a scalar in the emitted C, so
    ``divide()`` in a function body must become ``obs_divide``, not
    ``obs_divide()``.
    """
    lookup: dict[str, tuple[str, bool]] = dict(_BUILTIN_IDENT_MAP)
    for name, idx in param_idx.items():
        lookup[name] = (f"p[{idx}]", True)
    if use_arrays:
        for name, oi in obs_idx.items():
            lookup[name] = (f"obs[{oi}]", True)
        for fi, (_, fname, _) in enumerate(functions):
            lookup[fname] = (f"func[{fi}]", True)
    else:
        for name in obs_idx:
            lookup[name] = (f"obs_{_safe_c_name(name)}", True)
        for _, fname, _ in functions:
            lookup[fname] = (f"func_{_safe_c_name(fname)}", True)
    return lookup


def _translate_expr(expr: str, lookup: dict[str, tuple[str, bool]]) -> str:
    """Translate a .net function expression (ExprTk grammar) to C code.

    Mirrors the model-based ``_translate_expr_to_c`` pipeline so the .net
    codegen path produces the same numerics as the ExprTk interpreter for
    every BNG-supported expression construct: power (``^``), conditionals
    (``if(c,a,b)``), word-form logicals (``and``/``or``/``not``), constants
    (``_pi``/``_e``), and ``abs``/``ln``/``rint``. The .net path uses the
    ``obs_<Name>`` / ``func_<Name>`` local-variable naming emitted by
    ``generate_rhs_c``; species are referenced only via observables.

    ``lookup`` is the prebuilt identifier table from ``_build_ident_lookup`` —
    shared across all function bodies so it is built once, not per call (the
    second GH #161 quadratic). Single-pass identifier rewriting — see
    ``_translate_expr_to_c`` for the Issue-#25 motivation.
    """
    c_expr = _replace_if_calls(expr)
    # Float-ify integer literals before subscripts appear (see
    # _translate_expr_to_c) so ExprTk's ``1/2`` == 0.5 survives into C.
    c_expr = _floatify_int_literals(c_expr)

    def _repl(m: re.Match) -> str:
        name = m.group(1)
        empty_call = m.group(2)
        entry = lookup.get(name)
        if entry is None:
            return m.group(0)
        rep, eats_call = entry
        if empty_call is not None:
            return rep if eats_call else rep + empty_call
        return rep

    c_expr = _IDENT_OR_EMPTY_CALL_RE.sub(_repl, c_expr)
    c_expr = _replace_power_op(c_expr)
    return c_expr


def _rate_elementary(
    pname: str,
    sf: float,
    reactants: list,
    param_idx: dict,
    func_idx: dict,
) -> str:
    """Generate C expression for elementary rate: k * sf * ∏ y[ri].

    Reactant index 0 marks a null reactant (synthesis reaction); skip it
    so we don't emit an out-of-bounds y[-1] read.
    """
    parts = []
    if pname in param_idx:
        parts.append(f"p[{param_idx[pname]}]")
    else:
        parts.append(f"/* UNKNOWN_PARAM {pname} */ 0.0")

    if sf != 1.0:
        if sf == int(sf):
            parts.insert(0, str(int(sf)))
        else:
            parts.insert(0, str(sf))

    for ri in reactants:
        if ri > 0:
            parts.append(f"y[{ri - 1}]")

    return " * ".join(parts)


def _rate_functional(
    fname: str,
    sf: float,
    reactants: list,
    func_idx: dict,
    use_array: bool = False,
) -> str:
    """Generate C expression for functional rate: func * sf * ∏ y[ri].

    Reactant index 0 marks a null reactant (synthesis reaction); skip it
    so we don't emit an out-of-bounds y[-1] read.

    ``use_array`` references the function value as ``func[idx]`` (the packed
    array passed to a Tier-1 NOINLINE block) instead of the ``func_<name>``
    local — the blocks live outside bngsim_codegen_rhs and cannot see its locals.
    """
    parts = []
    safe = _safe_c_name(fname)
    if use_array and fname in func_idx:
        parts.append(f"func[{func_idx[fname]}]")
    else:
        parts.append(f"func_{safe}")

    if sf != 1.0:
        if sf == int(sf):
            parts.insert(0, str(int(sf)))
        else:
            parts.insert(0, str(sf))

    for ri in reactants:
        if ri > 0:
            parts.append(f"y[{ri - 1}]")

    return " * ".join(parts)


def _mm_sfree_c_lines(km_c: str, e_idx: int, s_idx: int, indent: str) -> list[str]:
    """C lines declaring ``Km``/``E``/``S``/``delta``/``Dmm``/``sFree``/``KpsF``
    for one tQSSA Michaelis–Menten reaction — the single source of the
    free-substrate root *and of the guard* for every emitter on this path
    (GH #89, GH #93).

    ``sFree`` is the positive root of ``x² − delta·x − Km·S = 0``. Writing it as
    ``½(delta + D)`` subtracts two nearly-equal positive numbers once
    ``delta < 0``, losing about two significant digits per decade of
    ``|delta|/√(4·Km·S)`` — no correct digit left at 1e8, in the *rate* and so in
    everything derived from it. The conjugate form ``2·Km·S/(D − delta)``
    multiplies out to the same value with no subtraction. ``delta ≥ 0`` keeps the
    textbook form, which is already cancellation-free, bit-for-bit.

    ``KpsF`` (``Km + sFree``) is declared here rather than spelled out at each
    use because it is the rate's denominator *and* the one guard every emitter
    tests — GH #93 was filed because the RHS emitter and the derivative emitters
    disagreed about that guard, so they now cannot: the RHS is live iff
    ``KpsF > 0``, and the partials are live iff ``KpsF > 0 && Dmm > 0``.
    ``sFree`` is deliberately *not* clamped to 0; see the header note in
    ``include/bngsim/mm_jacobian.hpp`` for why the negative branch is the correct
    smooth continuation and why the denominator is the only real degeneracy.
    """
    return [
        f"{indent}double Km = {km_c}, E = y[{e_idx}], S = y[{s_idx}];",
        f"{indent}double delta = S - Km - E;",
        f"{indent}double Dmm = sqrt(delta*delta + 4.0*Km*S);",
        f"{indent}double sFree = (delta >= 0.0) ? 0.5*(delta + Dmm)",
        f"{indent}    : ((Dmm - delta) > 0.0 ? 2.0*Km*S/(Dmm - delta) : 0.0);",
        f"{indent}double KpsF = Km + sFree;",
    ]


def _mm_rate_lines(
    kcat_c: str,
    km_c: str,
    sf: float,
    e_idx: int,
    s_idx: int,
    indent: str = "    ",
) -> list[str]:
    """C lines assigning the Michaelis–Menten tQSSA ``rate`` (a braced block, so
    the locals cannot collide with the caller's).

    ``e_idx``/``s_idx`` are 0-based species indices — the enzyme is the *first*
    reactant and the substrate the second, matching ``src/model.cpp``'s MM branch.

    The ``KpsF > 0`` ternary is the same guard the derivative emitters below use,
    from the same helper (GH #93). It is not a non-negativity clamp on ``sFree``:
    a negative ``sFree`` (i.e. ``S < 0``) still produces a rate here, the smooth
    restoring continuation that ``_mm_jacobian_groups`` differentiates.
    """
    sf_c = f"{sf} * " if sf != 1.0 else ""
    return [
        f"{indent}{{",
        *_mm_sfree_c_lines(km_c, e_idx, s_idx, indent + "    "),
        f"{indent}    rate = KpsF > 0.0 ? {sf_c}{kcat_c} * sFree * E / KpsF : 0.0;",
        f"{indent}}}",
    ]


# ─── Sensitivity RHS code generation ────────────────────────────────────


def generate_sens_rhs_c(net_path: str, *, emit_term_scale: bool = False) -> str | None:
    """Generate C code for the CVODES sensitivity RHS callback.

    The sensitivity equation for parameter p_iS is:
        ySdot = J * yS + df/dp_{iS}
    where J = df/dy is the Jacobian and df/dp_{iS} is the partial
    derivative of each species' RHS w.r.t. the iS-th parameter.

    For Elementary reactions v_r = k_r * sf * ∏ x_j^{m_j}:
        df_i/dk_r = S[i][r] * sf * ∏ x_j^{m_j}  (rate without k_r)
        J[i][j]   = S[i][r] * k_r * sf * m_j * x_j^{m_j-1} * ∏_{l≠j} x_l^{m_l}

    For Functional/MM: returns None (fall back to CVODES internal FD).

    The only symbolic work here is the derived-rate-constant chain rule below; it
    shares the GH #90 build-time budget with the model path, so a .net carrying
    enough ``# ConstantExpression`` rate constants declines rather than hangs.

    Parameters
    ----------
    net_path : str
        Path to the .net file.

    Returns
    -------
    str or None
        C source code, or None if model has non-Elementary reactions.
    """
    model = _parse_net_file(net_path)
    _validate_net_model_for_codegen(model, net_path)
    params = model["parameters"]
    species = model["species"]
    reactions = model["reactions"]
    model["observables"]
    functions = model["functions"]

    n_sp = len(species)
    n_params = len(params)

    # GH #90: one deadline for this build's symbolic work, resolved before it.
    from bngsim._jacobian import _DerivationBudgetExceeded

    deadline = _sens_derivation_deadline(n_sp)

    # Build name→index maps
    param_idx = {name: i for i, (_, name, _, _) in enumerate(params)}
    func_names = {name for _, name, _ in functions}

    # Check: all reactions must be Elementary for analytical sensitivity RHS
    rate_const_names: set[str] = set()
    for _, _reactants, _products, rate_law, _ in reactions:
        kind = _classify_rate_law(rate_law, func_names)
        if kind[0] != "elementary":
            return None  # Fall back to CVODES internal FD
        rate_const_names.add(kind[1])

    # Identify fixed species
    fixed_sp = {sp[0] - 1 for sp in species if sp[3]}

    # Build mapping from derived (constant-expression) parameter name to
    # ``{primary_param_name: C-expression-for-∂p_d/∂primary}``. When BNG2.pl
    # emits a rate law like ``chi_r1*kon_CSH2`` or ``5/MEK`` it stores the
    # value as a derived parameter ``_rateLaw{N}``. Without this expansion,
    # the codegen sensitivity RHS treats ``_rateLaw{N}`` as an independent
    # rate constant and the sensitivities w.r.t. the underlying primary
    # parameters are wrong (issue #2). The chain-rule contribution to
    # ``∂rate/∂primary`` is ``(∂p_d/∂primary) * sf * ∏y^m``.
    primary_param_names = {name for (_, name, expr, is_const) in params if is_const}
    derived_exprs = {name: expr for (_, name, expr, is_const) in params if not is_const and expr}
    derived_expansion: dict[str, dict[str, str]] = {}
    # GH #99: one memo for the whole loop, so a DAG node shared by several rate
    # constants is differentiated once.
    derived_jac_cache: dict[str, tuple[dict[str, str] | None, str | None]] = {}
    for _, name, expr, is_const in params:
        # Only a derived parameter that actually serves as a reaction's rate
        # constant reaches the sens RHS, so only those are differentiated — and
        # only those can invalidate it. A derived parameter used solely by an
        # observable or a function (a reporting quantity like a mean or a
        # standard deviation) has no bearing here.
        if is_const or name not in rate_const_names:
            continue
        try:
            jac, reason = _derived_param_jacobian_checked(
                expr,
                primary_param_names,
                param_idx,
                derived_exprs=derived_exprs,
                deadline=deadline,
                cache=derived_jac_cache,
                name=name,
            )
        except _DerivationBudgetExceeded:
            # GH #90: decline to CVODES' difference quotient rather than let the
            # chain-rule derivation run unbounded (see generate_sens_from_model).
            _warn_functional_sens_rhs_refused(
                _sens_budget_decline_reason(n_sp, f"deriving the rate constant {name} = {expr!r}")
            )
            return None
        if reason is not None:
            # Issue #56: emitting the RHS without this chain rule would report
            # ∂/∂primary as exactly zero. Refuse the analytic RHS instead so the
            # caller falls back to CVODES' internal difference quotient, which
            # is slower but right.
            _warn_sens_rhs_refused(name, expr, reason)
            return None
        if jac is not None:
            derived_expansion[name] = jac

    # Build reaction data structure for code generation
    rxn_data = []
    for _, reactants, products, rate_law, _ in reactions:
        kind = _classify_rate_law(rate_law, func_names)
        _, pname, sf = kind
        pidx = param_idx.get(pname, -1)

        # Net stoichiometry: for each species, compute net change
        stoich: dict[int, int] = {}  # 0-based species index → net coefficient
        for ri in reactants:
            if ri > 0:
                si = ri - 1
                stoich[si] = stoich.get(si, 0) - 1
        for pi in products:
            if pi > 0:
                si = pi - 1
                stoich[si] = stoich.get(si, 0) + 1

        # Reactant multiplicities (0-based)
        rmult = Counter(ri - 1 for ri in reactants if ri > 0)

        # Resolve any chain rule for derived rate-constant parameters.
        # Each entry ``(primary_param_idx, dpd_dprimary_c_expr)`` carries
        # the primary parameter's index and the C source for
        # ``∂p_d/∂primary``; the dfdp emit then multiplies by ``sf * ∏y^m``.
        derived_terms: list[tuple] = []
        if pname in derived_expansion:
            for primary_name, c_expr in derived_expansion[pname].items():
                p_idx_k = param_idx.get(primary_name, -1)
                if p_idx_k < 0:
                    continue
                derived_terms.append((p_idx_k, c_expr))

        rxn_data.append(
            {
                "param_idx": pidx,
                "stat_factor": sf,
                "stoich": stoich,
                "reactant_mult": dict(rmult),  # {sp_idx: multiplicity}
                "reactants_raw": [ri for ri in reactants if ri > 0],
                "derived_terms": derived_terms,
            }
        )

    # No ``value_lines_fn`` (GH #65): every derivative this path emits is a
    # ``p[]``/``y[]`` expression, so none can reference obs[]/func[] and the
    # context is never asked for. The .net parse also carries neither the
    # observable-entry nor the table-function shapes ``_emit_observable_lines`` /
    # ``_emit_function_lines`` consume — the same reason ``generate_combined_c``
    # sources its Jacobian from the *model*, not from the .net. Should a caller
    # ever produce an obs-referencing term here, _emit_sens_rhs_body declines
    # rather than emitting C that names an undeclared array.
    return _emit_sens_rhs_body(rxn_data, n_sp, n_params, fixed_sp, emit_term_scale=emit_term_scale)


def _emit_sens_rhs_body(
    rxn_data: list[dict],
    n_sp: int,
    n_params: int,
    fixed_sp: set[int],
    *,
    value_lines_fn: Callable[[], tuple[list[str], list[str]] | None] | None = None,
    functional_dfdp: bool = False,
    functional_jacv_groups: list[list[str]] | None = None,
    emit_term_scale: bool = False,
) -> str | None:
    """Emit the C source for `bngsim_dfdp`, `bngsim_jac_vec`, and
    `bngsim_codegen_sens_rhs` from a normalized reaction-data structure.

    Each entry of ``rxn_data`` is a dict with keys:
        param_idx     : int — 0-based index of the rate-constant parameter,
                        or -1 if the reaction has no scalar rate constant
                        (e.g., functional/MM, which the caller must filter
                        out before invoking this helper).
        stat_factor   : float
        stoich        : dict[int, int] — 0-based species index → net coeff
        reactant_mult : dict[int, int] — 0-based species index → multiplicity
        derived_terms : list[(primary_param_idx, dpd_dprimary_c_expr)]
                        — chain-rule contributions for derived rate constants;
                        empty for the model-based path (see issue #15).
        row_divisor   : dict[int, (live_volume_idx0, static_divisor,
                        static_divisor_param_idx0)] — optional, the GH #160
                        cross-compartment volume divide for the rows that have one
                        (see :func:`_psvs_row_divisor`). Absent or empty ⇒ every row
                        accumulates undivided, the shape every single-compartment
                        model has.
        amount_factor_c : str | None — optional, the GH #75 ``∏ V_c^mult`` amount
                        factor as one C factor (see :func:`_amount_factor_c`), or
                        ``None``/absent for a reaction that carries none. The
                        ``.net`` path never sets it.

    Both ``generate_sens_rhs_c`` (.net path) and ``generate_sens_from_model``
    (model path) feed this helper, so the emitted C is byte-identical for the
    same normalized input.

    ``value_lines_fn`` (GH #65) supplies the ``obs[]``/``func[]`` recomputation a
    Functional ``∂f/∂p`` reads. An Elementary rate law is ``k·sf·∏y^m``, whose
    parameter derivative is written purely in ``p[]``/``y[]``, so the emitted
    switch never mentions ``obs[``/``func[`` and the thunk is **never called** —
    Elementary models pay nothing, at emit time or at run time, and their source
    is byte-identical to the pre-#65 output. When a caller does emit a derivative
    that reads them, the thunk is invoked once and must return
    ``(obs_value_lines, func_value_lines)`` from the shared
    ``_emit_observable_lines`` / ``_emit_function_lines`` emitters, so derivative
    and value can never diverge; ``bngsim_dfdp`` then gains the corresponding
    ``const double*`` parameters (independently, mirroring the analytical
    Jacobian's block signature) and the driver computes the arrays once per call
    before dispatching.

    Returns ``None`` — decline, never a silently zeroed derivative — when the
    switch needs values the caller cannot deliver (no thunk, or a thunk that
    declines because a needed function dispatches through ``data->tfun_eval``,
    which this RHS's ``CodegenSensUserData`` cannot reach).

    ``functional_dfdp`` (GH #66) only labels the emitted source: ``rxn_data``
    carrying a Functional ∂func/∂p needs no different emission, so this just tells
    the header which rate-law classes the switch covers. Default ``False`` ⇒
    byte-identical output.

    ``functional_jacv_groups`` (GH #67) supplies the other half — the Functional
    reactions' ``J·v`` contributions, already emitted as balanced ``{ … }`` line
    groups by :func:`_functional_jacobian_groups` with the matvec fused into the
    scatter (``Jv_out[i] += coeff·dj·v[j]``). They are appended to the Elementary
    groups this helper builds from ``rxn_data``, so ``bngsim_jac_vec`` covers the
    whole model with no ``n×n`` buffer and no second derivation. The groups are
    also what decides whether ``bngsim_jac_vec`` takes ``obs``/``func``/``t``:
    Elementary bodies read none of the three, so an Elementary model's signature —
    and its whole emitted source — is unchanged.
    """
    lines: list[str] = []
    _emit = lines.append

    # Tier-1: bngsim_jac_vec is the giant straight-line function here (one block
    # per reaction); chunk it for large models. bngsim_dfdp is a switch (each
    # case is its own small basic block) which compiles fine flat, so it is left
    # alone. Same threshold as the RHS ⇒ a model that chunks one chunks both.
    chunk = _should_chunk(len(rxn_data))
    block_size = _chunk_block_size()

    # ── Resolve the obs[]/func[] context the switch will need (GH #65) ──
    # Decided *before* anything is emitted, because it fixes bngsim_dfdp's
    # signature, which is written above the body.
    #
    # Read off the input rather than off the emitted text. Every other term in a
    # case is built here — `_build_geom_terms` produces only numeric literals and
    # y[…], and the scatter is dfdp_out[…] += v — so a derived term's C
    # expression is the *only* place obs[]/func[] can enter. Deciding from the
    # data costs O(#derived terms) (zero for the overwhelming majority of
    # reactions); scanning the emitted body would cost O(source size), which at
    # genome scale is ~100k lines and shows up in emit time.
    #
    # An Elementary derivative is written purely in p[]/y[], so both flags stay
    # False, the thunk is never called, and every line below is skipped — the
    # source is byte-identical to the pre-#65 emitter.
    #
    # GH #67: bngsim_jac_vec's Functional groups read the same two arrays, so the
    # need is the union of the two consumers, while each function's own signature
    # is decided by what *it* reads — a model whose ∂f/∂p is parameter-free but
    # whose J·v is not (or the reverse) carries neither array where it is unused.
    _derived_exprs = [t[1] for rxn in rxn_data for t in rxn.get("derived_terms", ())]
    fjacv_groups = [list(g) for g in (functional_jacv_groups or ())]
    _fjacv_text = "\n".join("\n".join(g) for g in fjacv_groups)
    dfdp_need_obs = any("obs[" in e for e in _derived_exprs)
    dfdp_need_func = any("func[" in e for e in _derived_exprs)
    jacv_need_obs = "obs[" in _fjacv_text
    jacv_need_func = "func[" in _fjacv_text
    need_obs = dfdp_need_obs or jacv_need_obs
    need_func = dfdp_need_func or jacv_need_func
    obs_in: list[str] = []
    obs_fs: list[str] = []
    func_in: list[str] = []
    func_fs: list[str] = []
    if need_obs or need_func:
        values = value_lines_fn() if value_lines_fn is not None else None
        if values is None:
            # No context available (the .net path), or the caller declined (a
            # tfun-backed function value). Refuse the analytic RHS rather than
            # emit a derivative that cannot compile — CVODES' internal
            # difference quotient is slower but right (the #56 precedent).
            return None
        obs_value_lines, func_value_lines = values
        # Function bodies are written in obs[] symbols, so obs[] is emitted
        # whenever a func[] that references one is — otherwise only when the
        # switch itself reads it, so no unused array is ever declared.
        want_obs = need_obs or (need_func and any("obs[" in ln for ln in func_value_lines))
        if want_obs:
            if not obs_value_lines:
                return None
            _sens_obs_sig, _sens_obs_args = _obs_blk_sig(obs_value_lines)
            obs_in, obs_fs = _shard_value_lines(
                obs_value_lines,
                chunk=chunk,
                fn_prefix="sens_obs_blk",
                signature_params=_sens_obs_sig,
                call_args=_sens_obs_args,
            )
        if need_func:
            if not func_value_lines:
                return None
            # A model with no observables declares no obs[] array; the blocks
            # still take the parameter (one fixed signature) and never read it.
            func_in, func_fs = _shard_value_lines(
                func_value_lines,
                chunk=chunk,
                fn_prefix="sens_func_blk",
                signature_params=_SENS_FUNC_BLK_SIG,
                call_args=_SENS_FUNC_BLK_ARGS if want_obs else _SENS_FUNC_BLK_ARGS_NO_OBS,
            )

    # ── Header ──────────────────────────────────────────────────────
    _emit("/* Auto-generated CVODES sensitivity RHS - DO NOT EDIT */")
    _emit("/* Analytical sensitivity RHS for Elementary models */")
    if functional_dfdp:
        _emit("/* GH #66/#67: both halves below also cover Functional rate laws —")
        _emit("   bngsim_dfdp carries the analytic d(func)/dp and bngsim_jac_vec the")
        _emit("   fused Functional J*v — so this bngsim_codegen_sens_rhs is complete")
        _emit("   for this model. Rate laws with a condition or a non-smooth builtin")
        _emit("   never reach here; those models decline to CVODES' difference")
        _emit("   quotient (GH #68). */")
    _emit("")
    _emit("#include <math.h>")
    _emit("#include <string.h>")
    _emit("")
    _emit(f"#define N_SPECIES {n_sp}")
    _emit(f"#define N_PARAMS  {n_params}")
    _emit("")
    # Emit unconditionally so BNGSIM_EXPORT is defined even when this sensitivity
    # source compiles without a chunked RHS ahead of it (lanl/bngsim #5). The
    # #ifndef guards no-op it when the combined .so already defined the macros.
    for ln in _CODEGEN_PRELUDE_LINES:
        _emit(ln)
    _emit("")

    # ── df/dp function: computes partial derivatives w.r.t. one parameter ──
    # Group reactions by every parameter index that contributes to dfdp[*][iP]:
    # - direct contributions from reactions whose rate constant is p_iP itself
    # - chain-rule contributions for reactions whose rate constant is a derived
    #   parameter (e.g., _rateLaw{N}) that depends on p_iP
    rxns_by_param: dict[int, list[tuple]] = {}
    for rxn in rxn_data:
        pidx = rxn["param_idx"]
        if pidx >= 0:
            rxns_by_param.setdefault(pidx, []).append(("direct", rxn))
        for term in rxn.get("derived_terms", []):
            primary_pidx, dpd_dprimary_c = term
            rxns_by_param.setdefault(primary_pidx, []).append(("derived", rxn, dpd_dprimary_c))
        # GH #55: a Michaelis–Menten ∂rate/∂p is closed form but needs locals
        # (sFree and friends), so it arrives as C lines that assign ``v`` rather
        # than as an expression the geometry gets multiplied into — its rate is
        # not k·sf·∏y^m, so there is no geometry to multiply. Never present for
        # the .net path or for an Elementary/Functional model, so their emission
        # is untouched.
        for primary_pidx, v_lines in rxn.get("mm_terms", []):
            rxns_by_param.setdefault(primary_pidx, []).append(("mm", rxn, v_lines))

    def _build_geom_terms(rxn) -> list[str]:
        sf = rxn["stat_factor"]
        terms: list[str] = []
        if sf != 1.0:
            terms.append(str(int(sf)) if sf == int(sf) else str(sf))
        # GH #75: amount_valued reactants enter by their amount (stored × V_c),
        # so the rate carries the constant ∏ V_c^mult. None ⇒ no term emitted
        # (byte-identical for .net / V=1 / hOSU=false).
        amount_factor_c = rxn.get("amount_factor_c")
        if amount_factor_c is not None:
            terms.append(amount_factor_c)
        for sp_idx, mult in sorted(rxn["reactant_mult"].items()):
            for _ in range(mult):
                terms.append(f"y[{sp_idx}]")
        return terms

    def _scatter_v(rxn, *, skip_zero: bool, out: str, abs_terms: bool) -> None:
        """Accumulate the ∂rate/∂p held in ``v`` into every affected row.

        The one place ``dfdp_out`` is written, so the GH #160 cross-compartment
        divide cannot reach one caller and miss the other. A row listed in
        ``row_divisor`` carries the same divide the RHS applies to its ``rate``:
        folded into the coefficient when the compartment volume is static (so the
        emitted arithmetic is one multiply, as it is for every other row), or a
        runtime divide by the live volume when the compartment is itself an ODE
        state. Rows without an entry — every row of every single-compartment
        model — emit exactly the pre-#160 text.

        ``abs_terms`` (issue #177) emits the same scatter with every contribution
        taken in magnitude, which is what makes ``bngsim_dfdp_term_scale`` a
        mirror rather than a second derivation: the coefficient, the volume
        divide and the reaction set are read from the same ``rxn`` here, so a
        row's term scale cannot describe a different sum than the row's value.
        ``+=`` and ``-=`` collapse to one ``+= fabs(v)`` form, because the whole
        point is the sum that does *not* cancel.
        """
        rowdiv = rxn.get("row_divisor") or {}
        for sp_idx, coeff in sorted(rxn["stoich"].items()):
            if skip_zero and coeff == 0:
                continue
            div = rowdiv.get(sp_idx)
            if abs_terms:
                # A zero row coefficient contributes no term to the value, so it
                # contributes no roundoff either — skip it rather than emit a
                # `0.0 * fabs(v)`.
                if coeff == 0:
                    continue
                if div is not None:
                    live_idx, sdiv, sdiv_param = div
                    if live_idx >= 0:
                        _emit(
                            f"        {out}[{sp_idx}] += {_jac_c_float(abs(coeff))} * fabs(v) / "
                            f"(y[{live_idx}] > 0.0 ? y[{live_idx}] : {sdiv!r});"
                        )
                    elif sdiv_param >= 0:
                        # A volume is positive, so |coeff|/V == |coeff/V| and this
                        # stays the magnitude mirror of the value branch below.
                        _emit(
                            f"        {out}[{sp_idx}] += {_jac_c_float(abs(coeff))} "
                            f"/ p[{sdiv_param}] * fabs(v);"
                        )
                    else:
                        _emit(
                            f"        {out}[{sp_idx}] += "
                            f"{_jac_c_float(abs(coeff / sdiv))} * fabs(v);"
                        )
                elif coeff in (1, -1):
                    _emit(f"        {out}[{sp_idx}] += fabs(v);")
                else:
                    _emit(f"        {out}[{sp_idx}] += {abs(coeff)} * fabs(v);")
                continue
            if div is not None:
                live_idx, sdiv, sdiv_param = div
                if live_idx >= 0:
                    _emit(
                        f"        {out}[{sp_idx}] += {_jac_c_float(coeff)} * v / "
                        f"(y[{live_idx}] > 0.0 ? y[{live_idx}] : {sdiv!r});"
                    )
                elif sdiv_param >= 0:
                    # (#170 stage 2) One correctly-rounded divide of the same two
                    # doubles the Python fold performed, then the same multiply.
                    _emit(
                        f"        {out}[{sp_idx}] += {_jac_c_float(coeff)} / p[{sdiv_param}] * v;"
                    )
                else:
                    _emit(f"        {out}[{sp_idx}] += {_jac_c_float(coeff / sdiv)} * v;")
            elif coeff == 1:
                _emit(f"        {out}[{sp_idx}] += v;")
            elif coeff == -1:
                _emit(f"        {out}[{sp_idx}] -= v;")
            elif coeff > 0:
                _emit(f"        {out}[{sp_idx}] += {coeff} * v;")
            else:
                _emit(f"        {out}[{sp_idx}] += ({coeff}) * v;")

    def _emit_dfdp_switch(out: str, abs_terms: bool) -> None:
        """The ``switch (iP)`` shared by ``bngsim_dfdp`` and its term scale.

        One traversal of ``rxns_by_param`` drives both emissions, so the two
        functions cannot come to describe different reaction sets — the failure
        shape that has bitten every paired computation site in this emitter.
        """
        _emit("    double v;")
        _emit("    switch (iP) {")
        for pidx in sorted(rxns_by_param.keys()):
            if pidx < 0:
                continue
            _emit(f"    case {pidx}:")
            for entry in rxns_by_param[pidx]:
                kind = entry[0]
                rxn = entry[1]
                if kind == "mm":
                    for _ln in entry[2]:
                        _emit(_ln)
                    # A Michaelis–Menten enzyme sits on both sides, so its net
                    # stoichiometry is always 0. The Elementary branch below emits
                    # `+= (0) * v` for such a spectator; skipping it here keeps
                    # every MM reaction from carrying one dead line per column,
                    # and Elementary emission is left exactly as it was.
                    _scatter_v(rxn, skip_zero=True, out=out, abs_terms=abs_terms)
                    continue
                geom = _build_geom_terms(rxn)
                if kind == "direct":
                    parts = list(geom) if geom else ["1.0"]
                else:
                    # chain rule: rate uses derived param p_d = f(primaries).
                    # ∂rate/∂p_iP = (∂p_d/∂p_iP) * sf * ∏y^m
                    #
                    # GH #65: this is also the shape a Functional ∂f/∂p takes —
                    # ∂f_i/∂p = Σ_r stat_r·netstoich_ir·(∂func_r/∂p)·∏R_r is the same
                    # "(derivative expression) × geometry" product with ∂func_r/∂p in
                    # place of ∂p_d/∂primary. That expression is the one that may be
                    # written in obs[]/func[] symbols, which is why the signature
                    # above is decided from these terms.
                    _, _, dpd_dprimary_c = entry
                    parts = [f"({dpd_dprimary_c})", *geom]

                _emit(f"        v = {' * '.join(parts)};")
                _scatter_v(rxn, skip_zero=False, out=out, abs_terms=abs_terms)
            _emit("        break;")

        _emit("    default:")
        _emit("        break;  /* parameter not a rate constant - dfdp = 0 */")
        _emit("    }")
        _emit("")

    def _emit_dfdp_signature(fn_name: str, out: str) -> None:
        # GH #65: obs and func are appended independently — the way the analytical
        # Jacobian picks its shard-block signature — so a derivative that reads only
        # one does not carry the other, and an Elementary body carries neither
        # (byte-identical to the pre-#65 two-line signature). Continuation lines pack
        # two parameters each, mirroring bngsim_output_sens_dfdp one level up.
        _emit(f"static void {fn_name}(int iP, double t, const double* y,")
        _params = ["const double* p"]
        if dfdp_need_obs:
            _params.append("const double* obs")
        if dfdp_need_func:
            _params.append("const double* func")
        _params.append(f"double* {out}")
        # Align continuation lines under the open paren: len("static void ") +
        # the name + "(". Keeps bngsim_dfdp's emitted text byte-identical.
        pad = " " * (len(fn_name) + 13)
        for _i in range(0, len(_params), 2):
            _pair = ", ".join(_params[_i : _i + 2])
            _tail = ") {" if _i + 2 >= len(_params) else ","
            _emit(f"{pad}{_pair}{_tail}")
        _emit(f"    memset({out}, 0, N_SPECIES * sizeof(double));")
        _emit("")

    _emit("/* Compute df/dp_{iP} - partial derivative of RHS w.r.t. parameter iP.")
    _emit("   dfdp_out[i] = sum over reactions r where p_{iP} is the rate constant:")
    _emit("     S[i][r] * sf * product_of_reactant_concs (rate without k_r)")
    _emit("   For derived rate constants (e.g., _rateLaw{N} = chi_X*kon_Y or 5/MEK),")
    _emit("   the chain-rule contributions to each primary parameter are also emitted */")
    _emit_dfdp_signature("bngsim_dfdp", "dfdp_out")
    _emit_dfdp_switch("dfdp_out", abs_terms=False)

    # Zero fixed species
    if fixed_sp:
        _emit("    /* Zero fixed species */")
        for si in sorted(fixed_sp):
            _emit(f"    dfdp_out[{si}] = 0.0;")
    _emit("}")
    _emit("")

    # ── Term scale of df/dp_{iP}: the same sum, taken in magnitude (issue #177) ──
    #
    # Emitted only for a sensitivity run. The switch is O(parameters × reactions)
    # and on a large Functional model it is a real fraction of the compile —
    # BIOMD0000000496 goes from an 18 MB .so to 29 MB, +5.6 s of one-time clang —
    # while a run that never asks for a sensitivity can never call it. Same
    # reasoning, and the same `_want_output_sens` signal, as the GH #198
    # output-sensitivity block one level up.
    if emit_term_scale:
        _emit("/* Term scale of df/dp_{iP} (issue #177): scale_out[i] is the sum of the")
        _emit("   MAGNITUDES of the very terms bngsim_dfdp sums into dfdp_out[i].")
        _emit("")
        _emit("   dfdp_out[i] is an accumulation of signed contributions, one per")
        _emit("   (reaction, row), and on a model whose species differ by many orders")
        _emit("   those contributions cancel: `dfdp_out[i] = 1e18 - 1e18` is reported as")
        _emit("   0 but carries ~eps*2e18 of roundoff. The VALUE says nothing about the")
        _emit("   size of what cancelled, so the sensitivity error test cannot tell a")
        _emit("   genuinely-zero row from a catastrophically-cancelled one and shrinks h")
        _emit("   chasing accuracy the arithmetic does not have. eps*scale_out[i] is that")
        _emit("   roundoff, per row and per column, and it is what the caller floors the")
        _emit("   sensitivity absolute tolerance with.")
        _emit("")
        _emit("   The decomposition is per (reaction, row) — it does not reach inside a")
        _emit("   single reaction's own dv expression. That is where the cancellation")
        _emit("   this addresses lives: between reactions writing the same row. */")
        _emit_dfdp_signature("bngsim_dfdp_term_scale", "scale_out")
        _emit_dfdp_switch("scale_out", abs_terms=True)

        # A fixed species has no ODE row at all, so no terms and no roundoff.
        if fixed_sp:
            _emit("    /* Zero fixed species */")
            for si in sorted(fixed_sp):
                _emit(f"    scale_out[{si}] = 0.0;")
        _emit("}")
        _emit("")

    # ── Jacobian-vector product: J * v ──────────────────────────────
    # Build one line-group per contributing reaction (those with a scalar rate
    # constant and ≥1 reactant); reactions without are skipped (no group). The
    # body is then either spliced inline (flat, byte-identical) or split into
    # NOINLINE jacv_blk_* helpers (chunked) — see _emit_chunked_blocks.
    jacv_groups: list[list[str]] = []
    for rxn_idx, rxn in enumerate(rxn_data):
        pidx = rxn["param_idx"]
        sf = rxn["stat_factor"]
        stoich = rxn["stoich"]
        rmult = rxn["reactant_mult"]

        if pidx < 0 or not rmult:
            continue

        grp: list[str] = []
        g = grp.append
        g(f"    /* Reaction {rxn_idx + 1} */")

        # GH #75: amount_valued reactants enter the rate by their amount, so
        # ∂rate/∂x_j carries the same constant ∏ V_c^mult (the y-derivative is
        # w.r.t. the stored value, and rate = k·sf·(∏V_c^m)·∏x^m). None ⇒ no
        # factor (byte-identical for .net / V=1 / hOSU=false).
        amount_factor_c = rxn.get("amount_factor_c")

        # For each unique reactant species j:
        for sp_j, m_j in sorted(rmult.items()):
            # dv_r/dx_j = k * af * sf * m_j * x_j^{m_j-1} * ∏_{l≠j} x_l^{m_l}
            parts = [f"p[{pidx}]"]
            if amount_factor_c is not None:
                parts.append(amount_factor_c)
            if sf != 1.0:
                if sf == int(sf):
                    parts.append(str(int(sf)))
                else:
                    parts.append(str(sf))
            parts.append(str(m_j))

            # x_j^{m_j - 1}
            for _ in range(m_j - 1):
                parts.append(f"y[{sp_j}]")

            # ∏_{l≠j} x_l^{m_l}
            for sp_l, m_l in sorted(rmult.items()):
                if sp_l != sp_j:
                    for _ in range(m_l):
                        parts.append(f"y[{sp_l}]")

            g(f"    dv_dxj = {' * '.join(parts)};")
            g(f"    contrib = dv_dxj * v[{sp_j}];")

            for sp_i, coeff in sorted(stoich.items()):
                if coeff == 1:
                    g(f"    Jv_out[{sp_i}] += contrib;")
                elif coeff == -1:
                    g(f"    Jv_out[{sp_i}] -= contrib;")
                elif coeff > 0:
                    g(f"    Jv_out[{sp_i}] += {coeff} * contrib;")
                else:
                    g(f"    Jv_out[{sp_i}] += ({coeff}) * contrib;")
        jacv_groups.append(grp)

    # GH #67: the Functional reactions' J·v contributions, appended after every
    # Elementary group so an Elementary model's accumulation order — and its whole
    # emitted body — is untouched. These groups are self-contained ``{ … }`` blocks
    # declaring their own locals, so they mix into the chunked blocks below without
    # needing anything from the Elementary preamble.
    jacv_groups.extend(fjacv_groups)

    # The Functional groups are the only ones that read t / obs[] / func[]; the
    # Elementary ones are written purely in y[]/p[]/v[], so the block signature is
    # byte-identical whenever there are none (mirrors how the analytical Jacobian
    # picks _blk_sig from what its bodies actually reference).
    _jacv_blk_sig = ["const double* y", "const double* p", "const double* v"]
    _jacv_blk_args = ["y", "p", "v"]
    if fjacv_groups:
        _jacv_blk_sig.insert(0, "double t")
        _jacv_blk_args.insert(0, "t")
    if jacv_need_obs:
        _jacv_blk_sig.append("const double* obs")
        _jacv_blk_args.append("obs")
    if jacv_need_func:
        _jacv_blk_sig.append("const double* func")
        _jacv_blk_args.append("func")
    _jacv_blk_sig.append("double* Jv_out")
    _jacv_blk_args.append("Jv_out")

    jacv_block_defs: list[str] = []
    jacv_call_lines: list[str] = []
    jacv_block_protos: list[str] = []
    if chunk:
        jacv_block_defs, jacv_call_lines, jacv_block_protos = _emit_chunked_blocks(
            jacv_groups,
            fn_prefix="jacv_blk",
            signature_params=", ".join(_jacv_blk_sig),
            call_args=", ".join(_jacv_blk_args),
            block_size=block_size,
            preamble=("double dv_dxj, contrib;",),
        )
        # Prototypes (kept in the driver TU) precede the definitions so
        # bngsim_jac_vec can call the blocks after they are lifted into separate
        # units by compile_rhs (GH #160).
        for ln in jacv_block_protos:
            _emit(ln)
        _emit("")
        for ln in jacv_block_defs:
            _emit(ln)
        _emit("")

    _emit("/* Compute J * v (Jacobian-vector product).")
    _emit("   J[i][j] = sum over reactions r: S[i][r] * dv_r/dx_j")
    _emit("   For elementary: dv_r/dx_j = k_r * sf * m_j * x_j^{m_j-1} * prod_{l!=j} x_l^{m_l}")
    if fjacv_groups:
        _emit("   For functional (GH #67): the same per-species chain rule and")
        _emit("   per-observable product rule the analytical Jacobian emits, with the")
        _emit("   matvec fused into the scatter — no n*n matrix is ever formed.")
    _emit("   Output: Jv_out[i] = sum_j J[i][j] * v[j] */")
    # Two parameters per line, like bngsim_dfdp above, so the Elementary
    # (t, y) / (p, v) / (Jv_out) packing is unchanged byte for byte.
    _jacv_params = ["double t", "const double* y", "const double* p", "const double* v"]
    if jacv_need_obs:
        _jacv_params.append("const double* obs")
    if jacv_need_func:
        _jacv_params.append("const double* func")
    _jacv_params.append("double* Jv_out")
    for _i in range(0, len(_jacv_params), 2):
        _pair = ", ".join(_jacv_params[_i : _i + 2])
        _head = "static void bngsim_jac_vec(" if _i == 0 else " " * 27
        _tail = ") {" if _i + 2 >= len(_jacv_params) else ","
        _emit(f"{_head}{_pair}{_tail}")
    _emit("    memset(Jv_out, 0, N_SPECIES * sizeof(double));")
    _emit("")
    if chunk:
        _emit("    (void)t;")
        lines.extend(jacv_call_lines)
    else:
        _emit("    double dv_dxj, contrib;")
        for grp in jacv_groups:
            lines.extend(grp)

    # Zero fixed species
    if fixed_sp:
        _emit("")
        _emit("    /* Zero fixed species */")
        for si in sorted(fixed_sp):
            _emit(f"    Jv_out[{si}] = 0.0;")

    _emit("}")
    _emit("")

    # ── Complete sensitivity RHS: ySdot = J * yS + df/dp_{iS} ──────
    # The obs[]/func[] fill blocks (GH #65, chunked models only) live at file
    # scope ahead of the driver, like the jacv blocks above; compile_rhs lifts
    # both into parallel translation units on the sentinel comments.
    for ln in (*obs_fs, *func_fs):
        _emit(ln)
    if obs_fs or func_fs:
        _emit("")

    _emit("/* CVODES sensitivity RHS (CVSensRhs1Fn signature).")
    _emit("   Computes: ySdot = J(t,y) * yS + df/dp_{iS}")
    _emit("   where iS is the sensitivity index and plist maps iS to param index. */")
    _emit("")
    _emit("typedef struct {")
    _emit("    double* param_values;  /* runtime parameter array */")
    _emit("    int* plist;            /* plist[iS] = parameter index for sensitivity iS */")
    _emit("    int n_sens;            /* number of sensitivity parameters */")
    _emit("} CodegenSensUserData;")
    _emit("")
    _emit("BNGSIM_EXPORT int bngsim_codegen_sens_rhs(int Ns, double t,")
    _emit("                            double* y, double* ydot,")
    _emit("                            int iS, double* yS, double* ySdot,")
    _emit("                            void* user_data,")
    _emit("                            double* tmp1, double* tmp2) {")
    _emit("    CodegenSensUserData* data = (CodegenSensUserData*)user_data;")
    _emit("    double* p = data->param_values;")
    _emit("    int iP = data->plist[iS];  /* actual parameter index */")
    _emit("")
    if obs_in or func_in:
        # GH #65: the same emitters the RHS uses, so a Functional ∂f/∂p and the
        # RHS it differentiates never read divergent intermediates. Computed per
        # call — CVSensRhs1Fn is invoked once per sensitivity column, so an
        # Ns-column step recomputes these Ns times. Deduplicating that would mean
        # caller-owned buffers on CodegenSensUserData (an ABI widening mirrored in
        # two C++ translation units); the block is a handful of statements on the
        # models this serves, so it is recomputed instead.
        _emit("    /* Observables / functions (needed by Functional df/dp) */")
        for ln in (*obs_in, *func_in):
            _emit(ln)
        _emit("")
    _dfdp_ctx = "".join(
        s for s, want in ((", obs", dfdp_need_obs), (", func", dfdp_need_func)) if want
    )
    _jacv_ctx = "".join(
        s for s, want in ((", obs", jacv_need_obs), (", func", jacv_need_func)) if want
    )
    _emit("    /* 1. Compute df/dp_{iP} */")
    _emit("    double dfdp[N_SPECIES];")
    _emit(f"    bngsim_dfdp(iP, t, y, p{_dfdp_ctx}, dfdp);")
    _emit("")
    _emit("    /* 2. Compute J * yS */")
    _emit("    double Jv[N_SPECIES];")
    _emit(f"    bngsim_jac_vec(t, y, p, yS{_jacv_ctx}, Jv);")
    _emit("")
    _emit("    /* 3. ySdot = J * yS + df/dp */")
    _emit("    for (int i = 0; i < N_SPECIES; ++i) {")
    _emit("        ySdot[i] = Jv[i] + dfdp[i];")
    _emit("    }")
    _emit("")
    _emit("    return 0;")
    _emit("}")
    _emit("")

    # ── Term scale of the sensitivity RHS (issue #177) ──────────────────────
    if not emit_term_scale:
        return "\n".join(lines) + "\n"

    # A separate entry point rather than an extra output on the RHS above: the
    # RHS is called once per column per step and this is called a handful of
    # times per run, so folding the magnitude sum into it would put a fabs and an
    # add on the hot path for every row of every step. The emitted arithmetic of
    # bngsim_codegen_sens_rhs is byte-identical to the pre-#177 text.
    #
    # This reports the ∂f/∂p half only. The J·yS half's terms are Σ_j|J_ij||s_j|,
    # which the caller already has an analytical Jacobian to form, and on the
    # models this defect appears in it is the ∂f/∂p sum that cancels: the states
    # differ by many orders and ∂f/∂p is a difference of large fluxes, while the
    # rows whose |s_j| are large are not the rows whose |s_i| has decayed to zero.
    _emit("/* Term scale of the sensitivity RHS's ∂f/∂p column (issue #177).")
    _emit("   scale_out[i] = Σ|terms| accumulated into row i of ∂f/∂p_{plist[iS]}.")
    _emit("   eps * scale_out[i] is the roundoff floor of that row: below it, the")
    _emit("   sensitivity error test is asking for accuracy float64 cannot deliver")
    _emit("   and CVODES shrinks h without bound. Same user_data as the RHS. */")
    _emit("BNGSIM_EXPORT int bngsim_codegen_sens_term_scale(int Ns, double t,")
    _emit("                            double* y, int iS, double* scale_out,")
    _emit("                            void* user_data) {")
    _emit("    CodegenSensUserData* data = (CodegenSensUserData*)user_data;")
    _emit("    double* p = data->param_values;")
    _emit("    int iP = data->plist[iS];  /* actual parameter index */")
    _emit("    (void)Ns;")
    _emit("")
    if obs_in or func_in:
        # The same recomputation the RHS does, for the same reason: a Functional
        # ∂f/∂p is written in obs[]/func[] symbols.
        _emit("    /* Observables / functions (needed by Functional df/dp) */")
        for ln in (*obs_in, *func_in):
            _emit(ln)
        _emit("")
    _emit(f"    bngsim_dfdp_term_scale(iP, t, y, p{_dfdp_ctx}, scale_out);")
    _emit("    return 0;")
    _emit("}")

    return "\n".join(lines) + "\n"


def _codegen_emit_flags(model, emit_jac: bool) -> tuple[bool, bool, bool, bool]:
    """``(want_jac, want_outputs, want_output_sens)`` for the .net codegen append,
    from cheap O(1) model flags — never generates source, so a .net cache hit stays
    a few stat()s.

    ``want_term_scale`` (issue #177): append the ∂f/∂p term scale only for a
    sensitivity run — the same ``_want_output_sens`` signal ``want_output_sens``
    reads, but WITHOUT its has-functions condition, because a model with no
    functions at all is exactly the shape the #177 reproduction has. It must reach
    the cache key below or a .so compiled for a plain run would be reused for a
    sensitivity run and silently lack the symbol (the issue #51 inertness trap).

    ``want_jac`` (GH #162): append the compiled analytical Jacobian only when an
    analytical Jacobian is wanted (``emit_jac`` — i.e. ``jacobian`` in
    ``auto``/``analytical``; ``fd``/``jax`` keep the .net RHS Jacobian-free), the
    interpreted analytical Jacobian is complete (so the compiled scatter matches it),
    and the ``BNGSIM_NO_CODEGEN_JAC`` A/B hatch is off.

    ``want_outputs`` (GH #136/#163): append the compiled output evaluator whenever
    the model has at least one observable or function and references no ``rateOf``
    csymbol — exactly the two cases ``generate_outputs_from_model`` *emits* (it
    declines on no-obs-no-func and on rateOf). This is INDEPENDENT of the Jacobian
    gate: ``fd``/``jax`` runs record observables too. ``uses_rateof`` is a (slight)
    conservative over-decline — a model with rateOf only in event triggers (never in
    functions) could in principle be emitted, but those decline cleanly to the
    interpreted recorder, which is correct. Gating the emit on this exact flag keeps
    the cache key and the emitted symbols in lock-step: ``want_outputs`` ⇒
    ``generate_outputs_from_model`` returns non-None.
    """
    core = model._core if (model is not None and hasattr(model, "_core")) else model
    if core is None:
        return False, False, False, False
    want_jac = bool(
        emit_jac
        and core.analytical_jacobian_complete
        and os.environ.get("BNGSIM_NO_CODEGEN_JAC") != "1"
    )
    want_outputs = bool((core.n_observables + core.n_functions) > 0 and not core.uses_rateof)
    # GH #198: append the expression output-sensitivity evaluator only for a
    # sensitivity run (the Simulator stashes _want_output_sens on the model before
    # codegen) AND only when there are functions to differentiate — generate_
    # output_sens_from_model declines (returns None) for the no-function /
    # no-user-function / rateOf / embedded-tfun cases, so gate on the same
    # has-functions signal to keep the cache key and emitted symbol in lock-step.
    want_output_sens = bool(
        want_outputs and core.n_functions > 0 and getattr(model, "_want_output_sens", False)
    )
    want_term_scale = bool(getattr(model, "_want_output_sens", False))
    return want_jac, want_outputs, want_output_sens, want_term_scale


def functional_sens_rhs_enabled() -> bool:
    """Whether the analytic sensitivity RHS may cover Functional rate laws (GH #67).

    ``BNGSIM_NO_FUNCTIONAL_SENS_RHS=1`` restores the pre-#67 behaviour — a
    Functional model back on CVODES' internal difference quotient — for an A/B,
    mirroring ``BNGSIM_NO_CODEGEN_JAC`` for the compiled Jacobian.

    Read through this one predicate everywhere, because the hatch has to reach two
    places that must agree: the emitter, and the **.net cache key**. The .net key is
    built from the file's bytes plus cheap flags, not from the generated source, so
    a hatch that only the emitter honoured would let a hatched run collide with a
    .so compiled without it (and the other way round) — a silently wrong A/B. The
    model-based path hashes its source, so it is safe either way; keying both off
    this makes that not a thing to remember.
    """
    return os.environ.get("BNGSIM_NO_FUNCTIONAL_SENS_RHS") != "1"


def generate_combined_c(
    net_path: str,
    model=None,
    emit_jac: bool = True,
    emit_outputs: bool = True,
    emit_output_sens: bool = False,
    emit_term_scale: bool = False,
) -> tuple[str, bool]:
    """Generate C source with RHS, sensitivity RHS (if possible), and — when the
    built model is supplied — the analytical Jacobian (GH #162), the output
    evaluator (GH #136/#163), and the expression output-sensitivity evaluator
    (GH #198).

    Returns ``(c_source, has_sens_rhs)``.

    ``model`` is the built model (``Model`` or its ``_core``) for this *same* .net.
    When given, model-based callbacks are appended after the RHS (in the same
    RHS, sens, Jacobian, outputs, output-sens order as
    ``generate_combined_from_model``):

    * the analytical Jacobian (``generate_jacobian_from_model`` — dense, or sparse
      CSC for KLU-routed models) when ``emit_jac`` — so a .net-loaded large sparse
      model gets a **compiled** per-step Jacobian instead of the interpreted one
      (GH #162);
    * the output evaluator (``generate_outputs_from_model`` — ``bngsim_codegen_outputs``)
      when ``emit_outputs`` — so the warm recording loop fills the per-row observable
      and function buffers with one compiled call instead of re-walking the ExprTk
      trees for every observable/function at every output row (GH #136). Unlike the
      Jacobian, this applies to *every* ``jacobian`` strategy (GH #163).
    * the expression output-sensitivity evaluator (``generate_output_sens_from_model``
      — ``bngsim_codegen_output_sens``) when ``emit_output_sens`` — the GH #198
      chain-rule ``d func/dθ``. Gated separately because its build-time expression
      differentiation is expensive and only a sensitivity run needs it; the .net
      cache key carries the flag (``prepare_codegen``).

    The append is sound because the .net RHS already emits the ``CodegenUserData``
    typedef and the ``N_SPECIES``/``N_OBS``/``N_FUNC`` macros both callbacks reuse,
    and the .net parse and the built model agree on species/parameter/observable
    ordering (the model is built from the .net). ``model=None`` keeps the historical
    RHS(+sens)-only output byte-for-byte. A ``None`` from either emitter (an
    incomplete/un-emittable Jacobian or the A/B hatch; a rateOf / no-obs-no-func
    model for outputs) simply omits that symbol — never a partial/wrong one, and the
    simulator falls back to the interpreted Jacobian / interpreted recorder.
    """
    rhs_code = generate_rhs_c(net_path)
    sens_code = generate_sens_rhs_c(net_path, emit_term_scale=emit_term_scale)
    if sens_code is None and model is not None and functional_sens_rhs_enabled():
        # GH #67: the .net emitter reads rate laws as text and has no rate-law
        # expression to differentiate, so it declines every Functional model. The
        # built model does — and this is the path a .net-loaded model actually
        # takes, so without this hook #67 would reach only the SBML/Antimony
        # entry points. Same append-from-the-model shape as the Jacobian and the
        # output evaluators above, and sound for the same reason: the model is
        # built from this .net, so the two agree on species/parameter ordering.
        # Only ever tried once the .net path has already declined, so an
        # all-Elementary model's source stays byte-for-byte what it was.
        sens_code = generate_sens_from_model(
            model, functional=True, emit_term_scale=emit_term_scale
        )
    parts = [rhs_code]
    if sens_code is not None:
        parts.append(sens_code)
    if model is not None:
        if emit_jac:
            jac_code = generate_jacobian_from_model(model)
            if jac_code is not None:
                parts.append(jac_code)
        if emit_outputs:
            outputs_code = generate_outputs_from_model(model)
            if outputs_code is not None:
                parts.append(outputs_code)
        # Expression output sensitivities (GH #198) are appended only for a
        # sensitivity run — the build-time differentiation is expensive and wasted
        # otherwise. The cache key carries emit_output_sens so a non-sensitivity
        # .so is never reused for a sensitivity run (see prepare_codegen).
        if emit_output_sens:
            output_sens_code = generate_output_sens_from_model(model)
            if output_sens_code is not None:
                parts.append(output_sens_code)
    return "\n".join(parts), sens_code is not None


# ─── Compilation + caching ───────────────────────────────────────────


def compute_model_hash(net_path: str) -> str:
    """Compute a hash of the .net file content for caching.

    The hash mixes in ``_CODEGEN_CACHE_KEY`` — the hand-maintained
    ``_CODEGEN_VERSION`` *and* a digest of the emitters' own source (issue #51)
    — so a codegen behavior change invalidates previously-cached .so files
    whether or not the constant was bumped. Any .tfun data files referenced by
    the .net's function block are also folded in, so editing a tfun's y-values
    triggers a recompile.
    """
    h = hashlib.sha256()
    h.update(_CODEGEN_CACHE_KEY.encode())
    h.update(b"\0")
    with open(net_path, "rb") as f:
        net_bytes = f.read()
    h.update(net_bytes)

    # Walk the function block for tfun('file.tfun', ...) references.
    # Resolve relative paths against the .net's directory; silently skip
    # missing files (the build will fail loudly when it hits them).
    net_dir = Path(net_path).parent
    for ref in _iter_tfun_file_refs(net_bytes.decode("utf-8", errors="replace")):
        ref_path = Path(ref)
        if not ref_path.is_absolute():
            ref_path = net_dir / ref_path
        try:
            with open(ref_path, "rb") as f:
                h.update(b"\0tfun\0")
                h.update(ref.encode("utf-8"))
                h.update(b"\0")
                h.update(f.read())
        except OSError:
            # Missing or unreadable — leave it out of the hash. The
            # downstream model load will surface the error.
            continue

    return h.hexdigest()[:16]


def _iter_tfun_file_refs(net_text: str):
    """Yield each filename argument from tfun('file', ...) inside a .net's
    functions block. Inline tfuns (tfun([…],[…],…)) and non-function uses
    are skipped.
    """
    in_functions = False
    for raw_line in net_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("begin functions"):
            in_functions = True
            continue
        if line.startswith("end functions"):
            in_functions = False
            continue
        if not in_functions:
            continue
        # Strip trailing comment
        comment = line.find("#")
        if comment >= 0:
            line = line[:comment]
        m = re.search(r"\btfun\s*\(\s*['\"]([^'\"]+)['\"]", line)
        if m:
            yield m.group(1)


def _shared_lib_suffix() -> str:
    """Return the platform-specific shared library file extension."""
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    elif system == "Windows":
        return ".dll"
    else:
        return ".so"


def _find_c_compiler() -> list[str]:
    """Find the best available C compiler and return its base command.

    Returns a list of command-line tokens for the compiler invocation.

    Search order:
      1. CC environment variable (user override)
      2. Platform defaults:
         - Windows: cl.exe (MSVC), then gcc (MinGW)
         - Unix: cc
    """
    import shutil

    # Honor CC environment variable
    cc_env = os.environ.get("CC")
    if cc_env:
        return [cc_env]

    system = platform.system()
    if system == "Windows":
        # Try MSVC cl.exe first, then MinGW gcc
        if shutil.which("cl"):
            return ["cl"]
        if shutil.which("gcc"):
            return ["gcc"]
        raise RuntimeError(
            "No C compiler found. Install Visual Studio Build Tools (cl.exe) or MinGW (gcc)."
        )
    else:
        # Unix: cc is the standard symlink to the default compiler
        return ["cc"]


def _resolve_codegen_timeout() -> float | None:
    """Return the cc timeout in seconds, honoring BNGSIM_CODEGEN_TIMEOUT.

    A value of ``0`` (or any non-positive number) disables the timeout.
    """
    raw = os.environ.get("BNGSIM_CODEGEN_TIMEOUT")
    if raw is None:
        return float(_DEFAULT_CODEGEN_TIMEOUT)
    try:
        val = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid BNGSIM_CODEGEN_TIMEOUT=%r; using default %d s",
            raw,
            _DEFAULT_CODEGEN_TIMEOUT,
        )
        return float(_DEFAULT_CODEGEN_TIMEOUT)
    return val if val > 0 else None


def _resolve_opt_flag(compiler_name: str, source_size: int, chunked: bool = False) -> str:
    """Pick the optimization flag for the compiler and C-source size.

    Defaults to a high level for small sources, a low level once the source
    crosses _CODEGEN_BIG_SOURCE_BYTES, and no optimization (-O0) once it crosses
    _CODEGEN_HUGE_SOURCE_BYTES (where -O1's compile time would blow the timeout).
    BNGSIM_CODEGEN_OPT overrides the choice with an explicit level: an integer
    0-3, or the words "high"/"low"/"none".

    ``chunked`` sources (Tier-1 NOINLINE blocks) sidestep the size-based downshift
    entirely: the size tiers exist only to tame the flat giant function, which
    chunking eliminates, so a chunked source compiles at the "medium" level
    (-O2 / /O2) at any size — full optimization in minutes, not -O0 forever.
    """
    is_msvc = compiler_name == "cl"
    # MSVC accepts /Od (off), /O1 (size) and /O2 (speed); cap levels accordingly.
    high = "/O2" if is_msvc else "-O3"
    medium = "/O2" if is_msvc else "-O2"
    low = "/O1" if is_msvc else "-O1"
    none = "/Od" if is_msvc else "-O0"

    override = os.environ.get("BNGSIM_CODEGEN_OPT")
    if override is not None:
        token = override.strip().lower()
        if token == "high":
            return high
        if token == "low":
            return low
        if token == "none":
            return none
        if token.isdigit():
            level = int(token)
            if is_msvc:
                # MSVC has no /O0 or /O3; map 0→/Od, 1→/O1, ≥2→/O2.
                return "/Od" if level == 0 else ("/O1" if level == 1 else "/O2")
            return f"-O{min(level, 3)}"
        logger.warning(
            "Ignoring invalid BNGSIM_CODEGEN_OPT=%r; using size-based default",
            override,
        )

    if chunked:
        return medium
    if source_size > _CODEGEN_HUGE_SOURCE_BYTES:
        return none
    return low if source_size > _CODEGEN_BIG_SOURCE_BYTES else high


def _build_compile_cmd(c_path: Path, so_path: Path, opt_flag: str) -> list[str]:
    """Build the platform-specific compile command.

    Parameters
    ----------
    c_path : Path
        Path to the C source file.
    so_path : Path
        Path to the output shared library.
    opt_flag : str
        Optimization flag for the detected compiler (e.g. "-O3" / "/O2").

    Returns
    -------
    list[str]
        Command-line tokens for subprocess.run().
    """
    compiler = _find_c_compiler()
    compiler_name = Path(compiler[0]).stem.lower()

    if compiler_name == "cl":
        # MSVC: cl <opt> /LD /Fe:<output> <input> /link /DLL
        return compiler + [
            opt_flag,
            "/LD",
            f"/Fe:{so_path}",
            str(c_path),
            "/link",
            "/DLL",
        ]
    else:
        # GCC / Clang (Unix): cc <opt> -shared -fPIC -o <output> <input> -lm
        return compiler + [
            opt_flag,
            "-shared",
            "-fPIC",
            "-o",
            str(so_path),
            str(c_path),
            "-lm",
        ]


def get_cached_so(model_hash: str) -> Path | None:
    """Return path to cached shared library if it exists."""
    suffix = _shared_lib_suffix()
    so_path = CACHE_DIR / f"rhs_{model_hash}{suffix}"
    if so_path.exists():
        return so_path
    return None


def _allocation_cpu_count() -> int:
    """CPUs this process may actually run on — the Slurm/cgroup allocation, not
    the machine's core count (GH #160).

    ``os.cpu_count()`` reports every core on the node; on a shared HPC node that
    would spawn ~Nnode compilers inside a small cgroup → throttled and antisocial.
    The affinity mask (Linux ``sched_getaffinity``) reflects the cpuset/cgroup the
    kernel actually enforces; ``SLURM_CPUS_PER_TASK`` is honored as an additional
    cap. Falls back to ``os.cpu_count()`` only where neither is available
    (e.g. macOS has no ``sched_getaffinity`` — fine, that is the laptop case).
    """
    n: int | None = None
    getaff = getattr(os, "sched_getaffinity", None)
    if getaff is not None:
        try:
            n = len(getaff(0))
        except OSError:
            n = None
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            s = int(slurm)
        except ValueError:
            s = 0
        if s > 0:
            n = s if n is None else min(n, s)
    if n is None or n < 1:
        n = os.cpu_count() or 1
    return n


def _read_int_file(path: str) -> int | None:
    """Read a single integer from ``path``; None if missing/unparsable. cgroup
    memory files use the literal ``max`` for "no limit" — treated as None."""
    try:
        with open(path) as f:
            raw = f.read().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _available_memory_bytes() -> int | None:
    """Best-effort available RAM for this allocation, or None if undeterminable.

    Honors cgroup limits (the HPC case) so the memory cap reflects the job's real
    budget, not the node's total RAM, and folds in ``/proc/meminfo`` MemAvailable
    (actual free). On macOS, where neither exists, falls back to ``vm_stat`` so
    the parallel-compile memory cap applies on a laptop too (GH #168 follow-up:
    an unbounded cold codegen under memory pressure could otherwise overcommit
    and get jetsam-killed). Returns the minimum of whatever is readable, or None
    if nothing is — in which case the caller uses the CPU cap alone.
    """
    candidates: list[int] = []
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        v = _read_int_file(path)
        # cgroup v1 encodes "unlimited" as a huge sentinel (~PAGE_COUNTER_MAX),
        # not "max" — ignore implausibly large values.
        if v is not None and 0 < v < (1 << 62):
            candidates.append(v)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    # macOS has no /proc or cgroup; derive available RAM from vm_stat instead.
    if not candidates and platform.system() == "Darwin":
        mac = _macos_available_memory_bytes()
        if mac is not None and mac > 0:
            candidates.append(mac)
    return min(candidates) if candidates else None


def _macos_available_memory_bytes() -> int | None:
    """Available RAM on macOS, parsed from ``vm_stat`` as the page-count sum of
    free + inactive + speculative + purgeable (its MemAvailable analogue).

    Deliberately conservative — it omits the compressor's reclaimable pages — so
    it under-counts rather than over-counts available memory, which is the safe
    direction for a parallel-compile job cap (under-subscribe RAM, never OOM).
    Returns None if ``vm_stat`` is missing or unparseable, so the caller falls
    back to the CPU cap alone.
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 4096
    pages = 0
    matched = False
    for key in ("free", "inactive", "speculative", "purgeable"):
        mm = re.search(rf"Pages {key}:\s+(\d+)\.", out)
        if mm:
            pages += int(mm.group(1))
            matched = True
    return pages * page if matched else None


def _per_job_memory_bytes() -> int:
    """Estimated peak RSS of one shard compile, honoring BNGSIM_CODEGEN_MEM_PER_JOB
    (megabytes)."""
    raw = os.environ.get("BNGSIM_CODEGEN_MEM_PER_JOB", "").strip()
    if raw:
        try:
            mb = float(raw)
        except ValueError:
            mb = 0.0
        if mb > 0:
            return int(mb * 1024 * 1024)
        logger.warning(
            "Ignoring invalid BNGSIM_CODEGEN_MEM_PER_JOB=%r; using %d MB",
            raw,
            _DEFAULT_SHARD_MEM_MB,
        )
    return _DEFAULT_SHARD_MEM_MB * 1024 * 1024


def _resolve_codegen_jobs(n_units: int) -> int:
    """How many compilers to run in parallel for a sharded build (GH #160).

    Allocation-aware (never ``os.cpu_count()``) and memory-bounded, capped at the
    unit count. ``BNGSIM_CODEGEN_JOBS`` overrides: a positive integer is a hard
    core cap (``1`` ⇒ serial, no parallelism); ``auto``/``0``/empty selects the
    allocation-aware default. The memory cap always applies (raise
    ``BNGSIM_CODEGEN_MEM_PER_JOB`` to loosen it) so an over-large override cannot
    OOM a node.
    """
    if n_units <= 1:
        return 1

    raw = os.environ.get("BNGSIM_CODEGEN_JOBS", "").strip().lower()
    explicit: int | None = None
    if raw and raw not in ("auto", "0"):
        try:
            explicit = int(raw)
        except ValueError:
            logger.warning(
                "Ignoring invalid BNGSIM_CODEGEN_JOBS=%r; using allocation-aware auto",
                raw,
            )
        else:
            if explicit < 1:
                explicit = None

    cpu = explicit if explicit is not None else _allocation_cpu_count()
    jobs = max(1, min(cpu, n_units))

    avail = _available_memory_bytes()
    if avail is not None:
        per_job = _per_job_memory_bytes()
        mem_jobs = max(1, int(avail // per_job))
        if mem_jobs < jobs:
            logger.info(
                "Codegen: capping parallel compile to %d job(s) by memory "
                "(%.1f GB available / %d MB per job; %d core(s) allowed)",
                mem_jobs,
                avail / 1e9,
                per_job // (1024 * 1024),
                cpu,
            )
            jobs = mem_jobs
    return jobs


def _split_sharded_source(c_source: str) -> tuple[str, list[str]] | None:
    """Split a chunked codegen source into ``(driver_src, [unit_src, ...])`` for
    parallel compilation, or None if it carries no shard blocks.

    The driver holds everything except the NOINLINE block bodies (headers,
    prototypes, dispatchers); each unit holds a fixed group of blocks prefixed
    with the source's own header (everything before the first block), so the unit
    sees every macro/typedef/file-scope table the blocks reference (e.g. the
    model-based ``inv_vf``) and compiles standalone. The blocks have external
    linkage and the driver carries their prototypes, so the units' ``.o`` files
    resolve the driver's calls at link time.

    The partition is a pure function of the source — independent of the job count
    — so the linked ``.so`` is identical no matter how many compilers run (the
    GH #160 determinism constraint). The job count only sets concurrency.
    """
    lines = c_source.split("\n")
    try:
        first = lines.index(_SHARD_BLOCK_OPEN)
    except ValueError:
        return None

    # Source prefix (declarations, macros, file-scope tables) — prepended to
    # every unit and retained by the driver. Self-adjusting: whatever the
    # generator emits before the blocks is exactly what the blocks may reference.
    preamble = "\n".join(lines[:first])

    driver_lines: list[str] = list(lines[:first])
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for ln in lines[first:]:
        if ln == _SHARD_BLOCK_OPEN:
            current = []
            continue
        if ln == _SHARD_BLOCK_CLOSE:
            if current is not None:
                blocks.append(current)
                current = None
            continue
        if current is not None:
            current.append(ln)
        else:
            driver_lines.append(ln)

    if not blocks:
        return None

    driver_src = "\n".join(driver_lines)
    units: list[str] = []
    for start in range(0, len(blocks), _SHARD_UNIT_BLOCKS):
        group = blocks[start : start + _SHARD_UNIT_BLOCKS]
        body = "\n".join("\n".join(blk) for blk in group)
        units.append(f"{preamble}\n\n{body}\n")
    return driver_src, units


# Fixed timestamp stamped onto shard objects before linking so the linker embeds
# no per-build mtime (macOS records input mtimes in the debug map) — part of
# making the .so byte-identical across builds and job counts (GH #160). 1980-01-01;
# some tools reject a zero epoch.
_SHARD_REPRO_EPOCH = 315532800


def _repro_link_flags() -> list[str]:
    """Linker flags that drop per-build non-determinism so the sharded .so is
    byte-identical regardless of job count (GH #160).

    Linux: ``--build-id=none`` — guards against toolchains that stamp a *random*
    build id (the GNU ld default is a content hash, already deterministic, but
    some setups use a UUID). Safe because Linux ``dlopen`` does not require it.

    macOS: nothing. ld64's ``LC_UUID`` is a content hash, so it is already
    deterministic once the inputs are (relative names + normalized mtimes, below)
    — and it must be kept, because macOS ``dlopen`` *rejects* a dylib that is
    missing ``LC_UUID``.

    Sharding is gated to gcc/clang on these two platforms, so no other linker
    sees these flags.
    """
    if platform.system() == "Linux":
        return ["-Wl,--build-id=none"]
    return []


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and every process it spawned (GH #166).

    A compiler driver (``cc``/``clang``/``cl``) execs a backend — ``clang -cc1``
    on POSIX, ``c1``/``c2`` under MSVC — as a separate process. ``subprocess``'s
    own timeout/kill only signals the immediate driver, so the backend is
    reparented to PID 1 and keeps compiling, pegging a core for tens of minutes
    on a genome-scale source. We launch each compile in its own session /
    process group (see ``_run_compile``) so the whole tree can be torn down."""
    if os.name == "posix":
        # already gone, or reaped between getpgid and killpg
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    else:
        # CREATE_NEW_PROCESS_GROUP only governs Ctrl-C/Ctrl-Break delivery, not
        # tree teardown — taskkill /T reaps the MSVC backend children.
        with contextlib.suppress(FileNotFoundError, OSError):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()  # belt-and-suspenders for the direct child


def _run_compile(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a compile/link command so a timeout *or* an abort kills the whole
    process group, not just the immediate compiler driver (GH #166).

    Drop-in for ``subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
    timeout=timeout)``: same ``CompletedProcess`` result, and ``TimeoutExpired``
    is still raised on timeout — but only after the backend grandchildren
    (``clang -cc1`` &c.) have been reaped, so nothing survives the call. Any
    abort mid-compile (KeyboardInterrupt, a SIGTERM handler) tears the group
    down too, since ``start_new_session`` detaches the children from the
    terminal's foreground group and our explicit kill is then the only reaper."""
    popen_kwargs: dict = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        # CREATE_NEW_PROCESS_GROUP is Windows-only; this branch only runs there.
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        proc.communicate()  # reap the killed group before propagating
        raise
    except BaseException:
        _kill_process_group(proc)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _compile_one_object(
    compiler: list[str],
    opt_flag: str,
    work: Path,
    c_name: str,
    o_name: str,
    timeout: float | None,
) -> None:
    """Compile one ``.c`` to ``.o`` (``cc <opt> -fPIC -c``) from within ``work``
    using relative names, then normalize the object's mtime. Relative names +
    fixed mtimes keep the linked .so reproducible (the linker would otherwise
    embed absolute input paths and per-build timestamps). Raises on failure."""
    cmd = compiler + [opt_flag, "-fPIC", "-c", "-o", o_name, c_name]
    result = _run_compile(cmd, cwd=work, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Codegen shard compilation failed ({c_name}):\n{result.stderr}")
    os.utime(work / o_name, (_SHARD_REPRO_EPOCH, _SHARD_REPRO_EPOCH))


def _compile_sharded(
    driver_src: str,
    units: list[str],
    tmp_so_path: Path,
    opt_flag: str,
    jobs: int,
    timeout: float | None,
    compiler: list[str],
) -> None:
    """Compile the driver + shard units to ``.o`` with up to ``jobs`` concurrent
    ``cc -c`` processes, then link them into ``tmp_so_path``.

    Object compiles run in a thread pool — each thread blocks in ``subprocess``,
    which releases the GIL, so ``jobs`` compilers run genuinely in parallel.

    The result is byte-identical regardless of job count (the GH #160 determinism
    constraint): the source partition is job-count-independent, the link order is
    fixed (driver first, then units in index order), and the build is made
    reproducible — compiles run from the scratch dir with relative names, object
    mtimes are normalized, and a reproducibility linker flag drops the content
    UUID / build id and the linked library's name is relative. The scratch
    directory is removed on the way out.
    """
    import concurrent.futures
    import shutil
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="bngsim_shard_", dir=CACHE_DIR))
    try:
        names: list[tuple[str, str]] = []  # (c_name, o_name), link order
        # Always UTF-8: the generated source carries non-ASCII comment glyphs
        # (→, −, ·). Path.write_text defaults to the locale encoding, which is
        # cp1252 on Windows and raises UnicodeEncodeError on those bytes.
        (work / "driver.c").write_text(driver_src, encoding="utf-8")
        names.append(("driver.c", "driver.o"))
        for i, unit_src in enumerate(units):
            (work / f"unit_{i:04d}.c").write_text(unit_src, encoding="utf-8")
            names.append((f"unit_{i:04d}.c", f"unit_{i:04d}.o"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [
                ex.submit(_compile_one_object, compiler, opt_flag, work, cn, on, timeout)
                for cn, on in names
            ]
            # Surface the first failure (cancels not-yet-started compiles).
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        linked = f"linked{tmp_so_path.suffix}"
        link_cmd = (
            compiler
            + ["-shared", "-fPIC"]
            + _repro_link_flags()
            + ["-o", linked]
            + [on for _, on in names]
            + ["-lm"]
        )
        result = _run_compile(link_cmd, cwd=work, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Codegen shard link failed:\n{result.stderr}")
        os.replace(work / linked, tmp_so_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def compile_rhs(c_source: str, model_hash: str) -> Path:
    """Compile C source to shared library, cached by model hash.

    Uses platform-aware compiler detection and shared library naming.
    NO -ffast-math for strict IEEE 754 compliance.

    Large chunked sources are compiled in parallel: the NOINLINE blocks are split
    into independent translation units, compiled with an allocation-aware,
    memory-bounded pool of ``cc -c``, and linked into the ``.so`` (GH #160). A
    1-core allocation (or ``BNGSIM_CODEGEN_JOBS=1``) takes the unchanged serial
    path; MSVC always does.

    Parameters
    ----------
    c_source : str
        Complete C source code from generate_rhs_c().
    model_hash : str
        Hash of the .net file content.

    Returns
    -------
    Path
        Path to the compiled shared library.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    suffix = _shared_lib_suffix()
    so_path = CACHE_DIR / f"rhs_{model_hash}{suffix}"

    # Check cache
    if so_path.exists():
        logger.debug("Using cached codegen lib: %s", so_path)
        return so_path

    # Compile to process-unique temp paths, then os.replace() into the cached
    # name. Concurrent Dask workers race to build the same model_hash; writing
    # to shared paths risks a compiler reading a half-written .c or a caller
    # loading a partially-linked .so. os.replace() is atomic on POSIX/Windows.
    token = f"{os.getpid()}_{next(_compile_counter)}"
    tmp_so_path = CACHE_DIR / f"rhs_{model_hash}.{token}{suffix}"

    compiler = _find_c_compiler()
    compiler_name = Path(compiler[0]).stem.lower()
    # A chunked source (Tier-1 NOINLINE blocks) compiles at -O2 at any size; the
    # marker is on the first lines of the combined source (RHS is always first).
    chunked = _CHUNK_MARKER in c_source[:512]
    opt_flag = _resolve_opt_flag(compiler_name, len(c_source), chunked=chunked)
    timeout = _resolve_codegen_timeout()

    # Decide serial vs. sharded parallel compile. MSVC keeps the single-shot
    # serial path (the HPC parallel case targets gcc/clang); a chunked source with
    # ≥2 units and a multi-core allocation shards.
    sharded = _split_sharded_source(c_source) if (chunked and compiler_name != "cl") else None
    jobs = _resolve_codegen_jobs(len(sharded[1])) if sharded is not None else 1

    try:
        if sharded is not None and jobs > 1:
            driver_src, units = sharded
            logger.info(
                "Compiling codegen RHS (%s): sharded — %d unit(s), %d parallel job(s)",
                opt_flag,
                len(units),
                jobs,
            )
            _compile_sharded(driver_src, units, tmp_so_path, opt_flag, jobs, timeout, compiler)
            os.replace(tmp_so_path, so_path)
        else:
            c_path = CACHE_DIR / f"rhs_{model_hash}.{token}.c"
            # UTF-8: generated source has non-ASCII comment glyphs; the locale
            # default (cp1252 on Windows) would raise UnicodeEncodeError.
            c_path.write_text(c_source, encoding="utf-8")
            try:
                cmd = _build_compile_cmd(c_path, tmp_so_path, opt_flag)
                logger.info("Compiling codegen RHS (%s): %s", opt_flag, " ".join(cmd))
                result = _run_compile(cmd, timeout=timeout)
                if result.returncode != 0:
                    raise RuntimeError(f"Codegen compilation failed:\n{result.stderr}")
                os.replace(tmp_so_path, so_path)
            finally:
                c_path.unlink(missing_ok=True)
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(
            f"Codegen compilation timed out after {timeout:g} s "
            f"({len(c_source) / 1e6:.1f} MB C source at {opt_flag}). The RHS, the "
            "analytical Jacobian, and the output evaluator (including their "
            "observable/function computation) are all sharded into parallel "
            "translation units (GH #160/#165), so the wall is roughly the slowest "
            "unit plus the link — a timeout at this size usually means too few "
            "compile jobs (a low core count or a small allocation), not a single "
            "serial bottleneck. Give it more cores or raise BNGSIM_CODEGEN_JOBS, "
            "raise the budget with BNGSIM_CODEGEN_TIMEOUT (seconds; 0 disables the "
            "limit), or skip codegen for this run with Simulator(..., codegen=False) "
            "(integrates on the interpreted RHS, no compile step)."
        ) from err
    except FileNotFoundError as err:
        raise RuntimeError(
            "C compiler not found. Install Xcode Command Line Tools "
            "(macOS), gcc (Linux), or Visual Studio Build Tools (Windows)."
        ) from err
    finally:
        tmp_so_path.unlink(missing_ok=True)

    logger.info("Compiled codegen lib: %s", so_path)
    return so_path


# ─── Model-based codegen for SBML/Antimony ──────────────────────────────


def _expr_to_c(
    expr: str,
    param_names: list[str],
    species_names: list[str],
    obs_names: list[str],
    func_names: list[str],
) -> str:
    """Translate an ExprTk expression string to a C expression string.

    Handles:
    - Parameter references → p[idx]
    - Species references → y[idx]
    - Observable references → obs[idx]
    - Function references → func[idx]
    - if(cond, t, f) → ((cond) ? (t) : (f))
    - time() / t() → t
    - ^ (power) → pow(a, b)
    - and / or / not → && / || / !
    - Standard math functions pass through to C math.h
    - _pi → M_PI, _e → M_E

    Parameters
    ----------
    expr : str
        ExprTk expression string.
    param_names : list[str]
        Parameter names (order = index).
    species_names : list[str]
        Species names (order = index).
    obs_names : list[str]
        Observable names (order = index).
    func_names : list[str]
        Function names (order = index).

    Returns
    -------
    str
        C expression string.
    """
    # Build name→C-ref maps
    param_map = {name: f"p[{i}]" for i, name in enumerate(param_names)}
    species_map = {name: f"y[{i}]" for i, name in enumerate(species_names)}
    obs_map = {name: f"obs[{i}]" for i, name in enumerate(obs_names)}
    func_map = {name: f"func[{i}]" for i, name in enumerate(func_names)}

    lookup = _build_ident_lookup_model(param_map, species_map, obs_map, func_map)
    return _translate_expr_to_c(expr, lookup)


_IDENT_OR_EMPTY_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)(\s*\(\s*\))?")
_PAREN_AFTER_RE = re.compile(r"\s*\(")

# A bare integer literal: a digit run not glued to an identifier, a decimal
# point, or an exponent. The ``(?<![\w.])`` / ``(?![\w.])`` guards keep us
# off identifier suffixes (``A_1``, ``p2``), float mantissas/fractions
# (``1.5`` → neither ``1`` nor ``5`` matches), and bare-``e`` exponents
# (``1e9`` → ``1`` precedes ``e``∈``\w`` and ``9`` follows it). The extra
# ``(?<![eE][-+])`` guard rejects the *signed* exponent digits a ``[\w.]``
# lookbehind can't see past — without it ``2.5E-3`` would become the invalid
# ``2.5E-3.0``.
_INT_LITERAL_RE = re.compile(r"(?<![\w.])(?<![eE][-+])\d+(?![\w.])")


def _floatify_int_literals(expr: str) -> str:
    """Append ``.0`` to every bare integer literal in an ExprTk expression.

    ExprTk evaluates all arithmetic in ``double``, so ``1/2`` is ``0.5``.
    C does integer division on two integer literals, so the *same* string
    compiled by the codegen path makes ``1/2`` collapse to ``0`` — silently
    zeroing any rate law that carries a rational constant (e.g. a sigmoid
    ``(1/2)*(x/sqrt(x^2+1)+1)`` from an SBML ``functionDefinition`` rendered
    as ``(1/2)``). Promoting each integer literal to a floating literal makes
    the generated C match ExprTk's double semantics. Array subscripts
    (``p[0]``, ``y[5]``) are introduced *after* this pass by identifier
    substitution, so their indices are never touched. ``pow()`` exponents
    accept doubles, so ``x^2`` → ``x^2.0`` → ``pow(x, 2.0)`` is unaffected.

    Only matters for models routed through codegen (the SBML loader auto-
    enables it at ≥256 species, and any sensitivity workflow). Surfaced by
    MODEL1112100000 (1012-species WUSCHEL model): every ``Wus_*`` synthesis
    used a ``Sigma`` sigmoid whose leading ``(1/2)`` codegen'd to ``0``, so
    all ``Wus`` species froze at their initial value under the codegen RHS
    while the ExprTk RHS (and RoadRunner) grew them correctly.
    """
    return _INT_LITERAL_RE.sub(lambda m: m.group(0) + ".0", expr)


# ExprTk → C builtins for bare-identifier and word-form-operator replacements.
# Values are (replacement, eats_empty_parens). Func/obs/species/param maps are
# merged on top, all with eats_empty_parens=True: BNGL accepts a zero-arg call
# (`name()`) wherever the bareword is valid, and BNG2.pl preserves whichever
# form the user wrote when emitting the .net (issue #28). Every one of those
# names denotes a *scalar* in the generated C — `obs[3]`, `p[7]`, `y[2]`,
# `func[0]` — so the trailing `()` must be dropped, exactly as
# ``ExprTkEvaluator::compile`` does via ``strip_empty_parens`` (src/expression.cpp)
# for every name registered as a scalar variable. Leaving it in emits
# `obs[3]()`, which C rejects with "called object type 'double' is not a
# function". eats_empty_parens=False survives only for the entries that really
# are C *functions* (fabs/log/round/fmax/fmin) or operators (&&/||/!), where
# `name()` is not valid ExprTk in the first place and the parens must survive
# for the arguments that follow.
_BUILTIN_IDENT_MAP: dict[str, tuple[str, bool]] = {
    "time": ("t", True),
    "t": ("t", True),
    # Registered as remapped *constants* on the ExprTk evaluator, so
    # strip_empty_parens() strips `_pi()` → `_pi` there too.
    "_pi": ("M_PI", True),
    "_e": ("M_E", True),
    "and": ("&&", False),
    "or": ("||", False),
    "not": ("!", False),
    "ln": ("log", False),
    "rint": ("round", False),
    "abs": ("fabs", False),
    # ExprTk max/min have no C equivalent under those names; <math.h> spells
    # them fmax/fmin. The loader emits nested binary max()/min() for n-ary forms,
    # so the binary C builtins suffice. (Both are ExprTk-reserved, so they can
    # never be user-defined model symbols that would need to win the lookup.)
    "max": ("fmax", False),
    "min": ("fmin", False),
}


def _build_ident_lookup_model(
    param_map: dict[str, str],
    species_map: dict[str, str],
    obs_map: dict[str, str],
    func_map: dict[str, str],
    rateof_map: dict[str, str] | None = None,
) -> dict[str, tuple[str, bool]]:
    """Build the identifier → (C-reference, eats-empty-call) table the
    model-based ``_translate_expr_to_c`` rewrites function/observable bodies
    against.

    Built ONCE per emitter (``_emit_function_lines``) and reused for every body.
    Building it per call is O(n_bodies × (n_params + n_species + n_obs + n_funcs))
    — the model-based twin of the GH #161 ``_translate_expr`` quadratic. On a
    genome-scale model that is ~245k entries (species included) rebuilt for each
    of ~18k function bodies. Issue #25 already hoisted the source *maps* out of
    the per-call path; this hoists the combined lookup the maps feed.

    Priority (later overrides earlier, so it wins): parameter < species <
    observable < function, then rateOf accessors (GH #106) — matching the prior
    cascade.

    Every model name eats a trailing ``()`` (issue #28 — see
    ``_BUILTIN_IDENT_MAP``): each resolves to a scalar in the emitted C, so
    ``divide()`` in a function body must become ``obs[1]``, not ``obs[1]()``.
    """
    lookup: dict[str, tuple[str, bool]] = dict(_BUILTIN_IDENT_MAP)
    for name, rep in param_map.items():
        lookup[name] = (rep, True)
    for name, rep in species_map.items():
        lookup[name] = (rep, True)
    for name, rep in obs_map.items():
        lookup[name] = (rep, True)
    for name, rep in func_map.items():
        lookup[name] = (rep, True)
    # rateOf accessors (GH #106): rate_of__<species> → current_derivs[idx], a
    # plain variable read (not a call). The RHS emitter declares and fills
    # current_derivs via the two-pass probe; non-rateOf models pass nothing.
    if rateof_map:
        for name, rep in rateof_map.items():
            lookup[name] = (rep, True)
    return lookup


def _translate_expr_to_c(expr: str, lookup: dict[str, tuple[str, bool]]) -> str:
    """Translate an ExprTk expression to C in a single identifier pass.

    Pre-Issue-#25 implementation ran one ``re.sub`` per name in each map,
    which was O(names × expr_len) per call and dominated large-model
    codegen — ~600 ms / expression × 1000 functions = ~10 min just for
    function-body translation on MODEL1009150002-scale models. The current
    implementation tokenizes once with a single regex and uses dict lookup
    in the replacement callback, dropping that to a few ms per expression.

    ``lookup`` is the prebuilt identifier table from ``_build_ident_lookup_model``
    — shared across all bodies so it is built once, not per call (GH #161;
    rebuilding the ~245k-entry table per body was quadratic at genome scale).
    Its precedence is function > observable > species > parameter > built-in
    (time, _pi, _e, and, or, not, ln, rint, abs).
    """
    # if() must be expanded first so nested ternary structure is correct
    # before identifier rewriting touches anything.
    result = _replace_if_calls(expr)
    # Promote integer literals to floats BEFORE identifier substitution
    # introduces array subscripts (``p[0]``/``y[5]``), so ``1/2`` becomes
    # ``1.0/2.0`` (= 0.5 in C) instead of integer-dividing to 0.
    result = _floatify_int_literals(result)

    def _repl(m: re.Match) -> str:
        name = m.group(1)
        empty_call = m.group(2)
        entry = lookup.get(name)
        if entry is None:
            # Unknown identifier (e.g. math.h funcs like sin, exp, pow,
            # sqrt, sinh, log10) — leave the whole match untouched so the
            # following '(' or arguments survive.
            return m.group(0)
        rep, eats_call = entry
        if empty_call is not None:
            return rep if eats_call else rep + empty_call
        return rep

    result = _IDENT_OR_EMPTY_CALL_RE.sub(_repl, result)

    # Replace ^ with pow() — handle a^b patterns
    result = _replace_power_op(result)
    return result


def _replace_if_calls(expr: str) -> str:
    """Replace ExprTk if(cond, true_val, false_val) with C ternary.

    ExprTk: if(cond, t, f)
    C:      ((cond) ? (t) : (f))

    Handles nested if() calls correctly by matching parentheses.
    """
    result = []
    i = 0
    while i < len(expr):
        # Look for 'if(' pattern
        m = re.match(r"\bif\s*\(", expr[i:])
        if m:
            start = i + m.start()
            # Copy everything before 'if'
            result.append(expr[i:start])
            # Find the matching closing paren
            paren_start = i + m.end() - 1  # position of '('
            args = _split_if_args(expr, paren_start)
            if args and len(args) == 3:
                cond, true_val, false_val = args
                # Recursively process each argument
                cond = _replace_if_calls(cond.strip())
                true_val = _replace_if_calls(true_val.strip())
                false_val = _replace_if_calls(false_val.strip())
                result.append(f"(({cond}) ? ({true_val}) : ({false_val}))")
                # Find end of the if(...) expression
                end = _find_matching_paren(expr, paren_start)
                i = end + 1
            else:
                # Malformed if(), pass through
                result.append(expr[i])
                i += 1
        else:
            result.append(expr[i])
            i += 1
    return "".join(result)


def _split_if_args(expr: str, paren_pos: int) -> list[str] | None:
    """Split if(a, b, c) into [a, b, c] respecting nested parens."""
    depth = 0
    args = []
    current = []
    i = paren_pos + 1  # skip opening '('

    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            if depth == 0:
                args.append("".join(current))
                return args
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    return None  # unmatched


def _find_matching_paren(expr: str, open_pos: int) -> int:
    """Find the position of the matching closing paren."""
    depth = 0
    i = open_pos
    while i < len(expr):
        if expr[i] == "(":
            depth += 1
        elif expr[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(expr) - 1


def _replace_power_op(expr: str) -> str:
    """Replace a^b with pow(a, b) in a C expression.

    Handles:
    - simple: x^2 → pow(x, 2)
    - variable: x^y → pow(x, y)
    - parenthesized: (a+b)^2 → pow((a+b), 2)
    - array ref: p[0]^2 → pow(p[0], 2)
    - nested: a^(b^c) and (a^b)^c — both inner powers are translated
    """
    if "^" not in expr:
        return expr

    result: list[str] = []
    i = 0
    chars = expr

    while i < len(chars):
        if chars[i] == "^":
            # Find the base (everything to the left that's part of this operand).
            # The base was emitted left-to-right, so any ^ inside it has
            # already been rewritten — no recursion needed there.
            base = _extract_base_left(result)
            # Find the exponent on the right. The exponent slice is copied
            # from the source verbatim, so a nested ^ inside (e.g. the
            # ((10^hn1)+1) appearing inside a^((10^hn1)+1)) would survive
            # untouched. Recurse to translate any such inner powers.
            exp_str, end = _extract_exp_right(chars, i + 1)
            exp_str = _replace_power_op(exp_str)
            result.append(f"pow({base}, {exp_str})")
            i = end
        else:
            result.append(chars[i])
            i += 1

    return "".join(result)


def _extract_base_left(result_chars: list[str]) -> str:
    """Extract the base operand from the left side of ^.

    Pops characters from result_chars and returns the base string.
    """
    # Skip (and discard) any whitespace emitted between the base and ^.
    # Spaced operators like ``x ^ 2`` or ``(a+b) ^ c`` are natural in
    # hand-authored rate laws; without this the trailing space would be the
    # last char and neither branch below would match → empty base → pow(, e).
    while result_chars and result_chars[-1].isspace():
        result_chars.pop()

    if not result_chars:
        return "0"

    collected = []
    # Check if the last char is ')' or ']' — find matching open
    last = result_chars[-1]

    if last == ")" or last == "]":
        close_ch = last
        open_ch = "(" if close_ch == ")" else "["
        depth = 0
        while result_chars:
            ch = result_chars.pop()
            collected.append(ch)
            if ch == close_ch:
                depth += 1
            elif ch == open_ch:
                depth -= 1
                if depth == 0:
                    break
        # Also collect the function/array name before the paren
        while result_chars and (result_chars[-1].isalnum() or result_chars[-1] in "_"):
            collected.append(result_chars.pop())
    else:
        # Collect identifier or number
        while result_chars and (result_chars[-1].isalnum() or result_chars[-1] in "_."):
            collected.append(result_chars.pop())

    collected.reverse()
    return "".join(collected)


def _extract_exp_right(expr: str, start: int) -> tuple[str, int]:
    """Extract the exponent operand from the right side of ^.

    Returns (exponent_string, end_position).
    """
    i = start
    # Skip whitespace
    while i < len(expr) and expr[i].isspace():
        i += 1

    if i >= len(expr):
        return "0", i

    # Check for '(' — find matching close
    if expr[i] == "(":
        end = _find_matching_paren(expr, i)
        return expr[i : end + 1], end + 1
    # Check for unary minus
    if expr[i] == "-":
        i += 1
        start_num = i
        while i < len(expr) and (expr[i].isalnum() or expr[i] in "_.[]"):
            i += 1
        return f"-{expr[start_num:i]}", i
    # Collect identifier or number
    start_tok = i
    while i < len(expr) and (expr[i].isalnum() or expr[i] in "_.[]"):
        i += 1
    return expr[start_tok:i], i


def _amount_volume_factors(species: list[dict]) -> tuple[dict[int, float], dict[int, int]]:
    """The GH #75 per-species amount factor, as ``(av_factor, av_param)``.

    An ``amount_valued`` (SBML ``hasOnlySubstanceUnits``) species participates by
    its *amount* (stored × V_c), so every emitter scales it by its compartment
    volume: the Elementary rate's ``∏ V_c^mult`` amount factor, each observable
    weight, and the ``∂/∂x`` chain factor. ``av_factor[i]`` is that V_c.

    ``av_param[i]`` (issue #170 stage 2) is the 0-based index of the
    compartment-size *parameter* V_c IS, when the loader bound one. The emitters
    read ``p[k]`` there instead of baking the load-time number, which is what lets
    a ``set_param`` on the volume reach an already-generated source — otherwise the
    write is honoured by the interpreted engine (which re-derives
    ``Species::volume_factor``) and dropped by the compiled one, the invisible
    path-dependence #164 refused over. A species with no live parameter (``.net``,
    a promoted or assignment-rule compartment) is absent here and keeps the literal.

    A species is listed at all only when it carries a factor that can matter:
    ``V_c != 1`` — the literal case, exactly the pre-#170 membership rule, so a
    model with no live volume emits byte-identical text — *or* a live parameter,
    where a V_c of 1.0 must still be emitted as ``p[k]`` because the write that
    moves it off 1.0 comes later. ``× p[k]`` at the load-time value *is* ``× V_c``,
    so the arithmetic at the nominal point is unchanged either way.

    One builder for all seven emitters (the RHS, the two Jacobian paths, the
    sensitivity RHS and its value lines, the output evaluator and the output
    sensitivities) — the same reason :func:`_psvs_row_divisor` is one lookup.
    """
    av_factor: dict[int, float] = {}
    av_param: dict[int, int] = {}
    for i, s in enumerate(species):
        if not s.get("amount_valued", False):
            continue
        vf = float(s.get("volume_factor", 1.0))
        k = int(s.get("volume_param_idx0", -1))
        if vf == 1.0 and k < 0:
            continue
        av_factor[i] = vf
        if k >= 0:
            av_param[i] = k
    return av_factor, av_param


def _av_c(i: int, av_factor: dict, av_param: dict, fmt=repr) -> str:
    """Species ``i``'s amount factor as C — the live ``p[k]`` (issue #170 stage 2)
    or the load-time literal, formatted by ``fmt`` to match its call site's style."""
    k = av_param.get(i, -1)
    return f"p[{k}]" if k >= 0 else fmt(av_factor[i])


def _amount_factor_c(terms: list[tuple[int, float]], fmt=repr) -> str | None:
    """A reaction's ``∏ V_c^mult`` over its amount-valued reactants, as one C
    factor — or ``None`` when there is nothing to emit.

    ``terms`` is ordered, one entry per amount-valued reactant *occurrence*
    (multiplicity included), as ``(volume_param_idx0, volume_factor)``; a negative
    index means the load-time literal.

    The pre-#170 emission folded the whole product in Python and emitted the single
    resulting literal. A live compartment size cannot be folded, so the factors go
    out as **one parenthesised product in the same order** — C's left-associative
    ``*`` then evaluates exactly the sequence the Python fold did, which is what
    keeps the rate bit-identical at the load-time volume. (Emitting them as loose
    factors of the surrounding rate product would re-associate the multiply and
    move the last digit.) With no live factor the folded literal is emitted, so
    every model without a writable compartment size keeps its text.
    """
    if any(k >= 0 for k, _v in terms):
        parts = [f"p[{k}]" if k >= 0 else fmt(v) for k, v in terms]
        return parts[0] if len(parts) == 1 else "(" + " * ".join(parts) + ")"
    fold = 1.0
    for _k, v in terms:
        fold *= v
    return fmt(fold) if fold != 1.0 else None


def _psvs_row_divisor(species: list[dict], si: int) -> tuple[int, float, int]:
    """The compartment-volume divisor of one cross-compartment accumulation row,
    as ``(live_volume_idx0, static_divisor, static_divisor_param_idx0)``.

    A ``per_species_volume_scaling`` reaction's kinetic law evaluates to
    amount/time while every species it touches stores amount/V_c with a V_c of
    its own, so each accumulation row divides by *that* species's compartment
    volume. ``live_volume_idx0 >= 0`` means the compartment was promoted to an
    ODE state (a variable volume, GH #171): the divisor is the live ``y[live]``,
    falling back to the static value when it is not positive. Otherwise the
    divisor is the static ``volume_factor`` — read from ``p[static_divisor_param_idx0]``
    when the loader bound the compartment size to a writable parameter (issue #170
    stage 2), else the load-time literal. The two are mutually exclusive: a
    promoted compartment is integrator state, not a parameter. A row divides by
    1.0 — an ordinary same-volume row — when it has neither and ``volume_factor``
    is 1.0.

    One lookup for every emitter that scatters such a row: the RHS
    (:func:`generate_rhs_from_model`, which *defines* the divide), the Jacobian /
    ``J·v`` reconstruction (:func:`_functional_jacobian_groups`) and the
    sensitivity ``∂f/∂p`` (:func:`generate_sens_from_model`). Deriving it twice is
    how a row the RHS divides ends up with a derivative that does not — the defect
    family GH #119 is about, one scale factor wide and invisible at V = 1. Mirrors
    ``compute_derivs_core``'s ``species_divisor`` and
    ``AffectedRow::static_divisor_param_idx0`` in ``src/model.cpp``.
    """
    sp = species[si]
    return (
        int(sp.get("ode_live_volume_idx0", -1)),
        float(sp.get("volume_factor", 1.0) or 1.0),
        int(sp.get("volume_param_idx0", -1)),
    )


def generate_rhs_from_model(model) -> str:
    """Generate a C source file implementing the CVODE RHS from a built model.

    Works with ANY model (SBML, Antimony, .net) by extracting data from
    the already-built C++ NetworkModel via the codegen_data() binding.
    All reaction types are supported: Elementary, Functional, MichaelisMenten.

    Parameters
    ----------
    model : Model or NetworkModel
        A built BNGsim model (from any input format).

    Returns
    -------
    str
        Complete C source code for the CVODE RHS callback.
    """
    # Get the core C++ model
    core = model._core if hasattr(model, "_core") else model
    data = core.codegen_data()

    params = data["parameters"]
    species = data["species"]
    observables = data["observables"]
    functions = data["functions"]
    reactions = data["reactions"]
    # tfun specs are present iff the binding exposes them; older builds
    # without the model-based tfun support omit the key entirely.
    tfun_specs = data.get("table_functions", [])

    n_sp = len(species)
    n_params = len(params)

    # GH #75: per-species amount factor (volume_factor for amount_valued
    # species, else 1.0). An amount_valued species participates by its amount
    # (stored × V_c): observables read it as an amount (mirrors
    # NetworkModel::update_observables) and an Elementary rate carries the
    # constant ∏_{amount_valued reactants} V_c^mult (mirrors
    # compute_species_factor / AnalyticalJacobianData::amount_factor). Empty for
    # .net / V=1 / hOSU=false ⇒ byte-identical codegen output. ``av_param`` (issue
    # #170 stage 2) names the compartment-size parameter each factor IS, so the
    # emitted C reads it live instead of baking this load's volume.
    av_factor, av_param = _amount_volume_factors(species)
    n_obs = len(observables)
    n_func = len(functions)

    # Build name→index maps
    param_names = [p["name"] for p in params]
    species_names = [s["name"] for s in species]
    obs_names = [o["name"] for o in observables]
    func_names = [f["name"] for f in functions]
    # Reused per-name → C-reference dicts, hoisted out of every _expr_to_c
    # call to avoid rebuilding the same maps thousands of times on large
    # SBML models. Issue #25.
    _param_map = {name: f"p[{i}]" for i, name in enumerate(param_names)}
    _species_map = {name: f"y[{i}]" for i, name in enumerate(species_names)}
    _obs_map = {name: f"obs[{i}]" for i, name in enumerate(obs_names)}
    _func_map = {name: f"func[{i}]" for i, name in enumerate(func_names)}
    # Name → reaction-rate function index (used per reaction loop below).
    _func_idx_by_name: dict[str, int] = {name: i for i, name in enumerate(func_names)}

    # GH #106: rateOf support. A function body that references a
    # rate_of__<species> token needs the live dx/dt — emit the two-pass probe
    # and a current_derivs[] buffer, and resolve each token to current_derivs[i].
    # rateOf reaches the RHS only via functions (rate-rule / assignment-rule
    # bodies); event triggers are not in the codegen RHS. Empty map ⇒ byte-
    # identical to pre-#106 for every non-rateOf model.
    # GH #231 (sub-cluster 3): a hasOnlySubstanceUnits=true species in a
    # constant-volume compartment reports rateOf as the amount-rate
    # volume_factor·d(conc)/dt (current_derivs holds d(conc)/dt) — mirror the
    # engine's refresh_rateof_derivs scaling so codegen and interpreted agree.
    # Flagged set is empty for .net / V=1 / hOSU=false ⇒ byte-identical.
    uses_rateof = any(_RATEOF_PREFIX in f["expression"] for f in functions)
    _rateof_map = None
    if uses_rateof:
        _rateof_map = {}
        for i, name in enumerate(species_names):
            ref = f"current_derivs[{i}]"
            if species[i].get("report_rateof_amount", False):
                # (#170 stage 2) …read from p[k] when the compartment size is a
                # writable parameter, so a volume write moves the reported
                # amount-rate on the compiled path too.
                _kvol = int(species[i].get("volume_param_idx0", -1))
                _vscale = (
                    f"p[{_kvol}]"
                    if _kvol >= 0
                    else _jac_c_float(species[i].get("volume_factor", 1.0))
                )
                ref = f"({_vscale} * {ref})"
            _rateof_map[f"{_RATEOF_PREFIX}{name}"] = ref

    # Fixed species (0-based)
    fixed_sp = {i for i, s in enumerate(species) if s["fixed"]}

    # Map each tfun-backed BNGL function name to its (tf_id, C index expr).
    # Index expr matches the locals emitted below: bare ``t`` for time,
    # ``p[idx]`` for parameter-indexed, ``obs[idx]`` for observable-indexed.
    tfun_call_by_name: dict[str, tuple[int, str]] = {}
    for tf_id, spec in enumerate(tfun_specs):
        kind = spec["index_kind"]
        if kind == "time":
            idx_c = "t"
        elif kind == "parameter":
            idx_c = f"p[{spec['index_param_idx']}]"
        elif kind == "observable":
            idx_c = f"obs[{spec['index_obs_idx']}]"
        else:
            # Unrecognised index kind — let _expr_to_c fall through and
            # surface a compile error with the original expression text.
            continue
        tfun_call_by_name[spec["name"]] = (tf_id, idx_c)

    # ── Build per-reaction rate + scatter lines (one group per reaction) ────
    # Done before emission so the Tier-1 chunking decision can place the
    # NOINLINE block definitions at file scope, ahead of bngsim_codegen_rhs.
    # Below the threshold the groups are spliced inline → byte-identical to the
    # pre-chunking output. The reaction-rate / stoichiometry logic is unchanged;
    # only its destination (a group list vs. lines) differs.
    chunk = _should_chunk(len(reactions))
    block_size = _chunk_block_size()

    # Per-species 1.0/volume_factor table for cross-compartment unified emission
    # (per_species_volume_scaling=true). Built whenever any reaction is varvol so
    # the static-volume rows can reference it; whether the table is actually
    # EMITTED is decided after the reaction loop from real usage (GH #171: an
    # all-live reaction never reads it). Empty for V=1/.net ⇒ byte-identical.
    # The table is built AFTER the reaction loop, from the slots the emitted lines
    # actually read — see below.
    inv_vf_terms: list[str] = []

    rxn_groups: list[list[str]] = []
    for _rxn_i, rxn in enumerate(reactions):
        grp: list[str] = []
        g = grp.append
        rtype = rxn["type"]
        fname = rxn["function_name"]
        sf = rxn["stat_factor"]
        reactants = list(rxn["reactants"])  # 0-based species indices
        products = list(rxn["products"])  # 0-based species indices
        rate_params = list(rxn["rate_param_indices"])  # 0-based param indices
        # apply_species_factor is the BNGL-convention default for .net-loaded
        # reactions; SBML's unified emission marks it false because BNG's
        # writeSBML emits the kinetic-law function with the reactant factor
        # baked in (e.g. ``_rateLaw * S``). Older codegen_data() bindings
        # without the field default to true, preserving .net behavior.
        asf = bool(rxn.get("apply_species_factor", True))
        # per_species_volume_scaling: SBML cross-compartment unified emission.
        # When true, the per-species accumulator divides by each species's
        # volume_factor at ydot accumulation time (the kinetic-law function
        # evaluates to amount/time but storage is amount/V_c, and V_c can
        # differ across involved species). Defaults to false, preserving the
        # .net and uniform-V_s SBML behavior.
        psvs = bool(rxn.get("per_species_volume_scaling", False))

        # GH #75: amount_valued reactants enter the species factor by their
        # amount (stored × V_c), so the rate carries the constant ∏ V_c^mult.
        # Mirrors compute_species_factor_ode. None ⇒ no term emitted
        # (byte-identical for .net / V=1 / hOSU=false). Applies to the
        # species-factor product in both the elementary and the
        # apply_species_factor functional branches below. Issue #170 stage 2: a
        # live compartment size goes out as p[k] instead of folded — see
        # _amount_factor_c for why the product has to be parenthesised.
        amount_factor_c = _amount_factor_c(
            [(av_param.get(ri, -1), av_factor[ri]) for ri in reactants if ri in av_factor]
        )

        if rtype == "functional":
            # Rate = func[func_idx] * stat_factor [* ∏ y[reactants]].
            # Mirrors compute_rxn_rate() in src/model.cpp: the species
            # factor multiplication is gated on apply_species_factor.
            fidx = _func_idx_by_name.get(fname, -1)
            if fidx >= 0:
                parts: list[str] = []
                if sf != 1.0:
                    parts.append(str(int(sf)) if sf == int(sf) else str(sf))
                parts.append(f"func[{fidx}]")
                if asf:
                    if amount_factor_c is not None:
                        parts.append(amount_factor_c)
                    for ri in reactants:
                        parts.append(f"y[{ri}]")
                g(f"    rate = {' * '.join(parts)};  /* {fname} */")
            else:
                g(f"    rate = 0.0;  /* UNKNOWN func {fname} */")

        elif rtype == "elementary":
            # Rate = p[k_idx] * stat_factor * ∏ y[ri]
            parts = []
            if rate_params:
                parts.append(f"p[{rate_params[0]}]")
            else:
                parts.append("0.0")
            if sf != 1.0:
                if sf == int(sf):
                    parts.insert(0, str(int(sf)))
                else:
                    parts.insert(0, str(sf))
            if amount_factor_c is not None:
                parts.append(amount_factor_c)
            for ri in reactants:
                parts.append(f"y[{ri}]")
            g(f"    rate = {' * '.join(parts)};")

        elif rtype == "mm":
            # tQSSA Michaelis-Menten — the free substrate through the stable
            # quadratic root (GH #89), shared with generate_rhs_c.
            if len(rate_params) >= 2 and len(reactants) >= 2:
                for ln in _mm_rate_lines(
                    f"p[{rate_params[0]}]",
                    f"p[{rate_params[1]}]",
                    sf,
                    reactants[0],
                    reactants[1],
                ):
                    g(ln)
            else:
                g("    rate = 0.0;  /* malformed MM */")
        else:
            g(f"    rate = 0.0;  /* unknown type: {rtype} */")

        # Accumulate stoichiometry: subtract from reactants, add to products.
        # For SBML Functional, reactants is empty (stoichiometry is encoded as
        # separate per-species reactions with stat_factor = net coefficient) and
        # rate already includes the stat_factor. per_species_volume_scaling=true
        # divides by each affected species's volume_factor (cross-compartment).
        if psvs:
            # GH #171: a live-volume species (ode_live_volume_idx0 >= 0) divides by
            # the LIVE volume conc[L] (falling back to the static volume_factor when
            # conc[L] <= 0), mirroring compute_derivs_core's species_divisor. A
            # static-volume row keeps rate * inv_vf (byte-identical to pre-#171).
            # This is the divide _psvs_row_divisor names for the derivative
            # emitters; the reciprocal-table form here is what defines it.
            #
            # Issue #170 stage 2: when the static volume is a writable parameter the
            # row reads it live — as ``rate * (1.0 / p[k])``, NOT ``rate / p[k]``.
            # The table this replaces holds 1/V_c, so the row was one multiply by a
            # reciprocal; x*(1/V) and x/V differ in the last digit, and only the
            # first reproduces the pre-#170 value at the load-time volume. The
            # reciprocal is recomputed per row rather than hoisted into a mutable
            # table, so the chunked blocks' signatures (which already carry p) and
            # the `static const` table for every non-live row are untouched.
            def _psvs_divide(si: int) -> str:
                L, vf, kvol = _psvs_row_divisor(species, si)
                if L >= 0:
                    return f"rate / (y[{L}] > 0.0 ? y[{L}] : {vf!r})"
                if kvol >= 0:
                    return f"rate * (1.0 / p[{kvol}])"
                return f"rate * inv_vf[{si}]"

            for ri in reactants:
                if ri >= 0:
                    g(f"    ydot[{ri}] -= {_psvs_divide(ri)};")
            for pi in products:
                if pi >= 0:
                    g(f"    ydot[{pi}] += {_psvs_divide(pi)};")
        else:
            for ri in reactants:
                if ri >= 0:
                    g(f"    ydot[{ri}] -= rate;")
            for pi in products:
                if pi >= 0:
                    g(f"    ydot[{pi}] += rate;")
        g("")
        rxn_groups.append(grp)

    # Emit the inv_vf table only if a scatter line actually reads it (GH #171): an
    # all-live cross-compartment reaction (e.g. _C4_BOTH_RR, every affected row in
    # a variable-volume compartment) divides by conc[L] and never touches inv_vf,
    # so an unconditionally-emitted table would be an unused static (a -Werror
    # build failure). Detected from the emitted lines like rxn_needs_func below.
    #
    # The table is N_SPECIES long and indexed by species, so it has a slot for rows
    # that do NOT read it — and those slots are built from the SAME emitted text that
    # decides whether the table exists, rather than from a second rule about which
    # rows are live. That matters for issue #170 stage 2: a row whose compartment size
    # is a writable parameter reads ``1.0 / p[k]``, and leaving its load-time
    # reciprocal in the dead slot left the last volume literal in this source — so a
    # `set_param` on the volume moved the emitted TEXT, changing the `.so` cache key
    # and silently recompiling instead of being honoured by the binary the model was
    # loaded with (5 corpus models). A dead slot is 1.0, which no arithmetic reads;
    # deriving "dead" from the text is what keeps that claim true by construction
    # instead of by a rule two sites have to agree on.
    _inv_vf_read = {
        int(m.group(1))
        for grp in rxn_groups
        for ln in grp
        for m in re.finditer(r"inv_vf\[(\d+)\]", ln)
    }
    needs_inv_vf = bool(_inv_vf_read)
    if needs_inv_vf:
        inv_vf_terms = [
            repr(1.0 / (s.get("volume_factor", 1.0) or 1.0)) if i in _inv_vf_read else "1.0"
            for i, s in enumerate(species)
        ]

    # func[] is needed by a block only if some reaction in it reads func[idx]
    # (a Functional rate). Detected from the emitted lines so the block signature
    # matches exactly what the bodies reference.
    rxn_needs_func = any("func[" in ln for grp in rxn_groups for ln in grp)
    if rxn_needs_func:
        _rxn_sig = "const double* y, const double* p, const double* func, double* ydot"
        _rxn_args = "y, p, func, ydot"
    else:
        _rxn_sig = "const double* y, const double* p, double* ydot"
        _rxn_args = "y, p, ydot"
    rxn_block_defs: list[str] = []
    rxn_call_lines: list[str] = []
    rxn_block_protos: list[str] = []
    if chunk:
        rxn_block_defs, rxn_call_lines, rxn_block_protos = _emit_chunked_blocks(
            rxn_groups,
            fn_prefix="rxn_blk",
            signature_params=_rxn_sig,
            call_args=_rxn_args,
            block_size=block_size,
            preamble=("double rate;",),
        )

    # Shard the obs[]/func[] computation off the serial driver too (GH #165) —
    # except for rateOf models, whose func bodies read the function-local
    # current_derivs[] buffer from the two-pass probe, which a file-scope block
    # cannot see (rateOf models are small, so the flat inline emit is fine).
    chunk_obsfunc = chunk and not uses_rateof
    _rhs_obs_lines = (
        _emit_observable_lines(observables, av_factor, av_param) if observables else []
    )
    _obs_sig, _obs_args = _obs_blk_sig(_rhs_obs_lines)
    rhs_obs_in, rhs_obs_fs = _shard_value_lines(
        _rhs_obs_lines,
        chunk=chunk_obsfunc,
        fn_prefix="rhs_obs_blk",
        signature_params=_obs_sig,
        call_args=_obs_args,
    )
    rhs_func_in, rhs_func_fs = _shard_value_lines(
        _emit_function_lines(
            functions,
            tfun_call_by_name,
            _param_map,
            _species_map,
            _obs_map,
            _func_map,
            _rateof_map,
        )
        if functions
        else [],
        chunk=chunk_obsfunc,
        fn_prefix="rhs_func_blk",
        signature_params=_FUNC_BLK_SIG,
        call_args=_FUNC_BLK_ARGS,
        preamble=_FUNC_BLK_PREAMBLE,
    )

    lines: list[str] = []
    _emit = lines.append

    # ── Header ────────────────────────────────────────────────────────
    _emit("/* Auto-generated by bngsim._codegen - DO NOT EDIT */")
    if chunk:
        _emit(_CHUNK_MARKER)
    _emit("/* Model-based codegen for SBML/Antimony/any input format */")
    _emit("")
    _emit("#include <math.h>")
    _emit("#include <stdlib.h>")
    _emit("#include <string.h>")
    _emit("")
    _emit("#ifndef M_PI")
    _emit("#define M_PI 3.14159265358979323846")
    _emit("#endif")
    _emit("#ifndef M_E")
    _emit("#define M_E 2.71828182845904523536")
    _emit("#endif")
    _emit("")
    # Layout MUST match CodegenUserDataForSO in cvode_simulator.cpp —
    # tfun_eval is invoked from this RHS for any tfun-backed function.
    _emit("typedef double (*TfunEvalFn)(int tf_id, double x, void* ctx);")
    _emit("typedef struct {")
    _emit("    double* param_values;")
    _emit("    void* tfun_ctx;")
    _emit("    TfunEvalFn tfun_eval;")
    _emit("} CodegenUserData;")
    _emit("")
    _emit(f"#define N_SPECIES {n_sp}")
    _emit(f"#define N_PARAMS  {n_params}")
    _emit(f"#define N_OBS     {n_obs}")
    _emit(f"#define N_FUNC    {n_func}")
    _emit("")

    # ── Tier-1 chunking: NOINLINE reaction blocks at file scope ───────────
    # Prototypes precede the definitions so the driver TU can call the blocks
    # after compile_rhs lifts their bodies into separate units (GH #160). The
    # blocks read inv_vf at file scope (vs. the local static in the flat path),
    # so hoist its table here when needed — it lands in the source prefix the
    # shard splitter prepends to every unit, so each unit still sees it.
    # Emit the prelude unconditionally so BNGSIM_EXPORT is defined for every
    # model (lanl/bngsim #5); BNGSIM_NOINLINE stays used only by chunked blocks.
    for ln in _CODEGEN_PRELUDE_LINES:
        _emit(ln)
    _emit("")
    if chunk:
        if needs_inv_vf:
            _emit("/* 1/volume_factor per species (cross-compartment unified emission) */")
            _emit(f"static const double inv_vf[N_SPECIES] = {{ {', '.join(inv_vf_terms)} }};")
            _emit("")
        for ln in (
            *rxn_block_protos,
            "",
            *rxn_block_defs,
            *rhs_obs_fs,
            *rhs_func_fs,
        ):
            _emit(ln)
        _emit("")

    # ── RHS function ──────────────────────────────────────────────────
    _emit(
        "BNGSIM_EXPORT int bngsim_codegen_rhs(double t, double* y, double* ydot, "
        "void* user_data) {"
    )
    _emit("    CodegenUserData* data = (CodegenUserData*)user_data;")
    _emit("    double* p = data->param_values;")
    _emit("")

    if uses_rateof:
        # GH #106: live instantaneous dx/dt for rate_of__<species> reads. One
        # probe is exact — every rateOf argument is a species whose derivative is
        # independent of the rateOf consumers (no algebraic loop), so pass 0
        # (current_derivs all zero) computes the argument species exactly; we
        # publish ydot→current_derivs and pass 1 recomputes with rateOf live.
        # Mirrors NetworkModel::compute_derivs (src/model.cpp).
        _emit("    /* rateOf (GH #106): live dx/dt buffer + two-pass probe */")
        _emit("    double current_derivs[N_SPECIES];")
        _emit("    memset(current_derivs, 0, N_SPECIES * sizeof(double));")
        _emit("    for (int _rateof_pass = 0; _rateof_pass < 2; ++_rateof_pass) {")
        _emit("")

    # Zero derivatives
    _emit("    memset(ydot, 0, N_SPECIES * sizeof(double));")
    _emit("")

    # Compute observables from species. The emission is factored out so the
    # analytical-Jacobian function (generate_jacobian_from_model) recomputes
    # obs[] / func[] with byte-identical semantics — its Functional derivatives
    # reference these same intermediates.
    if observables:
        _emit("    /* Compute observables */")
        for ln in rhs_obs_in:
            _emit(ln)
        _emit("")

    # Evaluate functions (in dependency order — same as model build order).
    # Tfun-backed functions short-circuit through the runtime callback;
    # their ExprTk expression has been rewritten to ``tfun_<name>()`` by
    # ModelBuilder, which would otherwise leak through _expr_to_c as an
    # undeclared C identifier and break the compile. Chunked: filled by the
    # rhs_func_blk_* shard blocks (non-rateOf only — see chunk_obsfunc).
    if functions:
        _emit("    /* Evaluate functions */")
        for ln in rhs_func_in:
            _emit(ln)
        _emit("")

    # Reactions. Chunked: call the file-scope NOINLINE blocks built above.
    # Flat (below threshold): splice the per-reaction groups inline, declaring
    # `double rate;` once and the inv_vf table as a function-local static — this
    # branch is byte-identical to the pre-chunking output.
    _emit("    /* Reactions */")
    if chunk:
        lines.extend(rxn_call_lines)
        _emit("")
    else:
        _emit("    double rate;")
        if needs_inv_vf:
            _emit("    /* 1/volume_factor per species (cross-compartment unified emission) */")
            _emit(f"    static const double inv_vf[N_SPECIES] = {{ {', '.join(inv_vf_terms)} }};")
        _emit("")
        for grp in rxn_groups:
            lines.extend(grp)

    # Zero derivatives for fixed species
    if fixed_sp:
        _emit("    /* Zero fixed species */")
        for si in sorted(fixed_sp):
            _emit(f"    ydot[{si}] = 0.0;")
        _emit("")

    if uses_rateof:
        # Publish this pass's RHS as the live derivative for the next pass; after
        # pass 1, ydot already holds the correct RHS (the copy is harmless).
        _emit("    memcpy(current_derivs, ydot, N_SPECIES * sizeof(double));")
        _emit("    }  /* _rateof_pass */")
        _emit("")

    _emit("    return 0;")
    _emit("}")

    return "\n".join(lines) + "\n"


def _c_scalar(x) -> str:
    """An observable coefficient as C, in the style the two observable emitters
    have always used: an integral value without a decimal point, anything else via
    ``str`` (which round-trips a double). Factored out so the folded coefficient and
    issue #170's ``factor*p[k]`` form are formatted identically."""
    xf = float(x)
    if xf == int(xf):
        return str(int(xf))
    return str(xf)


# Signature of the sharded observable-fill block (GH #165). Issue #170 stage 2: an
# amount-valued species whose compartment size is writable reads p[k] in its
# weight, so the block has to be handed the parameter vector; a model with no such
# species keeps the two-parameter form and its emitted text is unchanged.
_OBS_BLK_SIG_P = "const double* y, const double* p, double* obs"
_OBS_BLK_ARGS_P = "y, p, obs"


def _obs_blk_sig(obs_lines: list[str]) -> tuple[str, str]:
    """``(signature, call_args)`` for :func:`_shard_value_lines` over ``obs_lines``,
    read off the emitted text the way the RHS decides ``rxn_needs_func``."""
    if any("p[" in ln for ln in obs_lines):
        return _OBS_BLK_SIG_P, _OBS_BLK_ARGS_P
    return _OBS_BLK_SIG, _OBS_BLK_ARGS


def _emit_observable_lines(observables: list, av_factor: dict, av_param: dict) -> list[str]:
    """C lines computing ``double obs[N]; obs[i] = …`` from species (0-based).

    Shared by the RHS (``generate_rhs_from_model``) and the analytical-Jacobian
    (``generate_jacobian_from_model``) emitters so both reference identical
    observable intermediates — the Functional derivatives the Jacobian emits are
    written in these ``obs[i]`` symbols. GH #75 amount factor is folded into each
    coefficient (1.0 outside the hOSU-V≠1 set ⇒ byte-identical there).

    Issue #170 stage 2: a species whose compartment size is a writable parameter
    keeps the factor and the volume as separate factors, ``factor*p[k]*y[i]``. C's
    left-associative ``*`` evaluates that as ``(factor*p[k])*y[i]`` — the same two
    operations, in the same order, as the folded ``coef*y[i]`` — so the value is
    unchanged at the load-time volume and follows a later write. A line that reads
    ``p[]`` makes the sharded fill block take the parameter vector (see
    :func:`_obs_blk_sig`).
    """
    lines = [f"    double obs[{len(observables)}];"]
    for i, o in enumerate(observables):
        entries = o["entries"]
        if not entries:
            lines.append(f"    obs[{i}] = 0.0;  /* {o['name']} */")
            continue
        terms = []
        for sp_idx, factor in entries:
            if sp_idx in av_param:
                pref = "" if factor == 1.0 else _c_scalar(factor) + "*"
                terms.append(f"{pref}p[{av_param[sp_idx]}]*y[{sp_idx}]")
                continue
            coef = factor * av_factor.get(sp_idx, 1.0)
            if coef == 1.0:
                terms.append(f"y[{sp_idx}]")
            else:
                terms.append(f"{_c_scalar(coef)}*y[{sp_idx}]")
        lines.append(f"    obs[{i}] = {' + '.join(terms)};  /* {o['name']} */")
    return lines


def _topological_function_order(functions: list) -> list[int]:
    """Declaration indices of ``functions`` in dependency (topological) order.

    A C statement ``func[i] = … func[j] …`` must come *after* ``func[j] = …`` or
    it reads an uninitialised slot. The model's function list is in *declaration*
    order, which is not necessarily a dependency order: an SBML assignment rule
    can reference another rule declared after it (``a := b`` before ``b := …``).
    The interpreted engine sidesteps this — ``ModelBuilder`` topologically sorts
    its ``var_param_bindings`` so one ``evaluate_functions`` pass converges — but
    the codegen emitters walked declaration order and emitted a use-before-def
    that silently corrupted the RHS for such models. This mirrors ModelBuilder's
    Kahn sort (src/model_builder.cpp) so the emitted ``func[]`` block is ordered
    the same way.

    Returns the declaration indices in an order where every function follows the
    functions it references. Seeded in ascending index, so a model already in
    dependency order keeps its original order (the emitted C is byte-identical) —
    the entire real corpus, where the loader/BNG already emits rules
    topologically. A residual cycle (malformed input) appends its members in
    declaration order, matching ModelBuilder's fallback.
    """
    nf = len(functions)
    name_to_idx = {f["name"]: i for i, f in enumerate(functions)}
    successors: list[list[int]] = [[] for _ in range(nf)]
    in_degree = [0] * nf
    for i, f in enumerate(functions):
        deps: set[int] = set()
        for tok in re.findall(r"[A-Za-z_]\w*", f["expression"]):
            j = name_to_idx.get(tok)
            if j is not None and j != i:
                deps.add(j)
        for j in deps:
            successors[j].append(i)
            in_degree[i] += 1
    queue = [i for i in range(nf) if in_degree[i] == 0]
    order: list[int] = []
    placed = [False] * nf
    qi = 0
    while qi < len(queue):
        u = queue[qi]
        qi += 1
        order.append(u)
        placed[u] = True
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    for i in range(nf):
        if not placed[i]:
            order.append(i)
    return order


def _emit_function_lines(
    functions: list,
    tfun_call_by_name: dict,
    param_map: dict,
    species_map: dict,
    obs_map: dict,
    func_map: dict,
    rateof_map: dict | None = None,
) -> list[str]:
    """C lines computing ``double func[N]; func[i] = …`` in dependency order.

    Shared by the RHS and analytical-Jacobian emitters (see
    ``_emit_observable_lines``). ``rateof_map`` (GH #106) resolves
    rate_of__<species> tokens to current_derivs[idx]; only the RHS emitter
    passes it (the Jacobian emitter declines for rateOf models).

    Statements are emitted in dependency order (``_topological_function_order``)
    so a function that references another declared after it does not read an
    uninitialised ``func[]`` slot; the ``func[i]`` slot index stays the
    declaration index, so ``func_map`` references are unaffected.
    """
    # Built once and shared across every function body — rebuilding it per body
    # is the model-based GH #161 quadratic (see _build_ident_lookup_model).
    lookup = _build_ident_lookup_model(param_map, species_map, obs_map, func_map, rateof_map)
    lines = [f"    double func[{len(functions)}];"]
    for i in _topological_function_order(functions):
        f = functions[i]
        tfun_call = tfun_call_by_name.get(f["name"])
        if tfun_call is not None:
            tf_id, idx_c = tfun_call
            lines.append(
                f"    func[{i}] = data->tfun_eval({tf_id}, {idx_c}, "
                f"data->tfun_ctx);  /* {f['name']} */"
            )
            continue
        c_expr = _translate_expr_to_c(f["expression"], lookup)
        lines.append(f"    func[{i}] = {c_expr};  /* {f['name']} */")
    return lines


def _jac_c_float(x) -> str:
    """Format a number as a C double literal that always carries a decimal point
    (so a scatter coefficient never participates in C integer division)."""
    xf = float(x)
    if xf == int(xf) and abs(xf) < 1e15:
        return f"{int(xf)}.0"
    return repr(xf)


def _jac_vpow(s: int, av_factor: dict, av_param: dict) -> str:
    """C expression for a reactant's amount value ``y[s]`` (× V_s when the
    species is amount_valued with V≠1, as the live ``p[k]`` when that volume is a
    writable compartment size — issue #170 stage 2)."""
    if s not in av_factor:
        return f"y[{s}]"
    return f"({_av_c(s, av_factor, av_param, _jac_c_float)}*y[{s}])"


def _mm_v_lines(
    *, kcat_c: str, km_c: str, e_idx: int, s_idx: int, stat: float, wrt: str, chain_c: str | None
) -> list[str]:
    """C lines setting ``v`` to ∂(MM rate)/∂kcat or ∂(MM rate)/∂Km (GH #55).

    Written as a braced block with locals rather than one inline expression, and
    grouped exactly as ``_mm_jacobian_groups`` groups ∂rate/∂E and ∂rate/∂S,
    because the grouping is load-bearing:

        ∂rate/∂kcat = stat·E·sFree/(Km + sFree)   ( = rate/kcat)
        ∂rate/∂Km   = −kcat·stat·E·sFree/((Km + sFree)·D)   ( = −rate/D)

    Every algebraically-identical alternative tried so far is worse in float64.
    The "simplified" ∂/∂Km with the ½ factors cancelled loses every significant
    digit on log-spread parameters; the chain rule through
    ``∂sFree/∂Km = ½(−1 + (2S − δ)/D)`` cancels catastrophically in deep
    saturation (1e+10 relative error there, vs machine precision for the form
    above — GH #89). Both are pinned by
    ``test_the_emitted_grouping_is_the_stable_one``.

    ``chain_c`` is the #15/#41 factor ∂(kcat or Km)/∂primary when the rate
    constant is a derived parameter; ``None`` for the direct column.
    """
    tail = f" * ({chain_c})" if chain_c else ""
    stat_c = "" if stat == 1.0 else f"{_jac_c_float(stat)} * "
    out = ["        {"]
    if wrt == "Km":
        out.append(f"            double kcat = {kcat_c};")
    out += _mm_sfree_c_lines(km_c, e_idx, s_idx, "            ")
    out += [
        "            v = 0.0;",
        # The rate itself is 0 exactly where KpsF <= 0 (no enzyme, or Km == 0
        # with S < E), so both partials are genuinely 0 there — the same guard
        # _mm_rate_lines and _mm_jacobian_groups use, meaning the same thing
        # (GH #93). Note this is *not* the old `sFree > 0`: at S = 0 the rate is
        # 0 for every kcat/Km, so these two partials still vanish, but they do so
        # because sFree is 0 in the numerator, not because a guard says so.
        "            if (KpsF > 0.0 && Dmm > 0.0) {",
    ]
    if wrt == "kcat":
        out.append(f"                v = {stat_c}E*sFree/KpsF{tail};")
    else:
        out.append(f"                v = -kcat * {stat_c}E*sFree/(KpsF*Dmm){tail};")
    out += ["            }", "        }"]
    return out


def _mm_dfdp_terms(data, plan_mm, param_idx_by_name, primary_param_names, derived_exprs):
    """``∂(MM rate)/∂p`` for every Michaelis–Menten reaction, as C line-blocks.

    Returns ``({rxn_idx: [(param_idx, v_lines)]}, None)`` or ``({}, reason)`` to
    decline the whole model — ``CVodeSensInit1`` is all-or-nothing, as everywhere
    else on this path.

    The tQSSA rate is closed form, so unlike the Functional path (GH #66) there is
    no sympy here at all: ∂/∂kcat = rate/kcat and ∂/∂Km = −rate/D. Both were
    checked against ``sympy.diff`` symbolically and over random parameter points
    before being written down.

    Cross-checked against ``plan_mm`` entry by entry rather than trusted: the
    plan is what ``_mm_jacobian_groups`` builds ``J·v`` from, so if the two
    disagree about which species is the enzyme, this declines instead of emitting
    a ∂f/∂p that belongs to a different reaction than the ``J·v`` beside it.

    **Accuracy.** ``sFree`` used to be written ``½(δ + D)``, which cancels
    catastrophically once ``|δ| ≫ √(4·Km·S)`` — roughly two digits per decade, in
    the RHS and the analytical Jacobian as much as here, so a model in that regime
    had a wrong trajectory before it had a wrong gradient. GH #89 replaced the
    root and these partials' grouping both; see :func:`_mm_sfree_c_lines` and
    :func:`_mm_v_lines`. These partials remain exactly as accurate as the rate
    they differentiate and as the Jacobian's own ∂rate/∂E.
    """
    mm_rxns = [(i, r) for i, r in enumerate(data["reactions"]) if r["type"] == "mm"]
    if not mm_rxns:
        return {}, None
    if len(mm_rxns) != len(plan_mm):
        return {}, (
            f"the model has {len(mm_rxns)} Michaelis–Menten reaction(s) but the analytical "
            f"Jacobian plan describes {len(plan_mm)}, so ∂f/∂p and J·v would not be built "
            "from the same reactions"
        )

    params = data["parameters"]
    out: dict[int, list[tuple[int, list[str]]]] = {}
    # GH #99: one memo across every reaction's derived rate constants.
    derived_jac_cache: dict[str, tuple[dict[str, str] | None, str | None]] = {}
    for (rxn_idx, rxn), mt in zip(mm_rxns, plan_mm, strict=True):
        label = f"reaction {rxn_idx + 1} (Michaelis–Menten)"
        rate_params = list(rxn["rate_param_indices"])
        reactants = list(rxn["reactants"])
        if len(rate_params) < 2 or len(reactants) < 2:
            return {}, f"{label} does not carry the kcat/Km and enzyme/substrate pair"
        # model.cpp's MM branch: enzyme is reactant_indices[0], substrate [1].
        e_idx, s_idx = int(reactants[0]), int(reactants[1])
        kcat_i, km_i = int(rate_params[0]), int(rate_params[1])
        if (
            int(mt["e_idx"]) != e_idx
            or int(mt["s_idx"]) != s_idx
            or int(mt["kcat_param_idx0"]) != kcat_i
            or int(mt["km_param_idx0"]) != km_i
            or float(mt["stat_factor"]) != float(rxn["stat_factor"])
        ):
            return {}, (
                f"{label} disagrees with the analytical Jacobian plan about its enzyme, "
                "substrate, rate constants or statistical factor"
            )

        stat = float(rxn["stat_factor"])
        kcat_c, km_c = f"p[{kcat_i}]", f"p[{km_i}]"
        terms: list[tuple[int, list[str]]] = []
        for wrt, pidx in (("kcat", kcat_i), ("Km", km_i)):
            terms.append(
                (
                    pidx,
                    _mm_v_lines(
                        kcat_c=kcat_c,
                        km_c=km_c,
                        e_idx=e_idx,
                        s_idx=s_idx,
                        stat=stat,
                        wrt=wrt,
                        chain_c=None,
                    ),
                )
            )
            # #15/#41: a derived kcat/Km is re-derived whenever a primary moves,
            # so ∂f/∂primary carries (∂rate/∂p_d)·(∂p_d/∂primary). Losing that
            # reads downstream as an exact zero, so a failure declines (#56).
            pname = params[pidx]["name"]
            if pname not in derived_exprs:
                continue
            jac, why = _derived_param_jacobian_checked(
                derived_exprs[pname],
                primary_param_names,
                param_idx_by_name,
                derived_exprs=derived_exprs,
                cache=derived_jac_cache,
                name=pname,
            )
            if why is not None:
                return {}, (
                    f"{label}'s derived rate constant {pname} = {derived_exprs[pname]!r} "
                    f"could not be differentiated ({why})"
                )
            for primary_name, dpd_c in (jac or {}).items():
                k = param_idx_by_name.get(primary_name, -1)
                if k < 0:
                    continue
                terms.append(
                    (
                        k,
                        _mm_v_lines(
                            kcat_c=kcat_c,
                            km_c=km_c,
                            e_idx=e_idx,
                            s_idx=s_idx,
                            stat=stat,
                            wrt=wrt,
                            chain_c=dpd_c,
                        ),
                    )
                )
        out[rxn_idx] = terms
    return out, None


def _mm_jacobian_groups(plan_mm, add) -> list[list[str]] | None:
    """Reconstruct every Michaelis–Menten reaction's Jacobian contribution as a
    balanced ``{ … }`` C line-group, scattered through the caller's ``add``.

    The tQSSA rate is ``kcat·stat·E·sFree/(Km + sFree)`` with ``sFree`` the
    positive root of ``x² − δ·x − Km·S = 0``, ``δ = S − Km − E``,
    ``D = √(δ² + 4·Km·S)`` — closed form, so unlike the Functional path there is
    no symbolic differentiation here, just the two partials written out:

        ∂rate/∂E = kcat·stat·sFree/D
        ∂rate/∂S = kcat·stat·E·Km/((Km + sFree)·D)

    Both are single subtraction-free quotients, and that is load-bearing (GH #89).
    The obvious chain rule through ``sFree`` — ``∂sFree/∂E = ½(−1 − δ/D)``,
    ``∂sFree/∂S = ½(1 + (δ + 2·Km)/D)`` — is algebraically identical and cancels
    catastrophically wherever ``δ < 0`` with ``|δ| ≫ √(4·Km·S)``, which is the
    same regime that used to break ``sFree`` itself: measured against mpmath at
    60 digits, that grouping reached a relative error of 1e+10 in ∂rate/∂E on a
    deep-saturation sweep *after* the root was fixed, while these forms stay at
    machine precision. They fall out of differentiating the symmetric form of the
    rate — the tQSSA complex is ``c = ½(A − D)`` with ``A = E + S + Km`` and the
    same ``D = √(A² − 4·E·S)``, so ``∂c/∂E = (S − c)/D`` and ``S − c = sFree``.

    ``add(col, row, value_c, prefix)`` writes one accumulation of ``value_c``
    into entry ``(row, col)`` of ∂f/∂x and returns the C line, or ``None`` if the
    entry has no home (a sparse CSC-pattern miss); this then returns ``None``
    rather than a partial reconstruction. Same contract as
    :func:`_functional_jacobian_groups`, and for the same reason (GH #67):
    ``generate_jacobian_from_model`` passes a dense/CSC element writer and the
    sensitivity RHS passes :func:`_jacv_add`, which fuses the matvec — so
    ``bngsim_jac_vec`` covers Michaelis–Menten with one reconstruction, not two
    that can drift.

    Mirrors ``NetworkModel``'s own MM branch (``src/model.cpp``): the enzyme is
    ``reactant_indices[0]`` and the substrate ``[1]`` — the reverse of the
    obvious reading — and the ``KpsF > 0 && Dmm > 0`` guard is the interpreted
    path's own guard, testing the same denominator on the same side of the same
    inequality (GH #93).

    That guard used to be ``sFree > 0``, which was the *clamp* seen from the
    derivative side, and it made this function contradict the RHS emitted beside
    it: ``_mm_rate_lines`` never clamped, so for ``S < 0`` the artifact returned a
    varying rate while this said the rate was flat. It was also wrong at ``S = 0``
    — a state every zero-IC species starts in — where ``∂rate/∂S`` is
    ``kcat·stat·E/(Km + E)``, not 0, and the quotient below returns exactly that.
    """
    groups: list[list[str]] = []
    for mt in plan_mm:
        e = int(mt["e_idx"])
        s = int(mt["s_idx"])
        grp: list[str] = []
        g = grp.append
        g("    {")
        # Km/E/S are declared by _mm_sfree_c_lines; only kcat is extra here.
        g(f"        double kcat = p[{int(mt['kcat_param_idx0'])}];")
        for ln in _mm_sfree_c_lines(f"p[{int(mt['km_param_idx0'])}]", e, s, "        "):
            g(ln)
        g("        double dE = 0.0, dS = 0.0;")
        g("        if (KpsF > 0.0 && Dmm > 0.0) {")
        g(f"            double Cmm = kcat * {_jac_c_float(mt['stat_factor'])};")
        g("            dE = Cmm*sFree/Dmm;")
        g("            dS = Cmm*E*Km/(KpsF*Dmm);")
        g("        }")
        for col, affected, deriv in ((e, mt["e_affected"], "dE"), (s, mt["s_affected"], "dS")):
            if not affected:
                continue
            g(f"        if ({deriv} != 0.0) {{")
            for row_i, coeff in affected:
                line = add(col, int(row_i), f"{_jac_c_float(coeff)} * {deriv}", "            ")
                if line is None:
                    return None
                g(line)
            g("        }")
        g("    }")
        groups.append(grp)
    return groups


def _functional_jacobian_groups(
    core, data, add, deadline: float | None = None
) -> tuple[list[list[str]], ...] | None:
    """Reconstruct every Functional reaction's Jacobian contribution as balanced
    ``{ … }`` C line-groups, scattered through the caller's ``add``.

    Returns ``(per_species, per_species_volume, per_observable)`` in the order
    ``fill_dense_analytical_jacobian`` accumulates them, or ``None`` to decline —
    an un-emittable derivative, an unresolvable rate-law function, or an ``add``
    that could not place an entry. Never a partial reconstruction.

    ``add(col, row, value_c, prefix)`` writes one accumulation of ``value_c`` into
    entry ``(row, col)`` of ∂f/∂x and returns the C line, or ``None`` if that entry
    has no home (the sparse Jacobian's CSC-pattern miss). Everything above it is
    matrix-shape agnostic, which is the point: ``generate_jacobian_from_model``
    passes a dense/CSC element writer, and the sensitivity RHS (GH #67) passes one
    that fuses the matvec — ``Jv_out[row] += value·v[col]`` — so ``bngsim_jac_vec``
    covers Functional models with no ``n×n`` buffer and no second derivation. The
    two consumers cannot drift because there is one reconstruction.

    Mirrors ``bngsim._jacobian.attach_functional_jacobian``: the per-species chain
    rule and the per-observable product rule follow ``set_functional_jacobian`` /
    ``scatter_functional_observable_terms``, with the derivative math from the
    native saturable C emitters (GH #151).

    ``deadline`` (GH #90) bounds the SymPy fallback of that derivative math. This
    is a *re*-derivation — ``attach_functional_jacobian`` already ran it at load
    under the #95 budget — but a re-derivation that ignores its own clock, and on
    a model whose load-time attach was itself cut off there is no earlier bound to
    inherit. The sensitivity RHS passes its build's deadline; the Jacobian emitter
    passes ``None`` and is unchanged.
    """
    from collections import Counter

    from bngsim._jacobian import (
        _TIME_SYM,
        build_per_species_c,
        differentiate_rate_law_c,
    )

    params = data["parameters"]
    species = data["species"]
    observables = data["observables"]
    functions = data["functions"]
    reactions_cd = data["reactions"]

    func_idx_by_name = {f["name"]: i for i, f in enumerate(functions)}

    # GH #75 amount factor (mirrors generate_rhs_from_model).
    av_factor, av_param = _amount_volume_factors(species)

    # ── Symbol resolver for sympy_to_c ──────────────────────────────────────
    # Map each free-symbol name in a derivative (observable / constant param /
    # time placeholder) to the C intermediate the caller computes. Keyword-named
    # identifiers were aliased by the symbolic core, so key the map by the aliased
    # name. Observables are registered last so they win on a name collision
    # (matching the interpreted ExprTk variable binding).
    #
    # Python keywords only, and that is correct rather than lucky (GH #108): this
    # map IS `sympy_to_c`'s resolve callback, so no symbol name is ever printed
    # and a C reserved word never has to survive the printer. Widening it to
    # `_sympy_symbol_alias_map` here would be inert; see `_alias_keyword_param`.
    def _alias(n: str) -> str:
        return _alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n

    c_ref: dict[str, str] = {}
    for i, p in enumerate(params):
        c_ref[_alias(p["name"])] = f"p[{i}]"
    for i, o in enumerate(observables):
        c_ref[_alias(o["name"])] = f"obs[{i}]"

    def resolve_symbol(name: str):
        if name == _TIME_SYM:
            return "t"
        return c_ref.get(name)

    # ── Functional context (read like attach_functional_jacobian) ───────────
    ctx = core.functional_jacobian_context()
    frxns = ctx.get("functional_reactions") or []
    func_map = dict(ctx["function_map"])
    obs_groups = {name: [(int(si), float(f)) for si, f in grp] for name, grp in ctx["observables"]}
    species_meta = {i: (bool(av), float(vf)) for i, (av, vf) in enumerate(ctx["species_meta"])}
    # (#170 stage 2) Same map attach_functional_jacobian builds, for the same reason:
    # a writable compartment size stays a symbol in the per-species derivative rather
    # than being folded into its coefficient at emit time. Empty ⇒ unchanged text.
    species_volume_sym = {i: n for i, n in enumerate(ctx.get("species_volume_param") or ()) if n}
    constants = set(ctx["constant_names"])
    ctx_obs_names = set(obs_groups)

    def _net_affected(rxn) -> list[tuple[int, float, int, float, int]]:
        """Affected rows as (i, coeff, live_idx, static_divisor, sdiv_param) —
        mirrors set_functional_jacobian (GH #171). coeff = stat_factor·net_stoich is
        UNFOLDED: the volume divide is deferred to the scatter so a variable-volume
        row uses the live volume conc[live_idx]. A static-volume / non-varvol row
        has live_idx = -1 and divides by static_divisor (volume_factor for a
        per_species_volume_scaling row, else 1.0 — folded at emit, byte-identical
        to the pre-#171 output), or by p[sdiv_param] when that volume is a writable
        compartment size (issue #170 stage 2)."""
        stat = float(rxn["stat_factor"])
        psvs = bool(rxn["per_species_volume_scaling"])
        net: dict[int, float] = {}
        for si in rxn["reactant_idx0"]:
            if si >= 0:
                net[si] = net.get(si, 0.0) - 1.0
        for si in rxn["product_idx0"]:
            if si >= 0:
                net[si] = net.get(si, 0.0) + 1.0
        out = []
        for i, c_i in net.items():
            if c_i == 0.0:
                continue
            row = _psvs_row_divisor(species, i) if psvs else (-1, 1.0, -1)
            live_idx, static_divisor, sdiv_param = row
            out.append((i, stat * c_i, live_idx, static_divisor, sdiv_param))
        return out

    per_species_groups: list[list[str]] = []
    per_species_volume_groups: list[list[str]] = []
    per_observable_groups: list[list[str]] = []
    missed = False

    # Scatter one existing-column contribution coeff·dj into (row i, col sp_j),
    # applying the GH #171 volume divide. A static-volume / non-varvol row folds
    # coeff/static_divisor at emit (static_divisor=1.0 for non-varvol ⇒ the
    # unchanged coeff, byte-identical to pre-#171). A live-volume row defers to a
    # runtime divide by conc[live_idx] (fallback static_divisor when ≤0),
    # mirroring fill_*_analytical_jacobian's `a.coeff * dv / divisor`.
    #
    # Issue #170 stage 2: a writable compartment size cannot be folded, so the
    # divide moves to run time as `coeff / p[k] * dj`. That is the same single
    # correctly-rounded division of the same two doubles the Python fold did,
    # followed by the same multiply — so the emitted value is unchanged at the
    # load-time volume, unlike hoisting a reciprocal would be.
    def _scatter_existing(sp_j, i, coeff, live_idx, sdiv, sdiv_param, rhs):
        nonlocal missed
        if live_idx >= 0:
            div = f"(y[{live_idx}] > 0.0 ? y[{live_idx}] : {sdiv!r})"
            line = add(sp_j, i, f"{_jac_c_float(coeff)} * {rhs} / {div}", "        ")
        elif sdiv_param >= 0:
            line = add(sp_j, i, f"{_jac_c_float(coeff)} / p[{sdiv_param}] * {rhs}", "        ")
        else:
            line = add(sp_j, i, f"{_jac_c_float(coeff / sdiv)} * {rhs}", "        ")
        if line is None:
            missed = True
        return line

    # Per-species (SBML) block — emitted for all reactions first so the scatter
    # accumulation order matches fill_dense_analytical_jacobian (all species_
    # terms, then all volume_terms, then all observable_terms).
    for rxn in frxns:
        has_sf = bool(rxn["apply_species_factor"]) and len(rxn["reactant_idx0"]) > 0
        if has_sf:
            continue
        affected = _net_affected(rxn)
        terms = build_per_species_c(
            rxn["rate_expr"],
            func_map,
            obs_groups,
            species_meta,
            constants,
            resolve_symbol,
            deadline,
            species_volume_sym,
        )
        if terms is None:
            return None
        if not affected:
            continue  # set_functional_jacobian drops empty-affected terms
        for sp_j, c_deriv in terms:
            grp = [
                f"    {{ /* per-species rxn {int(rxn['rxn_idx'])} col {sp_j} */",
                f"        double dj = {c_deriv};",
            ]
            for i, coeff, live_idx, sdiv, sdiv_param in affected:
                line = _scatter_existing(sp_j, i, coeff, live_idx, sdiv, sdiv_param, "dj")
                if line is not None:
                    grp.append(line)
            grp.append("    }")
            per_species_groups.append(grp)

    # GH #171: the new ∂/∂V_live column for cross-compartment variable-volume
    # reactions — −(stat·netstoichᵢ·func)/V_live² at (row i, col live_idx). Built
    # once per per_species_volume_scaling reaction from the affected rows directly
    # (independent of the ∂func derivatives: the bare-law k·A·B has no ∂func/∂V_live
    # term, so the whole column is this contribution). Mirrors the volume_terms
    # scatter in fill_*_analytical_jacobian; func is func[fidx] (= the reaction's
    # bound rate parameter). A live (row, col) with no home means the
    # reconstruction disagrees with the caller's matrix shape → decline.
    for rxn in frxns:
        if not bool(rxn["per_species_volume_scaling"]):
            continue
        live_rows = [
            (i, coeff, live_idx)
            for i, coeff, live_idx, _sd, _sdp in _net_affected(rxn)
            if live_idx >= 0
        ]
        if not live_rows:
            continue
        rxn_idx = int(rxn["rxn_idx"])
        fname = reactions_cd[rxn_idx]["function_name"]
        fidx = func_idx_by_name.get(fname, -1)
        if fidx < 0:
            return None
        grp = [
            f"    {{ /* varvol column rxn {rxn_idx} func {fname} */",
            f"        double fv = func[{fidx}];",
        ]
        for i, coeff, live_idx in live_rows:
            line = add(
                live_idx,
                i,
                f"{_jac_c_float(-coeff)} * fv / (y[{live_idx}] * y[{live_idx}])",
                f"        if (y[{live_idx}] > 0.0) ",
            )
            if line is None:
                missed = True
            else:
                grp.append(line)
        grp.append("    }")
        per_species_volume_groups.append(grp)

    # Per-observable (.net) block — rate = func(observables) · ∏R.
    for rxn in frxns:
        has_sf = bool(rxn["apply_species_factor"]) and len(rxn["reactant_idx0"]) > 0
        if not has_sf:
            continue
        affected = _net_affected(rxn)
        rxn_idx = int(rxn["rxn_idx"])
        od = differentiate_rate_law_c(
            rxn["rate_expr"], func_map, ctx_obs_names, constants, resolve_symbol, deadline
        )
        if od is None:
            return None
        fname = reactions_cd[rxn_idx]["function_name"]
        fidx = func_idx_by_name.get(fname, -1)
        if fidx < 0:
            return None
        observable_k = od  # [(obs_name, c_str)]
        d_c = [c for _obs_name, c in od]
        if not affected:
            continue
        rmult = Counter(si for si in rxn["reactant_idx0"] if si >= 0)

        # Columns keyed by species j: term A from observable groups, term B from
        # reactant membership — mirrors set_functional_jacobian's ensure_col.
        # a_terms carries the coefficient as C *text* rather than a number: with a
        # writable compartment size (issue #170 stage 2) the amount factor cannot be
        # folded into it and goes out as `factor*p[k]`, whose left-associative
        # evaluation is the same multiply-then-multiply the fold was. The
        # zero-coefficient skip then keys off the GroupEntry factor, since a live
        # p[k] is not a compile-time zero.
        cols: dict[int, dict] = {}
        for k, (obs_name, _se) in enumerate(observable_k):
            for sp_j, factor in obs_groups[obs_name]:
                if sp_j in av_param:
                    if factor == 0.0:
                        continue
                    g_c = f"{_jac_c_float(factor)}*p[{av_param[sp_j]}]"
                else:
                    gcoef = factor * av_factor.get(sp_j, 1.0)
                    if gcoef == 0.0:
                        continue
                    g_c = _jac_c_float(gcoef)
                cols.setdefault(sp_j, {"a_terms": [], "is_reactant": False, "mult_j": 0})[
                    "a_terms"
                ].append((k, g_c))
        for s, m in rmult.items():
            col = cols.setdefault(s, {"a_terms": [], "is_reactant": False, "mult_j": 0})
            col["is_reactant"] = True
            col["mult_j"] = m

        grp = [
            f"    {{ /* per-observable rxn {rxn_idx} func {fname} */",
            f"        double f = func[{fidx}];",
        ]
        p_parts = [_jac_vpow(s, av_factor, av_param) for s, m in rmult.items() for _ in range(m)]
        grp.append(f"        double P = {' * '.join(p_parts) if p_parts else '1.0'};")
        for k, c in enumerate(d_c):
            grp.append(f"        double d{k} = {c};")
        for sp_j, col in cols.items():
            aj = " + ".join(f"{g}*d{k}" for (k, g) in col["a_terms"])
            grp.append("        {")
            grp.append(f"            double val = ({aj if aj else '0.0'}) * P;")
            if col["is_reactant"]:
                dp_parts = [_jac_c_float(col["mult_j"])]
                for s, m in rmult.items():
                    for _ in range(m - 1 if s == sp_j else m):
                        dp_parts.append(_jac_vpow(s, av_factor, av_param))
                if sp_j in av_factor:
                    dp_parts.append(_av_c(sp_j, av_factor, av_param, _jac_c_float))
                grp.append(f"            val += f * ({' * '.join(dp_parts)});")
            # Per-observable (.net) reactions are never per_species_volume_scaling,
            # so live_idx is always -1 and static_divisor 1.0 — coeff is the
            # unfolded stat·net_stoich (byte-identical to the pre-#171 folded value).
            for i, coeff, _live_idx, _sdiv, _sdp in affected:
                line = add(sp_j, i, f"{_jac_c_float(coeff)} * val", "            ")
                if line is None:
                    missed = True
                else:
                    grp.append(line)
            grp.append("        }")
        grp.append("    }")
        per_observable_groups.append(grp)

    # An entry the caller could not place means the reconstruction and the
    # caller's matrix shape disagree — decline rather than emit a Jacobian
    # missing those entries (the interpreted path stays correct).
    if missed:
        return None
    return per_species_groups, per_species_volume_groups, per_observable_groups


def generate_jacobian_from_model(model) -> str | None:
    """Emit the analytical Jacobian callback — a C mirror of the model's
    ``NetworkModel::fill_*_analytical_jacobian`` (src/model.cpp). GH #76 Task 4,
    GH #162.

    Two forms, selected by how the CVODE solver routes the model (mirrors
    cvode_simulator.cpp ``use_sparse``):

      * **Dense** (``bngsim_codegen_jac``, GH #76) — column-major
        ``jac[j*N_SPECIES + i] = ∂f_i/∂x_j`` (matching ``SUNDenseMatrix_Data``),
        mirroring ``fill_dense_analytical_jacobian``.
      * **Sparse CSC** (``bngsim_codegen_jac_sparse``, GH #162) — fills the
        nnz-length CSC value array ``jac_data[data_idx]``, mirroring
        ``fill_sparse_analytical_jacobian``, for large sparse/KLU models where a
        dense ``n×n`` emit is infeasible (a 75k-species dense Jacobian ≈ 45 GB).

    The emitted function reuses the ``CodegenUserData`` typedef and the
    ``N_SPECIES``/``N_OBS``/``N_FUNC`` macros declared by the RHS source it is
    appended to (``generate_combined_from_model`` always prepends it).

    Returns the C source, or ``None`` to *decline* — the simulator then keeps the
    interpreted analytical / finite-difference Jacobian. Declines when:

      * the interpreted analytical Jacobian is not complete for this model, so a
        compiled Jacobian is never emitted where the interpreted dispatch would
        not use one (this also guarantees the compiled scatter matches the
        FD-self-checked interpreted assembly);
      * any Functional derivative cannot be emitted as C (an un-resolvable symbol
        or an un-representable construct) — never ship a partial/wrong Jacobian;
      * (sparse) a Functional ``(col, row)`` falls outside the CSC pattern, which
        would mean the Python reconstruction and the C++ sparsity disagree.

    Elementary + Michaelis–Menten contributions come from the C++ scatter plan
    (``codegen_jacobian_plan``): dense uses the pre-resolved rows, sparse uses the
    parallel ``affected_csc`` data indices. Functional contributions are
    reconstructed from ``functional_jacobian_context()`` exactly as
    ``bngsim._jacobian.attach_functional_jacobian`` does — the per-species chain
    rule and the per-observable product rule mirror ``set_functional_jacobian`` /
    ``scatter_functional_observable_terms``, with the derivative math emitted via
    the native saturable C emitters (``bngsim._jacobian.build_per_species_c`` /
    ``differentiate_rate_law_c``, GH #151) and scattered by CSC index for sparse.
    """
    import os

    # Escape hatch: force the interpreted analytical / FD Jacobian by declining to
    # emit the compiled one (A/B the feature; mirrors
    # BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0 for the interpreted path).
    if os.environ.get("BNGSIM_NO_CODEGEN_JAC") == "1":
        return None

    core = model._core if hasattr(model, "_core") else model
    plan = core.codegen_jacobian_plan()
    if not plan["available"]:
        return None
    ns = int(plan["n_species"])
    if ns <= 0:
        return None
    # GH #162: emit the CSC *sparse* Jacobian (bngsim_codegen_jac_sparse) for
    # models the CVODE solver routes to the sparse KLU path, and the dense one
    # (bngsim_codegen_jac) otherwise. A dense n×n emit is infeasible at scale (a
    # 75k-species dense Jacobian is ~45 GB), so large sparse models need the
    # nnz-length CSC form. The structural gate mirrors cvode_simulator.cpp
    # `use_sparse` (ns >= SPARSE_THRESHOLD=50, density < SPARSE_DENSITY_MAX=0.10,
    # non-empty pattern, KLU build). The runtime-only factors (force_dense,
    # jacobian="jax") only *relax* sparse routing, and a structurally-sparse model
    # run dense simply finds no bngsim_codegen_jac symbol and falls back to the
    # interpreted dense Jacobian — never a wrong one.
    nnz = int(plan["nnz"])
    is_sparse = bool(plan["has_klu"]) and ns >= 50 and nnz > 0 and float(plan["density"]) < 0.10

    # Scatter target for a single contribution. Dense writes the column-major
    # jac[col*N_SPECIES + row]; sparse writes the CSC value slot jac_data[csc].
    # Elementary/MM carry their CSC indices in the plan (affected_csc), so they use
    # ``_lv`` directly; Functional terms come back from _functional_jacobian_groups,
    # which resolves (col, row) -> csc lazily through ``_jac_add``.
    if is_sparse:
        import numpy as np

        col_ptrs = plan.get("col_ptrs")
        row_indices = plan.get("row_indices")
        if col_ptrs is None or row_indices is None:
            # Stale core without the CSC plan → decline (interpreted sparse Jac).
            return None
        col_ptrs = np.asarray(col_ptrs)
        row_indices = np.asarray(row_indices)

        _col_row_to_csc: dict[int, dict[int, int]] = {}

        def _csc_of(col: int, row: int) -> int:
            """CSC data index of (row, col); -1 on a miss.

            Columns are indexed lazily — only the columns Functional terms touch —
            so a genome-scale matrix never materializes its full nnz map. A miss
            means the Python reconstruction disagrees with the CSC pattern; the
            caller then declines rather than ship a partial/wrong Jacobian.
            """
            m = _col_row_to_csc.get(col)
            if m is None:
                if col < 0 or col + 1 >= len(col_ptrs):
                    return -1
                lo = int(col_ptrs[col])
                hi = int(col_ptrs[col + 1])
                m = {r: lo + off for off, r in enumerate(row_indices[lo:hi].tolist())}
                _col_row_to_csc[col] = m
            csc = m.get(int(row))
            return -1 if csc is None else csc

        def _lv(col: int, row: int, csc) -> str:
            return f"jac_data[{csc}]"

        def _jac_add(col: int, row: int, value_c: str, prefix: str) -> str | None:
            csc = _csc_of(col, row)
            return None if csc < 0 else f"{prefix}jac_data[{csc}] += {value_c};"
    else:

        def _lv(col: int, row: int, csc) -> str:
            return f"jac[{col}*N_SPECIES + {row}]"

        def _jac_add(col: int, row: int, value_c: str, prefix: str) -> str | None:
            return f"{prefix}jac[{col}*N_SPECIES + {row}] += {value_c};"

    data = core.codegen_data()
    params = data["parameters"]
    species = data["species"]
    observables = data["observables"]
    functions = data["functions"]
    reactions_cd = data["reactions"]

    param_names = [p["name"] for p in params]
    species_names = [s["name"] for s in species]
    obs_names = [o["name"] for o in observables]
    func_names = [f["name"] for f in functions]

    # GH #75 amount factor (mirrors generate_rhs_from_model).
    av_factor, av_param = _amount_volume_factors(species)

    # Maps + tfun dispatch for the obs[]/func[] recomputation (same as the RHS).
    _param_map = {name: f"p[{i}]" for i, name in enumerate(param_names)}
    _species_map = {name: f"y[{i}]" for i, name in enumerate(species_names)}
    _obs_map = {name: f"obs[{i}]" for i, name in enumerate(obs_names)}
    _func_map = {name: f"func[{i}]" for i, name in enumerate(func_names)}
    tfun_specs = data.get("table_functions", [])
    tfun_call_by_name: dict[str, tuple[int, str]] = {}
    for tf_id, spec in enumerate(tfun_specs):
        kind = spec["index_kind"]
        if kind == "time":
            idx_c = "t"
        elif kind == "parameter":
            idx_c = f"p[{spec['index_param_idx']}]"
        elif kind == "observable":
            idx_c = f"obs[{spec['index_obs_idx']}]"
        else:
            continue
        tfun_call_by_name[spec["name"]] = (tf_id, idx_c)

    # ── Functional contributions (GH #76/#171) ──────────────────────────────
    # Reconstructed by the shared builder the sensitivity RHS's bngsim_jac_vec
    # also uses (GH #67); the only difference between the two consumers is where
    # a contribution lands, which is what `_jac_add` supplies.
    groups = _functional_jacobian_groups(core, data, _jac_add)
    if groups is None:
        return None
    per_species_groups, per_species_volume_groups, per_observable_groups = groups

    # GH #171: the varvol column groups also read func[] (and the obs[] the func
    # recomputation depends on), so they count toward has_functional / need_func —
    # otherwise func[] would be referenced but never computed for a psvs model
    # whose only Functional terms are per-species (no per-observable block).
    has_functional = bool(per_species_groups or per_species_volume_groups or per_observable_groups)
    need_func = bool(per_observable_groups or per_species_volume_groups)

    # ── Elementary + Michaelis–Menten contribution groups ───────────────────
    # Each reaction's scatter is a balanced ``{ … }`` group, built here (rather
    # than emitted inline) so the same groups feed both the flat splice and the
    # NOINLINE shard wrap below.
    elem_groups: list[list[str]] = []
    for erxn in plan["elementary"]:
        grp: list[str] = []
        g = grp.append
        ksf_parts = [f"p[{int(erxn['rate_param_idx0'])}]"]
        sf = float(erxn["stat_factor"])
        af = float(erxn["amount_factor"])
        if sf != 1.0:
            ksf_parts.append(_jac_c_float(sf))
        # (#170 stage 2) amount_volume_terms is the same product `amount_factor`
        # folded, in the same order, but per-factor — non-empty only when one of the
        # compartment sizes is writable, so every other model still emits the folded
        # literal and its text is unchanged.
        af_c = _amount_factor_c(
            [(int(k), float(v)) for k, v in erxn.get("amount_volume_terms") or []],
            _jac_c_float,
        )
        if af_c is not None:
            ksf_parts.append(af_c)
        elif af != 1.0:
            ksf_parts.append(_jac_c_float(af))
        g(f"    {{ double k_sf = {' * '.join(ksf_parts)};")
        for pr in erxn["reactants"]:
            j = int(pr["species_idx"])
            m_j = int(pr["multiplicity"])
            dv_parts = ["k_sf"]
            if m_j != 1:
                dv_parts.append(str(m_j))
            dv_parts.extend(f"y[{j}]" for _ in range(m_j - 1))
            for oi, om in pr["others"]:
                dv_parts.extend(f"y[{int(oi)}]" for _ in range(int(om)))
            g(f"        {{ double dv = {' * '.join(dv_parts)};")
            affected_csc = pr.get("affected_csc")
            for ai, (row_i, stoich) in enumerate(pr["affected"]):
                csc = int(affected_csc[ai][0]) if is_sparse else None
                g(f"          {_lv(j, int(row_i), csc)} += {_jac_c_float(stoich)} * dv;")
            g("        }")
        g("    }")
        elem_groups.append(grp)

    mm_groups = _mm_jacobian_groups(plan["mm"], _jac_add)
    if mm_groups is None:
        return None

    # ── Tier-1 chunking decision (GH #165) ──────────────────────────────────
    # The whole analytical Jacobian otherwise lands in the single serial *driver*
    # translation unit (it sits outside the RHS NOINLINE blocks), which is the
    # compile wall at genome scale (GH #165). At/above the same reaction-count
    # threshold the RHS chunks at, wrap each contribution group in a NOINLINE
    # shard block so the parallel shard compile (GH #160) splits the Jacobian
    # across cores too. Below the threshold the groups are spliced inline and the
    # emitted Jacobian is byte-identical to the pre-#165 flat one.
    # Order mirrors fill_dense_analytical_jacobian: Elementary, MM, then Functional
    # (species_terms, volume_terms, observable_terms). The varvol column groups sit
    # after per-species so a shared entry (row i, col V_live) — the explicit-factor
    # #172 case, where ∂func/∂V_live and the new column cancel — accumulates in the
    # same order as the interpreted scatter.
    contrib_groups = (
        elem_groups
        + mm_groups
        + per_species_groups
        + per_species_volume_groups
        + per_observable_groups
    )
    chunk = _should_chunk(len(reactions_cd)) and bool(contrib_groups)

    # The shard blocks read y/p (and, for Functional terms, the recomputed
    # obs[]/func[] locals) and write the scatter target (dense jac / sparse
    # jac_data). Detect the obs/func dependency from the emitted bodies —
    # elementary/MM need neither — so the block signature matches exactly what the
    # bodies reference (mirrors the RHS rxn_needs_func detection). A body that
    # reads obs[i]/func[i] guarantees the driver computes that array (it is only
    # emitted for Functional terms, and only then does obs[/func[ appear).
    out_param = "jac_data" if is_sparse else "jac"
    contrib_text = "\n".join("\n".join(grp) for grp in contrib_groups)
    _blk_sig = ["const double* y", "const double* p"]
    _blk_args = ["y", "p"]
    if "obs[" in contrib_text:
        _blk_sig.append("const double* obs")
        _blk_args.append("obs")
    if "func[" in contrib_text:
        _blk_sig.append("const double* func")
        _blk_args.append("func")
    _blk_sig.append(f"double* {out_param}")
    _blk_args.append(out_param)

    jac_block_defs: list[str] = []
    jac_call_lines: list[str] = []
    jac_block_protos: list[str] = []
    if chunk:
        jac_block_defs, jac_call_lines, jac_block_protos = _emit_chunked_blocks(
            contrib_groups,
            fn_prefix=("jac_sparse_blk" if is_sparse else "jac_blk"),
            signature_params=", ".join(_blk_sig),
            call_args=", ".join(_blk_args),
            block_size=_chunk_block_size(),
        )

    # The obs[]/func[] recomputation the Functional derivatives read is itself a
    # large basic block; shard it too (GH #165) so it does not become the driver
    # wall. Flat below the chunk threshold (byte-identical).
    _jac_obs_lines = (
        _emit_observable_lines(observables, av_factor, av_param)
        if (has_functional and observables)
        else []
    )
    _jac_obs_sig, _jac_obs_args = _obs_blk_sig(_jac_obs_lines)
    jac_obs_in, jac_obs_fs = (
        _shard_value_lines(
            _jac_obs_lines,
            chunk=chunk,
            fn_prefix="jac_obs_blk",
            signature_params=_jac_obs_sig,
            call_args=_jac_obs_args,
        )
        if _jac_obs_lines
        else ([], [])
    )
    jac_func_in, jac_func_fs = (
        _shard_value_lines(
            _emit_function_lines(
                functions, tfun_call_by_name, _param_map, _species_map, _obs_map, _func_map
            ),
            chunk=chunk,
            fn_prefix="jac_func_blk",
            signature_params=_FUNC_BLK_SIG,
            call_args=_FUNC_BLK_ARGS,
            preamble=_FUNC_BLK_PREAMBLE,
        )
        if (need_func and functions)
        else ([], [])
    )

    # ── Assemble the function body ──────────────────────────────────────────
    lines: list[str] = []
    _emit = lines.append
    _emit("")

    # Chunked: the NOINLINE contribution blocks live at file scope before the
    # callback, with forward prototypes the driver TU calls. The BNGSIM_NOINLINE
    # macro and the shared typedef / N_* macros come from the RHS source this
    # Jacobian is appended to — chunking is gated on the same reaction count, so a
    # chunked Jacobian always rides a chunked RHS that defined them, and the shard
    # splitter prepends that RHS header to every unit.
    if chunk:
        for ln in (*jac_block_protos, "", *jac_block_defs, *jac_obs_fs, *jac_func_fs):
            _emit(ln)
        _emit("")
    if is_sparse:
        _emit("/* -- Analytical Jacobian, sparse CSC (GH #162) -----------------------")
        _emit("   Fills the nnz-length CSC value array jac_data[data_idx]. C mirror of")
        _emit("   NetworkModel::fill_sparse_analytical_jacobian. Reuses the RHS")
        _emit("   CodegenUserData typedef and the N_SPECIES/N_OBS/N_FUNC macros. */")
        _emit(
            "BNGSIM_EXPORT int bngsim_codegen_jac_sparse(double t, double* y, double* jac_data, "
            "void* user_data) {"
        )
        _emit("    CodegenUserData* data = (CodegenUserData*)user_data;")
        _emit("    double* p = data->param_values;")
        _emit(f"    memset(jac_data, 0, {nnz} * sizeof(double));")
        _emit("    (void)t; (void)p;")
        _emit("")
    else:
        _emit("/* -- Analytical Jacobian (GH #76 Task 4) -----------------------------")
        _emit("   Dense, column-major: jac[j*N_SPECIES + i] = d f_i / d x_j. C mirror of")
        _emit("   NetworkModel::fill_dense_analytical_jacobian. Reuses the RHS")
        _emit("   CodegenUserData typedef and the N_SPECIES/N_OBS/N_FUNC macros. */")
        _emit(
            "BNGSIM_EXPORT int bngsim_codegen_jac(double t, double* y, double* jac, "
            "void* user_data) {"
        )
        _emit("    CodegenUserData* data = (CodegenUserData*)user_data;")
        _emit("    double* p = data->param_values;")
        _emit("    memset(jac, 0, N_SPECIES * N_SPECIES * sizeof(double));")
        _emit("    (void)t; (void)p;")
        _emit("")

    if jac_obs_in:
        _emit("    /* Observables (state-coupled; needed by Functional derivatives) */")
        for ln in jac_obs_in:
            _emit(ln)
        _emit("")
    if jac_func_in:
        _emit("    /* Functions (needed by the per-observable product rule) */")
        for ln in jac_func_in:
            _emit(ln)
        _emit("")

    if chunk:
        # Call the file-scope NOINLINE shard blocks in fill order. The contribution
        # math lives in the blocks (lifted into parallel translation units by
        # compile_rhs); the driver just dispatches.
        _emit("    /* Jacobian contributions (NOINLINE shard blocks, GH #165) */")
        lines.extend(jac_call_lines)
        _emit("")
    else:
        # Flat: splice the contribution groups inline, byte-identical to the
        # pre-#165 Jacobian (every model below the chunk threshold is untouched).
        if elem_groups:
            _emit("    /* Elementary mass-action (closed form) */")
            for grp in elem_groups:
                lines.extend(grp)
            _emit("")
        if mm_groups:
            _emit("    /* Michaelis-Menten (tQSSA) closed form */")
            for grp in mm_groups:
                lines.extend(grp)
            _emit("")
        if per_species_groups:
            _emit("    /* Functional per-species (SBML chain rule) */")
            for grp in per_species_groups:
                lines.extend(grp)
            _emit("")
        if per_species_volume_groups:
            _emit("    /* Functional per-species varvol column (GH #171) */")
            for grp in per_species_volume_groups:
                lines.extend(grp)
            _emit("")
        if per_observable_groups:
            _emit("    /* Functional per-observable (.net product rule) */")
            for grp in per_observable_groups:
                lines.extend(grp)
            _emit("")

    if plan["fixed_rows"]:
        _emit("    /* Fixed (boundary-condition) species rows -> 0 */")
        if is_sparse:
            # Zero every CSC entry whose row is a fixed-species row. row_indices is
            # indexed by data index, so the matching positions ARE the data indices
            # (np.nonzero returns them ascending ⇒ deterministic emit). Mirrors the
            # col-major scan in fill_sparse_analytical_jacobian.
            fixed_set = sorted(int(r) for r in plan["fixed_rows"])
            fixed_arr = np.array(fixed_set, dtype=row_indices.dtype)
            fixed_csc = np.nonzero(np.isin(row_indices, fixed_arr))[0]
            for fixed_i in fixed_csc:
                _emit(f"    jac_data[{int(fixed_i)}] = 0.0;")
        else:
            for row in plan["fixed_rows"]:
                _emit(
                    f"    for (int j = 0; j < N_SPECIES; ++j) jac[j*N_SPECIES + {int(row)}] = 0.0;"
                )
        _emit("")

    # The only return statement this emitter produces, on any branch — and two
    # consumers depend on that. `steady_state.cpp`'s SteadyStateRhs fills are
    # `void` and drop the code, where the `cvode_simulator.cpp` mirrors propagate
    # it; see the return-value section of `include/bngsim/codegen_abi.hpp` for why
    # the split is safe and what has to change first if you add an early
    # `return <nonzero>;` here. `test_emitted_c_has_no_nonzero_return` fails if
    # you do, which is the intended way to find this note.
    _emit("    return 0;")
    _emit("}")
    return "\n".join(lines) + "\n"


# ─── GH #66: analytic ∂f/∂p for Functional rate laws ───────────────────────
#
# A Functional reaction's rate mirrors ``compute_rxn_rate`` (src/model.cpp) and
# the ``rtype == "functional"`` branch of ``generate_rhs_from_model``:
#
#     rate_r = stat_r · func_r(obs, p, t) · [af_r · ∏_j y_j^{m_j}]
#
# with the bracketed species factor present only when ``apply_species_factor``.
# That factor is parameter-free, and an observable is a *fixed* linear
# combination of species (so ∂obs/∂p = 0 at fixed y), which leaves
#
#     ∂f_i/∂p = Σ_r stat_r · netstoich_ir · (∂func_r/∂p) · af_r · ∏R_r
#
# — no product rule and no chain rule through observable groups. That is the very
# shape ``_emit_sens_rhs_body`` already emits for a derived rate constant
# (``(∂p_d/∂primary) × geometry``, #15/#41), so these terms are handed to it
# through the existing ``derived_terms`` channel with ∂func_r/∂p in place of
# ∂p_d/∂primary, and no emitter surgery is needed. What is new is only the
# differentiation below.


# Call heads an inlined rate law may still contain: the ExprTk builtins the
# symbolic core knows how to parse and re-emit. Anything else — a table function
# (``tfun_<table>(…)``), a user function ``_inline_functions`` could not resolve,
# an SBML helper — would parse as an *applied undefined function*, whose
# derivative sympy renders as ``Subs(Derivative(…))``. That is not a symbol the
# free-symbol check below can see, so it is rejected here by name instead, where
# the message can say which call blocked the model. Read from the symbolic core
# rather than re-listed, so the two cannot drift (the #56 lesson).
def _exprtk_call_heads() -> frozenset[str]:
    from bngsim._jacobian import _EXPRTK_TO_SYMPY_FUNC

    return frozenset(_EXPRTK_TO_SYMPY_FUNC)


# ``_pi`` / ``_e`` are ExprTk's math constants, not model parameters (they appear
# in no ``codegen_data()["parameters"]`` entry), so they need their own resolution
# — same pair, same C spellings, as ``differentiate_expression_output_partials``.
# Looked up only *after* the parameter/observable map, so a model that really does
# declare a parameter by one of these names keeps it.
_MATH_CONSTANT_C = {"_pi": "M_PI", "_e": "M_E"}


def _carry_reason_class(inner: str | None, wrapped: str) -> str:
    """Re-tag ``wrapped`` with ``inner``'s reason class, if it has one.

    A decline reason is a plain ``str`` almost everywhere; the one exception,
    :class:`~bngsim._switch_sensitivity.UncompensatedCrossingReason`, is what
    tells :func:`_warn_functional_sens_rhs_refused` that CVODES' difference
    quotient is not a correct fallback either. Every site that adds context to a
    reason has to route through here or it silently downgrades the warning.
    """
    from bngsim._switch_sensitivity import UncompensatedCrossingReason

    if isinstance(inner, UncompensatedCrossingReason):
        return UncompensatedCrossingReason(wrapped)
    return wrapped


def _warn_functional_sens_rhs_refused(reason: str) -> None:
    """Report that the analytic sensitivity RHS was declined over a Functional
    reaction (GH #66, following the #56 precedent).

    ``CVodeSensInit1`` takes ONE sensitivity-RHS callback for every column, so a
    single undifferentiable rate law has to decline the whole model — there is no
    per-reaction fallback to mix in. Saying so out loud is the point: the
    alternative this avoids is emitting ``∂func/∂p = 0`` for that reaction, which
    reads downstream as a converged gradient of exactly zero. Every decline routes
    through here, so none of them can be the quiet one.

    **The fallback is not always correct, and the message must not say it is.**
    For an underivable rate law or an exhausted derivation budget it is: the
    problem is smooth and the difference quotient answers the same question more
    slowly. For an
    :class:`~bngsim._switch_sensitivity.UncompensatedCrossingReason` it is not —
    the thing being reported is a branch crossing nobody compensates, and the
    difference quotient integrates the variational equation straight through it,
    dropping the very jump the analytic path was declined for. On AMICI's
    ``nested_events`` that is the saltation factor ``f⁺/f⁻``, and every parameter
    column comes back a factor of two low after the crossing (issue #146). Until
    issue #150 supplies the jump, this warning is the only thing standing between
    that and a number a caller would take at face value.
    """
    from bngsim._switch_sensitivity import UncompensatedCrossingReason

    if isinstance(reason, UncompensatedCrossingReason):
        logger.warning(
            "Forward sensitivity: %s. The analytic sensitivity RHS is declined for this "
            "model, and CVODES' internal difference quotient — which is used instead — "
            "does NOT recover the missing term: it integrates the variational equation "
            "smoothly through a crossing whose time moves, so wherever that condition "
            "actually crosses during the run, EVERY sensitivity column is wrong there "
            "and after it by the crossing's jump. Validate against a finite difference "
            "of the trajectory before relying on these columns. Tracked in issue #150.",
            reason,
        )
        return
    logger.warning(
        "Forward sensitivity: %s, so the analytic sensitivity RHS is declined for "
        "this model and CVODES' internal difference quotient is used instead "
        "(correct, but slower).",
        reason,
    )


class _FunctionalDfdpScope(NamedTuple):
    """Everything :func:`_functional_rate_law_partials` needs that does not vary
    per reaction — assembled once per model by :func:`_functional_dfdp_terms`.

    ``c_ref`` maps a (keyword-aliased) symbol name to the C intermediate holding
    its value; ``param_of_alias`` is the subset of those symbols that are real
    differentiation variables, i.e. parameters not shadowed by an observable and
    not function-bound.

    ``switch_scope`` is GH #68's gate context: non-``None`` only for a model that
    has a condition to gate *and* whose clock/parameter view could be assembled.
    ``None`` keeps the pre-#68 behaviour — every condition declines the model.

    ``deadline`` is GH #90's build-time derivation bound, shared by every rate law
    of the model (see :func:`_sens_derivation_deadline`); ``None`` is unbounded.

    ``derived_jac_cache`` is GH #99's ``∂p_d/∂primary`` memo, likewise one per
    model, so a derived parameter read by many rate laws is walked once. Pass a
    fresh dict per model — a ``None`` default would be right for a scope built
    per expression, and this one is not.
    """

    func_map: dict[str, str]
    c_ref: dict[str, str]
    param_of_alias: dict[str, str]
    param_idx_by_name: dict[str, int]
    primary_param_names: set[str]
    derived_exprs: dict[str, str]
    switch_scope: SwitchConditionScope | None = None
    deadline: float | None = None
    derived_jac_cache: dict[str, tuple[dict[str, str] | None, str | None]] | None = None
    # (#170 stage 3) ``aliased observable name → {volume_param_idx0: ∂obs/∂V as C}``.
    # Empty unless the model has an amount-valued species whose compartment size
    # is writable — see :func:`_observable_volume_weights`.
    obs_volume_weights: dict[str, dict[int, str]] = {}


def _functional_rate_law_partials(
    rate_expr: str, scope: _FunctionalDfdpScope
) -> tuple[list[tuple[int, str]] | None, str | None]:
    """``∂(rate law)/∂p`` for every parameter it reads, as C source.

    Returns ``([(param_idx, c_expr)], None)`` — possibly empty, which is the
    *success* case for a rate law with no parameter dependence at all — or
    ``(None, reason)`` naming what blocked it. Never returns a partial list: a
    rate law is either fully differentiated or refused (#56). The one exception
    it raises is ``scope.deadline``'s :class:`_DerivationBudgetExceeded` (GH #90),
    which is deliberately not a ``reason``: a reason is memoized per rate-law text
    by the caller, and a wall-clock expiry is a property of the build, not of the
    expression.

    The derivative is taken w.r.t. every parameter symbol surviving inlining,
    *including* a derived (ConstantExpression) one, whose own column is that
    direct partial — mirroring how the Elementary path emits ``case iP`` for a
    ``_rateLaw{N}``. Each derived parameter then contributes
    ``(∂func/∂p_d)·(∂p_d/∂primary)`` to its primaries' columns (#15/#41).
    """
    from bngsim._jacobian import (
        _EMPTY_CALL_RE,
        _IDENT_CALL_RE,
        _TIME_SYM,
        _exprtk_to_sympy,
        _inline_functions,
        sympy_to_c,
        unsupported_expr_construct,
    )

    try:
        import sympy as sp
    except ImportError:
        return None, "sympy is not installed"

    # GH #90: bound the whole build, but check on entry as well as per-parameter —
    # inlining, the construct scan and ``_exprtk_to_sympy`` are themselves
    # unbounded work on a large enough law, so a check only at the ``sp.diff``
    # loop would let a single rate law overshoot arbitrarily.
    _check_derivation_deadline(scope.deadline)

    inlined = _inline_functions(rate_expr, scope.func_map)
    if inlined is None:
        return None, "function references form a cycle or nest deeper than 64 levels"

    # Reject on the *inlined* text so a construct hidden inside a referenced
    # function is caught, not just one written in the rate law itself.
    reason = unsupported_expr_construct(inlined, allow_conditions=scope.switch_scope is not None)
    if reason is not None:
        return None, f"uses unsupported construct: {reason}"

    # GH #68: with the conditional class waived above, the ``if()`` survives to
    # sympy, which differentiates the ``Piecewise`` w.r.t. a condition-only
    # parameter to a clean ``0`` — no Dirac delta, so neither ``_is_emittable``
    # nor anything downstream would notice. That ``0`` is the correct in-branch
    # answer only when issue #48 supplies the crossing jump for the rest of it,
    # which it does for a recognized clock threshold and nothing else. Ask the
    # detector's own recognizer, so the gate cannot admit a condition the
    # detector would not compensate.
    if scope.switch_scope is not None:
        from bngsim._switch_sensitivity import uncompensated_condition_reason

        # Its own name, not the `why` the derived-parameter loop below reuses:
        # this one is an UncompensatedCrossingReason, which is what carries "the
        # difference-quotient fallback is wrong too" to the warning (#146).
        crossing_why = uncompensated_condition_reason(inlined, scope.switch_scope)
        if crossing_why is not None:
            return None, crossing_why

    # Strip the two zero-argument forms ``_preprocess_exprtk`` accepts
    # (``time()``/``t()`` and an observable written as a call, #28) before looking
    # for call heads, so neither is mistaken for an unknown function.
    probe = _EMPTY_CALL_RE.sub(r"\1", re.sub(r"\b(?:time|t)\s*\(\s*\)", " ", inlined))
    known = _exprtk_call_heads()
    if scope.switch_scope is not None:
        # ``if`` is a recognized head, just not through _EXPRTK_TO_SYMPY_FUNC:
        # _exprtk_to_sympy rewrites it to a sympy ``Piecewise`` before parsing
        # (_translate_bngl_if_to_piecewise), so it never reaches a function
        # lookup. Admitted only alongside the gate above, so a model whose
        # conditions were NOT cleared still reports it as unsupported.
        known = known | {"if"}
    stray = sorted({m.group(1) for m in _IDENT_CALL_RE.finditer(probe)} - known)
    if stray:
        return None, "calls unsupported function(s): " + ", ".join(stray)

    sym_expr = _exprtk_to_sympy(inlined)
    if sym_expr is None:
        return None, "could not be parsed for differentiation"

    def resolve_symbol(name: str) -> str | None:
        if name == _TIME_SYM:
            return "t"
        mapped = scope.c_ref.get(name)
        return mapped if mapped is not None else _MATH_CONSTANT_C.get(name)

    allowed = set(scope.c_ref) | {_TIME_SYM} | set(_MATH_CONSTANT_C)
    free = {str(s) for s in sym_expr.free_symbols}
    unknown = free - allowed
    if unknown:
        return None, "references unrecognized symbol(s): " + ", ".join(sorted(unknown))

    terms: list[tuple[int, str]] = []
    for a in sorted(free & set(scope.param_of_alias)):
        pname = scope.param_of_alias[a]
        _check_derivation_deadline(scope.deadline)
        deriv = sp.diff(sym_expr, sp.Symbol(a))
        if deriv == 0:
            continue
        c_expr = sympy_to_c(deriv, resolve_symbol)
        if c_expr is None:
            return None, (
                f"the derivative w.r.t. {pname} is not representable in C "
                "(non-differentiable or unsupported function)"
            )
        terms.append((scope.param_idx_by_name[pname], c_expr))

        # #15/#41: a derived (ConstantExpression) parameter is re-derived from its
        # primaries whenever one of them moves, so ∂f/∂primary carries
        # (∂func/∂p_d)·(∂p_d/∂primary) on top of any direct occurrence. A lost
        # chain rule here reads downstream as an exact zero, so a failure declines
        # the model rather than shipping the truncated gradient (#56).
        if pname not in scope.derived_exprs:
            continue
        jac, why = _derived_param_jacobian_checked(
            scope.derived_exprs[pname],
            scope.primary_param_names,
            scope.param_idx_by_name,
            derived_exprs=scope.derived_exprs,
            deadline=scope.deadline,
            cache=scope.derived_jac_cache,
            name=pname,
        )
        if why is not None:
            return None, (
                f"the derived parameter {pname} = {scope.derived_exprs[pname]!r} it "
                f"reads could not be differentiated ({why})"
            )
        for primary_name, dpd_c in (jac or {}).items():
            k = scope.param_idx_by_name.get(primary_name, -1)
            if k >= 0:
                terms.append((k, f"({c_expr}) * ({dpd_c})"))

    # ── (#170 stage 3) the amount-conversion channel of ∂(rate law)/∂V ──────
    #
    # A compartment size reaches this rate law by a road no symbol of it names.
    # An amount-valued species enters an observable by its AMOUNT, so the emitted
    # weight is the compartment size itself (``obs[k] = Σ factor·p[V]·y[j]``) —
    # the volume is the units the law's inputs are quoted in, not a term of the
    # law. ``sp.diff`` w.r.t. the size sees only the explicit occurrences, so
    # without this the column is short by ``Σ_k (∂rate/∂obs_k)·(∂obs_k/∂V)``,
    # which is the same "per-species amount conversion" the Elementary path picks
    # up as its ``∏ V_c^m`` geometry factor. Measured at 14% on a Michaelis–Menten
    # law over an hOSU species at V = 3, and it does not vanish as V → 1.
    #
    # ``obs_volume_weights`` is empty for every model without a writable
    # compartment size holding an amount-valued species — including all of .net —
    # so this loop does not run and the emitted text is unchanged.
    for a in sorted(free & set(scope.obs_volume_weights)):
        _check_derivation_deadline(scope.deadline)
        deriv = sp.diff(sym_expr, sp.Symbol(a))
        if deriv == 0:
            continue
        c_expr = sympy_to_c(deriv, resolve_symbol)
        if c_expr is None:
            return None, (
                f"the derivative w.r.t. the observable {a} — needed for the "
                "compartment-size column, because that observable reads an "
                "amount-valued species whose size is writable (issue #170) — is "
                "not representable in C"
            )
        for k, w_c in sorted(scope.obs_volume_weights[a].items()):
            terms.append((k, f"({c_expr}) * ({w_c})"))
    return terms, None


def _observable_volume_weights(species, observables, alias) -> dict[str, dict[int, str]]:
    """``∂obs_k/∂V_c`` as C, per (aliased observable name, compartment-size param).

    An amount-valued species contributes ``factor · V_c · y[j]`` to an observable
    (mirrors ``_emit_observable_lines``' ``coef = factor * av_factor[...]``), so
    differentiating w.r.t. that size leaves ``factor · y[j]`` — summed over the
    species of that compartment, in species-index order so the emitted sum
    associates the way the value's does.

    Returns ``{}`` — and therefore costs nothing and changes no emitted text —
    unless the model has an amount-valued species whose compartment size is a
    writable parameter (issue #170 stage 2's ``volume_param_idx0``). That is every
    ``.net`` model, every hOSU=false SBML model, and every model loaded before
    #170.
    """
    _, av_param = _amount_volume_factors(species)
    if not av_param:
        return {}
    out: dict[str, dict[int, str]] = {}
    for o in observables:
        per_vol: dict[int, list[str]] = {}
        for si, factor in o.get("entries", ()):
            k = av_param.get(int(si), -1)
            f = float(factor)
            if k < 0 or f == 0.0:
                continue
            per_vol.setdefault(k, []).append(
                f"y[{int(si)}]" if f == 1.0 else f"{f!r} * y[{int(si)}]"
            )
        if per_vol:
            out[alias(o["name"])] = {
                k: (v[0] if len(v) == 1 else "(" + " + ".join(v) + ")") for k, v in per_vol.items()
            }
    return out


def _functional_dfdp_terms(
    core, data, deadline: float | None = None
) -> tuple[dict[int, list[tuple[int, str]]], str | None]:
    """Differentiate every Functional rate law w.r.t. every parameter it reads.

    Returns ``({reaction_idx: [(param_idx, c_expr_for_∂func/∂p)]}, None)``, or
    ``({}, reason)`` to decline the **whole model** — the ``CVodeSensInit1``
    callback is all-or-nothing, so there is no such thing as declining one
    reaction. A reaction with no parameter dependence at all maps to ``[]``,
    which is a *success* (a genuinely zero ∂f/∂p column), never a decline.

    The rate law is read from ``functional_jacobian_context()`` — the same source
    ``generate_jacobian_from_model`` reconstructs the analytical Jacobian from —
    and flattened by the same helpers:

    * :func:`bngsim._jacobian._inline_functions` resolves nested user functions,
    * :func:`_inline_derived_param_refs` is *not* applied to the rate law; a
      derived parameter stays a symbol so its own column is the direct partial
      (matching how the Elementary path emits ``case iP`` for a ``_rateLaw{N}``),
      and the #15/#41 chain rule to its primaries is added as extra terms via
      :func:`_derived_param_jacobian_checked` — which is also what keeps a
      *nested* derived parameter reachable.

    Every rejection carries a reason naming what blocked it (#56 precedent): the
    ``abs``/``min``/``max``/``floor``/``ceil``/``round`` constructs
    :func:`bngsim._jacobian.unsupported_expr_construct` already rejects for #198
    (reused rather than re-spelled), a condition whose crossing moves with a
    parameter that issue #48 does not compensate (GH #68 — the ``if()`` /
    comparison / logical class is waived by ``allow_conditions`` and re-gated by
    :func:`bngsim._switch_sensitivity.uncompensated_condition_reason`), an
    unresolved call head, a free symbol that is neither observable nor parameter
    nor ``time`` (this is what catches ``rateOf``, whose accessor is an evaluator
    variable and not a model parameter), a derivative sympy cannot render as C,
    and a derived parameter whose own Jacobian was lost.

    ``deadline`` (GH #90) bounds the sympy work: this loop runs one ``sp.diff``
    per (distinct rate law, parameter it reads) pair, the one axis on this path
    that grows super-linearly with model size, so an unbudgeted genome-scale
    Functional model would appear to hang the build rather than decline. Expiry
    declines like any other reason, naming how far it got.
    """
    params = data["parameters"]
    observables = data["observables"]
    functions = data["functions"]
    reactions = data["reactions"]

    from bngsim._jacobian import _DerivationBudgetExceeded, has_condition_construct

    ctx = core.functional_jacobian_context()
    frxn_by_idx = {int(r["rxn_idx"]): r for r in (ctx.get("functional_reactions") or [])}
    func_map = dict(ctx["function_map"])

    param_idx_by_name = {p["name"]: i for i, p in enumerate(params)}
    primary_param_names = {p["name"] for p in params if p.get("is_const", True)}
    derived_exprs = {
        p["name"]: p.get("expression", "")
        for p in params
        if not p.get("is_const", True) and p.get("expression", "")
    }

    # Python keywords only: every derivative below prints through
    # `sympy_to_c(expr, resolve)`, which resolves each symbol to a C reference and
    # never prints a name, so C reserved words need no alias here (GH #108 — the
    # rule is written out in `_alias_keyword_param`). The wide map is for
    # `sp.ccode` output, which this function never produces; the derived-parameter
    # chain rule it splices in below brings its own, already round-tripped.
    def _alias(n: str) -> str:
        return _alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n

    # Symbol → C, and (separately) symbol → the parameter it is a derivative
    # variable for. Registration order is ``generate_jacobian_from_model``'s:
    # parameters first, observables last so an observable wins a name collision,
    # matching the interpreted ExprTk variable binding. A shadowed parameter is
    # dropped from the differentiation set too — otherwise a model with an
    # observable and a parameter of the same name would report ∂f/∂p for a symbol
    # that actually resolves to the observable.
    c_ref: dict[str, str] = {}
    param_of_alias: dict[str, str] = {}
    for i, p in enumerate(params):
        a = _alias(p["name"])
        c_ref[a] = f"p[{i}]"
        param_of_alias[a] = p["name"]
    for j, o in enumerate(observables):
        a = _alias(o["name"])
        c_ref[a] = f"obs[{j}]"
        param_of_alias.pop(a, None)
    # A function-bound synthetic parameter (the ``betaI`` a .net rate law names)
    # holds no independent value — ``_inline_functions`` replaces the token with
    # the function body, so it is never a differentiation variable.
    for f in functions:
        param_of_alias.pop(_alias(f["name"]), None)

    # GH #68: a condition-bearing rate law is emittable only behind the
    # switch-time detector's own clock-threshold recognizer. Building that view
    # costs two RHS probes and a pass over the parameter table, so skip it
    # entirely for the condition-free majority (GH #67's population), where the
    # construct pre-scan rejects a condition before this could matter. Scanned
    # over the reaction rate expressions as well as the function bodies: for a
    # ``.net`` model the rate expression *is* one of the bodies, but the check
    # should not depend on that. A model whose scope cannot be assembled keeps
    # the pre-#68 behaviour — conditions decline it.
    switch_scope = None
    conditional_text = list(func_map.values()) + [
        str(r.get("rate_expr", "")) for r in frxn_by_idx.values()
    ]
    if any(has_condition_construct(body) for body in conditional_text):
        from bngsim._switch_sensitivity import switch_condition_scope

        try:
            switch_scope = switch_condition_scope(core, ctx)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "GH #68: switch-condition scope unavailable (%s); conditions decline", exc
            )

    scope = _FunctionalDfdpScope(
        func_map=func_map,
        c_ref=c_ref,
        param_of_alias=param_of_alias,
        param_idx_by_name=param_idx_by_name,
        primary_param_names=primary_param_names,
        derived_exprs=derived_exprs,
        switch_scope=switch_scope,
        deadline=deadline,
        derived_jac_cache={},
        # (#170 stage 3) The compartment size an observable's amount conversion
        # hides. Empty ⇒ the derivative loop that reads it never runs.
        obs_volume_weights=_observable_volume_weights(data["species"], observables, _alias),
    )

    # A rule-generated network reuses one rate-law expression across many
    # reactions; the sympy differentiation is the expensive step, so memoize on
    # the expression text (the only per-reaction input the differentiation reads).
    cache: dict[str, tuple[list[tuple[int, str]] | None, str | None]] = {}
    out: dict[int, list[tuple[int, str]]] = {}

    def _decline(reason: str) -> tuple[dict[int, list[tuple[int, str]]], str]:
        _warn_functional_sens_rhs_refused(reason)
        return {}, reason

    for rxn_idx, rxn in enumerate(reactions):
        rtype = rxn["type"]
        if rtype == "elementary":
            continue
        label = f"reaction {rxn_idx + 1} ({rxn['function_name']})"
        if rtype == "mm":
            # Michaelis–Menten is closed form and handled by _mm_dfdp_terms, which
            # the caller runs over the whole model at once (it has to cross-check
            # the analytical Jacobian plan entry by entry).
            continue
        if rtype != "functional":
            return _decline(
                f"{label} has rate-law type {rtype!r}, which has no analytic ∂f/∂p yet"
            )
        frxn = frxn_by_idx.get(rxn_idx)
        if frxn is None:
            # functional_jacobian_context() skips a reaction whose function name
            # resolves to no expression — there is nothing to differentiate.
            return _decline(f"{label} has no resolvable rate-law expression")
        # GH #160: a cross-compartment reaction used to decline the whole model
        # here, because the ∂f/∂p scatter had no form for the per-row compartment
        # divide. It has one now — the same ``row_divisor`` the RHS and the J·v
        # reconstruction already apply — so ∂func/∂p is derived like any other
        # rate law and the divide happens where the row lands, not here.
        rate_expr = frxn["rate_expr"]
        hit = cache.get(rate_expr)
        if hit is None:
            try:
                hit = _functional_rate_law_partials(rate_expr, scope)
            except _DerivationBudgetExceeded:
                # GH #90. The deadline is checked on entry to the differentiation
                # as well as per-parameter, so this doubles as the between-laws
                # check — and it costs nothing on a cache hit, which is what a
                # rule-generated network is almost entirely made of.
                return _decline(
                    _sens_budget_decline_reason(
                        len(data["species"]),
                        f"after {len(cache)} distinct rate law(s), at {label}",
                    )
                )
            cache[rate_expr] = hit
        terms, why = hit
        if terms is None:
            wrapped = (
                f"the Functional rate law for {label} ({rate_expr!r}) could not be "
                f"differentiated — {why or 'unknown reason'}"
            )
            # Adding the reaction's name must not lose *which kind* of reason this
            # is: an f-string over an UncompensatedCrossingReason is a plain str,
            # and the warning would then promise a correct difference-quotient
            # fallback for the one class where the fallback is wrong too (#146).
            return _decline(_carry_reason_class(why, wrapped))
        out[rxn_idx] = terms
    return out, None


def _jacv_add(col: int, row: int, value_c: str, prefix: str) -> str:
    """``_functional_jacobian_groups`` scatter that fuses the matvec (GH #67).

    The analytical Jacobian writes ``∂f_row/∂x_col`` into a matrix element; the
    sensitivity RHS only ever needs that element multiplied by ``v[col]`` and
    summed into ``Jv_out[row]``. Both indices are known at emit time, so the fusion
    is a left-hand-side rewrite: no ``n×n`` scratch buffer inside the CVODES
    callback (267 KiB on the widest Functional corpus model), no per-column memset
    of it, O(nnz) work instead of O(n²), and no ``CodegenSensUserData`` widening.
    Never declines — every entry has a home in a dense vector.
    """
    return f"{prefix}Jv_out[{row}] += ({value_c}) * v[{col}];"


def generate_sens_from_model(
    model, *, functional: bool = False, emit_term_scale: bool = False
) -> str | None:
    """Generate C source for the CVODES analytical sensitivity RHS from a
    built model, parallel to ``generate_sens_rhs_c`` (.net path).

    Returns ``None`` if any reaction is non-Elementary — the caller then
    falls back to RHS-only codegen and CVODES uses internal FD.

    ``functional`` lifts that gate for Functional rate laws, emitting both halves
    of their sensitivity RHS: the analytic ``∂f/∂p`` derived by
    :func:`_functional_dfdp_terms` (GH #66), and the ``J·yS`` reconstructed by
    :func:`_functional_jacobian_groups` with the matvec fused into the scatter
    (GH #67). ``generate_combined_from_model`` sets it; it stays a keyword so the
    pre-#67 behaviour is one argument away for an A/B, and so the .net path (which
    has no model to read a rate law off) keeps declining as it always has.

    A Functional model is emitted only when **every** rate law it must
    differentiate is smooth algebra, or is conditional in a way issue #48 already
    compensates. A non-smooth builtin (``abs``/``min``/``max``/``floor``/
    ``ceil``/``round``), or a condition whose crossing moves with a parameter and
    is not a recognized clock threshold (GH #68), declines the whole model —
    ``CVodeSensInit1`` takes one callback for every column, so a single such law
    taints all of them — with a warning naming what blocked it. sympy
    differentiates a ``Piecewise`` w.r.t. a condition-only parameter to a clean
    ``0``, so nothing downstream would catch a bad condition, which is exactly why
    the pre-scan rejects on tokens and the gate is a separate, explicit check.

    Closes #15: parameters whose ``is_const`` field is False (derived
    expressions like ``_rateLaw_<rid>`` synthesized by the SBML loader for
    products of constant SBML parameters, or arbitrary BNGL
    ``# ConstantExpression`` rate constants surfaced via ``add_parameter(...,
    is_expression=True)``) get chain-rule expansion via sympy. Each derived
    rate constant ``p_d = expr(primary_1, primary_2, ...)`` contributes
    ``(∂p_d/∂primary_k) * sf * ∏y^m`` to the sensitivity of every primary
    parameter that appears in ``expr``, exactly mirroring the .net path's
    ``derived_expansion`` machinery.

    Every symbolic derivation this triggers — the Functional ∂func/∂p and both
    flavours of derived-parameter chain rule — shares one wall-clock budget
    (GH #90), so a model whose ``sp.diff`` work does not finish in time declines
    with a warning and falls back to CVODES' internal difference quotient instead
    of hanging the build.
    """
    core = model._core if hasattr(model, "_core") else model
    data = core.codegen_data()

    params = data["parameters"]
    species = data["species"]
    reactions = data["reactions"]

    n_sp = len(species)
    n_params = len(params)

    # GH #90: one deadline for the whole build, resolved before any sympy runs.
    from bngsim._jacobian import _DerivationBudgetExceeded

    deadline = _sens_derivation_deadline(n_sp)

    # Bail if any reaction is non-Elementary — analytical sens RHS is only
    # defined for k * sf * ∏y^m kinetics. Same constraint as
    # generate_sens_rhs_c (line 762-765) for the .net path.
    # ``function_name`` is the key the derived-expansion lookup below uses, so
    # collect the same key here (not the parameter index) when deciding which
    # derived parameters can reach this RHS.
    rate_const_names: set[str] = set()
    for rxn_idx, rxn in enumerate(reactions):
        # GH #160: the cross-compartment per-row volume divide is covered on the
        # Functional path only. ∂f/∂p gets it from ``row_divisor`` below, and
        # J·yS from _functional_jacobian_groups (which also supplies the ∂/∂V_live
        # column a variable volume needs). An Elementary or Michaelis–Menten
        # reaction carrying the flag would reach neither — its J·v comes from the
        # undivided loop in _emit_sens_rhs_body — so decline rather than emit half
        # a divide. No loader produces one today; the check is what keeps that
        # true rather than assumed.
        if rxn.get("per_species_volume_scaling", False) and rxn["type"] != "functional":
            _warn_functional_sens_rhs_refused(
                f"reaction {rxn_idx + 1} ({rxn['function_name']}) is cross-compartment "
                f"(per-species volume scaling) with rate-law type {rxn['type']!r}, whose "
                "J*yS has no form for the per-species compartment divide"
            )
            return None
        if rxn["type"] != "elementary":
            if not functional:
                return None
            # A Functional reaction's ``function_name`` names the rate-law
            # *function*, not a rate-constant parameter, so it must not seed the
            # derived-rate-constant expansion below. Its ∂func/∂p (including any
            # chain rule through a derived parameter the law reads) comes from
            # _functional_dfdp_terms instead.
            continue
        rate_const_names.add(rxn.get("function_name", ""))

    # GH #66: the Functional half of ∂f/∂p. Derived before anything is emitted so
    # an undifferentiable rate law declines the model with a warning naming it,
    # rather than reaching the emitter and contributing a silent zero. All-
    # Elementary models never enter here (and the gate above already returned).
    functional_terms: dict[int, list[tuple[int, str]]] = {}
    functional_jacv_groups: list[list[str]] = []
    if functional:
        functional_terms, decline = _functional_dfdp_terms(core, data, deadline)
        if decline is not None:
            return None
        if functional_terms:
            # GH #67: the other half — J·yS over the Functional reactions. Same
            # reconstruction the compiled analytical Jacobian uses, with the matvec
            # fused into the scatter, so ∂f/∂x is derived once for both consumers.
            # A decline here is the #151 emitters' ("this derivative is not
            # representable in C"), which the ∂f/∂p pass cannot see: it
            # differentiates w.r.t. parameters, this one w.r.t. species.
            # GH #90: this half runs sympy too (the #151 native emitters cover the
            # saturable family, everything else falls through to it), so it shares
            # the build's deadline rather than being the one unbounded derivation
            # left on the path.
            try:
                groups = _functional_jacobian_groups(core, data, _jacv_add, deadline)
            except _DerivationBudgetExceeded:
                _warn_functional_sens_rhs_refused(
                    _sens_budget_decline_reason(
                        n_sp, "deriving J*yS over the Functional reactions"
                    )
                )
                return None
            if groups is None:
                _warn_functional_sens_rhs_refused(
                    "a Functional rate law's derivative with respect to the species it "
                    "reads could not be emitted as C, so J*yS would be incomplete"
                )
                return None
            functional_jacv_groups = [g for gs in groups for g in gs]

    fixed_sp = {i for i, s in enumerate(species) if s["fixed"]}
    mm_terms_by_rxn: dict[int, list[tuple[int, list[str]]]] = {}

    # Chain-rule expansion for derived rate constants (#15). Each codegen_data
    # parameter carries ``is_const`` (False ⇒ derived) and ``expression``
    # (e.g. ``"kt * Bmax"`` for the SBML loader's synthesized rate constants
    # or ``"5/MEK"`` for BNGL ``# ConstantExpression`` lines). The Jacobian
    # ``∂p_d/∂primary`` is computed via sympy and then rewritten with
    # ``primary -> p[idx]`` so it can be inlined into ``bngsim_dfdp``.
    param_idx_by_name = {p["name"]: i for i, p in enumerate(params)}
    primary_param_names = {p["name"] for p in params if p.get("is_const", True)}
    derived_exprs = {
        p["name"]: p.get("expression", "")
        for p in params
        if not p.get("is_const", True) and p.get("expression", "")
    }
    # GH #55: the Michaelis–Menten half. Closed form, so no sympy and no
    # per-reaction cache — but it needs the analytical Jacobian plan, both for
    # ``J·v`` and to cross-check which species is the enzyme, so a model whose
    # plan is unavailable declines rather than guessing. Derived after the
    # Functional pass so a model carrying both is refused by whichever fails
    # first, with that pass's reason.
    if functional:
        plan = core.codegen_jacobian_plan()
        mm_rxn_count = sum(1 for r in reactions if r["type"] == "mm")
        if mm_rxn_count:
            if not plan.get("available"):
                _warn_functional_sens_rhs_refused(
                    f"the model has {mm_rxn_count} Michaelis–Menten reaction(s) but no "
                    "analytical Jacobian plan to build their J*yS from"
                )
                return None
            mm_terms_by_rxn, mm_decline = _mm_dfdp_terms(
                data, plan["mm"], param_idx_by_name, primary_param_names, derived_exprs
            )
            if mm_decline is not None:
                _warn_functional_sens_rhs_refused(mm_decline)
                return None
            mm_jacv = _mm_jacobian_groups(plan["mm"], _jacv_add)
            if mm_jacv is None:  # pragma: no cover - a dense vector never misses
                _warn_functional_sens_rhs_refused(
                    "a Michaelis–Menten Jacobian entry had no home in J*yS"
                )
                return None
            functional_jacv_groups = functional_jacv_groups + mm_jacv

    derived_expansion: dict[str, dict[str, str]] = {}
    # GH #99: one memo for the whole loop (see generate_sens_rhs_c). Separate
    # from the Functional pass's — that one is scoped to its own scope object,
    # and both are pure functions of the same DAG, so neither can disagree.
    derived_jac_cache: dict[str, tuple[dict[str, str] | None, str | None]] = {}
    for p in params:
        # As in generate_sens_rhs_c: only a derived parameter that is some
        # reaction's rate constant reaches this RHS, so only those are
        # differentiated, and only those can invalidate it.
        if p.get("is_const", True) or p["name"] not in rate_const_names:
            continue
        expr = p.get("expression", "")
        if not expr:
            continue
        try:
            jac, reason = _derived_param_jacobian_checked(
                expr,
                primary_param_names,
                param_idx_by_name,
                derived_exprs=derived_exprs,
                deadline=deadline,
                cache=derived_jac_cache,
                name=p["name"],
            )
        except _DerivationBudgetExceeded:
            # GH #90: same budget as the Functional pass above, and the same
            # outcome — decline to CVODES' difference quotient rather than let a
            # model with thousands of derived rate constants hang the build. This
            # loop is reached by Elementary-only models too, which have no other
            # sympy on this path.
            _warn_functional_sens_rhs_refused(
                _sens_budget_decline_reason(
                    n_sp, f"deriving the rate constant {p['name']} = {expr!r}"
                )
            )
            return None
        if reason is not None:
            # Issue #56 — see generate_sens_rhs_c: a dropped chain rule here
            # reads downstream as a hard zero, so refuse the analytic RHS and
            # let CVODES' internal difference quotient answer correctly.
            _warn_sens_rhs_refused(p["name"], expr, reason)
            return None
        if jac is not None:
            derived_expansion[p["name"]] = jac

    # Build the normalized rxn_data shape consumed by _emit_sens_rhs_body.
    # Reactant/product indices from codegen_data() are already 0-based, unlike
    # the .net path which carries 1-based indices and shifts them here.
    # GH #75: per-species amount factor (volume_factor for amount_valued
    # species, else 1.0). An amount_valued reactant participates in the rate by
    # its amount (stored × V_c), so a reaction's rate carries the constant
    # ∏_{amount_valued reactants} V_c^mult — exactly mirroring the C++
    # AnalyticalJacobianData::ReactionTerms::amount_factor. 1.0 for .net / V=1 /
    # hOSU=false ⇒ byte-identical codegen.
    av_factor, av_param = _amount_volume_factors(species)
    # (#170 stage 3) The rate's non-geometry factor, as C — ``p[k]`` for an
    # Elementary rate constant, ``func[i]`` for a Functional rate law. The
    # storage-half derivatives below are the rate times a volume exponent, so
    # they need the same handle on the rate that generate_rhs_from_model uses to
    # build it. Name → function index, the map that emitter builds too.
    func_idx_by_name = {f["name"]: i for i, f in enumerate(data.get("functions", []))}
    # Synthetic rxn_data entries for the ∂/∂V of the cross-compartment row
    # divide, appended after every real reaction so the Elementary J·v groups
    # (which number themselves by rxn_data index) keep their text.
    volume_storage_rows: list[dict] = []

    rxn_data: list[dict] = []
    for rxn_idx, rxn in enumerate(reactions):
        is_elementary = rxn["type"] == "elementary"
        rate_params = list(rxn["rate_param_indices"])
        # GH #66: a Functional reaction has no scalar rate constant, so it
        # contributes no *direct* ∂f/∂k term (and no bngsim_jac_vec group) —
        # only the ∂func/∂p terms collected below.
        pidx = rate_params[0] if (rate_params and is_elementary) else -1
        sf = rxn["stat_factor"]
        reactants = list(rxn["reactants"])
        products = list(rxn["products"])

        stoich: dict[int, int] = {}
        for ri in reactants:
            stoich[ri] = stoich.get(ri, 0) - 1
        for pi in products:
            stoich[pi] = stoich.get(pi, 0) + 1

        # GH #66: the species factor ∏R (and with it the GH #75 amount factor)
        # multiplies the rate only when apply_species_factor is set — SBML's
        # unified emission bakes the reactant factor into the kinetic law
        # instead. Mirrors the ``rtype == "functional"`` branch of
        # generate_rhs_from_model, so the geometry the derivative is multiplied
        # by is the geometry the RHS used. Always set for Elementary.
        # GH #55: a Michaelis–Menten rate already carries E and sFree(S), so it
        # has no ∏R geometry to be multiplied by — its terms arrive as ``mm_terms``
        # (which bypass _build_geom_terms) and its J·v comes from the shared
        # Jacobian builder, so an empty rmult also keeps it out of the Elementary
        # jac_vec loop.
        with_species_factor = is_elementary or (
            rxn["type"] != "mm" and bool(rxn.get("apply_species_factor", True))
        )
        rmult = Counter(reactants) if with_species_factor else Counter()

        amount_factor_c = (
            _amount_factor_c(
                [(av_param.get(ri, -1), av_factor[ri]) for ri in reactants if ri in av_factor]
            )
            if with_species_factor
            else None
        )

        # Resolve the rate-constant param's name so we can look up any
        # chain-rule expansion. ``rxn["function_name"]`` is the name passed
        # to add_reaction(..., "elementary", name); it matches a parameter
        # entry by name (the param's index lookup in C++ used the same key).
        rate_pname = rxn.get("function_name", "")
        derived_terms: list[tuple] = []
        if rate_pname in derived_expansion:
            for primary_name, c_expr in derived_expansion[rate_pname].items():
                p_idx_k = param_idx_by_name.get(primary_name, -1)
                if p_idx_k < 0:
                    continue
                derived_terms.append((p_idx_k, c_expr))
        # ∂func_r/∂p rides the same channel: _emit_sens_rhs_body multiplies each
        # entry's C expression by the geometry above and scatters it by net
        # stoichiometry, which is exactly Σ_r stat_r·netstoich_ir·(∂func_r/∂p)·∏R_r.
        derived_terms.extend(functional_terms.get(rxn_idx, ()))

        # ── (#170 stage 3) the STORAGE half of ∂f/∂V ───────────────────────
        #
        # A compartment size reaches ``f`` by two roads. The kinetic-law road —
        # the volume as a symbol in a rate law — is already differentiated: it
        # arrives above as ``_rateLaw_<rid> = k·(C/_V0_C)^n``'s chain rule, or as
        # ∂func/∂C. The *storage* road is the one #170's refusal names, and it is
        # the two factors the emitters put in by hand because they are a
        # convention rather than a symbol the model wrote:
        #
        #   f_i = Σ_r (ν_ir / V_i) · rate_r,   rate_r = base_r · sf · ∏V_c^{m_c} · ∏x^m
        #          ↑ the psvs row divide        ↑ the GH #75 amount factor
        #
        # Both are ``p[]`` reads since stage 2, so both are differentiable; until
        # stage 3 neither contributed a term and the column was the kinetic half
        # alone. ∂/∂V_c of the amount factor is ``rate·m_c/V_c``, which is the
        # existing "(derivative expression) × geometry" shape with ``base_r·m_c/V_c``
        # as the expression — so it rides ``derived_terms`` and needs no new
        # emission. The row divide is handled below, where the row set differs.
        #
        # ``base_c`` is None only where the RHS itself emits no rate (an
        # Elementary reaction with no rate constant, an unresolved function) or
        # for Michaelis–Menten, whose rate carries no amount factor and which the
        # psvs gate above already declines — so a None here always means "this
        # reaction reads no volume", never "a term was dropped".
        if is_elementary:
            base_c = f"p[{rate_params[0]}]" if rate_params else None
        elif rxn["type"] == "functional":
            _fidx = func_idx_by_name.get(rate_pname, -1)
            base_c = f"func[{_fidx}]" if _fidx >= 0 else None
        else:
            base_c = None

        if base_c is not None and amount_factor_c is not None:
            # One exponent per live compartment size across the reactant
            # occurrences — the same list _amount_factor_c folds into the factor,
            # counted rather than multiplied. A static (non-live) factor has no
            # index and contributes no column.
            for _vc, _m in sorted(
                Counter(av_param[ri] for ri in reactants if ri in av_param).items()
            ):
                _num = f"{_m}.0 * " if _m != 1 else ""
                derived_terms.append((_vc, f"{_num}{base_c} / p[{_vc}]"))

        entry = {
            "param_idx": pidx,
            "stat_factor": sf,
            "stoich": stoich,
            "reactant_mult": dict(rmult),
            "derived_terms": derived_terms,
            "amount_factor_c": amount_factor_c,
        }
        # GH #160: a cross-compartment reaction's law evaluates to amount/time
        # while each affected species stores amount/V_c, so every accumulation
        # row divides by its own compartment volume — the same divide the RHS
        # emits and the J·v reconstruction folds into its scatter. Only rows that
        # actually divide are recorded, so a per_species_volume_scaling reaction
        # whose volumes all happen to be 1 emits the unchanged text.
        if rxn.get("per_species_volume_scaling", False):
            row_divisor = {}
            for si in stoich:
                live_idx, sdiv, sdiv_param = _psvs_row_divisor(species, si)
                if live_idx >= 0 or sdiv_param >= 0 or sdiv != 1.0:
                    row_divisor[si] = (live_idx, sdiv, sdiv_param)
            if row_divisor:
                entry["row_divisor"] = row_divisor
                # (#170 stage 3) ∂/∂V_c of that divide. Row i contributes
                # ν_i·rate/V_i, so ∂/∂V_i is −ν_i·rate/V_i² — a term the rows
                # divided by *some other* compartment do not get, which is why
                # this cannot ride the entry above (its scatter covers every
                # row). It is emitted as its own rxn_data entry with the same
                # geometry, the stoichiometry restricted to the rows that divide
                # by this size, and one derived term ``−base/V_c``: the ordinary
                # scatter then applies its own ν_i/V_i and lands exactly on
                # −ν_i·rate/V_i². Reusing the entry shape rather than adding an
                # emission keeps the obs[]/func[] signature resolution, the
                # chunking and the issue #177 term-scale mirror in one place.
                # ``param_idx = -1`` keeps it out of the Elementary J·v loop,
                # the same way a Functional reaction stays out.
                if base_c is not None:
                    _by_size: dict[int, dict[int, int]] = {}
                    for _si, (_live, _sdiv, _sdivp) in row_divisor.items():
                        if _live < 0 and _sdivp >= 0:
                            _by_size.setdefault(_sdivp, {})[_si] = stoich[_si]
                    for _vc, _rows in sorted(_by_size.items()):
                        volume_storage_rows.append(
                            {
                                "param_idx": -1,
                                "stat_factor": sf,
                                "stoich": _rows,
                                "reactant_mult": dict(rmult),
                                "derived_terms": [(_vc, f"-{base_c} / p[{_vc}]")],
                                "amount_factor_c": amount_factor_c,
                                "row_divisor": {_si: row_divisor[_si] for _si in _rows},
                            }
                        )
        mm_terms = mm_terms_by_rxn.get(rxn_idx)
        if mm_terms:
            entry["mm_terms"] = mm_terms
        rxn_data.append(entry)

    rxn_data.extend(volume_storage_rows)

    src = _emit_sens_rhs_body(
        rxn_data,
        n_sp,
        n_params,
        fixed_sp,
        value_lines_fn=lambda: _sens_value_lines(data),
        functional_dfdp=bool(functional_terms) or bool(mm_terms_by_rxn),
        emit_term_scale=emit_term_scale,
        functional_jacv_groups=functional_jacv_groups,
    )
    if src is None and functional_terms:
        # Every Functional rate law differentiated, but the emitter could not give
        # the switch the obs[]/func[] values it is written in — a table function
        # reachable only through ``data->tfun_eval``, or a rateOf body (GH #65).
        # Declining is right; doing it silently is not, since this is the one
        # decline the per-reaction loop above cannot see coming.
        _warn_functional_sens_rhs_refused(
            "every Functional rate law was differentiated, but the observable/function "
            "values they read cannot be recomputed inside the sensitivity RHS (a table "
            "function or rateOf)"
        )
    return src


def _sens_value_lines(data: dict) -> tuple[list[str], list[str]] | None:
    """The ``obs[]``/``func[]`` recomputation ``bngsim_dfdp`` reads (GH #65).

    Returns ``(obs_value_lines, func_value_lines)`` from the same emitters the
    RHS, the analytical Jacobian and the output evaluator use, so the values a
    derivative is written against are the values the RHS computed. Returns
    ``None`` — decline the whole analytic sensitivity RHS — when a function
    cannot be evaluated from ``(t, y, p)`` alone:

    * a **whole-body table function**, which ``_emit_function_lines`` emits as
      ``data->tfun_eval(tf_id, x, data->tfun_ctx)``. That callback lives on
      ``CodegenUserData``; ``bngsim_codegen_sens_rhs`` is handed a
      ``CodegenSensUserData``, which has neither field.
    * an **embedded tfun wrapper**, which even the value codegen declines
      (``generate_outputs_from_model``) because the ``tfun_<table>(...)`` token
      would be emitted as an undeclared C call.
    * **rateOf**, which needs the live dx/dt buffer no sensitivity-RHS call has.

    Called lazily and at most once per emit, and only when the ∂f/∂p switch
    actually references ``obs[``/``func[`` — an Elementary model never does, so
    it never pays the translation cost and never reaches these declines.
    """
    species = data["species"]
    observables = data["observables"]
    functions = data["functions"]
    tfun_specs = data.get("table_functions", [])

    if any(_RATEOF_PREFIX in f["expression"] for f in functions):
        return None

    tfun_names = [spec["name"] for spec in tfun_specs]
    for f in functions:
        if f["name"] in tfun_names:
            return None  # whole-body tfun → data->tfun_eval, unreachable here
        if any(f"tfun_{tname}(" in f["expression"] for tname in tfun_names):
            return None  # embedded wrapper → the value codegen declines too

    # GH #75 amount factor, exactly as generate_rhs_from_model folds it in.
    av_factor, av_param = _amount_volume_factors(species)
    param_map = {p["name"]: f"p[{i}]" for i, p in enumerate(data["parameters"])}
    species_map = {s["name"]: f"y[{i}]" for i, s in enumerate(species)}
    obs_map = {o["name"]: f"obs[{j}]" for j, o in enumerate(observables)}
    func_map = {f["name"]: f"func[{m}]" for m, f in enumerate(functions)}

    obs_lines = _emit_observable_lines(observables, av_factor, av_param) if observables else []
    # tfun_call_by_name is empty by construction — every tfun-backed function was
    # declined above, so no emitted line can reference ``data``.
    func_lines = (
        _emit_function_lines(functions, {}, param_map, species_map, obs_map, func_map, None)
        if functions
        else []
    )
    return obs_lines, func_lines


def generate_outputs_from_model(model) -> str | None:
    """Emit ``bngsim_codegen_outputs`` — the compiled observable/expression
    output evaluator (GH #136).

    At each trajectory output row the recorder needs every observable total and
    every function value. The interpreted path re-walks the ExprTk trees for all
    of them once per row, which dominates wall time on large models with many
    observables/expressions (the integration itself is a small fraction). This
    emits a C function that computes the identical quantities with the SAME
    ``_emit_observable_lines`` / ``_emit_function_lines`` the RHS uses, writing
    them into caller-provided ``obs_out`` / ``func_out`` arrays in model order
    (``obs_out[i]`` == ``model.observables()[i].total``; ``func_out[i]`` ==
    ``model.function_names()[i]``). The CvodeSimulator warm path calls it instead
    of the interpreted ``update_observables`` + ``evaluate_functions`` pass.

    Returns ``None`` (⇒ the simulator keeps the interpreted recording path) when
    there is nothing to compile (no observables and no functions) or when the
    model references ``rateOf`` — a rateOf function body needs the live dx/dt
    (``current_derivs``), which only the RHS two-pass probe produces, so it
    cannot be evaluated standalone at an output point.

    Reuses the same emission helpers as ``generate_rhs_from_model`` so the
    compiled output values are byte-identical to the codegen RHS's internal
    obs[]/func[] intermediates (and within solver tolerance of the interpreted
    path, exactly like the codegen RHS itself).
    """
    core = model._core if hasattr(model, "_core") else model
    data = core.codegen_data()

    params = data["parameters"]
    species = data["species"]
    observables = data["observables"]
    functions = data["functions"]
    tfun_specs = data.get("table_functions", [])

    n_obs = len(observables)
    n_func = len(functions)
    if n_obs == 0 and n_func == 0:
        return None

    # rateOf functions need the live dx/dt buffer the RHS probe publishes; an
    # output-point evaluation has no RHS pass, so decline and let the simulator
    # fall back to the interpreted recorder for these (rare) models.
    if any(_RATEOF_PREFIX in f["expression"] for f in functions):
        return None

    # GH #75 amount factor (see generate_rhs_from_model) — folded into the
    # observable coefficients so an amount-valued species contributes its amount.
    av_factor, av_param = _amount_volume_factors(species)

    param_names = [p["name"] for p in params]
    species_names = [s["name"] for s in species]
    obs_names = [o["name"] for o in observables]
    func_names = [f["name"] for f in functions]
    _param_map = {name: f"p[{i}]" for i, name in enumerate(param_names)}
    _species_map = {name: f"y[{i}]" for i, name in enumerate(species_names)}
    _obs_map = {name: f"obs[{i}]" for i, name in enumerate(obs_names)}
    _func_map = {name: f"func[{i}]" for i, name in enumerate(func_names)}

    # tfun-backed functions dispatch through the runtime callback, exactly as in
    # the RHS (index expr: bare ``t`` / ``p[idx]`` / ``obs[idx]``).
    tfun_call_by_name: dict[str, tuple[int, str]] = {}
    for tf_id, spec in enumerate(tfun_specs):
        kind = spec["index_kind"]
        if kind == "time":
            idx_c = "t"
        elif kind == "parameter":
            idx_c = f"p[{spec['index_param_idx']}]"
        elif kind == "observable":
            idx_c = f"obs[{spec['index_obs_idx']}]"
        else:
            continue
        tfun_call_by_name[spec["name"]] = (tf_id, idx_c)

    # Embedded (wrapper-form) tfun — a tfun call nested inside arithmetic, e.g.
    # ``(tfun('drive') + 5)/k`` — is rewritten by the model to a synthetic helper
    # reference ``tfun_<table>(...)`` (e.g. ``tfun_f_complex__tfun0()``). This reuses
    # ``_emit_function_lines``, which only resolves a *whole-body* tfun (via
    # ``tfun_call_by_name`` → a ``data->tfun_eval`` callback); it has no inline
    # placeholder-substitution pass like the .net RHS (``generate_rhs_c`` L1109), so
    # it would emit that ``tfun_<table>()`` token as an undeclared C call. Decline so
    # the interpreted recorder is kept — both when appended onto the .net RHS (which
    # compiles standalone, GH #163) and the model-based RHS (which cannot emit the
    # embedded form either). Mirrors the rateOf / no-obs-no-func declines above.
    _tfun_table_names = [spec["name"] for spec in tfun_specs]
    for f in functions:
        if f["name"] in tfun_call_by_name:
            continue  # whole-body tfun → a data->tfun_eval callback, handled below
        if any(f"tfun_{tname}(" in f["expression"] for tname in _tfun_table_names):
            return None

    # At genome scale the obs[]/func[] computation is a large basic block — shard
    # it off the serial driver into NOINLINE blocks (GH #165), gated on the same
    # reaction count the RHS chunks at (so the BNGSIM_NOINLINE macro + N_* macros
    # this is appended after are present). Flat below the threshold (byte-identical).
    chunk = _should_chunk(len(data["reactions"]))
    _out_obs_lines = (
        _emit_observable_lines(observables, av_factor, av_param) if observables else []
    )
    _out_obs_sig, _out_obs_args = _obs_blk_sig(_out_obs_lines)
    out_obs_in, out_obs_fs = (
        _shard_value_lines(
            _out_obs_lines,
            chunk=chunk,
            fn_prefix="out_obs_blk",
            signature_params=_out_obs_sig,
            call_args=_out_obs_args,
        )
        if observables
        else ([], [])
    )
    out_func_in, out_func_fs = (
        _shard_value_lines(
            _emit_function_lines(
                functions, tfun_call_by_name, _param_map, _species_map, _obs_map, _func_map, None
            ),
            chunk=chunk,
            fn_prefix="out_func_blk",
            signature_params=_FUNC_BLK_SIG,
            call_args=_FUNC_BLK_ARGS,
            preamble=_FUNC_BLK_PREAMBLE,
        )
        if functions
        else ([], [])
    )

    lines: list[str] = []
    _emit = lines.append

    # Chunked: the obs/func fill blocks live at file scope before the callback.
    for ln in (*out_obs_fs, *out_func_fs):
        _emit(ln)
    if out_obs_fs or out_func_fs:
        _emit("")

    _emit("/* Output evaluation (GH #136): observables + function values per")
    _emit("   trajectory output row. Reuses the RHS CodegenUserData typedef and")
    _emit("   N_* macros emitted by generate_rhs_from_model (appended after it). */")
    _emit("BNGSIM_EXPORT int bngsim_codegen_outputs(double t, double* y, double* obs_out,")
    _emit("                           double* func_out, void* user_data) {")
    _emit("    CodegenUserData* data = (CodegenUserData*)user_data;")
    _emit("    double* p = data->param_values;")
    # A model may have observables but no parameter-referencing expressions (or
    # vice versa); silence unused-parameter noise rather than depend on which
    # symbols a given model happens to reference.
    _emit("    (void)t; (void)y; (void)p; (void)obs_out; (void)func_out;")
    _emit("")

    if observables:
        for ln in out_obs_in:
            _emit(ln)
        _emit("    for (int _i = 0; _i < N_OBS; ++_i) obs_out[_i] = obs[_i];")
        _emit("")

    if functions:
        for ln in out_func_in:
            _emit(ln)
        _emit("    for (int _i = 0; _i < N_FUNC; ++_i) func_out[_i] = func[_i];")
        _emit("")

    _emit("    return 0;")
    _emit("}")
    return "\n".join(lines) + "\n"


def _is_auto_rate_law(name: str) -> bool:
    """True for BNG2.pl's auto-generated ``_rateLawN`` function names — the
    internal rate-law intermediates filtered out of the user-facing expression
    columns. Mirrors ``bngsim._result._is_auto_rate_law`` (kept local to avoid a
    circular import)."""
    return name.startswith("_rateLaw") and name[len("_rateLaw") :].isdigit()


def _output_sens_analysis_key(core) -> tuple:
    """Memo key for :func:`_analyze_output_sens` (GH #97).

    Cheap by construction — four counters off the built model plus the budget
    override — because avoiding ``codegen_data()`` and the sympy behind it is the
    whole point. The counters are a structural guard, not a content hash: the
    analysis is a pure function of the model's *shape* (function bodies, parameter
    expressions, and the species/parameter/observable ordering its emitted
    ``y[i]``/``p[k]`` references are indices into), and none of that changes after
    load — ``set_param`` writes values, which never reach the emitted partials.

    The budget tag is in the key for the same reason it is in the ``.net`` ``.so``
    key (:func:`_sens_budget_cache_tag`): an expiry changes which functions come
    back supported, so an analysis made under one budget must not be served to a
    caller that set another.
    """
    return (
        core.n_species,
        core.n_parameters,
        core.n_observables,
        core.n_functions,
        _sens_budget_cache_tag(),
    )


def _analyze_output_sens(model) -> dict:
    """Per-function output-sensitivity analysis (GH #198), memoized on the model.

    The single source of truth shared by the C emitter
    (:func:`generate_output_sens_from_model`) and the Python support-map accessor
    (:func:`output_sens_support`), so the emitted code and the Result's targeted
    error metadata never diverge.

    For every global function it differentiates the body w.r.t. each *directly
    referenced* symbol (species / observable / parameter / earlier function) via
    :func:`bngsim._jacobian.differentiate_expression_output_partials` — no
    inlining, so the caller can assemble the chain rule
    ``df/dθ = Σ ∂f/∂s·ds/dθ`` over every dependency kind. Derived (expression)
    parameters get the #15 chain rule (``∂p_d/∂primary`` via
    :func:`_derived_param_jacobian_checked`); a function referencing a derived
    param whose Jacobian could not be derived is marked unsupported (#198 fails
    loudly rather than silently dropping the term).

    Returns a dict with ``decline`` (non-None ⇒ the whole codegen is declined,
    mirroring :func:`generate_outputs_from_model`'s rateOf / embedded-tfun /
    no-function declines), ``func_infos`` (per-function ``{name, supported,
    reason, partials}`` in declaration order), and the emitter context. **Treat
    the returned dict as read-only** — every caller now shares one instance.

    Memoized on the model (GH #97) for cost *and* for correctness. Both callers
    above run it, so a sensitivity workflow paid it twice — 85.7 s twice on the
    worst BioModels model. Once the analysis is wall-clock-budgeted that stops
    being merely wasteful: a budget makes it no longer a pure function of the
    model, so two independent evaluations can cut at different functions, and the
    emitted C would carry a NaN sentinel for a function the support map reports as
    supported. One evaluation, one cut, one answer.
    """
    core = model._core if hasattr(model, "_core") else model
    key = _output_sens_analysis_key(core)
    memo = getattr(model, "_output_sens_analysis", None)
    if memo is not None and memo[0] == key:
        return memo[1]
    analysis = _compute_output_sens_analysis(model, core)
    # A bare ``NetworkModel`` core has no slot to hold it and simply re-analyzes;
    # both callers that matter pass the Model.
    if hasattr(model, "_output_sens_analysis"):
        model._output_sens_analysis = (key, analysis)
    return analysis


def _compute_output_sens_analysis(model, core) -> dict:
    """The uncached body of :func:`_analyze_output_sens` — call that instead."""
    from bngsim._jacobian import (
        _DerivationBudgetExceeded,
        differentiate_expression_output_partials,
    )

    data = core.codegen_data()

    params = data["parameters"]
    species = data["species"]
    observables = data["observables"]
    functions = data["functions"]
    tfun_specs = data.get("table_functions", [])

    n_func = len(functions)
    base = {
        "params": params,
        "species": species,
        "observables": observables,
        "functions": functions,
        "reactions": data["reactions"],
    }

    if n_func == 0:
        return {"decline": "model has no global functions", "func_infos": [], **base}
    # rateOf needs the live dx/dt buffer the RHS two-pass probe publishes; an
    # output-point evaluation has none — the value codegen declines here too, so
    # the internal obs[]/func[] recomputation cannot be emitted standalone.
    if any(_RATEOF_PREFIX in f["expression"] for f in functions):
        return {
            "decline": "model uses rateOf (live derivatives unavailable at an output point)",
            "func_infos": [],
            **base,
        }

    # Whole-body tfun functions dispatch through data->tfun_eval (a *value*
    # callback only; table functions are intentionally not differentiated, so
    # their output sensitivity is unsupported). Embedded tfun wrappers make even
    # the value codegen decline; mirror that decline.
    tfun_call_by_name: dict[str, tuple[int, str]] = {}
    for tf_id, spec in enumerate(tfun_specs):
        kind = spec["index_kind"]
        if kind == "time":
            idx_c = "t"
        elif kind == "parameter":
            idx_c = f"p[{spec['index_param_idx']}]"
        elif kind == "observable":
            idx_c = f"obs[{spec['index_obs_idx']}]"
        else:
            continue
        tfun_call_by_name[spec["name"]] = (tf_id, idx_c)
    _tfun_table_names = [spec["name"] for spec in tfun_specs]
    for f in functions:
        if f["name"] in tfun_call_by_name:
            continue
        if any(f"tfun_{tname}(" in f["expression"] for tname in _tfun_table_names):
            return {
                "decline": "model uses embedded table-function wrappers (value codegen declines)",
                "func_infos": [],
                **base,
            }

    # Identifier → C-reference maps (identical to the value path).
    param_names = [p["name"] for p in params]
    species_names = [s["name"] for s in species]
    obs_names = [o["name"] for o in observables]
    func_names = [f["name"] for f in functions]
    param_map = {n: f"p[{i}]" for i, n in enumerate(param_names)}
    species_map = {n: f"y[{i}]" for i, n in enumerate(species_names)}
    obs_map = {n: f"obs[{i}]" for i, n in enumerate(obs_names)}
    func_map = {n: f"func[{i}]" for i, n in enumerate(func_names)}

    # Derived-parameter chain rule (#15 machinery): ∂p_d/∂primary as C strings.
    param_idx_by_name = {n: i for i, n in enumerate(param_names)}
    primary_param_names = {p["name"] for p in params if p.get("is_const", True)}
    derived_exprs = {
        p["name"]: p.get("expression", "")
        for p in params
        if not p.get("is_const", True) and p.get("expression", "")
    }
    # Only USER functions are selectable — the auto-generated _rateLawN rate-law
    # intermediates are filtered out of the result block and never addressed by a
    # selector. A genome-scale model can carry thousands of them and running sympy
    # on each would dominate codegen, so differentiate only user functions and
    # what they transitively reference, pruned here via a cheap regex reference
    # graph before any sympy. Functions outside that closure are recorded with a
    # zero placeholder (they are filtered out of the user-facing block anyway).
    #
    # This runs BEFORE the derived-parameter chain rule below, even though the
    # function loop needs it later: it is pure regex, it can decline the whole
    # model, and (GH #97) the step count it yields is what sizes the budget both
    # phases run under. A model with no user-selectable function used to pay the
    # derived-parameter sympy before finding that out.
    known_symbols = set(param_map) | set(species_map) | set(obs_map) | set(func_map)
    name_to_idx = {f["name"]: i for i, f in enumerate(functions)}
    refs: list[set[int]] = [set() for _ in range(n_func)]
    n_syms: list[int] = []
    for i, f in enumerate(functions):
        toks = set(re.findall(r"[A-Za-z_]\w*", f["expression"]))
        n_syms.append(len(toks & known_symbols))
        for tok in toks:
            j = name_to_idx.get(tok)
            if j is not None and j != i:
                refs[i].add(j)
    relevant: set[int] = set()
    stack = [i for i, f in enumerate(functions) if not _is_auto_rate_law(f["name"])]
    while stack:
        i = stack.pop()
        if i in relevant:
            continue
        relevant.add(i)
        stack.extend(refs[i])
    if not relevant:
        return {
            "decline": "model has no user-selectable global functions",
            "func_infos": [],
            **base,
        }

    # GH #97: everything below here is sympy, so the clock starts here. One
    # deadline for the whole analysis (its own, not the ∂f/∂p phase's), sized by
    # the step count above and checked on entry to each expression and before each
    # sp.diff. ``budget_reason`` is the per-function unsupported reason once it
    # expires — set once, then every remaining function takes it without doing any
    # further work.
    n_steps = _output_sens_derivation_steps(relevant, n_syms, derived_exprs, set(param_names))
    deadline = _output_sens_derivation_deadline(n_steps)
    budget_reason: str | None = None

    derived_expansion: dict[str, dict[str, str]] = {}
    # GH #99: this loop covers *every* derived parameter, so with one memo the
    # whole DAG is differentiated exactly once — which is also what makes the
    # step count above the real cost rather than an upper bound on it.
    derived_jac_cache: dict[str, tuple[dict[str, str] | None, str | None]] = {}
    for n_expanded, (dname, dexpr) in enumerate(derived_exprs.items()):
        try:
            jac, _ = _derived_param_jacobian_checked(
                dexpr,
                primary_param_names,
                param_idx_by_name,
                derived_exprs=derived_exprs,
                deadline=deadline,
                cache=derived_jac_cache,
                name=dname,
            )
        except _DerivationBudgetExceeded:
            budget_reason = _output_sens_budget_reason(
                f"expanded {n_expanded} of {len(derived_exprs)} derived parameters", n_steps
            )
            break
        if jac is not None:
            derived_expansion[dname] = jac

    # Per-function differentiation. status ∈ {"ok", "unsupported", "skipped"}:
    # "ok" emits the chain rule; "unsupported" emits a NaN sentinel and a reason
    # the Result raises at selection time; "skipped" (outside the user-function
    # closure) is filtered out of the block and left at the caller's zero.
    func_infos: list[dict] = []
    n_done = 0
    for i, f in enumerate(functions):
        name = f["name"]
        if i not in relevant:
            func_infos.append(
                {"name": name, "status": "skipped", "reason": None, "partials": None}
            )
            continue
        if name in tfun_call_by_name:
            func_infos.append(
                {
                    "name": name,
                    "status": "unsupported",
                    "reason": "table-function output sensitivities are not supported "
                    "(table functions are not differentiated)",
                    "partials": None,
                }
            )
            continue
        # GH #97: past the deadline nothing more is attempted — not even the cheap
        # construct scan, since accumulating cost over many functions is the case
        # this budget exists to bound. Every function derived BEFORE the deadline
        # keeps working; the rest fail loudly and specifically, the same way an
        # undifferentiable one does.
        if budget_reason is not None:
            func_infos.append(
                {"name": name, "status": "unsupported", "reason": budget_reason, "partials": None}
            )
            continue
        try:
            partials, reason = differentiate_expression_output_partials(
                f["expression"],
                species_cref=species_map,
                observable_cref=obs_map,
                param_cref=param_map,
                function_cref=func_map,
                deadline=deadline,
            )
        except _DerivationBudgetExceeded:
            budget_reason = _output_sens_budget_reason(
                f"derived {n_done} of {len(relevant)} functions", n_steps
            )
            func_infos.append(
                {"name": name, "status": "unsupported", "reason": budget_reason, "partials": None}
            )
            continue
        n_done += 1
        if reason is not None:
            func_infos.append(
                {"name": name, "status": "unsupported", "reason": reason, "partials": None}
            )
            continue
        bad = next(
            (
                pn
                for pn in partials["param"]
                if pn not in primary_param_names and derived_expansion.get(pn) is None
            ),
            None,
        )
        if bad is not None:
            func_infos.append(
                {
                    "name": name,
                    "status": "unsupported",
                    "reason": f"references derived parameter {bad!r} whose "
                    "primary-parameter Jacobian could not be derived",
                    "partials": None,
                }
            )
            continue
        func_infos.append({"name": name, "status": "ok", "reason": None, "partials": partials})

    # Transitive unsupported: an ok function with a nonzero partial w.r.t. an
    # unsupported function is itself unsupported (its NaN would propagate
    # silently). Propagate in dependency order so deps are final when reached.
    # An ok function only references functions inside the closure, so a referenced
    # function is never "skipped" — only "ok" or "unsupported".
    for i in _topological_function_order(functions):
        info = func_infos[i]
        if info["status"] != "ok":
            continue
        for dep in info["partials"]["function"]:
            j = name_to_idx.get(dep)
            if j is not None and func_infos[j]["status"] == "unsupported":
                info["status"] = "unsupported"
                info["partials"] = None
                info["reason"] = f"depends on unsupported function {dep!r}"
                break

    if budget_reason is not None:
        # GH #97: a build that quietly turned every d(function)/dθ into a NaN is
        # not something to discover at selection time. Said once here, with the
        # override, in addition to the per-function reason the Result raises.
        logger.warning(
            "Expression output sensitivities: %s. The functions derived before the "
            "deadline are unaffected; the rest raise this reason when selected.",
            budget_reason,
        )

    return {
        "decline": None,
        "func_infos": func_infos,
        "param_map": param_map,
        "species_map": species_map,
        "obs_map": obs_map,
        "func_map": func_map,
        "param_idx_by_name": param_idx_by_name,
        "primary_param_names": primary_param_names,
        "derived_expansion": derived_expansion,
        "tfun_call_by_name": tfun_call_by_name,
        **base,
    }


def output_sens_support(model) -> dict[str, str | None]:
    """``{function_name: unsupported_reason_or_None}`` for #198 expression output
    sensitivities, from the same analysis the C emitter uses.

    A ``None`` value means the function's output sensitivity is computed by the
    codegen ``.so``; a string is the actionable reason it is not (an unsupported
    construct, a table function, or a whole-model decline). Threaded onto the
    :class:`Result` by the Simulator so a selector can fail loudly with the
    specific reason instead of a bare empty-block error.
    """
    analysis = _analyze_output_sens(model)
    if analysis["decline"] is not None:
        return {f["name"]: analysis["decline"] for f in analysis["functions"]}
    return {info["name"]: info["reason"] for info in analysis["func_infos"]}


def _emit_obs_sens_lines(
    observables: list,
    av_factor: dict,
    av_param: dict,
    ss: str,
    out: str,
    *,
    column_param: str | None = None,
) -> list[str]:
    """C lines computing ``out[j] = Σ_i c_ji·ss[i]`` — the linear observable
    output sensitivity for one sensitivity column whose species derivatives are
    in ``ss`` (``dx_i/dθ``). The coefficient ``c_ji`` folds the GroupEntry factor
    and the GH #75 amount-volume scaling identically to ``_emit_observable_lines``
    and the #197 C++ runtime path, so observable and expression sensitivities
    stay consistent — including issue #170 stage 2's live ``factor*p[k]`` form.

    ``column_param`` (issue #170 stage 3) is a C expression naming *which*
    parameter this column differentiates. When the coefficient ``c_ji`` is itself
    a writable compartment size — an amount-valued species, whose observable is
    quoted in amounts — that column carries the direct ``∂c_ji/∂V·x_i`` on top of
    the chain rule, and this is what selects it at run time. Omitted (or a model
    with no live amount factor) ⇒ not one extra emitted line, so every model
    without an hOSU species on a writable size keeps its text.
    """
    lines = []
    for j, o in enumerate(observables):
        entries = o["entries"]
        if not entries:
            lines.append(f"        {out}[{j}] = 0.0;")
            continue
        terms = []
        direct: dict[int, list[str]] = {}
        for sp_idx, factor in entries:
            if sp_idx in av_param:
                pref = "" if factor == 1.0 else _c_scalar(factor) + "*"
                terms.append(f"{pref}p[{av_param[sp_idx]}]*{ss}[{sp_idx}]")
                if column_param is not None and factor != 0.0:
                    direct.setdefault(av_param[sp_idx], []).append(f"{pref}y[{sp_idx}]")
                continue
            coef = factor * av_factor.get(sp_idx, 1.0)
            if coef == 1.0:
                terms.append(f"{ss}[{sp_idx}]")
            else:
                terms.append(f"{_c_scalar(coef)}*{ss}[{sp_idx}]")
        lines.append(f"        {out}[{j}] = {' + '.join(terms)};")
        for k, parts in sorted(direct.items()):
            lines.append(f"        if ({column_param} == {k})")
            lines.append(f"            {out}[{j}] += {' + '.join(parts)};")
    return lines


def generate_output_sens_from_model(model) -> str | None:
    """Emit ``bngsim_codegen_output_sens`` — the compiled observable + expression
    output-sensitivity evaluator (GH #198).

    At each output row of the cold (CVODES sensitivity) path the simulator hands
    this function the per-column state sensitivities ``state_sens[c][i] =
    dx_i/dθ_c`` (the parameter-axis ``dx/dp`` columns followed by the IC-axis
    ``dx/dY(0)`` columns), and it fills ``func_sens_out[c*N_FUNC + m] =
    d func_m/dθ_c`` via the chain rule

        df/dθ = Σ_i ∂f/∂x_i·dx_i/dθ + Σ_j ∂f/∂obs_j·dobs_j/dθ
              + Σ_k ∂f/∂p_k·dp_k/dθ + Σ_m ∂f/∂f_m·df_m/dθ.

    Observable derivatives ``dobs_j/dθ = Σ_i c_ji·dx_i/dθ`` are recomputed
    internally (and written to ``obs_sens_out`` when non-NULL) since the function
    derivatives depend on them. The parameter term is the Kronecker-δ plus the
    derived-parameter chain (``plist[c]`` selects the differentiated parameter;
    IC columns carry the sentinel ``>= N_PARAMS`` and skip it). ``obs[]`` / ``func[]``
    are recomputed with the SAME emitters as the RHS/value codegen so derivative
    and value never diverge.

    Returns ``None`` (⇒ no symbol; expression selectors raise) when
    :func:`_analyze_output_sens` declines (no functions / rateOf / embedded tfun).
    Unsupported functions are emitted with a ``NaN`` sentinel so a result is never
    silently wrong; the Result raises the targeted reason at selection time.
    """
    analysis = _analyze_output_sens(model)
    if analysis["decline"] is not None:
        return None

    func_infos = analysis["func_infos"]
    species = analysis["species"]
    observables = analysis["observables"]
    functions = analysis["functions"]
    tfun_call_by_name = analysis["tfun_call_by_name"]
    param_map = analysis["param_map"]
    species_map = analysis["species_map"]
    obs_map = analysis["obs_map"]
    func_map = analysis["func_map"]
    param_idx_by_name = analysis["param_idx_by_name"]
    primary_param_names = analysis["primary_param_names"]
    derived_expansion = analysis["derived_expansion"]

    n_obs = len(observables)

    species_idx = {n: i for i, n in enumerate(s["name"] for s in species)}
    obs_idx = {o["name"]: j for j, o in enumerate(observables)}
    func_idx = {f["name"]: i for i, f in enumerate(functions)}

    av_factor, av_param = _amount_volume_factors(species)

    # Value recomputation (same emitters as the RHS / value codegen).
    obs_value_lines = (
        _emit_observable_lines(observables, av_factor, av_param) if observables else []
    )
    func_value_lines = _emit_function_lines(
        functions, tfun_call_by_name, param_map, species_map, obs_map, func_map, None
    )

    # Parameter-axis contributions, grouped by the differentiated parameter index
    # K so the switch in bngsim_output_sens_dfdp mirrors bngsim_dfdp: a direct
    # δ term for every parameter a function references (primary or derived), plus
    # the derived chain through each derived param's primaries.
    contributions_by_k: dict[int, list[tuple[int, str]]] = {}
    for m, info in enumerate(func_infos):
        if info["status"] != "ok":
            continue
        for pname, dpartial in info["partials"]["param"].items():
            contributions_by_k.setdefault(param_idx_by_name[pname], []).append((m, dpartial))
            if pname not in primary_param_names:
                for primary, dpd in derived_expansion[pname].items():
                    contributions_by_k.setdefault(param_idx_by_name[primary], []).append(
                        (m, f"({dpartial}) * ({dpd})")
                    )

    lines: list[str] = []
    _emit = lines.append

    _emit("/* GH #198 - observable + expression output sensitivities. Reuses the")
    _emit("   RHS CodegenUserData typedef and N_* macros (appended after the RHS). */")
    _emit("")
    # df/dp helper: per-function param-axis contribution for parameter iP.
    _emit("static void bngsim_output_sens_dfdp(int iP, double t, const double* y,")
    _emit("                                    const double* p, const double* obs,")
    _emit("                                    const double* func, double* dfdp_out) {")
    _emit("    (void)t; (void)y; (void)p; (void)obs; (void)func;")
    _emit("    for (int _m = 0; _m < N_FUNC; ++_m) dfdp_out[_m] = 0.0;")
    _emit("    switch (iP) {")
    for k in sorted(contributions_by_k):
        _emit(f"    case {k}:")
        for m, c_expr in contributions_by_k[k]:
            _emit(f"        dfdp_out[{m}] += {c_expr};")
        _emit("        break;")
    _emit("    default:")
    _emit("        break;")
    _emit("    }")
    _emit("}")
    _emit("")

    _emit(
        "BNGSIM_EXPORT int bngsim_codegen_output_sens(double t, const double* y, const double* p,"
    )
    _emit("                               const double* const* state_sens, const int* plist,")
    _emit("                               int n_sens, double* obs_sens_out,")
    _emit("                               double* func_sens_out, void* user_data) {")
    _emit("    CodegenUserData* data = (CodegenUserData*)user_data;")
    _emit("    (void)t; (void)y; (void)p; (void)data; (void)obs_sens_out;")
    _emit("")
    # Recompute obs[]/func[] from (y, p, t) — byte-consistent with the values.
    for ln in obs_value_lines:
        _emit(ln)
    for ln in func_value_lines:
        _emit(ln)
    _emit("")
    if n_obs > 0:
        _emit(f"    double obs_sens_c[{n_obs}];")
    _emit("    double dfdp[N_FUNC];")
    _emit("    for (int _c = 0; _c < n_sens; ++_c) {")
    _emit("        const double* ss = state_sens[_c];")
    _emit("        double* fs = func_sens_out + (size_t)_c * N_FUNC;")
    if n_obs > 0:
        for ln in _emit_obs_sens_lines(
            observables,
            av_factor,
            av_param,
            ss="ss",
            out="obs_sens_c",
            # (#170 stage 3) An IC column carries the params.size() sentinel here,
            # so it matches no compartment index and takes no direct term —
            # correct: ∂obs/∂x_k(0) holds the volume fixed.
            column_param="plist[_c]",
        ):
            _emit(ln)
        _emit("        if (obs_sens_out) {")
        _emit("            for (int _j = 0; _j < N_OBS; ++_j)")
        _emit("                obs_sens_out[(size_t)_c * N_OBS + _j] = obs_sens_c[_j];")
        _emit("        }")
    # obs[] is only declared when the model has observables; with none, no
    # function (hence no parameter partial) references it, so NULL is safe.
    _obs_arg = "obs" if n_obs > 0 else "NULL"
    _emit("        if (plist[_c] >= 0 && plist[_c] < N_PARAMS)")
    _emit(f"            bngsim_output_sens_dfdp(plist[_c], t, y, p, {_obs_arg}, func, dfdp);")
    _emit("        else")
    _emit("            for (int _m = 0; _m < N_FUNC; ++_m) dfdp[_m] = 0.0;")
    # Per-function chain rule in dependency order so fs[l] is set before use.
    # "skipped" functions (outside the user closure) are left at the caller's
    # zero (func_sens_out must be zeroed on entry) and filtered out downstream.
    for m in _topological_function_order(functions):
        info = func_infos[m]
        if info["status"] == "skipped":
            continue
        if info["status"] == "unsupported":
            _emit(f"        fs[{m}] = NAN;  /* {info['name']}: unsupported */")
            continue
        partials = info["partials"]
        terms: list[str] = []
        for sname, c_expr in partials["species"].items():
            terms.append(f"({c_expr}) * ss[{species_idx[sname]}]")
        for oname, c_expr in partials["observable"].items():
            terms.append(f"({c_expr}) * obs_sens_c[{obs_idx[oname]}]")
        terms.append(f"dfdp[{m}]")
        for fname, c_expr in partials["function"].items():
            terms.append(f"({c_expr}) * fs[{func_idx[fname]}]")
        _emit(f"        fs[{m}] = {' + '.join(terms)};  /* {info['name']} */")
    _emit("    }")
    _emit("    return 0;")
    _emit("}")
    return "\n".join(lines) + "\n"


def generate_combined_from_model(model, emit_output_sens: bool = False) -> tuple[str, bool]:
    """Generate combined RHS + sensitivity RHS from a built model.

    Returns ``(c_source, has_sens_rhs)``. Mirrors ``generate_combined_c``
    for the .net path so the model-based pipeline emits the same combined
    .so when sensitivity is supported.

    The analytical Jacobian callback (``bngsim_codegen_jac`` dense, GH #76 Task 4,
    or ``bngsim_codegen_jac_sparse`` CSC, GH #162) and the output evaluator
    (``bngsim_codegen_outputs``, GH #136) are appended when the model qualifies
    (see ``generate_jacobian_from_model`` / ``generate_outputs_from_model``); both
    reuse the RHS block's
    ``CodegenUserData`` typedef and ``N_*`` macros, so they are emitted after the
    RHS. A ``None`` from either emitter simply omits that symbol — the simulator
    then keeps the interpreted analytical / FD Jacobian and the interpreted
    output-recording path, respectively. ``has_sens_rhs`` reflects only the
    sensitivity RHS, unchanged by the Jacobian or output emitters.

    The expression output-sensitivity evaluator (``bngsim_codegen_output_sens``,
    GH #198) is appended only when ``emit_output_sens`` is set (a sensitivity run),
    because its build-time expression differentiation is expensive on large
    functional models and is wasted when no sensitivity is requested. It is
    independent of ``has_sens_rhs`` — it consumes whatever state sensitivities
    CVODES produced (analytical sens RHS or internal FD).

    GH #67: the sensitivity RHS is asked for Functional rate laws too. The gate is
    per **model**, not per requested parameter — ``CVodeSensInit1`` installs one
    callback for every sensitivity column, so there is nothing finer to key on —
    and it is ``generate_sens_from_model``'s own: it emits only when every
    Functional law it must differentiate is smooth algebra, and declines (loudly,
    to CVODES' difference quotient) on the ``if()``/comparison class GH #68 owns
    and on the non-smooth builtins that are permanently out of scope. Set
    ``BNGSIM_NO_FUNCTIONAL_SENS_RHS=1`` to force the pre-#67 behaviour — a
    Functional model back on the difference quotient — for an A/B.
    """
    rhs_code = generate_rhs_from_model(model)
    sens_code = generate_sens_from_model(
        model,
        functional=functional_sens_rhs_enabled(),
        emit_term_scale=bool(getattr(model, "_want_output_sens", False)),
    )
    jac_code = generate_jacobian_from_model(model)
    outputs_code = generate_outputs_from_model(model)
    parts = [rhs_code]
    if sens_code is not None:
        parts.append(sens_code)
    if jac_code is not None:
        parts.append(jac_code)
    if outputs_code is not None:
        parts.append(outputs_code)
    if emit_output_sens:
        output_sens_code = generate_output_sens_from_model(model)
        if output_sens_code is not None:
            parts.append(output_sens_code)
    return "\n".join(parts), sens_code is not None


def _canon_update(h, obj) -> None:
    """Feed *obj* into the hasher *h* as a canonical, injective byte stream.

    Every value is type-tagged and every variable-length one is length-prefixed,
    so no two distinct structures can produce the same bytes. That is not
    pedantry: without the length prefix ``{"a": "b:c"}`` and ``{"a:b": "c"}``
    serialize alike, and a collision in the codegen cache key loads the wrong
    ``.so`` — silent numerical corruption rather than a slow build.

    Floats go in as their IEEE-754 bit pattern rather than ``repr``: exact by
    construction, and it keeps ``-0.0`` apart from ``0.0``. ``bool`` is matched
    before ``int`` because it is a subclass of it. Dict keys are sorted, so
    Python's insertion order never reaches the digest, and an unhandled type
    raises instead of silently hashing its ``id()``.
    """

    def tagged(tag: bytes, payload: bytes) -> None:
        h.update(tag)
        h.update(str(len(payload)).encode())
        h.update(b":")
        h.update(payload)

    if obj is None:
        h.update(b"n")
    elif obj is True:
        h.update(b"T")
    elif obj is False:
        h.update(b"F")
    elif isinstance(obj, int):
        tagged(b"i", str(obj).encode())
    elif isinstance(obj, float):
        h.update(b"f")
        h.update(struct.pack("<d", obj))
    elif isinstance(obj, str):
        tagged(b"s", obj.encode("utf-8"))
    elif isinstance(obj, bytes):
        tagged(b"y", obj)
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        h.update(str(len(obj)).encode())
        h.update(b":")
        for item in obj:
            _canon_update(h, item)
    elif isinstance(obj, dict):
        h.update(b"{")
        h.update(str(len(obj)).encode())
        h.update(b":")
        for key in sorted(obj):
            _canon_update(h, key)
            _canon_update(h, obj[key])
    else:
        raise TypeError(f"codegen cache key cannot serialize {type(obj).__name__}")


# The value the model-based cache key deliberately drops. Everything else
# codegen_data() carries is structure the emitters read; a parameter's current
# VALUE is not — the whole design of the generated C is that parameters are read
# from the runtime ``p[]`` array rather than baked as literals (see this module's
# docstring), which is exactly what lets one .so serve every point of a fit.
# Dropping it is what makes the key stable across a parameter scan; the one path
# by which a value *can* still reach the emitted source — the issue #68
# switch-condition gate — is carried as a verdict by switch_gate_cache_digest.
_CODEGEN_KEY_DROPPED_PARAM_FIELDS = ("value",)


def compute_model_codegen_hash(model, *, emit_output_sens: bool = False) -> str:
    """Compute the model-based codegen cache key from model STRUCTURE (issue #174).

    The key the compiled ``.so`` is cached under, for the model-based path
    (SBML/Antimony, and any :class:`~bngsim.Model` built through
    ``ModelBuilder``). It is derived from cheap O(model) reads — a few C++
    accessors — and never generates source, so a warm cache resolves without
    paying the RHS/sensitivity derivation that dominates ``Simulator``
    construction (97% of it on ``Smith_BMCSystBiol2013``). That is the whole
    point of the issue: hashing the generated source made the cache skip only
    the ``cc`` compile.

    The inputs are exactly what ``generate_combined_from_model`` and everything
    it calls read off the model:

    * ``codegen_data()`` minus each parameter's current *value* (see
      ``_CODEGEN_KEY_DROPPED_PARAM_FIELDS``) — this carries the attachment
      vector ``is_const``, which is live state a ``set_param`` on a derived
      parameter moves (issue #188) and which the source genuinely depends on;
    * ``codegen_jacobian_plan()`` — including ``available``, so a model whose
      analytical Jacobian has not been attached yet never shares a key with the
      same model after ``prepare_analytical_jacobian``;
    * ``functional_jacobian_context()`` — the rate-law text the Functional
      derivation differentiates;
    * the process-scoped emit decisions: ``emit_output_sens`` (the caller's
      ``_want_output_sens``), the GH #67 and GH #90 hatches, ``BNGSIM_NO_CODEGEN_JAC``,
      and the *resolved* chunking policy (resolved, so ``on`` and ``true`` do not
      make two keys for one source);
    * :func:`bngsim._switch_sensitivity.switch_gate_cache_digest` — the one
      verdict in the pipeline that reads parameter values and species initial
      concentrations.

    ``_CODEGEN_CACHE_KEY`` is mixed in first, so an emitter edit invalidates
    every key here whether or not ``_CODEGEN_VERSION`` was bumped (issue #51).
    Note this key closes the hole that constant's docstring names: a C++ change
    that alters ``codegen_data()`` is not in the source digest, but the data
    itself is now in the key.
    """
    from bngsim._switch_sensitivity import switch_gate_cache_digest

    core = model._core if hasattr(model, "_core") else model
    data = dict(core.codegen_data())
    data["parameters"] = [
        {k: v for k, v in p.items() if k not in _CODEGEN_KEY_DROPPED_PARAM_FIELDS}
        for p in data["parameters"]
    ]
    ctx = core.functional_jacobian_context()

    h = hashlib.sha256()
    h.update(_CODEGEN_CACHE_KEY.encode())
    h.update(b"\0model_structural_v1\0")
    _canon_update(h, data)
    _canon_update(h, core.codegen_jacobian_plan())
    _canon_update(h, ctx)
    _canon_update(
        h,
        (
            bool(emit_output_sens),
            functional_sens_rhs_enabled(),
            _sens_budget_cache_tag(),
            os.environ.get("BNGSIM_NO_CODEGEN_JAC") == "1",
            _chunk_threshold(),
            _chunk_block_size(),
        ),
    )
    _canon_update(h, switch_gate_cache_digest(core, ctx))
    return h.hexdigest()[:16]


# ─── Codegen wall-time recorder (T0.3) ─────────────────────────────────────────
#
# The rr_parity harness used to reconstruct codegen time by running a model
# twice and subtracting (slow + noisy). Instead, each prepare_* entry point below
# records the wall seconds it actually spent — the cc compile for the .so paths, a
# few stat()s on a cache hit, source generation for the JIT paths — so a single
# run exposes the setup cost directly. This is pure setup instrumentation: the
# per-step RHS/Jacobian hot path is never touched. The value is stashed on the
# Model (the unambiguous owner, surviving the load → construct → run handoff) and
# mirrored to a thread-local so the .net-path entry points (which take a path, not
# a Model) can still surface their time to the constructing Simulator.
_codegen_timing = threading.local()


def _record_codegen_sec(model, sec: float, cache_hit: bool | None = None) -> None:
    """Record the most recent codegen wall time AND whether the compiled .so was
    reused from the on-disk cache, on this thread and on ``model`` when one is
    available (model-based prepare paths).

    ``cache_hit`` is ``True`` when ``get_cached_so`` (or the .net memo) resolved an
    existing .so without recompiling, ``False`` when a fresh ``cc`` compile ran,
    and ``None`` when no .so was involved at all (the MIR source-only paths, or a
    codegen failure). This is the definitive cache signal — not inferred from the
    wall time, which a model-based cache hit still spends on source generation."""
    _codegen_timing.last_sec = float(sec)
    _codegen_timing.last_cache_hit = cache_hit
    if model is not None:
        # Defensive: every Model carries these slots, but a caller that somehow
        # passes a slotted object without them should not break codegen.
        with contextlib.suppress(AttributeError, TypeError):  # pragma: no cover
            model._codegen_sec = float(sec)
            model._codegen_cache_hit = cache_hit


def last_codegen_sec() -> float:
    """Wall seconds the most recent ``prepare_*`` codegen on this thread spent
    (``0.0`` if none has run). See :attr:`bngsim.Simulator.last_codegen_sec`."""
    return float(getattr(_codegen_timing, "last_sec", 0.0))


def last_codegen_cache_hit() -> bool | None:
    """Whether the most recent ``prepare_*`` codegen on this thread reused a cached
    .so (``True``), compiled fresh (``False``), or involved no .so (``None``). See
    :attr:`bngsim.Simulator.codegen_cache_hit`."""
    return getattr(_codegen_timing, "last_cache_hit", None)


def prepare_model_codegen(model) -> Path | None:
    """Generate C code from a built model, compile, and return .so path.

    This is the main entry point for model-based codegen.
    Works with any model (SBML, Antimony, .net loaded via ModelBuilder).
    Emits combined RHS + analytical sensitivity RHS when every reaction is
    Elementary; otherwise falls back to RHS-only and CVODES uses internal FD.

    Parameters
    ----------
    model : Model
        A built BNGsim model.

    Returns
    -------
    Path or None
        Path to compiled .so, or None if codegen fails.
    """
    t0 = time.perf_counter()
    cache_hit: bool | None = None
    try:
        c_source, has_sens = generate_combined_from_model(
            model, emit_output_sens=bool(getattr(model, "_want_output_sens", False))
        )
        model_hash = hashlib.sha256(c_source.encode()).hexdigest()[:16]

        # Check cache
        cached = get_cached_so(model_hash)
        if cached is not None:
            logger.debug("Model codegen cache hit: %s", cached)
            cache_hit = True
            return cached

        if has_sens:
            logger.info(
                "Model codegen: combined RHS + sensitivity RHS (%d chars)",
                len(c_source),
            )
        else:
            logger.info(
                "Model codegen: RHS only (Functional/MM model, %d chars)",
                len(c_source),
            )
        cache_hit = False
        return compile_rhs(c_source, model_hash)
    except Exception as e:
        logger.warning("Model codegen failed: %s", e)
        return None
    finally:
        _record_codegen_sec(model, time.perf_counter() - t0, cache_hit)


def prepare_codegen_source(net_path: str, model=None, emit_jac: bool = True) -> str:
    """Generate the combined codegen C source for a .net model (GH #78).

    The in-process MIR micro-JIT backend consumes this string directly instead
    of compiling it to a .so with ``cc`` and dlopen'ing the result. It is the
    SAME C source ``prepare_codegen`` compiles — RHS plus analytical sensitivity
    RHS when every reaction is Elementary, plus (when ``model`` is supplied) the
    analytical Jacobian (GH #162, gated by ``emit_jac``) and the output evaluator
    (GH #136/#163) — so the JIT'd code is numerically identical to the cc-compiled
    one and the JIT backend resolves the same compiled symbols. The emit flags are
    derived from the SAME cheap model predicates ``prepare_codegen`` uses for its
    cache key (``_codegen_emit_flags``), so the JIT and cc paths emit byte-identical
    source for a given model. No caching: c2mir JIT is ~1-2 ms, far cheaper than the
    SHA-256 + filesystem round-trip a cache would add.
    """
    t0 = time.perf_counter()
    try:
        parsed = _parse_net_file(net_path)
        _validate_net_model_for_codegen(parsed, net_path)
        want_jac, want_outputs, want_output_sens, want_term_scale = _codegen_emit_flags(
            model, emit_jac
        )
        c_source, _ = generate_combined_c(
            net_path,
            model,
            emit_jac=want_jac,
            emit_outputs=want_outputs,
            emit_output_sens=want_output_sens,
            emit_term_scale=want_term_scale,
        )
        return c_source
    finally:
        _record_codegen_sec(None, time.perf_counter() - t0)


def prepare_model_codegen_source(model) -> str | None:
    """Generate the combined codegen C source for a built model (GH #78).

    Model-based analogue of ``prepare_codegen_source``: the same combined RHS +
    sensitivity RHS + analytical Jacobian source ``prepare_model_codegen``
    compiles, returned as a string for the in-process MIR micro-JIT. Returns
    ``None`` (matching ``prepare_model_codegen``) if source generation fails.
    """
    t0 = time.perf_counter()
    try:
        c_source, _ = generate_combined_from_model(
            model, emit_output_sens=bool(getattr(model, "_want_output_sens", False))
        )
        return c_source
    except Exception as e:
        logger.warning("Model codegen source generation failed: %s", e)
        return None
    finally:
        _record_codegen_sec(model, time.perf_counter() - t0)


def prepare_ssa_propensity_lib(model, *, force_recompile: bool = False) -> str | None:
    """Compile the STRUCTURE-specialized SSA propensity vector to a cached .so.

    ``force_recompile`` (measurement only) deletes any cached ``.so`` first so the
    call pays — and times — the real one-time cc compile rather than a disk-cache
    hit; production callers leave it False (cache reuse is the whole point).

    GH #190. ``NetworkModel.emit_ssa_propensity_source_structure`` emits one C
    function, ``bngsim_ssa_propensities(const double* x, const double* p, double*
    a)``, that reads each reaction's rate constant from the runtime ``p[]`` array
    (only the structural factor stat·svf is baked). Because no parameter VALUE
    appears in the source, the .so cache key depends only on the model STRUCTURE:
    it compiles ONCE per model and is reused across every parameter point (a fit)
    and every replicate (an ensemble) — no per-point recompile, no value-keyed
    cache explosion. Compiled through the same ``cc -O3`` path the ODE codegen
    uses (``compile_rhs``), content-cached on disk. The cc kernel matches MIR's
    end-to-end (memory-bound in the SSA loop), so this is how bngsim gets the
    RR-parity recompute-all SSA path with NO MIR.

    Returns the .so path as a string, or ``None`` when the model is not fully
    mass-action (``n_unsupported > 0`` — the JIT'd vector would be incomplete) or
    compilation fails (the caller then keeps the interpreted ``compute_propensity``
    path). The C++ ``SsaSimulator`` makes the final eligibility/size decision; this
    just provides the artifact.
    """
    core = getattr(model, "_core", model)
    try:
        from bngsim._bngsim_core import emit_ssa_propensity_source_structure

        src, n_unsupported = emit_ssa_propensity_source_structure(core)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("emit_ssa_propensity_source_structure failed: %s", e)
        return None
    if n_unsupported != 0 or not src:
        return None
    # Content hash (the source fully determines the .so) mixed with the codegen
    # version and a tag, so the propensity .so never collides with an RHS .so and
    # a codegen-behavior bump invalidates stale files. v2 = structure-specialized
    # signature (params runtime arg); invalidates any v1 value-specialized .so.
    h = hashlib.sha256()
    h.update(_CODEGEN_CACHE_KEY.encode())
    h.update(b"ssa_propensity_v2_structure")
    h.update(src.encode())
    model_hash = "ssaprop_" + h.hexdigest()[:16]
    if force_recompile:
        cached = get_cached_so(model_hash)
        if cached is not None and cached.exists():
            with contextlib.suppress(OSError):
                cached.unlink()
    try:
        return str(compile_rhs(src, model_hash))
    except Exception as e:
        logger.warning(
            "SSA propensity codegen compile failed (%s); using interpreted propensities", e
        )
        return None


# Process-local memo for prepare_codegen (T2). Without it, every
# Simulator(codegen=True, net_path=...) construction on an UNCHANGED .net
# re-reads, re-parses, and SHA-256-hashes the file (two full reads + parse +
# hash) only to resolve an already-cached .so — pure overhead under PyBNF's
# construct-Simulator-per-eval pattern. The memo maps the .net's absolute path
# to (so_path, dep_stamps, codegen_version); the fast path returns so_path after
# only re-stat()ing the .net and any .tfun files it folds into the hash — no
# read, no parse, no hash. dep_stamps captures the same file set
# compute_model_hash() folds into the cache key, so editing the .net or any
# referenced .tfun changes an mtime and forces a recompute, exactly matching the
# no-memo behavior. _CODEGEN_CACHE_KEY is part of the validity test so a codegen
# behavior change invalidates stale memo entries too — including one that edits an
# emitter without bumping _CODEGEN_VERSION (issue #51).
# Keyed by (net_abspath, want_jac, want_outputs, want_output_sens,
# functional_sens): the compiled Jacobian (GH #162), output evaluator (GH #163),
# and expression output-sensitivity evaluator (GH #198) are independent
# content-distinct callbacks, and the GH #67 A/B hatch changes the sensitivity RHS
# in place — so every flag is part of the key, and an entry for one combination
# must never satisfy another.
_PREPARE_CODEGEN_MEMO: dict[
    tuple[str, bool, bool, bool, bool, bool, str], tuple[Path, tuple[tuple[str, int], ...], str]
] = {}
_PREPARE_CODEGEN_MEMO_LOCK = threading.Lock()


def _codegen_dep_stamps(net_path: str) -> tuple[tuple[str, int], ...]:
    """(abspath, mtime_ns) for the .net and every .tfun it references.

    Mirrors the file set ``compute_model_hash`` folds into the cache key, so the
    memo invalidates on exactly the same edits. Reads the .net once; called only
    on the cold (memo-miss) path, so it adds no cost to the fast path.
    """
    net_abs = os.path.abspath(net_path)
    stamps: list[tuple[str, int]] = [(net_abs, os.stat(net_abs).st_mtime_ns)]
    net_dir = Path(net_path).parent
    try:
        with open(net_path, "rb") as f:
            net_text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return tuple(stamps)
    for ref in _iter_tfun_file_refs(net_text):
        ref_path = Path(ref)
        if not ref_path.is_absolute():
            ref_path = net_dir / ref_path
        try:
            stamps.append((os.path.abspath(ref_path), os.stat(ref_path).st_mtime_ns))
        except OSError:
            # Missing/unreadable tfun: compute_model_hash silently skips it too.
            # Its absence is folded into the hash, so a later add → recompile.
            continue
    return tuple(stamps)


def _codegen_dep_stamps_unchanged(dep_stamps: tuple[tuple[str, int], ...]) -> bool:
    """True iff every recorded dependency still has its recorded mtime."""
    for path, mtime in dep_stamps:
        try:
            if os.stat(path).st_mtime_ns != mtime:
                return False
        except OSError:
            return False
    return True


def prepare_codegen(net_path: str, model=None, emit_jac: bool = True) -> Path:
    """Generate C code, compile, and return .so path (with caching).

    This is the main entry point for the codegen pipeline.
    It generates combined RHS + sensitivity RHS when possible.
    The sensitivity RHS is included for all-Elementary models (analytical
    df/dp + J*v), and — when ``model`` is supplied (GH #67) — for a Functional
    model whose rate laws are smooth algebra, reconstructed from the built model
    because the .net text alone has no expression to differentiate. MM models, and
    Functional ones carrying a condition or a non-smooth builtin, still get the RHS
    only, and CVODES uses internal FD for sensitivity.

    GH #162: when ``model`` (the built model for this .net) is supplied, ``emit_jac``
    is set, and its analytical Jacobian is complete, the compiled analytical Jacobian
    (``bngsim_codegen_jac`` dense / ``bngsim_codegen_jac_sparse`` CSC) is appended so
    a .net-loaded large sparse model gets a compiled per-step Jacobian rather than
    the interpreted fallback.

    GH #163: when ``model`` is supplied and it has observables/functions and no
    ``rateOf`` csymbol, the compiled output evaluator (``bngsim_codegen_outputs``,
    GH #136) is appended so the warm recording loop fills the per-row observable +
    function buffers with one compiled call instead of the interpreted ExprTk pass.
    This is INDEPENDENT of ``emit_jac`` — outputs are emitted for every ``jacobian``
    strategy (``fd``/``jax`` record observables too).

    The cache key gains a ``:codegen_jac``, ``:codegen_outputs``, and/or
    ``:codegen_output_sens`` (GH #198) suffix so a .so carrying any of these
    callbacks never collides with one without it. The suffixes
    key off cheap O(1) model flags (not the generated source — ``_codegen_emit_flags``),
    so a cross-process .so cache hit still avoids regenerating the (large) RHS source;
    a Jacobian derivation that fails the GH #95 budget reports
    ``analytical_jacobian_complete == False`` and drops the ``:codegen_jac`` suffix —
    no cache poisoning.

    Parameters
    ----------
    net_path : str
        Path to the .net file.
    model : optional
        The built model (``Model`` or ``NetworkModel``) for this .net. When an
        analytical Jacobian is wanted (``emit_jac``), the caller must have prepared
        it (``prepare_analytical_jacobian``). Pass ``None`` to keep RHS(+sens)-only.
    emit_jac : bool
        Whether to append the analytical Jacobian (``jacobian`` in ``auto``/
        ``analytical``). Does not affect the output evaluator, which is emitted
        whenever ``model`` qualifies.

    Returns
    -------
    Path
        Path to the compiled shared library.
    """
    t0 = time.perf_counter()
    cache_hit: bool | None = None
    try:
        net_key = os.path.abspath(net_path)

        # Two independent compiled-callback decisions, both from cheap O(1) model
        # flags (no RHS source-gen): the analytical Jacobian (GH #162, gated by
        # emit_jac + completeness + A/B hatch) and the output evaluator (GH #163,
        # whenever the model has obs/func and no rateOf — independent of emit_jac).
        want_jac, want_outputs, want_output_sens, want_term_scale = _codegen_emit_flags(
            model, emit_jac
        )
        # The GH #67 hatch is process-scoped, not file-scoped, so it belongs in the
        # in-process memo key as well as the on-disk one below — a test that flips
        # it mid-process must not be handed the other variant's .so. GH #90's
        # derivation-budget override is process-scoped in exactly the same way and
        # decides exactly the same thing (whether the analytic sens RHS is emitted),
        # so it rides along in both keys.
        memo_key = (
            net_key,
            want_jac,
            want_outputs,
            want_output_sens,
            want_term_scale,
            functional_sens_rhs_enabled(),
            _sens_budget_cache_tag(),
        )

        # Fast path (T2): an unchanged .net (and its .tfun deps) resolves to the
        # already-cached .so via a few stat() calls, skipping the re-read +
        # re-parse + SHA-256 the cold path below performs.
        with _PREPARE_CODEGEN_MEMO_LOCK:
            entry = _PREPARE_CODEGEN_MEMO.get(memo_key)
        if entry is not None:
            memo_so, dep_stamps, ver = entry
            if (
                ver == _CODEGEN_CACHE_KEY
                and memo_so.exists()
                and _codegen_dep_stamps_unchanged(dep_stamps)
            ):
                logger.debug("Codegen memo hit: %s", memo_so)
                cache_hit = True  # memo resolved an existing .so, no recompile
                return memo_so

        parsed = _parse_net_file(net_path)
        _validate_net_model_for_codegen(parsed, net_path)

        model_hash = compute_model_hash(net_path)
        # Distinct cache key per appended-callback combination; cheap to derive (no
        # RHS source-gen), so cross-process cache hits stay fast. The ":codegen_jac"
        # form is byte-identical to GH #162 so a Jacobian-only .so still hits its
        # existing cache entry; ":codegen_outputs" is appended independently.
        suffix = ""
        if want_jac:
            suffix += ":codegen_jac"
        if want_outputs:
            suffix += ":codegen_outputs"
        if want_output_sens:
            suffix += ":codegen_output_sens"
        if want_term_scale:
            suffix += ":sens_term_scale"
        # GH #67: the A/B hatch changes the emitted source but nothing else in the
        # key, so it needs its own namespace. Appended only when the hatch is SET,
        # so the default key — and every .so already in the cache — is unchanged.
        if not functional_sens_rhs_enabled():
            suffix += ":no_functional_sens"
        suffix += _sens_budget_cache_tag()
        if suffix:
            model_hash = hashlib.sha256((model_hash + suffix).encode()).hexdigest()[:16]

        # Check cache first
        cached = get_cached_so(model_hash)
        if cached is not None:
            logger.debug("Codegen cache hit: %s", cached)
            cache_hit = True
            so_path = cached
        else:
            # Generate combined RHS + sensitivity RHS (+ Jacobian / + output
            # evaluator when wanted). model=None when neither is wanted keeps the
            # historical RHS(+sens)-only source byte-for-byte.
            c_source, has_sens = generate_combined_c(
                net_path,
                model
                if (want_jac or want_outputs or want_output_sens or want_term_scale)
                else None,
                emit_jac=want_jac,
                emit_outputs=want_outputs,
                emit_output_sens=want_output_sens,
                emit_term_scale=want_term_scale,
            )
            extra = ", ".join(
                n
                for n, on in (
                    ("analytical Jacobian", want_jac),
                    ("outputs", want_outputs),
                    ("output sensitivities", want_output_sens),
                )
                if on
            )
            extra_note = f" + {extra}" if extra else ""
            if has_sens:
                logger.info("Codegen: combined RHS + sensitivity RHS (analytical)%s", extra_note)
            else:
                logger.info("Codegen: RHS only (Functional/MM model, no sens RHS)%s", extra_note)
            cache_hit = False
            so_path = compile_rhs(c_source, model_hash)

        with _PREPARE_CODEGEN_MEMO_LOCK:
            _PREPARE_CODEGEN_MEMO[memo_key] = (
                so_path,
                _codegen_dep_stamps(net_path),
                _CODEGEN_CACHE_KEY,
            )
        return so_path
    finally:
        _record_codegen_sec(None, time.perf_counter() - t0, cache_hit)
