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
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

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
_CODEGEN_VERSION = "21"

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
    ``parse_expr`` and round-tripped back to ``p[idx]`` after differentiation."""
    return f"_BNG_KW_{name}"


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
    f(a1prime)`` where ``a1prime = kcr`` — is flattened here so the downstream
    sympy Jacobian sees an expression in primary parameters only. This is what
    lets the forward-sensitivity chain rule reach through nested derived
    parameters (issue #41). Without it, ``_compute_derived_param_jacobian``
    rejects the nested expression (a non-primary free symbol) and silently
    drops the ``primary -> derived -> derived -> rate`` contribution.

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
    defining expression. When supplied, nested derived references in ``expr``
    are inlined down to primaries first (issue #41), so ``p_d = f(p_e)`` with
    ``p_e`` itself derived still yields the full chain rule. Omitting it (or
    passing ``None``) preserves the pre-#41 behavior of rejecting any
    non-primary free symbol.

    Two preprocessing passes (issue #27) widen the set of expressions that
    yield an analytic Jacobian instead of the silent zero-contribution fallback:

    1. BNGL ``if(c, t, f)`` is rewritten to ``Piecewise((t, c), (f, True))``
       so sympy differentiates the conditional analytically. The boundary
       delta is sympy's standard Piecewise convention.
    2. Primary parameter names that happen to be Python keywords (e.g.
       ``lambda`` in ``ode/scaling_example.bngl``) are aliased to safe
       placeholders before ``parse_expr`` and round-tripped back to
       ``p[idx]`` on the way out.

    Returns
    -------
    dict[str, str] or None
        ``{primary_name: c_expr_for_partial}`` covering every primary whose
        partial derivative is non-zero. Primary names appearing in the C
        expression have already been rewritten as ``p[idx]``.
    """
    s = expr.strip()
    if not s:
        return None
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return None

    # Pass 0 (issue #41): flatten nested derived-parameter references so the
    # expression is expressed purely in primaries before differentiation. A
    # no-op for single-level derived params (whose expressions carry no derived
    # name tokens), so single-level output stays byte-identical.
    if derived_exprs:
        s = _inline_derived_param_refs(s, derived_exprs)

    # Pass 1: BNGL if(c, t, f) → sympy Piecewise. Applied to the raw string
    # so whole-word matching of ``if`` sees the source as written.
    s_pre = _translate_bngl_if_to_piecewise(s)

    referenced = sorted(p for p in primary_param_names if re.search(rf"\b{re.escape(p)}\b", s_pre))
    if not referenced:
        return None

    # Pass 2: alias Python-keyword-named primaries so parse_expr can tokenize
    # the expression. Sort by length descending so e.g. an ``if_thresh`` param
    # is not partially matched by the alias of ``if``.
    sym_name_of: dict[str, str] = {
        p: (_alias_keyword_param(p) if p in _PY_KEYWORD_PARAM_NAMES else p) for p in referenced
    }
    s_aliased = s_pre
    for p_name in sorted(referenced, key=len, reverse=True):
        if p_name in _PY_KEYWORD_PARAM_NAMES:
            s_aliased = re.sub(
                rf"\b{re.escape(p_name)}\b",
                sym_name_of[p_name],
                s_aliased,
            )

    # Bind every primary's sympy symbol so sympy never reaches for built-in
    # constants or functions of the same name (e.g., ``E``, ``S``). Also bind
    # ``Piecewise`` so the if-translation in pass 1 resolves to sympy's class.
    sym_map: dict[str, sp.Symbol] = {sym_name_of[p]: sp.Symbol(sym_name_of[p]) for p in referenced}
    local_dict: dict = dict(sym_map)
    local_dict["Piecewise"] = sp.Piecewise

    try:
        sym_expr = parse_expr(s_aliased, local_dict=local_dict, evaluate=True)
    except Exception:
        # Anything still unparseable (malformed BNGL, unsupported call, etc.)
        # → fall back to the no-analytic-Jacobian path.
        return None

    # Reject if the expression introduced any free symbol that isn't a
    # known primary parameter (e.g., a derived param appearing inside another
    # derived param's expression — out of scope for this chain rule).
    allowed_sym_names = {sym_name_of[p] for p in referenced}
    free = {str(sym) for sym in sym_expr.free_symbols}
    if not free.issubset(allowed_sym_names):
        return None

    # For round-tripping the ccode output: map each (possibly-aliased) sympy
    # symbol name back to ``p[idx]`` using the ORIGINAL primary's index.
    # Sort by sympy-name length so longer (aliased) names substitute first.
    sub_pairs: list[tuple[str, int]] = sorted(
        ((sym_name_of[p], param_idx[p]) for p in referenced if p in param_idx),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )

    result: dict[str, str] = {}
    for p_name in referenced:
        deriv = sp.diff(sym_expr, sym_map[sym_name_of[p_name]])
        if deriv == 0:
            continue
        c_str = sp.ccode(deriv)
        for sym_n, idx in sub_pairs:
            c_str = re.sub(
                rf"\b{re.escape(sym_n)}\b",
                f"p[{idx}]",
                c_str,
            )
        result[p_name] = c_str
    return result or None


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
        if kind[0] == "elementary":
            _, pname, sf = kind
            rate_expr = _rate_elementary(pname, sf, reactants, param_idx, func_idx)
        elif kind[0] == "functional":
            _, fname, sf = kind
            rate_expr = _rate_functional(fname, sf, reactants, func_idx, use_array=chunk)
        elif kind[0] == "mm":
            _, kcat, km, sf = kind
            rate_expr = _rate_mm(kcat, km, sf, reactants, param_idx)
        else:
            rate_expr = "0.0"
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
    """
    lookup: dict[str, tuple[str, bool]] = dict(_BUILTIN_IDENT_MAP)
    for name, idx in param_idx.items():
        lookup[name] = (f"p[{idx}]", False)
    if use_arrays:
        for name, oi in obs_idx.items():
            lookup[name] = (f"obs[{oi}]", False)
        for fi, (_, fname, _) in enumerate(functions):
            lookup[fname] = (f"func[{fi}]", True)
    else:
        for name in obs_idx:
            lookup[name] = (f"obs_{_safe_c_name(name)}", False)
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


def _rate_mm(
    kcat: str,
    km: str,
    sf: float,
    reactants: list,
    param_idx: dict,
) -> str:
    """Generate C expression for Michaelis-Menten tQSSA rate."""
    kcat_c = f"p[{param_idx[kcat]}]" if kcat in param_idx else "0.0"
    km_c = f"p[{param_idx[km]}]" if km in param_idx else "0.0"

    if len(reactants) >= 2:
        e_idx = reactants[0] - 1
        s_idx = reactants[1] - 1
    else:
        return "0.0"

    sf_c = f"{sf} * " if sf != 1.0 else ""

    return (
        f"({sf_c}{kcat_c} * (0.5 * ((y[{s_idx}] - {km_c} - y[{e_idx}]) + "
        f"sqrt((y[{s_idx}] - {km_c} - y[{e_idx}]) * "
        f"(y[{s_idx}] - {km_c} - y[{e_idx}]) + "
        f"4.0 * {km_c} * y[{s_idx}]))) * y[{e_idx}] / "
        f"({km_c} + 0.5 * ((y[{s_idx}] - {km_c} - y[{e_idx}]) + "
        f"sqrt((y[{s_idx}] - {km_c} - y[{e_idx}]) * "
        f"(y[{s_idx}] - {km_c} - y[{e_idx}]) + "
        f"4.0 * {km_c} * y[{s_idx}]))))"
    )


# ─── Sensitivity RHS code generation ────────────────────────────────────


def generate_sens_rhs_c(net_path: str) -> str | None:
    """Generate C code for the CVODES sensitivity RHS callback.

    The sensitivity equation for parameter p_iS is:
        ySdot = J * yS + df/dp_{iS}
    where J = df/dy is the Jacobian and df/dp_{iS} is the partial
    derivative of each species' RHS w.r.t. the iS-th parameter.

    For Elementary reactions v_r = k_r * sf * ∏ x_j^{m_j}:
        df_i/dk_r = S[i][r] * sf * ∏ x_j^{m_j}  (rate without k_r)
        J[i][j]   = S[i][r] * k_r * sf * m_j * x_j^{m_j-1} * ∏_{l≠j} x_l^{m_l}

    For Functional/MM: returns None (fall back to CVODES internal FD).

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

    # Build name→index maps
    param_idx = {name: i for i, (_, name, _, _) in enumerate(params)}
    func_names = {name for _, name, _ in functions}

    # Check: all reactions must be Elementary for analytical sensitivity RHS
    for _, _reactants, _products, rate_law, _ in reactions:
        kind = _classify_rate_law(rate_law, func_names)
        if kind[0] != "elementary":
            return None  # Fall back to CVODES internal FD

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
    for _, name, expr, is_const in params:
        if is_const:
            continue
        jac = _compute_derived_param_jacobian(
            expr, primary_param_names, param_idx, derived_exprs=derived_exprs
        )
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

    return _emit_sens_rhs_body(rxn_data, n_sp, n_params, fixed_sp)


def _emit_sens_rhs_body(
    rxn_data: list[dict],
    n_sp: int,
    n_params: int,
    fixed_sp: set[int],
) -> str:
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

    Both ``generate_sens_rhs_c`` (.net path) and ``generate_sens_from_model``
    (model path) feed this helper, so the emitted C is byte-identical for the
    same normalized input.
    """
    lines: list[str] = []
    _emit = lines.append

    # Tier-1: bngsim_jac_vec is the giant straight-line function here (one block
    # per reaction); chunk it for large models. bngsim_dfdp is a switch (each
    # case is its own small basic block) which compiles fine flat, so it is left
    # alone. Same threshold as the RHS ⇒ a model that chunks one chunks both.
    chunk = _should_chunk(len(rxn_data))
    block_size = _chunk_block_size()

    # ── Header ──────────────────────────────────────────────────────
    _emit("/* Auto-generated CVODES sensitivity RHS - DO NOT EDIT */")
    _emit("/* Analytical sensitivity RHS for Elementary models */")
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
    _emit("/* Compute df/dp_{iP} - partial derivative of RHS w.r.t. parameter iP.")
    _emit("   dfdp_out[i] = sum over reactions r where p_{iP} is the rate constant:")
    _emit("     S[i][r] * sf * product_of_reactant_concs (rate without k_r)")
    _emit("   For derived rate constants (e.g., _rateLaw{N} = chi_X*kon_Y or 5/MEK),")
    _emit("   the chain-rule contributions to each primary parameter are also emitted */")
    _emit("static void bngsim_dfdp(int iP, double t, const double* y,")
    _emit("                        const double* p, double* dfdp_out) {")
    _emit("    memset(dfdp_out, 0, N_SPECIES * sizeof(double));")
    _emit("")

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

    _emit("    double v;")
    _emit("    switch (iP) {")

    def _build_geom_terms(rxn) -> list[str]:
        sf = rxn["stat_factor"]
        terms: list[str] = []
        if sf != 1.0:
            terms.append(str(int(sf)) if sf == int(sf) else str(sf))
        # GH #75: amount_valued reactants enter by their amount (stored × V_c),
        # so the rate carries the constant ∏ V_c^mult. 1.0 ⇒ no term emitted
        # (byte-identical for .net / V=1 / hOSU=false).
        amount_factor = rxn.get("amount_factor", 1.0)
        if amount_factor != 1.0:
            terms.append(repr(amount_factor))
        for sp_idx, mult in sorted(rxn["reactant_mult"].items()):
            for _ in range(mult):
                terms.append(f"y[{sp_idx}]")
        return terms

    for pidx in sorted(rxns_by_param.keys()):
        if pidx < 0:
            continue
        _emit(f"    case {pidx}:")
        for entry in rxns_by_param[pidx]:
            kind = entry[0]
            rxn = entry[1]
            geom = _build_geom_terms(rxn)
            if kind == "direct":
                parts = list(geom) if geom else ["1.0"]
            else:
                # chain rule: rate uses derived param p_d = f(primaries).
                # ∂rate/∂p_iP = (∂p_d/∂p_iP) * sf * ∏y^m
                _, _, dpd_dprimary_c = entry
                parts = [f"({dpd_dprimary_c})", *geom]

            _emit(f"        v = {' * '.join(parts)};")
            for sp_idx, coeff in sorted(rxn["stoich"].items()):
                if coeff == 1:
                    _emit(f"        dfdp_out[{sp_idx}] += v;")
                elif coeff == -1:
                    _emit(f"        dfdp_out[{sp_idx}] -= v;")
                elif coeff > 0:
                    _emit(f"        dfdp_out[{sp_idx}] += {coeff} * v;")
                else:
                    _emit(f"        dfdp_out[{sp_idx}] += ({coeff}) * v;")
        _emit("        break;")

    _emit("    default:")
    _emit("        break;  /* parameter not a rate constant - dfdp = 0 */")
    _emit("    }")
    _emit("")

    # Zero fixed species
    if fixed_sp:
        _emit("    /* Zero fixed species */")
        for si in sorted(fixed_sp):
            _emit(f"    dfdp_out[{si}] = 0.0;")
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
        # w.r.t. the stored value, and rate = k·sf·(∏V_c^m)·∏x^m). 1.0 ⇒ no
        # factor (byte-identical for .net / V=1 / hOSU=false).
        amount_factor = rxn.get("amount_factor", 1.0)

        # For each unique reactant species j:
        for sp_j, m_j in sorted(rmult.items()):
            # dv_r/dx_j = k * af * sf * m_j * x_j^{m_j-1} * ∏_{l≠j} x_l^{m_l}
            parts = [f"p[{pidx}]"]
            if amount_factor != 1.0:
                parts.append(repr(amount_factor))
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

    jacv_block_defs: list[str] = []
    jacv_call_lines: list[str] = []
    jacv_block_protos: list[str] = []
    if chunk:
        jacv_block_defs, jacv_call_lines, jacv_block_protos = _emit_chunked_blocks(
            jacv_groups,
            fn_prefix="jacv_blk",
            signature_params="const double* y, const double* p, const double* v, double* Jv_out",
            call_args="y, p, v, Jv_out",
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
    _emit("   Output: Jv_out[i] = sum_j J[i][j] * v[j] */")
    _emit("static void bngsim_jac_vec(double t, const double* y,")
    _emit("                           const double* p, const double* v,")
    _emit("                           double* Jv_out) {")
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
    _emit("    /* 1. Compute df/dp_{iP} */")
    _emit("    double dfdp[N_SPECIES];")
    _emit("    bngsim_dfdp(iP, t, y, p, dfdp);")
    _emit("")
    _emit("    /* 2. Compute J * yS */")
    _emit("    double Jv[N_SPECIES];")
    _emit("    bngsim_jac_vec(t, y, p, yS, Jv);")
    _emit("")
    _emit("    /* 3. ySdot = J * yS + df/dp */")
    _emit("    for (int i = 0; i < N_SPECIES; ++i) {")
    _emit("        ySdot[i] = Jv[i] + dfdp[i];")
    _emit("    }")
    _emit("")
    _emit("    return 0;")
    _emit("}")

    return "\n".join(lines) + "\n"


def _codegen_emit_flags(model, emit_jac: bool) -> tuple[bool, bool, bool]:
    """``(want_jac, want_outputs, want_output_sens)`` for the .net codegen append,
    from cheap O(1) model flags — never generates source, so a .net cache hit stays
    a few stat()s.

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
        return False, False, False
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
    return want_jac, want_outputs, want_output_sens


def generate_combined_c(
    net_path: str,
    model=None,
    emit_jac: bool = True,
    emit_outputs: bool = True,
    emit_output_sens: bool = False,
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
    sens_code = generate_sens_rhs_c(net_path)
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

    The hash mixes in ``_CODEGEN_VERSION`` so a codegen behavior change
    invalidates previously-cached .so files automatically. Any .tfun data
    files referenced by the .net's function block are also folded in, so
    editing a tfun's y-values triggers a recompile.
    """
    h = hashlib.sha256()
    h.update(_CODEGEN_VERSION.encode())
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
# merged on top with eats_empty_parens=True only for funcs.
_BUILTIN_IDENT_MAP: dict[str, tuple[str, bool]] = {
    "time": ("t", True),
    "t": ("t", True),
    "_pi": ("M_PI", False),
    "_e": ("M_E", False),
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
    """
    lookup: dict[str, tuple[str, bool]] = dict(_BUILTIN_IDENT_MAP)
    for name, rep in param_map.items():
        lookup[name] = (rep, False)
    for name, rep in species_map.items():
        lookup[name] = (rep, False)
    for name, rep in obs_map.items():
        lookup[name] = (rep, False)
    for name, rep in func_map.items():
        lookup[name] = (rep, True)
    # rateOf accessors (GH #106): rate_of__<species> → current_derivs[idx], a
    # plain variable read (not a call). The RHS emitter declares and fills
    # current_derivs via the two-pass probe; non-rateOf models pass nothing.
    if rateof_map:
        for name, rep in rateof_map.items():
            lookup[name] = (rep, False)
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

    # GH #171: per-species LIVE compartment-volume index (ode_live_volume_idx0).
    # A cross-compartment variable-volume reaction divides each affected species's
    # ydot accumulation by the live volume conc[ode_live_volume_idx0] (a real ODE
    # state — the promoted compartment species) rather than the static
    # 1.0/volume_factor. -1 (the default, every static-volume / .net species) ⇒
    # use inv_vf, byte-identical to pre-#171. Case-4 models were declined here
    # before #171 (Part 2); the interpreted RHS handled them.
    species_live = {i: int(s.get("ode_live_volume_idx0", -1)) for i, s in enumerate(species)}

    # GH #75: per-species amount factor (volume_factor for amount_valued
    # species, else 1.0). An amount_valued species participates by its amount
    # (stored × V_c): observables read it as an amount (mirrors
    # NetworkModel::update_observables) and an Elementary rate carries the
    # constant ∏_{amount_valued reactants} V_c^mult (mirrors
    # compute_species_factor / AnalyticalJacobianData::amount_factor). Empty for
    # .net / V=1 / hOSU=false ⇒ byte-identical codegen output.
    av_factor = {
        i: float(s.get("volume_factor", 1.0))
        for i, s in enumerate(species)
        if s.get("amount_valued", False) and float(s.get("volume_factor", 1.0)) != 1.0
    }
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
                ref = f"({_jac_c_float(species[i].get('volume_factor', 1.0))} * {ref})"
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
    inv_vf_terms: list[str] = []
    if any(rxn.get("per_species_volume_scaling", False) for rxn in reactions):
        for s in species:
            vf = s.get("volume_factor", 1.0) or 1.0
            inv_vf_terms.append(repr(1.0 / vf))

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
        # Mirrors compute_species_factor_ode. 1.0 ⇒ no term emitted
        # (byte-identical for .net / V=1 / hOSU=false). Applies to the
        # species-factor product in both the elementary and the
        # apply_species_factor functional branches below.
        amount_factor = 1.0
        for ri in reactants:
            amount_factor *= av_factor.get(ri, 1.0)

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
                    if amount_factor != 1.0:
                        parts.append(repr(amount_factor))
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
            if amount_factor != 1.0:
                parts.append(repr(amount_factor))
            for ri in reactants:
                parts.append(f"y[{ri}]")
            g(f"    rate = {' * '.join(parts)};")

        elif rtype == "mm":
            # tQSSA Michaelis-Menten
            if len(rate_params) >= 2 and len(reactants) >= 2:
                kcat_c = f"p[{rate_params[0]}]"
                km_c = f"p[{rate_params[1]}]"
                e_idx = reactants[0]
                s_idx = reactants[1]
                sf_c = f"{sf} * " if sf != 1.0 else ""
                g(
                    f"    rate = ({sf_c}{kcat_c} * (0.5 * ((y[{s_idx}] - {km_c} - y[{e_idx}]) + "
                    f"sqrt((y[{s_idx}] - {km_c} - y[{e_idx}]) * "
                    f"(y[{s_idx}] - {km_c} - y[{e_idx}]) + "
                    f"4.0 * {km_c} * y[{s_idx}]))) * y[{e_idx}] / "
                    f"({km_c} + 0.5 * ((y[{s_idx}] - {km_c} - y[{e_idx}]) + "
                    f"sqrt((y[{s_idx}] - {km_c} - y[{e_idx}]) * "
                    f"(y[{s_idx}] - {km_c} - y[{e_idx}]) + "
                    f"4.0 * {km_c} * y[{s_idx}]))));"
                )
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
            def _psvs_divide(si: int) -> str:
                L = species_live.get(si, -1)
                if L >= 0:
                    vf = float(species[si].get("volume_factor", 1.0) or 1.0)
                    return f"rate / (y[{L}] > 0.0 ? y[{L}] : {vf!r})"
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
    needs_inv_vf = any("inv_vf[" in ln for grp in rxn_groups for ln in grp)

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
    rhs_obs_in, rhs_obs_fs = _shard_value_lines(
        _emit_observable_lines(observables, av_factor) if observables else [],
        chunk=chunk_obsfunc,
        fn_prefix="rhs_obs_blk",
        signature_params=_OBS_BLK_SIG,
        call_args=_OBS_BLK_ARGS,
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


def _emit_observable_lines(observables: list, av_factor: dict) -> list[str]:
    """C lines computing ``double obs[N]; obs[i] = …`` from species (0-based).

    Shared by the RHS (``generate_rhs_from_model``) and the analytical-Jacobian
    (``generate_jacobian_from_model``) emitters so both reference identical
    observable intermediates — the Functional derivatives the Jacobian emits are
    written in these ``obs[i]`` symbols. GH #75 amount factor is folded into each
    coefficient (1.0 outside the hOSU-V≠1 set ⇒ byte-identical there).
    """
    lines = [f"    double obs[{len(observables)}];"]
    for i, o in enumerate(observables):
        entries = o["entries"]
        if not entries:
            lines.append(f"    obs[{i}] = 0.0;  /* {o['name']} */")
            continue
        terms = []
        for sp_idx, factor in entries:
            coef = factor * av_factor.get(sp_idx, 1.0)
            if coef == 1.0:
                terms.append(f"y[{sp_idx}]")
            elif coef == int(coef):
                terms.append(f"{int(coef)}*y[{sp_idx}]")
            else:
                terms.append(f"{coef}*y[{sp_idx}]")
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


def _jac_vpow(s: int, av_factor: dict) -> str:
    """C expression for a reactant's amount value ``y[s]`` (× V_s when the
    species is amount_valued with V≠1)."""
    return f"({_jac_c_float(av_factor[s])}*y[{s}])" if s in av_factor else f"y[{s}]"


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
    from collections import Counter

    # Escape hatch: force the interpreted analytical / FD Jacobian by declining to
    # emit the compiled one (A/B the feature; mirrors
    # BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0 for the interpreted path).
    if os.environ.get("BNGSIM_NO_CODEGEN_JAC") == "1":
        return None

    # GH #151: per-species / per-observable C-derivative emission routes through
    # native-first helpers (saturable family, no SymPy) with a SymPy fallback.
    from bngsim._jacobian import (
        _TIME_SYM,
        build_per_species_c,
        differentiate_rate_law_c,
    )

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
    # Elementary/MM carry their CSC indices in the plan (affected_csc); Functional
    # terms resolve (col, row) -> csc lazily through the CSC structure below.
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
        _csc_miss = False

        def _csc_of(col: int, row: int) -> int:
            """CSC data index of (row, col); -1 (and a decline flag) on a miss.

            Columns are indexed lazily — only the columns Functional terms touch —
            so a genome-scale matrix never materializes its full nnz map. A miss
            means the Python reconstruction disagrees with the CSC pattern; the
            caller then declines rather than ship a partial/wrong Jacobian.
            """
            nonlocal _csc_miss
            m = _col_row_to_csc.get(col)
            if m is None:
                if col < 0 or col + 1 >= len(col_ptrs):
                    _csc_miss = True
                    return -1
                lo = int(col_ptrs[col])
                hi = int(col_ptrs[col + 1])
                m = {r: lo + off for off, r in enumerate(row_indices[lo:hi].tolist())}
                _col_row_to_csc[col] = m
            csc = m.get(int(row))
            if csc is None:
                _csc_miss = True
                return -1
            return csc

        def _lv(col: int, row: int, csc) -> str:
            return f"jac_data[{csc}]"
    else:

        def _lv(col: int, row: int, csc) -> str:
            return f"jac[{col}*N_SPECIES + {row}]"

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
    func_idx_by_name = {name: i for i, name in enumerate(func_names)}

    # GH #75 amount factor (mirrors generate_rhs_from_model).
    av_factor = {
        i: float(s.get("volume_factor", 1.0))
        for i, s in enumerate(species)
        if s.get("amount_valued", False) and float(s.get("volume_factor", 1.0)) != 1.0
    }
    species_vol = {i: (float(s.get("volume_factor", 1.0)) or 1.0) for i, s in enumerate(species)}
    # GH #171: per-species live compartment-volume index (mirrors generate_rhs).
    species_live = {i: int(s.get("ode_live_volume_idx0", -1)) for i, s in enumerate(species)}

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

    # ── Symbol resolver for sympy_to_c ──────────────────────────────────────
    # Map each free-symbol name in a derivative (observable / constant param /
    # time placeholder) to the C intermediate the Jacobian function computes.
    # Keyword-named identifiers were aliased by the symbolic core, so key the
    # map by the aliased name. Observables are registered last so they win on a
    # name collision (matching the interpreted ExprTk variable binding).
    def _alias(n: str) -> str:
        return _alias_keyword_param(n) if n in _PY_KEYWORD_PARAM_NAMES else n

    c_ref: dict[str, str] = {}
    for i, name in enumerate(param_names):
        c_ref[_alias(name)] = f"p[{i}]"
    for i, name in enumerate(obs_names):
        c_ref[_alias(name)] = f"obs[{i}]"

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
    constants = set(ctx["constant_names"])
    ctx_obs_names = set(obs_groups)

    def _net_affected(rxn) -> list[tuple[int, float, int, float]]:
        """Affected rows as (i, coeff, live_idx, static_divisor) — mirrors
        set_functional_jacobian (GH #171). coeff = stat_factor·net_stoich is
        UNFOLDED: the volume divide is deferred to the scatter so a variable-volume
        row uses the live volume conc[live_idx]. A static-volume / non-varvol row
        has live_idx = -1 and divides by static_divisor (volume_factor for a
        per_species_volume_scaling row, else 1.0 — folded at emit, byte-identical
        to the pre-#171 output)."""
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
            live_idx = species_live.get(i, -1) if psvs else -1
            static_divisor = species_vol.get(i, 1.0) if psvs else 1.0
            out.append((i, stat * c_i, live_idx, static_divisor))
        return out

    # Reconstruct the Functional contributions. Build them first so an
    # un-emittable derivative short-circuits to None before any C is assembled.
    # Each contribution is a balanced ``{ … }`` group so it can be spliced inline
    # (flat) or wrapped whole in a NOINLINE shard block (chunked, GH #165).
    per_species_groups: list[list[str]] = []
    per_species_volume_groups: list[list[str]] = []
    per_observable_groups: list[list[str]] = []

    # Scatter one existing-column contribution coeff·dj into (row i, col sp_j),
    # applying the GH #171 volume divide. A static-volume / non-varvol row folds
    # coeff/static_divisor at emit (static_divisor=1.0 for non-varvol ⇒ the
    # unchanged coeff, byte-identical to pre-#171). A live-volume row defers to a
    # runtime divide by conc[live_idx] (fallback static_divisor when ≤0),
    # mirroring fill_*_analytical_jacobian's `a.coeff * dv / divisor`.
    def _scatter_existing(sp_j, i, coeff, live_idx, sdiv, rhs):
        csc = _csc_of(sp_j, i) if is_sparse else None
        lv = _lv(sp_j, i, csc)
        if live_idx >= 0:
            div = f"(y[{live_idx}] > 0.0 ? y[{live_idx}] : {sdiv!r})"
            return f"        {lv} += {_jac_c_float(coeff)} * {rhs} / {div};"
        return f"        {lv} += {_jac_c_float(coeff / sdiv)} * {rhs};"

    # Per-species (SBML) block — emitted for all reactions first so the scatter
    # accumulation order matches fill_dense_analytical_jacobian (all species_
    # terms, then all volume_terms, then all observable_terms).
    for rxn in frxns:
        has_sf = bool(rxn["apply_species_factor"]) and len(rxn["reactant_idx0"]) > 0
        if has_sf:
            continue
        affected = _net_affected(rxn)
        terms = build_per_species_c(
            rxn["rate_expr"], func_map, obs_groups, species_meta, constants, resolve_symbol
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
            for i, coeff, live_idx, sdiv in affected:
                grp.append(_scatter_existing(sp_j, i, coeff, live_idx, sdiv, "dj"))
            grp.append("    }")
            per_species_groups.append(grp)

    # GH #171: the new ∂/∂V_live column for cross-compartment variable-volume
    # reactions — −(stat·netstoichᵢ·func)/V_live² at (row i, col live_idx). Built
    # once per per_species_volume_scaling reaction from the affected rows directly
    # (independent of the ∂func derivatives: the bare-law k·A·B has no ∂func/∂V_live
    # term, so the whole column is this contribution). Mirrors the volume_terms
    # scatter in fill_*_analytical_jacobian; func is func[fidx] (= the reaction's
    # bound rate parameter). A live (row, col) missing from the CSC pattern means
    # the reconstruction disagrees with the C++ sparsity → decline (via _csc_miss).
    for rxn in frxns:
        if not bool(rxn["per_species_volume_scaling"]):
            continue
        live_rows = [
            (i, coeff, live_idx) for i, coeff, live_idx, _sd in _net_affected(rxn) if live_idx >= 0
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
            csc = _csc_of(live_idx, i) if is_sparse else None
            lv = _lv(live_idx, i, csc)
            grp.append(
                f"        if (y[{live_idx}] > 0.0) {lv} += "
                f"{_jac_c_float(-coeff)} * fv / (y[{live_idx}] * y[{live_idx}]);"
            )
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
            rxn["rate_expr"], func_map, ctx_obs_names, constants, resolve_symbol
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
        cols: dict[int, dict] = {}
        for k, (obs_name, _se) in enumerate(observable_k):
            for sp_j, factor in obs_groups[obs_name]:
                gcoef = factor * av_factor.get(sp_j, 1.0)
                if gcoef == 0.0:
                    continue
                cols.setdefault(sp_j, {"a_terms": [], "is_reactant": False, "mult_j": 0})[
                    "a_terms"
                ].append((k, gcoef))
        for s, m in rmult.items():
            col = cols.setdefault(s, {"a_terms": [], "is_reactant": False, "mult_j": 0})
            col["is_reactant"] = True
            col["mult_j"] = m

        grp = [
            f"    {{ /* per-observable rxn {rxn_idx} func {fname} */",
            f"        double f = func[{fidx}];",
        ]
        p_parts = [_jac_vpow(s, av_factor) for s, m in rmult.items() for _ in range(m)]
        grp.append(f"        double P = {' * '.join(p_parts) if p_parts else '1.0'};")
        for k, c in enumerate(d_c):
            grp.append(f"        double d{k} = {c};")
        for sp_j, col in cols.items():
            aj = " + ".join(f"{_jac_c_float(g)}*d{k}" for (k, g) in col["a_terms"])
            grp.append("        {")
            grp.append(f"            double val = ({aj if aj else '0.0'}) * P;")
            if col["is_reactant"]:
                dp_parts = [_jac_c_float(col["mult_j"])]
                for s, m in rmult.items():
                    for _ in range(m - 1 if s == sp_j else m):
                        dp_parts.append(_jac_vpow(s, av_factor))
                if sp_j in av_factor:
                    dp_parts.append(_jac_c_float(av_factor[sp_j]))
                grp.append(f"            val += f * ({' * '.join(dp_parts)});")
            # Per-observable (.net) reactions are never per_species_volume_scaling,
            # so live_idx is always -1 and static_divisor 1.0 — coeff is the
            # unfolded stat·net_stoich (byte-identical to the pre-#171 folded value).
            for i, coeff, _live_idx, _sdiv in affected:
                csc = _csc_of(sp_j, i) if is_sparse else None
                grp.append(f"            {_lv(sp_j, i, csc)} += {_jac_c_float(coeff)} * val;")
            grp.append("        }")
        grp.append("    }")
        per_observable_groups.append(grp)

    # A Functional (col, row) that is not in the CSC pattern means the Python
    # reconstruction and the C++ sparsity disagree — decline rather than emit a
    # Jacobian missing those entries (the interpreted sparse path stays correct).
    if is_sparse and _csc_miss:
        return None

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
        grp = []
        g = grp.append
        ksf_parts = [f"p[{int(erxn['rate_param_idx0'])}]"]
        sf = float(erxn["stat_factor"])
        af = float(erxn["amount_factor"])
        if sf != 1.0:
            ksf_parts.append(_jac_c_float(sf))
        if af != 1.0:
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

    mm_groups: list[list[str]] = []
    for mt in plan["mm"]:
        e = int(mt["e_idx"])
        s = int(mt["s_idx"])
        grp = []
        g = grp.append
        g("    {")
        g(
            f"        double kcat = p[{int(mt['kcat_param_idx0'])}], "
            f"Km = p[{int(mt['km_param_idx0'])}];"
        )
        g(f"        double E = y[{e}], S = y[{s}];")
        g("        double delta = S - Km - E;")
        g("        double Dmm = sqrt(delta*delta + 4.0*Km*S);")
        g("        double sFree = 0.5*(delta + Dmm);")
        g("        double dE = 0.0, dS = 0.0;")
        g("        if (sFree > 0.0 && Dmm > 0.0) {")
        g("            double dsF_dE = 0.5*(-1.0 - delta/Dmm);")
        g("            double dsF_dS = 0.5*(1.0 + (delta + 2.0*Km)/Dmm);")
        g(f"            double Cmm = kcat * {_jac_c_float(mt['stat_factor'])};")
        g("            double KpsF = Km + sFree;")
        g("            double common = Cmm*E*Km/(KpsF*KpsF);")
        g("            dE = Cmm*sFree/KpsF + common*dsF_dE;")
        g("            dS = common*dsF_dS;")
        g("        }")
        if mt["e_affected"]:
            g("        if (dE != 0.0) {")
            e_csc = mt.get("e_affected_csc")
            for ai, (row_i, coeff) in enumerate(mt["e_affected"]):
                csc = int(e_csc[ai][0]) if is_sparse else None
                g(f"            {_lv(e, int(row_i), csc)} += {_jac_c_float(coeff)} * dE;")
            g("        }")
        if mt["s_affected"]:
            g("        if (dS != 0.0) {")
            s_csc = mt.get("s_affected_csc")
            for ai, (row_i, coeff) in enumerate(mt["s_affected"]):
                csc = int(s_csc[ai][0]) if is_sparse else None
                g(f"            {_lv(s, int(row_i), csc)} += {_jac_c_float(coeff)} * dS;")
            g("        }")
        g("    }")
        mm_groups.append(grp)

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
    jac_obs_in, jac_obs_fs = (
        _shard_value_lines(
            _emit_observable_lines(observables, av_factor),
            chunk=chunk,
            fn_prefix="jac_obs_blk",
            signature_params=_OBS_BLK_SIG,
            call_args=_OBS_BLK_ARGS,
        )
        if (has_functional and observables)
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

    _emit("    return 0;")
    _emit("}")
    return "\n".join(lines) + "\n"


def generate_sens_from_model(model) -> str | None:
    """Generate C source for the CVODES analytical sensitivity RHS from a
    built model, parallel to ``generate_sens_rhs_c`` (.net path).

    Returns ``None`` if any reaction is non-Elementary — the caller then
    falls back to RHS-only codegen and CVODES uses internal FD.

    Closes #15: parameters whose ``is_const`` field is False (derived
    expressions like ``_rateLaw_<rid>`` synthesized by the SBML loader for
    products of constant SBML parameters, or arbitrary BNGL
    ``# ConstantExpression`` rate constants surfaced via ``add_parameter(...,
    is_expression=True)``) get chain-rule expansion via sympy. Each derived
    rate constant ``p_d = expr(primary_1, primary_2, ...)`` contributes
    ``(∂p_d/∂primary_k) * sf * ∏y^m`` to the sensitivity of every primary
    parameter that appears in ``expr``, exactly mirroring the .net path's
    ``derived_expansion`` machinery.
    """
    core = model._core if hasattr(model, "_core") else model
    data = core.codegen_data()

    params = data["parameters"]
    species = data["species"]
    reactions = data["reactions"]

    n_sp = len(species)
    n_params = len(params)

    # Bail if any reaction is non-Elementary — analytical sens RHS is only
    # defined for k * sf * ∏y^m kinetics. Same constraint as
    # generate_sens_rhs_c (line 762-765) for the .net path.
    for rxn in reactions:
        if rxn["type"] != "elementary":
            return None

    fixed_sp = {i for i, s in enumerate(species) if s["fixed"]}

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
    derived_expansion: dict[str, dict[str, str]] = {}
    for p in params:
        if p.get("is_const", True):
            continue
        expr = p.get("expression", "")
        if not expr:
            continue
        jac = _compute_derived_param_jacobian(
            expr, primary_param_names, param_idx_by_name, derived_exprs=derived_exprs
        )
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
    av_factor = {
        i: float(s.get("volume_factor", 1.0))
        for i, s in enumerate(species)
        if s.get("amount_valued", False) and float(s.get("volume_factor", 1.0)) != 1.0
    }

    rxn_data: list[dict] = []
    for rxn in reactions:
        rate_params = list(rxn["rate_param_indices"])
        pidx = rate_params[0] if rate_params else -1
        sf = rxn["stat_factor"]
        reactants = list(rxn["reactants"])
        products = list(rxn["products"])

        stoich: dict[int, int] = {}
        for ri in reactants:
            stoich[ri] = stoich.get(ri, 0) - 1
        for pi in products:
            stoich[pi] = stoich.get(pi, 0) + 1

        rmult = Counter(reactants)

        amount_factor = 1.0
        for ri in reactants:
            amount_factor *= av_factor.get(ri, 1.0)

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

        rxn_data.append(
            {
                "param_idx": pidx,
                "stat_factor": sf,
                "stoich": stoich,
                "reactant_mult": dict(rmult),
                "derived_terms": derived_terms,
                "amount_factor": amount_factor,
            }
        )

    return _emit_sens_rhs_body(rxn_data, n_sp, n_params, fixed_sp)


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
    av_factor = {
        i: float(s.get("volume_factor", 1.0))
        for i, s in enumerate(species)
        if s.get("amount_valued", False) and float(s.get("volume_factor", 1.0)) != 1.0
    }

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
    out_obs_in, out_obs_fs = (
        _shard_value_lines(
            _emit_observable_lines(observables, av_factor),
            chunk=chunk,
            fn_prefix="out_obs_blk",
            signature_params=_OBS_BLK_SIG,
            call_args=_OBS_BLK_ARGS,
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


def _analyze_output_sens(model) -> dict:
    """Per-function output-sensitivity analysis (GH #198).

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
    :func:`_compute_derived_param_jacobian`); a function referencing a derived
    param whose Jacobian could not be derived is marked unsupported (#198 fails
    loudly rather than silently dropping the term).

    Returns a dict with ``decline`` (non-None ⇒ the whole codegen is declined,
    mirroring :func:`generate_outputs_from_model`'s rateOf / embedded-tfun /
    no-function declines), ``func_infos`` (per-function ``{name, supported,
    reason, partials}`` in declaration order), and the emitter context.
    """
    from bngsim._jacobian import differentiate_expression_output_partials

    core = model._core if hasattr(model, "_core") else model
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
    derived_expansion: dict[str, dict[str, str]] = {}
    for p in params:
        if p.get("is_const", True):
            continue
        expr = p.get("expression", "")
        if not expr:
            continue
        jac = _compute_derived_param_jacobian(
            expr, primary_param_names, param_idx_by_name, derived_exprs=derived_exprs
        )
        if jac is not None:
            derived_expansion[p["name"]] = jac

    # Only USER functions are selectable — the auto-generated _rateLawN rate-law
    # intermediates are filtered out of the result block and never addressed by a
    # selector. A genome-scale model can carry thousands of them and running sympy
    # on each would dominate codegen, so differentiate only user functions and
    # what they transitively reference, pruned here via a cheap regex reference
    # graph before any sympy. Functions outside that closure are recorded with a
    # zero placeholder (they are filtered out of the user-facing block anyway).
    name_to_idx = {f["name"]: i for i, f in enumerate(functions)}
    refs: list[set[int]] = [set() for _ in range(n_func)]
    for i, f in enumerate(functions):
        for tok in re.findall(r"[A-Za-z_]\w*", f["expression"]):
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

    # Per-function differentiation. status ∈ {"ok", "unsupported", "skipped"}:
    # "ok" emits the chain rule; "unsupported" emits a NaN sentinel and a reason
    # the Result raises at selection time; "skipped" (outside the user-function
    # closure) is filtered out of the block and left at the caller's zero.
    func_infos: list[dict] = []
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
        partials, reason = differentiate_expression_output_partials(
            f["expression"],
            species_cref=species_map,
            observable_cref=obs_map,
            param_cref=param_map,
            function_cref=func_map,
        )
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


def _emit_obs_sens_lines(observables: list, av_factor: dict, ss: str, out: str) -> list[str]:
    """C lines computing ``out[j] = Σ_i c_ji·ss[i]`` — the linear observable
    output sensitivity for one sensitivity column whose species derivatives are
    in ``ss`` (``dx_i/dθ``). The coefficient ``c_ji`` folds the GroupEntry factor
    and the GH #75 amount-volume scaling identically to ``_emit_observable_lines``
    and the #197 C++ runtime path, so observable and expression sensitivities
    stay consistent."""
    lines = []
    for j, o in enumerate(observables):
        entries = o["entries"]
        if not entries:
            lines.append(f"        {out}[{j}] = 0.0;")
            continue
        terms = []
        for sp_idx, factor in entries:
            coef = factor * av_factor.get(sp_idx, 1.0)
            if coef == 1.0:
                terms.append(f"{ss}[{sp_idx}]")
            elif coef == int(coef):
                terms.append(f"{int(coef)}*{ss}[{sp_idx}]")
            else:
                terms.append(f"{coef}*{ss}[{sp_idx}]")
        lines.append(f"        {out}[{j}] = {' + '.join(terms)};")
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

    av_factor = {
        i: float(s.get("volume_factor", 1.0))
        for i, s in enumerate(species)
        if s.get("amount_valued", False) and float(s.get("volume_factor", 1.0)) != 1.0
    }

    # Value recomputation (same emitters as the RHS / value codegen).
    obs_value_lines = _emit_observable_lines(observables, av_factor) if observables else []
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
        for ln in _emit_obs_sens_lines(observables, av_factor, ss="ss", out="obs_sens_c"):
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
    """
    rhs_code = generate_rhs_from_model(model)
    sens_code = generate_sens_from_model(model)
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


def compute_model_codegen_hash(model) -> str:
    """Compute a SHA-256 hash of model structure for codegen caching.

    The hash is based on the generated C source code (RHS + sensitivity
    RHS when available), so models with identical structure share the same
    compiled .so and a change in either generator invalidates the cache.
    """
    c_source, _ = generate_combined_from_model(model)
    return hashlib.sha256(c_source.encode()).hexdigest()[:16]


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
        want_jac, want_outputs, want_output_sens = _codegen_emit_flags(model, emit_jac)
        c_source, _ = generate_combined_c(
            net_path,
            model,
            emit_jac=want_jac,
            emit_outputs=want_outputs,
            emit_output_sens=want_output_sens,
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
    h.update(_CODEGEN_VERSION.encode())
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
# no-memo behavior. _CODEGEN_VERSION is part of the validity test so a codegen
# behavior bump invalidates stale memo entries too.
# Keyed by (net_abspath, want_jac, want_outputs, want_output_sens): the compiled
# Jacobian (GH #162), output evaluator (GH #163), and expression output-sensitivity
# evaluator (GH #198) are independent content-distinct callbacks, so all three flags
# are part of the key — an entry for one combination must never satisfy another.
_PREPARE_CODEGEN_MEMO: dict[
    tuple[str, bool, bool, bool], tuple[Path, tuple[tuple[str, int], ...], str]
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
    df/dp + J*v). For Functional/MM models, only the RHS is generated
    and CVODES uses internal FD for sensitivity.

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
        want_jac, want_outputs, want_output_sens = _codegen_emit_flags(model, emit_jac)
        memo_key = (net_key, want_jac, want_outputs, want_output_sens)

        # Fast path (T2): an unchanged .net (and its .tfun deps) resolves to the
        # already-cached .so via a few stat() calls, skipping the re-read +
        # re-parse + SHA-256 the cold path below performs.
        with _PREPARE_CODEGEN_MEMO_LOCK:
            entry = _PREPARE_CODEGEN_MEMO.get(memo_key)
        if entry is not None:
            memo_so, dep_stamps, ver = entry
            if (
                ver == _CODEGEN_VERSION
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
                model if (want_jac or want_outputs or want_output_sens) else None,
                emit_jac=want_jac,
                emit_outputs=want_outputs,
                emit_output_sens=want_output_sens,
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
                _CODEGEN_VERSION,
            )
        return so_path
    finally:
        _record_codegen_sec(None, time.perf_counter() - t0, cache_hit)
