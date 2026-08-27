"""Shared fixtures for bngsim Python tests."""

from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Test data lives in bngsim/tests/data/ — resolve via env var or relative to this file.
_DATA_DIR_ENV = os.environ.get("BNGSIM_TEST_DATA")


# ─── Test-owned artifact caches (issue #372) ──────────────────────────────────
#
# bngsim keeps two content-addressed caches under the user's home: compiled .so
# artifacts in ``~/.cache/bngsim/codegen`` and BNG2.pl-generated networks in
# ``~/.cache/bngsim/networks``. Both are resolved once at import from an env var,
# and until this hook existed nothing redirected either one at session scope — so
# every test that built a `codegen=True` Simulator or loaded a `.bngl` wrote into
# the developer's real cache and left it there. Two suite runs in one afternoon
# accounted for 303 live artifacts; the orphan pile behind them was 146 MB.
#
# Three things were wrong with that, in increasing order of severity:
#
#   * It accumulates, on laptops and on any CI runner with a persistent home.
#     `bngsim-cache prune --orphaned` cleans it up, but nothing should be filling
#     a user-facing cache as a side effect of running tests.
#   * It puts FABRICATED keys in a real directory. A test that stands in for
#     another install by monkeypatching ``_CODEGEN_CACHE_KEY`` leaves an artifact
#     under a key no install has ever had, and since #363 put the key in the
#     filename that artifact shows up as a row in somebody's `bngsim-cache info`.
#   * It couples runs. A test meaning to exercise a cold compile can be handed a
#     .so that a previous run — or another test — compiled under the same key.
#
# The redirect points both caches at a directory the SUITE owns instead. It is
# persistent rather than a per-run temp dir on purpose: artifacts stay warm
# across runs (a cold full suite measured 19m19s against roughly 14m warm), which
# is the whole reason a shared cache was tempting in the first place. What is
# given up — a guaranteed cold start every run — is available on demand, because
# the default location lives under ``.pytest_cache/d/`` and ``pytest
# --cache-clear`` wipes it.
#
# Individual tests still monkeypatch ``CACHE_DIR`` to a ``tmp_path`` when they
# need true isolation (a cold compile, a cache they can count entries in), and
# that keeps working: a function-scoped patch overrides the session value and
# restores it. This hook is the floor, not a replacement for those.

#: One knob for both caches: point it at scratch, at a CI-restored warm cache, or
#: at a throwaway directory for a guaranteed cold run. It overrides
#: ``BNGSIM_CODEGEN_CACHE_DIR`` / ``BNGSIM_BNGL_CACHE_DIR`` for the duration of a
#: pytest session rather than deferring to them — a developer who exports those
#: is pointing bngsim's real cache somewhere, and the suite has no more business
#: writing there than in ``~/.cache``.
_TEST_CACHE_ROOT_ENV = "BNGSIM_TEST_CACHE_DIR"

#: (env var read at import, module holding the resolved CACHE_DIR, subdirectory).
#: Both halves are set: the env var so subprocesses that import bngsim inherit the
#: redirect, the module attribute so it still holds if something imported bngsim
#: before this hook ran.
_REDIRECTED_CACHES: tuple[tuple[str, str, str], ...] = (
    ("BNGSIM_CODEGEN_CACHE_DIR", "bngsim._codegen", "codegen"),
    ("BNGSIM_BNGL_CACHE_DIR", "bngsim._bngl_loader", "networks"),
)

#: Set only when the cache root is a temp directory this session created, in
#: which case teardown removes it. A ``.pytest_cache`` root is left alone.
_EPHEMERAL_CACHE_ROOT: Path | None = None


def _test_cache_root(config: pytest.Config) -> tuple[Path, bool]:
    """Resolve where the suite's artifact caches live.

    Returns the directory and whether this session created it as a throwaway —
    kept a pure function of ``config`` + environment (the caller records the
    throwaway) so both branches can be exercised against a stub config without
    reaching into module state.
    """
    env = os.environ.get(_TEST_CACHE_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser(), False
    # ``getattr``, not ``config.cache``: disabling the plugin does not leave the
    # attribute None, it leaves the Config without one at all (the cacheprovider
    # is what sets it), so the direct read is an AttributeError on every CI leg.
    cache = getattr(config, "cache", None)
    if cache is not None:
        # Under ``.pytest_cache/d/``, which is what makes ``pytest --cache-clear``
        # the one-flag way to force the cold run this persistence gives up.
        return cache.mkdir("bngsim"), False
    # ``-p no:cacheprovider``, which every CI leg passes. That is an instruction
    # not to write under the rootdir, so take a per-run temp dir and remove it at
    # the end — a cold run, which is what a fresh runner gets anyway.
    return Path(tempfile.mkdtemp(prefix="bngsim-test-cache-")), True


def _redirect_artifact_caches(config: pytest.Config) -> Path:
    """Point bngsim's compiled-artifact and network caches at a test-owned dir."""
    global _EPHEMERAL_CACHE_ROOT
    root, ephemeral = _test_cache_root(config)
    if ephemeral:
        _EPHEMERAL_CACHE_ROOT = root
    for env_var, module_name, subdir in _REDIRECTED_CACHES:
        target = root / subdir
        target.mkdir(parents=True, exist_ok=True)
        os.environ[env_var] = str(target)
        # Never let the redirect itself break collection: a bngsim that cannot be
        # imported is the preflight's story to tell, in its own words.
        with contextlib.suppress(Exception):
            importlib.import_module(module_name).CACHE_DIR = target
    return root


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove the cache root only when this session created a temp one."""
    if _EPHEMERAL_CACHE_ROOT is not None:
        shutil.rmtree(_EPHEMERAL_CACHE_ROOT, ignore_errors=True)


def pytest_configure(config: pytest.Config) -> None:
    """Cache redirect (issue #372), then the stale-binary preflight (issue #125).

    The redirect goes first because it has to land before the first bngsim import
    — ``CACHE_DIR`` is resolved once, at import, from the env vars it sets. See the
    block above.

    The whole point of the test suite is to make true statements about the
    code. But the editable install loads a separately-built _bngsim_core that
    does not auto-rebuild (#23), so a forgotten rebuild means the suite reports
    on OLD C++ — a green run that gets committed as a correctness verdict about
    code that isn't actually running (this is what burned GH #118). Print the
    loaded binary's identity, and refuse to run against a demonstrably stale
    one. Escape hatches: BNGSIM_ALLOW_STALE_CORE=1 (proceed with a warning) or
    BNGSIM_NO_BUILD_CHECK=1 (skip the check entirely).
    """
    root = _redirect_artifact_caches(config)
    # Named on every run, beside the binary identity line: a developer who wonders
    # why their ~/.cache stopped growing gets the answer without reading conftest.
    print(
        f"[bngsim] artifact caches: {root} (test-owned; ${_TEST_CACHE_ROOT_ENV} relocates)",
        file=sys.stderr,
        flush=True,
    )

    try:
        from bngsim import _build_provenance as bp
    except Exception:
        return  # never let the guard itself break collection
    prov = bp.gather()
    print(bp.identity_line(prov), file=sys.stderr, flush=True)
    report = bp.blocking_report(prov)
    if report:
        raise pytest.UsageError("\n" + report)


# ─── Skip audit ────────────────────────────────────────────────────────────────
#
# Sibling in spirit to the stale-binary preflight above: that guard stops the
# suite reporting on code that isn't running, this one stops it reporting on
# tests that aren't running. A skipped test and a passing test are the same
# character in the summary line, so a permanently-dead test is invisible.
#
# That is not hypothetical. test_sbml_reversible_split.py::test_copasi_abc_xml_loads
# resolved its fixture one directory level too high and had therefore NEVER run,
# on any machine, since it was written — the first draft of this audit is what
# surfaced it. Separately, three whole classes skipped on every CI leg we have
# because they imported PyBNF (lanl/bngsim#45); one had been red for months on the
# only boxes that could run it.
#
# The audience used to be the pre-push hook alone: every workflow pytest call was
# a curated file list, so nothing in CI ran the full suite and only a local push
# saw the whole table. Since issue #169 that is no longer true — python-tests.yml
# runs `python/tests` as a directory on ubuntu + macos-14 and sets
# BNGSIM_SKIP_AUDIT=strict, so an undeclared skip fails CI rather than printing a
# `??` a developer may not read. The hook still prints the table on every push;
# the difference is that the audit now has teeth on two platforms nobody is
# standing at.
#
# Undeclared reasons warn by default. Set BNGSIM_SKIP_AUDIT=strict to make them
# fail instead; BNGSIM_SKIP_AUDIT=off silences the block entirely. Strict stays
# opt-in rather than becoming the default because a developer box legitimately
# lacks things CI has, and a guard that cries wolf gets disabled — which would
# leave us worse off than a quiet one.
#
# Strict is now on for three jobs, not one. python-tests.yml runs everything in
# the default build; mir.yml and windows-tail.yml run curated lists in KLU-off
# builds. Curated was the original argument against turning it on — a curated leg
# skips a different subset than a whole-suite run — but that argument was about
# the TABLE, and strict does not fire on the table's size. It fires per reason,
# and issue #179 is what made per-reason safe: the reasons those legs emit are
# the build-variant ones, and they had drifted to 25 undeclared precisely because
# no default-build run can produce them. A KLU-off leg was the only place they
# could ever be checked, so it is the place strict earns the most.

# ─── Tiers ────────────────────────────────────────────────────────────────────
#
# One flat list could not express the difference issue #179 turned up, and the
# difference matters: "this build does not have KLU" and "this machine has no C
# compiler" are both legitimate skips locally, but only the first is legitimate
# in CI. A CI leg that silently lost `cc` would skip ~22 files' worth of codegen
# tests and report success — a false green of exactly the kind this audit exists
# to catch, waved through by the very list that is supposed to catch it.
#
# ANYWHERE   the skip is a statement about a deliberate configuration — a build
#            variant, an optional extra, a corpus that is not vendored, a
#            platform that genuinely lacks the feature. Fine everywhere.
# LOCAL_ONLY the skip is a statement about the ENVIRONMENT being incomplete. A
#            developer laptop is allowed to be incomplete; a CI leg is not, and
#            one that becomes incomplete has broken rather than adapted.
#
# Strict mode is the CI signal — nothing but a workflow sets BNGSIM_SKIP_AUDIT,
# so LOCAL_ONLY is enforced exactly where "this environment is incomplete" stops
# being an acceptable answer, and stays a quiet table row on a laptop. The
# empirical check that this does not just add red: none of the LOCAL_ONLY reasons
# below fires on any leg today. GitHub's windows-latest ships MinGW gcc, so even
# the compiler gates have never skipped there (0 across all four mir.yml legs).
_ANYWHERE = "anywhere"
_LOCAL_ONLY = "local-only"

# Declared skip reasons: (substring to match, tier, why this skip is legitimate).
# A skip whose reason matches none of these is reported as undeclared. Adding an
# entry is the point — it forces a new permanent skip to be justified in a diff
# rather than blending into the summary count.
_DECLARED_SKIPS: tuple[tuple[str, str, str], ...] = (
    # Build-configuration variants — the feature is genuinely absent from this build.
    ("without the MIR backend", _ANYWHERE, "MIR JIT is off unless -DBNGSIM_ENABLE_MIR=ON"),
    (
        "selected with BNGSIM_CODEGEN_JIT=mir",
        _ANYWHERE,
        "issue #413 lives in MIR's register allocator, so its reproducer needs the "
        "backend BUILT and SELECTED; mir.yml sets both and runs it there",
    ),
    ("KLU not compiled", _ANYWHERE, "KLU-off builds are a supported configuration"),
    (
        "KLU sparse solver not built",
        _ANYWHERE,
        "as above; the phrasing test_codegen_jacobian_sparse.py uses",
    ),
    (
        "requires a build without SuiteSparse/KLU",
        _ANYWHERE,
        "inverse of the above; KLU-off builds only",
    ),
    ("build has no SuiteSparse/KLU", _ANYWHERE, "as above; the other half of the same gate"),
    (
        "LAPACK-dense not built",
        _ANYWHERE,
        "LAPACK is optional; CMake degrades to the reference solver",
    ),
    # Same build-variant condition as the line above, phrased differently by a
    # different file: test_engine_choice_accessors.py says "LAPACK-dense not built
    # in this configuration", test_lapack_dense_solver.py says "build links no
    # BLAS dense backend". Only the first was ever declared, and the gap is
    # invisible on macOS (Accelerate is always found, so neither test skips) —
    # it surfaces only where find_package(LAPACK) comes up empty, which nothing
    # ran the full suite on until #169 added a Linux leg.
    ("no BLAS dense backend", _ANYWHERE, "as above; the other half of the same gate"),
    (
        "RuleMonkey compiled in",
        _ANYWHERE,
        "inverse-condition test; runs only on RuleMonkey-off builds",
    ),
    ("RuleMonkey not compiled in", _ANYWHERE, "RuleMonkey is a build-time opt-in"),
    # The NFsim half of the pair above. Issue #179 found NINE phrasings for these
    # two conditions across 29 sites ("NFsim not built", "bngsim compiled without
    # NFsim support", "no NFsim support", …) — drift away from a convention that
    # already existed here rather than the absence of one. They are consolidated
    # onto the two strings declared here; keep new gates on the same wording.
    ("NFsim not compiled in", _ANYWHERE, "NFsim is a build-time opt-in"),
    # Optional / developer-only Python dependencies.
    (
        "could not import",
        _ANYWHERE,
        "optional extra (h5py, jax, pandas, sympy, xarray, ...) absent",
    ),
    (
        "JAX not installed",
        _ANYWHERE,
        "hand-written twin of the importorskip above; jax is an extra",
    ),
    (
        "vivarium-core not installed",
        _ANYWHERE,
        "optional extra (bngsim[vivarium]); the process shell is opt-in. Spelled "
        "out rather than left to the generated text because that importorskip "
        "passes an explicit reason=, which suppresses it",
    ),
    ("roadrunner", _ANYWHERE, "DEVELOPER-ONLY reference engine; never a base dependency"),
    ("antimony", _ANYWHERE, "optional extra; loaders fall back to SBML"),
    # `scipy` was here and is gone: no hand-written reason contains it, so every
    # scipy skip is an importorskip-generated "could not import 'scipy'" that the
    # entry above already matches. A pattern that matches nothing is not free —
    # it reads as a fourth optional extra somebody decided about (#179).
    # External tools and corpora that are not vendored.
    ("BNG2.pl", _ANYWHERE, "external perl toolchain, not a bngsim dependency"),
    ("biomodels", _ANYWHERE, "BioModels corpus is fetched, not vendored ($BIOMODELS_SBML_DIR)"),
    ("benchmark", _ANYWHERE, "benchmark corpus lives outside the packaged tree"),
    ("abc.xml not at", _ANYWHERE, "fixture in a sibling PyBNF checkout; dev-only"),
    ("no .ant fixture available", _ANYWHERE, "antimony fixture is optional in this checkout"),
    # Platform, as opposed to environment: the feature is absent because of what
    # the OS/toolchain IS, not because this box is missing something.
    (
        "POSIX-specific",
        _ANYWHERE,
        "no Windows equivalent: process-group reaping, and the /bin/sh fake "
        "interpreters the pybind11-resolution probe walks (GH #288)",
    ),
    ("tomllib is 3.11+", _ANYWHERE, "stdlib module absent on 3.10, which is still supported"),
    ("gcc/clang only", _ANYWHERE, "the sharded compile path; MSVC takes the other branch"),
    # Source-tree vs installed-wheel context.
    ("installed wheel", _ANYWHERE, "source-tree-only guard, correctly inert against a wheel"),
    ("source root", _ANYWHERE, "version-consistency check needs the source tree"),
    (
        "no committed stub at",
        _ANYWHERE,
        "test_committed_stub_has_no_version_literal reads python/bngsim/"
        "_bngsim_core.pyi out of the source tree; a wheel/subtree checkout has no "
        "committed stub to inspect. Same source-tree-only class as the two above",
    ),
    (
        "CMake",
        _ANYWHERE,
        "CMakeCache cross-checks need a configured build dir; the "
        "pybind11-resolution probe (GH #288) needs a cmake binary, which a leg "
        "testing an installed wheel has no reason to carry",
    ),
    ("not in this checkout", _ANYWHERE, "packaging script absent from a wheel/subtree checkout"),
    ("explicitly bypassed via env", _ANYWHERE, "the escape hatch reporting that it was used"),
    # ── LOCAL_ONLY: an incomplete environment, which CI is not allowed to be ──
    #
    # The C-compiler family. Four phrasings over ~22 sites, consolidated to the
    # one substring below. On a laptop with no compiler this is a fair skip; on a
    # CI leg it is the false green the audit exists to catch, so strict mode ends
    # the run rather than printing a row. #179 argued this class specifically:
    # these have never skipped on any leg, so making them fatal costs nothing
    # today and buys the alarm on the day a leg loses its toolchain.
    # pybind11 looks like an optional extra and is not one: the whole-suite job
    # syncs `--extra dev`, which declares it (GH #229), so the only environment
    # that can produce this skip is a venv provisioned with a narrower extra
    # list. That makes it a statement about the environment, not the build —
    # and if the `dev` extra ever loses the declaration, this is the tier that
    # says so instead of quietly dropping the check that the rebuild helper can
    # still find pybind11. The curated strict legs (mir.yml, windows-tail.yml)
    # do not list that file, so it cannot fire there.
    (
        "pybind11 is not installed",
        _LOCAL_ONLY,
        "CI syncs --extra dev, which declares pybind11; absent only in a narrower venv",
    ),
    # test_benchmark_scoped_run_out_flag.py asks git which report artifacts are
    # tracked, so its table cannot rot into a list of files nobody commits. A ZIP
    # export carries benchmarks/ but no .git and genuinely cannot answer; a CI leg
    # that cannot is a checkout that has broken, and the trackedness check would
    # have stopped running silently.
    ("not a git checkout", _LOCAL_ONLY, "actions/checkout always leaves a work tree"),
    ("no C compiler", _LOCAL_ONLY, "a CI leg without cc has broken, not adapted"),
    ("codegen compile unavailable", _LOCAL_ONLY, "same condition, reported from the codegen side"),
    ("no codegen backend available", _LOCAL_ONLY, "as above"),
    # libsbml is a HARD dependency in pyproject (`python-libsbml>=5.20`) and
    # HAS_LIBSBML is a plain import check, so this cannot be a build variant — if
    # it fires, the install is broken, and in CI a broken install must not pass.
    ("requires libsbml", _LOCAL_ONLY, "libsbml is a base dependency, not an extra"),
    # Numerics that a given host does not reproduce. Narrow on purpose: this is a
    # licence to not run a test, so it is spelled out per case rather than left as
    # a general "the numbers came out differently here" escape.
    (
        "finite-difference Jacobian does not carry this fixture",
        _ANYWHERE,
        "lanl/bngsim#176: the FD retry is a second attempt, not a guarantee, so "
        "the tests that assert a successful rescue stand down where there is no "
        "rescue to assert. The contract itself (auto reproduces explicit-FD "
        "whatever FD does) is asserted unconditionally and does not skip, so this "
        "never hides the fallback going missing",
    ),
    (
        "returns a finite sensitivity tensor on this host",
        _ANYWHERE,
        "lanl/bngsim#389: the non-finite forward-sensitivity guard's real-model "
        "witnesses are floating-point events. BIOMD0000000480's blow-up is a "
        "knife edge that macOS x86_64 does not cross at any tolerance-floor "
        "setting, and #388's structural witnesses go away when #388 is fixed — "
        "both are 'nothing to refuse here', not 'the guard is broken'. The guard "
        "itself is tested on a synthesized tensor and never skips, so this cannot "
        "hide the refusal going missing",
    ),
    # Missing .net / .xml fixtures. Deliberately last and deliberately narrow:
    # this is the category that rots silently, so it matches the exact phrasings
    # in use rather than a blanket "not found".
    ("not present", _ANYWHERE, "optional fixture absent from this checkout"),
    ("not found", _ANYWHERE, "optional fixture absent from this checkout"),
    ("not available", _ANYWHERE, "optional fixture absent from this checkout"),
)

# ─── Corpus absence, reported as one number ───────────────────────────────────
#
# The three catch-alls above are the honest declaration for "a model file this
# checkout does not have", and they should stay: those skips ARE legitimate, and
# making them undeclared would fail BNGSIM_SKIP_AUDIT=strict on every CI leg and
# in every worktree, which is the cry-wolf failure the note above warns about.
#
# But being legitimate is not the same as being legible, and this family is
# illegible in two compounding ways:
#
#   * It FRAGMENTS. Each gate names its own model, so the ~47 skips a
#     corpus-less checkout produces arrive as some thirty rows reading `1` or
#     `2`, and the one number a reader actually needs — how much of the suite did
#     not run — appears nowhere in the table.
#   * It UNDERSTATES. A test parametrized over the corpus contributes ONE skip
#     however many models it would have covered:
#     ``test_codegen_structural_key.py::test_corpus_key_matches_source_under_perturbation``
#     runs 14 models with the corpus present and 0 without, and reports `1`
#     either way. So the count is a lower bound on lost coverage, not a measure
#     of it, and the footer below says so rather than implying otherwise.
#
# Why it is worth a line at all: the two places the corpus is always absent are a
# ``.claude/worktrees/*`` worktree and CI (it is 257 MB and gitignored), and the
# pre-push hook prints the same `pytest (bngsim)....Passed` from a worktree as
# from the main checkout — 3705 tests against 3765, no signal. GH #192 shipped a
# 77-model `.net` regression through exactly this gap. Its *vacuous-pass* half was
# fixed by gating on a model FILE rather than on the directory (which exists
# empty, because ``models/.gitignore`` is tracked); this is the other half.
#
# The classifier is a RULE, not a list of models: a reason counts as corpus
# absence when it says something is absent AND names a corpus. Phrasings drift —
# that is the whole lesson of the audit above — so a new gate that says
# "rr_parity corpus model FOO not present" is picked up with no edit here.
_CORPUS_ABSENCE_TOKENS: tuple[str, ...] = ("not present", "not found", "not available")
_CORPUS_NAME_TOKENS: tuple[str, ...] = ("corpus", "benchmark", "sbml", "rr_parity")

# Where each corpus comes from, for the remediation line. Ordered most-cited
# first; only the ones a run actually hit are printed.
_CORPUS_REMEDIES: tuple[tuple[str, str], ...] = (
    ("rr_parity", "python parity_checks/rr_parity/materialize.py"),
    ("benchmark", "see benchmarks/suites/*/fetch.py"),
)


def _is_corpus_absence(reason: str) -> bool:
    """Is this skip "a model corpus is not in this checkout"?

    Deliberately conjunctive. ``not present`` alone also covers a one-off fixture
    (``egfr_net.net not present``) that has nothing to do with a corpus, and
    ``sbml`` alone appears in build-variant reasons; requiring both halves keeps
    the count meaning what the footer claims it means.
    """
    low = reason.lower()
    return any(t in low for t in _CORPUS_ABSENCE_TOKENS) and any(
        t in low for t in _CORPUS_NAME_TOKENS
    )


def _skip_reason(report: object) -> str:
    """Pull the human reason out of a skip report, minus pytest's prefix."""
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
    else:
        reason = str(longrepr or "")
    return reason[len("Skipped: ") :] if reason.startswith("Skipped: ") else reason


def _tier_of(reason: str) -> str | None:
    """The tier of the first declared pattern this reason matches, or None.

    First match wins, and LOCAL_ONLY is checked first so that a reason matching
    both tiers is treated as the stricter one. Otherwise a broad ANYWHERE
    pattern like ``not available`` could quietly launder a specific LOCAL_ONLY
    reason back into "fine everywhere", which is how the flat list lost this
    distinction in the first place.
    """
    low = reason.lower()
    for tier in (_LOCAL_ONLY, _ANYWHERE):
        for pattern, entry_tier, _ in _DECLARED_SKIPS:
            if entry_tier == tier and pattern.lower() in low:
                return tier
    return None


def _audit_skips(
    terminalreporter: object,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return (reason -> count, undeclared subset, LOCAL_ONLY subset).

    Recomputed by each hook rather than shared through the stash: conftest's
    pytest_sessionfinish can run *before* the terminal reporter's (which is what
    invokes pytest_terminal_summary), so anything the summary stashes is not yet
    there when the exit code is decided.
    """
    counts: dict[str, int] = {}
    for report in getattr(terminalreporter, "stats", {}).get("skipped", []):
        reason = _skip_reason(report)
        counts[reason] = counts.get(reason, 0) + 1
    tiers = {reason: _tier_of(reason) for reason in counts}
    undeclared = {r: n for r, n in counts.items() if tiers[r] is None}
    local_only = {r: n for r, n in counts.items() if tiers[r] == _LOCAL_ONLY}
    return counts, undeclared, local_only


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(
    terminalreporter: object, exitstatus: object, config: pytest.Config
) -> None:
    """Print what did not run, and flag reasons nobody has declared."""
    mode = os.environ.get("BNGSIM_SKIP_AUDIT", "warn").lower()
    if mode == "off":
        return
    counts, undeclared, local_only = _audit_skips(terminalreporter)
    strict = mode == "strict"
    if not counts:
        return

    write = terminalreporter.write_line  # type: ignore[attr-defined]
    terminalreporter.write_sep("─", "skip audit")  # type: ignore[attr-defined]
    # Corpus absence is collapsed to one row: thirty rows reading `1` are how the
    # total stayed invisible. The per-test detail is still one `-rs` away.
    corpus = {r: n for r, n in counts.items() if _is_corpus_absence(r)}
    n_corpus = sum(corpus.values())
    for reason, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if reason in corpus:
            continue
        if reason in undeclared:
            mark = "??"
        elif reason in local_only:
            # `!!` rather than `??`: this one IS declared, and the complaint is
            # about where it fired, not about nobody having decided on it.
            mark = "!!"
        else:
            mark = "  "
        write(f" {mark} {n:>3}  {reason[:96]}")
    if n_corpus:
        plural = "reason" if len(corpus) == 1 else "reasons"
        write(
            f"    {n_corpus:>3}  model corpus absent from this checkout ({len(corpus)} {plural})"
        )
        write("")
        write(f" {n_corpus} test(s) did not run because a model corpus is not in this checkout.")
        for token, remedy in _CORPUS_REMEDIES:
            if any(token in r.lower() for r in corpus):
                write(f"   {token}:  {remedy}")
        write(" This is a LOWER BOUND on the coverage that did not execute: a test")
        write(" parametrized over the corpus contributes one skip however many models it")
        write(" would have covered. CI never has the corpus and neither does a worktree,")
        write(" so a green run from either says nothing about these paths (GH #192).")
    if undeclared:
        write("")
        write(f" {len(undeclared)} undeclared skip reason(s), marked ?? above.")
        write(" Add each to _DECLARED_SKIPS in python/tests/conftest.py with a rationale,")
        write(" or fix the test so it runs. A skip nobody declared is usually a test that")
        write(" stopped running without anyone deciding it should.")
    if local_only:
        write("")
        n_local = sum(local_only.values())
        write(f" {n_local} test(s) skipped for an INCOMPLETE ENVIRONMENT, marked !! above.")
        write(" These are declared and fine on a developer box — a missing C compiler, a")
        write(" missing base dependency — and are a defect in CI, where the environment is")
        if strict:
            write(" built rather than found. Under BNGSIM_SKIP_AUDIT=strict they end the run.")
        else:
            write(" built rather than found. BNGSIM_SKIP_AUDIT=strict would end the run here.")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run under BNGSIM_SKIP_AUDIT=strict for an undeclared skip, or for
    a LOCAL_ONLY skip — one that says the environment is incomplete, which is a
    statement a CI leg is not entitled to make."""
    if os.environ.get("BNGSIM_SKIP_AUDIT", "warn").lower() != "strict":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    _, undeclared, local_only = _audit_skips(reporter)
    # Only escalate a run that would otherwise pass; a real failure keeps its
    # own exit code, which is the more actionable one.
    if (undeclared or local_only) and exitstatus == 0:
        session.exitstatus = 1


@pytest.fixture(scope="session")
def skip_audit() -> SimpleNamespace:
    """The audit's own internals, for ``test_skip_audit.py`` (GH #222).

    Handed over as a fixture rather than imported: ``addopts`` sets
    ``--import-mode=importlib``, which does not put this directory on
    ``sys.path``, so ``import conftest`` from a test module does not resolve.
    """
    return SimpleNamespace(
        is_corpus_absence=_is_corpus_absence,
        declared_skips=_DECLARED_SKIPS,
        corpus_remedies=_CORPUS_REMEDIES,
        tier_of=_tier_of,
        ANYWHERE=_ANYWHERE,
        LOCAL_ONLY=_LOCAL_ONLY,
    )


@pytest.fixture(scope="session")
def artifact_caches() -> SimpleNamespace:
    """The cache redirect's own internals, for ``test_artifact_cache_isolation.py``
    (issue #372). Handed over as a fixture for the reason ``skip_audit`` above is."""
    return SimpleNamespace(
        root_env=_TEST_CACHE_ROOT_ENV,
        redirected=_REDIRECTED_CACHES,
        resolve_root=_test_cache_root,
    )


@pytest.fixture
def data_dir() -> Path:
    """Path to the C++ test data directory (shared with Phase A tests).

    Set BNGSIM_TEST_DATA env var to override (useful when running tests
    from outside the source tree).
    """
    if _DATA_DIR_ENV:
        d = Path(_DATA_DIR_ENV)
    else:
        # Walk up from python/tests/ → python/ → bngsim/ then into tests/data/
        d = Path(__file__).resolve().parent.parent.parent / "tests" / "data"
    assert d.is_dir(), f"Test data directory not found: {d}"
    return d


@pytest.fixture
def simple_decay_net(data_dir: Path) -> Path:
    """Path to simple_decay.net."""
    return data_dir / "simple_decay.net"


@pytest.fixture
def reversible_net(data_dir: Path) -> Path:
    """Path to two_species_reversible.net."""
    return data_dir / "two_species_reversible.net"


@pytest.fixture
def time_func_net(data_dir: Path) -> Path:
    """Path to time_dependent_func.net."""
    return data_dir / "time_dependent_func.net"


@pytest.fixture
def fixed_species_net(data_dir: Path) -> Path:
    """Path to fixed_species.net."""
    return data_dir / "fixed_species.net"


@pytest.fixture
def const_expr_net(data_dir: Path) -> Path:
    """Path to const_expr_setparam.net."""
    return data_dir / "const_expr_setparam.net"


@pytest.fixture
def expr_param_net(data_dir: Path) -> Path:
    """Path to expr_param_species.net."""
    return data_dir / "expr_param_species.net"


@pytest.fixture
def exprtk_reserved_words_net(data_dir: Path) -> Path:
    """Path to exprtk_reserved_words.net (issue #18 regression)."""
    return data_dir / "exprtk_reserved_words.net"


@pytest.fixture
def t_as_observable_net(data_dir: Path) -> Path:
    """Path to t_as_observable.net (issue #24 regression)."""
    return data_dir / "t_as_observable.net"


@pytest.fixture
def reactions_text_block_net(data_dir: Path) -> Path:
    """Path to reactions_text_block.net (issue #13 regression)."""
    return data_dir / "reactions_text_block.net"


@pytest.fixture
def obs_zero_arg_call_net(data_dir: Path) -> Path:
    """Path to obs_zero_arg_call.net (issue #28 regression)."""
    return data_dir / "obs_zero_arg_call.net"


@pytest.fixture
def obs_zero_arg_call_sens_net(data_dir: Path) -> Path:
    """Path to obs_zero_arg_call_sens.net (issue #28 on the codegen path)."""
    return data_dir / "obs_zero_arg_call_sens.net"


@pytest.fixture
def sign_as_parameter_net(data_dir: Path) -> Path:
    """Path to sign_as_parameter.net (sign-collision regression)."""
    return data_dir / "sign_as_parameter.net"


@pytest.fixture
def mratio_overflow_net(data_dir: Path) -> Path:
    """Path to test_mratio_overflow.net (issue #42 regression).

    Generated by BNG2.pl from BNGL-Models/my_models/ode/test_Mratio_1.bngl.
    The parameter block computes a confluent-hypergeometric mean/sdev with
    `a=-1000, b=9001, z=-10000` — the regime where bngsim's previous naive
    power-series mratio overflowed to nan.
    """
    return data_dir / "test_mratio_overflow.net"


@pytest.fixture
def homodimer_ssa_net(data_dir: Path) -> Path:
    """Path to homodimer_ssa.net."""
    return data_dir / "homodimer_ssa.net"


@pytest.fixture
def fractional_ssa_net(data_dir: Path) -> Path:
    """Path to fractional_ssa.net."""
    return data_dir / "fractional_ssa.net"


@pytest.fixture
def saturation_net(data_dir: Path) -> Path:
    """Path to saturation.net."""
    return data_dir / "saturation.net"


@pytest.fixture
def ssa_abc_net(data_dir: Path) -> Path:
    """Path to ssa_abc.net."""
    return data_dir / "ssa_abc.net"


@pytest.fixture
def mm_tqssa_net(data_dir: Path) -> Path:
    """Path to mm_tqssa.net (Michaelis-Menten tQSSA test)."""
    return data_dir / "mm_tqssa.net"


@pytest.fixture
def sat_rewrite_net(data_dir: Path) -> Path:
    """Path to sat_rewrite.net (legacy Sat → Functional loader rewrite fixture)."""
    return data_dir / "sat_rewrite.net"


@pytest.fixture
def hill_rewrite_net(data_dir: Path) -> Path:
    """Path to hill_rewrite.net (legacy Hill → Functional loader rewrite fixture)."""
    return data_dir / "hill_rewrite.net"


@pytest.fixture
def tfun_time_indexed_net(data_dir: Path) -> Path:
    """Path to tfun_time_indexed.net (tfun indexed by time)."""
    return data_dir / "tfun_time_indexed.net"


@pytest.fixture
def tfun_param_indexed_net(data_dir: Path) -> Path:
    """Path to tfun_param_indexed.net (tfun indexed by parameter)."""
    return data_dir / "tfun_param_indexed.net"


@pytest.fixture
def tfun_step_time_indexed_net(data_dir: Path) -> Path:
    """Path to tfun_step_time_indexed.net (step-interpolated tfun)."""
    return data_dir / "tfun_step_time_indexed.net"


@pytest.fixture
def tfun_uppercase_time_net(data_dir: Path) -> Path:
    """tfun_uppercase_time.net — `.tfun` header has `# Time  cumNcases()`
    and the tfun call passes `Time` as index. Regression for GH #35:
    bngsim should honor BNG's case-insensitive `time`/`t` and trailing
    `()` stripping conventions in `.tfun` headers.
    """
    return data_dir / "tfun_uppercase_time.net"


@pytest.fixture
def tfun_paren_param_net(data_dir: Path) -> Path:
    """tfun_paren_param.net — `.tfun` column-1 header is `drug_conc()`
    (with empty parens) but the tfun call passes the bare `drug_conc`
    parameter. Regression for GH #35: trailing `()` stripping must apply
    to param-indexed tfun headers too, not just time.
    """
    return data_dir / "tfun_paren_param.net"


@pytest.fixture
def wrap_single_net(data_dir: Path) -> Path:
    """Path to wrap_single.net (wrapper-form tfun: `(tfun('drive') + 5)/k_scale`).

    Regression fixture for GH #33. The .net came from BioNetGen 2.9.3 on
    wrap_single.bngl; sister `drive.tfun` lives in the same directory so the
    loader's relative-path resolver finds it.
    """
    return data_dir / "wrap_single.net"


@pytest.fixture
def cumncases_tfun(data_dir: Path) -> Path:
    """Path to cumNcases.tfun data file."""
    return data_dir / "cumNcases.tfun"


@pytest.fixture
def dose_response_tfun(data_dir: Path) -> Path:
    """Path to dose_response.tfun data file."""
    return data_dir / "dose_response.tfun"


@pytest.fixture
def nfsim_xml(data_dir: Path) -> Path:
    """Path to NFsim simple_system.xml test file."""
    return data_dir / "nfsim" / "simple_system.xml"


@pytest.fixture
def nfsim_funccols_xml(data_dir: Path) -> Path:
    """XML with a global function and a composite (function-of-function).

    `phos_ratio() = Xp/Xtot` references observables only (a GlobalFunction);
    `phos_percent() = 100*phos_ratio()` references another function (a
    CompositeFunction). Used to verify that both kinds surface as Result
    expression columns.
    """
    return data_dir / "nfsim" / "funccols.xml"


@pytest.fixture
def nfsim_param_prop_xml(data_dir: Path) -> Path:
    """XML mirroring simple_system but with derived <Parameter expr=...> chains.

    Used to verify set_param() propagation through dependent parameters
    (issue #20). The new params introduce: kon = kon_base*kon_scale,
    use_fast = if(kon_scale>=threshold,1,0), kon_eff = kon*(1-use_fast)+kon*100*use_fast.
    """
    return data_dir / "nfsim" / "param_propagation.xml"


@pytest.fixture
def nfsim_seed_concentration_xml(data_dir: Path) -> Path:
    """XML where seed-species concentrations reference parameters.

    Used to verify pre-init set_param() rewrites `<Species concentration="X">`
    through the override-resolved namespace before NFsim creates agents
    (issue #29). X_init=5000 drives X(p~0,y); Y_init=500 drives Y(x);
    X_total=X_init+Y_init exercises propagation through a derived parameter.
    """
    return data_dir / "nfsim" / "seed_concentration_param.xml"


@pytest.fixture
def nfsim_switchable_rate_xml(data_dir: Path) -> Path:
    """A(b)+B(b)<->A.B with a switchable rate and a derived seed amount (GH #44).

    ``kf`` (the binding rate) is 0 at load, so the bind rule would be dropped by
    NFsim's parser unless the session keeps zero-rate rules — this exercises Bug
    1 (a post-init ``set_param('kf', ...)`` must be able to activate the rule).
    Both seed species use ``concentration="Ntot"`` where ``Ntot = 100*scale``, a
    *derived* parameter — this exercises Bug 2 (a pre-init ``set_param('scale',
    ...)`` must re-derive the seed-species amount, matching NFsim). Shared by the
    NFsim and RuleMonkey session tests.
    """
    return data_dir / "nfsim" / "switchable_rate.xml"


@pytest.fixture
def dose_seed_precision_xml(data_dir: Path) -> Path:
    """L(b)+R(b)<->L.R in real BNG2 writeXML shape, for GH #115.

    Every ``<Parameter>`` carries both the collapsed ``value=`` and the symbolic
    ``expr=``, and the two disagree in the last digits for ``NA`` — the shape
    BNG2.pl emits for Avogadro's number, and the reason "re-evaluate the whole
    ``expr=`` graph" is not the same thing as "propagate an override".

    * ``LT = ((dose_nM*1e-9)*NA)*V_sim`` — a derived, *fractional* seed amount
      (1806.6422 at the default dose), so a pre-init ``set_param('dose_nM',
      ...)`` must re-derive it and the result must be rounded half-up (GH #44,
      GH #51).
    * ``RT = 300*rscale`` — a second derived seed amount that can be driven to
      an exactly-integral count, so a scan can cross fractional/integral in
      both directions.
    * ``kf = ((K*1e9)*kr)/(NA*V_sim)`` — a bimolecular rate constant *outside*
      the dose's dependency cone, which must keep its loaded ``value=``
      bit-for-bit under an override.

    RuleMonkey-only. The NFsim adapter still bakes overrides by re-evaluating
    the whole parameter graph into the XML, so it re-rounds ``kf`` to ``expr=``
    precision whenever any override is pending; only the seed *counts* agree
    across the two engines here.
    """
    return data_dir / "nfsim" / "dose_seed_precision.xml"


@pytest.fixture
def fractional_init_xml(data_dir: Path) -> Path:
    """Path to XML fixture with non-integer initial species count."""
    return data_dir / "nfsim" / "fractional_init.xml"


@pytest.fixture
def nfsim_malformed_xml(data_dir: Path) -> Path:
    """Path to malformed NFsim XML (triggers parse error)."""
    return data_dir / "nfsim" / "malformed.xml"


@pytest.fixture
def nfsim_empty_model_xml(data_dir: Path) -> Path:
    """XML from a BNGL file with only a parameters block — no molecule
    types, observables, or reactions, but with a ``simulate({method=>"nf",...})``
    action. BNG2.pl emits a ``.gdat`` with only the ``# time`` header and
    the requested time rows.

    Generated from ``BNGL-Models/my_models/nf/BSA_v1.bngl`` (an unfinished
    BNGL file that snuck into a corpus sweep). Used to verify that
    bngsim's NFsim path matches BNG2.pl's vacuous-success behavior and
    that ``Result.has_simulation_data`` reports ``False`` for the
    resulting empty Result — see issue #40 B-2.
    """
    return data_dir / "nfsim" / "empty_model.xml"


@pytest.fixture
def nfsim_composite_local_deps_xml(data_dir: Path) -> Path:
    """XML whose rate law is a CompositeFunction with local-function deps.

    Minimal BLBR-style model: A flips s 0->1 at a rate that depends on a
    molecule-scope observable Atot(a). BNG2.pl emits `a` as type="Local"
    on the synthesized `_rateLaw1` composite, so a scope-free
    evaluateOn(nullptr,nullptr,...) call hits the local-deps branch.

    Used by the post-Session-29 audit-closure tests: the bngsim probe in
    `resolve_output_functions` (src/nfsim_simulator.cpp) catches the
    resulting std::runtime_error and silently skips the composite. Before
    the Session-29 carry the same path called exit(1) and killed the
    process; see bngsim/dev/notes/session29_regression_audit.md.
    """
    return data_dir / "nfsim" / "composite_with_local_deps.xml"


@pytest.fixture
def nfsim_composite_reactant_count_dep_xml(data_dir: Path) -> Path:
    """XML whose rate law is a CompositeFunction with a reactant-count dep.

    Minimal model: `A(b) + A(b) -> A(b!1).A(b!1)` at rate `reactant_1()*k`.
    `reactant_1()` is declared as an empty local function (BNG emits it as a
    Function with an empty Expression, which the loader skips) and is the
    built-in reactant-count reference. The synthesized `_rateLaw1` composite
    therefore has n_lfs==0 but n_reactantCounts==1.

    GH #116: a scope-free evaluateOn(nullptr,nullptr,nullptr,0) — the probe in
    `resolve_output_functions` (src/nfsim_simulator.cpp) — used to pass the
    local-function guard (n_lfs==0) and then NULL-deref the reactant-count
    array, a native SIGSEGV during NfsimSession.initialize(). evaluateOn now
    throws a catchable error for a missing reactant-count context, so the probe
    skips the composite. Minimal isolation of actin_branch_forFitToData.
    """
    return data_dir / "nfsim" / "composite_with_reactant_count_dep.xml"


@pytest.fixture
def nfsim_tfun_xml(data_dir: Path) -> Path:
    """Path to NFsim TFUN XML fixture."""
    return data_dir / "nfsim" / "tfun_test" / "tfun_simple.xml"


@pytest.fixture
def nfsim_tfun_new_format_dir(data_dir: Path) -> Path:
    """Path to the new-format NFsim TFUN fixture directory."""
    return data_dir / "nfsim" / "tfun_new_format"


@pytest.fixture
def nfsim_compartment_xml(data_dir: Path) -> Path:
    """Path to NFsim XML fixture containing compartments (cBNGL smoke)."""
    return data_dir / "nfsim" / "compartment_test" / "simple_compartment_system.xml"


@pytest.fixture
def nfsim_sym_state_xml(data_dir: Path) -> Path:
    """XML with two same-named state-bearing components on one MoleculeType.

    `L(r~u~c~g, r~u~c~g)` — exercises NFsim's symmetric-site renaming
    (internally `r1`/`r2`) on a stateful equivalency class. Used to verify
    the species API resolves bare class-original names like `L(r~u,r~u)`
    (issue #21).
    """
    return data_dir / "nfsim" / "sym_state_sites.xml"


@pytest.fixture
def nfsim_sym_stateless_xml(data_dir: Path) -> Path:
    """XML with two same-named stateless components on one MoleculeType.

    `L(r,r)` — exercises the species API on a stateless equivalency class.
    """
    return data_dir / "nfsim" / "sym_stateless_sites.xml"
