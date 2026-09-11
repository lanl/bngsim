#!/usr/bin/env python3
"""Refresh or summarize the vendored NFsim tree from a local NFsim Git checkout."""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# The bngsim checkout itself. REPO_ROOT is its *parent*, which is a Git work
# tree only in the monorepo layout this script was written for -- in a
# standalone bngsim clone it is just the directory above the repo. Anything
# that shells out to git must use BNGSIM_ROOT. Matches the other four
# scripts/vendor_*.py.
BNGSIM_ROOT = REPO_ROOT / "bngsim"
VENDOR_DIR = BNGSIM_ROOT / "third_party" / "nfsim"
PATCHES_DIR = BNGSIM_ROOT / "scripts" / "nfsim_vendor_patches"
METADATA_NAME = "VENDOR.json"
DEFAULT_NFSIM_REPO = Path("/tmp/nfsim-vendor-candidate")
DEFAULT_VENDOR_REF = "bngsim/vendor"
DEFAULT_UPSTREAM_REMOTE = "https://github.com/RuleWorld/nfsim.git"
DEFAULT_UPSTREAM_BASE_REF = "origin/master"
TRIPWIRE_REMEDIATION = (
    "Direct edits to bngsim/third_party/nfsim/ are not supported. "
    "Apply changes via bngsim/scripts/vendor_nfsim.py instead:\n"
    "  1. Land the change on the NFsim-side `bngsim/vendor` branch.\n"
    "  2. Re-export the patch series under "
    "bngsim/scripts/nfsim_vendor_patches/.\n"
    "  3. Update CARRY_QUEUE in this script if a new topic is needed.\n"
    "  4. Run `vendor_nfsim.py --ref bngsim/vendor` to refresh the tree.\n"
    "See bngsim/scripts/NFSIM_VENDORING.md for the full workflow."
)

INCLUDED_PATHS = (
    "CMakeLists.txt",
    "LICENSE.txt",
    "README.md",
    "src",
)

PRUNED_PATHS = (
    "src/NFtest",
    "src/NFfunction/muParser",
    "src/NFfunction/exprtk",
    "src/NFutil/MTrand/mttest.cpp",
)

CARRY_QUEUE = (
    {
        "topic": "bngsim/carry-v1143-compat",
        "kind": "local",
        "status": (
            "BNGsim selector compatibility toggle for same-seed parity with "
            "NFsim v1.14.3 CLI behavior."
        ),
        "commits": [
            {
                "commit": "8d6900200e299f2e75227f3835528b4bf41e1419",
                "summary": "bngsim: add NFsim v1.14.3 selector compat toggle",
            },
        ],
    },
    {
        "topic": "bngsim/carry-exception-safety",
        "kind": "local",
        "status": (
            "Raise an exception instead of terminating when the global molecule limit is exceeded."
        ),
        "commits": [
            {
                "commit": "5ab913963446f3b191a6e8544bef1147c2522b0c",
                "summary": "bngsim: throw when molecule limit is exceeded",
            },
        ],
    },
    {
        "topic": "bngsim/carry-tfun-comments",
        "kind": "local",
        "status": "Ignore comment lines in TFUN data files used by BNG fixtures.",
        "commits": [
            {
                "commit": "9c4a11835dfb22a280afd84cdcdae67f6c93aa30",
                "summary": "bngsim: ignore comments in TFUN data files",
            },
        ],
    },
    {
        "topic": "bngsim/carry-subproject-macos-arch",
        "kind": "local",
        "status": (
            "Do not force universal macOS architectures inside vendored NFsim "
            "subprojects; the embedding build owns target-architecture choice."
        ),
        "commits": [
            {
                "commit": "e0ff8d3894ccd4c8850e522c4fa388a1df103bb3",
                "summary": "bngsim: don't force universal macOS arch in subprojects",
            },
        ],
    },
    {
        "topic": "bngsim/carry-exit-to-exception",
        "kind": "local",
        "status": (
            "Convert exit(1) calls to throw std::runtime_error across six "
            "NFsim files so malformed input and other fatal-error paths "
            "raise exceptions instead of killing the embedding host. "
            "Originally landed as PyBNF-Private bngsim 9c77d2ec (Session 29, "
            "Feb 22 2026) but written directly into bngsim/third_party/nfsim "
            "and silently reverted by 008529a7 on May 15 2026; now staged "
            "here so future vendor_nfsim.py refreshes preserve it."
        ),
        "commits": [
            {
                "commit": "e279193196a7516fd61b58ff0664e382b0c08e62",
                "summary": "bngsim: convert exit(1) calls to throw std::runtime_error",
            },
        ],
    },
    {
        "topic": "bngsim/carry-reserved-symbol-remap",
        "kind": "local",
        "status": (
            "Forward the mu::Parser ExprTk shim's BNG identifier -> ExprTk "
            "symbol mapping (the unconditional leading-underscore remap and the "
            "conditional reserved-symbol mangle, nfsim_funcparser.h) to the host "
            "single source bngsim::expr_compat (compute_registration_name + "
            "remap_name, exposed via <bngsim/expr_compat.hpp> and linked into the "
            "vendored NFsim target through bngsim::expression). Previously this "
            "logic was hand-ported into the shim alongside the same code in "
            "bngsim/src/expression.cpp; that dual ownership is the drift risk "
            "issue #49 flags. The shim now keeps only the thin mu::Parser "
            "adapter state (the per-parser mangled-name map and the call-form "
            "ambiguity check). The host reserved set is a superset of the shim's "
            "former one (it also guards the `time` alias and the u_* constant "
            "keys from GH #90); the extra names only add internally-consistent "
            "mangling, so behavior is unchanged on the full suite. Closes the "
            "reserved-symbol half of the BNG-compatibility split "
            "(internal#49, originally #64)."
        ),
        "commits": [
            {
                "commit": "575790268280a0050d5a954358ceb7c9dbe82c7c",
                "summary": "bngsim: forward mu::Parser reserved-symbol remap to host expr_compat",
            },
        ],
    },
    {
        "topic": "bngsim/carry-mratio-builtin",
        "kind": "local",
        "status": (
            "Forward the NFsim ExprTk mu::Parser shim's mratio(a,b,z) built-in — "
            "confluent hypergeometric ratio M(a+1,b+1,z)/M(a,b,z), BNG2.pl "
            "Perl2/Expression.pm — to the host single source "
            "bngsim::expr_compat::mratio (nfsim_funcparser.h). The modified-Lentz "
            "continued-fraction implementation was previously ported verbatim "
            "from bngsim/src/expression.cpp into the shim; now both the NFsim "
            "shim and the ODE/SSA engine evaluate it from one definition. The "
            "shim still registers `mratio` as a 3-arg ExprTk function; only the "
            "adapter body changes. Closes the mratio half of the "
            "BNG-compatibility split (internal#49, originally #64)."
        ),
        "commits": [
            {
                "commit": "97a28d6c7440b5e48651109e99322ca32915713f",
                "summary": "bngsim: forward mu::Parser mratio() built-in to host expr_compat",
            },
        ],
    },
    {
        "topic": "bngsim/carry-reactant-count-composite-guard",
        "kind": "local",
        "status": (
            "Guard CompositeFunction::evaluateOn against a missing reactant-count "
            "context when a composite depends on reactant_N(). bngsim probes "
            "composites scope-free to discover scalar output columns; without "
            "this guard, reactant-count-only composites dereference a NULL count "
            "buffer and segfault during initialize() instead of being skipped as "
            "non-scalar output functions (GH #116)."
        ),
        "commits": [
            {
                "commit": "4a2bfb1877aae432e37f5c67da29d4f872fe3a7b",
                "summary": "bngsim: guard reactant-count composite scope-free eval",
            },
        ],
    },
    {
        "topic": "bngsim/carry-products-selector-hard-stop",
        "kind": "local",
        "status": (
            "Restore the pre-PR-85 hard load failure for include_products()/"
            "exclude_products(). Upstream origin/master now parses product-side "
            "selector blocks, but BNGsim's regression contract is still to fail "
            "loudly rather than silently change semantics mid-refresh; keep the "
            "clear 'not yet enforced in NFsim' abort until the products-side "
            "selector implementation is finished and validated deliberately."
        ),
        "commits": [
            {
                "commit": "6f19aad5db25ab0ed16acfd9add116748f0a88ca",
                "summary": "bngsim: restore hard stop for include_products/exclude_products",
            },
        ],
    },
    {
        "topic": "bngsim/carry-prune-nftest-facade",
        "kind": "local",
        "status": (
            "Prune NFtest-only headers from NFsim.hh in the vendored facade. "
            "The generated vendor export still omits src/NFtest/, but upstream "
            "NFsim.hh includes those test harness headers directly; without this "
            "carry, embedding/library builds fail on missing NFtest paths even "
            "when NFSIM_BUILD_EXECUTABLE is off."
        ),
        "commits": [
            {
                "commit": "154f9e9dcd21c95b4cf06ccc1cc11982e158d7b6",
                "summary": "bngsim: prune NFtest headers from vendored facade",
            },
        ],
    },
    {
        "topic": "bngsim/carry-library-only-cli-trim",
        "kind": "local",
        "status": (
            "Trim standalone CLI and NFtest-only sources from NFsim's current "
            "top-level CMake build when BNGsim builds the vendored library with "
            "NFSIM_BUILD_LIBRARY=ON and NFSIM_BUILD_EXECUTABLE=OFF. The vendor "
            "export omits src/NFtest, so library-only builds must not compile "
            "NFsim.cpp, Scheduler.cpp, or the NFtest harness sources."
        ),
        "commits": [
            {
                "commit": "9ff506c7c0bec9383df5c1a2b7e431a6c50d7843",
                "summary": "bngsim: trim CLI sources from library-only NFsim build",
            },
        ],
    },
    {
        "topic": "bngsim/carry-step-to-cache-hook",
        "kind": "local",
        "status": (
            "Restore the public System::invalidateStepToCache() compatibility "
            "hook that bngsim's session wrapper uses after parameter and "
            "species mutations. Upstream NFsim still keeps the pending stepTo() "
            "cache internally, but the method is protected again; expose the "
            "existing cache reset logic so the host build and runtime stay "
            "compatible."
        ),
        "commits": [
            {
                "commit": "6a1728f561e00cfbf05139b8eba015ff4941688e",
                "summary": "bngsim: restore stepTo cache compatibility hook",
            },
        ],
    },
    {
        "topic": "bngsim/carry-keep-zero-rate-reactions",
        "kind": "local",
        "status": (
            "Keep reaction rules whose base rate is exactly zero at parse time "
            "in the System instead of dropping them, gated behind an opt-in "
            "System::keepZeroRateReactions flag threaded through "
            "initializeFromXML. bngsim's stateful session API changes a rate "
            "after initialize() (equilibrate, then switch a rate on); a dropped "
            "zero-rate rule can never be resurrected by a later setParameter(). "
            "A kept rule has zero propensity, so trajectories are unchanged; the "
            "default (off) preserves upstream behavior. See bngsim GH #44."
        ),
        "commits": [
            {
                "commit": "5c547470600d00b1bbd64c087705b905b2e007f4",
                "summary": (
                    "bngsim: keep zero-base-rate reactions so post-init "
                    "set_param can activate them"
                ),
            },
        ],
    },
)

SUMMARY_PREVIEW_LIMIT = 12


def run(cmd: list[str], cwd: Path | None = None, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=text,
    )


def git(nfsim_repo: Path, args: list[str]) -> str:
    return run(["git", "-C", str(nfsim_repo), *args]).stdout.strip()


def ordered_remotes(nfsim_repo: Path) -> list[str]:
    try:
        remotes = [remote for remote in git(nfsim_repo, ["remote"]).splitlines() if remote]
    except subprocess.CalledProcessError:
        return []

    ordered: list[str] = []
    for preferred in ("fork", "origin"):
        if preferred in remotes:
            ordered.append(preferred)
    for remote in remotes:
        if remote not in ordered:
            ordered.append(remote)
    return ordered


def ref_candidates(nfsim_repo: Path, ref: str) -> list[str]:
    candidates = [ref]
    if ref.startswith("refs/"):
        return candidates

    remotes = ordered_remotes(nfsim_repo)
    if not remotes:
        return candidates

    ref_prefix = ref.split("/", 1)[0]
    if ref_prefix in remotes:
        return candidates

    for remote in remotes:
        candidate = f"{remote}/{ref}"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_ref(nfsim_repo: Path, ref: str) -> tuple[str, str]:
    attempted: list[str] = []
    for candidate in ref_candidates(nfsim_repo, ref):
        try:
            commit = git(nfsim_repo, ["rev-parse", "--verify", f"{candidate}^{{commit}}"])
            return candidate, commit
        except subprocess.CalledProcessError:
            attempted.append(candidate)

    attempted_text = ", ".join(attempted) if attempted else ref
    raise RuntimeError(f"Could not resolve NFsim ref '{ref}'. Tried: {attempted_text}")


def safe_extract_tar(data: bytes, destination: Path) -> None:
    archive_path = destination / "nfsim.tar"
    archive_path.write_bytes(data)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = destination / member.name
            try:
                member_path.resolve().relative_to(destination.resolve())
            except ValueError as exc:
                raise RuntimeError(f"Refusing unsafe archive path: {member.name}") from exc
        archive.extractall(destination)
    archive_path.unlink()


def export_nfsim_tree(nfsim_repo: Path, commit: str, destination: Path) -> Path:
    source_dir = destination / "source"
    source_dir.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(nfsim_repo), "archive", "--format=tar", commit, *INCLUDED_PATHS],
        check=True,
        capture_output=True,
    )
    safe_extract_tar(archive.stdout, source_dir)

    for rel_path in PRUNED_PATHS:
        target = source_dir / rel_path
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    return source_dir


def ensure_clean_destination(force: bool) -> None:
    # Query git from BNGSIM_ROOT, not REPO_ROOT: REPO_ROOT is the *parent* of
    # the bngsim checkout, which is not itself a work tree in a standalone
    # clone -- `git -C <parent> status` then exits 128 and this guard aborts the
    # whole refresh instead of checking anything, pushing everyone to --force,
    # which skips the check outright. vendor_rulemonkey.py already fixed this.
    if force:
        return
    rel_vendor = VENDOR_DIR.relative_to(BNGSIM_ROOT)
    status = run(["git", "status", "--porcelain", "--", str(rel_vendor)], cwd=BNGSIM_ROOT).stdout
    if status.strip():
        raise RuntimeError(
            f"{VENDOR_DIR.relative_to(REPO_ROOT)} has uncommitted changes. "
            "Commit/stash them or rerun with --force."
        )


def remove_children(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_tree(source: Path, destination: Path) -> None:
    remove_children(destination)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def remote_url(nfsim_repo: Path, remote: str) -> str | None:
    try:
        return git(nfsim_repo, ["remote", "get-url", remote])
    except subprocess.CalledProcessError:
        return None


def canonical_upstream_remote(nfsim_repo: Path) -> str:
    for remote in ("upstream", "origin"):
        url = remote_url(nfsim_repo, remote)
        if url and "RuleWorld/nfsim" in url:
            return url
    return DEFAULT_UPSTREAM_REMOTE


def upstream_base_metadata(nfsim_repo: Path, commit: str) -> tuple[str, str | None]:
    try:
        resolved_base_ref, _ = resolve_ref(nfsim_repo, DEFAULT_UPSTREAM_BASE_REF)
    except RuntimeError:
        return DEFAULT_UPSTREAM_BASE_REF, None

    try:
        base_commit = git_merge_base(nfsim_repo, resolved_base_ref, commit)
    except subprocess.CalledProcessError:
        return resolved_base_ref, None

    return resolved_base_ref, base_commit


def metadata(nfsim_repo: Path, ref: str, commit: str) -> dict:
    base_ref, base_commit = upstream_base_metadata(nfsim_repo, commit)
    return {
        "name": "NFsim",
        "vendored_path": "bngsim/third_party/nfsim",
        "source": {
            "local_checkout": str(nfsim_repo),
            "upstream_remote": canonical_upstream_remote(nfsim_repo),
            "local_origin_remote": remote_url(nfsim_repo, "origin"),
            "fork_remote": remote_url(nfsim_repo, "fork"),
            "branch_or_ref": ref,
            "commit": commit,
            "base_ref": base_ref,
            "base_commit": base_commit,
        },
        "imported_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pruning": {
            "included_paths": list(INCLUDED_PATHS),
            "excluded_paths": list(PRUNED_PATHS),
        },
        "carry_queue": list(CARRY_QUEUE),
    }


def write_metadata(nfsim_repo: Path, ref: str, commit: str) -> None:
    data = metadata(nfsim_repo, ref, commit)
    metadata_path = VENDOR_DIR / METADATA_NAME
    metadata_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def load_vendor_metadata() -> dict | None:
    metadata_path = VENDOR_DIR / METADATA_NAME
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse {metadata_path}: {exc}") from exc


def files_differ(left: Path, right: Path) -> bool:
    if left.is_dir() != right.is_dir():
        return True
    if left.is_dir():
        return False
    return not filecmp.cmp(left, right, shallow=False)


def compare_trees(expected: Path, actual: Path) -> list[str]:
    differences: list[str] = []
    expected_paths = {
        path.relative_to(expected) for path in expected.rglob("*") if path.name != METADATA_NAME
    }
    actual_paths = {
        path.relative_to(actual) for path in actual.rglob("*") if path.name != METADATA_NAME
    }
    for rel_path in sorted(expected_paths | actual_paths):
        expected_path = expected / rel_path
        actual_path = actual / rel_path
        if rel_path not in expected_paths:
            differences.append(f"extra: {rel_path}")
        elif rel_path not in actual_paths:
            differences.append(f"missing: {rel_path}")
        elif files_differ(expected_path, actual_path):
            differences.append(f"changed: {rel_path}")
    return differences


def difference_counts(differences: list[str]) -> dict[str, int]:
    counts = {"changed": 0, "missing": 0, "extra": 0}
    for diff in differences:
        status, _, _ = diff.partition(":")
        counts[status] = counts.get(status, 0) + 1
    return counts


def path_in_scope(rel_path: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel_path == prefix or rel_path.startswith(f"{prefix}/") for prefix in prefixes)


def is_vendored_surface_path(rel_path: str) -> bool:
    return path_in_scope(rel_path, INCLUDED_PATHS) and not path_in_scope(rel_path, PRUNED_PATHS)


def git_lines(nfsim_repo: Path, args: list[str]) -> list[str]:
    output = git(nfsim_repo, args)
    return [line for line in output.splitlines() if line]


def git_merge_base(nfsim_repo: Path, left: str, right: str) -> str:
    return git(nfsim_repo, ["merge-base", left, right])


def git_ahead_behind_counts(nfsim_repo: Path, left: str, right: str) -> tuple[int, int]:
    output = git(nfsim_repo, ["rev-list", "--left-right", "--count", f"{left}...{right}"])
    left_count, right_count = output.split()
    return int(left_count), int(right_count)


def git_changed_paths(nfsim_repo: Path, left: str, right: str) -> list[str]:
    return git_lines(nfsim_repo, ["diff", "--name-only", f"{left}...{right}"])


def format_resolved_ref(requested_ref: str | None, resolved_ref: str, commit: str) -> str:
    if requested_ref and requested_ref != resolved_ref:
        return f"{resolved_ref} (resolved from {requested_ref}) @ {commit}"
    return f"{resolved_ref} @ {commit}"


def default_summary_baseline() -> tuple[str, str, str | None] | None:
    current_metadata = load_vendor_metadata()
    if not current_metadata:
        return None

    source = current_metadata.get("source", {})
    commit = source.get("commit")
    if not commit:
        return None

    label = source.get("branch_or_ref") or commit
    base_commit = source.get("base_commit")
    return label, commit, base_commit


def print_preview(header: str, items: list[str], limit: int = SUMMARY_PREVIEW_LIMIT) -> None:
    if not items:
        return
    preview = items[:limit]
    print(f"{header} ({len(preview)} of {len(items)}):")
    for item in preview:
        print(f"  {item}")


def print_summary(
    nfsim_repo: Path,
    requested_ref: str,
    resolved_ref: str,
    commit: str,
    differences: list[str],
    compare_ref: str | None,
) -> None:
    print("NFsim vendoring summary")
    print(f"Candidate ref: {format_resolved_ref(requested_ref, resolved_ref, commit)}")

    baseline = default_summary_baseline()
    if baseline is None:
        print(f"Current vendored source: unavailable (missing {METADATA_NAME})")
    else:
        current_label, current_commit, current_base_commit = baseline
        print(f"Current vendored source: {current_label} @ {current_commit}")
        if current_base_commit:
            print(f"Current vendored base commit: {current_base_commit}")

    print()
    print(f"Tree delta vs {VENDOR_DIR.relative_to(REPO_ROOT)}:")
    counts = difference_counts(differences)
    print(f"  total differences: {len(differences)}")
    print(f"  changed: {counts.get('changed', 0)}")
    print(f"  missing: {counts.get('missing', 0)}")
    print(f"  extra: {counts.get('extra', 0)}")
    if differences:
        print_preview("Sample tree differences", differences)
    else:
        print("  no differences")

    compare_label: str | None = None
    compare_commit: str | None = None
    compare_requested = compare_ref
    if compare_ref:
        compare_label, compare_commit = resolve_ref(nfsim_repo, compare_ref)
    elif baseline is not None:
        compare_label, compare_commit, _ = baseline

    if compare_label is None or compare_commit is None:
        return

    changed_paths = sorted(git_changed_paths(nfsim_repo, compare_commit, commit))
    vendored_paths = [path for path in changed_paths if is_vendored_surface_path(path)]
    left_count, right_count = git_ahead_behind_counts(nfsim_repo, compare_commit, commit)
    merge_base = git_merge_base(nfsim_repo, compare_commit, commit)

    print()
    print("Git range impact:")
    print(
        f"  compare ref: {format_resolved_ref(compare_requested, compare_label, compare_commit)}"
    )
    print(f"  merge base: {merge_base}")
    print(f"  compare-only commits: {left_count}")
    print(f"  candidate-only commits: {right_count}")
    print(f"  files changed: {len(changed_paths)}")
    print(f"  vendored-surface files changed: {len(vendored_paths)}")
    if vendored_paths:
        print_preview("Sample vendored-surface paths", vendored_paths)


def expected_carry_commit_count() -> int:
    return sum(len(topic["commits"]) for topic in CARRY_QUEUE)


def patch_series_files() -> list[Path]:
    return sorted(PATCHES_DIR.glob("*.patch"))


def patch_series_summaries() -> list[str]:
    summaries: list[str] = []
    for path in patch_series_files():
        with path.open() as fh:
            subject_parts: list[str] = []
            for line in fh:
                if line.startswith("Subject:"):
                    subject = line[len("Subject:") :].strip()
                    subject = re.sub(r"^\[PATCH[^\]]*\]\s*", "", subject)
                    subject_parts.append(subject)
                    continue
                if subject_parts:
                    if line.startswith(" "):
                        subject_parts.append(line.strip())
                        continue
                    break
        summaries.append(" ".join(subject_parts))
    return summaries


def verify_carry_queue_matches_patches() -> list[str]:
    """Cross-check CARRY_QUEUE against bngsim/scripts/nfsim_vendor_patches/.

    Returns a list of human-readable problems (empty if consistent).
    """
    problems: list[str] = []
    patches = patch_series_files()
    expected_n = expected_carry_commit_count()
    if len(patches) != expected_n:
        problems.append(
            f"patch series has {len(patches)} *.patch files but CARRY_QUEUE "
            f"lists {expected_n} commits — they must match"
        )

    patch_summaries = patch_series_summaries()
    patch_summary_set = {s for s in patch_summaries if s}
    for topic in CARRY_QUEUE:
        for entry in topic["commits"]:
            summary = entry.get("summary", "")
            if summary and summary not in patch_summary_set:
                problems.append(
                    f"CARRY_QUEUE topic {topic['topic']!r} references commit "
                    f"summary {summary!r}, but no patch in the series has that "
                    f"Subject line — has the patch been re-exported after a "
                    f"summary change?"
                )
    return problems


def rebuild_candidate_from_patches(nfsim_repo: Path) -> tuple[Path, str]:
    """Replay the patch series on top of origin/master in a temp checkout.

    Returns the temp checkout path and the resulting commit hash.
    """
    base_ref, base_commit = upstream_base_metadata(nfsim_repo, "HEAD")
    if base_commit is None:
        raise RuntimeError(
            f"Could not resolve upstream base ref {base_ref!r} in {nfsim_repo}. "
            "Ensure the canonical RuleWorld remote is configured and fetched."
        )

    tmp_dir = tempfile.mkdtemp(prefix="vendor-nfsim-strict-")
    tmp_repo = Path(tmp_dir) / "nfsim"
    run(["git", "clone", "--quiet", str(nfsim_repo), str(tmp_repo)])
    run(["git", "-C", str(tmp_repo), "checkout", "--quiet", "-B", "bngsim/vendor", base_commit])
    patches = patch_series_files()
    if not patches:
        raise RuntimeError(f"No *.patch files found in {PATCHES_DIR}")
    am_cmd = ["git", "-C", str(tmp_repo), "am", "--quiet", *[str(p) for p in patches]]
    completed = subprocess.run(am_cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        subprocess.run(["git", "-C", str(tmp_repo), "am", "--abort"], check=False)
        raise RuntimeError(
            "Patch series did not apply cleanly on top of "
            f"{base_ref} @ {base_commit}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    tip = git(tmp_repo, ["rev-parse", "HEAD"])
    return tmp_repo, tip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nfsim-repo",
        type=Path,
        default=DEFAULT_NFSIM_REPO,
        help="Local NFsim Git checkout to vendor from (default: /tmp/nfsim-vendor-candidate).",
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_VENDOR_REF,
        help=(
            "NFsim branch, tag, or commit to vendor. If a local branch is missing, "
            "remote-tracking refs (prefer fork/, then origin/) are tried automatically."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the generated tree with the current vendored tree without writing.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Report tree and git-range impact without writing.",
    )
    parser.add_argument(
        "--compare-ref",
        help=(
            "NFsim ref used as the git-range baseline in summary output. "
            f"Defaults to the current vendored source recorded in {METADATA_NAME}."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh even when the vendored tree has uncommitted changes.",
    )
    parser.add_argument(
        "--verify-clean",
        action="store_true",
        help=(
            "Tripwire mode: byte-compare bngsim/third_party/nfsim/ against the "
            "tree the carry queue produces, plus cross-check that "
            "bngsim/scripts/nfsim_vendor_patches/ matches CARRY_QUEUE. "
            "Exits non-zero on any drift — direct edits to the vendored tree "
            "are not allowed. Intended for CI."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "When combined with --verify-clean, also rebuild the candidate "
            "from origin/master + the patch series in a temp checkout and "
            "confirm the resulting tree matches the live candidate's "
            "bngsim/vendor tip. Verifies that the patch series is "
            "self-contained and reproducible."
        ),
    )
    args = parser.parse_args()
    if args.compare_ref:
        args.summary = True
    if args.strict and not args.verify_clean:
        parser.error("--strict requires --verify-clean")
    return args


def run_verify_clean(args: argparse.Namespace) -> int:
    nfsim_repo = args.nfsim_repo.expanduser().resolve()
    if not nfsim_repo.exists():
        print(
            f"NFsim checkout does not exist: {nfsim_repo}\n"
            f"Rebuild via the recipe in bngsim/scripts/NFSIM_VENDORING.md "
            '("Rebuild The Candidate Checkout"). Without it the tripwire '
            "cannot confirm what the carry queue would produce.",
            file=sys.stderr,
        )
        return 1

    resolved_ref, commit = resolve_ref(nfsim_repo, args.ref)
    rel_vendor = VENDOR_DIR.relative_to(REPO_ROOT)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="vendor-nfsim-verify-") as tmp:
        source = export_nfsim_tree(nfsim_repo, commit, Path(tmp))
        differences = compare_trees(source, VENDOR_DIR)
        if differences:
            lines = [f"DIRECT EDIT DETECTED in {rel_vendor}:"]
            for diff in differences[:80]:
                lines.append(f"  {diff}")
            if len(differences) > 80:
                lines.append(f"  ... {len(differences) - 80} more")
            lines.append("")
            lines.append(TRIPWIRE_REMEDIATION)
            failures.append("\n".join(lines))

    carry_problems = verify_carry_queue_matches_patches()
    if carry_problems:
        lines = ["CARRY_QUEUE inconsistent with patch series:"]
        for problem in carry_problems:
            lines.append(f"  - {problem}")
        failures.append("\n".join(lines))

    if args.strict:
        try:
            tmp_repo, rebuilt_tip = rebuild_candidate_from_patches(nfsim_repo)
        except RuntimeError as exc:
            failures.append(f"--strict reproducibility check failed: {exc}")
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="vendor-nfsim-rebuilt-") as tmp:
                    rebuilt_source = export_nfsim_tree(tmp_repo, rebuilt_tip, Path(tmp))
                    diff = compare_trees(rebuilt_source, VENDOR_DIR)
                    if diff:
                        lines = [
                            "PATCH SERIES NOT REPRODUCIBLE: replaying the patch "
                            "series against origin/master produces a tree that "
                            f"does not match {rel_vendor}:"
                        ]
                        for d in diff[:40]:
                            lines.append(f"  {d}")
                        if len(diff) > 40:
                            lines.append(f"  ... {len(diff) - 40} more")
                        failures.append("\n".join(lines))
            finally:
                shutil.rmtree(tmp_repo.parent, ignore_errors=True)

    if failures:
        for block in failures:
            print(block, file=sys.stderr)
            print("", file=sys.stderr)
        return 1

    n_patches = len(patch_series_files())
    msg = (
        f"{rel_vendor} matches {resolved_ref} @ {commit}; "
        f"CARRY_QUEUE consistent with {n_patches} *.patch files"
    )
    if args.strict:
        msg += "; patch series reproduces from origin/master"
    print(msg)
    return 0


def main() -> int:
    args = parse_args()
    if args.verify_clean:
        return run_verify_clean(args)

    nfsim_repo = args.nfsim_repo.expanduser().resolve()
    if not nfsim_repo.exists():
        raise RuntimeError(f"NFsim checkout does not exist: {nfsim_repo}")

    resolved_ref, commit = resolve_ref(nfsim_repo, args.ref)
    resolution_note = f" (resolved from {args.ref})" if resolved_ref != args.ref else ""

    with tempfile.TemporaryDirectory(prefix="vendor-nfsim-") as tmp:
        source = export_nfsim_tree(nfsim_repo, commit, Path(tmp))
        differences = compare_trees(source, VENDOR_DIR) if (args.check or args.summary) else []
        if args.summary:
            print_summary(
                nfsim_repo, args.ref, resolved_ref, commit, differences, args.compare_ref
            )
            if not args.check:
                return 0
        if args.check:
            if differences:
                print(
                    f"{VENDOR_DIR.relative_to(REPO_ROOT)} differs from "
                    f"{resolved_ref}{resolution_note} @ {commit}:"
                )
                for diff in differences[:80]:
                    print(f"  {diff}")
                if len(differences) > 80:
                    print(f"  ... {len(differences) - 80} more")
                return 1
            print(
                f"{VENDOR_DIR.relative_to(REPO_ROOT)} matches {resolved_ref}{resolution_note} @ {commit}"
            )
            return 0

        ensure_clean_destination(args.force)
        copy_tree(source, VENDOR_DIR)
        write_metadata(nfsim_repo, resolved_ref, commit)
        print(
            f"Refreshed {VENDOR_DIR.relative_to(REPO_ROOT)} from "
            f"{resolved_ref}{resolution_note} @ {commit}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"vendor_nfsim.py: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
