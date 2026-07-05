#!/usr/bin/env python3
"""Refresh or summarize the vendored RuleMonkey tree from a Git checkout.

The authoritative source is richardposner/RuleMonkey `main`. BNGsim vendors
only an explicit allowlist into bngsim/third_party/rulemonkey and never
RuleMonkey's third_party/ tree.
"""

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
BNGSIM_ROOT = REPO_ROOT / "bngsim"
VENDOR_DIR = BNGSIM_ROOT / "third_party" / "rulemonkey"
METADATA_NAME = "VENDOR.json"
DEFAULT_RULEMONKEY_REPO = Path("/tmp/rulemonkey-vendor-candidate")
DEFAULT_VENDOR_REF = "main"
DEFAULT_UPSTREAM_REMOTE = "https://github.com/richardposner/RuleMonkey.git"
SUMMARY_PREVIEW_LIMIT = 12
DISALLOWED_EXPORT_PREFIXES = ("third_party",)
PIN_PREFIX = "Pinned commit: "
RULEMONKEY_EXPRTK_VENDOR_NOTE = Path("third_party") / "bngsim_expr" / "VENDOR"
RULEMONKEY_MAIN_TARGET_STRIP_INCLUDE = (
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bngsim_expr/include"
)
REQUIRED_RULEMONKEY_CMAKE_SNIPPETS = (
    "if(TARGET bngsim::expression)",
    "target_link_libraries(rulemonkey PRIVATE bngsim::expression)",
    "third_party/bngsim_expr/src/expression.cpp",
    "third_party/exprtk",
)

# Allowlist of RuleMonkey paths copied into bngsim/third_party/rulemonkey/.
#
# `third_party/` is DELIBERATELY omitted. Since RuleMonkey#6 (the ExprTk
# swap), RuleMonkey vendors its own copies of ExprTk and bngsim's expression
# evaluator under third_party/exprtk/ + third_party/bngsim_expr/ — verbatim
# copies of bngsim's own third_party/exprtk/exprtk.hpp + src/expression.cpp.
# Copying those into bngsim would compile bngsim::ExprTkEvaluator twice →
# duplicate-symbol / ODR hazard across two static archives.
INCLUDED_PATHS = (
    "CMakeLists.txt",
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    "cmake",
    "cpp",
    "docs",
    "include",
)

PRUNED_PATHS = ("cpp/cli",)

# Files that RuleMonkey vendors verbatim from bngsim under third_party/bngsim_expr/
# (+ third_party/exprtk/) and must keep byte-identical to this tree. The drift
# guard (exprtk_sync_drift) compares each pair before vendoring; the metadata
# block's verified_against_bngsim_paths is derived from this map so the two never
# diverge. expr_compat.hpp is a dependency of expression.cpp since GH #49
# (expression.cpp #includes it and defines the expr_compat:: free functions it
# declares), so it must be tracked here too — a drift in it would otherwise slip
# the guard while breaking RuleMonkey's standalone bngsim_expr build.
EXPRTK_SYNC_FILES = {
    "third_party/exprtk/exprtk.hpp": BNGSIM_ROOT / "third_party" / "exprtk" / "exprtk.hpp",
    "third_party/bngsim_expr/include/bngsim/expression.hpp": (
        BNGSIM_ROOT / "include" / "bngsim" / "expression.hpp"
    ),
    "third_party/bngsim_expr/include/bngsim/expr_compat.hpp": (
        BNGSIM_ROOT / "include" / "bngsim" / "expr_compat.hpp"
    ),
    "third_party/bngsim_expr/src/expression.cpp": BNGSIM_ROOT / "src" / "expression.cpp",
}


def run(cmd: list[str], cwd: Path | None = None, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=text,
    )


def git(rulemonkey_repo: Path, args: list[str]) -> str:
    return run(["git", "-C", str(rulemonkey_repo), *args]).stdout.strip()


def git_lines(rulemonkey_repo: Path, args: list[str]) -> list[str]:
    output = git(rulemonkey_repo, args)
    return [line for line in output.splitlines() if line]


def normalize_remote_url(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    normalized = normalized.removesuffix(".git").rstrip("/")
    return normalized


def is_canonical_upstream_remote(url: str | None) -> bool:
    return normalize_remote_url(url) == normalize_remote_url(DEFAULT_UPSTREAM_REMOTE)


def remote_url(rulemonkey_repo: Path, remote: str) -> str | None:
    try:
        return git(rulemonkey_repo, ["remote", "get-url", remote])
    except subprocess.CalledProcessError:
        return None


def canonical_remote_info(rulemonkey_repo: Path) -> tuple[str | None, str | None]:
    remotes = git_lines(rulemonkey_repo, ["remote"])
    canonical = [
        remote
        for remote in remotes
        if is_canonical_upstream_remote(remote_url(rulemonkey_repo, remote))
    ]
    ordered: list[str] = []
    for preferred in ("origin", "upstream"):
        if preferred in canonical:
            ordered.append(preferred)
    for remote in canonical:
        if remote not in ordered:
            ordered.append(remote)
    if not ordered:
        return None, None
    remote = ordered[0]
    return remote, remote_url(rulemonkey_repo, remote)


def ordered_remotes(rulemonkey_repo: Path) -> list[str]:
    remotes = git_lines(rulemonkey_repo, ["remote"])
    if not remotes:
        return []

    ordered: list[str] = []
    canonical_remote, _ = canonical_remote_info(rulemonkey_repo)
    if canonical_remote:
        ordered.append(canonical_remote)

    for preferred in ("origin", "upstream"):
        if preferred in remotes and preferred not in ordered:
            ordered.append(preferred)

    for remote in remotes:
        if remote not in ordered:
            ordered.append(remote)
    return ordered


def ref_candidates(rulemonkey_repo: Path, ref: str) -> list[str]:
    if ref.startswith("refs/"):
        return [ref]

    remotes = ordered_remotes(rulemonkey_repo)
    if not remotes:
        return [ref]

    ref_prefix = ref.split("/", 1)[0]
    if ref_prefix in remotes:
        return [ref]

    candidates: list[str] = []
    for remote in remotes:
        candidate = f"{remote}/{ref}"
        if candidate not in candidates:
            candidates.append(candidate)
    if ref not in candidates:
        candidates.append(ref)
    return candidates


def resolve_ref(rulemonkey_repo: Path, ref: str) -> tuple[str, str]:
    attempted: list[str] = []
    for candidate in ref_candidates(rulemonkey_repo, ref):
        try:
            commit = git(rulemonkey_repo, ["rev-parse", "--verify", f"{candidate}^{{commit}}"])
            return candidate, commit
        except subprocess.CalledProcessError:
            attempted.append(candidate)

    attempted_text = ", ".join(attempted) if attempted else ref
    raise RuntimeError(f"Could not resolve RuleMonkey ref '{ref}'. Tried: {attempted_text}")


def working_tree_changes(rulemonkey_repo: Path) -> list[str]:
    return git_lines(rulemonkey_repo, ["status", "--short"])


def read_exprtk_vendor_pin(rulemonkey_repo: Path) -> str:
    vendor_note = rulemonkey_repo / RULEMONKEY_EXPRTK_VENDOR_NOTE
    if not vendor_note.exists():
        raise RuntimeError(f"Missing RuleMonkey vendor note: {vendor_note}")
    for line in vendor_note.read_text().splitlines():
        if line.startswith(PIN_PREFIX):
            return line[len(PIN_PREFIX) :].strip()
    raise RuntimeError(f"Could not find '{PIN_PREFIX}' in {vendor_note}")


def exprtk_sync_drift(rulemonkey_repo: Path) -> list[str]:
    drift: list[str] = []
    for rulemonkey_rel, bngsim_path in EXPRTK_SYNC_FILES.items():
        rulemonkey_path = rulemonkey_repo / rulemonkey_rel
        if not rulemonkey_path.exists():
            raise RuntimeError(f"Missing RuleMonkey ExprTk sync file: {rulemonkey_path}")
        if not bngsim_path.exists():
            raise RuntimeError(f"Missing BNGsim ExprTk sync file: {bngsim_path}")
        if rulemonkey_path.read_bytes() != bngsim_path.read_bytes():
            drift.append(f"{rulemonkey_rel} != {bngsim_path.relative_to(REPO_ROOT)}")
    return drift


def verify_source_checkout(rulemonkey_repo: Path) -> dict[str, str]:
    try:
        inside_work_tree = git(rulemonkey_repo, ["rev-parse", "--is-inside-work-tree"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"RuleMonkey checkout is not a Git worktree: {rulemonkey_repo}"
        ) from exc

    if inside_work_tree != "true":
        raise RuntimeError(f"RuleMonkey checkout is not a Git worktree: {rulemonkey_repo}")

    canonical_remote, canonical_remote_url = canonical_remote_info(rulemonkey_repo)
    if canonical_remote is None or canonical_remote_url is None:
        raise RuntimeError(
            "RuleMonkey checkout does not point at the canonical upstream remote "
            f"{DEFAULT_UPSTREAM_REMOTE}. Refresh /tmp/rulemonkey-vendor-candidate from upstream first."
        )

    changes = working_tree_changes(rulemonkey_repo)
    if changes:
        preview = "\n  ".join(changes[:10])
        extra = "" if len(changes) <= 10 else f"\n  ... {len(changes) - 10} more"
        raise RuntimeError(
            "RuleMonkey checkout has local changes. Vendor only from a clean candidate checkout.\n"
            f"  {preview}{extra}"
        )

    canonical_main_ref = f"{canonical_remote}/{DEFAULT_VENDOR_REF}"
    try:
        canonical_main_commit = git(
            rulemonkey_repo, ["rev-parse", "--verify", f"{canonical_main_ref}^{{commit}}"]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"RuleMonkey checkout is missing {canonical_main_ref}. Fetch/prune the canonical remote first."
        ) from exc

    exprtk_pin = read_exprtk_vendor_pin(rulemonkey_repo)
    drift = exprtk_sync_drift(rulemonkey_repo)
    if drift:
        drift_preview = "\n  ".join(drift)
        raise RuntimeError(
            "RuleMonkey standalone ExprTk/bngsim_expr vendoring has drifted from this BNGsim tree.\n"
            f"  Pinned BNGsim commit: {exprtk_pin}\n"
            f"  Drifted files:\n  {drift_preview}\n"
            "Refresh upstream RuleMonkey's third_party/bngsim_expr pin before vendoring it into BNGsim."
        )

    return {
        "canonical_remote": canonical_remote,
        "canonical_remote_url": canonical_remote_url,
        "canonical_main_ref": canonical_main_ref,
        "canonical_main_commit": canonical_main_commit,
        "head_ref": git(rulemonkey_repo, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "head_commit": git(rulemonkey_repo, ["rev-parse", "HEAD"]),
        "exprtk_pin": exprtk_pin,
    }


def path_in_scope(rel_path: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel_path == prefix or rel_path.startswith(f"{prefix}/") for prefix in prefixes)


def validate_export_contract() -> None:
    offenders = [
        path
        for path in (*INCLUDED_PATHS, *PRUNED_PATHS)
        if path_in_scope(path, DISALLOWED_EXPORT_PREFIXES)
    ]
    if offenders:
        joined = ", ".join(sorted(offenders))
        raise RuntimeError(
            "RuleMonkey export contract must never include or post-prune third_party/. "
            f"Offending paths: {joined}"
        )


def safe_extract_tar(data: bytes, destination: Path) -> None:
    archive_path = destination / "rulemonkey.tar"
    archive_path.write_bytes(data)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = destination / member.name
            try:
                member_path.resolve().relative_to(destination.resolve())
            except ValueError as exc:
                raise RuntimeError(f"Refusing unsafe archive path: {member.name}") from exc
        if sys.version_info >= (3, 12):
            archive.extractall(destination, filter="data")
        else:
            archive.extractall(destination)
    archive_path.unlink()


def extract_cmake_call_block(text: str, marker: str) -> tuple[int, int, str]:
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find '{marker}' in vendored RuleMonkey CMakeLists.txt")

    depth = 0
    seen_open = False
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
            seen_open = True
        elif char == ")":
            depth -= 1
            if seen_open and depth == 0:
                return start, index + 1, text[start : index + 1]

    raise RuntimeError(f"Could not parse '{marker}' in vendored RuleMonkey CMakeLists.txt")


def normalize_vendored_cmake(source_dir: Path) -> bool:
    cmake_path = source_dir / "CMakeLists.txt"
    text = cmake_path.read_text()
    start, end, block = extract_cmake_call_block(text, "target_include_directories(rulemonkey")
    pattern = rf"(?m)^[ \t]*{re.escape(RULEMONKEY_MAIN_TARGET_STRIP_INCLUDE)}\s*\n"
    normalized_block, count = re.subn(pattern, "", block, count=1)
    if count == 0:
        return False
    cmake_path.write_text(text[:start] + normalized_block + text[end:])
    return True


def validate_exported_tree(source_dir: Path) -> None:
    missing_required = [
        rel_path
        for rel_path in ("CMakeLists.txt", "include/rulemonkey/simulator.hpp")
        if not (source_dir / rel_path).exists()
    ]
    if missing_required:
        joined = ", ".join(missing_required)
        raise RuntimeError(f"Vendored RuleMonkey export is missing required files: {joined}")

    disallowed_paths = [
        str(path.relative_to(source_dir))
        for path in source_dir.rglob("*")
        if path_in_scope(str(path.relative_to(source_dir)), DISALLOWED_EXPORT_PREFIXES)
    ]
    if disallowed_paths:
        preview = ", ".join(sorted(disallowed_paths)[:12])
        raise RuntimeError(
            "Vendored RuleMonkey export unexpectedly contains third_party/. "
            f"Sample paths: {preview}"
        )

    cmake_path = source_dir / "CMakeLists.txt"
    text = cmake_path.read_text()
    for snippet in REQUIRED_RULEMONKEY_CMAKE_SNIPPETS:
        if snippet not in text:
            raise RuntimeError(
                "Vendored RuleMonkey CMake no longer contains the expected "
                f"bngsim::expression handoff snippet: {snippet}"
            )

    _, _, rulemonkey_include_block = extract_cmake_call_block(
        text, "target_include_directories(rulemonkey"
    )
    if "third_party/bngsim_expr/include" in rulemonkey_include_block:
        raise RuntimeError(
            "Vendored RuleMonkey main target still references "
            "third_party/bngsim_expr/include. BNGsim omits RuleMonkey/third_party "
            "and must rely on the host bngsim::expression target instead."
        )


def export_rulemonkey_tree(rulemonkey_repo: Path, commit: str, destination: Path) -> Path:
    source_dir = destination / "source"
    source_dir.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(rulemonkey_repo), "archive", "--format=tar", commit, *INCLUDED_PATHS],
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

    normalize_vendored_cmake(source_dir)
    validate_exported_tree(source_dir)
    return source_dir


def ensure_clean_destination(force: bool) -> None:
    if force:
        return
    rel_vendor = VENDOR_DIR.relative_to(REPO_ROOT)
    status = run(["git", "status", "--porcelain", "--", str(rel_vendor)], cwd=REPO_ROOT).stdout
    if status.strip():
        raise RuntimeError(
            f"{rel_vendor} has uncommitted changes. Commit/stash them or rerun with --force."
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


def metadata(rulemonkey_repo: Path, ref: str, commit: str, checkout_info: dict[str, str]) -> dict:
    # Record the local checkout with the home-dir prefix scrubbed to ``~`` so the
    # committed VENDOR.json stays machine- and username-independent (the commit +
    # authoritative_remote are what actually pin the vendored source).
    home = str(Path.home())
    local_checkout = str(rulemonkey_repo)
    if local_checkout.startswith(home):
        local_checkout = "~" + local_checkout[len(home) :]
    return {
        "name": "RuleMonkey",
        "vendored_path": "bngsim/third_party/rulemonkey",
        "source": {
            "authoritative_remote": DEFAULT_UPSTREAM_REMOTE,
            "authoritative_branch": DEFAULT_VENDOR_REF,
            "local_checkout": local_checkout,
            "upstream_remote": DEFAULT_UPSTREAM_REMOTE,
            "local_origin_remote": remote_url(rulemonkey_repo, "origin"),
            "local_upstream_remote": remote_url(rulemonkey_repo, "upstream"),
            "fork_remote": remote_url(rulemonkey_repo, "fork"),
            "branch_or_ref": ref,
            "commit": commit,
        },
        "imported_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pruning": {
            "included_paths": list(INCLUDED_PATHS),
            "excluded_paths": list(PRUNED_PATHS),
        },
        "exprtk_sync": {
            "rulemonkey_vendor_note": str(RULEMONKEY_EXPRTK_VENDOR_NOTE),
            "pinned_bngsim_commit": checkout_info["exprtk_pin"],
            "verified_against_bngsim_paths": [
                str(p.relative_to(REPO_ROOT)) for p in EXPRTK_SYNC_FILES.values()
            ],
        },
        "normalizations": [
            "Removed RuleMonkey's unconditional third_party/bngsim_expr include path from the rulemonkey target.",
        ],
        "notes": [
            "RuleMonkey is consumed through add_subdirectory(third_party/rulemonkey).",
            "BNGsim disables RuleMonkey CLI, tests, examples, and install rules at configure time.",
            "RuleMonkey third_party/ is intentionally not vendored into BNGsim.",
            "BNGsim requires the host-target handoff through bngsim::expression to avoid duplicate ExprTkEvaluator symbols.",
        ],
    }


def write_metadata(
    rulemonkey_repo: Path, ref: str, commit: str, checkout_info: dict[str, str]
) -> None:
    data = metadata(rulemonkey_repo, ref, commit, checkout_info)
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


def is_vendored_surface_path(rel_path: str) -> bool:
    return path_in_scope(rel_path, INCLUDED_PATHS) and not path_in_scope(rel_path, PRUNED_PATHS)


def git_merge_base(rulemonkey_repo: Path, left: str, right: str) -> str:
    return git(rulemonkey_repo, ["merge-base", left, right])


def git_ahead_behind_counts(rulemonkey_repo: Path, left: str, right: str) -> tuple[int, int]:
    output = git(rulemonkey_repo, ["rev-list", "--left-right", "--count", f"{left}...{right}"])
    left_count, right_count = output.split()
    return int(left_count), int(right_count)


def git_changed_paths(rulemonkey_repo: Path, left: str, right: str) -> list[str]:
    return git_lines(rulemonkey_repo, ["diff", "--name-only", f"{left}...{right}"])


def format_resolved_ref(requested_ref: str | None, resolved_ref: str, commit: str) -> str:
    if requested_ref and requested_ref != resolved_ref:
        return f"{resolved_ref} (resolved from {requested_ref}) @ {commit}"
    return f"{resolved_ref} @ {commit}"


def default_summary_baseline() -> tuple[str, str] | None:
    current_metadata = load_vendor_metadata()
    if not current_metadata:
        return None

    source = current_metadata.get("source", {})
    commit = source.get("commit")
    if not commit:
        return None

    label = (
        source.get("branch_or_ref")
        or source.get("authoritative_branch")
        or source.get("authoritative_remote")
        or commit
    )
    return label, commit


def print_preview(header: str, items: list[str], limit: int = SUMMARY_PREVIEW_LIMIT) -> None:
    if not items:
        return
    preview = items[:limit]
    print(f"{header} ({len(preview)} of {len(items)}):")
    for item in preview:
        print(f"  {item}")


def print_summary(
    rulemonkey_repo: Path,
    requested_ref: str,
    resolved_ref: str,
    commit: str,
    differences: list[str],
    compare_ref: str | None,
    checkout_info: dict[str, str],
) -> None:
    print("RuleMonkey vendoring summary")
    print(f"Candidate checkout: {rulemonkey_repo}")
    print(
        f"Canonical remote: {checkout_info['canonical_remote']} -> "
        f"{checkout_info['canonical_remote_url']}"
    )
    print(f"Candidate HEAD: {checkout_info['head_ref']} @ {checkout_info['head_commit']}")
    print(
        f"Canonical upstream main: {checkout_info['canonical_main_ref']} @ {checkout_info['canonical_main_commit']}"
    )
    print(f"ExprTk sync pin: {checkout_info['exprtk_pin']} (matches current BNGsim tree)")
    print(f"Candidate ref: {format_resolved_ref(requested_ref, resolved_ref, commit)}")

    baseline = default_summary_baseline()
    if baseline is None:
        print(f"Current vendored source: unavailable (missing {METADATA_NAME})")
    else:
        current_label, current_commit = baseline
        print(f"Current vendored source: {current_label} @ {current_commit}")

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
        compare_label, compare_commit = resolve_ref(rulemonkey_repo, compare_ref)
    elif baseline is not None:
        compare_label, compare_commit = baseline

    if compare_label is None or compare_commit is None:
        return

    changed_paths = sorted(git_changed_paths(rulemonkey_repo, compare_commit, commit))
    vendored_paths = [path for path in changed_paths if is_vendored_surface_path(path)]
    left_count, right_count = git_ahead_behind_counts(rulemonkey_repo, compare_commit, commit)
    merge_base = git_merge_base(rulemonkey_repo, compare_commit, commit)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rulemonkey-repo",
        type=Path,
        default=DEFAULT_RULEMONKEY_REPO,
        help=(f"RuleMonkey Git checkout to vendor from (default: {DEFAULT_RULEMONKEY_REPO})."),
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_VENDOR_REF,
        help=(
            "RuleMonkey branch, tag, or commit to vendor. Unqualified refs resolve against "
            "the canonical upstream remote before local branches."
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
        help="Print a no-write summary of the vendoring impact before refreshing.",
    )
    parser.add_argument(
        "--compare-ref",
        help=(
            "RuleMonkey ref used as the git-range baseline in summary output. "
            "Defaults to the current vendored commit from VENDOR.json."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh even when the vendored tree has uncommitted changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rulemonkey_repo = args.rulemonkey_repo.expanduser()
    if not rulemonkey_repo.is_absolute():
        rulemonkey_repo = (Path.cwd() / rulemonkey_repo).resolve()
    if not rulemonkey_repo.exists():
        raise RuntimeError(f"RuleMonkey checkout does not exist: {rulemonkey_repo}")

    validate_export_contract()
    checkout_info = verify_source_checkout(rulemonkey_repo)
    resolved_ref, commit = resolve_ref(rulemonkey_repo, args.ref)
    if args.ref == DEFAULT_VENDOR_REF and commit != checkout_info["canonical_main_commit"]:
        raise RuntimeError(
            f"Requested ref '{args.ref}' resolved to {commit}, but canonical upstream main is "
            f"{checkout_info['canonical_main_ref']} @ {checkout_info['canonical_main_commit']}. "
            "Fetch/prune the candidate checkout and retry."
        )

    resolution_note = f" (resolved from {args.ref})" if resolved_ref != args.ref else ""

    with tempfile.TemporaryDirectory(prefix="vendor-rulemonkey-") as tmp:
        source = export_rulemonkey_tree(rulemonkey_repo, commit, Path(tmp))
        differences = compare_trees(source, VENDOR_DIR) if (args.check or args.summary) else []

        if args.summary:
            print_summary(
                rulemonkey_repo,
                args.ref,
                resolved_ref,
                commit,
                differences,
                args.compare_ref,
                checkout_info,
            )
            if not args.check:
                print()

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
                f"{VENDOR_DIR.relative_to(REPO_ROOT)} matches "
                f"{resolved_ref}{resolution_note} @ {commit}"
            )
            return 0

        ensure_clean_destination(args.force)
        copy_tree(source, VENDOR_DIR)
        write_metadata(rulemonkey_repo, resolved_ref, commit, checkout_info)
        print(
            f"Refreshed {VENDOR_DIR.relative_to(REPO_ROOT)} from "
            f"{resolved_ref}{resolution_note} @ {commit}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"vendor_rulemonkey.py: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
