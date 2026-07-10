#!/usr/bin/env python3
"""Refresh or summarize the vendored ExprTk header from a Git checkout.

BNGsim treats the author-maintained GitHub mirror as the canonical vendoring
input because it provides immutable commits and a restartable checkout-based
workflow. The official project homepage and download remain on partow.net.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BNGSIM_ROOT = REPO_ROOT / "bngsim"
VENDOR_DIR = BNGSIM_ROOT / "third_party" / "exprtk"
HEADER_NAME = "exprtk.hpp"
METADATA_NAME = "VENDOR.json"
DEFAULT_EXPRTK_REPO = Path("/tmp/exprtk-vendor-candidate")
DEFAULT_VENDOR_REF = "master"
DEFAULT_UPSTREAM_REMOTE = "https://github.com/ArashPartow/exprtk.git"
OFFICIAL_PROJECT_HOMEPAGE = "https://www.partow.net/programming/exprtk/index.html"
OFFICIAL_PROJECT_DOWNLOAD = "https://www.partow.net/downloads/exprtk.zip"
BNGSIM_WRAPPER_ALIASES = ("ln", "rint", "sign", "mratio", "time")
REQUIRED_HEADER_TOKENS = (
    "static const std::string reserved_words[]",
    "static const std::string reserved_symbols[]",
    "#ifndef exprtk_disable_caseinsensitivity",
    "#ifndef exprtk_disable_string_capabilities",
    "#ifndef exprtk_disable_rtl_io_file",
    "#ifndef exprtk_disable_rtl_vecops",
    "class ifunction : public function_traits",
    "allow_zero_parameters()",
    "set_max_stack_depth",
)


def run(cmd: list[str], cwd: Path | None = None, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=text,
    )


def git(exprtk_repo: Path, args: list[str]) -> str:
    return run(["git", "-C", str(exprtk_repo), *args]).stdout.strip()


def git_lines(exprtk_repo: Path, args: list[str]) -> list[str]:
    output = git(exprtk_repo, args)
    return [line for line in output.splitlines() if line]


def normalize_remote_url(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.removesuffix(".git").rstrip("/")


def is_canonical_upstream_remote(url: str | None) -> bool:
    return normalize_remote_url(url) == normalize_remote_url(DEFAULT_UPSTREAM_REMOTE)


def remote_url(exprtk_repo: Path, remote: str) -> str | None:
    try:
        return git(exprtk_repo, ["remote", "get-url", remote])
    except subprocess.CalledProcessError:
        return None


def canonical_remote_info(exprtk_repo: Path) -> tuple[str | None, str | None]:
    remotes = git_lines(exprtk_repo, ["remote"])
    canonical = [
        remote
        for remote in remotes
        if is_canonical_upstream_remote(remote_url(exprtk_repo, remote))
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
    return remote, remote_url(exprtk_repo, remote)


def ordered_remotes(exprtk_repo: Path) -> list[str]:
    remotes = git_lines(exprtk_repo, ["remote"])
    if not remotes:
        return []

    ordered: list[str] = []
    canonical_remote, _ = canonical_remote_info(exprtk_repo)
    if canonical_remote:
        ordered.append(canonical_remote)

    for preferred in ("origin", "upstream"):
        if preferred in remotes and preferred not in ordered:
            ordered.append(preferred)

    for remote in remotes:
        if remote not in ordered:
            ordered.append(remote)
    return ordered


def ref_candidates(exprtk_repo: Path, ref: str) -> list[str]:
    if ref.startswith("refs/"):
        return [ref]

    remotes = ordered_remotes(exprtk_repo)
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


def resolve_ref(exprtk_repo: Path, ref: str) -> tuple[str, str]:
    attempted: list[str] = []
    for candidate in ref_candidates(exprtk_repo, ref):
        try:
            commit = git(exprtk_repo, ["rev-parse", "--verify", f"{candidate}^{{commit}}"])
            return candidate, commit
        except subprocess.CalledProcessError:
            attempted.append(candidate)

    attempted_text = ", ".join(attempted) if attempted else ref
    raise RuntimeError(f"Could not resolve ExprTk ref '{ref}'. Tried: {attempted_text}")


def working_tree_changes(exprtk_repo: Path) -> list[str]:
    return git_lines(exprtk_repo, ["status", "--short"])


def verify_source_checkout(exprtk_repo: Path) -> dict[str, str]:
    try:
        inside_work_tree = git(exprtk_repo, ["rev-parse", "--is-inside-work-tree"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ExprTk checkout is not a Git worktree: {exprtk_repo}") from exc

    if inside_work_tree != "true":
        raise RuntimeError(f"ExprTk checkout is not a Git worktree: {exprtk_repo}")

    canonical_remote, canonical_remote_url = canonical_remote_info(exprtk_repo)
    if canonical_remote is None or canonical_remote_url is None:
        raise RuntimeError(
            "ExprTk checkout does not point at the canonical upstream remote "
            f"{DEFAULT_UPSTREAM_REMOTE}. Refresh /tmp/exprtk-vendor-candidate from upstream first."
        )

    changes = working_tree_changes(exprtk_repo)
    if changes:
        preview = "\n  ".join(changes[:10])
        extra = "" if len(changes) <= 10 else f"\n  ... {len(changes) - 10} more"
        raise RuntimeError(
            "ExprTk checkout has local changes. Vendor only from a clean candidate checkout.\n"
            f"  {preview}{extra}"
        )

    header_path = exprtk_repo / HEADER_NAME
    if not header_path.exists():
        raise RuntimeError(f"ExprTk checkout is missing {HEADER_NAME}: {header_path}")

    canonical_ref = f"{canonical_remote}/{DEFAULT_VENDOR_REF}"
    try:
        canonical_commit = git(
            exprtk_repo, ["rev-parse", "--verify", f"{canonical_ref}^{{commit}}"]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ExprTk checkout is missing {canonical_ref}. Fetch/prune the canonical remote first."
        ) from exc

    return {
        "canonical_remote": canonical_remote,
        "canonical_remote_url": canonical_remote_url,
        "canonical_ref": canonical_ref,
        "canonical_commit": canonical_commit,
        "head_commit": git(exprtk_repo, ["rev-parse", "HEAD"]),
    }


def read_vendor_metadata() -> dict | None:
    metadata_path = VENDOR_DIR / METADATA_NAME
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def header_bytes_from_commit(exprtk_repo: Path, commit: str) -> bytes:
    return run(
        ["git", "-C", str(exprtk_repo), "show", f"{commit}:{HEADER_NAME}"], text=False
    ).stdout


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_string_array(header_text: str, array_name: str) -> list[str]:
    pattern = re.compile(
        rf"static const std::string {re.escape(array_name)}\[\]\s*=\s*\{{(?P<body>.*?)\n\s*\}};",
        re.DOTALL,
    )
    match = pattern.search(header_text)
    if not match:
        raise RuntimeError(f"Could not extract ExprTk array '{array_name}' from {HEADER_NAME}")
    return re.findall(r'"([^"]+)"', match.group("body"))


def extract_header_metadata(header_bytes: bytes) -> dict:
    header_text = header_bytes.decode("utf-8")
    reserved_words = extract_string_array(header_text, "reserved_words")
    reserved_symbols = extract_string_array(header_text, "reserved_symbols")
    missing_tokens = [token for token in REQUIRED_HEADER_TOKENS if token not in header_text]

    if missing_tokens:
        missing_preview = "\n  ".join(missing_tokens)
        raise RuntimeError(
            "ExprTk header no longer exposes one or more wrapper-critical tokens.\n"
            f"  {missing_preview}\n"
            "Refresh the BNGsim compatibility analysis before vendoring this revision."
        )

    return {
        "sha256": sha256_hex(header_bytes),
        "bytes": len(header_bytes),
        "reserved_words": reserved_words,
        "reserved_symbols": reserved_symbols,
        "required_header_tokens": list(REQUIRED_HEADER_TOKENS),
    }


def header_diff_summary(vendored_header: Path, candidate_bytes: bytes) -> str:
    if not vendored_header.exists():
        return "vendored exprtk.hpp is missing"
    if vendored_header.read_bytes() == candidate_bytes:
        return "vendored exprtk.hpp already matches the resolved upstream ref"

    fd, temp_path = tempfile.mkstemp(prefix="exprtk-candidate-", suffix=".hpp")
    os.close(fd)
    try:
        Path(temp_path).write_bytes(candidate_bytes)
        diff = subprocess.run(
            ["git", "diff", "--no-index", "--shortstat", "--", temp_path, str(vendored_header)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if diff.returncode not in (0, 1):
            raise RuntimeError(diff.stdout.strip() or "git diff --no-index failed")
        return diff.stdout.strip() or "vendored exprtk.hpp differs"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def build_vendor_metadata(
    exprtk_repo: Path,
    checkout_info: dict[str, str],
    resolved_ref: str,
    commit: str,
    header_metadata: dict,
) -> dict:
    try:
        tags = git_lines(exprtk_repo, ["tag", "--points-at", commit])
    except subprocess.CalledProcessError:
        tags = []

    return {
        "name": "ExprTk",
        "vendored_path": "bngsim/third_party/exprtk",
        "source": {
            "authoritative_remote": DEFAULT_UPSTREAM_REMOTE,
            "authoritative_branch": DEFAULT_VENDOR_REF,
            "official_project_homepage": OFFICIAL_PROJECT_HOMEPAGE,
            "official_project_download": OFFICIAL_PROJECT_DOWNLOAD,
            "local_checkout": str(exprtk_repo),
            "upstream_remote": checkout_info["canonical_remote_url"],
            "local_origin_remote": remote_url(exprtk_repo, "origin"),
            "local_upstream_remote": remote_url(exprtk_repo, "upstream"),
            "fork_remote": None,
            "branch_or_ref": resolved_ref,
            "commit": commit,
            "tags": tags,
        },
        "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {
            "header": {
                "path": "bngsim/third_party/exprtk/exprtk.hpp",
                "sha256": header_metadata["sha256"],
                "bytes": header_metadata["bytes"],
            }
        },
        "guardrails": {
            "reserved_words": header_metadata["reserved_words"],
            "reserved_symbols": header_metadata["reserved_symbols"],
            "required_header_tokens": header_metadata["required_header_tokens"],
            "bngsim_wrapper_aliases": list(BNGSIM_WRAPPER_ALIASES),
        },
        "local_carries": [],
        "notes": [
            "The Partow site is ExprTk's official project homepage and download location.",
            "BNGsim pins the author-maintained GitHub mirror because it provides immutable commits for restartable vendoring.",
            "BNGsim-specific compatibility behavior lives in include/bngsim/expression.hpp and src/expression.cpp, not in exprtk.hpp.",
            "Refresh third_party/exprtk only through bngsim/scripts/vendor_exprtk.py.",
        ],
    }


def metadata_drift(
    vendor_metadata: dict | None,
    commit: str,
    header_metadata: dict,
) -> list[str]:
    if vendor_metadata is None:
        return [f"missing {VENDOR_DIR / METADATA_NAME}"]

    drift: list[str] = []
    source = vendor_metadata.get("source", {})
    header = vendor_metadata.get("files", {}).get("header", {})
    guardrails = vendor_metadata.get("guardrails", {})

    expectations = [
        (
            "source.authoritative_remote",
            source.get("authoritative_remote"),
            DEFAULT_UPSTREAM_REMOTE,
        ),
        ("source.authoritative_branch", source.get("authoritative_branch"), DEFAULT_VENDOR_REF),
        (
            "source.official_project_homepage",
            source.get("official_project_homepage"),
            OFFICIAL_PROJECT_HOMEPAGE,
        ),
        (
            "source.official_project_download",
            source.get("official_project_download"),
            OFFICIAL_PROJECT_DOWNLOAD,
        ),
        ("source.upstream_remote", source.get("upstream_remote"), DEFAULT_UPSTREAM_REMOTE),
        ("source.commit", source.get("commit"), commit),
        ("files.header.path", header.get("path"), "bngsim/third_party/exprtk/exprtk.hpp"),
        ("files.header.sha256", header.get("sha256"), header_metadata["sha256"]),
        ("files.header.bytes", header.get("bytes"), header_metadata["bytes"]),
        (
            "guardrails.reserved_words",
            guardrails.get("reserved_words"),
            header_metadata["reserved_words"],
        ),
        (
            "guardrails.reserved_symbols",
            guardrails.get("reserved_symbols"),
            header_metadata["reserved_symbols"],
        ),
        (
            "guardrails.required_header_tokens",
            guardrails.get("required_header_tokens"),
            header_metadata["required_header_tokens"],
        ),
        (
            "guardrails.bngsim_wrapper_aliases",
            guardrails.get("bngsim_wrapper_aliases"),
            list(BNGSIM_WRAPPER_ALIASES),
        ),
        ("local_carries", vendor_metadata.get("local_carries"), []),
    ]

    for label, actual, expected in expectations:
        if actual != expected:
            drift.append(f"{label}: expected {expected!r}, found {actual!r}")

    return drift


def write_refresh(exprtk_repo: Path, header_bytes: bytes, metadata: dict) -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    (VENDOR_DIR / HEADER_NAME).write_bytes(header_bytes)
    (VENDOR_DIR / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")


def print_summary(
    exprtk_repo: Path,
    checkout_info: dict[str, str],
    resolved_ref: str,
    commit: str,
    header_metadata: dict,
    vendor_metadata: dict | None,
) -> None:
    vendored_header = VENDOR_DIR / HEADER_NAME
    current_header_sha = (
        sha256_hex(vendored_header.read_bytes()) if vendored_header.exists() else "missing"
    )
    current_commit = None
    if vendor_metadata:
        current_commit = vendor_metadata.get("source", {}).get("commit")

    print(f"ExprTk candidate: {exprtk_repo}")
    print(
        "Canonical remote: "
        f"{checkout_info['canonical_remote']} -> {checkout_info['canonical_remote_url']}"
    )
    print(f"Candidate HEAD: {checkout_info['head_commit']}")
    print(f"Resolved source ref: {resolved_ref}")
    print(f"Resolved source commit: {commit}")
    print(f"Official homepage: {OFFICIAL_PROJECT_HOMEPAGE}")
    print(f"Official download: {OFFICIAL_PROJECT_DOWNLOAD}")
    print(f"Current vendored commit: {current_commit or 'missing'}")
    print(f"Current vendored sha256: {current_header_sha}")
    print(f"Candidate sha256: {header_metadata['sha256']}")
    print(
        f"Header diff: {header_diff_summary(vendored_header, header_bytes_from_commit(exprtk_repo, commit))}"
    )
    print(
        "Upstream reserved identifiers: "
        f"{len(header_metadata['reserved_words'])} reserved_words, "
        f"{len(header_metadata['reserved_symbols'])} reserved_symbols"
    )
    print(f"Wrapper aliases: {', '.join(BNGSIM_WRAPPER_ALIASES)}")
    print("Required header tokens: OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exprtk-repo",
        type=Path,
        default=DEFAULT_EXPRTK_REPO,
        help="ExprTk checkout to vendor from (default: /tmp/exprtk-vendor-candidate)",
    )
    ap.add_argument(
        "--ref",
        default=DEFAULT_VENDOR_REF,
        help="ExprTk ref/tag/commit to vendor on refresh (default: master)",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify the vendored header + metadata match the resolved upstream ref; write nothing",
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="preview vendoring details for the resolved upstream ref; write nothing",
    )
    args = ap.parse_args()

    exprtk_repo = args.exprtk_repo.expanduser()
    checkout_info = verify_source_checkout(exprtk_repo)
    resolved_ref, commit = resolve_ref(exprtk_repo, args.ref)
    header_bytes = header_bytes_from_commit(exprtk_repo, commit)
    header_metadata = extract_header_metadata(header_bytes)
    vendor_metadata = read_vendor_metadata()

    if args.summary:
        print_summary(
            exprtk_repo, checkout_info, resolved_ref, commit, header_metadata, vendor_metadata
        )
        return 0

    if args.check:
        vendored_header = VENDOR_DIR / HEADER_NAME
        drift: list[str] = []
        if not vendored_header.exists():
            drift.append(f"missing {vendored_header}")
        elif vendored_header.read_bytes() != header_bytes:
            drift.append(
                f"vendored exprtk.hpp differs from resolved upstream {resolved_ref} ({commit})"
            )
        drift.extend(metadata_drift(vendor_metadata, commit, header_metadata))
        if drift:
            print("vendor_exprtk: DRIFT detected:", file=sys.stderr)
            for item in drift:
                print(f"  {item}", file=sys.stderr)
            print(
                "Run bngsim/scripts/vendor_exprtk.py to refresh the vendored header and VENDOR.json.",
                file=sys.stderr,
            )
            return 1

        print(
            "vendor_exprtk: OK — vendored exprtk.hpp and VENDOR.json match "
            f"{resolved_ref} ({commit})"
        )
        return 0

    metadata = build_vendor_metadata(
        exprtk_repo, checkout_info, resolved_ref, commit, header_metadata
    )
    write_refresh(exprtk_repo, header_bytes, metadata)
    print(
        "vendor_exprtk: refreshed bngsim/third_party/exprtk/exprtk.hpp and VENDOR.json "
        f"from {resolved_ref} ({commit})"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"vendor_exprtk: {exc}")
