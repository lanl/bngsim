#!/usr/bin/env python3
"""Refresh, summarize, or check the vendored MIR micro-JIT tree.

MIR (https://github.com/vnmakarov/mir) is vendored as a *pruned* source tree
under bngsim/third_party/mir for the GH #78 in-process codegen JIT backend.
Unlike the other vendored trees this one was historically hand-maintained; this
script makes the prune set and VENDOR.json re-anchoring reproducible.

Modes:

  --check    (offline) verify the checked-in tree still matches the per-file
             SHA256 anchors in VENDOR.json and the structural invariants. Needs
             no MIR checkout; this is the CI / test tripwire.
  --summary  preview the resolved upstream ref against the pinned baseline.
  (default)  re-prune the tree from a MIR checkout at --ref and rewrite the
             dynamic VENDOR.json fields (commit, imported_at_utc, files{}).

The prune is a denylist: every upstream file is vendored except tests,
benchmarks, standalone CLI/test drivers, docs (other than LICENSE/README.md),
build files, and CI/shell scripts. All target backends are kept so the same
tree builds on every host arch (see VENDOR.json `pruning`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BNGSIM_ROOT = REPO_ROOT / "bngsim"
VENDOR_DIR = BNGSIM_ROOT / "third_party" / "mir"
METADATA_NAME = "VENDOR.json"
CMAKE_LISTS = BNGSIM_ROOT / "CMakeLists.txt"

DEFAULT_MIR_REPO = Path("/tmp/mir-vendor-candidate")
DEFAULT_VENDOR_REF = "master"
DEFAULT_UPSTREAM_REMOTE = "https://github.com/vnmakarov/mir.git"

# BNGsim-owned files in the vendored tree — never written or removed by a
# refresh, never imported from upstream.
BNGSIM_FILES = {"VENDOR.json", "README.md.bngsim"}

# The three translation units CMake compiles (VENDOR.json build.translation_units).
TRANSLATION_UNITS = ("mir.c", "mir-gen.c", "c2mir/c2mir.c")

# Files whose SHA256 is anchored in VENDOR.json files{} (order preserved on write).
ANCHORED_FILES = (
    "mir.c",
    "mir-gen.c",
    "c2mir/c2mir.c",
    "mir.h",
    "mir-gen.h",
    "c2mir/c2mir.h",
    "LICENSE",
)

# Structural invariants for --check: these must be present, these must be absent.
REQUIRED_PRESENT = ANCHORED_FILES
FORBIDDEN_PRESENT = (
    "sieve.c",
    "mir-bin-driver.c",
    "mir-bin-run.c",
    "mir-gen-stub.c",
    "c2mir/c2mir-driver.c",
    "CMakeLists.txt",
    "GNUmakefile",
)

# ── Prune denylist (see VENDOR.json `pruning`) ────────────────────────────────
# A file is vendored unless one of these matches. Validated to reproduce the
# checked-in 87-file tree exactly against the pinned upstream commit.
EXCLUDED_TOP_DIRS = {
    ".github",
    "adt-tests",
    "c-benchmarks",
    "c-tests",
    "llvm2mir",
    "mir2c",
    "mir-tests",
    "mir-utils",
}
EXCLUDED_SUFFIXES = {".svg", ".sh", ".yml", ".yaml"}
EXCLUDED_BASENAMES = {".clang-format", ".gitignore", "GNUmakefile", "CMakeLists.txt"}
EXCLUDED_EXACT_PATHS = {
    "sieve.c",
    "mir-bin-driver.c",
    "mir-bin-run.c",
    "mir-gen-stub.c",
    "c2mir/c2mir-driver.c",
}


def is_vendored_path(rel: str) -> bool:
    """True if upstream relative path `rel` belongs in the pruned tree."""
    parts = rel.split("/")
    if parts[0] in EXCLUDED_TOP_DIRS:
        return False
    if rel in EXCLUDED_EXACT_PATHS:
        return False
    base = parts[-1]
    if base in EXCLUDED_BASENAMES:
        return False
    suffix = base[base.rfind(".") :] if "." in base else ""
    if suffix in EXCLUDED_SUFFIXES:
        return False
    # Drop docs except README.md (LICENSE has no extension and is kept above).
    if base.endswith(".md") and base != "README.md":
        return False
    return True


# ── Git helpers (mirrors vendor_exprtk.py conventions) ────────────────────────
def run(cmd: list[str], cwd: Path | None = None, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )


def git(repo: Path, args: list[str]) -> str:
    return run(["git", "-C", str(repo), *args]).stdout.strip()


def git_lines(repo: Path, args: list[str]) -> list[str]:
    return [line for line in git(repo, args).splitlines() if line]


def normalize_remote_url(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.removesuffix(".git").rstrip("/")


def is_canonical_upstream_remote(url: str | None) -> bool:
    return normalize_remote_url(url) == normalize_remote_url(DEFAULT_UPSTREAM_REMOTE)


def remote_url(repo: Path, remote: str) -> str | None:
    try:
        return git(repo, ["remote", "get-url", remote])
    except subprocess.CalledProcessError:
        return None


def canonical_remote_info(repo: Path) -> tuple[str | None, str | None]:
    remotes = git_lines(repo, ["remote"])
    canonical = [r for r in remotes if is_canonical_upstream_remote(remote_url(repo, r))]
    for preferred in ("origin", "upstream"):
        if preferred in canonical:
            return preferred, remote_url(repo, preferred)
    if canonical:
        return canonical[0], remote_url(repo, canonical[0])
    return None, None


def ordered_remotes(repo: Path) -> list[str]:
    remotes = git_lines(repo, ["remote"])
    ordered: list[str] = []
    canonical_remote, _ = canonical_remote_info(repo)
    if canonical_remote:
        ordered.append(canonical_remote)
    for preferred in ("origin", "upstream"):
        if preferred in remotes and preferred not in ordered:
            ordered.append(preferred)
    for remote in remotes:
        if remote not in ordered:
            ordered.append(remote)
    return ordered


def ref_candidates(repo: Path, ref: str) -> list[str]:
    if ref.startswith("refs/"):
        return [ref]
    remotes = ordered_remotes(repo)
    if not remotes or ref.split("/", 1)[0] in remotes:
        return [ref]
    candidates = [f"{remote}/{ref}" for remote in remotes]
    if ref not in candidates:
        candidates.append(ref)
    return candidates


def resolve_ref(repo: Path, ref: str) -> tuple[str, str]:
    attempted: list[str] = []
    for candidate in ref_candidates(repo, ref):
        try:
            commit = git(repo, ["rev-parse", "--verify", f"{candidate}^{{commit}}"])
            return candidate, commit
        except subprocess.CalledProcessError:
            attempted.append(candidate)
    raise RuntimeError(f"Could not resolve MIR ref '{ref}'. Tried: {', '.join(attempted) or ref}")


def verify_source_checkout(repo: Path) -> dict[str, str]:
    try:
        inside = git(repo, ["rev-parse", "--is-inside-work-tree"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"MIR checkout is not a Git worktree: {repo}") from exc
    if inside != "true":
        raise RuntimeError(f"MIR checkout is not a Git worktree: {repo}")

    changes = git_lines(repo, ["status", "--short"])
    if changes:
        preview = "\n  ".join(changes[:10])
        extra = "" if len(changes) <= 10 else f"\n  ... {len(changes) - 10} more"
        raise RuntimeError(
            "MIR checkout has local changes. Vendor only from a clean candidate checkout.\n"
            f"  {preview}{extra}"
        )
    if not (repo / "mir.c").exists() or not (repo / "c2mir" / "c2mir.c").exists():
        raise RuntimeError(f"MIR checkout is missing mir.c / c2mir/c2mir.c: {repo}")

    canonical_remote, canonical_url = canonical_remote_info(repo)
    return {
        "canonical_remote": canonical_remote or "",
        "canonical_remote_url": canonical_url or "",
        "head_commit": git(repo, ["rev-parse", "HEAD"]),
    }


# ── Content helpers ───────────────────────────────────────────────────────────
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_vendor_metadata() -> dict:
    metadata_path = VENDOR_DIR / METADATA_NAME
    if not metadata_path.exists():
        raise RuntimeError(f"missing {metadata_path}")
    return json.loads(metadata_path.read_text())


def upstream_tracked_files(repo: Path, commit: str) -> list[str]:
    return git_lines(repo, ["ls-tree", "-r", "--name-only", commit])


def file_bytes_from_commit(repo: Path, commit: str, rel: str) -> bytes:
    return run(["git", "-C", str(repo), "show", f"{commit}:{rel}"], text=False).stdout


def anchored_checksums_from_tree() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in ANCHORED_FILES:
        path = VENDOR_DIR / rel
        out[rel] = sha256_hex(path.read_bytes()) if path.exists() else "missing"
    return out


# ── Modes ─────────────────────────────────────────────────────────────────────
def do_check() -> int:
    """Offline: the checked-in tree must match its own VENDOR.json + invariants."""
    metadata = read_vendor_metadata()
    drift: list[str] = []

    recorded = metadata.get("files", {})
    actual = anchored_checksums_from_tree()
    for rel in ANCHORED_FILES:
        want = recorded.get(rel, {}).get("sha256")
        got = actual[rel]
        if want is None:
            drift.append(f"files.{rel}: missing sha256 in VENDOR.json")
        elif got == "missing":
            drift.append(f"files.{rel}: file absent from vendored tree")
        elif got != want:
            drift.append(f"files.{rel}.sha256: expected {want}, found {got}")

    for rel in REQUIRED_PRESENT:
        if not (VENDOR_DIR / rel).exists():
            drift.append(f"required file absent: {rel}")
    for rel in FORBIDDEN_PRESENT:
        if (VENDOR_DIR / rel).exists():
            drift.append(f"pruned file present (should be excluded): {rel}")
    for rel in BNGSIM_FILES:
        if not (VENDOR_DIR / rel).exists():
            drift.append(f"BNGsim file missing: {rel}")

    if metadata.get("local_carries") != []:
        drift.append(f"local_carries: expected [], found {metadata.get('local_carries')!r}")

    if drift:
        print("vendor_mir: DRIFT detected:", file=sys.stderr)
        for item in drift:
            print(f"  {item}", file=sys.stderr)
        print(
            "Re-run bngsim/scripts/vendor_mir.py to re-vendor and re-anchor VENDOR.json.",
            file=sys.stderr,
        )
        return 1

    commit = metadata.get("source", {}).get("commit", "?")
    print(f"vendor_mir: OK — bngsim/third_party/mir matches VENDOR.json ({commit[:12]})")
    return 0


def do_summary(repo: Path, resolved_ref: str, commit: str, checkout_info: dict[str, str]) -> int:
    metadata = read_vendor_metadata()
    pinned = metadata.get("source", {}).get("commit", "missing")

    all_upstream = upstream_tracked_files(repo, commit)
    kept = [p for p in all_upstream if is_vendored_path(p)]
    vendored = [
        str(p.relative_to(VENDOR_DIR))
        for p in VENDOR_DIR.rglob("*")
        if p.is_file() and str(p.relative_to(VENDOR_DIR)) not in BNGSIM_FILES
    ]

    candidate_anchors = {rel: sha256_hex(file_bytes_from_commit(repo, commit, rel)) for rel in ANCHORED_FILES}
    recorded = {rel: metadata.get("files", {}).get(rel, {}).get("sha256") for rel in ANCHORED_FILES}
    changed = [rel for rel in ANCHORED_FILES if candidate_anchors[rel] != recorded[rel]]

    print(f"MIR candidate: {repo}")
    print(f"Canonical remote: {checkout_info['canonical_remote']} -> {checkout_info['canonical_remote_url']}")
    print(f"Candidate HEAD: {checkout_info['head_commit']}")
    print(f"Resolved source ref: {resolved_ref}")
    print(f"Resolved source commit: {commit}")
    print(f"Pinned baseline commit: {pinned}")
    print(f"Pin moves: {'yes' if pinned != commit else 'no'}")
    print(f"Upstream files at ref: {len(all_upstream)}")
    print(f"Kept after prune: {len(kept)} (current vendored tree: {len(vendored)})")
    if changed:
        print(f"Anchored files that would change: {', '.join(changed)}")
    else:
        print("Anchored files that would change: none")
    return 0


def do_write(repo: Path, resolved_ref: str, commit: str) -> int:
    metadata = read_vendor_metadata()

    upstream = upstream_tracked_files(repo, commit)
    kept = [p for p in upstream if is_vendored_path(p)]
    if not kept or "mir.c" not in kept or "c2mir/c2mir.c" not in kept:
        raise RuntimeError("prune produced an empty / incomplete tree; refusing to write")

    # Materialize the kept files from the resolved commit (not the worktree).
    written: set[str] = set()
    for rel in kept:
        dest = VENDOR_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file_bytes_from_commit(repo, commit, rel))
        written.add(rel)

    # Remove stale upstream files no longer kept (e.g. dropped across a bump),
    # preserving BNGsim-owned files.
    for path in sorted(VENDOR_DIR.rglob("*"), reverse=True):
        if path.is_file():
            rel = str(path.relative_to(VENDOR_DIR))
            if rel not in written and rel not in BNGSIM_FILES:
                path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    # Update only the dynamic VENDOR.json fields; preserve the hand-authored
    # purpose / guardrails / pruning / notes.
    metadata.setdefault("source", {})
    metadata["source"]["branch_or_ref"] = resolved_ref
    metadata["source"]["commit"] = commit
    metadata["imported_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata["files"] = {
        rel: {"sha256": sha256_hex((VENDOR_DIR / rel).read_bytes())} for rel in ANCHORED_FILES
    }
    (VENDOR_DIR / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )

    print(
        f"vendor_mir: re-vendored bngsim/third_party/mir ({len(kept)} files) and "
        f"re-anchored VENDOR.json from {resolved_ref} ({commit})"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--mir-repo",
        type=Path,
        default=DEFAULT_MIR_REPO,
        help="MIR checkout to vendor from (default: /tmp/mir-vendor-candidate)",
    )
    ap.add_argument(
        "--ref",
        default=DEFAULT_VENDOR_REF,
        help="MIR ref/tag/commit to vendor on refresh (default: master)",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="offline: verify the vendored tree matches VENDOR.json + invariants; write nothing",
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="preview the resolved upstream ref against the pinned baseline; write nothing",
    )
    args = ap.parse_args()

    # --check is offline and needs no MIR checkout (the CI / test tripwire).
    if args.check:
        return do_check()

    repo = args.mir_repo.expanduser()
    checkout_info = verify_source_checkout(repo)
    resolved_ref, commit = resolve_ref(repo, args.ref)

    if args.summary:
        return do_summary(repo, resolved_ref, commit, checkout_info)
    return do_write(repo, resolved_ref, commit)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"vendor_mir: {exc}")
