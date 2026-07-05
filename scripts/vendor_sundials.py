#!/usr/bin/env python3
"""Refresh or summarize the pinned SUNDIALS release metadata for BNGsim.

BNGsim does not check SUNDIALS source into the repository. Managed builds fetch
the official SUNDIALS release archive recorded in
`bngsim/third_party/sundials/VENDOR.json`. This script is the supported path
for refreshing or checking that metadata against a candidate release tarball.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BNGSIM_ROOT = REPO_ROOT / "bngsim"
VENDOR_DIR = BNGSIM_ROOT / "third_party" / "sundials"
METADATA_NAME = "VENDOR.json"
DEFAULT_CANDIDATE_DIR = Path("/tmp/sundials-vendor-candidate")
DEFAULT_REPO_URL = "https://github.com/LLNL/sundials.git"
DEFAULT_RELEASE_TAG = "v7.2.1"
DEFAULT_TAG_OBJECT = "35a984a216f6ea1db0315a506114673d90994407"
DEFAULT_TAG_COMMIT = "2dcb3e018b4c4cfe824bff09eb52184ed083e368"
DEFAULT_RELEASE_ASSET_NAME = "sundials-7.2.1.tar.gz"

REQUIRED_COMPONENT_PATHS = (
    "CMakeLists.txt",
    "README.md",
    "cmake/SundialsBuildOptionsPre.cmake",
    "cmake/SundialsTPLOptions.cmake",
    "cmake/SundialsSetupCompilers.cmake",
    "cmake/tpl/SundialsPOSIXTimers.cmake",
    "cmake/macros/SundialsAddLibrary.cmake",
    "src/CMakeLists.txt",
    "src/cvodes/CMakeLists.txt",
    "src/kinsol/CMakeLists.txt",
    "src/nvector/serial/CMakeLists.txt",
    "src/sunmatrix/dense/CMakeLists.txt",
    "src/sunmatrix/sparse/CMakeLists.txt",
    "src/sunlinsol/dense/CMakeLists.txt",
    "src/sunlinsol/klu/CMakeLists.txt",
)

REQUIRED_TEXT_TOKENS = {
    "cmake/SundialsBuildOptionsPre.cmake": [
        "sundials_option(SUNDIALS_BUILD_WITH_MONITORING BOOL",
        "sundials_option(SUNDIALS_BUILD_WITH_PROFILING BOOL",
        "sundials_option(BUILD_CVODES BOOL",
        "sundials_option(BUILD_KINSOL BOOL",
    ],
    "cmake/SundialsTPLOptions.cmake": [
        "sundials_option(ENABLE_KLU BOOL",
    ],
    "cmake/SundialsSetupCompilers.cmake": [
        "if(SUNDIALS_POSIX_TIMERS AND POSIX_TIMERS_NEED_POSIX_C_SOURCE)",
    ],
    "cmake/tpl/SundialsPOSIXTimers.cmake": [
        "set(POSIX_TIMERS_NEED_POSIX_C_SOURCE TRUE)",
        "set(SUNDIALS_POSIX_TIMERS TRUE)",
        "set(SUNDIALS_POSIX_TIMERS FALSE)",
    ],
    "cmake/macros/SundialsAddLibrary.cmake": [
        "add_library(SUNDIALS::${_export_name} ALIAS",
    ],
    "src/CMakeLists.txt": [
        "if(BUILD_CVODES)",
        "if(BUILD_KINSOL)",
    ],
    "src/cvodes/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/kinsol/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/nvector/serial/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/sunmatrix/dense/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/sunmatrix/sparse/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/sunlinsol/dense/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/sunlinsol/klu/CMakeLists.txt": [
        "sundials_add_library(",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help=(
            "Path to the official SUNDIALS release archive. "
            "Defaults to /tmp/sundials-vendor-candidate/<current asset name>."
        ),
    )
    parser.add_argument(
        "--repo-url",
        default=None,
        help=f"Canonical upstream Git remote. Default: {DEFAULT_REPO_URL}",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=f"Release tag for the archive. Default: {DEFAULT_RELEASE_TAG}",
    )
    parser.add_argument(
        "--tag-object",
        default=None,
        help="Annotated tag object SHA for the release tag.",
    )
    parser.add_argument(
        "--tag-commit",
        default=None,
        help="Peeled commit SHA for the release tag.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a no-write summary of the candidate archive versus VENDOR.json.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the candidate archive does not match the checked-in metadata.",
    )
    return parser.parse_args()


def read_vendor_metadata() -> dict | None:
    metadata_path = VENDOR_DIR / METADATA_NAME
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def normalize_repo_url(url: str) -> str:
    normalized = url.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.removesuffix(".git").rstrip("/")


def candidate_archive_path(args: argparse.Namespace, current_metadata: dict | None) -> Path:
    if args.archive is not None:
        return args.archive

    asset_name = DEFAULT_RELEASE_ASSET_NAME
    if current_metadata is not None:
        asset_name = current_metadata["source"]["authoritative_release_asset_name"]
    return DEFAULT_CANDIDATE_DIR / asset_name


def release_asset_name(archive_path: Path) -> str:
    if not archive_path.name.endswith(".tar.gz"):
        raise RuntimeError(f"Expected a .tar.gz SUNDIALS release archive, got: {archive_path.name}")
    return archive_path.name


def version_from_asset_name(asset_name: str) -> str:
    match = re.fullmatch(r"sundials-(\d+\.\d+\.\d+)\.tar\.gz", asset_name)
    if not match:
        raise RuntimeError(
            "Could not parse SUNDIALS version from archive name "
            f"'{asset_name}'. Expected sundials-X.Y.Z.tar.gz."
        )
    return match.group(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_archive_name(name: str) -> str:
    normalized = name.removeprefix("./").rstrip("/")
    return normalized


def archive_root_dir(member_names: set[str]) -> str:
    roots = {
        name.split("/", 1)[0]
        for name in member_names
        if name
    }
    if len(roots) != 1:
        roots_text = ", ".join(sorted(roots))
        raise RuntimeError(f"Expected a single top-level directory in the archive, found: {roots_text}")
    return next(iter(roots))


def read_text_member(archive: tarfile.TarFile, member_name: str) -> str:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise RuntimeError(f"Archive member is not a readable file: {member_name}")
    return extracted.read().decode("utf-8")


def parse_cmake_version(cmake_text: str) -> str:
    major = re.search(r'set\(PACKAGE_VERSION_MAJOR "([^"]+)"\)', cmake_text)
    minor = re.search(r'set\(PACKAGE_VERSION_MINOR "([^"]+)"\)', cmake_text)
    patch = re.search(r'set\(PACKAGE_VERSION_PATCH "([^"]+)"\)', cmake_text)
    label = re.search(r'set\(PACKAGE_VERSION_LABEL "([^"]*)"\)', cmake_text)
    if not major or not minor or not patch or not label:
        raise RuntimeError("Could not parse PACKAGE_VERSION_* fields from the SUNDIALS CMakeLists.txt")
    version = f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
    if label.group(1):
        version = f"{version}-{label.group(1)}"
    return version


def parse_readme_version(readme_text: str) -> str:
    match = re.search(r"Version ([0-9]+\.[0-9]+\.[0-9]+)", readme_text)
    if not match:
        raise RuntimeError("Could not parse the SUNDIALS version from README.md")
    return match.group(1)


def inspect_archive(archive_path: Path) -> dict:
    if not archive_path.exists():
        raise RuntimeError(f"Missing SUNDIALS release archive: {archive_path}")

    asset_name = release_asset_name(archive_path)
    expected_version = version_from_asset_name(asset_name)
    archive_sha256 = sha256_file(archive_path)
    archive_bytes = archive_path.stat().st_size

    with tarfile.open(archive_path, mode="r:gz") as archive:
        member_names = {
            normalize_archive_name(member.name)
            for member in archive.getmembers()
            if normalize_archive_name(member.name)
        }
        root_dir = archive_root_dir(member_names)

        missing_paths = [
            relpath for relpath in REQUIRED_COMPONENT_PATHS if f"{root_dir}/{relpath}" not in member_names
        ]
        if missing_paths:
            raise RuntimeError(
                "SUNDIALS release archive is missing required build paths:\n  "
                + "\n  ".join(missing_paths)
            )

        text_files: dict[str, str] = {}
        for relpath in REQUIRED_TEXT_TOKENS:
            text_files[relpath] = read_text_member(archive, f"{root_dir}/{relpath}")

        cmake_text = read_text_member(archive, f"{root_dir}/CMakeLists.txt")
        readme_text = read_text_member(archive, f"{root_dir}/README.md")

    detected_version = parse_cmake_version(cmake_text)
    readme_version = parse_readme_version(readme_text)
    if detected_version != expected_version:
        raise RuntimeError(
            f"Archive name version {expected_version} does not match CMake version {detected_version}"
        )
    if readme_version != expected_version:
        raise RuntimeError(
            f"Archive name version {expected_version} does not match README version {readme_version}"
        )
    if root_dir != f"sundials-{expected_version}":
        raise RuntimeError(
            f"Archive root directory {root_dir} does not match expected sundials-{expected_version}"
        )

    missing_tokens: dict[str, list[str]] = {}
    for relpath, expected_tokens in REQUIRED_TEXT_TOKENS.items():
        file_text = text_files[relpath]
        missing = [token for token in expected_tokens if token not in file_text]
        if missing:
            missing_tokens[relpath] = missing
    if missing_tokens:
        formatted = []
        for relpath, tokens in missing_tokens.items():
            formatted.append(f"{relpath}: {tokens}")
        raise RuntimeError(
            "SUNDIALS release archive is missing wrapper-critical tokens:\n  "
            + "\n  ".join(formatted)
        )

    return {
        "asset_name": asset_name,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "root_dir": root_dir,
        "detected_version": detected_version,
    }


def metadata_defaults(args: argparse.Namespace, current_metadata: dict | None, archive_info: dict) -> dict[str, str]:
    current_source = current_metadata["source"] if current_metadata else {}
    repo_url = args.repo_url or current_source.get("authoritative_repo", DEFAULT_REPO_URL)
    tag = args.tag or current_source.get("authoritative_release_tag", DEFAULT_RELEASE_TAG)
    tag_object = args.tag_object or current_source.get("tag_object", DEFAULT_TAG_OBJECT)
    tag_commit = args.tag_commit or current_source.get("tag_commit", DEFAULT_TAG_COMMIT)
    repo_web = normalize_repo_url(repo_url)

    return {
        "repo_url": repo_url,
        "tag": tag,
        "tag_object": tag_object,
        "tag_commit": tag_commit,
        "release_page": f"{repo_web}/releases/tag/{tag}",
        "release_asset_url": f"{repo_web}/releases/download/{tag}/{archive_info['asset_name']}",
    }


def build_metadata(archive_info: dict, source_info: dict[str, str]) -> dict:
    return {
        "name": "SUNDIALS",
        "vendored_path": "bngsim/third_party/sundials",
        "source": {
            "authoritative_repo": source_info["repo_url"],
            "authoritative_release_tag": source_info["tag"],
            "authoritative_release_asset_name": archive_info["asset_name"],
            "authoritative_release_asset_url": source_info["release_asset_url"],
            "authoritative_release_page": source_info["release_page"],
            "tag_object": source_info["tag_object"],
            "tag_commit": source_info["tag_commit"],
            "candidate_workspace": str(Path(archive_info["archive_path"]).parent),
            "candidate_archive": archive_info["archive_path"],
        },
        "imported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {
            "source_archive": {
                "path": "external:/tmp/sundials-vendor-candidate/<release tarball>",
                "asset_name": archive_info["asset_name"],
                "sha256": archive_info["archive_sha256"],
                "bytes": archive_info["archive_bytes"],
                "root_dir": archive_info["root_dir"],
                "detected_version": archive_info["detected_version"],
            }
        },
        "guardrails": {
            "required_component_paths": list(REQUIRED_COMPONENT_PATHS),
            "required_text_tokens": REQUIRED_TEXT_TOKENS,
        },
        "local_carries": [],
        "notes": [
            "BNGsim does not check SUNDIALS source into the repository.",
            "Managed builds fetch the official SUNDIALS release tarball pinned here via CMake FetchContent URL + URL_HASH.",
            "Set -DBNGSIM_USE_SYSTEM_SUNDIALS=ON to use an environment-managed SUNDIALS install instead of the pinned BNGsim fetch path.",
            "Refresh third_party/sundials/VENDOR.json only through bngsim/scripts/vendor_sundials.py.",
        ],
    }


def stable_metadata_view(metadata: dict) -> dict:
    stable = copy.deepcopy(metadata)
    stable.pop("imported_at_utc", None)
    return stable


def flatten_metadata(prefix: tuple[str, ...], value: object) -> dict[tuple[str, ...], str]:
    if isinstance(value, dict):
        flattened: dict[tuple[str, ...], str] = {}
        for key, item in value.items():
            flattened.update(flatten_metadata(prefix + (key,), item))
        return flattened
    if isinstance(value, list):
        return {prefix: json.dumps(value, indent=None, sort_keys=True)}
    return {prefix: json.dumps(value, sort_keys=True)}


def metadata_mismatches(current: dict | None, candidate: dict) -> list[str]:
    if current is None:
        return ["VENDOR.json does not exist yet."]

    current_flat = flatten_metadata((), stable_metadata_view(current))
    candidate_flat = flatten_metadata((), stable_metadata_view(candidate))
    mismatches: list[str] = []
    for key in sorted(set(current_flat) | set(candidate_flat)):
        if current_flat.get(key) != candidate_flat.get(key):
            dotted = ".".join(key)
            mismatches.append(
                f"{dotted}: current={current_flat.get(key, '<missing>')} "
                f"candidate={candidate_flat.get(key, '<missing>')}"
            )
    return mismatches


def print_summary(
    archive_path: Path,
    archive_info: dict,
    source_info: dict[str, str],
    current_metadata: dict | None,
    mismatches: list[str],
) -> None:
    print("SUNDIALS vendor summary")
    print(f"  archive: {archive_path}")
    print(f"  release tag: {source_info['tag']}")
    print(f"  tag object: {source_info['tag_object']}")
    print(f"  tag commit: {source_info['tag_commit']}")
    print(f"  release page: {source_info['release_page']}")
    print(f"  asset URL: {source_info['release_asset_url']}")
    print(f"  archive sha256: {archive_info['archive_sha256']}")
    print(f"  archive bytes: {archive_info['archive_bytes']}")
    print(f"  archive root: {archive_info['root_dir']}")
    print(f"  detected version: {archive_info['detected_version']}")
    print(f"  required component paths: {len(REQUIRED_COMPONENT_PATHS)}")
    print(f"  guarded text files: {len(REQUIRED_TEXT_TOKENS)}")
    if current_metadata is None:
        print("  current VENDOR.json: missing")
        return

    current_source = current_metadata["source"]
    current_archive = current_metadata["files"]["source_archive"]
    print("  current VENDOR.json:")
    print(f"    release tag: {current_source['authoritative_release_tag']}")
    print(f"    tag commit: {current_source['tag_commit']}")
    print(f"    asset URL: {current_source['authoritative_release_asset_url']}")
    print(f"    archive sha256: {current_archive['sha256']}")
    print(f"    detected version: {current_archive['detected_version']}")
    if mismatches:
        print(f"  mismatches vs VENDOR.json: {len(mismatches)}")
        for mismatch in mismatches[:20]:
            print(f"    - {mismatch}")
        if len(mismatches) > 20:
            print(f"    - ... {len(mismatches) - 20} more")
    else:
        print("  candidate matches the checked-in VENDOR.json")


def write_metadata(metadata: dict, current_metadata: dict | None) -> bool:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = VENDOR_DIR / METADATA_NAME
    if current_metadata is not None and stable_metadata_view(current_metadata) == stable_metadata_view(metadata):
        return False
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return True


def main() -> int:
    args = parse_args()
    current_metadata = read_vendor_metadata()
    archive_path = candidate_archive_path(args, current_metadata)

    try:
        archive_info = inspect_archive(archive_path)
    except RuntimeError as exc:
        if args.summary and not args.check:
            print(f"SUNDIALS vendor summary\n  archive: {archive_path}\n  error: {exc}")
            if current_metadata is not None:
                print("  current VENDOR.json is still present.")
            return 0
        print(f"error: {exc}", file=sys.stderr)
        return 1

    source_info = metadata_defaults(args, current_metadata, archive_info)
    candidate_metadata = build_metadata(archive_info, source_info)
    mismatches = metadata_mismatches(current_metadata, candidate_metadata)

    if args.summary:
        print_summary(archive_path, archive_info, source_info, current_metadata, mismatches)
        if not args.check:
            return 0

    if args.check:
        if mismatches:
            if not args.summary:
                print_summary(archive_path, archive_info, source_info, current_metadata, mismatches)
            return 1
        return 0

    wrote = write_metadata(candidate_metadata, current_metadata)
    if wrote:
        print(f"Updated {VENDOR_DIR / METADATA_NAME}")
    else:
        print(f"{VENDOR_DIR / METADATA_NAME} is already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
