#!/usr/bin/env python3
"""Put a sha256-pinned file at a path: keep the copy there, or download it.

The download half of .github/actions/cached-download (lanl/bngsim#1607). That
action restores ``--dest`` from the Actions cache first, so on a warm cache the
file is already there and this checks it and downloads nothing. On a miss it
downloads, retrying, and the action saves the result to the cache.

A file whose sha256 is not ``--sha256`` is never left at ``--dest``. A copy
already there with other bytes is replaced by a fresh download. A download with
other bytes is a failure, whatever the server said.

Writes ``downloaded=true|false`` to ``$GITHUB_OUTPUT`` when that is set.
Standard library only, so it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

#: Seconds to wait before each retry. Five attempts across about four minutes:
#: enough for a few failed requests, not for a long outage, which the cache is for.
RETRY_DELAYS = (10, 30, 60, 120)
TIMEOUT = 120
USER_AGENT = "bngsim-ci-cached-download"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download_once(url: str, part: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, open(part, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)


def download(url: str, dest: Path, sha256: str, delays=RETRY_DELAYS) -> None:
    """Download ``url`` to ``dest`` and check it, retrying after each delay."""
    part = dest.with_name(dest.name + ".part")
    attempts = len(delays) + 1
    for attempt in range(1, attempts + 1):
        try:
            _download_once(url, part)
        except Exception as exc:  # urllib raises several unrelated types
            part.unlink(missing_ok=True)
            print(f"download attempt {attempt} of {attempts} failed: {exc}", file=sys.stderr)
            if attempt == attempts:
                raise SystemExit(f"could not download {url}") from None
            time.sleep(delays[attempt - 1])
            continue
        got = sha256_of(part)
        if got != sha256:
            part.unlink()
            raise SystemExit(f"{url} has sha256 {got}; expected {sha256}")
        os.replace(part, dest)
        return


def ensure(url: str, sha256: str, dest: Path, delays=RETRY_DELAYS) -> bool:
    """Leave the pinned file at ``dest``. True if it had to be downloaded."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        got = sha256_of(dest)
        if got == sha256:
            print(f"{dest.name}: the cached copy matches sha256 {sha256}", file=sys.stderr)
            return False
        print(
            f"{dest.name}: the cached copy has sha256 {got}, not {sha256}; downloading again",
            file=sys.stderr,
        )
        dest.unlink()
    download(url, dest, sha256, delays)
    print(f"{dest.name}: downloaded from {url}, sha256 checked", file=sys.stderr)
    return True


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--dest", required=True, type=Path)
    args = p.parse_args(argv)
    sha256 = args.sha256.strip().lower()
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise SystemExit(f"--sha256 {args.sha256!r} is not a sha256 hex digest")
    downloaded = ensure(args.url, sha256, args.dest)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"downloaded={'true' if downloaded else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
