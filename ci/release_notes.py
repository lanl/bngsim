#!/usr/bin/env python3
"""Print the GitHub Release body for one version, from CHANGELOG.md.

Used by `.github/workflows/release.yml` on a tag push, so a published release
never again has to be created by hand. Emits the version's CHANGELOG section
behind a short preamble, truncated at a paragraph boundary if it would exceed
GitHub's release-body limit.

    python ci/release_notes.py 0.13.0 > notes.md
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# GitHub rejects a release body longer than this many characters.
LIMIT = 125_000
REPO = os.environ.get("GITHUB_REPOSITORY", "lanl/bngsim")


def section(changelog: str, version: str) -> str:
    """The lines between `## [version]` and the next `## [` heading."""
    lines = changelog.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^## \[{re.escape(version)}\]", line):
            start = i
            break
    if start is None:
        raise SystemExit(f"release_notes: no CHANGELOG section for {version}")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## ["):
            end = j
            break
    return "\n".join(lines[start + 1 : end]).strip("\n")


def build(changelog: str, version: str) -> str:
    body = section(changelog, version)
    url = f"https://github.com/{REPO}/blob/v{version}/CHANGELOG.md"
    preamble = (
        f"```\npip install bngsim=={version}\n```\n\n"
        f"[Full changelog for {version}]({url})\n\n---\n\n"
    )
    budget = LIMIT - len(preamble) - 400
    if len(body) > budget:
        cut = body.rfind("\n\n", 0, budget)
        body = body[: cut if cut > 0 else budget].rstrip()
        body += (
            f"\n\n---\n\n*These notes exceed GitHub's {LIMIT:,}-character "
            f"release-body limit and are truncated here. The complete entry is "
            f"in [CHANGELOG.md]({url}).*"
        )
    return preamble + body


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: release_notes.py VERSION [CHANGELOG.md]")
    version = sys.argv[1]
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("CHANGELOG.md")
    out = build(path.read_text(encoding="utf-8"), version)
    if len(out) > LIMIT:  # belt and braces — the workflow must not 422
        raise SystemExit(f"release_notes: body is {len(out)} chars, over {LIMIT}")
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
