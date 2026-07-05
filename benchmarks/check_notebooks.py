#!/usr/bin/env python3
"""Check shared benchmark notebooks for committed outputs and local paths."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


NOTEBOOK_ROOTS = (Path(__file__).resolve().parent / "suites" / "showcase",)
LOCAL_PATH_RE = re.compile(r"/Users/|/home/|~/|[A-Z]:\\\\")


def iter_notebooks() -> list[Path]:
    paths: list[Path] = []
    for root in NOTEBOOK_ROOTS:
        paths.extend(sorted(root.glob("*.ipynb")))
    return paths


def check_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    payload = json.loads(path.read_text())
    raw = path.read_text()
    if LOCAL_PATH_RE.search(raw):
        errors.append("contains a local absolute path")
    for idx, cell in enumerate(payload.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"cell {idx} has execution_count")
        if cell.get("outputs"):
            errors.append(f"cell {idx} has committed outputs")
    return errors


def main() -> int:
    failed = False
    for path in iter_notebooks():
        errors = check_notebook(path)
        if errors:
            failed = True
            for err in errors:
                print(f"{path}: {err}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
