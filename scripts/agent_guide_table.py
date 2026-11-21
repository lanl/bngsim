"""Regenerate the agent guide's translation table from FOREIGN_NAMES (issue #1699).

``python/bngsim/agent_guide.md`` holds the table between two markers, and
``test_agent_guide.py`` fails while it differs from what
:func:`bngsim._agent_guide.translation_table` generates. After a change to
``bngsim/_foreign_names.py``, run::

    python scripts/agent_guide_table.py
"""

from __future__ import annotations

from pathlib import Path

from bngsim import _agent_guide

GUIDE = Path(__file__).resolve().parents[1] / "python" / "bngsim" / _agent_guide.GUIDE_FILE


def main() -> int:
    text = GUIDE.read_text(encoding="utf-8")
    new = _agent_guide._with_table(text, _agent_guide.translation_table())
    if new == text:
        print(f"{GUIDE}: the table is current")
    else:
        GUIDE.write_text(new, encoding="utf-8")
        print(f"{GUIDE}: rewrote the translation table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
