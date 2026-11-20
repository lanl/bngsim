"""Wheel smoke test: the agent guide is in the wheel, and ``python -m bngsim`` prints it.

Issue #1699. ``bngsim.agent_guide()`` reads ``agent_guide.md`` from the
installed package, so the guide an agent reads offline is only there if the
wheel carries the file. The test suite runs against the source tree, where the
file always is; this runs against each built wheel, in cibuildwheel's
``test-command`` beside the other smoke tests.

bngsim alone: no compiler, no network, no data files. Run:
``python ci/wheel_smoke_agent_guide.py``.
"""

from __future__ import annotations

import subprocess
import sys

import bngsim

guide = bngsim.agent_guide()
assert guide.startswith("# bngsim agent guide\n"), guide[:80]
assert "translation-table" in guide and "**libRoadRunner**" in guide
skill = bngsim.agent_guide(skill=True)
assert skill.startswith("---\nname: bngsim\n") and skill.endswith(guide)

printed = subprocess.run(
    [sys.executable, "-m", "bngsim", "guide"], capture_output=True, check=True
).stdout.decode("utf-8")
assert printed.replace("\r\n", "\n") == guide, "python -m bngsim guide printed something else"
print(f"agent guide: {len(guide):,} characters, in the wheel and printed by python -m bngsim")
