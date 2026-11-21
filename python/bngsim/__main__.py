"""``python -m bngsim``: print the agent guide, or the same guide as a skill (#1699).

``python -m bngsim guide`` prints :func:`bngsim.agent_guide`, the one-page
guide to the installed version. ``python -m bngsim skill`` prints it with the
header that makes it a ``SKILL.md`` file, which agent harnesses that load
skills read when a task matches its description::

    python -m bngsim skill > ~/.claude/skills/bngsim/SKILL.md
"""

from __future__ import annotations

import sys

_USAGE = """\
usage: python -m bngsim {guide,skill}

  guide  print the BNGsim agent guide for this version (bngsim.agent_guide())
  skill  print the guide as a SKILL.md file for agent harnesses that load skills
"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args in (["-h"], ["--help"]):
        sys.stdout.write(_USAGE)
        return 0
    if args not in (["guide"], ["skill"]):
        sys.stderr.write(_USAGE)
        return 2
    from bngsim import agent_guide

    text = agent_guide(skill=args[0] == "skill")
    # As UTF-8 bytes whatever the console's encoding, so a redirect to SKILL.md
    # is UTF-8 on Windows too, and no character stops the print half way.
    out = getattr(sys.stdout, "buffer", None)
    if out is not None:
        out.write(text.encode("utf-8"))
        out.flush()
    else:  # pragma: no cover - a replaced stdout with no buffer
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
