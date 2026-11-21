"""``python -m bngsim``: the ``bngsim`` command (issues #1699, #1708).

The same command as the ``bngsim`` console script, reachable with only the
interpreter: ``python -m bngsim doctor``, ``python -m bngsim guide``,
``python -m bngsim inspect model.xml --json``. ``python -m bngsim --help`` lists
the commands; :mod:`bngsim._cli` has the rules for their output and exit codes.
"""

from __future__ import annotations

from bngsim._cli import main

if __name__ == "__main__":
    raise SystemExit(main())
