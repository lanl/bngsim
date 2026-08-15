"""Enable ``python -m bngsim.cache`` as an alias for the ``bngsim-cache`` CLI.

A separate module rather than a ``__main__`` guard inside the package, for the same
reason ``bngsim.convert`` has one: ``bngsim/__init__.py`` imports this package, so
``python -m`` on a single-file module would re-execute an already-imported module and
``runpy`` would warn about it on every invocation.

It also matters more here than elsewhere. ``bngsim-cache`` is a console script, so it
appears only after a (re)install — and the whole point of this tool is to be reachable
on a machine whose cache is already 2 GB, with whatever bngsim happens to be there.
"""

from __future__ import annotations

from bngsim.cache import main

if __name__ == "__main__":
    raise SystemExit(main())
