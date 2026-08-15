"""Re-export of the shipped BNG2.pl resolver, plus the script-only abort helper.

The resolver itself moved to :mod:`bngsim._bngpath` (GH #162), because
:meth:`bngsim.Model.from_bngl` needs the same lookup from inside the *shipped*
package and ``parity_checks/`` is not packaged. Read that module's header for
why one resolver exists at all — the short version is that six near-duplicates
disagreed about precedence, and a seventh had already appeared in
``bngsim.convert._bng2``.

Nothing here re-implements any of it: these names are the same objects, so a
monkeypatch of ``bngsim._bngpath._bundled_bngpath`` is seen by callers who
reached the resolver through ``_core``. What stays here is
:func:`require_bng`, which ends the process — a sweep/matrix entrypoint concern
that has no business in a library.
"""

from __future__ import annotations

import os

from bngsim._bngpath import (
    BUNDLED,
    ENV_BNG2_PL,
    ENV_BNGPATH,
    EXPLICIT,
    ON_PATH,
    BngResolution,
    resolve_bng,
    skip_reason,
)

__all__ = [
    "BUNDLED",
    "ENV_BNG2_PL",
    "ENV_BNGPATH",
    "EXPLICIT",
    "ON_PATH",
    "BngResolution",
    "require_bng",
    "resolve_bng",
    "skip_reason",
]


def require_bng(purpose: str, explicit: str | os.PathLike[str] | None = None) -> BngResolution:
    """Resolve BNG2.pl or exit with a message naming ``purpose`` and the trail.

    For the sweep/matrix entrypoints, which cannot proceed without it. Also
    exports ``$BNGPATH`` so child processes inherit the same resolution rather
    than repeating the lookup and possibly landing somewhere else.
    """
    import sys

    r = resolve_bng(explicit)
    if not r.ok:
        sys.exit(f"ABORT: {purpose}\n  {r.why_not()}")
    os.environ["BNGPATH"] = str(r.root)
    return r
