"""Single source of truth for locating the legacy BioNetGen install (BNG2.pl).

Why this module exists
----------------------
BNG2.pl was located by six near-duplicate helpers across eleven files, and they
disagreed about precedence. The test-side ones asked the *installed* PyBioNetGen
for its bundled copy first and consulted ``$BNGPATH`` / ``$BNG2_PL`` only from an
``except`` branch — so whenever ``bionetgen`` was importable an explicit
``export BNGPATH=...`` was silently **ignored**, and an install whose
``get_conf()`` returned no ``bngpath`` resolved to ``None`` with the env var
sitting there unread. The script-side ones did the reverse and never looked at
the bundled copy at all, so a perfectly good PyBioNetGen install still required
exporting a path by hand. Either way the failure surfaced as a bare "needs
BNG2.pl" with no indication of what had been looked for — which reads as "you
have no BioNetGen" on a machine that has three of them.

The rules here:

* **Explicit beats implicit.** An argument, then ``$BNG2_PL``, then ``$BNGPATH``,
  then ``BNG2.pl`` on ``$PATH``, then whatever PyBioNetGen bundles. Setting an
  env var always wins over an installed package, so an override is an override.
* **Every attempt is recorded.** :attr:`BngResolution.tried` carries one entry
  per mechanism, so a skip or abort can name what was searched and how to fix it
  instead of just reporting absence.
* **Nothing is hardcoded.** No ``/Users/...``, no guessed install directories —
  the resolution comes from the environment or from an installed package.

Why it is *shipped* (GH #162)
-----------------------------
It began under ``parity_checks/``, which is developer-only and not packaged. Then
:meth:`bngsim.Model.from_bngl` arrived and the shipped package needed the same
lookup — and ``bngsim.convert._bng2.find_bng2`` had already grown a seventh,
weaker copy of it (``$BNGPATH`` and ``$PATH`` only, no ``$BNG2_PL``, no bundled
PyBioNetGen, no trail). Promoting the module here rather than adding an eighth is
the whole point of the paragraphs above; ``parity_checks/_core/bngpath.py`` now
re-exports from this file, so there is still exactly one resolver.

The probe is deliberately **not** cached: the only expensive step is the first
``import bionetgen``, which ``sys.modules`` already memoizes, so re-resolving
stays cheap while continuing to see an env var a caller sets at runtime.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Mechanisms consulted, in precedence order. Kept as data so the "what was
# tried" report and the lookup itself can never drift apart.
ENV_BNG2_PL = "$BNG2_PL"
ENV_BNGPATH = "$BNGPATH"
ON_PATH = "BNG2.pl on $PATH"
BUNDLED = "PyBioNetGen bundled"
EXPLICIT = "explicit argument"


@dataclass(frozen=True)
class BngResolution:
    """Outcome of a BNG2.pl lookup, including the trail of what was tried."""

    bng2_pl: Path | None
    root: Path | None
    source: str | None
    tried: tuple[tuple[str, str], ...]
    has_perl: bool

    @property
    def ok(self) -> bool:
        """A usable BNG2.pl *and* the perl needed to run it."""
        return self.bng2_pl is not None and self.has_perl

    def why_not(self) -> str:
        """One-line, actionable reason this resolution is unusable.

        Names every mechanism that was consulted and what it yielded, so the
        reader can tell "nothing is installed" from "it's installed somewhere I
        didn't look" — the distinction the old bare message destroyed.
        """
        if self.ok:
            return ""
        if self.bng2_pl is not None and not self.has_perl:
            return (
                f"found BNG2.pl at {self.bng2_pl} but `perl` is not on PATH — "
                "BNG2.pl is a Perl script, so it cannot run without one "
                "(macOS and most Linux distributions ship perl; stock Windows "
                "does not)"
            )
        trail = "; ".join(f"{name}: {detail}" for name, detail in self.tried)
        return (
            f"no usable BNG2.pl — tried {trail}. "
            "Fix: `pip install 'bngsim[bngl]'` (PyBioNetGen bundles BNG2.pl), or "
            "point $BNGPATH at a BioNetGen folder containing BNG2.pl. For the "
            "parity_checks/ suite use `uv sync --extra dev --group parity` "
            "instead — it pins the exact PyBioNetGen commit that suite requires."
        )


def _bundled_bngpath() -> tuple[str | None, str]:
    """PyBioNetGen's bundled BNG dir, plus a note on why it's unavailable."""
    try:
        from bionetgen.main import get_conf
    except Exception as exc:  # not installed, or import blew up
        return None, f"bionetgen not importable ({type(exc).__name__})"
    try:
        path = get_conf().get("bngpath")
    except Exception as exc:
        return None, f"get_conf() failed ({type(exc).__name__})"
    if not path:
        return None, "installed but get_conf() reports no bngpath"
    return str(path), str(path)


def resolve_bng(explicit: str | os.PathLike[str] | None = None) -> BngResolution:
    """Locate BNG2.pl, explicit-first, recording every mechanism consulted.

    ``explicit`` (a BioNetGen folder or a direct BNG2.pl path) wins, then
    ``$BNG2_PL``, then ``$BNGPATH``, then a ``BNG2.pl`` on ``$PATH``, then
    PyBioNetGen's bundled copy. A candidate that does not exist on disk does not
    veto the ones after it — a stale env var falls through to a working install
    rather than poisoning the lookup.
    """
    candidates: list[tuple[str, str | None]] = [
        (EXPLICIT, str(explicit) if explicit else None),
        (ENV_BNG2_PL, os.environ.get("BNG2_PL")),
        (ENV_BNGPATH, os.environ.get("BNGPATH")),
        # A BNG2.pl the user put on PATH is an explicit act too, so it outranks
        # a package that merely happens to be installed. This is the mechanism
        # convert._bng2.find_bng2 had and this module did not; keeping it means
        # the promotion is a strict superset of every copy it replaces.
        (ON_PATH, shutil.which("BNG2.pl")),
    ]

    tried: list[tuple[str, str]] = []
    for name, raw in candidates:
        if not raw:
            tried.append((name, "unset" if name != ON_PATH else "not on PATH"))
            continue
        found = _bng2_pl_under(raw)
        if found is not None:
            return _resolved(found, name, tried)
        tried.append((name, f"{raw} (no BNG2.pl there)"))

    bundled, detail = _bundled_bngpath()
    if bundled:
        found = _bng2_pl_under(bundled)
        if found is not None:
            return _resolved(found, BUNDLED, tried)
        detail = f"{bundled} (no BNG2.pl there)"
    tried.append((BUNDLED, detail))

    return BngResolution(
        bng2_pl=None,
        root=None,
        source=None,
        tried=tuple(tried),
        has_perl=shutil.which("perl") is not None,
    )


def _bng2_pl_under(raw: str) -> Path | None:
    """BNG2.pl at ``raw`` — which may be the folder or the script itself."""
    p = Path(raw).expanduser()
    bng2 = p if p.is_file() else p / "BNG2.pl"
    return bng2 if bng2.is_file() else None


def _resolved(bng2_pl: Path, source: str, tried: list[tuple[str, str]]) -> BngResolution:
    return BngResolution(
        bng2_pl=bng2_pl,
        root=bng2_pl.parent,
        source=source,
        tried=(*tried, (source, str(bng2_pl))),
        has_perl=shutil.which("perl") is not None,
    )


def skip_reason(explicit: str | os.PathLike[str] | None = None) -> str | None:
    """``None`` when BNG2.pl is usable, else the actionable reason to skip."""
    r = resolve_bng(explicit)
    return None if r.ok else r.why_not()
