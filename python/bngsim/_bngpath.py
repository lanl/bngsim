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

The lookup is deliberately **not** cached: the only expensive step is the first
``import bionetgen``, which ``sys.modules`` already memoizes, so re-resolving
stays cheap while continuing to see an env var a caller sets at runtime.

Whether the ``perl`` on ``PATH`` can run the BNG2.pl that was found is a second
question (issue #1761). BNG2.pl needs Perl's core library (its line 34 is ``use
FindBin``), and Debian's ``perl-base``, the only Perl on slim container images,
leaves that library out: ``which("perl")`` says yes there, and the load dies with
Perl's own "Can't locate FindBin.pm". So :func:`resolve_bng` asks perl to compile
the script, ``perl -c BNG2.pl``, which loads every module BNG2.pl and its
``Perl2`` library ``use`` at compile time, runs only that compile-time code
(``BEGIN`` blocks and the ``use``d modules' top-level statements), not the
program, and takes well under a second. It runs in the environment BNG2.pl
itself runs in (``$BNGPATH`` and ``$BioNetGenRoot`` choose its ``Perl2``
library, and ``PERL_BADLANG=0`` keeps a locale warning out of its output), and a
success *is* remembered, per perl, script, script mtime and those two values,
since the suites resolve BNG2.pl at the import of many modules; a refusal is
asked again, so a Perl installed later in the same process is seen.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Mechanisms consulted, in precedence order. Kept as data so the "what was
# tried" report and the lookup itself can never drift apart.
ENV_BNG2_PL = "$BNG2_PL"
ENV_BNGPATH = "$BNGPATH"
ON_PATH = "BNG2.pl on $PATH"
BUNDLED = "PyBioNetGen bundled"
EXPLICIT = "explicit argument"

#: Opens every "nothing found" reason from :meth:`BngResolution.why_not`, so the
#: repository's test summaries can recognise a BNG2.pl skip by this constant
#: rather than by matching prose that may drift.
_NOT_FOUND = "no usable BNG2.pl"

#: How a bngsim *checkout* gets BNG2.pl (CONTRIBUTING.md). Kept out of
#: :meth:`BngResolution.why_not` on purpose (GH #1692): ``Model.from_bngl`` and
#: ``capabilities()`` show that text to wheel users, who have no uv project and
#: no ``parity_checks/``. ``parity_checks/_core/bngpath.require_bng`` and the
#: skip summaries of both test suites print it via :func:`_checkout_remedy`.
_CHECKOUT_REMEDY = (
    "In a bngsim checkout, get BNG2.pl with `uv sync --extra dev --group parity` "
    "or $BNGPATH rather than `pip install 'bngsim[bngl]'`, which the next "
    "`uv sync` removes."
)


def _checkout_remedy(reasons: Iterable[str]) -> list[str]:
    """Skip-summary lines for a checkout, or ``[]`` if no reason is a BNG2.pl miss.

    For the test suites' terminal summaries only; nothing in the library calls it.
    A found BNG2.pl with no ``perl`` is not a miss: it needs perl, not this.
    """
    n = sum(1 for reason in reasons if _NOT_FOUND in reason)
    if not n:
        return []
    return [
        f" {n} test(s) did not run because no BNG2.pl was found. The reason names",
        " the fix for a wheel install; this checkout needs a different one:",
        f"   {_CHECKOUT_REMEDY}",
    ]


@dataclass(frozen=True)
class BngResolution:
    """Outcome of a BNG2.pl lookup, including the trail of what was tried."""

    bng2_pl: Path | None
    root: Path | None
    source: str | None
    tried: tuple[tuple[str, str], ...]
    #: Whether a ``perl`` is on ``PATH`` at all (the stock-Windows question).
    has_perl: bool
    #: Why that perl cannot run the BNG2.pl found, as "<perl> said: <its first
    #: line>"; empty when it can, when there is no perl, or when nothing was found.
    perl_problem: str = ""

    @property
    def ok(self) -> bool:
        """A usable BNG2.pl *and* a perl that can run it."""
        return self.bng2_pl is not None and self.has_perl and not self.perl_problem

    def why_not(self) -> str:
        """One-line, actionable reason this resolution is unusable.

        Names every mechanism that was consulted and what it yielded, so the
        reader can tell "nothing is installed" from "it's installed somewhere I
        didn't look" — the distinction the old bare message destroyed.
        """
        if self.ok:
            return ""
        if self.bng2_pl is not None and self.perl_problem:
            # Not a miss (BNG2.pl is there) and not a missing interpreter.
            return _perl_problem_reason(self.bng2_pl, self.perl_problem)
        if self.bng2_pl is not None and not self.has_perl:
            return (
                f"found BNG2.pl at {self.bng2_pl} but `perl` is not on PATH — "
                "BNG2.pl is a Perl script, so it cannot run without one "
                "(macOS and most Linux distributions ship perl; stock Windows "
                "does not)"
            )
        trail = "; ".join(f"{name}: {detail}" for name, detail in self.tried)
        return (
            f"{_NOT_FOUND} — tried {trail}. "
            "Fix: `pip install 'bngsim[bngl]'` (PyBioNetGen bundles BNG2.pl), or "
            "point $BNGPATH at a BioNetGen folder containing BNG2.pl."
        )


#: Perl's "Can't locate X.pm", with ``X`` as a path (``XML/TreePP``).
_CANT_LOCATE = re.compile(r"Can't locate (\S+)\.pm\b")


def _perl_problem_reason(bng2_pl: Path, problem: str) -> str:
    """:meth:`BngResolution.why_not`'s text for a BNG2.pl the perl on PATH cannot compile.

    What Perl could not find decides the advice (issue #1761). BioNetGen's own
    modules sit in the ``Perl2`` folder beside BNG2.pl; when one of them is the
    missing one, ``$BNGPATH`` or ``$BioNetGenRoot``, which BNG2.pl's ``use lib``
    reads first, sent it to another library, and Perl is fine. Any other missing
    module is one of Perl's own, which Debian's perl-base, the only Perl on slim
    container images, leaves out. A refusal that names no module, and a perl that
    could not be run or did not finish, get neither diagnosis.
    """
    if _PERL_SAID not in problem:
        # perl could not be run or did not finish: the perl itself needs attention.
        return (
            f"found BNG2.pl at {bng2_pl} but the perl on PATH could not compile it "
            f"({problem}) — fix that perl, or put another `perl` first on PATH"
        )
    head = f"found BNG2.pl at {bng2_pl} but the perl on PATH cannot compile it ({problem})"
    missing = _CANT_LOCATE.search(problem)
    if missing is None:
        return f"{head} — check that BNG2.pl and its Perl2 library are complete and match"
    module = missing.group(1)
    try:
        own = (bng2_pl.parent / "Perl2" / f"{module}.pm").is_file()
    except OSError:
        own = False
    if own:
        return (
            f"{head} — {module}.pm is one of BioNetGen's own modules, in the Perl2 folder "
            "beside BNG2.pl, so $BNGPATH or $BioNetGenRoot points BNG2.pl at another "
            "library: unset them, or point them at the folder that holds BNG2.pl"
        )
    return (
        f"{head} — {module} is not one of BioNetGen's own modules but part of Perl's "
        "library, which Debian's perl-base, the only Perl on slim container images, "
        "leaves out (FindBin first). Install the full Perl: `apt-get install perl` "
        "(Debian/Ubuntu)"
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


#: Seconds allowed for ``perl -c BNG2.pl``. It takes well under a second; a perl
#: that hangs here would hang the load too, so the probe says so instead.
_PERL_CHECK_TIMEOUT = 60

#: Marks a ``perl_problem`` that is perl's own refusal to compile the script, as
#: against one where perl could not be run or did not finish; :meth:`why_not`
#: names the Perl package only for the former.
_PERL_SAID = " said: "

#: The ``(perl, script, mtime_ns, $BNGPATH, $BioNetGenRoot)`` keys whose perl
#: compiled the script. Successes only: a perl that refused is asked again on the
#: next resolution, so a full Perl installed in the same process (``apt-get`` in
#: a notebook) is seen at once, where perl's path and the script's mtime do not
#: change. On a broken image that costs one ``perl -c`` per resolution.
_PERL_COMPILES: set[tuple[str, str, int, str | None, str | None]] = set()


def _forget_perl_checks() -> None:
    """Drop the remembered successes (for tests that swap the perl or the script)."""
    _PERL_COMPILES.clear()


def _perl_status(bng2_pl: Path | None) -> tuple[bool, str]:
    """``(has_perl, perl_problem)`` for a resolution that found ``bng2_pl``.

    No ``perl`` on ``PATH`` is ``(False, "")``, the stock-Windows case. With no
    BNG2.pl there is nothing to compile, so a present perl is ``(True, "")``. With
    both, perl is asked to compile the script (issue #1761), in the environment
    BNG2.pl itself runs in: a relative ``$BNGPATH`` or ``$BioNetGenRoot`` made
    absolute against this process's directory (issue #1494), since either one
    chooses the ``Perl2`` library the script loads.
    """
    perl = shutil.which("perl")
    if perl is None or bng2_pl is None:
        return perl is not None, ""
    from bngsim._bngl_loader import _bng2_env

    env = _bng2_env()
    script = bng2_pl.absolute()  # as the load runs it; a relative one would key on cwd
    problem = _perl_cannot_compile(
        perl,
        str(script),
        _mtime_ns(script),
        env.get("BNGPATH"),
        env.get("BioNetGenRoot"),  # noqa: SIM112 - BNG2.pl's own spelling
    )
    return True, problem


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:  # pragma: no cover - the file was just resolved
        return -1


def _perl_cannot_compile(
    perl: str, bng2_pl: str, mtime_ns: int, bngpath: str | None, root_env: str | None
) -> str:
    """``""`` when ``perl -c bng2_pl`` succeeds, else why not.

    Compiling BNG2.pl loads every module it and its ``Perl2`` library ``use`` at
    compile time (``FindBin`` first, then ``Class::Struct``, ``Data::Dumper``,
    ``Storable`` and the rest), running only compile-time code (``BEGIN`` blocks
    and those modules' top-level statements), not the program. A perl that lacks
    any of them fails here on the first one, reported as "<perl> said: <its line>"
    (:func:`_first_perl_line`), the line a user would otherwise meet inside
    ``Model.from_bngl``'s error; the
    ``@INC`` listing Perl appends is dropped, since it says where Perl looked, not
    what to do. A perl that could not be run, or did not finish, is reported
    without that marker (see :data:`_PERL_SAID`).

    A success is remembered in :data:`_PERL_COMPILES`, per perl, script, script
    mtime and the two environment variables BNG2.pl's ``use lib`` reads, as the
    run will see them, because the test suites resolve BNG2.pl when many modules
    import. A failure is not, so a fix made in the same process is seen.
    """
    key = (perl, bng2_pl, mtime_ns, bngpath, root_env)
    if key in _PERL_COMPILES:
        return ""
    problem = _run_perl_check(perl, bng2_pl, bngpath, root_env)
    if not problem:
        _PERL_COMPILES.add(key)
    return problem


def _run_perl_check(perl: str, bng2_pl: str, bngpath: str | None, root_env: str | None) -> str:
    env = dict(os.environ)
    for var, value in (("BNGPATH", bngpath), ("BioNetGenRoot", root_env)):
        if value is None:
            env.pop(var, None)
        else:
            env[var] = value
    # Perl warns about an unusable locale before anything else (common on minimal
    # images that forward LANG); the warning would take the place of the line
    # that names the missing module. Compiling does not depend on the locale.
    env["PERL_BADLANG"] = "0"
    try:
        proc = subprocess.run(
            [perl, "-c", bng2_pl],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PERL_CHECK_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return f"{perl} -c did not finish within {_PERL_CHECK_TIMEOUT}s"
    except OSError as exc:
        return f"{perl} could not be run ({exc})"
    if proc.returncode == 0:
        return ""
    first = _first_perl_line(f"{proc.stderr}\n{proc.stdout}") or f"exit status {proc.returncode}"
    first = re.sub(r"\s*\(@INC (?:contains|entries checked):.*$", "", first)
    if len(first) > 240:
        first = first[:237] + "..."
    return f"{perl}{_PERL_SAID}{first}"


def _first_perl_line(text: str) -> str:
    """The line of perl's output that says why it refused, or ``""``.

    A "Can't locate" line wins wherever it is. Otherwise the first line that is not
    part of perl's locale warning ("perl: warning: …" and its indented lines),
    which ``PERL_BADLANG=0`` normally suppresses but a perl may still print.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for line in lines:
        if "Can't locate" in line:
            return line.strip()
    for line in lines:
        if line.startswith("perl: warning:") or line[:1].isspace():
            continue
        return line.strip()
    return ""


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

    has_perl, _ = _perl_status(None)
    return BngResolution(
        bng2_pl=None,
        root=None,
        source=None,
        tried=tuple(tried),
        has_perl=has_perl,
    )


def _bng2_pl_under(raw: str) -> Path | None:
    """BNG2.pl at ``raw`` — which may be the folder or the script itself."""
    p = Path(raw).expanduser()
    bng2 = p if p.is_file() else p / "BNG2.pl"
    return bng2 if bng2.is_file() else None


def _resolved(bng2_pl: Path, source: str, tried: list[tuple[str, str]]) -> BngResolution:
    has_perl, perl_problem = _perl_status(bng2_pl)
    return BngResolution(
        bng2_pl=bng2_pl,
        root=bng2_pl.parent,
        source=source,
        tried=(*tried, (source, str(bng2_pl))),
        has_perl=has_perl,
        perl_problem=perl_problem,
    )


def skip_reason(explicit: str | os.PathLike[str] | None = None) -> str | None:
    """``None`` when BNG2.pl is usable, else the actionable reason to skip."""
    r = resolve_bng(explicit)
    return None if r.ok else r.why_not()
