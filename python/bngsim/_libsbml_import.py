"""What to say when ``import libsbml`` fails (issue #1760).

python-libsbml is a base dependency of bngsim, so it is installed wherever bngsim
is. It can still fail to import: its Linux wheel links the system expat library
rather than carrying a copy, and Debian's slim container images
(``python:3.13-slim`` and the like) do not ship ``libexpat1``. There, ``import
libsbml`` raises a bare ``ImportError: libexpat.so.1: cannot open shared object
file`` from inside the package, and every SBML or Antimony load dies with it,
while ``import bngsim`` and ``.net`` models work, so nothing else points at the
cause. ``find_spec("libsbml")`` says the package is present, which is true and
beside the point.

This module turns that into a message that names what is installed, what did
not load, and the one command that fixes it; :mod:`bngsim._sbml_loader` raises
it from its own ``import libsbml``, and ``bngsim.HAS_LIBSBML`` and
``capabilities()`` read :func:`libsbml_problem` so that they say ``False`` where a
load would fail.
"""

from __future__ import annotations

_NOT_INSTALLED = (
    "python-libsbml is not installed, so SBML and Antimony models cannot be loaded. "
    "It is a dependency of bngsim: `pip install python-libsbml` (or reinstall bngsim) "
    "puts it back. Python said: {text}"
)

_NOT_LOADABLE = (
    "python-libsbml is installed, but its compiled module could not be loaded: {text}. "
)

_EXPAT_HINT = (
    "libSBML's Linux wheel reads the system expat library, which slim container images "
    "such as python:3.13-slim do not ship. Install it and retry: "
    "`apt-get install libexpat1` (Debian/Ubuntu)."
)

_OTHER_HINT = (
    "A shared library it needs is missing from this system, or the module was built "
    "for another Python; `python -c 'import libsbml'` shows the same failure."
)


def is_missing_package(exc: BaseException) -> bool:
    """``True`` for ``No module named 'libsbml'``: no package, not a failed load.

    Only the error that names ``libsbml`` itself counts. A ``ModuleNotFoundError``
    from inside libsbml's own import chain, named or not, means the package is
    there and something it loads is not.
    """
    return isinstance(exc, ModuleNotFoundError) and exc.name == "libsbml"


def libsbml_import_error(exc: BaseException) -> str:
    """The message for an ``ImportError`` that ``import libsbml`` raised.

    A missing *package* gets the install command. Anything else means the package
    is there and its compiled module did not load; the original text is quoted,
    and when it names expat the message says which system package supplies it.
    """
    text = str(exc)
    if is_missing_package(exc):
        return _NOT_INSTALLED.format(text=text)
    hint = _EXPAT_HINT if "expat" in text.lower() else _OTHER_HINT
    return _NOT_LOADABLE.format(text=text) + hint


def libsbml_import_failure(exc: BaseException) -> ImportError:
    """The exception to raise in place of ``exc``, with :func:`libsbml_import_error`.

    It keeps the kind and the name of what Python raised, so a caller's
    ``except ModuleNotFoundError`` or ``.name == "libsbml"`` still works: a
    missing package is a ``ModuleNotFoundError``, a failed load an ``ImportError``,
    both naming ``libsbml``. Raise it ``from exc``.
    """
    message = libsbml_import_error(exc)
    if is_missing_package(exc):
        return ModuleNotFoundError(message, name="libsbml")
    return ImportError(message, name="libsbml")


def libsbml_problem() -> str | None:
    """``None`` when ``import libsbml`` works here, else :func:`libsbml_import_error`.

    The import itself is what is tested, not the package's presence: on a slim
    image the package is present and the import fails. A success is memoised by
    ``sys.modules``; a failure is retried on each call, which costs one failed
    import and lets a fix made in the same process (an ``apt-get`` in a notebook)
    be seen.
    """
    try:
        import libsbml  # noqa: F401
    except ImportError as exc:
        return libsbml_import_error(exc)
    return None
