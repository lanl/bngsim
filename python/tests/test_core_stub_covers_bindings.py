"""Every pybind11 binding appears in the committed type stub.

``python/bngsim/_bngsim_core.pyi`` is machine-written by ``pybind11_stubgen``
and committed, because it is the *only* description of the compiled extension
that a type checker or an editor can read — neither can see into a ``.so``. It
is regenerated as a side effect of ``scripts/rebuild_editable.py``, and until
this test nothing checked that it matched the bindings.

It drifted the first time someone forgot: PR #306 added
``SolverOptions.set_crossing_stop_times`` without regenerating, so main
described an API missing a method it had. Two costs, neither loud:

  * every rebuild dirties the working tree with the regenerated hunk, which
    reads as noise from the build script rather than as a real omission;
  * mypy believes the stub. It reports ``"SolverOptions" has no attribute
    "set_crossing_stop_times"`` — *correct code, flagged* — for any caller that
    reaches the method through a typed reference.

mypy did not catch that drift itself only because the one caller passes ``opts``
as an **untyped** parameter, so inside the helper it is ``Any`` and no attribute
is checked. That is a thin thread to hang the invariant on: it holds only while
every call site stays untyped, which is the opposite of what anyone wants.

Names, not signatures, on purpose. Regenerating the stub in CI and diffing would
also catch a changed *signature*, but it needs a built extension on the leg and
``pybind11_stubgen`` output moves with the tool version and the platform — the
strictness would be paid for in flakes. The realistic failure is somebody adding
a binding and not regenerating, and a name-level check catches that with a
regex, in milliseconds, on every leg, with no build.
"""

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BINDINGS = _REPO / "src" / "_bngsim_core.cpp"
_STUB = _REPO / "python" / "bngsim" / "_bngsim_core.pyi"

# Each pybind11 spelling that publishes a *named* Python attribute. `.def(
# py::init<...>())` and the operator overloads carry no name literal and so
# match nothing, which is what we want.
_BINDING_PATTERNS = (
    r'\.def\(\s*"([A-Za-z_]\w*)"',
    r'\.def_static\(\s*"([A-Za-z_]\w*)"',
    r'\.def_readwrite\(\s*"([A-Za-z_]\w*)"',
    r'\.def_readonly\(\s*"([A-Za-z_]\w*)"',
    r'\.def_property\(\s*"([A-Za-z_]\w*)"',
    r'\.def_property_readonly\(\s*"([A-Za-z_]\w*)"',
)

# Dunders pybind11 synthesizes or that stubgen renders structurally rather than
# as a plain `def` line.
_EXEMPT = frozenset({"__init__", "__enter__", "__exit__", "__repr__", "__str__"})


def _bound_names() -> set[str]:
    source = _BINDINGS.read_text()
    names: set[str] = set()
    for pattern in _BINDING_PATTERNS:
        names |= set(re.findall(pattern, source))
    return names - _EXEMPT


def _declared_in_stub(name: str, stub: str) -> bool:
    # A method (`def name(`) or an attribute (`name: type`), at any indent.
    return (
        re.search(rf"(?m)^\s*(?:def\s+{re.escape(name)}\b|{re.escape(name)}\s*:)", stub)
        is not None
    )


@pytest.mark.skipif(
    not _BINDINGS.is_file(),
    reason="src/_bngsim_core.cpp is not in this checkout (installed package)",
)
def test_every_binding_is_declared_in_the_committed_stub():
    """A binding the stub does not declare is a mypy error waiting for a caller.

    Fix by rebuilding: ``python scripts/rebuild_editable.py`` regenerates the
    stub from the freshly built module. Commit the regenerated file with the
    binding that made it necessary.
    """
    stub = _STUB.read_text()
    missing = sorted(n for n in _bound_names() if not _declared_in_stub(n, stub))
    assert not missing, (
        f"{len(missing)} pybind11 binding(s) are missing from {_STUB.name}: "
        f"{missing}. Run `python scripts/rebuild_editable.py` and commit the "
        "regenerated stub alongside the binding."
    )


@pytest.mark.skipif(
    not _BINDINGS.is_file(),
    reason="src/_bngsim_core.cpp is not in this checkout (installed package)",
)
def test_the_scan_actually_finds_the_bindings():
    """Guard the guard: a regex that matched nothing would pass vacuously.

    The count is asserted as a floor rather than a number so adding bindings
    does not fail this, while a refactor that moves the binding block out of
    this file — or a pybind11 spelling change that stops matching — does.
    """
    names = _bound_names()
    assert len(names) > 150, f"only {len(names)} bindings found; the scan has gone blind"
    # Spot-check one binding of each kind that must be found by name.
    for expected in ("run", "set_crossing_stop_times", "rtol", "n_discontinuity_triggers"):
        assert expected in names, f"{expected!r} not seen by the binding scan"
