"""python-libsbml can be installed and still fail to import (issue #1760).

On a Debian slim image (``python:3.13-slim``) libsbml's Linux wheel wants the
system expat library, which the image lacks, so ``import libsbml`` raises a bare
``ImportError: libexpat.so.1: cannot open shared object file`` from inside the
package. Every SBML and Antimony load died with that line, ``HAS_LIBSBML`` and
``capabilities()`` said SBML was available (``find_spec`` finds the package), and
``Model.from_antimony_string`` reached libsbml before antimony and so never got
to say anything about either.

Oracles are the texts themselves: the message built for an error, the flag and
``capabilities()`` with the import made to fail the way the image does, and what
each entry point raises. The failing import is staged with a meta-path finder
whose loader raises the image's exact line; libsbml itself is present here.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from types import ModuleType

import bngsim
import pytest
from bngsim import _libsbml_import as lib
from bngsim import _sbml_loader

# The line python:3.13-slim gives (lanl/bngsim#1694's Linux pass, run 43671106019).
EXPAT = "libexpat.so.1: cannot open shared object file: No such file or directory"

# A real one-species document, so from_sbml's failure is the import, not the path.
SBML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">\n'
    '  <model id="m"><listOfCompartments><compartment id="c" size="1" constant="true"/>'
    "</listOfCompartments><listOfSpecies>"
    '<species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"'
    ' boundaryCondition="false" constant="false"/></listOfSpecies></model>\n'
    "</sbml>\n"
)


class _LoadFails:
    """A meta-path finder that claims ``name`` and fails to load it with ``exc``,
    the way a package whose compiled module cannot be loaded fails: the spec is
    found, ``exec_module`` raises."""

    def __init__(self, name: str, exc: BaseException) -> None:
        self.name, self.exc = name, exc

    def find_spec(self, name, path=None, target=None):
        return importlib.util.spec_from_loader(name, self) if name == self.name else None

    def create_module(self, spec) -> ModuleType | None:
        return None

    def exec_module(self, module) -> None:
        raise self.exc


# ─── The message for each kind of failure ───────────────────────────────────


def test_a_missing_system_library_is_named_with_its_package():
    msg = lib.libsbml_import_error(ImportError(EXPAT))
    assert msg.startswith(
        "python-libsbml is installed, but its compiled module could not be loaded"
    )
    assert EXPAT in msg  # the cause, quoted
    assert "apt-get install libexpat1" in msg  # the fix
    assert "slim" in msg  # where this happens


def test_an_uninstalled_libsbml_gets_the_install_command():
    exc = ModuleNotFoundError("No module named 'libsbml'", name="libsbml")
    msg = lib.libsbml_import_error(exc)
    assert "pip install python-libsbml" in msg
    assert "libexpat1" not in msg


def test_another_load_failure_is_quoted_without_the_expat_fix():
    msg = lib.libsbml_import_error(ImportError("_libsbml.so: wrong ELF class: ELFCLASS32"))
    assert "wrong ELF class" in msg
    assert "python-libsbml is installed" in msg
    assert "libexpat1" not in msg


@pytest.mark.parametrize("name", ["some_dep", None], ids=["named", "unnamed"])
def test_a_missing_dependency_of_libsbml_is_a_load_failure_not_a_missing_package(name):
    # ModuleNotFoundError for something libsbml itself imports, named or not: the
    # package is there, so "pip install python-libsbml" would be the wrong advice.
    exc = ModuleNotFoundError("No module named 'some_dep'", name=name)
    assert not lib.is_missing_package(exc)
    assert "pip install python-libsbml" not in lib.libsbml_import_error(exc)


def test_the_raised_exception_keeps_the_kind_and_the_name():
    missing = lib.libsbml_import_failure(
        ModuleNotFoundError("No module named 'libsbml'", name="libsbml")
    )
    assert isinstance(missing, ModuleNotFoundError) and missing.name == "libsbml"
    assert "pip install python-libsbml" in str(missing)
    broken = lib.libsbml_import_failure(ImportError(EXPAT))
    assert isinstance(broken, ImportError) and not isinstance(broken, ModuleNotFoundError)
    assert broken.name == "libsbml"
    assert "apt-get install libexpat1" in str(broken)


def test_here_libsbml_loads():
    """libsbml is a base dependency and this environment has it (control)."""
    assert lib.libsbml_problem() is None
    assert bngsim.HAS_LIBSBML is True
    assert bngsim.capabilities()["features"]["libsbml"] is True


# ─── With the import failing as it does on the image ────────────────────────

_SCRIPT = (
    textwrap.dedent(
        """
    import importlib.util, json, sys
    from types import ModuleType

    EXPAT = @EXPAT@

    class LoadFails:
        def find_spec(self, name, path=None, target=None):
            return importlib.util.spec_from_loader(name, self) if name == "libsbml" else None
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            raise ImportError(EXPAT)

    sys.meta_path.insert(0, LoadFails())
    out = {}
    import bngsim
    out["has_libsbml"] = bngsim.HAS_LIBSBML
    caps = bngsim.capabilities()
    keys = ("libsbml", "sbml_import", "antimony_import")
    out["features"] = {k: caps["features"][k] for k in keys}
    out["missing_libsbml"] = caps["missing"].get("libsbml", "")
    try:
        import bngsim._sbml_loader
    except ImportError as exc:
        out["loader"] = str(exc)
        out["loader_cause"] = str(exc.__cause__)
        out["loader_kind"] = type(exc).__name__
        out["loader_name"] = exc.name
    try:
        bngsim.Model.from_antimony_string("A -> B; k*A\\nk = 0.5\\nA = 100\\nB = 0\\n")
    except ImportError as exc:
        out["antimony"] = str(exc)
    import pathlib
    pathlib.Path("m.xml").write_text(@SBML@)
    try:
        bngsim.Model.from_sbml("m.xml")
    except ImportError as exc:
        out["sbml"] = str(exc)
    print(json.dumps(out))
    """
    )
    .replace("@EXPAT@", repr(EXPAT))
    .replace("@SBML@", repr(SBML))
)


@pytest.fixture(scope="module")
def on_a_slim_image(tmp_path_factory) -> dict:
    """``import bngsim`` and the entry points in a fresh interpreter whose
    ``import libsbml`` fails with the image's line (a subprocess, so this
    interpreter's libsbml and bngsim modules are untouched)."""
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path_factory.mktemp("slim"),
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_flag_and_capabilities_say_libsbml_is_unavailable(on_a_slim_image):
    out = on_a_slim_image
    # find_spec finds the package there too; what matters is that the import fails.
    assert out["has_libsbml"] is False
    assert out["features"] == {"libsbml": False, "sbml_import": False, "antimony_import": False}
    assert "apt-get install libexpat1" in out["missing_libsbml"]
    assert EXPAT in out["missing_libsbml"]


def test_the_loader_import_names_the_cause_and_the_fix(on_a_slim_image):
    out = on_a_slim_image
    assert "python-libsbml is installed" in out["loader"]
    assert "apt-get install libexpat1" in out["loader"]
    assert out["loader_cause"] == EXPAT  # chained, not swallowed
    # The kind and the name survive, for a caller's `except ModuleNotFoundError`
    # or `.name` check: a failed load is an ImportError, not a missing module.
    assert out["loader_kind"] == "ImportError"
    assert out["loader_name"] == "libsbml"


@pytest.mark.parametrize("entry", ["antimony", "sbml"])
def test_each_entry_point_raises_that_message(on_a_slim_image, entry):
    msg = on_a_slim_image[entry]
    assert "apt-get install libexpat1" in msg
    # Not the antimony extra's install hint: antimony was never reached.
    assert "bngsim[antimony]" not in msg


# ─── An antimony that is missing, or present and not loadable ───────────────


def test_import_antimony_keeps_the_install_hint_for_a_missing_package(monkeypatch):
    monkeypatch.delitem(sys.modules, "antimony", raising=False)
    missing = ModuleNotFoundError("No module named 'antimony'", name="antimony")
    monkeypatch.setattr(sys, "meta_path", [_LoadFails("antimony", missing), *sys.meta_path])
    with pytest.raises(ImportError, match=r"pip install 'bngsim\[antimony\]'") as info:
        _sbml_loader._import_antimony()
    assert info.value.__cause__ is None


def test_import_antimony_relays_a_load_failure_instead(monkeypatch):
    monkeypatch.delitem(sys.modules, "antimony", raising=False)
    broken = ImportError(
        "libantimony.so: cannot open shared object file: No such file or directory"
    )
    monkeypatch.setattr(sys, "meta_path", [_LoadFails("antimony", broken), *sys.meta_path])
    with pytest.raises(ImportError, match="antimony is installed, but") as info:
        _sbml_loader._import_antimony()
    assert "libantimony.so" in str(info.value)
    assert "bngsim[antimony]" not in str(info.value)
    assert info.value.__cause__ is broken


def test_importing_bngsim_does_not_import_libsbml():
    """The flag is lazy so that `import bngsim` keeps not paying for libsbml."""
    proc = subprocess.run(
        [sys.executable, "-c", "import bngsim, sys; print('libsbml' in sys.modules)"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.stdout.strip() == "False", proc.stderr
