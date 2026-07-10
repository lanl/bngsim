"""SSA propensity library export decoration (lanl/bngsim #6).

The structure-specialized SSA propensity vector (GH #190) is emitted as C by
``NetworkModel::emit_ssa_propensity_source_structure``, compiled to a shared
library by the Python codegen layer (``prepare_ssa_propensity_lib``), and then
resolved from that library BY NAME at run time::

    prop_lib->symbol<PropFn>("bngsim_ssa_propensities")   # GetProcAddress on Windows

An MSVC/MinGW DLL exports nothing unless the symbol is tagged
``__declspec(dllexport)``, so on Windows that lookup used to fail and the run
fell back silently to the interpreted propensity path — the same class of bug #5
fixed for the Python-side codegen entry points, but on the C++ emit side. The
emit now tags the function with a portable ``BNGSIM_EXPORT`` macro.

Three guards, coarsest reproduction first:

* a pure-string assertion on the emitted C — runs on every platform, needs no
  compiler, and pins both the decoration and its ``_WIN32`` form;
* a ``ctypes`` load of the compiled library that resolves the symbol by name —
  the direct analogue of the ``GetProcAddress`` call the C++ loader makes;
* an end-to-end assertion that the default SSA path engages the ``cc``-compiled
  propensity backend through bngsim's own ``DynamicLibrary`` — the failure the
  export bug actually produced was a quiet degrade to ``interpreted``.

The last two are Windows regressions specifically; on Unix the symbol exports by
default so they pass with or without the fix (they still guard against the emit
losing the decoration and breaking Windows). The first is the load-bearing
cross-platform pin.
"""

import ctypes
import os
import shutil

import bngsim
import pytest
from bngsim._bngsim_core import NetworkModel, emit_ssa_propensity_source_structure
from bngsim._codegen import prepare_ssa_propensity_lib

# Honor BNGSIM_TEST_DATA (run_tests.sh copies tests to a temp dir, breaking the
# __file__-relative path), matching test_codegen.py / test_mir_equivalence.py.
DATA = os.environ.get("BNGSIM_TEST_DATA") or os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "data"
)

# A single elementary mass-action reaction (A -> B, rate k1*A): the emitted
# propensity vector is fully covered (n_unsupported == 0) and the model is
# recompute-all eligible + within the size gate, so the default cc backend
# engages. No MM branch, so the source has no sqrt extern either.
_MASS_ACTION_NET = os.path.join(DATA, "simple_decay.net")

_CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc") or shutil.which("cl")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH")


def test_emitted_source_tags_export():
    """The emitted propensity C decorates the entry point for Windows export."""
    m = NetworkModel.from_net(_MASS_ACTION_NET)
    src, n_unsupported = emit_ssa_propensity_source_structure(m)
    assert n_unsupported == 0, (
        "simple_decay is elementary mass-action; the propensity vector must be "
        "fully covered so the export path is exercised"
    )
    # Portable export macro is defined and its Windows form is guarded by _WIN32.
    assert "#if defined(_WIN32)" in src
    assert "#define BNGSIM_EXPORT __declspec(dllexport)" in src
    assert "#define BNGSIM_EXPORT\n" in src  # empty (Unix) branch
    # The entry point the C++ loader resolves by name carries the decoration.
    assert "BNGSIM_EXPORT void bngsim_ssa_propensities(" in src


@needs_cc
def test_compiled_library_exports_symbol():
    """The compiled propensity library exports bngsim_ssa_propensities by name.

    ``ctypes.CDLL(...).bngsim_ssa_propensities`` does a ``GetProcAddress`` on
    Windows — exactly what the C++ ``DynamicLibrary::symbol`` loader does — and
    raises ``AttributeError`` if the symbol was not exported. This is the
    isolated reproduction of the #6 failure, independent of the SSA eligibility
    and size gates.
    """
    model = bngsim.Model.from_net(_MASS_ACTION_NET)
    so = prepare_ssa_propensity_lib(model, force_recompile=True)
    assert so, "expected a compiled propensity library for a mass-action model"
    lib = ctypes.CDLL(so)
    # Attribute access resolves the symbol; missing export -> AttributeError.
    assert lib.bngsim_ssa_propensities is not None


@needs_cc
def test_default_ssa_engages_cc_propensity_backend():
    """The default SSA path loads the .so and reports the cc backend.

    A failed by-name lookup in ssa_simulator.cpp degrades the run to the
    interpreted backend (the ``catch`` resets ``prop_backend`` to
    ``"interpreted"``), so asserting ``"cc"`` guards the export decoration
    through bngsim's own production loader end-to-end.
    """
    model = bngsim.Model.from_net(_MASS_ACTION_NET)
    sim = bngsim.Simulator(model, method="ssa")
    r = sim.run(t_span=(0, 10), n_points=11, seed=1)
    assert r.ssa_diagnostics["propensity_backend"] == "cc"
