"""Which codegen backend a forced run selects, for the tests that depend on it
(issue #1217).

Run with every model forced through the MIR JIT
(``BNGSIM_CODEGEN_JIT=mir BNGSIM_CODEGEN_THRESHOLD=1``), the whole suite is the
check that found #1208 and #1212. It is only a signal if it passes, and three
kinds of test cannot pass under it. None of them is about a wrong number:

* ``under_jit_no_cc_artifact``: the test is about what the cc backend leaves on
  disk (the ``.so``, its cache key, the redirected cache directory, the notes
  written beside the artifact). The JIT compiles in process and leaves none.
* ``under_jit_engine_name``: the test asserts the backend is ``"cc"``, and the
  JIT reports ``"mir"``.
* ``threshold_forced``: the test needs a model below the auto-codegen threshold
  to run interpreted, and a lowered ``BNGSIM_CODEGEN_THRESHOLD`` compiles it.

Each is a ``skipif`` with a reason ``conftest.py`` declares, so a forced run
reports them as declared skips rather than as failures.

A test that only *reaches* its subject through a compiled artifact is not one of
them. When the subject is a number (a trajectory, a sensitivity, rows that must
not cross-talk), the test asserts what either backend leaves (the ``.so`` or the
JIT's C source, as ``test_model_codegen_sensitivity.py`` does) and names the
backend with :func:`selected_codegen_backend`. Skipping it would drop the very
check the forced JIT run is for.

Import it the way the other sibling helpers in this directory are imported::

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _codegen_backend import under_jit_no_cc_artifact
"""

from __future__ import annotations

import os

import pytest
from bngsim._codegen import _codegen_jit_backend

# The default ``Simulator`` reads when the variable is unset (``_simulator.py``).
_DEFAULT_CODEGEN_THRESHOLD = 256


def _jit_selected() -> bool:
    return _codegen_jit_backend() != ""


def selected_codegen_backend() -> str:
    """What ``Simulator.codegen_backend`` reports for a compiled model in this run."""
    return _codegen_jit_backend() or "cc"


def _threshold_lowered() -> bool:
    # The Simulator's auto-codegen reads no threshold under BNGSIM_NO_CODEGEN,
    # so nothing below it compiles there whatever the threshold says.
    if os.environ.get("BNGSIM_NO_CODEGEN"):
        return False
    raw = os.environ.get("BNGSIM_CODEGEN_THRESHOLD", "")
    try:
        return raw != "" and int(raw) < _DEFAULT_CODEGEN_THRESHOLD
    except ValueError:
        return False


under_jit_no_cc_artifact = pytest.mark.skipif(
    _jit_selected(),
    reason="tests the cc backend's compiled artifact, which the codegen JIT "
    "selected with BNGSIM_CODEGEN_JIT does not write",
)

under_jit_engine_name = pytest.mark.skipif(
    _jit_selected(),
    reason="asserts the cc backend by name; the codegen JIT selected with "
    "BNGSIM_CODEGEN_JIT reports its own",
)

threshold_forced = pytest.mark.skipif(
    _threshold_lowered(),
    reason="needs a model below the auto-codegen threshold to run interpreted; "
    "a lowered BNGSIM_CODEGEN_THRESHOLD compiles it",
)
