"""Issue #326 — the two escape hatches that make the tolerance SHAPE one variable.

bngsim states a sensitivity's absolute tolerance as ``atolS[iS][i] =
atol·scale[i]/pbar[iS]``: ``scale[i] = max(|y_i(0)|, 1)`` non-dimensionalizes the
state axis (GH #214) and ``pbar[iS] = |p_iS|`` the parameter axis. AMICI states a
flat ``atol`` for every (row, column) and hands CVODES a ``NULL`` pbar
(``amici/src/solver.cpp`` 652 and 243). So on any model whose states or
parameters are not O(1), the two engines are not asking their sensitivity error
tests the same question — and #326 is a queue of models AMICI integrates and
bngsim does not.

Two hatches, read once at setup, both unset in shipped behaviour:

    BNGSIM_SENS_PBAR=unit     pbar -> 1.0          (AMICI's pbar)
    BNGSIM_SENS_ATOLS=flat    atolS -> atol        (AMICI's rule outright)

**Two, not one**, because the two factors move in opposite directions and a
single knob cannot attribute a rescue to either. ``BIOMD0000000879`` is the case
that forced it: ``N(0) = 2e10`` against ``delta = 1e4``, so dropping ``pbar``
loosens that column by 1e4 while dropping ``scale`` tightens it by 2e10. The
model is rescued by ``pbar=unit`` and *not* by ``atolS=flat`` (which needs the
issue #177 floor off as well), which is only a statement anyone can make because
the factors turn separately.

These change tolerances, never mathematics, so every test below checks the
closed form on both sides of the hatch as well as the step count. The model is
``S' = -k·S``, ``S(0) = S0``, run over ten e-foldings:

    S(t)     = S0·e^{−k·t}
    dS/dk(t) = −S0·t·e^{−k·t}
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

_HATCHES = ("BNGSIM_SENS_PBAR", "BNGSIM_SENS_ATOLS")


def _model(k: float, s0: float = 1.0) -> bngsim.Model:
    b = ModelBuilder()
    b.add_parameter("k", k)
    s_idx = b.add_species("S", s0)
    b.add_reaction([s_idx], [], "elementary", "k")
    return bngsim.Model(b.build())


def _run(monkeypatch, k: float, s0: float = 1.0, **hatches):
    """One run with exactly ``hatches`` set — every other hatch cleared."""
    for name in _HATCHES:
        monkeypatch.delenv(name, raising=False)
    for name, value in hatches.items():
        monkeypatch.setenv(name, value)
    sim = bngsim.Simulator(_model(k, s0), method="ode", sensitivity_params=["k"])
    # Ten e-foldings whatever k is, so the closed form below is never asked for
    # a number that has underflowed to zero.
    return sim.run(t_span=(0.0, 10.0 / k), n_points=11, rtol=1e-9, atol=1e-12)


def _check_closed_form(res, k: float, s0: float = 1.0):
    t = np.asarray(res.time)
    dk = np.asarray(res.sensitivities)[:, 0, 0]
    np.testing.assert_allclose(dk, -s0 * t * np.exp(-k * t), rtol=1e-5, atol=1e-9 / k)


def _steps(res) -> int:
    return int(res.solver_stats["n_steps"])


# ── BNGSIM_SENS_PBAR=unit ───────────────────────────────────────────────────
#
# Direction is forced, not observed-and-pinned: the hatch removes a division by
# |p|, so it loosens the column when |p| > 1 and tightens it when |p| < 1, and a
# looser error test takes no more steps. Testing both signs is what separates
# "the hatch does what it says" from "the hatch changed something".


def test_pbar_unit_loosens_a_large_parameter(monkeypatch):
    k = 1e6
    shipped = _run(monkeypatch, k)
    hatched = _run(monkeypatch, k, BNGSIM_SENS_PBAR="unit")
    assert _steps(hatched) < _steps(shipped)
    _check_closed_form(shipped, k)
    _check_closed_form(hatched, k)


def test_pbar_unit_tightens_a_small_parameter(monkeypatch):
    k = 1e-6
    shipped = _run(monkeypatch, k)
    hatched = _run(monkeypatch, k, BNGSIM_SENS_PBAR="unit")
    assert _steps(hatched) > _steps(shipped)
    _check_closed_form(shipped, k)
    _check_closed_form(hatched, k)


# ── BNGSIM_SENS_ATOLS=flat ──────────────────────────────────────────────────


def test_atols_flat_drops_the_state_scale_too(monkeypatch):
    """The factor ``pbar=unit`` leaves behind.

    With ``k = 1`` the pbar factor is inert, so anything this hatch changes is
    the ``scale[i]`` half. ``S(0) = 1e6`` makes the shipped tolerance 1e6x looser
    than the flat one, so flattening tightens it — the opposite direction from
    the large-parameter case above, on the same model shape.
    """
    k, s0 = 1.0, 1e6
    shipped = _run(monkeypatch, k, s0=s0)
    hatched = _run(monkeypatch, k, s0=s0, BNGSIM_SENS_ATOLS="flat")
    assert _steps(hatched) > _steps(shipped)
    _check_closed_form(shipped, k, s0)
    _check_closed_form(hatched, k, s0)


# ── Shipped behaviour is what you get without them ──────────────────────────


def test_an_unset_hatch_is_the_shipped_shape(monkeypatch):
    """The control: a run with the variables absent is the reference."""
    k = 1e6
    a = _run(monkeypatch, k)
    b = _run(monkeypatch, k)
    assert _steps(a) == _steps(b)
    np.testing.assert_array_equal(np.asarray(a.sensitivities), np.asarray(b.sensitivities))


@pytest.mark.parametrize(
    "hatch",
    [{"BNGSIM_SENS_PBAR": "magnitude"}, {"BNGSIM_SENS_ATOLS": "scaled"}, {"BNGSIM_SENS_PBAR": ""}],
    ids=["pbar_other_value", "atols_other_value", "pbar_empty"],
)
def test_only_the_documented_token_turns_a_hatch_on(monkeypatch, hatch):
    """A diagnostic that fires on any non-empty value is a trap.

    These hatches exist to be set in a sweep's environment beside a dozen others;
    one that read "set at all" rather than "set to this token" would silently
    change a corpus run that meant to name a different variable.
    """
    k = 1e6
    shipped = _run(monkeypatch, k)
    other = _run(monkeypatch, k, **hatch)
    assert _steps(other) == _steps(shipped)
    np.testing.assert_array_equal(
        np.asarray(other.sensitivities), np.asarray(shipped.sensitivities)
    )
