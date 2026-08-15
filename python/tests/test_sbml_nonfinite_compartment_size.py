"""Issue #353 — a non-finite compartment size must not poison the state.

``MODEL2002070001`` (a two-compartment gut-colonisation model) declares
``size="NaN"`` on both of its compartments and makes every species
``hasOnlySubstanceUnits="true"`` — the quantities are amounts, and no rate law
ever reads a compartment size. RoadRunner and AMICI both integrate it, in amount
units. bngsim stores a *concentration* (``amount / V``), so the ``NaN`` size
divided a well-specified ``initialAmount=10`` into ``nan`` and the whole state
went non-finite before a single step — the plain ODE run failed at ``flag=-9``,
naming the rate laws as if they were the cause.

The fix: a non-finite declared size is unusable as the amount↔concentration
divisor, so the loader substitutes a unit volume (and warns). An amount-only
species then loads as its declared amount and the model integrates in amounts,
matching both reference engines. A model with a *finite* size is untouched —
its amount is still stored as ``amount / V`` — so the substitution can only ever
turn a ``nan`` state into a usable one.

The real corpus model lives under the git-ignored ``parity_checks`` tree, which
is absent from a fresh checkout and from CI, so the model below reproduces its
essential shape self-containedly: NaN-sized compartments, amount-only species
whose initial conditions come by ``initialAmount`` and by ``initialAssignment``,
and rate-rule dynamics with a closed form to check against.
"""

from __future__ import annotations

import logging

import bngsim
import numpy as np
import pytest

# A rate-rule model in amount units. ``A`` grows as ``dA/dt = k*A`` (closed form
# ``A(t) = A0 * exp(k t)``); ``D`` is constant (``dD/dt = 0``). ``A``'s amount is
# declared with ``initialAmount``; ``D``'s with an ``initialAssignment`` — the two
# code paths that divide by the compartment volume. ``{msize}``/``{lsize}`` let a
# test swap the NaN sizes for finite ones without changing anything else.
MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="gut">
    <listOfCompartments>
      <compartment id="mucosa" size="{msize}" spatialDimensions="3" constant="true"/>
      <compartment id="lumen"  size="{lsize}" spatialDimensions="3" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="mucosa" initialAmount="10" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="D" compartment="lumen"  hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="D">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>7</cn></math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <rateRule variable="A">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>A</ci></apply></math>
      </rateRule>
      <rateRule variable="D">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0</cn></math>
      </rateRule>
    </listOfRules>
  </model>
</sbml>"""


def _nan_model():
    return MODEL.format(msize="NaN", lsize="NaN")


def _state(model):
    return dict(zip(model.species_names, np.asarray(model.get_state()), strict=False))


# ── the loaded state ─────────────────────────────────────────────────────────


def test_amount_only_ic_loads_as_its_amount_not_nan():
    """The bug in one row: an ``initialAmount=10`` amount-only species in a
    NaN-sized compartment loaded as ``nan``. It must load as 10 — the size is not
    a legitimate divisor when it is non-finite."""
    model = bngsim.Model.from_sbml_string(_nan_model())
    st = _state(model)
    assert st["A"] == 10.0
    assert st["D"] == 7.0  # the initialAssignment path, same defect
    assert all(np.isfinite(v) for v in st.values())


def test_the_substitution_is_announced(caplog):
    """Silently turning a NaN size into a usable one is exactly what issue #353
    argues against; the substitution must be logged, once per compartment, naming
    it."""
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml_string(_nan_model())
    warned = [r.getMessage() for r in caplog.records if "non-finite size" in r.getMessage()]
    assert len(warned) == 2
    assert any("mucosa" in m for m in warned)
    assert any("lumen" in m for m in warned)


# ── the integration ──────────────────────────────────────────────────────────


def test_the_plain_ode_run_integrates_in_amounts():
    """The whole point: the plain ODE run — sensitivities off — now completes and
    matches the closed form ``A(t) = A0 exp(k t)`` in amount units, where before
    it failed at ``t=0`` with a non-finite RHS. ``D`` stays at its declared
    amount."""
    model = bngsim.Model.from_sbml_string(_nan_model())
    result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 5.0), n_points=6)
    traj = dict(zip(model.species_names, np.asarray(result.species).T, strict=False))
    assert np.all(np.isfinite(np.asarray(result.species)))
    t = np.asarray(result.time)
    np.testing.assert_allclose(traj["A"], 10.0 * np.exp(0.1 * t), rtol=1e-6)
    np.testing.assert_allclose(traj["D"], 7.0, rtol=0, atol=1e-9)


# ── the regression guard: a finite size is untouched ─────────────────────────


def test_a_finite_size_still_divides_amount_by_volume():
    """The substitution fires only for a non-finite size. With ``mucosa`` at 2.0
    the amount-only ``A`` stores ``amount / V = 5`` (bngsim's storage convention,
    matching RoadRunner's ``[A]``), so the fix cannot have changed a model whose
    size was well-specified all along."""
    model = bngsim.Model.from_sbml_string(MODEL.format(msize="2.0", lsize="4.0"))
    st = _state(model)
    assert st["A"] == 5.0  # 10 / 2, unchanged by #353
    assert st["D"] == 1.75  # 7 / 4


def test_a_finite_size_warns_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="bngsim"):
        bngsim.Model.from_sbml_string(MODEL.format(msize="2.0", lsize="4.0"))
    assert not [r for r in caplog.records if "non-finite size" in r.getMessage()]


# ── cross-engine parity, where RoadRunner is available ───────────────────────


def test_matches_roadrunner_amounts():
    """RoadRunner integrates this shape in amount units (its *concentration*
    selections are ``nan`` — the same reporting artifact of a NaN size — but its
    amounts are finite). bngsim, storing ``amount / 1`` after the substitution,
    reports those same amounts."""
    rr = pytest.importorskip("roadrunner")
    src = _nan_model()

    r = rr.RoadRunner(src)
    r.integrator.setValue("relative_tolerance", 1e-10)
    r.integrator.setValue("absolute_tolerance", 1e-12)
    sids = list(r.model.getFloatingSpeciesIds())
    r.selections = ["time"] + sids
    ref = np.asarray(r.simulate(0, 5, 6))

    model = bngsim.Model.from_sbml_string(src)
    result = bngsim.Simulator(model, method="ode").run(
        sample_times=list(ref[:, 0]), rtol=1e-10, atol=1e-12
    )
    got = dict(zip(model.species_names, np.asarray(result.species).T, strict=False))
    for j, sid in enumerate(sids, start=1):
        np.testing.assert_allclose(got[sid], ref[:, j], rtol=1e-6, atol=1e-9)
