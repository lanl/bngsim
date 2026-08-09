"""Issue #203 — the ``params=None`` column set must be independent coordinates.

``compute_all_sensitivities(params=None)`` means "everything computable", and
:meth:`Result.gradient` contracts that tensor's whole parameter axis into one
``(n_params,)`` vector which its own docstring hands to
``scipy.optimize.minimize`` over a parameter vector of the same width. That is
only a gradient if no two columns describe the same physical effect.

A **derived** parameter breaks it. ``k = 2 * k_base`` reaches the trajectory
only through ``k_base``, and ``k_base``'s column is a *total* derivative through
``k`` — that is the chain rule #188 restored. So the two columns are the same
information twice, in exact proportion ``dk/dk_base``, and an optimizer told
both are free coordinates counts that effect twice.
``test_a_derived_column_is_the_primarys_column_rescaled`` is that statement with
no constant in it.

A synthesized ``_V0_<comp>`` breaks it differently: it is bngsim's record of a
compartment's size at load, which the rate constants in that compartment are
normalised against, so ``set_param`` refuses a value-changing write to it
(``test_the_load_time_volume_record_is_not_a_knob`` in
:mod:`test_compartment_size_live` owns that contract) and its column is the
derivative of a coordinate that cannot move on its own.

Both are now dropped from the default with a warning naming them — the pattern
issue #164 established in the same five lines for the compartment sizes
``set_param`` refuses — which makes the default exactly
:attr:`Model.primary_param_names` minus that unwritable residue, and therefore
the same list ``bngsim.jax.differentiable_solve`` differentiates over by default
(``flat=False``). Naming either class in ``params=[...]`` still returns its
column: an explicit ask is a statement that you want that derivative on its own
terms, which is what ``flat=True`` is.

The same hazard one level down is issue #155: the ``parameter`` axis of an
initial-condition sensitivity is a *total* derivative and must not be summed
with the ``ic`` axis.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

_REPRO = Path(__file__).resolve().parents[2] / "benchmarks" / "sbml_events" / "BIOMD0000000701.xml"
_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"
# Gate on a model file actually being present, not on the directory: the corpus
# is gitignored, so `models/` exists and is empty in a fresh worktree and in CI,
# and an `is_dir()` gate is how #192 shipped a 77-model regression.
_CORPUS_MODELS = sorted(_MODELS_DIR.glob("*/*.xml")) if _MODELS_DIR.is_dir() else []

_T_SPAN = (0.0, 2.0)
_N_POINTS = 5


# ── The self-contained model: `k = 2 * k_base`, S' = -k*S, S(0) = 100 ───────
#
# Closed form S(t) = 100 exp(-k t) with k = 1, so every column below has an
# exact oracle and nothing here is a finite difference:
#
#     dS/dk      = -100 t exp(-k t)          (k as an independent coordinate)
#     dS/dk_base = -100 t exp(-k t) dk/dk_base = -200 t exp(-t)   (total)
#
# The same shape as `_rateLaw1 = chi*kon` and as BIOMD0000000701's
# `_rateLaw_R16_fwd = alpha * konBT`, with a coefficient small enough to check
# by hand.


def _derived_model() -> bngsim.Model:
    b = ModelBuilder()
    b.add_parameter("k_base", 0.5)
    b.add_parameter("k", 0.0, expression="2 * k_base", is_expression=True)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    return bngsim.Model(b.build())


def _plain_model() -> bngsim.Model:
    """The control: same kinetics, no expression anywhere."""
    b = ModelBuilder()
    b.add_parameter("k", 1.0)
    s_idx = b.add_species("S", 100.0)
    b.add_reaction([s_idx], [], "elementary", "k")
    return bngsim.Model(b.build())


def _sens(model, **kw):
    return bngsim.Simulator(model, method="ode").compute_all_sensitivities(
        _T_SPAN, _N_POINTS, n_workers=1, **kw
    )


# ── The default column set ──────────────────────────────────────────────────


def test_the_default_drops_a_derived_parameter_and_says_so():
    m = _derived_model()
    assert m.param_names == ["k_base", "k"]

    with pytest.warns(UserWarning, match=r"skipping 1 derived parameter\(s\).*'k'"):
        res = _sens(m)

    assert res.sensitivity_params == ["k_base"]
    assert res.sensitivity_params == m.primary_param_names
    assert res.sensitivities.shape[2] == 1


def test_a_model_with_no_derived_parameters_is_untouched_and_silent():
    """The control that keeps the filter from being "drop something, always"."""
    m = _plain_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = _sens(m)

    assert res.sensitivity_params == m.param_names == ["k"]


def test_dropping_a_column_did_not_move_the_ones_kept():
    """``dS/dk_base`` is still the closed form, to solver tolerance.

    The filter removes columns; it must not perturb the remaining ones. Checked
    against the analytic solution rather than against the pre-#203 tensor, so
    this stays an oracle rather than a snapshot of whatever shipped.
    """
    res = _sens(_derived_model())
    t = np.asarray(res.time)
    got = np.asarray(res.sensitivities)[:, 0, 0]

    np.testing.assert_allclose(got, -200.0 * t * np.exp(-t), rtol=1e-6, atol=1e-9)


# ── Why those columns are not coordinates ───────────────────────────────────


def test_a_derived_column_is_the_primarys_column_rescaled():
    """The non-independence itself, with the constant taken from the model.

    ``dS/dk_base == (dk/dk_base) * dS/dk`` identically — so the derived column
    carries nothing the primary's column does not already carry, and a search
    direction built from both moves along that effect twice. ``dk/dk_base`` is
    measured off the model rather than written as ``2``, so this test is about
    the chain rule and not about this fixture's coefficient.
    """
    m = _derived_model()
    res = _sens(m, params=["k_base", "k"])
    assert res.sensitivity_params == ["k_base", "k"]
    S = np.asarray(res.sensitivities)

    # dk/dk_base, off the model: writing the primary re-derives the expression.
    kb0, k0 = m.get_param("k_base"), m.get_param("k")
    m.set_param("k_base", kb0 * 1.5)
    dk_dkbase = (m.get_param("k") - k0) / (kb0 * 0.5)
    m.set_param("k_base", kb0)
    assert dk_dkbase == pytest.approx(2.0)

    np.testing.assert_allclose(S[:, :, 0], dk_dkbase * S[:, :, 1], rtol=1e-8, atol=1e-12)


def test_every_default_column_is_a_coordinate_no_other_column_moves():
    """The invariant, stated as the property "independent" actually means.

    Write one column's parameter; no other column's parameter may move. The
    pre-#203 default fails this on the same model — writing ``k_base`` moves
    ``k``, which the second half asserts so that a regression restoring the old
    list cannot pass by making both halves equally wrong.
    """
    m = _derived_model()
    default = _sens(m).sensitivity_params

    for name in default:
        others = {n: m.get_param(n) for n in default if n != name}
        m.set_param(name, m.get_param(name) * 1.5)
        for n, v in others.items():
            assert m.get_param(n) == v, f"writing {name!r} moved {n!r}"

    # ...and the list that was the default before #203 is not such a set.
    m = _derived_model()
    k_before = m.get_param("k")
    m.set_param("k_base", m.get_param("k_base") * 1.5)
    assert m.get_param("k") != k_before, "the old default's columns were not independent"


def test_gradient_is_aligned_with_the_default_column_set():
    """:meth:`Result.gradient`'s docstring builds the fitted vector from
    ``model.primary_param_names`` and asserts ``result.sensitivity_params ==
    names``. That assertion is the contract; run it."""
    m = _derived_model()
    names = m.primary_param_names
    res = _sens(m)

    assert res.sensitivity_params == names
    grad = res.gradient(lambda species, time: np.ones_like(species))
    assert grad.shape == (len(names),)

    # The value, in closed form: sum_t dS/dk_base.
    t = np.asarray(res.time)
    assert grad[0] == pytest.approx(np.sum(-200.0 * t * np.exp(-t)), rel=1e-6)


def test_the_old_column_set_made_the_fisher_matrix_singular_by_construction():
    """The second consumer of the parameter axis, and the sharper symptom.

    ``fisher_information`` is ``Sᵀ Σ⁻¹ S`` over that axis. Two exactly
    proportional columns make it **rank-deficient by construction**, not by
    anything about the model — and the identifiability advice in the user guide
    (``eigvalsh``, smallest eigenvalues → least identifiable) reads that
    round-off eigenvalue as a scientific finding. The default column set has no
    such direction.
    """
    m = _derived_model()

    fim_old = np.asarray(_sens(m, params=["k_base", "k"]).fisher_information(sigma=0.1))
    ev = np.linalg.eigvalsh(fim_old)
    assert fim_old.shape == (2, 2)
    assert ev[0] / ev[-1] < 1e-14, f"expected a null direction, got eigenvalues {ev}"

    fim_new = np.asarray(_sens(m).fisher_information(sigma=0.1))
    assert fim_new.shape == (1, 1)
    assert np.linalg.matrix_rank(fim_new) == 1
    assert fim_new[0, 0] > 0.0


# ── The escape hatch ────────────────────────────────────────────────────────


def test_naming_a_derived_parameter_still_returns_its_column():
    """An explicit ask means "on its own terms" — the axis ``flat=True`` uses.

    And it is the right derivative: ``dS/dk = -100 t exp(-k t)``, the column
    that treats ``k`` as free rather than as ``2*k_base``.
    """
    res = _sens(_derived_model(), params=["k"])

    assert res.sensitivity_params == ["k"]
    t = np.asarray(res.time)
    np.testing.assert_allclose(
        np.asarray(res.sensitivities)[:, 0, 0],
        -100.0 * t * np.exp(-t),
        rtol=1e-6,
        atol=1e-9,
    )


def test_an_explicit_full_parameter_list_is_honored_unchanged():
    """``params=model.param_names`` still reproduces the pre-#203 tensor.

    The filter is a *default*, not a refusal — nothing about the explicit path
    changed, which is what keeps ``differentiable_solve(flat=True)`` working.
    """
    m = _derived_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = _sens(m, params=m.param_names)

    assert res.sensitivity_params == ["k_base", "k"]


# ── The load-time volume record (issue #170), on the same footing ───────────

# `k*A` is net compartment power -1, which the loader folds as `sf = k/V` — so
# the rate constant is normalised against the load-time volume and `_V0_C` is
# synthesized to hold it. (`C*k*A`, where the compartment cancels against the
# storage divide, produces no `_V0_` at all; see :mod:`test_compartment_size_live`,
# which tabulates both.)
_SBML_ONE_COMPARTMENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m">
    <listOfCompartments>
      <compartment id="C" size="2.5" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="C" initialConcentration="10"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="R" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_the_default_drops_the_load_time_volume_record_and_says_so():
    """``_V0_<comp>`` is not a knob, so it is not a column either.

    ``set_param`` refuses a value-changing write to it — moving it rescales
    every rate in the compartment while the volume stays put — so a gradient
    entry for it is one an optimizer would fit against nothing. The compartment
    size ``C`` itself is an ordinary writable parameter since #170 stage 1 and
    stays in the tensor, which is what makes this a narrow drop rather than "no
    compartment gradients".
    """
    m = bngsim.Model.from_sbml_string(_SBML_ONE_COMPARTMENT)
    (v0,) = [n for n in m.param_names if n.startswith("_V0_")]
    assert v0 not in m.primary_param_names

    with pytest.warns(UserWarning, match=r"skipping 1 internal parameter\(s\)"):
        res = _sens(m)

    assert v0 not in res.sensitivity_params
    assert "C" in res.sensitivity_params and "k" in res.sensitivity_params
    assert res.sensitivity_params == m.primary_param_names

    # Still available by name — the drop is a default, and the JAX `flat=True`
    # axis reaches it through the same explicit path.
    named = _sens(m, params=[v0])
    assert named.sensitivity_params == [v0]


# ── The issue's own model ───────────────────────────────────────────────────


@pytest.mark.skipif(not _REPRO.exists(), reason=f"benchmark model not present: {_REPRO}")
def test_biomd701_default_is_its_primary_parameters():
    """#203's reproducer: 77 columns, 6 of them ``_rateLaw_R1*_fwd = alpha*konB*``.

    ``alpha`` reaches the trajectory only through those six — #188 proved it by
    pinning them and watching ``d(x)/d(alpha)`` go to exactly zero — so its
    column is a total derivative through all six, and the tensor reported each
    of those effects twice.
    """
    m = bngsim.Model.load(str(_REPRO))
    derived = [n for n, f in zip(m.param_names, m.param_is_expression, strict=True) if f]
    assert len(derived) == 6 and all(d.startswith("_rateLaw_R") for d in derived)

    with pytest.warns(UserWarning, match="skipping 6 derived parameter"):
        res = bngsim.Simulator(m).compute_all_sensitivities((0.0, 100.0), 3, n_workers=1)

    assert res.sensitivity_params == m.primary_param_names
    assert len(res.sensitivity_params) == len(m.param_names) - 6
    assert not set(res.sensitivity_params) & set(derived)


# ── The corpus assumption the filter is built on ────────────────────────────


@pytest.mark.skipif(not _CORPUS_MODELS, reason=f"rr_parity corpus not present: {_MODELS_DIR}")
def test_corpus_the_three_skip_classes_stay_disjoint():
    """Derived, internal and write-refused partition cleanly, and something is left.

    The filter reports each dropped name under exactly one of three reasons, and
    it computes the derived list as a *residue* of ``primary_param_names`` so
    that stays true even if the classes start overlapping. This is the sweep
    that says they do not overlap today (0 of 10,048 dropped parameters carries
    two flags across the corpus), and — the part that matters more — that no
    model is left with an empty column set by the narrowing.
    """
    checked = 0
    for path in _CORPUS_MODELS[:120]:
        try:
            m = bngsim.Model.load(str(path))
        except Exception:  # a handful of corpus models do not load; not this test's job
            continue
        names = set(m.param_names)
        if not names:
            continue
        checked += 1
        derived = {n for n, f in zip(m.param_names, m.param_is_expression, strict=True) if f}
        internal = names & m._internal_param_names()
        refused = names & set(m.unwritable_compartment_size_params)

        assert not derived & internal, f"{path.parent.name}: derived ∩ internal"
        assert not derived & refused, f"{path.parent.name}: derived ∩ write-refused"
        assert not internal & refused, f"{path.parent.name}: internal ∩ write-refused"
        assert set(m.primary_param_names) == names - derived - internal
        assert set(m.primary_param_names) - refused, (
            f"{path.parent.name}: narrowing the default left no columns at all"
        )

    assert checked > 50, f"corpus sweep reached only {checked} models"
