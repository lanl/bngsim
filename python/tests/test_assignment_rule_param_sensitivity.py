"""Issue #329 — a parameter an ``<assignmentRule>`` defines is not a sensitivity column.

The #328 degeneracy census found three corpus models with a moving state
trajectory and a forward-sensitivity tensor that was exactly ``0.0`` for every
sampled parameter, and read it as issue #313 one construct over: a missing chain
rule through ``<assignmentRule>``, the way ``<initialAssignment>`` used to be
folded to a constant that nothing differentiated. Two separate things had to be
true for that reading, and neither is.

**The chain rule is there.** ``test_the_chain_rule_reaches_the_parameter_under_an_ar``
puts a parameter's *only* route to the right-hand side through an
``<assignmentRule>`` and checks ``dS/dp`` against a closed form — the exact shape
#313 described, differentiating correctly. #315 lifted ``<initialAssignment>``
because SBML evaluates it once, at load; an ``<assignmentRule>`` is a statement
about all time, and bngsim lowers it to a model **function** that the emitted
sensitivity right-hand side differentiates like any other expression.

**The zeros were true.** The three models' *actual* knobs really do have zero
influence on their states, which a finite difference through bngsim's own
trajectory settles without a reference engine
(``test_the_census_models_zeros_are_the_true_derivatives``). MODEL1006230116 is
the clearest: its one rate rule is ``d(Ca_sr)/dt = 1``, a constant, so every
parameter's derivative is zero and the state still spans 100.

What the census actually sampled was ``Model.param_names``, which on an
assignment-rule-driven model is mostly rule targets — 38 of 46 parameters in
BIOMD0000000126, 35 of 38 in BIOMD0000000266, 17 of 36 in MODEL1006230116. A
rule target is not a knob: bngsim binds a function to that slot and rewrites it
from the rule's expression before every derivative evaluation, so ``set_param``
refuses a value-changing write to it (#227/#266) and ``force_override`` does not
lift that refusal either. Issue #203 had already dropped these from
``compute_all_sensitivities(params=None)`` with a warning saying the column
"would be identically zero" — but naming one explicitly handed back that zero.
That silent zero is what made the census read a true answer as a bug, and it is
what #329 fixes: the explicit ask now raises, in the same three places the
compartment-size refusal (#164/#170) already does.

The refusal is narrow. ``_V0_<comp>`` is internal too and is deliberately left
answerable, because it genuinely appears in the emitted right-hand side; a
derived parameter (``param_is_expression``) is left answerable because
``force_override`` makes its column real, which is what
``bngsim.jax.differentiable_solve(flat=True)`` differentiates over. Only the
class where no write of any kind moves the coordinate is refused.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

_MODELS_DIR = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"

_T_SPAN = (0.0, 3.0)
_N_POINTS = 7


# ── The construct the issue hypothesized was broken ─────────────────────────
#
# `p` reaches the reaction ONLY through the assignmentRule target `q = 2*p`, so
# S(t) = exp(-2 p t) and dS/dp = -2 t exp(-2 p t) exactly. Nothing here is a
# finite difference.
_SBML_AR_CHAIN = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_chain">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p" value="0.3" constant="true"/>
      <parameter id="q" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="q">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>2</cn><ci>p</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="degrade" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>c</ci><ci>q</ci><ci>S</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

# Two rules deep: `p -> q = 2*p -> r = q*q`, so S(t) = exp(-4 p^2 t) and
# dS/dp = -8 p t exp(-4 p^2 t). One link differentiating is not the same claim
# as a chain of them differentiating, and the census models are chains.
_SBML_AR_TWO_DEEP = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="ar_two_deep">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="p" value="0.3" constant="true"/>
      <parameter id="q" value="0" constant="false"/>
      <parameter id="r" value="0" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="r">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>q</ci><ci>q</ci></apply>
        </math>
      </assignmentRule>
      <assignmentRule variable="q">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>2</cn><ci>p</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="degrade" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>c</ci><ci>r</ci><ci>S</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

# The narrowness control: `k*A` at net compartment power -1 folds as `k/V`, so
# the loader synthesizes `_V0_C` to hold the load-time size. Internal, refused by
# `set_param`, and deliberately NOT refused a sensitivity column (#203).
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


def _run(model, params):
    return bngsim.Simulator(model, method="ode", sensitivity_params=params).run(_T_SPAN, _N_POINTS)


# ── The chain rule the issue hypothesized was missing ───────────────────────


def test_the_chain_rule_reaches_the_parameter_under_an_ar():
    """``dS/dp`` where ``p`` reaches the RHS only through ``<assignmentRule>``.

    This is #313's shape one construct over, and the one the issue predicted
    would be identically zero. It is the closed form instead.
    """
    m = bngsim.Model.from_sbml_string(_SBML_AR_CHAIN)
    assert "q" in m.function_names, "the rule target must lower to a function"

    res = _run(m, ["p"])
    t = np.asarray(res.time)
    got = np.asarray(res.sensitivities)[:, 0, 0]

    np.testing.assert_allclose(got, -2.0 * t * np.exp(-0.6 * t), rtol=1e-6, atol=1e-9)
    assert np.abs(got).max() > 1.0, "a nonzero column is the whole point"


def test_the_chain_rule_survives_two_assignment_rules_in_series():
    """``p -> q = 2p -> r = q^2 -> rate``, against ``dS/dp = -8 p t exp(-4 p^2 t)``.

    Also pins the *dependency order*: the document declares ``r`` before the ``q``
    it reads, so a one-pass evaluator in document order would differentiate a
    stale ``q`` — the ordering hazard #315 hit for ``<initialAssignment>``.
    """
    m = bngsim.Model.from_sbml_string(_SBML_AR_TWO_DEEP)
    res = _run(m, ["p"])
    t = np.asarray(res.time)
    got = np.asarray(res.sensitivities)[:, 0, 0]

    p = 0.3
    np.testing.assert_allclose(got, -8.0 * p * t * np.exp(-4.0 * p * p * t), rtol=1e-6, atol=1e-9)


# ── The refusal: an AR target is not a coordinate ───────────────────────────


def test_naming_the_rule_target_raises_instead_of_answering_zero():
    m = bngsim.Model.from_sbml_string(_SBML_AR_CHAIN)
    with pytest.raises(ValueError, match="identically zero"):
        bngsim.Simulator(m, method="ode", sensitivity_params=["q"])


def test_the_refusal_covers_every_explicit_ask():
    """All three doors, so the refusal is not one entry point deep.

    ``compute_all_sensitivities(params=...)`` and
    ``steady_state(sensitivity_params=...)`` reach the same emitted sensitivity
    RHS by different routes; the constructor is only the most-used one.
    """
    m = bngsim.Model.from_sbml_string(_SBML_AR_CHAIN)

    with pytest.raises(ValueError, match=r"issue #329"):
        bngsim.Simulator(m, method="ode", sensitivity_params=["q"])
    with pytest.raises(ValueError, match=r"issue #329"):
        bngsim.Simulator(m, method="ode").compute_all_sensitivities(
            _T_SPAN, _N_POINTS, params=["q"], n_workers=1
        )
    with pytest.raises(ValueError, match=r"issue #329"):
        bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=["q"])


def test_a_mixed_request_is_refused_whole_and_names_only_the_bad_entry():
    """Asking for ``["p", "q"]`` refuses, and the message names ``q`` alone.

    A partial answer is the failure mode this replaces: it would put a real
    column and a structurally-zero one in the same tensor with nothing marking
    which was which.
    """
    m = bngsim.Model.from_sbml_string(_SBML_AR_CHAIN)
    with pytest.raises(ValueError) as exc:
        bngsim.Simulator(m, method="ode", sensitivity_params=["p", "q"])
    assert "['q']" in str(exc.value), str(exc.value)


def test_the_refusal_matches_what_set_param_refuses():
    """The same slot, refused by the same rule, for the same reason.

    A sensitivity column is the derivative of a write; where the write is
    refused and cannot even be forced, the column has nothing to be the
    derivative of. Tying the two together here keeps them from drifting apart.
    """
    m = bngsim.Model.from_sbml_string(_SBML_AR_CHAIN)

    with pytest.raises(ValueError, match=r"issue #227"):
        m.set_param("q", 99.0)
    # Not even the escape hatch that pins a derived parameter (#188).
    with pytest.raises(ValueError, match=r"issue #227"):
        m._core.set_param("q", 99.0, force_override=True)
    with pytest.raises(ValueError, match=r"issue #329"):
        bngsim.Simulator(m, method="ode", sensitivity_params=["q"])


# ── Narrowness: what is NOT swept up ────────────────────────────────────────


def test_the_load_time_volume_record_is_still_answerable():
    """``_V0_<comp>`` is internal too, and #203 shipped its explicit ask working.

    It differs from a function's slot in the way that matters here: it really is
    in the emitted RHS, so its column is not structurally zero. This refusal must
    not widen into it.
    """
    m = bngsim.Model.from_sbml_string(_SBML_ONE_COMPARTMENT)
    (v0,) = [n for n in m.param_names if n.startswith("_V0_")]
    assert v0 in m._internal_param_names() and v0 not in set(m.function_names)

    res = bngsim.Simulator(m, method="ode").compute_all_sensitivities(
        _T_SPAN, _N_POINTS, params=[v0], n_workers=1
    )
    assert res.sensitivity_params == [v0]


def test_the_default_column_set_still_only_warns():
    """``params=None`` keeps dropping-with-a-warning; only the explicit ask hardened.

    Raising there would make the method unusable on any assignment-rule-driven
    model for the sake of columns nobody asked for by name — the reason #164
    chose a warning for the compartment sizes in the first place.
    """
    m = bngsim.Model.from_sbml_string(_SBML_AR_CHAIN)
    with pytest.warns(UserWarning, match=r"skipping \d+ internal parameter\(s\)"):
        res = bngsim.Simulator(m, method="ode").compute_all_sensitivities(
            _T_SPAN, _N_POINTS, n_workers=1
        )
    assert "q" not in res.sensitivity_params
    assert res.sensitivity_params == m.primary_param_names


def test_a_model_with_no_rule_targets_is_untouched_and_silent():
    """The control that keeps this from being "refuse something, always"."""
    m = bngsim.Model.from_sbml_string(_SBML_ONE_COMPARTMENT)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = _run(m, ["k"])
    assert np.abs(np.asarray(res.sensitivities)).max() > 0.0


# ── The census models: the zeros were the truth ─────────────────────────────

# (model directory, file stem) for the three #329 named. Gated on the file, not
# the directory: the corpus is gitignored, so `models/` exists and is empty in a
# fresh worktree and in CI.
_CENSUS = [
    ("BIOMD0000000126", "BIOMD0000000126_url.xml"),
    ("BIOMD0000000266", "BIOMD0000000266_url.xml"),
    ("MODEL1006230116", "MODEL1006230116.xml"),
]


@pytest.mark.parametrize(("model_id", "fname"), _CENSUS)
def test_the_census_models_zeros_are_the_true_derivatives(model_id, fname):
    """Analytic ``0`` is confirmed by a finite difference through bngsim itself.

    The issue's own suggested first check, and the one that needs no reference
    engine: perturb a real knob, re-integrate, difference. If the analytic column
    is zero and the trajectory does not move either, the zero is the answer.

    The state DOES move — asserted here — so this is not a model that does
    nothing; it is a model whose motion does not depend on its parameters.
    """
    path = _MODELS_DIR / model_id / fname
    if not path.exists():
        pytest.skip(f"corpus model not present: {path}")
    src = path.read_text()

    t_span, n_points = (0.0, 10.0), 11

    base = bngsim.Simulator(bngsim.Model.from_sbml_string(src)).run(t_span, n_points)
    assert float(np.ptp(np.asarray(base.species))) > 1e-3, "the state must move"

    m = bngsim.Model.from_sbml_string(src)
    # Every rule target is refused rather than answered zero — the #329 fix, on
    # the models that motivated it.
    targets = sorted(m._internal_param_names() & set(m.function_names))
    assert targets, f"{model_id} is meant to be assignment-rule driven"
    with pytest.raises(ValueError, match=r"issue #329"):
        bngsim.Simulator(m, method="ode", sensitivity_params=[targets[0]])

    # And every real knob's analytic column matches a finite difference through
    # bngsim's own trajectory. Compartment sizes are excluded: #170 answers those
    # from a rebuild, not from a `set_param` difference.
    knobs = [p for p in m.primary_param_names if p not in set(m.compartment_size_params)]
    assert knobs, f"{model_id} must have at least one differentiable knob"

    for name in knobs:
        v = float(m.get_param(name))
        h = 1e-4 * (abs(v) or 1.0)

        def _at(value, _name=name):
            mm = bngsim.Model.from_sbml_string(src)
            mm.set_param(_name, value)
            return np.asarray(bngsim.Simulator(mm).run(t_span, n_points).species)

        fd = (_at(v + h) - _at(v - h)) / (2.0 * h)
        analytic = np.asarray(
            bngsim.Simulator(
                bngsim.Model.from_sbml_string(src), method="ode", sensitivity_params=[name]
            )
            .run(t_span, n_points)
            .sensitivities
        )[:, :, 0]

        scale = max(1.0, float(np.abs(fd).max()))
        np.testing.assert_allclose(
            analytic,
            fd,
            rtol=1e-4,
            atol=1e-6 * scale,
            err_msg=f"{model_id}: d(x)/d({name}) disagrees with its finite difference",
        )
