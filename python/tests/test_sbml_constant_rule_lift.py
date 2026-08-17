"""GH #385 — an ``<initialAssignment>`` may rest on an assignment rule, provided
it rests on the rule's *body* and the body is constant.

``BIOMD0000000587`` writes its knobs through COPASI's indirection::

    <initialAssignment symbol="ModelValue_0"> Theta            ModelValue_0 = Theta
    <assignmentRule variable="Alpha">  ModelValue_0/(24*3.344) Alpha        = Theta/80.256
    <initialAssignment symbol="ModelValue_1"> Alpha            ModelValue_1 = Theta/80.256

The lift took the first hop and refused the third, so ``ModelValue_1`` was
declared a **primary** parameter — an independent knob — when it is a function of
``Theta``. The *values* were right either way; what was lost was every
sensitivity term routed through it. On that model that is most of ``Theta``'s
column and half of ``rho_f``'s, enough to flip two of three signs against
RoadRunner and AMICI, both of which the issue measured at six significant
figures.

The refusal was deliberate and its reasoning was sound — for the case it was
written against. An assignment rule's *slot* is function-backed: the engine
rewrites it from the rule before every derivative evaluation, so a lifted
expression reading that slot re-derives from the rule's value at the last
integrated point rather than the ``t = 0`` fold the initialAssignment means.
``BIOMD0000000570``'s ``ModelValue_60 = O2c_bar`` is the case, and it would go
5.68 → 7.87 on the next write after a run.

What changed is that the lift now substitutes the rule's **body**, so the emitted
expression never mentions the slot — and it only does so when that body reaches
nothing but constant parameters, which is the condition under which the two
readings coincide anyway, because such a rule cannot move.
``O2c_bar`` is excluded by exactly that test: its rule reads species.

:class:`TestTheHazardIsStillExcluded` is the half that keeps this honest.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

_MODELS = Path(__file__).resolve().parents[2] / "parity_checks" / "rr_parity" / "models"

# Gate on the file, not the directory: the corpus is gitignored, so `models/`
# exists and is empty in a fresh worktree and in CI, and an `is_dir()` gate is how
# #192 shipped a 77-model regression.
_587 = _MODELS / "BIOMD0000000587" / "BIOMD0000000587_url.xml"


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bngsim.Model.from_sbml(str(path))


# ── The self-contained model: the three-hop chain, with nothing else in it ────
#
#   k_eff := <assignmentRule>  k_base * 2          (constant: k_base is a knob)
#   k_ia  := <initialAssignment> k_eff             (so k_ia == 2*k_base)
#   S' = -k_ia * S
#
# Written as SBML text rather than through ModelBuilder because the defect is in
# the SBML loader's lift, and a builder model never reaches it.
CHAIN_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="chain">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="100"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k_base" value="0.5" constant="true"/>
      <parameter id="k_eff"  value="1.0" constant="false"/>
      <parameter id="k_ia"   value="1.0" constant="true"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="k_eff">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k_base</ci><cn>2</cn></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfInitialAssignments>
      <initialAssignment symbol="k_ia">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>k_eff</ci></math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfReactions>
      <reaction id="decay" reversible="false">
        <listOfReactants>
          <speciesReference species="S" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>k_ia</ci><ci>S</ci><ci>c</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

RUN = {"t_span": (0.0, 2.0), "n_points": 5, "rtol": 1e-10, "atol": 1e-14}


@pytest.fixture
def chain(tmp_path):
    p = tmp_path / "chain.xml"
    p.write_text(CHAIN_SBML)
    return p


class TestTheChainIsFollowed:
    """``k_ia`` is a function of ``k_base``, so it is not a knob of its own."""

    def test_the_rule_target_is_not_primary(self, chain):
        m = _load(chain)
        assert "k_base" in m.primary_param_names
        assert "k_ia" not in m.primary_param_names, (
            "k_ia = k_eff = 2*k_base is a function of k_base, not an independent knob"
        )

    def test_the_value_is_unchanged(self, chain):
        """The lift is about the *derivative*. Reclassifying must not move what
        the parameter is worth."""
        m = _load(chain)
        assert m.get_param("k_ia") == pytest.approx(1.0, rel=1e-15)

    def test_writing_the_primary_moves_the_derived_one(self, chain):
        """The point of the classification: ``set_param`` on the knob propagates."""
        m = _load(chain)
        m.set_param("k_base", 1.5)
        assert m.get_param("k_ia") == pytest.approx(3.0, rel=1e-12)

    def test_the_identity_write_round_trips(self, chain):
        """The GH #570 hazard, stated directly: an optimizer writes the whole
        vector back every iteration, and that must move nothing."""
        m = _load(chain)
        before = {n: m.get_param(n) for n in m.param_names}
        m.set_params(dict(before))
        assert {n: m.get_param(n) for n in m.param_names} == before

    def test_the_sensitivity_carries_the_chain(self, chain):
        """``S(t) = 100·exp(-2·k_base·t)``, so ``dS/dk_base = -2t·S(t)``. Before
        the fix the column was identically zero: ``k_base`` reached the rate law
        only through ``k_ia``, and the chain rule stopped at the rule."""
        m = _load(chain)
        res = bngsim.Simulator(m, method="ode", sensitivity_params=["k_base"]).run(**RUN)
        S = np.asarray(res.sensitivities)[:, 0, 0]
        x = np.asarray(res.species)[:, 0]
        t = np.asarray(res.time)
        assert np.allclose(S, -2.0 * t * x, rtol=1e-6, atol=1e-9)
        assert np.abs(S).max() > 1.0, "a zero column would satisfy nothing above"


class TestTheHazardIsStillExcluded:
    """A rule whose body is *not* constant must stay out of the lift — that is
    the whole reason the exclusion existed."""

    def test_a_state_dependent_rule_is_not_substituted(self, tmp_path):
        """``k_eff := k_base * S`` moves with the trajectory, so ``k_ia``'s
        initialAssignment fold is a ``t = 0`` statement that a later write must
        not re-derive. It stays primary."""
        p = tmp_path / "state_rule.xml"
        p.write_text(
            CHAIN_SBML.replace(
                "<apply><times/><ci>k_base</ci><cn>2</cn></apply>",
                "<apply><times/><ci>k_base</ci><ci>S</ci></apply>",
            )
        )
        m = _load(p)
        assert "k_ia" in m.primary_param_names, (
            "a rule reading a species is not a constant expression and must not be "
            "substituted into the lift"
        )


@pytest.mark.skipif(not _587.exists(), reason="BIOMD0000000587 not available")
class TestTheCorpusWitness:
    """``BIOMD0000000587``, the model #385 was filed on."""

    def test_the_two_model_values_are_derived(self):
        m = _load(_587)
        primary = set(m.primary_param_names)
        assert {"Theta", "f", "rho_f"} <= primary
        assert not ({"ModelValue_1", "ModelValue_3"} & primary)

    def test_the_columns_match_a_finite_difference_of_the_trajectory(self):
        """The issue's own numbers, to which RoadRunner and AMICI agree to six
        figures: ``Theta`` −33.2326, ``f`` −37.80348, ``rho_f`` 334.1004. Before
        the fix bngsim returned +1.06159, −0.008265 and 180.43 — two of them the
        wrong sign.

        Asserted against a central difference of bngsim's own trajectory rather
        than against those constants, so this measures the chain rule rather than
        pinning three magic numbers to one corpus revision.

        The perturbation rewrites the ``value="…"`` attribute in the SBML **text**
        and reloads. It must not use ``set_param``: on the unfixed loader
        ``set_param("Theta", …)`` does not reach ``ModelValue_1`` either, so the
        reference would carry the very defect it is meant to detect and the two
        would agree — this test passed against unfixed code until the reference
        was moved into the text.
        """
        import re

        source = _587.read_text()
        cols = ["Theta", "f", "rho_f"]
        run = {"t_span": (0.0, 60.0), "n_points": 101, "rtol": 1e-9, "atol": 1e-12}

        def traj(text, tmp):
            tmp.write_text(text)
            return np.asarray(bngsim.Simulator(_load(tmp), method="ode").run(**run).species)

        S = np.asarray(
            bngsim.Simulator(_load(_587), method="ode", sensitivity_params=cols)
            .run(**run)
            .sensitivities
        )
        for j, name in enumerate(cols):
            pat = re.compile(rf'(<parameter[^>]*\bid="{re.escape(name)}"[^>]*\bvalue=")([^"]+)(")')
            mo = pat.search(source)
            assert mo is not None, f"{name} is not a <parameter value=...> in the text"
            p0 = float(mo.group(2))
            h = abs(p0) * 1e-3
            edited = [
                pat.sub(lambda _m, v=p0 + s * h: f"{_m.group(1)}{v!r}{_m.group(3)}", source, 1)
                for s in (+1, -1)
            ]
            tmp = self._tmp / f"{name}.xml"
            fd = (traj(edited[0], tmp) - traj(edited[1], tmp)) / (2 * h)
            scale = max(np.abs(fd).max(), np.abs(S[:, :, j]).max())
            assert scale > 1.0, f"{name}: nothing to compare"
            assert np.abs(S[:, :, j] - fd).max() / scale < 1e-3, name

    @pytest.fixture(autouse=True)
    def _tmpdir(self, tmp_path):
        self._tmp = tmp_path
