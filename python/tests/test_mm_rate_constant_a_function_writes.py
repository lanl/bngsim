"""A Michaelis-Menten reaction whose ``kcat`` or ``Km`` is a parameter that a
function of the same name writes (issue #931).

A Michaelis-Menten rate law reads its two constants as parameters. The
interpreted ODE right-hand side evaluates every function into the parameter
slot of its name first, so there the constant followed the function. Nothing
else did:

- the compiled right-hand side read the slot as it stood: with ``kcat() = kb``
  over a slot at 0 the reaction never ran, P(6) = 0 for 18 under
  ``codegen=True`` and in every forward-sensitivity run, which always uses the
  compiled right-hand side, with dP/dkb = 0 for 6;
- the compiled SSA propensities held the slot at its first value, 502 for 99
  with ``kcat() = kb*Aobs``, and the interpreted SSA did not know the reaction
  reads what the function reads (272 or 104, by how often the run was
  sampled);
- the Jacobian took the constant for one.

Nothing was logged in any of them. BNG2.pl writes no such model, and the
model is refused where it is built.

E = 1 and S = 1e6 with Km = 1, so the rate is kcat·S/(Km + S) to six digits and
P(6) = 6·kcat.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 kb 3
    2 kcat {kcat}
    3 Km {km}
end parameters
begin functions
    1 {function}
end functions
begin species
    1 E() 1
    2 S() 1e6
    3 P() 0
    4 Q() 0
end species
begin reactions
    1 3 4 kb
    2 1,2 1,3 {law}
end reactions
begin groups
    1 Sobs 2
end groups
"""


def _write(tmp_path, function="kcat() kb", law="MM kcat Km", kcat="0", km="1"):
    path = tmp_path / "mm.net"
    path.write_text(NET.format(function=function, law=law, kcat=kcat, km=km))
    return str(path)


def _made(sim):
    """P + Q at t = 6: what the Michaelis-Menten reaction made."""
    run = sim.run(t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12)
    names = list(run.species_names)
    end = np.asarray(run.species)[-1]
    return float(end[names.index("P()")] + end[names.index("Q()")])


SHAPES = {
    "kcat-at-0": dict(),
    "kcat-at-7": dict(kcat="7"),
    "kcat-equal-to-what-the-function-gives": dict(kcat="3"),
    "kcat-of-the-state": dict(function="kcat() kb*Sobs/1e6"),
    "kcat-of-the-time": dict(function="kcat() if(time()>3,2*kb,0)"),
    "km": dict(function="Km() kb/3", law="MM kb Km", km="5e6"),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_model_is_refused_where_it_is_built(tmp_path, shape):
    """Whatever the slot holds and whatever the function reads: the second
    reaction of the model, with the one before it left alone."""
    name = "Km" if shape == "km" else "kcat"
    with pytest.raises(
        bngsim.ModelError, match=rf"MichaelisMenten.*'{name}'.*issue #931"
    ) as caught:
        bngsim.Model.from_net(_write(tmp_path, **SHAPES[shape]))
    assert "reaction 1 " in str(caught.value)


def test_the_constant_as_a_parameter_that_is_an_expression(tmp_path):
    """Control. What the refusal points at: ``kcat`` as a parameter written in
    others. Interpreted and compiled agree, and the sensitivity is right."""
    text = NET.format(function="unused() kb*2", law="MM kcat Km", kcat="kb*2", km="1")
    path = tmp_path / "expr.net"
    path.write_text(text)
    for codegen in (False, True):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=codegen)
        assert _made(sim) == pytest.approx(36.0, rel=1e-5)
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["kb"]
    )
    run = sim.run(t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12)
    names = list(run.species_names)
    sens = np.asarray(run.sensitivities)[-1, :, 0]
    assert sens[names.index("P()")] + sens[names.index("Q()")] == pytest.approx(12.0, rel=1e-5)


def test_a_function_of_another_name_that_reads_the_constant(tmp_path):
    """Control. ``twice() = 2*kcat`` reads the parameter and does not write
    it."""
    path = _write(tmp_path, function="twice() 2*kcat", kcat="3")
    for codegen in (False, True):
        sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", codegen=codegen)
        assert _made(sim) == pytest.approx(18.0, rel=1e-5)


@pytest.mark.parametrize("law", ["Sat kcat Km", "kcat"], ids=["sat", "elementary"])
def test_the_other_rate_laws_follow_a_function_of_their_constant_s_name(tmp_path, law):
    """Control. The loader makes these two functional rate laws, which every
    engine reads through the function: the same compiled and interpreted."""
    compiled = _made(
        bngsim.Simulator(
            bngsim.Model.from_net(_write(tmp_path, law=law)), method="ode", codegen=True
        )
    )
    plain = _made(
        bngsim.Simulator(
            bngsim.Model.from_net(_write(tmp_path, law=law)), method="ode", codegen=False
        )
    )
    assert compiled == pytest.approx(plain, rel=1e-8)
    assert compiled > 0.0
