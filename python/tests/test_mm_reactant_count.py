"""A Michaelis-Menten reaction with more than two reactants (issue #1502).

Every engine reads an MM reaction's first two reactants, the enzyme and the
substrate, and nothing after them. ``S + E1 + E2 -> P + E1 + E2`` at
``MM kcat Km`` (S = 10, E1 = 1, E2 = 3) ran with E1 as the whole enzyme,
P(1) = 0.896350, and with E2 when the enzymes were listed the other way round,
P(1) = 2.574841, with no warning. run_network 3.0 reads the first reactant as
the substrate and the total of the others as the enzyme, P(1) = 3.318492;
BNG2.pl refuses the rule ("Michaelis-Menton type ratelaw require exactly 2
reactants").

Such a reaction is now refused, on every route that builds a model. Measured
with run_network on the files below: two reactants, a homodimer ``1,1``
among them, read the same here and there; fewer than two give a rate of 0 in
both, and keep loading. The functional rewrite the refusal asks for gives
run_network's 3.318492.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder
from bngsim.convert import net_to_sbml

REFUSAL = "a Michaelis-Menten rate law reads two, an enzyme and a substrate"
SPECIES = "begin species\n 1 S() 10\n 2 E1() 1\n 3 E2() 3\n 4 P() 0\n 5 E3() 2\nend species\n"
# P(1) from run_network 3.0, cvode at rtol 1e-10 and atol 1e-12, as it prints it.
RN_S_E1 = 0.896350382205
RN_HOMODIMER = 4.968136219291
RN_THREE = 3.318492133909  # the issue's three-reactant line
# RUN_NETWORK: bngsim at the same tolerances against those, measured on macOS
# arm64: 8e-14 for the same file, and 1.9e-10 for the functional rewrite against
# the three-reactant line, two integrations of one law. 1e-8 leaves room for
# another platform's solver; a dropped enzyme is 70% off.
RUN_NETWORK = 1e-8


def _net(reactants: str, products: str) -> str:
    return (
        "begin parameters\n 1 kcat 1\n 2 Km 1\nend parameters\n"
        + SPECIES
        + f"begin reactions\n 1 {reactants} {products} MM kcat Km\nend reactions\n"
        + "begin groups\n 1 Pobs 4\nend groups\n"
    )


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "m.net"
    path.write_text(text)
    return path


def _p_at_1(model: bngsim.Model) -> float:
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 1.0), n_points=2, rtol=1e-10, atol=1e-12
    )
    return float(np.asarray(result.observables)[-1, 0])


def _tqssa(s: float, e: float) -> float:
    """The closed-form tQSSA rate at kcat = Km = 1."""
    b = s + e + 1.0
    return 0.5 * (b - math.sqrt(b * b - 4.0 * s * e))


# ── Refused ──────────────────────────────────────────────────────────────────

MORE_THAN_TWO = {
    # The issue's two orders: E1 or E2 was read as the whole enzyme.
    "s-e1-e2": ("1,2,3", "4,2,3", 3),
    "s-e2-e1": ("1,3,2", "4,3,2", 3),
    # run_network 1.759815: an enzyme listed twice counts twice.
    "s-e1-e1": ("1,2,2", "4,2,2", 3),
    # run_network 4.483966.
    "four-reactants": ("1,2,3,5", "4,2,3,5", 4),
}


@pytest.mark.parametrize("case", sorted(MORE_THAN_TWO))
def test_an_mm_reaction_with_more_than_two_reactants_is_refused(tmp_path, case):
    """It loaded, and ran on its first two reactants alone."""
    reactants, products, n = MORE_THAN_TWO[case]
    path = _write(tmp_path, _net(reactants, products))
    with pytest.raises(bngsim.ModelError, match=REFUSAL) as caught:
        bngsim.Model.from_net(str(path))
    message = str(caught.value)
    assert f"reaction 0 (MichaelisMenten) has {n} reactants" in message
    assert "BNG2.pl refuses such a rule" in message and "run_network reads the first" in message
    with pytest.raises(ValueError, match=REFUSAL):
        bngsim.parse_net_file(str(path))


def test_every_route_that_builds_it_refuses_it(tmp_path):
    """``Model.load`` and ``net_to_sbml`` go through the loader; a model put
    together with ``ModelBuilder`` is checked at ``build()``. ``net_to_sbml``
    wrote a kinetic law without E2 and called the conversion fine."""
    path = str(_write(tmp_path, _net("1,2,3", "4,2,3")))
    with pytest.raises(bngsim.ModelError, match=REFUSAL):
        bngsim.Model.load(path)
    with pytest.raises(bngsim.ModelError, match=REFUSAL):
        net_to_sbml(path)
    builder = ModelBuilder()
    builder.add_parameter("kcat", 1.0)
    builder.add_parameter("Km", 1.0)
    species = (("S", 10.0), ("E1", 1.0), ("E2", 3.0), ("P", 0.0))
    s, e1, e2, p = (builder.add_species(n, v) for n, v in species)
    builder.add_reaction([s, e1, e2], [p, e1, e2], "mm", "kcat,Km")
    with pytest.raises(RuntimeError, match=REFUSAL):
        builder.build()


# ── Read as before ───────────────────────────────────────────────────────────

TWO = {
    # The closed-form rate at t = 0 is tQSSA(10, 1).
    "s-e1": ("1,2", "4,2", RN_S_E1, _tqssa(10.0, 1.0)),
    # The substrate is its own enzyme.
    "homodimer": ("1,1", "4,1", RN_HOMODIMER, _tqssa(10.0, 10.0)),
}


@pytest.mark.parametrize("case", sorted(TWO))
def test_two_reactants_read_as_run_network_reads_them(tmp_path, case):
    """Control: loaded on main too, to run_network's numbers."""
    reactants, products, p_at_1, rate_at_0 = TWO[case]
    model = bngsim.Model.from_net(str(_write(tmp_path, _net(reactants, products))))
    rhs = model.rhs(np.asarray(model.get_state()))
    assert rhs[3] == pytest.approx(rate_at_0, rel=1e-12)
    assert _p_at_1(model) == pytest.approx(p_at_1, rel=RUN_NETWORK)


@pytest.mark.parametrize("reactants", ["0", "1"])
def test_fewer_than_two_reactants_still_load_at_a_rate_of_0(tmp_path, reactants):
    """Control: loaded on main too. With no enzyme, the rate is 0 here and in
    run_network alike, so refusing it would refuse what main gets right."""
    model = bngsim.Model.from_net(str(_write(tmp_path, _net(reactants, "4"))))
    assert _p_at_1(model) == 0.0


def test_the_functional_rewrite_the_refusal_asks_for_is_run_network_s_reading(tmp_path):
    """Control: loads on main too. ``S -> P`` at the tQSSA rate over the
    enzymes' total gives the 3.318492 run_network gives the three-reactant
    line, and run_network gives this file the same."""
    text = (
        "begin parameters\n 1 kcat 1\n 2 Km 1\nend parameters\n" + SPECIES + "begin functions\n"
        " 1 sFree() 0.5*((Sobs-Km-Etot)+sqrt((Sobs-Km-Etot)^2+4*Km*Sobs))\n"
        " 2 f() kcat*sFree()*Etot/(Km+sFree())/Sobs\n"
        "end functions\n"
        "begin reactions\n 1 1 4 f\nend reactions\n"
        "begin groups\n 1 Pobs 4\n 2 Sobs 1\n 3 Etot 2,3\nend groups\n"
    )
    model = bngsim.Model.from_net(str(_write(tmp_path, text)))
    assert _p_at_1(model) == pytest.approx(RN_THREE, rel=RUN_NETWORK)
