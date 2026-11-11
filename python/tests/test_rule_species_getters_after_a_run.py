"""An assignment-rule species reads its live value through the model after a
run (issue #1510).

An SBML assignment-rule target is a ``fixed`` slot the integrator does not
move, and only the report (``Simulator._apply_ar_report_map``) reads the rule.
A rule linear in species is read as an observable, which nothing copied back
into the slot, and an SSA run copied back neither kind. So after a run,
``Model.get_concentration``, ``Model.get_state``, ``Simulator.get_state``,
``ReactionKernel.get_state``, ``CouplingMap.read`` and ``Simulator.snapshot``
returned the slot's t=0 value: ``S := 2*A`` read 20 where the rule gives
2.706706, while ``Result.species`` reported the live value.

The run now leaves each rule target at the value it reports for the last row.
The slot is inert, read by no rate law and no event, so nothing that follows
moves: a continuation run and a ``carry_sensitivities`` pre-equilibration are
the bits they were.

Oracles: the closed form A(t) = 10 e^(-t/2), and for SSA the rule at the
model's own A.
"""

from __future__ import annotations

import contextlib
import io
import math
import shutil
import warnings

import bngsim
import numpy as np
import pytest
from bngsim.coupling import CouplingMap

ANT = """
model m
  compartment c = 1
  species A in c = 10
  species S in c
  species N in c
  A => ; k*A
  k = 0.5
  S := 2*A
  N := A^2/(5 + A)
end
"""
# A(4) = 10 e^-2; S is linear (an observable), N is not (a function).
A4 = 10 * math.exp(-2.0)
RULES = {"S": lambda a: 2 * a, "N": lambda a: a * a / (5 + a)}

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
needs_cc = pytest.mark.skipif(_CC is None, reason="no C compiler on PATH for the codegen leg")


def _quiet():
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    stack.enter_context(warnings.catch_warnings())
    warnings.simplefilter("ignore")
    return stack


def _model(text=ANT):
    return bngsim.Model.from_antimony_string(text)


@pytest.mark.parametrize(
    "kw",
    [
        pytest.param({}, id="interpreted"),
        pytest.param({"codegen": True}, id="codegen", marks=needs_cc),
    ],
)
def test_the_getters_read_the_rule_after_an_ode_run(kw):
    m = _model()
    with _quiet():
        sim = bngsim.Simulator(m, method="ode", **kw)
        r = sim.run(t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12)
    state = dict(zip(m.species_names, m.get_state(), strict=True))
    sim_state = dict(zip(m.species_names, sim.get_state(), strict=True))
    for name, rule in RULES.items():
        want = r.species[-1, r.species_names.index(name)]
        assert want == pytest.approx(rule(A4), rel=1e-8)
        assert m.get_concentration(name) == want
        assert state[name] == want and sim_state[name] == want


def test_the_getters_read_the_rule_after_an_ssa_run():
    """Under SSA neither kind of rule was copied back: N read 6.666667, its
    t=0 value, too."""
    m = _model()
    with _quiet():
        r = bngsim.Simulator(m, method="ssa").run(t_span=(0, 4), n_points=5, seed=1)
    a = m.get_concentration("A")
    for name, rule in RULES.items():
        assert m.get_concentration(name) == pytest.approx(rule(a), rel=1e-12)
        assert m.get_concentration(name) == r.species[-1, r.species_names.index(name)]


def test_a_kernel_and_what_reads_through_it():
    """ReactionKernel.get_state, advance()'s return value and CouplingMap.read,
    which a co-simulation orchestrator reads S through."""
    with _quiet():
        k = bngsim.ReactionKernel(_model(), method="ode")
        for _ in range(3):
            k.advance(1.0, rtol=1e-10, atol=1e-12)
        last = k.advance(1.0, rtol=1e-10, atol=1e-12)
    i = list(k.state_names).index("S")
    assert last[i] == pytest.approx(2 * A4, rel=1e-8)
    assert k.get_state()[i] == last[i]
    assert CouplingMap(k.state_names, ["S"]).read(k)[0] == last[i]


def test_a_snapshot_taken_after_run_until():
    m = _model()
    with _quiet():
        sim = bngsim.Simulator(m, method="ode")
        sim.run_until(4.0, rtol=1e-10, atol=1e-12)
    assert sim.snapshot()["species"]["S"] == pytest.approx(2 * A4, rel=1e-8)


def test_an_amount_valued_target_in_a_sized_compartment():
    """hOSU=true in c = 2: the rule gives the target's amount, and the getter
    reads what Result.species reports for it, the same number."""
    m = _model(
        "model m\n compartment c = 2\n species A in c = 10\n"
        " substanceOnly species T in c\n A => ; k*A\n k = 0.5\n T := 3*A\nend\n"
    )
    with _quiet():
        r = bngsim.Simulator(m, method="ode").run(
            t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12
        )
    assert m.get_concentration("T") == r.species[-1, r.species_names.index("T")]


def test_the_net_written_from_the_sbml(tmp_path):
    """sbml_to_net output reloaded with Model.from_net rebuilds the report map
    from the .net (#515), and reaches the same getters."""
    antimony = pytest.importorskip("antimony")
    from bngsim.convert import sbml_to_net

    antimony.clearPreviousLoads()
    assert (
        antimony.loadAntimonyString(
            "model m\n compartment c = 1\n species A in c = 10\n species S in c\n"
            " A => ; k*A\n k = 0.5\n S := 2*A\nend\n"
        )
        >= 0
    )
    xml = tmp_path / "m.xml"
    xml.write_text(antimony.getSBMLString("m"))
    with _quiet():
        sbml_to_net(str(xml), str(tmp_path / "m.net"), validate=None)
        m = bngsim.Model.from_net(str(tmp_path / "m.net"))
        bngsim.Simulator(m, method="ode").run(t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12)
    assert m.get_concentration("S") == pytest.approx(2 * A4, rel=1e-8)


def test_the_state_round_trips_and_a_continuation_reads_on():
    """set_state(get_state()) gives the state back, and a run continuing from
    it reports the rule, and leaves it, at the new end."""
    m = _model()
    with _quiet():
        sim = bngsim.Simulator(m, method="ode")
        sim.run(t_span=(0, 4), n_points=5, rtol=1e-10, atol=1e-12)
        before = m.get_state()
        m.set_state(m.get_state())
        assert np.array_equal(m.get_state(), before)
        r = sim.run(t_span=(4, 8), n_points=5, rtol=1e-10, atol=1e-12)
    a8 = 10 * math.exp(-4.0)
    assert r.species[-1, r.species_names.index("A")] == pytest.approx(a8, rel=1e-7)
    assert m.get_concentration("S") == pytest.approx(2 * a8, rel=1e-7)


def test_a_pre_equilibration_carries_its_sensitivities_through_the_write():
    """The write goes through the core's set_concentration, which drops a
    pending carry-over seed, so the run puts the seed back around it. Without
    that, a carry_sensitivities run after a sensitivity run starts its columns
    from nothing. Two legs, 0 -> 3 and carried 3 -> 6, give the single run's
    dA/dk at t = 6."""
    ant = (
        "model m\n compartment c = 1\n species A in c = 10\n species S in c\n"
        " J0: -> A; ka\n J1: A -> ; k*A\n k = 0.5; ka = 2\n S := 2*A\nend\n"
    )
    with _quiet():
        one = bngsim.Simulator(_model(ant), method="ode", sensitivity_params=["k"])
        whole = one.run(t_span=(0, 6), n_points=7, rtol=1e-10, atol=1e-12)
        sim = bngsim.Simulator(_model(ant), method="ode", sensitivity_params=["k"])
        sim.run(t_span=(0, 3), n_points=4, rtol=1e-10, atol=1e-12)
        leg = sim.run(t_span=(3, 6), n_points=4, rtol=1e-10, atol=1e-12, carry_sensitivities=True)
    ia = whole.species_names.index("A")
    want = np.asarray(whole.sensitivities)[-1, ia, 0]
    assert np.asarray(leg.sensitivities)[-1, leg.species_names.index("A"), 0] == pytest.approx(
        want, rel=1e-6
    )
    assert want != 0.0


def test_a_run_on_a_model_without_rules_is_left_as_it_was():
    """Control: nothing to write, nothing written."""
    m = _model("species X = 3; R1: X -> ; 0.2*X\n")
    with _quiet():
        bngsim.Simulator(m, method="ode").run(t_span=(0, 2), n_points=3, rtol=1e-10, atol=1e-12)
    assert m.get_concentration("X") == pytest.approx(3 * math.exp(-0.4), rel=1e-8)
