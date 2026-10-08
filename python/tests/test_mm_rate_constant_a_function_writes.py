"""A Michaelis-Menten reaction whose ``kcat`` or ``Km`` is a parameter that a
function of the same name writes (issue #931).

The engine evaluates the function into the parameter's slot before each
right-hand side. The compiled right-hand side read the slot as the model was
loaded: with ``kcat() = kb`` over a slot at 0 the reaction never ran, and
P(6) came back 0 for 18 under ``codegen=True`` and in every forward-sensitivity
run, which always uses the compiled right-hand side, with dP/dkb = 0 for 6.
Nothing was logged.

Codegen declines such a model now. A plain run falls back to the interpreted
engine, which is right; ``codegen=True`` and a sensitivity request say why they
cannot be honoured.

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
end species
begin reactions
    1 1,2 1,3 {law}
end reactions
"""


def _model(tmp_path, function="kcat() kb", law="MM kcat Km", kcat="0", km="1"):
    path = tmp_path / "mm.net"
    path.write_text(NET.format(function=function, law=law, kcat=kcat, km=km))
    return bngsim.Model.from_net(str(path))


def _p_end(sim):
    run = sim.run(t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12)
    return float(np.asarray(run.species)[-1, list(run.species_names).index("P()")])


CASES = {
    "kcat-at-0": dict(),
    "kcat-at-7": dict(kcat="7"),
    "kcat-twice-kb": dict(function="kcat() kb*2"),
    "km": dict(function="Km() kb/3", law="MM kb Km", km="5e6"),
}
P_END = {"kcat-at-0": 18.0, "kcat-at-7": 18.0, "kcat-twice-kb": 36.0, "km": 18.0}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_compiled_right_hand_side_is_declined(tmp_path, case):
    """``codegen=True`` returned P(6) = 0 with the slot at 0 and 42 with it at
    7, for 18; and 3 for 18 with ``Km() = kb/3`` over a slot at 5e6."""
    with pytest.raises(RuntimeError, match=r"codegen declined.*issue #931"):
        bngsim.Simulator(_model(tmp_path, **CASES[case]), method="ode", codegen=True)


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_plain_run_reads_the_function(tmp_path, case):
    """Control. The interpreted engine, which a plain run of a model this
    size uses and which a declined model falls back to."""
    model = _model(tmp_path, **CASES[case])
    assert _p_end(bngsim.Simulator(model, method="ode")) == pytest.approx(P_END[case], rel=1e-5)
    model.reset()
    assert _p_end(bngsim.Simulator(model, method="ode", codegen=False)) == pytest.approx(
        P_END[case], rel=1e-5
    )


def test_a_forward_sensitivity_run_is_refused(tmp_path):
    """It integrated the compiled right-hand side: P(6) = 0 and dP/dkb = 0,
    for 18 and 6."""
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"issue #931"):
        bngsim.Simulator(_model(tmp_path), method="ode", sensitivity_params=["kb"])


def test_a_steady_state_sensitivity_is_refused(tmp_path):
    sim = bngsim.Simulator(_model(tmp_path), method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match=r"issue #931"):
        sim.steady_state(sensitivity_params=["kb"])


def test_the_source_for_the_jit_is_declined_too(tmp_path):
    from bngsim import _codegen

    model = _model(tmp_path)
    assert _codegen.prepare_model_codegen_source(model) is None
    assert "issue #931" in (_codegen.last_codegen_decline() or "")
    assert _codegen.prepare_model_codegen(model) is None
    assert "issue #931" in (_codegen.last_codegen_decline() or "")


@pytest.mark.parametrize("case", ["kcat-at-0", "km"])
def test_the_emitter_declines_for_a_caller_that_goes_round_the_entry_points(tmp_path, case):
    from bngsim import _codegen

    with pytest.raises(_codegen.CodegenDeclined, match=r"issue #931"):
        _codegen.generate_rhs_from_model(_model(tmp_path, **CASES[case]))


def test_a_constant_that_no_function_writes_is_compiled(tmp_path):
    """Control. ``MM kb Km`` over plain parameters, beside a function that is
    in no rate."""
    model = _model(tmp_path, function="unused() kb*2", law="MM kb Km")
    assert _p_end(bngsim.Simulator(model, method="ode", codegen=True)) == pytest.approx(
        18.0, rel=1e-5
    )
    model = _model(tmp_path, function="unused() kb*2", law="MM kb Km")
    sens = bngsim.Simulator(model, method="ode", sensitivity_params=["kb"])
    run = sens.run(t_span=(0.0, 6.0), n_points=3, rtol=1e-10, atol=1e-12)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("P()"), 0]
    assert got == pytest.approx(6.0, rel=1e-5)


@pytest.mark.parametrize("law", ["Sat kcat Km", "kcat"], ids=["sat", "elementary"])
def test_the_other_rate_laws_read_the_function_when_compiled(tmp_path, law):
    """Control. The loader makes these two functional rate laws, which the
    compiled right-hand side reads through the function: the same compiled
    and interpreted."""
    compiled = _p_end(bngsim.Simulator(_model(tmp_path, law=law), method="ode", codegen=True))
    plain = _p_end(bngsim.Simulator(_model(tmp_path, law=law), method="ode", codegen=False))
    assert compiled == pytest.approx(plain, rel=1e-8)
    assert compiled > 0.0
