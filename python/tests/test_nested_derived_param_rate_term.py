"""A derived parameter that reaches a rate through another derived one (issue #912).

A requested derived parameter is a sensitivity coordinate of its own: it is held
where it is set (``set_param(..., force_override=True)``) and everything defined
from it follows. With ``Q = 3*R0`` and ``Rt = 2*Q`` and a rate that reads ``Rt``,
the ``Q`` column therefore carries ``∂f/∂Rt · ∂Rt/∂Q``.

The compiled sensitivity right-hand side chained a derived rate constant to its
*primaries* only. ``Rt``'s own column had the direct term and ``R0``'s the full
chain, but ``Q``, in between, had neither: its column came back 0, or held only
its initial-condition part (issue #715), with no warning. BNG2.pl makes this
shape out of every compound rate law, ``_rateLaw1 = kp*Rt``, so the column of a
derived parameter named in one was missing.

Each expected column is a closed form. Where the model is not linear in time
the same column is also compared with a central difference of plain runs with
the parameter overridden.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

T = np.array([0.0, 1.0, 2.0])


def _run(tmp_path, text, params, name="m.net"):
    net = tmp_path / name
    net.write_text(text)
    model = bngsim.Model.from_net(net)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=list(params)).run(
        sample_times=list(T), rtol=1e-11, atol=1e-13
    )
    return run


def _columns(tmp_path, text, params, species=0):
    return np.asarray(_run(tmp_path, text, params).sensitivities)[:, species, :]


def _fd(tmp_path, text, param, species=0, rel=1e-3):
    """Central difference of plain runs with `param` overridden, at two steps
    and extrapolated. A step of 1e-6 left the solver's own error, 1e-12 of the
    state over the step, at the size of the tolerance it was compared within."""

    def at(value):
        net = tmp_path / "fd.net"
        net.write_text(text)
        model = bngsim.Model.from_net(net)
        model.set_param(param, value, force_override=True)
        run = bngsim.Simulator(model, method="ode").run(
            sample_times=list(T), rtol=1e-12, atol=1e-14
        )
        return np.asarray(run.species)[:, species]

    net = tmp_path / "fd0.net"
    net.write_text(text)
    v = bngsim.Model.from_net(net).get_param(param)

    def central(h):
        return (at(v + h) - at(v - h)) / (2 * h)

    h = rel * abs(v)
    return (4.0 * central(h / 2) - central(h)) / 3.0


# Rt = 1 sets B(0) and is the production rate; B decays at kd = 0.5. Per unit of
# Rt, B = e^(−t/2) + 2·(1 − e^(−t/2)) = 2 − e^(−t/2).
PRODUCED = """\
begin parameters
    1 R0  0.16666666666666666
    2 kd  0.5
    3 Q   3*R0
    4 Rt  2*Q
end parameters
begin species
    1 B() Rt
end species
begin reactions
    1 0 1 Rt
    2 1 0 kd
end reactions
"""
G = 2.0 - np.exp(-0.5 * T)


def test_the_parameter_in_the_middle_of_the_chain_gets_its_rate_term(tmp_path):
    """The issue's second case. The Q column was 2·e^(−t/2), its
    initial-condition part alone."""
    s = _columns(tmp_path, PRODUCED, ["Q", "Rt", "R0"])
    np.testing.assert_allclose(s[:, 0], 2.0 * G, rtol=1e-8)
    np.testing.assert_allclose(s[:, 1], G, rtol=1e-8)
    np.testing.assert_allclose(s[:, 2], 6.0 * G, rtol=1e-8)
    np.testing.assert_allclose(s[:, 0], _fd(tmp_path, PRODUCED, "Q"), rtol=1e-6, atol=1e-8)


def test_the_middle_column_alone(tmp_path):
    """Requested by itself: no other column to lean on."""
    s = _columns(tmp_path, PRODUCED, ["Q"])
    np.testing.assert_allclose(s[:, 0], 2.0 * G, rtol=1e-8)


# What BNG2.pl writes for `0 -> C() kp*Rt`.
COMPOUND = """\
begin parameters
    1 kp         0.1
    2 R0         2
    3 Rt         3*R0
    4 _rateLaw1  kp*Rt
end parameters
begin species
    1 C() 0
end species
begin reactions
    1 0 1 _rateLaw1
end reactions
"""


def test_a_parameter_inside_a_compound_rate_law(tmp_path):
    """The issue's first case: C = kp·Rt·t, and dC/dRt came back 0."""
    s = _columns(tmp_path, COMPOUND, ["Rt", "R0", "kp", "_rateLaw1"])
    np.testing.assert_allclose(s[:, 0], 0.1 * T, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(s[:, 1], 0.3 * T, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(s[:, 2], 6.0 * T, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(s[:, 3], T, rtol=1e-9, atol=1e-12)


DEEP = """\
begin parameters
    1 R0  1
    2 U   5*R0
    3 V   3*U
    4 W   2*V
end parameters
begin species
    1 C() 0
end species
begin reactions
    1 0 1 W
end reactions
"""


def test_every_parameter_on_a_longer_chain(tmp_path):
    """C = W·t with W = 2·V, V = 3·U, U = 5·R0."""
    s = _columns(tmp_path, DEEP, ["U", "V", "W", "R0"])
    for col, want in enumerate((6.0, 2.0, 1.0, 30.0)):
        np.testing.assert_allclose(s[:, col], want * T, rtol=1e-9, atol=1e-12)


DIAMOND = """\
begin parameters
    1 R0  1
    2 Q   2*R0
    3 P2  3*R0
    4 Rt  Q*P2
end parameters
begin species
    1 C() 0
end species
begin reactions
    1 0 1 Rt
end reactions
"""


def test_two_derived_parameters_that_meet_in_one_rate_constant(tmp_path):
    """C = Q·P2·t: dC/dQ = P2·t = 3t, dC/dP2 = Q·t = 2t, dC/dR0 = 12t."""
    s = _columns(tmp_path, DIAMOND, ["Q", "P2", "R0"])
    for col, want in enumerate((3.0, 2.0, 12.0)):
        np.testing.assert_allclose(s[:, col], want * T, rtol=1e-9, atol=1e-12)


FUNCTIONAL = """\
begin parameters
    1 R0  0.5
    2 Q   3*R0
    3 Rt  2*Q
end parameters
begin functions
    1 fB() Rt*Aobs
end functions
begin species
    1 A() 2
    2 B() 0
end species
begin reactions
    1 0 2 fB
end reactions
begin groups
    1 Aobs 1
end groups
"""


def test_a_functional_rate_law_that_reads_the_end_of_the_chain(tmp_path):
    """B = Rt·A·t with A = 2: dB/dQ = 4t, dB/dRt = 2t, dB/dR0 = 12t."""
    s = _columns(tmp_path, FUNCTIONAL, ["Q", "Rt", "R0"], species=1)
    for col, want in enumerate((4.0, 2.0, 12.0)):
        np.testing.assert_allclose(s[:, col], want * T, rtol=1e-9, atol=1e-12)


MICHAELIS = """\
begin parameters
    1 R0    0.5
    2 Q     3*R0
    3 kcat  2*Q
    4 Km    1
end parameters
begin species
    1 E() 1
    2 S() 5
    3 P() 0
end species
begin reactions
    1 1,2 1,3 MM kcat Km
end reactions
"""


def test_a_michaelis_menten_kcat_at_the_end_of_the_chain(tmp_path):
    """No closed form worth writing: each column against a central difference
    of plain runs with that parameter overridden. dP/dkcat is 2·dP/dQ."""
    s = _columns(tmp_path, MICHAELIS, ["Q", "kcat", "R0"], species=2)
    for col, name in enumerate(("Q", "kcat", "R0")):
        want = _fd(tmp_path, MICHAELIS, name, species=2)
        np.testing.assert_allclose(s[:, col], want, rtol=1e-6, atol=1e-9, err_msg=name)
    np.testing.assert_allclose(s[:, 0], 2.0 * s[:, 1], rtol=1e-9, atol=1e-12)
    assert abs(s[-1, 0]) > 0.1


OUTPUT = """\
begin parameters
    1 R0  0.5
    2 Q   3*R0
    3 Rt  2*Q
    4 kd  0.5
end parameters
begin functions
    1 scaled() Rt*Bobs
end functions
begin species
    1 B() 4
end species
begin reactions
    1 1 0 kd
end reactions
begin groups
    1 Bobs 1
end groups
"""


def test_an_output_expression_that_reads_the_end_of_the_chain(tmp_path):
    """scaled = Rt·B with B = 4·e^(−t/2): d(scaled)/dQ = 2·B, /dRt = B,
    /dR0 = 6·B."""
    run = _run(tmp_path, OUTPUT, ["Q", "Rt", "R0"])
    out = np.asarray(run.output_sensitivities("expression:scaled"))[:, 0, :]
    b = 4.0 * np.exp(-0.5 * T)
    for col, want in enumerate((2.0, 1.0, 6.0)):
        np.testing.assert_allclose(out[:, col], want * b, rtol=1e-8)


@pytest.mark.parametrize("params", [["R0"], ["Rt"], ["R0", "kd"]])
def test_columns_that_were_right_stay_as_they_were(tmp_path, params):
    """The primary and the rate constant itself, with no parameter between."""
    s = _columns(tmp_path, PRODUCED, params)
    want = {"R0": 6.0 * G, "Rt": G}
    for col, name in enumerate(params):
        if name in want:
            np.testing.assert_allclose(s[:, col], want[name], rtol=1e-8)
        else:
            np.testing.assert_allclose(
                s[:, col], _fd(tmp_path, PRODUCED, name), rtol=1e-6, atol=1e-8
            )
