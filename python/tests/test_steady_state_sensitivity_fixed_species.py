"""The steady-state sensitivity of a model with a ``$``-fixed species is
solved, not refused as singular (issue #961).

A fixed species has no equation: its row of J is 0. The columns are solved on
the unknowns, and a fixed species was kept among them in two cases: a model with
no conservation law and no mask, which took the full solve, and one with a law,
whose independent species it was among. Either way J was singular there, and
``steady_state(sensitivity_params=...)`` raised "dY_ss/dp does not exist" for
columns that exist. The fixed species is taken out of the unknowns now; its own
row is what a parameter sets it to.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

# X is made at k1 per S and lost at k2: X* = k1·S/k2.
NO_LAW = """begin parameters
    1 k1 2.0
    2 k2 0.5
    3 S0 3.0
end parameters
begin species
    1 $S() S0
    2 X() 0
end species
begin reactions
    1 1 1,2 k1
    2 2 0 k2
end reactions
"""

# A <-> B at kf, kr, and S turns A into B at k1: with a = kf + k1·S and
# A + B = 1, B* = a/(a + kr).
BESIDE_A_LAW = """begin parameters
    1 k1 2.0
    2 kf 1.0
    3 kr 0.25
    4 S0 3.0
end parameters
begin species
    1 $S() S0
    2 A() 1
    3 B() 0
end species
begin reactions
    1 2 3 kf
    2 3 2 kr
    3 1,2 1,3 k1
end reactions
"""


# Nothing is solved for once the fixed species is out: S alone, and S beside an X
# that S + X -> S + X leaves where it is, a law of its own.
ONLY_FIXED = """begin parameters
    1 S0 3.0
    2 k 1.0
end parameters
begin species
    1 $S() S0
end species
begin reactions
    1 1 1 k
end reactions
"""

NO_UNKNOWN_LEFT = """begin parameters
    1 S0 3.0
    2 X0 2.0
    3 k 1.0
end parameters
begin species
    1 $S() S0
    2 X() X0
end species
begin reactions
    1 1,2 1,2 k
end reactions
"""


def _model(tmp_path, text):
    path = tmp_path / "m.net"
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _columns(tmp_path, text, params, set_to=None):
    model = _model(tmp_path, text)
    for name, value in (set_to or {}).items():
        model.set_concentration(name, value)
    out = bngsim.Simulator(model, method="ode").steady_state(sensitivity_params=params)
    assert out.converged
    return np.asarray(out.sensitivity)


def test_a_fixed_species_with_no_law_is_solved(tmp_path):
    """Refused as singular. dS/d[k1, k2, S0] = [0, 0, 1]; dX = [S/k2, -k1·S/k2², k1/k2]."""
    k1, k2, s = 2.0, 0.5, 3.0
    want = [[0.0, 0.0, 1.0], [s / k2, -k1 * s / k2**2, k1 / k2]]
    np.testing.assert_allclose(
        _columns(tmp_path, NO_LAW, ["k1", "k2", "S0"]), want, rtol=1e-6, atol=1e-12
    )


def test_a_fixed_species_beside_a_conservation_law_is_solved(tmp_path):
    """Refused as singular, the fixed species among the law's independents."""
    k1, kf, kr, s = 2.0, 1.0, 0.25, 3.0
    a = kf + k1 * s
    d = (a + kr) ** 2
    db = [s * kr / d, kr / d, -a / d, k1 * kr / d]  # d/d[k1, kf, kr, S0]
    want = [[0.0, 0.0, 0.0, 1.0], [-x for x in db], db]
    np.testing.assert_allclose(
        _columns(tmp_path, BESIDE_A_LAW, ["k1", "kf", "kr", "S0"]), want, rtol=1e-6, atol=1e-12
    )


def test_a_model_with_no_fixed_species_is_solved_as_before(tmp_path):
    """Control. 0 -> X at k1, X -> 0 at k2: dX*/d[k1, k2] = [1/k2, -k1/k2²]."""
    text = (
        "begin parameters\n 1 k1 2.0\n 2 k2 0.5\nend parameters\n"
        "begin species\n 1 X() 0\nend species\n"
        "begin reactions\n 1 0 1 k1\n 2 1 0 k2\nend reactions\n"
    )
    np.testing.assert_allclose(_columns(tmp_path, text, ["k1", "k2"]), [[2.0, -8.0]], rtol=1e-6)


def test_a_fixed_species_set_by_hand_is_set_by_no_parameter(tmp_path):
    """Moved off its initial condition, S is where it was put whatever S0 is
    (issue #113), with no law and no mask too: at S = 5, dX = [S/k2, -k1·S/k2², 0]."""
    k1, k2, s = 2.0, 0.5, 5.0
    want = [[0.0, 0.0, 0.0], [s / k2, -k1 * s / k2**2, 0.0]]
    got = _columns(tmp_path, NO_LAW, ["k1", "k2", "S0"], set_to={"S()": s})
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(
    ("text", "params", "want"),
    [
        (ONLY_FIXED, ["S0", "k"], [[1.0, 0.0]]),
        (NO_UNKNOWN_LEFT, ["S0", "X0", "k"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ],
    ids=["only_fixed", "the_rest_in_a_law_of_its_own"],
)
def test_a_model_with_no_unknown_left_is_solved(tmp_path, text, params, want):
    """Each row is a seed or a total, with nothing to solve for. Every column
    was NaN and refused as singular."""
    np.testing.assert_allclose(_columns(tmp_path, text, params), want, atol=1e-12)


@pytest.mark.parametrize("method", ["integration", "newton"])
@pytest.mark.parametrize(
    ("text", "params", "state"),
    [
        (NO_LAW, ["k1", "k2", "S0"], [3.0, 12.0]),
        (BESIDE_A_LAW, ["k1", "kf", "kr", "S0"], [3.0, 0.25 / 7.25, 7.0 / 7.25]),
    ],
    ids=["no_law", "beside_a_law"],
)
def test_the_state_is_the_one_found_without_the_columns(tmp_path, method, text, params, state):
    """Control. The columns are solved at the state, and leave it where it was
    found without them: X* = 12, and B* = a/(a + kr) beside the law."""

    def solve(**kw):
        sim = bngsim.Simulator(_model(tmp_path, text), method="ode")
        return np.asarray(sim.steady_state(method=method, **kw).concentrations)

    alone = solve()
    np.testing.assert_allclose(alone, state, rtol=1e-7)
    np.testing.assert_array_equal(solve(sensitivity_params=params), alone)
