"""Issue #904, part 2: a window whose residual turns within a step.

``if(abs(Aobs-5) < 0.01, k, 0)`` and ``if((Aobs-5)^2 < 0.0001, k, 0)`` are the
window ``4.99 < A < 5.01``. Each is one comparison, and its residual
(``abs(A-5) - 0.01``, ``(A-5)^2 - 1e-4``) falls to zero and comes back up as A
passes 5: within one step it has the same sign at both ends, so a root on it
reports nothing, and the window was stepped over. Y came back 0 for 0.1 at
every tolerance, and with sensitivities ``(A-5)^2`` gave Y = 0 and dY/dk = 0.

Such an atom is now rooted at its edges, ``A - 5`` against ``0.01`` and
``-0.01``, each monotone along a flow that moves A one way, and on the
sensitivity path jumped there as any crossing is.

A = 2t, so A is in the window for 0.01 of time and Y = k * 0.01. With the rate
``k*A`` and the window centred on a parameter c, Y = k * 0.01 * c.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest
from bngsim._switch_sensitivity import _monotone_pieces

NET = """begin parameters
    1 k 10
    2 two 2
    3 c 5
end parameters
begin functions
    1 f() {cond}
end functions
begin species
    1 A() 0
    2 Y() 0
end species
begin reactions
    1 0 1 two
    2 0 2 f
end reactions
begin groups
    1 Aobs 1
end groups
"""

WINDOWS = [
    "if((Aobs>4.99)&&(Aobs<5.01),k,0)",
    "if(abs(Aobs-5)<0.01,k,0)",
    "if((Aobs-5)^2<0.0001,k,0)",
]
IDS = ["two-comparisons", "abs", "square"]


def _model(tmp_path, cond: str) -> bngsim.Model:
    path = tmp_path / "window.net"
    path.write_text(NET.format(cond=cond))
    return bngsim.Model.from_net(str(path))


@pytest.mark.parametrize("rtol", [1e-6, 1e-10])
@pytest.mark.parametrize("cond", WINDOWS, ids=IDS)
def test_a_window_one_comparison_spells_is_not_stepped_over(tmp_path, cond, rtol):
    run = bngsim.Simulator(_model(tmp_path, cond), method="ode").run(
        t_span=(0.0, 10.0), n_points=2, rtol=rtol, atol=1e-12
    )
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(0.1, rel=1e-5)


@pytest.mark.parametrize("cond", WINDOWS[1:], ids=IDS[1:])
def test_its_edges_are_the_roots(tmp_path, cond):
    roots = _model(tmp_path, cond).state_switch_root_conditions()
    assert len(roots) == 2
    assert all("Aobs-5" in r and "abs" not in r and "^" not in r for r in roots)


@pytest.mark.parametrize(
    "cond",
    ["if((Aobs>c-0.01)&&(Aobs<c+0.01),k*Aobs,0)", "if((Aobs-c)^2<0.0001,k*Aobs,0)"],
    ids=["two-comparisons", "square"],
)
def test_the_sensitivity_jumps_at_each_edge(tmp_path, cond):
    """Y = k * 0.01 * c: dY/dc = 0.1 moves both edges, dY/dk = 0.05."""
    sim = bngsim.Simulator(_model(tmp_path, cond), method="ode", sensitivity_params=["c", "k"])
    run = sim.run(t_span=(0.0, 10.0), n_points=2, rtol=1e-9, atol=1e-12)
    assert float(np.asarray(run.species)[-1, 1]) == pytest.approx(0.5, rel=1e-6)
    d_c, d_k = np.asarray(run.sensitivities)[-1, 1, :]
    assert d_c == pytest.approx(0.1, rel=1e-5)
    assert d_k == pytest.approx(0.05, rel=1e-6)


@pytest.mark.parametrize(
    ("atom", "pieces"),
    [
        ("abs(Aobs-5)<0.01", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-(0.01))"]),
        ("0.01>abs(Aobs-5)", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-(0.01))"]),
        ("abs(Aobs-c)<=w", ["(Aobs-c)<(w)", "(Aobs-c)>(-(w))"]),
        ("(Aobs-5)^2<0.0001", ["(Aobs-5)<(0.01)", "(Aobs-5)>(-0.01)"]),
        ("Aobs^2>4", ["(Aobs)<(2.0)", "(Aobs)>(-2.0)"]),
    ],
)
def test_the_pieces_of_a_turning_comparison(atom, pieces):
    assert _monotone_pieces(atom) == pieces


@pytest.mark.parametrize(
    "atom",
    [
        "Aobs<5",  # monotone already
        "abs(A)*abs(B)<1",  # not one abs()
        "abs(A)<abs(B)",  # turns on both sides
        "(A-5)^2<w",  # a square against a bound that is not a number
        "(A-5)^2<-1",  # against a negative one: no edge at all
        "A^2.5<1",  # not a square
    ],
)
def test_other_comparisons_are_left_as_they_are(atom):
    assert _monotone_pieces(atom) == []
