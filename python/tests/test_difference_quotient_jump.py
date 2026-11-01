"""lanl/bngsim#1002: a difference quotient that steps across a jump in a rate law.

Where a run or a steady-state solve has no closed-form Jacobian (``jacobian="fd"``,
or one law the closed form declines, which takes it away from every law), the
solver differences the right-hand side. From a state within one step of a
threshold of ``if``, a comparison or ``ceil``, the step crosses the threshold
and the quotient reads the jump over the step, Δf/h, where the row has a slope:
-1.3e6 for -0.02 beside ``if(X > 5, 0.04, 0.02)`` at X = 5. On main that did
two things, with no error or warning:

* CVODE's Newton iteration under the matrix took corrections a millionth of
  their size, called them converged, and kept the matrix: the run followed its
  predictor, X = 5 - 0.1 t, 166% off at t = 60 (negative by then).
* A species held at 0 under ``if(I > 0, ...)`` is stepped by CVODE's floor,
  far under 1e-12. Through that entry the LU's rounding put 1e-36 into I, which
  turned the condition true: 490% off, and a steady state 5x off with
  ``converged=True``.

bngsim now forms the dense quotient CVODE formed (bngsim/dq_jacobian.hpp), the
same matrix bit for bit, for runs and the steady-state march, and mends a row
whose step crosses a jump (bngsim/fd_jacobian.hpp); a run's colored sparse
quotient does the same. KINSOL's quotient is left as it was. Every closed form
here is piecewise exponential.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

import bngsim
import numpy as np
import pytest

RTOL = 1e-6
ATOL = 1e-9
# The runs here are within a few 1e-6 of their closed forms once nothing steps
# across a jump; main was off by 2e-2 to 5.
OK = 1e-4

_NET = """begin parameters
    1 kx 1
end parameters
begin species
{species}
end species
begin groups
{groups}
end groups
begin functions
{functions}
end functions
begin reactions
{reactions}
end reactions
"""


def _net(
    tmp_path: Path,
    species: list[str],
    groups: list[str],
    functions: list[str],
    reactions: list[str],
    *,
    declined: bool = False,
) -> bngsim.Model:
    """A .net model; ``declined`` adds a reaction on two species of its own whose
    law (``rint``) the closed form declines, so the whole model is differenced."""
    species, groups, functions, reactions = (
        list(species),
        list(groups),
        list(functions),
        list(reactions),
    )
    if declined:
        n = len(species)
        species += ["W() 1", "Z() 0"]
        groups += [f"Wobs {n + 1}"]
        functions += ["gdecl() 1e-3*rint(Wobs*10)"]
        reactions += [f"{n + 1} {n + 2} gdecl"]
    body = _NET.format(
        species="\n".join(f"    {i + 1} {s}" for i, s in enumerate(species)),
        groups="\n".join(f"    {i + 1} {g}" for i, g in enumerate(groups)),
        functions="\n".join(f"    {i + 1} {f}" for i, f in enumerate(functions)),
        reactions="\n".join(f"    {i + 1} {r}" for i, r in enumerate(reactions)),
    )
    path = tmp_path / f"m{len(list(tmp_path.iterdir()))}.net"
    path.write_text(body)
    model = bngsim.Model.from_net(str(path))
    return model


def _run(model: bngsim.Model, t_end: float, n: int, **kw) -> tuple[np.ndarray, np.ndarray]:
    result = bngsim.Simulator(model, method="ode", **kw).run(
        t_span=(0.0, t_end), n_points=n, rtol=RTOL, atol=ATOL
    )
    if kw.get("jacobian") != "fd":
        assert model.analytical_jacobian_decline is not None, "the closed form was not declined"
    return np.asarray(result.time), np.asarray(result.species)


def _worst(x: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(np.abs(x / exact - 1.0)))


# ── X decays at kx·k(X)·X, k = 0.04 above 5 and 0.02 at or below ────────────


def _threshold(tmp_path: Path, x0: float, cond: str = "Xobs > 5", **kw) -> bngsim.Model:
    return _net(
        tmp_path,
        [f"X() {x0!r}", "Y() 0"],
        ["Xobs 1"],
        [f"f() kx*if({cond}, 0.04, 0.02)"],
        ["1 2 f"],
        **kw,
    )


def _threshold_exact(t: np.ndarray, x0: float) -> np.ndarray:
    if x0 <= 5.0:
        return x0 * np.exp(-0.02 * t)
    t5 = math.log(x0 / 5.0) / 0.04
    return np.where(t <= t5, x0 * np.exp(-0.04 * t), 5.0 * np.exp(-0.02 * (t - t5)))


@pytest.mark.parametrize(
    ("declined", "kw"),
    [
        pytest.param(False, {"jacobian": "fd"}, id="fd"),
        pytest.param(True, {}, id="declined-law"),
        pytest.param(
            False, {"jacobian": "fd", "force_sparse_linear_solver": True}, id="fd-sparse"
        ),
    ],
)
def test_a_run_that_starts_on_a_threshold(tmp_path: Path, declined: bool, kw: dict) -> None:
    """X0 = 5 on `X > 5`, falling: the step up crosses the threshold. Main: 1.66
    with "fd" (X = 5 - 0.1 t), 2.1e-2 with a declined law, 1.66 on the colored
    sparse quotient."""
    t, s = _run(_threshold(tmp_path, 5.0, declined=declined), 60.0, 61, **kw)
    assert _worst(s[:, 0], _threshold_exact(t, 5.0)) < OK


@pytest.mark.parametrize(("declined", "kw"), [(False, {"jacobian": "fd"}), (True, {})])
def test_a_run_that_leaves_the_threshold_it_starts_on(
    tmp_path: Path, declined: bool, kw: dict
) -> None:
    """X0 = 5 on `X >= 5`: at the start the step crosses nothing, but X falls
    below 5 at once and the matrix formed a hair under it does. Main: 1.7 with
    "fd", 7.8e-2 with a declined law."""
    t, s = _run(_threshold(tmp_path, 5.0, cond="Xobs >= 5", declined=declined), 60.0, 61, **kw)
    assert _worst(s[:, 0], 5.0 * np.exp(-0.02 * t)) < OK


_FD_PATHS = pytest.mark.parametrize(
    "kw",
    [{"jacobian": "fd"}, {"jacobian": "fd", "force_sparse_linear_solver": True}],
    ids=["fd", "fd-sparse"],
)


@_FD_PATHS
def test_sensitivities_of_a_run_that_starts_on_a_threshold(tmp_path: Path, kw: dict) -> None:
    """X = 5·exp(-0.02·kx·t), so dX/dkx = -0.02·t·X, at most 1.84 in size. Main:
    the state 1.66 off and dX/dkx off by 4.19."""
    result = bngsim.Simulator(
        _threshold(tmp_path, 5.0), method="ode", sensitivity_params=["kx"], **kw
    ).run(t_span=(0.0, 60.0), n_points=61, rtol=RTOL, atol=ATOL)
    t = np.asarray(result.time)
    exact = 5.0 * np.exp(-0.02 * t)
    assert _worst(np.asarray(result.species)[:, 0], exact) < OK
    dxdk = np.asarray(result.sensitivities)[:, 0, 0]
    assert float(np.max(np.abs(dxdk + 0.02 * t * exact))) < 1e-3


@_FD_PATHS
def test_a_law_steep_enough_to_step_like_a_jump(tmp_path: Path, kw: dict) -> None:
    """kx·(0.03 + 0.01·tanh(1e10·(X − 5)))·X is smooth, but across the quotient's
    step of 7.5e-8 it is a jump from 0.02 to 0.04, and X0 = 5 sits on it: X =
    5·exp(-0.02 t) after a transient of 1e-10. Main: 1.66 off. A test of a rate
    law's operators would not have looked at this one."""
    model = _net(
        tmp_path,
        ["X() 5", "Y() 0"],
        ["Xobs 1"],
        ["f() kx*(0.03 + 0.01*tanh(1e10*(Xobs - 5)))"],
        ["1 2 f"],
    )
    t, s = _run(model, 60.0, 61, **kw)
    assert _worst(s[:, 0], 5.0 * np.exp(-0.02 * t)) < OK


def test_a_run_that_starts_on_a_threshold_that_recedes(tmp_path: Path) -> None:
    """X0 = Y0 = 5 on `X > Y` (rate kx·X, kx 0.02 at or below Y, −1 above), Y
    rising at 0.5: X rises too but never reaches Y, so X = 5·exp(0.02 t). X's
    step crosses its own law's threshold at the start. Main: 1.5e-2 off (round
    1 of the review)."""
    model = _net(
        tmp_path,
        ["X() 5", "Y() 5", "S() 1"],
        ["Xobs 1", "Yobs 2", "Sobs 3"],
        ["k() kx*if(Xobs > Yobs, -1, 0.02)", "p() 0.5/Sobs"],
        ["1 1,1 k", "3 2,3 p"],
        declined=True,
    )
    t, s = _run(model, 40.0, 41, jacobian="fd")
    assert _worst(s[:, 0], 5.0 * np.exp(0.02 * t)) < OK


def test_a_crossing_during_the_run(tmp_path: Path) -> None:
    """Control. X0 = 10 crosses 5 at t = 17.3: no matrix is formed within a step
    of the threshold. Right on main too."""
    t, s = _run(_threshold(tmp_path, 10.0), 60.0, 61, jacobian="fd")
    assert _worst(s[:, 0], _threshold_exact(t, 10.0)) < OK


# ── The 2026-10-19 comment: a ceil() law over the state ─────────────────────


def _ceil_exact(t: float, x: float = 10.0, at: float = 0.0) -> float:
    # X falls at 0.01·c between the steps of c = ceil(0.4·X): 0.04 to X = 7.5, then 0.03, ...
    while True:
        c = math.ceil(0.4 * x - 1e-12)
        leaves = (c - 1) / 0.4
        jump = at + math.log(x / leaves) / (0.01 * c)
        if t <= jump:
            return x * math.exp(-0.01 * c * (t - at))
        x, at = leaves, jump


@pytest.mark.parametrize(("declined", "kw"), [(False, {"jacobian": "fd"}), (True, {})])
def test_a_run_that_starts_on_a_step_of_ceil(tmp_path: Path, declined: bool, kw: dict) -> None:
    """X0 = 10 is on a step of ceil(0.4·X). Main: 3.4e-2 with "fd", 4.9e-2 with a
    declined law."""
    model = _net(
        tmp_path,
        ["X() 10", "Y() 0"],
        ["Xobs 1"],
        ["f() kx*0.01*ceil(Xobs*0.4)"],
        ["1 2 f"],
        declined=declined,
    )
    t, s = _run(model, 30.0, 31, **kw)
    assert _worst(s[:, 0], np.array([_ceil_exact(ti) for ti in t])) < OK


# ── A species held at 0 under `> 0` ──────────────────────────────────────────


def _held_inhibitor(tmp_path: Path, **kw) -> bngsim.Model:
    """I = 0 for good (nothing makes it); Y is made at 1 and lost at 0.05 while
    I = 0 (0.01 were I > 0). Steady state Y = 20."""
    return _net(
        tmp_path,
        ["I() 0", "Y() 10", "S() 1"],
        ["Iobs 1"],
        ["f() kx*if(Iobs > 0, 0.01, 0.05)", "p() 1"],
        ["2 0 f", "3 2,3 p"],
        **kw,
    )


@pytest.mark.parametrize(
    ("declined", "kw"),
    [(False, {"jacobian": "fd"}), (True, {})],
)
def test_a_species_held_at_zero_stays_there(tmp_path: Path, declined: bool, kw: dict) -> None:
    """I = 0 for good; Y is lost at 0.05 while I = 0 (0.01 were I > 0), so Y =
    10·exp(-0.05 t). Main put 1e-35 into I through the LU, and Y fell at 0.01:
    4.9 off at t = 60 with "fd", 10 with a declined law. The leak's sign is the
    LU's rounding (on other output grids main leaked below 0 and Y came out
    right), so I is also held to exactly 0."""
    model = _net(
        tmp_path,
        ["I() 0", "Y() 10"],
        ["Iobs 1"],
        ["f() kx*if(Iobs > 0, 0.01, 0.05)"],
        ["2 0 f"],
        declined=declined,
    )
    t, s = _run(model, 60.0, 121, **kw)
    assert _worst(s[:, 1], 10.0 * np.exp(-0.05 * t)) < OK
    assert np.all(s[:, 0] == 0.0)


@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_steady_state_of_a_species_held_at_zero(tmp_path: Path, method: str) -> None:
    """Main: Y = 100 for 20 with converged=True, by either method."""
    result = bngsim.Simulator(_held_inhibitor(tmp_path), method="ode", jacobian="fd").steady_state(
        method=method
    )
    assert result.converged
    c = np.asarray(result.concentrations)
    assert c[0] == 0.0
    assert abs(c[1] / 20.0 - 1.0) < OK


def _gate(tmp_path: Path, **kw) -> bngsim.Model:
    """X = 0 for good; Y is made at 0.5 while X = 0 (1 were X > 0), from a source
    S, and lost at kx·Y. Y(t) = 0.5 + 0.5·exp(-t)."""
    return _net(
        tmp_path,
        ["X() 0", "Y() 1", "S() 1"],
        ["Xobs 1", "Sobs 3"],
        ["g() if(Xobs > 0, 1, 0.5)/Sobs"],
        ["3 2,3 g", "2 0 kx"],
        **kw,
    )


@pytest.mark.parametrize(("declined", "kw"), [(False, {"jacobian": "fd"}), (True, {})])
def test_a_gate_held_at_zero(tmp_path: Path, declined: bool, kw: dict) -> None:
    """Main: Y went to 1 for 0.5, 100% off, with "fd" and with a declined law."""
    t, s = _run(_gate(tmp_path, declined=declined), 60.0, 61, **kw)
    assert np.all(s[:, 0] == 0.0)
    assert _worst(s[:, 1], 0.5 + 0.5 * np.exp(-t)) < OK


def test_a_gate_held_at_zero_under_equality(tmp_path: Path) -> None:
    """`X == 0` at X = 0: the steps both ways cross, so neither one-sided quotient
    is the row's slope and the entry is 0. Main: Y went to 1 for 0.5."""
    model = _net(
        tmp_path,
        ["X() 0", "Y() 1", "S() 1"],
        ["Xobs 1", "Sobs 3"],
        ["g() if(Xobs == 0, 0.5, 1)/Sobs"],
        ["3 2,3 g", "2 0 kx"],
    )
    t, s = _run(model, 60.0, 61, jacobian="fd")
    assert np.all(s[:, 0] == 0.0)
    assert _worst(s[:, 1], 0.5 + 0.5 * np.exp(-t)) < OK


def test_a_stiff_run_that_starts_on_a_threshold(tmp_path: Path) -> None:
    """X0 = 5 on `X > 5` with rates 2000 above and 1000 at or below: X =
    5·exp(-1000 t). Main: 1.73 off at t = 1e-3 (35% of X0)."""
    model = _net(
        tmp_path, ["X() 5", "Y() 0"], ["Xobs 1"], ["f() kx*if(Xobs > 5, 2000, 1000)"], ["1 2 f"]
    )
    result = bngsim.Simulator(model, method="ode", jacobian="fd").run(
        t_span=(0.0, 0.01), n_points=11, rtol=RTOL, atol=1e-12
    )
    t, x = np.asarray(result.time), np.asarray(result.species)[:, 0]
    assert float(np.max(np.abs(x - 5.0 * np.exp(-1000.0 * t)))) < 5.0 * OK


@pytest.mark.parametrize(("declined", "kw"), [(False, {"jacobian": "fd"}), (True, {})])
@pytest.mark.parametrize("method", ["integration", "newton"])
def test_the_steady_state_of_a_gate_held_at_zero(
    tmp_path: Path, declined: bool, kw: dict, method: str
) -> None:
    """Main: Y = 1 for 0.5 with converged=True, by either method."""
    result = bngsim.Simulator(_gate(tmp_path, declined=declined), method="ode", **kw).steady_state(
        method=method
    )
    assert result.converged
    c = np.asarray(result.concentrations)
    assert c[0] == 0.0
    assert abs(c[1] / 0.5 - 1.0) < OK


def test_a_run_that_starts_on_a_threshold_between_basins(tmp_path: Path) -> None:
    """X' = a(X) - 0.1·X, a = 1.1 above 5, 0.1 on [0.5, 5] and 0 below 0.5: from
    X0 = 5 the run falls to the middle steady state, X = 1 + 4·exp(-0.1 t). Main,
    with a declined law, followed its predictor below 0.5 and ended in the lower
    basin: X(200) = -6.4e-8."""
    model = _net(
        tmp_path,
        ["X() 5", "S() 1"],
        ["Xobs 1", "Sobs 2"],
        ["a() if(Xobs > 5, 1.1, if(Xobs < 0.5, 0.0, 0.1))/Sobs", "d() 0.1"],
        ["2 1,2 a", "1 0 d"],
        declined=True,
    )
    t, s = _run(model, 200.0, 41)
    assert _worst(s[:, 0], 1.0 + 4.0 * np.exp(-0.1 * t)) < OK


def test_a_model_with_no_jump_has_nothing_mended(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Control: MODEL1812040006 has no jump in it, and nothing of it is mended.
    Two smooth shapes read as jumps to a looser test. A species at 1.6e-8 is
    stepped by CVODE's floor, 2.5e-2, and its points back are negative amounts,
    where the saturating law it feeds is nothing like itself: 52 mends with
    those points taken. And a row whose terms cancel moves by a last bit of
    them: 2 mends against a floor of the row's value instead of its terms.
    Either moved the run off main's bits."""
    monkeypatch.setenv("BNGSIM_JAC_DEBUG", "1")
    model = bngsim.Model.from_sbml(str(data_dir / "MODEL1812040006.xml"))
    bngsim.Simulator(model, method="ode", jacobian="fd").run(
        t_span=(0.0, 100.0), n_points=101, rtol=RTOL, atol=ATOL
    )
    assert "across a jump" not in capfd.readouterr().err


def _split(data_dir: Path, tmp_path: Path, k_v_stim: str) -> tuple[str, str]:
    """#903's `split` twin of the #176 fixture, written beside the test: the
    voltage carried as V + V2, two species that trade at kx·time() and kx, its
    stimulus set to `k_v_stim`. Returns the network's path and its oracle's,
    the same network with each step replaced by the branch the exact trajectory
    takes."""
    step = "if(((-70+Voltage_Level)<-20),0.5,0.05)"
    text = (data_dir / "ltype_calcium_discontinuous_jacobian.net").read_text()
    swaps = [
        (
            "   16 k_v_stim        49.99999999999  # Constant\n",
            f"   16 k_v_stim        {k_v_stim}  # Constant\n",
        ),
        (
            "   19 _rateLaw2       0.01  # Constant\n",
            "   19 _rateLaw2       0.01  # Constant\n   20 kx 0.01\n",
        ),
        (f"    4 v_rec() {step}\n", f"    4 v_rec() {step}\n    5 xf() kx*time()\n"),
        (
            "   10 LTCC(b,g~C,loc~mem,p~P,s~I) 0\n",
            "   10 LTCC(b,g~C,loc~mem,p~P,s~I) 0\n   11 Voltage2() 0\n",
        ),
        (
            "   25 10 8 _rateLaw2 #_R8\n",
            "   25 10 8 _rateLaw2 #_R8\n   26 3 11 xf\n   27 11 3 kx\n   28 11 0 k_v_leak\n",
        ),
        ("    8 Voltage_Level        3\n", "    8 Voltage_Level        3,11\n"),
    ]
    for old, new in swaps:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    exact = text.replace(step, "0.5").replace("if((Phospho_LTCC>0),k_pka_shift,0)", "k_pka_shift")
    (tmp_path / "split.net").write_text(text)
    (tmp_path / "split_exact.net").write_text(exact)
    return str(tmp_path / "split.net"), str(tmp_path / "split_exact.net")


def _split_deviation(net: str, exact: str, **window) -> float:
    got = bngsim.Simulator(
        bngsim.Model.from_net(net),
        method="ode",
        jacobian=window.pop("jacobian"),
        force_sparse_linear_solver=window.pop("sparse"),
    ).run(**window)
    want = bngsim.Simulator(bngsim.Model.from_net(exact), method="ode").run(
        **{**window, "rtol": 1e-12, "atol": 1e-12}
    )
    g, w = np.asarray(got.observables), np.asarray(want.observables)
    scale = np.maximum(np.abs(w).max(axis=0), 1e-12)
    return float((np.abs(g - w) / scale).max())


def _split_mends(
    net: str, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], **window
) -> str:
    """What the run prints under BNGSIM_JAC_DEBUG: a line for each matrix it mends."""
    monkeypatch.setenv("BNGSIM_JAC_DEBUG", "1")
    capfd.readouterr()
    sparse = window.pop("sparse")
    with contextlib.suppress(bngsim.SimulationError):
        bngsim.Simulator(
            bngsim.Model.from_net(net),
            method="ode",
            jacobian="fd",
            force_sparse_linear_solver=sparse,
        ).run(**window)
    return capfd.readouterr().err


# Where `split`'s outcome lands moves with the platform's rounding: #176 records
# its parked regime as chaotic. On x86-64 Linux these runs, with nothing mended
# and so main's own matrices, came out 0.62 and 0.633 off at some of these
# tolerances, where on macOS they are right. So what the controls below hold is
# the matrix (the species that trade are never mended, as on main), not one
# platform's digits.
@pytest.mark.parametrize(
    ("rtol", "atol", "n_points"), [(1e-8, 1e-8, 301), (1e-7, 1e-10, 301), (3e-7, 1e-8, 601)]
)
def test_a_parked_sum_keeps_the_slope_that_holds_it(
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    rtol: float,
    atol: float,
    n_points: int,
) -> None:
    """Control. #903's `split`: V + V2 parks 1e-11 under the #176 step at 50.
    At t = 18.6 V leaves at 0.38 and V2 arrives at 0.38; V's step crosses the
    threshold, in the rows of the species the step gates, not V's own, so
    nothing of it is mended and the straddled slope still carries the run.
    Mended, the run crossed on macOS at these tolerances, 0.633 off at rtol 1e-7
    and 3e-7 (601 points), where main is right there."""
    net, exact = _split(data_dir, tmp_path, "49.99999999999")
    window = dict(t_span=(0.0, 150.0), n_points=n_points, rtol=rtol, atol=atol)
    assert "across a jump" not in _split_mends(net, monkeypatch, capfd, sparse=False, **window)
    if rtol == 1e-8:
        assert _split_deviation(net, exact, jacobian="fd", sparse=False, **window) < 1e-6


@pytest.mark.parametrize(
    ("rtol", "atol", "sparse"),
    [(1e-6, 1e-6, False), (1e-6, 1e-8, True), (3e-7, 1e-6, True), (1e-8, 1e-12, True)],
)
def test_a_sum_parked_on_its_threshold_is_not_mended(
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    rtol: float,
    atol: float,
    sparse: bool,
) -> None:
    """Control. `split` with the stimulus at exactly 50, so V + V2 parks on the
    step itself (the #176 fixture upstream): nothing is mended. With V's mends
    kept in the gated rows, a step back along the flow let them through on
    macOS and the run returned 0.633 off with no error at each of these
    tolerances, where main fails loudly there."""
    net, _ = _split(data_dir, tmp_path, "50.0")
    window = dict(t_span=(0.0, 150.0), n_points=301, rtol=rtol, atol=atol)
    assert "across a jump" not in _split_mends(net, monkeypatch, capfd, sparse=sparse, **window)


# ── The two steady-state quotients #1302 left as they were (issue #1335) ─────
#
# KINSOL's own difference quotient (``method="newton"`` with ``jacobian="fd"``;
# a declined law sends ``newton`` to the march instead) and the march's colored
# sparse one (KLU, forced here) have no mend for a step across a jump. Neither
# has been seen to need one: #1302 tried 72 and 24 variants and found no wrong
# answer. These pin that, from a start on a threshold and just either side of
# it, and on a species held at 0, against closed forms. Each asserts the path
# it took, so it cannot pass on another one.
#
# Three paths:
# * "KINSOL" starts KINSOL on the given state (``newton_burst=False``). The
#   default ``newton`` marches a burst into the root's basin first, so KINSOL's
#   quotient would be formed away from the jump.
# * "burst on KLU, then KINSOL" is the default ``newton`` with KLU forced: a
#   burst on the march's colored quotient, then KINSOL's polish, which always
#   factors densely (``linear_solver`` names the march's route).
# * "colored march" is the march alone.
#
# Every root here is exact: f is affine in X on each side of the threshold, so
# each solve lands on its root to rounding (relative error 0 on every row,
# measured). ROOT leaves a platform its last bits.
#
# A final state cannot see a wrong colored quotient: with it negated, every
# threshold row still lands on its root. The march's work can: on these rows it
# takes at most 236 RHS evaluations with the right quotient, and at least 414
# with it negated (7,896 on the gate, unconverged). MARCH_RHS sits between.
# The march's rows need KLU, so a KLU-off build skips them and their budget.
ROOT = 1e-12
MARCH_RHS = 350


def _source_and_switch(tmp_path: Path, s: float, x0: float) -> bngsim.Model:
    """X made at s and lost at kx·k(X)·X, k = 0.04 above 5 and 0.02 at or below.
    The one steady state is s/0.02 where that is at most 5, else s/0.04."""
    return _net(
        tmp_path,
        [f"X() {x0!r}", "S() 1"],
        ["Xobs 1", "Sobs 2"],
        ["f() kx*if(Xobs > 5, 0.04, 0.02)", f"src() {s!r}/Sobs"],
        ["1 0 f", "2 1,2 src"],
    )


def _kinsol(sim: bngsim.Simulator):
    """KINSOL from the model's own state, with no burst first."""
    return sim.steady_state_batch(params=[{}], method="newton", newton_burst=False)[0]


# (simulator options, how it is solved, the path it must take)
_PATHS = {
    "KINSOL": ({}, _kinsol, ("newton", "dense")),
    "burst on KLU, then KINSOL": (
        {"force_sparse_linear_solver": True},
        lambda sim: sim.steady_state(method="newton"),
        ("newton", "klu"),
    ),
    "colored march": (
        {"force_sparse_linear_solver": True},
        lambda sim: sim.steady_state(method="integration"),
        ("integration", "klu"),
    ),
}


_requires_klu = pytest.mark.skipif(
    not bngsim._bngsim_core.HAS_KLU, reason="build has no SuiteSparse/KLU"
)
_PATH_PARAMS = [
    pytest.param(name, marks=_requires_klu) if taken[1] == "klu" else name
    for name, (_, _, taken) in _PATHS.items()
]


def _solved_on(result, path: tuple[str, str]) -> None:
    assert result.converged
    assert (result.method_used, result.linear_solver) == path
    assert result.solver_jacobian_source == "finite-difference"
    if path[0] == "integration":
        assert result.n_rhs_evals <= MARCH_RHS


@pytest.mark.parametrize("path", _PATH_PARAMS)
@pytest.mark.parametrize("x0", [5.0, 5.0 + 1e-9, 5.0 - 1e-9], ids=["on", "above", "below"])
@pytest.mark.parametrize(("s", "exact"), [(0.08, 4.0), (0.4, 10.0)], ids=["lands below", "above"])
def test_a_steady_state_solved_from_a_threshold(
    tmp_path: Path, path: str, x0: float, s: float, exact: float
) -> None:
    sim_kw, solve, taken = _PATHS[path]
    model = _source_and_switch(tmp_path, s, x0)
    result = solve(bngsim.Simulator(model, method="ode", jacobian="fd", **sim_kw))
    _solved_on(result, taken)
    assert abs(float(np.asarray(result.concentrations)[0]) / exact - 1.0) < ROOT


@pytest.mark.parametrize("path", _PATH_PARAMS)
@pytest.mark.parametrize(
    ("model", "exact"), [(_gate, 0.5), (_held_inhibitor, 20.0)], ids=["gate", "inhibitor"]
)
def test_a_steady_state_beside_a_species_held_at_zero(tmp_path: Path, path: str, model, exact):
    """The run's two #1002 failures, on the two quotients the runs' mend does
    not reach: the held species stays exactly 0, and Y lands where it should."""
    sim_kw, solve, taken = _PATHS[path]
    result = solve(bngsim.Simulator(model(tmp_path), method="ode", jacobian="fd", **sim_kw))
    _solved_on(result, taken)
    c = np.asarray(result.concentrations)
    assert c[0] == 0.0
    assert abs(c[1] / exact - 1.0) < ROOT
