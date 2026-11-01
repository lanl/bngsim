"""A rate law that branches on an equality the state sits on is refused in a
sensitivity run where leaving it is a switch nothing jumps, not differentiated
to 0 (issues #1228, #1279).

``fY = if(Bobs==0, kb, 0)``, where B is made only once A = 10·e^(-a·t) falls to
q, at t* = ln(10/q)/a. The equality holds from the start until t*, so Y =
kb·t*, and the exact columns are dY/da = -kb·ln(10/q)/a² = -6.437752, dY/dq =
-kb/(a·q) = -1 and dY/dkb = t* = 3.218876.

``state_switch_conditions`` roots no equality, since one the state crosses
holds for an instant. The core admits the equality's surface (#381) and reads
its branch gap either side of it, where both sides are false. So leaving a
surface the state sat on was a switch nothing saw: the run came back with
dY/da = dY/dq = 0 and no warning. ``Bobs<=0``, the same switch, is located at
t* and refused as a tangency.

#1240 refused every run that starts on such a surface and stays, which also
refused runs main got right (#1279). The solver now roots where the state
leaves, and refuses there only where the rate laws reading the equality change
and no jump at that instant covers it. A stretch that outlasts the run, or that
a rooted condition inside the equality ends, or over which the rest of the
condition never holds, runs.

An equality the state only starts on and leaves at once, or crosses, or that
no rate law reads, runs as before.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest
from bngsim._exceptions import SensitivityUnsupportedError

A, Q, KB = 0.5, 2.0, 1.0
T_STAR = math.log(10 / Q) / A
EXACT = [-KB * math.log(10 / Q) / A**2, -KB / (A * Q), T_STAR]  # dY/d[a, q, kb]
# Of kb·∫A over (0, 8), for Y' = kb·A beside a branch that adds nothing.
_INT_A = 10 * (1 - math.exp(-8 * A)) / A
EXACT_INT_A = [KB * 10 * (8 * math.exp(-8 * A) / A - (1 - math.exp(-8 * A)) / A**2), 0.0, _INT_A]


def _model(tmp_path, f_y, b0=0.0, extra="", f_b="kp*if(q-Aobs>0,q-Aobs,0)"):
    path = tmp_path / "m.net"
    path.write_text(
        f"begin parameters\n 1 a {A}\n 2 q {Q}\n 3 kb {KB}\n 4 kp 1.0\nend parameters\n"
        f"begin functions\n 1 fB() {f_b}\n"
        f" 2 fY() {f_y}\n{extra}end functions\n"
        f"begin species\n 1 A() 10\n 2 B() {b0}\n 3 Y() 0\nend species\n"
        "begin reactions\n 1 1 0 a\n 2 0 2 fB\n 3 0 3 fY\nend reactions\n"
        "begin groups\n 1 Aobs 1\n 2 Bobs 2\nend groups\n"
    )
    return bngsim.Model.from_net(str(path))


def _columns(model, t_span=(0, 8)):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["a", "q", "kb"])
    result = sim.run(t_span=t_span, n_points=9, rtol=1e-10, atol=1e-12)
    return np.asarray(result.sensitivities)[-1, list(result.species_names).index("Y()"), :]


@pytest.mark.parametrize("cond", ["Bobs==0", "Bobs!=0"])
def test_an_equality_the_state_sits_on_is_refused_by_name(tmp_path, cond):
    with pytest.raises(SensitivityUnsupportedError, match=rf"'{cond}'.*issue #1228"):
        _columns(_model(tmp_path, f"if({cond},kb,0)"))


def test_the_same_switch_written_as_its_crossing_is_exact(tmp_path):
    """The control: the switch that ends the stretch, `Aobs>q`, is located and
    jumped, and its columns are the closed form."""
    np.testing.assert_allclose(_columns(_model(tmp_path, "if(Aobs>q,kb,0)")), EXACT, rtol=1e-6)


def test_the_same_switch_written_as_an_inequality_is_refused_at_its_tangency(tmp_path):
    """`Bobs<=0` names the same surface and is refused, as before: B leaves 0
    with no first-order motion. The equality now agrees with it."""
    with pytest.raises(Exception, match="rides that surface"):
        _columns(_model(tmp_path, "if(Bobs<=0,kb,0)"))


@pytest.mark.parametrize(
    ("cond", "want"),
    [("Aobs==10", [0.0, 0.0, 0.0]), ("Aobs!=10", [0.0, 0.0, 8.0])],
    ids=["holds at t = 0 alone", "false at t = 0 alone"],
)
def test_an_equality_the_state_starts_on_and_leaves_runs(tmp_path, cond, want):
    """A decays from 10 at once, so `Aobs==10` holds for the instant t = 0:
    Y gains nothing, and `Aobs!=10` is on for the whole run, Y = 8·kb."""
    np.testing.assert_allclose(_columns(_model(tmp_path, f"if({cond},kb,0)")), want, atol=1e-9)


def test_an_equality_the_state_only_crosses_runs(tmp_path):
    """B starts at 1 and grows, so `Bobs==0` never holds: Y = 0."""
    np.testing.assert_allclose(
        _columns(_model(tmp_path, "if(Bobs==0,kb,0)", b0=1.0)), [0.0, 0.0, 0.0], atol=1e-9
    )


def test_an_equality_no_rate_law_reads_runs(tmp_path):
    model = _model(tmp_path, "if(Aobs>q,kb,0)", extra=" 3 fOut() if(Bobs==0,1,0)\n")
    np.testing.assert_allclose(_columns(model), EXACT, rtol=1e-6)


def test_a_plain_run_is_unaffected(tmp_path):
    result = bngsim.Simulator(_model(tmp_path, "if(Bobs==0,kb,0)"), method="ode").run(
        t_span=(0, 8), n_points=9, rtol=1e-10, atol=1e-12
    )
    assert float(np.asarray(result.species)[-1, 2]) == pytest.approx(KB * T_STAR, rel=1e-6)


@pytest.mark.parametrize(
    ("f_y", "f_b", "t_span", "want"),
    [
        ("if(Bobs==0,kb,0)", "0", (0, 8), [0.0, 0.0, 8.0]),
        ("if(Bobs==0,kb,0)", None, (0, 2), [0.0, 0.0, 2.0]),
        ("if(fB()==0,kb,0)", None, (0, 8), EXACT),
        ("if(Bobs==0&&Aobs<1,kb,0)", None, (0, 8), [0.0, 0.0, 0.0]),
        ("if(Bobs==0&&Aobs<1,kb,0)+kb*Aobs", "if(time()>3,kp,0)", (0, 8), EXACT_INT_A),
    ],
    ids=[
        "B never leaves",
        "the run ends before t*",
        "a rooted condition inside the equality ends it",
        "the rest of the condition never holds as B leaves",
        "only a reaction that does not read the equality jumps as B leaves",
    ],
)
def test_a_stretch_whose_leaving_needs_no_jump_runs(tmp_path, f_y, f_b, t_span, want):
    """#1240 refused all five, where main's columns are exact (issue #1279).
    With A above 1 when B leaves, `Bobs==0&&Aobs<1` reads false either side.
    In the last, B's own production jumps on at t = 3 as B leaves: the leaving
    is judged by the laws that read the equality alone."""
    kw = {} if f_b is None else {"f_b": f_b}
    np.testing.assert_allclose(
        _columns(_model(tmp_path, f_y, **kw), t_span=t_span), want, rtol=1e-6, atol=1e-9
    )


@pytest.mark.parametrize(("q", "a"), [(1.5, 0.5), (1.5, 0.37), (1.6, 0.5), (2.0, 0.5)])
def test_a_leaving_a_switch_inside_the_equality_jumps_at_runs(tmp_path, q, a):
    """`if(fB()==0, kb, 0)` with fB gated by `q-Aobs>0`: that switch's jump at
    t* reads fY either side of its surface. Whether its restart lands on the
    equality's surface, so that the leaving is rooted just after and judged,
    or past it, is rounding: of these, (2.0, 0.5) lands past it on macOS and on
    it on Linux, which refused it there. dY/d[a, q, kb] = [-ln(10/q)/a²,
    -1/(a·q), ln(10/q)/a]·kb."""
    path = tmp_path / "m.net"
    path.write_text(
        f"begin parameters\n 1 a {a!r}\n 2 q {q!r}\n 3 kb {KB}\n 4 kp 1.0\nend parameters\n"
        "begin functions\n 1 fB() kp*if(q-Aobs>0,q-Aobs,0)\n 2 fY() if(fB()==0,kb,0)\n"
        "end functions\n"
        "begin species\n 1 A() 10\n 2 B() 0\n 3 Y() 0\nend species\n"
        "begin reactions\n 1 1 0 a\n 2 0 2 fB\n 3 0 3 fY\nend reactions\n"
        "begin groups\n 1 Aobs 1\n 2 Bobs 2\nend groups\n"
    )
    t_star = math.log(10 / q) / a
    want = [-KB * t_star / a, -KB / (a * q), t_star]
    np.testing.assert_allclose(_columns(bngsim.Model.from_net(str(path))), want, rtol=1e-6)


def test_a_held_equality_is_read_at_the_run_start_time(tmp_path):
    """B is made at once before t = 1, so a run from 0 leaves `Bobs==0` at once
    and runs, while a run from 5 holds B at 0 until A falls to q and is
    refused there. The flow read at t = 0 for both gave the second
    [0, 0, 3.218876] for [-6.4378, -1, 3.2189] by difference."""
    f_b = "if(time()<1,1,0)+kp*if(q-Aobs>0,q-Aobs,0)"
    np.testing.assert_allclose(
        _columns(_model(tmp_path, "if(Bobs==0,kb,0)", f_b=f_b), t_span=(0, 20)),
        [0.0, 0.0, 0.0],
        atol=1e-9,
    )
    with pytest.raises(SensitivityUnsupportedError, match=r"'Bobs==0'.*issue #1228"):
        _columns(_model(tmp_path, "if(Bobs==0,kb,0)", f_b=f_b), t_span=(5, 20))


@pytest.mark.parametrize("entry", ["run_batch", "compute_all_sensitivities"])
def test_every_entry_point_refuses_it_typed(tmp_path, entry):
    """compute_all_sensitivities retried the refused chunk column by column and
    reported a SimulationError that blamed the tolerance."""
    sim = bngsim.Simulator(
        _model(tmp_path, "if(Bobs==0,kb,0)"), method="ode", sensitivity_params=["a", "q", "kb"]
    )
    with pytest.raises(SensitivityUnsupportedError, match=r"'Bobs==0'.*issue #1228"):
        if entry == "run_batch":
            sim.run_batch(t_span=(0, 8), n_points=9, params=[{"a": A}, {"a": 0.6}])
        else:
            sim.compute_all_sensitivities(
                t_span=(0, 8), n_points=9, params=["a", "q", "kb"], chunk_size=1, n_workers=1
            )


def test_a_carried_sensitivity_seed_survives_the_check(tmp_path):
    """A rate law with an equality the state is not on, so the check probes and
    passes. Its set_state dropped the dx/dθ a pre-equilibration hands the next
    run (GH #210), and that run raised for want of it. A = k_prod/k_deg at the
    steady state: dA/dk_prod = 1/k_deg = 2, dA/dk_deg = -k_prod/k_deg² = -8."""
    path = tmp_path / "preequil.net"
    path.write_text(
        "begin parameters\n 1 k_prod 2.0\n 2 k_deg 0.5\nend parameters\n"
        "begin functions\n 1 fZ() if(Aobs==1000,1,0)\nend functions\n"
        "begin species\n 1 A() 0\n 2 Z() 0\nend species\n"
        "begin reactions\n 1 0 1 k_prod\n 2 1 0 k_deg\n 3 0 2 fZ\nend reactions\n"
        "begin groups\n 1 Aobs 1\nend groups\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=["k_prod", "k_deg"]
    )
    sim.run(t_span=(0, 200), n_points=3, steady_state=True)
    result = sim.run(t_span=(0, 3), n_points=5, carry_sensitivities=True)
    np.testing.assert_allclose(
        np.asarray(result.sensitivities)[:, 0, :], [[2.0, -8.0]] * 5, rtol=1e-6
    )
