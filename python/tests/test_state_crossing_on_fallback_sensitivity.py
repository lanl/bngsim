"""A rate law that switches where the state crosses a threshold, in a run left
on CVODES' difference quotient (issues #938 and #932).

One rate law the analytic sensitivity right-hand side declines, ``kc*max(Aobs,
0.5)``, puts every column of the model on CVODES' internal difference quotient.
That quotient reads the rate law at ``y + σ·s``. Beside a surface the state
crosses, ``y + σ·s`` is on the other branch for any column whose sensitivity
moves the state across it. Where the rate law jumps there, the difference is the
jump over ``σ``: the column takes part of the jump before the crossing, and the
jump applied at the crossing is added to it. dY/dk came back 14.52 for 10.2,
beside a warning that said the columns were wrong. Other models of the same
shape ended in CVODE's no-progress error just short of the crossing, or in the
wall clock.

Nothing at the crossing can put that right: the column is wrong before the run
gets there, and a run that ends short of the crossing returns it. So a model
with a crossing the quotient reads across is refused before a time course
starts. A crossing on literal time is not one, and neither is a counter that no
requested column moves.

Every expected value is a closed form.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

NET = """begin parameters
    1 A0 {A0!r}
    2 k 1.0
    3 thr 4.4
    4 kb 3.0
    5 kc 5.0
    6 tau 3.4
    7 P 1.3
end parameters
begin functions
    1 fY() {fy}
    2 fZ() {fz}
end functions
begin species
    1 A() A0
    2 Y() 0
    3 Z() 0
end species
begin reactions
    1 {a_rxn} k
    2 0 2 fY
    3 0 3 fZ
end reactions
begin groups
    1 Aobs 1
end groups
"""

DECLINED = "kc*max(Aobs,0.5)"
T_END = 6.0


def _simulator(tmp_path, fy, fz, params, decays=False):
    """A is made at rate k from 1, or, with ``decays``, decays at k·A from 10."""
    path = tmp_path / "m.net"
    path.write_text(
        NET.format(fy=fy, fz=fz, A0=10.0 if decays else 1.0, a_rxn="1 0" if decays else "0 1")
    )
    return bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=params)


def _y_columns(sim):
    run = sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-10, atol=1e-12, timeout=60)
    return np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), :]


def _refused(sim, names, rtol=1e-10):
    """Refused before the run starts, by an error that names the crossing."""
    assert not sim.has_analytic_sens_rhs
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938") as caught:
        sim.run(t_span=(0.0, T_END), n_points=3, rtol=rtol, atol=1e-2 * rtol, timeout=10)
    assert names in str(caught.value)


def test_a_counter_that_jumps_beside_a_declined_rate_law_is_refused(tmp_path):
    """A = A0 + k·t passes thr at 3.4, and Y = kb·(T − t*) from there:
    dY/dk = 10.2. It came back 14.52."""
    sim = _simulator(tmp_path, "if(Aobs>thr,kb,0)", DECLINED, ["k", "thr"])
    _refused(sim, "Aobs>thr")


def test_the_same_counter_thresholded_twice_is_refused(tmp_path):
    """The model of issue #938: a second condition on the same surface, through
    ``max()``. dY/dk came back 14.55."""
    sim = _simulator(tmp_path, "if(Aobs>thr,kb,0)", "if(max(Aobs-thr,0)>0,kc,0)", ["k", "thr"])
    _refused(sim, "Aobs")


def test_a_counter_against_a_literal_is_moved_too(tmp_path):
    """``Aobs > 4.4`` is a threshold nothing moves, crossed at a time the
    counter's own rate constant and initial amount both move. dY/dk and dY/dA0
    came back 14.52 and 4.27 for 10.2 and 3."""
    sim = _simulator(tmp_path, "if(Aobs>4.4,kb,0)", DECLINED, ["k", "A0"])
    _refused(sim, "Aobs>4.4")


@pytest.mark.parametrize("kc", ["1e6", "1e8"])
def test_a_counter_beside_a_flux_a_million_times_the_jump_is_refused(tmp_path, kc):
    """The declined rate law is ``kc·max(Aobs, 0.5)`` with kc at 1e6 and 1e8.
    A cut that refused at the crossing asked whether the rate law jumps there
    by more than a millionth of the largest flux in the model, and read a jump
    of 3 beside 4.4e6 as none: dY/dk came back 28.75 at rtol 1e-4 and 14.56 at
    1e-10, for 10.2."""
    sim = _simulator(tmp_path, "if(Aobs>thr,kb,0)", f"{kc}*max(Aobs,0.5)", ["k", "thr"])
    _refused(sim, "Aobs>thr")


def test_a_column_that_does_not_move_the_counter_runs(tmp_path):
    """Control. Only thr is requested: its value is held while the quotient is
    taken and no sensitivity moves A, so nothing reads across the threshold.
    dY/dthr = −kb/k."""
    sim = _simulator(tmp_path, "if(Aobs>thr,kb,0)", DECLINED, ["thr"])
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [-3.0], rtol=1e-8)


@pytest.mark.parametrize("param", ["k", "A0", "thr"])
def test_a_state_switch_that_jumps_is_refused(tmp_path, param):
    """A decays through thr and Y is made at kb from there. At rtol 1e-4 the run
    got to the crossing on the quotient: dY/dk came back 6.94 for 2.46,
    dY/dA0 −0.846 for −0.3 and dY/dthr 1.92 for 0.68, each with the jump taken
    nearly three times over. At tighter tolerances it stalled short of the
    crossing instead."""
    sim = _simulator(tmp_path, "if(Aobs<thr,kb,0)", DECLINED, [param], decays=True)
    _refused(sim, "Aobs<thr", rtol=1e-4)


def test_a_run_that_ends_short_of_the_crossing_is_refused_too(tmp_path):
    """The quotient reads across the surface before the state gets there, so a
    run that ends 1e-5 short of the crossing has part of the jump already:
    dY/dk = 0.59 for 0 at rtol 1e-4. Refused at the crossing, such a run was
    never refused."""
    sim = _simulator(tmp_path, "if(Aobs>thr,kb,0)", DECLINED, ["k", "thr"])
    assert not sim.has_analytic_sens_rhs
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run(t_span=(0.0, 3.4 - 1e-5), n_points=3, rtol=1e-4, atol=1e-6, timeout=10)


def test_a_step_call_on_the_state_is_refused(tmp_path):
    """``floor(Aobs/P)`` steps each time A passes a multiple of P, and no root
    is placed on a step call: the run ended in CVODE's no-progress error at the
    first one."""
    sim = _simulator(tmp_path, "kb", "kc*floor(Aobs/P)", ["k", "P"])
    _refused(sim, "floor(Aobs/P)")


def test_a_step_call_on_the_state_inside_a_condition_is_refused(tmp_path):
    """``if(floor(Aobs/P) > 2, 1, 2)``: the condition's atom reads the state
    through a step call. dZ/dk came back −67.9 for −49.5."""
    law = "kc*max(Aobs,0.5)*if(floor(Aobs/P)>2,1,2)"
    sim = _simulator(tmp_path, "kb", law, ["k"], decays=True)
    _refused(sim, "floor(Aobs/P)", rtol=1e-4)


def test_a_step_of_time_that_a_requested_parameter_moves_is_refused(tmp_path):
    """``kc·floor(time()/P)`` with P requested: the steps move with P, and
    nothing jumps a column at a step. dZ/dP came back −56.7 for −50 at rtol
    1e-4, and the run stalled at the first step at 1e-10."""
    sim = _simulator(tmp_path, "if(time()>tau,kb,0)", "kc*floor(time()/P)", ["P", "kc"])
    _refused(sim, "floor(time()/P)", rtol=1e-4)


STEP_TABLE = NET.replace(
    "    1 fY() {fy}", '    1 fY() tfun([0,2,4.4,8],[0,1,3,5],Aobs,method=>"{method}")'
)


@pytest.mark.parametrize("method", ["step", "linear"])
def test_a_table_that_steps_on_the_state_is_refused(tmp_path, method):
    """A table function read as a step, indexed by an observable: no condition,
    no step call and no root. dY/dk came back 4.42 for 2.46. Read with linear
    interpolation the same table is continuous, and runs."""
    path = tmp_path / "m.net"
    path.write_text(STEP_TABLE.format(method=method, fz="kc*Aobs", A0=10.0, a_rxn="1 0"))
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode", sensitivity_params=["k"])
    if method == "step":
        _refused(sim, "tfun", rtol=1e-4)
    else:
        assert not sim.has_analytic_sens_rhs
        sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-6, atol=1e-8, timeout=60)


ISSUE_932 = {
    # The condition reads rateOf.
    "rateof-condition": (
        "species A, Y; A = A0; Y = 0; A0 = 10; a = 0.5; thr = 1; kb = 3\n"
        "J0: A -> ; a*A\n"
        "J1: -> Y; piecewise(kb, rateOf(A) > -thr, 0)\n",
        ["thr"],
    ),
    # rateOf is only in a rule an event's trigger reads.
    "rateof-in-a-rule": (
        "species A, Y, Z, B; A = 10; Y = 0; Z = 0; B = 0; a = 0.5; thr = 1; q = 3\n"
        "J0: A -> ; a*A\n"
        "r := rateOf(A)\n"
        "J1: -> B; piecewise(1, A < q, 0)\n"
        "J2: -> Z; Y\n"
        "E: at (r > -thr): Y = 1\n",
        ["a", "thr", "q"],
    ),
    # No rateOf at all: any rate law the analytic path declines.
    "max-beside-it": (
        "species A, B, Z; A = 10; B = 0; Z = 0; a = 0.5; q = 3\n"
        "J0: A -> ; a*A\n"
        "J1: -> B; piecewise(1, A < q, 0)\n"
        "J2: -> Z; max(A, 0.1)\n",
        ["a", "q"],
    ),
}


@pytest.mark.parametrize("model", sorted(ISSUE_932))
def test_a_run_that_stalled_short_of_a_jump_is_refused_by_name(model):
    """The models of issue #932. Each ended in ``CVODE made no progress`` just
    before the crossing, as a solver failure. It is a refusal now, before the
    run, and it names the crossing and what was declined. The plain run of each
    is right."""
    text, params = ISSUE_932[model]
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=params
    )
    assert not sim.has_analytic_sens_rhs
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#932"):
        sim.run(sample_times=np.linspace(0.0, 6.0, 7), rtol=1e-10, atol=1e-12, timeout=10)


@pytest.mark.parametrize(
    ("fy", "fz", "params", "decays"),
    [
        ("if(Aobs<thr,kb,0)", DECLINED, ["kb", "kc"], True),
        ("if(Aobs>4.4,kb*(Aobs-4.4),0)", DECLINED, ["k"], False),
        ("if(Aobs<thr,kb*(thr-Aobs),0)", DECLINED, ["k", "thr"], True),
        ("if((Aobs-thr)<0,(-(Aobs-thr))*kb,0)", DECLINED, ["k", "thr"], True),
    ],
    ids=["columns-that-do-not-move-it", "a-ramp-on-a-counter", "a-ramp", "a-signed-rate"],
)
def test_crossings_the_quotient_gets_right_are_refused_all_the_same(
    tmp_path, fy, fz, params, decays
):
    """Four the difference quotient is right for, and main returns: columns
    that move neither the state nor the threshold, and rate laws that turn on
    from 0 at the crossing, so that there is no jump to read across. Whether a
    rate law jumps where the state will cross, and whether a column moves the
    crossing, are not known before the run; a model with a crossing on the
    state is refused on the quotient whichever it is."""
    sim = _simulator(tmp_path, fy, fz, params, decays=decays)
    _refused(sim, "Aobs")


def test_a_time_crossing_beside_a_declined_rate_law_runs(tmp_path):
    """Control. ``time() > tau`` on the difference quotient: tau is held while
    the quotient is taken, and Y = kb·(T − tau) gives dY/dtau = −kb."""
    sim = _simulator(tmp_path, "if(time()>tau,kb,0)", DECLINED, ["k", "tau"])
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [0.0, -3.0], atol=1e-8)


def test_a_fixed_step_of_time_beside_a_time_crossing_runs(tmp_path):
    """Control. ``floor(time()/7)`` is declined and steps at a fixed time."""
    sim = _simulator(tmp_path, "if(time()>tau,kb,0)", "kc*floor(time()/7)", ["tau", "kb"])
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [-3.0, T_END - 3.4], rtol=1e-8)


def test_a_step_of_time_whose_parameter_is_not_requested_runs(tmp_path):
    """Control. ``floor(time()/P)`` with only kc requested: the steps do not
    move. Z = kc·1.3·(0 + 1 + 2 + 3) + kc·4·(6 − 5.2), so dZ/dkc = 11."""
    sim = _simulator(tmp_path, "if(time()>tau,kb,0)", "kc*floor(time()/P)", ["kc"])
    assert not sim.has_analytic_sens_rhs
    run = sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-10, atol=1e-12, timeout=60)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Z()"), 0]
    assert got == pytest.approx(1.3 * 6.0 + 4.0 * 0.8, rel=1e-8)


def test_a_jump_on_the_analytic_path_runs(tmp_path):
    """Control. The same crossing with nothing declined: dY/dk = 10.2 and
    dY/dthr = −3."""
    sim = _simulator(tmp_path, "if(Aobs>thr,kb,0)", "kc*Aobs", ["k", "thr"])
    assert sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [10.2, -3.0], rtol=1e-8)


def test_a_declined_rate_law_with_no_crossing_runs(tmp_path):
    """Control. Nothing crosses: the difference quotient is right, and slower."""
    sim = _simulator(tmp_path, "kb", DECLINED, ["k", "kb"])
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [0.0, T_END], rtol=1e-8, atol=1e-10)


def test_the_plain_run_is_untouched(tmp_path):
    """Control. Y(6) = kb·(6 − 3.4)."""
    path = tmp_path / "m.net"
    path.write_text(NET.format(fy="if(Aobs>thr,kb,0)", fz=DECLINED, A0=1.0, a_rxn="0 1"))
    run = bngsim.Simulator(bngsim.Model.from_net(path), method="ode").run(
        t_span=(0.0, T_END), n_points=3, rtol=1e-10, atol=1e-12
    )
    assert np.asarray(run.species)[-1, list(run.species_names).index("Y()")] == pytest.approx(
        7.8, rel=1e-8
    )


def test_a_steady_state_solve_is_not_refused():
    """Control. A steady state reads ∂f/∂p at one state and integrates through
    nothing, step calls on the state included."""
    text = (
        "species A, B; A = 10; B = 0; a = 0.5; q = 3; ks = 1\n"
        "J0: A -> ; a*A\n"
        "Js: -> A; ks\n"
        "J1: -> B; piecewise(1, A < q, 0)\n"
        "Jd: B -> ; max(B, 0.1) + 0*floor(A)\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["a", "ks"]
    )
    assert not sim.has_analytic_sens_rhs
    sim.steady_state(sensitivity_params=["a", "ks"])
