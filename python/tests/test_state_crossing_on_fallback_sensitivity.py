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

What is refused is what the text says: every condition on the state, whatever
the law does where it flips, and a sign or a step written by dividing by an
``abs``, ``max`` or ``min``. Nothing is evaluated. A bend written with ``max``
or ``min`` runs; the same bend written as a condition is refused, where main
is right.

Every expected value is a closed form, or a central difference of plain runs
where that is said.
"""

from __future__ import annotations

import math

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
    no step call and no root. dY/dk came back 4.42 for 2.46.

    Control (the linear case). Read with linear interpolation the same table is
    continuous, and runs as it did."""
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


def test_a_jump_no_requested_column_moves_is_refused_all_the_same(tmp_path):
    """kb and kc move neither A nor thr, so neither column reads across the
    surface, and main returns them: dY/dkb = T − t* and dY/dkc = 0. Whether a
    column moves the crossing is not known before the run, and a model whose
    rate law jumps where the state crosses is refused on the quotient whichever
    it is."""
    sim = _simulator(tmp_path, "if(Aobs<thr,kb,0)", DECLINED, ["kb", "kc"], decays=True)
    _refused(sim, "Aobs<thr")


WIDER = """begin parameters
    1 A0 {A0!r}
    2 k {k!r}
    3 thr 4.4
    4 kb 3.0
    5 kc 5.0
    6 tau 3.4
    7 P 1.3
    8 one 1.0
    9 n {n!r}
   10 c -4.0
   11 g 1e-4
   12 idx 2*n
   13 near 1.001
end parameters
begin functions
{funcs}
end functions
begin species
    1 A() A0
    2 Y() 0
    3 Z() 0
    4 C() 0
    5 B() 1e-4
end species
begin reactions
    1 {a_rxn} k
    2 0 2 fY
    3 0 3 fZ
    4 0 4 one
end reactions
begin groups
    1 Aobs 1
    2 Cobs 4
    3 Bobs 5
end groups
"""


def _wider(tmp_path, fy, fz=DECLINED, decays=True, extra=(), k=1.0, n=0.0):
    """The model above with a counter C that nothing but its own rate moves,
    a species B that stays at 1e-4, an exponent n and a few more parameters."""
    funcs = [*extra, f"fY() {fy}", f"fZ() {fz}"]
    path = tmp_path / "wider.net"
    path.write_text(
        WIDER.format(
            funcs="\n".join(f"    {i} {f}" for i, f in enumerate(funcs, 1)),
            A0=10.0 if decays else 1.0,
            a_rxn="1 0" if decays else "0 1",
            k=k,
            n=n,
        )
    )
    return bngsim.Model.from_net(path)


def _refused_run(model, params, **kwargs):
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
    assert not sim.has_analytic_sens_rhs
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-4, atol=1e-6, timeout=20, **kwargs)


def _ramp_columns():
    """dY/dk and dY/dthr of Y' = kb·(thr − A) from where A = 10·exp(−k·t)
    passes thr."""
    a0, k, thr, kb = 10.0, 1.0, 4.4, 3.0
    t_star = math.log(a0 / thr) / k
    tail = math.exp(-k * T_END)
    return [
        kb * (thr * t_star / k + thr / k**2 - a0 * T_END * tail / k - a0 * tail / k**2),
        kb * (T_END - t_star),
    ]


BENDS_PROVED = {
    "a-ramp": "if(Aobs<thr,kb*(thr-Aobs),0)",
    "a-signed-rate": "if((Aobs-thr)<0,(-(Aobs-thr))*kb,0)",
    "a-ramp-whose-condition-divides-by-a-parameter": "if((Aobs-thr)/tau<0,kb*(thr-Aobs),0)",
    "a-ramp-over-a-floor": "if(Aobs<thr,(thr-Aobs)*kb*max(Aobs,0.01)/max(Aobs,0.01),0)",
    "the-greater-of-two-written-out": "kb*(if(thr>Aobs,thr,Aobs)-Aobs)",
}


@pytest.mark.parametrize("case", sorted(BENDS_PROVED))
def test_a_bend_that_is_proved_one_runs(tmp_path, case):
    """Control. A decays through thr and the rate law turns on from 0 there.
    One branch is 0 and the other a multiple of the difference the condition
    compares, or the two branches are the two sides of the comparison, and
    nothing in the law divides by what could be 0: continuous by what it is
    made of, and the quotient is right across a bend."""
    sim = _simulator(tmp_path, BENDS_PROVED[case], DECLINED, ["k", "thr"], decays=True)
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), _ramp_columns(), rtol=1e-6)


def test_a_ramp_from_a_counter_s_threshold_runs(tmp_path):
    """Control. The rate law turns on as a ramp from the counter's threshold:
    Y = kb·k·(T − t*)²/2 with t* = (thr − A0)/k, so
    dY/dk = kb·(T − t*)²/2 + kb·(T − t*)·t* = 36.66."""
    sim = _simulator(tmp_path, "if(Aobs>4.4,kb*(Aobs-4.4),0)", DECLINED, ["k"])
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [36.66], rtol=1e-6)


def test_a_window_written_as_a_product_runs(tmp_path):
    """Control. ``(A − 4.4)·(A − 5) < 0`` is 0 with each of its two factors,
    and the law has both. Against a central difference of plain runs."""
    law = "if((Aobs-4.4)*(Aobs-5)<0,kb*(Aobs-4.4)*(5-Aobs),0)"

    def build(k):
        return _wider(tmp_path, law, k=k)

    sim = bngsim.Simulator(build(1.0), method="ode", sensitivity_params=["k"])
    assert not sim.has_analytic_sens_rhs
    run = sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-10, atol=1e-12, timeout=60)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), 0]

    def plain(k):
        out = bngsim.Simulator(build(k), method="ode").run(
            t_span=(0.0, T_END), n_points=3, rtol=1e-12, atol=1e-14, timeout=60
        )
        return np.asarray(out.species)[-1, list(out.species_names).index("Y()")]

    def slope(h):
        return (plain(1.0 + h) - plain(1.0 - h)) / (2.0 * h)

    assert got == pytest.approx((4.0 * slope(5e-4) - slope(1e-3)) / 3.0, rel=1e-5)


BENDS_NOT_PROVED = {
    # Continuous, each of them, and main is right on it.
    "a-guard": "if(Aobs>0,kb/Aobs,0)",
    "a-ramp-as-a-root": "if(Aobs<thr,kb*sqrt(thr-Aobs),0)",
    "a-ramp-to-a-power-that-is-no-whole-number": "if(Aobs<thr,kb*(thr-Aobs)^P,0)",
    "an-onset-through-a-function": "if(Aobs<thr,kb*(1-exp(-(thr-Aobs)/2)),0)",
    "a-clamp-written-as-two-conditions": "kb*if(Aobs>0,if(Aobs<thr,Aobs,thr),0)",
    "a-ramp-over-what-is-not-known-nonzero": "if(Aobs<thr,kb*(thr-Aobs)/(1+Cobs),0)",
    "a-condition-scaled-on-both-sides": "if(2*Aobs<2*thr,kb*(thr-Aobs),0)",
}


@pytest.mark.parametrize("case", sorted(BENDS_NOT_PROVED))
def test_a_bend_that_is_not_proved_one_is_refused(tmp_path, case):
    """Refused here, where main is right. Each of these laws is continuous
    where its condition flips, and is not one of the two shapes that is
    proved so. Whether a law bends or jumps is not told by evaluating it: a
    reading that tried took a jump in proportion to ``exp(−2000·B)`` for none
    where B was large, a jump beside a slope of 1e8 for rounding, and a ramp
    to a power of 0 for a ramp. What is not proved is refused."""
    model = _wider(tmp_path, BENDS_NOT_PROVED[case])
    _refused_run(model, ["k", "thr"])


@pytest.mark.parametrize(
    ("law", "proved"),
    [
        ("if(X<thr,kb*(thr-X),0)", True),
        ("if(X<thr,0,kb*(X-thr))", True),
        ("if(X<thr,kb,0)", False),
        ("if(X<thr,kb*(thr-X),kb)", False),
        ("if((X-thr)<0,(-(X-thr))*kb,0)", True),
        ("if(2*X<0,(-X)*kb,0)", True),
        ("if((2*X-1)<0,(-(2*X-1))/max(Y,0.01),0)", True),
        ("if((2*X-1)<0,(-(2*X-1))/max(Y,zero),0)", False),
        ("if(X<thr,X,thr)", True),
        ("if(X>thr,X,thr)", True),
        ("if(X<thr,thr,X)", True),
        ("if(X<thr,X,Y)", False),
        ("if(X<thr&&Y>0,X,thr)", False),
        # A power is 0 with its base where it is a number of 1 or more.
        ("if(X<thr,kb*(thr-X)^2,0)", True),
        ("if(X<thr,kb*(thr-X)^1,0)", True),
        ("if(X<thr,kb*(thr-X)^pos,0)", True),
        ("if(X<thr,kb*(thr-X)^half,0)", False),
        ("if(X<thr,kb*(thr-X)^1.5,0)", False),
        ("if(X<thr,kb*(thr-X)*abs(Y)^1.5,0)", True),
        ("if(X<thr,kb*(thr-X)^zero,0)", False),
        ("if(X<thr,kb*(thr-X)^0.5,0)", False),
        ("if(X<thr,kb*(thr-X)^Y,0)", False),
        ("if(X<thr,kb*sqrt(thr-X),0)", False),
        # A root has no value below 0, and a comparison with no number is
        # false: the law drops from −kb to 0 where X passes thr.
        ("if((sqrt(X-thr)-1)<0,kb*(sqrt(X-thr)-1),0)", False),
        ("if(((X-thr)^0.5-1)<0,kb*((X-thr)^0.5-1),0)", False),
        ("if(X<thr,kb*(thr-X)*sqrt(Y),0)", False),
        ("if(X<thr,kb*(thr-X)*sqrt(Y*Y+pos),0)", True),
        # Both branches 0 where the comparison flips.
        ("if(X>thr,kb*(X-thr),neg*(X-thr))", True),
        ("if(X>thr,kb*(X-thr),-kb*(X-thr))", True),
        ("if(X>thr,kb*(X-thr),neg)", False),
        # A branch that is 0 as a parameter is, or a product with one.
        ("if(X<thr,kb*(thr-X),zero)", True),
        ("if(X<thr,kb*(thr-X),pos*zero)", True),
        # A condition that is a number, true where it is not 0.
        ("if(max(X,thr)-X,kb,0)", False),
        ("if(X<thr&&(Y-1),kb*(thr-X),0)", False),
        # A call that is not known to be continuous.
        ("if(X<thr,kb*(thr-X)*clamp(0,Y,1),0)", False),
        ("if(X<thr,kb*abs(thr-X),0)", False),
        # What the product is of besides: continuous, or not known to be.
        ("if(X<thr,kb*(thr-X)*exp(-Y),0)", True),
        ("if(X<thr,kb*(thr-X)/Y,0)", False),
        ("if(X<thr,kb*(thr-X)/pos,0)", True),
        ("if(X<thr,kb*(thr-X)/zero,0)", False),
        # A sign turns what is known of it: pos less what is below 0 is above 0.
        ("if(X<thr,kb*(thr-X)/(pos-(-exp(Y))),0)", True),
        ("if(X<thr,kb*(thr-X)/(pos-exp(Y)),0)", False),
        ("if(X<thr,kb*(thr-X),0)+kb/Y", False),
        ("if(X<thr,kb*(thr-X)*log(Y),0)", False),
        ("if(X<thr,kb*(thr-X)*log(Y*Y+pos),0)", True),
        ("if(X<thr,kb*(thr-X)*tan(Y),0)", False),
        ("if(X<thr,kb*(thr-X)*Y^neg,0)", False),
        ("if(X<thr,kb*(thr-X)*pow(Y,neg),0)", False),
        ("if(X<thr,kb*(thr-X)*(Y*Y+pos)^neg,0)", True),
        ("if(X<thr,kb*(thr-X)*floor(T),0)", True),
        ("if(X<thr,kb*(thr-X)*floor(Y),0)", False),
        ("if(X<thr,kb*(thr-X)*(Y%2),0)", False),
        # Every factor the comparison is 0 with.
        ("if((X-4.4)*(X-5)<0,kb*(X-4.4)*(5-X),0)", True),
        ("if((X-4.4)*(X-5)<0,kb*(X-4.4),0)", False),
        ("if((X-thr)/pos<0,kb*(thr-X),0)", True),
        ("if(X>0,kb/X,0)", False),
        # Each comparison of a condition, through and, or and not.
        ("if(X>0&&Y>0,kb*X*Y,0)", True),
        ("if(X>0&&Y>0,kb*X,0)", False),
        ("if(X>0||Y>0,kb*X*Y,0)", True),
        ("if(not(X>0),0,kb*X)", True),
        # Every condition in the law, and none on a clock.
        ("if(X<thr,kb*(thr-X),0)*if(Y>1,2,3)", False),
        ("if(X<thr,kb*(thr-X),0)*if(T>1,2,3)", True),
        ("if(X>0,if(X<pos,X,pos),0)", False),
        ("if(if(X<thr,Y-1,1-Y)>0,kb*(if(X<thr,Y-1,1-Y))^2,0)", False),
        ("if(X==thr,kb,0)", False),
        ("if(2*X<2*thr,kb*(thr-X),0)", False),
        ("kb*max(X,0.5)", True),
    ],
)
def test_which_rate_law_is_proved_to_bend(law, proved):
    """What :func:`_only_bends` proves: X and Y are the state, T a clock,
    ``pos``, ``zero`` and ``neg`` parameters above, at and below 0."""
    from bngsim._switch_sensitivity import _only_bends, _syntax_tree

    values = {"kb": 3.0, "thr": 4.4, "pos": 2.0, "zero": 0.0, "neg": -2.0, "half": 0.5}
    assert _only_bends(_syntax_tree(law), values, frozenset({"T"}), set()) is proved


@pytest.mark.parametrize(
    "law",
    ["kb*max(thr-Aobs,0)", "kb*(thr-min(Aobs,thr))", "kb*(abs(thr-Aobs)+(thr-Aobs))/2"],
    ids=["max", "min", "abs"],
)
def test_the_same_bend_written_with_a_choice_runs(tmp_path, law):
    """Control. The ramp from thr written with ``max``, ``min`` and ``abs``:
    continuous by what it is made of, with no condition and no division."""
    sim = _simulator(tmp_path, law, DECLINED, ["k", "thr"], decays=True)
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), _ramp_columns(), rtol=1e-6)


@pytest.mark.parametrize(
    "law",
    ["kb*abs(Aobs-thr)", "if((Aobs-thr)>0,kb*(Aobs-thr),-kb*(Aobs-thr))"],
    ids=["abs", "written-as-a-condition"],
)
def test_a_magnitude_runs(tmp_path, law):
    """Control. ``kb·abs(Aobs − thr)`` is continuous where A is thr, and so is
    the declined law beside it, ``kc·max(Aobs, 0.5)``: bends, which the
    quotient is right across. Written as a condition, both branches are 0
    where it flips. dY/dk in closed form."""
    sim = _simulator(tmp_path, law, DECLINED, ["k"])
    assert not sim.has_analytic_sens_rhs
    k, thr, kb = 1.0, 4.4, 3.0
    t_star = (thr - 1.0) / k
    # dY/dk = kb·(∫ t dt over [t*, T] − ∫ t dt over [0, t*]).
    want = kb * ((T_END**2 - t_star**2) / 2.0 - t_star**2 / 2.0)
    np.testing.assert_allclose(_y_columns(sim), [want], rtol=1e-6)


SIGNS = {
    # 1 above thr and 0 below it, each: Y' = kb from the crossing on.
    "a-ratio-to-its-magnitude": "kb*(1+(Aobs-thr)/abs(Aobs-thr))/2",
    "a-magnitude-over-what-it-is-of": "kb*(1+abs(Aobs-thr)/(Aobs-thr))/2",
    "the-greater-over-the-difference": "kb*(max(Aobs,thr)-thr)/(Aobs-thr)",
    "the-greater-of-it-and-0-over-it": "kb*max(Aobs-thr,0)/(Aobs-thr)",
    "the-lesser-of-it-and-0-over-it": "kb*(1-min(Aobs-thr,0)/(Aobs-thr))",
    "a-negative-power": "kb*(1+abs(Aobs-thr)*(Aobs-thr)^(-1))/2",
    "a-power-that-is-a-parameter": "kb*(1+abs(Aobs-thr)*(Aobs-thr)^c)/2",
    "a-factor-no-column-moves-on-each-side": "kb*(1+(2*(Aobs-thr))/abs(3*(Aobs-thr))*1.5)/2",
    "with-a-cusp-beside-it": "kb*(1+(Aobs-thr)/abs(Aobs-thr)*(1+sqrt(abs(Aobs-thr))))/2",
    "with-a-call-the-scan-does-not-know": "kb*(1+(Aobs-thr)/abs(Aobs-thr))*erf(1)",
    "a-product-of-two": "kb*(Aobs-4.4)*(Aobs-5)/abs((Aobs-4.4)*(Aobs-5))",
    "one-factor-of-a-product": "kb*abs((Aobs-4.4)*(Aobs+5))/(Aobs-4.4)",
}


@pytest.mark.parametrize("case", sorted(SIGNS))
def test_a_sign_written_as_a_quotient_is_refused(tmp_path, case):
    """``(X − thr)/abs(X − thr)`` is −1 below thr and 1 above it, with no
    condition written and no step call: a division by what an ``abs``,
    ``max`` or ``min`` is 0 at, or flips at. dY/dk came back 18.37 for 10.2
    at a loose tolerance, and the run ended in CVODE's no-progress error at a
    tight one."""
    model = _wider(tmp_path, SIGNS[case], decays=False)
    _refused_run(model, ["k"])


def test_a_sign_of_a_counter_nothing_moves_runs(tmp_path):
    """Control. A is made at a rate of 1 and only kb is requested: A is a
    clock, read at the same value in both terms of the quotient, and the
    sign of ``Aobs − thr`` flips at an instant no column moves.
    Y = kb·(T − 3.4)."""
    law = "kb*(1+(Aobs-thr)/abs(Aobs-thr))/2"
    sim = bngsim.Simulator(
        _wider(tmp_path, law, decays=False), method="ode", sensitivity_params=["kb"]
    )
    assert not sim.has_analytic_sens_rhs
    run = sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-8, atol=1e-10, timeout=60)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), 0]
    assert got == pytest.approx(T_END - 3.4, rel=1e-4)


def test_a_sign_of_the_time_that_a_requested_parameter_moves_is_refused(tmp_path):
    """``(time() − tau)/abs(time() − tau)`` with tau requested: nothing stops
    at tau and nothing holds it while the quotient is taken. dY/dtau came
    back −0.656 for −3."""
    law = "kb*0.5*(1+(time()-tau)/abs(time()-tau))"
    _refused_run(_wider(tmp_path, law, decays=False), ["tau"])


CHOICES_THAT_BEND = {
    "a-floor-under-a-divisor": "kb*Aobs/max(Aobs,0.01)",
    "a-magnitude-in-a-sum-it-divides-by": "kb/(1+abs(Aobs-thr))",
    "a-root-of-a-magnitude-in-a-sum": "kb*Aobs/(1+0.5*(Aobs+sqrt(abs(Aobs*Aobs-thr))))",
    "a-clamp": "kb*max(0,min(Aobs,thr))",
    "the-lesser-of-two-rates": "min(kb*Aobs/(thr+Aobs),kb*Aobs/(1+Aobs))",
    "a-positive-power-of-a-magnitude": "kb*abs(thr-Aobs)^1.5",
}


@pytest.mark.parametrize("case", sorted(CHOICES_THAT_BEND))
def test_a_choice_that_only_bends_runs(tmp_path, case):
    """Control. A division beside an ``abs``, ``max`` or ``min`` that is not
    by what it is 0 at: a floor that is a number above 0, a magnitude in a
    sum, a lesser of two quotients. Against a central difference of plain
    runs, Richardson-extrapolated."""

    def build(k):
        return _wider(tmp_path, CHOICES_THAT_BEND[case], k=k)

    sim = bngsim.Simulator(build(1.0), method="ode", sensitivity_params=["k"])
    assert not sim.has_analytic_sens_rhs
    run = sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-10, atol=1e-12, timeout=60)
    got = np.asarray(run.sensitivities)[-1, list(run.species_names).index("Y()"), 0]

    def plain(k):
        out = bngsim.Simulator(build(k), method="ode").run(
            t_span=(0.0, T_END), n_points=3, rtol=1e-12, atol=1e-14, timeout=60
        )
        return np.asarray(out.species)[-1, list(out.species_names).index("Y()")]

    def slope(h):
        return (plain(1.0 + h) - plain(1.0 - h)) / (2.0 * h)

    want = (4.0 * slope(5e-4) - slope(1e-3)) / 3.0
    assert got == pytest.approx(want, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize(
    ("law", "found"),
    [
        ("kb*abs(X-thr)", False),
        ("kb*(X-thr)/abs(X-thr)", True),
        ("kb*abs(X-thr)/(X-thr)", True),
        ("kb*abs(2*(X-thr))/(3*(thr-X))", True),
        ("kb*max(X,thr)", False),
        ("kb*(max(X,thr)-thr)/(X-thr)", True),
        ("kb*(max(X,thr)-thr)/(thr-X)", True),
        ("kb*max(X-thr,0)/(X-thr)", True),
        ("kb*max(0,X-thr)/(X-thr)", True),
        ("kb*min(X,thr,Y)/(X-Y)", True),
        ("kb*min(X,thr,Y)/(X-1)", False),
        ("kb*X/max(X,0.01)", False),
        ("kb*X/max(X,pos)", False),
        ("kb*X/max(X,neg)", True),
        ("kb*X/max(X,-0.01)", True),
        ("kb*X/min(X,-0.01)", False),
        ("kb*X/(pos+abs(X))", False),
        ("kb*X/(neg+abs(X))", False),
        ("kb*X/(abs(X)*pos)", True),
        ("kb*X/(abs(X)+1)^2", False),
        ("kb*X/abs(X)^2", True),
        ("kb*X/sqrt(abs(X))", True),
        ("kb*X*abs(X)^(-1)", True),
        ("kb*X*abs(X)^neg", True),
        ("kb*X*abs(X)^pos", False),
        ("kb*X*abs(X)^Y", True),
        ("kb*X*pow(abs(X),-1)", True),
        ("kb*X/exp(abs(X))", False),
        ("kb*X/(abs(X)*abs(X)+1)", False),
        ("kb*X/(-(1+abs(X)))", False),
        ("kb*abs(X*Y)/Y", True),
        ("kb*abs(X*Y)/(X+Y)", False),
        # On the time alone a choice flips at an instant no column moves.
        ("kb*X*(T-3)/abs(T-3)", False),
        # And with a requested parameter beside the time, one does.
        ("kb*X*(T-asked)/abs(T-asked)", True),
        ("kb*X*(T-pos)/abs(T-pos)", False),
        # A parameter alone is no state.
        ("kb*X*(pos-3)/abs(pos-3)", False),
        # A sum that the choice is in is not asked about.
        ("kb*X/(abs(X)+0*Y)", False),
        # A choice this does not know: any division by what may be 0.
        ("kb*clamp(0,X-thr,1)/(X-thr)", True),
        ("kb*clamp(0,X-thr,1)/(X*X+pos)", False),
        ("kb*clamp(0,X-thr,1)", False),
        # A conditional the law divides by is 0 where a branch of it is.
        ("kb*(1+(X-thr)/if(T>0.2,abs(X-thr),1))", True),
        ("kb*(1+(X-thr)/if(pos>0.5,abs(X-thr),1))", True),
        ("kb*(1+(X-thr)/if(T>0.2,1+abs(X-thr),1))", False),
        # A choice under a root, a logarithm or an exponential, with no sum
        # over it.
        ("kb*X/sqrt(abs(X))", True),
        ("kb*X/sqrt(1+abs(X))", False),
        ("kb*X/tanh(abs(X))", True),
        # exp of what has no value is 0, and so is a power of it.
        ("kb*(1+X/exp(log(abs(X))))", True),
        ("kb*(1+X/2^log2(abs(X)))", True),
        ("kb*(1+X/exp(-abs(X)))", False),
        # The greater of e and −e is its magnitude.
        ("kb*(1+max(X-thr,-(X-thr))/(X-thr))", True),
        ("kb*(1+max(-(X-thr),X-thr)/(X-thr))", True),
        # 0 as a parameter is, or a product with one.
        ("kb*max(X-thr,zero)/(X-thr)", True),
        ("kb*max(X-thr,0*kb)/(X-thr)", True),
        ("kb*X/max(X,zero)", True),
        # Over parameters alone: a bend the quotient straddles for the whole
        # run where a requested one is within a quarter of itself of it.
        ("kb*X*min(pos,near)", True),
        ("kb*X*min(pos,asked)", False),
        ("kb*X*abs(near-pos)", True),
        ("kb*X*max(kb,pos)", False),
        ("kb*X*max(thr,near-pos)", False),
        # A power whose sign is what its parameters make it.
        ("kb*X*abs(X)^(asked-1)", False),
        ("kb*X*abs(X)^(pos-3)", True),
        # A requested parameter is moved as the state is.
        ("kb*X*(asked-3)/abs(asked-3)", True),
        ("kb*X/abs(asked)", False),
        ("kb*X*abs(X)^(Y*Y)", False),
        ("kb*X/max(Y*Y+pos,X)", False),
        ("kb*X/max(Y*X+pos,X)", True),
        # Nothing is assumed of the state: X + pos is not known to be above 0.
        ("kb*Y/max(X+pos,0)", True),
        ("kb*Y/max(pos-X,0)", True),
        ("kb*Y/max(pos*X,0)", True),
        ("kb*Y/max(pos/X,0)", True),
        ("kb*Y/max(X^3,0)", True),
        ("kb*X/max(Y^3+pos,X)", True),
        ("kb*X/min(X,pos)", True),
        ("kb*X/max(-pos,X)", True),
        ("kb*X/max(if(Y>1,pos,X),X)", True),
        # What is known above 0: an even power, a magnitude and an
        # exponential beside a number above 0, and a choice between two such.
        ("kb*X/max(Y^2+pos,X)", False),
        ("kb*X/max(pos+abs(Y),X)", False),
        ("kb*X/max(exp(Y),X)", False),
        ("kb*X/max(if(Y>1,pos,2),X)", False),
        # Through the numerator of a quotient, and through a sign.
        ("kb*X/(abs(X)/pos)", True),
        ("kb*X/(-abs(X))", True),
        # Through a root: a step written as the greater of e and 0 over the
        # root of e squared.
        ("kb*max(X-thr,0)/sqrt((X-thr)*(X-thr))", True),
        ("kb*max(X-thr,0)/sqrt((X-thr)*(X-thr)+pos)", False),
    ],
)
def test_which_quotient_is_named(law, found):
    """What :func:`_quotient_across_a_choice` names: X and Y are the state,
    T a clock, ``pos`` and ``asked`` parameters above 0 and ``neg`` one
    below, with ``asked`` requested."""
    from bngsim._switch_sensitivity import _quotient_across_a_choice, _syntax_tree

    values = {"kb": 3.0, "thr": 4.4, "pos": 2.0, "neg": -2.0, "asked": 3.0, "zero": 0.0}
    values["near"] = 2.001
    got = _quotient_across_a_choice(_syntax_tree(law), values, frozenset({"T"}), {"asked", "near"})
    assert (got is not None) is found


def test_a_long_rate_law_is_read(tmp_path):
    """Control. 3,000 terms in a sum, with a ``max`` among them: read without
    going as deep as the law is long."""
    from bngsim._switch_sensitivity import _quotient_across_a_choice, _syntax_tree

    law = "+".join(["kb*max(X,0.5)"] + [f"X/({i}+thr)" for i in range(1, 3000)])
    tree = _syntax_tree(law)
    assert tree is not None
    assert _quotient_across_a_choice(tree, {"kb": 3.0, "thr": 4.4}, frozenset(), set()) is None
    sign = law + "+X/abs(X)"
    found = _quotient_across_a_choice(
        _syntax_tree(sign), {"kb": 3.0, "thr": 4.4}, frozenset(), set()
    )
    assert found == "X / abs(X)"


def test_a_scan_that_fails_refuses(tmp_path, monkeypatch):
    """A rate law that cannot be read for a crossing is not let through."""
    from bngsim import _switch_sensitivity

    def broken(*args, **kwargs):
        raise RuntimeError("no scope")

    sim = _simulator(tmp_path, "kb", DECLINED, ["k"])
    monkeypatch.setattr(_switch_sensitivity, "fallback_crossing", broken)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="could not be read"):
        sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-6, atol=1e-8, timeout=20)


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


@pytest.mark.parametrize(
    ("fy", "fz", "params"),
    [("if(Aobs>thr,kb,0)", DECLINED, ["k", "thr"]), ("kb", "kc*floor(time()/P)", ["P", "kc"])],
    ids=["a-state-switch", "a-step-of-time-the-period-moves"],
)
def test_every_column_at_once_is_refused_by_its_own_columns(tmp_path, fy, fz, params):
    """``compute_all_sensitivities(params=...)`` on a simulator built with no
    columns asked the refusal about the simulator's columns, none, and not
    its own: no counter was moved and no parameter requested. The model of
    issue #938 returned 14.52 for 10.2 through it."""
    sim = bngsim.Simulator(_wider(tmp_path, fy, fz, decays=False), method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.compute_all_sensitivities(
            t_span=(0.0, T_END), n_points=3, params=params, rtol=1e-4, atol=1e-6
        )


def test_one_simulator_is_asked_again_for_other_columns(tmp_path):
    """The sign of ``Aobs − thr`` with A a counter: with kb requested no
    column moves A, and the law's answer is kept. Asked next for k, which
    moves A, the simulator gave the answer it had kept."""
    law = "kb*(1+(Aobs-thr)/abs(Aobs-thr))/2"
    sim = bngsim.Simulator(_wider(tmp_path, law, decays=False), method="ode")
    sim.compute_all_sensitivities(
        t_span=(0.0, T_END), n_points=3, params=["kb"], rtol=1e-4, atol=1e-6
    )
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.compute_all_sensitivities(
            t_span=(0.0, T_END), n_points=3, params=["k"], rtol=1e-4, atol=1e-6
        )


@pytest.mark.parametrize(
    ("fy", "fz", "params"),
    [
        ("kb", "kc*floor(Cobs/P)", ["P", "kc"]),
        ("if(floor(Cobs/P)>2,kb,0)", DECLINED, ["P"]),
    ],
    ids=["outside-a-condition", "in-a-condition"],
)
def test_a_step_of_a_counter_nothing_moves_is_a_step_of_the_clock(tmp_path, fy, fz, params):
    """``floor(Cobs/P)`` with C a counter no column moves steps where
    ``floor(time()/P)`` does, and its steps move with P: dZ/dP came back
    −56.72 for −50, and −16.17 for −9 with the step in a condition."""
    _refused_run(_wider(tmp_path, fy, fz, decays=False), params)


def test_a_step_table_indexed_through_a_function_is_refused(tmp_path):
    """``fIdx() = Aobs*1`` owns a parameter slot, and a table indexed by it was
    taken as indexed by a parameter: dY/dk = 0 for −3.698."""
    table = 'tfun([0,2,4.4,8],[0,1,3,5],fIdx,method=>"step")'
    _refused_run(_wider(tmp_path, table, "kc*Aobs", extra=["fIdx() Aobs*1"]), ["k"])


@pytest.mark.parametrize(
    ("law", "n"),
    [
        # 0 for the first seven units of time.
        ("if(Aobs<thr,kb*floor(time()/7),0)", 0.0),
        # 2 at n = 5, and 0 at a value of n between 0.5 and 2.
        ("if(Aobs<thr,kb*floor(n/2),0)", 5.0),
        # A jump of kb at n = 0, and a ramp at n = 1.
        ("if(Aobs<thr,kb*(thr-Aobs)^n,0)", 0.0),
        # Most of kb at g = 1e-4, and nothing at g between 0.5 and 2.
        ("if(Aobs<thr,kb*exp(-2000*g),0)", 0.0),
        # A second flip, at Aobs = 2, for c = −4 and for no c above 0.
        ("if((Aobs-thr)*(Aobs*Aobs+c)<0,kb*(thr-Aobs),0)", 0.0),
        # Most of kb where B is, at 1e-4, and nothing for B between 0.5 and 2.
        ("if(Aobs<thr,kb*exp(-2000*Bobs),0)", 0.0),
    ],
    ids=[
        "a-step-of-time",
        "a-step-of-a-parameter",
        "an-exponent-of-0",
        "a-factor",
        "a-flip",
        "a-factor-of-the-state",
    ],
)
def test_a_jump_that_goes_with_what_the_parameters_are_is_refused(tmp_path, law, n):
    """Each of these laws bends where A is thr for parameters, a time or a
    state between 0.5 and 2, and jumps at the values the model has. A reading
    that put every symbol between 0.5 and 2 found each continuous, and one
    that read parameters at their values and the state near where it is
    found the last so too."""
    _refused_run(_wider(tmp_path, law, n=n), ["k", "thr"])


def test_the_refusal_is_asked_again_when_a_parameter_changes(tmp_path):
    """A is made at rate k: at k = 1 it is a counter, which no requested
    column moves, and its switch at thr is jumped (issue #48). At k = 2 it is
    a state like any other. A simulator that had run at k = 1 kept the first
    answer and returned dY/dthr = −4.228 for −1.5."""
    model = _wider(tmp_path, "if(Aobs>thr,kb,0)", decays=False)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr"])
    sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-4, atol=1e-6, timeout=20)
    model.set_param("k", 2.0)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-4, atol=1e-6, timeout=20)


@pytest.mark.parametrize("depth", [30, 70])
def test_a_condition_behind_many_assignment_rules_is_refused(depth):
    """A jump reached through a chain of assignment rules. Past 64 of them
    the law is not written out, was read as its bare name, and held no
    condition: dB/da = 40.72 for 14.45."""
    text = (
        "species A, B, Z; A = 10; B = 0; Z = 0; a = 0.5; q = 3; kb = 3;\n"
        "J0: A -> ; a*A\nJ2: -> Z; max(A, 0.1)\nf0 := piecewise(kb, A<q, 0)\n"
    )
    text += "".join(f"f{i} := f{i - 1} + 0\n" for i in range(1, depth))
    model = bngsim.Model.from_antimony_string(text + f"J1: -> B; f{depth - 1}\n")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["a", "q"])
    with pytest.raises(bngsim.SensitivityUnsupportedError):
        sim.run(sample_times=[0.0, 3.0, 6.0], rtol=1e-4, atol=1e-6, timeout=20)


@pytest.mark.parametrize(
    "law",
    [
        "if(max(Aobs-thr,0)==0,kb,0)",
        "if(if(Aobs<thr,1,0)==1,kb,0)",
        "if(min(Aobs,thr)==Aobs,kb,0)",
        "if(abs(Aobs-thr)+(thr-Aobs)!=0,kb,0)",
    ],
    ids=["a-max", "a-condition", "a-min", "an-abs"],
)
def test_an_equality_that_holds_over_an_interval_is_refused(tmp_path, law):
    """``max(Aobs − thr, 0) == 0`` holds for every A up to thr, where an
    equality between two smooth expressions holds at one value. Taken for
    one of those, the model ran: dY/dk = 4.424 and dY/dthr = 1.225 for 2.463
    and 0.682."""
    sim = _simulator(tmp_path, law, DECLINED, ["k", "thr"], decays=True)
    _refused(sim, "Aobs", rtol=1e-4)


BELOW_ZERO = {
    # x = cos(w·t) passes 0, and −0.5, again and again.
    "a-rate-rule-through-0": (
        "x = 1; y = 0; Y = 0; Z = 0; w = 1; kb = 3\nx' = w*y\ny' = -w*x\n"
        "Y' = piecewise(kb, x > 0, 0)\nZ' = max(x, -2)\n",
        ["w"],
    ),
    "a-rate-rule-through-a-number-below-0": (
        "x = 1; y = 0; Y = 0; Z = 0; w = 1; kb = 3; c = 0.5\nx' = w*y\ny' = -w*x\n"
        "Y' = piecewise(kb, x > -c, 0)\nZ' = max(x, -2)\n",
        ["w"],
    ),
    "a-sign-of-what-passes-below-0": (
        "x = 1; y = 0; Y = 0; Z = 0; w = 1; kb = 3; c = 0.5\nx' = w*y\ny' = -w*x\n"
        "Y' = kb*(1 + (x + c)/abs(x + c))\nZ' = max(x, -2)\n",
        ["w"],
    ),
    # A falls at a constant rate and passes 0 at t = 3.
    "a-species-consumed-at-a-constant-rate": (
        "species A, Y, Z; A = 3; Y = 0; Z = 0; k0 = 1; kb = 3\nJ0: A -> ; k0\n"
        "J1: -> Y; piecewise(kb, A > 0, 0)\nJ2: -> Z; max(A, -5)\n",
        ["k0"],
    ),
}


@pytest.mark.parametrize("case", sorted(BELOW_ZERO))
def test_a_condition_on_what_goes_below_zero_is_refused(case):
    """``x > 0`` was taken to hold for good where x starts above 0, as a
    guard on a concentration does. A variable under a rate rule, and a
    species consumed at a constant rate, pass 0: dY/dw came back 14.13 for
    9.42, and dY/dk0 −13.49 for −9. Nothing is assumed of the state."""
    text, params = BELOW_ZERO[case]
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=params
    )
    assert not sim.has_analytic_sens_rhs
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run(sample_times=[0.0, 3.0, 6.0], rtol=1e-8, atol=1e-10, timeout=20)


def test_a_batch_row_is_asked_about_with_its_own_parameters(tmp_path):
    """A is made at rate k: a counter at k = 1, which the one requested
    column, thr, does not move, and a state like any other at k = 2. The
    batch was asked about once, at the model's own k = 1, and the row at
    k = 2 returned dY/dthr = −4.228 for −1.5."""
    model = _wider(tmp_path, "if(Aobs>thr,kb,0)", decays=False)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr"])
    assert not sim.has_analytic_sens_rhs
    rows = sim.run_batch(
        params=[{"k": 1.0}], t_span=(0.0, T_END), n_points=3, rtol=1e-8, atol=1e-10
    )
    got = np.asarray(rows[0].sensitivities)[-1, list(rows[0].species_names).index("Y()"), 0]
    assert got == pytest.approx(-3.0, rel=1e-6)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run_batch(params=[{"k": 2.0}], t_span=(0.0, T_END), n_points=3, rtol=1e-4, atol=1e-6)


def test_a_divisor_is_asked_about_again_when_a_parameter_changes_sign(tmp_path):
    """``kb/max(thr − Aobs, g)``: a floor under the divisor at g = 1e-4, and
    a division by what is 0 wherever A is above thr at g = 0."""
    model = _wider(tmp_path, "kb/max(thr-Aobs,g)")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["kb"])
    assert not sim.has_analytic_sens_rhs
    sim.run(t_span=(0.0, 1.0), n_points=3, rtol=1e-6, atol=1e-8, timeout=20)
    model.set_param("g", 0.0)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run(t_span=(0.0, 1.0), n_points=3, rtol=1e-6, atol=1e-8, timeout=20)


@pytest.mark.parametrize("n", [1.0 + 1e-6, 1.0, 1.0 - 1e-6, 1.001, 0.9])
def test_a_condition_a_requested_parameter_is_near_flipping_is_refused(tmp_path, n):
    """``if(n > 1, kb, 0)`` holds one way for the whole run, and is a surface
    in n. The quotient reads the law at n moved by up to its size times the
    root of the tolerance, either way, and near 1 one of the two is on the
    other branch: dY/dn came back 901 at n = 1 + 1e-6, 1.8e10 at 1, 2.0 at
    1 − 1e-6 and 0.9 at 1.001, for 0."""
    _refused_run(_wider(tmp_path, "if(n>1,kb,0)", n=n), ["n", "k"])


@pytest.mark.parametrize(("n", "params"), [(1.5, ["n", "k"]), (0.5, ["n", "k"]), (1.0, ["k"])])
def test_a_condition_on_a_parameter_far_from_flipping_runs(tmp_path, n, params):
    """Control. n half of itself from where the comparison flips, and n on
    it with only k requested: nothing is moved across. Y is kb·T or 0
    whatever k is."""
    sim = bngsim.Simulator(
        _wider(tmp_path, "if(n>1,kb,0)", n=n), method="ode", sensitivity_params=params
    )
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [0.0] * len(params), atol=1e-9)


@pytest.mark.parametrize(
    ("law", "n"),
    [
        ("if(n==1,kb,0)", 1.0),
        ("kb*floor(n)", 5.0),
        ("kb*(1+(n-1)/abs(n-1))/2", 1.0 + 1e-6),
        ('tfun([0,2,4.4,8],[0,1,3,5],n,method=>"step")', 2.0),
    ],
    ids=["an-equality-that-holds", "a-step", "a-sign", "a-step-table"],
)
def test_a_jump_in_a_requested_parameter_is_refused(tmp_path, law, n):
    """The same in an equality that holds where n is, a step call on n, a
    sign of ``n − 1`` written as a quotient and a step table indexed by n."""
    _refused_run(_wider(tmp_path, law, n=n), ["n", "k"])


@pytest.mark.parametrize(
    ("law", "param", "value"),
    [("if((n-1)*(n-1.1)<0,kb,0)", "n", 0.95), ("if(g>0.1,kb,0)", "g", 0.0)],
    ids=["a-window-a-tenth-wide", "a-parameter-at-0"],
)
def test_how_far_a_requested_parameter_is_moved(tmp_path, law, param, value):
    """Refused here, where main is right at an ordinary tolerance. The
    comparison is asked about with the parameter moved a quarter of itself
    and by halves of that down, either way, which is the quotient's reach at
    a relative tolerance of 0.06: a window from 1 to 1.1 is found from
    n = 0.95, though it is the same on both sides of it. A parameter at 0 is
    moved by a quarter."""
    model = _wider(tmp_path, law, n=0.95)
    model.set_param(param, value)
    _refused_run(model, [param, "k"])


def test_an_equality_on_a_parameter_that_does_not_hold_runs(tmp_path):
    """Control. ``n == 1`` at n = 1.1 holds at one value of n, which the
    quotient does not read at."""
    sim = bngsim.Simulator(
        _wider(tmp_path, "if(n==1,kb,0)", n=1.1), method="ode", sensitivity_params=["n", "k"]
    )
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [0.0, 0.0], atol=1e-9)


def test_a_rate_law_that_does_not_parse_is_refused(tmp_path, monkeypatch):
    """A rate law with an ``abs``, ``max`` or ``min`` in it that is not read
    is one whose quotients are not known."""
    from bngsim import _switch_sensitivity

    sim = _simulator(tmp_path, "kb*abs(Aobs-thr)", DECLINED, ["k"])
    monkeypatch.setattr(_switch_sensitivity, "_syntax_tree", lambda expr: None)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="not read"):
        sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-6, atol=1e-8, timeout=20)


def test_a_bend_is_proved_again_when_a_parameter_changes_sign(tmp_path):
    """``if(Aobs < thr, (thr − Aobs)·kb/max(Aobs, g), 0)``: a ramp over a
    floor at g = 1e-4, and over what may be 0 at g = 0, where it is not
    proved a bend."""
    model = _wider(tmp_path, "if(Aobs<thr,(thr-Aobs)*kb/max(Aobs,g),0)")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    assert not sim.has_analytic_sens_rhs
    sim.run(t_span=(0.0, 1.0), n_points=3, rtol=1e-6, atol=1e-8, timeout=20)
    model.set_param("g", 0.0)
    model.reset()
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run(t_span=(0.0, 1.0), n_points=3, rtol=1e-6, atol=1e-8, timeout=20)


def test_what_is_kept_between_runs_does_not_grow(tmp_path):
    """What is made of each rate law's text, and the last answer for it,
    whatever the parameters are set to: a fit that changes them ten thousand
    times keeps what it kept after the first run."""
    model = _wider(tmp_path, "kb*abs(Aobs-thr)", decays=False)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    sizes = set()
    for kb in (3.0, 3.5, 4.0, 4.5, -1.0):
        model.set_param("kb", kb)
        model.reset()
        sim.run(t_span=(0.0, 1.0), n_points=3, rtol=1e-6, atol=1e-8, timeout=20)
        sizes.add(len(sim._fallback_scan_cache))
    assert len(sizes) == 1 and sizes.pop() <= 4


def test_a_branch_scan_that_fails_refuses(tmp_path, monkeypatch):
    """The scan for a branch crossing whose time moves (issue #414) comes
    before this one, and where it failed the model was run: nothing after it
    was asked."""
    from bngsim import _switch_sensitivity

    def broken(*args, **kwargs):
        raise RuntimeError("no scope")

    sim = _simulator(tmp_path, "if(Aobs>thr,kb,0)", DECLINED, ["k"])
    monkeypatch.setattr(_switch_sensitivity, "model_uncompensated_crossing_reason", broken)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="could not be read"):
        sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-6, atol=1e-8, timeout=20)


def test_an_equality_that_holds_a_step_holds_over_an_interval(tmp_path):
    """``if(floor(Aobs/thr) == 0, kb, 0)`` with nothing declined, so on the
    analytic path: kb while A is under thr. An equality over the state was
    taken to hold on no interval, and with a step call in it it holds on one:
    the columns came back [0, 0] for [2.463, 0.682], with nothing logged. It
    is not admitted, and the run is refused (issue #414)."""
    sim = _simulator(tmp_path, "if(floor(Aobs/thr)==0,kb,0)", "kc*Aobs", ["k", "thr"], True)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="issue #414"):
        sim.run(t_span=(0.0, T_END), n_points=3, rtol=1e-8, atol=1e-10, timeout=20)


def test_an_equality_on_the_state_is_refused(tmp_path):
    """Refused here, where main is right. ``if(Aobs == 4.4, kb, 2*kb)`` holds
    at one value of A, which A passes through, and dY/dk is 0. An equality
    that the state sits on is not told from one it passes through."""
    sim = _simulator(tmp_path, "if(Aobs==4.4,kb,2*kb)", DECLINED, ["k"], True)
    _refused(sim, "Aobs==4.4", rtol=1e-4)


def test_an_equality_the_state_sits_on_is_refused():
    """B is 0 until A falls past q, and is made from there: ``B == 0`` holds
    over an interval, and stops holding where A passes q. Taken to hold at
    one state and nowhere else, the model ran: dY/da = 0 and dY/dq = 0 for
    −14.45 and −2."""
    text = (
        "species A, B, Y; A = 10; B = 0; Y = 0; a = 0.5; q = 3; kb = 3; kp = 1\n"
        "J0: A -> ; a*A\nJB: -> B; kp*max(q - A, 0)\nJ1: -> Y; piecewise(kb, B == 0, 0)\n"
    )
    sim = bngsim.Simulator(
        bngsim.Model.from_antimony_string(text), method="ode", sensitivity_params=["a", "q"]
    )
    assert not sim.has_analytic_sens_rhs
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
        sim.run(sample_times=[0.0, 3.0, 6.0], rtol=1e-8, atol=1e-10, timeout=20)


MORE_SIGNS = {
    # 1 above thr and 0 below it: dY/dk came back −2.94 to 8.84 for 2.46 or 4.93.
    "a-clamp-over-what-it-is-of": "kb*clamp(0,thr-Aobs,1)/(thr-Aobs)",
    "a-magnitude-in-a-conditional-on-the-time": (
        "kb*(1+(thr-Aobs)/if(time()>0.2,abs(thr-Aobs),1))"
    ),
    "a-magnitude-in-a-conditional-on-a-parameter": (
        "kb*(1+(thr-Aobs)/if(one>0.5,abs(thr-Aobs),1))"
    ),
    "a-magnitude-through-a-logarithm": "kb*(1+(thr-Aobs)/exp(log(abs(thr-Aobs))))",
    "the-greater-of-it-and-its-negative": "kb*(1+max(thr-Aobs,-(thr-Aobs))/(thr-Aobs))",
    "the-greater-of-it-and-a-parameter-at-0": "kb*max(thr-Aobs,n)/(thr-Aobs)",
}


@pytest.mark.parametrize("case", sorted(MORE_SIGNS))
def test_a_sign_written_as_a_quotient_is_refused_however_the_choice_is_written(tmp_path, case):
    """The same sign with its choice written as a ``clamp``, inside a
    conditional on the time or on a parameter, through ``exp(log(abs(e)))``,
    as ``max(e, −e)``, and against a parameter that is 0."""
    _refused_run(_wider(tmp_path, MORE_SIGNS[case]), ["k", "thr"])


def test_a_step_table_indexed_by_what_is_made_of_a_requested_parameter_is_refused(tmp_path):
    """``idx = 2*n`` with n requested and the table's index on a step edge,
    4.4004: the table named a requested parameter and not one made of it.
    dY/dn came back 18 for 0."""
    table = 'tfun([0,2,4.4,8],[0,1,3,5],idx,method=>"step")'
    _refused_run(_wider(tmp_path, table, n=2.2002), ["n"])


def test_a_bend_in_a_requested_parameter_that_is_close_is_refused(tmp_path):
    """``min(one, near)·Aobs`` with near = 1.001: one is the lesser, and the
    column for near is 0. The quotient moves near by more than a thousandth
    at rtol 1e-4 and reads the other side of the bend for the whole run:
    dY/d(near) came back 1.41 for 0."""
    _refused_run(_wider(tmp_path, "kb*min(one,near)*Aobs"), ["near", "k"])


def test_a_bend_in_a_parameter_that_is_far_runs(tmp_path):
    """Control. ``min(one, tau)·Aobs`` with tau = 3.4: the column for tau is 0."""
    model = _wider(tmp_path, "kb*min(one,tau)*Aobs")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["tau"])
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [0.0], atol=1e-9)


def test_a_root_past_the_edge_of_where_it_has_a_value_is_not_proved_a_bend(tmp_path):
    """``if((sqrt(Aobs−thr)−1) < 0, kb·(sqrt(Aobs−thr)−1), 0)``: one branch 0
    and the other the difference the condition compares. Below thr the root
    has no value, the comparison is false, and the law drops from −kb to 0.
    Proved a bend, it ran: dY/dk = 0.2687 and dY/dthr = 0.0170 for 0.2133
    and 0.0456."""
    law = "if((sqrt(Aobs-thr)-1)<0,kb*(sqrt(Aobs-thr)-1),0)"
    _refused_run(_wider(tmp_path, law), ["k", "thr"])


def test_a_batch_row_s_refusal_is_a_refusal(tmp_path):
    """A row that is refused raises what ``run`` raises for its model, in a
    batch run in threads as in one run in order."""
    model = _wider(tmp_path, "if(Aobs>thr,kb,0)", decays=False)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["thr"])
    for workers in (None, 2):
        with pytest.raises(bngsim.SensitivityUnsupportedError, match="#938"):
            sim.run_batch(
                params=[{"k": 2.0}, {"k": 3.0}],
                t_span=(0.0, T_END),
                n_points=3,
                rtol=1e-4,
                atol=1e-6,
                num_processors=workers,
            )
