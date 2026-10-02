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


def test_a_jump_no_requested_column_moves_is_refused_all_the_same(tmp_path):
    """kb and kc move neither A nor thr, so neither column reads across the
    surface, and main returns them: dY/dkb = T − t* and dY/dkc = 0. Whether a
    column moves the crossing is not known before the run, and a model whose
    rate law jumps where the state crosses is refused on the quotient whichever
    it is."""
    sim = _simulator(tmp_path, "if(Aobs<thr,kb,0)", DECLINED, ["kb", "kc"], decays=True)
    _refused(sim, "Aobs<thr")


def test_a_counter_threshold_the_rate_law_does_not_jump_at_runs(tmp_path):
    """Control. The rate law turns on as a ramp from the counter's threshold:
    Y = kb·k·(T − t*)²/2 with t* = (thr − A0)/k, so
    dY/dk = kb·(T − t*)²/2 + kb·(T − t*)·t* = 36.66. A bend, which the
    quotient reads across without harm."""
    sim = _simulator(tmp_path, "if(Aobs>4.4,kb*(Aobs-4.4),0)", DECLINED, ["k"])
    assert not sim.has_analytic_sens_rhs
    np.testing.assert_allclose(_y_columns(sim), [36.66], rtol=1e-6)


@pytest.mark.parametrize(
    "law",
    ["if(Aobs<thr,kb*(thr-Aobs),0)", "if((Aobs-thr)<0,(-(Aobs-thr))*kb,0)"],
    ids=["ramp", "signed-rate"],
)
def test_a_crossing_that_does_not_jump_runs_on_the_difference_quotient(tmp_path, law):
    """Control. A decays through thr and the rate law turns on from 0 there, as
    a ramp, or in the signed-rate idiom. The rate law is continuous wherever
    its condition flips, which is asked of it before the run, at three points
    of the condition's surface: there is no jump for the quotient to straddle,
    and the columns are right on it."""
    sim = _simulator(tmp_path, law, DECLINED, ["k", "thr"], decays=True)
    assert not sim.has_analytic_sens_rhs
    a0, k, thr, kb = 10.0, 1.0, 4.4, 3.0
    t_star = math.log(a0 / thr) / k
    tail = math.exp(-k * T_END)
    want = [
        kb * (thr * t_star / k + thr / k**2 - a0 * T_END * tail / k - a0 * tail / k**2),
        kb * (T_END - t_star),
    ]
    np.testing.assert_allclose(_y_columns(sim), want, rtol=1e-6)


@pytest.mark.parametrize(
    ("law", "atom", "continuous"),
    [
        ("if((k*(thr-Aobs))<0,(-(k*(thr-Aobs)))/max(Bobs,0.01),0)", "(k*(thr-Aobs))<0", True),
        ("if(Aobs<thr,kb*(thr-Aobs)^2,0)", "Aobs<thr", True),
        ("if(Aobs<thr,kb*(thr-Aobs)+1e6*Aobs,1e6*Aobs)", "Aobs<thr", True),
        ("if((Aobs>thr)&&(Bobs<2),kb*(Aobs-thr),0)", "Aobs>thr", True),
        # The same law jumps where its other comparison flips.
        ("if((Aobs>thr)&&(Bobs<2),kb*(Aobs-thr),0)", "Bobs<2", False),
        # A jump in proportion to another species: nothing where that one is 0.
        ("if(Aobs<thr,kb*Bobs,0)", "Aobs<thr", False),
        # In proportion to a symbol the condition itself reads.
        ("if(k*(Aobs-thr)<0,k*kb,0)", "k*(Aobs-thr)<0", False),
        # A jump of a thousandth beside a term that moves by 1.3 over the hair.
        ("if(Aobs<thr,kb,0)*1e-3+1e6*Aobs", "Aobs<thr", False),
        # Continuous and not a bend: its slope has no bound at the surface.
        ("if(Aobs<thr,kb*sqrt(thr-Aobs),0)", "Aobs<thr", False),
        ("if(rateOf(A)>-thr,kb,0)", "rateOf(A)>-thr", False),
        ("kc*if(floor(Aobs/P)>2,1,2)", "floor(Aobs/P)>2", False),
        # The branch not taken has no value a hair below the threshold: its
        # square root is of a negative number.
        ("if(Aobs<thr,0,kb*(Aobs-thr)*sqrt(Aobs-thr+1e-9))", "Aobs<thr", True),
        # The branch taken is: no value there, and nothing to compare.
        ("if(Aobs<thr,kb*(thr-Aobs)*sqrt(Aobs-thr),0)", "Aobs<thr", False),
        # A parameter named with a Python keyword.
        ("if(lambda<thr,kb*(thr-lambda),0)", "lambda<thr", True),
        ("if(lambda<thr,kb,0)", "lambda<thr", False),
        # A call and an operator that are not read as doubles.
        ("if(Aobs<thr,kb*(thr-Aobs)*mratio(1,2,Aobs),0)", "Aobs<thr", False),
        ("if(Aobs<thr,kb*(thr-Aobs)*(Bobs%2),0)", "Aobs<thr", False),
        # A saturation whose half-point is three hairs from the surface: not
        # a ramp over the hair, and a bend.
        ("if(Aobs<thr,kb*(thr-Aobs)/((thr-Aobs)+3e-6),0)", "Aobs<thr", True),
        # A jump ten times what the ramp beside it does over a hair, and one
        # a thousandth of it: the second is inside what the law does between
        # the two readings it is carried to the flip from, and is not seen.
        ("if(Aobs<thr,kb*(thr-Aobs)+1e-5,0)", "Aobs<thr", False),
        ("if(Aobs<thr,kb*(thr-Aobs)+1e-9,0)", "Aobs<thr", True),
        # Powers under 1: 0.9 leaves the surface slowly enough, 0.2 does not.
        ("if(Aobs<thr,kb*(thr-Aobs)^0.9,0)", "Aobs<thr", True),
        ("if(Aobs<thr,kb*(thr-Aobs)^0.2,0)", "Aobs<thr", False),
        # A jump in proportion to something that is 0 at the values picked.
        ("if(Aobs<thr,kb*max(Bobs-3,0),0)", "Aobs<thr", False),
        ("if(Aobs<thr,kb*if(Bobs>5,1,0),0)", "Aobs<thr", False),
        # A ramp that is clamped is a bend, and is not found one: the clamp
        # is taken as a symbol of its own.
        ("if(Aobs<thr,min(kb*(thr-Aobs),cap),0)", "Aobs<thr", False),
        # The lesser of two written as a conditional.
        ("kb*if(Aobs<Bobs,Aobs,Bobs)", "Aobs<Bobs", True),
        # The atom under a not: the branches change places.
        ("if(not(Aobs>=thr),kb*(thr-Aobs),0)", "Aobs>=thr", True),
        ("if(not(Aobs>=thr),kb,0)", "Aobs>=thr", False),
        # A step call in the comparison: it flips where the step does.
        ("if(floor(Aobs/P)>2,kb*(floor(Aobs/P)-2),0)", "floor(Aobs/P)>2", False),
        # The greater of two written as a conditional, inside the comparison
        # of a signed rate: the outer condition holds the inner atom only
        # through the inner conditional's value.
        (
            "if((g*if(QR>QL,QR,QL)-QR)>0,(g*if(QR>QL,QR,QL)-QR),0)",
            "QR>QL",
            True,
        ),
        (
            "if((g*if(QR>QL,QR,2*QL)-QR)>0,(g*if(QR>QL,QR,2*QL)-QR),0)",
            "QR>QL",
            False,
        ),
        # The law is 0/0 on the flip itself, at that one double, and has a
        # value either side of it.
        (
            "if(X<x0,exp(sp*((x0-X)/x0)^2),exp(sn*((X-x0)/x0)^2))*exp(((X-x0)/abs(X-x0))*h*((X-x0)/x0)^2)",
            "X<x0",
            True,
        ),
        (
            "if(X<x0,exp(sp*((x0-X)/x0)^2),2*exp(sn*((X-x0)/x0)^2))*exp(((X-x0)/abs(X-x0))*h*((X-x0)/x0)^2)",
            "X<x0",
            False,
        ),
        # A clamp written as two conditionals: a bend, and not found one. The
        # inner choice is free where the outer condition flips.
        ("if(X>0,if(X<n,X,n),0)", "X>0", False),
        # Either of two conditions: the first flips to no effect where the
        # second holds, and where it does not the branches meet.
        ("if((Aobs>thr)||(Bobs<2),kb,kb*(1+(Aobs-thr)))", "Aobs>thr", True),
        ("if((Aobs>thr)||(Bobs<2),kb,kb*(1+(Aobs-thr)))", "Bobs<2", False),
        # A jump times a factor that is 0 on the surface: the law is
        # continuous though the conditional is not.
        ("if(R<0,0,if(R>0,1,0.5))*kb*R", "R<0", True),
        ("if(R<0,0,if(R>0,1,0.5))*kb*(R+1)", "R<0", False),
        # A step call in the comparison that does not step at the values
        # tried: the threshold jumps where Z reaches 1e12.
        ("if(X>thr+floor(Z/1e12+0.5),kb*(X-thr),0)", "X>thr+floor(Z/1e12+0.5)", False),
        # A comparison the law does not hold.
        ("if(Aobs<thr,kb*(thr-Aobs),0)", "Bobs<thr", False),
        # The jump is along the second symbol the comparison reads.
        ("if((A-1)*(Z-2)>0,kb*(A-1),0)", "(A-1)*(Z-2)>0", False),
        # The comparison flips twice along B: a bend first, then a jump.
        ("if((B-0.1)*(B-3)>0,kb*(B-0.1),0)", "(B-0.1)*(B-3)>0", False),
        # A magnitude that holds a conditional on the atom is one choice,
        # made the same way either side.
        ("kb*abs(if(A<B,A,B)-c)", "A<B", True),
        # A jump in proportion to a magnitude less its argument: 0 where the
        # argument is negative, as it is at the values tried.
        ("if(A<thr,kb*(abs(Y-3)-(3-Y)),0)", "A<thr", False),
        # A condition with the atom and its denial: never true.
        ("if((A<thr)&&not(A<thr),kb,0)", "A<thr", True),
        # Two conditionals on one atom, one a bend and one a jump.
        ("if(Aobs<thr,kb*(thr-Aobs),0)+if(Aobs<thr,kb,0)", "Aobs<thr", False),
        # A pole of the condition and of the law: passed over, and the bend
        # where the numerator is 0 is what is left.
        ("if(((A-B)/(U+R))>0,(A-B)/(U+R),0)", "((A-B)/(U+R))>0", True),
        # 1 − 1/(1 + exp(−x)) is 0 or one ulp of 1 far out, and the law steps
        # by that: inside what its own arithmetic rounds by.
        (
            "if((a*(1-1/(1+exp(-(Q-off)/tmp)))*(S-Q))>0,a*(1-1/(1+exp(-(Q-off)/tmp)))*(S-Q),0)",
            "(a*(1-1/(1+exp(-(Q-off)/tmp)))*(S-Q))>0",
            True,
        ),
        # A threshold written as a count of molecules.
        ("if(Aobs>250000,kb*(Aobs-250000),0)", "Aobs>250000", True),
        ("if(Aobs>250000,kb,0)", "Aobs>250000", False),
    ],
)
def test_whether_a_rate_law_is_continuous_where_its_condition_flips(law, atom, continuous):
    """The question the refusal turns on, asked of the rate law's text."""
    from bngsim._switch_sensitivity import _continuous_across

    assert _continuous_across(law, atom) is continuous


GATES = "if(T<4,1,if(T>=16,if(T<20,1,0),0))"


SIGNED = f"if((({GATES})*fA-V)>0,(({GATES})*fA-V),0)"


@pytest.mark.parametrize(
    ("law", "atom", "held", "continuous"),
    [
        (SIGNED, f"(({GATES})*fA-V)>0", {"T"}, True),
        (f"if((({GATES})*fA-V)>0,kb,0)", f"(({GATES})*fA-V)>0", {"T"}, False),
        # With the clock moved, the gates are what is asked about, and each
        # is a jump.
        (SIGNED, f"(({GATES})*fA-V)>0", set(), True),
        (SIGNED, "T<4", set(), False),
    ],
    ids=["signed-rate", "jump", "signed-rate-on-a-moved-clock", "a-gate-on-a-moved-clock"],
)
def test_a_gate_schedule_inside_the_condition_is_read_without_sympy(
    monkeypatch, law, atom, held, continuous
):
    """A condition that holds a conditional on a clock T: sympy puts it into a
    canonical form as it parses, at 2 s a rate law, and one corpus model
    (mt_music_sequencer) has twelve of them. The law is read as plain doubles.
    The gates inside the comparison are symbols of their own: where one of
    them flips, the comparison jumps, and that is the gate's own flip."""
    from bngsim import _jacobian
    from bngsim._switch_sensitivity import _continuous_across

    def parse(expr):
        raise AssertionError(f"parsed through sympy: {expr[:40]}")

    monkeypatch.setattr(_jacobian, "_exprtk_to_sympy", parse)
    assert _continuous_across(law, atom, held=frozenset(held)) is continuous


def test_a_condition_is_not_moved_along_what_is_held():
    """``(A − thr)/tau < 0`` flips along tau where tau is 0, across a pole,
    and the ramp does not meet 0 there. tau is a parameter: no run moves it
    and no column perturbs it across 0, and it is held."""
    from bngsim._switch_sensitivity import _continuous_across

    law, atom = "if((A-thr)/tau<0,kb*(thr-A),0)", "(A-thr)/tau<0"
    assert _continuous_across(law, atom, held=frozenset({"tau", "thr", "kb"}))
    assert not _continuous_across(law, atom)


def test_a_law_that_underflows_to_zero_does_not_jump():
    """``1/(1 + exp(x))`` is 1e-306 and then exactly 0 where the exponential
    overflows, and the signed rate that carries it goes from 1e-306 to 0
    there. Under 1e-292 a double is within gradual underflow of 0."""
    from bngsim._switch_sensitivity import _continuous_across

    v = "a/(1+exp((QL-QR)/tmp))"
    held = frozenset({"tmp", "a"})
    assert _continuous_across(f"if(({v})>0,{v},0)", f"({v})>0", held=held)
    # The condition flips there in the engine's arithmetic too, and a law
    # that is 5 more on one side of it steps by 5: what the exponential
    # rounds by near the largest double is not what the law rounds by.
    assert not _continuous_across(f"if(({v})>0,{v}+5,0)", f"({v})>0", held=held)


def test_a_ramp_whose_condition_divides_by_a_parameter_runs(tmp_path):
    """Control. The same ramp in a model: its condition changes sign with the
    parameter it divides by, which nothing in a run moves."""
    sim = _simulator(
        tmp_path, "if((Aobs-thr)/tau<0,kb*(thr-Aobs),0)", DECLINED, ["k", "thr"], True
    )
    assert not sim.has_analytic_sens_rhs
    a0, k, thr, kb = 10.0, 1.0, 4.4, 3.0
    t_star = math.log(a0 / thr) / k
    tail = math.exp(-k * T_END)
    want = [
        kb * (thr * t_star / k + thr / k**2 - a0 * T_END * tail / k - a0 * tail / k**2),
        kb * (T_END - t_star),
    ]
    np.testing.assert_allclose(_y_columns(sim), want, rtol=1e-6)


@pytest.mark.parametrize(
    ("law", "atom", "continuous"),
    [
        # The atom flips at B = 0.1 and at B = 3: a bend at 3, a jump at 0.1.
        ("if((B-0.1)*(B-3)>0,kb*(B-3),0)", "(B-0.1)*(B-3)>0", False),
        # A bend at both.
        ("if((B-0.1)*(B-3)>0,kb*(B-0.1)*(B-3),0)", "(B-0.1)*(B-3)>0", True),
        # The comparison tends to 0 far out and reaches it only at B = 0.1,
        # where the law jumps. A secant from B between 0.5 and 2 runs outward.
        ("if((B-0.1)*exp(-3*B)>0,kb,0)", "(B-0.1)*exp(-3*B)>0", False),
        # A flip across a pole, with the law finite either side.
        ("if(1/(B-0.7)>0,kb,2*kb)", "1/(B-0.7)>0", False),
        # A condition no value of its symbols flips.
        ("if((A*A+thr*thr)<0,kb,0)", "(A*A+thr*thr)<0", False),
        # The magnitude of a signed quantity: two flips, a bend at each.
        ("if(abs(V)>c,kb*(abs(V)-c),0)", "abs(V)>c", True),
        ("if(abs(V)>c,kb*(V-c),0)", "abs(V)>c", False),
    ],
)
def test_every_flip_of_a_condition_is_asked_about(law, atom, continuous):
    """A comparison can flip at more than one value of a symbol, and the law
    can bend at one and jump at another. Asked only at the root a secant came
    to from a value between 0.5 and 2, the first of these was found
    continuous, and the third at a root that is none."""
    from bngsim._switch_sensitivity import _continuous_across

    assert _continuous_across(law, atom) is continuous


@pytest.mark.parametrize(
    "law",
    ["kb*(1+(Aobs-thr)/abs(Aobs-thr))/2", "kb*(max(Aobs,thr)-thr)/(Aobs-thr)"],
    ids=["a-ratio-to-its-magnitude", "a-ratio-of-the-greater"],
)
def test_a_jump_written_with_no_condition_is_refused(tmp_path, law):
    """``(X − thr)/abs(X − thr)`` is −1 below thr and 1 above it, with no
    condition written and no step call. ``abs``, ``max`` and ``min`` are
    choices the law makes on the state, each with a surface, and each is
    asked about as a condition is. dY/dk came back 18.37 for 10.2 at a loose
    tolerance, and the run ended in CVODE's no-progress error at a tight one."""
    _refused(_simulator(tmp_path, law, DECLINED, ["k"]), "Aobs")


@pytest.mark.parametrize(
    ("law", "want"),
    [
        # Y = kb·((T − t*)² + t*²)/2 with A = 1 + k·t crossing thr at t* = 3.4/k.
        ("kb*abs(Aobs-thr)", 3.0 * (6.0 * (6.0 - 3.4) * 6.0 / 2.0 - 3.4 * 3.4 * 6.0 / 2.0) / 6.0),
    ],
    ids=["a-magnitude"],
)
def test_a_choice_that_bends_runs_on_the_difference_quotient(tmp_path, law, want):
    """Control. ``kb·abs(Aobs − thr)`` is continuous where A is thr, and so is
    the declined law beside it, ``kc·max(Aobs, 0.5)``: bends, which the
    quotient is right across. dY/dk in closed form."""
    sim = _simulator(tmp_path, law, DECLINED, ["k"])
    assert not sim.has_analytic_sens_rhs
    k, thr, kb = 1.0, 4.4, 3.0
    t_star = (thr - 1.0) / k
    # dY/dk = kb·(∫ t dt over [t*, T] − ∫ t dt over [0, t*]).
    want = kb * ((T_END**2 - t_star**2) / 2.0 - t_star**2 / 2.0)
    np.testing.assert_allclose(_y_columns(sim), [want], rtol=1e-6)


@pytest.mark.parametrize(
    ("law", "bends"),
    [
        ("kb*abs(X-thr)", None),
        ("kb*(X-thr)/abs(X-thr)", "abs(X - thr)"),
        ("kb*max(X,thr)", None),
        ("kb*(max(X,thr)-thr)/(X-thr)", "max(X, thr)"),
        ("kb*min(X,thr,c)", None),
        # A clamp written with the two calls is a bend at each.
        ("kb*max(0,min(X,n))", None),
        # A choice on the time alone flips at an instant.
        ("kb*X*(T-3)/abs(T-3)", None),
        # A step call inside a choice.
        ("kb*abs(floor(X)-thr)", "abs(floor(X) - thr)"),
    ],
)
def test_which_choice_a_rate_law_jumps_across(law, bends):
    from bngsim._switch_sensitivity import _choice_jump

    assert _choice_jump(law, {}, frozenset({"T", "kb"})) == bends


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
    "expr",
    [
        "if(a<b,a*b,a/b)+max(a,b,c)-min(a,c)+abs(a-c)",
        "exp(-a/b)*log(c)+ln(a)+log10(b)+log2(c)+sqrt(a*b)",
        "sin(a)+cos(b)+tan(c)+asin(a/4)+acos(b/4)+atan(c)+sinh(a)+cosh(b)+tanh(c)",
        "a^b-(a-c)^2+(-a)^3+2^-b+_pi*_e",
        "if((a>b)&&(b<=c),1,2)+if((a>=b)||not(c!=a),3,4)",
        "(a<b)+(b<c)*2",
        "1-1/(1+exp(-(a-b)*40))",
        "sqrt(a-b-c)+log(a-b-c)+1/(a-a)+(a-b-c)^0.5",
    ],
)
def test_the_two_readings_of_a_rate_law_agree(expr):
    """A law is read twice: compiled, for the many readings of a condition
    along a symbol, and walked, for a value with what it rounds by. The two
    give one value, a NaN where the other gives a NaN."""
    from bngsim._switch_sensitivity import _float_form, _float_rounding, _float_tree

    tree = _float_tree(expr)
    evaluate, names, _ = _float_form(tree)
    for values in ({"a": 1.7, "b": 0.6, "c": 0.9}, {"a": 0.3, "b": 1.9, "c": 1.1}):
        point = {name: values[name] for name in names}
        compiled = float(evaluate(point))
        walked, size = _float_rounding(tree, point)
        assert compiled == float(walked) or (math.isnan(compiled) and math.isnan(walked))
        assert size >= 0.0 and math.isfinite(size)
