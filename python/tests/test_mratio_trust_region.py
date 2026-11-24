"""Issue #453 — mratio must refuse arguments it cannot compute correctly.

``mratio(a, b, z)`` is ``M(a+1,b+1,z) / M(a,b,z)``, evaluated by Gauss's
continued fraction with the modified Lentz method. Outside a certain region the
fraction converges to something that is not the ratio: errors reached a factor
of a thousand and in places the sign was wrong, with no warning and no failure.

Three findings shaped the fix, and each is the reason a cheaper fix does not
work.

* The approach to the false limit is indistinguishable from the approach to the
  true one. ``|Delta - 1|`` decays smoothly and geometrically into both. So no
  stopping test can separate them, however many consecutive hits it demands.
* Iterating longer does not help either. At 120 digits the same recurrence does
  reach the true value, but only after about 1600 steps where double precision
  settles at about 84; by then rounding has frozen the iterates.
* So the decision has to be made from the arguments, before the fraction runs.

The region it is trusted in, measured against a 40 digit reference, is all for
``b > 0`` only (issue #768): every grid behind it had a positive ``b``, and with
``b`` negative the fraction is wrong inside each clause. Within that:

* ``a`` a negative integer. Safe by construction rather than by
  measurement: the odd partial numerator carries ``z*(a+k)`` and reaches exactly
  zero at ``k = -a``, so the fraction terminates and computes a finite exact sum.
  Not ``a = 0`` (issue #768): the odd numerators start at ``k = 1``, so none of
  them vanishes there.
* ``a <= 0`` with ``z <= 0``. This is where BNG models live, since a model builds
  ``a = -min(AT, BT)`` and ``z = -1/Keq``, and it stays true when a fit moves the
  counts off the integers. Except past ``z = -20`` with ``b - a < 1`` (issue
  #1080), where the fraction loses about ``10*eps*sqrt(-z)/(b - a)`` of the
  ratio, which is everything once ``b`` and ``a`` are both tiny. Those are summed
  as two series out to ``z = -700`` and left to the asymptotic expansion past
  that. That is the very last section below.
* ``|z| <= 20``, clean over 3400 hostile triples where the first wrong answer
  anywhere is at ``|z| = 60``. Clean of false limits, that is: issue #1080
  found it losing digits there, up to 4.4e-7, when ``a`` is tiny next to ``b``
  or ``b`` itself is tiny, and those arguments are now summed as two series
  before this rule is asked. That is the last section below.
* ``b*b >= 64*|z*a|``, which says the odd partial numerators start small,
  and for a positive ``z`` also ``2*z <= b``, which is the same statement
  about the even ones. That second half is issue #456: without it this
  clause admitted a corner where the fraction was wrong 60 times, since
  the even numerators carry ``z*(a-b-k)`` and so are about ``z/b`` for a
  large ``b`` however small ``z*a`` is. Nor does it admit, past ``z = -20``,
  a positive ``a`` below a ``b`` under 1 (issue #1793): a tiny enough ``a``
  passed the bound, and the fraction lost there what it loses in the corner of
  the ``a <= 0`` clause, so that corner now takes it.

Everything else is refused by the fraction. That gives up answers it would have
got right, because in the uncertain region it is right most of the time and
there is no way to tell which. A refusal that says so is worth more than a
number that is usually right.

Issue #456 later narrowed what that gives up, by sending those arguments to an
asymptotic expansion that can vouch for its own accuracy before refusing them.
That is in test_mratio_asymptotic.py. It changed no answer the fraction was
already giving, which is what test_the_regime_models_use_still_answers below
holds it to.
"""

from __future__ import annotations

import contextlib
import io
import math

import bngsim
import pytest

NET = """begin parameters
    1 a  {a!r}  # Constant
    2 b  {b!r}  # Constant
    3 zz {z!r}  # Constant
end parameters
begin functions
    1 law() mratio(a,b,zz)
end functions
begin species
    1 A() 1
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 A                    1
end groups
"""


def _value(tmp_path, a: float, b: float, z: float) -> float:
    """mratio(a, b, z) through the engine, or the exception it refuses with."""
    p = tmp_path / "m.net"
    p.write_text(NET.format(a=a, b=b, z=z))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(p))
    core = model._core if hasattr(model, "_core") else model
    return core._eval_functions(0.0, [1.0])["law"]


# ── The arguments that were silently wrong ───────────────────────────────────

# a, b, z, and what the ratio really is (mpmath at 40 digits). The middle column
# is what the fraction used to return.
WAS_WRONG = [
    (10.0, 2.5, -10000.0, 0.249812331, 0.0002502129154),
    (10.0, 0.5, -10000.0, 0.04995245718, 5.005261319e-05),
    (5.0, 0.5, -10000.0, -0.02565865026, 5.002753167e-05),
    (50.0, 2.5, -1000.0, 0.047495986, 0.0026350702),
]


@pytest.mark.parametrize("a, b, z, old, true_value", WAS_WRONG)
def test_the_wrong_answers_are_never_returned_again(tmp_path, a, b, z, old, true_value):
    """Each of these returned ``old`` where the ratio is ``true_value``.

    The third row is the one that makes the case on its own: it returned a
    negative number where the answer is positive.

    What this asks for is that the wrong value never comes back. It does not ask
    for a refusal, because the refusal was the remedy and not the goal: issue
    #456 added an asymptotic expansion that gets three of these four right, and
    only the last is still refused. Which of the two happens is pinned argument
    by argument in test_mratio_asymptotic.py. The contract here is the one that
    has to hold whatever else changes.
    """
    assert abs(old - true_value) / true_value > 1.0  # the error really was huge
    try:
        got = _value(tmp_path, a, b, z)
    except Exception as excinfo:
        message = str(excinfo)
        assert "not reliable" in message
        # The message has to leave the reader somewhere to go: the region the
        # fraction is reliable in and the arguments BNG builds, with no issue
        # number to look up.
        assert "The fraction is reliable for b > 0" in message
        assert "BNG builds a = -min(AT,BT)" in message
        assert "#" not in message
        assert "negative integer" in message
        return
    assert got == pytest.approx(true_value, rel=1e-9)


# ── The region models actually use, which must still answer ──────────────────

# BNG builds a = -min(AT, BT) and b = max - min + 1 and z = -1/Keq. The first
# row is test_Mratio_1 as published; the rest are where a fit takes it, with the
# counts no longer integers. Reference values are mpmath at 40 digits.
BNG_REGIME = [
    (-1000.0, 9001.0, -10000.0, 0.46128328365),
    (-1000.0, 9001.0, -1000.0, 0.891196668786),
    (-1000.37, 9000.63, -10000.0, 0.461268477014),
    (-999.25, 9001.75, -2702.7, 0.754511007597),
    (-250.5, 9750.5, -10000.0, 0.490565895369),
]


@pytest.mark.parametrize("a, b, z, expect", BNG_REGIME)
def test_the_regime_models_use_still_answers(tmp_path, a, b, z, expect):
    """Integer counts and the continuous path a fit takes through them.

    A rule stated only as a bound on ``|z*a|/b*b`` passes every grid check and
    still refuses these, which is why the rule has the ``a <= 0 and z <= 0``
    clause: a fit moves the counts off the integers and the model has to keep
    working while it does.
    """
    got = _value(tmp_path, a, b, z)
    assert got == pytest.approx(expect, rel=1e-9)


# ── Each clause of the rule carries its own witness ──────────────────────────


def test_a_negative_integer_terminates_the_fraction(tmp_path):
    """z > 20 as well, with a small b, which only this clause admits."""
    assert _value(tmp_path, -7.0, 3.0, 0.0) == pytest.approx(1.0, rel=1e-12)
    assert _value(tmp_path, -2.0, 5.0, -0.5) == pytest.approx(26.0 / 29.0, rel=1e-12)
    assert _value(tmp_path, -3.0, 2.5, 10.0) == pytest.approx(-0.774436090226, rel=1e-9)
    # a = -3 with z = +30: past |z| <= 20, and b*b = 6.25 is under 64*|z*a| =
    # 5760, so every other clause refuses it and only this one admits it. The
    # asymptotic route cannot stand in either: a is a pole of Gamma(a).
    assert _value(tmp_path, -3.0, 2.5, 30.0) == pytest.approx(-0.0994800693240901, rel=1e-12)


def test_a_small_partial_numerator_is_admitted(tmp_path):
    """Positive a, which only the contraction clause admits."""
    assert _value(tmp_path, 0.5, 101.0, 1.0) == pytest.approx(1.00985005864, rel=1e-9)
    assert _value(tmp_path, 2.0, 101.0, -30.0) == pytest.approx(0.775054308812, rel=1e-9)


def test_the_same_arguments_with_a_smaller_b_are_refused(tmp_path):
    """The pair to the test above: only b changed, and the answer is now a refusal.

    b*b against 64*|z*a| is the whole difference between these two, so this is
    what shows the rule is doing the work rather than the sign of a.
    """
    with pytest.raises(Exception, match="not reliable"):
        _value(tmp_path, 2.0, 2.5, -30.0)


# ── Issue #768: a = 0, and a negative b ──────────────────────────────────────

# a, b, z, what the fraction returned, and the ratio. For a = 0 the ratio is
# M(1, b+1, z), since M(0, b, z) = 1: (e^z - 1)/z at b = 1, and
# sqrt(pi) e^z erf(sqrt z) / (2 sqrt z) at b = 1/2. The rest are mpmath at 60
# digits.
WAS_WRONG_768 = [
    (0.0, 1.0, 100.0, 1.2821328238e15, 2.6881171418161353e41),
    (0.0, 1.0, 50.0, 3.4399420235e15, 1.0369411057174145e20),
    (0.0, 1.0, 30.0, 3.5627049156e11, 3.562158193841154e11),
    (0.0, 0.5, 40.0, -4.0919618279e15, 3.2983308827274492e16),
    (0.0, 50.0, 300.0, 8.08041719198144, 8.229182757884219e70),
    (
        -22.52202123513951,
        -25.181368299781333,
        -12.004134759218255,
        2.853393829675684,
        -2.3371245658875206,
    ),
    (-19.96, -28.47, -14.64, 1.1254407074935024, 0.778240700764792),
    (-1.29, -19.03, 9.24, 45.465477953425015, 45.56636138975769),
]


@pytest.mark.parametrize("a, b, z, old, true_value", WAS_WRONG_768)
def test_a_zero_or_a_negative_b_is_not_answered_wrongly(tmp_path, a, b, z, old, true_value):
    """Trusted by the clause for a non-positive integer a, which never
    terminates at a = 0, or by clauses only ever measured with b > 0. An a = 0
    with z > 0 is now summed exactly as M(1, b+1, z); a negative b is refused,
    none of these being one the asymptotic route can vouch for."""
    assert abs(old - true_value) / abs(true_value) > 1e-4
    if a == 0.0:
        assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-13)
        return
    with pytest.raises(Exception, match=r"not reliable.*The fraction is reliable for b > 0"):
        _value(tmp_path, a, b, z)


@pytest.mark.parametrize(
    "a, b, z, true_value",
    [
        # a = 0 and z > 0, summed: the fraction's |z| <= 20 clause was 8.7e-9
        # off at the second, and 2z <= b admitted the first.
        (0.0, 100.0, 30.0, 1.4200972358613853),
        (0.0, 1.0, 20.0, 24258259.720489514),  # (e^20 - 1)/20
        (0.0, 1.0, 1e-3, 1.0005001667083417),
        (0.0, 1.0, -100.0, 0.01),  # a <= 0 and z <= 0: (1 - e^-100)/100
        (0.0, 3.0, 0.0, 1.0),  # the same clause at z = 0, where M(1, b+1, 0) = 1
        (-16.0, 25.44, -15.76, 0.5501014741976908),  # a negative integer, b > 0
    ],
)
def test_a_zero_where_another_route_holds_answers(tmp_path, a, b, z, true_value):
    assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-12)


def test_a_zero_past_the_largest_double_is_infinite(tmp_path):
    """(e^800 - 1)/800 is past 1.8e308: +inf is the ratio rounded."""
    assert _value(tmp_path, 0.0, 1.0, 800.0) == float("inf")


def test_a_zero_the_series_cannot_finish_is_refused(tmp_path):
    """With b past about 5e8 and z near b the terms fall too slowly for the
    series' million-term cap (here it would need 7.1e6). It gives up with a NaN
    and the call goes on: the rule refuses a = 0 with 2*z > b, and the asymptotic
    route refuses a = 0 outright.

    The fraction would not have been wrong at this point. Before issue #768 it
    answered 1253314.4707011, and the ratio is 1253314.4706489 (mpmath, summed
    over 1.05e7 terms): 4.2e-11 off. So this pins a refusal, not a correction.
    The refusal holds because nothing vouches for the fraction at a = 0 with
    2*z > b: where it was checked (b up to 100, z from 20 to 400) it was wrong at
    168 of 200 points, mratio(0, 1, 100) among them."""
    with pytest.raises(Exception, match=r"not reliable.*The fraction is reliable for b > 0"):
        _value(tmp_path, 0.0, 1e12, 1e12)


ISSUE_768_NET = """begin parameters
    1 scale 9.643749239819589e-21
end parameters
begin functions
    1 f() mratio(0,1,50)*scale
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 f
end reactions
begin groups
    1 Aobs 1
end groups
"""


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
def test_the_issues_model_follows_the_closed_form(tmp_path, codegen):
    """``A' = mratio(0, 1, 50) / 1.0369e20`` from 0, so A(1) = 1. Both engines
    returned 3.3e-05 with no error."""
    p = tmp_path / "m768.net"
    p.write_text(ISSUE_768_NET)
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(p))
        sim = bngsim.Simulator(model, method="ode", codegen=codegen)
        result = sim.run(t_span=(0.0, 1.0), n_points=3)
    # The compiled leg has to run the generated C, or it tests the interpreter twice.
    assert sim.codegen_backend in (("cc", "mir") if codegen else ("exprtk",))
    assert float(result.species[-1][0]) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
def test_a_negative_b_in_a_rate_law_fails_rather_than_returning_a_wrong_trajectory(
    tmp_path, codegen
):
    """The interpreter raises the refusal; the generated C returns NaN, and the
    run's non-finite witness re-evaluates the model and carries the same text."""
    p = tmp_path / "m768b.net"
    p.write_text(ISSUE_768_NET.replace("mratio(0,1,50)*scale", "mratio(-22.52,-25.18,-12.0)"))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(p))
        sim = bngsim.Simulator(model, method="ode", codegen=codegen)
        with pytest.raises(
            bngsim.SimulationError, match=r"not reliable.*The fraction is reliable for b > 0"
        ):
            sim.run(t_span=(0.0, 1.0), n_points=3)
    assert sim.codegen_backend in (("cc", "mir") if codegen else ("exprtk",))


# ── Issue #1080: digits the fraction lost inside |z| <= 20 ──────────────────

# a, b, z, what the fraction returned, and the ratio (mpmath at 60 digits). No
# false limit here: the fraction converged to the ratio, but lost about
# 1e-14*|R| of it, and R reaches 1e9 where a is tiny next to a b below 1 and z
# is near 20. The fourth and fifth rows are a tiny b near z = -20, the fourth
# through the a <= 0, z <= 0 clause. The last is a b far below anything a model
# builds, where the fraction returned its Lentz floor, 1e-32.
WAS_OFF_1080 = [
    (-1e-12, 0.026664156824421394, 19.310533421401022, 223548140.61039492, 223548187.99851523),
    (
        5.1457398429446604e-14,
        0.002765315544365488,
        19.907142257727095,
        434265223.3081783,
        434265294.9192469,
    ),
    (
        -2.6999339278500113e-11,
        0.011537625782194286,
        19.682954101484214,
        1648055069.4106143,
        1648055787.7495603,
    ),
    (
        -8.607573956569877e-15,
        1.535796353807133e-08,
        -19.72305014003269,
        3.541811340992023e-09,
        3.5418112969725804e-09,
    ),
    (
        9.032951159523816e-08,
        2.0956113825578895e-08,
        -19.34191963409733,
        -5.592377139114178e-11,
        -5.592377814347665e-11,
    ),
    (0.5, 1e-300, 10.0, 9.999999999999999e-33, 2.1056310242163857e-300),
]


@pytest.mark.parametrize("a, b, z, old, true_value", WAS_OFF_1080)
def test_the_digits_the_fraction_lost_are_summed_back(tmp_path, a, b, z, old, true_value):
    """Each of these was answered by the fraction and came out more than 1e-8
    off. For b > 0, 0 < |z| <= 20 and |alpha| < 1, with alpha = a for z > 0 and
    b - a for z < 0, both Kummer series are summed instead, and their terms after
    the first all share a sign, so nothing cancels at these arguments."""
    assert abs(old - true_value) / abs(true_value) > 1e-8
    assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-13, abs=0.0)


def test_near_a_pole_the_error_is_what_an_ulp_of_a_does(tmp_path):
    """M(a, b, z) is 6.5e-5 here, near a zero at z = 18.37, so the ratio is
    6.9e11 and the denominator's sum cancels. That cancellation is the problem's
    own: one ulp of a moves the ratio by 2.5e-12 (mpmath), and the sum is off
    by about that. The fraction was off by 2.8e-4, a hundred million ulps."""
    a, b, z = -5.016391657033421e-09, 0.2277418763764943, 18.373535269367025
    assert _value(tmp_path, a, b, z) == pytest.approx(691719131795.04684, rel=1e-11, abs=0.0)


@pytest.mark.parametrize(
    "a, b, z, true_value",
    [
        # Just outside the box, one on each edge, where the fraction answers as
        # before and is right to rounding. mpmath at 80 digits and more.
        (1.0, 0.026664156824421394, 19.310533421401022, 0.028008147640145812),  # alpha = 1
        (-1.0, 0.5, 10.0, -1.0 / 19.0),  # a negative integer: the fraction terminates
        (-1e-12, 1.0, -19.5, 0.051282051107778356),  # z < 0 with b - a just above 1
        # |z| past 20, where the box goes on only for 0 < b - a < 1 with b < 1
        # (the last sections below): a positive a with b = 1, then b - a just
        # above 1, near and far.
        (1e-6, 1.0, -20.5, 0.04878049025292664),
        (-1e-12, 1.0, -25.0, 0.039999999999442815),
        (-0.5, 1.5, -1000.0, 0.0014985007496251875),
    ],
)
def test_the_fraction_still_answers_outside_the_box(tmp_path, a, b, z, true_value):
    assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-13, abs=0.0)


# ── Issue #1080: past z = -20, where b - a < 1 ──────────────────────────────

# a, b, z, what the fraction returned, and the ratio (mpmath at 80 digits and
# more, two ways: hyp1f1 after Kummer's transformation, and the transformed
# series summed directly). All in the a <= 0, z <= 0 clause, which used to
# trust the fraction at any z. It converged to the ratio but lost about
# 10*eps*sqrt(-z)/(b - a) of it. The first row is the issue's own example and
# the second its a = 0. The next two were the worst of 20000 random triples
# with b in [1e-12, 10], |a| in [1e-15, 10] and z in [-2000, 0], and both are
# past z = -700, so the asymptotic route answers them, the second through its
# exact M(0, b, z) = 1. The fifth is off by a factor of two. The last is a b far
# below anything a model builds, where the fraction returned its Lentz floor.
WAS_OFF_1080_PAST_20 = [
    (-1e-12, 1e-9, -25.0, 5.5620506912562485e-11, 5.562054719123343e-11),
    (0.0, 1e-9, -30.0, 3.4620717894130465e-11, 3.462069802043962e-11),
    (
        -2.2816251335577544e-13,
        1.048356393428415e-12,
        -1576.5643547958143,
        6.572841358026786e-16,
        6.653849543435732e-16,
    ),
    (
        0.0,
        1.1766052422876225e-12,
        -784.9671045705651,
        1.5179125913905077e-15,
        1.5008374136500115e-15,
    ),
    (
        -2.4013304216952906e-13,
        1.3563113178612664e-12,
        -786022.5708171733,
        3.680405664315373e-18,
        1.7255395630598238e-18,
    ),
    (0.0, 1e-300, -100.0, 1.292767299104767e-17, 3.720075976020836e-44),
]


@pytest.mark.parametrize("a, b, z, old, true_value", WAS_OFF_1080_PAST_20)
def test_past_z_minus_20_a_tiny_b_minus_a_is_not_answered_wrongly(
    tmp_path, a, b, z, old, true_value
):
    """Out to z = -700 both Kummer series are summed, after the transformation
    that makes every term of both positive: alpha = b - a is at least b when
    a <= 0. Past it the asymptotic route answers, which in this corner keeps a
    branch that outweighs the dropped one by e^(-z) and so vouches for itself."""
    assert abs(old - true_value) / true_value > 1e-8
    assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-13, abs=0.0)


@pytest.mark.parametrize(
    "a, b, z, true_value",
    [
        # The same corner where b - a is not tiny and the fraction was right to
        # rounding. These have to keep answering: on either side of z = -700,
        # and an a = 0 past it, which the asymptotic route refused until it was
        # told that M(0, b, z) = 1 is exact. mpmath at 80 digits.
        (-0.25, 0.5, -700.0, 0.0007145411817709659),
        (-0.25, 0.5, -701.0, 0.0007135215003868878),
        (-0.5, 0.3, -1000.0, 0.0003000600420655688),
        (0.0, 0.5, -1e4, 5.000250037509378e-05),
        # The fraction did not converge at all here, within 100000 steps, and
        # raised. The ratio is b/(-z) to a part in 1e9.
        (-1e-12, 1e-9, -1e9, 1.0000000010000001e-18),
    ],
)
def test_past_z_minus_20_what_the_fraction_got_right_still_answers(tmp_path, a, b, z, true_value):
    assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-13, abs=0.0)


def test_past_z_minus_700_with_b_minus_a_near_the_smallest_double_is_refused(tmp_path):
    """The ratio is 1.41e-303 and the fraction returned 1.42e-18. Past z = -700
    the series is not tried, since its terms could pass the largest double, and
    the asymptotic route cannot vouch here: the branch it keeps is about
    b/(-z) = 1.4e-303 and the one it drops about e^z = 4.5e-309, which leaves
    3e-6 of the answer unaccounted for. That is where the refusal is left, and
    only for b - a below about 1e-290."""
    with pytest.raises(Exception, match=r"not reliable.*The fraction is reliable for b > 0"):
        _value(tmp_path, 0.0, 1e-300, -710.0)


# ── Issue #1793: the same corner, with a tiny positive a ─────────────────────

# a, b, z, what the fraction returned, and the ratio (mpmath at 80 digits after
# Kummer's transformation, and at 150 without it, which agree). The corner
# above was written for a <= 0. A positive a reached the fraction past z = -20
# through the b*b >= 64*|z*a| clause instead, which a tiny enough a passes, and
# there the fraction lost what it loses in the corner, about
# 10*eps*sqrt(-z)/b. The first row is just past z = -20, where at z = -20
# itself the series already answered; the next four are the issue's, the last
# of them 5.7% off. The last is the worst of 20000 random triples with b below
# 1, off by a factor of 7000 and with the wrong sign. The series takes the
# first three, and the asymptotic route the rest, past z = -700.
WAS_OFF_1793 = [
    (1e-30, 2e-12, -20.000001, 2.0612571457511304e-09, 2.061257156856199e-09),
    (1e-30, 2e-12, -25.0, 1.3971417420721947e-11, 1.3971436819759902e-11),
    (1e-30, 2e-12, -100.0, 2.0204345020266157e-14, 2.0204125055496302e-14),
    (1e-30, 2e-12, -1000.0, 1.992426431996472e-15, 2.0020040120482374e-15),
    (1e-30, 2e-12, -30000.0, 7.046624128694702e-17, 6.666888903705185e-17),
    (
        1.6746677479385269e-143,
        1.0464791789235133e-15,
        -6549120.474099903,
        -1.1238198263826546e-18,
        1.5978929428026101e-22,
    ),
]


@pytest.mark.parametrize("a, b, z, old, true_value", WAS_OFF_1793)
def test_past_z_minus_20_a_tiny_positive_a_is_not_answered_wrongly(
    tmp_path, a, b, z, old, true_value
):
    """The corner is now 0 < b - a < 1 with b < 1. For a <= 0 that is the old
    one, since there b - a < 1 already puts b below 1, and it adds every a
    between 0 and b for a b below 1. a < b keeps alpha = b - a positive, so
    every term of both series is still positive and nothing cancels."""
    assert abs(old - true_value) / true_value > 1e-9
    assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-13, abs=0.0)


@pytest.mark.parametrize(
    "a, b, z, true_value",
    [
        # In the corner, where the fraction was right because b is not small, or
        # where nothing answered: both are summed now. mpmath at 80 digits.
        (1e-6, 0.5, -20.5, 0.02503515280316255),
        (0.4999999999, 0.5, -25.0, 0.29234219600186745),  # refused before
        # A b of 1 or more stays with the fraction where it was trusted. With
        # b - a < 1 the b*b clause admits there only a tiny a beside a b just
        # over 1, or a b past about 64*|z|, and both measured clean, worst
        # 2.3e-14 and 4.4e-16 over 5000 triples each. Past z = -700 nothing
        # else answers the second kind, which is why the corner stops at b = 1
        # rather than taking every a < b.
        (1999.5, 2000.0, -25.0, 0.9999968370848997),
        (99999.5, 100000.0, -1000.0, 0.9999999494954699),
        # And an a above b, which the corner never takes: alpha is negative.
        (1500.5, 1500.0, -20.5, 1.0000046155575413),
    ],
)
def test_past_z_minus_20_with_a_positive_a_what_was_right_still_answers(
    tmp_path, a, b, z, true_value
):
    assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-13, abs=0.0)


def test_past_z_minus_700_an_a_too_small_for_the_expansion_is_refused_either_side(tmp_path):
    """mratio(5e-324, 0.5, -701) is 7.14e-4, and the fraction got it right,
    since b is not small; with a tiny b it was as wrong at this a as at 1e-30.
    Past z = -700 the corner has only the asymptotic route, and in its
    denominator the second term, a*(a-b+1)/(-z), underflows to zero, which its
    zero-term rule reads as a series that never decayed. So it refuses, as it
    already did for the same a below zero. That takes an a below about 1e-300.
    Short of -700 the series answers both."""
    for a in (5e-324, -5e-324):
        with pytest.raises(Exception, match=r"not reliable.*The fraction is reliable for b > 0"):
            _value(tmp_path, a, 0.5, -701.0)
        assert _value(tmp_path, a, 0.5, -699.0) == pytest.approx(
            0.0007158203491378445, rel=1e-13, abs=0.0
        )


# a, b, z, with b - a subnormal in each. The first three are a positive a one
# ulp below b. Summed, they came out 2.0e-6, 9.6e-7 and 2e-2 off (mpmath at 120
# digits) until the corner was held to a normal b - a; before issue #1793 they
# were refused. The last is the old corner, a <= 0 with a subnormal b, which
# was summed 1.1e-4 off before issue #1793 and is refused now with the rest.
SUBNORMAL_ALPHA = [
    (2.225073858507201e-308, 2.2250738585072014e-308, -100.0),
    (9.9999999999995e-311, 1e-310, -20.5),
    (9.99999994e-316, 1e-315, -20.5),
    (-5e-324, 1e-322, -21.0),
]


@pytest.mark.parametrize("a, b, z", SUBNORMAL_ALPHA)
def test_past_z_minus_20_a_subnormal_b_minus_a_is_refused(tmp_path, a, b, z):
    """The series takes b - a only from the smallest normal double up. A
    subnormal alpha makes alpha*x and the terms just after it subnormal, each
    rounded to a multiple of 4.9e-324, and every later term is built from them.
    The rule refuses the fraction there, and the asymptotic route cannot vouch
    for a b - a that small either."""
    assert 0.0 < b - a < 2.2250738585072014e-308
    with pytest.raises(Exception, match=r"not reliable.*The fraction is reliable for b > 0"):
        _value(tmp_path, a, b, z)


# ── Issue #1080: a b below about 1e-22, wherever the fraction is trusted ─────

# a, b, z, what the fraction returned, and the ratio (mpmath at 80 digits and
# more, hyp1f1 directly and after Kummer's transformation, which agree). The
# fraction starts from tiny = 1e-32 in place of zero, so its first C stands for
# infinity and the next is 1 + p_2/C_1, which should be exactly 1. Taken
# literally it is 1 + p_2*1e-32, with p_2 = z*(a-b)/(b*(b+1)), and a tiny b makes
# that anything at all. One row per clause it reached: a <= 0 with z <= 0 (near
# and past z = -20), a negative integer, whose fraction terminates and was still
# 7e-7 off, and |z| <= 20 with each sign of z. The last has p_2 just under the
# largest double, where the ratio is itself near the smallest normal one.
WAS_OFF_1080_TINY_B = [
    (
        -4.119676222070549,
        2.4225094582386945e-279,
        -1.4540899212539904,
        9.999999999999999e-33,
        8.515249528874852e-280,
    ),
    (
        -3.689525822973658,
        5.126352093327822e-300,
        -491.8281574812038,
        1e-32,
        1.0366673205419992e-302,
    ),
    (-3.0, 1e-25, 10.0, -1.347825086956521e-26, -1.3478260869565218e-26),
    (-20.0, 1e-200, 15.0, 1e-32, -3.8469703304037625e-202),
    (2.5, 1e-25, 10.0, 4.884009591187713e-26, 4.884008591187709e-26),
    (12.0, 1e-100, -5.0, 9.999999999999999e-33, 7.757425751152755e-102),
    (
        8.48517191235676,
        3.06959761947884e-309,
        -0.05844426302774795,
        9.999999999999999e-33,
        -4.331194060173115e-309,
    ),
]


@pytest.mark.parametrize("a, b, z, old, true_value", WAS_OFF_1080_TINY_B)
def test_a_tiny_b_no_longer_leaks_through_the_fractions_start(tmp_path, a, b, z, old, true_value):
    """The step after the first now takes C_2 = 1. For |p_2| below 5.5e15,
    1 + p_2*1e-32 already rounded to 1, so no other argument changes by a bit;
    BNG's never come near, since they have b >= 1."""
    assert abs(old - true_value) / abs(true_value) > 1e-8
    assert _value(tmp_path, a, b, z) == pytest.approx(true_value, rel=1e-13, abs=0.0)


@pytest.mark.parametrize(
    "b, z, true_value",
    [
        # M(0, b, z) = 1, so the ratio is M(1, b+1, z): mpmath at 100 digits.
        (-2.5, -500.0, -0.005035318510732385),
        (-0.5, -100.0, -0.005076943751970561),
        (-7.3, -2000.0, -0.003665218300681953),
    ],
)
def test_a_zero_with_a_negative_b_far_out_is_answered(tmp_path, b, z, true_value):
    """A side effect of the asymptotic route taking M(0, b, z) = 1 as exact
    (issue #1080). Its zero-term rule used to refuse every a = 0 because the
    denominator's expansion stops after its first term, and a b <= 0 asks that
    route whatever z is, so these were refusals. Now they are answers."""
    assert _value(tmp_path, 0.0, b, z) == pytest.approx(true_value, rel=1e-13, abs=0.0)


def test_a_p_2_past_the_largest_double_still_comes_back_nan(tmp_path):
    """Here p_2 = z*(a-b)/(b*(b+1)) = -1.4e310 overflows. The fraction's second
    step keeps the old arithmetic in that case, so the call returns the NaN it
    always did, which the non-finite warner reports. It does not take C_2 = 1:
    with D_2 = 1/(1 + p_2) = 0 that would make the answer a quiet, wrong 0."""
    assert math.isnan(_value(tmp_path, -5.5, 1e-310, 0.25))
