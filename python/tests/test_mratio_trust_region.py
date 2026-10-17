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
  counts off the integers.
* ``|z| <= 20``, clean over 3400 hostile triples where the first wrong answer
  anywhere is at ``|z| = 60``.
* ``b*b >= 64*|z*a|``, which says the odd partial numerators start small,
  and for a positive ``z`` also ``2*z <= b``, which is the same statement
  about the even ones. That second half is issue #456: without it this
  clause admitted a corner where the fraction was wrong 60 times, since
  the even numerators carry ``z*(a-b-k)`` and so are about ``z/b`` for a
  large ``b`` however small ``z*a`` is.

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
        # The message has to leave the reader somewhere to go.
        assert "#453" in message
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
    with pytest.raises(Exception, match=r"not reliable.*#768"):
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
        with pytest.raises(bngsim.SimulationError, match=r"not reliable.*#768"):
            sim.run(t_span=(0.0, 1.0), n_points=3)
    assert sim.codegen_backend in (("cc", "mir") if codegen else ("exprtk",))
