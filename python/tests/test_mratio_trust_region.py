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

The region it is trusted in, measured against a 40 digit reference:

* ``a`` a non-positive integer. Safe by construction rather than by
  measurement: the odd partial numerator carries ``z*(a+k)`` and reaches exactly
  zero at ``k = -a``, so the fraction terminates and computes a finite exact sum.
* ``a <= 0`` with ``z <= 0``. This is where BNG models live, since a model builds
  ``a = -min(AT, BT)`` and ``z = -1/Keq``, and it stays true when a fit moves the
  counts off the integers.
* ``b*b >= 64*|z*a|``, which says the partial numerators start small.

Everything else is refused. That gives up answers the fraction would have got
right, because in the uncertain region it is right most of the time and there is
no way to tell which. A refusal that says so is worth more than a number that is
usually right.
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
def test_the_wrong_answers_are_refused(tmp_path, a, b, z, old, true_value):
    """Each of these returned ``old`` where the ratio is ``true_value``.

    The third row is the one that makes the case on its own: it returned a
    negative number where the answer is positive.
    """
    assert abs(old - true_value) / true_value > 1.0  # the error really was huge
    with pytest.raises(Exception) as excinfo:
        _value(tmp_path, a, b, z)
    message = str(excinfo.value)
    assert "not reliable" in message
    # The message has to leave the reader somewhere to go.
    assert "#453" in message
    assert "non-positive integer" in message


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


def test_a_non_positive_integer_terminates_the_fraction(tmp_path):
    """z > 0 as well, which only this clause admits."""
    assert _value(tmp_path, -7.0, 3.0, 0.0) == pytest.approx(1.0, rel=1e-12)
    assert _value(tmp_path, -2.0, 5.0, -0.5) == pytest.approx(26.0 / 29.0, rel=1e-12)
    # a = -3 with z = +10: refused by the other two clauses, admitted by this one.
    assert _value(tmp_path, -3.0, 2.5, 10.0) == pytest.approx(-0.774436090226, rel=1e-9)


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
