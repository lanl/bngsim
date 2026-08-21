"""Issue #456 — narrow what mratio refuses by routing to its asymptotic expansion.

Issue #453 made ``mratio(a, b, z)`` refuse the arguments its continued fraction
cannot be trusted with, which removed a class of silently wrong answers. The
cost was that it also refused arguments the fraction would have got right, since
in the uncertain region there is no way to tell which ones those are.

This adds a second method that can vouch for its own answer. For a large ``|z|``
the ratio has an asymptotic expansion in which the Gamma factors cancel between
numerator and denominator, which is the whole point: the individual Kummer
functions overflow a double long before their ratio does. The two series are
divergent, so each is summed to its smallest term, and that term is an estimate
of how far the sum is from the function. That estimate is what makes the route
safe to add. It is used only where the estimate is small, and the refusal is
kept everywhere else.

Measured over 5955 argument triples against a 40 to 60 digit reference:

    ================================  =======  =======  ========
    ..                                answers    wrong  refused
    ================================  =======  =======  ========
    before                               3191       60     46.4%
    after                                4259        0     28.5%
    ================================  =======  =======  ========

The zero is the property that must not be given up, and it is the reason every
test below that pins a new answer pins it against a high precision reference
rather than against the fraction.

The 60 wrong answers were already there before this change; see
``test_the_positive_z_corner_that_used_to_leak`` for what they were.
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


# ── Refusals that are now answers ────────────────────────────────────────────

# a, b, z, and the ratio to 12 digits (mpmath at 50). Every one of these was a
# refusal before this change: the fraction is not trusted there, and nothing
# else was tried. The first three are the arguments issue #453 opened with,
# where the fraction returned 0.2498, 0.04995 and -0.02566 against the values
# below, the last of those with the wrong sign.
RECOVERED = [
    (10.0, 2.5, -10000.0, 0.000250212915408),
    (10.0, 0.5, -10000.0, 5.00526131851e-05),
    (5.0, 0.5, -10000.0, 5.00275316722e-05),
    (10.0, 2.5, -5000.0, 0.000500853331575),
    (10.0, 2.5, -2000.0, 0.00125536494995),
    (25.5, 0.25, -100000.0, 2.5006565964e-06),
    (3.7, 50.5, -3000.0, 0.0165798233853),
    (0.5, 101.0, 300.0, 134.158438486),
    (-0.5, 901.0, 10000.0, -1639.52290997),
    (-7.3, 2.5, 300.0, -0.330947555309),
]


@pytest.mark.parametrize("a, b, z, expect", RECOVERED)
def test_the_expansion_answers_where_the_fraction_was_refused(tmp_path, a, b, z, expect):
    assert _value(tmp_path, a, b, z) == pytest.approx(expect, rel=1e-9)


def test_the_recovered_answers_are_not_the_wrong_ones_the_fraction_had(tmp_path):
    """The three arguments issue #453 was reported with, and what they returned.

    Answering again is only an improvement if the answer is the right one. The
    fraction was wrong here by a factor of a thousand, and on the third row it
    also had the wrong sign, so a route that merely reproduced it would look
    like a fix and be the original defect.
    """
    for a, b, z, fraction_used_to_say, true_value in [
        (10.0, 2.5, -10000.0, 0.249812331, 0.000250212915408),
        (10.0, 0.5, -10000.0, 0.04995245718, 5.00526131851e-05),
        (5.0, 0.5, -10000.0, -0.02565865026, 5.00275316722e-05),
    ]:
        got = _value(tmp_path, a, b, z)
        assert got == pytest.approx(true_value, rel=1e-9)
        assert abs(got - fraction_used_to_say) / abs(fraction_used_to_say) > 0.9


# ── What the expansion will not vouch for, and why ───────────────────────────


def test_a_dropped_branch_that_is_the_whole_answer_is_refused(tmp_path):
    """``b - a`` a non-positive integer, where the term being kept is absent.

    For a negative z the expansion keeps the algebraic branch, whose coefficient
    carries 1 over Gamma(b-a). At a non-positive integer that is zero, so the
    branch it keeps is not there and what is left is the exponentially small one
    it drops. ``b == a`` is the clearest case: the ratio is then M(a+1,a+1,z)
    over M(a,a,z), which is exp(z)/exp(z), exactly 1. Reading the expansion
    literally there returns about 5e-06.

    Nothing tests for this directly. lgamma is +inf at exactly those points, so
    the estimate of the dropped branch goes to +inf and the refusal follows.
    """
    for a, b in [(5.0, 5.0), (7.3, 7.3), (9.0, 4.0)]:
        with pytest.raises(Exception, match="not reliable"):
            _value(tmp_path, a, b, -1000.0)


def test_a_series_that_never_decayed_is_refused(tmp_path):
    """A series can stop early without having shown that it converges.

    At ``b = a + 1`` the second parameter of both series is ``a - b + 1 = 0``, so
    every term after the first is exactly zero and the sum is 1. Treating that
    as an exact result claims an accuracy the series never demonstrated: the
    answer collapses to ``b / (-z)``, which is 301/21 = 14.33 for the first row
    below, where the ratio is 0.9998.

    The estimate is the last term actually added rather than the zero that ended
    the series, so a series that stopped after one term reports an error of 1 and
    is refused. A series that did decay before terminating keeps its answer.
    """
    for z in (-21.0, -30.0, -60.0, -100.0):
        with pytest.raises(Exception, match="not reliable"):
            _value(tmp_path, 300.0, 301.0, z)


def test_what_neither_method_can_do_is_still_refused(tmp_path):
    """The expansion narrows the refusal, it does not remove it.

    These need a larger ``|z|`` before the series decays far enough to vouch for
    itself, and the fraction is not trusted there either, so the answer is still
    that there is no answer.
    """
    for a, b, z in [(50.0, 2.5, -1000.0), (1.0, 901.0, 3000.0), (2.0, 2.5, -30.0)]:
        with pytest.raises(Exception) as excinfo:
            _value(tmp_path, a, b, z)
        message = str(excinfo.value)
        assert "not reliable" in message
        # The message has to leave the reader somewhere to go.
        assert "#453" in message and "#456" in message


# ── The corner that was leaking before any of this ───────────────────────────

# a, b, z, the ratio, and what the fraction returned. Every one of these was
# ANSWERED before this change, with the value in the last column.
USED_TO_LEAK = [
    (1.0, 901.0, 3000.0, 630.7, -0.4286691565),
    (0.5, 101.0, 300.0, 134.158438486, 1.098669683),
    (-0.5, 901.0, 10000.0, -1639.52290997, -0.09901589115),
    (100.5, 9001.0, 10000.0, 14.4807353513, -5.546010988),
]


@pytest.mark.parametrize("a, b, z, true_value, used_to_say", USED_TO_LEAK)
def test_the_positive_z_corner_that_used_to_leak(tmp_path, a, b, z, true_value, used_to_say):
    """Silently wrong answers that issue #453's rule let through.

    The ``b*b >= 64*|z*a|`` clause bounds only the odd partial numerators of the
    fraction, which carry ``z*(a+k)``. The even ones carry ``z*(a-b-k)``, so for
    a large b they are about ``z/b`` however small ``z*a`` is, and the clause
    never looked at them. Sixty of 3191 answers were wrong because of it, some
    with the wrong sign, and all of them had a positive z.

    Where the fraction turns is sharp and it is at ``z = b``: over 597 triples it
    is clean through ``z/b = 1.02`` and 12 of 47 are wrong at 1.05. A negative z
    does not need the companion bound and does not get one, measured clean out to
    ``|z|/b = 300``. So the clause now also asks for ``2*z <= b`` when z is
    positive, which is a factor of two under a threshold whose position is
    understood rather than merely observed.

    Two of these four are now answered correctly by the expansion instead. The
    other two are refused. Either is acceptable; returning the last column is
    not.
    """
    assert abs(used_to_say - true_value) / abs(true_value) > 0.9  # it really was wrong
    try:
        got = _value(tmp_path, a, b, z)
    except Exception as excinfo:
        assert "not reliable" in str(excinfo)
        return
    assert got == pytest.approx(true_value, rel=1e-9)


# ── The fraction keeps priority where it is trusted ──────────────────────────

# The expansion is only ever consulted for arguments that were going to be
# refused, so nothing that answers today can change. These are the arguments the
# other mratio tests pin, and they still go through the fraction.
UNCHANGED = [
    (-1000.0, 9001.0, -10000.0, 0.461283283652),
    (2.0, 101.0, -30.0, 0.775054308812),
    (0.5, 101.0, 1.0, 1.00985005864),
    (300.0, 301.0, -5.0, 0.999944079156),
    (-10.0, 21.0, -50.0, 0.27332389077),
    (-100.0, 901.0, -1000.0, 0.46164590186),
]


@pytest.mark.parametrize("a, b, z, expect", UNCHANGED)
def test_the_trusted_region_is_untouched(tmp_path, a, b, z, expect):
    assert _value(tmp_path, a, b, z) == pytest.approx(expect, rel=1e-9)


# ── The two methods meet without a step ──────────────────────────────────────


@pytest.mark.parametrize("a, b", [(0.5, 5.0), (0.5, 9.0)])
def test_the_handover_between_the_two_methods_is_seamless(tmp_path, a, b):
    """Either side of ``|z| = 20``, where the fraction stops and the expansion starts.

    A rate law whose ``z`` rides a species crosses this boundary during a run, so
    a step here would be a discontinuity in the right-hand side that the
    integrator would have to fight. There is none, because both methods are
    right: the two values below differ only by the function's own slope across
    the 1e-4 gap between them.

    Checked against mpmath at 50 digits rather than against each other, since two
    methods agreeing is also what a shared mistake looks like.
    """
    fraction = _value(tmp_path, a, b, -20.0)
    expansion = _value(tmp_path, a, b, -20.0001)
    true_at_20, true_just_past = {
        (0.5, 5.0): (0.21055996499992646, 0.21055908836289424),
        (0.5, 9.0): (0.32214104134640261, 0.32213990742627571),
    }[(a, b)]
    assert fraction == pytest.approx(true_at_20, rel=1e-12)
    assert expansion == pytest.approx(true_just_past, rel=1e-9)
    # No step: the gap between them is the slope, not a change of method.
    assert abs(expansion - fraction) < 1e-5
