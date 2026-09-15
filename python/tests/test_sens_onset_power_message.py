"""Issue #545: a forward-sensitivity run that fails on a derivative says so.

The model is ``X' = k(t)·X`` with ``k = k0 + k1·s^(a-1)·(1-s)`` over a window
``s = (t - on)/D`` that is 0 outside it. For ``1 < a < 2`` the rate law is finite and
continuous everywhere, but its derivative with respect to the onset goes as
``s^(a-2)``: infinite at the onset, and unbounded just past it. Both failures that
follow used to be reported as something else:

* **Opening on** ``time() >= on``, bngsim stops on the crossing and restarts there
  (issue #305), so the first sensitivity RHS call is at the onset. The run stopped
  with ``CV_FIRST_SRHSFUNC_ERR`` and the advice to constrain a species — though every
  species and every rate law is finite there.
* **Opening on** ``time() > on`` keeps that instant off the pulse, but the unbounded
  forcing just past it still collapses the step. The run said "made no progress" and
  blamed an ``if()`` rate jump, or failed the error test and said nothing at all.

Now the first names the sensitivity column and says ``∂f/∂on`` is the non-finite half,
and the second says the step gave out where the run had just restarted. Both end with
the shape and its remedies.

The first failure stays a failure: the sensitivity RHS evaluates the true derivative at
the onset. A finite value there would remove it but not the second, and on some models
would turn a run that fails into one that finishes with a wrong column and no warning,
so #545 declined it.
"""

from __future__ import annotations

import bngsim
import pytest

# BNG2.pl 2.9.3's network for the issue's pulse.bngl, the onset comparison left open.
PULSE_NET = """\
begin parameters
    1 k0  0.1  # Constant
    2 k1  1  # Constant
    3 a   {a}  # Constant
    4 on  10  # Constant
    5 D   20  # Constant
end parameters
begin functions
    1 s() if(((time(){onset}on)&&(time()<=(on+D))),((time()-on)/D),0)
    2 k() k0+((k1*(s()^(a-1)))*(1-s()))
end functions
begin species
    1 X() 1
end species
begin reactions
    1 1 1,1 k #_R1
end reactions
"""

RESTART_HINT = "the step gave out where it had just restarted, at t=10"


def _failure(tmp_path, *, onset, a, rtol, atol, params=("k0", "on")):
    path = tmp_path / f"pulse_{'ge' if onset == '>=' else 'gt'}_{a}.net"
    path.write_text(PULSE_NET.format(onset=onset, a=a))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=list(params)
    )
    with pytest.raises(Exception) as exc:  # noqa: PT011 - the message is what is under test
        sim.run(t_span=(0.0, 40.0), n_points=41, rtol=rtol, atol=atol)
    return " ".join(str(exc.value).split())


def test_the_onset_derivative_names_its_column_not_a_species(tmp_path):
    """``on`` is the second column, so the name has to come from the column that
    went non-finite rather than from the first one."""
    msg = _failure(tmp_path, onset=">=", a=1.5, rtol=1e-8, atol=1e-8)
    assert "The compiled sensitivity RHS returned a non-finite value at t=10" in msg, msg
    assert "Every rate law and every species is finite there" in msg, msg
    assert "the sensitivity column for parameter 'on' (row X())" in msg, msg
    assert "The non-finite half is ∂f/∂on itself" in msg, msg
    assert "(issue #545)" in msg, msg
    assert "Constrain the species" not in msg, msg
    assert "'k0'" not in msg, msg  # only the column that went non-finite


@pytest.mark.parametrize(
    ("a", "rtol", "atol"),
    [(1.1, 1e-8, 1e-12), (1.2, 1e-8, 1e-8)],
    ids=["a=1.1 (stalls)", "a=1.2 (fails the error test)"],
)
def test_a_step_that_gives_out_past_a_strict_onset_says_it_had_just_restarted(
    tmp_path, a, rtol, atol
):
    """No value goes non-finite on this side of the onset, so no witness is left to
    explain it; the restart is. Asserted whichever way the step gives out — the
    stall and the error-test failure both carry the note — because which of the two
    a given platform hits is not what the issue is about."""
    msg = _failure(tmp_path, onset=">", a=a, rtol=rtol, atol=atol)
    assert "made no progress" in msg or "CV_ERR_FAILURE" in msg, msg
    assert RESTART_HINT in msg, msg
    assert "(issue #545)" in msg, msg


def test_a_rate_law_that_is_itself_infinite_keeps_the_domain_advice(tmp_path):
    """The counterpart: with ``a < 1`` the rate law is infinite wherever ``s`` is 0,
    so the value right-hand side fails first, and its message is about the law."""
    msg = _failure(tmp_path, onset=">=", a=0.5, rtol=1e-8, atol=1e-8)
    assert "Constrain the species, or write the law so it is defined there" in msg, msg
    assert "Every rate law and every species is finite there" not in msg, msg
    assert "issue #545" not in msg, msg
