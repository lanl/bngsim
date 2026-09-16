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

Issue #549 added the *other* way a finite rate law hands back a non-finite derivative:
a value that is an ordinary number the emitted arithmetic could not reach, because an
``exp`` in it overflowed. The pulse note alone asserted the singularity as the
explanation, and two corpus models that had overflowed instead sent their readers
looking for a pulse that was not there. It is said only where a value actually went
non-finite — an overflow leaves a NaN, and so always leaves a witness, which is
exactly what the restart hint is the absence of.

On this model neither failure happens any more: from the onset on, the solver
integrates the comoving column ``V = S + c·f`` in place of ``S`` (issue #545;
``test_sens_comoving_onset.py``), and its forcing is bounded. The messages are for the
runs that cannot take that column — one with an event, or a crossing moved at a shift
no case was emitted for — so these tests reach them with ``BNGSIM_SENS_COMOVING=0``,
which keeps every column plain. The sensitivity RHS itself still evaluates the true
derivative at the onset; #545 declined a finite value there, and the comoving column
never needs one.
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


def _failure(tmp_path, *, onset, a, rtol, atol, params=("k0", "on"), monkeypatch=None):
    if monkeypatch is not None:
        # The plain column: see the module docstring.
        monkeypatch.setenv("BNGSIM_SENS_COMOVING", "0")
    path = tmp_path / f"pulse_{'ge' if onset == '>=' else 'gt'}_{a}.net"
    path.write_text(PULSE_NET.format(onset=onset, a=a))
    sim = bngsim.Simulator(
        bngsim.Model.from_net(str(path)), method="ode", sensitivity_params=list(params)
    )
    with pytest.raises(Exception) as exc:  # noqa: PT011 - the message is what is under test
        sim.run(t_span=(0.0, 40.0), n_points=41, rtol=rtol, atol=atol)
    return " ".join(str(exc.value).split())


def test_the_onset_derivative_names_its_column_not_a_species(tmp_path, monkeypatch):
    """``on`` is the second column, so the name has to come from the column that
    went non-finite rather than from the first one."""
    msg = _failure(tmp_path, onset=">=", a=1.5, rtol=1e-8, atol=1e-8, monkeypatch=monkeypatch)
    assert "The compiled sensitivity RHS returned a non-finite value at t=10" in msg, msg
    assert "Every rate law and every species is finite there" in msg, msg
    assert "the sensitivity column for parameter 'on' (row X())" in msg, msg
    assert "The non-finite half is ∂f/∂on itself" in msg, msg
    assert "(issue #545)" in msg, msg
    assert "Constrain the species" not in msg, msg
    assert "'k0'" not in msg, msg  # only the column that went non-finite
    # ...and the other way a derivative goes non-finite, so the note above reads as
    # one of two explanations rather than as the diagnosis (issue #549).
    assert "The other way is a value that is perfectly ordinary" in msg, msg
    assert "(issue #549)" in msg, msg


@pytest.mark.parametrize(
    ("a", "rtol", "atol"),
    [(1.1, 1e-8, 1e-12), (1.2, 1e-8, 1e-8)],
    ids=["a=1.1 (stalls)", "a=1.2 (fails the error test)"],
)
def test_a_step_that_gives_out_past_a_strict_onset_says_it_had_just_restarted(
    tmp_path, monkeypatch, a, rtol, atol
):
    """No value goes non-finite on this side of the onset, so no witness is left to
    explain it; the restart is. Asserted whichever way the step gives out — the
    stall and the error-test failure both carry the note — because which of the two
    a given platform hits is not what the issue is about."""
    msg = _failure(tmp_path, onset=">", a=a, rtol=rtol, atol=atol, monkeypatch=monkeypatch)
    assert "made no progress" in msg or "CV_ERR_FAILURE" in msg, msg
    assert RESTART_HINT in msg, msg
    assert "(issue #545)" in msg, msg
    # Every value stayed finite here, so an overflow — which would have left a NaN
    # and a witness — is not on the table, and saying so would be the same
    # misdirection #549 removed, one note over.
    assert "(issue #549)" not in msg, msg


def test_a_rate_law_that_is_itself_infinite_keeps_the_domain_advice(tmp_path):
    """The counterpart: with ``a < 1`` the rate law is infinite wherever ``s`` is 0,
    so the value right-hand side fails first, and its message is about the law."""
    msg = _failure(tmp_path, onset=">=", a=0.5, rtol=1e-8, atol=1e-8)
    assert "Constrain the species, or write the law so it is defined there" in msg, msg
    assert "Every rate law and every species is finite there" not in msg, msg
    assert "issue #545" not in msg, msg
