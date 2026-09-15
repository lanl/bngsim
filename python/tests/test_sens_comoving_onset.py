"""Issue #545: forward sensitivity past the onset of a pulse that rises from zero.

``X' = k(t)·X`` with ``k = k0 + k1·g·s^(a-1)·(1-s)`` over a window ``s = (t - on)/D``
that is 0 outside it, and ``g = 1`` or ``1 + on/10``. For ``1 < a < 2`` the rate law is
continuous, but ``∂f/∂on`` goes as ``s^(a-2)``: unbounded past the onset, where a plain
sensitivity column stalls or finishes wrong. From the onset on, the solver integrates
``V = S + c·f`` instead, with ``c = ∂t*/∂on = 1``. Its forcing is the derivative of ``f``
along ``on`` and the clock together, which is bounded, and ``S = V − c·f`` wherever ``S``
is read.

Every column is checked against the closed form: ``ln X = k0·t + k1·g·D·B(U)``, with
``U = clip((t-on)/D, 0, 1)`` and ``B(U) = U^a/a − U^(a+1)/(a+1)``. On main every run
below with ``a < 2`` raised.
"""

from __future__ import annotations

import re

import bngsim
import numpy as np
import pytest

K0, K1, D = 0.1, 1.0, 20.0
PARAMS = ("k0", "on", "a", "D")

TIME_NET = """\
begin parameters
    1 k0  0.1
    2 k1  1
    3 a   {a}
    4 on  10
    5 D   20
end parameters
begin functions
    1 s() if(((time(){onset}on)&&(time()<=(on+D))),((time()-on)/D),0)
    2 k() k0+((k1*{height}(s()^(a-1)))*(1-s()))
end functions
begin species
    1 X() 1
end species
begin reactions
    1 1 1,1 k #_R1
end reactions
"""

# BNG2.pl 2.9.3's network for the same pulse on a counter clock, `0 -> counter() 1`
# read back as the observable `t`: how a BNGL model keeps time (SIR_v4's shape).
COUNTER_NET = """\
begin parameters
    1 k0         0.1
    2 k1         1.0
    3 a          {a}
    4 on         10.0
    5 D          20.0
    6 _rateLaw1  1
end parameters
begin functions
    1 s() if(((t{onset}on)&&(t<=(on+D))),((t-on)/D),0)
    2 k() k0+((k1*{height}(s()^(a-1)))*(1-s()))
end functions
begin species
    1 X() 1.0
    2 counter() 0
end species
begin reactions
    1 1 1,1 k #_R1
    2 0 2 _rateLaw1 #_R2
end reactions
begin groups
    1 t                    2
end groups
"""


def _pulse_terms(t, a, start, g=1.0):
    """``(B, pulse, dB/da, dU/dD)`` of one window opening at ``start``."""
    n = a - 1.0
    inside = (t >= start) & (t <= start + D)
    U = np.clip((t - start) / D, 0.0, 1.0)
    logU = np.log(np.where(U > 0, U, 1.0))
    B = U ** (n + 1) / (n + 1) - U ** (n + 2) / (n + 2)
    dB_da = np.where(
        U > 0,
        U ** (n + 1) * (logU / (n + 1) - 1 / (n + 1) ** 2)
        - U ** (n + 2) * (logU / (n + 2) - 1 / (n + 2) ** 2),
        0.0,
    )
    return (
        B,
        np.where(inside, U**n * (1.0 - U), 0.0),
        dB_da,
        np.where(inside, -(t - start) / D**2, 0.0),
    )


def closed_form(a, scaled, t, on=10.0):
    g, dg = (1.0 + on / 10.0, 0.1) if scaled else (1.0, 0.0)
    B, pulse, dB_da, dU_dD = _pulse_terms(t, a, on)
    X = np.exp(K0 * t + K1 * g * D * B)
    dlnX = {
        "k0": t,
        "on": K1 * dg * D * B - K1 * g * pulse,
        "a": K1 * g * D * dB_da,
        "D": K1 * g * (B + D * pulse * dU_dD),
    }
    return {p: X * v for p, v in dlnX.items()}


def _simulator(tmp_path, template, *, onset, a, scaled):
    path = tmp_path / "pulse.net"
    path.write_text(template.format(onset=onset, a=a, height="(1+(on/10))*" if scaled else ""))
    model = bngsim.Model.from_net(str(path))
    return model, bngsim.Simulator(model, method="ode", sensitivity_params=list(PARAMS))


def _worst(model, result, exact, params=PARAMS, species="X()"):
    i = list(model.species_names).index(species)
    S = np.asarray(result.sensitivities)
    return {
        p: float(np.max(np.abs(S[:, i, j] - exact[p])) / np.max(np.abs(exact[p])))
        for j, p in enumerate(params)
    }


@pytest.mark.parametrize("scaled", [False, True], ids=["height k1", "height k1*(1+on/10)"])
@pytest.mark.parametrize("a", [1.1, 1.5])
@pytest.mark.parametrize("onset", [">=", ">"])
@pytest.mark.parametrize("clock", ["time", "counter"])
def test_every_column_matches_the_closed_form_past_the_onset(tmp_path, clock, onset, a, scaled):
    model, sim = _simulator(
        tmp_path, TIME_NET if clock == "time" else COUNTER_NET, onset=onset, a=a, scaled=scaled
    )
    result = sim.run(t_span=(0.0, 40.0), n_points=41, rtol=1e-8, atol=1e-10)
    assert sim.has_analytic_sens_rhs, sim.sens_rhs_decline_reason
    errors = _worst(model, result, closed_form(a, scaled, np.asarray(result.time)))
    # The run's own accuracy, not a loose bound: the smooth pulse (a = 2.5) lands
    # within 5e-7 of the closed form at this tolerance, and so does every cell here.
    assert max(errors.values()) < 1e-5, errors


def test_the_same_run_with_every_column_plain_still_fails(tmp_path, monkeypatch):
    """The A/B: the comoving column is what carries the run past the onset."""
    model, sim = _simulator(tmp_path, TIME_NET, onset=">", a=1.2, scaled=False)
    sim.run(t_span=(0.0, 40.0), n_points=41, rtol=1e-8, atol=1e-10)
    monkeypatch.setenv("BNGSIM_SENS_COMOVING", "0")
    model.reset()
    with pytest.raises(Exception):  # noqa: PT011, B017 - how it fails is #547's subject
        sim.run(t_span=(0.0, 40.0), n_points=41, rtol=1e-8, atol=1e-10)


def test_a_run_that_ends_inside_the_window_carries_s_not_v(tmp_path):
    """A run ending at t=20 ends with the column still comoving. What it hands the
    next run is S: seeded with V, the t=20 row would be off by k(20)·X(20)."""
    model, sim = _simulator(tmp_path, TIME_NET, onset=">=", a=1.5, scaled=True)
    sim.run(t_span=(0.0, 20.0), n_points=21, rtol=1e-8, atol=1e-10)
    result = sim.run(
        t_span=(20.0, 40.0), n_points=21, rtol=1e-8, atol=1e-10, carry_sensitivities=True
    )
    errors = _worst(model, result, closed_form(1.5, True, np.asarray(result.time)))
    assert max(errors.values()) < 1e-5, errors


# The same pulse through libSBML, which is where an event can be written.
PULSE_ANTIMONY = """\
model pulse
  species X = 1;
  k0 = {k0}; k1 = {k1}; a = {a}; on = 10; D = {D};
  s := piecewise((time - on)/D, (time >= on) && (time <= on + D), 0);
  J0: => X; (k0 + k1*s^(a - 1)*(1 - s))*X;
{event}end
"""


def test_an_event_keeps_every_column_plain():
    """A model with an event gets no comoving column at all. The event jump
    differentiates at the pre-event state, which `V` is not, so rather than compose
    the two bngsim leaves every column plain — and the onset column then meets the
    forcing again. The user guide says so under "the model has no events"."""
    a = 1.2
    model = bngsim.Model.from_antimony_string(
        PULSE_ANTIMONY.format(k0=K0, k1=K1, a=a, D=D, event="")
    )
    assert model.n_events == 0
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(PARAMS))
    result = sim.run(t_span=(0.0, 40.0), n_points=41, rtol=1e-8, atol=1e-10)
    errors = _worst(model, result, closed_form(a, False, np.asarray(result.time)), species="X")
    assert max(errors.values()) < 1e-5, errors

    # One event, firing past the window and assigning X to itself, changes no value
    # this run computes. It is still enough to keep every column plain.
    evented = bngsim.Model.from_antimony_string(
        PULSE_ANTIMONY.format(k0=K0, k1=K1, a=a, D=D, event="  E1: at (time > 35): X = X;\n")
    )
    assert evented.n_events == 1
    sim = bngsim.Simulator(evented, method="ode", sensitivity_params=list(PARAMS))
    with pytest.raises(Exception):  # noqa: PT011, B017 - how it fails is #547's subject
        sim.run(t_span=(0.0, 40.0), n_points=41, rtol=1e-8, atol=1e-10)


# A window that opens on a condition reading live state: `t - Z >= on`, with `Z` a
# species nothing produces, so the window opens at `on` and moves with it one for
# one. bngsim cannot solve that crossing for a time, so it is the issue #150 path —
# a root CVODE has to reach — and the column enters the comoving frame there rather
# than at a crossing it stopped on.
STATE_SWITCH_NET = """\
begin parameters
    1 k0  0.1
    2 k1  1
    3 a   {a}
    4 on  10
    5 D   20
end parameters
begin functions
    1 s() if((((time()-Zo)>=on)&&(time()<=(on+D))),((time()-on)/D),0)
    2 k() k0+((k1*(s()^(a-1)))*(1-s()))
end functions
begin species
    1 X() 1
    2 Z() 0
end species
begin reactions
    1 1 1,1 k #_R1
end reactions
begin groups
    1 Zo                   2
end groups
"""


def test_a_window_opening_on_a_state_switch_enters_the_frame_at_its_root(tmp_path, monkeypatch):
    """`a = 1.9` keeps the forcing mild enough that CVODE accepts the step that
    brackets the root. Sharper than that and no step spanning the onset passes the
    error test, so the run wedges one ulp short of a root it never reaches — which
    no column, comoving or plain, can do anything about, and which is why the clock
    guards below are recognized as crossings instead."""
    from bngsim._switch_sensitivity import state_switch_conditions

    a = 1.9
    path = tmp_path / "state_pulse.net"
    path.write_text(STATE_SWITCH_NET.format(a=a))
    model = bngsim.Model.from_net(str(path))
    assert state_switch_conditions(model._core) == ["(time()-Zo)>=on"]
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(PARAMS))
    result = sim.run(t_span=(0.0, 40.0), n_points=41, rtol=1e-8, atol=1e-10)
    assert sim.has_analytic_sens_rhs, sim.sens_rhs_decline_reason
    errors = _worst(model, result, closed_form(a, False, np.asarray(result.time)))
    assert max(errors.values()) < 1e-5, errors

    monkeypatch.setenv("BNGSIM_SENS_COMOVING", "0")
    model.reset()
    with pytest.raises(Exception):  # noqa: PT011, B017 - how it fails is #547's subject
        sim.run(t_span=(0.0, 40.0), n_points=41, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    ("law", "has_case"),
    [
        ("k0+((k1*(s()^(a-1)))*(1-s()))", True),
        ("k0+((k1*(s()^0.5))*(1-s()))", True),
        ("k0+if((time()>=on),(k1*((1-exp((on-time())))^(a-1))),0)", True),
        ("if((time()>=on),k1,k0)", False),
        ("k0+((k1*(s()^1.5))*(1-s()))", False),
        ("k0+if((time()>=on),(k1*((1+exp((on-time())))^(-a))),0)", False),
        ("k0+(k1*exp((-((time()-on)^2.0))))", False),
    ],
    ids=["s^(a-1)", "s^0.5", "(1-exp)^(a-1)", "step", "s^1.5", "logistic^(-a)", "gaussian"],
)
def test_only_a_power_singular_at_its_crossing_gets_a_case(tmp_path, law, has_case):
    """A case is for a power whose base can reach 0 with an exponent that makes its
    derivative blow up there. A step has no power; ``s^1.5`` has a bounded
    derivative; a logistic's base is never 0; a Gaussian's ``^2.0`` is a square.
    Each of those keeps the source it had."""
    from bngsim._codegen import generate_sens_from_model

    path = tmp_path / "law.net"
    path.write_text(
        TIME_NET.format(onset=">=", a=1.5, height="").replace("k0+((k1*(s()^(a-1)))*(1-s()))", law)
    )
    source = generate_sens_from_model(
        bngsim.Model.from_net(str(path)), functional=True, emit_term_scale=True
    )
    assert source is not None
    if has_case:
        # `on` is parameter 3, and its one case shifts the clock at c = 1.
        assert re.search(r"case 3:\n\s+if \(k == 0\) \{ \*c_out = 1\.0; return \d+; \}", source)
    else:
        assert "comoving" not in source


# Two seasons on a counter clock, the second opening through a year selection:
# `start()` is `d1` in the first year and `365+d2` after it, which is how SIR_v4
# writes its onsets. The condition reads the clock on both sides, so until #545 it
# went to the issue #150 state path, which roots on the crossing — and CVODE never
# accepts a step to that root when the rate past it rises as s^(a-1).
SEASONS_NET = """\
begin parameters
    1 k0         0.01
    2 k1         0.05
    3 a          {a}
    4 d1         100
    5 d2         132
    6 D          20
    7 _rateLaw1  1
end parameters
begin functions
    1 start() if((t<365),d1,(365+d2))
    2 s() if(((t>start())&&(t<=(start()+D))),((t-start())/D),0)
    3 k() k0+((k1*(s()^(a-1)))*(1-s()))
end functions
begin species
    1 X() 1.0
    2 counter() 0
end species
begin reactions
    1 1 1,1 k #_R1
    2 0 2 _rateLaw1 #_R2
end reactions
begin groups
    1 t                    2
end groups
"""


def test_an_onset_written_through_a_year_selection_is_a_crossing_it_stops_on(tmp_path):
    from bngsim._switch_sensitivity import compute_switch_time_sens, state_switch_conditions

    a = 1.2
    path = tmp_path / "seasons.net"
    path.write_text(SEASONS_NET.format(a=a))
    model = bngsim.Model.from_net(str(path))
    assert state_switch_conditions(model._core) == []
    records, _pinned = compute_switch_time_sens(model._core, ["d2"], 0.0, 700.0, True)
    assert [(round(r[0], 9), r[3]) for r in records if r[3] != [0.0]] == [
        (497.0, [1.0]),
        (517.0, [1.0]),
    ]

    params = ("k0", "d1", "d2", "a")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    result = sim.run(t_span=(0.0, 700.0), n_points=701, rtol=1e-8, atol=1e-10)
    t = np.asarray(result.time)
    k0, k1 = 0.01, 0.05
    B1, pulse1, dB1, _ = _pulse_terms(t, a, 100.0)
    B2, pulse2, dB2, _ = _pulse_terms(t, a, 497.0)
    X = np.exp(k0 * t + k1 * D * (B1 + B2))
    exact = {
        "k0": X * t,
        "d1": X * (-k1 * pulse1),
        "d2": X * (-k1 * pulse2),
        "a": X * (k1 * D * (dB1 + dB2)),
    }
    errors = _worst(model, result, exact, params)
    assert max(errors.values()) < 1e-5, errors
