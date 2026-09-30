"""Every forward-sensitivity term against a central difference, one term per model.

Issue #788 tracks sensitivity terms that are missing or wrong at events, switches,
clocks, comoving frames and initial-condition seeds. Most are separate terms, so
the family's shared asset is a derivation (``docs/development/sensitivity-terms.md``)
and an oracle: a model on which the term is the only one in play, and a central
difference of plain runs — runs with no sensitivities, which never reach the
jump, clock or comoving code — to hold its column to. ``_fd_sens.py`` is that
difference, with its step chosen from the measured noise floor and a refusal at
the two places issue #368 showed a difference misleads.

Each term below is a :class:`Term`: the model text with the column's parameter
left as a slot, filled with the model's own spelling for the analytic run and
with a literal value for each difference run. A literal is also what a derived
parameter's column means (the free axis a pinned value makes real), so the same
substitution serves primaries, derived parameters and initial values alike, and
the reference never calls ``set_param``, which issue #708 is about.

A term that agrees is a plain test. A term with an open defect is a strict xfail
naming the issue, after that issue's documented symptom was reproduced; the
symptom each one shows is in the Term's ``note``. Every analytic run passes
``codegen=True`` and asserts the backend that built it, so the MIR leg checks
the JIT as well as ``cc``, and both ``sensitivity_method`` arms run.
"""

from __future__ import annotations

import functools
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field

import bngsim
import numpy as np
import pytest
from bngsim import SimulationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _fd_sens import (  # noqa: E402
    CentralDifference,
    ReferenceRefused,
    central_difference,
    mismatches,
)

_BACKEND = "mir" if os.environ.get("BNGSIM_CODEGEN_JIT", "").strip().lower() == "mir" else "cc"
METHODS = ("staggered", "simultaneous")

# The difference runs a decade and more tighter than any analytic column it
# judges, so the floor being divided by the step is the oracle's, not the
# column's (the #368 lesson).
_FD_RTOL = 1e-12
_MAX_STEPS = 10**6


@dataclass(frozen=True)
class Term:
    """One column on one model.

    ``text`` holds ``{slot}`` where the column's parameter (or, for an ``ic``
    column, the species' initial value) is written. ``spelling`` fills it for the
    analytic run and ``value`` is what it evaluates to there; each difference
    run writes a literal instead. ``compare`` picks the sample rows to judge,
    for a model whose defect lives next to a sample that sits on a crossing.
    ``refused`` is for a defect that shows as a refusal rather than a number:
    an analytic run failing with a message it matches raises
    :class:`AnalyticRunRefused`, and nothing else does.
    """

    id: str
    loader: str  # "antimony" | "net"
    text: str
    slot: str
    spelling: str
    value: float
    sample_times: tuple[float, ...]
    species: tuple[str, ...]
    param: str | None = None  # the sensitivity_params name; None for an ic column
    ic: str | None = None  # the sensitivity_ic species
    scale: float | None = None
    rtol: float = 1e-10
    atol: float = 1e-12
    compare: tuple[int, ...] | None = None
    refused: str | None = None
    note: str = ""
    analytic: Callable[[Term, str], np.ndarray] | None = field(default=None, compare=False)

    def model(self, fill: str) -> bngsim.Model:
        text = self.text.format(**{self.slot: fill})
        if self.loader == "antimony":
            return bngsim.Model.from_antimony_string(text)
        return _net_model(text)


# Removed when the interpreter exits. One file per distinct text, so no two
# models in a run are ever loaded from the same path.
_NET_DIR = tempfile.TemporaryDirectory(prefix="sens_fd_oracle_")


def _net_model(text: str) -> bngsim.Model:
    path = os.path.join(_NET_DIR.name, f"m{abs(hash(text)):x}.net")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return bngsim.Model.from_net(path)


class AnalyticRunRefused(Exception):
    """The analytic run failed with the message its Term's ``refused`` names.

    Its own class, so a strict xfail for a refusal is satisfied by that run's
    failure and not by any other: a reference run's error, or a Simulator that
    could not be built, is a different exception and fails the test.
    """


def _rows(term: Term, names: list[str]) -> list[int]:
    return [names.index(s) for s in term.species]


def _plain(term: Term, value: float, rtol: float) -> np.ndarray:
    """One difference run: a fresh model with the slot written as a literal.

    ``codegen=False`` so the reference is integrated by the interpreter, the path
    the sensitivity machinery never runs on.
    """
    model = term.model(repr(float(value)))
    r = bngsim.Simulator(model, method="ode", codegen=False).run(
        sample_times=list(term.sample_times), rtol=rtol, atol=rtol * 1e-2, max_steps=_MAX_STEPS
    )
    return np.asarray(r.species)[:, _rows(term, list(r.species_names))]


@functools.cache
def _reference(term: Term) -> CentralDifference:
    return central_difference(
        lambda v, rtol: _plain(term, v, rtol), term.value, rtol=_FD_RTOL, scale=term.scale
    )


def _simulator(term: Term, model: bngsim.Model, method: str) -> bngsim.Simulator:
    sim = bngsim.Simulator(
        model,
        method="ode",
        sensitivity_params=[term.param] if term.param else None,
        sensitivity_ic=[term.ic] if term.ic else None,
        sensitivity_method=method,
        codegen=True,
    )
    if sim.codegen_backend != _BACKEND:
        # Not an assert: a strict xfail names raises=AssertionError, and a wrong
        # backend must fail it rather than pass for the defect it names.
        pytest.fail(f"{term.id}: built on {sim.codegen_backend!r}, expected {_BACKEND!r}")
    return sim


def _column(term: Term, result) -> np.ndarray:
    S = np.asarray(result.sensitivities_ic if term.ic else result.sensitivities)
    return S[:, _rows(term, list(result.species_names)), 0]


def _analytic(term: Term, method: str) -> np.ndarray:
    if term.analytic is not None:
        return term.analytic(term, method)
    sim = _simulator(term, term.model(term.spelling), method)
    try:
        r = sim.run(
            sample_times=list(term.sample_times),
            rtol=term.rtol,
            atol=term.atol,
            max_steps=_MAX_STEPS,
        )
    except SimulationError as e:
        if term.refused and re.search(term.refused, str(e)):
            raise AnalyticRunRefused(str(e)) from e
        raise
    return _column(term, r)


def _check(term: Term, method: str) -> None:
    fd = _reference(term)
    S = _analytic(term, method)
    rows = list(term.compare) if term.compare is not None else list(range(len(term.sample_times)))
    sub = CentralDifference(
        fd.value[rows], fd.err[rows], fd.step[rows], fd.kink[rows], fd.noise[rows], fd.typical
    )
    times = [term.sample_times[i] for i in rows]

    def label(cell: tuple) -> str:
        return f"{term.species[int(cell[1])]} at t={times[int(cell[0])]}"

    bad = mismatches(S[rows], sub, term.value, rtol=term.rtol, atol=term.atol, labels=label)
    assert not bad, (
        f"{term.id} [{method}]: d/d{term.param or term.ic} disagrees with the central "
        "difference of plain runs:\n  " + "\n  ".join(bad)
    )


def _xfail(issue: int, why: str, raises=AssertionError):
    return pytest.mark.xfail(reason=f"issue #{issue}: {why}", strict=True, raises=raises)


# ─── Models ─────────────────────────────────────────────────────────────────
# Kept as small as the term allows: every other term is absent, so a column that
# disagrees names its term. Parameter values are the ones the issues report.
# Every value a term might vary is a ``{placeholder}``; :func:`_fill` writes all
# but the column's own, which stays a slot for the analytic spelling and the
# difference's literals.


def _fill(template: str, slot: str, **fill: str) -> str:
    return template.format(**{**fill, slot: "{" + slot + "}"})


_MM = """
species S = 10, P = 0
Vm = 2; Km = {Km}
R: S -> P; Vm*S/(Km + S)
"""

_IC_NET = """begin parameters
    1 R0   {R0}
    2 kd   0.5
    3 Rt   {Rt}
end parameters
begin species
    1 B() {B0}
end species
begin reactions
    1 1 0 kd
end reactions
"""

_DERIVED_RATE_NET = """begin parameters
    1 k0   {k0}
    2 kd   {kd}
end parameters
begin functions
    1 rf() kd*{body}
end functions
begin species
    1 A() 10
end species
begin reactions
    1 1 0 rf
end reactions
begin groups
    1 Atot 1
end groups
"""

_EVENT_FIXED = """
species X = 10
k = {k}; c = 1
J: X -> ; k*X
E: at (time >= 2): X = X/2 + c
"""

_EVENT_TIME = """
species A = 1, B = 0
k = 0.5; T0 = {T0}
J: A -> ; k*A
E: at (time >= T0): B = {body}
"""

_EVENT_STATE = """
species A = 1, B = 0
k = {k}; thr = 0.5
J: A -> ; k*A
E: at (A < thr): B = {body}
"""

# The reset lands below the threshold, so the trigger stays true and the event
# fires once.
_EVENT_RESET = """
species A = 10
k = 0.5; thr = {thr}
J: A -> ; k*A
E: at (A < thr): A = A/2 - 1
"""

_RATEOF = """
species A, Y
A = A0; Y = 0; A0 = 10; a = {a}; tau = 3
J0: A -> ; a*A
E: at (time >= tau): Y = {body}
"""

_BATCH = """
species X; X = 0; p = {p}
J0: -> X; p
{first}
{second}
"""
_ELO = "Elo: at (time >= 2), priority = 1: X = 3*p"
_EHI = "Ehi: at (time >= 2), priority = 2: X = 5*p"

_T0_FIRE = """
species A = 10, B = 0
k1 = {k1}
R1: A -> B; k1*A
E1: at (time >= {fire}), t0=false: A = 8
"""

_COINCIDENT = """
species X, Y; X = 0; Y = 0; a = 2; k = 0.5; tau = {tau}
J0: -> X; a
J1: -> Y; piecewise(k*X, time >= {switch}, 0)
E1: at (time >= tau): X = 0
"""

_TIME_SWITCH_NET = """begin parameters
    1 tau {tau}
    2 k   2
end parameters
begin functions
    1 r() if(time()>=tau,k,0)
end functions
begin species
    1 X() 0
end species
begin reactions
    1 0 1 r
end reactions
"""

_STATE_SWITCH_NET = """begin parameters
    1 A0    {A0}
    2 a     0.5
    3 thr   2
    4 kb    3
    5 B0    {B0}
    6 kdeg  0.1
end parameters
begin functions
    1 fY() if(Aobs<thr,kb,0)
end functions
begin species
    1 A() A0
    2 Y() 0
    3 B() B0
end species
begin reactions
    1 1 0 a
    2 0 2 fY
    3 3 0 kdeg
end reactions
begin groups
    1 Aobs 1
end groups
"""

# A counter clock: `0 -> C() rc` with rc = 1 is BNGL's way of reading time,
# through a group conventionally called `t`. No source species: with one, the
# RHS probe that recognizes a clock perturbs the source too and sees no clock.
_COUNTER_NET = """begin parameters
    1 c0     {c0}
    2 rc     {rc}
    3 sigma  {sigma}
    4 k      2
end parameters
begin functions
    1 rate_X() if(t>=sigma,k,0)
end functions
begin species
    1 C() {C0}
    2 X() 0
end species
begin reactions
    1 0 1 rc
    2 0 2 rate_X
end reactions
begin groups
    1 t 1
end groups
"""

_FAKE_CLOCK_NET = """begin parameters
    1 kf    1
    2 kr    1
    3 thr   {thr}
    4 kin   2
    5 kd    0.3
    6 A0    {A0}
end parameters
begin functions
    1 pulse() if(Bo>=thr,kin,0)
end functions
begin species
    1 A() A0
    2 B() 0
    3 Y() 0
end species
begin reactions
    1 1 2 kf
    2 2 1 kr
    3 0 3 pulse
    4 3 0 kd
end reactions
begin groups
    1 Bo 2
end groups
"""

# Issue #545's pulse on a counter clock: k0 + k1·s^(a-1)·(1-s) over s = (t-on)/D.
_PULSE_NET = """begin parameters
    1 k0    0.1
    2 k1    2.0
    3 a     {a}
    4 lam   3.0
    5 on    {on}
    6 D     {D}
    7 kdeg  0.3
    8 k2    0.5
    9 _rateLaw1 1
end parameters
begin functions
    1 s() (t-on)/D
    2 prod() k0+if(t>=on,if(t<=(on+D),k1*{shape},0),0)
end functions
begin species
    1 X() 0
    2 Tc() 0
end species
begin reactions
    1 0 1 prod
    2 1 0 kdeg
    3 0 2 _rateLaw1
{extra}end reactions
begin groups
    1 t 2
end groups
"""
_OPENING = "(s()^(a-1))*(1-s())"
_CLOSING = "s()*((1-s())^(a-1))"
_CLOCK_FED = "    4 2 2,1 k2\n"


def _pulse(slot: str, **fill: str) -> str:
    return _fill(
        _PULSE_NET,
        slot,
        **{"a": "1.8", "on": "3.0", "D": "4.0", "shape": _OPENING, "extra": "", **fill},
    )


def _reused_after_override(term: Term, method: str) -> np.ndarray:
    """Issue #708's fitting loop: one Simulator, a derived parameter overridden
    between runs. ``k2 = 2*k1`` on the first run and pinned on the second."""
    model = term.model("2*k1")
    sim = _simulator(term, model, method)
    sim.run(sample_times=list(term.sample_times), rtol=term.rtol, atol=term.atol)
    model.reset()
    model.set_param("k2", term.value)
    model.reset()
    r = sim.run(sample_times=list(term.sample_times), rtol=term.rtol, atol=term.atol)
    return _column(term, r)


_STALE = "species A = 10, B = 0; k1 = 1; k2 = {k2}; R1: A -> B; k2*A*A"
_T = (0.0, 0.5, 1.0, 2.0, 3.0)
_PULSE_T = (0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0)  # off the onset (3) and close (7)
_SWITCH_T = (0.0, 1.0, 2.0, 2.5, 4.0, 5.0, 6.0)  # off tau = 3 and tau + 0.5

TERMS: list = [
    # ── The seed and the variational RHS ─────────────────────────────────────
    Term(
        "rhs-dfdp",
        "antimony",
        _MM,
        "Km",
        "3.0",
        3.0,
        (0.0, 1.0, 2.0, 4.0, 8.0),
        ("S", "P"),
        param="Km",
        note="∂f/∂p through a nonlinear law, nothing else",
    ),
    Term(
        "ic-seed-param",
        "net",
        _fill(_IC_NET, "R0", Rt="3*R0", B0="R0"),
        "R0",
        "10",
        10.0,
        _T,
        ("B()",),
        param="R0",
        note="∂x(0)/∂p for a species seeded by a primary",
    ),
    Term(
        "ic-seed-axis",
        "net",
        _fill(_IC_NET, "B0", R0="10", Rt="3*R0"),
        "B0",
        "10",
        10.0,
        _T,
        ("B()",),
        ic="B()",
        note="the sensitivity_ic axis: an identity seed",
    ),
    Term(
        "ic-seed-derived-via-primary",
        "net",
        _fill(_IC_NET, "R0", Rt="3*R0", B0="Rt"),
        "R0",
        "10",
        10.0,
        _T,
        ("B()",),
        param="R0",
        note="a primary reaching an initial value through a derived parameter (#43)",
    ),
    Term(
        "ic-seed-derived-own-column",
        "net",
        _fill(_IC_NET, "Rt", R0="10", B0="Rt"),
        "Rt",
        "3*R0",
        30.0,
        _T,
        ("B()",),
        param="Rt",
        note="the derived IC parameter's own column (#715)",
    ),
    Term(
        "rhs-derived-chain",
        "net",
        _fill(_DERIVED_RATE_NET, "k0", kd="2*k0", body="1"),
        "k0",
        "0.5",
        0.5,
        _T,
        ("A()",),
        param="k0",
        note="∂f/∂p chained through a derived rate constant (#2, #188)",
    ),
    Term(
        "rhs-derived-own-column",
        "net",
        _fill(_DERIVED_RATE_NET, "kd", k0="0.5", body="1"),
        "kd",
        "2*k0",
        1.0,
        _T,
        ("A()",),
        param="kd",
        note="a derived parameter's own column on the analytic RHS",
    ),
    Term(
        "dq-primary",
        "net",
        _fill(_DERIVED_RATE_NET, "kd", k0="0.5", body="abs(Atot)/Atot"),
        "kd",
        "1.0",
        1.0,
        _T,
        ("A()",),
        param="kd",
        note="abs() declines the analytic RHS, so CVODES' difference quotient carries it",
    ),
    pytest.param(
        Term(
            "dq-derived-own-column",
            "net",
            _fill(_DERIVED_RATE_NET, "kd", k0="0.5", body="abs(Atot)/Atot"),
            "kd",
            "2*k0",
            1.0,
            _T,
            ("A()",),
            param="kd",
            note="column exactly 0 against -10 t exp(-t)",
        ),
        marks=_xfail(707, "the DQ RHS sync re-derives the probed derived parameter"),
    ),
    pytest.param(
        Term(
            "stale-artifact-after-override",
            "antimony",
            _STALE,
            "k2",
            "5.0",
            5.0,
            tuple(np.linspace(0.0, 1.0, 11)),
            ("A",),
            param="k2",
            analytic=_reused_after_override,
            note="d/dk2 is 0 on the reused Simulator, whose sensitivity RHS still chains k2 to k1",
        ),
        marks=_xfail(708, "the compiled sensitivity RHS is reused after set_param"),
    ),
    # ── Events ───────────────────────────────────────────────────────────────
    Term(
        "event-fixed-time",
        "antimony",
        _EVENT_FIXED,
        "k",
        "0.5",
        0.5,
        (0.0, 1.0, 3.0, 4.0),
        ("X",),
        param="k",
        note="s⁺ = ∂h/∂x·s⁻ + ∂h/∂p at a fixed time (GH #212)",
    ),
    Term(
        "event-time-trigger",
        "antimony",
        _fill(_EVENT_TIME, "T0", body="A"),
        "T0",
        "1.0",
        1.0,
        (0.0, 0.5, 2.0, 3.0),
        ("B",),
        param="T0",
        note="a time trigger T0 moves: h_x(s⁻ + f⁻τ) − f⁺τ with τ = 1 (#49)",
    ),
    Term(
        "event-state-trigger",
        "antimony",
        _EVENT_RESET,
        "thr",
        "5.0",
        5.0,
        (0.0, 1.0, 2.0, 3.0),
        ("A",),
        param="thr",
        note="a state trigger: τ = −(∂g/∂p + g_x·s⁻)/(g_x·f⁻) (#144)",
    ),
    pytest.param(
        Term(
            "event-assignment-reads-time",
            "antimony",
            _fill(_EVENT_STATE, "k", body="time + 1"),
            "k",
            "0.5",
            0.5,
            (0.0, 1.0, 2.0, 3.0),
            ("B",),
            param="k",
            note="dB/dk is 0 against -ln2/k² = -2.7726",
        ),
        marks=_xfail(735, "the event jump drops (∂h/∂t)·∂t*/∂p"),
    ),
    pytest.param(
        Term(
            "event-assignment-reads-time-clock",
            "antimony",
            _fill(_EVENT_TIME, "T0", body="time"),
            "T0",
            "1.0",
            1.0,
            (0.0, 0.5, 2.0, 3.0),
            ("B",),
            param="T0",
            note="dB/dT0 is 0 against 1",
        ),
        marks=_xfail(735, "the event jump drops (∂h/∂t)·∂t*/∂p"),
    ),
    Term(
        "event-assignment-reads-state",
        "antimony",
        _fill(_RATEOF, "a", body="-a*A"),
        "a",
        "0.5",
        0.5,
        (0.0, 1.0, 2.0, 4.0, 6.0),
        ("Y",),
        param="a",
        note="the explicit twin of the rateOf case",
    ),
    pytest.param(
        Term(
            "event-assignment-reads-rateof",
            "antimony",
            _fill(_RATEOF, "a", body="rateOf(A)"),
            "a",
            "0.5",
            0.5,
            (0.0, 1.0, 2.0, 4.0, 6.0),
            ("Y",),
            param="a",
            note="dY/da is 0 against 1.115651",
        ),
    ),
    Term(
        "event-batch-survivor-declared-last",
        "antimony",
        _fill(_BATCH, "p", first=_EHI, second=_ELO),
        "p",
        "1.0",
        1.0,
        (0.0, 1.0, 2.5, 3.0, 4.0),
        ("X",),
        param="p",
        note="the event executed last is also declared last, so the jump's walk is right",
    ),
    pytest.param(
        Term(
            "event-batch-priority-order",
            "antimony",
            _fill(_BATCH, "p", first=_ELO, second=_EHI),
            "p",
            "1.0",
            1.0,
            (0.0, 1.0, 2.5, 3.0, 4.0),
            ("X",),
            param="p",
            note="dX/dp(4) is 7, the event last by index, against 5, the one executed last",
        ),
        marks=_xfail(722, "the batch jump walks declaration order, the state priority order"),
    ),
    Term(
        "event-fires-after-t-start",
        "antimony",
        _fill(_T0_FIRE, "k1", fire="0.25"),
        "k1",
        "0.5",
        0.5,
        (0.0, 1.0, 2.0),
        ("A", "B"),
        param="k1",
        note="the same reset, fired a quarter of a time unit after t_start",
    ),
    pytest.param(
        Term(
            "event-fires-at-t-start",
            "antimony",
            _fill(_T0_FIRE, "k1", fire="0"),
            "k1",
            "0.5",
            0.5,
            (0.0, 1.0, 2.0),
            ("A", "B"),
            param="k1",
            refused="CV_CONV_FAILURE|CV_FIRST_SRHSFUNC_ERR",
            note="refused with CV_CONV_FAILURE: s⁻ read before CVODES's first step is NaN",
        ),
        marks=_xfail(
            717,
            "capture_event_sens reads s⁻ before CVODES has stepped",
            raises=AnalyticRunRefused,
        ),
    ),
    Term(
        "event-beside-clock-switch",
        "antimony",
        _fill(_COINCIDENT, "tau", switch="tau + 0.5"),
        "tau",
        "3.0",
        3.0,
        _SWITCH_T,
        ("X", "Y"),
        param="tau",
        note="an event and a clock switch tau moves, half a time unit apart",
    ),
    pytest.param(
        Term(
            "event-coincident-with-clock-switch",
            "antimony",
            _fill(_COINCIDENT, "tau", switch="tau"),
            "tau",
            "3.0",
            3.0,
            _SWITCH_T,
            ("X", "Y"),
            param="tau",
            note="dY/dtau(6) is 0 against -k·a·(T-tau) = -3",
        ),
        marks=_xfail(767, "the event jump reads f⁻ on the switch's after-branch"),
    ),
    # ── Rate-law switches ────────────────────────────────────────────────────
    Term(
        "time-switch",
        "net",
        _TIME_SWITCH_NET,
        "tau",
        "3.0",
        3.0,
        (0.0, 1.0, 2.0, 2.5, 4.0, 6.0, 10.0),
        ("X()",),
        param="tau",
        scale=1.0,
        note="s⁺ = s⁻ + (f⁻ − f⁺)·∂t*/∂p at a crossing known a priori (#48)",
    ),
    pytest.param(
        Term(
            "time-switch-just-after-an-output",
            "net",
            _TIME_SWITCH_NET,
            "tau",
            "30.00000005",
            30.00000005,
            # The core takes a switch as reached within 1e-9·max(1, horizon) of
            # it, so the horizon of 100 makes that window 1e-7, and the crossing
            # 5e-8 past the output at t=30 falls inside it. At a horizon of 40 it
            # would not, and the column comes out right.
            tuple(float(t) for t in range(0, 101, 5)),
            ("X()",),
            param="tau",
            scale=1.0,
            # t=30 itself sits 5e-8 before the crossing, inside every step the
            # difference can take, so it is a kink cell; the defect shows at every
            # sample past it.
            compare=tuple(i for i in range(21) if i != 6),
            note="dX/dtau is 0 at every sample against -k = -2",
        ),
        marks=_xfail(737, "a switch within 1e-9·horizon after an output is taken at the output"),
    ),
    Term(
        "state-switch",
        "net",
        _fill(_STATE_SWITCH_NET, "A0", B0="1"),
        "A0",
        "10",
        10.0,
        (0.0, 1.0, 2.0, 4.0, 5.0, 6.0),
        ("Y()",),
        param="A0",
        note="the #150 saltation term at a state crossing whose time moves with A0",
    ),
    pytest.param(
        Term(
            "state-switch-beside-a-large-pool",
            "net",
            _fill(_STATE_SWITCH_NET, "A0", B0="1e8"),
            "A0",
            "10",
            10.0,
            (0.0, 1.0, 2.0, 4.0, 5.0, 6.0),
            ("Y()",),
            param="A0",
            note="dY/dA0 is 0 against -0.6: a 1e8 pool nearby makes the jump read as roundoff",
        ),
        marks=_xfail(763, "the continuity test scales the jump by max|f| over every species"),
    ),
    # ── Counter clocks ───────────────────────────────────────────────────────
    Term(
        "counter-clock-threshold",
        "net",
        _fill(_COUNTER_NET, "sigma", c0="0.5", rc="1", C0="c0"),
        "sigma",
        "3",
        3.0,
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        ("X()",),
        param="sigma",
        note="the clock branch of #48: ∂t*/∂sigma = 1/rc",
    ),
    pytest.param(
        Term(
            "counter-clock-seed",
            "net",
            _fill(_COUNTER_NET, "c0", rc="1", sigma="3", C0="c0"),
            "c0",
            "0.5",
            0.5,
            (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            ("X()",),
            param="c0",
            note="dX/dc0 is 0 against k/rc = 2",
        ),
        marks=_xfail(725, "the counter-clock jump drops the clock's own sensitivity"),
    ),
    pytest.param(
        Term(
            "counter-clock-rate",
            "net",
            _fill(_COUNTER_NET, "rc", c0="0.5", sigma="3", C0="c0"),
            "rc",
            "1",
            1.0,
            (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            ("X()",),
            param="rc",
            note="dX/drc is 0 against k(sigma-c0)/rc² = 5",
        ),
        marks=_xfail(725, "the counter-clock jump drops the clock's own sensitivity"),
    ),
    pytest.param(
        Term(
            "counter-clock-ic-axis",
            "net",
            _fill(_COUNTER_NET, "C0", c0="0.5", rc="1", sigma="3"),
            "C0",
            "0.5",
            0.5,
            (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            ("X()",),
            ic="C()",
            note="dX/dC(0) is 0 against 2: an IC column is never jumped",
        ),
        marks=_xfail(725, "the counter-clock jump drops the clock's own sensitivity"),
    ),
    Term(
        "not-a-clock",
        "net",
        _fill(_FAKE_CLOCK_NET, "thr", A0="1.001"),
        "thr",
        "0.4",
        0.4,
        tuple(np.linspace(0.0, 5.0, 11)),
        ("Y()",),
        param="thr",
        note="A0 = 1.001 moves B's rate off 1, so the #150 state path takes it",
    ),
    Term(
        "not-a-clock-read-as-one",
        "net",
        _fill(_FAKE_CLOCK_NET, "thr", A0="1"),
        "thr",
        "0.4",
        0.4,
        tuple(np.linspace(0.0, 5.0, 11)),
        ("Y()",),
        param="thr",
        note="B of A<->B at kf = kr has slope 1 at both probes and is still not a clock (#733)",
    ),
    # ── The comoving onset frame (#545) ──────────────────────────────────────
    Term(
        "comoving-onset",
        "net",
        _pulse("on"),
        "on",
        "3.0",
        3.0,
        _PULSE_T,
        ("X()",),
        param="on",
        note="V = S + c·f past an onset rising as s^(a-1), a = 1.8",
    ),
    pytest.param(
        Term(
            "comoving-onset-clock-fed-rate",
            "net",
            _pulse("on", a="3.0", extra=_CLOCK_FED),
            "on",
            "3.0",
            3.0,
            _PULSE_T,
            ("X()",),
            param="on",
            note="Tc() -> Tc() + X() k2 adds nothing that moves with on; dX/don flips sign",
        ),
        marks=_xfail(749, "beta drops c·dF/dclock for rates that read the clock species"),
    ),
    Term(
        "comoving-onset-loose",
        "net",
        _pulse("on", a="1.2"),
        "on",
        "3.0",
        3.0,
        _PULSE_T,
        ("X()",),
        param="on",
        rtol=1e-4,
        atol=1e-6,
        note="a primary onset's comoving column at rtol 1e-4",
    ),
    pytest.param(
        Term(
            "comoving-onset-derived",
            "net",
            _pulse("on", a="1.2"),
            "on",
            "lam*1.0",
            3.0,
            _PULSE_T,
            ("X()",),
            param="on",
            rtol=1e-4,
            atol=1e-6,
            note="13% low at rtol 1e-4: the derived onset's column never gets a comoving case",
        ),
        marks=_xfail(750, "a derived onset column is left on the plain, singular path"),
    ),
    Term(
        "comoving-closing-edge-regular",
        "net",
        _pulse("D", a="1.5", shape=_CLOSING),
        "D",
        "4.0",
        4.0,
        _PULSE_T,
        ("X()",),
        param="D",
        rtol=1e-8,
        atol=1e-10,
        note="a window closing as (1-s)^0.5, whose forcing stays integrable",
    ),
    pytest.param(
        Term(
            "comoving-closing-edge",
            "net",
            _pulse("D", a="1.1", shape=_CLOSING),
            "D",
            "4.0",
            4.0,
            _PULSE_T,
            ("X()",),
            param="D",
            rtol=1e-8,
            atol=1e-10,
            note="dX/dD(10) is 0.43999986 against 0.44162066, -0.37%, flat in rtol",
        ),
        marks=_xfail(760, "the frame starts at the crossing, after the closing edge's approach"),
    ),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("term", TERMS, ids=lambda term: term.id)
def test_column_matches_central_difference(term: Term, method: str) -> None:
    _check(term, method)


# ─── The oracle itself ──────────────────────────────────────────────────────
# Checked against closed forms with a synthetic solver error, so what is tested
# is the step choice and the bound, not bngsim. The error is deterministic and
# rough in the parameter, the shape a run's error has when a perturbation
# changes the step sequence, and scales with the tolerance the way the noise
# probe assumes.

_TS = np.array([0.0, 0.5, 1.0, 2.0, 4.0])


def _noisy(x: np.ndarray, v: float, rtol: float) -> np.ndarray:
    return x * (1.0 + rtol * np.sin(1.0e7 * v + np.arange(x.size)))


def _decay(v: float, rtol: float) -> np.ndarray:
    return _noisy(10.0 * np.exp(-v * _TS), v, rtol)


class TestTheOracle:
    def test_its_bound_holds_on_a_closed_form(self):
        fd = central_difference(_decay, 0.5, rtol=1e-12)
        exact = -10.0 * _TS * np.exp(-0.5 * _TS)
        assert np.all(np.abs(fd.value - exact) <= fd.err)
        assert not fd.kink.any()
        # ...and it is tight enough to judge a column at the default resolution.
        assert mismatches(exact, fd, 0.5, rtol=1e-10, atol=1e-12) == []

    def test_a_fixed_step_below_the_floor_is_what_it_avoids(self):
        # Issue #368's first trap: a step of 1e-6·p on a trajectory accurate to
        # 1e-6 returns noise over 2h. The ladder's quotient still meets the
        # closed form within its own bound.
        rtol = 1e-6
        h = 1e-6 * 0.5
        fixed = (_decay(0.5 + h, rtol) - _decay(0.5 - h, rtol)) / (2.0 * h)
        exact = -10.0 * _TS * np.exp(-0.5 * _TS)
        assert np.max(np.abs(fixed - exact)) > 1.0
        fd = central_difference(_decay, 0.5, rtol=rtol)
        assert np.all(np.abs(fd.value - exact) <= fd.err)

    def test_a_sample_on_a_crossing_is_refused(self):
        # Issue #368's second trap: x = k·max(t - tau, 0) has a kink in tau at
        # t = tau, where a central difference is exactly half the jump at every
        # step. The cell is flagged, and a comparison over it refuses.
        def ramp(v, rtol):
            return _noisy(2.0 * np.maximum(_TS - v, 0.0), v, rtol)

        fd = central_difference(ramp, 1.0, rtol=1e-12)
        node = list(_TS).index(1.0)
        assert fd.value[node] == pytest.approx(-1.0, rel=1e-6)
        assert fd.kink[node] and fd.kink.sum() == 1
        with pytest.raises(ReferenceRefused, match="kink"):
            mismatches(np.where(_TS > 1.0, -2.0, 0.0), fd, 1.0, rtol=1e-10, atol=1e-12)

    def test_a_zero_column_is_still_judged(self):
        # A parameter the outputs do not depend on: the true column is 0, which
        # has no peak to judge against, so the typical scale is the yardstick. A
        # small but real nonzero analytic value fails; the exact zero passes.
        def flat(v, rtol):
            return _noisy(10.0 * np.exp(-0.5 * _TS), v, rtol)

        fd = central_difference(flat, 2.0, rtol=1e-12)
        assert mismatches(np.zeros(_TS.size), fd, 2.0, rtol=1e-10, atol=1e-12) == []
        wrong = np.full(_TS.size, 1e-4 * fd.typical)
        assert mismatches(wrong, fd, 2.0, rtol=1e-10, atol=1e-12)

    def test_a_reference_too_coarse_to_judge_refuses(self):
        fd = central_difference(_decay, 0.5, rtol=1e-4)
        with pytest.raises(ReferenceRefused, match="too coarse"):
            mismatches(fd.value, fd, 0.5, rtol=1e-10, atol=1e-12, resolution=1e-9)

    def test_a_ladder_of_any_ratio_extrapolates(self):
        # The one-sided extrapolation uses each pair of steps' own ratio, so a
        # ladder dividing by 3 keeps an honest bound and still finds the kink.
        ladder = tuple(0.1 / 3.0**k for k in range(9))
        fd = central_difference(_decay, 0.5, rtol=1e-12, rel_steps=ladder)
        exact = -10.0 * _TS * np.exp(-0.5 * _TS)
        assert np.all(np.abs(fd.value - exact) <= fd.err) and not fd.kink.any()

        def ramp(v, rtol):
            return _noisy(2.0 * np.maximum(_TS - v, 0.0), v, rtol)

        kinked = central_difference(ramp, 1.0, rtol=1e-12, rel_steps=ladder)
        assert kinked.kink.sum() == 1 and kinked.kink[list(_TS).index(1.0)]

    def test_a_ladder_must_decrease(self):
        with pytest.raises(ValueError, match="strictly decreasing"):
            central_difference(_decay, 0.5, rel_steps=(0.1, 0.2, 0.05))

    def test_a_loose_column_is_still_held_to_one_percent(self):
        # 100·rtol alone would pass a column twice the true value at rtol 1e-2.
        fd = central_difference(_decay, 0.5, rtol=1e-12)
        exact = -10.0 * _TS * np.exp(-0.5 * _TS)
        assert mismatches(2.0 * exact, fd, 0.5, rtol=1e-2, atol=1e-12)
        assert mismatches(1.005 * exact, fd, 0.5, rtol=1e-2, atol=1e-12) == []

    def test_a_refusal_cannot_satisfy_a_known_defect_xfail(self):
        # The xfails above name raises=AssertionError; a reference that could not
        # judge must fail them rather than count as the defect.
        assert not issubclass(ReferenceRefused, AssertionError)
