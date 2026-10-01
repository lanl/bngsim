"""Every way a rate reads a clock, in a comoving onset column (issues #749, #750).

A pulse ``k1·s^(a-1)·(1-s)`` opening at ``on`` has an onset column that is
integrated as ``V = S + c·f`` with ``c = ∂t*/∂on`` (issue #545). The clock rows of
``V`` are held at zero, so the forcing has to carry ``c·∂f/∂clock`` for every
rate that reads a clock species.

It carried that term only for a rate law whose text names the clock (#749).
Mass action over the clock, a rate law multiplied by it, a rate law reading an
observable that sums it with another species, and a Michaelis-Menten rate with
the clock as its enzyme all lost theirs, and the onset column came back wrong by
up to several times its size, with the wrong sign, at every tolerance.

A column asked for a *derived* onset, ``on = lam*1.0``, got no comoving case at
all (#750): 13% low at rtol 1e-4 with no warning, and a solver failure at the
default tolerance.

Every column is compared with a central difference of plain runs, taken at two
steps and extrapolated. Where the added reaction makes nothing that moves with
the onset, it is also compared with the column of the model without it.
"""

from __future__ import annotations

import bngsim
import numpy as np
import pytest

T = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]  # off the onset (3) and the close (7)

NET = """begin parameters
    1 k0    0.1
    2 k1    2.0
    3 a     {a}
{chain}
    6 D     4.0
    7 kdeg  0.3
    8 k2    0.5
    9 _rateLaw1 1
   10 kd2   0.05
   11 kcat  0.2
   12 Km    1.5
end parameters
begin functions
    1 s() (t-{onset})/D
    2 prod() k0+if(t>={onset},if(t<=({onset}+D),k1*(s()^(a-1))*(1-s()),0),0)
{funcs}end functions
begin species
    1 X() 0
    2 Tc() 0
    3 P() 0
end species
begin reactions
    1 {source} prod
    2 1 0 kdeg
    3 0 2 _rateLaw1
{extra}end reactions
begin groups
    1 t 2
{groups}end groups
"""
PRIMARY = "    4 lam   3.0\n    5 on    3.0"


def _net(a="1.2", chain=PRIMARY, onset="on", funcs="", extra="", groups="", source="0 1") -> str:
    return NET.format(
        a=a, chain=chain, onset=onset, funcs=funcs, extra=extra, groups=groups, source=source
    )


def _model(tmp_path, text):
    path = tmp_path / "m.net"
    path.write_text(text)
    return bngsim.Model.from_net(path)


def _columns(tmp_path, text, params, rtol=1e-8, atol=1e-10):
    sim = bngsim.Simulator(_model(tmp_path, text), method="ode", sensitivity_params=list(params))
    run = sim.run(sample_times=T, rtol=rtol, atol=atol)
    return np.asarray(run.sensitivities)[:, 0, :]


def _fd(tmp_path, text, param):
    """dX/d(param) by a central difference of plain runs, extrapolated from two
    steps. A derived parameter is held where it is set, as its column holds it."""
    value = _model(tmp_path, text).get_param(param)

    def at(v):
        model = _model(tmp_path, text)
        model.set_param(param, v, force_override=True)
        run = bngsim.Simulator(model, method="ode").run(sample_times=T, rtol=1e-12, atol=1e-14)
        return np.asarray(run.species)[:, 0]

    def central(h):
        return (at(value + h) - at(value - h)) / (2 * h)

    h = 1e-3 * abs(value)
    return (4.0 * central(h / 2) - central(h)) / 3.0


def _close(got, want, rel):
    np.testing.assert_allclose(got, want, rtol=0, atol=rel * np.max(np.abs(want)))


# ─── #749: a rate that reads the clock outside a rate law's text ─────────────

# Each adds a reaction to the pulse model. `same` marks the ones that make
# nothing which moves with the onset, so dX/d(on) is the pulse model's own.
CLOCK_READERS = {
    "mass-action-catalyst": dict(extra="    4 2 2,1 k2\n", same=True),
    "mass-action-clock-squared": dict(extra="    4 2,2 2,2,1 kd2\n", same=True),
    "mass-action-loss-with-clock": dict(extra="    4 1,2 2 kd2\n", same=False),
    "rate-law-times-clock": dict(funcs="    3 fB() k2*1.0\n", extra="    4 2 2,1 fB\n", same=True),
    "rate-law-reads-a-sum-with-the-clock": dict(
        funcs="    3 fT() kd2*Tot\n", extra="    4 1 0 fT\n", groups="    2 Tot 1,2\n", same=False
    ),
    "rate-law-reads-a-sum-with-twice-the-clock": dict(
        funcs="    3 fT() kd2*Tot\n",
        extra="    4 1 0 fT\n",
        groups="    2 Tot 1,2*2\n",
        same=False,
    ),
    "pulse-law-times-clock": dict(source="2 2,1", same=False),
    "pulse-law-also-reads-a-sum": dict(
        funcs="    3 fP() prod()*Tot/(1+Tot)\n",
        extra="    4 0 1 fP\n",
        groups="    2 Tot 1,2\n",
        same=False,
    ),
    "michaelis-menten-clock-enzyme": dict(extra="    4 2,1 2,3 MM kcat Km\n", same=False),
    # The clock in the substrate's place, and returned, so it stays a clock. X is
    # the enzyme and the product.
    "michaelis-menten-clock-substrate": dict(extra="    4 1,2 1,2,1 MM kcat Km\n", same=False),
}


@pytest.mark.parametrize("a", ["1.2", "3.0"])
@pytest.mark.parametrize("reader", sorted(CLOCK_READERS))
def test_the_onset_column_carries_every_rate_that_reads_the_clock(tmp_path, reader, a):
    """dX/d(on) with a second rate that reads the clock species. With
    ``Tc() -> Tc() + X() k2`` the column came back −1.28 for 0.18 at a = 1.2, and
    −1.41 for 0.05 at a = 3, where the pulse's own derivative is smooth."""
    spec = dict(CLOCK_READERS[reader])
    same = spec.pop("same")
    text = _net(a=a, **spec)
    got = _columns(tmp_path, text, ["on"])[:, 0]
    _close(got, _fd(tmp_path, text, "on"), 1e-5)
    if same:
        _close(got, _columns(tmp_path, _net(a=a), ["on"])[:, 0], 1e-5)


@pytest.mark.parametrize("reader", ["mass-action-catalyst", "rate-law-reads-a-sum-with-the-clock"])
def test_a_column_with_no_comoving_case_is_unchanged(tmp_path, reader):
    """Control. The window length D moves no singular power of this pulse, so its
    column is a plain one, and was right."""
    spec = dict(CLOCK_READERS[reader])
    spec.pop("same")
    text = _net(**spec)
    _close(_columns(tmp_path, text, ["D"])[:, 0], _fd(tmp_path, text, "D"), 1e-5)


def test_a_loose_tolerance_keeps_the_clock_terms(tmp_path):
    """The same column at rtol 1e-4. The missing term was not a tolerance error:
    it was −1.28 for 0.18 here too."""
    text = _net(extra="    4 2 2,1 k2\n")
    got = _columns(tmp_path, text, ["on"], rtol=1e-4, atol=1e-6)[:, 0]
    _close(got, _fd(tmp_path, text, "on"), 5e-3)


# ─── #750: a derived onset parameter's own column ───────────────────────────

# (the chain, how the onset is written, the columns asked for, what the first
# column is of the others: d(first)/d(other))
CHAINS = {
    "on = lam*1.0": ("    4 lam   3.0\n    5 on    lam*1.0", "on", ["on", "lam"], [1.0]),
    "on = 2*lam": ("    4 lam   1.5\n    5 on    2*lam", "on", ["on", "lam"], [2.0]),
    "on = mid+1, mid = 2*lam": (
        "    4 lam   1.0\n    5 mid   2*lam\n   13 on    mid+1",
        "on",
        ["on", "mid", "lam"],
        [1.0, 2.0],
    ),
    "the onset written as lam+off, off = 2*q": (
        "    4 lam   1.0\n    5 q     1.0\n   13 off   2*q",
        "(lam+off)",
        ["off", "lam", "q"],
        [1.0, 2.0],
    ),
    # `lambda` is a Python keyword: its symbol is an alias inside the derivation.
    "on = lambda*1.0": ("    4 lambda 3.0\n    5 on    lambda*1.0", "on", ["on", "lambda"], [1.0]),
}


@pytest.mark.parametrize("rtol,atol,rel", [(1e-4, 1e-6, 2e-2), (1e-8, 1e-10, 1e-5)])
@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_a_derived_onset_gets_its_own_comoving_column(tmp_path, chain, rtol, atol, rel):
    """Each parameter on the chain to the onset, asked for alone. The derived
    ``on`` was 13% to 18% low at rtol 1e-4 and stalled CVODE at the default."""
    spec, onset, params, _ratios = CHAINS[chain]
    text = _net(chain=spec, onset=onset)
    for name in params:
        got = _columns(tmp_path, text, [name], rtol=rtol, atol=atol)[:, 0]
        _close(got, _fd(tmp_path, text, name), rel)


@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_the_columns_of_a_chain_together(tmp_path, chain):
    """Asked for in one run, each column is a fixed multiple of the first: the
    onset moves at d(on)/d(lam) per unit of lam."""
    spec, onset, params, ratios = CHAINS[chain]
    text = _net(chain=spec, onset=onset)
    cols = _columns(tmp_path, text, params)
    _close(cols[:, 0], _fd(tmp_path, text, params[0]), 1e-5)
    for k, ratio in enumerate(ratios, start=1):
        _close(cols[:, k], ratio * cols[:, 0], 1e-6)


@pytest.mark.parametrize(
    "reader",
    [
        "mass-action-catalyst",
        "rate-law-reads-a-sum-with-twice-the-clock",
        "michaelis-menten-clock-enzyme",
    ],
)
def test_a_shift_other_than_one_scales_the_clock_terms(tmp_path, reader):
    """``on = 2*lam``: the onset moves at 2 per unit of lam, so lam's column is
    integrated with c = 2 and every clock term in it is doubled. Its column is
    twice the derived onset's own."""
    spec = dict(CLOCK_READERS[reader])
    spec.pop("same")
    text = _net(chain=CHAINS["on = 2*lam"][0], **spec)
    cols = _columns(tmp_path, text, ["on", "lam"])
    _close(cols[:, 0], _fd(tmp_path, text, "on"), 1e-5)
    _close(cols[:, 1], _fd(tmp_path, text, "lam"), 1e-5)
    _close(cols[:, 1], 2.0 * cols[:, 0], 1e-6)


def test_the_primaries_keep_their_cases_when_the_derived_axes_run_out_of_budget(
    tmp_path, monkeypatch
):
    """There is one derived axis per derived parameter a singular base reaches,
    so that is where a build's derivation budget goes. A first cut derived them
    in the same pass as the primaries, and an expiry there took the primary's
    case with it: its column fell back to the plain one, 15% off at rtol 1e-4
    with a thousand derived parameters feeding one onset.

    The emitted source is read directly. Nothing is compiled, so the build this
    test starves is not left in the cache for another test to find."""
    from bngsim import _codegen
    from bngsim._jacobian import _DerivationBudgetExceeded

    model = _model(tmp_path, _net(chain=CHAINS["on = 2*lam"][0]))

    def cases():
        source = _codegen.generate_sens_from_model(
            model._core, functional=True, emit_term_scale=True
        )
        return source.count("*c_out =")

    assert cases() == 2  # lam's and the derived onset's own

    def out_of_budget(*_args, **_kwargs):
        raise _DerivationBudgetExceeded

    monkeypatch.setattr(_codegen, "_comoving_derived_axes", out_of_budget)
    assert cases() == 1


# ─── The same through SBML, where a species can be an amount ────────────────

SBML_T = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]
SBML_PULSE = (
    "k0 + piecewise(piecewise(k1*(({clock}-on)/D)^(a-1)*(1-({clock}-on)/D), "
    "{clock} <= on + D, 0), {clock} >= on, 0)"
)
COUNTER = """
model counter
  compartment C = {size};
  substanceOnly species T in C;
  species X in C{more};
  X = 0; T = 0;
  {rule}
  k0 = 0.1; k1 = 2; a = {a}; on = 3; D = 4; kdeg = 0.3; k2 = 0.5; kd2 = 0.05;
  Jt: -> T; C;
  J1: -> X; {pulse};
  J2: X -> ; kdeg*X;
  {extra}
end
"""


def _counter(a, size="1", more="", rule="", extra=""):
    return COUNTER.format(
        a=a, size=size, more=more, rule=rule, extra=extra, pulse=SBML_PULSE.format(clock="T")
    )


def _sbml_column(text, times=SBML_T):
    model = bngsim.Model.from_antimony_string(text)
    run = bngsim.Simulator(model, method="ode", sensitivity_params=["on"]).run(
        sample_times=list(times), rtol=1e-8, atol=1e-10
    )
    return np.asarray(run.sensitivities)[:, list(run.species_names).index("X"), 0]


def _sbml_fd(text, times=SBML_T):
    def at(value):
        model = bngsim.Model.from_antimony_string(text)
        model.set_param("on", value)
        run = bngsim.Simulator(model, method="ode").run(
            sample_times=list(times), rtol=1e-12, atol=1e-14
        )
        return np.asarray(run.species)[:, list(run.species_names).index("X")]

    def central(h):
        return (at(3.0 + h) - at(3.0 - h)) / (2 * h)

    return (4.0 * central(1.5e-3) - central(3e-3)) / 3.0


@pytest.mark.parametrize("a", ["1.8", "3.0"])
@pytest.mark.parametrize("extra", ["", "J3: T -> T + X; k2*T;"], ids=["alone", "with-a-reader"])
def test_an_amount_in_a_compartment_of_size_two_is_not_a_unit_rate_clock(a, extra):
    """T is declared in substance units and made at C per unit time, so what a
    rate law reads under its name is 2·t. Its stored value moves at 1, and that
    was taken for the clock: the onset `T >= 3` was jumped at t = 3, where it is
    crossed at 1.5, and the column was wrong with or without a second reader."""
    text = _counter(a, size="2", extra=extra)
    _close(_sbml_column(text), _sbml_fd(text), 1e-5)


def test_a_rate_law_that_reads_a_sum_whose_weight_is_a_live_size():
    """Tot := T + X with T an amount in a compartment whose size is a parameter:
    T's weight in Tot is that size, which a shift has no number for. A rate law
    reads Tot, so the model gets no comoving case and the plain column answers.
    With the case and without the term the column was 15% off at a = 3."""
    text = _counter("3.0", more=", Tot in C", rule="Tot := T + X;", extra="J3: X -> ; kd2*X*Tot;")
    _close(_sbml_column(text), _sbml_fd(text), 1e-5)


def test_a_sum_no_rate_law_reads_does_not_cost_the_model_its_case():
    """Control. The same Tot with nothing reading it. At a = 1.5 the pulse's
    onset column needs its comoving case at this tolerance, and keeps it: a
    first cut dropped the plan for any such observable, read or not, and the
    run stalled at the onset."""
    text = _counter("1.5", more=", Tot in C", rule="Tot := T + X;")
    _close(_sbml_column(text), _sbml_fd(text), 1e-5)


def test_a_row_divided_by_a_compartment_that_is_the_clock():
    """The pulse is gated on a compartment that grows at 1, and a transfer puts
    X's row over that compartment's live size. The row moves with the clock in
    a way no comoving case writes, so the model gets none. With one, the column
    was 46% off at rtol 1e-8."""
    text = """
model growing
  compartment C1 = 1, C2 = 2;
  C1' = 1;
  species X in C1, Y in C2;
  X = 0; Y = 1;
  k0 = 0.1; k1 = 2; a = 3.0; on = 3; D = 4; kdeg = 0.3; kt = 0.4;
  Jx: Y => X; kt*Y;
  J1: -> X; {pulse};
  J2: X -> ; kdeg*X;
end
""".format(pulse=SBML_PULSE.format(clock="C1"))
    # C1 = 1 + t crosses the onset at t = 2 and the close at t = 6: off both.
    times = [0.0, 1.0, 2.5, 4.0, 5.0, 6.5, 8.0, 10.0]
    _close(_sbml_column(text, times), _sbml_fd(text, times), 1e-5)
