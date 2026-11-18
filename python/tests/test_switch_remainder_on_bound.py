"""Coinciding switch times where one has no parameter of its own (issue #1822).

A coinciding crossing is told apart from the others on its instant by a
parameter only its own threshold reads (issue #375). With ``sigma = t0 +
t_delta`` and ``t_delta = 0``, which is where a bounded optimizer leaves a
parameter that wants to be smaller, the crossing at ``t0`` has none: ``t0``
moves ``sigma`` too. bngsim refused the run, and a fit that had clamped
``t_delta`` to its bound stopped there.

Separating all but one crossing of an instant is enough. The one left over is
given the instant's merged jump, both crossings moving with ``t0``, and
``sigma`` its own jump.

Where a rate law composes the two, a product of their gates, the solution has
a kink in ``t_delta`` at 0. The ``t_delta`` column is then the derivative for
``t_delta`` increasing, the direction that delays ``sigma`` past ``t0``, and
bngsim says so in a warning. Every model here has a closed form on each side.
"""

from __future__ import annotations

import math
import warnings

import bngsim
import pytest
from bngsim._exceptions import SensitivityRefusedForParametersError, SensitivityUnsupportedError

# Looked up, not imported: on a build without the fix each test then fails on
# its own, on what it asserts, where an import would stop the whole file.
OneSidedSensitivityWarning = getattr(bngsim, "OneSidedSensitivityWarning", UserWarning)

T_END = 6.0
CLOCKS = ["time()", "t"]


def _net(tmp_path, clock: str, parameters: list[str], laws: dict[str, str], species: int = 1):
    """Species X1..Xn, each decaying at its own rate law, from 100.

    ``clock`` is ``time()`` or ``t``, a counter species that rises at rate 1
    from 0, which is how the model of the issue keeps its time.
    """
    params = "".join(f"    {i} {p}\n" for i, p in enumerate([*parameters, "one 1"], 1))
    funcs = "".join(
        f"    {i} {name}() {law.replace('CLOCK', clock)}\n"
        for i, (name, law) in enumerate(laws.items(), 1)
    )
    counter = species + 1
    text = (
        f"begin parameters\n{params}end parameters\n"
        f"begin functions\n{funcs}end functions\n"
        "begin species\n"
        + "".join(f"    {i} X{i}() 100\n" for i in range(1, species + 1))
        + f"    {counter} counter() 0\nend species\n"
        "begin reactions\n"
        + "".join(f"    {i} {i} 0 {name} #_R{i}\n" for i, name in enumerate(laws, 1))
        + f"    {len(laws) + 1} 0 {counter} one #_clock\nend reactions\n"
        "begin groups\n"
        + "".join(f"    {i} X{i}obs {i}\n" for i in range(1, species + 1))
        + f"    {counter} t {counter}\nend groups\n"
    )
    path = tmp_path / "m.net"
    path.write_text(text, encoding="utf-8")
    return bngsim.Model.from_net(str(path))


def _run(model, params: list[str]):
    """(result, the warnings of the new category it raised)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bngsim.Simulator(model, method="ode", sensitivity_params=params).run(
            (0.0, T_END), 7, rtol=1e-11, atol=1e-13
        )
    return result, [w for w in caught if issubclass(w.category, OneSidedSensitivityWarning)]


def _column(result, species: int, param: str) -> float:
    return float(result.sensitivities[-1, species, result.sensitivity_params.index(param)])


SHIFTED = ["1 t0 3", "2 t_delta 0", "3 sigma t0+t_delta", "4 k 0.5", "5 k2 0.25"]
SHIFTED = [p.split(" ", 1)[1] for p in SHIFTED]


# ─── Two rate laws, one gate each: the crossings add ─────────────────────────


@pytest.mark.parametrize("clock", CLOCKS)
def test_two_laws_on_one_instant_get_their_own_jumps(tmp_path, clock):
    # X1 decays from t0 and X2 from sigma. t0 moves both; t_delta moves X2 alone.
    model = _net(
        tmp_path,
        clock,
        SHIFTED,
        {"a": "if(CLOCK>=t0,k,0)", "b": "if(CLOCK>=sigma,k2,0)"},
        species=2,
    )
    result, raised = _run(model, ["t0", "t_delta"])
    x1 = 100 * math.exp(-0.5 * (T_END - 3))
    x2 = 100 * math.exp(-0.25 * (T_END - 3))
    assert result.species[-1, 0] == pytest.approx(x1, rel=1e-8)
    assert _column(result, 0, "t0") == pytest.approx(0.5 * x1, rel=1e-6)
    assert _column(result, 0, "t_delta") == pytest.approx(0.0, abs=1e-6)
    assert _column(result, 1, "t0") == pytest.approx(0.25 * x2, rel=1e-6)
    assert _column(result, 1, "t_delta") == pytest.approx(0.25 * x2, rel=1e-6)
    # No rate law reads the two together: there is no kink, and no warning.
    assert raised == [] and result.one_sided_sensitivities == []


@pytest.mark.parametrize("clock", CLOCKS)
def test_two_gates_summed_in_one_law_commute_and_are_two_sided(tmp_path, clock):
    # One law reads both gates, and adds them: the rate is the same whichever
    # comes first, so the t_delta column is an ordinary derivative.
    model = _net(tmp_path, clock, SHIFTED, {"a": "if(CLOCK>=t0,k,0)+if(CLOCK>=sigma,k2,0)"})
    result, raised = _run(model, ["t0", "t_delta", "k"])
    x = 100 * math.exp(-0.75 * (T_END - 3))
    assert result.species[-1, 0] == pytest.approx(x, rel=1e-8)
    assert _column(result, 0, "t0") == pytest.approx(0.75 * x, rel=1e-6)
    assert _column(result, 0, "t_delta") == pytest.approx(0.25 * x, rel=1e-6)
    assert _column(result, 0, "k") == pytest.approx(-(T_END - 3) * x, rel=1e-6)
    assert raised == [] and result.one_sided_sensitivities == []


# ─── One rate law, the product of the two gates: a kink in t_delta ───────────
# X decays at k from max(t0, sigma). With t_delta raised sigma is the later and
# dX/dt_delta = k·X; with it lowered t0 is, and dX/dt_delta = 0.

PRODUCT = {"a": "if(CLOCK>=t0,1,0)*if(CLOCK>=sigma,k,0)"}


@pytest.mark.parametrize("clock", CLOCKS)
def test_a_product_of_the_two_gates_gives_the_side_that_delays_the_switch(tmp_path, clock):
    model = _net(tmp_path, clock, SHIFTED, PRODUCT)
    result, raised = _run(model, ["t0", "t_delta", "k"])
    x = 100 * math.exp(-0.5 * (T_END - 3))
    assert result.species[-1, 0] == pytest.approx(x, rel=1e-8)
    assert _column(result, 0, "t_delta") == pytest.approx(0.5 * x, rel=1e-6)  # not 0
    # t0 moves both crossings, on either side of the kink.
    assert _column(result, 0, "t0") == pytest.approx(0.5 * x, rel=1e-6)
    assert _column(result, 0, "k") == pytest.approx(-(T_END - 3) * x, rel=1e-6)
    assert result.one_sided_sensitivities == [(3.0, {"t_delta": "increasing"})]
    assert len(raised) == 1
    message = str(raised[0].message)
    assert "t=3" in message and "'t_delta' increasing" in message and "#1822" in message
    assert "'t0'" not in message and "'k'" not in message


def test_the_warning_names_the_line_that_ran_the_model(tmp_path):
    model = _net(tmp_path, "time()", SHIFTED, PRODUCT)
    _, raised = _run(model, ["t_delta"])
    assert [w.filename for w in raised] == [__file__]


def test_the_warning_can_be_made_an_error(tmp_path):
    model = _net(tmp_path, "time()", SHIFTED, PRODUCT)
    with warnings.catch_warnings():
        warnings.simplefilter("error", OneSidedSensitivityWarning)
        with pytest.raises(OneSidedSensitivityWarning, match="'t_delta' increasing"):
            bngsim.Simulator(model, method="ode", sensitivity_params=["t_delta"]).run(
                (0.0, T_END), 7
            )


@pytest.mark.parametrize("clock", CLOCKS)
def test_a_threshold_that_falls_with_its_parameter_is_delayed_by_lowering_it(tmp_path, clock):
    # sigma = t0 - t_delta: the direction that delays sigma is t_delta decreasing,
    # and on that side X decays from sigma = t0 - t_delta: dX/dt_delta = -k·X.
    params = ["t0 3", "t_delta 0", "sigma t0-t_delta", "k 0.5"]
    model = _net(tmp_path, clock, params, PRODUCT)
    result, raised = _run(model, ["t0", "t_delta"])
    x = 100 * math.exp(-0.5 * (T_END - 3))
    assert _column(result, 0, "t_delta") == pytest.approx(-0.5 * x, rel=1e-6)
    assert _column(result, 0, "t0") == pytest.approx(0.5 * x, rel=1e-6)
    assert result.one_sided_sensitivities == [(3.0, {"t_delta": "decreasing"})]
    assert len(raised) == 1 and "'t_delta' decreasing" in str(raised[0].message)


@pytest.mark.parametrize("clock", CLOCKS)
def test_off_the_bound_the_column_is_two_sided_and_there_is_no_warning(tmp_path, clock):
    params = ["t0 3", "t_delta 0.5", "sigma t0+t_delta", "k 0.5"]
    model = _net(tmp_path, clock, params, PRODUCT)
    result, raised = _run(model, ["t0", "t_delta"])
    x = 100 * math.exp(-0.5 * (T_END - 3.5))
    assert _column(result, 0, "t_delta") == pytest.approx(0.5 * x, rel=1e-6)
    assert _column(result, 0, "t0") == pytest.approx(0.5 * x, rel=1e-6)
    assert raised == [] and result.one_sided_sensitivities == []


# ─── A window of no width: the second bound of the issue's model ─────────────
# X decays at l0 on [sigma, tau1) and at l1 from tau1. With t_delta2 raised the
# window opens and dX/dt_delta2 = (l1 - l0)·X; with it lowered tau1 comes first,
# the window never opens, and dX/dt_delta2 = l1·X.

WINDOW_PARAMS = [
    "t0 1",
    "t_delta 2",
    "t_delta2 0",
    "sigma t0+t_delta",
    "tau1 sigma+t_delta2",
    "l0 0.5",
    "l1 0.2",
]
WINDOW = {"a": "if((CLOCK>=sigma)&&(CLOCK<tau1),l0,if(CLOCK>=tau1,l1,0))"}


@pytest.mark.parametrize("clock", CLOCKS)
def test_a_window_of_no_width_gives_the_side_on_which_it_opens(tmp_path, clock):
    model = _net(tmp_path, clock, WINDOW_PARAMS, WINDOW)
    result, raised = _run(model, ["t0", "t_delta", "t_delta2", "l0", "l1"])
    x = 100 * math.exp(-0.2 * (T_END - 3))
    assert result.species[-1, 0] == pytest.approx(x, rel=1e-8)
    assert _column(result, 0, "t_delta2") == pytest.approx((0.2 - 0.5) * x, rel=1e-6)  # not 0.2·X
    assert _column(result, 0, "t0") == pytest.approx(0.2 * x, rel=1e-6)
    assert _column(result, 0, "t_delta") == pytest.approx(0.2 * x, rel=1e-6)
    assert _column(result, 0, "l0") == pytest.approx(0.0, abs=1e-6)
    assert _column(result, 0, "l1") == pytest.approx(-(T_END - 3) * x, rel=1e-6)
    assert result.one_sided_sensitivities == [(3.0, {"t_delta2": "increasing"})]
    assert len(raised) == 1


# ─── What is still refused ───────────────────────────────────────────────────


@pytest.mark.parametrize("clock", CLOCKS)
def test_two_crossings_with_no_parameter_of_their_own_are_refused(tmp_path, clock):
    # t_delta = t_delta2 = 0: t0 moves all three, t_delta moves sigma and tau1.
    params = [p if not p.startswith("t_delta ") else "t_delta 0" for p in WINDOW_PARAMS]
    model = _net(
        tmp_path,
        clock,
        params,
        {"a": "if(CLOCK>=t0,1,0)*" + WINDOW["a"]},
    )
    with pytest.raises(SensitivityUnsupportedError) as refusal:
        _run(model, ["t0", "t_delta", "t_delta2"])
    message = str(refusal.value)
    assert "2 of the 3 crossings here have no such parameter" in message
    assert "a parameter on a bound makes them so" in message and "#1822" in message
    assert "coincidence" not in message


@pytest.mark.parametrize("clock", CLOCKS)
def test_two_composed_crossings_that_each_have_a_parameter_are_refused_as_before(tmp_path, clock):
    # tA = tB by value. Either can be moved alone, neither comes first, and the
    # run is on a kink in both (issue #951).
    model = _net(
        tmp_path,
        clock,
        ["tA 3", "tB 3", "k 0.5"],
        {"a": "if(CLOCK>=tA,1,0)*if(CLOCK>=tB,k,0)"},
    )
    with pytest.raises(SensitivityRefusedForParametersError, match="do not commute"):
        _run(model, ["tA", "tB"])


@pytest.mark.parametrize("clock", CLOCKS)
def test_three_on_an_instant_with_two_composed_isolated_ones_are_refused(tmp_path, clock):
    # t0 is left over; s1 and s2 each have a parameter, and one law multiplies
    # all three gates. Each isolated one would be read after the other.
    model = _net(
        tmp_path,
        clock,
        ["t0 3", "d1 0", "d2 0", "s1 t0+d1", "s2 t0+d2", "k 0.5"],
        {"a": "if(CLOCK>=t0,1,0)*if(CLOCK>=s1,1,0)*if(CLOCK>=s2,k,0)"},
    )
    with pytest.raises(SensitivityRefusedForParametersError, match="do not commute"):
        _run(model, ["t0", "d1", "d2"])


@pytest.mark.parametrize("clock", CLOCKS)
def test_three_on_an_instant_that_add_are_separated(tmp_path, clock):
    # The same three thresholds, one law each: nothing is composed, t0 is the
    # one left over and takes what s1 and s2 do not.
    model = _net(
        tmp_path,
        clock,
        ["t0 3", "d1 0", "d2 0", "s1 t0+d1", "s2 t0+d2", "k 0.5"],
        {"a": "if(CLOCK>=t0,k,0)", "b": "if(CLOCK>=s1,k,0)", "c": "if(CLOCK>=s2,k,0)"},
        species=3,
    )
    result, raised = _run(model, ["t0", "d1", "d2"])
    x = 100 * math.exp(-0.5 * (T_END - 3))
    expected = {"t0": (1, 1, 1), "d1": (0, 1, 0), "d2": (0, 0, 1)}
    for param, row in expected.items():
        for species, on in enumerate(row):
            assert _column(result, species, param) == pytest.approx(
                on * 0.5 * x, rel=1e-6, abs=1e-6
            )
    assert raised == [] and result.one_sided_sensitivities == []


@pytest.mark.parametrize("clock", CLOCKS)
def test_a_later_switch_on_its_own_is_not_reported_as_one_sided(tmp_path, clock):
    # The kink at 3, then an ordinary switch at 5 in another law. Only the
    # first is on a kink; the second has a derivative on both sides.
    params = ["t0 3", "t_delta 0", "sigma t0+t_delta", "k 0.5", "late 5", "k2 0.25"]
    laws = {**PRODUCT, "b": "if(CLOCK>=late,k2,0)"}
    model = _net(tmp_path, clock, params, laws, species=2)
    result, raised = _run(model, ["t0", "t_delta", "late"])
    x2 = 100 * math.exp(-0.25 * (T_END - 5))
    assert _column(result, 1, "late") == pytest.approx(0.25 * x2, rel=1e-6)
    assert _column(result, 1, "t_delta") == pytest.approx(0.0, abs=1e-6)
    assert result.one_sided_sensitivities == [(3.0, {"t_delta": "increasing"})]
    assert len(raised) == 1 and "'late'" not in str(raised[0].message)


def test_on_a_clock_a_requested_parameter_moves_the_left_over_rule_is_not_used(tmp_path):
    # The counter rises at `one`, and `one` is asked for: every crossing on the
    # counter moves with it, and a column there is not a jump times dt*/dp alone.
    model = _net(
        tmp_path,
        "t",
        SHIFTED,
        {"a": "if(CLOCK>=t0,k,0)", "b": "if(CLOCK>=sigma,k2,0)"},
        species=2,
    )
    with pytest.raises(SensitivityUnsupportedError, match="cannot be told apart"):
        _run(model, ["t0", "t_delta", "one"])


# ─── Refused where the instant's merged jump would leave a crossing out ──────
# The crossing left over is given the merged jump, read by nudging its clock 64
# ulp across it. A crossing the nudge does not flip is not in it: the t0
# column came back 0 for 11.8, with no error (found in review).

TWO_LAWS = {"a": "if(CLOCK>=t0,k,0)", "b": "if(CLOCK>=sigma,k2,0)"}


@pytest.mark.parametrize("t_delta", ["6e-14", "1e-13", "1e-12"])
def test_crossings_past_the_nudge_and_within_the_instant_are_refused(tmp_path, t_delta):
    params = ["t0 3", f"t_delta {t_delta}", "sigma t0+t_delta", "k 0.5", "k2 0.25"]
    model = _net(tmp_path, "time()", params, TWO_LAWS, species=2)
    with pytest.raises(SensitivityUnsupportedError, match="ulp from it"):
        _run(model, ["t0", "t_delta"])


@pytest.mark.parametrize("t_delta", ["0", "3e-14"])
def test_crossings_within_the_nudge_take_the_merged_jump(tmp_path, t_delta):
    params = ["t0 3", f"t_delta {t_delta}", "sigma t0+t_delta", "k 0.5", "k2 0.25"]
    model = _net(tmp_path, "time()", params, TWO_LAWS, species=2)
    result, _ = _run(model, ["t0", "t_delta"])
    x2 = 100 * math.exp(-0.25 * (T_END - 3))
    assert _column(result, 1, "t0") == pytest.approx(0.25 * x2, rel=1e-6)


@pytest.mark.parametrize(
    "laws",
    [
        {"a": "if(time()>=t0,k,0)", "b": "if(t>=sigma,k2,0)"},
        {"a": "if(t>=t0,k,0)", "b": "if(time()>=sigma,k2,0)"},
        {"a": "if(time()>=t0,1,0)*if(t>=sigma,k,0)"},
        {"a": "if(time()>=t0,k,0)", "b": "if(time()>=sigma,k2,0)", "c": "if(t>=tc,k2,0)"},
    ],
    ids=[
        "time-then-counter",
        "counter-then-time",
        "a-product-on-two-clocks",
        "a-third-gate-elsewhere",
    ],
)
def test_a_crossing_on_another_clock_is_refused(tmp_path, laws):
    params = ["t0 3", "t_delta 0", "sigma t0+t_delta", "tc 3", "k 0.5", "k2 0.25"]
    model = _net(tmp_path, "time()", params, laws, species=len(laws))
    requested = ["t0", "t_delta", "tc"] if "c" in laws else ["t0", "t_delta"]
    with pytest.raises(SensitivityUnsupportedError, match="on another clock"):
        _run(model, requested)


def test_a_window_that_closes_as_a_singular_power_is_refused(tmp_path):
    # X decays at k·(sigma − t)^0.2 on [t0, sigma), so X = 100·exp(−k·d^1.2/1.2)
    # and the derivative for d increasing is 0. What a hair past the instant
    # reads goes as the hair to the 0.2 and is no jump: −2.9 came back (#949).
    params = ["t0 3", "d 0", "sigma t0+d", "k 0.5"]
    law = {"a": "if(CLOCK>=t0,if(CLOCK<sigma,k*(sigma-CLOCK)^0.2,0),0)"}
    model = _net(tmp_path, "time()", params, law)
    with pytest.raises(SensitivityRefusedForParametersError, match="is not a jump"):
        _run(model, ["t0", "d"])


# ─── Results made from several runs keep the entries ─────────────────────────


@pytest.mark.parametrize("n_workers", [1, 3])
def test_chunked_sensitivities_keep_the_entries_and_warn_once_at_the_caller(tmp_path, n_workers):
    model = _net(tmp_path, "time()", SHIFTED, PRODUCT)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bngsim.Simulator(model, method="ode").compute_all_sensitivities(
            (0.0, T_END), 7, params=["k", "t0", "t_delta"], chunk_size=1, n_workers=n_workers
        )
    raised = [w for w in caught if issubclass(w.category, OneSidedSensitivityWarning)]
    assert result.one_sided_sensitivities == [(3.0, {"t_delta": "increasing"})]
    assert [w.filename for w in raised] == [__file__]
    x = 100 * math.exp(-0.5 * (T_END - 3))
    assert _column(result, 0, "t_delta") == pytest.approx(0.5 * x, rel=1e-6)


def test_a_result_cut_by_a_stop_condition_keeps_the_entries(tmp_path):
    # The run is cut at the first sample past X1 < 50; the result it carries
    # is built again from the run's, and keeps what the run read.
    model = _net(tmp_path, "time()", SHIFTED, PRODUCT)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["t_delta"])
    sim.add_stop_condition("X1obs < 50")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OneSidedSensitivityWarning)
        with pytest.raises(bngsim.StopConditionMet) as met:
            sim.run((0.0, T_END), 61)
    result = met.value.result
    assert result.time[-1] < T_END  # cut short
    assert result.one_sided_sensitivities == [(3.0, {"t_delta": "increasing"})]


@pytest.mark.parametrize("clock", CLOCKS)
def test_a_reading_that_is_no_jump_at_one_instant_does_not_refuse_the_next(tmp_path, clock):
    # At 2 a law that rises from its switch as a power, beside a gate it does
    # not compose with: no kink, so what the hair reads there, which goes as
    # the hair to the 1.5 and is no jump, is not asked to be one. At 4 the product of two gates on a bound: one-sided, as alone.
    params = [
        "t0 4",
        "t_delta 0",
        "sigma t0+t_delta",
        "k 0.5",
        "a0 2",
        "ad 0",
        "sa a0+ad",
        "k3 0.1",
    ]
    laws = {
        "a": "if(CLOCK>=t0,1,0)*if(CLOCK>=sigma,k,0)",
        "b": "if(CLOCK>=a0,k3,0)",
        "c": "if(CLOCK>=sa,k3*(CLOCK-sa)^1.5,0)",
    }
    model = _net(tmp_path, clock, params, laws, species=3)
    # a0 is asked for, so the pair at 2 is read as ordered, sa after a0.
    result, raised = _run(model, ["t_delta", "ad", "a0"])
    assert (4.0, {"t_delta": "increasing"}) in result.one_sided_sensitivities
    x = 100 * math.exp(-0.5 * (T_END - 4))
    assert _column(result, 0, "t_delta") == pytest.approx(0.5 * x, rel=1e-6)
