"""AMICI is run through the instants its event triggers switch at (issue #1542).

AMICI rewrites ``time == c`` as ``(time >= c) && (time <= c)``, whose trigger
function ``min(t - c, c - t)`` touches 0 at c without changing sign, so it fires
the event only when one of its output times lands on c. bngsim fires it wherever
c falls (#1015), so on a job grid with no sample there the sweep read AMICI's
miss as a bngsim divergence: BIOMD0000000816, 817, 928 and 1059 went PASS ->
DIFF. ``_amici_common.amici_ode(..., through_event_instants=True)`` gives AMICI
the instants ``_rr_common.event_instants`` finds, as #1024 did for RoadRunner,
and reports the job's own grid.

The fifth #1542 model, BIOMD0000000285, is the other AMICI miss. At t = 0 AMICI
reads a trigger whose residual is exactly 0 as true, so ``p38death > 0`` with
p38death(0) = 0 and the L2V4 default initialValue = true never fires. Instants
cannot help there (it reads a species, not the clock), so it is an
``amici_dispositions`` entry, and ``test_amici_reads_a_zero_residual_as_true_at_the_start``
is that entry's premise.

``amici_value_of``, ``with_event_instants`` and ``amici_run``'s opt-in are
tested without AMICI. The others build small models with AMICI (cached on disk
under ``amici_parity/amici_cache`` after the first run) and skip where it is not
installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libsbml")

import _amici_common as ac  # noqa: E402

T = '<csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>'
MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def _species(ids) -> str:
    return "".join(
        f'<species id="{i}" compartment="C" initialConcentration="0" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        for i in ids
    )


def _event(eid: str, trigger: str, target: str, initial_value: str = "false") -> str:
    return (
        f'<event id="{eid}" useValuesFromTriggerTime="true">'
        f'<trigger initialValue="{initial_value}" persistent="true">'
        f"<math {MATH}>{trigger}</math></trigger>"
        f'<listOfEventAssignments><eventAssignment variable="{target}">'
        f"<math {MATH}><cn>1</cn></math></eventAssignment></listOfEventAssignments>"
        "</event>"
    )


def _sbml(species, events, *, params: str = "", extra: str = "", size: float = 1.0) -> str:
    """An L3V2 model in compartment C of ``size``: ``species`` at 0, the
    constant ``params``, ``extra`` (initial assignments, rules) and ``events``."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments>
   <compartment id="C" spatialDimensions="3" size="{size}" constant="true"/>
  </listOfCompartments>
  <listOfSpecies>{_species(species)}</listOfSpecies>
  <listOfParameters>{params}</listOfParameters>
  {extra}
  <listOfEvents>{"".join(events)}</listOfEvents>
 </model>
</sbml>
"""


def _param(pid: str, value: float) -> str:
    return f'<parameter id="{pid}" value="{value}" constant="true"/>'


def _initial_assignment(pid: str, value: float) -> str:
    return (
        f'<listOfInitialAssignments><initialAssignment symbol="{pid}">'
        f"<math {MATH}><cn>{value}</cn></math></initialAssignment>"
        "</listOfInitialAssignments>"
    )


def _eq_time(rhs: str) -> str:
    return f"<apply><eq/>{T}{rhs}</apply>"


# --------------------------------------------------------------------------- #
# amici_value_of, against a stand-in for AMICI's model
# --------------------------------------------------------------------------- #
class _StandInModel:
    """The four accessors ``amici_value_of`` reads, with AMICI's semantics: free
    parameters (unscaled) and fixed parameters, each as ids and values in order."""

    def __init__(self, free: dict[str, float], fixed: dict[str, float] | None = None):
        self._free, self._fixed = free, fixed or {}

    def get_free_parameter_ids(self):
        return list(self._free)

    def get_unscaled_parameters(self):
        return list(self._free.values())

    def get_fixed_parameter_ids(self):
        return list(self._fixed)

    def get_fixed_parameters(self):
        return list(self._fixed.values())


def test_value_of_reads_amicis_values_then_the_sbml_where_amici_has_none():
    """AMICI's own value wins (it is after the initialAssignment), under AMICI's
    rename of a colliding id; an id AMICI folded away is read from the SBML
    unless an initialAssignment sets it, in which case it gives no value."""
    params = _param("k", 9.0) + _param("p", 0.75) + _param("f", 4.0)
    params += _param("h", 2.5) + _param("g", 1.0)
    sbml = _sbml(["S"], [], params=params, extra=_initial_assignment("g", 7.0), size=0.63)
    model = _StandInModel({"k": 3.0, "amici_p": 0.53}, {"f": 5.0})
    value_of = ac.amici_value_of(model, sbml)
    assert value_of("k") == 3.0  # AMICI's, not the SBML's 9
    assert value_of("p") == 0.53  # AMICI renamed p to amici_p
    assert value_of("f") == 5.0  # a fixed parameter
    assert value_of("C") == 0.63  # a compartment AMICI keeps no parameter for
    assert value_of("h") == 2.5  # a parameter AMICI keeps none for, unassigned
    assert value_of("g") is None  # ... and one whose start value AMICI would know
    assert value_of("nope") is None


def test_a_rename_is_not_read_when_the_sbml_has_that_id_itself():
    """With both ``p`` and ``amici_p`` in the SBML, AMICI's ``amici_p`` may be
    either, so ``p`` is not read from it (the no-guessing rule of
    ``normalize_amici_names``); it falls back to the SBML's own value."""
    sbml = _sbml(["S"], [], params=_param("p", 0.75) + _param("amici_p", 0.2))
    value_of = ac.amici_value_of(_StandInModel({"amici_p": 0.2}), sbml)
    assert value_of("p") == 0.75
    assert value_of("amici_p") == 0.2


# --------------------------------------------------------------------------- #
# AMICI runs
# --------------------------------------------------------------------------- #
_GRID = (0.0, 1.0, 11)  # samples every 0.1; no event instant falls on one


def _ulps_below(x: float, n: int) -> float:
    for _ in range(n):
        x = float(np.nextafter(x, -np.inf))
    return x


# Equality triggers on the clock, against: a number (A); a parameter whose
# initialAssignment moves it, 0.75 -> 0.53, which AMICI renames to amici_p (B);
# the compartment's size, which AMICI keeps no parameter for (D); a number
# written to 17 digits, which AMICI reads at 15, 0.733333333333333 (E); a
# number in e-notation with an 18-digit mantissa, which AMICI rebuilds as
# 1.46610990695848 * 10**-1 = 0.14661099069584801, neither the exact double
# (0.14661099069584832) nor its 15-digit rounding (0.146610990695848) (H); and a
# parameter 6 ulps below the grid time 0.3, an instant AMICI cannot be given
# because it fails a run with output times that close (G).
_Q = _ulps_below(0.3, 6)
_H_AT = 0.14661099069584801
_TIME_SBML = _sbml(
    ["A", "B", "D", "E", "G", "H"],
    [
        _event("EA", _eq_time("<cn>0.33</cn>"), "A"),
        _event("EB", _eq_time("<ci>p</ci>"), "B"),
        _event("ED", _eq_time("<ci>C</ci>"), "D"),
        _event("EE", _eq_time("<cn>0.73333333333333331</cn>"), "E"),
        _event("EG", _eq_time("<ci>q</ci>"), "G"),
        _event("EH", _eq_time('<cn type="e-notation"> 1.46610990695848304 <sep/> -1 </cn>'), "H"),
    ],
    params=_param("p", 0.75) + _param("q", _Q),
    extra=_initial_assignment("p", 0.53),
    size=0.63,
)

# ``B > 0`` with B(0) = 0 and dB/dt = 1, once per initialValue (F_true,
# F_false), and three triggers that are never true, each with initialValue =
# false: ``D > 0`` with D(0) = 0 and dD/dt = -1 (N_falls), ``Z > 0`` with Z = 0
# throughout (N_flat), and ``B < 0`` (N_rises). Last, ``L > 0`` with initialValue
# = true, L held at 0 until t = 0.5 and rising after (F_late): every reading of
# the SBML start rule fires that one, just after 0.5. The triggers read species
# only, so the model has no event instants.
_GT0 = "<apply><gt/><ci>{}</ci><cn>0</cn></apply>"
_LATE_RATE = (
    f"<piecewise><piece><cn>0</cn><apply><lt/>{T}<cn>0.5</cn></apply></piece>"
    "<otherwise><cn>1</cn></otherwise></piecewise>"
)
_START_SBML = _sbml(
    ["B", "D", "Z", "L", "F_true", "F_false", "N_falls", "N_flat", "N_rises", "F_late"],
    [
        _event("Etrue", _GT0.format("B"), "F_true", "true"),
        _event("Efalse", _GT0.format("B"), "F_false", "false"),
        _event("Efalls", _GT0.format("D"), "N_falls", "false"),
        _event("Eflat", _GT0.format("Z"), "N_flat", "false"),
        _event("Erises", "<apply><lt/><ci>B</ci><cn>0</cn></apply>", "N_rises", "false"),
        _event("Elate", _GT0.format("L"), "F_late", "true"),
    ],
    extra=(
        "<listOfRules>"
        f'<rateRule variable="B"><math {MATH}><cn>1</cn></math></rateRule>'
        f'<rateRule variable="D"><math {MATH}><cn>-1</cn></math></rateRule>'
        f'<rateRule variable="L"><math {MATH}>{_LATE_RATE}</math></rateRule>'
        "</listOfRules>"
    ),
)


def _column(run, name):
    t, v, names, _ = run
    return v[:, list(names).index(name)]


def _amici(sbml: str, **kw):
    pytest.importorskip("amici")
    return ac.amici_ode(sbml, *_GRID, 1e-9, 1e-12, **kw)


def test_amici_skips_a_time_equality_between_its_samples():
    """The premise: if this starts failing, AMICI has learned to fire these
    events on its own and the instants are no longer needed."""
    run = _amici(_TIME_SBML)
    for name in ("A", "B", "D", "E", "G", "H"):
        assert np.all(_column(run, name) == 0.0), name
    assert run[3]["event_instants"] == 0


def test_through_the_instants_it_fires_them_and_reports_the_job_grid():
    run = _amici(_TIME_SBML, through_event_instants=True)
    t = run[0]
    np.testing.assert_array_equal(t, np.linspace(*_GRID))
    assert run[1].shape[0] == len(t)
    fires = (("A", 0.33), ("B", 0.53), ("D", 0.63), ("E", 0.733333333333333), ("H", _H_AT))
    for name, at in fires:
        col = _column(run, name)
        assert np.all(col[t < at] == 0.0) and np.all(col[t > at] == 1.0), name
    # G's instant is left out, so AMICI runs and misses it, as on the grid alone.
    assert np.all(_column(run, "G") == 0.0)
    assert run[3]["event_instants"] == 5


def test_an_instant_next_to_another_output_time_is_left_out():
    """AMICI can fail a run whose output times sit a few ulps apart once an event
    fires at the first (``_TIME_SBML``'s G). An instant that close to a grid time,
    or to an instant already kept, is not added; the grid is kept whole."""
    grid = np.linspace(*_GRID)
    instants = [0.33, _ulps_below(0.3, 6), float(np.nextafter(0.5, 1.0)), 1.0, 0.71]
    instants.append(float(np.nextafter(0.71, 1.0)))
    out = ac.with_event_instants(grid, instants)
    np.testing.assert_array_equal(out, np.sort(np.concatenate([grid, [0.33, 0.71]])))
    np.testing.assert_array_equal(ac.with_event_instants(grid, []), grid)
    # The separation is 1e-12 * max(1, |t|): absolute below 1, relative above.
    sep_700 = 1e-12 * 700
    wide = np.array([0.0, 0.1, 700.0, 1400.0])
    near = [0.1 + 5e-13, 0.1 + 2e-12, 700.0 + 0.5 * sep_700, 700.0 + 2 * sep_700]
    out = ac.with_event_instants(wide, near)
    np.testing.assert_array_equal(out, np.sort(np.concatenate([wide, [near[1], near[3]]])))


def test_a_model_without_time_triggers_runs_as_before():
    plain = _amici(_START_SBML)
    through = _amici(_START_SBML, through_event_instants=True)
    np.testing.assert_array_equal(plain[0], through[0])
    np.testing.assert_array_equal(plain[1], through[1])
    assert through[3]["event_instants"] == 0


def test_amici_reads_a_zero_residual_as_true_at_the_start():
    """The premise of the ``BIOMD0000000285:ode`` disposition. At t = 0 AMICI
    reads a trigger whose residual is exactly 0 as true, for ``>`` as for ``>=``.
    With initialValue = true, ``B > 0`` then never fires, and neither does
    ``L > 0`` with L rising only from t = 0.5, which every reading of the SBML
    start rule fires. With initialValue = false AMICI fires ``B > 0`` at t = 0
    itself, and also three triggers that are never true. bngsim and RoadRunner
    fire F_late after 0.5 and none of the three. Those two are the evidence
    that does not lean on bngsim's reading of the start rule. If this starts
    failing, AMICI has changed here, and the disposition will report itself
    STALE or DRIFTED."""
    run = _amici(_START_SBML)
    assert np.all(_column(run, "F_true") == 0.0)
    assert np.all(_column(run, "F_late") == 0.0)
    assert _column(run, "L")[-1] == pytest.approx(0.5, rel=1e-6)  # L itself is right
    assert np.all(_column(run, "F_false") == 1.0)  # row 0 included: fired at t = 0
    for name in ("N_falls", "N_flat", "N_rises"):
        assert np.all(_column(run, name) == 1.0), name


# --------------------------------------------------------------------------- #
# amici_run opts in
# --------------------------------------------------------------------------- #
def test_amici_run_runs_amici_through_the_instants(monkeypatch):
    """The sweep's worker asks for the instants and says so in the row, as
    rr_run does ('RR through N event instants'). Both engines are stand-ins, so
    this runs without AMICI."""
    import amici_run as ar

    grid = np.linspace(*_GRID)
    values = np.zeros((len(grid), 1))
    seen = {}

    def fake_amici_ode(xml, t_start, t_end, n_points, rtol, atol, **kw):
        seen.update(kw)
        return grid, values, ["S"], {"event_instants": 2}

    monkeypatch.setattr(ac, "measure_warmup", lambda: {})
    monkeypatch.setattr(ac, "set_amici_quiet", lambda: None)
    monkeypatch.setattr(ac, "bn_ode", lambda *a, **k: (grid, values, ["S"], {}))
    monkeypatch.setattr(ac, "amici_ode", fake_amici_ode)
    spec = {
        "key": "M|ode",
        "model_id": "M",
        "method": "ode",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "xml": "unused.xml",
        "params": {"t_start": 0.0, "t_end": 1.0, "n_points": 11, "rtol": 1e-9, "atol": 1e-12},
    }
    out: list[dict] = []

    class _Queue:
        put = staticmethod(out.append)

    ar._worker(spec, _Queue())
    assert seen == {"through_event_instants": True}
    (res,) = out
    assert res["status"] == "pass"
    assert res["comment"].endswith("; AMICI through 2 event instants")


# --------------------------------------------------------------------------- #
# The sensitivity job runs AMICI through the instants too (issue #1571)
# --------------------------------------------------------------------------- #
def _amici_sens(tmp_path, sbml: str, **kw):
    # A compiled AMICI model reads amici.sim.sundials at import; the runner's
    # warmup imports it before any model loads.
    pytest.importorskip("amici.sim.sundials")
    import _amici_sens as asens

    xml = tmp_path / "m.xml"
    xml.write_text(sbml)
    ids, built = asens.amici_free_parameter_ids(str(xml))
    return asens.amici_sens(built, *_GRID, 1e-9, 1e-12, ids[:1], "staggered", sbml=sbml, **kw)


def test_amicis_sensitivity_run_fires_them_through_the_instants(tmp_path):
    """Without the instants AMICI's sensitivity run misses the equalities as its
    ODE run does (816, 817 and 928 read DIFF in the sensitivity job, state DIFF
    included); through them it fires each one and reports the job grid, its
    sensitivities included."""
    plain = _amici_sens(tmp_path, _TIME_SBML)
    assert np.all(plain[1] == 0.0) and plain[4]["event_instants"] == 0
    t, x, sx, names, timing = _amici_sens(tmp_path, _TIME_SBML, through_event_instants=True)
    np.testing.assert_array_equal(t, np.linspace(*_GRID))
    assert x.shape[0] == sx.shape[0] == len(t)
    names = ac.normalize_amici_names(list(names))
    for name, at in (("A", 0.33), ("B", 0.53), ("D", 0.63), ("E", 0.733333333333333)):
        col = x[:, names.index(name)]
        assert np.all(col[t < at] == 0.0) and np.all(col[t > at] == 1.0), name
    assert timing["event_instants"] == 5


def test_the_sens_worker_runs_amici_through_the_instants(monkeypatch, tmp_path):
    """The sensitivity sweep asks for them and says so in the row, as amici_run
    does. Both engines are stand-ins, so this runs without AMICI."""
    import amici_sens_run as asr

    grid = np.linspace(*_GRID)
    x = np.column_stack([np.exp(-grid), 1.0 - np.exp(-grid)])
    run = (grid, x, x[:, :, None] * 0.5, ["S", "R"])
    seen = {}

    def fake_amici_sens(*a, **kw):
        seen.update(kw)
        return (*run, {"integrate_sec": 0.01, "event_instants": 2, "strict_onset_samples": []})

    xml = tmp_path / "m.xml"
    xml.write_text(_TIME_SBML)
    monkeypatch.setattr(asr.asens, "measure_warmup", lambda include_amici=True: {})
    monkeypatch.setattr(asr.asens, "set_amici_quiet", lambda: None)
    monkeypatch.setattr(asr.asens, "amici_free_parameter_ids", lambda xml: (["p"], "built"))
    monkeypatch.setattr(
        asr.asens, "shared_sensitivity_params", lambda *a, **k: (["p"], {"p": "p"}, 1)
    )
    monkeypatch.setattr(asr.asens, "amici_sens", fake_amici_sens)
    monkeypatch.setattr(asr.asens, "bn_sens", lambda *a, **k: (*run, {}))
    spec = {
        "key": "M:sens:staggered",
        "model_id": "M",
        "method": "sens/staggered",
        "sens_method": "staggered",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "xml": str(xml),
        "params": {"t_start": 0.0, "t_end": 1.0, "n_points": 11, "rtol": 1e-9, "atol": 1e-12},
        "param_cap": 20,
        "param_budget": 20_000,
        "max_steps": 10_000,
        "config_env": {},
    }
    out: list[dict] = []

    class _Queue:
        put = staticmethod(out.append)

    asr._worker(spec, _Queue())
    assert seen["through_event_instants"] is True
    (res,) = out
    assert res["status"] == "pass"
    assert res["comment"].endswith("; AMICI through 2 event instants")
    # A recorded entry says how AMICI was run, so a replay of one made without
    # the instants is refused.
    assert asr.archive_identity(spec)["through_event_instants"] is True
