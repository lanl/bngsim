"""GH #247 — non-finite initialAssignment / parameter values (INF / -INF / NaN).

Two failure modes lived under one issue:

* the loader's section-0 fixpoint loop decided "did this fold move the value?"
  with ``abs(new - old) > 1e-30``, which is NaN-hostile — ``abs(NaN - 0.0)`` is
  NaN and ``NaN > 1e-30`` is False, so a ``<notanumber/>`` initialAssignment was
  treated as "no change" and the target kept its stale raw value. ``<infinity/>``
  survived only because ``abs(inf - 0)`` is ``inf > 1e-30``. Suite case 00950's
  ``R`` came out 0.0 instead of NaN;
* the suite grader compared these sentinels with a numeric tolerance, which can
  never hold for inf/NaN — that fix lives in ``test_sbml_suite_grading.py``.

These tests pin the *engine* half: a non-finite initialAssignment lands as the
IEEE value, and — critically for suite cases 01811 / 01813 — an id merely
*spelled* ``INF`` / ``NaN`` stays a normal finite symbol, distinct from the
``<infinity/>`` / ``<notanumber/>`` MathML constants.
"""

import math

import bngsim

_HDR = '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'


def _ia_params(assignments, *, raw=None):
    """Build a params-only model. ``assignments`` is ``[(id, mathml)]`` turned
    into initialAssignments; ``raw`` is ``{id: value_str}`` for parameters that
    also carry a plain ``value=`` (used for the same-spelled-id cases). Returns
    ``{id: folded value}``."""
    raw = raw or {}
    ids = [pid for pid, _ in assignments] + [pid for pid in raw if pid not in dict(assignments)]
    params = "".join(
        f'<parameter id="{pid}" constant="true"'
        + (f' value="{raw[pid]}"' if pid in raw else "")
        + "/>"
        for pid in ids
    )
    ias = "".join(
        f'<initialAssignment symbol="{pid}">'
        f'<math xmlns="http://www.w3.org/1998/Math/MathML">{mathml}</math>'
        f"</initialAssignment>"
        for pid, mathml in assignments
    )
    sbml = (
        f'{_HDR}\n<model id="m">'
        f"<listOfParameters>{params}</listOfParameters>"
        f"<listOfInitialAssignments>{ias}</listOfInitialAssignments>"
        f"</model></sbml>"
    )
    model = bngsim.Model.from_sbml_string(sbml)
    return {pid: _get_param(model, pid) for pid in ids}


def _get_param(model, pid):
    # bngsim renames ids that collide with reserved tokens (e.g. inf/nan and
    # their case variants) to an ``_ant_`` form, exactly as the suite harness
    # undoes. Resolve either spelling so the test reads the same value the
    # grader would.
    for name in (pid, f"_ant_{pid}"):
        if name in model.param_names:
            return model.get_param(name)
    raise KeyError(pid)


_INF = "<infinity/>"
_NEG_INF = "<apply><minus/><infinity/></apply>"
_NAN = "<notanumber/>"


def test_infinity_initial_assignment():
    assert _ia_params([("P", _INF)])["P"] == math.inf


def test_negative_infinity_initial_assignment():
    assert _ia_params([("Q", _NEG_INF)])["Q"] == -math.inf


def test_notanumber_initial_assignment():
    # The regressed case: <notanumber/> was dropped, leaving the raw 0.0.
    assert math.isnan(_ia_params([("R", _NAN)])["R"])


def test_case_00950_shape_all_three():
    vals = _ia_params([("P", _INF), ("Q", _NEG_INF), ("R", _NAN)])
    assert vals["P"] == math.inf
    assert vals["Q"] == -math.inf
    assert math.isnan(vals["R"])


def test_literal_nonfinite_parameter_values():
    # Case 00951: parameters carry INF / -INF / NaN directly as value=.
    sbml = (
        f'{_HDR}\n<model id="m"><listOfParameters>'
        f'<parameter id="P" value="INF" constant="true"/>'
        f'<parameter id="Q" value="-INF" constant="true"/>'
        f'<parameter id="R" value="NaN" constant="true"/>'
        f"</listOfParameters></model></sbml>"
    )
    m = bngsim.Model.from_sbml_string(sbml)
    assert m.get_param("P") == math.inf
    assert m.get_param("Q") == -math.inf
    assert math.isnan(m.get_param("R"))


def test_ids_spelled_inf_are_ordinary_finite_symbols():
    # Case 01811: ids INF/Inf/inf are finite parameters; <ci> to them resolves
    # to those finite values, while <infinity/> is the numeric constant. The
    # spelling of an id must never be confused with the MathML constant.
    vals = _ia_params(
        [
            ("a", "<ci>INF</ci>"),
            ("b", "<ci>Inf</ci>"),
            ("c", "<ci>inf</ci>"),
            ("d", _INF),
        ],
        raw={"INF": "10", "Inf": "1", "inf": "0.1"},
    )
    assert vals["INF"] == 10.0
    assert vals["Inf"] == 1.0
    assert vals["inf"] == 0.1
    assert vals["a"] == 10.0
    assert vals["b"] == 1.0
    assert vals["c"] == 0.1
    assert vals["d"] == math.inf  # the <infinity/> constant, not id 'inf'


def test_ids_spelled_nan_are_ordinary_finite_symbols():
    # Case 01813: ids NAN/NaN/nan are finite parameters; <notanumber/> is the
    # NaN constant, kept distinct from any id spelled 'nan'.
    vals = _ia_params(
        [
            ("p", "<ci>NAN</ci>"),
            ("q", "<ci>NaN</ci>"),
            ("r", "<ci>nan</ci>"),
            ("s", _NAN),
        ],
        raw={"NAN": "0.1", "NaN": "0.01", "nan": "0.001"},
    )
    assert vals["NAN"] == 0.1
    assert vals["NaN"] == 0.01
    assert vals["nan"] == 0.001
    assert vals["p"] == 0.1
    assert vals["q"] == 0.01
    assert vals["r"] == 0.001
    assert math.isnan(vals["s"])  # the <notanumber/> constant, not id 'nan'
