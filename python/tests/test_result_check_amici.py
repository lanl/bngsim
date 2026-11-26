"""``Result.check()``'s ``"sensitivity_second_engine"``: a run's forward
sensitivities against AMICI's, from the model file (issue #1923).

The oracle is a closed form: for ``A' = -kA``, ``dA/dk = -t·A0·e^{-kt}``. Most
tests here hand the check a stand-in for AMICI that returns the closed form
(:class:`_ClosedForm`), so they run wherever bngsim does, and AMICI is no
dependency of bngsim. The few marked by ``_amici`` build real AMICI models,
two small ones, cached in the suite's own directory, and skip where ``import
amici`` fails.

A wrong sensitivity is made to order three ways:

* in the run alone (a solve that is not converged), which the tolerance tier
  sees too;
* in every sensitivity solve, with the loader moving ``k`` twice as far as it
  is asked to (:func:`_loader_moves_k_twice`). The trajectory at the file's
  ``k`` is right, the forward sensitivities and every difference quotient of
  runs agree with each other, and both are twice the truth: the #144 shape, a
  defect the loader and the sensitivity code share, which only a second engine
  sees;
* in AMICI's answer, for the checks of the rule itself.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from bngsim import Model, Result, Simulator, _check, _check_amici, _check_sensitivity, examples

NAME = "sensitivity_second_engine"
K0, A0 = 0.5, 100.0
_ROOT = Path(__file__).resolve().parents[2]


# ─── Models ──────────────────────────────────────────────────────────────────


def _write_sbml(
    path: Path,
    *,
    species: list[tuple[str, float, bool]],
    params: dict[str, float],
    reactions: list[tuple[str, list[str], list[str], str, dict[str, float]]],
    size: float = 1.0,
    initial: dict[str, str] | None = None,
    rules: dict[str, str] | None = None,
    functions: dict[str, str] | None = None,
    events: list[tuple[str, dict[str, str]]] | None = None,
) -> Path:
    """An SBML L3V2 file: one compartment ``C``; *species* as ``(id, initial
    value, hasOnlySubstanceUnits)`` (an amount where that is true, else a
    concentration); *reactions* as ``(id, reactants, products, rate law, local
    parameters)``; formulas in libSBML's L3 syntax."""
    import libsbml

    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId("m")

    def math(formula: str):
        node = libsbml.parseL3Formula(formula)
        assert node is not None, libsbml.getLastParseL3Error()
        return node

    c = m.createCompartment()
    c.setId("C"), c.setSize(size), c.setConstant(True), c.setSpatialDimensions(3)
    for sid, value, hosu in species:
        s = m.createSpecies()
        s.setId(sid), s.setCompartment("C"), s.setHasOnlySubstanceUnits(hosu)
        s.setBoundaryCondition(False), s.setConstant(False)
        (s.setInitialAmount if hosu else s.setInitialConcentration)(value)
    for pid, value in params.items():
        p = m.createParameter()
        p.setId(pid), p.setValue(value), p.setConstant(pid not in (rules or {}))
    for fid, formula in (functions or {}).items():
        f = m.createFunctionDefinition()
        f.setId(fid), f.setMath(math(formula))
    for sym, formula in (initial or {}).items():
        ia = m.createInitialAssignment()
        ia.setSymbol(sym), ia.setMath(math(formula))
    for var, formula in (rules or {}).items():
        r = m.createAssignmentRule()
        r.setVariable(var), r.setMath(math(formula))
    for rid, reactants, products, law, local in reactions:
        r = m.createReaction()
        r.setId(rid), r.setReversible(False)
        for sid in reactants:
            ref = r.createReactant()
            ref.setSpecies(sid), ref.setStoichiometry(1.0), ref.setConstant(True)
        for sid in products:
            ref = r.createProduct()
            ref.setSpecies(sid), ref.setStoichiometry(1.0), ref.setConstant(True)
        kl = r.createKineticLaw()
        for lid, value in local.items():
            lp = kl.createLocalParameter()
            lp.setId(lid), lp.setValue(value)
        kl.setMath(math(law))
    for i, (trigger, assignments) in enumerate(events or []):
        e = m.createEvent()
        e.setId(f"e{i}"), e.setUseValuesFromTriggerTime(True)
        t = e.createTrigger()
        t.setMath(math(trigger)), t.setPersistent(True), t.setInitialValue(False)
        for var, formula in assignments.items():
            a = e.createEventAssignment()
            a.setVariable(var), a.setMath(math(formula))
    path.write_text(libsbml.writeSBMLToString(doc), encoding="utf-8")
    return path


def _decay(path: Path | None = None, **kw) -> Result:
    """``A' = -kA`` from ``A(0) = 100`` with ``k = 0.5`` (bngsim's bundled
    SBML), sensitivities to ``k``."""
    model = Model.load(str(path or examples.path("decay", format="sbml")))
    sim = Simulator(model, method="ode", sensitivity_params=kw.pop("params", ["k"]), **kw)
    return sim.run(t_span=(0, 10), n_points=11)


def _by_name(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["checks"]}


def _verdict(r: Result, *checks: str) -> dict:
    return _by_name(r.check([*checks, NAME]))[NAME]


def _scaled(r: Result, column: int, factor: float, row: int | None = None) -> Result:
    """*r* with one parameter column (or one series of it) scaled."""
    out = copy.copy(r)
    s = np.asarray(r._sensitivities, dtype=float).copy()
    if row is None:
        s[:, :, column] *= factor
    else:
        s[:, row, column] *= factor
    out._sensitivities = s
    return out


def _loader_moves_k_twice(monkeypatch) -> None:
    """A loader that moves ``k`` twice as far from the file's value as it is
    asked to: the run at the file's ``k`` is right, its forward sensitivities
    to ``k`` are doubled, and so is every difference quotient of runs that
    move ``k`` through ``set_param``. The two agree with each other, and the
    tolerance tier sees a doubled column at every tolerance alike."""
    real_set = Model.set_param

    def set_param(self, name, value, *a, **kw):
        if name == "k":
            value = K0 + 2.0 * (float(value) - K0)
        return real_set(self, name, value, *a, **kw)

    real_run = Simulator.run

    def run(self, *a, **kw):
        out = real_run(self, *a, **kw)
        if "k" in list(out._sensitivity_params):
            out = _scaled(out, list(out._sensitivity_params).index("k"), 2.0)
        return out

    monkeypatch.setattr(Model, "set_param", set_param)
    monkeypatch.setattr(Simulator, "run", run)


# ─── A stand-in for AMICI ────────────────────────────────────────────────────


class _ClosedForm:
    """What the check reads of a built AMICI model, answering with closed
    forms: each state ``s`` decays at ``k`` from ``x0[s]`` (an amount where
    *amounts* gives the compartment's size), with ``ds/dk = -t·s``. AMICI's
    names: ``k`` is ``amici_k``."""

    def __init__(
        self,
        x0: dict[str, float],
        *,
        k: float = K0,
        free: dict[str, float] | None = None,
        state_ids: list[str] | None = None,
        commit: str | None = None,
        fail: str = "",
        edit=None,
    ) -> None:
        self.x0, self.k = x0, k
        self.free_ids = list((free or {"amici_k": k}).keys())
        self.free_values = list((free or {"amici_k": k}).values())
        self.state_ids = state_ids or list(x0)
        self.version, self.commit = "0.0-test", commit
        self.build_seconds, self.cached = 0.0, True
        self.fail, self.edit = fail, edit
        self.asked: dict = {}

    def simulate(self, ids, t0, times, *, rtol, atol, atol_sens, method):
        self.asked = dict(ids=list(ids), t0=t0, rtol=rtol, atol=atol, atol_sens=atol_sens)
        if self.fail:
            return self.fail, np.empty((0, 0)), np.empty((0, 0, 0))
        t = np.asarray(times, dtype=float)[:, None]
        x0 = np.asarray([self.x0[s.removeprefix("amici_")] for s in self.state_ids])[None, :]
        x = x0 * np.exp(-self.k * t)
        sx = np.zeros((t.size, x.shape[1], len(ids)))
        for c, p in enumerate(ids):
            if p == "amici_k":
                sx[:, :, c] = -t * x
        if self.edit is not None:
            x, sx = self.edit(x, sx)
        return "", x, sx


@pytest.fixture
def engine(monkeypatch):
    """Stand AMICI in with a :class:`_ClosedForm`; returns a setter."""
    held: dict = {}

    def use(fake: _ClosedForm) -> _ClosedForm:
        held["engine"] = fake
        return fake

    monkeypatch.setattr(_check_amici, "_import_amici", lambda: SimpleNamespace())
    monkeypatch.setattr(_check_amici, "build_engine", lambda amici, sbml: held["engine"])
    use(_ClosedForm({"A": A0}))
    return use


# ─── Where it runs ───────────────────────────────────────────────────────────


def test_all_adds_it_for_a_sensitivity_run_and_the_default_does_not():
    plain = Simulator(examples.load("decay", format="sbml"), method="ode").run(
        t_span=(0, 10), n_points=11
    )
    assert NAME not in _by_name(plain.check("all"))
    names = [c["name"] for c in _decay(params=["k"]).check()["checks"]]
    assert NAME not in names
    assert NAME in _check.SENSITIVITY
    (v,) = plain.check([NAME])["checks"]
    assert v["status"] == "unsupported" and "computed no sensitivities" in v["note"]


def test_without_amici_it_skips_and_says_why(monkeypatch):
    monkeypatch.setitem(sys.modules, "amici", None)
    rep = _decay().check(["sensitivity", NAME])
    v = _by_name(rep)[NAME]
    assert v["status"] == "skip"
    assert "AMICI is not installed" in v["note"] and "pip install amici" in v["note"]
    assert "not checked against a second engine" in v["note"]
    # A skip is no finding: the default tier passes, and so does the report.
    assert rep["ok"] is True
    assert json.loads(json.dumps(rep, allow_nan=False)) == rep


def test_a_net_model_and_a_steady_state_run_are_not_its_to_judge():
    r = Simulator(
        examples.load("decay", format="net"), method="ode", sensitivity_params=["k"]
    ).run(t_span=(0, 10), n_points=11)
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "unsupported" and "'net'" in v["note"]
    model = examples.load("decay", format="sbml")
    r = Simulator(model, method="ode", sensitivity_params=["k"]).run(
        t_span=(0, 200), n_points=11, steady_state=True
    )
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "unsupported" and "steady state" in v["note"]


def test_a_run_another_engine_cannot_make_again_from_the_file_is_skipped(tmp_path, engine):
    sbml = examples.path("decay", format="sbml")
    # Started from a saved state.
    m = Model.load(str(sbml))
    m.set_concentration("A", 3.0)
    m.save_concentrations()
    r = Simulator(m, method="ode", sensitivity_params=["k"]).run(t_span=(0, 5), n_points=6)
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "skip" and "file's initial state" in v["note"], v
    # A parameter that is not the file's.
    m = Model.load(str(sbml))
    m.set_param("k", 0.7)
    r = Simulator(m, method="ode", sensitivity_params=["k"]).run(t_span=(0, 5), n_points=6)
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "skip" and "k = 0.7" in v["note"], v
    # Sample times.
    m = Model.load(str(sbml))
    r = Simulator(m, method="ode", sensitivity_params=["k"]).run(sample_times=[0, 1, 2, 4])
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "skip" and "plain t_span run is made again in AMICI" in v["note"]
    # A model loaded from text names no file.
    m = Model.from_sbml_string(sbml.read_text())
    r = Simulator(m, method="ode", sensitivity_params=["k"]).run(t_span=(0, 5), n_points=6)
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "skip" and "no model file" in v["note"]
    # The file changed after the load, though its parameters did not.
    path = tmp_path / "decay.xml"
    path.write_text(sbml.read_text(), encoding="utf-8")
    r = _decay(path)
    path.write_text(sbml.read_text().replace("<ci> k </ci>", "<ci> k </ci> "), encoding="utf-8")
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "skip" and "changed after" in v["note"], v
    # Initial-amount columns alone: AMICI gives sensitivities to parameters.
    r = _decay(params=[], sensitivity_ic=["A"])
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "skip" and "no column AMICI can give" in v["note"]
    assert "parameters only" in v["note"]


def test_a_model_with_an_event_or_a_switch_is_not_compared(tmp_path, engine):
    base = dict(species=[("A", A0, False)], params={"k": K0, "ton": 5.0})
    cases = {
        "event": dict(
            reactions=[("J1", ["A"], [], "k*A", {})],
            events=[("time >= ton", {"A": "A + 100"})],
        ),
        "a piecewise": dict(reactions=[("J1", ["A"], [], "piecewise(k*A, time < ton, 0)", {})]),
        "max": dict(reactions=[("J1", ["A"], [], "k*max(A, 0)", {})]),
        "names an argument twice": dict(
            reactions=[("J1", ["A"], [], "k*f(A, 1)", {})],
            functions={"f": "lambda(x, x, x)"},
        ),
    }
    for what, kw in cases.items():
        path = _write_sbml(tmp_path / f"{what.replace(' ', '_')}.xml", **base, **kw)
        (v,) = _decay(path).check([NAME])["checks"]
        assert v["status"] == "skip" and what in v["note"], (what, v)
    # The bundled bolus model: a dose event at t = 5.
    r = Simulator(
        examples.load("bolus", format="sbml"), method="ode", sensitivity_params=["k"]
    ).run(t_span=(0, 10), n_points=11)
    (v,) = r.check([NAME])["checks"]
    assert v["status"] == "skip" and "event" in v["note"]


@pytest.mark.parametrize(
    ("where", "formula", "what"),
    [
        ("rule", "k*(ton > 1)", "a comparison"),
        ("rule", "k*piecewise(1, (ton > 1) && (ton < 9), 0)", "a piecewise"),
        ("initial", "k*floor(ton)", "floor"),
        ("initial", "ceil(ton)", "ceiling"),
        ("law", "k*A*min(A, ton)", "min"),
        ("law", "k*rem(A, ton)", "a remainder"),
        ("law", "k*quotient(A, ton)", "a quotient"),
        ("law", "k*delay(A, 1)", "a delay"),
        ("function", "lambda(x, piecewise(x, x > 0, 0))", "a piecewise"),
        ("function", "lambda(x, x^2)", None),
        ("law", "k*A*exp(-time)/(1 + abs(sin(A)))^2", None),
    ],
)
def test_the_scan_names_what_switches(tmp_path, where, formula, what):
    import libsbml

    kw: dict = dict(species=[("A", A0, False)], params={"k": K0, "ton": 5.0, "kk": 0.0})
    law = "kk*A"
    if where == "rule":
        kw["rules"] = {"kk": formula}
    elif where == "initial":
        kw["initial"] = {"kk": formula}
    elif where == "law":
        law = formula
    else:
        kw["functions"] = {"f": formula}
        law = "k*f(A)"
    path = _write_sbml(tmp_path / "m.xml", reactions=[("J1", ["A"], [], law, {})], **kw)
    doc = libsbml.readSBMLFromString(path.read_text())
    assert doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) == 0
    why = _check_amici.not_comparable(doc.getModel())
    if what is None:
        assert why is None
    else:
        assert why is not None and f" has {what}: " in why, why
    assert "no model" in _check_amici.not_comparable(None)


# ─── The rule, against a closed form ─────────────────────────────────────────


def test_a_closed_form_passes(engine):
    r = _decay()
    t = np.asarray(r.time)
    exact = -t * A0 * np.exp(-K0 * t)
    assert np.allclose(np.asarray(r.sensitivities)[:, 0, 0], exact, rtol=1e-6, atol=1e-8)
    fake = engine(_ClosedForm({"A": A0}, commit=_check_amici.PINNED_COMMIT))
    rep = r.check([NAME])
    (v,) = rep["checks"]
    assert v["status"] == "pass", v
    assert v["margin"] < 0.1 and v["value"] < 1e-5
    assert v["columns"] == [
        {"name": "k", "kind": "parameter", "worst_species": "A", "value": v["margin"]}
    ]
    assert v["n_judged"] == 1 and v["not_judged"] == {} and v["species_not_compared"] == {}
    assert v["state_diff"] is False and v["state_margin"] < 0.1
    assert v["engine"]["pinned"] is True and "parity suite pins" not in v["note"]
    assert v["changed"] == {"engine": f"AMICI 0.0-test ({_check_amici.PINNED_COMMIT[:9]})"}
    # AMICI was asked for bngsim's k under its own name, at the run's rtol, and
    # held at least as tight as bngsim's floor on every series.
    cols = _check_sensitivity._Columns(_check._Context(r, r.record, None))
    assert fake.asked["ids"] == ["amici_k"] and fake.asked["rtol"] == r.record["run"]["rtol"]
    assert fake.asked["atol_sens"] == pytest.approx(float(cols.atol.min()))
    assert rep["ok"] is True
    assert json.loads(json.dumps(rep, allow_nan=False)) == rep


def test_a_defect_the_loader_and_the_quotient_share_fails_here_only(monkeypatch, engine):
    _loader_moves_k_twice(monkeypatch)
    r = _decay()
    t = np.asarray(r.time)
    # The injection holds: the run is right, its sensitivity twice the truth.
    assert np.allclose(np.asarray(r.species)[:, 0], A0 * np.exp(-K0 * t), rtol=1e-6)
    assert np.allclose(
        np.asarray(r.sensitivities)[1:, 0, 0], -2.0 * t[1:] * A0 * np.exp(-K0 * t[1:]), rtol=1e-5
    )
    rep = r.check(["sensitivity", "sensitivity_fd", NAME])
    by = _by_name(rep)
    assert by["sensitivity"]["status"] == "pass", by["sensitivity"]
    assert by["sensitivity_fd"]["status"] == "pass", by["sensitivity_fd"]
    v = by[NAME]
    assert v["status"] == "diff" and v["worst_column"] == "dA/dk"
    assert v["state_diff"] is False and v["diff_not_converged"] is False
    assert v["value"] == pytest.approx(0.5, rel=1e-3)
    assert rep["ok"] is False
    assert "bngsim and AMICI disagree on this model's forward sensitivities" in rep["next"]


def test_a_column_wrong_in_the_run_alone_fails(engine):
    bad = _scaled(_decay(), 0, 1.01)
    v = _verdict(bad)
    assert v["status"] == "diff" and v["worst_column"] == "dA/dk"
    assert v["value"] == pytest.approx(0.01 / 1.01, rel=1e-2)
    # A thousandth of a percent is within the allowance.
    assert _verdict(_scaled(_decay(), 0, 1 + 1e-7))["status"] == "pass"


def test_a_small_error_in_every_solve_is_past_the_allowance(monkeypatch, engine):
    # Every solve off alike: the tighter run agrees with the run, so its drift
    # forgives nothing, and the allowance is the tolerance's alone.
    real = Simulator.run

    def off_by(factor):
        def run(self, *a, **kw):
            out = real(self, *a, **kw)
            return _scaled(out, 0, factor) if len(out._sensitivity_params) else out

        monkeypatch.setattr(Simulator, "run", run)

    off_by(1 + 1e-4)
    v = _verdict(_decay())
    assert v["status"] == "diff" and 1.0 < v["margin"] < 10.0, v
    off_by(1 + 1e-5)
    assert _verdict(_decay())["status"] == "pass"


def test_a_column_the_run_gives_as_zero_is_judged_on_amicis_scale(engine):
    v = _verdict(_scaled(_decay(), 0, 0.0))
    assert v["status"] == "diff" and v["margin"] > 100, v


def test_a_small_series_beside_a_large_one_is_judged_on_its_own_peak(tmp_path, engine):
    path = _write_sbml(
        tmp_path / "two.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {}), ("J2", ["B"], [], "k*B", {})],
    )
    engine(_ClosedForm({"A": A0, "B": 0.1}))
    r = _decay(path)
    assert _verdict(r)["status"] == "pass"
    # dB/dk is a thousandth of dA/dk: off by 1% there is far under the
    # largest series' scale, and still a finding.
    v = _verdict(_scaled(r, 0, 1.01, row=1))
    assert v["status"] == "diff" and v["worst_column"] == "dB/dk", v


def test_noise_near_zero_is_judged_on_its_columns_scale(tmp_path, engine):
    # B is held at 0.1 and k moves nothing of it: bngsim's dB/dk is 0, within
    # its tolerance of zero, which the run does not resolve.
    path = _write_sbml(
        tmp_path / "held.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {})],
    )
    r = _decay(path)
    assert np.abs(np.asarray(r.sensitivities)[:, 1, 0]).max() == 0.0

    def amici_says(value):
        def edit(x, sx):
            x, sx = x.copy(), sx.copy()
            x[:, 1], sx[:, 1, 0] = 0.1, value
            return x, sx

        return edit

    # AMICI's noise there, far past that tolerance and a millionth of the
    # column's largest series, is no finding.
    engine(_ClosedForm({"A": A0, "B": 0.1}, edit=amici_says(1e-4)))
    v = _verdict(r)
    assert v["status"] == "pass", v
    assert v["not_resolved"] == ["dB/dk"] and v["n_not_resolved"] == 1
    # A departure as large as a hundredth of the column's scale is one.
    engine(_ClosedForm({"A": A0, "B": 0.1}, edit=amici_says(1.0)))
    v = _verdict(r)
    assert v["status"] == "diff" and v["worst_column"] == "dB/dk", v


def test_a_state_amici_disagrees_on_is_a_finding_about_the_trajectory(engine):
    def off(x, sx):
        return x * 1.01, sx

    engine(_ClosedForm({"A": A0}, edit=off))
    rep = _decay().check([NAME])
    (v,) = rep["checks"]
    assert v["status"] == "diff" and v["state_diff"] is True and v["state_margin"] > 1
    assert "the states disagree" in v["note"]
    assert "disagree on this model's trajectory" in rep["next"]


def test_a_value_the_run_gives_as_not_finite_is_a_finding_and_amicis_a_skip(engine):
    bad = _decay()
    s = np.asarray(bad._sensitivities, dtype=float).copy()
    s[3, 0, 0] = np.nan
    bad._sensitivities = s
    v = _verdict(bad)
    assert v["status"] == "diff" and v["margin"] > 1e300, v

    def nan(x, sx):
        sx = sx.copy()
        sx[3, 0, 0] = np.nan
        return x, sx

    engine(_ClosedForm({"A": A0}, edit=nan))
    v = _verdict(_decay())
    assert v["status"] == "skip" and "not finite" in v["note"]


def test_a_series_not_converged_at_the_runs_tolerance_gets_no_drift(tmp_path, monkeypatch, engine):
    path = _write_sbml(
        tmp_path / "two.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {}), ("J2", ["B"], [], "k*B", {})],
    )
    engine(_ClosedForm({"A": A0, "B": 0.1}))
    r = _decay(path)
    real = _check_sensitivity._tier1

    def tier1(ctx, cols):
        move, keep, d10, held = real(ctx, cols)
        # dB/dk moves past the bound at the tighter tolerance, by as much as
        # it is off: a drift that, kept, would forgive the very error it is.
        move, d10 = move.copy(), d10.copy()
        move[1], d10[1] = 0.01, 1.0
        return move, keep, d10, held

    monkeypatch.setattr(_check_sensitivity, "_tier1", tier1)
    assert _verdict(r)["status"] == "pass"
    rep = _scaled(r, 0, 1.5, row=1).check([NAME])
    (v,) = rep["checks"]
    assert v["status"] == "diff" and v["worst_column"] == "dB/dk", v
    assert v["n_not_converged"] == 1 and v["diff_not_converged"] is True
    assert "not converged at the run's tolerance" in rep["next"]
    # With the trajectory off as well, the diff is not the tolerance's alone.

    def off(x, sx):
        return x * 1.01, sx

    engine(_ClosedForm({"A": A0, "B": 0.1}, edit=off))
    rep = _scaled(r, 0, 1.5, row=1).check([NAME])
    (v,) = rep["checks"]
    assert v["state_diff"] is True and v["diff_not_converged"] is False
    assert "disagree on this model's trajectory" in rep["next"]


def test_amicis_failures_are_skips_not_findings(engine):
    engine(_ClosedForm({"A": A0}, fail="status -1 (AMICI_TOO_MUCH_WORK)"))
    v = _verdict(_decay())
    assert v["status"] == "skip" and "AMICI could not run it" in v["note"]
    assert "TOO_MUCH_WORK" in v["note"]


# ─── What AMICI names, and what it holds ─────────────────────────────────────


def test_ids_are_matched_without_guessing():
    file_ids = {"k", "x", "J2_kl", "amici_y"}
    assert _check_amici.amici_ids(["amici_k", "J2_kl", "amici_x", "amici_y"], file_ids) == [
        "k",
        "J2_kl",
        "x",
        "amici_y",  # the file's own id
    ]
    # Two of AMICI's ids that would be one of the file's: neither is matched.
    assert _check_amici.amici_ids(["amici_x", "x"], {"x"}) == [None, None]
    assert _check_amici.bngsim_ids(["_lp_J2_kl", "k1", "J2_kl"]) == [None, "k1", None]
    assert _check_amici.bngsim_ids(["_lp_J2_kl", "k1"]) == ["J2_kl", "k1"]


def test_a_column_amici_has_no_free_parameter_for_is_named(engine):
    engine(_ClosedForm({"A": A0}, free={"other": 1.0}))
    v = _verdict(_decay())
    assert v["status"] == "skip" and "no column AMICI can give" in v["note"]
    assert "AMICI has no free parameter for it" in v["note"]
    engine(_ClosedForm({"A": A0}, free={"amici_k": 0.25}))
    v = _verdict(_decay())
    assert v["status"] == "skip" and "AMICI reads it as 0.25" in v["note"]


def test_an_amount_amici_holds_is_compared_as_a_concentration(tmp_path, engine):
    # A in a compartment of size 2, held as an amount (hasOnlySubstanceUnits):
    # AMICI integrates the amount, 100·e^{-kt}; bngsim reports half of it.
    path = _write_sbml(
        tmp_path / "amount.xml",
        species=[("A", A0, True)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {})],
        size=2.0,
    )
    r = _decay(path)
    assert np.allclose(np.asarray(r.species)[:, 0], 0.5 * A0 * np.exp(-K0 * np.asarray(r.time)))
    engine(_ClosedForm({"A": A0}))
    v = _verdict(r)
    assert v["status"] == "pass" and v["species_not_compared"] == {}, v


def test_an_amount_in_a_compartment_an_initial_assignment_sizes_is_not_compared(tmp_path, engine):
    # The file declares C's size as 1 and sets it to 2 at the start: the
    # declared size is not the size, and no guess is divided by.
    path = _write_sbml(
        tmp_path / "sized.xml",
        species=[("A", A0, True)],
        params={"k": K0, "s": 1.0},
        reactions=[("J1", ["A"], [], "k*A", {})],
        size=1.0,
        initial={"C": "2*s"},
    )
    r = _decay(path)
    assert np.allclose(np.asarray(r.species)[:, 0], 0.5 * A0 * np.exp(-K0 * np.asarray(r.time)))
    engine(_ClosedForm({"A": A0}))
    v = _verdict(r)
    assert v["status"] == "skip" and "no species AMICI integrates is bngsim's" in v["note"], v
    assert "whose size the file does not give as a constant" in v["note"]


def test_a_species_amici_holds_no_state_for_is_named(tmp_path, engine):
    path = _write_sbml(
        tmp_path / "two.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {}), ("J2", ["B"], [], "k*B", {})],
    )
    engine(_ClosedForm({"A": A0}))
    v = _verdict(_decay(path))
    assert v["status"] == "pass"
    assert v["species_not_compared"] == {"B": "AMICI integrates no state for it"}
    assert "1 species not compared" in v["note"]


def test_the_report_names_an_install_the_parity_suite_does_not_pin(engine):
    engine(_ClosedForm({"A": A0}, commit=None))
    v = _verdict(_decay())
    assert v["engine"] == {
        "name": "AMICI",
        "version": "0.0-test",
        "commit": None,
        "pinned": False,
        "cached": True,
        "build_seconds": 0.0,
        "atol_sensitivities": v["engine"]["atol_sensitivities"],
    }
    assert (
        f"not the commit bngsim's AMICI parity suite pins ({_check_amici.PINNED_COMMIT[:9]})"
        in (v["note"])
    )


def test_the_pin_is_the_parity_suites():
    pin = json.loads((_ROOT / "parity_checks" / "amici_parity" / "AMICI_PIN.json").read_text())
    assert pin["commit"] == _check_amici.PINNED_COMMIT


def test_the_version_and_commit_come_from_the_install_record(monkeypatch):
    import importlib.metadata as md

    record = json.dumps({"url": "x", "vcs_info": {"vcs": "git", "commit_id": "abc123"}})
    dist = SimpleNamespace(read_text=lambda name: record if name == "direct_url.json" else None)
    monkeypatch.setattr(md, "distribution", lambda name: dist)
    assert _check_amici.amici_version(SimpleNamespace(__version__="9.9")) == ("9.9", "abc123")
    dist = SimpleNamespace(read_text=lambda name: None)
    monkeypatch.setattr(md, "distribution", lambda name: dist)
    assert _check_amici.amici_version(SimpleNamespace(__version__="9.9")) == ("9.9", None)


# ─── The build ───────────────────────────────────────────────────────────────


class _FakeAmici:
    """An ``amici`` module whose build writes a directory, and whose model is
    a :class:`_ClosedForm`'s ids."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.builds: list[Path] = []
        self.paths: list[str] = []
        self.__version__ = "0.0-fake"
        outer = self

        class SbmlImporter:
            def __init__(self, sbml, from_file):
                assert from_file is False

            def sbml2amici(self, name, out, **kw):
                assert kw["generate_sensitivity_code"] is True
                assert kw["compute_conservation_laws"] is False
                outer.paths.append(os.environ["PATH"])
                if outer.fail is not None:
                    raise outer.fail
                Path(out, name).mkdir(parents=True)
                outer.builds.append(Path(out))

        self.SbmlImporter = SbmlImporter

    def import_model_module(self, name, where):
        assert Path(where, name).is_dir()
        model = SimpleNamespace(
            get_free_parameter_ids=lambda: ["amici_k"],
            get_unscaled_parameters=lambda: [K0],
            get_state_ids=lambda: ["A"],
        )
        return SimpleNamespace(get_model=lambda: model)


def test_a_build_is_cached_whole_and_the_tools_are_found(monkeypatch, tmp_path):
    import sysconfig

    monkeypatch.setattr(_check_amici, "CACHE_DIR", tmp_path / "amici")
    monkeypatch.setenv("PATH", "/usr/bin")
    fake = _FakeAmici()
    first = _check_amici.build_engine(fake, "<sbml/>")
    assert isinstance(first, _check_amici.Engine) and first.cached is False
    assert first.free_ids == ["amici_k"] and first.state_ids == ["A"]
    # Built in a staging directory and renamed into place; none is left over.
    (entry,) = list((tmp_path / "amici").iterdir())
    assert entry.name.startswith("bngsim_amici_") and fake.builds[0].name.startswith(".building_")
    # SWIG and CMake are found in this environment, and PATH is put back.
    assert fake.paths[0].split(os.pathsep)[0] == sysconfig.get_path("scripts")
    assert os.environ["PATH"] == "/usr/bin"
    again = _check_amici.build_engine(fake, "<sbml/>")
    assert again.cached is True and len(fake.builds) == 1
    other = _check_amici.build_engine(fake, "<sbml other/>")
    assert other.cached is False and len(list((tmp_path / "amici").iterdir())) == 2
    # The key is the text and the install, so an AMICI upgrade builds anew.
    fake.__version__ = "0.1-fake"
    assert _check_amici.build_engine(fake, "<sbml/>").cached is False


def test_a_build_that_fails_is_a_skip_and_leaves_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(_check_amici, "CACHE_DIR", tmp_path / "amici")
    why = _check_amici.build_engine(_FakeAmici(RuntimeError("swig not found")), "<sbml/>")
    assert why == "AMICI could not build it: RuntimeError: swig not found"
    assert list((tmp_path / "amici").iterdir()) == []


def test_the_build_cache_is_the_suites_own():
    cache = Path(_check_amici.CACHE_DIR)
    assert not cache.is_relative_to(Path.home() / ".cache" / "bngsim")
    assert os.environ["BNGSIM_AMICI_CACHE_DIR"] == str(cache)


# ─── Against AMICI itself ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _amici():
    return pytest.importorskip("amici.sim.sundials")


def test_amici_agrees_with_the_closed_form_and_a_shared_defect_fails(_amici, monkeypatch):
    import amici

    r = _decay()
    rep = r.check(["sensitivity", "sensitivity_fd", NAME])
    v = _by_name(rep)[NAME]
    assert v["status"] == "pass", v
    assert v["margin"] < 0.1 and v["state_margin"] < 0.1
    assert v["engine"]["version"] == amici.__version__
    assert v["changed"]["engine"].startswith(f"AMICI {amici.__version__}")
    pinned = v["engine"]["commit"] == _check_amici.PINNED_COMMIT
    assert v["engine"]["pinned"] is pinned
    assert ("parity suite pins" in v["note"]) is not pinned
    assert rep["ok"] is True
    # The build is kept whole, and the next check reads it.
    assert _verdict(r)["engine"]["cached"] is True
    entries = [p.name for p in Path(_check_amici.CACHE_DIR).iterdir()]
    assert entries and all(e.startswith("bngsim_amici_") for e in entries), entries
    # A column the run alone has wrong.
    assert _verdict(_scaled(r, 0, 1.01))["status"] == "diff"
    # A defect the loader and the sensitivity code share: only AMICI sees it.
    _loader_moves_k_twice(monkeypatch)
    by = _by_name(_decay().check(["sensitivity", "sensitivity_fd", NAME]))
    assert by["sensitivity"]["status"] == "pass" and by["sensitivity_fd"]["status"] == "pass"
    assert by[NAME]["status"] == "diff" and by[NAME]["worst_column"] == "dA/dk"
    assert by[NAME]["value"] == pytest.approx(0.5, rel=1e-3)


def test_amici_lines_up_locals_amounts_renamed_ids_and_initial_assignments(_amici, tmp_path):
    # J1 moves A (a concentration) to B (an amount in a compartment of size 2);
    # J2 moves B to x (AMICI's amici_x) at a local kl; J3 drains x at kb, which
    # an initial assignment sets to 2·k1; A starts at A0 by another.
    path = _write_sbml(
        tmp_path / "lineup.xml",
        species=[("A", 10.0, False), ("B", 4.0, True), ("x", 1.0, False)],
        params={"k1": 0.3, "kb": 0.0, "A0": 10.0},
        reactions=[
            ("J1", ["A"], ["B"], "k1*A*C", {}),
            ("J2", ["B"], ["x"], "kl*B", {"kl": 0.2}),
            ("J3", ["x"], [], "kb*x*C", {}),
        ],
        size=2.0,
        initial={"kb": "2*k1", "A": "A0"},
    )
    model = Model.load(str(path))
    r = Simulator(
        model,
        method="ode",
        sensitivity_params=["k1", "_lp_J2_kl", "A0", "kb"],
        sensitivity_ic=["B"],
    ).run(t_span=(0, 5), n_points=11)
    v = _verdict(r)
    assert v["status"] == "pass", v
    assert v["n_judged"] == 3 and set(v["not_judged"]) == {"B", "kb"}
    assert "AMICI has no free parameter" in v["not_judged"]["kb"]
    assert v["species_not_compared"] == {} and v["state_margin"] < 0.1
    # Each judged column is one bngsim and AMICI both move, A0's through the
    # initial assignment it sets.
    cols = {c["name"] for c in v["columns"]}
    assert cols == {"k1", "_lp_J2_kl", "A0"}
    assert np.abs(np.asarray(r.sensitivities)[:, :, 2]).max() > 0.1
    # A wrong A0 column, the #144 shape, fails.
    bad = _verdict(_scaled(r, 2, 1.01))
    assert bad["status"] == "diff" and bad["worst_column"].endswith("/dA0"), bad
