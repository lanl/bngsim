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
    run = {k: kw.pop(k) for k in ("rtol",) if k in kw}
    sim = Simulator(model, method="ode", sensitivity_params=kw.pop("params", ["k"]), **kw)
    return sim.run(t_span=(0, 10), n_points=11, **run)


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
        self.calls: list[dict] = []
        #: Added to the sensitivities of the run at the run's own tolerance:
        #: AMICI's own error there, which its tighter run does not have.
        self.loose_error = 0.0
        self.loose_state_error = 0.0
        self.nan_in_reference = False
        #: Applied to the run at the run's own tolerance only.
        self.loose_edit = None

    @property
    def asked(self) -> dict:
        """The first run's settings: AMICI at the run's tolerance."""
        return self.calls[0]

    def simulate(self, ids, t0, times, *, rtol, atol, atol_sens, method):
        self.calls.append(
            dict(ids=list(ids), t0=t0, rtol=rtol, atol=atol, atol_sens=atol_sens, method=method)
        )
        if self.fail and (self.fail != "tight" or len(self.calls) > 1):
            failure = "status -1 (AMICI_TOO_MUCH_WORK)" if self.fail == "tight" else self.fail
            return failure, np.empty((0, 0)), np.empty((0, 0, 0))
        t = np.asarray(times, dtype=float)[:, None]
        x0 = np.asarray([self.x0[s.removeprefix("amici_")] for s in self.state_ids])[None, :]
        x = x0 * np.exp(-self.k * t)
        sx = np.zeros((t.size, x.shape[1], len(ids)))
        for c, p in enumerate(ids):
            if p == "amici_k":
                sx[:, :, c] = -t * x
        if self.edit is not None:
            x, sx = self.edit(x, sx)
        if len(self.calls) == 1:
            sx = sx * (1 + self.loose_error)
            x = x * (1 + self.loose_state_error)
            if self.loose_edit is not None:
                x, sx = self.loose_edit(x, sx)
        elif self.nan_in_reference:
            sx = sx.copy()
            sx[2, 0, 0] = np.nan
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
    assert fake.asked["method"] == "staggered" == r.record["run"]["sensitivity_method"]
    fake = engine(_ClosedForm({"A": A0}))
    _verdict(_decay(sensitivity_method="simultaneous"))
    assert fake.asked["method"] == "simultaneous"
    # And again at a tenth of the run's tolerance, the reference.
    (_, tight) = fake.calls
    assert tight["rtol"] == pytest.approx(fake.asked["rtol"] / 10)
    assert tight["atol"] == pytest.approx(fake.asked["atol"] / 10)
    assert tight["atol_sens"] == pytest.approx(fake.asked["atol_sens"] / 10)
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


def test_amicis_own_move_at_the_runs_tolerance_widens_the_allowance(monkeypatch, engine):
    # Every bngsim solve off by 1e-4 of dA/dk: past the allowance where AMICI
    # does not move with the tolerance; within it where AMICI itself moves by
    # as much between the run's tolerance and a tenth of it, as a correct
    # engine there may.
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        return _scaled(out, 0, 1 + 1e-4) if len(out._sensitivity_params) else out

    monkeypatch.setattr(Simulator, "run", run)
    assert _verdict(_decay())["status"] == "diff"
    fake = engine(_ClosedForm({"A": A0}))
    fake.loose_error = 1e-4
    v = _verdict(_decay())
    assert v["status"] == "pass", v
    # It widens: it does not forgive a reading off by far more.

    def run_far(self, *a, **kw):
        out = real(self, *a, **kw)
        return _scaled(out, 0, 1.01) if len(out._sensitivity_params) else out

    fake = engine(_ClosedForm({"A": A0}))
    fake.loose_error = 1e-4
    monkeypatch.setattr(Simulator, "run", run_far)
    assert _verdict(_decay())["status"] == "diff"


def test_a_disagreement_the_tighter_run_does_not_show_is_the_tolerances(engine):
    # Off in the run alone, by 1%: the tenfold tighter run agrees with AMICI.
    rep = _scaled(_decay(), 0, 1.01).check([NAME])
    (v,) = rep["checks"]
    assert v["status"] == "diff" and v["diff_not_converged"] is True, v
    assert "the tolerance's, not the reading's" in v["note"]
    assert "not converged at the run's tolerance" in rep["next"]


def test_the_reference_is_amicis_tighter_run(engine):
    # AMICI at the run's tolerance off by 5% of dA/dk, and of A: its run at a
    # tenth of the tolerance is the one judged against, and a correct run
    # passes. The series AMICI moves past a thousandth of their peak on are
    # named, as their allowance is that wide.
    fake = engine(_ClosedForm({"A": A0}))
    fake.loose_error = 0.05
    fake.loose_state_error = 0.05
    v = _verdict(_decay())
    assert v["status"] == "pass", v
    assert v["amici_not_converged"] == ["dA/dk"] and v["n_amici_not_converged"] == 1
    assert "AMICI itself moves past 0.001 of their peak" in v["note"]
    assert "(dA/dk)" in v["note"]
    # Ten times AMICI's move: a reading off by twice the truth is still one.
    fake = engine(_ClosedForm({"A": A0}))
    fake.loose_error = 0.05
    assert _verdict(_scaled(_decay(), 0, 2.0))["status"] == "diff"


def test_the_states_reference_is_amicis_tighter_run_too(monkeypatch, engine):
    # bngsim's A off by 1.05% in every solve; AMICI's A off by 0.1% at the
    # run's tolerance, the same way, and exact at a tenth of it. Against the
    # tighter run the gap is 1.05 times ten times AMICI's move: a diff.
    real = Simulator.run

    def run(self, *a, **kw):
        out = copy.copy(real(self, *a, **kw))
        out._species = np.asarray(out._species, dtype=float) * (1 + 1.05e-2)
        return out

    monkeypatch.setattr(Simulator, "run", run)
    fake = engine(_ClosedForm({"A": A0}))
    fake.loose_state_error = 1e-3
    v = _verdict(_decay())
    assert v["status"] == "diff" and v["state_diff"] is True, v
    assert 1.0 < v["state_margin"] < 1.1, v


def test_a_species_the_run_does_not_resolve_is_judged_on_its_whole_scale(tmp_path, engine):
    # B is held at 1e-11, within 100 times its atol of zero, which the run
    # does not resolve: AMICI's B 1e-8 off is past B's own allowance, and no
    # finding on the run's whole scale.
    path = _write_sbml(
        tmp_path / "trace.xml",
        species=[("A", A0, False), ("B", 1e-11, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {})],
    )
    r = _decay(path)
    ctx = _check._Context(r, r.record, None)
    atol_b = float(ctx.atols[1])
    assert not _check.resolved(np.array([1e-11]), np.array([atol_b]), r.record["run"]["rtol"])[0]
    assert _check.PATH_ATOL_FACTOR * atol_b < 1e-8

    def off(x, sx):
        x = x.copy()
        x[:, 1] = 1e-11 + 1e-8
        return x, sx

    engine(_ClosedForm({"A": A0, "B": 1e-11}, edit=off))
    v = _verdict(r)
    assert v["status"] == "pass" and v["state_diff"] is False, v


def test_a_value_amicis_reference_run_gives_as_not_finite_is_a_skip(engine):
    fake = engine(_ClosedForm({"A": A0}))
    fake.nan_in_reference = True
    v = _verdict(_decay())
    assert v["status"] == "skip" and "not finite" in v["note"], v


def test_a_species_not_converged_is_judged_on_the_runs_whole_scale(tmp_path, monkeypatch, engine):
    # B is a thousandth of A. Its tighter run moves by 1% of it, so the run
    # does not converge B; AMICI's B, 0.5% off, is judged on the run's whole
    # scale, A's, as every path judges such a species, and is no finding.
    path = _write_sbml(
        tmp_path / "two.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {}), ("J2", ["B"], [], "k*B", {})],
    )
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("rtol", 1.0) < self.record["run"]["rtol"]:
            out = copy.copy(out)
            x = np.asarray(out._species, dtype=float).copy()
            x[:, 1] *= 1.01
            out._species = x
        return out

    monkeypatch.setattr(Result, "rerun", rerun)

    def off(x, sx):
        x = x.copy()
        x[:, 1] *= 1.005
        return x, sx

    engine(_ClosedForm({"A": A0, "B": 0.1}, edit=off))
    rep = _decay(path).check([NAME])
    (v,) = rep["checks"]
    assert v["status"] == "pass" and v["state_diff"] is False, v


def test_a_species_wrong_in_every_solve_is_the_readings_though_not_converged(
    tmp_path, monkeypatch, engine
):
    # B is a thousandth of A, and the run does not converge it (its tighter
    # run moves by 1%). B 20% off in every solve, the tighter run's too, is
    # the reading's.
    path = _write_sbml(
        tmp_path / "two.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {}), ("J2", ["B"], [], "k*B", {})],
    )
    real_rerun = Result.rerun

    def rerun(self, **changes):
        out = real_rerun(self, **changes)
        if changes.get("rtol", 1.0) < self.record["run"]["rtol"]:
            out = copy.copy(out)
            x = np.asarray(out._species, dtype=float).copy()
            x[:, 1] *= 1.01
            out._species = x
        return out

    def b_off(r, factor):
        out = copy.copy(r)
        x = np.asarray(out._species, dtype=float).copy()
        x[:, 1] *= factor
        out._species = x
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    engine(_ClosedForm({"A": A0, "B": 0.1}))
    real_run = Simulator.run
    monkeypatch.setattr(
        Simulator, "run", lambda self, *a, **kw: b_off(real_run(self, *a, **kw), 1.2)
    )
    v = _verdict(_decay(path))
    assert v["status"] == "diff" and v["state_diff"] is True, v
    assert v["diff_not_converged"] is False


def test_a_series_wrong_in_every_solve_is_the_readings_though_not_converged(monkeypatch, engine):
    # dA/dk twice the truth in every solve, at a tolerance where the series
    # is not converged either: bngsim's tighter run is as far from AMICI, so
    # the diff is not the tolerance's.
    _loader_moves_k_twice(monkeypatch)
    rep = _decay(rtol=1e-3).check([NAME])
    (v,) = rep["checks"]
    assert v["status"] == "diff" and v["diff_not_converged"] is False, v
    assert "Report it" in rep["next"]


def test_amicis_own_move_widens_the_states_allowance_too(monkeypatch, engine):
    real = Simulator.run

    def run(self, *a, **kw):
        out = copy.copy(real(self, *a, **kw))
        out._species = np.asarray(out._species, dtype=float) * (1 + 1e-4)
        return out

    monkeypatch.setattr(Simulator, "run", run)
    v = _verdict(_decay())
    assert v["status"] == "diff" and v["state_diff"] is True, v
    fake = engine(_ClosedForm({"A": A0}))
    fake.loose_state_error = 1e-4
    v = _verdict(_decay())
    assert v["status"] == "pass", v


def test_amici_failing_at_the_tighter_tolerance_is_a_skip(engine):
    engine(_ClosedForm({"A": A0}, fail="tight"))
    v = _verdict(_decay())
    assert v["status"] == "skip" and "at a tenth of the run's tolerance" in v["note"], v


def test_amici_starts_where_the_run_starts(engine):
    model = Model.load(str(examples.path("decay", format="sbml")))
    r = Simulator(model, method="ode", sensitivity_params=["k"]).run(t_span=(2, 7), n_points=6)
    fake = engine(_ClosedForm({"A": A0}))
    _verdict(r)
    assert fake.asked["t0"] == 2.0 and all(c["t0"] == 2.0 for c in fake.calls)


def test_an_antimony_model_is_read_as_its_sbml(engine):
    pytest.importorskip("antimony")
    model = Model.load(str(examples.path("decay", format="antimony")))
    r = Simulator(model, method="ode", sensitivity_params=["k"]).run(t_span=(0, 10), n_points=11)
    v = _verdict(r)
    assert v["status"] == "pass", v


def test_a_local_parameter_beside_a_global_of_its_amici_name_is_not_judged(tmp_path, engine):
    # J2's local kl is AMICI's J2_kl, unless the file has a global J2_kl: then
    # the global keeps the name, and the local is J2_kl_1.
    path = _write_sbml(
        tmp_path / "collide.xml",
        species=[("A", A0, False), ("B", 1.0, False)],
        params={"k": K0, "J2_kl": 0.7},
        reactions=[
            ("J1", ["A"], [], "k*A", {}),
            ("J2", ["B"], [], "kl*B", {"kl": 0.2}),
            ("J3", ["B"], [], "J2_kl*B", {}),
        ],
    )
    model = Model.load(str(path))
    r = Simulator(model, method="ode", sensitivity_params=["_lp_J2_kl"]).run(
        t_span=(0, 5), n_points=6
    )
    engine(_ClosedForm({"A": A0, "B": 1.0}, free={"k": K0, "J2_kl": 0.7, "J2_kl_1": 0.2}))
    v = _verdict(r)
    assert v["status"] == "skip" and "no column AMICI can give" in v["note"], v
    assert "global parameter J2_kl as well" in v["note"]
    r = Simulator(Model.load(str(path)), method="ode", sensitivity_params=["J2_kl", "k"]).run(
        t_span=(0, 5), n_points=6
    )
    engine(_ClosedForm({"A": A0, "B": 1.0}, free={"amici_k": K0, "J2_kl": 0.7, "J2_kl_1": 0.2}))
    v = _verdict(r)
    assert v["not_judged"] == {
        "J2_kl": "a reaction's local parameter has its name in AMICI's naming too"
    }, v


def test_a_column_the_run_gives_as_zero_is_judged_on_amicis_scale(engine):
    v = _verdict(_scaled(_decay(), 0, 0.0))
    assert v["status"] == "diff" and v["margin"] > 100, v
    # Off by the whole of AMICI's peak, not by nothing on bngsim's.
    assert v["value"] == pytest.approx(1.0, rel=1e-6)


def test_the_looser_runs_move_widens_a_series_drift(monkeypatch, engine):
    # AMICI off by 5e-4 of the series: past what the tighter run's move
    # allows, and within ten times half the move at twice the tolerance.
    def off(x, sx):
        return x, sx * (1 + 5e-4)

    engine(_ClosedForm({"A": A0}, edit=off))
    r = _decay()
    assert _verdict(r)["status"] == "diff"
    real = Result.rerun

    def rerun(self, **changes):
        out = real(self, **changes)
        if changes.get("rtol", 0) > self.record["run"]["rtol"]:
            out = _scaled(out, 0, 1 + 2e-4)
        return out

    monkeypatch.setattr(Result, "rerun", rerun)
    v = _verdict(r)
    assert v["status"] == "pass" and v["margin"] > 0.1, v
    # A looser run that is far off itself widens a series by no more than a
    # thousandth of its peak: AMICI off by 5% is still a finding.

    def far(x, sx):
        return x, sx * 1.05

    engine(_ClosedForm({"A": A0}, edit=far))

    def rerun_far(self, **changes):
        out = real(self, **changes)
        if changes.get("rtol", 0) > self.record["run"]["rtol"]:
            out = _scaled(out, 0, 1.1)
        return out

    monkeypatch.setattr(Result, "rerun", rerun_far)
    assert _verdict(r)["status"] == "diff"


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


def test_noise_near_zero_is_judged_on_its_columns_scale(tmp_path, monkeypatch, engine):
    # B is held at 0.1 and k moves nothing of it: the true dB/dk is 0.
    path = _write_sbml(
        tmp_path / "held.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {})],
    )
    r = _decay(path)
    assert np.abs(np.asarray(r.sensitivities)[:, 1, 0]).max() == 0.0
    cols = _check_sensitivity._Columns(_check._Context(r, r.record, None))
    floor = float(cols.atol[1, 0])

    def amici_says(value):
        def edit(x, sx):
            x, sx = x.copy(), sx.copy()
            x[:, 1], sx[:, 1, 0] = 0.1, value
            return x, sx

        return edit

    # Each engine's noise within 100 of the series' tolerances of zero, on
    # opposite sides: 160 tolerances apart, past what the series' own scale
    # allows, and no finding on its column's.
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        if len(out._sensitivity_params) and out._sensitivities.shape[1] == 2:
            out = copy.copy(out)
            s = np.asarray(out._sensitivities, dtype=float).copy()
            s[:, 1, 0] = -80.0 * floor
            out._sensitivities = s
        return out

    monkeypatch.setattr(Simulator, "run", run)
    noisy = _decay(path)
    engine(_ClosedForm({"A": A0, "B": 0.1}, edit=amici_says(80.0 * floor)))
    v = _verdict(noisy)
    assert v["status"] == "pass", v
    assert v["not_resolved"] == ["dB/dk"] and v["n_not_resolved"] == 1
    # Nor is AMICI's own move there named as wide: a series neither engine
    # resolves has no peak to read it on.
    fake = engine(_ClosedForm({"A": A0, "B": 0.1}, edit=amici_says(0.0)))
    fake.loose_edit = amici_says(50.0 * floor)
    v = _verdict(noisy)
    assert v["status"] == "pass" and v["amici_not_converged"] == [], v
    monkeypatch.setattr(Simulator, "run", real)
    # A value AMICI resolves where the run gives 0 is a finding, however small
    # beside the column's largest series: a dependency the loader dropped.
    engine(_ClosedForm({"A": A0, "B": 0.1}, edit=amici_says(1e-4)))
    v = _verdict(r)
    assert v["status"] == "diff" and v["worst_column"] == "dB/dk", v
    assert v["not_resolved"] == []


def test_a_series_dropped_in_every_solve_is_judged_on_amicis_scale(tmp_path, monkeypatch, engine):
    # dB/dk is a thousandth of dA/dk, and every bngsim solve gives it as 0.
    path = _write_sbml(
        tmp_path / "two.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {}), ("J2", ["B"], [], "k*B", {})],
    )
    engine(_ClosedForm({"A": A0, "B": 0.1}))
    real = Simulator.run

    def run(self, *a, **kw):
        out = real(self, *a, **kw)
        return _scaled(out, 0, 0.0, row=1) if len(out._sensitivity_params) else out

    monkeypatch.setattr(Simulator, "run", run)
    for rtol in (1e-8, 1e-6, 1e-4):
        v = _verdict(_decay(path, rtol=rtol))
        assert v["status"] == "diff" and v["worst_column"] == "dB/dk", (rtol, v)
    # Dropped in the run alone, the tighter run's move is the whole series:
    # read on AMICI's peak, it is no drift to forgive the drop with.
    monkeypatch.setattr(Simulator, "run", real)
    for rtol in (1e-8, 1e-6, 1e-4):
        v = _verdict(_scaled(_decay(path, rtol=rtol), 0, 0.0, row=1))
        assert v["status"] == "diff" and v["worst_column"] == "dB/dk", (rtol, v)
        assert v["diff_not_converged"] is True


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
    assert "1 series not finite in the run where AMICI's are" in v["note"]
    assert "1.8e+308" not in v["note"]

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


def test_a_budget_spent_on_bngsims_reruns_builds_nothing(monkeypatch, engine):
    real = _check._Context._ensure_spread

    def spread_then_spent(self):
        out = real(self)
        self.deadline = 0.0  # the reruns took the whole budget
        return out

    monkeypatch.setattr(_check._Context, "_ensure_spread", spread_then_spent)
    built = []
    monkeypatch.setattr(_check_amici, "build_engine", lambda amici, sbml: built.append(sbml))
    v = _verdict(_decay())
    assert v["status"] == "timeout" and built == [], v


def test_a_run_with_no_converged_species_judges_nothing_and_builds_nothing(monkeypatch, engine):
    r = _decay(rtol=0.1)
    assert r.check(["tolerance"])["converged"] is False
    built = []
    monkeypatch.setattr(_check_amici, "build_engine", lambda amici, sbml: built.append(sbml))
    v = _verdict(r)
    assert v["status"] == "no_base" and built == [], v


def test_a_load_with_compartment_sizes_is_skipped(tmp_path, engine):
    path = _write_sbml(
        tmp_path / "sized.xml",
        species=[("A", A0, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {})],
    )
    m = Model.load(str(path), compartment_sizes={"C": 1.0})
    r = Simulator(m, method="ode", sensitivity_params=["k"]).run(t_span=(0, 10), n_points=11)
    v = _verdict(r)
    assert v["status"] == "skip" and "compartment_sizes=" in v["note"], v


def test_amicis_failures_are_skips_not_findings(engine):
    engine(_ClosedForm({"A": A0}, fail="status -1 (AMICI_TOO_MUCH_WORK)"))
    v = _verdict(_decay())
    assert v["status"] == "skip" and "AMICI could not run it" in v["note"]
    assert "TOO_MUCH_WORK" in v["note"]

    def short(x, sx):
        return x[:-1], sx[:-1]

    engine(_ClosedForm({"A": A0}, edit=short))
    v = _verdict(_decay())
    assert v["status"] == "skip" and "AMICI returned" in v["note"], v


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


@pytest.mark.parametrize("size", [2.0, 1e-6])
def test_an_amount_amici_holds_is_compared_as_a_concentration(tmp_path, engine, size):
    # A held as an amount (hasOnlySubstanceUnits) in a compartment of *size*:
    # AMICI integrates the amount, A0·size·e^{-kt}; bngsim reports A0·e^{-kt}.
    path = _write_sbml(
        tmp_path / "amount.xml",
        species=[("A", A0 * size, True)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {})],
        size=size,
    )
    r = _decay(path)
    assert np.allclose(np.asarray(r.species)[:, 0], A0 * np.exp(-K0 * np.asarray(r.time)))
    fake = engine(_ClosedForm({"A": A0 * size}))
    v = _verdict(r)
    assert v["status"] == "pass" and v["species_not_compared"] == {}, v
    # AMICI is held on its own scale as tight as bngsim holds the
    # concentration: an amount's tolerance is the concentration's times the
    # size, or a small compartment would hold AMICI that many times looser;
    # and in a large one, no looser than the run's own.
    ctx = _check._Context(r, r.record, None)
    cols = _check_sensitivity._Columns(ctx)
    assert fake.asked["atol"] == pytest.approx(float(ctx.atols[0]) * min(size, 1.0))
    assert fake.asked["atol_sens"] == pytest.approx(float(cols.atol[0, 0]) * min(size, 1.0))


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
    # And one a rule sets.
    path = _write_sbml(
        tmp_path / "ruled.xml",
        species=[("A", A0, True)],
        params={"k": K0, "s": 1.0},
        reactions=[("J1", ["A"], [], "k*A", {})],
        size=1.0,
        rules={"C": "2*s"},
    )
    v = _verdict(_decay(path))
    assert v["status"] == "skip" and "whose size the file does not give" in v["note"], v


def test_a_species_amici_holds_no_state_for_is_named(tmp_path, engine):
    path = _write_sbml(
        tmp_path / "two.xml",
        species=[("A", A0, False), ("B", 0.1, False)],
        params={"k": K0},
        reactions=[("J1", ["A"], [], "k*A", {}), ("J2", ["B"], [], "k*B", {})],
    )
    fake = engine(_ClosedForm({"A": A0}))
    r = _decay(path)
    v = _verdict(r)
    assert v["status"] == "pass"
    assert v["species_not_compared"] == {"B": "no state of AMICI's has its name"}
    assert "1 species not compared" in v["note"]
    # AMICI's tolerances are scalars over every state, compared or not: no
    # looser than the run's smallest, B's here.
    ctx = _check._Context(r, r.record, None)
    cols = _check_sensitivity._Columns(ctx)
    assert cols.atol[1, 0] < cols.atol[0, 0]
    assert fake.asked["atol_sens"] == pytest.approx(float(cols.atol[:, 0].min()))
    assert fake.asked["atol"] == pytest.approx(float(np.min(ctx.atols)))


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
        "atol": v["engine"]["atol"],
        "atol_sensitivities": v["engine"]["atol_sensitivities"],
    }
    assert (
        f"not the commit bngsim's AMICI parity suite pins ({_check_amici.PINNED_COMMIT[:9]})"
        in (v["note"])
    )


def test_second_engine_skips_a_file_that_changed_after_the_load_too(tmp_path):
    pytest.importorskip("roadrunner")
    sbml = examples.path("decay", format="sbml")
    path = tmp_path / "decay.xml"
    path.write_text(sbml.read_text(), encoding="utf-8")
    r = Simulator(Model.load(str(path)), method="ode").run(t_span=(0, 5), n_points=6)
    assert r.check(["second_engine"])["checks"][0]["status"] == "pass"
    path.write_text(sbml.read_text().replace("<ci> k </ci>", "<ci> k </ci> "), encoding="utf-8")
    (v,) = r.check(["second_engine"])["checks"]
    assert v["status"] == "skip" and v["note"] == _check.FILE_CHANGED, v


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
        self.log = ""
        self.builds: list[Path] = []
        self.paths: list[str] = []
        self.__version__ = "0.0-fake"
        outer = self

        class SbmlImporter:
            def __init__(self, sbml, from_file):
                assert from_file is False

            def sbml2amici(self, name, out, **kw):
                import logging

                assert kw["generate_sensitivity_code"] is True
                assert kw["compute_conservation_laws"] is False
                outer.paths.append(os.environ["PATH"])
                if outer.log:
                    logging.getLogger("amici.importers.sbml").error(outer.log)
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


def test_a_build_that_fails_is_a_skip_and_leaves_nothing(monkeypatch, tmp_path, capfd):
    import logging

    monkeypatch.setattr(_check_amici, "CACHE_DIR", tmp_path / "amici")
    why = _check_amici.build_engine(_FakeAmici(RuntimeError("swig not found")), "<sbml/>")
    assert why == "AMICI could not build it: RuntimeError: swig not found"
    assert list((tmp_path / "amici").iterdir()) == []
    # AMICI's own log of a failure is kept off the terminal, and its first
    # error goes in the note instead of "see the messages above".
    log = logging.getLogger("amici")
    handler = logging.StreamHandler()
    monkeypatch.setattr(log, "handlers", [handler])
    fake = _FakeAmici(RuntimeError("SBML Document failed to load (see error messages above)"))
    general = "libSBML SBML component consistency (Error): " + "the rule's general text " * 20
    fake.log = general + "\n\nThe species 'S' is not a modifier of 'J1'\n"
    why = _check_amici.build_engine(fake, "<sbml/>")
    assert why.endswith("... The species 'S' is not a modifier of 'J1')"), why
    assert "(AMICI's first error: libSBML SBML component consistency (Error): the rule" in why
    assert len(why) < 500
    assert log.handlers == [handler]
    out, err = capfd.readouterr()
    assert "not a modifier" not in out + err


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific: os.kill(pid, 0) probes a pid")
def test_a_killed_builds_staging_directory_is_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(_check_amici, "CACHE_DIR", tmp_path / "amici")
    old = os.path.getmtime(tmp_path) - 2 * _check_amici._STALE_STAGING_SECONDS
    dead = tmp_path / "amici" / ".building_bngsim_amici_0123_999999999"
    alive = tmp_path / "amici" / f".building_bngsim_amici_4567_{os.getppid()}"
    young = tmp_path / "amici" / ".building_bngsim_amici_89ab_999999998"
    for d in (dead, alive, young):
        d.mkdir(parents=True)
    for d in (dead, alive):
        os.utime(d, (old, old))
    assert isinstance(_check_amici.build_engine(_FakeAmici(), "<sbml/>"), _check_amici.Engine)
    # Gone and old: removed. Alive, or touched lately (a build in another
    # container, whose pid this host does not know): left alone.
    assert not dead.exists() and alive.exists() and young.exists()


def test_the_build_cache_is_the_suites_own():
    cache = Path(_check_amici.CACHE_DIR)
    assert not cache.is_relative_to(Path.home() / ".cache" / "bngsim")
    assert os.environ["BNGSIM_AMICI_CACHE_DIR"] == str(cache)


# ─── Against AMICI itself ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _amici():
    # Any ImportError, not only a missing package: an install that does not
    # import is no install. (importorskip's exc_type= needs pytest 8.2.)
    try:
        import amici.sim.sundials as ss
    except ImportError as e:
        pytest.skip(f"could not import 'amici.sim.sundials': {e}")
    return ss


def test_amici_agrees_with_the_closed_form_and_a_shared_defect_fails(_amici, monkeypatch, capfd):
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
    # The solver is set as the run was: forward sensitivities, the run's
    # corrector method and rtol, at least as tight as bngsim on every series.
    engine = _check_amici.build_engine(amici, examples.path("decay", format="sbml").read_text())
    assert engine.cached is True and engine.free_ids == ["amici_k"]
    s = engine.solver(rtol=1e-7, atol=1e-11, atol_sens=1e-13, method="simultaneous")
    assert (s.get_relative_tolerance(), s.get_absolute_tolerance()) == (1e-7, 1e-11)
    assert (s.get_relative_tolerance_fsa(), s.get_absolute_tolerance_fsa()) == (1e-7, 1e-13)
    assert s.get_max_steps() == _check_amici.MAX_STEPS
    assert s.get_internal_sensitivity_method() == _amici.InternalSensitivityMethod_simultaneous
    assert s.get_sensitivity_method() == _amici.SensitivityMethod_forward
    s = engine.solver(rtol=1e-7, atol=1e-11, atol_sens=1e-13, method="staggered")
    assert s.get_internal_sensitivity_method() == _amici.InternalSensitivityMethod_staggered
    # The parameter scale is set to none whatever the model held: dx/dp, not
    # dx/d ln p.
    engine.model.set_parameter_scale([_amici.ParameterScaling_ln])
    failure, x_am, sx_am = engine.simulate(
        ["amici_k"],
        0.0,
        np.linspace(0, 10, 11),
        rtol=1e-8,
        atol=1e-10,
        atol_sens=1e-8,
        method="staggered",
    )
    assert failure == "" and list(engine.model.get_parameter_scale()) == [
        _amici.ParameterScaling_none
    ]
    t = np.linspace(0, 10, 11)
    assert np.allclose(sx_am[:, 0, 0], -t * A0 * np.exp(-K0 * t), rtol=1e-5, atol=1e-6)
    # The build is kept whole, and the next check reads it.
    assert _verdict(r)["engine"]["cached"] is True
    entries = [p.name for p in Path(_check_amici.CACHE_DIR).iterdir()]
    assert entries and all(e.startswith("bngsim_amici_") for e in entries), entries
    # A column the run alone has wrong.
    assert _verdict(_scaled(r, 0, 1.01))["status"] == "diff"
    # AMICI's own failure is a skip, and its log stays off the terminal: what
    # AMICI's console handler would have printed is in the note instead.
    import logging

    printed: list[logging.LogRecord] = []
    console = logging.Handler()
    console.emit = printed.append  # type: ignore[method-assign]
    monkeypatch.setattr(logging.getLogger("amici"), "handlers", [console])
    monkeypatch.setattr(_check_amici, "MAX_STEPS", 3)
    capfd.readouterr()
    v = _verdict(r)
    assert v["status"] == "skip" and "AMICI could not run it" in v["note"], v
    assert "TOO_MUCH_WORK" in v["note"] and "AMICI's first error:" in v["note"]
    assert not [rec for rec in printed if rec.levelno >= logging.ERROR], printed
    out, err = capfd.readouterr()
    assert "FORWARD_FAILURE" not in out + err
    monkeypatch.undo()
    # A defect the loader and the sensitivity code share: only AMICI sees it.
    _loader_moves_k_twice(monkeypatch)
    by = _by_name(_decay().check(["sensitivity", "sensitivity_fd", NAME]))
    assert by["sensitivity"]["status"] == "pass" and by["sensitivity_fd"]["status"] == "pass"
    assert by[NAME]["status"] == "diff" and by[NAME]["worst_column"] == "dA/dk"
    assert by[NAME]["value"] == pytest.approx(0.5, rel=1e-3)


def test_amici_lines_up_locals_amounts_renamed_ids_and_initial_assignments(_amici, tmp_path):
    # J1 moves A (a concentration) to B (an amount in a compartment of size
    # 1e-6, which AMICI integrates on that scale); J2 moves B to x (AMICI's
    # amici_x) at a local kl; J3 drains x at kb, which an initial assignment
    # sets to 2·k1; A starts at A0 by another.
    path = _write_sbml(
        tmp_path / "lineup.xml",
        species=[("A", 10.0, False), ("B", 4e-6, True), ("x", 1.0, False)],
        params={"k1": 0.3, "kb": 0.0, "A0": 10.0},
        reactions=[
            ("J1", ["A"], ["B"], "k1*A*C", {}),
            ("J2", ["B"], ["x"], "kl*B", {"kl": 0.2}),
            ("J3", ["x"], [], "kb*x*C", {}),
        ],
        size=1e-6,
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
