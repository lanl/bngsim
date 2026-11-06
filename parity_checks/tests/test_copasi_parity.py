"""copasi_parity: what both engines are asked to solve, and the rows COPASI gets wrong (issue #1550).

Three copasi_parity rows went PASS -> DIFF between bngsim 0.15.1 and 0.17.0. None
was bngsim, and RoadRunner agrees with bngsim on every cell of all three:

  * BIOMD0000000569 was the harness. ``bn_ode`` starts the model's initial state
    at the time it is given, and COPASI integrates from the model's own initial
    time whatever output start it is given. copasi_run gave bngsim the SED-ML
    outputStartTime (1e-5) where rr_run gives the initialTime (0), so bngsim
    solved the model shifted by 1e-5 and COPASI did not. #873's stricter differ
    stopped forgiving it. The same shift made 12 of the 19 jobs whose window
    starts after the model does score DIFF already at 0.15.1.
  * BIOMD0000000183: COPASI's SBML import drops a power of a parameter from
    four rate laws it rewrites as mass action.
  * BIOMD0000000285: COPASI does not fire a trigger that sits at exactly 0 until
    it turns true (unless it leaves 0 with a nonzero slope at t = 0).

The last two are COPASI-calibrated ``INVALID_REFERENCE`` entries in
``copasi_dispositions.py``. On the way, copasi_run takes the two #873 changes the
other runners took (the run's atol in the movement gate, the shared row comment),
and ``co_ode`` reads SBML text without taking it for a path.

The harness tests stand both engines in, so they run without COPASI. The two
tests that pin COPASI's defects need COPASI and RoadRunner and skip without them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from _core import Outcome, differ, read_manifest

_PC = Path(__file__).resolve().parent.parent
if str(_PC / "copasi_parity") not in sys.path:
    sys.path.insert(0, str(_PC / "copasi_parity"))

import _copasi_common as cc  # noqa: E402
import copasi_dispositions as cd  # noqa: E402
import copasi_run  # noqa: E402

_GRID = np.linspace(0.0, 1.0, 11)


def _molar_trajectory(n=201):
    """BioModels in molar units: a species at 3e-5 that moves by 1e-8 sets the
    scale, and one that moves its whole range sits at 9.3e-11, a hundred atol at
    the suite's 1e-12 (test_differ's case)."""
    t = np.linspace(0.0, 100.0, n)
    return t, np.column_stack(
        [3.1e-5 - 1e-8 * (1 - np.exp(-t / 10)), 9.3e-11 * (1 - np.exp(-t / 5))]
    )


class _Queue:
    def __init__(self):
        self.out: list[dict] = []

    def put(self, res):
        self.out.append(res)


def _spec(params: dict) -> dict:
    return {
        "key": "M:ode",
        "model_id": "M",
        "method": "ode",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "xml": "unused.xml",
        "params": {"t_end": 1.0, "n_points": 11, "rtol": 1e-9, "atol": 1e-12, **params},
    }


@pytest.fixture
def stand_ins(monkeypatch):
    """Both engines stood in; each records the arguments it was called with and
    returns the trajectory the test put in ``runs``."""
    calls: dict[str, tuple] = {}
    runs = {"bn": (_GRID, np.zeros((len(_GRID), 1)), ["S"]), "co": None}

    def fake_bn(xml, t_start, t_end, n_points, rtol, atol, **kw):
        calls["bn"] = (t_start, t_end, n_points, rtol, atol)
        return (*runs["bn"], {})

    def fake_co(xml, t_start, t_end, n_points, rtol, atol):
        calls["co"] = (t_start, t_end, n_points, rtol, atol)
        return (*(runs["co"] or runs["bn"]), {})

    monkeypatch.setattr(cc, "measure_warmup", lambda: {})
    monkeypatch.setattr(cc, "set_copasi_quiet", lambda: None)
    monkeypatch.setattr(cc, "bn_ode", fake_bn)
    monkeypatch.setattr(cc, "co_ode", fake_co)
    return calls, runs


# --------------------------------------------------------------------------- #
# Both engines solve the same problem
# --------------------------------------------------------------------------- #
def test_both_engines_integrate_from_the_initial_time(stand_ins):
    """BIOMD0000000569's window is [1e-5, 1e-2] with the model starting at 0. COPASI
    integrates from 0 whatever output start it is given; bn_ode starts the model's
    initial state at the time it is given. Handing bngsim 1e-5 solved the model
    shifted by one output step: COPASI's row i was bngsim's row i + 1, and #873's
    differ scored that DIFF 0.356 (621 hard cells)."""
    calls, _ = stand_ins
    q = _Queue()
    copasi_run._worker(_spec({"initial_time": 0.0, "t_start": 1e-5, "t_end": 0.01}), q)
    assert calls["bn"][0] == 0.0
    assert calls["co"][0] == 0.0
    assert q.out[0]["status"] == "pass"


def test_a_job_without_an_initial_time_starts_at_t_start(stand_ins):
    """Job files predating the field integrate from ``t_start``, as rr_run's do."""
    calls, _ = stand_ins
    copasi_run._worker(_spec({"t_start": 2.0, "t_end": 3.0}), _Queue())
    assert calls["bn"][0] == calls["co"][0] == 2.0


def test_the_runner_reads_the_initial_time_from_every_offset_window_job():
    """The 19 corpus jobs whose window starts after the model does all start at
    0; the helper gives that, not the output start, for each."""
    _meta, jobs = read_manifest(_PC / "rr_parity" / "ode_jobs.json")
    offset = [
        j for j in jobs if j.params.get("initial_time", j.params["t_start"]) != j.params["t_start"]
    ]
    assert {"BIOMD0000000042", "BIOMD0000000569", "BIOMD0000000627"} <= {
        j.model_id for j in offset
    }
    for j in offset:
        assert copasi_run.initial_time(j.params) == j.params["initial_time"] < j.params["t_start"]


def test_the_movement_gate_gets_the_runs_atol(stand_ins):
    """#873 left a cell within ten atol out of the movement gate and passed the
    run's atol from the rr, amici and bng runners; copasi_run passed none, so
    solver noise on a near-zero species could fail a COPASI row alone. Through the
    worker, as a sweep runs it: 3.7 atol apart on a 9.3e-11 species passes."""
    _, runs = stand_ins
    t, ref = _molar_trajectory()
    a = ref.copy()
    a[:, 1] *= 1.04
    runs["bn"] = (t, a, ["big", "small"])
    runs["co"] = (t, ref, ["big", "small"])
    assert not differ.deterministic_verdict(a, ref)["passed"]  # what copasi_run used to ask
    q = _Queue()
    copasi_run._worker(
        _spec({"t_start": 0.0, "t_end": 100.0, "n_points": len(t), "atol": 1e-12}), q
    )
    assert q.out[0]["status"] == "pass", q.out[0]["comment"]


def test_the_row_comment_is_the_shared_builder():
    """rr and amici write their rows with ``differ.ode_row_comment`` (#873), which
    reports the worst error over movement and NO SIGNAL; copasi_run's own string
    had neither."""
    t, ref = _molar_trajectory()
    _, _, comment, _, _ = copasi_run._compare_ode(
        (t, ref, ["big", "small"]), (t, ref, ["big", "small"]), atol=1e-12
    )
    v = differ.deterministic_verdict(ref, ref, atol=1e-12)
    assert comment == differ.ode_row_comment(v, 2)
    assert "err/movement" in comment


def test_sbml_text_is_not_taken_for_a_path(tmp_path):
    """``Path(text).exists()`` raises ``File name too long`` on SBML text longer
    than a file name may be; amici_ode and rr_ode recognize text first, and now
    co_ode does too."""
    text = "<?xml version='1.0'?>" + "<sbml>" + " " * 5000 + "</sbml>"
    assert cc.sbml_source(text) == (None, text)
    f = tmp_path / "m.xml"
    f.write_text(text)
    assert cc.sbml_source(str(f)) == (str(f), None)


# --------------------------------------------------------------------------- #
# The COPASI-calibrated dispositions
# --------------------------------------------------------------------------- #
def test_the_issues_rows_are_authored_and_the_harness_row_is_not():
    """183 and 285 are COPASI's answer; 569 was the harness and PASSes once both
    engines start at 0 (one sub-atol cell of 21021 fails, and is exempt), so an entry for it would only
    report itself STALE."""
    assert set(cd.INVALID_REFERENCE) == {"BIOMD0000000183:ode", "BIOMD0000000285:ode"}
    assert cd.disposition_for("BIOMD0000000569", "ode") is None


def test_every_authored_key_names_a_model_the_corpus_builds():
    _meta, jobs = read_manifest(_PC / "rr_parity" / "ode_jobs.json")
    ids = {j.model_id for j in jobs}
    for regime in cd.REGIMES:
        assert cd.stale_keys(ids, regime) == []
    assert cd.stale_keys(set(), "ode") == sorted(cd.ALL_KEYS)


def test_every_entry_cites_a_reason_an_issue_and_an_oracle_that_is_not_bngsim():
    """bngsim must never adjudicate its own case: every entry names RoadRunner,
    the third engine that shares no code with COPASI or bngsim."""
    for key, entry in cd.INVALID_REFERENCE.items():
        assert entry.get("issue"), key
        assert "RoadRunner" in entry["reason"], key


@pytest.mark.parametrize("key", sorted(cd.INVALID_REFERENCE))
def test_an_entry_disposes_of_the_diff_it_was_written_against(key):
    """Through AMICI's ``apply_disposition``, the one mechanism both suites use:
    the DIFF becomes a non-scoring REFERENCE_FAILED with the ``invalid_result``
    class, not a PASS, and at its recorded max_rel it is not DRIFTED."""
    model_id, _, regime = key.rpartition(":")
    disp = cd.disposition_for(model_id, regime)
    outcome, comment, refusal, state = cd.apply_disposition(
        disp, Outcome.DIFF, cd.INVALID_REFERENCE[key]["max_rel"], "row"
    )
    assert (outcome, refusal, state) == (
        Outcome.REFERENCE_FAILED,
        cd.INVALID_RESULT_REFUSAL,
        "applied",
    )
    assert "DRIFTED" not in comment
    assert comment.startswith("invalid reference (GH #1550): ")


def test_a_row_that_agrees_on_its_own_reports_the_entry_stale():
    disp = cd.disposition_for("BIOMD0000000183", "ode")
    outcome, comment, refusal, state = cd.apply_disposition(disp, Outcome.PASS, 0.0, "row")
    assert (outcome, refusal, state) == (Outcome.PASS, None, "stale")
    assert "STALE" in comment


def test_copasi_run_disposes_of_the_rows_it_runs(monkeypatch, tmp_path):
    """The runner end to end, engines stood in: an authored DIFF is reported as
    REFERENCE_FAILED / invalid_result with the reason in its comment, a PASS
    beside it is untouched, and the report counts what was applied."""
    natural = {
        "BIOMD0000000183": ("diff", 0.6197),
        "BIOMD0000000285": ("diff", 1.0),
        "BIOMD0000000569": ("pass", 0.0),
    }

    def fake_schedule(specs, worker, **kw):
        out = []
        for s in specs:
            status, value = natural[s["model_id"]]
            out.append(
                {
                    **{k: s[k] for k in ("key", "model_id", "method", "metric", "tol")},
                    "status": status,
                    "value": value,
                    "comment": "21 sp; fail 0/21021",
                    "exception": "",
                    "wall_sec": 0.1,
                }
            )
        return out

    monkeypatch.setattr(cc, "schedule", fake_schedule)
    # The SBML corpus is gitignored; the runner only checks the files exist.
    monkeypatch.setattr(cc, "model_path", lambda suite_dir, job: Path(__file__))
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["copasi_run.py", "--models", ",".join(natural), "--out", str(out), "--workers", "1"],
    )
    assert copasi_run.main() == 0
    report = json.loads(out.read_text())
    rows = {r["model_id"]: r for r in report["results"]}
    for mid in ("BIOMD0000000183", "BIOMD0000000285"):
        assert rows[mid]["outcome"] == str(Outcome.REFERENCE_FAILED), mid
        assert rows[mid]["reference_refusal"] == cd.INVALID_RESULT_REFUSAL
        assert "GH #1550" in rows[mid]["comment"]
    assert rows["BIOMD0000000569"]["outcome"] == str(Outcome.PASS)
    assert report["_meta"]["dispositions"]["counts"] == {"invalid_reference:applied": 2}
    assert cd.INVALID_RESULT_REFUSAL in report["_meta"]["reference_refusal_breakdown"]["settled"]


# --------------------------------------------------------------------------- #
# The COPASI defects the entries rest on (need COPASI and RoadRunner)
# --------------------------------------------------------------------------- #
_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2"><model id="m">
<listOfCompartments><compartment id="c" size="1" spatialDimensions="3" constant="true"/></listOfCompartments>"""
_MATH = '<math xmlns="http://www.w3.org/1998/Math/MathML">{}</math>'
_TIME = (
    "<csymbol encoding='text' definitionURL='http://www.sbml.org/sbml/symbols/time'>t</csymbol>"
)


def _species(sid, conc, boundary=False):
    return (
        f'<species id="{sid}" compartment="c" initialConcentration="{conc}" '
        f'hasOnlySubstanceUnits="false" boundaryCondition="{str(boundary).lower()}" '
        'constant="false"/>'
    )


def _mass_action_sbml(power: bool) -> str:
    """S -> P at c * S * ka * bb^(1/2) = 0.5 at t = 0 (or c * S * ka * bb = 0.25)."""
    factor = (
        "<apply><power/><ci>bb</ci><apply><divide/><cn>1</cn><cn>2</cn></apply></apply>"
        if power
        else "<ci>bb</ci>"
    )
    law = f"<apply><times/><ci>c</ci><ci>S</ci><ci>ka</ci>{factor}</apply>"
    return (
        _HEAD
        + f"<listOfSpecies>{_species('S', 1)}{_species('P', 0)}</listOfSpecies>"
        + '<listOfParameters><parameter id="ka" value="1" constant="true"/>'
        + '<parameter id="bb" value="0.25" constant="true"/></listOfParameters>'
        + '<listOfReactions><reaction id="r" reversible="false">'
        + '<listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>'
        + '<listOfProducts><speciesReference species="P" stoichiometry="1" constant="true"/></listOfProducts>'
        + f"<kineticLaw>{_MATH.format(law)}</kineticLaw></reaction></listOfReactions></model></sbml>"
    )


def _need_copasi() -> None:
    """Skip without COPASI, and import it only through ``_import_copasi``.

    ``pytest.importorskip("COPASI")`` would import it bare, which drops the
    process ``LC_CTYPE`` to ``C``; every later test in the same worker that reads
    a UTF-8 file with the default encoding then fails on it. ``_import_copasi``
    saves and restores the locale around the import, as the sweep does.
    """
    if importlib.util.find_spec("COPASI") is None:
        pytest.skip("python-copasi is not installed")
    cc._import_copasi()


def _copasi_rate(xml: str) -> tuple[str, float]:
    COPASI = cc._import_copasi()
    dm = COPASI.CRootContainer.addDatamodel()
    try:
        assert dm.importSBMLFromString(xml)
        mod = dm.getModel()
        mod.applyInitialValues()
        r = mod.getReaction(0)
        return r.getFunction().getObjectName(), r.getFlux()
    finally:
        COPASI.CRootContainer.removeDatamodel(dm)


def test_copasi_drops_a_power_of_a_parameter_from_a_rate_it_reads_as_mass_action():
    """The premise of the ``BIOMD0000000183:ode`` entry, on one reaction. COPASI
    imports ``c * S * ka * bb^(1/2)`` as 'Mass action (irreversible)' with k1 = ka
    and loses the power: rate 1.0, where the SBML and RoadRunner say 0.5. The
    control, ``c * S * ka * bb``, it reads right. If this starts failing, COPASI
    has changed here, and the entry will report itself STALE or DRIFTED."""
    _need_copasi()
    roadrunner = pytest.importorskip("roadrunner")
    law, rate = _copasi_rate(_mass_action_sbml(power=True))
    assert law == "Mass action (irreversible)"
    assert rate == pytest.approx(1.0)
    rr = roadrunner.RoadRunner(_mass_action_sbml(power=True))
    assert rr.model.getReactionRates()[0] == pytest.approx(0.5)
    assert _copasi_rate(_mass_action_sbml(power=False))[1] == pytest.approx(0.25)


def _trigger_sbml(rate_math: str, initial_value: str) -> str:
    """B' = rate_math, B(0) = 0, and an event on ``B > 0`` that sets F = 1."""
    one = _MATH.format("<cn>1</cn>")
    return (
        _HEAD
        + f"<listOfSpecies>{_species('B', 0)}{_species('F', 0, boundary=True)}</listOfSpecies>"
        + f'<listOfRules><rateRule variable="B">{_MATH.format(rate_math)}</rateRule></listOfRules>'
        + '<listOfEvents><event id="e" useValuesFromTriggerTime="true">'
        + f'<trigger initialValue="{initial_value}" persistent="true">'
        + _MATH.format("<apply><gt/><ci>B</ci><cn>0</cn></apply>")
        + f'</trigger><listOfEventAssignments><eventAssignment variable="F">{one}'
        + "</eventAssignment></listOfEventAssignments></event></listOfEvents></model></sbml>"
    )


_HELD_THEN_RISING = (
    f"<piecewise><piece><cn>0</cn><apply><lt/>{_TIME}<cn>0.5</cn></apply></piece>"
    "<otherwise><cn>1</cn></otherwise></piecewise>"
)


def _fired(run) -> bool:
    t, v, names = run[:3]
    return bool(v[-1, list(names).index("F")] == 1.0)


@pytest.mark.parametrize("initial_value", ["true", "false"])
def test_copasi_misses_a_trigger_that_sits_on_its_threshold(initial_value):
    """The premise of the ``BIOMD0000000285:ode`` entry. ``B > 0`` with B held at 0
    until t = 0.5 and rising after turns true at 0.5 under every reading of the
    SBML start rule, and B = t^2 / 2 makes it true at every t > 0, which with
    initialValue = false every reading fires too. COPASI fires neither, with
    either initialValue; RoadRunner fires both. The control: B leaving 0 with
    slope 1 COPASI does fire, so the event itself reaches COPASI's output. If
    this starts failing, COPASI has changed here, and the entry will report
    itself STALE or DRIFTED."""
    _need_copasi()
    pytest.importorskip("roadrunner")
    from rr_parity import _rr_common as rc

    def both(rate):
        xml = _trigger_sbml(rate, initial_value)
        return (
            _fired(cc.co_ode(xml, 0.0, 1.0, 101, 1e-9, 1e-12)),  # SBML text, not a path
            _fired(rc.rr_ode(xml, 0.0, 1.0, 101, 1e-9, 1e-12)),
        )

    assert both(_HELD_THEN_RISING) == (False, True)
    assert both(_TIME) == (False, True)
    assert both("<cn>1</cn>") == (True, True)
