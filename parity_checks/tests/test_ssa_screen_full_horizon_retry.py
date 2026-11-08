"""The SSA screen's partial-horizon fallback, where its window is the whole horizon.

A full-horizon bngsim ensemble with one replicate over ``REP_WALL_CAP`` falls back
to partial-horizon adjudication, whose search starts at ``t_end``. Where every
replicate finishes there, the ensemble is the full one (the same seeds against the
same reference), and a pass with cells compared is the full verdict, not a
provisional partial one: BIOMD0000001026 and 1040 read "partial t=0→100", their
whole horizon, when one replicate ran slow on one runner (issue #1582).

The model decays A from 100 copies and makes B at a constant rate, so both
engines agree and plenty of cells are above the low-count floor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import ssa_screen as SS

T_END = 20.0


def _sbml(path: Path) -> str:
    libsbml = pytest.importorskip("libsbml")

    doc = libsbml.SBMLDocument(3, 2)
    m = doc.createModel()
    m.setId("decay_and_make")
    c = m.createCompartment()
    c.setId("C")
    c.setSize(1)
    c.setConstant(True)
    for sid, ic in (("A", 100.0), ("B", 0.0)):
        sp = m.createSpecies()
        sp.setId(sid)
        sp.setCompartment("C")
        sp.setInitialAmount(ic)
        sp.setHasOnlySubstanceUnits(True)
        sp.setBoundaryCondition(False)
        sp.setConstant(False)
    for pid, v in (("kd", 0.1), ("p", 5.0)):
        par = m.createParameter()
        par.setId(pid)
        par.setValue(v)
        par.setConstant(True)
    for rid, reactant, product, law in (("J0", "A", None, "kd*A"), ("J1", None, "B", "p")):
        r = m.createReaction()
        r.setId(rid)
        r.setReversible(False)
        for sid, make in ((reactant, r.createReactant), (product, r.createProduct)):
            if sid:
                ref = make()
                ref.setSpecies(sid)
                ref.setStoichiometry(1)
                ref.setConstant(True)
        r.createKineticLaw().setMath(libsbml.parseL3Formula(law))
    path.write_text(libsbml.writeSBMLToString(doc))
    return str(path)


class _Queue:
    def __init__(self):
        self.items: list = []

    def put(self, item):
        self.items.append(item)


def _run(tmp_path, monkeypatch, slow_full_horizon_calls: int) -> tuple[dict, list[float]]:
    """The worker's last row, with the first ``slow_full_horizon_calls`` ensembles
    over the whole horizon timing out, as one slow replicate makes them."""
    pytest.importorskip("roadrunner")
    import bngsim

    real = SS.rc.bn_ssa_replicates
    horizons: list[float] = []

    def one_slow_replicate(xml, t_start, t_end, *args, **kwargs):
        horizons.append(float(t_end))
        if float(t_end) == T_END and horizons.count(T_END) <= slow_full_horizon_calls:
            raise bngsim.SimulationTimeout("a replicate ran past REP_WALL_CAP")
        return real(xml, t_start, t_end, *args, **kwargs)

    monkeypatch.setattr(SS.rc, "bn_ssa_replicates", one_slow_replicate)
    monkeypatch.setattr(SS.rc, "measure_ssa_warmup", lambda **k: {})
    spec = {
        "name": "decay_and_make",
        "source": "synthetic",
        "sbml_path": _sbml(tmp_path / "m.xml"),
        "exists": True,
        "t_end": T_END,
        "n_steps": 10,
        "effort": "low",
    }
    q = _Queue()
    SS._worker(
        spec,
        30,
        2000,
        SS.DEFAULT_MEAN_Z_TOL,
        5,
        SS.DEFAULT_MIN_MEAN_COUNT,
        SS.DEFAULT_SELF_Z_TOL,
        q,
    )
    return q.items[-1], horizons


def test_a_fallback_that_finishes_the_whole_horizon_is_the_full_pass(tmp_path, monkeypatch):
    row, horizons = _run(tmp_path, monkeypatch, slow_full_horizon_calls=1)
    assert horizons.count(T_END) == 2  # the full attempt, then the search at t_end
    assert row["status"] == "pass", row.get("error") or row.get("note")
    assert row["full_horizon_on_retry"] is True
    assert "partial_horizon" not in row
    assert int(row["n_compared"]) > 0


def test_a_fallback_to_a_shorter_window_stays_partial(tmp_path, monkeypatch):
    # Control: slow at t_end twice, the search goes on to t_end/10, and the
    # verdict there is the provisional one it always was.
    row, horizons = _run(tmp_path, monkeypatch, slow_full_horizon_calls=2)
    assert row["status"] == "partial", row.get("error") or row.get("note")
    assert row["partial_horizon"] == pytest.approx(T_END / 10)
    assert "full_horizon_on_retry" not in row
