"""The evidence behind six rr_parity dispositions that used to cite notes never in
this repository.

Six KNOWN_ARTIFACT entries, from a 2026-05-29 triage, scored their DIFF rows as
PASS on the strength of a ledger and notes under a directory that never existed
here. A PASS counts as validated, so each claim is re-derived from evidence that
runs from this repository, with an oracle that is neither bngsim nor RoadRunner's
default run:

* BIOMD0000000879 and MODEL7908934508 stay KNOWN_ARTIFACT. Each has narrow
  piecewise-in-time pulses that RoadRunner's default run steps over. RoadRunner
  run so that it cannot (a bounded step, and for 879 a restart at every pulse
  edge), at two step bounds that agree, is the reference; bngsim passes against
  it and RoadRunner's own run does not.
* BIOMD0000000373, 376 and 682, long-horizon oscillators, and MODEL2310250001,
  whose isotopologue split has no unique solution once a pool's total crosses 0,
  move to NO_GROUND_TRUTH (non-scoring). For each, neither engine agrees with
  itself as the tolerance tightens to rtol 1e-13, so no answer over the job's
  horizon can be checked, and a PASS there checked nothing.

``scipy_oracle.py`` reads none of these models (rate rules, an event, piecewise),
so it is not used. The model files are the gitignored rr_parity corpus, so those
tests skip where it is absent; the rest need neither the corpus nor an engine.
"""

from __future__ import annotations

import importlib.util
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from _core import differ, read_manifest

_RR = Path(__file__).resolve().parent.parent / "rr_parity"
_THIS = "test_rr_known_artifact_evidence.py"


def _load_rr_overrides():
    """rr_parity's overrides.py, by path: bng_parity has an ``overrides`` module too,
    and a bare import resolves to whichever directory reached ``sys.path`` first."""
    spec = importlib.util.spec_from_file_location("rr_parity_overrides_ka", _RR / "overrides.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ov = _load_rr_overrides()

# The two entries that stay KNOWN_ARTIFACT, and how the reference is run. ``edges``
# are the pulse edges RoadRunner is restarted at (None: a step bound alone suffices).
# ``probe`` is (species, t, the reference's value, RoadRunner's default value),
# each checked to 1%. ``max_em`` bounds bngsim's err/movement against either
# reference, and ``refs_em`` the two references' against each other.
_DOSES = (0.0, 21.0, 42.0, 63.0, 84.0, 105.0, 126.0)
PULSES = {
    "BIOMD0000000879:ode": {
        "edges": tuple(e for d in _DOSES for e in (d, d + 0.125)),
        "probe": ("N", 42.0, 5.11805e9, 1.1795e10),
        "max_em": 1e-5,
        "refs_em": 1e-5,
    },
    "MODEL7908934508:ode": {
        "edges": None,
        "probe": ("CaM4", 1.0, 6.049e-6, None),
        "max_em": 3e-3,
        "refs_em": 1e-3,
    },
}
STEP_BOUNDS = (0.05, 0.01)

# The four NO_GROUND_TRUTH entries. An oscillator's ``agree_rows``: the engines agree
# at the suite's tolerance over this many leading rows of the job's grid, where the
# trajectory is still pinned; ``amplitude``: every species' min and max agree
# between engines within this fraction of its range.
OSCILLATORS = {
    "BIOMD0000000373:ode": {"agree_rows": 18, "amplitude": 0.02},
    "BIOMD0000000376:ode": {"agree_rows": 12, "amplitude": 0.03},
    "BIOMD0000000682:ode": {"agree_rows": 365, "amplitude": 0.01},
}
SPLIT = "MODEL2310250001:ode"
# The gluconate isotopologues and the 6PG pair they feed: no unique solution once
# GNT_e_tot crosses 0. GAP and PYR, downstream of 6PG, move by up to 10% of their
# range and are left out of both lists.
UNSETTLED = ("GNT_e_0", "GNT_e_1", "_6PG_0", "_6PG_1")
DOWNSTREAM = ("GAP_0", "GAP_1", "PYR_0", "PYR_1")
TIGHT = ((1e-12, 1e-18), (1e-13, 1e-20))


# --------------------------------------------------------------------------- #
# The entries (no corpus needed)
# --------------------------------------------------------------------------- #
def test_no_entry_cites_a_note_that_is_not_in_this_repository():
    """The triage's notes lived under a dev/ directory this repository never had."""
    assert "dev/" not in (_RR / "overrides.py").read_text()


def test_the_six_entries_are_where_their_evidence_puts_them():
    assert set(PULSES) <= set(ov.KNOWN_ARTIFACT)
    assert set(ov.NO_GROUND_TRUTH) == {*OSCILLATORS, SPLIT}
    assert not set(ov.NO_GROUND_TRUTH) & set(ov.KNOWN_ARTIFACT)


@pytest.mark.parametrize("key", sorted({*PULSES, *OSCILLATORS, SPLIT}))
def test_each_reason_names_the_test_that_runs_its_evidence(key):
    entry = ov.KNOWN_ARTIFACT.get(key) or ov.NO_GROUND_TRUTH[key]
    assert f"parity_checks/tests/{_THIS}" in entry["reason"]


def test_a_key_in_two_reclassifiers_is_refused(monkeypatch):
    """One row, one disposition: bngsim checked and right, or no ground truth."""
    key = next(iter(PULSES))
    monkeypatch.setitem(ov.NO_GROUND_TRUTH, key, {"issue": None, "reason": "r"})
    with pytest.raises(ValueError, match="KNOWN_ARTIFACT and NO_GROUND_TRUTH"):
        ov.overrides_for(*key.split(":"))


# --------------------------------------------------------------------------- #
# Running the models
# --------------------------------------------------------------------------- #
@cache
def _jobs() -> dict:
    return {f"{j.model_id}:{j.method}": j for j in read_manifest(_RR / "ode_jobs.json")[1]}


def _model(key: str):
    job = _jobs()[key]
    path = _RR / job.model
    if not path.is_file():
        pytest.skip(f"rr_parity corpus model for {key} not present")
    for mod in ("bngsim", "roadrunner"):
        pytest.importorskip(mod)
    p = job.params
    return path.read_text(), float(p.get("initial_time", p["t_start"])), p


@cache
def _bngsim(key: str, rtol: float, atol: float):
    """bngsim as rr_run runs it."""
    import _rr_common as rc

    xml, t0, p = _model(key)
    return rc.bn_ode(xml, t0, p["t_end"], p["n_points"], rtol, atol)[:3]


@cache
def _roadrunner(key: str, rtol: float, atol: float):
    """RoadRunner as rr_run runs it."""
    import _rr_common as rc

    xml, t0, p = _model(key)
    return rc.rr_ode(xml, t0, p["t_end"], p["n_points"], rtol, atol, through_event_instants=True)[
        :3
    ]


def _suite(key: str, engine):
    import _rr_common as rc

    return engine(key, rc.DEFAULT_RTOL, rc.DEFAULT_ATOL)


@cache
def _bounded_roadrunner(key: str, max_step: float, edges: tuple | None):
    """RoadRunner at the suite's tolerance that cannot step over a pulse: its step
    bounded by ``max_step``, and, with ``edges``, restarted at each one, one
    ``simulate`` call per interval of the job's grid and the edges.

    Each interval that ends at an edge ends one ulp short of it. A closed condition
    such as ``time >= 21`` already holds AT 21, and RoadRunner with a bounded step,
    which must end a step exactly there, stalls (CV_TOO_MUCH_WORK). Rows on the
    job's grid are returned, with the selections ``rr_ode`` makes."""
    import _rr_common as rc
    import roadrunner

    rc.set_rr_quiet()
    xml, t0, p = _model(key)
    r = roadrunner.RoadRunner(xml)
    r.integrator = "cvode"
    r.integrator.relative_tolerance = rc.DEFAULT_RTOL
    r.integrator.absolute_tolerance = rc.DEFAULT_ATOL
    r.integrator.maximum_time_step = max_step
    r.integrator.maximum_num_steps = 1_000_000
    boundary = set(r.model.getBoundarySpeciesIds())
    sel = [f"[{s}]" if s in boundary else s for s in r.timeCourseSelections]
    assert sel[0] == "time" and not boundary  # neither model has a boundary species
    r.timeCourseSelections = sel
    grid = np.linspace(t0, p["t_end"], p["n_points"])
    if edges is None:
        out = np.asarray(r.simulate(t0, p["t_end"], p["n_points"]))
    else:
        cuts = set(edges)
        points = sorted(set(grid.tolist()) | {e for e in cuts if t0 < e < p["t_end"]})
        on_grid = set(grid.tolist())
        rows = []
        for a, b in zip(points[:-1], points[1:], strict=True):
            end = float(np.nextafter(b, a)) if b in cuts else b
            seg = np.asarray(r.simulate(a, end, 2))
            if not rows:
                rows.append(seg[0])
            if b in on_grid:
                rows.append(seg[-1])
        out = np.vstack(rows)
    assert out.shape[0] == grid.size
    names = [c[1:-1] if c.startswith("[") else c for c in sel[1:]]
    return out[:, 0], out[:, 1:], names


def _compare(a, b, *, columns=None, rows=None) -> tuple[str, float]:
    """The suite's verdict of ``a`` against ``b`` (``pass`` / ``diff``) and its
    err/movement, over all common columns and rows unless narrowed. Over all of
    them it is rr_run's own ``_compare_ode``."""
    import _rr_common as rc
    import rr_run

    ai, bi, common = rc.align_common(a[2], b[2])
    if columns is not None:
        keep = [k for k, n in enumerate(common) if n in columns]
        ai, bi = [ai[k] for k in keep], [bi[k] for k in keep]
    sl = slice(None) if rows is None else slice(0, rows)
    v = differ.deterministic_verdict(a[1][sl, ai], b[1][sl, bi], atol=rc.DEFAULT_ATOL)
    status = "pass" if v["passed"] else "diff"
    if columns is None and rows is None:
        assert rr_run._compare_ode(a, b, atol=rc.DEFAULT_ATOL)[0] == status
    return status, v["max_err_over_movement"]


def _at(run, name: str, t: float) -> float:
    return float(run[1][int(np.argmin(np.abs(run[0] - t))), run[2].index(name)])


def _worst_over_scale(a, b) -> dict[str, float]:
    """Each common column's largest |a - b|, over b's range (or a millionth of its
    peak, for a column that holds still)."""
    import _rr_common as rc

    ai, bi, common = rc.align_common(a[2], b[2])
    x, y = a[1][:, ai], b[1][:, bi]
    scale = np.maximum(np.ptp(y, axis=0), 1e-6 * np.abs(y).max(axis=0))
    return dict(zip(common, (np.abs(x - y).max(axis=0) / scale).tolist(), strict=True))


# --------------------------------------------------------------------------- #
# KNOWN_ARTIFACT: RoadRunner steps over narrow pulses, bngsim resolves them
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", sorted(PULSES))
def test_two_step_bounds_give_one_reference(key):
    """RoadRunner at maximum_time_step 0.05 and 0.01 agree: the reference converged."""
    spec = PULSES[key]
    coarse, fine = (_bounded_roadrunner(key, h, spec["edges"]) for h in STEP_BOUNDS)
    status, em = _compare(coarse, fine)
    assert status == "pass" and em <= spec["refs_em"], em
    name, t, value, _ = spec["probe"]
    assert _at(fine, name, t) == pytest.approx(value, rel=1e-2)


@pytest.mark.parametrize("key", sorted(PULSES))
def test_bngsim_passes_against_the_reference(key):
    """bngsim, run as rr_run runs it, passes against both references."""
    spec = PULSES[key]
    bn = _suite(key, _bngsim)
    for h in STEP_BOUNDS:
        status, em = _compare(bn, _bounded_roadrunner(key, h, spec["edges"]))
        assert status == "pass" and em <= spec["max_em"], (h, em)
    name, t, value, _ = spec["probe"]
    assert _at(bn, name, t) == pytest.approx(value, rel=1e-2)


@pytest.mark.parametrize("key", sorted(PULSES))
def test_roadrunner_on_its_own_steps_over_the_pulses(key):
    """RoadRunner as rr_run runs it fails against the reference, and so the row is
    a DIFF that the entry reclassifies: the outlier is RoadRunner's own run."""
    spec = PULSES[key]
    rr = _suite(key, _roadrunner)
    assert _compare(rr, _bounded_roadrunner(key, STEP_BOUNDS[1], spec["edges"]))[0] == "diff"
    assert _compare(_suite(key, _bngsim), rr)[0] == "diff"
    name, t, value, own = spec["probe"]
    if own is None:  # it misses the pulse outright: off by orders of magnitude
        assert abs(_at(rr, name, t)) < 1e-3 * value
    else:
        assert _at(rr, name, t) == pytest.approx(own, rel=1e-2)


# --------------------------------------------------------------------------- #
# NO_GROUND_TRUTH: no tolerance pins the trajectory over the job's horizon
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("engine", ["bngsim", "roadrunner"])
@pytest.mark.parametrize("key", sorted(OSCILLATORS))
def test_neither_engine_agrees_with_itself_as_the_tolerance_tightens(key, engine):
    """At rtol 1e-12 and again at 1e-13, an engine's two runs DIFF, by as much as
    the engines differ from each other: the phase out to t_end is not pinned."""
    run = _bngsim if engine == "bngsim" else _roadrunner
    status, em = _compare(*(run(key, *tol) for tol in TIGHT))
    assert status == "diff" and em >= 0.5, em


@pytest.mark.parametrize("key", sorted(OSCILLATORS))
def test_the_oscillators_agree_where_the_trajectory_is_pinned(key):
    """At the suite's tolerance the engines agree over the leading rows, and on
    every species' amplitude over the whole horizon; the row is still a DIFF."""
    spec = OSCILLATORS[key]
    bn, rr = _suite(key, _bngsim), _suite(key, _roadrunner)
    assert _compare(bn, rr)[0] == "diff"
    assert _compare(bn, rr, rows=spec["agree_rows"])[0] == "pass"
    import _rr_common as rc

    ai, bi, common = rc.align_common(bn[2], rr[2])
    for i, j, name in zip(ai, bi, common, strict=True):
        x, y = bn[1][:, i], rr[1][:, j]
        span = np.ptp(y)
        assert abs(x.min() - y.min()) <= spec["amplitude"] * span, name
        assert abs(x.max() - y.max()) <= spec["amplitude"] * span, name


def test_the_isotopic_rate_law_is_singular_where_a_total_crosses_zero():
    """Every rate is isotopic_irreversible(v, S, Stot) = v*S/Stot, and the glucose
    and gluconate totals pass through 0 between t = 8 and 9."""
    libsbml = pytest.importorskip("libsbml")
    xml, _, _ = _model(SPLIT)
    model = libsbml.readSBMLFromString(xml).getModel()
    (fd,) = [
        f for f in model.getListOfFunctionDefinitions() if f.getId() == "isotopic_irreversible"
    ]
    assert libsbml.formulaToL3String(fd.getMath()) == "lambda(v, S, Stot, v * S / Stot)"
    bn = _suite(SPLIT, _bngsim)
    for total in ("GLC_e_tot", "GNT_e_tot"):
        assert _at(bn, total, 8.0) > 0 > _at(bn, total, 9.0), total


@pytest.mark.parametrize("engine", ["bngsim", "roadrunner"])
def test_the_isotopologue_split_converges_at_no_tolerance(engine):
    """At rtol 1e-12 and again at 1e-13, the gluconate split and the 6PG split it
    feeds move by more than a tenth of their range in either engine, while every
    column agrees through t = 8 and the 17 columns outside the split agree to
    1e-4 of theirs."""
    run = _bngsim if engine == "bngsim" else _roadrunner
    a, b = (run(SPLIT, *tol) for tol in TIGHT)
    worst = _worst_over_scale(a, b)
    for name in UNSETTLED:
        assert worst[name] > 0.1, (name, worst[name])
    settled = {n: w for n, w in worst.items() if n not in UNSETTLED + DOWNSTREAM}
    assert len(settled) == 17 and max(settled.values()) < 1e-4, settled
    rows_to_8 = int(np.searchsorted(a[0], 8.0)) + 1
    assert _compare(a, b, rows=rows_to_8)[0] == "pass"


def test_the_engines_agree_outside_the_split():
    """The row is a DIFF only in the split; on the other 17 columns bngsim and
    RoadRunner pass against each other at the suite's tolerance."""
    bn, rr = _suite(SPLIT, _bngsim), _suite(SPLIT, _roadrunner)
    assert _compare(bn, rr)[0] == "diff"
    rest = set(bn[2]) - set(UNSETTLED) - set(DOWNSTREAM)
    assert len(rest) == 17
    assert _compare(bn, rr, columns=rest)[0] == "pass"
