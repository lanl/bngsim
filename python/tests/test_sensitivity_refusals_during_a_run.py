"""A forward-sensitivity run the solver refuses partway is a declared refusal by type.

The solver refuses a sensitivity run where it reaches a construct whose
derivative it cannot produce: an event that fires as its trigger leaves its
threshold flat (issue #934), a compartment that jumps twice at one instant
(#1181), a delay a ``run_batch`` row made positive (#144), and others. Most of
those refusals were thrown as ``std::runtime_error`` and reached Python as a
plain ``SimulationError``, which a caller, or a parity suite (BIOMD0000000285's
AMICI sensitivity row scored EXCEPTION), could tell from a solver crash only by
its text. Issue #1279 typed one of them. The ones reached whatever columns are
requested are now ``SensitivityRefusedDuringRunError``: a
``SensitivityUnsupportedError`` by type, and still the ``SimulationError`` an
existing ``except`` caught.

A refusal whose condition turns on a requested column (a crossing or a fire the
column moves, two times it parts) is ``SensitivityRefusedForParametersError``
(issue #1435), a subclass: ``compute_all_sensitivities`` retries a chunk refused
that way one column at a time (GH #243) and names the columns that run alone,
while it re-raises any other ``SensitivityUnsupportedError`` at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import bngsim
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src" / "cvode_simulator.cpp"

# X = Y = t, so X^2 > c leaves c = 0 at second order: no first-order motion.
FLAT_START = (
    "species X = 0, F = 0, G = 0; c = 0; k = 1\nX' = k;\nG' = F;\n"
    "E: at (X^2 > c), t0=false: F = 1000;\n"
)


def _flat_start_sim(**kw) -> bngsim.Simulator:
    return bngsim.Simulator(
        bngsim.Model.from_antimony_string(FLAT_START), method="ode", sensitivity_params=["c"], **kw
    )


def test_the_type_is_a_declared_refusal_and_the_simulation_error_it_was():
    cls = bngsim.SensitivityRefusedDuringRunError
    assert issubclass(cls, bngsim.SensitivityUnsupportedError)
    assert issubclass(cls, bngsim.SimulationError)
    assert "SensitivityRefusedDuringRunError" in bngsim.__all__


def test_a_refusal_the_run_reaches_is_raised_by_type_with_its_message():
    """BIOMD0000000285's death triggers are this refusal: the trigger leaves its
    threshold flat, and the time it fires has no derivative."""
    with pytest.raises(bngsim.SensitivityUnsupportedError) as refused:
        _flat_start_sim().run(sample_times=[0.0, 1.0, 2.0], rtol=1e-10, atol=1e-12)
    assert isinstance(refused.value, bngsim.SimulationError)
    assert re.search(r"no first-order motion \(issue #934\)", str(refused.value))


def test_a_batch_row_whose_delay_turns_positive_is_refused_by_type():
    """The pre-run gate passes a delay that is 0 at the values it sees (issue
    #835); a run_batch row that sets it positive reaches the fire, where the
    delayed apply has no sensitivity jump (issue #144)."""
    model = bngsim.Model.from_antimony_string(
        "species X = 0; k = 1; d = 0\nX' = -k*X;\nE: at d after (time > 1): X = 5;\n"
    )
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k"])
    with pytest.raises(bngsim.SensitivityRefusedDuringRunError, match="delayed events"):
        sim.run_batch(params=[{"d": 0.5}], t_span=(0, 3), n_points=4)


def test_compute_all_sensitivities_raises_the_refusal_without_retrying_columns(monkeypatch):
    """A refusal that holds for any parameter fails each column alone too, so
    the chunk's refusal is raised as it is, not after a column-by-column retry
    that reports every column unresolvable."""
    sim = bngsim.Simulator(bngsim.Model.from_antimony_string(FLAT_START), method="ode")
    retried = []
    original = bngsim.Simulator._recover_failed_chunk
    monkeypatch.setattr(
        bngsim.Simulator,
        "_recover_failed_chunk",
        lambda self, *a, **k: retried.append(a) or original(self, *a, **k),
    )
    with pytest.raises(bngsim.SensitivityRefusedDuringRunError, match=r"issue #934"):
        sim.compute_all_sensitivities(params=["c", "k"], t_span=(0, 2), n_points=3, chunk_size=2)
    assert retried == []


# The refusals this change types, each by a fragment of its message found
# nowhere else in the solver. Each is reached whatever columns are requested:
# what throws reads whether the run has sensitivities and the trajectory, not
# which parameters move what (and a per-parameter probe of the refusal tests
# found no column that ran alone for any of them a test reaches; no test
# reaches the two caps).
TYPED = {
    "tangentially at t=": "#144, a trigger met tangentially",
    "after more state-dependent rate-law switches than": "#945/#1025, crossings over the cap",
    "after more event fires than are kept": "#945/#1025, fires over the cap (issue #1435)",
    "it resizes a compartment after an earlier event": "#936",
    "the state is held on the surface of a state-dependent": "#926",
    "and a probe step of": "#763, a probe step past a crossing",
    "bngsim refuses rather than apply it": "#763, a jump applied twice",
    "which starts exactly on its threshold": "#934",
    "times at one instant, and its concentrations": "#1181",
    "a strict comparison, and event": "#947/#144",
    "delayed events is not yet supported": "#144, a delay made positive",
    "and another compartment's jump of the same instant": "#1054, two rescales",
    "where a compartment's size jumps, and its values, frozen": "#1054, a frozen value",
    "events triggering events": "#205/#144, a cascade",
    "within roundoff after an output point": "#950",
    "crosses at exactly the instant t=": "#150, a switch on an event",
    "triggers within the step the solver takes past": "#150, a fire inside the step",
}


# Typed before this change, by #1279, and kept so: it does turn on a column
# where a restart lands on the stop (a moving one is refused, a fixed one not),
# and making it a runtime_error would move its parity rows from UNSUPPORTED to
# EXCEPTION.
TYPED_BEFORE = {"the state, which sat on that equality": "#1228/#1279, a held equality left"}

_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')


def _argument_end(source: str, open_paren: int) -> int:
    """Where the parenthesised argument opened at ``open_paren`` closes, string
    literals skipped."""
    depth, i = 0, open_paren
    while i < len(source):
        literal = _LITERAL.match(source, i)
        if literal:
            i = literal.end()
            continue
        depth += {"(": 1, ")": -1}.get(source[i], 0)
        if depth == 0:
            return i
        i += 1
    raise ValueError("unbalanced")


def _thrown_as(source: str, fragment: str) -> str:
    """The type of the throw whose message holds ``fragment``: the throw whose
    argument it is in, or, for a message built in a stream first, the next
    throw after it."""
    return _thrown_at(source, source.index(fragment))


def _enclosing_throw(source: str, at: int) -> str | None:
    """The type of the throw whose argument holds the text at ``at``, if one does."""
    for m in re.finditer(r"throw ((?:\w+::)*\w+)\(", source[: at + 1]):
        if m.end() - 1 < at <= _argument_end(source, m.end() - 1):
            return m.group(1).removeprefix("bngsim::")
    return None


def _thrown_at(source: str, at: int) -> str:
    """The type of the throw whose message holds the text at ``at``."""
    enclosing = _enclosing_throw(source, at)
    if enclosing is not None:
        return enclosing
    m = re.compile(r"throw ((?:\w+::)*\w+)\(").search(source, at)
    return m.group(1).removeprefix("bngsim::")


def test_the_refusals_reached_whatever_columns_are_asked_are_thrown_typed():
    """Read from the source: each refusal in TYPED is SensitivityUnsupported, and
    no other throw is. A refusal whose condition turns on a requested column (a
    crossing or a fire it moves, a pair of times it parts) is its
    SensitivityRefusedForParameters instead, whatever its message says to do:
    compute_all_sensitivities' per-column retry can run the other columns, and
    says which ones do."""
    source = SOURCE.read_text()
    typed = {**TYPED, **TYPED_BEFORE}
    for fragment in typed:
        assert source.count(fragment) == 1, fragment
    untyped = [f for f in typed if _thrown_as(source, f) != "SensitivityUnsupported"]
    assert not untyped, untyped
    thrown = re.findall(r"throw (?:bngsim::)?SensitivityUnsupported\(", source)
    assert len(thrown) == len(typed)


# Issue #1435: the refusals whose condition turns on the columns requested. Two
# events on one instant: E0 on the clock, E1 when q, which k drives, reaches
# 2.5. k moves E1's time and not E0's, so their order flips across k and the
# result has no derivative there (issues #49, #1277). Nothing reads u, so its
# column runs.
ONE_INSTANT = (
    "species A, B; A = 3; B = 3; q = 0; k = 1; u = 1\nq' = k; A' = 1; B' = 1\n"
    "E0: at (time >= 2.5): A = A + 1\nE1: at (q >= 2.5): B = B/2\n"
)
GRID = dict(sample_times=[0.0, 1.0, 2.0, 3.0], rtol=1e-10, atol=1e-12)


def _one_instant(**kw) -> bngsim.Simulator:
    return bngsim.Simulator(bngsim.Model.from_antimony_string(ONE_INSTANT), method="ode", **kw)


def test_the_refusal_for_the_parameters_asked_is_a_refusal_during_the_run():
    cls = bngsim.SensitivityRefusedForParametersError
    assert issubclass(cls, bngsim.SensitivityRefusedDuringRunError)
    assert issubclass(cls, bngsim.SimulationError)
    assert "SensitivityRefusedForParametersError" in bngsim.__all__


def test_a_refusal_that_turns_on_the_parameters_asked_is_raised_by_type():
    """It was a plain SimulationError, told from a solver failure only by its
    text. The column the condition does not turn on runs."""
    with pytest.raises(
        bngsim.SensitivityRefusedForParametersError, match=r"issues #49, #1277"
    ) as refused:
        _one_instant(sensitivity_params=["k"]).run(**GRID)
    assert isinstance(refused.value, bngsim.SimulationError)
    assert _one_instant(sensitivity_params=["u"]).run(**GRID).has_sensitivities


def test_compute_all_sensitivities_retries_such_a_chunk_and_reports_it_by_type(monkeypatch):
    """The chunk is retried a column at a time, as an untyped failure was, and
    the columns refused alone are reported as a refusal: by type, with no advice
    to loosen tolerances or regroup, which suits a solver failure."""
    retried = []
    original = bngsim.Simulator._recover_failed_chunk
    monkeypatch.setattr(
        bngsim.Simulator,
        "_recover_failed_chunk",
        lambda self, *a, **k: retried.append(a) or original(self, *a, **k),
    )
    with pytest.raises(bngsim.SensitivityRefusedForParametersError) as refused:
        _one_instant().compute_all_sensitivities(
            params=["k", "u"], t_span=(0, 3), n_points=4, chunk_size=2
        )
    assert len(retried) == 1
    text = str(refused.value)
    assert "Column(s) ['k'] are refused on their own; ['u'] integrated fine alone." in text
    assert text.endswith("Re-run with params=['u'] for the remaining 1 column(s).")
    assert "loosen" not in text and "CVODES error test" not in text


@pytest.mark.parametrize("num_processors", [1, 2])
def test_a_run_batch_row_refused_for_its_parameters_is_named(num_processors):
    """The refusal turns on the row's values, so a scan says which row: at k = 2
    E1 fires at t = 1.25, apart from E0, and at k = 1 the two share t = 2.5."""
    sim = _one_instant(sensitivity_params=["k"])
    with pytest.raises(
        bngsim.SensitivityRefusedForParametersError,
        match=r"^run_batch row 1: Forward sensitivity: events 'E0' and 'E1'",
    ):
        sim.run_batch(
            params=[{"k": 2.0}, {"k": 1.0}],
            t_span=(0, 3),
            n_points=4,
            num_processors=num_processors,
        )


DECAY = "species X = 1; k1 = 1; k2 = 1\nX' = -k1*X - k2*X;\n"


def _scripted(monkeypatch, outcomes: dict) -> bngsim.Simulator:
    """A Simulator whose sensitivity chunks raise what ``outcomes`` gives their
    columns, and run otherwise."""
    original = bngsim.Simulator._run_sensitivity_chunk

    def run(self, params, *a, **k):
        failure = outcomes.get(tuple(params))
        if failure is not None:
            raise failure
        return original(self, params, *a, **k)

    monkeypatch.setattr(bngsim.Simulator, "_run_sensitivity_chunk", run)
    return bngsim.Simulator(bngsim.Model.from_antimony_string(DECAY), method="ode")


def _all(sim: bngsim.Simulator, chunk_size: int = 2):
    return sim.compute_all_sensitivities(
        params=["k1", "k2"], t_span=(0, 1), n_points=3, chunk_size=chunk_size
    )


def _refused(what: str) -> Exception:
    return bngsim.SensitivityRefusedForParametersError(f"Forward sensitivity: {what}")


def test_a_column_that_fails_alone_otherwise_reports_the_failure_it_was(monkeypatch):
    sim = _scripted(
        monkeypatch,
        {
            ("k1", "k2"): _refused("the pair"),
            ("k1",): _refused("k1 alone"),
            ("k2",): bngsim.SimulationError("CVODE failed"),
        },
    )
    with pytest.raises(bngsim.SimulationError) as failed:
        _all(sim)
    assert not isinstance(failed.value, bngsim.SensitivityUnsupportedError)
    assert "or loosen rtol/atol" in str(failed.value)


def test_every_column_refused_alone_leaves_no_re_run_to_advise(monkeypatch):
    sim = _scripted(
        monkeypatch,
        {("k1", "k2"): _refused("the pair"), ("k1",): _refused("k1"), ("k2",): _refused("k2")},
    )
    with pytest.raises(bngsim.SensitivityRefusedForParametersError) as refused:
        _all(sim)
    assert str(refused.value).endswith("Column(s) ['k1', 'k2'] are refused on their own.")


def test_columns_refused_alone_are_a_refusal_whatever_failed_the_chunk(monkeypatch):
    """The report is about the columns alone: a pair that failed as a solver
    failure, then one column refused alone and the other run, is a refusal."""
    sim = _scripted(
        monkeypatch,
        {("k1", "k2"): bngsim.SimulationError("CVODE failed"), ("k1",): _refused("k1 alone")},
    )
    with pytest.raises(bngsim.SensitivityRefusedForParametersError) as refused:
        _all(sim)
    assert "Column(s) ['k1'] are refused on their own; ['k2'] integrated fine alone." in str(
        refused.value
    )


def test_a_column_the_model_refuses_alone_is_that_refusal(monkeypatch):
    """A refusal no column changes, met by one column alone: no column runs, so
    it is raised as it is, as the chunk's own would be."""
    model_wide = bngsim.SensitivityRefusedDuringRunError("Forward sensitivity: the model")
    sim = _scripted(monkeypatch, {("k1", "k2"): _refused("the pair"), ("k1",): model_wide})
    with pytest.raises(bngsim.SensitivityRefusedDuringRunError) as refused:
        _all(sim)
    assert refused.value is model_wide


def test_a_column_run_alone_and_refused_is_reported_by_type(monkeypatch):
    sim = _scripted(monkeypatch, {("k1",): _refused("k1 alone")})
    with pytest.raises(bngsim.SensitivityRefusedForParametersError) as refused:
        _all(sim, chunk_size=1)
    assert "Re-run with params=['k2']" in str(refused.value)


def test_a_chunk_too_wide_to_retry_is_reported_by_type(monkeypatch):
    sim = _scripted(monkeypatch, {("k1", "k2"): _refused("the pair")})
    monkeypatch.setattr(bngsim.Simulator, "_MAX_COLUMNS_FOR_SOLO_RETRY", 1)
    with pytest.raises(bngsim.SensitivityRefusedForParametersError) as refused:
        _all(sim)
    text = str(refused.value)
    assert "which of them is refused alone is unknown" in text and "loosen" not in text


# The one run-time message naming forward sensitivity that is not a refusal: a
# guard on a state the solver should not reach, which asks for a report. Typed,
# a parity suite would score it as a declared refusal and never see it.
UNTYPED = {"' was found triggered at t=": "GH #205/#910, a fire found and not located"}

# Throws of SensitivityRefusedForParameters: one per message below but #951's
# two, kept as a reason and thrown at one place. The type says a smaller set of
# columns may run, so the per-column retry is worth its cost; a refusal reached
# whatever columns are asked is in TYPED, raised at once.
REFUSED_FOR_PARAMETERS = 28

_KEPT = re.compile(r"(\w+) = msg\.str\(\);")


def _refusal_thrown_as(source: str, at: int) -> str:
    """``_thrown_at``, and for a message kept as a reason before any throw (#951's
    ``not_commuting``), the type of the throw of that reason."""
    enclosing = _enclosing_throw(source, at)
    if enclosing is not None:
        return enclosing
    kept = _KEPT.search(source, at)
    throw = re.compile(r"throw ").search(source, at)
    if kept and throw and kept.start() < throw.start():
        name = re.escape(kept.group(1))
        m = re.compile(rf"throw ((?:\w+::)*\w+)\({name}\)").search(source, kept.end())
        return m.group(1).removeprefix("bngsim::")
    return _thrown_at(source, at)


def test_every_refusal_the_run_reaches_has_a_type():
    """Issue #1435: each refusal the solver's source builds as "Forward
    sensitivity: ..." is thrown typed, so no caller reads its text."""
    source = SOURCE.read_text()
    starts = [m.start() for m in re.finditer(r'"Forward sensitivity: ', source)]
    assert len(starts) >= 45
    for fragment in UNTYPED:
        assert source.count(fragment) == 1, fragment
    untyped_at = {source.index(f) for f in UNTYPED}
    wrong = []
    for at in starts:
        kind = _refusal_thrown_as(source, at)
        guard = any(0 < u - at < 200 for u in untyped_at)
        want = (
            {"std::runtime_error"}
            if guard
            else {
                "SensitivityUnsupported",
                "SensitivityRefusedForParameters",
            }
        )
        if kind not in want:
            wrong.append((source.count("\n", 0, at) + 1, kind))
    assert not wrong, wrong
    thrown = re.findall(r"throw (?:bngsim::)?SensitivityRefusedForParameters\(", source)
    assert len(thrown) == REFUSED_FOR_PARAMETERS
