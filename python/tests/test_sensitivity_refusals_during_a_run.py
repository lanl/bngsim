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
column moves, two times it parts) stays a plain ``SimulationError``, whatever
its message advises: ``compute_all_sensitivities`` retries a failed chunk one
column at a time (GH #243) and names the columns that run alone, while it
re-raises a ``SensitivityUnsupportedError`` at once.
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
# found no column that ran alone for any of them).
TYPED = {
    "tangentially at t=": "#144, a trigger met tangentially",
    "after more state-dependent rate-law switches than": "#945/#1025, crossings over the cap",
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
    at = source.index(fragment)
    for m in re.finditer(r"throw ((?:\w+::)*\w+)\(", source[: at + 1]):
        if m.end() - 1 < at <= _argument_end(source, m.end() - 1):
            return m.group(1).removeprefix("bngsim::")
    m = re.compile(r"throw ((?:\w+::)*\w+)\(").search(source, at)
    return m.group(1).removeprefix("bngsim::")


def test_the_refusals_reached_whatever_columns_are_asked_are_thrown_typed():
    """Read from the source: each refusal in TYPED is SensitivityUnsupported, and
    no other throw is. A refusal whose condition turns on a requested column (a
    crossing or a fire it moves, a pair of times it parts) stays a
    runtime_error, whatever its message says to do: compute_all_sensitivities'
    per-column retry can run the other columns, and says which ones do."""
    source = SOURCE.read_text()
    typed = {**TYPED, **TYPED_BEFORE}
    for fragment in typed:
        assert source.count(fragment) == 1, fragment
    untyped = [f for f in typed if _thrown_as(source, f) != "SensitivityUnsupported"]
    assert not untyped, untyped
    thrown = re.findall(r"throw (?:bngsim::)?SensitivityUnsupported\(", source)
    assert len(thrown) == len(typed)
