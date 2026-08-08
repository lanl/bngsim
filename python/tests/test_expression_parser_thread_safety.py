"""Issue #201 — the ExprTk parser is shared across clones and must be serialized.

``ExprTkEvaluator::Impl`` shares one ``exprtk::parser`` across every clone of a
model, to avoid re-constructing the ~100KB template object per clone. The comment
justifying that said the parser "is stateless between compile() calls", which is
true *between* sequential calls and false *during* one: ``compile()`` drives a
lexer, a token scanner and an error list that all live in the parser object.

Two threads compiling through one parser therefore corrupt each other. The
observed failures on ``compute_all_sensitivities``'s default fan-out were 0 of 8
runs surviving — SIGSEGV, SIGABRT and SIGBUS, plus an ``ERR244 - Expected ')'``
which is ExprTk's way of reporting an *unregistered symbol* when one compile's
symbol resolution is clobbered by another.

**Why it needs an integration running, not just concurrent clones.** Concurrent
``clone()`` alone cannot trip it — the GIL serializes it (1200 concurrent clones of
this very model are clean). The second compiler has to be running with the GIL
released, and one is: ``NetworkModel::event_trigger_residual_expr`` is a lazy memo
that compiles on first use, and ``cvode_simulator.cpp`` calls it from inside the
integration. So the trip is one thread compiling GIL-free inside a run while
another compiles inside ``clone()`` — which is why this reproduces on an *events*
model under a parallel sensitivity job and nowhere else.

**The subprocess, and why there is no in-process companion.** Half the failure
modes are a signal, not an exception, so an in-process assertion cannot observe
them — a SIGSEGV takes the whole pytest session down and reports nothing useful.
Each trial therefore runs in its own process and the assertion is on the exit
status. A faster in-process test of the same mechanism (clone on one thread while
another integrates) was written and then **deleted**: rebuilt without the lock it
still passed at 400 iterations, so it could not fail, and a test that cannot fail
is worse than no test — it reads as coverage. What survives is calibrated: 6 of 6
trials die on the unlocked build (SIGSEGV, 3x SIGABRT, SIGBUS, one ERR247).

**The repeat count.** A race that has been fixed and a race that merely did not
fire look identical in one run. Six trials is not proof either; it is calibrated
against the measured pre-fix rate of 0 survivors in 8, where P(6 false passes) is
negligible. The real guarantee is that the parser now travels with its mutex.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
# Tracked (214 files under benchmarks/sbml_events), unlike the gitignored
# parity_checks corpus — so this runs in CI and in a worktree, not just here.
_EVENTS_MODEL = _REPO / "benchmarks" / "sbml_events" / "BIOMD0000000701.xml"

_TRIALS = 6

# Runs the issue's reproducer at its DEFAULT width. `params=None` is the case the
# issue points at: nothing in the suite exercised it, which is how a 0-of-8 defect
# in a public API's default configuration survived.
_REPRO = """
import warnings; warnings.filterwarnings("ignore")
import bngsim
m = bngsim.Model.load({path!r})
r = bngsim.Simulator(m).compute_all_sensitivities((0.0, 100.0), 21)
assert r.sensitivities.shape[0] == 21, r.sensitivities.shape
print("OK", r.sensitivities.shape)
"""


requires_events_model = pytest.mark.skipif(
    not _EVENTS_MODEL.is_file(), reason="benchmarks/sbml_events corpus not present"
)


@requires_events_model
def test_parallel_compute_all_sensitivities_survives_its_default_fanout():
    """The reported bug, at the width that was never covered.

    ``compute_all_sensitivities`` chunks at ``chunk_size=2`` and fans out to
    ``min(n_chunks, cpu_count)`` threads *by default*, so this is the default path
    for any model with enough parameters. 71 parameters here -> 36 chunks.
    """
    script = _REPRO.format(path=str(_EVENTS_MODEL))
    failures = []
    for trial in range(_TRIALS):
        p = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
        )
        if p.returncode != 0:
            how = f"signal {-p.returncode}" if p.returncode < 0 else f"exit {p.returncode}"
            failures.append(f"trial {trial}: {how}\n{p.stderr[-700:]}")
    assert not failures, (
        f"{len(failures)} of {_TRIALS} parallel sensitivity runs died — the shared "
        f"ExprTk parser is being compiled through by more than one thread (issue "
        f"#201).\n\n" + "\n\n".join(failures)
    )
