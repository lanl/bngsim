"""Issues #201 and #257 — no two model clones may share ExprTk state.

``ExprTkEvaluator::Impl`` used to share one ``exprtk::parser`` across every clone
of a model, to avoid re-constructing the ~100KB template object per clone. The
comment justifying that said the parser "is stateless between compile() calls",
which is true *between* sequential calls and false *during* one: ``compile()``
drives a lexer, a token scanner and an error list that all live in the parser
object.

Two threads compiling through one parser therefore corrupt each other. The
observed failures on ``compute_all_sensitivities``'s default fan-out were 0 of 8
runs surviving — SIGSEGV, SIGABRT and SIGBUS, plus an ``ERR244 - Expected ')'``
which is ExprTk's way of reporting an *unregistered symbol* when one compile's
symbol resolution is clobbered by another.

**#201's fix — a mutex around compile() — was necessary and not sufficient, and
this test measured the difference.** It went on failing about 10% of the time
(2 of 20 consecutive runs on ``main``), with a *fourth* signature: SIGTRAP, the
macOS malloc freelist check, raised in ``NetworkModel::clone()`` while another
thread sat in ``ExprTkEvaluator::evaluate()`` under ``CVode``. That thread was
not compiling, so no lock on ``compile()`` was ever going to cover it. The cause
(issue #257) is that ``exprtk::parser::compile()`` ends by copying the compiled
expression's symbol tables into the parser and never clears them, so a shared
parser holds a strong handle on one clone's symbol table and drops it from
whichever thread compiles next — through a refcount that is a plain
``std::size_t``. Clones now share no ExprTk state at all, and this test went from
18 of 20 to 25 of 25. The C++ side of the coverage is
``test_evaluator_clone_shares_no_exprtk_state`` and
``test_concurrent_clone_compile_evaluate`` in ``tests/test_bngsim.cpp``.

**Why it needs an integration running, not just concurrent clones.** Concurrent
``clone()`` alone cannot trip it — the GIL serializes it (1200 concurrent clones of
this very model are clean). The second compiler has to be running with the GIL
released, and one is: ``NetworkModel::event_trigger_residual_expr`` is a lazy memo
that compiles on first use, and ``cvode_simulator.cpp`` calls it from inside the
integration. So the trip is one thread compiling GIL-free inside a run while
another compiles inside ``clone()`` — which is why this reproduces on an *events*
model under a parallel sensitivity job and nowhere else.

**The subprocess, and why the in-process companion is in C++.** Half the failure
modes are a signal, not an exception, so an in-process assertion cannot observe
them — a SIGSEGV takes the whole pytest session down and reports nothing useful.
Each trial therefore runs in its own process and the assertion is on the exit
status. A faster in-process *Python* test of the same mechanism (clone on one
thread while another integrates) was written and then **deleted**: rebuilt without
the lock it still passed at 400 iterations, so it could not fail, and a test that
cannot fail is worse than no test — it reads as coverage. The GIL is why: it
serializes the clone-side compiles, leaving only the handful of GIL-free lazy
memos per run to collide. The #257 companion is in C++ for exactly that reason —
no GIL, so 8 threads collide freely and the pre-fix design dies in 8 of 8 runs
rather than 1 in 10.

**The repeat count.** A race that has been fixed and a race that merely did not
fire look identical in one run. Six trials is not proof either; it is calibrated
against the measured pre-fix rate of 0 survivors in 8, where P(6 false passes) is
negligible. The real guarantee is that no ExprTk state crosses a clone boundary.
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
        f"{len(failures)} of {_TRIALS} parallel sensitivity runs died — ExprTk state "
        f"is crossing a model-clone boundary. A signal here means memory, not logic: "
        f"SIGSEGV/SIGABRT/SIGBUS is a parser compiled through by more than one thread "
        f"(issue #201), SIGTRAP is the macOS malloc freelist check firing on a heap "
        f"already corrupted by a shared symbol table (issue #257).\n\n" + "\n\n".join(failures)
    )
