"""A forward-sensitivity tensor that comes back non-finite is refused (issue #384).

CVODES can return ``CV_SUCCESS`` with NaN in the sensitivity vectors, and the
reason is structural rather than unlucky: the error test is a comparison, and
every comparison against NaN is false, so the machinery whose job is to reject
the value cannot reject it once it has arrived. The run then reports a clean
solve and hands back a poisoned gradient.

``BIOMD0000000480`` is the case. Its ``parameter_63`` column tracks AMICI to six
significant figures for 963 of 1001 output points, and then all 41 of its rows
go NaN together while the state trajectory stays finite and
``n_err_test_fails`` reads 0.

**The blow-up itself is not fixed here, and is not a logic error in any one
component.** It is admitted by the issue #177 tolerance floor's relaxation, but
every perturbation of that floor — including ones that only tighten it — moves
the failure somewhere else rather than removing it. Sweeping the floor's time
scale gives 0 non-finite cells at 9e-4, 1517 at 1e-3, 0 at 1.05e-3, 33661 at
1.2e-3 and 0 at 1.5e-3: a knife edge, not a rule with a bug in it. What is
fixable, and what these tests pin, is that the failure must not be silent.
"""

import os
import re
from pathlib import Path

import bngsim
import numpy as np
import pytest

# The manifest settings this model fails under. n_points is load-bearing: CVODES
# steps to each output point, so 201 or 101 points walk a different trajectory
# and both come back finite. A test that "simplified" it would pass vacuously.
T_SPAN = (0.0, 10.0)
N_POINTS = 1001
RUN = {"rtol": 1e-9, "atol": 1e-12, "max_steps": 100000}


def _model_path():
    p = Path(__file__).resolve().parents[2] / "benchmarks" / "sbml_events" / "BIOMD0000000480.xml"
    if not p.exists():  # pragma: no cover - the corpus ships with the repo
        pytest.skip("BIOMD0000000480 not available")
    return str(p)


def _targets(m):
    """The parameter list amici_parity negotiates: budget-capped, no
    function-backed slots and no compartment sizes (both are refused by name,
    which would mask what this module is testing)."""
    cap = max(1, 20000 // max(len(m.species_names), 1))
    skip = (m._internal_param_names() & set(m.function_names)) | set(m.compartment_size_params)
    return [n for n in m.primary_param_names if n not in skip][:cap]


def _run(**env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        m = bngsim.Model.from_sbml(_model_path())
        sim = bngsim.Simulator(
            m, method="ode", sensitivity_params=_targets(m), sensitivity_method="staggered"
        )
        return sim.run(t_span=T_SPAN, n_points=N_POINTS, **RUN)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestTheRefusal:
    def test_a_nonfinite_tensor_is_refused_rather_than_returned(self):
        """The defect was a silent NaN: this run used to succeed."""
        with pytest.raises(bngsim.SimulationError, match="non-finite"):
            _run()

    def test_the_message_localizes_the_failure(self):
        """A caller has to be able to act on it: which column, and when.

        The column is what to drop to get a usable run out of the same model,
        and the time is where to start a bisection.

        The exact crossing time is deliberately NOT asserted. This is a knife
        edge — the same sweep that shows 0 non-finite cells at tau 1.05e-3 and
        33661 at 1.2e-3 also moves the instant across platforms, and an earlier
        version of this test pinned macOS's ``t=9.64`` and failed on both Linux
        legs while the refusal itself fired identically. What is portable is the
        SHAPE of the diagnostic: a column, a count, and a located output point.
        """
        with pytest.raises(bngsim.SimulationError) as exc:
            _run()
        msg = str(exc.value)
        assert "parameter_63" in msg
        assert re.search(r"\d+ cell\(s\)", msg)
        assert re.search(rf"output point t=[\d.eE+-]+ \(index \d+ of {N_POINTS}\)", msg)

    def test_the_message_carries_both_remedies(self):
        """The two things a caller can do next, and the counter that explains
        why the solver looked clean. Asserted because a diagnostic that stops
        naming them is a diagnostic that stops being actionable."""
        with pytest.raises(bngsim.SimulationError) as exc:
            _run()
        msg = str(exc.value)
        assert "n_sens_err_test_fails" in msg
        assert "BNGSIM_SENS_ERROR_FLOOR=0" in msg

    def test_the_remedy_the_message_advertises_works(self):
        """The diagnostic names ``BNGSIM_SENS_ERROR_FLOOR=0``. If that stops
        being true the message is worse than none, so it is asserted, not
        described — and it also pins the attribution to the issue #177 floor."""
        r = _run(BNGSIM_SENS_ERROR_FLOOR="0")
        assert np.isfinite(np.asarray(r.sensitivities)).all()


class TestTheSensitivityCounters:
    """``n_err_test_fails`` counts the STATE solve. Reading it as "the solve was
    clean" is how a poisoned column got called healthy, so CVODES' separate
    sensitivity counters are reported separately (issue #384)."""

    def test_the_state_counter_does_not_see_the_sensitivity_rejections(self):
        r = _run(BNGSIM_SENS_ERROR_FLOOR="0")
        stats = r.solver_stats
        assert stats["n_err_test_fails"] == 0
        assert stats["n_sens_err_test_fails"] > 0, (
            "the run whose state error test never fails is exactly the one whose "
            "sensitivity error test fails dozens of times — if this ever reads 0, "
            "the counter is being read off the wrong solve again"
        )

    def test_both_counters_are_present_and_zero_without_sensitivities(self):
        m = bngsim.Model.from_sbml(_model_path())
        r = bngsim.Simulator(m, method="ode").run(t_span=T_SPAN, n_points=11, **RUN)
        assert r.solver_stats["n_sens_err_test_fails"] == 0
        assert r.solver_stats["n_sens_nonlin_conv_fails"] == 0
