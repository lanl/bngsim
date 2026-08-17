"""GH #395 — what a non-finite forward-sensitivity value does must not depend on
which column it lands in.

CVODES reduces the per-column weighted norms to a single number with a
comparison (``cvSensNorm``, ``src/cvodes/cvodes.c``)::

    nrm = cv_mem->cv_cvals[0];
    for (is = 1; is < Ns; is++)
      if (cv_mem->cv_cvals[is] > nrm) { nrm = cv_mem->cv_cvals[is]; }

Every comparison against NaN is false. So that reduction **propagates** a NaN in
column 0 — it is the seed and nothing replaces it — and **discards** one in any
later column, which never wins the comparison. Both the staggered corrector's
convergence test and the sensitivity error test read that one number, so the
same NaN either stalls the corrector into ``CV_CONV_FAILURE`` or is invisible to
the solver and rides to the output scan, decided by nothing but the parameter's
position in ``sensitivity_params``.

Two places had to change, because a non-finite value reaches the solver by two
routes and only one of them is a value the solver computed:

* **the sensitivity RHS** — ``cvode_codegen_sens_rhs`` now returns the
  recoverable code on a non-finite ``ySdot`` instead of passing it through, so
  CVODES cuts ``h`` and retries at the point of production rather than inferring
  something from a norm that cannot see it. A NaN the predictor caused by
  overshooting a domain boundary for one step is then *rescued* rather than
  refused, and one that is really the model fails with
  ``CV_REPTD_SRHSFUNC_ERR`` / ``CV_FIRST_SRHSFUNC_ERR`` naming the time.
* **the initial seed** — ``dx/dtheta(0)`` is checked before ``CVodeSensInit1``.
  A NaN there is not something a step can fix, and it is not the dynamics: it
  comes from an initial-condition chain rule or from a carry-over installed by
  ``set_pending_sensitivity_seed``. It is refused by name.

The tests below pin the *symmetry*, not a particular flag, wherever the outcome
could be host-dependent. That is deliberate: GH #389 is a live case of a
sensitivity blow-up that reproduces on one host's arithmetic and not another's,
and a test that asserted the failure rather than the invariance would be a
platform tripwire.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

# `dose = 0` and a rate law of `sqrt(dose)`. The *value* is finite there
# (`sqrt(0)` is 0, the model just gets no dose); the derivative
# `d/d(dose) kcat*sqrt(dose) = kcat/(2*sqrt(dose))` is `+inf`. So this is a model
# whose state trajectory is clean and whose sensitivity RHS is non-finite on
# every call — deterministically, on every platform, with no dependence on where
# the predictor happens to put a state.
SQRT_AT_ZERO = """\
begin parameters
    1 dose   0.0  # Constant
    2 kcat   1.0  # Constant
    3 kdeg   0.3  # Constant
end parameters
begin functions
    1 uptake() kcat*sqrt(dose)
    2 clear()  kdeg
end functions
begin species
    1 A() 1.0
end species
begin reactions
    1 0 1 uptake #_R1
    2 1 0 clear #_R2
end reactions
begin groups
    1 Atot 1
end groups
"""

RUN = {"t_span": (0, 5), "n_points": 6, "rtol": 1e-9, "atol": 1e-12}


@pytest.fixture
def sqrt_net(tmp_path):
    net = tmp_path / "sqrt_at_zero.net"
    net.write_text(SQRT_AT_ZERO)
    return net


def _outcome(net, params):
    """``("raised", <flag or message head>)`` or ``("returned", tensor)``."""
    model = bngsim.Model.from_net(net)
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
    try:
        return "returned", np.asarray(sim.run(**RUN).sensitivities)
    except Exception as exc:  # noqa: BLE001 - the class is what is under test
        return "raised", str(exc)


class TestARhsThatIsNonFiniteOnEveryCall:
    """The deterministic half: the derivative is ``+inf`` from the first call,
    so nothing about the trajectory or the host decides what happens."""

    def test_the_trajectory_itself_is_clean(self, sqrt_net):
        """Without which this model would be testing the wrong thing — the
        defect is a sensitivity that is non-finite while the state is not."""
        model = bngsim.Model.from_net(sqrt_net)
        species = np.asarray(bngsim.Simulator(model, method="ode").run(**RUN).species)
        assert np.all(np.isfinite(species))

    @pytest.mark.parametrize(
        "params",
        [
            ["dose", "kcat", "kdeg"],  # the offending column first
            ["kcat", "dose", "kdeg"],  # ...in the middle
            ["kcat", "kdeg", "dose"],  # ...last
        ],
    )
    def test_every_position_fails_the_same_way(self, sqrt_net, params):
        kind, detail = _outcome(sqrt_net, params)
        assert kind == "raised"
        assert "CV_FIRST_SRHSFUNC_ERR" in detail or "CV_REPTD_SRHSFUNC_ERR" in detail
        # ...and it names where to look, rather than reporting a bare flag.
        assert "sensitivity RHS returned a non-finite value" in detail

    def test_the_position_used_to_decide_the_answer(self, sqrt_net, monkeypatch):
        """The A/B, so this file cannot pass vacuously if the fix is reverted:
        with the pre-#395 pass-through restored, the same model fails with two
        *different* CVODES flags depending on where ``dose`` sits."""
        monkeypatch.setenv("BNGSIM_SENS_NONFINITE_RECOVER", "0")
        first = _outcome(sqrt_net, ["dose", "kcat", "kdeg"])
        last = _outcome(sqrt_net, ["kcat", "kdeg", "dose"])

        assert first[0] == "raised" and last[0] == "raised"
        flags = {
            next((tok for tok in detail.split() if tok.startswith("flag=")), "")
            for _, detail in (first, last)
        }
        assert len(flags) == 2, f"expected the pre-fix asymmetry, got {flags}"


class TestASeedThatIsNonFinite:
    """The other route in: a NaN already sitting in ``dx/dtheta(0)``, which no
    step can remove. This is GH #395's own reproducer."""

    NET = Path(__file__).resolve().parents[2] / "tests" / "data" / "preequil_prod_deg.net"
    PARAMS = ["k_prod", "k_deg"]

    def _seeded(self, col):
        model = bngsim.Model.from_net(self.NET)
        seed = np.zeros((model._core.n_species, len(self.PARAMS)))
        seed[2, col] = np.nan  # row Ad(): a pure sink, whose row nothing reads
        model._core.set_pending_sensitivity_seed(seed, self.PARAMS)
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=self.PARAMS)
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the message is the assertion
            sim.run(t_span=(0, 5), n_points=11, rtol=1e-11, atol=1e-13, carry_sensitivities=True)
        return str(excinfo.value)

    @pytest.mark.parametrize("col", [0, 1])
    def test_it_is_refused_by_name_wherever_it_sits(self, col):
        message = self._seeded(col)
        assert "non-finite initial seed" in message
        # the column it is in and the row it is in, both named
        assert f"'{self.PARAMS[col]}'" in message
        assert "Ad()" in message

    def test_both_columns_give_the_same_diagnosis(self):
        """The invariant, stated directly. Before GH #395 column 0 gave
        ``CV_CONV_FAILURE`` from the corrector and column 1 ran to completion and
        was caught by the GH #384 output scan — two different reports of one
        poisoned seed."""
        first, second = self._seeded(0), self._seeded(1)
        assert first.replace("k_prod", "P") == second.replace("k_deg", "P")


class TestATransientNonFiniteIsRecovered:
    """``BIOMD0000000833`` is the corpus witness: its ``n`` column NaNs because
    CVODES' predictor puts a species at ``-3.75e-36`` for one internal step, and
    the run recovers once the step is rejected rather than the NaN being kept.

    The assertion is the GH #395 invariance, not the rescue: whether a given
    host's arithmetic visits that state at all is exactly the kind of thing
    GH #389 is open about, so a test that demanded the blow-up would be a
    platform tripwire. Invariance holds either way.
    """

    MODEL = (
        Path(__file__).resolve().parents[2]
        / "parity_checks"
        / "rr_parity"
        / "models"
        / "BIOMD0000000833"
        / "DiCamillo2016.xml"
    )
    RUN = {"t_span": (0.0, 60.0), "n_points": 101, "rtol": 1e-9, "atol": 1e-12}

    def _columns(self, params):
        if not self.MODEL.exists():  # pragma: no cover - corpus ships with the repo
            pytest.skip("BIOMD0000000833 not available")
        from bngsim import _sbml_loader as sl

        model = sl.load_sbml_string(self.MODEL.read_text())
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
        try:
            result = sim.run(max_steps=100000, **self.RUN)
        except Exception:  # noqa: BLE001
            return None
        return np.asarray(result.sensitivities)[:, :, params.index("n")]

    def test_the_column_is_the_same_first_or_last(self):
        first = self._columns(["n", "Kd_pkc", "Kd_akt"])
        last = self._columns(["Kd_pkc", "Kd_akt", "n"])

        assert (first is None) == (last is None), (
            "the same NaN was fatal in one column position and not the other"
        )
        if first is not None:
            assert np.array_equal(first, last, equal_nan=True)
