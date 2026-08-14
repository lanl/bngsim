"""Issue #339 — the per-solve step budget, and that both engines get the same one.

#331 raised each model's parameter count from a flat 20 to a coupled-state
budget. On the first full sweep carrying it, **8 models went `PASS` →
`REFERENCE_FAILED`**: AMICI ran out of integrator steps at the higher `Np` and
the models lost their oracle entirely.

Both engines default to 10,000 steps — bngsim's ``Simulator.run(max_steps=)`` and
AMICI's ``Solver.set_max_steps`` — and the harness set neither, so the budgets
were already equal and the rows were an honest engine-capability difference
(bngsim solved all eight within the same budget). What made it the wrong result
is that 10,000 was calibrated for a coupled system of ``n_species*(20+1)`` and
was being spent on up to ``n_species*(306+1)``.

**The budget is flat, not scaled to the coupled system as the issue proposed.**
All eight were probed at 10k / 100k / 1M before choosing:

    BIOMD0000000832 (Np 56)   10k AMICI_ERROR       -> 100k ok   0.6 s
    BIOMD0000000061 (Np 69)   10k TOO_MUCH_WORK     -> 100k ok   0.9 s
    BIOMD0000000667 (Np 83)   10k TOO_MUCH_WORK     -> 100k ok   4.4 s
    BIOMD0000000474 (Np 150)  10k TOO_MUCH_WORK     -> 100k ok  15.9 s
    MODEL2401050001 (Np 161)  10k TOO_MUCH_WORK     -> 100k ok  11.9 s
    MODEL2202020001 (Np 188)  10k TOO_MUCH_WORK     -> 100k ok   5.0 s
    MODEL0911120000 (Np 33)   fails at 10k, 100k AND 1M — NaN in sxdot[7]
    MODEL1701170001 (Np 135)  fails at all three    — NaN in sxdot[0]

Six recover; two never will, and their `TOO_MUCH_WORK` was a symptom of a NaN in
AMICI's own sensitivity RHS rather than of the step budget. The coupled size does
not predict the need either: the *smallest* system of the eight (9 species x 34)
is the one no budget rescues, while 37 x 189 clears 100k in 5 seconds. Step count
tracks stiffness, not width, so scaling by width would be a rule the evidence
does not support.

1M is not chosen because the ceiling is paid by the models it cannot help: a
recovering model costs ~nothing extra (the 10k run burned its budget before
failing anyway) while `MODEL0911120000` goes 0.2 s → 3.1 s → 32.6 s across the
three budgets. The per-job ``--timeout`` remains the real bound on a runaway.

What these tests protect is **symmetry**. The suite rests on both engines being
solved under identical conditions; a budget raised for the reference alone would
hand the oracle room the engine under test does not get, and would silently
convert an engine-capability difference into a harness artifact.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import _amici_sens as asens  # isort: skip  (suite dir is on sys.path via conftest)
import amici_sens_run as run  # isort: skip


# ── The budget reaches both engines, and it is the same number ──────────────


def test_both_engine_wrappers_take_a_step_budget():
    """Neither engine may be left on its own default while the other is raised."""
    for fn in (asens.bn_sens, asens.amici_sens):
        sig = inspect.signature(fn)
        assert "max_steps" in sig.parameters, f"{fn.__name__} has no max_steps"
        assert sig.parameters["max_steps"].default == asens.DEFAULT_SENS_MAX_STEPS


def test_the_worker_hands_both_engines_the_same_budget(monkeypatch):
    """The load-bearing property, asserted on the worker rather than by reading it.

    Both calls are fed from one local read of the spec, so this fails if a future
    edit raises one side and not the other.
    """
    seen: dict[str, int] = {}

    def _fake_bn(*args, **kwargs):
        seen["bngsim"] = kwargs["max_steps"]
        raise RuntimeError("stop after recording")

    def _fake_am(*args, **kwargs):
        seen["amici"] = kwargs["max_steps"]
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(asens, "bn_sens", _fake_bn)
    monkeypatch.setattr(asens, "amici_sens", _fake_am)
    _run_worker_to_the_engine_calls(monkeypatch, spec_extra={"max_steps": 4321})

    assert seen == {"bngsim": 4321, "amici": 4321}


def test_a_spec_without_the_key_falls_back_to_the_default(monkeypatch):
    """A resumed run whose checkpoint predates the field must still be symmetric."""
    seen: dict[str, int] = {}

    def _rec(name):
        def _f(*args, **kwargs):
            seen[name] = kwargs["max_steps"]
            raise RuntimeError("stop after recording")

        return _f

    monkeypatch.setattr(asens, "bn_sens", _rec("bngsim"))
    monkeypatch.setattr(asens, "amici_sens", _rec("amici"))
    _run_worker_to_the_engine_calls(monkeypatch, spec_extra={})

    assert seen == {
        "bngsim": asens.DEFAULT_SENS_MAX_STEPS,
        "amici": asens.DEFAULT_SENS_MAX_STEPS,
    }


_MINIMAL_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="1"
               hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="R" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>c</ci><ci>k1</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _run_worker_to_the_engine_calls(monkeypatch, *, spec_extra: dict):
    """Drive ``_worker`` far enough to reach both engine calls, with no real solve.

    Only the AMICI build and the two engine calls are stubbed, so this needs
    neither the gitignored corpus nor a ~15 s C++ compile. The bngsim model load
    is *real*, against the minimal document above — stubbing
    ``bngsim._sbml_loader`` through ``sys.modules`` is not reliable here, because
    ``import bngsim._sbml_loader as ...`` resolves through the parent package's
    attribute once any other test has imported it, and would silently fall
    through to the real loader depending on test order.
    """
    monkeypatch.setattr(asens, "amici_free_parameter_ids", lambda xml: (["k1"], object()))
    monkeypatch.setattr(
        asens, "shared_sensitivity_params", lambda *a, **k: (["k1"], {"k1": "k1"}, 1)
    )

    class _Q:
        def __init__(self):
            self.items = []

        def put(self, r):
            self.items.append(r)

    spec = {
        "key": "M:sens:staggered",
        "model_id": "M",
        "method": "sens/staggered",
        "sens_method": "staggered",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "xml": _MINIMAL_SBML,
        "params": {
            "t_start": 0.0,
            "t_end": 1.0,
            "n_points": 3,
            "rtol": 1e-9,
            "atol": 1e-12,
        },
        "cap": 60.0,
        "param_cap": 20,
        "param_budget": 20_000,
        "config_env": {},
        **spec_extra,
    }
    run._worker(spec, _Q())


# ── The budget rides on the spec, and is recorded on the row ────────────────


class _Job:
    """The attributes ``make_specs`` reads off a manifest job.

    ``model`` is a suite-relative path ``model_path`` resolves; it is never
    opened here, since ``make_specs`` only builds the spec.
    """

    model_id = "M"
    model = "models/M/M.xml"
    params = {"t_start": 0.0, "t_end": 1.0, "n_points": 3}
    overrides: list = []


def test_make_specs_stamps_the_budget_on_every_job():
    """Read from the spec, not from a module constant inside the worker — so the
    number a row was solved under is a property of the job, not of the process."""

    specs, _ = run.make_specs(
        [_Job()],
        ["staggered", "simultaneous"],
        rtol=1e-9,
        atol=1e-12,
        timeout=None,
        param_cap=0,
        param_budget=20_000,
        config_env={},
        max_steps=7777,
    )
    assert specs and all(s["max_steps"] == 7777 for s in specs)


def test_make_specs_defaults_to_the_measured_budget():
    specs, _ = run.make_specs(
        [_Job()],
        ["staggered"],
        rtol=1e-9,
        atol=1e-12,
        timeout=None,
        param_cap=0,
        param_budget=20_000,
        config_env={},
    )
    assert specs[0]["max_steps"] == asens.DEFAULT_SENS_MAX_STEPS


def test_the_default_is_above_both_engines_own_default():
    """10,000 is what each engine picks on its own; the point is to exceed it.

    Pinned as an inequality rather than a literal so raising the budget later
    does not have to touch this test, but lowering it back to a no-op does.
    """
    assert asens.DEFAULT_SENS_MAX_STEPS > 10_000


# ── The AMICI side actually applies it ──────────────────────────────────────


def test_amici_sens_sets_the_budget_on_its_solver():
    """Asserted through a recording double: the real call needs a C++ compile."""
    ss = pytest.importorskip("amici.sim.sundials")
    rec: list = []

    class _RData:
        status = 0
        cpu_time = 1.0
        x = np.zeros((3, 1))
        sx = np.zeros((3, 1, 1))
        state_ids = ["A"]

    class _Solver:
        def set_relative_tolerance(self, v): ...
        def set_absolute_tolerance(self, v): ...
        def set_sensitivity_order(self, v): ...
        def set_sensitivity_method(self, v): ...
        def set_internal_sensitivity_method(self, v): ...

        def set_max_steps(self, v):
            rec.append(int(v))

        def get_linear_solver(self):
            return int(getattr(ss, "LinearSolver_KLU", 2))

    class _Model:
        def get_free_parameter_ids(self):
            return ["k1"]

        def set_parameter_scale(self, v): ...
        def set_parameter_list(self, v): ...
        def create_solver(self):
            return _Solver()

        def set_t0(self, t): ...
        def set_timepoints(self, ts): ...

        def simulate(self, solver=None):
            return _RData()

    zero = dict.fromkeys(
        ("parse_sec", "interpret_sec", "jac_derive_sec", "codegen_sec", "compile_sec"), 0.0
    )
    built = (_Model(), {**zero, "load_sec": 0.0}, True)

    asens.amici_sens(built, 0.0, 1.0, 3, 1e-9, 1e-12, ["k1"], "staggered", max_steps=54321)
    assert rec == [54321], f"solver.set_max_steps was not called with the budget: {rec}"
