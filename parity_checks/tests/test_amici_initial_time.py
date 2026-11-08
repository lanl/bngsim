"""Both engines of the AMICI ODE job start at the SED-ML initial time (#1564).

On 19 corpus jobs the SED-ML outputStartTime (``t_start``) is later than the
initialTime (``initial_time``): BIOMD0000000042 on [120, 300], 569 on
[1e-5, 1e-2], 834 on [400, 700]. Each engine takes the start it is handed as the
time the model's initial state holds at, so ``amici_run`` handing both of them
``t_start`` compared two runs of the model's initial state placed at the output
start: nothing before it happened, 834's ``H := 1.8`` at ``t >= 200`` included.
rr_run has integrated from ``initial_time`` since GH #19, and copasi_run since
#1550. The sensitivity job's worker is pinned in ``test_amici_reference_archive``.

These run without AMICI: both engines are stand-ins.
"""

from __future__ import annotations

import _amici_common as ac
import amici_run as ar
import numpy as np


def test_the_initial_time_is_the_jobs_initial_time_or_its_start():
    assert ac.initial_time({"initial_time": 0.0, "t_start": 120.0}) == 0.0
    # A job file predating the field integrates from its t_start, as before.
    assert ac.initial_time({"t_start": 120.0}) == 120.0


def test_both_engines_of_the_ode_job_start_at_the_initial_time(monkeypatch):
    t = np.linspace(0.0, 300.0, 11)
    x = np.column_stack([np.exp(-t / 100.0), 1.0 - np.exp(-t / 100.0)])
    seen = {}

    def engine(name):
        def run(xml, t0, t_end, n_points, rtol, atol, **kw):
            seen[name] = t0
            assert (t_end, n_points) == (300.0, 11)
            return np.linspace(t0, t_end, n_points), x, ["A", "B"], {}

        return run

    monkeypatch.setattr(ac, "measure_warmup", lambda: {})
    monkeypatch.setattr(ac, "set_amici_quiet", lambda: None)
    monkeypatch.setattr(ac, "bn_ode", engine("bngsim"))
    monkeypatch.setattr(ac, "amici_ode", engine("amici"))
    spec = {
        "key": "BIOMD0000000042|ode",
        "model_id": "BIOMD0000000042",
        "method": "ode",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "xml": "unused.xml",
        "params": {
            "initial_time": 0.0,
            "t_start": 120.0,
            "t_end": 300.0,
            "n_points": 11,
            "rtol": 1e-9,
            "atol": 1e-12,
        },
    }
    out: list[dict] = []

    class _Queue:
        put = staticmethod(out.append)

    ar._worker(spec, _Queue())
    (res,) = out
    assert res["status"] == "pass"
    assert seen == {"bngsim": 0.0, "amici": 0.0}
