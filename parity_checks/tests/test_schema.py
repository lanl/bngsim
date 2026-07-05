"""Lock the _core report schema's forward/backward compatibility (T6.1).

JobResult.from_dict must tolerate a report written by an older suite version —
in particular one that still carries the retired per-engine wall_bn/wall_rr
fields (dropped once the per-phase ``timing`` dict superseded them). A plain
``cls(**d)`` would raise TypeError on those stale keys; from_dict drops unknown
keys instead, so old reports still load.
"""

from __future__ import annotations

import dataclasses

from _core import JobResult


def test_from_dict_ignores_retired_wall_fields():
    # An old-shape record: valid fields plus the retired wall_bn/wall_rr.
    old = {
        "model_id": "BIOMD0000000003",
        "method": "ode",
        "reference_engine": "roadrunner",
        "outcome": "PASS",
        "metric": "max_rel_err",
        "value": 0.0,
        "wall_bn": 0.012,  # retired
        "wall_rr": 0.034,  # retired
    }
    r = JobResult.from_dict(old)  # must not raise
    assert r.model_id == "BIOMD0000000003"
    assert r.outcome == "PASS"
    assert r.value == 0.0
    # The retired keys are dropped, not smuggled onto the instance.
    assert not hasattr(r, "wall_bn")
    assert not hasattr(r, "wall_rr")


def test_from_dict_drops_arbitrary_unknown_keys():
    r = JobResult.from_dict(
        {
            "model_id": "m",
            "method": "ode",
            "reference_engine": "rr",
            "outcome": "DIFF",
            "some_future_field": 123,
        }
    )
    assert r.outcome == "DIFF"
    assert not hasattr(r, "some_future_field")


def test_to_dict_roundtrips_known_fields():
    r = JobResult(
        model_id="m",
        method="ode",
        reference_engine="roadrunner",
        outcome="PASS",
        wall_sec=0.046,
        timing={"bngsim": {"integrate_sec": 0.01}},
    )
    r2 = JobResult.from_dict(r.to_dict())
    assert r2 == r
    # wall_bn/wall_rr are gone from the dataclass entirely.
    names = {f.name for f in dataclasses.fields(JobResult)}
    assert "wall_bn" not in names and "wall_rr" not in names
    assert "wall_sec" in names and "timing" in names
