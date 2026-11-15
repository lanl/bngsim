"""A protocol that runs with numpy values can be written (issue #1750).

A step may hold numpy scalars: a seed from ``rng.integers()``, a count, a
sweep's values from ``np.linspace``. ``run_protocol`` runs them, but
``ProtocolSpec.to_json`` handed them to ``json.dumps``, which raised
``TypeError: Object of type int64 is not JSON serializable``, and so did
``write_sedml_protocol``, which embeds that JSON in its annotation.
``EvaluationSpec.to_json`` already wrote a numpy seed as the int it is.

Oracle: the same protocol with Python numbers in place of the numpy ones.
Written, it must give the same bytes; read back, the same protocol.
"""

from __future__ import annotations

import json
import re
import warnings

import bngsim
import numpy as np
import pytest
from bngsim.convert import (
    Experiment,
    ProtocolSpec,
    StateChange,
    write_bngl_protocol,
    write_sedml_protocol,
)

NET = """begin parameters
    1 kp 200.0
    2 kd 0.5
end parameters
begin species
    1 A() 0
end species
begin reactions
    1 0 1 kp
    2 1 0 kd
end reactions
"""


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "m.net"
    path.write_text(NET)
    return bngsim.Model.from_net(str(path))


def _run(seed=42, n_points=11) -> Experiment:
    return Experiment(
        kind="simulate",
        method="ssa",
        t_span=(0.0, 10.0),
        n_points=n_points,
        extra={"seed": seed},
    )


def _spec(*steps) -> ProtocolSpec:
    return ProtocolSpec(steps=tuple(steps), source=None)


# Each case: (the step with numpy values, the same step with Python values).
CASES = {
    "an int64 seed (the issue)": (_run(seed=np.int64(42)), _run(seed=42)),
    "an int64 n_points": (_run(n_points=np.int64(11)), _run(n_points=11)),
    "an int32 n_points": (_run(n_points=np.int32(11)), _run(n_points=11)),
    "a uint64 seed at its top": (
        _run(seed=np.uint64(2**64 - 1)),
        _run(seed=2**64 - 1),
    ),
    "float64 times": (
        Experiment(
            kind="simulate",
            method="ode",
            t_span=(np.float64(0.0), np.float64(2.5)),
            n_points=3,
        ),
        Experiment(kind="simulate", method="ode", t_span=(0.0, 2.5), n_points=3),
    ),
    "a scan's int64 points and float32 bounds": (
        Experiment(
            kind="scan",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            scan_parameter="kd",
            scan_min=np.float32(0.1),
            scan_max=np.float32(1.0),
            scan_points=np.int64(3),
        ),
        Experiment(
            kind="scan",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            scan_parameter="kd",
            scan_min=float(np.float32(0.1)),
            scan_max=1.0,
            scan_points=3,
        ),
    ),
    "a sweep from np.linspace, and a numpy bool": (
        Experiment(
            kind="scan",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            scan_parameter="kp",
            scan_values=tuple(np.linspace(1.0, 3.0, 3)),
            reset_between=np.bool_(False),
        ),
        Experiment(
            kind="scan",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            scan_parameter="kp",
            scan_values=(1.0, 2.0, 3.0),
            reset_between=False,
        ),
    ),
    "a numpy array inside extra": (
        Experiment(
            kind="simulate",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            extra={"seed": 7, "marks": np.array([1, 2])},
        ),
        Experiment(
            kind="simulate",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            extra={"seed": 7, "marks": [1, 2]},
        ),
    ),
    "numpy ints in a list inside extra": (
        Experiment(
            kind="simulate",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            extra={"seed": 7, "marks": [np.int64(1), np.int64(2)]},
        ),
        Experiment(
            kind="simulate",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            extra={"seed": 7, "marks": [1, 2]},
        ),
    ),
    "numpy ints in a tuple inside extra": (
        Experiment(
            kind="simulate",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            extra={"seed": 7, "pair": (np.int64(1), np.int64(2))},
        ),
        Experiment(
            kind="simulate",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=2,
            extra={"seed": 7, "pair": (1, 2)},
        ),
    ),
    "a set_parameter of a float64": (
        StateChange(kind="set_parameter", target="kd", value=np.float64(0.25)),
        StateChange(kind="set_parameter", target="kd", value=0.25),
    ),
    "a set_concentration of an int64": (
        StateChange(kind="set_concentration", target="A()", value=np.int64(3)),
        StateChange(kind="set_concentration", target="A()", value=3),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_to_json_writes_the_python_values_bytes(case):
    with_numpy, plain = CASES[case]
    assert _spec(with_numpy).to_json() == _spec(plain).to_json()


@pytest.mark.parametrize("case", list(CASES))
def test_to_dict_holds_python_values_only(case):
    def walk(v):
        assert not isinstance(v, (np.generic, np.ndarray)), type(v)
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(_spec(CASES[case][0]).to_dict())


@pytest.mark.parametrize("case", list(CASES))
def test_it_reads_back_as_the_python_values_protocol(case):
    with_numpy, plain = CASES[case]
    assert ProtocolSpec.from_json(_spec(with_numpy).to_json()) == ProtocolSpec.from_json(
        _spec(plain).to_json()
    )


def test_the_issues_protocol_is_written_and_runs_the_same_read_back(model, tmp_path):
    spec = _spec(_run(seed=np.int64(42)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the no-SBML-target warning is beside the point
        (ran,) = bngsim.run_protocol(model, spec)
        xml = write_sedml_protocol(spec, tmp_path / "p.sedml", model=model)
        (again,) = bngsim.run_protocol(model, ProtocolSpec.from_json(spec.to_json()))
    assert np.array_equal(ran.result.species, again.result.species)
    payload = re.search(r"<(?:\w+:)?protocol[^>]*>(.*?)</(?:\w+:)?protocol>", xml, re.S)
    assert payload is not None
    seed = json.loads(payload.group(1).replace("&quot;", '"'))["steps"][0]["extra"]["seed"]
    assert seed == 42 and type(seed) is int


@pytest.mark.parametrize("case", list(CASES))
def test_each_writer_writes_the_python_values_text(case, model):
    # Main quoted an int64 seed in the actions block (seed=>"42") and wrote a
    # numpy bool as "True"; under numpy 2 a repr would read "np.int64(1)".
    # A run after it: the SED-ML writer refuses a protocol with none.
    with_numpy, plain = CASES[case]
    a, b = _spec(with_numpy, _run(seed=1)), _spec(plain, _run(seed=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert write_bngl_protocol(a) == write_bngl_protocol(b)
        assert write_sedml_protocol(a, model=model) == write_sedml_protocol(b, model=model)
        assert not re.search(r"\bnp\.\w+\(", write_bngl_protocol(a))


@pytest.mark.parametrize("seed", [2**53 + 1, 2**64 - 1, np.uint64(2**64 - 1)])
def test_a_seed_past_a_doubles_integers_is_written_exactly(seed):
    # Through float(), 2**64 - 1 was written as 2**64, out of a seed's range.
    from bngsim.convert import parse_bngl_protocol

    text = write_bngl_protocol(_spec(_run(seed=seed)))
    assert re.search(rf"seed=>{int(seed)}[,}}]", text)
    back = parse_bngl_protocol("begin model\nend model\ngenerate_network()\n" + text)
    assert back.experiments[0].extra["seed"] == int(seed)


def test_a_scan_with_float32_bounds_runs_the_doubles_points(model):
    # Held as float32, the bounds made float32 points: 0.1 + k*delta in float32.
    with_numpy, plain = CASES["a scan's int64 points and float32 bounds"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (a,) = bngsim.run_protocol(model, _spec(with_numpy))
        (b,) = bngsim.run_protocol(model, _spec(plain))
    assert a.scan_values == b.scan_values
    assert all(type(v) is float for v in a.scan_values)
    for x, y in zip(a.scan_results, b.scan_results, strict=True):
        assert np.array_equal(x.species, y.species)
