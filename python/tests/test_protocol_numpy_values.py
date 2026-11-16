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
import math
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


# ── an array given as a field; what a step keeps as it was given ────────────


def _scan(**kw) -> Experiment:
    base = dict(kind="scan", method="ode", t_span=(0.0, 1.0), n_points=2, scan_parameter="kp")
    return Experiment(**{**base, **kw})


def _ode(**kw) -> Experiment:
    return Experiment(
        **{**dict(kind="simulate", method="ode", t_span=(0.0, 1.0), n_points=2), **kw}
    )


# Each: (a step given an array for a tuple field, the step given the tuple).
ARRAY_FIELDS = {
    "scan_values": (
        _scan(scan_values=np.linspace(1.0, 3.0, 3)),
        _scan(scan_values=(1.0, 2.0, 3.0)),
    ),
    "t_span": (_ode(t_span=np.array([0.0, 1.0])), _ode()),
    "sample_times": (
        _ode(sample_times=np.array([0.0, 0.5, 1.0], dtype=np.float32)),
        _ode(sample_times=(0.0, 0.5, 1.0)),
    ),
}


@pytest.mark.parametrize("name", list(ARRAY_FIELDS))
def test_an_array_given_as_a_field_is_the_tuple_the_field_holds(name):
    # tolist() alone left a list: the step did not equal the one given the
    # tuple, nor the one its own JSON reads back as, and did not hash.
    with_array, plain = ARRAY_FIELDS[name]
    held = getattr(with_array, name)
    assert type(held) is tuple and all(type(v) is float for v in held)
    assert with_array == plain
    assert hash(getattr(with_array, name)) == hash(getattr(plain, name))
    spec = _spec(with_array)
    assert ProtocolSpec.from_json(spec.to_json()) == spec
    assert spec.to_json() == _spec(plain).to_json()


@pytest.mark.parametrize("case", list(CASES))
def test_a_step_given_numpy_values_equals_the_python_valued_step(case):
    with_numpy, plain = CASES[case]
    assert with_numpy == plain


def test_a_step_given_no_numpy_value_holds_what_it_was_given():
    # Every mapping, list and tuple was rebuilt, numpy in it or not: `extra`
    # became a copy (main held the caller's dict) and a plain dict whatever
    # mapping it was given.
    import types
    from collections import OrderedDict

    given = {"seed": 1, "marks": [1, 2], "pair": (1, 2)}
    step = _ode(extra=given)
    assert step.extra is given and step.extra["marks"] is given["marks"]
    given["seed"] = 2
    assert step.extra["seed"] == 2
    frozen = types.MappingProxyType({"seed": 1})
    assert _ode(extra=frozen).extra is frozen
    assert type(_ode(extra=OrderedDict(seed=1)).extra) is OrderedDict
    span = (0.0, 1.0)
    assert _ode(t_span=span).t_span is span
    steps = (_ode(),)
    assert ProtocolSpec(steps=steps).steps is steps


@pytest.mark.parametrize(
    "given, plain",
    [
        ({"marks": np.array([np.int64(1), 2], dtype=object)}, {"marks": [1, 2]}),
        ({"by_run": {np.int64(1): 2}}, {"by_run": {1: 2}}),
        ({"tol": np.longdouble(1.5)}, {"tol": 1.5}),
        ({"tols": np.array([1.5, 2.5], dtype=np.longdouble)}, {"tols": [1.5, 2.5]}),
    ],
    ids=["an object array", "a numpy key", "a longdouble", "a longdouble array"],
)
def test_numpy_values_that_item_and_tolist_leave_numpy_are_written(given, plain):
    # item() of a longdouble is the longdouble, tolist() of an object array
    # keeps its elements, and a mapping's keys were not looked at.
    with_numpy, twin = _ode(extra=given), _ode(extra=plain)
    assert with_numpy == twin
    assert _spec(with_numpy).to_json() == _spec(twin).to_json()


@pytest.mark.parametrize(
    "field", ["generation_step", "lossy_before_generation", "dropped_before_generation"]
)
def test_a_protocols_own_numpy_counts_are_written(field, model):
    # The protocol's fields go to the same json.dumps as its steps'.
    with_numpy = ProtocolSpec(steps=(_run(),), **{field: np.int64(1)})
    plain = ProtocolSpec(steps=(_run(),), **{field: 1})
    assert type(getattr(with_numpy, field)) is int
    assert with_numpy == plain and with_numpy.to_json() == plain.to_json()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert write_sedml_protocol(with_numpy, model=model) == write_sedml_protocol(
            plain, model=model
        )


# ── integers in the actions block: exact, and never a new error ─────────────

_HEAD = "begin model\nend model\ngenerate_network()\n"
_GOOGOL4 = "1" + "0" * 400  # an integer past a double's range
_LONG = "9" * 5000  # past Python's 4300-digit limit for int()


@pytest.mark.parametrize("digits", [_GOOGOL4, _LONG], ids=["1e400", "5000 digits"])
def test_an_integer_past_a_doubles_range_reads_as_it_did(digits):
    # Read as the int it spells, it raised OverflowError at the next float()
    # (t_end, a set's value, sample_times, par_max), and past 4300 digits int()
    # itself raised ValueError. Main read each as inf; so does this.
    from bngsim.convert import parse_bngl_protocol

    def read(actions):
        return parse_bngl_protocol(_HEAD + actions + "\n")

    run = read(f'simulate({{method=>"ode",t_end=>{digits},n_steps=>10}})').experiments[0]
    assert run.t_span == (0.0, math.inf)
    (change,) = read(f'setParameter("k",{digits})').steps
    assert change.value == math.inf
    seeded = read(f'simulate({{method=>"ssa",t_end=>1,n_steps=>1,seed=>{digits}}})')
    assert seeded.experiments[0].extra["seed"] == math.inf
    timed = read(f'simulate({{method=>"ode",sample_times=>[0,1,{digits}]}})').experiments[0]
    assert timed.sample_times == (0.0, 1.0, math.inf)
    scan = read(
        'parameter_scan({method=>"ode",parameter=>"k",par_min=>1,'
        f"par_max=>{digits},n_scan_pts=>3,t_end=>1,n_steps=>1}})"
    ).experiments[0]
    assert scan.scan_max == math.inf
    negative = read(f'setParameter("k",-{digits})').steps[0]
    assert negative.value == -math.inf


@pytest.mark.parametrize(
    "token, value",
    [
        ("0" * 5000 + "7", 7),  # finite, but too many digits for int()
        ("9" * 308, int("9" * 308)),  # the longest run of digits a double holds
        ("-18446744073709551615", -(2**64 - 1)),
        ("+9007199254740993", 2**53 + 1),
    ],
    ids=["leading zeros", "308 digits", "negative", "signed"],
)
def test_an_integer_a_double_can_hold_reads_exactly(token, value):
    from bngsim.convert import parse_bngl_protocol

    spec = parse_bngl_protocol(
        _HEAD + f'simulate({{method=>"ssa",t_end=>1,n_steps=>1,seed=>{token}}})\n'
    )
    seed = spec.experiments[0].extra["seed"]
    assert seed == value and type(seed) is int
    float(seed)  # what every reader of an action's number does next
