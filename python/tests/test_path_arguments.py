"""``Simulator(xml_path=...)`` and ``EvaluationSpec(model_source=...)`` take the
:class:`pathlib.Path` that :func:`bngsim.examples.path` hands out (issue #1924).

``Simulator(model, method="nf" / "rm", xml_path=Path)`` passed the ``Path``
straight to the network-free core, which raised a raw pybind11 ``TypeError``
naming a class the caller never called. ``EvaluationSpec`` accepted a ``Path``
and failed only later, at ``to_json()``. Oracle: the same call with the
``str``, which both always took; a spec built from the ``Path`` must be the spec
built from the ``str`` (equal, and the same JSON).
"""

from __future__ import annotations

import json
import pathlib

import bngsim
import pytest
from bngsim import EvaluationSpec, UsageError

XML = bngsim.examples.path("binding", "bng-xml")
NET = bngsim.examples.path("binding", "net")
RUN = {"t_span": (0.0, 10.0), "n_points": 3, "seed": 7}


@pytest.fixture
def model():
    return bngsim.examples.load("binding")


def _has(method: str) -> bool:
    from bngsim import _bngsim_core as core

    return bool(getattr(core, "HAS_NFSIM" if method == "nf" else "HAS_RULEMONKEY", False))


@pytest.mark.parametrize("method", ["nf", "rm"])
def test_a_network_free_simulator_takes_the_path_examples_hands_out(model, method):
    if not _has(method):
        pytest.skip(("NFsim" if method == "nf" else "RuleMonkey") + " not compiled in")
    assert isinstance(XML, pathlib.Path)
    from_path = bngsim.Simulator(model, method=method, xml_path=XML).run(**RUN)
    from_str = bngsim.Simulator(model, method=method, xml_path=str(XML)).run(**RUN)
    assert from_path.observable_names == from_str.observable_names
    for name in from_str.observable_names:
        assert list(from_path.observables[name]) == list(from_str.observables[name])


@pytest.mark.parametrize("method", ["nf", "rm"])
@pytest.mark.parametrize("bad", [b"m.xml", 3])
def test_an_xml_path_that_is_no_path_is_refused_by_name(model, method, bad):
    if not _has(method):
        pytest.skip(("NFsim" if method == "nf" else "RuleMonkey") + " not compiled in")
    with pytest.raises(UsageError, match="xml_path must be a str or os.PathLike") as err:
        bngsim.Simulator(model, method=method, xml_path=bad)
    assert err.value.code == "usage.invalid_argument"


@pytest.mark.parametrize("method", ["nf", "rm"])
@pytest.mark.parametrize("missing", ["", None])
def test_no_xml_path_is_still_the_missing_argument(model, method, missing):
    if not _has(method):
        pytest.skip(("NFsim" if method == "nf" else "RuleMonkey") + " not compiled in")
    with pytest.raises(UsageError, match="requires xml_path") as err:
        bngsim.Simulator(model, method=method, xml_path=missing)
    assert err.value.code == "usage.missing_argument"


def test_an_ode_simulator_still_ignores_an_empty_xml_path(model):
    bngsim.Simulator(model, method="ode", xml_path=None)
    bngsim.Simulator(model, method="ode", xml_path="")
    bngsim.Simulator(model, method="ode")


def test_a_spec_from_a_path_is_the_spec_from_its_str():
    from_path = EvaluationSpec(model_source=NET, model_format="net")
    from_str = EvaluationSpec(model_source=str(NET), model_format="net")
    assert isinstance(from_path.model_source, str)
    assert from_path == from_str
    assert from_path.to_json() == from_str.to_json()
    assert json.loads(from_path.to_json())["model_source"] == str(NET)
    assert EvaluationSpec.from_json(from_path.to_json()) == from_path


def test_a_spec_from_a_path_evaluates():
    spec = EvaluationSpec(model_source=NET, model_format="net", t_span=(0.0, 1.0), n_points=3)
    assert spec.evaluate().species.shape[0] == 3


def test_a_path_for_inline_model_text_is_refused():
    with pytest.raises(UsageError, match="carries the model text") as err:
        EvaluationSpec(model_source=pathlib.Path("m.xml"), model_format="sbml_string")
    assert err.value.code == "usage.invalid_argument"


@pytest.mark.parametrize("bad", [b"m.net", 3, None])
def test_a_model_source_that_is_no_str_is_refused_when_built(bad):
    with pytest.raises(UsageError, match="model_source must be a str") as err:
        EvaluationSpec(model_source=bad)
    assert err.value.code == "usage.invalid_argument"
