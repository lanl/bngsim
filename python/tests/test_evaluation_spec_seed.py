"""An EvaluationSpec carries a seed, and SED-ML's KISAO:0000488 is it (issue #1517).

``EvaluationSpec`` accepts ``method="ssa"`` and is documented to give the same
arrays on any node, but it had no seed field and ``evaluate()`` took none: each
``evaluate()`` of one SSA spec, and of its JSON round trip, drew a fresh seed
from entropy. ``read_sedml`` dropped a document's random seed
(``KISAO:0000488``), ``read_sedml_protocol`` read no algorithm parameter of a
stochastic run, and the writers wrote none, so ``read_omex(...).load_protocol()``
ran a seeded BNGL ``simulate`` on fresh seeds.

A spec now has ``seed``, carried by ``to_dict``/``from_dict``/JSON and SED-ML,
and emitted only when set, so an unseeded spec keeps its bytes. An unseeded
stochastic spec still draws a fresh seed on each evaluation, as
``Simulator.run`` does, and reports it on ``Result.seed``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ConversionError
from bngsim._seed import derived_seeds
from bngsim.convert import (
    ProtocolSpec,
    net_to_omex,
    read_omex,
    read_sedml,
    read_sedml_protocol,
    write_sedml,
    write_sedml_protocol,
)
from bngsim.convert._protocol import Experiment

SEED = 42

# The key set an unseeded spec has always had: adding the field must not move
# any existing spec's content key.
_KEYS_BEFORE = {
    "model_source",
    "model_format",
    "method",
    "t_span",
    "n_points",
    "params",
    "sensitivity_params",
    "sensitivity_ic",
    "sensitivity_method",
    "outputs",
    "rtol",
    "atol",
    "max_steps",
    "model_sha256",
}


@pytest.fixture
def net(reversible_net: Path) -> str:
    return str(reversible_net)


def _spec(net: str, **kw) -> bngsim.EvaluationSpec:
    base = {
        "model_source": net,
        "model_format": "net",
        "method": "ssa",
        "t_span": (0.0, 10.0),
        "n_points": 6,
    }
    base.update(kw)
    return bngsim.EvaluationSpec(**base)


def _run(net: str, seed: int) -> bngsim.Result:
    model = bngsim.Model.from_net(net)
    return bngsim.Simulator(model, method="ssa").run(t_span=(0.0, 10.0), n_points=6, seed=seed)


# ── The spec ─────────────────────────────────────────────────────────────────


def test_a_seeded_spec_is_one_trajectory_everywhere(net):
    """Two evaluations, and one through JSON as a worker gets it, give the run
    ``Simulator.run(seed=42)`` gives."""
    spec = _spec(net, seed=SEED)
    a = spec.evaluate()
    b = bngsim.EvaluationSpec.from_json(spec.to_json()).evaluate()
    ref = _run(net, SEED)
    assert a.seed == b.seed == SEED
    np.testing.assert_array_equal(a.species, b.species)
    np.testing.assert_array_equal(a.species, ref.species)


def test_the_seed_is_in_the_wire_format(net):
    spec = _spec(net, seed=SEED)
    assert spec.to_dict()["seed"] == SEED
    back = bngsim.EvaluationSpec.from_dict(json.loads(spec.to_json()))
    assert back == spec


def test_two_seeds_are_two_content_keys(net):
    assert _spec(net, seed=1).to_json() != _spec(net, seed=2).to_json()


def test_an_unseeded_spec_keeps_its_bytes(net):
    """Control: passes on main. ``seed`` is emitted only when set."""
    for method in ("ode", "ssa"):
        assert set(_spec(net, method=method).to_dict()) == _KEYS_BEFORE


def test_an_unseeded_stochastic_spec_reports_the_seed_it_drew(net):
    """It draws a fresh seed, as ``Simulator.run`` does; the seed it reports
    reruns its trajectory."""
    spec = _spec(net)
    first = spec.evaluate()
    again = replace(spec, seed=first.seed).evaluate()
    np.testing.assert_array_equal(first.species, again.species)


@pytest.mark.parametrize("bad", [-1, 2**64, 1.5, True, "7"])
def test_a_seed_that_is_no_seed_is_refused(net, bad):
    """By name. (Main refused every ``seed=`` with a ``TypeError``.)"""
    with pytest.raises(ValueError, match="seed must be an integer"):
        _spec(net, seed=bad)


def test_a_64_bit_seed_is_kept_exactly(net):
    """Control on the bound: 2**64 - 1 is a seed."""
    assert _spec(net, seed=2**64 - 1).seed == 2**64 - 1


def test_the_two_legs_of_an_output_start_spec_do_not_share_a_stream(net):
    """The leg up to the output start runs at the seed, the output leg at the
    second seed drawn from it, so the output leg is not a replay of the first."""
    spec = _spec(net, seed=SEED, t_span=(0.0, 10.0), output_start=5.0)
    a, b = spec.evaluate(), spec.evaluate()
    np.testing.assert_array_equal(a.species, b.species)
    assert a.seed == derived_seeds(SEED, 2)[1] != SEED


def test_an_ode_spec_reads_the_seed_too(net):
    """An event-free ODE run is deterministic with or without one, and the
    spec passes it (as the event tie-break seed, GH #242) without changing it."""
    plain = _spec(net, method="ode").evaluate()
    seeded = _spec(net, method="ode", seed=SEED).evaluate()
    np.testing.assert_array_equal(plain.species, seeded.species)


# ── SED-ML ───────────────────────────────────────────────────────────────────


def _sedml(
    seed_value: str | None, kisao: str = "KISAO:0000029", seed_id: str = "KISAO:0000488"
) -> str:
    params = (
        ""
        if seed_value is None
        else f'<listOfAlgorithmParameters><algorithmParameter kisaoID="{seed_id}" '
        f'value="{seed_value}"/></listOfAlgorithmParameters>'
    )
    return (
        '<sedML xmlns="http://sed-ml.org/sed-ml/level1/version3" level="1" version="3">'
        '<listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0" '
        f'outputEndTime="10" numberOfPoints="5"><algorithm kisaoID="{kisao}">{params}</algorithm>'
        "</uniformTimeCourse></listOfSimulations>"
        '<listOfModels><model id="m" language="urn:sedml:language:sbml" source="m.xml"/>'
        "</listOfModels>"
        '<listOfTasks><task id="t" modelReference="m" simulationReference="s"/></listOfTasks>'
        "</sedML>"
    )


def test_read_sedml_takes_the_document_s_seed(net):
    """The issue's document: it ran on a fresh seed, not 42."""
    spec = read_sedml(_sedml("42"), model_source=net, model_format="net")
    assert spec.seed == SEED
    np.testing.assert_array_equal(spec.evaluate().species, _run(net, SEED).species)


@pytest.mark.parametrize(("value", "seed"), [("42.0", 42), ("  7 ", 7)])
def test_an_integral_seed_reads_as_written(net, value, seed):
    assert read_sedml(_sedml(value), model_source=net, model_format="net").seed == seed


def test_the_underscore_spelling_of_the_term_is_read(net):
    """``KISAO_0000488``, as some BioModels documents write a term."""
    spec = read_sedml(_sedml("42", seed_id="KISAO_0000488"), model_source=net, model_format="net")
    assert spec.seed == SEED


@pytest.mark.parametrize("value", ["4.5", "-1", "abc", str(2**64)])
def test_a_document_seed_that_is_no_seed_is_refused(net, value):
    with pytest.raises(ConversionError, match="random seed"):
        read_sedml(_sedml(value), model_source=net, model_format="net")


def test_a_document_without_a_seed_reads_none(net):
    """Control: passes on main."""
    assert (
        getattr(read_sedml(_sedml(None), model_source=net, model_format="net"), "seed", None)
        is None
    )


def test_write_sedml_round_trips_a_seed_past_2_to_the_53(net):
    """Read as an int: 2**60 + 1 does not survive a float."""
    spec = _spec(net, seed=2**60 + 1)
    text = write_sedml(spec, model_source=net)
    assert 'kisaoID="KISAO:0000488"' in text
    assert read_sedml(text, model_source=net, model_format="net").seed == 2**60 + 1


def test_write_sedml_writes_no_seed_for_an_unseeded_spec(net):
    """Control: passes on main."""
    assert "KISAO:0000488" not in write_sedml(_spec(net), model_source=net)


# ── The protocol route and OMEX ──────────────────────────────────────────────


def test_read_sedml_protocol_takes_a_stochastic_run_s_seed():
    """A document with no bngsim annotation, as another tool writes one: its
    seed reaches the experiment, where ``run_protocol`` reads it."""
    proto = read_sedml_protocol(_sedml("42"))
    [exp] = [s for s in proto.steps if isinstance(s, Experiment)]
    assert exp.method == "ssa" and exp.extra.get("seed") == SEED


def test_write_sedml_protocol_writes_a_stochastic_run_s_seed():
    proto = ProtocolSpec(
        steps=[
            Experiment(
                kind="simulate", method="ssa", t_span=(0.0, 10.0), n_points=6, extra={"seed": 42}
            )
        ]
    )
    assert 'kisaoID="KISAO:0000488" value="42"' in write_sedml_protocol(proto)


def test_a_seed_the_protocol_cannot_name_is_said(recwarn):
    """A symbolic seed has no number to write; the annotation keeps it."""
    proto = ProtocolSpec(
        steps=[
            Experiment(
                kind="simulate", method="ssa", t_span=(0.0, 10.0), n_points=6, extra={"seed": "s0"}
            )
        ]
    )
    with pytest.warns(bngsim.ConversionWarning, match="seed=>s0"):
        text = write_sedml_protocol(proto)
    assert "KISAO:0000488" not in text


def test_an_omex_protocol_keeps_its_seed(net, tmp_path):
    """``read_omex(...).load_protocol()`` reads the standard SED-ML, which held
    no seed."""
    omex = tmp_path / "m.omex"
    net_to_omex(net, omex, protocol=_spec(net, seed=SEED))
    spec = read_omex(omex, extract_dir=tmp_path / "x").load_protocol()
    assert spec.seed == SEED
