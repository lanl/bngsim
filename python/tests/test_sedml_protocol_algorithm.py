"""Issue #1500 — a protocol's SED-ML names each run's method and tolerances.

``write_sedml_protocol`` (behind ``net_to_omex(bngl=...)`` and
``bngsim-net2sbml --sidecar``) wrote every ``<algorithm>`` bare, so a
simulate's ``atol``/``rtol`` lived only in the bngsim annotation, and filed
every method outside ode/ssa/nf, PLA and PSA included, under Gillespie's direct
method. A standards reader, bngsim's own ``read_omex(...).load_protocol()``
among them, ran another experiment: an ``rtol=>1e-12`` run came back at default
tolerances, 8.8e-7 off the closed form instead of 7e-10, and a PLA run came
back as an exact SSA, with nothing said.

Model: A -> 0 at k = 1, A(0) = 1000, so A(10) = 1000 exp(-10).
"""

from __future__ import annotations

import math
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile

import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim.convert import (
    net_to_omex,
    read_omex,
    read_sedml,
    read_sedml_protocol,
    write_sedml_protocol,
)
from bngsim.convert._protocol import Experiment, ProtocolSpec

NET = """begin parameters
    1 k 1
end parameters
begin species
    1 A() 1000
end species
begin reactions
    1 1 0 k
end reactions
begin groups
    1 Atot 1
end groups
"""
BNGL = """begin model
begin parameters
k 1
end parameters
begin molecule types
A()
end molecule types
begin seed species
A() 1000
end seed species
begin observables
Molecules Atot A()
end observables
begin reaction rules
A() -> 0 k
end reaction rules
end model
generate_network({overwrite=>1})
"""
CLOSED = 1000 * math.exp(-10)


def _algorithms(xml: str) -> list[tuple[str, dict[str, str]]]:
    """Each uniformTimeCourse's (kisaoID, {parameter kisaoID: value})."""
    out = []
    for el in ET.fromstring(xml).iter():
        if el.tag.rsplit("}", 1)[-1] == "algorithm":
            params = {
                p.get("kisaoID"): p.get("value")
                for p in el.iter()
                if p.tag.rsplit("}", 1)[-1] == "algorithmParameter"
            }
            out.append((el.get("kisaoID"), params))
    return out


def _without_annotation(xml: str) -> str:
    return re.sub(r"<annotation>.*?</annotation>", "", xml, flags=re.S)


def _protocol(method: str, **extra) -> ProtocolSpec:
    return ProtocolSpec(
        steps=(
            Experiment(
                kind="simulate", method=method, t_span=(0.0, 10.0), n_points=11, extra=extra
            ),
        )
    )


def test_an_omex_of_a_tight_ode_run_reloads_at_its_tolerances(tmp_path):
    (tmp_path / "m.net").write_text(NET)
    action = 'simulate({method=>"ode",t_end=>10,n_steps=>10,atol=>1e-14,rtol=>1e-12})'
    (tmp_path / "m.bngl").write_text(BNGL + action + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        net_to_omex(tmp_path / "m.net", tmp_path / "m.omex", bngl=tmp_path / "m.bngl", gate="none")
    with zipfile.ZipFile(tmp_path / "m.omex") as zf:
        sedml = next(zf.read(n).decode() for n in zf.namelist() if n.endswith(".sedml"))
    assert _algorithms(sedml) == [
        ("KISAO:0000019", {"KISAO:0000209": "1e-12", "KISAO:0000211": "1e-14"})
    ]
    with read_omex(tmp_path / "m.omex") as archive:
        spec = archive.load_protocol()
        assert (spec.method, spec.rtol, spec.atol) == ("ode", 1e-12, 1e-14)
        a10 = float(spec.evaluate().observables["A"][-1])
    # At the Simulator's defaults this was 8.8e-7 off.
    assert abs(a10 - CLOSED) / CLOSED < 1e-8


def test_an_ode_run_without_tolerances_carries_run_networks():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        text = write_sedml_protocol(_protocol("ode"))
    assert _algorithms(text) == [
        ("KISAO:0000019", {"KISAO:0000209": "1e-08", "KISAO:0000211": "1e-08"})
    ]


@pytest.mark.parametrize(
    ("method", "kisao"),
    [("ssa", "KISAO:0000029"), ("nf", "KISAO:0000263"), ("pla", "KISAO:0000524")],
)
def test_each_method_gets_its_own_term(method, kisao):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        text = write_sedml_protocol(_protocol(method))
    # Tolerances are an ODE run's; a stochastic one has none to carry.
    assert _algorithms(text) == [(kisao, {})]


@pytest.mark.parametrize(
    "protocol", [_protocol("psa", poplevel=50), _protocol("ssa", poplevel=50)]
)
def test_psa_has_no_term_and_says_so(protocol):
    with pytest.warns(ConversionWarning, match="runs PSA .* has no KiSAO term"):
        text = write_sedml_protocol(protocol)
    assert _algorithms(text) == [("KISAO:0000029", {})]


@pytest.mark.parametrize(
    ("protocol", "said"),
    [
        (_protocol("pla", pla_config="fEuler|pre-neg:sb|eps=0.03"), "pla_config=>"),
        (_protocol("ode", steady_state=1), "steady_state=>1"),
        (_protocol("ode", rtol="tol"), "rtol=>tol, which cannot be read without the model"),
    ],
)
def test_what_the_algorithm_cannot_hold_is_named(protocol, said):
    with pytest.warns(ConversionWarning, match=re.escape(said)):
        text = write_sedml_protocol(protocol)
    if "rtol" in said:
        # The readable tolerance is still written; the other is not guessed.
        assert _algorithms(text) == [("KISAO:0000019", {"KISAO:0000211": "1e-08"})]


def test_the_standard_elements_alone_rebuild_the_run():
    protocol = _protocol("ode", atol=1e-14, rtol=1e-12)
    (exp,) = read_sedml_protocol(_without_annotation(write_sedml_protocol(protocol))).experiments
    assert exp.method == "ode"
    assert exp.extra == {"rtol": 1e-12, "atol": 1e-14}
    for method in ("nf", "pla"):
        bare = _without_annotation(write_sedml_protocol(_protocol(method)))
        assert read_sedml_protocol(bare).experiments[0].method == method


def test_read_sedml_refuses_pla_and_runs_nfsim_as_the_exact_ssa():
    with pytest.raises(ConversionError, match="no PLA method"):
        read_sedml(write_sedml_protocol(_protocol("pla")))
    assert read_sedml(write_sedml_protocol(_protocol("nf"))).method == "ssa"


def _foreign(kisao: str) -> str:
    return f"""<sedML xmlns="http://sed-ml.org/sed-ml/level1/version4" level="1" version="4">
 <listOfModels><model id="m" language="urn:sedml:language:sbml" source="m.xml"/></listOfModels>
 <listOfSimulations><uniformTimeCourse id="s" initialTime="0" outputStartTime="0"
   outputEndTime="10" numberOfSteps="10"><algorithm kisaoID="{kisao}"/></uniformTimeCourse>
 </listOfSimulations>
 <listOfTasks><task id="t" modelReference="m" simulationReference="s"/></listOfTasks>
</sedML>"""


@pytest.mark.parametrize("kisao", ["KISAO:0000694", "KISAO_0000694", "KISAO_0000019"])
def test_the_ode_solver_terms_biomodels_writes_read_as_ode(kisao):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        assert read_sedml(_foreign(kisao)).method == "ode"
        assert read_sedml_protocol(_foreign(kisao)).experiments[0].method == "ode"


def test_an_unknown_term_is_named_by_both_readers():
    # A tau-leaping term the protocol reader turned into method=>"ode" unsaid.
    with pytest.warns(ConversionWarning, match="unmapped algorithm KiSAO term 'KISAO:0000039'"):
        read_sedml(_foreign("KISAO:0000039"))
    with pytest.warns(ConversionWarning, match="unmapped algorithm KiSAO term 'KISAO:0000039'"):
        assert read_sedml_protocol(_foreign("KISAO:0000039")).experiments[0].method == "ode"
