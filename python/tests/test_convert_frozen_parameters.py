"""A parameter folded at load is not written as a free one (issue #1792).

The SBML loader evaluates some expressions once, at load, and the model holds
the number: a stoichiometry written over a parameter, for one. Every parameter
such a number was read from is in ``Model.frozen_params``, and ``set_param``
refuses to change it. ``sbml_to_net`` and ``sbml_to_bngl`` wrote the number and
wrote the parameter as any other, so the model loaded from the file took the
write: with no effect where nothing else read the parameter, and with half an
effect where a rate law still did.

Now a frozen parameter that nothing in the written network reads is left out,
and one that something still reads carries a ``folded_at_load=`` comment that
``Model.from_net`` and ``Model.from_bngl`` read back.

Oracles: closed forms of two small models; for a model of the BioModels
corpus, the SBML model's own list and run; BNG2.pl for what it does with a
written file.

* ``A -> F B`` at rate ``k A``, ``A(0) = 1``, ``B(0) = 0``:
  ``B(t) = F (1 - exp(-k t))``. ``F`` is the stoichiometry, the value of
  ``f`` when the SBML is loaded.
* The same with ``B -> 0`` at rate ``f kd B``, where ``f`` is read live:
  ``dB/dt = F k exp(-k t) - g B`` with ``g = f kd``, so
  ``B(t) = F k (exp(-g t) - exp(-k t)) / (k - g)`` for ``g != k``.
  The model that follows a write to ``f`` in full has ``F = f``; a write that
  moves the rate law alone leaves ``F`` at its load value.
"""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace

import bngsim
import pytest
from bngsim import ParameterError
from bngsim._model import _folded_at_load_in
from bngsim._net_reader import build_model_from_parsed, parse_net_file
from bngsim.convert import _frozen, sbml_to_bngl, sbml_to_net
from bngsim.convert._bng2 import find_bng2

needs_sbml = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML loading requires libsbml")

_FOUND = find_bng2()
BNG2 = Path(_FOUND).resolve() if _FOUND else None
HAS_BNG2 = BNG2 is not None and BNG2.is_file() and bool(shutil.which("perl"))
needs_bng2 = pytest.mark.skipif(not HAS_BNG2, reason="BNG2.pl not found")

K, KD = 1.0, 0.25
REPO = Path(__file__).resolve().parents[2]
BIOMD613 = REPO / "benchmarks" / "sbml_events" / "BIOMD0000000613.xml"
MODEL1509 = REPO / "benchmarks" / "sbml_events" / "MODEL1509020000.xml"
COMPARTMENTAL = REPO / "benchmarks" / "models" / "net" / "ode" / "rec_dim_comp.net"


def _sbml(f: float, *, read: bool) -> str:
    """``A -> f B`` with the stoichiometry of B written over the parameter f;
    with *read*, also ``B -> 0`` at ``f kd B``, which reads f live."""
    decay = """
   <reaction id="R2" reversible="false">
    <listOfReactants><speciesReference species="B"/></listOfReactants>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><times/><ci>f</ci><ci>kd</ci><ci>B</ci></apply></math></kineticLaw>
   </reaction>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
 <model id="stoich">
  <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
  <listOfSpecies>
   <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"/>
   <species id="B" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="{K}"/><parameter id="kd" value="{KD}"/><parameter id="f" value="{f}"/>
  </listOfParameters>
  <listOfReactions>
   <reaction id="R1" reversible="false">
    <listOfReactants><speciesReference species="A"/></listOfReactants>
    <listOfProducts><speciesReference species="B"><stoichiometryMath>
      <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>f</ci></math>
    </stoichiometryMath></speciesReference></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
   </reaction>{decay if read else ""}
  </listOfReactions>
 </model>
</sbml>
"""


def _b_unread(big_f: float, t: float) -> float:
    return big_f * (1.0 - math.exp(-K * t))


def _b_read(big_f: float, f: float, t: float) -> float:
    g = f * KD
    return big_f * K * (math.exp(-g * t) - math.exp(-K * t)) / (K - g)


WRITERS = [
    pytest.param(sbml_to_net, ".net", id="net"),
    pytest.param(sbml_to_bngl, ".bngl", id="bngl", marks=needs_bng2),
]


def _convert(tmp_path: Path, writer, ext: str, f: float, *, read: bool, **options):
    """The SBML at *f* written by *writer*: ``(report, file, warnings)``."""
    source = tmp_path / "s.xml"
    source.write_text(_sbml(f, read=read), encoding="utf-8")
    out = tmp_path / f"s{ext}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = writer(str(source), str(out), **options)
    return report, out, [str(w.message) for w in caught]


def _load(path: Path) -> bngsim.Model:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if path.suffix == ".net":
            return bngsim.Model.from_net(str(path))
        return bngsim.Model.from_bngl(str(path), cache=False)


def _b_at(model: bngsim.Model, t: float) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = bngsim.Simulator(model, method="ode").run(t_span=(0, t), n_points=3)
    names = [n.split("::")[-1].split("(")[0] for n in result.species_names]
    return float(result.species[-1][names.index("B")])


def _parameters(text: str) -> dict[str, str]:
    """The parameters block of a written file: name -> the rest of its line."""
    block = text.split("begin parameters")[1].split("end parameters")[0]
    lines = [line.split(None, 2) for line in block.splitlines() if line.strip()]
    assert [int(index) for index, _, _ in lines] == list(range(1, len(lines) + 1))
    return {name: rest for _, name, rest in lines}


# Tolerance of the runs against a closed form. The solver's default relative
# and absolute tolerances are 1e-8, and the largest error measured over the
# two time units of these runs is 2.1e-7, where the rate law reads f (9.4e-9
# where nothing does; BNG2.pl's row for the same model is within 1.1e-7).
# 1e-6 leaves a factor of about 5 over that and is more than five orders
# below the smallest effect tested (a third of the value).
REL = 1e-6


# --- the source model, as a control -----------------------------------------


@needs_sbml
@pytest.mark.parametrize("read", [False, True])
def test_the_sbml_model_refuses_the_write(tmp_path, read):
    source = tmp_path / "s.xml"
    source.write_text(_sbml(2, read=read), encoding="utf-8")
    model = bngsim.Model.from_sbml(str(source))
    assert model.frozen_params == ["f"]
    with pytest.raises(ParameterError, match="cannot be changed after the model is loaded"):
        model.set_param("f", 3.0)
    want = _b_read(2, 2, 2.0) if read else _b_unread(2, 2.0)
    assert _b_at(model, 2.0) == pytest.approx(want, rel=REL)


# --- nothing reads it: left out ----------------------------------------------


@needs_sbml
@pytest.mark.parametrize(("writer", "ext"), WRITERS)
@pytest.mark.parametrize("f", [1, 2])
def test_a_frozen_parameter_nothing_reads_is_left_out(tmp_path, writer, ext, f):
    report, out, said = _convert(tmp_path, writer, ext, f, read=False)
    parameters = _parameters(out.read_text(encoding="utf-8"))
    assert "f" not in parameters and {"k", "kd"} <= set(parameters)
    # Said once, as something dropped, by the parameter's name.
    (note,) = [n for n in report.dropped if "folded" in n]
    assert "(f)" in note
    assert [w for w in said if "folded" in w] == [f"s.xml: dropping {note}"]
    assert not report.lossy and not report.bngsim_only and report.ok
    # The report names it, and counts the parameters the file has.
    assert (report.folded_left_out, report.folded_marked) == (["f"], [])
    assert report.n_parameters == len(parameters)

    model = _load(out)
    assert "f" not in model.param_names and model.frozen_params == []
    with pytest.raises(ParameterError, match="'f' not found"):
        model.set_param("f", 3.0)
    # The network is the one the SBML gives at the f it was loaded with.
    assert _b_at(model, 2.0) == pytest.approx(_b_unread(f, 2.0), rel=REL)


@needs_sbml
def test_the_round_trip_checks_pass_with_a_parameter_left_out(tmp_path):
    # The L1 counts compare the source's parameters with the reloaded file's,
    # which has one fewer by design.
    report, _, _ = _convert(tmp_path, sbml_to_net, ".net", 2, read=False, validate="L1")
    assert report.structural is not None and report.structural.passed, report.structural
    assert report.structural.n_parameters[0] == report.structural.n_parameters[1]
    assert report.ok
    report, _, _ = _convert(tmp_path, sbml_to_net, ".net", 2, read=False, validate="full")
    assert report.validation is not None and report.ok, report.validation
    # The full gate's L1 records the same count, without gating on it.
    first = report.validation.level("L1")
    assert first is not None and first.status == "pass"
    assert first.metrics["n_parameters"][0] == first.metrics["n_parameters"][1], first.metrics


RATE_OF = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
 <model id="m">
  <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
  <listOfSpecies>
   <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true"
            boundaryCondition="false" constant="false"/>
   <species id="B" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"
            boundaryCondition="false" constant="false"/>
  </listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="1" constant="true"/>
   <parameter id="f" value="2" constant="true"/>
   <parameter id="p" value="0" constant="false"/>
  </listOfParameters>
  <listOfInitialAssignments>
   <initialAssignment symbol="sr">
    <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>f</ci></math>
   </initialAssignment>
  </listOfInitialAssignments>
  <listOfRules>
   <assignmentRule variable="p"><math xmlns="http://www.w3.org/1998/Math/MathML"><apply>
    <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/rateOf">rateOf</csymbol>
    <ci>A</ci></apply></math></assignmentRule>
  </listOfRules>
  <listOfReactions>
   <reaction id="R1" reversible="false">
    <listOfReactants>
     <speciesReference species="A" stoichiometry="1" constant="true"/>
    </listOfReactants>
    <listOfProducts><speciesReference id="sr" species="B" constant="true"/></listOfProducts>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><times/><ci>k</ci><ci>A</ci></apply></math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
"""


@needs_sbml
def test_the_report_of_a_network_that_does_not_load_back_names_the_parameters(tmp_path):
    # A rule over rateOf, which the .net cannot hold: lenient conversion writes
    # a file that does not load back and returns before any round-trip check.
    # That report carries the lists and the count as the other does. Two
    # parameters are frozen here, f and the stoichiometry's own id, and neither
    # is read.
    source = tmp_path / "s.xml"
    source.write_text(RATE_OF, encoding="utf-8")
    out = tmp_path / "s.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = bngsim.Model.from_sbml(str(source))
        report = sbml_to_net(str(source), str(out), strict=False)
    assert loaded.frozen_params == ["f", "sr"] and loaded.n_parameters == 7
    assert report.structural is None and report.rhs_faithful is False and report.lossy
    with pytest.raises(bngsim.ModelError):
        bngsim.Model.from_net(str(out))
    assert (report.folded_left_out, report.folded_marked) == (["f", "sr"], [])
    assert report.n_parameters == 5
    assert not {"f", "sr"} & set(_parameters(out.read_text(encoding="utf-8")))


@needs_sbml
@needs_bng2
def test_bng2pl_refuses_a_write_to_the_parameter_left_out(tmp_path):
    _, out, _ = _convert(tmp_path, sbml_to_bngl, ".bngl", 2, read=False)
    text = out.read_text(encoding="utf-8")

    def run(actions: str) -> subprocess.CompletedProcess:
        (tmp_path / "m.bngl").write_text(text + actions, encoding="utf-8")
        return subprocess.run(
            ["perl", str(BNG2), "m.bngl"],
            cwd=tmp_path,
            env={**os.environ, "BNGPATH": str(BNG2.parent)},
            capture_output=True,
            text=True,
            timeout=300,
        )

    generate = "generate_network({overwrite=>1})\n"
    assert run(generate).returncode == 0  # the file itself is one BNG2.pl takes
    refused = run(generate + 'setParameter("f", 3)\n')
    assert refused.returncode != 0
    assert "Parameter f not defined" in refused.stdout + refused.stderr


# --- something still reads it: marked ----------------------------------------


@needs_sbml
@pytest.mark.parametrize(("writer", "ext"), WRITERS)
def test_a_frozen_parameter_still_read_is_refused_on_the_reloaded_model(tmp_path, writer, ext):
    report, out, said = _convert(tmp_path, writer, ext, 2, read=True)
    parameters = _parameters(out.read_text(encoding="utf-8"))
    assert parameters["f"].split("#")[0].strip() == "2.0"
    assert "folded_at_load=the stoichiometry of 'B' in reaction 'R1'" in parameters["f"]
    assert [n for n in parameters if "folded_at_load" in parameters[n]] == ["f"]
    (note,) = [n for n in report.bngsim_only if "folded" in n]
    assert "(f)" in note and not [n for n in report.dropped if "folded" in n]
    # Warned once, whatever strict is, as what only bngsim reads.
    (warned,) = [w for w in said if "folded" in w]
    assert warned.startswith("s.xml: a ") and "only bngsim reads" in warned
    assert warned.endswith(note)
    assert not report.lossy and report.ok
    assert (report.folded_left_out, report.folded_marked) == ([], ["f"])
    assert report.n_parameters == len(parameters)

    model = _load(out)
    assert model.frozen_params == ["f"]
    with pytest.raises(ParameterError) as refused:
        model.set_param("f", 3.0)
    message = str(refused.value)
    assert "the stoichiometry of 'B' in reaction 'R1'" in message
    assert "the document the file was written from, and write the file again" in message
    assert model.get_param("f") == 2.0
    # The value it holds is no change, and the run is the SBML's at f = 2.
    model.set_param("f", 2.0)
    assert _b_at(model, 2.0) == pytest.approx(_b_read(2, 2, 2.0), rel=REL)
    # A clone refuses too.
    with pytest.raises(ParameterError):
        model.clone().set_param("f", 3.0)


@needs_sbml
def test_what_the_refusal_prevents_and_what_it_points_to(tmp_path):
    # With the comment taken off, the file is what was written before: the
    # write is taken and moves the rate law alone, F = 2 with f = 3. The answer
    # of the model at f = 3 is the SBML's with f = 3, converted again.
    _, out, _ = _convert(tmp_path, sbml_to_net, ".net", 2, read=True)
    bare = tmp_path / "bare.net"
    lines = out.read_text(encoding="utf-8").split("\n")
    bare.write_text(
        "\n".join(line.split("  # folded_at_load=")[0] for line in lines), encoding="utf-8"
    )
    half = _load(bare)
    assert half.frozen_params == []
    half.set_param("f", 3.0)
    half_moved, right = _b_read(2, 3, 2.0), _b_read(3, 3, 2.0)
    assert abs(half_moved - right) > 0.3 * right  # the two answers are far apart
    assert _b_at(half, 2.0) == pytest.approx(half_moved, rel=REL)

    again = tmp_path / "again"
    again.mkdir()
    _, converted, _ = _convert(again, sbml_to_net, ".net", 3, read=True)
    assert _b_at(_load(converted), 2.0) == pytest.approx(right, rel=REL)


@needs_sbml
def test_a_marked_network_written_again_is_marked_once(tmp_path):
    _, out, _ = _convert(tmp_path, sbml_to_net, ".net", 2, read=True)
    first = out.read_text(encoding="utf-8")
    again = tmp_path / "again.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bngsim.convert.write_net(_load(out), str(again))
    second = _parameters(again.read_text(encoding="utf-8"))["f"]
    assert second == _parameters(first)["f"]
    assert second.count("folded_at_load=") == 1 and "comment says" not in second
    assert _load(again).frozen_params == ["f"]


@needs_sbml
def test_a_frozen_parameter_only_an_initial_value_reads_is_marked(tmp_path):
    # B starts at f (an initialAssignment) and f is B's stoichiometry too. The
    # species line names f, so f stays: left out, the network would name a
    # parameter it does not declare. Closed form: B(t) = f + F (1 - exp(-k t)).
    source = tmp_path / "s.xml"
    source.write_text(
        _sbml(2, read=False).replace(
            "</listOfReactions>",
            '</listOfReactions>\n  <listOfInitialAssignments><initialAssignment symbol="B">'
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>f</ci></math>'
            "</initialAssignment></listOfInitialAssignments>",
        ),
        encoding="utf-8",
    )
    out = tmp_path / "s.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sbml_to_net(str(source), str(out), validate="L1")
    text = out.read_text(encoding="utf-8")
    species = text.split("begin species")[1].split("end species")[0].splitlines()
    assert ["B", "f"] in [line.split()[1:3] for line in species if line.strip()]
    assert "folded_at_load=" in _parameters(text)["f"]
    assert report.structural is not None and report.structural.passed
    model = _load(out)
    assert model.frozen_params == ["f"]
    assert model.get_concentration("B") == 2.0
    assert _b_at(model, 2.0) == pytest.approx(2.0 + _b_unread(2, 2.0), rel=REL)


@needs_sbml
@pytest.mark.skipif(not BIOMD613.is_file(), reason="BIOMD0000000613 is not in this checkout")
def test_a_corpus_model_reloads_with_the_frozen_parameters_the_sbml_has(tmp_path):
    # BIOMD0000000613: k3, k4 and Pic0 go into initial values an
    # initialAssignment sets, and rate laws read each. The converted model took
    # set_param("k3", 2 k3), which moved the rate laws and not the initial
    # value. Oracle: the SBML model's own list, and its run.
    out = tmp_path / "m.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = bngsim.Model.from_sbml(str(BIOMD613))
        report = sbml_to_net(str(BIOMD613), str(out), strict=False, validate="L1")
    frozen = loaded.frozen_params
    assert frozen
    model = _load(out)
    assert model.frozen_params == frozen
    assert report.structural is not None and report.structural.passed
    marked = [name for name, _ in _folded_at_load_in(out.read_text(encoding="utf-8"))]
    assert sorted(marked) == sorted(frozen)
    (note,) = [n for n in report.bngsim_only if "folded" in n]
    assert all(name in note for name in frozen)
    with pytest.raises(ParameterError):
        model.set_param("k3", 2 * model.get_param("k3"))
    # With nothing written the two run alike up to t = 10: the model's one
    # event, which the .net does not carry, acts later. The same equations on
    # both sides, so to the solver's last digits (1e-9 relative).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = bngsim.Simulator(loaded).run(t_span=(0.0, 10.0), n_points=11)
        b = bngsim.Simulator(model).run(t_span=(0.0, 10.0), n_points=11)
    for name in a.species_names:
        x, y = a.species[name], b.species[name]
        assert (abs(x - y) <= 1e-9 * abs(x) + 1e-300).all(), name


@needs_sbml
@pytest.mark.skipif(not MODEL1509.is_file(), reason="MODEL1509020000 is not in this checkout")
def test_a_corpus_model_is_written_without_the_frozen_parameters_nothing_reads(tmp_path):
    # MODEL1509020000: parameter_1 and parameter_2 size compartments and no
    # rate law, rule or initial value of the written network names either.
    # Oracle: the SBML model's own list; the file's own parameters block.
    out = tmp_path / "m.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = bngsim.Model.from_sbml(str(MODEL1509))
        report = sbml_to_net(str(MODEL1509), str(out), strict=False, validate="L1")
    assert loaded.frozen_params == ["parameter_1", "parameter_2"]
    assert (report.folded_left_out, report.folded_marked) == (loaded.frozen_params, [])
    text = out.read_text(encoding="utf-8")
    parameters = _parameters(text)
    assert not set(loaded.frozen_params) & set(parameters) and "folded_at_load" not in text
    assert (loaded.n_parameters, report.n_parameters) == (86, 84)
    (note,) = [n for n in report.dropped if "folded" in n]
    assert "(parameter_1, parameter_2)" in note
    assert report.structural is not None and report.structural.passed
    model = _load(out)
    assert model.frozen_params == []
    with pytest.raises(ParameterError, match="'parameter_1' not found"):
        model.set_param("parameter_1", 2.0)


def _bng2pl_pair(work: Path, actions: str) -> subprocess.CompletedProcess:
    """BNG2.pl on m.net in *work* and then the lines of an actions block."""
    lines = [line for line in actions.splitlines() if "generate_network" not in line]
    body = "\n".join(lines).replace("begin actions", "").replace("end actions", "")
    (work / "run.bngl").write_text(f'readFile({{file=>"m.net"}})\n{body}\n', encoding="utf-8")
    return subprocess.run(
        ["perl", str(BNG2), "run.bngl"],
        cwd=work,
        env={**os.environ, "BNGPATH": str(BNG2.parent)},
        capture_output=True,
        text=True,
        timeout=300,
    )


ACTIONS = """begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>2})
setParameter("f", 3)
simulate({method=>"ode",t_start=>1,t_end=>2,n_steps=>2,continue=>1})
end actions
"""
# The same with the value f already has, which is no change.
SAME = ACTIONS.replace('setParameter("f", 3)', 'setParameter("f", 2)')
SCAN = (
    "begin actions\ngenerate_network({overwrite=>1})\n"
    'parameter_scan({parameter=>"f",par_min=>1,par_max=>3,n_scan_pts=>3,'
    'method=>"ode",t_start=>0,t_end=>2,n_steps=>2})\n'
    "end actions\n"
)
# ACTIONS on the network that still reads f, in a reader that takes the write:
# to t = 1 as the model, B(1) = _b_read(2, 2, 1) and A(1) = exp(-K). From there
# the rate law has f = 3 (g = 3 KD) and the stoichiometry is still 2:
# B(2) = B(1) exp(-g) + 2 K A(1) (exp(-g) - exp(-K)) / (K - g).
_G3 = 3.0 * KD
B_WITH_THE_RATE_LAW_MOVED = _b_read(2, 2, 1.0) * math.exp(-_G3) + 2.0 * K * math.exp(-K) * (
    math.exp(-_G3) - math.exp(-K)
) / (K - _G3)


def _b_last(runs) -> float:
    """B at the end of the last run of a protocol."""
    last = runs[-1].result
    names = [n.split("::")[-1].split("(")[0] for n in last.species_names]
    return float(last.species[-1][names.index("B")])


def _archive_to_net(work: Path, actions: str, *, read: bool):
    """An archive of the model at f = 2 whose SED-ML holds *actions*, through
    ``omex_to_net``: ``(report, the .net, the actions block, the warnings)``."""
    from bngsim.convert import omex_to_net, parse_bngl_protocol, write_omex, write_sedml_protocol

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # what SED-ML's standard elements leave out
        sedml = write_sedml_protocol(parse_bngl_protocol(actions), model_source="model.xml")
        write_omex(work / "m.omex", sbml=_sbml(2, read=read), sedml=sedml)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = omex_to_net(work / "m.omex", work / "m.net", gate="L1")
    block = Path(report.bngl_out).read_text(encoding="utf-8")
    return report, work / "m.net", block, [str(w.message) for w in caught]


@needs_sbml
@pytest.mark.parametrize(
    ("actions", "starts", "written"),
    [
        (ACTIONS, 'setParameter of "f" sets', 'setParameter("f",3)'),
        (SAME, 'setParameter of "f" sets', 'setParameter("f",2)'),
        (SCAN, 'parameter_scan of "f" scans', 'parameter=>"f"'),
    ],
    ids=["setParameter", "setParameter to the value held", "parameter_scan"],
)
@pytest.mark.parametrize(
    ("read", "says"),
    [
        (False, "a parameter the network does not declare"),
        (True, "that the network or its actions still read"),
    ],
    ids=["left out", "marked"],
)
def test_an_archives_action_on_a_frozen_parameter_is_said_in_the_report(
    tmp_path, read, says, actions, starts, written
):
    # The archive's SED-ML sets f, or scans it; the loaded SBML model refuses
    # a write that changes f. The .net leaves f out, or marks it; either way
    # the actions written beside it name f, and the report and the actions
    # file say what that comes to.
    from bngsim.convert import omex_to_net, parse_bngl_protocol, write_omex, write_sedml_protocol

    report, out, block, warned = _archive_to_net(tmp_path, actions, read=read)
    (note,) = [n for n in report.lossy if "folded" in n]
    assert note.startswith(f"actions: {starts}") and says in note
    assert "as the loaded model refuses a write that changes it" in note
    assert read or "so the action is refused, or moves nothing, where the block is run" in note
    assert not read or "bngsim refuses the action where it changes the value" in note
    assert [w for w in warned if "folded into" in w][-1] == note[9:]
    assert f"# bngsim: {note[9:]}\n" in block and written in block
    # The loaded SBML model refuses a write that changes f, and takes one of
    # the value f has.
    source = tmp_path / "s.xml"
    source.write_text(_sbml(2, read=read), encoding="utf-8")
    with pytest.raises(ParameterError):
        bngsim.Model.from_sbml(str(source)).set_param("f", 3.0)
    bngsim.Model.from_sbml(str(source)).set_param("f", 2.0)
    # And so does the model the pair is run on, where it has f.
    with pytest.raises(ParameterError):
        _load(out).set_param("f", 3.0)
    # The pair, run. Where f is left out, bngsim refuses the action as one on
    # a parameter the network has not, whatever the value. Where f is marked,
    # it refuses the action as a write to a parameter folded at load, and
    # runs the one that sets the value f has.
    protocol = parse_bngl_protocol(Path(report.bngl_out))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if read and actions is SAME:
            # B -> 0 at f kd B is read live with f = 2 throughout.
            runs = bngsim.run_protocol(_load(out), protocol)
            assert _b_last(runs) == pytest.approx(_b_read(2, 2, 2.0), rel=REL)
        else:
            with pytest.raises(ParameterError if read else bngsim.ConversionError):
                bngsim.run_protocol(_load(out), protocol)
    # An archive whose actions set another parameter gets no such note.
    plain = write_sedml_protocol(
        parse_bngl_protocol(actions.replace('"f"', '"k"')), model_source="model.xml"
    )
    other = tmp_path / "other.omex"
    write_omex(other, sbml=_sbml(2, read=read), sedml=plain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = omex_to_net(other, tmp_path / "other.net", gate="L1")
    assert not [n for n in clean.lossy if "folded" in n]


@needs_sbml
@needs_bng2
@pytest.mark.parametrize(
    "actions", [ACTIONS, SAME, SCAN], ids=["setParameter", "to the value held", "parameter_scan"]
)
@pytest.mark.parametrize("read", [False, True], ids=["left out", "marked"])
def test_bng2pl_on_an_archives_action_on_a_frozen_parameter(tmp_path, read, actions):
    # The pair of the test above, in BNG2.pl. Where f is marked it takes the
    # action, which moves the rate law that still reads f and not the
    # stoichiometry. Where f is left out it aborts on the name (2.9.3, and
    # 2.5.1 for a setParameter), or scans a name the network has not without
    # moving anything (2.5.1): each point of this scan, which starts every
    # point from the initial state, is then the folded f = 2.
    _, _, block, _ = _archive_to_net(tmp_path, actions, read=read)
    done = _bng2pl_pair(tmp_path, block)
    said = done.stdout + done.stderr
    if read:
        assert done.returncode == 0, said[-1000:]
        if actions is not SCAN:
            # The last row of its species file: the time, A, B.
            row = (tmp_path / "run.cdat").read_text(encoding="utf-8").splitlines()[-1].split()
            want = _b_read(2, 2, 2.0) if actions is SAME else B_WITH_THE_RATE_LAW_MOVED
            assert float(row[0]) == 2.0
            assert float(row[2]) == pytest.approx(want, rel=REL)
    elif done.returncode:
        assert "Parameter f not defined" in said, said[-1000:]
    else:
        assert actions is SCAN, said[-1000:]
        # Its scan file: a row for each point, with f, A and B at t = 2, in
        # nine digits.
        rows = (tmp_path / "run_f.scan").read_text(encoding="utf-8").splitlines()[1:]
        assert [float(row.split()[0]) for row in rows] == [1.0, 2.0, 3.0]
        for row in rows:
            assert float(row.split()[2]) == pytest.approx(_b_unread(2, 2.0), rel=REL)


READS = """begin actions
generate_network({overwrite=>1})
simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>2})
setConcentration("A()", %s)
simulate({method=>"ode",t_start=>1,t_end=>2,n_steps=>2,continue=>1})
end actions
"""
# A -> 2 B at A from A = 1: A(1) = exp(-1), B(1) = 2 (1 - exp(-1)). A is then
# set to f/2 = 1, and B(2) = B(1) + 2 (1 - exp(-1)).
B_AFTER_THE_READ = 4.0 * (1.0 - math.exp(-1.0))


@needs_sbml
def test_a_parameter_the_actions_read_stays_in_the_network(tmp_path):
    # The archive's SED-ML sets A to f*0.5. Reading a frozen parameter is no
    # write: the loaded SBML model runs it. Nothing in the network reads f, and
    # left out of the .net it made the action fail where the pair ran right
    # before. It stays, marked, and a write to it is still refused.
    from bngsim.convert import parse_bngl_protocol

    work = tmp_path / "reads"
    work.mkdir()
    report, out, block, _ = _archive_to_net(work, READS % '"f*0.5"', read=False)
    assert (report.folded_left_out, report.folded_marked) == ([], ["f"])
    assert not report.lossy and not [n for n in report.dropped if "folded" in n]
    assert "folded_at_load=" in _parameters(out.read_text(encoding="utf-8"))["f"]
    assert 'setConcentration("A","f*0.5")' in block
    model = _load(out)
    assert model.frozen_params == ["f"]
    with pytest.raises(ParameterError):
        model.clone().set_param("f", 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = bngsim.run_protocol(model, parse_bngl_protocol(Path(report.bngl_out)))
    assert _b_last(runs) == pytest.approx(B_AFTER_THE_READ, rel=REL)

    # The same archive with a number for the value reads nothing: f is left out.
    number = tmp_path / "number"
    number.mkdir()
    plain = _archive_to_net(number, READS % "1", read=False)[0]
    assert (plain.folded_left_out, plain.folded_marked) == (["f"], [])


@needs_sbml
@needs_bng2
def test_bng2pl_runs_the_pair_whose_actions_read_the_parameter(tmp_path):
    # The pair of the test above, in BNG2.pl: f is in the network, so the
    # action that reads it evaluates.
    _, _, block, _ = _archive_to_net(tmp_path, READS % '"f*0.5"', read=False)
    done = _bng2pl_pair(tmp_path, block)
    assert done.returncode == 0, (done.stdout + done.stderr)[-1000:]
    # The last row of its species file: the time, A, B.
    row = (tmp_path / "run.cdat").read_text(encoding="utf-8").splitlines()[-1].split()
    assert float(row[0]) == 2.0
    assert float(row[2]) == pytest.approx(B_AFTER_THE_READ, rel=REL)


@needs_sbml
def test_sbml_to_net_keeps_what_actions_read_however_it_is_told(tmp_path):
    # Three ways the names reach the writer: given outright, read off the
    # SED-ML sbml_to_net is handed, and, in an archive of two SED-ML files,
    # read off the one that is not its master (which omex_to_net passes on).
    from bngsim.convert import omex_to_net, parse_bngl_protocol, write_omex, write_sedml_protocol
    from bngsim.convert._omex import _FMT_SEDML

    source = tmp_path / "s.xml"
    source.write_text(_sbml(2, read=False), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reading = write_sedml_protocol(
            parse_bngl_protocol(READS % '"f*0.5"'), model_source="model.xml"
        )
        plain = write_sedml_protocol(parse_bngl_protocol(READS % "1"), model_source="model.xml")
        (tmp_path / "reading.sedml").write_text(reading, encoding="utf-8")
        alone = sbml_to_net(str(source), str(tmp_path / "alone.net"))
        told = sbml_to_net(str(source), str(tmp_path / "told.net"), read_by_actions={"f", "x"})
        handed = sbml_to_net(
            str(source), str(tmp_path / "handed.net"), sedml=str(tmp_path / "reading.sedml")
        )
        write_omex(
            tmp_path / "two.omex",
            sbml=_sbml(2, read=False),
            sedml=plain,
            extra=[("./second.sedml", reading, _FMT_SEDML)],
        )
        two = omex_to_net(tmp_path / "two.omex", tmp_path / "two.net", gate="L1")
    assert (alone.folded_left_out, alone.folded_marked) == (["f"], [])
    for report, name in ((told, "told.net"), (handed, "handed.net"), (two, "two.net")):
        assert (report.folded_left_out, report.folded_marked) == ([], ["f"]), name
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "folded_at_load=" in _parameters(text)["f"], name
        assert _load(tmp_path / name).frozen_params == ["f"], name
        assert [n for n in report.bngsim_only if "(f)" in n], name


@needs_sbml
def test_the_names_the_actions_read_are_taken_in_any_form(tmp_path):
    # write_net uses the names twice, for its notes and for the file: an
    # iterator has to serve both, or the warning says marked and the file
    # leaves the parameter out. A str is one name, not its letters.
    source = tmp_path / "s.xml"
    source.write_text(_sbml(2, read=False), encoding="utf-8")
    model = bngsim.Model.from_sbml(str(source))

    def written(given):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            text = bngsim.convert.write_net(model, read_by_actions=given)
        (warned,) = [str(w.message) for w in caught if "folded" in str(w.message)]
        return _parameters(text), warned

    for given in ({"f"}, ["f"], iter(["f"]), (n for n in ("x", "f")), "f"):
        parameters, warned = written(given)
        assert "folded_at_load=" in parameters["f"], given
        assert "only bngsim reads" in warned and "(f)" in warned, given
        assert "dropping" not in warned, given
    for given in ((), "kf", ["k"]):  # none of these names f
        parameters, warned = written(given)
        assert "f" not in parameters, given
        assert "dropping" in warned and "(f)" in warned, given

    def lists(given):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = sbml_to_net(str(source), read_by_actions=given)
        return report.folded_left_out, report.folded_marked

    assert lists(iter(["f"])) == ([], ["f"]) and lists("f") == ([], ["f"])
    assert lists("kf") == (["f"], [])

    # The two capability reports take the names the same way.
    from bngsim.convert import bngl_capability_report, capability_report

    for report in (capability_report, bngl_capability_report):
        kept = report(model, read_by_actions=iter(["f"]))
        assert [n for n in kept["bngsim_only"] if "(f)" in n], report.__name__
        assert not [n for n in kept["dropped"] if "folded" in n], report.__name__
        gone = report(model, read_by_actions="kf")
        assert [n for n in gone["dropped"] if "(f)" in n], report.__name__
        assert not [n for n in gone["bngsim_only"] if "folded" in n], report.__name__
    assert _frozen.as_names("kf") == {"kf"} and _frozen.as_names(iter("kf")) == {"k", "f"}


def _bngl_whose_actions_read_f(tmp_path: Path):
    """``write_bngl`` of the model at f = 2 with a protocol that sets A to
    ``f*0.5``: ``(the model, the steps before and after, the file, its text,
    the warnings of a folded parameter)``."""
    from bngsim.convert import Experiment, ProtocolSpec, StateChange, write_bngl

    source = tmp_path / "s.xml"
    source.write_text(_sbml(2, read=False), encoding="utf-8")
    model = bngsim.Model.from_sbml(str(source))
    first = Experiment(kind="simulate", method="ode", t_span=(0.0, 1.0), n_points=3)
    then = Experiment(
        kind="simulate", method="ode", t_span=(1.0, 2.0), n_points=3, extra={"continue": 1}
    )
    reads = StateChange(kind="set_concentration", target="A", value_expr="f*0.5")
    out = tmp_path / "reads.bngl"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = write_bngl(model, str(out), protocol=ProtocolSpec(steps=(first, reads, then)))
    said = [str(w.message) for w in caught if "folded" in str(w.message)]
    return model, (first, then), out, text, said


@needs_sbml
def test_write_bngl_keeps_a_parameter_its_actions_read(tmp_path):
    from bngsim.convert import ProtocolSpec, StateChange, write_bngl

    model, (first, then), _, text, said = _bngl_whose_actions_read_f(tmp_path)
    sets = StateChange(kind="set_concentration", target="A", value=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        number = write_bngl(model, protocol=ProtocolSpec(steps=(first, sets, then)))
        alone = write_bngl(model)
    assert "folded_at_load=" in _parameters(text)["f"]
    assert 'setConcentration("@comp1:A()","f*0.5")' in text
    # The warning is of what the file does: f marked, and nothing dropped.
    (warned,) = said
    assert "only bngsim reads" in warned and "(f)" in warned and "dropping" not in warned
    # With a number for the value, or with no actions, nothing reads f.
    assert "f" not in _parameters(number) and "f" not in _parameters(alone)


@needs_sbml
@needs_bng2
def test_the_bngl_whose_actions_read_the_parameter_runs(tmp_path):
    # The file of the test above, run: BNG2.pl generates its network and
    # bngsim runs the actions on it.
    out = _bngl_whose_actions_read_f(tmp_path)[2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = bngsim.run_bngl(str(out), cache=False)
    assert _b_last(runs) == pytest.approx(B_AFTER_THE_READ, rel=REL)


def test_the_names_the_actions_read():
    from bngsim.convert import Experiment, ProtocolSpec, StateChange

    steps = (
        StateChange(kind="set_concentration", target="A", value_expr="f*0.5 + 2g"),
        StateChange(kind="set_parameter", target="h", value=3.0),  # set, not read
        StateChange(kind="set_parameter", target="k", value_expr="1e-3*m"),
        Experiment(
            kind="scan",
            method="ode",
            t_span=(0.0, 1.0),
            n_points=3,
            scan_parameter="s",  # scanned, not read
            scan_min=1.0,
            scan_max=3.0,
            scan_points=3,
            extra={"stop_if": "B>q"},
        ),
    )
    assert _frozen.names_read(ProtocolSpec(steps=steps)) == {"f", "g", "m", "B", "q"}
    assert _frozen.names_read(None) == frozenset()
    assert _frozen.names_read(ProtocolSpec(steps=())) == frozenset()


def test_only_a_setparameter_on_a_folded_parameter_gets_a_note():
    # The notes, by themselves: one for each folded parameter the actions set
    # or scan, however often; none for a setConcentration on a species of that
    # name, for a simulate, nor for a parameter that was not folded.
    from bngsim.convert import Experiment, ProtocolSpec, StateChange, _actions_on_folded

    report = SimpleNamespace(folded_left_out=["f"], folded_marked=["g"])

    def notes(*steps):
        return _actions_on_folded(ProtocolSpec(steps=tuple(steps)), report)

    set_f = StateChange(kind="set_parameter", target="f", value=3.0)
    set_g = StateChange(kind="set_parameter", target="g", value=3.0)
    assert notes(StateChange(kind="set_concentration", target="f", value=1.0)) == []
    assert notes(StateChange(kind="set_parameter", target="k", value=1.0)) == []
    assert notes(StateChange(kind="save_parameters")) == []
    left_out, marked = notes(set_f, set_g, set_f)
    assert left_out.startswith('setParameter of "f" sets') and "does not declare" in left_out
    assert marked.startswith('setParameter of "g" sets') and "or its actions still read" in marked

    def run(kind, parameter=None):
        return Experiment(
            kind=kind,
            method="ode",
            t_span=(0.0, 1.0),
            n_points=3,
            scan_parameter=parameter,
            scan_min=1.0,
            scan_max=3.0,
            scan_points=3,
        )

    assert notes(run("simulate"), run("scan", "k")) == []
    scanned, swept, again = notes(run("scan", "f"), run("bifurcate", "g"), run("scan", "f"), set_f)
    assert scanned.startswith('parameter_scan of "f" scans') and "does not declare" in scanned
    assert swept.startswith('bifurcate of "g" scans') and "or its actions still read" in swept
    assert again.startswith('setParameter of "f" sets')


@needs_sbml
@needs_bng2
def test_bng2pl_reads_a_marked_network(tmp_path):
    # The mark is a comment on a line where BNG2.pl writes comments of its own.
    _, out, _ = _convert(tmp_path, sbml_to_net, ".net", 2, read=True)
    work = tmp_path / "bng2"
    work.mkdir()
    shutil.copy(out, work / "m.net")
    (work / "run.bngl").write_text(
        'readFile({file=>"m.net"})\nwriteNET({prefix=>"again"})\n', encoding="utf-8"
    )
    done = subprocess.run(
        ["perl", str(BNG2), "run.bngl"],
        cwd=work,
        env={**os.environ, "BNGPATH": str(BNG2.parent)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout[-1000:]
    again = (work / "again.net").read_text(encoding="utf-8")
    assert _parameters(again)["f"].split("#")[0].strip() in ("2", "2.0")


# --- what is not this change's ---------------------------------------------------


@needs_sbml
@pytest.mark.parametrize(("writer", "ext"), [(sbml_to_net, ".net"), (sbml_to_bngl, ".bngl")])
def test_a_model_with_no_frozen_parameter_is_written_with_no_mark_and_no_note(
    tmp_path, writer, ext
):
    # The same model with the stoichiometry a number: f is free, and unread.
    source = tmp_path / "s.xml"
    text = _sbml(2, read=False)
    start, end = text.index("<stoichiometryMath>"), text.index("</stoichiometryMath>")
    source.write_text(text[:start] + text[end + len("</stoichiometryMath>") :], encoding="utf-8")
    assert bngsim.Model.from_sbml(str(source)).frozen_params == []
    out = tmp_path / f"s{ext}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = writer(str(source), str(out))
    written = out.read_text(encoding="utf-8")
    assert "folded_at_load" not in written and "f" in _parameters(written)
    assert not [n for n in report.dropped + report.bngsim_only if "folded" in n]
    assert not [w for w in caught if "folded" in str(w.message)]


@pytest.mark.skipif(not COMPARTMENTAL.is_file(), reason="rec_dim_comp.net is not in this checkout")
def test_the_volumes_bng2pl_folds_are_written_as_before():
    # A compartmental network's volumes are frozen too, by BNG2.pl
    # (unit_conversion=). write_net writes each as it did: that loss is not
    # this change's.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_net(str(COMPARTMENTAL))
        text = bngsim.convert.write_net(model, strict=False)
    assert model.frozen_params == ["vol_EC", "vol_PM"]
    assert not _frozen.has_folded(model)
    parameters = _parameters(text)
    assert (parameters["vol_EC"], parameters["vol_PM"]) == ("39.0", "0.01")
    assert "folded_at_load" not in text


# --- the comment, read ---------------------------------------------------------

NET = """\
# a comment naming folded_at_load=nothing
begin parameters
    1 k 1.0  # Constant
    2 f 2.0  # folded_at_load=a number  # and a second comment
    3 g k*2  # ConstantExpression
#   4 h 1.0  # folded_at_load=a line commented out
    4 j 1.0  # folded_at_load =  spaced out\r
    5 m 1.0  # folded_at_load=
end parameters
begin species
    1 A 1.0  # folded_at_load=not a parameter line
end species
begin reactions
    1 1 0 k  # folded_at_load=nor this
end reactions
"""


def test_the_comment_is_read_off_parameter_lines_only():
    assert _folded_at_load_in(NET) == [("f", "a number"), ("j", "spaced out")]
    assert _folded_at_load_in(NET.replace("folded_at_load", "folded")) == []
    assert _folded_at_load_in(BNGL_FORMS) == [(name, f"reason {name}") for name in "fghjmnpst"]
    # Lines that declare nothing name nothing.
    nothing = (
        "begin parameters\n  =  # folded_at_load=x\n  : f 2  # folded_at_load=y\nend parameters\n"
    )
    assert _folded_at_load_in(nothing) == []


# The forms of a parameter line BNG2.pl reads in a .bngl: no index, an index,
# an = between the name and the value, a label before the name.
BNGL_FORMS = """\
begin model
begin parameters
  k 1
  f 2 * k  # folded_at_load=reason f
  7 g 1  # folded_at_load=reason g
  h=2.5  # folded_at_load=reason h
  9 j=2.5  # folded_at_load=reason j
  m = 2.5  # folded_at_load=reason m
  lbl: n 2.5  # folded_at_load=reason n
  11 lbl2: p 2.5  # folded_at_load=reason p
  lbl3 : s 2.5  # folded_at_load=reason s
  13 lbl4 : t 2.5  # folded_at_load=reason t
  other:  # folded_at_load=a label and nothing after it
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 1
end seed species
begin reaction rules
  A() -> 0 k*f*g*h*j*m*n*p*s*t
end reaction rules
end model
"""


@needs_bng2
def test_bng2pl_reads_every_form_of_parameter_line_the_comment_is_read_off(tmp_path):
    # Oracle for the forms above: BNG2.pl declares each of the nine in the
    # network it generates (without the last line, which is no declaration and
    # which it refuses), and from_bngl lists each as frozen.
    text = "\n".join(line for line in BNGL_FORMS.split("\n") if "other:" not in line)
    path = tmp_path / "forms.bngl"
    path.write_text(text + "generate_network({overwrite=>1})\n", encoding="utf-8")
    net = tmp_path / "forms.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_bngl(str(path), cache=False, net_out=str(net))
    declared = _parameters(net.read_text(encoding="utf-8"))
    assert set("fghjmnpst") <= set(declared)
    assert not {"lbl", "lbl2", "lbl3", "lbl4"} & set(declared)
    assert model.frozen_params == list("fghjmnpst")


def test_a_comment_written_by_hand_freezes_the_parameter(tmp_path):
    path = tmp_path / "hand.net"
    path.write_text(NET, encoding="utf-8")
    model = bngsim.Model.from_net(str(path))
    assert model.frozen_params == ["f", "j"]
    with pytest.raises(ParameterError, match="for a number, as the file's folded_at_load comment"):
        model.set_param("f", 5.0)
    model.set_param("k", 5.0)  # any other parameter is written as before
    assert model.get_param("k") == 5.0

    # The same through the dictionary of the file, and not without its entry.
    parsed = parse_net_file(str(path))
    assert parsed["folded_at_load"] == [("f", "a number"), ("j", "spaced out")]
    assert build_model_from_parsed(parsed).frozen_params == ["f", "j"]
    parsed["folded_at_load"] = [("f", "a number"), ("not_a_parameter", "x")]
    built = build_model_from_parsed(parsed)
    assert built.frozen_params == ["f"] and set(built._frozen_params) == {"f"}
    del parsed["folded_at_load"]
    assert build_model_from_parsed(parsed).frozen_params == []
    parsed["folded_at_load"] = ["f"]
    with pytest.raises(bngsim.InvalidModelError, match="folded_at_load"):
        build_model_from_parsed(parsed)


@needs_sbml
@needs_bng2
def test_a_bngl_comment_is_read_though_bng2pl_drops_it(tmp_path):
    _, out, _ = _convert(tmp_path, sbml_to_bngl, ".bngl", 2, read=True)
    net = tmp_path / "generated.net"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bngsim.Model.from_bngl(str(out), cache=False, net_out=str(net))
    # BNG2.pl's network has the parameter and not the comment.
    generated = net.read_text(encoding="utf-8")
    assert "folded_at_load" not in generated and " f " in generated
    assert model.frozen_params == ["f"]
    assert bngsim.Model.from_net(str(net)).frozen_params == []


# --- the text, folded ------------------------------------------------------------

TEXT = """\
# f1 f2 f3 f4 in a comment
begin parameters
    1 k 1.0
    2 f1 2.0
    3 f2 f1*2
    4 f3 3.0
    5 f4 4.0
    6 Ve 10.0
    7 r k*f3
end parameters
begin reactions
    1 1 0 r
end reactions
"""


def test_fold_leaves_out_what_only_parameters_left_out_read():
    # f2 reads f1 and nothing reads f2: both go. f3 is read by r: marked. f4 is
    # named in a comment only: gone. Ve was folded by BNG2.pl: not this
    # module's. Indices are counted again.
    frozen = {
        "f1": "one",
        "f2": "two",
        "f3": "three,\n  on two lines",
        "f4": "four",
        "Ve": "the volume factor (reaction 1: unit_conversion=1/Ve)",
        "absent": "x",
    }
    text, left_out, marked = _frozen.fold(SimpleNamespace(_frozen_params=frozen), TEXT)
    assert (left_out, marked) == (["f1", "f2", "f4"], ["f3"])
    assert text.split("\n")[1:7] == [
        "begin parameters",
        "    1 k 1.0",
        "    2 f3 3.0  # folded_at_load=three, on two lines",
        "    3 Ve 10.0",
        "    4 r k*f3",
        "end parameters",
    ]
    assert _folded_at_load_in(text) == [("f3", "three, on two lines")]
    # Nothing frozen, nothing changed.
    assert _frozen.fold(SimpleNamespace(_frozen_params={}), TEXT) == (TEXT, [], [])
    assert _frozen.fold(SimpleNamespace(), TEXT) == (TEXT, [], [])
    only_bng2pl = SimpleNamespace(_frozen_params={"Ve": frozen["Ve"]})
    assert _frozen.fold(only_bng2pl, TEXT) == (TEXT, [], [])
    assert not _frozen.has_folded(only_bng2pl)


def test_fold_reads_whole_names_and_not_the_lines_that_open_a_block():
    # e, e5, species and end are named nowhere but in a number's exponent
    # (1e-30, and 2.e5 in r) and in the lines that open and close a block: each
    # is left out. ke is not e, e2 is read by r, and R is the second molecule
    # of a species. q, w and z are read after a number, which the .net reader
    # takes for a product (2q is 2*q). k.f, a name with a dot, is read by r2.
    text = """\
begin parameters
    1 e 2.0
    2 species 3.0
    3 end 4.0
    4 ke 1e-30
    5 e2 1.5e+3
    6 r ke*e2*2.e5+2q+2.5w+ke*2z
    7 e5 2.0
    8 R 6.0
    9 q 7.0
    10 w 8.0
    11 z 9.0
    12 k.f 1.0
    13 r2 k.f*2
end parameters
begin species
    1 A(e~0) 1.0
    2 L(r!1).R(l!1) 1.0
end species
begin reactions
    1 1 0 r
end reactions
"""
    names = ("e", "species", "end", "e2", "e5", "R", "q", "w", "z", "k.f")
    frozen = {name: "x" for name in names}
    folded, left_out, marked = _frozen.fold(SimpleNamespace(_frozen_params=frozen), text)
    # A(e~0) names e, a component of that name: e stays, on the safe side.
    assert (left_out, marked) == (["species", "end", "e5"], ["e", "e2", "R", "q", "w", "z", "k.f"])
    without = _frozen.fold(
        SimpleNamespace(_frozen_params=frozen), text.replace("A(e~0)", "A(x~0)")
    )
    assert without[1:] == (["e", "species", "end", "e5"], ["e2", "R", "q", "w", "z", "k.f"])
    # A name that actions written beside the file read stays, whatever the
    # file itself reads.
    kept = _frozen.fold(SimpleNamespace(_frozen_params=frozen), text, {"species", "absent"})
    assert kept[1:] == (["end", "e5"], ["e", "species", "e2", "R", "q", "w", "z", "k.f"])
    assert "    1 e 2.0  # folded_at_load=x" in folded.split("\n")


def test_the_report_keeps_its_older_fields_in_place():
    # The fields of the report before this change, in their order (read off
    # the release it is built on): a report built by position, or read as a
    # tuple, means what it did. The two new fields come after them.
    from bngsim.convert import ConversionReport

    older = [
        "source",
        "out_path",
        "output_text",
        "n_species",
        "n_reactions",
        "n_parameters",
        "n_observables",
        "dropped",
        "lossy",
        "structural",
        "max_rhs_delta",
        "max_ar_report_delta",
        "rhs_faithful",
        "validation",
        "protocol",
        "bngl_out",
        "bngsim_only",
        "sedml_lossy",
    ]
    names = [f.name for f in dataclasses.fields(ConversionReport)]
    assert names[: len(older)] == older
    assert {"folded_left_out", "folded_marked"} <= set(names[len(older) :])


def test_the_notes_name_every_parameter():
    dropped, bngsim_only = _frozen.notes(["a", "b"], ["c", "d", "e", "f"])
    assert "2 parameter(s)" in dropped[0] and "(a, b)" in dropped[0]
    assert "4 parameter(s)" in bngsim_only[0] and "(c, d, e, f)" in bngsim_only[0]
    # Whose frozen_params the note means, and that the actions count as readers.
    assert "(Model.frozen_params of the model written)" in dropped[0]
    assert "which the written network or its actions still read (c, d, e, f)" in bngsim_only[0]
    assert "as it does on the model written" in bngsim_only[0]
    assert _frozen.notes([], []) == ([], [])
