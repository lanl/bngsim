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

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML loading requires libsbml")

_FOUND = find_bng2()
BNG2 = Path(_FOUND).resolve() if _FOUND else None
HAS_BNG2 = BNG2 is not None and BNG2.is_file() and bool(shutil.which("perl"))
needs_bng2 = pytest.mark.skipif(not HAS_BNG2, reason="BNG2.pl not found")

K, KD = 1.0, 0.25
REPO = Path(__file__).resolve().parents[2]
BIOMD613 = REPO / "benchmarks" / "sbml_events" / "BIOMD0000000613.xml"
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


# Tolerance of the runs against a closed form: the solver's default relative
# tolerance is 1e-8, and its global error over two time units of these smooth
# decays stays within a few times that. 1e-6 leaves two orders of margin and is
# four orders below the smallest effect tested (a third of the value).
REL = 1e-6


# --- the source model, as a control -----------------------------------------


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

    model = _load(out)
    assert "f" not in model.param_names and model.frozen_params == []
    with pytest.raises(ParameterError, match="'f' not found"):
        model.set_param("f", 3.0)
    # The network is the one the SBML gives at the f it was loaded with.
    assert _b_at(model, 2.0) == pytest.approx(_b_unread(f, 2.0), rel=REL)


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

    model = _load(out)
    assert model.frozen_params == ["f"]
    with pytest.raises(ParameterError) as refused:
        model.set_param("f", 3.0)
    message = str(refused.value)
    assert "the stoichiometry of 'B' in reaction 'R1'" in message
    assert "the SBML document the file was converted from, and convert it again" in message
    assert model.get_param("f") == 2.0
    # The value it holds is no change, and the run is the SBML's at f = 2.
    model.set_param("f", 2.0)
    assert _b_at(model, 2.0) == pytest.approx(_b_read(2, 2, 2.0), rel=REL)
    # A clone refuses too.
    with pytest.raises(ParameterError):
        model.clone().set_param("f", 3.0)


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
    # A .bngl parameter line may have no index.
    bngl = (
        "begin model\nbegin parameters\n"
        "  f 2 * k # folded_at_load=x y\n"
        "  7 g 1 # folded_at_load=z\n"
        "end parameters\nend model\n"
    )
    assert _folded_at_load_in(bngl) == [("f", "x y"), ("g", "z")]


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


def test_the_notes_name_every_parameter():
    dropped, bngsim_only = _frozen.notes(["a", "b"], ["c", "d", "e", "f"])
    assert "2 parameter(s)" in dropped[0] and "(a, b)" in dropped[0]
    assert "4 parameter(s)" in bngsim_only[0] and "(c, d, e, f)" in bngsim_only[0]
    assert _frozen.notes([], []) == ([], [])
