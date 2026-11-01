"""A species an assignment rule sets, written to .net (issue #1274).

The writer emits it ``$``-fixed, beside the observable (a linear rule) or the
function (any other) that holds the rule under the species' own name. bngsim's
reader takes the pair as the rule (``_ar_report_map_from_net``, #515), so the
file is exact for bngsim. BioNetGen's ``run_network`` holds the species at its
initial value for the whole run instead. Its rate laws read the name, which is
the observable or the function, so they read the rule: only the species' own
column is wrong there. The writer now says so, and still writes the file.
"""

from __future__ import annotations

import subprocess
import warnings

import bngsim
import numpy as np
import pytest
from bngsim._bngpath import resolve_bng
from bngsim.convert import capability_report, sbml_to_net, validate_conversion, write_net

_BNG = resolve_bng()

RULE = "species A = 10; species B; k = 0.5; J: A -> ; k*A; B := 2*A"
# The rule species read by a rate law, C's.
READ = "species A = 10; species B; species C = 0; k = 0.5; J: A -> ; k*A; J2: -> C; B; B := {}"


def _write(antimony: str, path, **kw):
    model = bngsim.Model.from_antimony_string(antimony)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_net(model, str(path), **kw)
    return [str(w.message) for w in caught if issubclass(w.category, bngsim.ConversionWarning)]


@pytest.mark.parametrize(
    ("antimony", "kind"),
    [
        (RULE, "a linear rule, held by an observable"),
        (
            "species A = 10; species B; k = 0.5; J: A -> ; k*A; B := A*A",
            "any other, by a function",
        ),
    ],
)
def test_a_rule_species_is_warned_and_still_written(tmp_path, antimony, kind):
    path = tmp_path / "rule.net"
    said = _write(antimony, path)
    assert path.is_file()
    rule = [m for m in said if "only bngsim reads as written" in m]
    assert len(rule) == 1, said
    assert "'B'" in rule[0] and "run_network holds each at its initial value" in rule[0]
    assert "only those species' own columns can be wrong" in rule[0]


def test_strict_does_not_refuse_it(tmp_path):
    """The file is exact for bngsim's own reader: warned, never refused, so the
    default strict=True still writes it."""
    path = tmp_path / "rule.net"
    _write(RULE, path, strict=True)
    assert path.is_file()


def test_a_refused_file_is_not_warned_as_written(tmp_path):
    """Where strict refuses the model for something else (an amount-valued
    species in a compartment of volume 2), no file is written, and the note
    that says how one is written is not given."""
    model = bngsim.Model.from_antimony_string(
        "compartment c = 2; species A in c = 10; substanceOnly species B in c; "
        "k = 0.5; J: A -> ; k*A; B := 2*A"
    )
    assert capability_report(model)["bngsim_only"]  # the rule species is there
    path = tmp_path / "rule.net"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(bngsim.ConversionError, match="amount-valued"):
            write_net(model, str(path), strict=True)
    assert not path.exists()
    assert not [w for w in caught if "only bngsim reads as written" in str(w.message)]


@pytest.mark.parametrize(
    "antimony",
    [
        "species A = 10; species B = 0; k = 0.5; J: A -> B; k*A",
        # a boundary species a reaction makes is written $-fixed too, and is set
        # by no rule: held, as written, by bngsim and run_network alike
        "species A = 10; $B = 3; k = 0.5; J: A -> B; k*A",
    ],
    ids=["no-rule", "boundary-species"],
)
def test_a_model_with_no_rule_species_is_not_warned(tmp_path, antimony):
    path = tmp_path / "plain.net"
    said = _write(antimony, path)
    if "$B" in antimony:
        assert "$B 3.0" in path.read_text()  # the control is a $-fixed species
    assert not [m for m in said if "only bngsim reads as written" in m], said


def test_the_report_names_it_apart_from_lossy():
    report = capability_report(bngsim.Model.from_antimony_string(RULE))
    assert len(report["bngsim_only"]) == 1
    assert "'B'" in report["bngsim_only"][0]
    assert report["lossy"] == []


def _sbml(tmp_path):
    import antimony

    antimony.loadAntimonyString(f"model m\n  {RULE}\nend")
    sbml = tmp_path / "rule.xml"
    sbml.write_text(antimony.getSBMLString("m"))
    return sbml


@pytest.mark.parametrize("entry", ["sbml_to_net", "validate_conversion"])
def test_the_conversion_reports_carry_it(tmp_path, entry):
    """The reports sbml_to_net and validate_conversion return carry the note,
    and their summaries print it, as they do what was dropped: a caller reading
    the report, and not the warnings, is told too."""
    sbml = _sbml(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.ConversionWarning)
        if entry == "sbml_to_net":
            report = sbml_to_net(sbml, tmp_path / "rule.net")
        else:
            report = validate_conversion(sbml, levels=("L0", "L1"), out_dir=tmp_path)
            assert report.to_dict()["bngsim_only"] == report.bngsim_only
    assert len(report.bngsim_only) == 1 and "'B'" in report.bngsim_only[0]
    assert f"bngsim-only: {report.bngsim_only[0]}" in report.summary()


def test_bngsim_reads_the_rule_back(tmp_path):
    path = tmp_path / "rule.net"
    _write(RULE, path)
    r = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
        t_span=(0, 5), n_points=2
    )
    names = list(r.species_names)
    a, b = np.asarray(r.species)[-1, [names.index("A"), names.index("B")]]
    assert b == pytest.approx(2 * a, rel=1e-6)


@pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")
@pytest.mark.parametrize(
    ("rule", "initial"), [("2*A", 20.0), ("A*A", 100.0)], ids=["observable", "function"]
)
def test_run_network_holds_only_its_column(tmp_path, rule, initial):
    """What the warning says, as run_network does it (with -g, as BNG2.pl runs
    it, so that the functions can read the groups): B is held at its initial
    value, and C, whose rate law reads B, is bngsim's, since the name it reads is
    the observable or function that holds the rule."""
    path = tmp_path / "rule.net"
    _write(READ.format(rule), path)
    run_network = _BNG.root / "bin" / "run_network"
    if not run_network.is_file():
        pytest.skip(f"BNG2.pl unavailable: no run_network at {run_network}")
    out = tmp_path / "rn"
    run = subprocess.run(
        [str(run_network), "-o", str(out), "-p", "cvode", "-a", "1e-12", "-r", "1e-10"]
        + ["-g", str(path), str(path), "5", "1"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert run.returncode == 0, run.stdout[-600:] + run.stderr[-600:]
    a, b, c = np.loadtxt(f"{out}.cdat")[-1, 1:]
    assert b == pytest.approx(initial)  # B, held
    r = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode").run(
        t_span=(0, 5), n_points=2, rtol=1e-10, atol=1e-12
    )
    names = list(r.species_names)
    want = np.asarray(r.species)[-1, [names.index("A"), names.index("C")]]
    np.testing.assert_allclose([a, c], want, rtol=1e-6)
