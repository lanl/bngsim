"""A species an assignment rule sets survives the cBNGL round-trip (issue #1400).

``write_bngl`` refused every model with such a species, saying "the flat .net
round-trip cannot carry the rule". The rule was carried: it is written as the
function of the species' own name, which every rate law reads, so the other
species' dynamics were exact (2.6e-13 through BNG2.pl, 0 through bngsim's
reload). Only the species' own column was lost: written ``$``-fixed, it was
held at its initial value by BNG2.pl and by bngsim reloading BNG2.pl's ``.net``,
which names it as a pattern (``@comp1::$B()``) that never collides with its
rule's name, so the reader that rebuilds the rule (#515) never found the pair.

The species is now written ``__ar_<rule>``, beside the function that carries
its rule. bngsim's reader pairs the two and reports the column; the ``bng2``
gate checks that column as well as the right-hand side, which a held species
cannot show (dy/dt = 0 on both sides). BNG2.pl's own run still holds it, so it
is a bngsim-only note, warned, as the flat ``.net`` writer's is (#1274).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import ConversionError, ConversionWarning
from bngsim.convert import bngl_capability_report, sbml_to_bngl, write_bngl
from bngsim.convert._bng2 import find_bng2

pytestmark = pytest.mark.skipif(not bngsim.HAS_LIBSBML, reason="SBML conversion requires libsbml")

_BNG2 = find_bng2()
needs_bng2 = pytest.mark.skipif(
    _BNG2 is None or shutil.which("perl") is None, reason="BNG2.pl / perl not available"
)

_RATES = "k1 = 0.5; k2 = 0.3; J1: A -> ; k1*A; J2: -> C; k2*B;"
_HEAD = "species A in c = 10; species C in c = 0;"
MODELS = {
    "linear": f"compartment c = 1; {_HEAD} species B in c; B := 2*A; {_RATES}",
    "nonlinear": f"compartment c = 1; {_HEAD} species B in c; B := A*A; {_RATES}",
    "volume 2": f"compartment c = 2; {_HEAD} species B in c; B := 2*A; {_RATES}",
    "consumed": (
        f"compartment c = 1; {_HEAD} species D in c = 0;"
        f" species $B in c; B := 2*A; k3 = 0.2; J3: B -> D; k3*B; {_RATES}"
    ),
}


def _sbml(tmp_path: Path, text: str) -> Path:
    import antimony

    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(text) >= 0, antimony.getLastError()
    path = tmp_path / "m.xml"
    path.write_text(antimony.getSBMLString(antimony.getMainModuleName()))
    return path


def _bng2(bngl: str, action: str) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "m.bngl").write_text(bngl + "\n" + action + "\n")
    subprocess.run(
        [str(_BNG2), "--outdir", str(d), str(d / "m.bngl")],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return d


def _end(model: bngsim.Model) -> dict[str, float]:
    r = bngsim.Simulator(model, method="ode").run(
        t_span=(0, 5), n_points=11, rtol=1e-10, atol=1e-12
    )
    names = [n.split("::")[-1].removesuffix("()").removeprefix("__ar_") for n in r.species_names]
    return dict(zip(names, np.asarray(r.species)[-1].tolist(), strict=True))


# --------------------------------------------------------------------------- #
# without BioNetGen
# --------------------------------------------------------------------------- #
def test_the_species_is_written_beside_its_rule_and_warned_not_refused(tmp_path):
    model = bngsim.Model.from_sbml(str(_sbml(tmp_path, MODELS["nonlinear"])))
    caps = bngl_capability_report(model)
    assert caps["lossy"] == []
    assert any("__ar_" in n for n in caps["bngsim_only"]), caps["bngsim_only"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = write_bngl(model, strict=True)
    assert any(issubclass(w.category, ConversionWarning) for w in caught)
    assert "@comp1:$__ar_B() " in text and "\n    B() (A*A)" in text


def test_a_species_that_cannot_be_marked_is_still_refused(tmp_path):
    """Control: a parameter already named ``__ar_B`` takes the mark, so the
    species is written unmarked, and bngsim could not read it back."""
    text = MODELS["linear"] + " __ar_B = 1;"
    model = bngsim.Model.from_sbml(str(_sbml(tmp_path, text)))
    caps = bngl_capability_report(model)
    assert any("assignment-rule species" in n for n in caps["lossy"]), caps
    with pytest.raises(ConversionError, match="cannot be marked"):
        write_bngl(model, strict=True)


def test_a_model_with_no_reactions_is_refused(tmp_path):
    """BNG2.pl aborts on a model with no reaction rules. BIOMD0000000566 and
    567 are assignment-rule species alone: refused for those before this, they
    would now be written, and the file would not build."""
    model = bngsim.Model.from_sbml(
        str(_sbml(tmp_path, "compartment c = 1; species A in c = 10; species B in c; B := 2*A;"))
    )
    assert any("no reactions" in n for n in bngl_capability_report(model)["lossy"])
    with pytest.raises(ConversionError, match="no reactions"):
        write_bngl(model, strict=True)


def test_the_reader_pairs_only_a_marked_species(tmp_path):
    """A hand-written .net: a fixed ``__ar_X()`` beside a function X reports
    through X; a fixed ``Y()`` beside a function Y does not (a BNGL species
    name is a pattern, and only the mark says it is a rule's); nor does an
    ``__ar_Z()`` that is not fixed, which the writer never writes."""
    net = """\
begin parameters
    1 k 1
end parameters
begin species
    1 @c::$__ar_X() 7
    2 @c::$Y() 7
    3 @c::A() 3
    4 @c::__ar_Z() 7
end species
begin reactions
    1 3 0 k
end reactions
begin groups
    1 Aobs 3
end groups
begin functions
    1 X() 2*Aobs
    2 Y() 5*Aobs
    3 Z() 3*Aobs
end functions
"""
    path = tmp_path / "m.net"
    path.write_text(net)
    amap = bngsim.Model.from_net(str(path))._ar_report_map
    assert amap == {"@c::__ar_X()": ("expression", "X", 1.0)}


def test_the_column_check_reads_a_reordered_reload_in_its_own_order(tmp_path):
    """``_ar_report_delta`` on a reload whose species BNG2.pl put in another
    order: read at the same state through ``back_perm``, it agrees; read in the
    source's order, it does not."""
    from bngsim.convert._validate import _ar_report_delta

    src = bngsim.Model.from_sbml(str(_sbml(tmp_path, MODELS["nonlinear"])))
    assert list(src.species_names) == ["A", "C", "B"]
    net = """\
begin parameters
    1 k1 0.5
    2 k2 0.3
end parameters
begin species
    1 @c::$__ar_B() 100
    2 @c::C() 0
    3 @c::A() 10
end species
begin reactions
    1 3 0 k1
    2 0 2 J2
end reactions
begin groups
    1 Aobs 3
end groups
begin functions
    1 B() Aobs*Aobs
    2 J2() k2*B
end functions
"""
    path = tmp_path / "back.net"
    path.write_text(net)
    back = bngsim.Model.from_net(str(path))
    names = {"A": "@c::A()", "C": "@c::C()", "B": "@c::__ar_B()"}
    perm = [2, 1, 0]  # source index -> reload index
    assert _ar_report_delta(src, back, back_names=names, back_perm=perm) <= 1e-12
    assert _ar_report_delta(src, back, back_names=names) > 1e-3


# --------------------------------------------------------------------------- #
# through BNG2.pl
# --------------------------------------------------------------------------- #
@needs_bng2
@pytest.mark.parametrize("case", sorted(MODELS))
def test_every_column_comes_back_through_bng2_pl(tmp_path, case):
    """BNG2.pl's .net, reloaded: every species, the rule's included, ends as the
    source does. On main the rule species was held at its initial value (20 for
    1.64 on the linear model)."""
    src = bngsim.Model.from_sbml(str(_sbml(tmp_path, MODELS[case])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConversionWarning)
        bngl = write_bngl(src, strict=True)
    d = _bng2(bngl, "generate_network({overwrite=>1})")
    want, got = _end(src), _end(bngsim.Model.from_net(str(d / "m.net")))
    assert set(got) == set(want)
    for name, value in want.items():
        assert got[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name


@needs_bng2
def test_the_bng2_gate_checks_the_column_and_passes(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConversionWarning)
        rep = sbml_to_bngl(_sbml(tmp_path, MODELS["nonlinear"]), validate="bng2")
    assert rep.rhs_faithful is True
    assert rep.max_ar_report_delta is not None and rep.max_ar_report_delta <= 1e-9
    assert rep.bngsim_only and not rep.lossy


@needs_bng2
def test_bng2_pls_own_run_still_holds_the_column(tmp_path):
    """Why the note is bngsim-only: BNG2.pl's own simulation keeps the species
    $-fixed at its initial value (20), while the other species agree. If BNG2.pl
    ever carries the column, this flags the note as stale."""
    src = bngsim.Model.from_sbml(str(_sbml(tmp_path, MODELS["linear"])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConversionWarning)
        bngl = write_bngl(src, strict=True)
    d = _bng2(
        bngl,
        'generate_network({overwrite=>1})\nsimulate({method=>"ode",t_end=>5,n_steps=>10,'
        "atol=>1e-12,rtol=>1e-10})",
    )
    # .cdat names its columns S1..Sn, in the order of BNG2.pl's own .net.
    order = [
        n.split("::")[-1].removesuffix("()")
        for n in bngsim.Model.from_net(str(d / "m.net")).species_names
    ]
    last_row = [ln for ln in (d / "m.cdat").read_text().splitlines() if ln.strip()][-1]
    last = dict(zip(order, map(float, last_row.split()[1:]), strict=True))
    want = _end(src)
    assert last["__ar_B"] == 20.0
    assert last["A"] == pytest.approx(want["A"], rel=1e-6)
    assert last["C"] == pytest.approx(want["C"], rel=1e-6)
