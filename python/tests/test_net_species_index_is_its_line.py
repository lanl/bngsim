"""A `.net` species index that is not its line is refused (issue #1512).

Reactions and groups name a species by its index, and the loader reads that
index as the species' line. ``2 B() 5`` above ``1 A() 1`` was read by line, so
the file's one reaction, ``1 1 0 k``, degraded B(): Aobs(1) = 1.839397, where
the file declares species 1 to be A() and the closed form is e^-1 = 0.367879.

BioNetGen's two readers disagree on such a file: run_network 3.0 places a
species at the index it declares, and BNG2.pl 2.9.3's readFile at its line
(``SpeciesList.pm`` strips the index). Measured with both on the files below:
each one refused here is one the two read differently, or one run_network
crashes on (SIGSEGV) or refuses; each one that loads here both read to
Aobs(1) = 0.367879. The one refusal stricter than both is an index past
2**32, which run_network's integer wraps to the line it is on.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim.convert import net_to_sbml

E1 = math.exp(-1.0)
REFUSAL = "reactions and groups name a species by its index"


def _net(species: list[str], rxn: str = "1 1 0 k", grp: str = "1 Aobs 1") -> str:
    lines = "".join(f"    {line}\n" for line in species)
    return (
        "begin parameters\n    1 k 1\nend parameters\n"
        f"begin species\n{lines}end species\n"
        f"begin reactions\n    {rxn}\nend reactions\n"
        f"begin groups\n    {grp}\nend groups\n"
    )


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "m.net"
    path.write_text(text)
    return path


def _aobs(model: bngsim.Model) -> float:
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, 1.0), n_points=2, rtol=1e-10, atol=1e-12
    )
    return float(np.asarray(result.observables)[-1, 0])


# ── Refused ──────────────────────────────────────────────────────────────────


def test_the_issue_s_file_is_refused(tmp_path):
    """``2 B() 5`` above ``1 A() 1``: run_network gives Aobs = 0.367879,
    BNG2.pl 1.839397, and bngsim gave 1.839397 with no word."""
    path = _write(tmp_path, _net(["2 B() 5", "1 A() 1"]))
    with pytest.raises(bngsim.ModelError, match=REFUSAL) as caught:
        bngsim.Model.from_net(str(path))
    message = str(caught.value)
    assert "'2 B() 5' has index 2 on the species block's line 1" in message
    assert "run_network at its index" in message and "readFile at its line" in message


def test_every_route_that_reads_the_file_refuses_it(tmp_path):
    """``parse_net_file`` raises the ValueError it documents. ``net_to_sbml``
    returned a report that called the misread model fine, since its check
    compares bngsim with itself."""
    path = str(_write(tmp_path, _net(["2 B() 5", "1 A() 1"])))
    with pytest.raises(ValueError, match=REFUSAL):
        bngsim.parse_net_file(path)
    with pytest.raises(bngsim.ModelError, match=REFUSAL):
        bngsim.Model.load(path)
    with pytest.raises(bngsim.ModelError, match=REFUSAL):
        net_to_sbml(path)


REFUSED = {
    # run_network SIGSEGV; BNG2.pl Aobs = 0.367879.
    "a-gap": ["1 A() 1", "3 B() 5"],
    # run_network 1.839397 (the second line overwrites); BNG2.pl 0.367879.
    "a-duplicate": ["1 A() 1", "1 B() 5"],
    # run_network SIGSEGV; BNG2.pl 0.367879.
    "zero-based": ["0 A() 1", "1 B() 5"],
    # run_network 0.367879; BNG2.pl 1.839397.
    "an-unindexed-line-then-1": ["B() 5", "1 A() 1"],
    # The two put B() and C() on different lines: with a reaction and a group
    # on species 2, run_network degrades B() (1.839397), BNG2.pl C() (0.735759).
    "three-species-permuted": ["1 A() 1", "3 C() 2", "2 B() 5"],
    # Stricter than both: run_network's int wraps 2**32 + 1 to 1.
    "an-index-past-2-to-the-32": ["4294967297 A() 1", "2 B() 5"],
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_an_index_that_is_not_its_line_is_refused(tmp_path, case):
    path = _write(tmp_path, _net(REFUSED[case]))
    with pytest.raises(bngsim.ModelError, match=REFUSAL):
        bngsim.Model.from_net(str(path))


def test_an_offset_block_is_refused(tmp_path):
    """Indices 2, 3, with the reaction and the group on 2: run_network refuses
    ("No reactions in the network"), BNG2.pl reads A() as species 1 and gives
    1.839397."""
    path = _write(tmp_path, _net(["2 A() 1", "3 B() 5"], rxn="1 2 0 k", grp="1 Aobs 2"))
    with pytest.raises(bngsim.ModelError, match="has index 2 on the species block's line 1"):
        bngsim.Model.from_net(str(path))


def test_a_seed_species_block_is_held_to_it_too(tmp_path):
    """BNG2.pl reads ``seed species`` as the species block, and so does the loader."""
    text = _net(["2 B() 5", "1 A() 1"]).replace("species\n", "seed species\n", 2)
    text = text.replace("begin seed species\n    1 k", "begin species\n    1 k")
    path = _write(tmp_path, text)
    with pytest.raises(bngsim.ModelError, match="on the seed species block's line 1"):
        bngsim.Model.from_net(str(path))


# ── Read as before ───────────────────────────────────────────────────────────

READ = {
    "in-order": ["1 A() 1", "2 B() 5"],
    "unindexed": ["A() 1", "B() 5"],
    "an-unindexed-line-then-2": ["A() 1", "2 B() 5"],
    "1-then-an-unindexed-line": ["1 A() 1", "B() 5"],
    "leading-zeros": ["01 A() 1", "02 B() 5"],
    "leading-zeros-on-one": ["001 A() 1", "2 B() 5"],
    "a-tab-after-the-index": ["1\tA() 1", "2\tB() 5"],
}


@pytest.mark.parametrize("case", sorted(READ))
def test_an_index_that_is_its_line_reads_as_before(tmp_path, case):
    """Control: loaded on main too, to the closed form both referees give."""
    model = bngsim.Model.from_net(str(_write(tmp_path, _net(READ[case]))))
    assert list(model.species_names) == ["A()", "B()"]
    assert _aobs(model) == pytest.approx(E1, rel=1e-7)


def test_a_written_network_loads_back(tmp_path):
    """Control: ``write_net`` numbers each species by its line, so a model
    read from an unindexed block round-trips."""
    from bngsim.convert import write_net

    model = bngsim.Model.from_net(str(_write(tmp_path, _net(["A() 1", "B() 5"]))))
    out = tmp_path / "rt.net"
    write_net(model, out)
    back = bngsim.Model.from_net(str(out))
    assert list(back.species_names) == ["A()", "B()"]
    assert _aobs(back) == pytest.approx(E1, rel=1e-7)
