"""Issue #1366: ``Model.from_bngl`` says when the source's first run starts elsewhere.

The actions after a source's ``generate_network`` are the author's experiment,
and ``from_bngl`` loads the network without them (intended, documented). Most
are harmless to drop, but one that sets a parameter or a concentration *before
the first* ``simulate`` means the model loaded is not the model the author first
simulated, and nothing said so. ``ode/wofsy_goldstein.bngl`` sets ``alpha`` from
0.01 to 0.1 before its first scan; ``egfr_net_red.bngl`` sets ``egf(r)`` to 0
before equilibrating.

Two tiers, as in ``test_bngl_loading.py``: the scan of the actions needs no
BioNetGen; whether a change changes the loaded model needs the network.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim._bngl_loader import state_changes_before_first_run
from bngsim._bngpath import resolve_bng

_REPO = Path(__file__).resolve().parents[2]
_BNG = resolve_bng()
needs_bng2 = pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")

MODEL = """\
begin model
begin parameters
  k   1
  A0  100
  k2  0
end parameters
begin molecule types
  A(b)
  B(x,y,z)
end molecule types
begin seed species
  A(b) A0
  B(x,y,z) 5
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A(b) -> 0 k
end reaction rules
end model
"""


def _verbs(actions: str) -> list[tuple[str, list[str]]]:
    return state_changes_before_first_run(MODEL + actions)


# --------------------------------------------------------------------------- #
# the scan, without BioNetGen
# --------------------------------------------------------------------------- #
def test_a_change_before_the_first_run_is_found():
    actions = 'generate_network({overwrite=>1})\nsetParameter("k",2)\nsimulate({t_end=>1})\n'
    assert _verbs(actions) == [("setParameter", ["k", "2"])]


def test_a_change_before_generation_is_applied_not_dropped():
    actions = 'setParameter("k",2)\ngenerate_network({overwrite=>1})\nsimulate({t_end=>1})\n'
    assert _verbs(actions) == []


def test_a_change_between_two_generation_calls_is_dropped_too():
    """``generation_source`` cuts at the first call, so the change after it never
    reaches the network the second call would have built."""
    actions = (
        'generate_network({overwrite=>1})\nsetParameter("k",2)\n'
        "generate_network({overwrite=>1})\nsimulate({t_end=>1})\n"
    )
    assert _verbs(actions) == [("setParameter", ["k", "2"])]


def test_a_change_after_the_first_run_is_a_later_phase():
    """``nfkb``'s dose after its first phase: the first run starts as declared."""
    actions = (
        'generate_network({overwrite=>1})\nsimulate({t_end=>1})\nsetParameter("k",2)\n'
        "simulate({t_end=>1})\n"
    )
    assert _verbs(actions) == []


def test_save_then_reset_before_the_only_run_changes_nothing():
    """The seven curated files' ``saveConcentrations(); resetConcentrations()``."""
    actions = (
        "generate_network({overwrite=>1})\nsaveConcentrations()\nresetConcentrations()\n"
        "simulate({t_end=>1})\n"
    )
    assert _verbs(actions) == []


def test_a_reset_undoes_a_change_made_after_the_save():
    actions = (
        'generate_network({overwrite=>1})\nsaveConcentrations()\nsetConcentration("A(b)",0)\n'
        'resetConcentrations()\nsaveParameters()\nsetParameter("k",2)\nresetParameters()\n'
        "simulate({t_end=>1})\n"
    )
    assert _verbs(actions) == []


def test_a_reset_keeps_a_change_made_before_the_save():
    actions = (
        'generate_network({overwrite=>1})\nsetConcentration("A(b)",0)\nsaveConcentrations()\n'
        "resetConcentrations()\n"
        'parameter_scan({parameter=>"k",par_min=>1,par_max=>2,n_scan_pts=>2})\n'
    )
    assert _verbs(actions) == [("setConcentration", ["A(b)", "0"])]


def test_no_run_or_no_generation_is_nothing_to_say():
    assert _verbs('generate_network({overwrite=>1})\nsetParameter("k",2)\n') == []
    assert _verbs('setParameter("k",2)\nsimulate({t_end=>1})\n') == []


# --------------------------------------------------------------------------- #
# against the network
# --------------------------------------------------------------------------- #
def _load(tmp_path: Path, actions: str, *, protocol: bool = False):
    path = tmp_path / "m.bngl"
    path.write_text(MODEL + actions)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = bngsim.Model.from_bngl(str(path), protocol=protocol, cache=False)
    return out, [w for w in caught if issubclass(w.category, bngsim.ConversionWarning)]


@needs_bng2
def test_a_change_the_loaded_model_does_not_hold_is_warned(tmp_path):
    actions = (
        'generate_network({overwrite=>1})\nsetParameter("k",2)\nsetConcentration("A(b)",0)\n'
        "simulate({t_end=>1})\n"
    )
    _model, caught = _load(tmp_path, actions)
    assert len(caught) == 1
    text = str(caught[0].message)
    assert "setParameter('k', '2')" in text and "setConcentration('A(b)', '0')" in text
    assert "protocol=True" in text


# The network BNG2.pl generates from MODEL, written by hand, so the judging of
# each action against a network needs no BioNetGen (only the three tests that
# go through from_bngl itself do).
NET = """\
begin parameters
    1 k   1
    2 A0  100
    3 k2  0
end parameters
begin species
    1 A(b) A0
    2 B(x,y,z) 5
end species
begin reactions
    1 1 0 k
end reactions
begin groups
    1 Atot 1
end groups
"""


def _changes(tmp_path: Path, actions: str) -> list[str]:
    from bngsim._bngl_loader import dropped_state_changes

    net = tmp_path / "m.net"
    net.write_text(NET)
    return dropped_state_changes(bngsim.Model.from_net(str(net)), MODEL + actions)


def _before_a_run(change: str) -> str:
    return f"generate_network({{overwrite=>1}})\n{change}\nsimulate({{t_end=>1}})\n"


@pytest.mark.parametrize(
    "change",
    [
        'setParameter("k2",0)',
        'setConcentration("A(b)",100)',
        'setConcentration("A(b)","A0")',
        'setConcentration("B(z,x,y)",5)',
        'addConcentration("A(b)",0)',
    ],
    ids=[
        "a parameter to its value",
        "a species to its amount",
        "to the expression it was seeded from",
        "components in another order",
        "adding 0",
    ],
)
def test_a_change_to_what_the_model_already_holds_is_not_named(tmp_path, change):
    """Control: 53 of the 94 repo files whose actions set something before their
    first run set it to what the model already holds (``egfr_path`` sets EGF to
    its declared 0). A warning there would be wrong."""
    assert _changes(tmp_path, _before_a_run(change)) == []


def test_adding_to_a_species_is_named(tmp_path):
    assert _changes(tmp_path, _before_a_run('addConcentration("A(b)",5)')) == [
        "addConcentration('A(b)', '5')"
    ]


def test_only_the_actions_that_change_something_are_named(tmp_path):
    actions = _before_a_run('setParameter("k2",0)\nsetParameter("k",3)')
    assert _changes(tmp_path, actions) == ["setParameter('k', '3')"]


def test_a_species_the_network_does_not_name_counts_as_a_change(tmp_path):
    """Where the effect cannot be read, say so rather than stay quiet."""
    assert _changes(tmp_path, _before_a_run('setConcentration("A(b!1).A(b!1)",3)')) == [
        "setConcentration('A(b!1).A(b!1)', '3')"
    ]


@needs_bng2
def test_protocol_true_returns_the_actions_instead_of_warning(tmp_path):
    actions = 'generate_network({overwrite=>1})\nsetParameter("k",2)\nsimulate({t_end=>1})\n'
    (_model, _spec), caught = _load(tmp_path, actions, protocol=True)
    assert caught == []


@needs_bng2
def test_wofsy_goldstein_from_the_corpus_is_warned_and_egfr_path_is_not():
    """The corpus cases the issue lists: wofsy_goldstein's alpha (0.01 -> 0.1)
    reaches its first scan; egfr_path's EGF -> 0 is its declared amount."""
    wofsy = _REPO / "benchmarks" / "models" / "bngl" / "ode" / "wofsy_goldstein.bngl"
    egfr = _REPO / "benchmarks" / "models" / "bngl" / "ode" / "egfr_path.bngl"
    messages = {}
    for path in (wofsy, egfr):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bngsim.Model.from_bngl(str(path))
        messages[path.name] = [
            str(w.message) for w in caught if issubclass(w.category, bngsim.ConversionWarning)
        ]
    assert messages["egfr_path.bngl"] == []
    (text,) = messages["wofsy_goldstein.bngl"]
    assert "setParameter('alpha', '0.1')" in text and "'beta'" not in text
