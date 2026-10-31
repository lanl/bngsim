"""Issue #1373: method='nf' and method='rm' start from the model's species amounts.

``Model.set_concentration`` reached neither backend: both are seeded from the
XML's ``<Species>`` list, so with ``L(r)`` set to 0 SSA started from Lfree = 0
and nf and rm from the XML's 300, with no warning. Each run now hands the
backend the amount of every species the model holds at an amount other than the
one it was loaded with, by the XML seed species of the same name
(``bngsim/_network_free_params.py``); an amount it cannot be told is refused.

``model_parameters.{net,xml}`` (#1360): L + R <-> LR, R bound phosphorylates,
RP dephosphorylates. Seeds L(r) = L0 = 300 and R(Y~0,l) = R0 = 200; the XML
lists all five species. Lfree is L(r); RP is the two phosphorylated species.
"""

from __future__ import annotations

import re
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._network_free_params import _unclamped, species_amounts, xml_seed_species

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"
NET = DATA / "model_parameters.net"
XML = DATA / "model_parameters.xml"
FREE_L = "L(r)"
FREE_RP = "R(Y~P,l)"
BOUND = "L(r!1).R(Y~0,l!1)"


def _available(method: str) -> bool:
    if method == "nf":
        return bool(getattr(bngsim, "HAS_NFSIM", False))
    try:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        return bool(HAS_RULEMONKEY)
    except ImportError:
        return False


# Literal reasons, the ones the other network-free tests give: the skip audit
# (test_skip_audit.py) reads each reason from the source and cannot read an
# f-string's.
METHODS = [
    pytest.param(
        "nf", marks=pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
    ),
    pytest.param(
        "rm",
        marks=pytest.mark.skipif(not _available("rm"), reason="RuleMonkey not compiled in"),
    ),
]


def _observable(result, name: str) -> np.ndarray:
    return np.asarray(result.observables)[:, list(result.observable_names).index(name)]


def _run(method: str, model, *, xml: Path = XML, seed: int = 5, t_end: float = 20.0):
    sim = bngsim.Simulator(model, method=method, xml_path=str(xml))
    return sim.run(t_span=(0, t_end), n_points=3, seed=seed)


def _model(**amounts: float) -> bngsim.Model:
    model = bngsim.Model.from_net(str(NET))
    for name, value in amounts.items():
        model.set_concentration(name, value)
    return model


@pytest.mark.parametrize("method", METHODS)
def test_a_species_the_model_empties_starts_empty(method):
    """The issue's repro. Before the fix, nf and rm started from Lfree = 300."""
    lfree = _observable(_run(method, _model(**{FREE_L: 0.0})), "Lfree")
    assert np.all(lfree == 0.0), lfree


@pytest.mark.parametrize("method", METHODS)
def test_the_amount_set_is_the_amount_started_from(method):
    """A free ligand of 50 and a free RP of 40, the second a species whose
    XML concentration is a literal 0 and not a parameter."""
    r = _run(method, _model(**{FREE_L: 50.0, FREE_RP: 40.0}))
    assert _observable(r, "Lfree")[0] == 50.0
    assert _observable(r, "RP")[0] == 40.0


@pytest.mark.parametrize("method", METHODS)
def test_ssa_and_the_network_free_runs_start_alike(method):
    """Every species set at once, complexes included, against SSA's start."""
    amounts = {FREE_L: 120.0, "R(Y~0,l)": 70.0, BOUND: 30.0, FREE_RP: 15.0}
    ssa = bngsim.Simulator(_model(**amounts), method="ssa").run(t_span=(0, 20), n_points=3, seed=5)
    r = _run(method, _model(**amounts))
    for obs in ("Lfree", "RP"):
        assert _observable(r, obs)[0] == _observable(ssa, obs)[0], obs


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("amount", "count"), [(49.5, 50), (49.4, 49)])
def test_a_fractional_amount_rounds_half_up_as_every_seed_does(method, amount, count):
    """GH #51's policy, the one SSA and the XML's own seeds follow."""
    r = _run(method, _model(**{FREE_L: amount}))
    assert _observable(r, "Lfree")[0] == count


@pytest.mark.parametrize("method", METHODS)
def test_set_after_the_simulator_is_built_and_between_runs(method):
    """Read at each run: the Simulator outlives the write."""
    model = _model()
    sim = bngsim.Simulator(model, method=method, xml_path=str(XML))
    assert _observable(sim.run(t_span=(0, 20), n_points=3, seed=5), "Lfree")[0] == 300.0
    model.set_concentration(FREE_L, 7.0)
    assert _observable(sim.run(t_span=(0, 20), n_points=3, seed=5), "Lfree")[0] == 7.0
    model.set_concentration(FREE_L, 300.0)
    assert _observable(sim.run(t_span=(0, 20), n_points=3, seed=5), "Lfree")[0] == 300.0


@pytest.mark.parametrize("method", METHODS)
def test_a_species_amount_outranks_the_parameter_it_was_seeded_from(method):
    """L(r) reads L0 in the XML. Set L0 and the seed follows it (#1360); set
    L(r) as well and the species' own amount wins, as on the network."""
    model = _model()
    model.set_param("L0", 100.0)
    assert _observable(_run(method, model), "Lfree")[0] == 100.0
    model.set_concentration(FREE_L, 25.0)
    assert _observable(_run(method, model), "Lfree")[0] == 25.0


@pytest.mark.parametrize("method", METHODS)
def test_a_run_on_the_network_hands_its_end_state_on(method):
    """The model's state after an SSA run is the state it holds, complexes and
    all, and the network-free run starts there as a second SSA run would."""
    model = _model()
    ssa = bngsim.Simulator(model, method="ssa").run(t_span=(0, 20), n_points=3, seed=5)
    r = _run(method, model)
    for obs in ("Lfree", "RP"):
        assert _observable(r, obs)[0] == _observable(ssa, obs)[-1], obs


@pytest.mark.parametrize("method", METHODS)
def test_a_clone_keeps_what_its_parent_loaded(method):
    model = _model(**{FREE_L: 0.0})
    assert _observable(_run(method, model.clone()), "Lfree")[0] == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_an_unchanged_model_runs_as_before(method):
    """Control: nothing changed, nothing handed over, the same trajectory as a
    model whose amount was written back to its loaded value."""
    plain = _run(method, _model())
    rewritten = _run(method, _model(**{FREE_L: 300.0}))
    np.testing.assert_array_equal(np.asarray(plain.observables), np.asarray(rewritten.observables))
    assert species_amounts(_model(), xml_seed_species(XML), method) == {}


def _xml_without(tmp_path: Path, species_id: str) -> Path:
    text = XML.read_text()
    text, n = re.subn(rf'\n *<Species id="{species_id}" .*?</Species>', "", text, flags=re.S)
    assert n == 1
    out = tmp_path / "seeds.xml"
    out.write_text(text)
    return out


@pytest.mark.parametrize("method", METHODS)
def test_an_amount_for_a_species_the_xml_does_not_seed_is_refused(method, tmp_path):
    """The backend would start it at 0. Before the fix, the run went on from 0."""
    xml = _xml_without(tmp_path, "S3")
    with pytest.raises(ValueError, match=re.escape(f"cannot start species {BOUND!r}")):
        _run(method, _model(**{BOUND: 30.0}), xml=xml)


@pytest.mark.parametrize("method", METHODS)
def test_a_complex_an_ssa_run_left_is_refused_with_the_way_out(method, tmp_path):
    """The usual way to meet the refusal: an SSA run on the same model, which
    writes its end state there, leaves a complex the XML does not seed. The
    message names the run and model.reset(), which lets the run go on, where it
    used to say only to rewrite the XML or run SSA."""
    xml = _xml_without(tmp_path, "S3")
    model = _model()
    bngsim.Simulator(model, method="ssa").run(t_span=(0, 20), n_points=3, seed=5)
    with pytest.raises(ValueError, match=r"an SSA, PSA or ODE run on this model left it there"):
        _run(method, model, xml=xml)
    model.reset()
    assert _observable(_run(method, model, xml=xml), "Lfree")[0] == 300.0


@pytest.mark.parametrize("method", METHODS)
def test_zero_for_a_species_the_xml_does_not_seed_is_what_it_starts_from(method, tmp_path):
    """Control for the refusal: 0 is what the backend starts it from."""
    xml = _xml_without(tmp_path, "S3")
    model = _model()
    model._loaded_state = (300.0, 200.0, 5.0, 0.0, 0.0)  # loaded at 5, now 0
    _run(method, model, xml=xml)


def test_a_name_two_seed_species_share_is_refused(tmp_path):
    text = XML.read_text().replace('name="R(Y~P,l)"', 'name="L(r)"')
    xml = tmp_path / "twice.xml"
    xml.write_text(text)
    seeds = xml_seed_species(xml)
    assert seeds[FREE_L] is None
    with pytest.raises(ValueError, match="more than one of the XML's seed species"):
        species_amounts(_model(**{FREE_L: 3.0}), seeds, "nf")


@pytest.mark.parametrize("method", METHODS)
def test_the_backend_refuses_an_id_it_has_no_seed_species_for(method, monkeypatch):
    """The C++ side does not drop an amount it could not place either."""
    import bngsim._network_free_params as nfp

    monkeypatch.setattr(nfp, "species_amounts", lambda *_a, **_k: {"S99": 5.0})
    with pytest.raises((RuntimeError, bngsim.SimulationError), match="S99"):
        _run(method, _model())


@pytest.mark.parametrize("method", METHODS)
def test_a_species_at_its_loaded_amount_keeps_it_when_its_seed_parameter_moves(method):
    """L(r) reads L0. With the state saved as the baseline, the model keeps
    L(r) = 300 when L0 goes to 0 (Model.set_param), and SSA starts there.
    Handed L0 = 0 and nothing for L(r), which sat at its amount at load, the
    backend re-derived the seed and started from 0."""
    model = _model()
    model.save_concentrations()
    model.set_param("L0", 0.0)
    assert model.get_state()[0] == 300.0
    ssa = bngsim.Simulator(model.clone(), method="ssa").run(t_span=(0, 20), n_points=3, seed=5)
    assert _observable(ssa, "Lfree")[0] == 300.0
    assert _observable(_run(method, model), "Lfree")[0] == 300.0


FIXED_NET = DATA / "fixed_seed.net"
FIXED_XML = DATA / "fixed_seed.xml"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("amount", [10.0, 0.0])
def test_a_fixed_seed_species_is_found_under_the_name_the_model_gives_it(method, amount):
    """The .net loader names $X(a) X(a), and the XML keeps the $. Unmatched,
    10 was refused as a species the XML does not seed, and 0 was dropped as
    one, so the run started from the XML's 50."""
    model = bngsim.Model.from_net(str(FIXED_NET))
    model.set_concentration("X(a)", amount)
    ssa = bngsim.Simulator(model.clone(), method="ssa").run(t_span=(0, 20), n_points=3, seed=5)
    r = _run(method, model, xml=FIXED_XML)
    assert _observable(r, "Xfree")[0] == amount == _observable(ssa, "Xfree")[0]


def test_a_fixed_species_name_is_read_as_the_net_loader_reads_it():
    assert _unclamped("$X(a)") == "X(a)"
    assert _unclamped("@c::$X(a)") == "@c::X(a)"
    assert _unclamped("@c::X(a)") == "@c::X(a)"
    assert _unclamped("X(a)") == "X(a)"
    assert xml_seed_species(FIXED_XML) == {"A(x)": "S1", "X(a)": "S2", "A(x!1).X(a!1)": "S3"}


@pytest.mark.parametrize("method", METHODS)
def test_an_amount_an_ode_run_leaves_just_below_zero_starts_at_zero(method):
    """SSA rounds -1e-12 to 0, as every seed is rounded. Handed on as it was,
    both backends refused it as a negative amount."""
    model = _model()
    state = model.get_state().copy()
    state[0] = -1e-12
    model.set_state(state)
    assert species_amounts(model, xml_seed_species(XML), method) == {"S1": 0.0}
    assert _observable(_run(method, model), "Lfree")[0] == 0.0


def test_an_amount_that_is_no_count_is_refused_by_name():
    model = _model()
    state = model.get_state().copy()
    state[0] = -3.0
    model.set_state(state)
    with pytest.raises(
        ValueError, match=r"cannot start species .L\(r\). at .*: it rounds to no number"
    ):
        species_amounts(model, xml_seed_species(XML), "nf")


def test_an_xml_that_cannot_be_read_refuses_a_changed_amount(tmp_path):
    """As for parameters (#1360): which seed species there are is not known, so
    a changed amount is refused rather than dropped, 0 included."""
    bad = tmp_path / "bad.xml"
    bad.write_text("<sbml><model>")
    assert xml_seed_species(bad) is None
    assert species_amounts(_model(), None, "nf") == {}
    with pytest.raises(bngsim.ModelError, match="could not be read"):
        species_amounts(_model(**{FREE_L: 0.0}), None, "nf")
