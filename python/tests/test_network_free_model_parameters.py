"""Issue #1360: method='nf' and method='rm' run on the model's parameters.

NFsim and RuleMonkey are built from the BNG XML alone, and ``Model.set_param``
reached neither: with the only phosphorylation rule switched off (``kph=0``),
SSA gave RP = 0 and nf and rm a few hundred phosphorylations, and with ``L0=0``
both still started from 300 free ligand. Each run now hands the backend every
parameter the model holds at a value other than the one it was loaded with
(``bngsim/_network_free_params.py``).

``model_parameters.{bngl,net,xml}``: L + R <-> LR at kp, km; R bound
phosphorylates at ``kph2 = kph*kv*NA*V/3`` with ``kv = 1/(NA*V)``, so kph2 =
kph/3; RP dephosphorylates at kdeph. Seeds L0 = 300, R0 = 200. The XML prints
``value=`` as ``%.8g``: ``NA``, ``kv`` and ``kph2`` read there other than the
model holds them, with nobody having changed them.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._network_free_params import parameter_overrides, xml_parameter_names

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "nfsim"
NET = DATA / "model_parameters.net"
XML = DATA / "model_parameters.xml"


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


def _run(method: str, writes: dict[str, float], *, after: bool = False, seed: int = 5):
    model = bngsim.Model.from_net(str(NET))
    if not after:
        model.set_params(writes)
    sim = bngsim.Simulator(model, method=method, xml_path=str(XML))
    if after:
        model.set_params(writes)
    return sim.run(t_span=(0, 20), n_points=3, seed=seed)


@pytest.mark.parametrize("after", [False, True], ids=["before the Simulator", "after it"])
@pytest.mark.parametrize("method", METHODS)
def test_a_rate_constant_the_model_sets_reaches_the_backend(method, after):
    """kph = 0 switches off the only phosphorylation: no RP at any time.
    Before the fix, nf ended at RP = 59 and rm at 74."""
    rp = _observable(_run(method, {"kph": 0.0}, after=after), "RP")
    assert np.all(rp == 0.0), rp


@pytest.mark.parametrize("method", METHODS)
def test_a_seed_amount_the_model_sets_reaches_the_backend(method):
    """L0 = 0 seeds no ligand. Before the fix, both started from 300."""
    lfree = _observable(_run(method, {"L0": 0.0}), "Lfree")
    assert np.all(lfree == 0.0), lfree


@pytest.mark.parametrize("method", METHODS)
def test_a_value_set_back_runs_as_the_xml_does(method):
    """Control: a model whose parameter was changed and set back runs as one
    never changed, to the bit, on one Simulator: the overrides of a run are
    the model's at that run, not every one ever made."""
    model = bngsim.Model.from_net(str(NET))
    sim = bngsim.Simulator(model, method=method, xml_path=str(XML))
    model.set_param("kph", 0.0)
    off = sim.run(t_span=(0, 20), n_points=3, seed=5)
    model.set_param("kph", 0.5)
    back = sim.run(t_span=(0, 20), n_points=3, seed=5)
    fresh = _run(method, {})
    assert np.all(_observable(off, "RP") == 0.0)
    np.testing.assert_array_equal(np.asarray(back.observables), np.asarray(fresh.observables))
    assert _observable(fresh, "RP")[-1] > 0.0


@pytest.mark.parametrize("method", METHODS)
def test_a_model_nobody_changed_runs_as_the_xml_does(method):
    """Control: an unchanged model runs as the backend runs the XML with no
    overrides, to the bit. That NA, kv and kph2 are not handed on, which would
    move a rate in its ninth digit, below what these runs show, is
    test_a_model_nobody_changed_hands_the_backend_nothing."""
    import bngsim._bngsim_core as core

    times = core.TimeSpec()
    times.t_start, times.t_end, times.n_points = 0.0, 20.0, 3
    # By name, and only the one this case runs: a build has the class only when
    # it has the backend, and the other may be compiled out.
    backend = getattr(core, "NfsimSimulator" if method == "nf" else "RuleMonkeySimulator")
    for seed in (1, 5, 9):
        raw = backend(str(XML)).run(times, seed, 0.0)
        ours = _run(method, {}, seed=seed)
        assert list(ours.observable_names) == list(raw.observable_names)
        np.testing.assert_array_equal(
            np.asarray(ours.observables), np.asarray(raw.observable_data)
        )


def test_a_model_nobody_changed_hands_the_backend_nothing():
    """Control: NA, kv and kph2 read in the XML at eight digits, other than the
    model holds them. Compared with the XML, all three would be pushed, and an
    unchanged model would run on other rate constants than before."""
    names = xml_parameter_names(XML)
    model = bngsim.Model.from_net(str(NET))
    assert {"NA", "kv", "kph2"} <= names
    assert parameter_overrides(model, names) == {}


def test_a_change_past_the_xmls_digits_is_handed_on():
    """A change in the ninth digit prints as the XML's value, and is still a
    change; a derived parameter the model re-derives goes with its input; a
    clone carries its parent's values at load, not its own."""
    names = xml_parameter_names(XML)
    model = bngsim.Model.from_net(str(NET))
    model.set_param("kp", 0.0100000001)
    assert parameter_overrides(model, names) == {"kp": 0.0100000001}
    model.set_param("kp", 0.01)
    assert parameter_overrides(model, names) == {}
    model.set_param("NA", 6e23)
    expected = {"NA": 6e23, "kv": pytest.approx(1.0 / (6e23 * 1.3e-15), rel=1e-15)}
    assert parameter_overrides(model, names) == expected
    assert parameter_overrides(model.clone(), names) == expected


def _kph_off_kph2_kept(**kwargs) -> tuple[bngsim.Model, float]:
    """kph = 0, with kph2 written back to its value at load: the model holds
    kph2 there, overriding its expression, so R still phosphorylates."""
    model = bngsim.Model.from_net(str(NET))
    kph2 = model.get_param("kph2")
    model.set_param("kph", 0.0)
    model.set_param("kph2", kph2, **kwargs)
    return model, kph2


def test_a_derived_parameter_the_model_holds_is_handed_on():
    """kph2 is at its value at load, and is still handed on: the backend, handed
    kph = 0, would re-derive kph2 = 0 from it. A pin at the expression's own
    value, with nothing else changed, hands nothing: the XML derives the same."""
    names = xml_parameter_names(XML)
    model, kph2 = _kph_off_kph2_kept()
    assert parameter_overrides(model, names) == {"kph": 0.0, "kph2": kph2}
    pinned = bngsim.Model.from_net(str(NET))
    pinned.set_param("kph2", kph2, force_override=True)
    assert parameter_overrides(pinned, names) == {}
    pinned.set_param("kph", 0.0)
    assert parameter_overrides(pinned, names) == {"kph": 0.0, "kph2": kph2}


@pytest.mark.parametrize("method", METHODS)
def test_a_derived_parameter_the_model_holds_reaches_the_backend(method):
    """SSA keeps phosphorylating at kph2 = 1/6, ending at RP = 59. Before the
    fix, nf and rm were handed kph = 0 alone and ran no RP at any row."""
    model, _ = _kph_off_kph2_kept()
    sim = bngsim.Simulator(model, method=method, xml_path=str(XML))
    rp = _observable(sim.run(t_span=(0, 20), n_points=3, seed=5), "RP")
    assert rp[-1] > 0.0, rp


def test_an_unreadable_xml_refuses_a_changed_model():
    """Which parameters an XML that cannot be parsed declares is not known.
    Before, it declared none, and every change was dropped without a word."""
    assert xml_parameter_names(DATA / "malformed.xml") is None
    model = bngsim.Model.from_net(str(NET))
    assert parameter_overrides(model, None) == {}
    model.set_param("kph", 0.0)
    with pytest.raises(bngsim.ModelError, match=r"\['kph', 'kph2'\], but the XML could not be"):
        parameter_overrides(model, None)


@pytest.mark.skipif(not _available("nf"), reason="NFsim not compiled in")
def test_an_unreadable_xml_refuses_the_run_before_the_backend_reads_it():
    """The constructor takes such an XML (NFsim reads it at the run), so the
    run is where the refusal is: ahead of NFsim's own parse."""
    model = bngsim.Model.from_net(str(NET))
    sim = bngsim.Simulator(model, method="nf", xml_path=str(DATA / "malformed.xml"))
    model.set_param("kph", 0.0)
    with pytest.raises(bngsim.SimulationError, match="could not be read"):
        sim.run(t_span=(0, 20), n_points=3, seed=5)


@pytest.mark.parametrize("method", METHODS)
def test_run_batch_refuses_a_network_free_method_by_name(method):
    """run_batch has no network-free branch. It failed at the first point as
    "Unknown method: nfsim" (or rulemonkey)."""
    sim = bngsim.Simulator(bngsim.Model.from_net(str(NET)), method=method, xml_path=str(XML))
    with pytest.raises(ValueError, match=rf"run_batch does not run method='{method}'"):
        sim.run_batch(t_span=(0, 20), n_points=3, params=[{"kph": 0.0}])
