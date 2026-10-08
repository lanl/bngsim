"""A compartment volume of a cBNGL model, which BNG2.pl writes into the rate
constants as a number (issue #711).

For a reaction of a compartmental model BNG2.pl evaluates the volume factor and
writes ``0.1*kb`` for ``1/Ve`` at ``Ve = 10``, with the expression in a comment,
``#_R1 unit_conversion=1/Ve``. The loader read the number. ``Ve`` stayed a
parameter that no rate read: a write took the value and moved nothing, and its
sensitivity column was an exact 0.

The networks below are what BNG2.pl 2.9.3 writes for one model at ``Ve = 10``
and at ``Ve = 20``: ``L + R <-> LR`` between the extracellular volume and the
membrane, and ``X + X -> X`` in the cytoplasm. The second is the oracle for the
first: Lf(5) is 14.903 at 10 and 16.756 at 20.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngpath import resolve_bng

NET = """begin parameters
    1 Vc  5  # Constant
    2 Ve  {Ve}  # Constant
    3 Sm  1  # Constant
    4 kb  0.1  # Constant
    5 ku  0.05  # Constant
    6 kt  0.2  # Constant
end parameters
begin species
    1 @EC::L(r) 20
    2 @PM::R(l) 10
    3 @CP::X() 5
    4 @PM::L(r!1)@EC.R(l!1) 0
end species
begin reactions
    1 1,2 4 {factor}*kb #_R1 unit_conversion=1/Ve
    2 3,3 3 0.2*kt #_R2 unit_conversion=1/Vc
    3 4 1,2 ku #_reverse__R1
end reactions
begin groups
    1 Lf                   1
    2 Xc                   3
end groups
"""

BNGL = """begin model
begin parameters
  Vc 5
  Ve 10
  Sm 1
  kb 0.1
  ku 0.05
  kt 0.2
end parameters
begin compartments
  EC 3 Ve
  PM 2 Sm EC
  CP 3 Vc PM
end compartments
begin molecule types
  L(r)
  R(l)
  X()
end molecule types
begin seed species
  @EC:L(r) 20
  @PM:R(l) 10
  @CP:X() 5
end seed species
begin observables
  Molecules Lf @EC:L(r)
  Molecules Xc @CP:X()
end observables
begin reaction rules
  L(r) + R(l) <-> L(r!1).R(l!1) kb, ku
  X() + X() -> X() kt
end reaction rules
end model
"""

RUN = {"t_span": (0.0, 5.0), "n_points": 6, "rtol": 1e-10, "atol": 1e-12}
_BNG = resolve_bng()
needs_bng2 = pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")


def _net(tmp_path: Path, text: str, name: str = "cb.net") -> bngsim.Model:
    path = tmp_path / name
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


def _at(tmp_path: Path, ve: float) -> bngsim.Model:
    return _net(tmp_path, NET.format(Ve=ve, factor=1.0 / ve), f"cb{ve:g}.net")


def _lf(model: bngsim.Model, name: str = "Lf") -> float:
    model.reset()
    result = bngsim.Simulator(model, method="ode").run(**RUN)
    return float(np.asarray(result.observables)[-1, list(result.observable_names).index(name)])


# ── What is recorded ─────────────────────────────────────────────────────────


def test_the_volumes_a_unit_conversion_names_are_listed(tmp_path):
    """``Sm`` is in no reaction's factor here, and is not listed."""
    assert _at(tmp_path, 10).frozen_params == ["Vc", "Ve"]


def test_the_networks_at_two_volumes_differ(tmp_path):
    """Control. What a write of ``Ve`` would have to reproduce."""
    assert _lf(_at(tmp_path, 10)) == pytest.approx(14.903192, abs=1e-5)
    assert _lf(_at(tmp_path, 20)) == pytest.approx(16.756035, abs=1e-5)


def test_what_a_derived_volume_reads_is_listed_too(tmp_path):
    """``vol = 4*r`` set the number, and so did ``r``."""
    text = NET.format(Ve=10, factor=0.1).replace(
        "    6 kt  0.2  # Constant\n",
        "    6 kt  0.2  # Constant\n    7 r  2.5  # Constant\n"
        "    8 vol  4*r  # ConstantExpression\n",
    )
    text = text.replace("unit_conversion=1/Ve", "unit_conversion=1/vol")
    model = _net(tmp_path, text)
    assert model.frozen_params == ["Vc", "r", "vol"]
    with pytest.raises(bngsim.ParameterError, match="'vol', which is read for the volume factor"):
        model.set_param("r", 5.0)
    with pytest.raises(bngsim.ParameterError, match="unit_conversion=1/vol"):
        model.set_param("vol", 20.0)


def test_the_exponent_of_a_number_is_not_a_name(tmp_path):
    """``1/(6.0221e+23*reacvol)`` names ``reacvol``. A parameter called ``e``
    is not named by the ``e`` of the number."""
    text = NET.format(Ve=10, factor=0.1).replace(
        "    6 kt  0.2  # Constant\n",
        "    6 kt  0.2  # Constant\n    7 e  2.5  # Constant\n    8 reacvol  1e-3  # Constant\n",
    )
    text = text.replace("unit_conversion=1/Ve", "unit_conversion=1/(6.0221e+23*reacvol)")
    model = _net(tmp_path, text)
    assert model.frozen_params == ["Vc", "reacvol"]
    model.set_param("e", 3.0)


def test_a_comment_outside_the_reactions_names_nothing(tmp_path):
    text = NET.format(Ve=10, factor=0.1).replace(
        "    4 kb  0.1  # Constant\n", "    4 kb  0.1  # Constant unit_conversion=1/kb\n"
    )
    assert _net(tmp_path, text).frozen_params == ["Vc", "Ve"]


def test_a_network_with_no_such_comment_has_none(tmp_path):
    """Control. A network that is not compartmental is as it was."""
    text = NET.format(Ve=10, factor=0.1)
    for comment in (" #_R1 unit_conversion=1/Ve", " #_R2 unit_conversion=1/Vc"):
        text = text.replace(comment, "")
    model = _net(tmp_path, text)
    assert model.frozen_params == []
    model.set_param("Ve", 20.0)


# ── The write ────────────────────────────────────────────────────────────────


def test_a_write_of_a_volume_is_refused(tmp_path):
    """``set_param("Ve", 20)`` took the value and Lf(5) stayed 14.903, where
    the network generated at 20 gives 16.756."""
    model = _at(tmp_path, 10)
    with pytest.raises(bngsim.ParameterError, match="#711") as refusal:
        model.set_param("Ve", 20.0)
    message = str(refusal.value)
    assert "unit_conversion=1/Ve" in message
    assert "reaction 1" in message
    assert "the BNGL source and generate the network again" in message
    assert model.get_param("Ve") == 10.0
    with pytest.raises(bngsim.ParameterError, match="unit_conversion=1/Vc"):
        model.set_param("Vc", 4.0)


def test_a_write_of_the_value_it_holds_is_no_change(tmp_path):
    """Control. A whole parameter vector goes back in."""
    model = _at(tmp_path, 10)
    before = _lf(model)
    model.set_param("Ve", 10.0)
    model.set_params({name: model.get_param(name) for name in model.param_names})
    assert _lf(model) == before


def test_set_params_writes_nothing_when_it_refuses_one(tmp_path):
    model = _at(tmp_path, 10)
    with pytest.raises(bngsim.ParameterError, match="#711"):
        model.set_params({"kb": 0.2, "Ve": 20.0})
    assert model.get_param("kb") == 0.1


def test_a_clone_refuses_it_too(tmp_path):
    clone = _at(tmp_path, 10).clone()
    assert clone.frozen_params == ["Vc", "Ve"]
    with pytest.raises(bngsim.ParameterError, match="#711"):
        clone.set_param("Ve", 20.0)


def test_a_scan_over_a_volume_is_refused(tmp_path):
    """``parameter_scan("Ve", [10, 20, 40])`` returned Lf(5) = 14.903 three
    times."""
    sim = bngsim.Simulator(_at(tmp_path, 10), method="ode")
    with pytest.raises(bngsim.ParameterError, match="#711"):
        sim.parameter_scan("Ve", [10.0, 20.0, 40.0], t_span=(0.0, 5.0), n_points=6)


def test_a_batch_row_that_writes_a_volume_is_refused(tmp_path):
    sim = bngsim.Simulator(_at(tmp_path, 10), method="ode")
    with pytest.raises(bngsim.ParameterError, match="#711"):
        sim.run_batch(params=[{"Ve": 20.0}], t_span=(0.0, 5.0), n_points=6)


def test_a_rate_constant_is_written_as_before(tmp_path):
    """Control."""
    model = _at(tmp_path, 10)
    before = _lf(model)
    model.set_param("kb", 0.2)
    assert _lf(model) < before - 1.0
    model.set_param("Sm", 3.0)


# ── The column ───────────────────────────────────────────────────────────────


def test_a_sensitivity_column_for_a_volume_is_refused(tmp_path):
    """dLf/dVe came back 0 at every time, beside dLf/dkb = -28.87. Differences
    of the two networks put it near 0.19."""
    model = _at(tmp_path, 10)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#711") as refusal:
        bngsim.Simulator(model, method="ode", sensitivity_params=["Ve", "kb"])
    assert "unit_conversion=1/Ve" in str(refusal.value)
    sim = bngsim.Simulator(model, method="ode")
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#711"):
        sim.compute_all_sensitivities(params=["Vc"], **RUN)


def test_the_other_columns_are_as_they_were(tmp_path):
    """Control."""
    sim = bngsim.Simulator(_at(tmp_path, 10), method="ode", sensitivity_params=["kb"])
    column = np.asarray(sim.run(**RUN).output_sensitivities("Lf"))[-1, 0, 0]
    assert column == pytest.approx(-28.87133, abs=1e-4)


def test_every_column_at_once_leaves_the_volumes_out_and_says_so(tmp_path):
    sim = bngsim.Simulator(_at(tmp_path, 10), method="ode")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.compute_all_sensitivities(**RUN)
    assert list(result.sensitivity_params) == ["Sm", "kb", "ku", "kt"]
    said = [str(w.message) for w in caught if "#711" in str(w.message)]
    assert len(said) == 1
    assert "['Vc', 'Ve']" in said[0]


# ── Through BNG2.pl ──────────────────────────────────────────────────────────


@needs_bng2
def test_a_model_loaded_from_bngl_refuses_it(tmp_path):
    """``Model.from_bngl`` reads the network BNG2.pl writes, comments and all."""
    path = tmp_path / "cb.bngl"
    path.write_text(BNGL)
    model = bngsim.Model.from_bngl(str(path), cache=False)
    assert model.frozen_params == ["Vc", "Ve"]
    assert _lf(model) == pytest.approx(14.903192, abs=1e-5)
    with pytest.raises(bngsim.ParameterError, match="unit_conversion=1/Ve"):
        model.set_param("Ve", 20.0)
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="#711"):
        bngsim.Simulator(model, method="ode", sensitivity_params=["Ve"])


def test_the_corpus_s_compartmental_networks():
    """The in-tree networks BNG2.pl wrote for compartmental models."""
    root = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"
    if not (root / "rec_dim_comp.net").exists():
        pytest.skip("benchmark corpus not present")
    assert bngsim.Model.from_net(str(root / "rec_dim_comp.net")).frozen_params == [
        "vol_EC",
        "vol_PM",
    ]
    assert bngsim.Model.from_net(str(root / "mwc.net")).frozen_params == ["reacvol"]
    assert bngsim.Model.from_net(str(root / "energy_example1.net")).frozen_params == [
        "PI",
        "rad_cell",
        "vol_CP",
    ]
