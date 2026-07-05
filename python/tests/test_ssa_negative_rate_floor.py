"""GH #110 — SSA literal rate-law boundary behavior.

The exact SSA used to silently floor a reactant's count at zero after a fire
while still adding the products, which (a) manufactured molecules from nothing
on a constant-flux conversion and (b) made the SSA disagree with the CVODE path
(which integrates the literal rate law and has no non-negativity constraint).

The fix makes both engines agree at the level of means:

  - No floor: a species count goes negative exactly as the literal rate law
    dictates, matching CVODE. Non-negativity is the modeler's responsibility.
  - Sign-split firing: a reaction whose rate law evaluates negative fires in
    REVERSE (products -> reactants) with propensity |rate|, so the SSA's
    expected drift equals the ODE RHS (nu * v) at every state.
  - Both boundary events are surfaced as a filterable ``SsaBoundaryWarning`` and
    recorded on ``result.ssa_diagnostics`` (never silent).

Oracles: the deterministic ODE trajectory (mean tracking) and exact mass
conservation (the bug broke conservation; the fix restores it).
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

# ── SBML builders ─────────────────────────────────────────────────────


def _sbml(species_block: str, product_block: str, math: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
 <model id="m">
  <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
  <listOfSpecies>{species_block}</listOfSpecies>
  <listOfParameters><parameter id="k" value="10"/></listOfParameters>
  <listOfReactions>
   <reaction id="R" reversible="false">
    <listOfReactants><speciesReference species="X"/></listOfReactants>{product_block}
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">{math}</math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>"""


_K = "<ci>k</ci>"  # constant-flux (zeroth-order) rate law: independent of X
_NEG_K = "<apply><minus/><cn>0</cn><ci>k</ci></apply>"  # constant NEGATIVE rate -> reverse

# Pure sink: X -> 0 at constant rate k. Identical to the sys-bio/roadrunner#1320
# reproducer. X must go negative (no floor), tracking the ODE X(t)=X0-k*t.
SINK = _sbml(
    '<species id="X" compartment="c" initialAmount="20" hasOnlySubstanceUnits="true"/>',
    "",
    _K,
)

# Constant-flux conversion X -> Y. X+Y must stay at its initial total (the bug
# inflated it: products added while the reactant was floored).
CONV = _sbml(
    '<species id="X" compartment="c" initialAmount="20" hasOnlySubstanceUnits="true"/>'
    '<species id="Y" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>',
    '<listOfProducts><speciesReference species="Y"/></listOfProducts>',
    _K,
)

# Negative constant rate law on X -> Y. The ODE runs it backward (dX/dt=+k,
# dY/dt=-k); the SSA must fire Y -> X to match. X starts at 0, Y at 30.
REV = _sbml(
    '<species id="X" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>'
    '<species id="Y" compartment="c" initialAmount="30" hasOnlySubstanceUnits="true"/>',
    '<listOfProducts><speciesReference species="Y"/></listOfProducts>',
    _NEG_K,
)

# Plain mass-action X -> Y (rate k*X). Propensity vanishes at X=0, so this never
# floors, never goes negative, never reverses — the control that must stay clean.
MASS_ACTION = _sbml(
    '<species id="X" compartment="c" initialAmount="100" hasOnlySubstanceUnits="true"/>'
    '<species id="Y" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true"/>',
    '<listOfProducts><speciesReference species="Y"/></listOfProducts>',
    "<apply><times/><ci>k</ci><ci>X</ci></apply>",
)


def _col(res, name: str) -> np.ndarray:
    sp = np.asarray(res.species)
    return sp[:, list(res.species_names).index(name)].ravel()


def _ode(sbml: str, species: str, t_end: float, n: int) -> np.ndarray:
    res = bngsim.Simulator(bngsim.Model.from_sbml_string(sbml), method="ode").run(
        t_span=(0, t_end), n_points=n
    )
    return _col(res, species)


def _ssa_mean(sbml: str, species: str, t_end: float, n: int, n_seeds: int) -> np.ndarray:
    acc = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
        for s in range(1, n_seeds + 1):
            sim = bngsim.Simulator(
                bngsim.Model.from_sbml_string(sbml), method="ssa", strict_ssa=False
            )
            acc += _col(sim.run(t_span=(0, t_end), n_points=n, seed=s), species)
    return acc / n_seeds


# ── No floor: SSA goes negative and tracks the ODE ────────────────────


def test_sink_ssa_goes_negative_no_floor():
    """X -> 0 constant flux: SSA X must go negative (no floor), like the ODE."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
        sim = bngsim.Simulator(bngsim.Model.from_sbml_string(SINK), method="ssa", strict_ssa=False)
        x = _col(sim.run(t_span=(0, 10), n_points=11, seed=1), "X")
    assert x.min() < 0, "X should go negative once the constant-flux sink exhausts it"


def test_sink_ssa_mean_tracks_ode():
    """SSA <X(t)> equals the ODE X(t)=20-10t, including the negative tail."""
    ode = _ode(SINK, "X", 10, 11)
    ssa = _ssa_mean(SINK, "X", 10, 11, n_seeds=300)
    # Poisson std of cumulative fires by t is sqrt(k*t) ~ sqrt(100)=10; /sqrt(300)
    # std error ~ 0.6. A 2.5-unit band is a comfortable, non-flaky bound.
    assert np.max(np.abs(ssa - ode)) < 2.5


def test_conv_conserves_total_no_manufacturing():
    """X -> Y constant flux: X+Y stays at the initial 20 (the bug inflated it)."""
    for seed in range(1, 11):
        sim = bngsim.Simulator(bngsim.Model.from_sbml_string(CONV), method="ssa", strict_ssa=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
            r = sim.run(t_span=(0, 10), n_points=11, seed=seed)
        total = _col(r, "X") + _col(r, "Y")
        assert np.allclose(total, 20.0), f"seed {seed}: X+Y drifted to {total}"


# ── Sign-split: negative rate law fires in reverse, mean-faithful ─────


def test_reverse_fire_mean_tracks_ode():
    """Negative-rate X -> Y runs backward; SSA mean tracks ODE X(t)=k*t."""
    ode = _ode(REV, "X", 5, 6)
    ssa = _ssa_mean(REV, "X", 5, 6, n_seeds=300)
    assert np.max(np.abs(ssa - ode)) < 2.0


def test_reverse_fire_conserves_total():
    """Reverse firing applies the full +/- step to both sides: X+Y == 30."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(REV), method="ssa", strict_ssa=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
        r = sim.run(t_span=(0, 5), n_points=6, seed=1)
    assert np.allclose(_col(r, "X") + _col(r, "Y"), 30.0)


# ── Diagnostics + warnings (never silent) ─────────────────────────────


def test_diagnostics_record_negative_crossing_and_reverse():
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(REV), method="ssa", strict_ssa=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
        r = sim.run(t_span=(0, 5), n_points=6, seed=1)
    diag = r.ssa_diagnostics
    assert diag["n_reverse_fires"] > 0
    assert diag["first_reverse_reaction"].startswith("R")
    assert "X -> Y" in diag["first_reverse_reaction"]
    # Y starts at 30 and is consumed by the reverse flux until it goes negative.
    assert diag["n_negative_crossings"] >= 1
    assert diag["first_negative_species"] == "Y"


def test_warns_on_negative_crossing():
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(SINK), method="ssa", strict_ssa=False)
    with pytest.warns(bngsim.SsaBoundaryWarning, match="went negative"):
        sim.run(t_span=(0, 10), n_points=11, seed=1)


def test_warns_on_reverse_fire():
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(REV), method="ssa", strict_ssa=False)
    with pytest.warns(bngsim.SsaBoundaryWarning, match="fired in reverse"):
        sim.run(t_span=(0, 5), n_points=6, seed=1)


def test_warning_is_filterable_to_error():
    """Library-clean: callers can promote the warning to an exception."""
    sim = bngsim.Simulator(bngsim.Model.from_sbml_string(SINK), method="ssa", strict_ssa=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.SsaBoundaryWarning)
        with pytest.raises(bngsim.SsaBoundaryWarning):
            sim.run(t_span=(0, 10), n_points=11, seed=1)


# ── Control: ordinary mass-action stays clean and silent ──────────────


def test_mass_action_stays_clean_and_silent():
    """Mass-action propensity vanishes at 0: no floor, no negative, no reverse."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.SsaBoundaryWarning)  # any warning -> failure
        sim = bngsim.Simulator(bngsim.Model.from_sbml_string(MASS_ACTION), method="ssa")
        r = sim.run(t_span=(0, 10), n_points=11, seed=1)
    sp = np.asarray(r.species)
    assert sp.min() >= 0.0
    assert np.allclose(sp.sum(axis=1), 100.0)  # X+Y conserved
    assert r.ssa_diagnostics["n_negative_crossings"] == 0
    assert r.ssa_diagnostics["n_reverse_fires"] == 0


def test_non_ssa_backend_has_empty_diagnostics():
    """ODE results carry default (zero/empty) ssa_diagnostics, never warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", bngsim.SsaBoundaryWarning)
        r = bngsim.Simulator(bngsim.Model.from_sbml_string(SINK), method="ode").run(
            t_span=(0, 10), n_points=11
        )
    diag = r.ssa_diagnostics
    assert diag["n_negative_crossings"] == 0
    assert diag["n_reverse_fires"] == 0
    assert diag["first_negative_species"] == ""
    assert diag["first_reverse_reaction"] == ""
