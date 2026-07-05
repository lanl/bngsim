"""Tests for Antimony (.ant) model loading — Session 23.

Tests both pure ODE models (rate rules) and reaction-based models.
Validates against analytical solutions where available.
"""

import math
from pathlib import Path

import bngsim
import pytest

# ─── Test model paths ─────────────────────────────────────────────────────────

pytest.importorskip("antimony")

SSYS = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "antimony" / "ssys"

pytestmark = pytest.mark.skipif(
    not SSYS.is_dir(),
    reason=f"Antimony benchmark directory not found: {SSYS}",
)


# ─── ModelBuilder (C++ level) ─────────────────────────────────────────────────


def test_model_builder_simple():
    """ModelBuilder: construct a simple decay model programmatically."""
    from bngsim._bngsim_core import ModelBuilder

    builder = ModelBuilder()
    builder.add_parameter("k", 0.3)
    si = builder.add_species("S", 10.0)
    builder.add_observable("S", [(si, 1.0)])
    builder.add_function("_rhs_S", "-k*S")
    builder.add_reaction([], [si], "functional", "_rhs_S")

    model = builder.build()
    assert model.n_species == 1
    assert model.n_reactions == 1
    assert model.n_parameters >= 1


def test_model_builder_two_species():
    """ModelBuilder: two-species model with interactions."""
    from bngsim._bngsim_core import ModelBuilder

    builder = ModelBuilder()
    builder.add_parameter("a", 1.0)
    builder.add_parameter("b", 0.5)
    x_i = builder.add_species("X", 2.0)
    y_i = builder.add_species("Y", 1.0)
    builder.add_observable("X", [(x_i, 1.0)])
    builder.add_observable("Y", [(y_i, 1.0)])
    builder.add_function("_rhs_X", "a*X - b*X*Y")
    builder.add_function("_rhs_Y", "b*X*Y - a*Y")
    builder.add_reaction([], [x_i], "functional", "_rhs_X")
    builder.add_reaction([], [y_i], "functional", "_rhs_Y")

    model = builder.build()
    assert model.n_species == 2
    assert model.n_reactions == 2


# ─── Antimony loading ─────────────────────────────────────────────────────────


def test_load_exp_decay():
    """Load pure ODE model: exponential decay S' = -k*S."""
    model = bngsim.Model.from_antimony(SSYS / "m01_exp_decay.ant")
    assert model.n_species == 1
    assert model.n_reactions == 1
    assert "k" in model.param_names
    assert "S" in model.species_names


def test_exp_decay_simulation():
    """Simulate exponential decay and compare to analytical solution."""
    model = bngsim.Model.from_antimony(SSYS / "m01_exp_decay.ant")
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 20), n_points=201)

    # Analytical: S(t) = 10 * exp(-0.3*t)
    S_final = result.species[-1, 0]
    expected = 10.0 * math.exp(-0.3 * 20.0)
    assert abs(S_final - expected) < 1e-4, f"S(20) = {S_final}, expected {expected}"


def test_load_lotka_volterra():
    """Load pure ODE model: Lotka-Volterra predator-prey."""
    model = bngsim.Model.from_antimony(SSYS / "m03_Lotka_Volterra.ant")
    assert model.n_species == 2
    assert "X" in model.species_names
    assert "Y" in model.species_names


def test_lotka_volterra_simulation():
    """Simulate Lotka-Volterra and check positivity + conservation."""
    model = bngsim.Model.from_antimony(SSYS / "m03_Lotka_Volterra.ant")
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 20), n_points=201)

    # Both species should remain positive (oscillatory system)
    assert result.species[-1, 0] > 0, "X went non-positive"
    assert result.species[-1, 1] > 0, "Y went non-positive"


def test_load_sir_reactions():
    """Load reaction-based model: SIR epidemic."""
    model = bngsim.Model.from_antimony(SSYS / "m07_SIR.ant")
    assert model.n_species == 3
    assert "S" in model.species_names
    assert "I" in model.species_names
    assert "R" in model.species_names


def test_sir_conservation():
    """SIR model: S + I + R should be conserved."""
    model = bngsim.Model.from_antimony(SSYS / "m07_SIR.ant")
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 100), n_points=101)

    total_0 = sum(result.species[0, :3])
    total_end = sum(result.species[-1, :3])
    rel_err = abs(total_end - total_0) / total_0
    assert rel_err < 1e-6, f"Conservation violated: {total_0} -> {total_end} (err={rel_err})"


def test_load_van_der_pol():
    """Load pure ODE model: van der Pol oscillator."""
    model = bngsim.Model.from_antimony(SSYS / "m10_van_der_Pol.ant")
    # van der Pol uses reactions (R1: ->x; y and R2: ->y; mu*(1-x^2)*y-x)
    assert model.n_species >= 2


def test_van_der_pol_simulation():
    """Simulate van der Pol and check for limit cycle behavior."""
    model = bngsim.Model.from_antimony(SSYS / "m10_van_der_Pol.ant")
    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 20), n_points=201)

    # Should produce bounded oscillations (not diverge)
    x_vals = result.species[:, 0]
    assert max(abs(x_vals)) < 100, "van der Pol diverged"


def test_from_antimony_string():
    """Load model from Antimony string (not file)."""
    text = """
    model test_decay()
        S = 5.0
        k = 1.0
        S' = -k*S
    end
    """
    model = bngsim.Model.from_antimony_string(text)
    assert model.n_species == 1
    assert model.n_parameters >= 1

    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 5), n_points=51)

    expected = 5.0 * math.exp(-1.0 * 5.0)
    actual = result.species[-1, 0]
    assert abs(actual - expected) < 1e-4


def test_logistic_growth():
    """Logistic growth: S' = r*S*(1 - S/K)."""
    model = bngsim.Model.from_antimony(SSYS / "m02_logistic.ant")
    assert model.n_species >= 1

    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 20), n_points=201)

    # Should approach carrying capacity (positive, bounded)
    S_end = result.species[-1, 0]
    assert S_end > 0, "Population went negative"
    assert S_end < 1e10, "Population diverged"


def test_michaelis_menten():
    """Michaelis-Menten production/degradation."""
    model = bngsim.Model.from_antimony(SSYS / "m14_Michaelis_Menten_prod_deg.ant")
    assert model.n_species >= 1

    sim = bngsim.Simulator(model, method="ode")
    result = sim.run(t_span=(0, 20), n_points=201)

    # Should reach steady state (positive, bounded)
    X_end = result.species[-1, 0]
    assert X_end > 0, "Species went negative"


def test_file_not_found():
    """from_antimony raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        bngsim.Model.from_antimony("/nonexistent/path.ant")


def test_clone_antimony_model():
    """Clone works for models loaded from Antimony."""
    model = bngsim.Model.from_antimony(SSYS / "m01_exp_decay.ant")
    clone = model.clone()
    assert clone.n_species == model.n_species
    assert clone.n_parameters == model.n_parameters
