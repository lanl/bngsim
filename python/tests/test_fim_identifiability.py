"""Tensor-only Fisher Information / model identifiability (GH #202).

Generalizes ``Result.fisher_information`` from a species-only FIM to one built
over **named outputs** (``species:``/``observable:``/``expression:`` selectors)
and adds ``Result.identifiability`` — the eigenvalue / practical-identifiability
("sloppiness") readout of the FIM. This is a pure function of the model + the
output-sensitivity tensor: no measurement data, no residuals, no objective.

Verified against the linear algebra and finite differences:
- FIM over species vs observable vs expression selectors, and that a
  species-only request matches the original species-only method;
- the eigen-readout flags a deliberately non-identifiable toy (degenerate
  parameter pair);
- singular / rank-deficient FIMs warn and return NaN where inversion is invalid.
"""

import os
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest

_env = os.environ.get("BNGSIM_TEST_DATA")
DATA_DIR = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def _net(name: str) -> str:
    p = DATA_DIR / name
    assert p.exists(), f"Test data not found: {p}"
    return str(p)


def _reversible(params=("kf", "kr")):
    """Identifiable two-species reversible model with a 2-param sensitivity run."""
    model = bngsim.Model.from_net(_net("two_species_reversible.net"))
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=list(params))
    return sim.run(t_span=(0, 10), n_points=101)


# ── Generalized FIM over named-output selectors ────────────────────────────


class TestFimOverSelectors:
    def test_species_selectors_match_species_only(self):
        """FIM over every species selector == the species-only method (back-compat)."""
        r = _reversible()
        fim_default = r.fisher_information(sigma=1.0)
        sels = [f"species:{n}" for n in r.species_names]
        fim_sel = r.fisher_information(outputs=sels, sigma=1.0)
        np.testing.assert_allclose(fim_sel, fim_default, rtol=1e-12, atol=0)

    def test_observable_selectors_shape_and_manual(self):
        """FIM over observable selectors matches a manual tensor contraction."""
        r = _reversible()
        sels = [f"observable:{n}" for n in r.observable_names]
        fim = r.fisher_information(outputs=sels, sigma=1.0)
        assert fim.shape == (2, 2)
        S = r.output_sensitivities(sels)  # (nt, n_out, n_param)
        manual = np.einsum("toi,toj->ij", S, S)
        np.testing.assert_allclose(fim, manual, rtol=1e-12, atol=0)

    def test_symmetric_psd_over_observables(self):
        r = _reversible()
        sels = [f"observable:{n}" for n in r.observable_names]
        fim = r.fisher_information(outputs=sels, sigma=1.0)
        np.testing.assert_allclose(fim, fim.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(fim) >= -1e-6 * np.linalg.norm(fim))

    def test_per_output_sigma_scaling(self):
        """1/σ² scaling holds for per-output σ over selectors."""
        r = _reversible()
        sels = [f"observable:{n}" for n in r.observable_names]
        n_out = len(sels)
        fim1 = r.fisher_information(outputs=sels, sigma=np.ones(n_out))
        fim2 = r.fisher_information(outputs=sels, sigma=2.0 * np.ones(n_out))
        np.testing.assert_allclose(fim2, fim1 / 4.0, rtol=1e-10)

    def test_per_output_sigma_wrong_length(self):
        r = _reversible()
        sels = [f"observable:{n}" for n in r.observable_names]
        with pytest.raises(ValueError, match="must match n_outputs"):
            r.fisher_information(outputs=sels, sigma=np.ones(len(sels) + 1))

    def test_single_selector_string(self):
        """A bare selector string (not a list) is accepted."""
        r = _reversible()
        fim = r.fisher_information(outputs="observable:C_total", sigma=1.0)
        assert fim.shape == (2, 2)


# ── Mixed species/observable/expression selectors (codegen) ────────────────


class TestFimExpressionSelectors:
    @pytest.fixture(autouse=True)
    def _force_codegen(self, monkeypatch):
        """Expression output sensitivities require the compiled ``.so``."""
        monkeypatch.setenv("BNGSIM_CODEGEN_THRESHOLD", "1")
        monkeypatch.delenv("BNGSIM_NO_CODEGEN", raising=False)

    def _chain(self):
        m = bngsim.Model.from_net(_net("expr_sens_chain.net"))
        sim = bngsim.Simulator(m, method="ode", sensitivity_params=["k1", "k2", "scale", "eps"])
        return sim.run(t_span=(0, 8), n_points=9, rtol=1e-11, atol=1e-13, max_steps=10**6)

    def test_mixed_kind_selectors(self):
        """A FIM over species + observable + expression selectors at once."""
        r = self._chain()
        sels = ["species:A()", "observable:A_obs", "expression:scaled"]
        fim = r.fisher_information(outputs=sels, sigma=1.0)
        assert fim.shape == (4, 4)
        S = r.output_sensitivities(sels)
        manual = np.einsum("toi,toj->ij", S, S)
        np.testing.assert_allclose(fim, manual, rtol=1e-12, atol=0)

    def test_expression_only_identifiability(self):
        r = self._chain()
        rep = r.identifiability(outputs=["expression:scaled", "expression:ratio"], sigma=0.5)
        assert rep.fim.shape == (4, 4)
        assert rep.parameters == ["k1", "k2", "scale", "eps"]


# ── Identifiability readout on an identifiable model ───────────────────────


class TestIdentifiabilityReadout:
    def test_full_rank_report(self):
        r = _reversible()
        rep = r.identifiability(sigma=1.0)
        assert rep.rank == 2
        assert rep.is_identifiable
        assert rep.non_identifiable_directions == []
        assert np.all(rep.identifiable)
        assert np.isfinite(rep.condition_number)
        assert rep.parameters == ["kf", "kr"]

    def test_eigenvalues_ascending_and_nonneg(self):
        rep = _reversible().identifiability(sigma=1.0)
        assert np.all(np.diff(rep.eigenvalues) >= 0)
        assert np.all(rep.eigenvalues >= 0)

    def test_eigenvectors_aligned_with_eigenvalues(self):
        """``fim @ v_i == λ_i v_i`` — columns align with their eigenvalues."""
        rep = _reversible().identifiability(sigma=1.0)
        for i in range(rep.fim.shape[0]):
            v = rep.eigenvectors[:, i]
            np.testing.assert_allclose(rep.fim @ v, rep.eigenvalues[i] * v, atol=1e-3)

    def test_cramer_rao_is_fim_inverse_when_full_rank(self):
        rep = _reversible().identifiability(sigma=1.0)
        np.testing.assert_allclose(rep.cramer_rao_bound, np.linalg.inv(rep.fim), rtol=1e-8)
        # CRB is a lower bound on parameter variance: diagonal is positive.
        assert np.all(np.diag(rep.cramer_rao_bound) > 0)

    def test_condition_number_matches_eigen_ratio(self):
        rep = _reversible().identifiability(sigma=1.0)
        expected = rep.eigenvalues[-1] / rep.eigenvalues[0]
        np.testing.assert_allclose(rep.condition_number, expected, rtol=1e-10)

    def test_repr_is_concise(self):
        rep = _reversible().identifiability(sigma=1.0)
        text = repr(rep)
        assert "IdentifiabilityReport" in text and "rank=2" in text
        # The full FIM / eigenvector arrays are NOT dumped into the repr.
        assert "\n" not in text


# ── Deliberately non-identifiable toy (degenerate parameter pair) ──────────


class TestNonIdentifiableToy:
    def _degenerate(self):
        model = bngsim.Model.from_net(_net("degenerate_decay.net"))
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1", "k2"])
        return sim.run(t_span=(0, 20), n_points=201)

    def test_sensitivities_are_degenerate(self):
        """dY/dk1 == dY/dk2: k1, k2 enter only through their sum keff = k1 + k2."""
        r = self._degenerate()
        sens = r.sensitivities  # (nt, n_species, n_params)
        np.testing.assert_allclose(sens[:, :, 0], sens[:, :, 1], atol=1e-10)

    def test_eigenvalues_flag_sloppy_direction(self):
        r = self._degenerate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rep = r.identifiability(sigma=1.0)
        assert rep.rank == 1
        assert not rep.is_identifiable
        assert rep.non_identifiable_directions == [0]
        # The sloppy eigenvector is the antisymmetric k1−k2 combination.
        v = rep.eigenvectors[:, 0]
        np.testing.assert_allclose(np.abs(v), [np.sqrt(0.5)] * 2, atol=1e-6)
        assert v[0] * v[1] < 0  # opposite signs ⇒ k1 − k2 direction

    def test_singular_warns_and_crb_is_nan(self):
        r = self._degenerate()
        with pytest.warns(RuntimeWarning, match="rank-deficient"):
            rep = r.identifiability(sigma=1.0)
        assert np.all(np.isnan(rep.cramer_rao_bound))
        assert not np.isfinite(rep.condition_number)

    def test_fim_itself_does_not_warn(self):
        """The bare FIM is well-defined even when singular — only its inverse warns."""
        r = self._degenerate()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fim = r.fisher_information(sigma=1.0)
        assert fim.shape == (2, 2)


# ── rtol surfaces sloppy-but-nonzero directions ────────────────────────────


class TestSloppyThreshold:
    def test_default_rtol_keeps_full_rank(self):
        """The reversible model is ill-conditioned but numerically full rank."""
        rep = _reversible().identifiability(sigma=1.0)
        assert rep.rank == 2

    def test_large_rtol_flags_small_direction(self):
        """A looser rtol surfaces the small (sloppy) eigen-direction."""
        r = _reversible()
        rep_default = r.identifiability(sigma=1.0)
        ratio = rep_default.eigenvalues[0] / rep_default.eigenvalues[-1]
        # Choose rtol between the eigenvalue ratio and 1 so the small one is flagged.
        rtol = float(np.sqrt(ratio))
        assert ratio < rtol < 1.0
        with pytest.warns(RuntimeWarning, match="rank-deficient"):
            rep = r.identifiability(sigma=1.0, rtol=rtol)
        assert rep.rank == 1
        assert rep.non_identifiable_directions == [0]
        assert np.all(np.isnan(rep.cramer_rao_bound))


# ── IC-axis FIM ────────────────────────────────────────────────────────────


class TestFimIcAxis:
    def test_ic_axis_fim(self):
        model = bngsim.Model.from_net(_net("two_species_reversible.net"))
        sim = bngsim.Simulator(model, method="ode", sensitivity_ic=["A()", "B()"])
        r = sim.run(t_span=(0, 10), n_points=101)
        fim = r.fisher_information(axis="ic", sigma=1.0)
        assert fim.shape == (2, 2)
        np.testing.assert_allclose(fim, fim.T, atol=1e-9)
        rep = r.identifiability(axis="ic", sigma=1.0)
        assert rep.parameters == ["A()", "B()"]


# ── Error handling ─────────────────────────────────────────────────────────


class TestFimErrors:
    def test_no_sensitivity_raises(self):
        model = bngsim.Model.from_net(_net("simple_decay.net"))
        sim = bngsim.Simulator(model, method="ode")
        r = sim.run(t_span=(0, 10), n_points=11)
        with pytest.raises(ValueError, match="No sensitivity data"):
            r.fisher_information(sigma=1.0)

    def test_no_ic_sensitivity_raises(self):
        model = bngsim.Model.from_net(_net("simple_decay.net"))
        sim = bngsim.Simulator(model, method="ode", sensitivity_params=["k1"])
        r = sim.run(t_span=(0, 10), n_points=11)
        with pytest.raises(ValueError, match="No sensitivity data"):
            r.fisher_information(axis="ic", sigma=1.0)

    def test_invalid_axis_raises(self):
        r = _reversible()
        with pytest.raises(ValueError, match="axis must be 'parameter' or 'ic'"):
            r.fisher_information(axis="bogus", sigma=1.0)

    def test_scalar_sigma_nonpositive_raises(self):
        r = _reversible()
        with pytest.raises(ValueError, match="sigma must be > 0"):
            r.fisher_information(sigma=0.0)
        with pytest.raises(ValueError, match="sigma must be > 0"):
            r.fisher_information(sigma=-1.0)

    def test_2d_sigma_raises(self):
        r = _reversible()
        with pytest.raises(ValueError, match="scalar or 1-D"):
            r.fisher_information(sigma=np.ones((2, 2)))

    def test_batch_result_rejected(self):
        """The FIM is per single simulation; a stacked batch is refused."""
        r = _reversible()
        batch = bngsim.Result.squeeze([r, r])
        assert batch.species.ndim == 3
        with pytest.raises(ValueError, match="single-simulation"):
            batch.fisher_information(sigma=1.0)
