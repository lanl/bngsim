"""Repo-wide stochastic initial-count rounding policy (GH #51).

bngsim integerizes a fractional molecule count with round-half-up (nearest, ties
away from zero) at every continuous→discrete boundary it controls, matching
BNG2.pl ``run_network`` SSA and COPASI. These tests pin that policy at three of
those boundaries:

* the :func:`bngsim.round_half_up` helper (the single source for Python),
* the NFsim / RuleMonkey cold-start seed concentrations bngsim rounds into the
  XML the network-free loaders parse (which otherwise truncate toward zero), and
* the session count mutations (warm continue / dosing / scans).

The representative values and exact-half cases come straight from the issue, so
a future parity failure can tell a real simulator difference from an
integerization-policy difference.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest
from bngsim import round_half_up

# value → expected round-half-up count (the issue's regression matrix:
# representative fractions plus the exact halves that distinguish round-half-up
# from Python's bankers' round()).
ROUNDING_MATRIX = {
    0.389: 0,
    0.10001: 0,
    5.7: 6,
    155.6747: 156,
    466.98: 467,
    0.5: 1,
    1.5: 2,
    2.5: 3,
    3.5: 4,
}


def _has_nfsim() -> bool:
    return getattr(bngsim, "HAS_NFSIM", False)


def _has_rulemonkey() -> bool:
    return getattr(bngsim, "HAS_RULEMONKEY", False)


# ─── The helper itself ───────────────────────────────────────────────────────


class TestRoundHalfUp:
    @pytest.mark.parametrize("value,expected", list(ROUNDING_MATRIX.items()))
    def test_matrix(self, value: float, expected: int):
        assert round_half_up(value) == expected
        assert isinstance(round_half_up(value), int)

    def test_diverges_from_bankers_round_at_halves(self):
        # The whole point: ties go away from zero, not to even. Python's round()
        # would give 0, 2, 2 here.
        assert [round_half_up(x) for x in (0.5, 1.5, 2.5)] == [1, 2, 3]

    @pytest.mark.parametrize("n", [0, 1, 5, 156, 467, 1_000_000])
    def test_idempotent_on_integers(self, n: int):
        assert round_half_up(n) == n
        assert round_half_up(float(n)) == n

    def test_negative_ties_away_from_zero(self):
        # Never arises for populations, but the rule is defined and symmetric.
        assert round_half_up(-0.5) == -1
        assert round_half_up(-2.5) == -3
        assert round_half_up(-5.7) == -6

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_raises(self, bad: float):
        with pytest.raises(ValueError):
            round_half_up(bad)


# ─── NFsim cold-start seed concentrations ────────────────────────────────────


@pytest.mark.skipif(not _has_nfsim(), reason="bngsim compiled without NFsim support")
class TestNfsimColdStartRounding:
    def test_literal_fractional_concentration_rounds_up(self, fractional_init_xml: Path):
        # <Parameter n_init value="1.9"/> drives <Species concentration="n_init">.
        # The vendored loader would truncate to 1; bngsim rounds the seed to 2.
        with bngsim.NfsimSession(str(fractional_init_xml)) as nf:
            nf.initialize(seed=1)
            assert nf.get_molecule_count("X") == 2

    @pytest.mark.parametrize("value,expected", list(ROUNDING_MATRIX.items()))
    def test_override_to_fractional_rounds_at_seed(
        self, nfsim_seed_concentration_xml: Path, value: float, expected: int
    ):
        # set_param drives the seed-species parameter to a fractional value; the
        # agent population must be the round-half-up count, not the truncation.
        with bngsim.NfsimSession(str(nfsim_seed_concentration_xml)) as nf:
            nf.set_param("X_init", value)
            nf.initialize(seed=42)
            assert nf.get_molecule_count("X") == expected

    def test_integer_concentration_unchanged(self, nfsim_seed_concentration_xml: Path):
        # No fractional seed, no overrides: rounding is a no-op and the baseline
        # XML counts are preserved exactly (guards the fast path).
        with bngsim.NfsimSession(str(nfsim_seed_concentration_xml)) as nf:
            nf.initialize(seed=42)
            assert nf.get_molecule_count("X") == 5000
            assert nf.get_molecule_count("Y") == 500


# ─── NFsim warm / dosing count mutations ─────────────────────────────────────


@pytest.mark.skipif(not _has_nfsim(), reason="bngsim compiled without NFsim support")
class TestNfsimWarmCountRounding:
    @pytest.mark.parametrize(
        "value,delta",
        [(5.7, 6), (2.5, 3), (1.5, 2), (155.6747, 156), (0.6, 1)],
    )
    def test_add_molecules_rounds_fractional_delta(
        self, nfsim_seed_concentration_xml: Path, value: float, delta: int
    ):
        with bngsim.NfsimSession(str(nfsim_seed_concentration_xml)) as nf:
            nf.initialize(seed=42)
            before = nf.get_molecule_count("X")
            nf.add_molecules("X", value)
            assert nf.get_molecule_count("X") == before + delta

    @pytest.mark.parametrize("value", [0.4, 0.10001, 0.389])
    def test_add_molecules_rounding_to_zero_raises(
        self, nfsim_seed_concentration_xml: Path, value: float
    ):
        # round-half-up(0.4) == 0, so the "must be positive" guard still fires —
        # the rounding happens before the guard, consistently with the policy.
        with bngsim.NfsimSession(str(nfsim_seed_concentration_xml)) as nf:
            nf.initialize(seed=42)
            with pytest.raises(ValueError):
                nf.add_molecules("X", value)


# ─── RuleMonkey cold-start seed concentrations ───────────────────────────────


@pytest.mark.skipif(not _has_rulemonkey(), reason="bngsim compiled without RuleMonkey support")
class TestRuleMonkeyColdStartRounding:
    def test_literal_fractional_concentration_rounds_up(self, fractional_init_xml: Path):
        with bngsim.RuleMonkeySession(str(fractional_init_xml)) as rm:
            rm.initialize(seed=1)
            assert rm.get_molecule_count("X") == 2

    def test_integer_concentration_unchanged(self, nfsim_seed_concentration_xml: Path):
        with bngsim.RuleMonkeySession(str(nfsim_seed_concentration_xml)) as rm:
            rm.initialize(seed=42)
            assert rm.get_molecule_count("X") == 5000
            assert rm.get_molecule_count("Y") == 500
