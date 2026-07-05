"""Regression test for issue #42 — `mratio()` nan-on-overflow.

The previous `mratio_impl()` in `src/expression.cpp` was a direct
power-series summation of M(a,b,z) and M(a+1,b+1,z), then a ratio.
For the BNG2.pl-supported parameter range hit by `test_Mratio_1.bngl`
(`a=-1000, b=9001, z=-10000`) the intermediate partial sums peaked at
~1.5e308 and overflowed `double` → inf/inf = nan, which silently
propagated through `U1_U0` / `U2_U1` / `C_mean` / `C_sdev`.

The fix replaces the naive series with the modified-Lentz continued
fraction that BNG2.pl's `Util::Mratio` uses (Hlavacek 2018 Fortran,
Harris 2019 Perl). Lentz works with per-step ratios Δ_j ≈ 1, so
partial values stay O(1) and the same parameter range converges in a
few hundred iterations.

Reference values come from BNG2.pl's own `test_Mratio_1_ode.gdat`
output and are documented verbatim in the issue.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from bngsim import Model


def test_mratio_overflow_params_finite(mratio_overflow_net: Path) -> None:
    """The downstream `mratio`-dependent params load to finite values
    (the primary symptom in #42 was silent nan propagation)."""
    model = Model.from_net(mratio_overflow_net)
    for name in ("U1_U0", "U2_U1", "C_mean", "C_sdev"):
        value = model.get_param(name)
        assert math.isfinite(value), f"{name} = {value!r} (expected finite)"


def test_mratio_overflow_params_match_bng(mratio_overflow_net: Path) -> None:
    """The Lentz CF reproduces BNG2.pl's `test_Mratio_1_ode.gdat` values.

    Per the issue, BNG2.pl prints `C_theory=487.5199603907`,
    `C_upper=518.7261409425`, `C_lower=456.3137798390`. The values
    below match those to every printed digit.
    """
    model = Model.from_net(mratio_overflow_net)

    assert model.get_param("U1_U0") == pytest.approx(-5.124800396093e-05, rel=1e-12)
    assert model.get_param("C_mean") == pytest.approx(487.51996039074, rel=1e-12)
    assert model.get_param("C_sdev") == pytest.approx(15.60309027589, rel=1e-12)
