"""What Jacobian the KINSOL polish actually uses.

The docs said, in four places, that ``method="newton"`` polishes "with an
analytical Jacobian". It does not: ``solve_by_newton`` never calls
``KINSetJacFn``, so KINSOL installs its own difference-quotient Jacobian and
each setup costs one RHS evaluation per unknown. ``jacobian=`` reaches the
consumers that need the matrix itself — ``dY_ss/dp`` and the issue #78 stability
certificate — not the polish.

That is a cost, not a defect, and wiring the closed form in would change how the
polish converges, so it wants its own measurement rather than a drive-by. This
test pins the behavior the documentation now describes: if someone installs a
Jacobian function, the RHS count collapses and this fails, which is the prompt
to update the prose with it.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest

_NETS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode"
_SHP2 = "SHP2_base_model.net"  # 149 species, 2 conservation laws -> 147 unknowns


def _net(name: str) -> str:
    path = _NETS_DIR / name
    if not _NETS_DIR.is_dir():
        pytest.skip(f"benchmark net corpus not available: {_NETS_DIR}")
    assert path.is_file(), f"vendored net missing from tracked corpus: {path}"
    return str(path)


def test_polish_differences_its_own_jacobian():
    """One RHS evaluation per unknown, per Jacobian setup.

    Measured on the cleanest possible case: seed the model AT its steady state,
    so the two-tier ladder takes its early-exit branch and runs exactly ONE
    KINSOL polish with no integration at all. The model's analytical Jacobian is
    complete here, so the #78 certificate costs no RHS evaluations and every
    call beyond the polish's own iterations is the difference quotient.
    """
    model = bngsim.Model.from_net(_net(_SHP2))
    sim = bngsim.Simulator(model, method="ode")

    settled = np.asarray(sim.steady_state(method="integration").concentrations, dtype=float)
    laws = model._core.conservation_laws
    n_unknowns = len(laws["independent"]) if laws["n_laws"] else model._core.n_species
    assert model._core.analytical_jacobian_complete, "this measurement needs a closed-form J"

    model._core.set_state(settled)
    model._core.reset_rhs_counters()
    ss = sim.steady_state(method="newton")

    assert ss.converged
    assert ss.method_used == "newton"
    assert ss.n_steps <= 3, "seeded at the root, the polish should converge immediately"

    # KINSOL's own func-eval counter sees only the residual evaluations; the
    # difference-quotient columns show up at the model.
    dq_evals = model._core.rhs_eval_count - ss.n_rhs_evals
    assert dq_evals >= n_unknowns, (
        f"expected >= {n_unknowns} difference-quotient RHS evaluations from one "
        f"Jacobian setup, saw {dq_evals}. If KINSetJacFn was installed, this is "
        f"the docs' cue: the polish no longer differences its own Jacobian."
    )
    assert dq_evals < 3 * n_unknowns, (
        f"{dq_evals} RHS evaluations is more than a few Jacobian setups over "
        f"{n_unknowns} unknowns — the polish is costing more than documented."
    )


def test_jacobian_option_does_not_change_the_polished_root():
    """``jacobian=`` does not reach the polish, so it cannot move its answer.

    It selects how the certificate and ``dY_ss/dp`` build their matrix; both
    "auto" (closed form here) and "fd" must accept the same root.
    """
    roots = {}
    for strategy in ("auto", "fd"):
        model = bngsim.Model.from_net(_net(_SHP2))
        sim = bngsim.Simulator(model, method="ode", jacobian=strategy)
        ss = sim.steady_state(method="newton")
        assert ss.converged
        assert ss.root_stability in ("stable", "undetermined")
        roots[strategy] = np.asarray(ss.concentrations, dtype=float)
    np.testing.assert_allclose(roots["auto"], roots["fd"], rtol=1e-8)
