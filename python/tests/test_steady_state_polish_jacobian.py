"""What Jacobian the steady-state solver tiers actually use (issue #127).

Both tiers install one. ``SteadyStateMarcher``'s constructor calls
``CVodeSetJacFn`` and ``solve_by_newton`` calls ``KINSetJacFn`` whenever the
model has a closed form and ``jacobian=`` asks for it, so ``jacobian=`` now
reaches the *solvers* as well as the two consumers that need the matrix itself
(``dY_ss/dp`` and the issue #78 stability certificate).

What this file pins is the gate and its consequence, not a wall-clock number:
the difference-quotient columns disappear under ``jacobian="auto"`` and come
back under ``jacobian="fd"``, in both tiers, and the root does not move between
them. Before #127 neither callback existed and the polish differenced its own
Jacobian at one RHS evaluation per unknown per setup — measured here as the
``"fd"`` half, which is still exactly what that escape hatch selects.
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


def _assert_same_root(a, b, msg: str = "") -> None:
    """The two solves landed on the same steady state.

    Scale-relative, because a per-component relative comparison answers the
    wrong question here. ``jacobian=`` now selects the *Newton matrix*, so the
    two polishes are genuinely different iterations converging to the same root:
    they agree to ~1e-9 of the state's own magnitude, which on a species sitting
    at 1e-15 of that magnitude is a 0.2% relative difference and on one that
    integration left at 4.7e-10 while the polish drove it to exactly 0 is 100%.
    Neither is a different root. The tolerance is on the state's scale.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7 * scale, err_msg=msg)


def _seeded_polish(jacobian: str):
    """One KINSOL polish, no integration, with the model's RHS calls counted.

    Seeding the model AT its steady state makes the two-tier ladder take its
    early-exit branch: exactly one polish runs, and the #78 certificate reads
    the closed form without evaluating the RHS at all. Every model-level RHS
    call beyond the ones KINSOL reports is therefore a difference-quotient
    Jacobian column.
    """
    model = bngsim.Model.from_net(_net(_SHP2))
    sim = bngsim.Simulator(model, method="ode", jacobian=jacobian)

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
    return ss, model._core.rhs_eval_count - ss.n_rhs_evals, n_unknowns


def test_polish_installs_the_closed_form_jacobian():
    """No difference-quotient columns: KINSOL is handed the matrix instead."""
    ss, dq_evals, n_unknowns = _seeded_polish("auto")

    assert ss.solver_jacobian_source == "analytical"
    assert dq_evals < n_unknowns, (
        f"{dq_evals} RHS evaluations beyond KINSOL's own over {n_unknowns} unknowns — "
        f"that is a difference-quotient Jacobian, so KINSetJacFn did not take."
    )


def test_jacobian_fd_still_differences_the_polish():
    """``jacobian="fd"`` is the escape hatch, and it still reaches KINSOL.

    This is the pre-#127 behavior, kept selectable: one RHS evaluation per
    unknown per Jacobian setup.
    """
    ss, dq_evals, n_unknowns = _seeded_polish("fd")

    assert ss.solver_jacobian_source == "finite-difference"
    assert dq_evals >= n_unknowns, (
        f"expected >= {n_unknowns} difference-quotient RHS evaluations from one "
        f"Jacobian setup under jacobian='fd', saw {dq_evals}."
    )
    assert dq_evals < 3 * n_unknowns, (
        f"{dq_evals} RHS evaluations is more than a few Jacobian setups over "
        f"{n_unknowns} unknowns."
    )


def test_march_installs_the_closed_form_jacobian():
    """The CVODE march too — its setups are the other half of issue #127.

    Counted as a ratio rather than an absolute: the two strategies march the
    same trajectory to the same criterion, so the difference in model-level RHS
    calls is the difference-quotient Jacobian and nothing else.
    """
    evals = {}
    for strategy in ("auto", "fd"):
        model = bngsim.Model.from_net(_net(_SHP2))
        sim = bngsim.Simulator(model, method="ode", jacobian=strategy)
        model._core.reset_rhs_counters()
        ss = sim.steady_state(method="integration")
        assert ss.converged
        assert ss.solver_jacobian_source == (
            "analytical" if strategy == "auto" else "finite-difference"
        )
        evals[strategy] = model._core.rhs_eval_count

    assert evals["auto"] < evals["fd"], (
        f"the march evaluated the RHS {evals['auto']} times with the closed-form "
        f"Jacobian and {evals['fd']} times differencing it — CVodeSetJacFn did not take."
    )


def test_jacobian_option_does_not_change_the_polished_root():
    """``jacobian=`` now reaches the polish, but it may not move its answer.

    The Newton matrix decides how the iteration gets there; the accepted root is
    whatever satisfies the same ``||f||_2/n < tol``, and both strategies have to
    land on it.
    """
    roots = {}
    for strategy in ("auto", "fd"):
        model = bngsim.Model.from_net(_net(_SHP2))
        sim = bngsim.Simulator(model, method="ode", jacobian=strategy)
        ss = sim.steady_state(method="newton")
        assert ss.converged
        assert ss.root_stability in ("stable", "undetermined")
        roots[strategy] = np.asarray(ss.concentrations, dtype=float)
    _assert_same_root(roots["auto"], roots["fd"])


def test_reduced_projection_agrees_with_the_differenced_reduced_residual():
    """The reduced polish solves the same system either way (issue #127).

    KINSOL's difference quotient differentiates the *reduced* residual directly,
    so it gets the conservation-law projection for free. The closed-form fill is
    of the full ``ns x ns`` system and has to be projected by hand
    (``ss_reduce_jacobian``, shared with ``dY_ss/dp`` and the #78 certificate).
    A projection that did not match would be a Newton matrix for a different
    system: seeded away from the root, the two would take different iteration
    counts to different residuals.
    """
    laws_seen = False
    out = {}
    for strategy in ("auto", "fd"):
        model = bngsim.Model.from_net(_net(_SHP2))
        laws = model._core.conservation_laws
        laws_seen = laws["n_laws"] > 0
        sim = bngsim.Simulator(model, method="ode", jacobian=strategy)

        # A seed off the root, but in its basin: scale the IC so the polish has
        # real Newton work to do rather than confirming a root it was handed.
        ic = np.asarray(model._core.get_state(), dtype=float)
        model._core.set_state(ic * 1.5)
        ss = sim.steady_state(method="newton")
        assert ss.converged
        out[strategy] = ss

    assert laws_seen, "this test is about the conservation-law reduction"
    _assert_same_root(out["auto"].concentrations, out["fd"].concentrations)
    # An exact Newton matrix cannot need materially more iterations than a
    # difference quotient of the same system; a mis-projected one would.
    assert out["auto"].n_steps <= out["fd"].n_steps + 2, (
        f"the closed-form polish took {out['auto'].n_steps} nonlinear iterations "
        f"against the differenced polish's {out['fd'].n_steps} — the reduced "
        f"projection does not match the residual KINSOL solves."
    )
