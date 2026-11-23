"""A cyclic function graph is solved, not swept (issue #621).

Functions that read each other are a simultaneous system. `evaluate_functions()`
used to walk them once per RHS evaluation, in whatever order the sort's cycle
fallback left — a Gauss-Seidel sweep with no convergence check and no fixed
number of steps. The value a function held therefore depended on how many times
the RHS had been called: `Model.rhs(y, t)` returned a different number on every
call at the same state, and `compute_derivs` was not a function of `(t, y)` at
all. Below a loop gain of 1 it crept toward the answer without arriving; above 1
it ran away.

`build()` now groups the functions into strongly-connected components, and a
group of two or more is solved by Newton on the residual `g(x) = F(x) - x`
rather than iterated. Newton is what the corpus forces: `MODEL1006230117`'s group
is LINEAR in its unknowns, so it has one exact solution — and a linear system is
exactly where Newton lands on it whatever the gain, while fixed-point iteration
diverges for any gain above 1. That model's gain is ~36; before this it reached
`inf` before t = 1e-13 and the solve died there.

A group of ONE — every function in every corpus model but that one — is a single
evaluation in dependency order, i.e. precisely the walk it has always been, so
nothing acyclic pays for this or changes.

The compiled path cannot express the solve, so codegen declines a cyclic graph
and the caller falls back to the interpreted engine. Before that, it emitted
`func[i] = ...` in an order that cannot exist and read an uninitialised `func[]`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _source_root import bngsim_source_root  # noqa: E402

CORPUS = "benchmarks/suites/biomodels/data/sbml_downloads/MODEL1006230117.xml"


def _cycle_model(f_expr: str, g_expr: str):
    """S decaying at rate `f`, where f and g read each other."""
    b = ModelBuilder()
    b.add_parameter("k", 1.0, "", False)
    b.add_species("S", 1.0, False, 1.0)
    b.add_function("f", f_expr)
    b.add_function("g", g_expr)
    b.add_reaction([0], [], "functional", "f", 1.0, True)
    return bngsim.Model(_core=b.build())


def _corpus_model():
    root = bngsim_source_root()
    if root is None or not (root / CORPUS).is_file():
        pytest.skip("vendored BioModels SBML corpus not present")
    return bngsim.Model.from_sbml(str(root / CORPUS))


# ─── The reported defect ─────────────────────────────────────────────────────


def test_rhs_is_a_function_of_state_not_of_call_count():
    """The headline. Five calls at one state used to give five different
    numbers, growing ~36x each time."""
    m = _corpus_model()
    y = np.array(m.get_state())
    values = [float(np.asarray(m.rhs(y, 0.0))[0]) for _ in range(12)]
    assert len(set(values)) == 1, (
        f"rhs() returned {len(set(values))} distinct values at one state: "
        f"spread {max(values) - min(values):g}"
    )


def test_a_convergent_cycle_lands_on_the_fixed_point_not_near_it():
    """`f = 0.5g+1`, `g = 0.5f+1` has the solution f = g = 2, so dS/dt = -2.

    Gain 0.5, so the old sweep converged — but only asymptotically, a different
    answer every call: -1.0, -1.75, -1.9375, -1.984375, ... It never reached -2.
    """
    m = _cycle_model("0.5*g + 1", "0.5*f + 1")
    y = np.array(m.get_state())
    for _ in range(4):
        assert float(np.asarray(m.rhs(y, 0.0))[0]) == pytest.approx(-2.0, abs=1e-9)


def test_a_divergent_cycle_is_solved_rather_than_run_away():
    """`f = 2g+1`, `g = 2f+1` solves to f = g = -1, so dS/dt = +1.

    Loop gain 4: fixed-point iteration diverges from any start, so no amount of
    sweeping finds this. Newton does, which is the whole reason it is Newton.
    """
    m = _cycle_model("2*g + 1", "2*f + 1")
    y = np.array(m.get_state())
    assert float(np.asarray(m.rhs(y, 0.0))[0]) == pytest.approx(1.0, abs=1e-9)


def test_the_corpus_model_reaches_its_closed_form_solution():
    """MODEL1006230117's caveolar group is a star: R reads LR/LRG/RG and each of
    those reads only R, so R = R_Total/(1 + LR/R + LRG/R + RG/R). The engine must
    land on that number, not on whatever a sweep left."""
    m = _corpus_model()
    m.rhs(np.array(m.get_state()), 0.0)
    suffix = "_caveolar_beta_1_adrenergic_receptor_module"
    assert m.get_param("R" + suffix) == pytest.approx(0.0171002, rel=1e-5)


def test_the_corpus_model_integrates():
    """It could not be simulated at all: the RHS reached `inf` by t ~ 9e-14 and
    CVODE aborted with CV_REPTD_RHSFUNC_ERR."""
    m = _corpus_model()
    r = bngsim.Simulator(m, method="ode").run(
        t_span=(0.0, 10.0), n_points=5, rtol=1e-8, atol=1e-10
    )
    species = np.asarray(r.species)
    assert np.all(np.isfinite(species))
    assert species[-1, 0] > species[0, 0]  # it actually moves


# ─── Scale invariance (issue #782) ───────────────────────────────────────────
#
# The solve seeds x = 0 and used to test convergence against an absolute
# 1e-12 before any Newton step, so a group whose values were all <= ~1e-12 was
# "solved" at F(0) — the answer with every other member read as 0 — and marked
# converged, silently. A cyclic model has no preferred scale; these pin that.

_SCALES = [1.0, 1e-9, 1e-11, 3e-12, 1.5e-12, 1e-12, 1e-13, 1e-20]

_DECAY_NET = """begin parameters
    1 A0 {A0}
end parameters
begin functions
    1 F1() Atot-F2()
    2 F2() 0.5*F1()
    3 kr() F2()/Atot
end functions
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kr
end reactions
begin groups
    1 Atot 1
end groups
"""


@pytest.mark.parametrize("a0", _SCALES)
def test_a_cyclic_decay_does_not_depend_on_its_scale(tmp_path, a0):
    """F2 = Atot/3 in closed form, so kr = 1/3 and A = A0*exp(-t/3) for ANY A0.

    Before the fix: A0 <= 1e-12 never decayed ([1, 1, 1, 1]); A0 = 3e-12 and
    1.5e-12 decayed until A reached 1e-12 and then froze there.
    """
    net = tmp_path / "cyc.net"
    net.write_text(_DECAY_NET.format(A0=a0))
    r = bngsim.Simulator(bngsim.Model.from_net(str(net)), "ode").run(
        (0.0, 6.0), 4, rtol=1e-10, atol=a0 * 1e-12
    )
    t = np.asarray(r.time)
    ratio = np.asarray(r.species)[:, 0] / a0
    np.testing.assert_allclose(ratio, np.exp(-t / 3.0), rtol=1e-8)


@pytest.mark.parametrize("rt", [1.0, 1e-13, 1e-30, 1e30])
def test_a_nonlinear_cycle_does_not_depend_on_its_scale(rt):
    """`f = Rt - g`, `g = f*f/Rt` solves to f = Rt*(sqrt(5)-1)/2 at any Rt.

    Nonlinear, so it also needs the FD Jacobian step to scale with the group:
    a fixed `1e-7*(|x|+1)` step is ~1e6x a 1e-13 group.
    """
    b = ModelBuilder()
    b.add_parameter("Rt", rt, "", False)
    b.add_species("S", 1.0, False, 1.0)
    b.add_function("f", "Rt - g")
    b.add_function("g", "f*f/Rt")
    b.add_reaction([0], [], "functional", "f", 1.0, True)
    m = bngsim.Model(_core=b.build())
    y = np.array(m.get_state())
    values = [float(np.asarray(m.rhs(y, 0.0))[0]) for _ in range(3)]
    assert len(set(values)) == 1
    assert -values[0] / rt == pytest.approx((5**0.5 - 1) / 2, rel=1e-10)


def test_the_corpus_model_keeps_its_ratios_at_a_tiny_receptor_total():
    """Issue #782's corpus reproduction: at R_Total = 1e-13 the muscarinic group
    reported R/R_Total = 1.0 and RG = 0. The ratios must match R_Total = 1e-9."""
    suffix = "_caveolar_muscarinic_receptor_module"

    def ratios(total):
        m = _corpus_model()
        m.set_param("R_Total" + suffix, total)
        m.rhs(np.array(m.get_state()), 0.0)
        return np.array([m.get_param(n + suffix) / total for n in ("R", "RG")])

    np.testing.assert_allclose(ratios(1e-13), ratios(1e-9), rtol=1e-8)


# ─── What must not change ────────────────────────────────────────────────────


def test_an_acyclic_function_graph_is_still_one_pass_and_exact():
    """Every group is a singleton, so this is the same ordered walk as before —
    including GH #76's case, a function declared before the one it reads."""
    b = ModelBuilder()
    b.add_parameter("k", 2.0, "", False)
    b.add_species("S", 1.0, False, 1.0)
    b.add_function("a", "b * 3")  # declared BEFORE what it reads
    b.add_function("b", "k")
    b.add_reaction([0], [], "functional", "a", 1.0, True)
    m = bngsim.Model(_core=b.build())
    y = np.array(m.get_state())
    values = [float(np.asarray(m.rhs(y, 0.0))[0]) for _ in range(3)]
    assert values == [pytest.approx(-6.0)] * 3  # a = 3b = 3k = 6


def test_codegen_declines_a_cyclic_graph_instead_of_emitting_use_before_def():
    """A cyclic graph has no assignment order, so the emitted C read an
    uninitialised `func[]` — 12 such reads on the corpus model, and the compiled
    run died on a non-finite RHS at t=0 while the interpreted one was correct.
    Declining falls back to the engine that solves it."""
    from bngsim._codegen import CodegenDeclined, _topological_function_order

    cyclic = [
        {"name": "f", "expression": "0.5*g + 1"},
        {"name": "g", "expression": "0.5*f + 1"},
    ]
    with pytest.raises(CodegenDeclined, match=r"reference each other"):
        _topological_function_order(cyclic)

    acyclic = [{"name": "a", "expression": "b*3"}, {"name": "b", "expression": "2"}]
    assert _topological_function_order(acyclic) == [1, 0]


def test_a_decline_is_not_blamed_on_an_earlier_failed_build(tmp_path, monkeypatch):
    """codegen=True's refusal names the cause ``last_codegen_error()`` reports, and
    the decline records none, so the decline has to clear it. It used to return
    before the clear: after one failed build on this thread, a cyclic model was
    refused with that build's exception as the reason. The failure is injected in
    ``generate_combined_from_model``, which the cc and MIR JIT paths both call."""
    import bngsim._codegen as cg

    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")

    def fail(*args, **kwargs):
        raise RuntimeError("simulated build failure")

    with monkeypatch.context() as mp:
        mp.setattr(cg, "generate_combined_from_model", fail)
        with pytest.raises(RuntimeError, match="simulated build failure"):
            bngsim.Simulator(_cycle_model("k", "1"), method="ode", codegen=True)

    with pytest.raises(RuntimeError, match="declined this model") as refused:
        bngsim.Simulator(_cycle_model("0.5*g + 1", "0.5*f + 1"), method="ode", codegen=True)
    assert "simulated build failure" not in str(refused.value)
    assert cg.last_codegen_error() is None


def test_the_refusal_prints_the_recorded_decline_reason(tmp_path, monkeypatch):
    """A decline records its reason (``last_codegen_decline()``), and the refusal
    prints that reason rather than inferring one from a missing error -- which
    named the #621 cycle and would have named it for any decline added later. A
    decline with a reason of its own is injected where the cycle check runs."""
    import bngsim._codegen as cg

    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")

    def decline(functions, tfun_call_by_name=None):
        raise cg.CodegenDeclined("a decline nothing else has ever raised")

    with monkeypatch.context() as mp:
        mp.setattr(cg, "_topological_function_order", decline)
        with pytest.raises(RuntimeError, match="a decline nothing else has ever raised"):
            bngsim.Simulator(_cycle_model("k", "1"), method="ode", codegen=True)
        assert cg.last_codegen_decline() == "a decline nothing else has ever raised"

    # The real cycle records its own, and a later successful build clears it.
    with pytest.raises(RuntimeError, match="a cyclic function graph has no assignment order"):
        bngsim.Simulator(_cycle_model("0.5*g + 1", "0.5*f + 1"), method="ode", codegen=True)
    assert "a cyclic function graph has no assignment order" in cg.last_codegen_decline()
    bngsim.Simulator(_cycle_model("k", "1"), method="ode", codegen=True)
    assert cg.last_codegen_decline() is None


def test_a_sensitivity_run_on_a_declined_model_names_the_decline(tmp_path, monkeypatch):
    """The sensitivity refusal inferred too: a build that returned nothing with no
    error was reported as rate laws that "could not be differentiated", and the
    only such build is the #621 decline, which has nothing to do with that."""
    import bngsim._codegen as cg
    from bngsim import SensitivityUnsupportedError

    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path / "cg")
    with pytest.raises(SensitivityUnsupportedError, match="declined this model") as refused:
        bngsim.Simulator(
            _cycle_model("0.5*g + 1", "0.5*f + 1"), method="ode", sensitivity_params=["k"]
        )
    assert "a cyclic function graph has no assignment order" in str(refused.value)
    assert "could not be differentiated" not in str(refused.value)


def test_forcing_codegen_on_a_cyclic_model_still_gives_the_right_trajectory():
    """End to end: with the auto-codegen threshold dropped so the cyclic model
    qualifies, the decline has to actually reach the fallback."""
    m = _corpus_model()
    prev = os.environ.get("BNGSIM_CODEGEN_THRESHOLD")
    os.environ["BNGSIM_CODEGEN_THRESHOLD"] = "1"
    try:
        r = bngsim.Simulator(m, method="ode").run(
            t_span=(0.0, 10.0), n_points=5, rtol=1e-8, atol=1e-10
        )
    finally:
        if prev is None:
            os.environ.pop("BNGSIM_CODEGEN_THRESHOLD", None)
        else:
            os.environ["BNGSIM_CODEGEN_THRESHOLD"] = prev
    assert np.all(np.isfinite(np.asarray(r.species)))
    assert not getattr(m, "_codegen_so_path", ""), "codegen should have declined"
