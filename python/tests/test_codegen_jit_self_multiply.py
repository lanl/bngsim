"""Issue #413 — a rate law that squares a value must not smash the JIT's stack.

``x = x * x`` reaches MIR's register allocator as the three-operand
``dmul r, r, r``. When ``r`` is spilled, ``try_spilled_reg_mem`` (mir-gen.c)
rewrites every operand naming it into a memory operand and records the rewritten
indices in ``int op_nums[MAX_INSN_RELOAD_MEM_OPS]`` — a **two**-element array,
bounded upstream by ``gen_assert`` alone. ``gen_assert`` is ``assert``, so
``-DNDEBUG`` — every release build, including bngsim's — deletes it, and the
third match stores past the end of the array. glibc's stack protector catches
that as ``*** stack smashing detected ***`` and raises SIGABRT, which no
``except`` can see and which kills the whole pytest session; a toolchain that
leaves that frame unguarded corrupts it silently instead. **The platforms that
passed were never evidence of safety** — macOS and Windows simply lay the frame
out differently.

The fix is a bounds check carried locally in ``third_party/mir/mir-gen.c``; it
declines the memory operand once the table is full, which is the same fallback
the function already takes when ``target_insn_ok_p`` rejects the rewrite. See
``VENDOR.json`` → ``local_carries`` and ``test_mir_vendoring.py``.

Two things this file is careful about, both learned from the investigation:

* **Sensitivity is not the trigger, it is what turns codegen on.** The original
  reproducer needed ``sensitivity_params`` only because a four-species model is
  far below the auto-codegen threshold, so a plain ODE run never JITs anything.
  The same ``x*x`` in the same plain RHS overflows the same array once it *is*
  JIT'd. Both are covered below.
* **The overflow needs register pressure, not just the multiply.** A one-line
  ``ydot[0] = t*t`` does not spill and does not reach the bug, so the fixture is
  a whole rate law inside a real model rather than the smallest expression that
  contains a square.

The end-to-end tests are what actually reproduced #413; they are meaningful only
where MIR is built and selected, and are skipped otherwise. The emitted-source
test runs everywhere and keeps the fixture honest — if the codegen ever stops
emitting a self-multiply here, the reproducer would pass while covering nothing.
"""

from __future__ import annotations

import os

import bngsim
import numpy as np
import pytest
from bngsim import _codegen as cg

pytest.importorskip("sympy")

# `beta*I*time()*time()` — the minimal law from the #413 comment matrix. The
# counter species and unused parameters are what the investigation's variant
# matrix shared; they keep the emitted functions big enough for the allocator to
# spill, which is the precondition the bug needs.
SQUARED_TIME = """\
begin parameters
    1 S0      1000  # Constant
    2 I0      1  # Constant
    3 beta    0.002  # Constant
    4 gamma   0.15  # Constant
    5 sigma   3.0  # Constant
    6 kclock  1  # Constant
    7 thresh  40.0  # Constant
end parameters
begin functions
    1 betaI() BODY
end functions
begin species
    1 person(state~S) S0
    2 person(state~I) I0
    3 person(state~R) 0
    4 counter() 0
end species
begin reactions
    1 1 2 betaI #_R1
    2 2 3 gamma #_R2
    3 0 4 kclock #_R3
end reactions
begin groups
    1 S                    1
    2 I                    2
    3 R                    3
    4 t                    4
end groups
"""

# Both shapes the #413 matrix found, plus the linear-in-t control that passed on
# every leg before the fix and must keep passing after it.
BODIES = {
    # No if(), no threshold, no comparison — just t*t. Admitted by the #68 gate,
    # so this one also emits bngsim_codegen_sens_rhs.
    "bare-square": "beta*I*time()*time()",
    # Refused by the #68 gate (a clock threshold quadratic in t), so it emits no
    # sens RHS at all — and smashed anyway. Kept because it is the shape that
    # first surfaced this, and because it exercises the square inside a ternary.
    # Since issue #414 a *sensitivity* run over it is refused up front (an
    # uncompensated moving crossing), so its self-multiply-under-JIT coverage runs
    # through the plain-ODE leg below instead; the sensitivity leg asserts the
    # refusal (test_the_square_in_condition_sensitivity_run_is_refused).
    "square-in-condition": "if(time()*time()>=thresh,beta,0)*I",
    # Linear in t: no self-multiply, never affected. The negative control.
    "linear": "if(time()>=6.3,beta,0)*I",
}

MIR_SELECTED = getattr(bngsim, "HAS_MIR", False) and (
    os.environ.get("BNGSIM_CODEGEN_JIT", "").strip().lower() == "mir"
)
needs_mir_jit = pytest.mark.skipif(
    not MIR_SELECTED,
    reason="needs a -DBNGSIM_ENABLE_MIR=ON build selected with BNGSIM_CODEGEN_JIT=mir",
)


def _model(tmp_path, body, name):
    net = tmp_path / f"{name}.net"
    net.write_text(SQUARED_TIME.replace("BODY", body))
    return bngsim.Model.from_net(net)


def _self_multiplies(c_source: str) -> int:
    """Occurrences of a value multiplied by itself, in the forms the emitter
    produces for ``time()*time()``: ``t*t`` from the value path and ``(t)*(t)``
    from the differentiated path."""
    return c_source.count("t*t") + c_source.count("(t)*(t)")


@pytest.mark.parametrize("key", ["bare-square", "square-in-condition"])
def test_the_fixture_emits_a_self_multiply(tmp_path, key):
    """Backend-independent: keeps the reproducers below honest.

    If the emitter ever rewrites ``time()*time()`` to something that is not a
    register times itself (``pow(t, 2)``, a common subexpression hoisted to a
    different register), these models stop reaching the allocator path #413 is
    about and the runs below would pass while covering nothing.
    """
    model = _model(tmp_path, BODIES[key], key)
    model._want_output_sens = True
    src = cg.prepare_codegen_source(model._net_path, model, emit_jac=True)
    assert _self_multiplies(src) > 0, (
        f"{key}: the emitted C no longer multiplies a value by itself, so it no "
        f"longer reproduces #413. Pick a rate law that does."
    )


def test_the_control_emits_no_self_multiply(tmp_path):
    model = _model(tmp_path, BODIES["linear"], "linear")
    model._want_output_sens = True
    src = cg.prepare_codegen_source(model._net_path, model, emit_jac=True)
    assert _self_multiplies(src) == 0


@needs_mir_jit
@pytest.mark.parametrize("key", ["bare-square", "linear"])
def test_a_sensitivity_run_does_not_smash_the_stack(tmp_path, key):
    """The #413 reproducer. Before the fix this did not fail — it aborted the
    interpreter, so there is nothing subtler to assert than "it returned".

    ``square-in-condition`` is deliberately absent: issue #414 refuses a
    sensitivity run over its uncompensated moving crossing before the solve, so
    it can no longer reach the JIT here. Its identical self-multiply-under-JIT is
    covered by the plain-ODE leg below and its refusal by
    ``test_the_square_in_condition_sensitivity_run_is_refused``."""
    model = _model(tmp_path, BODIES[key], key)
    params = ["beta", "gamma"]
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=params)
    run = sim.run(t_span=(0.0, 12.0), n_points=61, rtol=1e-11, atol=1e-11)

    sens = np.asarray(run.sensitivities)
    assert sens.shape == (61, model._core.n_species, len(params))
    assert np.isfinite(sens).all()


def test_the_square_in_condition_sensitivity_run_is_refused(tmp_path):
    """Issue #414. ``if(time()*time()>=thresh, ...)`` is a clock threshold #48's
    affine solver cannot invert for t* and #150 cannot root on (it reads no live
    state), so its crossing moves uncompensated and the difference quotient the
    declined analytic RHS falls back to is wrong at it. bngsim refuses the
    sensitivity run rather than return that gradient. Backend-independent (the
    refusal is raised before any solve), so unlike the reproducers above this
    needs no MIR JIT — which is also why the self-multiply this shape carries is
    exercised under JIT through the plain-ODE leg instead."""
    model = _model(tmp_path, BODIES["square-in-condition"], "square-in-condition")
    sim = bngsim.Simulator(model, method="ode", sensitivity_params=["beta", "gamma"])
    with pytest.raises(bngsim.SensitivityUnsupportedError, match="crossing time moves"):
        sim.run(t_span=(0.0, 12.0), n_points=61, rtol=1e-11, atol=1e-11)


@needs_mir_jit
@pytest.mark.parametrize("key", ["bare-square", "square-in-condition"])
def test_a_plain_ode_run_over_the_same_source_does_not_smash(tmp_path, key):
    """Sensitivity was never the trigger — it was what forced codegen on.

    A four-species model is far under the auto-codegen threshold, so the plain
    ODE leg of the original matrix ran interpreted and could not have smashed.
    ``codegen=True`` JITs the plain RHS over the identical rate law, and the
    overflow is in ``bngsim_codegen_rhs`` there.
    """
    model = _model(tmp_path, BODIES[key], key)
    sim = bngsim.Simulator(model, method="ode", codegen=True)
    # Without this the test degrades into the interpreted RHS and asserts
    # nothing — which is exactly how the original matrix read "plain ODE is
    # clean" off a run that had never JIT'd anything.
    assert sim._codegen_c_source, "codegen=True did not produce JIT source"

    run = sim.run(t_span=(0.0, 12.0), n_points=61, rtol=1e-11, atol=1e-11)

    obs = np.asarray(run.observables)
    assert obs.shape == (61, run.n_observables)
    assert np.isfinite(obs).all()
