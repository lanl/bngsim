"""Issue #451 — mratio has to reach the generated C as C.

``mratio`` is in the engine's reserved function list, so a model may call it,
but C has no such function and the generated source called it by name. The
compile failed with "call to undeclared function", which took down an explicit
``codegen=True`` run and every forward sensitivity run of the same model.
Issue #448 dealt with the five reserved functions that are one-line
expressions; this one is a loop, so the generated source carries the loop.

src/expression.cpp stays the single source of truth. The tests below hold the
port to that: the two are asked for the same numbers over a swept argument and
have to agree exactly, not approximately.

The routine does not answer everywhere, and the port copies which arguments it
will and will not answer for rather than diverging on one side. See issue #453,
which decided that, and issue #456, which narrowed it by adding a second method
for arguments the fraction is refused.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _BUILTIN_IDENT_MAP, _CODEGEN_PRELUDE_LINES

NET = """begin parameters
    1 a  {a!r}  # Constant
    2 b  {b!r}  # Constant
    3 k  0.05  # Constant
end parameters
begin functions
    1 law() {body}
end functions
begin species
    1 A() {a0!r}
end species
begin reactions
    1 1 0 law #_R1
end reactions
begin groups
    1 A                    1
end groups
"""


def _model(tmp_path, body: str, a: float, b: float, a0: float):
    p = tmp_path / "m.net"
    p.write_text(NET.format(a=a, b=b, body=body, a0=a0))
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(str(p))


def _run(tmp_path, body, a, b, a0, codegen, t_end=30.0):
    m = _model(tmp_path, body, a, b, a0)
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Simulator(m, method="ode", codegen=codegen).run(
            t_span=(0.0, t_end), n_points=61
        )


def _value(tmp_path, a, b, z):
    """mratio(a, b, z) straight out of the interpreter, at one point."""
    m = _model(tmp_path, f"mratio(a,b,{z!r})", a, b, 1.0)
    core = m._core if hasattr(m, "_core") else m
    return core._eval_functions(0.0, [1.0])["law"]


# ── The helper reaches the generated source ──────────────────────────────────


def test_the_helper_is_in_every_generated_source():
    """One definition, carried by the prelude every emitter writes out."""
    text = "\n".join(_CODEGEN_PRELUDE_LINES)
    assert "static double bngsim_mratio(double a, double b, double z)" in text
    # Guarded, because the sensitivity source and the right-hand side source can
    # land in one translation unit.
    assert "#ifndef BNGSIM_MRATIO_DEFINED" in text
    # No exception to throw in C, so the cap returns a value the caller already
    # knows how to react to.
    assert "return NAN;" in text


def test_the_engine_name_maps_to_the_helper():
    assert _BUILTIN_IDENT_MAP["mratio"] == ("bngsim_mratio", False)


# ── The port computes what the C++ computes ──────────────────────────────────

# Each case sweeps one argument through the species, so a single trajectory
# covers a range rather than a point. The last two move a and b, which the
# real models hold fixed, precisely because nothing else here would. The two
# with a positive first argument carry a large b, which is what keeps them
# inside the region the fraction is trusted in (issue #453); a positive a with a
# small b and a large z is refused now, and has its own test below.
CASES = [
    ("z rides the species", -3.0, 5.0, "k*mratio(a,b,-A)", 50.0),
    ("small z", -10.0, 21.0, "k*mratio(a,b,-A/10)", 20.0),
    ("positive a", 2.0, 101.0, "k*mratio(a,b,-A)", 30.0),
    ("non-integer a and b", 0.5, 101.0, "k*mratio(a,b,A/20)", 15.0),
    ("the large-argument case", -1000.0, 9001.0, "k*mratio(a,b,-A*100)", 100.0),
    # Outside the region the fraction is trusted in, so every point of this one
    # goes through the asymptotic expansion added by issue #456 rather than the
    # loop. Without it the compiled copy of that expansion is never run, and
    # c2mir has already been caught miscompiling this routine once (see the note
    # on `odd = 1 - odd` in _codegen.py).
    ("the asymptotic route", 10.0, 2.5, "k*mratio(a,b,-A*100)", 50.0),
    ("a rides the species", -3.0, 9.0, "k*mratio(-A,b,-20)", 12.0),
    ("b rides the species", -4.0, 0.0, "k*mratio(a,A+2,-8)", 10.0),
]


@pytest.mark.parametrize("tag, a, b, body, a0", CASES, ids=[c[0] for c in CASES])
def test_the_compiled_path_matches_the_interpreter(tmp_path, tag, a, b, body, a0):
    interpreted = np.asarray(_run(tmp_path, body, a, b, a0, codegen=False).species)[:, 0]
    compiled = np.asarray(_run(tmp_path, body, a, b, a0, codegen=True).species)[:, 0]
    # Not exact equality, because there is more than one compiled backend. Under
    # cc the two are bit-identical, measured. The MIR JIT may contract or reorder
    # a floating-point operation, and over a summing loop inside an integration
    # that reaches a few parts in 1e14. The bound below is three orders above the
    # worst seen and still nowhere near a real disagreement: the wrong algorithm
    # in issue #453 is off by a factor of a thousand, not by 1e-14.
    assert compiled == pytest.approx(interpreted, rel=1e-11, abs=0.0)
    # …and the trajectory moved, so the agreement is not two flat lines.
    assert abs(interpreted[-1] - interpreted[0]) > 1e-6


def test_a_sensitivity_run_builds_and_falls_back(tmp_path):
    """The shape the issue was reported in, and the fallback it should reach.

    Nothing here teaches the differentiation layer what mratio means, so the
    analytic sensitivity right-hand side is still declined and CVODES' own
    difference quotient is used. That was always the intended answer. What was
    wrong is that the rate law failed to compile, so the run never got there.
    """
    m = _model(tmp_path, "k*mratio(a,b,-A)", -3.0, 5.0, 50.0)
    with contextlib.redirect_stderr(io.StringIO()):
        result = bngsim.Simulator(m, method="ode", sensitivity_params=["k"]).run(
            t_span=(0.0, 20.0), n_points=11
        )
    sens = np.asarray(result.sensitivities)[:, 0, 0]
    assert np.all(np.isfinite(sens))
    assert np.any(sens != 0.0)
    assert m._codegen_sens_decline is not None


# ── Known values, so a rewrite that drifts is caught by more than agreement ──

# mpmath.hyp1f1(a+1, b+1, z) / mpmath.hyp1f1(a, b, z) at 40+ digits. Held here as
# literals so this does not need mpmath, and so that a change to the algorithm
# has to face the reference rather than only the other copy of itself.
REFERENCE = [
    (-3.0, 5.0, -2.0, 0.66787003611),
    (-10.0, 21.0, -50.0, 0.27332389077),
    (-100.0, 901.0, -1000.0, 0.46164590186),
    (-1000.0, 9001.0, -10000.0, 0.46128328365),
    (0.5, 101.0, 1.0, 1.00985005864),
]


@pytest.mark.parametrize("a, b, z, expect", REFERENCE)
def test_known_values_in_the_regime_models_use(tmp_path, a, b, z, expect):
    assert _value(tmp_path, a, b, z) == pytest.approx(expect, rel=1e-9)


def test_both_copies_refuse_and_both_say_why(tmp_path):
    """The region from issue #453, where the answer is a refusal in both copies.

    For a positive first argument with a large negative third one the fraction
    settles on something that is not the ratio, and here the asymptotic
    expansion added by issue #456 cannot vouch for a value either, so neither
    copy answers. The interpreter raises and names the region. The generated C
    cannot raise, so its helper returns NaN and the run fails on the right-hand
    side — and the failure carries the same explanation, because describing a
    non-finite witness re-evaluates the model at that state and passes the
    refusal on.

    Without that the compiled path reported a bare ``CV_FIRST_RHSFUNC_ERR`` and
    nothing about mratio, which is a worse answer than the wrong number it
    replaced was to diagnose.
    """
    body = "k*mratio(50,2.5,-1000)"
    messages = {}
    for codegen in (False, True):
        with pytest.raises(Exception) as excinfo:
            _run(tmp_path, body, 1.0, 1.0, 10.0, codegen=codegen, t_end=1.0)
        messages[codegen] = str(excinfo.value)
    for codegen, message in messages.items():
        assert "not reliable" in message, f"codegen={codegen} said nothing about the refusal"
        assert "#453" in message and "#456" in message
        assert "mratio" in message
    # The compiled one reaches it the long way round, through the witness.
    assert "non-finite" in messages[True]
