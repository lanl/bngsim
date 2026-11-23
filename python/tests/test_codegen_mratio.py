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
import re

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
    # a = 0 with z > 0 is summed as M(1, b+1, z) and never reaches the loop
    # (issue #768), so this is the only case that runs the compiled copy of that
    # series. mratio(0, 1, z) is (e^z - 1)/z, and the exp(-A) keeps the rate
    # tame while z sweeps from 60 down to near 0.
    ("a = 0, summed", 0.0, 1.0, "40*k*mratio(a,b,A)*exp(-A)", 60.0),
    # For b > 0, 0 < |z| <= 20 and |alpha| < 1 both Kummer series are summed
    # and the loop is never reached (issue #1080), so these two are what run the
    # compiled copy of that routine, one per sign of z: a tiny a with z near 20,
    # where the fraction lost up to 4e-7, and a tiny b with z near -20, which
    # takes Kummer's transformation. The exp() cancels the ratio's e^z.
    ("tiny a, both series", -1e-12, 0.026664156824421394, "k*mratio(a,b,A)*exp(-A)", 20.0),
    ("tiny b, both series, z < 0", -1e-12, 1e-9, "k*mratio(a,b,-A)*exp(A)", 20.0),
    # Past z = -20 with b - a < 1 the fraction is refused and the series goes on
    # to z = -700, the asymptotic route past it (issue #1080). A decays from 1500
    # to 334, so each of these crosses from one to the other halfway. The ratio
    # is about b/A, which the A/b undoes. The a = 0 one runs the asymptotic
    # route's exact M(0, b, z) = 1, which nothing else in the file reaches.
    ("tiny b - a, past z = -20", -1e-12, 1e-9, "k*mratio(a,b,-A)*A/b", 1500.0),
    ("a = 0, tiny b, past z = -20", 0.0, 1e-9, "k*mratio(a,b,-A)*A/b", 1500.0),
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


@pytest.mark.parametrize(
    "body",
    [
        "k*mratio(50,2.5,-1000)",
        # A negative b, which every clause of the rule now refuses (issue #768).
        "k*mratio(-22.52,-25.18,-12)",
        # Past z = -700 with b - a far below anything a model builds (issue
        # #1080): the fraction returned its Lentz floor there.
        "k*mratio(0,1e-300,-710)",
    ],
    ids=["positive a, large negative z", "negative b", "tiny b - a past z = -700"],
)
def test_both_copies_refuse_and_both_say_why(tmp_path, body):
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

    The negative b is the same contract for issue #768: both copies returned a
    finite wrong value there. This file is the one the MIR and Windows legs run,
    so it is what checks the compiled copy's ``b > 0`` test on those backends.
    """
    messages = {}
    for codegen in (False, True):
        with pytest.raises(Exception) as excinfo:
            _run(tmp_path, body, 1.0, 1.0, 10.0, codegen=codegen, t_end=1.0)
        messages[codegen] = str(excinfo.value)
    for codegen, message in messages.items():
        assert "not reliable" in message, f"codegen={codegen} said nothing about the refusal"
        assert "a model reaching this is using mratio in its own way" in message
        assert not re.search(r"#\d", message), message
        assert "mratio" in message
    # The compiled one reaches it the long way round, through the witness.
    assert "non-finite" in messages[True]


# ── Issue #1080: the series route, pinned to the reference on both engines ──

CONSTANT_RATE_NET = """begin parameters
    1 a  {a!r}  # Constant
    2 b  {b!r}  # Constant
    3 zz {z!r}  # Constant
    4 scale {scale!r}  # Constant
end parameters
begin functions
    1 f() mratio(a,b,zz)*scale
end functions
begin species
    1 A() 0
end species
begin reactions
    1 0 1 f
end reactions
begin groups
    1 Aobs 1
end groups
"""


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
@pytest.mark.parametrize(
    "a, b, z, true_value",
    [
        # mpmath at 60 digits. The fraction returned 2.235481406e8 at the first
        # (2.1e-7 off) and was 1.2e-8 off at the second.
        (-1e-12, 0.026664156824421394, 19.310533421401022, 223548187.99851523),
        (
            -8.607573956569877e-15,
            1.535796353807133e-08,
            -19.72305014003269,
            3.5418112969725804e-09,
        ),
    ],
    ids=["z > 0", "z < 0"],
)
def test_the_series_route_matches_the_reference_on_both_engines(
    tmp_path, codegen, a, b, z, true_value
):
    """``A' = mratio(a, b, z) / true_value`` from 0, so A(1) = 1, and a constant
    right-hand side integrates to rounding. Agreement between the two copies
    alone would pass if both were wrong the same way; this holds the compiled
    one to the reference itself, ten thousand times tighter than the old error."""
    p = tmp_path / "m1080.net"
    p.write_text(CONSTANT_RATE_NET.format(a=a, b=b, z=z, scale=1.0 / true_value))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(p))
        sim = bngsim.Simulator(model, method="ode", codegen=codegen)
        result = sim.run(t_span=(0.0, 1.0), n_points=3)
    # The compiled leg has to run the generated C, or it tests the interpreter twice.
    assert sim.codegen_backend in (("cc", "mir") if codegen else ("exprtk",))
    assert float(result.species[-1][0]) == pytest.approx(1.0, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
@pytest.mark.parametrize(
    "a, b, z, true_value",
    [
        # mpmath at 80 digits and more. The fraction was 7.2e-7 off at the
        # first, 1.1e-2 at the second and a factor of two at the third.
        (-1e-12, 1e-9, -25.0, 5.562054719123343e-11),
        (0.0, 1.1766052422876225e-12, -784.9671045705651, 1.5008374136500115e-15),
        (
            -2.4013304216952906e-13,
            1.3563113178612664e-12,
            -786022.5708171733,
            1.7255395630598238e-18,
        ),
    ],
    ids=["the series past z = -20", "a = 0 past z = -700", "a factor of two off"],
)
def test_past_z_minus_20_both_engines_match_the_reference(tmp_path, codegen, a, b, z, true_value):
    """The same constant-rate check as above, for the corner of the a <= 0,
    z <= 0 clause where b - a < 1 (issue #1080): the series to z = -700 and the
    asymptotic route past it, which in the generated C are what the trust rule's
    new clause hands these to."""
    p = tmp_path / "m1080b.net"
    p.write_text(CONSTANT_RATE_NET.format(a=a, b=b, z=z, scale=1.0 / true_value))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(p))
        sim = bngsim.Simulator(model, method="ode", codegen=codegen)
        result = sim.run(t_span=(0.0, 1.0), n_points=3)
    assert sim.codegen_backend in (("cc", "mir") if codegen else ("exprtk",))
    assert float(result.species[-1][0]) == pytest.approx(1.0, rel=1e-11, abs=0.0)


@pytest.mark.parametrize("codegen", [False, True], ids=["interpreted", "compiled"])
@pytest.mark.parametrize(
    "a, b, z, true_value",
    [
        # mpmath at 80 digits and more. The fraction returned 1e-32 at the
        # first and was 1.2e-6 and 2.1e-7 off at the other two.
        (-4.119676222070549, 2.4225094582386945e-279, -1.4540899212539904, 8.515249528874852e-280),
        (-3.0, 1e-25, -10.0, 8.55421686746988e-27),
        (2.5, 1e-25, 10.0, 4.884008591187709e-26),
    ],
    ids=["a <= 0, z <= 0", "a negative integer", "|z| <= 20, z > 0"],
)
def test_a_tiny_b_both_engines_match_the_reference(tmp_path, codegen, a, b, z, true_value):
    """The same constant-rate check, for a b far below 1e-22 with b - a >= 1,
    where the fraction itself answers and its second step is now exactly 1
    (issue #1080). This runs the compiled copy of that step."""
    p = tmp_path / "m1080c.net"
    p.write_text(CONSTANT_RATE_NET.format(a=a, b=b, z=z, scale=1.0 / true_value))
    with contextlib.redirect_stderr(io.StringIO()):
        model = bngsim.Model.from_net(str(p))
        sim = bngsim.Simulator(model, method="ode", codegen=codegen)
        result = sim.run(t_span=(0.0, 1.0), n_points=3)
    assert sim.codegen_backend in (("cc", "mir") if codegen else ("exprtk",))
    assert float(result.species[-1][0]) == pytest.approx(1.0, rel=1e-11, abs=0.0)
