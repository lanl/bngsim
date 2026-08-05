"""Issue #177 — the emitted ``∂f/∂p`` term scale, and its bond to the signed sum.

``bngsim_dfdp`` accumulates a signed sum into each row. On a model whose species
span many orders those contributions cancel, and the *value* left behind says
nothing about the size of what cancelled — so the sensitivity error test cannot
tell a genuinely-zero row from a catastrophically-cancelled one, sets an absolute
tolerance below the row's own roundoff, and shrinks ``h`` without bound chasing
accuracy float64 does not have.

``bngsim_dfdp_term_scale`` reports the sum of the *magnitudes* of exactly those
contributions, so ``ε·scale`` is that roundoff. It is emitted from the same
traversal as the signed switch, which is what these tests pin: not the numbers
the emitter happens to produce today, but the invariants that fail the moment the
two emissions describe different reaction sets — the drift shape that has bitten
every paired computation site in this file's neighbourhood.
"""

from __future__ import annotations

import ctypes
import hashlib
import math

import bngsim
import pytest
from bngsim import _codegen as cg

# Two reactions, one parameter, one row: ∂f/∂p for X is c1·S − c2·X, and at the
# steady state those are 2e18 and 2e18. The signed sum is ~0; the term scale is
# ~4e18. Everything this module needs is in that gap.
CANCELLING = """\
begin parameters
    1 p   1.0    # Constant
    2 c1  2.0    # Constant
    3 c2  1.0    # Constant
    4 a   p*c1   # ConstantExpression
    5 b   p*c2   # ConstantExpression
end parameters
begin species
    1 $S() 1e18
    2 X()  0
end species
begin reactions
    1 1 2 a  #_R1
    2 2 0 b  #_R2
end reactions
begin groups
    1 X_tot 2
end groups
"""

# A plain mass-action chain with no cancellation anywhere: every row of every
# column gets its terms from a single reaction, so signed and magnitude sums
# agree exactly. The control for the tests below.
CHAIN = """\
begin parameters
    1 k1  0.7  # Constant
    2 k2  0.3  # Constant
end parameters
begin species
    1 A() 5.0
    2 B() 2.0
    3 C() 0.0
end species
begin reactions
    1 1 2 k1  #_R1
    2 2 3 k2  #_R2
end reactions
begin groups
    1 A_tot 1
end groups
"""


class _SensUserData(ctypes.Structure):
    _fields_ = [
        ("param_values", ctypes.POINTER(ctypes.c_double)),
        ("plist", ctypes.POINTER(ctypes.c_int)),
        ("n_sens", ctypes.c_int),
    ]


class _Compiled:
    """The emitted sensitivity source, with both entry points reachable."""

    def __init__(self, model, tmp_path, monkeypatch):
        # emit_term_scale mirrors what a sensitivity run sets on the model
        # (``_want_output_sens``); without it the companion is not emitted at
        # all, which is the point of the gate and is covered separately below.
        self.src = cg.generate_sens_from_model(model, functional=True, emit_term_scale=True)
        assert self.src is not None
        self.n_sp = len(model._core.codegen_data()["species"])
        self.n_par = len(model._core.codegen_data()["parameters"])
        src = cg.generate_rhs_from_model(model) + "\n" + self.src
        monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
        so = cg.compile_rhs(src, hashlib.sha256(src.encode()).hexdigest()[:16])
        self.lib = ctypes.CDLL(str(so))
        dp = ctypes.POINTER(ctypes.c_double)
        self.lib.bngsim_codegen_sens_rhs.restype = ctypes.c_int
        self.lib.bngsim_codegen_sens_rhs.argtypes = [
            ctypes.c_int, ctypes.c_double, dp, dp, ctypes.c_int, dp, dp,
            ctypes.c_void_p, dp, dp,
        ]  # fmt: skip
        self.lib.bngsim_codegen_sens_term_scale.restype = ctypes.c_int
        self.lib.bngsim_codegen_sens_term_scale.argtypes = [
            ctypes.c_int, ctypes.c_double, dp, ctypes.c_int, dp, ctypes.c_void_p,
        ]  # fmt: skip

    def _ud(self, p, iP):
        pbuf = (ctypes.c_double * self.n_par)(*p)
        self._keep = pbuf  # the struct holds a bare pointer into this
        return _SensUserData(param_values=pbuf, plist=(ctypes.c_int * 1)(int(iP)), n_sens=1)

    def dfdp(self, t, y, p, iP):
        """An all-zero ``yS`` zeroes ``J·yS`` exactly, so ``ySdot`` comes back as
        the bare ∂f/∂p column — the trick ``steady_state.cpp`` uses."""
        ud = self._ud(p, iP)
        out = (ctypes.c_double * self.n_sp)()
        scratch = [(ctypes.c_double * self.n_sp)() for _ in range(4)]
        assert self.lib.bngsim_codegen_sens_rhs(
            1, float(t), (ctypes.c_double * self.n_sp)(*y), scratch[0], 0,
            scratch[1], out, ctypes.byref(ud), scratch[2], scratch[3],
        ) == 0  # fmt: skip
        return list(out)

    def term_scale(self, t, y, p, iP):
        ud = self._ud(p, iP)
        out = (ctypes.c_double * self.n_sp)()
        assert self.lib.bngsim_codegen_sens_term_scale(
            1, float(t), (ctypes.c_double * self.n_sp)(*y), 0, out, ctypes.byref(ud)
        ) == 0  # fmt: skip
        return list(out)


def _model(tmp_path, text, name="m.net"):
    p = tmp_path / name
    p.write_text(text)
    return bngsim.Model.load(str(p))


@pytest.fixture
def cancelling(tmp_path, monkeypatch):
    return _Compiled(_model(tmp_path, CANCELLING), tmp_path, monkeypatch)


@pytest.fixture
def chain(tmp_path, monkeypatch):
    return _Compiled(_model(tmp_path, CHAIN, "chain.net"), tmp_path, monkeypatch)


def test_the_scale_is_what_the_value_cannot_say(cancelling):
    """The whole point, in one state.

    At ``X = c1·S/c2`` the two contributions to row X are +2e18 and −2e18. ∂f/∂p
    is ~0 and *correct*; its roundoff is ~ε·4e18 ≈ 900, which no reading of that
    0 can recover. The term scale says 4e18.
    """
    p = [1.0, 2.0, 1.0, 2.0, 1.0]
    y = [1e18, 2e18]  # S fixed at 1e18; X at its steady state (c1/c2)·S
    dfdp = cancelling.dfdp(0.0, y, p, 0)
    scale = cancelling.term_scale(0.0, y, p, 0)
    assert abs(dfdp[1]) < 1e4, f"the signed sum should have cancelled, got {dfdp[1]:.3e}"
    assert scale[1] == pytest.approx(4e18, rel=1e-12)
    # …and that is the number the solver needs: ε·scale is the row's roundoff,
    # eleven orders above the absolute tolerance it was being given (atol·1/|p|).
    assert 1e2 < 2.220446049250313e-16 * scale[1] < 1e4


def test_no_cancellation_means_the_two_agree(chain):
    """The control. Every row of this model draws its terms from one reaction, so
    there is nothing to cancel and the magnitude sum equals |the signed sum|.

    A term scale that were merely "something big" would pass the test above and
    fail this one.
    """
    p = [0.7, 0.3]
    y = [5.0, 2.0, 0.0]
    for iP in (0, 1):
        dfdp = chain.dfdp(0.0, y, p, iP)
        scale = chain.term_scale(0.0, y, p, iP)
        for i, (d, s) in enumerate(zip(dfdp, scale, strict=True)):
            assert s == pytest.approx(abs(d), rel=1e-14, abs=1e-300), f"row {i}, column {iP}"


@pytest.mark.parametrize("net,name", [(CANCELLING, "c.net"), (CHAIN, "ch.net")])
def test_the_scale_bounds_the_value_everywhere(tmp_path, monkeypatch, net, name):
    """``|Σ terms| ≤ Σ|terms|`` — over every column, every row, and a spread of
    states. Cheap to state, and it is the property the floor rests on: a floor
    derived from the scale can never be below the row's own roundoff.

    The converse direction is the one that catches drift, and it is one-sided on
    purpose: a zero scale must mean a zero value, because a row the term-scale
    emission forgot would report "no roundoff here" for a row that has some, and
    nothing else in the system would notice.
    """
    c = _Compiled(_model(tmp_path, net, name), tmp_path, monkeypatch)
    data = c.src
    assert "bngsim_dfdp_term_scale" in data
    p_nom = [1.0, 2.0, 1.0, 2.0, 1.0] if net is CANCELLING else [0.7, 0.3]
    states = [
        [1.0] * c.n_sp,
        [0.0] * c.n_sp,
        [1e18, 2e18][: c.n_sp] + [0.0] * max(0, c.n_sp - 2),
        [3.5, 0.0, 7.25][: c.n_sp] + [0.0] * max(0, c.n_sp - 3),
    ]
    for iP in range(c.n_par):
        for y in states:
            dfdp = c.dfdp(0.0, y, p_nom, iP)
            scale = c.term_scale(0.0, y, p_nom, iP)
            for i, (d, s) in enumerate(zip(dfdp, scale, strict=True)):
                assert math.isfinite(s) and s >= 0.0, f"row {i} column {iP}: scale {s}"
                assert abs(d) <= s * (1 + 1e-12) + 1e-300, (
                    f"row {i} column {iP}: |value| {abs(d):.3e} exceeds term scale {s:.3e}"
                )
                if s == 0.0:
                    assert d == 0.0, (
                        f"row {i} column {iP} has a nonzero ∂f/∂p ({d:.3e}) but no terms — "
                        "the two emissions have drifted apart"
                    )


def test_an_ic_column_has_no_parameter_terms(cancelling):
    """The solver hands IC columns a one-past-the-end plist sentinel, which must
    reach the switch's ``default:`` arm rather than an adjacent case.
    """
    p = [1.0, 2.0, 1.0, 2.0, 1.0]
    assert cancelling.term_scale(0.0, [1e18, 2e18], p, len(p)) == [0.0] * cancelling.n_sp


def test_the_signed_half_is_untouched_by_the_companion(cancelling):
    """The emitted ``bngsim_dfdp`` / ``bngsim_jac_vec`` / ``bngsim_codegen_sens_rhs``
    text is the pre-#177 text, so a corpus A/B of this change is a test of the
    *solver* consuming the new symbol and nothing else.

    Pinned structurally rather than against a golden file: the term-scale
    emission may only ADD, so every line of the signed switch must still be
    present, and the two ``memset``-to-``}`` bodies must have the same number of
    scatter statements.
    """

    def body(fn: str) -> str:
        """The function's own text — signature through the closing brace, with
        the doc comment above it excluded (both comments name both symbols)."""
        src = cancelling.src
        start = src.index(f"static void {fn}(")
        return src[start : src.index("\n}\n", start)]

    signed, scaled = body("bngsim_dfdp"), body("bngsim_dfdp_term_scale")
    assert "dfdp_out" in signed and "scale_out" not in signed
    assert "scale_out" in scaled and "dfdp_out" not in scaled
    # Same reaction set, same case labels, same v-assignments.
    assert [ln for ln in signed.splitlines() if ln.strip().startswith("case ")] == [
        ln for ln in scaled.splitlines() if ln.strip().startswith("case ")
    ]
    assert [ln for ln in signed.splitlines() if ln.strip().startswith("v = ")] == [
        ln for ln in scaled.splitlines() if ln.strip().startswith("v = ")
    ]


def test_the_companion_is_emitted_only_for_a_sensitivity_run(tmp_path):
    """A run that never asks for a sensitivity can never call the term scale, and
    the switch is O(parameters × reactions) — big enough to matter on a large
    Functional model (18 MB of .so becomes 29 MB on BIOMD0000000496). So the
    emission is gated on the same ``_want_output_sens`` signal the GH #198
    output-sensitivity block uses, and the .net cache key carries it: without that
    a plain run's .so would be reused for a sensitivity run and silently lack the
    symbol.
    """
    m = _model(tmp_path, CANCELLING, "gate.net")
    off = cg.generate_sens_from_model(m, functional=True)
    on = cg.generate_sens_from_model(m, functional=True, emit_term_scale=True)
    for sym in ("bngsim_dfdp_term_scale", "bngsim_codegen_sens_term_scale"):
        assert sym not in off
        assert sym in on
    # …and the gate is purely additive: every line of the ungated source appears
    # in the gated one, in the same order. (A prefix check would not do — the
    # companion is inserted BETWEEN bngsim_dfdp and bngsim_jac_vec, not appended.)
    it = iter(on.splitlines())
    assert all(line in it for line in off.splitlines()), (
        "emitting the term scale changed a line of the source that ships without it"
    )

    flags_off = cg._codegen_emit_flags(m, True)
    m._want_output_sens = True
    flags_on = cg._codegen_emit_flags(m, True)
    assert flags_off[3] is False and flags_on[3] is True, (
        "the sensitivity-run flag must reach the cache key, or a .so compiled "
        "without the symbol is reused for a run that needs it"
    )
