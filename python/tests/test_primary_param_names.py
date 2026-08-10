"""Issue #227 — ``primary_param_names`` must be exactly the model's knobs.

The accessor's own docstring says to hand this list to an external optimizer or
sampler, and ``bngsim.jax.differentiable_solve`` takes it as the default
differentiation set (``flat=False``). It was wrong in both directions at once,
and ``benchmarks/models/net/ode/SIR.net`` — 23 lines — showed both.

**A literal-valued constant was omitted.** The loader decided "derived" by
whether the value text parses as a float, so ``gamma 1/7`` was derived and the
list dropped it. But ``1/7`` references nothing: there is no primary underneath
it to be fitted instead, ``set_param`` moves the trajectory through it, and a
sensitivity column for it is live. BNG2.pl draws the line the other way and
annotates that line ``# Constant``, reserving ``# ConstantExpression`` for a
value that names another parameter — which is the rule #181 gave the codegen
``.net`` parser, so the loader and the codegen disagreed about the same 626
lines. Classification now happens once, in ``ModelBuilder::build()``, *after* the
expression is evaluated: a parameter whose expression names nothing is folded to
the constant it is, which is also what keeps ``gamma`` at ``1/7`` rather than the
``1.0`` a partial ``stod("1/7")`` leaves behind.

**Every function name was listed as a knob.** Each function gets a parameter slot
to hold its evaluated value, and the engine rewrites that slot from the
function's own expression before every derivative evaluation. So ``set_param``
was accepted, ``get_param`` echoed the new value back, the trajectory did not
move, and ``jax.grad`` spent a coordinate returning exactly ``0.0`` forever. The
slot is now ``is_internal`` — the flag #170 gave ``_V0_<comp>``, for the same
reason and with the same two consequences: out of this list, and a
value-changing write is refused instead of silently discarded.

The other half of that binding — a function whose name is a parameter the *input*
declared, which is how an SBML ``<assignmentRule>`` arrives — was issues #256 and
#266 (the same defect reported from the SBML side and the ``.net`` side), and is
fixed now: the function binds to the *declared* slot and ``evaluate_functions()``
overwrites it exactly as it overwrites a synthesized one, so the number in the
``parameters`` block is the seed that slot holds until the function first
evaluates, never a knob.

Why the sweeps below did not catch it, which is the lesson worth keeping: they
glob ``*.net``, and the shape is overwhelmingly an SBML phenomenon. Zero of the
140 tracked ``.net`` files carry it; **107 of 327 tracked SBML models do**
(``BIOMD0000000613`` alone leaked 141 names). The gap was never
tracked-vs-untracked, it was one input format's sweep standing in for both — so
``tests/data/shadowed_function_param.net`` now carries the shape in the format
these sweeps actually read.
"""

from __future__ import annotations

import math
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._bngsim_core import ModelBuilder

_SIR = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "net" / "ode" / "SIR.net"
_NET_TREE = Path(__file__).resolve().parents[2]
#: The #266 shape, in the format these sweeps read. See the module docstring for
#: why a tracked fixture was needed: the defect lives in SBML `<assignmentRule>`
#: models (107 of 327 tracked), and zero tracked `.net` files carried it.
_SHADOWED = _NET_TREE / "tests" / "data" / "shadowed_function_param.net"
_NETS = (
    sorted(p for p in _NET_TREE.rglob("*.net") if "build" not in p.parts)
    if _NET_TREE.is_dir()
    else []
)


# ── A self-contained model carrying one of each kind ────────────────────────
#
# Two independent decays, so every column below has a closed form and nothing
# here is a finite difference:
#
#     A' = -k_const A,  A(0) = 100,  k_const = "1/7"    -> A(t) = 100 exp(-t/7)
#     B' = -k_der   B,  B(0) = 100,  k_der   = "2*k_base"
#
# `k_const` is the `gamma 1/7` shape: arithmetic that names nothing. `k_der` is
# the `_rateLaw{N} = chi*kon` shape that the whole derived-parameter machinery
# exists for. `pi_`, `t_dep` and `obs_ref` are the three ways an expression can
# fail to be a load-time constant, one of which (`pi_`) is not one.


def _mixed_model() -> bngsim.Model:
    b = ModelBuilder()
    b.add_parameter("k_const", 0.0, expression="1/7", is_expression=True)
    b.add_parameter("k_base", 0.5)
    b.add_parameter("k_der", 0.0, expression="2*k_base", is_expression=True)
    b.add_parameter("pi_", 0.0, expression="2*asin(1)", is_expression=True)
    a = b.add_species("A", 100.0)
    s_b = b.add_species("B", 100.0)
    b.add_reaction([a], [], "elementary", "k_const")
    b.add_reaction([s_b], [], "elementary", "k_der")
    b.add_observable("Atot", [(a, 1.0)])
    b.add_parameter("obs_ref", 0.0, expression="2*Atot", is_expression=True)
    b.add_parameter("t_dep", 0.0, expression="2*time()", is_expression=True)
    return bngsim.Model(b.build())


def _function_model() -> bngsim.Model:
    """``C' = -flux``, ``flux() = kf*Ctot`` — the `betaI` shape."""
    b = ModelBuilder()
    b.add_parameter("kf", 0.25)
    c = b.add_species("C", 100.0)
    b.add_observable("Ctot", [(c, 1.0)])
    b.add_function("flux", "kf*Ctot")
    b.add_reaction([c], [], "functional", "flux")
    return bngsim.Model(b.build())


def _run(model, t_end=4.0, n=5):
    """Integrate from the initial conditions the *current* parameters imply."""
    model.reset()
    return np.asarray(bngsim.Simulator(model, method="ode").run((0.0, t_end), n).species)


# ── Direction 1: a constant written as arithmetic is a knob ─────────────────


def test_a_referenceless_expression_is_a_primary_and_holds_its_value():
    """``1/7`` names nothing, so it is a constant — and it is exactly ``1/7``.

    The value assertion is not incidental. The flag is what gets the expression
    compiled at all, so the demotion has to happen after the evaluation; done in
    the other order this parameter would hold the ``1.0`` that ``stod("1/7")``
    stops at, and the trajectory below would be wrong by a factor of seven with
    every flag reporting correctly.
    """
    m = _mixed_model()
    kinds = dict(zip(m.param_names, m.param_is_expression, strict=True))

    assert kinds["k_const"] is False
    assert "k_const" in m.primary_param_names
    assert m.get_param("k_const") == 1.0 / 7.0

    # `2*asin(1)` is the `pi = 2*asin(1)` line BNG2.pl calls `# Constant`: a
    # built-in function of a literal is still a literal.
    assert kinds["pi_"] is False
    assert "pi_" in m.primary_param_names
    assert m.get_param("pi_") == math.pi


def test_a_reference_is_what_makes_a_parameter_derived():
    """The three ways an expression can name something that moves.

    ``k_der`` names a parameter — the chain rule the derived machinery exists
    for. ``obs_ref`` names an observable and ``t_dep`` names the clock; neither
    is a load-time constant, so neither may be folded into one even though
    neither has a *parameter* underneath it either.
    """
    m = _mixed_model()
    kinds = dict(zip(m.param_names, m.param_is_expression, strict=True))
    primary = set(m.primary_param_names)

    for name in ("k_der", "obs_ref", "t_dep"):
        assert kinds[name] is True, f"{name} references a live symbol"
        assert name not in primary


def test_the_omitted_knob_reaches_the_trajectory():
    """``A(t) = 100 exp(-t/7)``, and writing ``k_const`` moves it accordingly.

    This is the claim the omission contradicted: the parameter is a working knob
    by every measure other than the list. The oracle is the closed-form solution,
    at both the nominal value and a written one.
    """
    m = _mixed_model()
    t = np.linspace(0.0, 4.0, 5)

    A = _run(m)[:, 0]
    np.testing.assert_allclose(A, 100.0 * np.exp(-t / 7.0), rtol=1e-6)

    m.set_param("k_const", 0.5)
    assert m.get_param("k_const") == 0.5
    A2 = _run(m)[:, 0]
    np.testing.assert_allclose(A2, 100.0 * np.exp(-0.5 * t), rtol=1e-6)


def test_the_omitted_knob_gets_a_sensitivity_column():
    """``dA/dk_const = -100 t exp(-k t)`` — the column a fit was never handed.

    A column existed the whole time when asked for by name; what the defect cost
    was the *default*, which is ``primary_param_names``. So this asks for the
    default and checks that this parameter's column is in it and is the analytic
    derivative.
    """
    m = _mixed_model()
    res = bngsim.Simulator(m, method="ode").compute_all_sensitivities((0.0, 4.0), 5, n_workers=1)

    assert "k_const" in res.sensitivity_params
    col = res.sensitivity_params.index("k_const")
    t = np.linspace(0.0, 4.0, 5)
    got = np.asarray(res.sensitivities)[:, 0, col]
    np.testing.assert_allclose(got, -100.0 * t * np.exp(-t / 7.0), rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not _SIR.exists(), reason=f"benchmark model not present: {_SIR}")
def test_sir_is_the_issues_own_reproduction():
    """The 23-line model from #227, both directions in four assertions."""
    m = bngsim.Model.load(str(_SIR))

    assert m.param_names == ["S0", "I0", "beta", "gamma", "betaI"]
    # `beta = 1/S0` names a parameter and stays derived; `gamma = 1/7` does not.
    assert m.param_is_expression == [False, False, True, False, False]
    assert m.primary_param_names == ["S0", "I0", "gamma"]
    assert m.get_param("gamma") == 1.0 / 7.0


# ── The rule is BNG2.pl's own, and now only one place decides it ────────────


@pytest.mark.skipif(not _NETS, reason=f".net models not present under {_NET_TREE}")
def test_the_loader_agrees_with_the_codegen_net_parser():
    """One rule, two readers — the disagreement #227 reported is closed.

    ``_classify_parameter_kinds`` (#181) is the codegen ``.net`` parser's answer
    to the same question, reached from the file text. This is the loaded model's
    answer, reached from ``ModelBuilder``. They partition the parameter block the
    same way on every ``.net`` in the tree, which is what makes the model-based
    and text-based codegen paths emit the same sensitivity RHS for a file.

    Function slots are excluded because they are not in the parameters block at
    all — the loader synthesizes them, and #227 flags them ``is_internal``.
    """
    from bngsim._codegen import _parse_net_file

    compared = 0
    for path in _NETS:
        try:
            parsed = _parse_net_file(str(path))
            m = bngsim.Model.load(str(path))
        except Exception:  # a few fixtures are deliberately malformed
            continue
        text_derived = {name for _, name, _, is_const in parsed["parameters"] if not is_const}
        model_derived = {
            n
            for n, f in zip(m.param_names, m.param_is_expression, strict=True)
            if f and n not in m._internal_param_names()
        }
        if not parsed["parameters"]:
            continue
        compared += 1
        assert text_derived == model_derived, f"{path.name}: {text_derived ^ model_derived}"

    assert compared > 50, f"only {compared} .net models compared"


# ── Direction 2: a function is not a knob ───────────────────────────────────


def test_a_functions_slot_is_not_a_knob():
    m = _function_model()

    assert m.function_names == ["flux"]
    # The slot stays a parameter *index*: the runtime parameter array is sized by
    # it and every sensitivity plist indexes into it. What changed is the claim
    # that it is a knob.
    assert "flux" in m.param_names
    assert "flux" not in m.primary_param_names
    assert dict(zip(m.param_names, m.param_is_internal, strict=True))["flux"] is True
    assert m.primary_param_names == ["kf"]


def test_writing_a_functions_slot_is_refused_and_the_message_names_the_reason():
    """It used to be accepted, and *that* was the defect.

    ``get_param`` echoed the write back and the trajectory did not move, which is
    the worst shape a no-op can take. The refusal is value-changing only, so a
    full parameter-vector round trip — the sequence ``Result.gradient``'s own
    docstring recommends — still goes through untouched.
    """
    m = _function_model()
    before = _run(m)

    with pytest.raises(ValueError, match=r"flux.*function"):
        m.set_param("flux", 3.0)
    assert m.get_param("flux") != 3.0

    # An unchanged write is not a change: the round trip still works.
    m.set_params({n: m.get_param(n) for n in m.param_names})
    np.testing.assert_array_equal(_run(m), before)


def test_the_default_sensitivity_set_drops_the_slot_and_says_which_kind():
    m = _function_model()
    with pytest.warns(UserWarning, match=r"hold a FUNCTION's evaluated value"):
        res = bngsim.Simulator(m, method="ode").compute_all_sensitivities(
            (0.0, 4.0), 5, n_workers=1
        )
    assert res.sensitivity_params == ["kf"] == m.primary_param_names


@pytest.mark.skipif(not _NETS, reason=f".net models not present under {_NET_TREE}")
def test_no_net_model_in_the_tree_leaks_a_function_name():
    """The leak was universal — the issue sampled 327 function-carrying models
    and found every one leaking every function name — so a sweep is the right
    shape of check, and a model *with* functions has to be reached or the sweep
    proves nothing."""
    with_functions = 0
    for path in _NETS:
        try:
            m = bngsim.Model.load(str(path))
        except Exception:
            continue
        if not m.function_names:
            continue
        with_functions += 1
        leaked = set(m.function_names) & set(m.primary_param_names)
        assert not leaked, f"{path.name}: {sorted(leaked)}"

    assert with_functions > 20, (
        f"only {with_functions} function-carrying models reached under {_NET_TREE}"
    )
    # `_NETS` is an rglob of the repo root, so the corpus is whatever `.net`
    # files this machine happens to have — 140 in a clean checkout, 677+ on a box
    # that has run the benchmark suites. That is what made #266 red on one
    # developer's pre-push hook and green in CI, so name the one file the sweep
    # must always have found: without it, a green run here means only that
    # nothing on *this* machine carries the shape.
    assert _SHADOWED in _NETS, (
        f"{_SHADOWED.name} is missing from the {len(_NETS)} .net files found under "
        f"{_NET_TREE} — this sweep cannot see the #266 shape without it"
    )


# ── The two flags stay disjoint, and together they define the list ──────────


@pytest.mark.skipif(not _NETS, reason=f".net models not present under {_NET_TREE}")
def test_primary_is_param_names_minus_the_two_flags():
    """``compute_all_sensitivities`` computes its derived list as the *residue*
    of this identity, so a name that carried both flags would be reported under
    neither reason. Before #227 the internal class was empty on every ``.net``
    model, which made the disjointness free; this is the sweep that says it
    survived the class being populated."""
    checked = 0
    for path in _NETS:
        try:
            m = bngsim.Model.load(str(path))
        except Exception:
            continue
        names = set(m.param_names)
        if not names:
            continue
        checked += 1
        derived = {n for n, f in zip(m.param_names, m.param_is_expression, strict=True) if f}
        internal = {n for n, f in zip(m.param_names, m.param_is_internal, strict=True) if f}
        assert not derived & internal, f"{path.name}: derived ∩ internal"
        assert set(m.primary_param_names) == names - derived - internal, path.name

    assert checked > 50, f"only {checked} .net models checked under {_NET_TREE}"


# ── A declared row a same-named function shadows (issues #256 / #266) ────────
#
# Driven from a tracked fixture rather than the sweeps above, which cannot be
# relied on to contain the shape: they glob `*.net`, and this is an SBML
# `<assignmentRule>` phenomenon that reaches `.net` only through conversion.


@pytest.fixture
def shadowed():
    return bngsim.Model.load(str(_SHADOWED))


def test_shadowed_literal_row_is_not_a_knob(shadowed):
    """`supply 0.0` + `supply()` — the shape #227 left behind.

    A literal, so nothing about its *syntax* marks it; the only thing that makes
    it a non-knob is that a function is bound to its slot.
    """
    flags = dict(zip(shadowed.param_names, shadowed.param_is_internal, strict=True))
    assert flags["supply"] is True
    assert "supply" not in shadowed.primary_param_names
    assert set(shadowed.primary_param_names) == {"kdeg", "kmax"}


def test_shadowed_arithmetic_row_is_internal_and_not_derived(shadowed):
    """`recycle 1/4` + `recycle()` — excluded before, but for the wrong reason.

    The `.net` reader guesses ``is_expression`` from the value text, so this row
    was already kept out of the list — and reported to the user as a *derived*
    parameter, "not independent of its primaries" (issue #203's warning). It is
    not derived: nothing recomputes it from primaries, a function overwrites it.
    The two flags also have to stay disjoint, since ``primary_param_names`` is
    the residue of subtracting both.
    """
    internal = dict(zip(shadowed.param_names, shadowed.param_is_internal, strict=True))
    derived = dict(zip(shadowed.param_names, shadowed.param_is_expression, strict=True))
    assert internal["recycle"] is True
    assert derived["recycle"] is False


def test_the_seed_is_a_seed_and_the_function_drives_the_run(shadowed):
    """What makes the row a seed rather than a knob, shown rather than asserted.

    ``supply`` is the rate of the zero-order source feeding A. Its declared value
    is ``0.0``: a run that used the number in the ``parameters`` block would leave
    A at zero forever. A moves, so the function's value is what reaches the RHS.
    """
    assert shadowed.get_param("supply") == 0.0

    res = bngsim.Simulator(shadowed).run(t_span=(0.0, 5.0), n_points=3)
    a = np.asarray(res.species)[:, shadowed.species_names.index("A()")]

    assert a[0] == 0.0
    assert a[-1] > 1.0, "A never grew — the 0.0 seed reached the RHS, not the function"


def test_set_param_refuses_the_shadowed_row(shadowed):
    """It used to be accepted, echoed back by ``get_param``, and discarded.

    The message has to work for a row the model *declared* — the pre-#266 text
    said the name "is not a parameter of the model", which is exactly what a
    reader looking at their own ``parameters`` block would dispute.
    """
    with pytest.raises(Exception, match=r"a function of that name owns this slot"):
        shadowed.set_param("supply", 10.0)

    with pytest.raises(Exception, match=r"issue #266"):
        shadowed.set_param("recycle", 10.0)
