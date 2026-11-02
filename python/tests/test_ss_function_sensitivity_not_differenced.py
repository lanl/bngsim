"""A function's steady-state sensitivity is not differenced (issue #1002).

``steady_state(sensitivity_params=...)`` assembles d(func)/dp from the compiled
output-sensitivity chain rule (GH #198), the one a time course uses. Where that
has no derivative for a function (an ``if()``, an ``abs()``, a ``max()``, and
the others its reason names), the row was differenced: each species stepped by
1.5e-8 of the state's largest concentration, each parameter by √eps of itself.
#1100 and #1294 refused the solve where two or three steps disagreed, and a
quotient can be wrong at every step a check takes. The rows are not
differenced now. Since issue #1330 a row the chain rule leaves is taken by
dual numbers through the interpreter, exactly, and returned; at a steady state
on a function's change of branch it is taken from each side, and refused where
the two differ. A model with no dual evaluation (a ring of functions that read
each other, rateOf) leaves such a row NaN, and ``output_sensitivities`` refuses
it by name, with the chain rule's reason, as after a run: the ``nodual``
fixture puts a model there (BNGSIM_NO_DUAL_NUMBERS). dY_ss/dp and every function
the chain rule answers are returned as before.

Every expected value is a closed form.
"""

from __future__ import annotations

import math

import bngsim
import numpy as np
import pytest

K1 = 0.21
SQRT_EPS = 1.4901161193847656e-8
# The floored step of a species at or near 0 beside G at 1e8.
H = SQRT_EPS * 1e8

# 0 -> X at s, X -> 0 at k1: X* = s/k1 and dX*/ds = 1/k1. G is made at sg and
# removed at 1, so G* = sg and dG*/ds = 0. The functions are the slot.
NET = """begin parameters
    1 s   {s!r}
    2 k1  0.21
    3 sg  1e8
    4 dg  1.0
    5 q   0.0
end parameters
begin species
    1 X() 0
    2 G() 1e8
end species
begin groups
    1 Xo 1
    2 Go 2
end groups
begin functions
{functions}end functions
begin reactions
    1 0 1 s
    2 1 0 k1
    3 0 2 sg
    4 2 0 dg
end reactions
"""


def _steady(tmp_path, functions, s=1.3, params=("s",)):
    path = tmp_path / "m.net"
    path.write_text(NET.format(s=s, functions=functions))
    sim = bngsim.Simulator(bngsim.Model.from_net(path), method="ode")
    return sim.steady_state(sensitivity_params=list(params))


@pytest.fixture
def nodual(monkeypatch):
    """A model with no dual evaluation, whose rows the chain rule leaves are
    NaN (issue #1330)."""
    monkeypatch.setenv("BNGSIM_NO_DUAL_NUMBERS", "1")


def _refused(out, name, construct):
    with pytest.raises(ValueError, match=rf"'expression:{name}' has no steady-state") as err:
        out.output_sensitivities([name])
    assert f"uses unsupported construct: {construct}" in str(err.value)
    assert "issue #1002" in str(err.value)


def test_a_secant_the_same_at_both_steps_is_exact(tmp_path):
    """``hx = abs(X)·abs(X)/(K² + X²) + G`` at X* = 0 with K = h/√2, where
    h = 1.49 is the floored step: the secants over h and over h/2 are both
    2/(3h), so the two steps agreed, and main returned dhx/ds = 2/(3h·k1) =
    2.13 at the default settings with no error. The slope at X = 0 is 0 and
    G does not move, so dhx/ds = 0, which dual numbers give: X* = 0 is the
    kink of abs, and the row is the same from either side of it."""
    k = H / math.sqrt(2.0)
    out = _steady(tmp_path, f"    1 hx() abs(Xo)*abs(Xo)/({k!r}^2+Xo^2) + Go\n", s=0.0)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [1 / K1, 0.0], rtol=1e-9)
    assert out.sens_output_source == "automatic-differentiation"
    assert out.output_sensitivities(["hx"])[0, 0] == 0.0


def test_a_secant_the_same_at_both_steps_with_no_dual_evaluation_is_refused(tmp_path, nodual):
    """The same with no dual evaluation: the row is not differenced, and is
    refused by name with the chain rule's reason."""
    k = H / math.sqrt(2.0)
    out = _steady(tmp_path, f"    1 hx() abs(Xo)*abs(Xo)/({k!r}^2+Xo^2) + Go\n", s=0.0)
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [1 / K1, 0.0], rtol=1e-9)
    assert out.sens_output_source == "none"
    assert np.isnan(out.sensitivities_expressions).all()
    _refused(out, "hx", "abs()")


@pytest.mark.parametrize("frac", [1e-5, 1e-6])
def test_a_kink_within_the_step_is_exact(tmp_path, frac):
    """``kx = if(X > c, X - c, 0) + G`` with c a hair beyond X* = s/k1: the
    slope at X* is 0, so dkx/ds = 0. The floored step crossed the kink, and
    the half step too, so main returned dkx/ds = 4.762 (1/k1) with no error
    where the kink was 1e-5·h or nearer. Dual numbers read the branch X* is
    on, and give 0."""
    c = 1.3 / K1 + frac * H
    out = _steady(tmp_path, f"    1 kx() if(Xo > {c!r}, Xo - {c!r}, 0) + Go\n")
    np.testing.assert_allclose(np.asarray(out.sensitivity)[:, 0], [1 / K1, 0.0], rtol=1e-9)
    assert out.output_sensitivities(["kx"])[0, 0] == 0.0


def test_a_kink_within_the_step_with_no_dual_evaluation_is_refused(tmp_path, nodual):
    c = 1.3 / K1 + 1e-5 * H
    out = _steady(tmp_path, f"    1 kx() if(Xo > {c!r}, Xo - {c!r}, 0) + Go\n")
    _refused(out, "kx", "if() conditional")


def test_a_condition_on_a_parameter_at_0_is_refused(tmp_path):
    """``fq = if(q > 0, 10, 0) + X`` at q = 0 jumps there, and has no
    derivative in q. Its explicit term was differenced at a step of √eps in
    q, across the jump: main returned dfq/dq = 10/√eps = 6.7e8 with no error.
    The corpus has the same: Chitnis2008's ``sigma_v = if(Baseline_dummy_variable
    == 0, 0.33, 0.5)``, a scenario switch at 0, came back 1.1e7. Dual numbers
    meet the tie at q = 0, take the row from each side, and find the value
    jumping between them: it is refused by name (issue #1330)."""
    out = _steady(tmp_path, "    1 fq() if(q > 0, 10, 0) + Xo\n", params=("s", "q"))
    np.testing.assert_allclose(np.asarray(out.sensitivity)[0], [1 / K1, 0.0], rtol=1e-9)
    with pytest.raises(
        ValueError,
        match=r"'expression:fq' has no steady-state derivative: the steady state "
        r"sits where it changes branch.*#1330",
    ):
        out.output_sensitivities(["fq"])


def test_a_condition_on_a_parameter_at_0_with_no_dual_evaluation_is_refused(tmp_path, nodual):
    out = _steady(tmp_path, "    1 fq() if(q > 0, 10, 0) + Xo\n", params=("s", "q"))
    _refused(out, "fq", "if() conditional")


def test_a_compiled_function_beside_a_declined_one_is_returned(tmp_path):
    """``lin = 3·X + G`` has a compiled derivative, d(lin)/ds = 3/k1, and is
    returned, as on main, beside ``kx``, which main returned 4.762 for 0, and
    whose row dual numbers give exactly; the observable block is the linear
    projection it was."""
    c = 1.3 / K1 + 1e-5 * H
    out = _steady(
        tmp_path,
        f"    1 kx() if(Xo > {c!r}, Xo - {c!r}, 0) + Go\n    2 lin() 3*Xo + Go\n",
    )
    assert out.sens_output_source == "codegen+automatic-differentiation"
    np.testing.assert_allclose(out.output_sensitivities(["lin"]), [[3 / K1]], rtol=1e-9)
    np.testing.assert_allclose(out.output_sensitivities(["observable:Xo"]), [[1 / K1]], rtol=1e-9)
    assert out.output_sensitivities(["kx"])[0, 0] == 0.0


def test_a_compiled_function_beside_a_declined_one_with_no_dual_evaluation(tmp_path, nodual):
    c = 1.3 / K1 + 1e-5 * H
    out = _steady(
        tmp_path,
        f"    1 kx() if(Xo > {c!r}, Xo - {c!r}, 0) + Go\n    2 lin() 3*Xo + Go\n",
    )
    assert out.sens_output_source == "partial"
    np.testing.assert_allclose(out.output_sensitivities(["lin"]), [[3 / K1]], rtol=1e-9)
    _refused(out, "kx", "if() conditional")


def test_every_declined_function_is_returned(tmp_path):
    """At X* = 6.19, where neither the kink of ``abs`` nor that of ``max`` is,
    the two declined functions are returned, exact: d|X|/ds = dmax(X, 2)/ds =
    1/k1. Main returned both right too, differenced; Step 2 of #1002 refused
    them, at the cost of rows that were right (issue #1330)."""
    out = _steady(
        tmp_path,
        "    1 ab() abs(Xo)\n    2 lin() 3*Xo\n    3 mx() max(Xo, 2)\n",
    )
    np.testing.assert_allclose(
        out.output_sensitivities(["ab", "lin", "mx"])[:, 0], [1 / K1, 3 / K1, 1 / K1], rtol=1e-12
    )


def test_every_declined_function_is_named_at_once(tmp_path, nodual):
    """With no dual evaluation, two declined functions in one selection are
    refused in one message, each with its own construct; the compiled one
    between them is not named."""
    out = _steady(
        tmp_path,
        "    1 ab() abs(Xo)\n    2 lin() 3*Xo\n    3 mx() max(Xo, 2)\n",
    )
    with pytest.raises(ValueError) as err:
        out.output_sensitivities(["ab", "lin", "mx"])
    message = str(err.value)
    assert "'expression:ab'" in message and "abs()" in message
    assert "'expression:mx'" in message and "max()" in message
    assert "'expression:lin'" not in message


# ``P`` is made at ``kp`` and lost at ``kd*P`` (P* = 5, dP*/d(kp, kd) = (5, -25));
# ``Z`` collects what P loses at ``kz*P``, has no steady value, and is left out
# by ``mask=``.
_MASKED = (
    "compartment D = 1; species P in D; P = 0; kp = 1; kd = 0.2;\n"
    "R3: => P; kp; R4: P => ; kd*P;\n"
    "species Z in D; Z = 0; kz = 0.1; R5: P => Z; kz*P;\n"
)


def _masked_steady(text):
    m = bngsim.Model.from_antimony_string(text)
    sim = bngsim.Simulator(m, method="ode")
    return sim.steady_state(
        sensitivity_params=["kp", "kd"], mask=~np.asarray(m.is_pure_sink()), tol=1e-12
    )


def test_a_compiled_function_alone_that_reads_a_masked_species_names_it():
    """Main named Z too (with ``sens_output_source`` "finite-difference").
    ``V := 2*Z`` is the model's only function, so
    no row of the block is answered (``"none"``), though the chain rule ran: V's
    row is NaN because Z has none. The refusal names Z and the mask; an earlier
    version of this change said there was no chain rule."""
    ss = _masked_steady(_MASKED + "V := 2*Z;\n")
    assert ss.sens_output_source == "none"
    with pytest.raises(ValueError, match=r"'expression:V'.*reads Z, which mask= left out"):
        ss.output_sensitivities(["V"])


def test_a_read_the_step_cannot_see_is_refused_not_zero():
    """``V := 1e20 + Z`` with Z masked. Main stepped Z, saw ``1e20 + Z`` not
    move, took the partial for 0 and returned dV/d(kp, kd) = (0, 0) with no
    error; Z has no steady value, so V has no derivative. It is refused, and
    the refusal says that the step did not see the read."""
    ss = _masked_steady(_MASKED + "V := 1e20 + Z;\n")
    assert np.isnan(ss.sensitivities_expressions).all()
    with pytest.raises(
        ValueError,
        match=r"'expression:V'.*not found to read a species mask= left out of the solve "
        r"\(Z\)\. Either it reads one through a term too small",
    ):
        ss.output_sensitivities(["V"])


def test_a_row_nan_of_its_own_in_the_compiled_code_is_exact():
    """``V := sqrt(W) + P`` with W held at 0 and Z masked: V reads no masked
    species, and its compiled row is NaN for inf·0 (the slope of sqrt at 0
    times W's row of 0). Main differenced it, right: (10/3, -100/9). Dual
    numbers leave the slope of a quantity that does not move out of the
    product, and give that exactly (issue #1330)."""
    ss = _masked_steady(_MASKED + "species W in D; W = 0;\nV := sqrt(W) + P;\n")
    np.testing.assert_allclose(ss.output_sensitivities(["V"])[0], [1 / 0.3, -1 / 0.09], rtol=1e-9)


def test_a_row_nan_of_its_own_beside_a_mask_is_not_said_to_read_it(nodual):
    """The same with no dual evaluation: it is refused, and the refusal does
    not say that V reads Z: it gives both reasons a row can have here."""
    ss = _masked_steady(_MASKED + "species W in D; W = 0;\nV := sqrt(W) + P;\n")
    with pytest.raises(ValueError, match=r"'expression:V'.*not found to read") as err:
        ss.output_sensitivities(["V"])
    assert "or its derivative is not a number at this state" in str(err.value)


@pytest.mark.parametrize(
    "rule, found",
    [("Z*Z + P", True), ("1e20 + Z*P", False)],
    ids=["read", "unseen"],
)
def test_a_rule_target_whose_rule_reads_a_masked_species_says_so(rule, found):
    """``species T := <rule>`` with Z masked: T's row is its rule's, and the
    rule reads Z, so T has no steady value. Main refused ``species:T`` saying
    T reads no masked species, and for ``1e20 + Z*P``, whose read a step of Z
    cannot see, returned (0, 0) with no error. Both are refused, naming the
    rule, and Z where the step saw it."""
    ss = _masked_steady(_MASKED + f"species T in D; T := {rule};\n")
    with pytest.raises(ValueError, match=r"'species:T'.*its rule 'T'") as err:
        ss.output_sensitivities(["species:T"])
    if found:
        assert "its rule 'T' reads Z, which mask= left out" in str(err.value)
    else:
        assert "its rule 'T' was not found to read" in str(err.value)
    np.testing.assert_allclose(
        ss.output_sensitivities(["species:P"])[0], [1 / 0.3, -1 / 0.09], rtol=1e-7
    )


def test_a_function_the_derivation_budget_cut_is_exact(monkeypatch, tmp_path):
    """Where the build-time derivation runs out of its budget (GH #97), the
    compiled code has no derivative for a function. Main differenced it (right
    here: T = P², (50, -250)); dual numbers give it exactly, with no budget
    (issue #1330). So for a compartment sized by such a rule: T = 2P in
    C = 1 + P is the concentration 2P/(1 + P), and d/dθ of it is
    2/(1 + P)²·dP/dθ = (5/18, -25/18) at P = 5."""
    from bngsim import _codegen as cg

    monkeypatch.setenv("BNGSIM_SENS_DERIV_BUDGET_S", "1e-9")
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
    base = (
        "compartment D = 1; species P in D; P = 0; kpb = 1; kdb = 0.2;\n"
        "R3: => P; kpb; R4: P => ; kdb*P;\n"
    )
    m = bngsim.Model.from_antimony_string(base + "species T in D; T := P*P;\n")
    ss = bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=["kpb", "kdb"])
    np.testing.assert_allclose(ss.output_sensitivities(["species:P"])[0], [5, -25], rtol=1e-6)
    np.testing.assert_allclose(ss.output_sensitivities(["species:T"])[0], [50, -250], rtol=1e-9)

    sized = base + "C := 1 + P;\nsubstanceOnly species T in C; T := 2*P;\n"
    m = bngsim.Model.from_antimony_string(sized)
    ss = bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=["kpb", "kdb"])
    np.testing.assert_allclose(
        ss.output_sensitivities(["species:T"])[0], [5 / 18, -25 / 18], rtol=1e-9
    )


def test_a_function_the_derivation_budget_cut_is_refused_with_that_reason(
    monkeypatch, tmp_path, nodual
):
    """With no dual evaluation it is refused by name, with the budget's reason
    readable in the sentence, and a compartment sized by such a rule refuses
    the solve without advice to write the rule without a construct, which is
    not what is wrong with it."""
    from bngsim import _codegen as cg

    monkeypatch.setenv("BNGSIM_SENS_DERIV_BUDGET_S", "1e-9")
    monkeypatch.setattr(cg, "CACHE_DIR", tmp_path)
    base = (
        "compartment D = 1; species P in D; P = 0; kpb = 1; kdb = 0.2;\n"
        "R3: => P; kpb; R4: P => ; kdb*P;\n"
    )
    m = bngsim.Model.from_antimony_string(base + "species T in D; T := P*P;\n")
    ss = bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=["kpb", "kdb"])
    np.testing.assert_allclose(ss.output_sensitivities(["species:P"])[0], [5, -25], rtol=1e-6)
    with pytest.raises(
        ValueError, match=r"'species:T'.*has none for its rule 'T' \(the build-time"
    ) as err:
        ss.output_sensitivities(["species:T"])
    assert "derivation exceeded" in str(err.value)

    sized = base + "C := 1 + P;\nsubstanceOnly species T in C; T := 2*P;\n"
    m = bngsim.Model.from_antimony_string(sized)
    with pytest.raises(bngsim.SimulationError, match=r"its size C has no steady-state") as err:
        bngsim.Simulator(m, method="ode").steady_state(sensitivity_params=["kpb", "kdb"])
    assert "derivation exceeded" in str(err.value)
    assert "written without that construct" not in str(err.value)
