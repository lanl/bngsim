"""``differentiable_solve`` takes ``atol`` as ``Simulator.run`` does, and its
default warns as ``run``'s does (issue #1420).

``run()`` raises :class:`bngsim.AtolScaleWarning` only where the caller chose no
atol (#1361). ``differentiable_solve`` declared ``atol = 1e-8`` and passed that
on, so ``run()`` read it as the caller's choice and never warned. That covered
two cases where the default's per-species scale does not reach: a model whose
species all start at 0, which came back 12% off, and the run a gradient takes on
a model whose sensitivities fall back to CVODES' difference quotient, which keeps
the scalar. ``float(atol)`` also refused every other form ``run`` takes:
``"auto"``, an ``ExactAtol``, a ``TrackingAtol``, a vector, and ``"tracking"``,
the remedy the warning names. The default is ``None`` now, which is
``Simulator``'s default with its warning, and every form is passed on.
"""

from __future__ import annotations

import os
import warnings

import bngsim
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from bngsim.jax import differentiable_solve  # noqa: E402

jax.config.update("jax_enable_x64", True)

# A made at kp from nothing and lost at kd, and B made from A: A(t) =
# (kp/kd)(1 - exp(-kd t)), which settles at 1e-9, from a start where every
# species is 0. The default per-species atol has no scale to take, so it is the
# scalar 1e-8, above both species all run.
FROM_ZERO = """begin parameters
    1 kp 1e-12
    2 kd 1e-3
end parameters
begin species
    1 A() 0
    2 B() 0
end species
begin reactions
    1 0 1 kp
    2 1 0 kd
    3 1 1,2 kd
    4 2 0 kd
end reactions
"""
KP, KD, T_END = 1e-12, 1e-3, 5000.0
A_END = KP / KD * (1.0 - np.exp(-KD * T_END))
# dA(T)/dkp and dA(T)/dkd of the closed form.
A_END_GRAD = [
    (1.0 - np.exp(-KD * T_END)) / KD,
    -KP / KD**2 * (1.0 - np.exp(-KD * T_END)) + KP / KD * T_END * np.exp(-KD * T_END),
]

# #1361's binding model, species near 1e-9 M. Each takes its own scale under the
# default, the same ceiling as an explicit 1e-8.
BINDING = """begin parameters
    1 kf 1e6
    2 kr 1e-3
end parameters
begin species
    1 A() 1e-9
    2 B() 2e-9
    3 C() 0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
end reactions
"""

# The binding model plus D made at a rate law the
# analytic sensitivity RHS declines (clamp with a parameter for its upper
# bound), which puts every column on CVODES' difference quotient. A plain run
# takes each species' scale from the start and gives A(5000) = 4.142e-10; a run
# with sensitivities on the quotient keeps the scalar (Simulator._run_atol).
BINDING_ON_THE_QUOTIENT = """begin parameters
    1 kf 1e6
    2 kr 1e-3
    3 kc 1e-15
    4 big 1e300
end parameters
begin functions
    1 fD() kc*clamp(0.5,Aobs,big)
end functions
begin species
    1 A() 1e-9
    2 B() 2e-9
    3 C() 0
    4 D() 0
end species
begin reactions
    1 1,2 3 kf
    2 3 1,2 kr
    3 0 4 fD
end reactions
begin groups
    1 Aobs 1
end groups
"""
BINDING_A_END = 4.142e-10  # A(5000) at rtol 1e-12, atol 1e-22


def _load(tmp_path, text):
    path = tmp_path / "model.net"
    path.write_text(text)
    return bngsim.Model.from_net(str(path))


@pytest.fixture
def from_zero(tmp_path):
    return _load(tmp_path, FROM_ZERO)


def _scale_warnings(caught):
    return [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]


def _solve(model, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = differentiable_solve(model, jnp.array([KP, KD]), (0.0, T_END), 51, **kw)
    return np.asarray(y), _scale_warnings(caught)


@pytest.mark.parametrize("kw", [{}, {"chunk_size": 1}], ids=["one call", "chunked"])
def test_the_default_atol_warns_where_the_start_gives_no_scale(from_zero, kw):
    """It came back 12% off, with no warning, on either path."""
    y, scale = _solve(from_zero, **kw)
    assert len(scale) == 1
    assert "'A()', 'B()'" in str(scale[0].message)
    assert "atol='tracking'" in str(scale[0].message)
    assert abs(y[-1, 0] - A_END) > 0.1 * A_END  # the trajectory the warning is about


def test_an_atol_the_caller_chose_is_not_second_guessed(from_zero):
    _, scale = _solve(from_zero, atol=1e-8)
    assert scale == []


@pytest.mark.parametrize("kw", [{}, {"chunk_size": 1}], ids=["one call", "chunked"])
def test_the_remedy_the_warning_names_is_taken_and_differentiated(from_zero, kw):
    """``atol="tracking"`` was refused. It gives the closed form, value and
    gradient, with no warning."""
    y, scale = _solve(from_zero, atol="tracking", **kw)
    assert scale == []
    assert y[-1, 0] == pytest.approx(A_END, rel=1e-6)

    def a_end(p):
        return differentiable_solve(from_zero, p, (0.0, T_END), 51, atol="tracking", **kw)[-1, 0]

    grad = np.asarray(jax.grad(a_end)(jnp.array([KP, KD])))
    np.testing.assert_allclose(grad, A_END_GRAD, rtol=1e-6)


@pytest.mark.parametrize(
    "atol",
    [
        None,
        1e-8,
        "auto",
        "tracking",
        bngsim.ExactAtol(1e-8),
        bngsim.TrackingAtol(decades=6),
        [1e-20, 1e-20, 1e-20],
        np.array([1e-20, 1e-20, 1e-20]),
    ],
    ids=["None", "scalar", "auto", "tracking", "ExactAtol", "TrackingAtol", "list", "array"],
)
def test_every_atol_run_takes_reaches_run_as_given(tmp_path, atol):
    """The trajectory is ``Simulator.run``'s at the same atol, to the bit, on
    the binding model, where ``ExactAtol(1e-8)`` and the ceiling ``1e-8``
    differ. Every form but a number and None raised before."""
    p = jnp.array([1e6, 1e-3])
    y = differentiable_solve(_load(tmp_path, BINDING), p, (0.0, 5000.0), 51, atol=atol)
    run = bngsim.Simulator(_load(tmp_path, BINDING), method="ode").run(
        t_span=(0.0, 5000.0), n_points=51, atol=atol
    )
    np.testing.assert_array_equal(np.asarray(y), np.asarray(run.species))


@pytest.mark.parametrize("atol", ["fine", True], ids=["unknown token", "bool"])
def test_an_atol_run_does_not_take_is_refused(from_zero, atol):
    """By ``run``, which names what it takes. ``float(True)`` ran at 1.0."""
    with pytest.raises(ValueError, match=r"run\(atol=\.\.\.\)"):
        differentiable_solve(from_zero, jnp.array([KP, KD]), (0.0, 1.0), 3, atol=atol)


def test_the_warning_names_the_callers_line(from_zero):
    """Not a line of JAX's custom-JVP machinery, which the solve runs inside:
    ``jax/_src/api_util.py`` for the plain call and under ``jax.grad`` alike."""

    def a_end(p):
        return differentiable_solve(from_zero, p, (0.0, T_END), 51)[-1, 0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a_end(jnp.array([KP, KD]))
        jax.grad(a_end)(jnp.array([KP, KD]))
    scale = _scale_warnings(caught)
    assert len(scale) == 2
    assert {os.path.realpath(w.filename) for w in scale} == {os.path.realpath(__file__)}


def test_a_gradient_on_the_difference_quotient_is_warned(tmp_path):
    """The run a gradient takes on a model whose sensitivities fall back to the
    difference quotient keeps the scalar, and A goes negative under it. Passed
    on as 1e-8, that run said nothing. The plain call takes each species' scale
    and is right and quiet."""
    model = _load(tmp_path, BINDING_ON_THE_QUOTIENT)
    p = jnp.array([model.get_param(n) for n in model.primary_param_names])

    def a_end(q, **kw):
        return differentiable_solve(model, q, (0.0, 5000.0), 51, **kw)[-1, 0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plain = float(a_end(p))
    assert _scale_warnings(caught) == []
    assert plain == pytest.approx(BINDING_A_END, rel=1e-3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value, _ = jax.value_and_grad(a_end)(p)
    assert len(_scale_warnings(caught)) == 1
    assert float(value) < 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jax.value_and_grad(lambda q: a_end(q, atol=1e-8))(p)
    assert _scale_warnings(caught) == []


def test_the_default_trajectory_and_gradient_are_unchanged(tmp_path):
    """``None`` resolves to the Simulator's own default, the 1e-8 that was
    passed before: the numbers are the same bits."""
    model = _load(tmp_path, BINDING)

    def total(p, **kw):
        return jnp.sum(differentiable_solve(model, p, (0.0, 5000.0), 51, **kw))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.AtolScaleWarning)
        p = jnp.array([model.get_param(n) for n in model.primary_param_names])
        default = jax.value_and_grad(total)(p)
        explicit = jax.value_and_grad(lambda q: total(q, atol=1e-8))(p)
    assert float(default[0]) == float(explicit[0])
    np.testing.assert_array_equal(np.asarray(default[1]), np.asarray(explicit[1]))
