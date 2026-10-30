"""``differentiable_solve`` at its default atol warns as ``Simulator`` does
(issue #1420).

#1379 warns where the default scalar atol sits above species that matter, and
only where the caller chose no atol. ``differentiable_solve`` declared ``atol =
1e-8`` and passed that on, so ``run()`` read it as the caller's choice and kept
quiet: #1361's binding model in molar units came back A(5000) = -8.7e-10, for
4.1e-10, with no warning, and the gradient was taken through that trajectory.
The default is ``None`` now, which is ``Simulator``'s default with its warning;
``"auto"`` is passed on as ``run`` takes it.
"""

from __future__ import annotations

import warnings

import bngsim
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from bngsim.jax import differentiable_solve  # noqa: E402

jax.config.update("jax_enable_x64", True)

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
PARAMS = (1e6, 1e-3)
A_REFERENCE = 4.142e-10  # A(5000) at rtol 1e-10, atol 1e-16


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "binding.net"
    path.write_text(BINDING)
    return bngsim.Model.from_net(str(path))


def _solve(model, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = differentiable_solve(model, jnp.array(PARAMS), (0.0, 5000.0), 51, **kw)
    scale = [w for w in caught if issubclass(w.category, bngsim.AtolScaleWarning)]
    return np.asarray(y), scale


@pytest.mark.parametrize("kw", [{}, {"chunk_size": 1}], ids=["one call", "chunked"])
def test_the_default_atol_warns_on_a_molar_model(model, kw):
    """It came back with no warning on either path."""
    y, scale = _solve(model, **kw)
    assert len(scale) == 1
    assert y[-1, 0] < 0  # the trajectory the warning is about


def test_an_atol_the_caller_chose_is_not_second_guessed(model):
    _, scale = _solve(model, atol=1e-8)
    assert scale == []


def test_auto_scales_the_tolerance_and_is_differentiated(model):
    """``float("auto")`` raised before."""
    y, scale = _solve(model, atol="auto")
    assert scale == []
    assert y[-1, 0] == pytest.approx(A_REFERENCE, rel=1e-3)

    def a_end(p):
        return differentiable_solve(model, p, (0.0, 5000.0), 51, atol="auto")[-1, 0]

    grad = np.asarray(jax.grad(a_end)(jnp.array(PARAMS)))
    assert np.all(np.isfinite(grad)) and grad[0] < 0 < grad[1]


def test_the_default_trajectory_and_gradient_are_unchanged(model):
    """``None`` resolves to the Simulator's own default, the 1e-8 that was
    passed before: the numbers are the same bits."""

    def total(p, **kw):
        return jnp.sum(differentiable_solve(model, p, (0.0, 5000.0), 51, **kw))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bngsim.AtolScaleWarning)
        p = jnp.array(PARAMS)
        default = jax.value_and_grad(total)(p)
        explicit = jax.value_and_grad(lambda q: total(q, atol=1e-8))(p)
    assert float(default[0]) == float(explicit[0])
    np.testing.assert_array_equal(np.asarray(default[1]), np.asarray(explicit[1]))


def test_an_atol_string_other_than_auto_is_refused(model):
    with pytest.raises(ValueError, match="'auto' or None"):
        differentiable_solve(model, jnp.array(PARAMS), (0.0, 1.0), 3, atol="fine")
