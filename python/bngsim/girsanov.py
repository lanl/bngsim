"""Likelihood-ratio (Girsanov) scores and ensemble gradients from SSA runs (GH #616).

An exact SSA run that recorded its per-reaction statistics
(``Simulator(model, method="ssa", reaction_stats=True)``) carries, at every
output time ``t`` and for every reaction ``r``, the firing count ``N_r(t)`` and
the integrated propensity ``∫₀ᵗ a_r ds``. For a mass-action reaction with rate
constant ``c``, ``a_r = c · h_r(x)`` and the score of the path law with respect
to ``c`` is

    W_c(t) = (N_r(t) − ∫₀ᵗ a_r ds) / c              for ∂/∂c
    W_c(t) = ln(10) · (N_r(t) − ∫₀ᵗ a_r ds)         for ∂/∂log₁₀ c

summed over the reactions ``c`` is the rate constant of. For any function ``f``
of the state at ``t``, ``∂E[f(X_t)]/∂c = E[f(X_t) · W_c(t)]``; and because
``E[W] = 0``, the sample covariance of ``f`` and ``W`` over replicates estimates
the same gradient with a variance smaller by a factor of the observable's squared
mean-to-standard-deviation ratio. That covariance is what :func:`gradient`
computes, with its standard error.

A parameter that enters the rate constants through the model's parameter
expressions (``k = 2*k__FREE``, the ``__FREE`` alias form a fitting tool
writes) is scored through the chain rule, ``W_θ = Σ_r (∂c_r/∂θ / c_r)
(N_r − ∫ a_r ds)``, with ``∂c_r/∂θ`` taken from the model's own parameter
evaluation.

Scope: elementary (mass-action) rate laws. A functional or Michaelis--Menten
rate law makes ``∂ log a_r/∂c`` state-dependent, which the two recorded
accumulators cannot resolve, so :func:`scores` refuses such a model rather than
return a number that is wrong for it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bngsim._model import Model
    from bngsim._result import Result

__all__ = ["gradient", "rate_constant_reactions", "scores"]


def _reaction_table(model: Model) -> tuple[list[str], list[dict]]:
    d = model._core.codegen_data()
    names = [p["name"] for p in d["parameters"]]
    return names, list(d["reactions"])


def rate_constant_reactions(model: Model) -> dict[str, list[int]]:
    """Map each parameter that is an elementary rate constant to its reactions.

    Returns ``{parameter name: [0-based reaction indices]}`` over the model's
    elementary (mass-action) reactions; a parameter that is the rate constant of
    several reactions maps to all of them. Functional and Michaelis--Menten
    reactions do not appear.
    """
    names, reactions = _reaction_table(model)
    out: dict[str, list[int]] = {}
    for i, rxn in enumerate(reactions):
        if rxn["type"] != "elementary" or len(rxn["rate_param_indices"]) != 1:
            continue
        out.setdefault(names[rxn["rate_param_indices"][0]], []).append(i)
    return out


def _rate_constant_derivatives(
    model: Model, param: str, rate_constants: Sequence[str]
) -> dict[str, float]:
    """``∂c/∂param`` for each rate constant ``c``, from the model's own evaluation.

    A rate constant that *is* the parameter has derivative exactly 1. Every
    other one is re-derived by the model at ``param ± h`` (the model evaluates
    its parameter expressions in dependency order on every write), so the
    central difference is exact to rounding for the alias and linear forms a
    fitting tool writes and accurate to about 1e-10 relative otherwise. The
    parameter is restored before returning.
    """
    theta = float(model.get_param(param))
    h = 1e-6 * abs(theta) if theta != 0.0 else 1e-6
    others = [c for c in rate_constants if c != param]
    plus: dict[str, float] = {}
    minus: dict[str, float] = {}
    if others:
        try:
            model.set_param(param, theta + h)
            plus = {c: float(model.get_param(c)) for c in others}
            model.set_param(param, theta - h)
            minus = {c: float(model.get_param(c)) for c in others}
        finally:
            model.set_param(param, theta)
    out = {c: (plus[c] - minus[c]) / (2.0 * h) for c in others}
    if param in rate_constants:
        out[param] = 1.0
    return out


def scores(
    model: Model,
    result: Result,
    params: Sequence[str],
    *,
    log10: bool = False,
) -> NDArray[np.float64]:
    """Likelihood-ratio scores of the path law for named rate constants.

    Parameters
    ----------
    model : Model
        The model the run was made from, at the parameter values it ran with.
    result : Result
        A run, or a stacked batch of replicates, made with
        ``reaction_stats=True``.
    params : sequence of str
        Parameters to score. Each must set at least one elementary reaction's
        rate constant, directly or through the parameter expressions the rate
        constants are defined by (``k = 2*k__FREE``); the chain rule
        ``∂c_r/∂θ`` is applied from the model's own parameter evaluation, which
        perturbs and restores ``θ`` on the model.
    log10 : bool, optional
        Score with respect to ``log₁₀`` of each parameter instead of the
        parameter itself. Default ``False``.

    Returns
    -------
    ndarray
        ``W[..., t, p]`` for each output time and each parameter in ``params``:
        shape ``(n_times, n_params)`` for a single run, ``(n_sims, n_times,
        n_params)`` for a batch.

    Raises
    ------
    ValueError
        If the result carries no per-reaction statistics, if the model has a
        functional or Michaelis--Menten reaction (its score is not a function of
        the recorded statistics), or if a parameter sets no elementary
        reaction's rate constant.
    """
    if not result.has_reaction_stats:
        from bngsim._error_codes import pass_argument
        from bngsim._exceptions import UsageError

        raise UsageError(
            "scores() needs the per-reaction statistics: run with "
            "Simulator(model, method='ssa', reaction_stats=True)",
            remedies=[pass_argument("reaction_stats", True)],
        )
    names, reactions = _reaction_table(model)
    non_elementary = [
        f"{result.reaction_labels[i] if i < len(result.reaction_labels) else i}: {rxn['type']}"
        for i, rxn in enumerate(reactions)
        if rxn["type"] != "elementary"
    ]
    if non_elementary:
        from bngsim._exceptions import UsageError

        raise UsageError(
            "scores() is defined for mass-action rate laws only; the score of a "
            "functional or Michaelis-Menten reaction depends on the state at every "
            "firing, which the recorded statistics do not carry. Non-elementary: "
            + "; ".join(non_elementary)
        )
    by_param = rate_constant_reactions(model)
    counts = np.asarray(result.reaction_firing_counts, dtype=float)
    integrals = np.asarray(result.reaction_propensity_integrals, dtype=float)
    if counts.shape[-1] != len(reactions):
        from bngsim._exceptions import UsageError

        raise UsageError(
            f"result has statistics for {counts.shape[-1]} reactions but the model "
            f"has {len(reactions)}; score against the model the run was made from"
        )
    compensated = counts - integrals  # N_r(t) − ∫ a_r ds, per reaction
    rate_constants = sorted(by_param)
    out = np.zeros(compensated.shape[:-1] + (len(params),))
    for j, name in enumerate(params):
        if name not in names:
            from bngsim._exceptions import UsageError

            raise UsageError(
                f"{name!r} is not a parameter of the model",
                code="param.not_found",
                details={"name": name},
            )
        derivs = _rate_constant_derivatives(model, name, rate_constants)
        block = np.zeros(compensated.shape[:-1])
        touched = False
        for c, dc in derivs.items():
            if dc == 0.0:
                continue
            value = float(model.get_param(c))
            if value == 0.0:
                from bngsim._exceptions import UsageError

                raise UsageError(
                    f"rate constant {c!r} is 0, so the score with respect to a parameter "
                    "that sets it is undefined"
                )
            touched = True
            block += (dc / value) * compensated[..., by_param[c]].sum(axis=-1)
        if not touched:
            known = ", ".join(rate_constants) or "(none)"
            from bngsim._exceptions import UsageError

            raise UsageError(
                f"{name!r} sets no elementary reaction's rate constant, directly or through "
                f"a parameter expression; the rate constants are: {known}"
            )
        if log10:
            out[..., j] = math.log(10.0) * float(model.get_param(name)) * block
        else:
            out[..., j] = block
    return out


def gradient(
    values: NDArray[np.float64],
    score: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Ensemble estimate of ``∂E[f]/∂θ`` and its standard error, from scores.

    Parameters
    ----------
    values : ndarray
        ``f`` per replicate: shape ``(n_sims, n_times, n_outputs)`` (for
        example a stacked ``Result.observables``) or ``(n_sims, n_times)``.
    score : ndarray
        The matching :func:`scores`, shape ``(n_sims, n_times, n_params)``.

    Returns
    -------
    (gradient, stderr)
        Each of shape ``(n_times, n_outputs, n_params)``, or
        ``(n_times, n_params)`` when ``values`` is 2-D. The estimate is the
        sample covariance over replicates of ``f`` and the score; ``E[W] = 0``
        makes it unbiased for the gradient, and centring ``f`` removes the
        variance the raw product ``f · W`` would carry from ``E[f]²``. The
        standard error is that of the mean of the centred products.
    """
    values = np.asarray(values, dtype=float)
    score = np.asarray(score, dtype=float)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[:, :, None]
    if values.ndim != 3 or score.ndim != 3 or values.shape[:2] != score.shape[:2]:
        from bngsim._exceptions import UsageError

        raise UsageError(
            "values must be (n_sims, n_times[, n_outputs]) and score (n_sims, n_times, "
            f"n_params) over the same replicates and times; got {values.shape} and "
            f"{score.shape}"
        )
    n = values.shape[0]
    if n < 2:
        from bngsim._exceptions import UsageError

        raise UsageError("gradient() needs at least two replicates")
    centred = values - values.mean(axis=0, keepdims=True)
    products = centred[:, :, :, None] * score[:, :, None, :]  # (sims, times, outputs, params)
    grad = products.sum(axis=0) / (n - 1)
    stderr = products.std(axis=0, ddof=1) / math.sqrt(n)
    if squeeze:
        return grad[:, 0, :], stderr[:, 0, :]
    return grad, stderr
