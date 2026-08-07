"""Per-species absolute tolerance for the ODE solver (issue #196).

A scalar ``atol`` is one number asked to mean the same thing for every state
variable. On a model whose species span decades that is not a tolerance, it is
a choice of which end of the model to resolve. ``Brannmark_JBC2010`` is the
worked example the issue is filed from: ``IRp`` starts at 1.8e-09 and ``X`` at
1.0e+01, 9.8 decades apart, and the two ends want absolute tolerances of
1.8e-17 and 1.0e-07. Pinned at the tight end the model does not integrate at
all; pinned at the loose end ``IRp`` sits under the noise floor for the whole
run. There is no scalar between them that is a tolerance for both, which is
why the answer is a vector and not a better scalar.

CVODE has taken one since forever (``CVodeSVtolerances``), and bngsim already
uses the vector form one axis over — the sensitivity columns go in through
``CVodeSensSVtolerances``. This module is the state-axis half: the ordering
and length contract, and the ``atol="auto"`` derivation that saves every
caller from reimplementing the same heuristic against the same models.

The vector is **positional**: entry ``i`` is the absolute tolerance for species
``i``, ordered like :attr:`bngsim.Model.species_names`. A length mismatch is an
error rather than a broadcast, because the failure mode of the alternative is
silent — species ``i`` held to the number written for species ``j``, with a
plausible trajectory to show for it.

``AUTO``, :func:`derive_atol` and :func:`normalize_atol_vector` are re-exported
from the package namespace (``bngsim.AUTO``, ``bngsim.derive_atol``,
``bngsim.normalize_atol_vector``) and *that* is the spelling to import — issue
#212. Do not import from ``bngsim._atol`` across a repository boundary.
:func:`is_scalar_atol` and ``AtolLike`` stay internal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

# Token accepted wherever ``atol`` is: derive the vector from the model's own
# initial state instead of making the caller supply one. Exported as
# ``bngsim.AUTO`` (#212) so `hasattr(bngsim, "AUTO")` feature-detects the whole
# per-species capability — the version string cannot, since the checkout that
# first carried #196 still declared 0.12.2.
AUTO = "auto"

# What every ``atol=`` argument in the public API accepts. A float is the
# pre-#196 scalar and behaves exactly as it did; a sequence is the per-species
# vector; ``"auto"`` asks bngsim to derive one from the model.
AtolLike = Union[float, Sequence[float], "NDArray[np.float64]", Literal["auto"]]

__all__ = ["AUTO", "AtolLike", "derive_atol", "is_scalar_atol", "normalize_atol_vector"]


def is_scalar_atol(atol: object) -> bool:
    """Whether ``atol`` is the plain scalar tolerance (the pre-#196 form)."""
    return isinstance(atol, (int, float)) and not isinstance(atol, bool)


def derive_atol(
    state: Sequence[float] | NDArray[np.float64],
    rtol: float,
    *,
    floor: float | None = None,
) -> NDArray[np.float64]:
    """Build a per-species absolute tolerance from a state vector.

    The rule is ``atol[i] = rtol * max(|y[i]|, floor)``: every species is
    resolved to ``rtol`` of *its own* magnitude, which is what a scalar cannot
    say and what every model spanning decades needs said.

    This derives from **the state you hand it**, which is the difference that
    matters between this function and :meth:`bngsim.Simulator.auto_atol` (and
    ``atol="auto"``, which is the same thing): those read the model's *live*
    state, the one the next ``run()`` would start from. Reach for this one when
    the tolerance has to be a constant of the *model* rather than of the point
    being integrated — a parameter fit that moves initial conditions is the
    case. There, derive once from the nominal state, hold the vector for the
    whole fit, and pass it as ``atol=``; ``"auto"`` would re-derive at every
    evaluation and put a step in the objective wherever the derivation crossed
    a rounding boundary, which is invisible in the usual way (the objective
    still looks right and only the search behaves oddly).

    ``floor`` exists for exactly one case — a species whose value here is zero
    has no magnitude to scale, and ``atol[i] = 0`` would put it under pure
    relative error control, which CVODE will not integrate the moment that
    species is genuinely zero. The default takes the smallest strictly positive
    entry of ``state``: a species with no scale of its own is treated as living
    at the smallest scale the model actually exhibits. That is the tight
    choice, deliberately — it can cost step size on a zero-initialized species
    that later grows, but it never quietly drops one below the noise floor,
    and it is the direction a caller can measure and then override.

    Parameters
    ----------
    state : array_like
        Species values to scale from, ordered like
        :attr:`bngsim.Model.species_names`. Normally the model's initial
        concentrations.
    rtol : float
        Relative tolerance the run will use. Must be > 0.
    floor : float, optional
        Magnitude to substitute for a species whose ``|y[i]|`` is below it.
        Must be > 0. Defaults to the smallest strictly positive ``|y[i]|``,
        or ``1.0`` when every entry is zero.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of ``len(state)`` absolute tolerances.

    See Also
    --------
    bngsim.Simulator.auto_atol : the same rule against the model's live state.
    bngsim.normalize_atol_vector : validate a vector you built yourself.

    Notes
    -----
    A tolerance derived from *initial* values still cannot see a species that
    starts at order one and decays to something tiny; that is a within-species,
    over-time mismatch, and the CVODE construct for it is ``CVodeWFtolerances``
    (a user-supplied error-weight function), which #196 explicitly leaves out
    of scope. What this removes is the cross-species compromise.

    Examples
    --------
    Derive once from the nominal state, then hold it across a fit that moves
    initial conditions::

        nominal = model.get_state()                     # before any fitting
        atol = bngsim.derive_atol(nominal, rtol=1e-8)   # a constant of the model

        for theta in search:
            model.set_params(theta)
            result = sim.run(t_span=(0, 100), rtol=1e-8, atol=atol)
    """
    y = np.abs(np.asarray(state, dtype=np.float64).ravel())
    if not np.all(np.isfinite(y)):
        raise ValueError(
            "cannot derive a per-species atol from a state containing NaN or inf; "
            "the magnitudes it would be scaled from are not numbers."
        )
    rtol = float(rtol)
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be finite and > 0 to derive a per-species atol, got {rtol!r}")

    if floor is None:
        positive = y[y > 0.0]
        scale_floor = float(positive.min()) if positive.size else 1.0
    else:
        scale_floor = float(floor)
        if not np.isfinite(scale_floor) or scale_floor <= 0.0:
            raise ValueError(f"floor must be finite and > 0, got {floor!r}")

    return rtol * np.maximum(y, scale_floor)


def normalize_atol_vector(
    atol: Sequence[float] | NDArray[np.float64],
    n_species: int,
    species_names: Sequence[str] | None = None,
    *,
    where: str = "atol",
) -> list[float]:
    """Validate a caller-supplied per-species ``atol`` and return it as a list.

    Every ``atol=`` entry point runs a vector through this before handing it to
    the solver, so calling it yourself buys nothing except *when* the error
    arrives. That is the point of it being public (issue #212): a caller that
    assembles its own vector — from a nominal state, from a per-species clamp,
    from a table — can take the contract check once at setup rather than at the
    first ``run()``, which on a fit is a long way from where the vector was
    built.

    Parameters
    ----------
    atol : array_like
        The candidate vector.
    n_species : int
        Length the model requires (:attr:`bngsim.Model.n_species`).
    species_names : sequence of str, optional
        Used only to make the length error concrete about which ordering the
        vector was supposed to be in (:attr:`bngsim.Model.species_names`).
    where : str
        Names the caller in the error message. Defaults to ``"atol"``; pass
        your own label to point the message at your call site.

    Returns
    -------
    list of float
        The same values as plain Python floats.

    Raises
    ------
    ValueError
        If the vector is not 1-D, is the wrong length, or holds an entry that
        is not finite and >= 0.

    See Also
    --------
    bngsim.derive_atol : build the vector this validates.
    """
    arr = np.asarray(atol, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"{where}: a per-species absolute tolerance must be 1-D, got shape {arr.shape}."
        )
    if arr.size != n_species:
        hint = ""
        if species_names is not None and n_species:
            shown = ", ".join(str(n) for n in list(species_names)[:3])
            if n_species > 3:
                shown += ", ..."
            hint = f" Species order is [{shown}] (Model.species_names)."
        raise ValueError(
            f"{where}: per-species absolute tolerance has {arr.size} entries but the "
            f"model has {n_species} species. The vector is positional — entry i is the "
            f"tolerance for species i — so a length mismatch is rejected rather than "
            f"broadcast or truncated.{hint}"
        )
    bad = ~(np.isfinite(arr) & (arr >= 0.0))
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        name = ""
        if species_names is not None and i < len(species_names):
            name = f" ('{list(species_names)[i]}')"
        raise ValueError(
            f"{where}: entry {i}{name} is {arr[i]!r}; every per-species absolute "
            f"tolerance must be finite and >= 0. CVODE rejects a negative abstol "
            f"outright, and a NaN would silently disable error control on that species."
        )
    return [float(v) for v in arr]
