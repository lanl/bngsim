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

The vector fixes the cross-species half and stops there: whatever number
species ``i`` gets, it keeps for the whole run. :class:`TrackingAtol` (issue
#213) is the over-time half — the same vector, re-evaluated against the state
being integrated rather than against ``t=0``, through CVODE's
``CVodeWFtolerances``.

``AUTO``, ``TRACKING``, :class:`TrackingAtol`, :func:`derive_atol` and
:func:`normalize_atol_vector` are re-exported from the package namespace
(``bngsim.AUTO``, ``bngsim.TrackingAtol``, ``bngsim.derive_atol``,
``bngsim.normalize_atol_vector``) and *that* is the spelling to import — issue
#212. Do not import from ``bngsim._atol`` across a repository boundary.
:func:`is_scalar_atol` and ``AtolLike`` stay internal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

# Token accepted wherever ``atol`` is: derive the vector from the model's own
# initial state instead of making the caller supply one. Exported as
# ``bngsim.AUTO`` (#212) so `hasattr(bngsim, "AUTO")` feature-detects the whole
# per-species capability — the version string cannot, since the checkout that
# first carried #196 still declared 0.12.2.
AUTO = "auto"

# Token accepted wherever ``atol`` is: the tracking mode of issue #213, at its
# default depth. Sugar for ``TrackingAtol()``, which is the spelling to reach
# for when the depth or the ceiling needs saying.
TRACKING = "tracking"

# How many decades below its own ceiling a species keeps being resolved
# relatively, when the caller does not say. See TrackingAtol.decades for what
# picked it.
DEFAULT_TRACKING_DECADES = 12.0


@dataclass(frozen=True)
class TrackingAtol:
    """An absolute tolerance that follows the trajectory (issue #213).

    A per-species vector (:func:`derive_atol`, ``atol="auto"``) removes the
    *cross-species* compromise: a model whose species span decades no longer
    has to pick one number that is a tolerance for the largest and the smallest
    at once. It does not touch the *within-species, over-time* version of the
    same problem. Whatever number species ``i`` is given, it keeps for the whole
    run — so a species that starts at order one and decays to something tiny
    outgrows its own tolerance partway through and stops being error-controlled
    from there on.

    CVODE's construct for that is ``CVodeWFtolerances``: instead of a fixed
    vector, the integrator is handed a callback that computes the error weights
    at the state actually being integrated. This is that mode, and the rule it
    installs is::

        atol_i(y) = clamp(rtol * |y_i|, ceiling_i * 10**-decades, ceiling_i)

    Read it as: hold every species to ``rtol`` of the magnitude it currently
    has, never looser than the vector already asked for, and stop tightening
    once it has fallen ``decades`` decades below where it started.

    Two consequences worth knowing before reaching for it:

    * **It is not free.** Resolving a decay over twelve decades is real work.
      On ``tests/data/deep_decay.net`` the step count goes 162 → 605. What it
      buys is an answer instead of noise: that species' value at the end of the
      run is 3.8e-10 under the scalar tolerance and 9.36e-14 under tracking,
      against an analytical 9.357623e-14.
    * **It cannot conjure digits that are not there.** A species whose value is
      formed as a *difference* of large fluxes carries absolute roundoff of
      order ``eps * flux``, and asking for relative accuracy below that will
      collapse the step size rather than sharpen the answer. That is what
      ``decades`` bounds, and why it is finite.

    Parameters
    ----------
    decades : float, optional
        How far below its ceiling a species keeps being resolved relatively.
        ``0`` reduces the rule to the ceiling vector exactly — the mode is a
        strict extension of :func:`derive_atol`, not a different rule that
        happens to agree somewhere. The default, 12, is where the measured
        accuracy stops improving. On ``tests/data/deep_decay.net``, worst
        relative error in the decaying species against its analytical value:

        =========  =========
        decades    rel. err.
        =========  =========
        0          1.8e+04
        3          3.6e+01
        6          9.5e-02
        8          2.0e-03
        12         2.6e-06
        16         5.0e-06
        20         9.4e-06
        =========  =========

        Past 12 it gets slightly *worse*, which is the tell that the limit
        there is roundoff rather than the tolerance — and 12 also stays clear of
        the ``eps``-relative floor where a coupled model's own roundoff starts
        failing error tests.
    ceiling : float, sequence of float, or ``"auto"``, optional
        The tolerance being tracked below — the same per-species vector the
        non-tracking path takes. ``"auto"`` (the default) derives it from the
        model's live state exactly as ``atol="auto"`` does; a float broadcasts
        to every species; a sequence is used as given. Every entry must be
        strictly positive: a zero ceiling scales to a zero floor, and a species
        then at exactly zero would have an infinite error weight.

    See Also
    --------
    bngsim.derive_atol : the ceiling this tracks below, built from a state.
    bngsim.Simulator.auto_atol : the same, against the model's live state.

    Examples
    --------
    >>> result = sim.run(t_span=(0, 30), atol=bngsim.TrackingAtol())
    >>> result = sim.run(t_span=(0, 30), atol="tracking")      # same thing
    >>> result = sim.run(t_span=(0, 30), atol=bngsim.TrackingAtol(decades=6))
    """

    decades: float = DEFAULT_TRACKING_DECADES
    ceiling: float | Sequence[float] | NDArray[np.float64] | Literal["auto"] = "auto"

    def __post_init__(self) -> None:
        decades = float(self.decades)
        if not np.isfinite(decades) or decades < 0.0:
            raise ValueError(
                f"TrackingAtol.decades must be finite and >= 0, got {self.decades!r}. "
                "A tracking tolerance with no floor is pure relative error control, "
                "which has no weight to give a species sitting at exactly zero."
            )
        object.__setattr__(self, "decades", decades)


# What every ``atol=`` argument in the public API accepts. A float is the
# pre-#196 scalar and behaves exactly as it did; a sequence is the per-species
# vector; ``"auto"`` asks bngsim to derive one from the model; a
# :class:`TrackingAtol` (or ``"tracking"``) re-evaluates that vector against the
# trajectory instead of against the initial state.
AtolLike = Union[
    float,
    Sequence[float],
    "NDArray[np.float64]",
    Literal["auto"],
    Literal["tracking"],
    TrackingAtol,
]

__all__ = [
    "AUTO",
    "DEFAULT_TRACKING_DECADES",
    "TRACKING",
    "AtolLike",
    "TrackingAtol",
    "derive_atol",
    "is_scalar_atol",
    "normalize_atol_vector",
]


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
    over-time mismatch, and what this removes is the cross-species compromise.
    :class:`bngsim.TrackingAtol` (issue #213) is the other half — it takes the
    vector built here as a *ceiling* and re-evaluates it against the trajectory,
    so a decaying species keeps a tolerance that means something for it.

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
