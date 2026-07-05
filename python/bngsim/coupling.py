"""bngsim.coupling — the hardened state-exchange layer (GH #102 Stage 1).

Stage 0 gave an external orchestrator a clean per-step drive of a *single*
bngsim network (:class:`bngsim.ReactionKernel`: ``set_state → advance(dt) →
get_state``). Stage 1 is the **exchange layer** that makes a *real two-subset
hybrid split* correct and ergonomic — when an orchestrator couples, say, a
deterministic ODE subset and a stochastic SSA subset over a set of shared
species, the raw storage vectors that flow across that boundary need units,
addressing, discretization, and conservation handled explicitly. This module
provides those primitives, all in count/amount space and all framework-agnostic:

* :class:`UnitConverter` — bulk **count ↔ concentration ↔ amount** conversion
  over the whole state vector, using each species's load-time ``volume_factor``
  with an optional **live-volume override** on the concentration views (so a
  framework that grows a compartment can still read/write concentrations
  correctly even though the SSA engine bakes a static volume).
* :class:`CouplingMap` — shared-species **name ↔ index** addressing, so the
  orchestrator reads/writes only the *coupling subset* by name across two
  subsets whose species are ordered differently.
* :class:`DiscreteExchange` / :func:`round_to_counts` — an explicit, inspectable
  **rounding policy** at the SSA/NFsim hand-off, with leak accounting (an
  error-feedback *carry* so repeated continuous→discrete round-trips do not
  shed mass on average).
* :class:`ConservationLedger` / :func:`moiety_total` — **conservation / no-leak**
  checks across the exchange boundary.
* :class:`Divider` — molecule **partitioning at cell division** (a pure
  count-space op; binomial / multinomial / deterministic, exact integer
  conservation) and :func:`get_compartment_volume` / :func:`set_compartment_volume`
  for **framework volume-growth → compartment-volume coupling**.
* :func:`make_subset_model` — the **ODE-subset-as-model** helper: reconstruct a
  reaction-subset model with the other operator's species marked ``fixed=True``,
  so a *static* partition needs no native subset integration.

Units, in one place
-------------------
bngsim stores each species as a raw *storage* value (what
:meth:`bngsim.Model.get_state` returns). The load-time per-species
``volume_factor`` ``V_c`` is the storage→amount factor::

    amount        = storage * volume_factor          # molecule number; volume-invariant
    concentration = amount / V = storage * V_c / V    # needs a volume V (default V_c → storage)
    count         = round(amount)                     # the discrete view of amount

For ``.net`` / ``V=1`` models ``volume_factor == 1``, so storage == amount ==
count and concentration == count / V. Because **amount (and therefore count) is
volume-invariant**, two subsets exchange shared-species state in *count/amount*
space and the conversion never needs a live volume. The live-volume override is
required only at a *concentration*-speaking boundary (a framework reporting
mol/L into a compartment it is itself growing): there the SSA subset's baked
``V_c`` is stale, so :meth:`UnitConverter.to_concentrations` /
:meth:`UnitConverter.from_concentrations` accept the current ``volume``. (The
SSA *propensities* still use the baked ``V_c`` — tracking live volume inside the
stochastic dynamics is Stage 2; the exchange layer is correct in count space
regardless.)

See ``benchmarks/kernel/operator_split_example.py`` for the headline two-subset
operator-split acceptance demo and ``python/tests/test_coupling.py`` for the
conservation / round-trip / divider invariants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from bngsim._exceptions import BngsimError
from bngsim._model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from bngsim.kernel import ReactionKernel

__all__ = [
    "ConservationError",
    "ConservationLedger",
    "CouplingMap",
    "DiscreteExchange",
    "Divider",
    "RoundingPolicy",
    "UnitConverter",
    "get_compartment_volume",
    "make_subset_model",
    "moiety_total",
    "round_to_counts",
    "set_compartment_volume",
]

RoundingPolicy = Literal["nearest", "floor", "ceil", "stochastic"]
"""Discretization rule for the continuous→count hand-off.

``"nearest"`` rounds half away from zero (matching the SSA engine's entry
rounding, :func:`round_initial_population_to_storage` in ``ssa_simulator.cpp``);
``"floor"`` / ``"ceil"`` truncate; ``"stochastic"`` rounds up with probability
equal to the fractional part (unbiased in expectation, needs an RNG).
"""


class ConservationError(BngsimError):
    """A conserved moiety drifted beyond tolerance across the exchange boundary."""


def _resolve_model(obj: Model | ReactionKernel) -> Model:
    """Accept a :class:`Model` or anything exposing ``.model`` (a kernel)."""
    if isinstance(obj, Model):
        return obj
    inner = getattr(obj, "model", None)
    if isinstance(inner, Model):
        return inner
    raise TypeError(f"expected a bngsim.Model or ReactionKernel, got {type(obj).__name__}")


def _species_codegen(model: Model) -> list[dict]:
    """Per-species codegen records (name/fixed/volume_factor/amount_valued/...).

    Ordered like :attr:`Model.species_names` and :meth:`Model.get_state` — the
    *full* species vector, with no GH #71 ``reported`` filtering (the bulk state
    API moves every species, reported or not).
    """
    return list(model._core.codegen_data()["species"])


# ─── Unit conversion ─────────────────────────────────────────────────────────


class UnitConverter:
    """Bulk count ↔ concentration ↔ amount conversion over the state vector.

    Wraps the per-species storage→amount factor vector ``volume_factor`` (``V_c``;
    :attr:`Species.volume_factor`, baked at load time) and converts a whole
    ``get_state`` / ``set_state`` storage vector to and from the units an external
    orchestrator works in. See the module docstring for the unit algebra; the
    short version is ``amount = storage * V_c`` (volume-invariant) and
    ``concentration = amount / V`` (needs a volume).

    Build one with :meth:`from_model` (or :meth:`from_kernel`); it is immutable
    and cheap to keep alongside a kernel.

    Parameters
    ----------
    volume_factors : array_like
        Per-species ``V_c``, ordered like the model's state vector. Must be
        finite and > 0.
    names : sequence of str, optional
        Species names parallel to ``volume_factors`` (for error messages and
        :meth:`for_species`). Not required for the array conversions.
    """

    __slots__ = ("_vf", "_n", "_names", "_name_to_idx")

    def __init__(self, volume_factors: ArrayLike, *, names: Sequence[str] | None = None) -> None:
        vf = np.asarray(volume_factors, dtype=np.float64)
        if vf.ndim != 1:
            raise ValueError(f"volume_factors must be 1-D, got shape {vf.shape}")
        if not np.all(np.isfinite(vf)) or np.any(vf <= 0.0):
            raise ValueError("volume_factors must all be finite and > 0")
        self._vf = vf
        self._n = vf.shape[0]
        self._names = list(names) if names is not None else None
        if self._names is not None:
            if len(self._names) != self._n:
                raise ValueError(
                    f"names length {len(self._names)} != volume_factors length {self._n}"
                )
            self._name_to_idx = {n: i for i, n in enumerate(self._names)}
        else:
            self._name_to_idx = {}

    @classmethod
    def from_model(cls, model: Model | ReactionKernel) -> UnitConverter:
        """Gather the per-species ``volume_factor`` vector from a model/kernel."""
        m = _resolve_model(model)
        species = _species_codegen(m)
        vf = [float(s.get("volume_factor", 1.0)) for s in species]
        names = [str(s["name"]) for s in species]
        return cls(vf, names=names)

    #: Alias — a kernel exposes ``.model``, so :meth:`from_model` already accepts
    #: it; kept for symmetry with the rest of the API.
    from_kernel = from_model

    @property
    def volume_factors(self) -> NDArray[np.float64]:
        """Per-species ``V_c`` (a copy)."""
        return self._vf.copy()

    @property
    def n_species(self) -> int:
        """Length of the state vector this converter expects."""
        return self._n

    @property
    def names(self) -> list[str] | None:
        """Species names parallel to the state vector, if supplied."""
        return list(self._names) if self._names is not None else None

    def _check_len(self, arr: NDArray[np.float64], what: str) -> NDArray[np.float64]:
        if arr.ndim != 1 or arr.shape[0] != self._n:
            raise ValueError(
                f"{what} must be a 1-D array of length {self._n}, got shape {arr.shape}"
            )
        return arr

    def _volume(self, volume: ArrayLike | None) -> NDArray[np.float64]:
        """Resolve a live-volume override to a per-species vector.

        ``None`` ⇒ the baked ``V_c``. A scalar broadcasts to every species (a
        single growing compartment, the common case). A length-``n`` array is a
        per-species live volume.
        """
        if volume is None:
            return self._vf
        v = np.asarray(volume, dtype=np.float64)
        if v.ndim == 0:
            v = np.full(self._n, float(v))
        elif v.shape != (self._n,):
            raise ValueError(
                f"volume override must be a scalar or a length-{self._n} array, "
                f"got shape {v.shape}"
            )
        if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
            raise ValueError("volume override must be finite and > 0")
        return v

    # amount ↔ storage — volume-invariant; no override (amount is how storage is
    # *defined*: storage = amount / V_c, so amount = storage * V_c always).

    def to_amounts(self, storage: ArrayLike) -> NDArray[np.float64]:
        """Storage vector → molecule amounts (``storage * V_c``)."""
        s = self._check_len(np.asarray(storage, dtype=np.float64), "storage")
        return s * self._vf

    def from_amounts(self, amounts: ArrayLike) -> NDArray[np.float64]:
        """Molecule amounts → storage vector (``amounts / V_c``)."""
        a = self._check_len(np.asarray(amounts, dtype=np.float64), "amounts")
        return a / self._vf

    # count ↔ storage — amount, discretized. Rounding is a *policy*; this is the
    # plain nearest-integer view. Use DiscreteExchange for leak-accounted rounding.

    def to_counts(
        self,
        storage: ArrayLike,
        *,
        policy: RoundingPolicy = "nearest",
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Storage vector → integer molecule counts (rounded amounts)."""
        return round_to_counts(self.to_amounts(storage), policy=policy, rng=rng)

    def from_counts(self, counts: ArrayLike) -> NDArray[np.float64]:
        """Integer molecule counts → storage vector (``counts / V_c``)."""
        return self.from_amounts(counts)

    # concentration ↔ storage — needs a volume. Default volume = V_c recovers the
    # at-load concentration (== storage); pass the live volume to track growth.

    def to_concentrations(
        self, storage: ArrayLike, *, volume: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Storage vector → concentrations (``storage * V_c / volume``).

        ``volume=None`` returns the at-load concentration (== ``storage``). Pass
        the **current** compartment volume to report concentrations against a
        framework-grown compartment whose baked ``V_c`` is now stale.
        """
        return self.to_amounts(storage) / self._volume(volume)

    def from_concentrations(
        self, concentrations: ArrayLike, *, volume: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Concentrations → storage vector (``concentration * volume / V_c``).

        Inverse of :meth:`to_concentrations`; pass the same ``volume`` you would
        read the concentration against.
        """
        c = self._check_len(np.asarray(concentrations, dtype=np.float64), "concentrations")
        return c * self._volume(volume) / self._vf

    def for_species(self, names: Sequence[str]) -> UnitConverter:
        """A sub-converter over ``names`` (must have been supplied at build)."""
        if self._names is None:
            raise ValueError("this UnitConverter has no species names; pass names= to subset it")
        try:
            idx = [self._name_to_idx[n] for n in names]
        except KeyError as e:
            raise KeyError(f"species {e.args[0]!r} not in this converter") from None
        return UnitConverter(self._vf[idx], names=list(names))

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        uniform = "uniform" if np.allclose(self._vf, self._vf[0]) else "mixed"
        return f"UnitConverter(n_species={self._n}, V_c={uniform})"


# ─── Shared-species addressing ───────────────────────────────────────────────


class CouplingMap:
    """Name ↔ index addressing of the shared coupling subset within a state vector.

    Two coupled subsets share species *by name* but order their state vectors
    independently. A :class:`CouplingMap` pins the index of each shared name in
    one subset's vector, so the orchestrator gathers / scatters only the coupling
    subset and exchanges it in a single canonical order across subsets.

    Parameters
    ----------
    all_names : sequence of str
        The full state-vector ordering of one subset (its ``species_names``).
    shared_names : sequence of str
        The coupling subset, in the canonical exchange order. Every name must
        appear in ``all_names``; duplicates are rejected.
    """

    __slots__ = ("_shared", "_idx", "_n_full")

    def __init__(self, all_names: Sequence[str], shared_names: Sequence[str]) -> None:
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        if len(name_to_idx) != len(all_names):
            raise ValueError("all_names contains duplicate species names")
        shared = list(shared_names)
        if len(set(shared)) != len(shared):
            raise ValueError("shared_names contains duplicates")
        missing = [n for n in shared if n not in name_to_idx]
        if missing:
            raise KeyError(f"shared species not in the state vector: {missing}")
        self._shared = shared
        self._idx = np.array([name_to_idx[n] for n in shared], dtype=np.intp)
        self._n_full = len(all_names)

    @classmethod
    def from_model(cls, model: Model | ReactionKernel, shared_names: Sequence[str]) -> CouplingMap:
        """Build from a model/kernel's :attr:`species_names`."""
        m = _resolve_model(model)
        return cls(m.species_names, shared_names)

    from_kernel = from_model

    @property
    def names(self) -> list[str]:
        """The shared species, in exchange order."""
        return list(self._shared)

    @property
    def indices(self) -> NDArray[np.intp]:
        """Indices of the shared species into the full state vector (a copy)."""
        return self._idx.copy()

    @property
    def n_shared(self) -> int:
        """Number of shared (coupling) species."""
        return self._idx.shape[0]

    @property
    def n_full(self) -> int:
        """Length of the full state vector this map addresses."""
        return self._n_full

    def _check_full(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        if state.ndim != 1 or state.shape[0] != self._n_full:
            raise ValueError(
                f"state must be a 1-D array of length {self._n_full}, got shape {state.shape}"
            )
        return state

    def gather(self, state: ArrayLike) -> NDArray[np.float64]:
        """Extract the shared subset from a full state vector, in exchange order."""
        s = self._check_full(np.asarray(state, dtype=np.float64))
        return s[self._idx]

    def scatter(
        self, state: ArrayLike, values: ArrayLike, *, copy: bool = True
    ) -> NDArray[np.float64]:
        """Write ``values`` (exchange order) into the shared slots of ``state``.

        Returns the updated full vector. With ``copy=True`` (default) ``state`` is
        not mutated; ``copy=False`` writes in place and returns the same array.
        """
        s = self._check_full(np.asarray(state, dtype=np.float64))
        v = np.asarray(values, dtype=np.float64)
        if v.shape != (self.n_shared,):
            raise ValueError(f"values must be a length-{self.n_shared} array, got shape {v.shape}")
        if copy:
            s = s.copy()
        s[self._idx] = v
        return s

    def read(self, source: Model | ReactionKernel) -> NDArray[np.float64]:
        """Pull the shared subset straight from a live model/kernel's state."""
        m = _resolve_model(source)
        return self.gather(m.get_state())

    def write(self, target: Model | ReactionKernel, values: ArrayLike) -> None:
        """Inject ``values`` into the shared slots of a live model/kernel's state."""
        m = _resolve_model(target)
        m.set_state(self.scatter(m.get_state(), values, copy=False))

    def __len__(self) -> int:
        return self.n_shared

    def __repr__(self) -> str:
        return f"CouplingMap(n_shared={self.n_shared}, n_full={self._n_full})"


# ─── Discrete rounding policy + leak accounting ──────────────────────────────


def round_to_counts(
    amounts: ArrayLike,
    policy: RoundingPolicy = "nearest",
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Discretize continuous amounts to integer molecule counts (stateless).

    Parameters
    ----------
    amounts : array_like
        Continuous molecule amounts (the volume-invariant exchange currency;
        use :meth:`UnitConverter.to_amounts` to get here from storage).
    policy : {"nearest", "floor", "ceil", "stochastic"}
        See :data:`RoundingPolicy`. ``"nearest"`` rounds half away from zero,
        matching the SSA engine's entry rounding so an explicit hand-off and the
        implicit one agree. ``"stochastic"`` rounds up with probability equal to
        the fractional part (unbiased), and needs ``rng``.
    rng : numpy.random.Generator, optional
        Required for ``policy="stochastic"``; ignored otherwise.

    Returns
    -------
    ndarray
        Integer-valued ``float64`` counts (same dtype as the state vector, so it
        feeds straight back through :meth:`UnitConverter.from_counts`).
    """
    a = np.asarray(amounts, dtype=np.float64)
    if policy == "nearest":
        # Half away from zero — matches round_initial_population_to_storage.
        return np.where(a >= 0.0, np.floor(a + 0.5), np.ceil(a - 0.5))
    if policy == "floor":
        return np.floor(a)
    if policy == "ceil":
        return np.ceil(a)
    if policy == "stochastic":
        if rng is None:
            raise ValueError("policy='stochastic' requires an rng (numpy.random.Generator)")
        lo = np.floor(a)
        frac = a - lo
        return lo + (rng.random(a.shape) < frac).astype(np.float64)
    raise ValueError(f"unknown rounding policy {policy!r}")


class DiscreteExchange:
    """Explicit, inspectable rounding at the SSA/NFsim hand-off with leak accounting.

    The SSA engine already rounds whatever continuous storage it is handed at the
    next ``advance`` (:func:`round_initial_population_to_storage`), but that entry
    rounding is *implicit* and silent — repeatedly injecting continuous amounts
    and reading integer counts back sheds the fractional remainder every step,
    which leaks mass over a long hybrid run. :class:`DiscreteExchange` makes the
    boundary explicit and conserving: it carries the per-species fractional
    residual forward (error feedback / dithering), so the discrete counts track
    the continuous amounts with bounded, *accounted* error rather than a silent
    downward drift.

    Parameters
    ----------
    n_species : int
        Length of the amount vectors handed across the boundary.
    policy : {"nearest", "floor", "ceil", "stochastic"}
        Per-step rounding rule (:data:`RoundingPolicy`).
    dither : bool, optional
        Carry the fractional residual into the next step (default ``True``). With
        dithering the cumulative leak stays bounded by the carry; without it,
        each step rounds independently and leak can accumulate — set ``False``
        only when you explicitly want memoryless rounding.
    nonneg : bool, optional
        Clamp counts at 0 (default ``True``) so the SSA boundary never receives a
        negative population. When dithering, a clamp keeps the unrepresentable
        deficit in the carry, so conservation accounting is undisturbed.
    rng : numpy.random.Generator, optional
        Used by ``policy="stochastic"``.

    Examples
    --------
    >>> dx = DiscreteExchange(3, policy="nearest")
    >>> counts = dx.discretize([0.4, 0.4, 0.4])   # rounds down this step
    >>> later = dx.discretize([0.4, 0.4, 0.4])    # carry has built up to 0.8 → rounds up
    >>> dx.leak                                    # net mass injected vs the continuous input
    """

    __slots__ = (
        "_n",
        "_policy",
        "_dither",
        "_nonneg",
        "_rng",
        "_carry",
        "_leak",
        "_last_residual",
    )

    def __init__(
        self,
        n_species: int,
        *,
        policy: RoundingPolicy = "nearest",
        dither: bool = True,
        nonneg: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_species < 0:
            raise ValueError("n_species must be >= 0")
        self._n = int(n_species)
        self._policy: RoundingPolicy = policy
        self._dither = bool(dither)
        self._nonneg = bool(nonneg)
        self._rng = rng
        self._carry = np.zeros(self._n, dtype=np.float64)
        self._leak = 0.0
        self._last_residual = np.zeros(self._n, dtype=np.float64)

    def discretize(self, amounts: ArrayLike) -> NDArray[np.float64]:
        """Round ``amounts`` to integer counts, carrying the residual forward.

        Returns integer-valued ``float64`` counts of length ``n_species``. Updates
        :attr:`carry` (dithering), :attr:`last_residual` (this step's
        ``counts - amounts``) and :attr:`leak` (cumulative net mass discretization
        has added or removed since construction / :meth:`reset`).
        """
        a = np.asarray(amounts, dtype=np.float64)
        if a.shape != (self._n,):
            raise ValueError(f"amounts must be a length-{self._n} array, got shape {a.shape}")
        biased = a + self._carry if self._dither else a
        counts = round_to_counts(biased, self._policy, rng=self._rng)
        if self._nonneg:
            counts = np.maximum(counts, 0.0)
        if self._dither:
            # carry = what we owe; a clamp leaves the unrepresentable deficit here.
            self._carry = biased - counts
        self._last_residual = counts - a
        self._leak += float(self._last_residual.sum())
        return counts

    @property
    def carry(self) -> NDArray[np.float64]:
        """Current per-species fractional residual buffer (a copy)."""
        return self._carry.copy()

    @property
    def last_residual(self) -> NDArray[np.float64]:
        """``counts - amounts`` from the most recent :meth:`discretize` (a copy)."""
        return self._last_residual.copy()

    @property
    def leak(self) -> float:
        """Cumulative net molecules added (>0) or removed (<0) by rounding.

        With ``dither=True`` this stays bounded (it equals ``-carry.sum()``); a
        growing magnitude under ``dither=False`` is exactly the silent SSA-entry
        leak this boundary is meant to surface.
        """
        return self._leak

    def reset(self) -> None:
        """Clear the carry, leak ledger, and last residual."""
        self._carry = np.zeros(self._n, dtype=np.float64)
        self._leak = 0.0
        self._last_residual = np.zeros(self._n, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"DiscreteExchange(n_species={self._n}, policy={self._policy!r}, "
            f"dither={self._dither}, leak={self._leak:.3g})"
        )


# ─── Conservation / no-leak checks ───────────────────────────────────────────


def moiety_total(state: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Total of a conserved moiety: ``weights · state`` (or ``state.sum()``).

    A conserved moiety of a reaction network is a left-null-space vector ``w`` of
    the stoichiometry matrix; ``w · n`` is invariant under the reactions. The
    default ``weights=None`` is the all-ones vector — the total molecule count of
    a closed transfer network, the moiety an operator split must not leak.
    """
    s = np.asarray(state, dtype=np.float64)
    if weights is None:
        return float(s.sum())
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != s.shape:
        raise ValueError(f"weights shape {w.shape} != state shape {s.shape}")
    return float(np.dot(w, s))


class ConservationLedger:
    """Track a conserved moiety across exchange-boundary round-trips (GH #102).

    Records the moiety total at a baseline state and on every subsequent state,
    so an operator-split loop can assert the shared moiety is preserved across
    ``get_state → orchestrator → set_state`` exchanges and the discrete rounding.

    Parameters
    ----------
    weights : array_like, optional
        Moiety weight vector (see :func:`moiety_total`). Default: total count.
    atol, rtol : float, optional
        Absolute / relative tolerance for :meth:`check` and
        :meth:`assert_conserved`. The drift bound is ``atol + rtol * |baseline|``.
    name : str, optional
        Label used in error messages.
    """

    __slots__ = ("_weights", "_atol", "_rtol", "_name", "_baseline", "_last", "_max_drift", "_n")

    def __init__(
        self,
        weights: ArrayLike | None = None,
        *,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        name: str = "total",
    ) -> None:
        self._weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self._atol = float(atol)
        self._rtol = float(rtol)
        self._name = name
        self._baseline: float | None = None
        self._last: float | None = None
        self._max_drift = 0.0
        self._n = 0

    def record(self, state: ArrayLike) -> float:
        """Record the moiety total of ``state``; set the baseline on first call.

        Returns the moiety total.
        """
        total = moiety_total(state, self._weights)
        if self._baseline is None:
            self._baseline = total
        self._last = total
        self._max_drift = max(self._max_drift, abs(total - self._baseline))
        self._n += 1
        return total

    def _tol(self) -> float:
        base = 0.0 if self._baseline is None else abs(self._baseline)
        return self._atol + self._rtol * base

    def check(self, state: ArrayLike) -> tuple[bool, float]:
        """Record ``state`` and return ``(within_tolerance, signed_drift)``."""
        total = self.record(state)
        drift = total - (self._baseline if self._baseline is not None else total)
        return abs(drift) <= self._tol(), drift

    def assert_conserved(self, state: ArrayLike) -> float:
        """Record ``state``; raise :class:`ConservationError` if drift exceeds tol.

        Returns the signed drift from baseline.
        """
        ok, drift = self.check(state)
        if not ok:
            raise ConservationError(
                f"moiety {self._name!r} drifted by {drift:.6g} from baseline "
                f"{self._baseline:.6g} (tolerance {self._tol():.3g}) after "
                f"{self._n} records"
            )
        return drift

    @property
    def baseline(self) -> float | None:
        """The first recorded moiety total (``None`` before any record)."""
        return self._baseline

    @property
    def last(self) -> float | None:
        """The most recently recorded moiety total."""
        return self._last

    @property
    def max_abs_drift(self) -> float:
        """Largest absolute drift from baseline seen so far."""
        return self._max_drift

    @property
    def n_records(self) -> int:
        """Number of states recorded."""
        return self._n

    def __repr__(self) -> str:
        base = "unset" if self._baseline is None else f"{self._baseline:.6g}"
        return (
            f"ConservationLedger(name={self._name!r}, baseline={base}, "
            f"max_abs_drift={self._max_drift:.3g}, n={self._n})"
        )


# ─── Cell-division divider ───────────────────────────────────────────────────


class Divider:
    """Partition molecule counts across daughter cells at division (GH #102).

    Cell division is a pure **count-space** operation with no analogue elsewhere
    in bngsim: the molecules of each species are dealt out among the daughters,
    exactly conserving the parent total (``sum(daughters) == parent`` for every
    partitioned species). It composes on top of :class:`UnitConverter` — convert
    storage → counts, divide, then convert each daughter's counts back to storage
    via :meth:`UnitConverter.from_counts` and inject with ``set_state``. Volume
    is the orchestrator's to halve (see :func:`set_compartment_volume`).

    Parameters
    ----------
    method : {"binomial", "multinomial", "deterministic"}
        ``"binomial"`` / ``"multinomial"`` (synonyms) deal each molecule to a
        uniformly-random daughter — the physically faithful stochastic split,
        exactly conserving. ``"deterministic"`` splits as evenly as possible
        (floor share + largest-remainder distribution of the leftover), also
        exactly conserving; reproducible without an RNG.
    rng : numpy.random.Generator, optional
        Required for the stochastic methods.
    """

    __slots__ = ("_method", "_rng")

    def __init__(
        self,
        *,
        method: Literal["binomial", "multinomial", "deterministic"] = "binomial",
        rng: np.random.Generator | None = None,
    ) -> None:
        if method not in ("binomial", "multinomial", "deterministic"):
            raise ValueError(f"unknown divide method {method!r}")
        self._method = method
        self._rng = rng

    def divide(
        self,
        counts: ArrayLike,
        n_daughters: int = 2,
        *,
        partition_mask: ArrayLike | None = None,
    ) -> list[NDArray[np.float64]]:
        """Partition integer ``counts`` into ``n_daughters`` daughter vectors.

        Parameters
        ----------
        counts : array_like
            Non-negative, integer-valued molecule counts (e.g. from
            :meth:`UnitConverter.to_counts`).
        n_daughters : int, optional
            Number of daughters (default 2).
        partition_mask : array_like of bool, optional
            ``True`` where a species is a partitioned molecule pool (split and
            conserved); ``False`` where it is shared environment (copied
            identically to every daughter, not split). Default: partition all.

        Returns
        -------
        list of ndarray
            ``n_daughters`` integer-valued ``float64`` count vectors. For every
            partitioned species the daughters sum exactly to the parent count.
        """
        if n_daughters < 1:
            raise ValueError("n_daughters must be >= 1")
        a = np.asarray(counts, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError(f"counts must be 1-D, got shape {a.shape}")
        if np.any(a < 0):
            raise ValueError("counts must be non-negative")
        rounded = np.rint(a)
        if not np.allclose(a, rounded):
            raise ValueError("counts must be integer-valued; discretize before dividing")
        ints = rounded.astype(np.int64)
        n = ints.shape[0]

        if partition_mask is None:
            mask = np.ones(n, dtype=bool)
        else:
            mask = np.asarray(partition_mask, dtype=bool)
            if mask.shape != (n,):
                raise ValueError(f"partition_mask must be length {n}, got shape {mask.shape}")

        if self._method == "deterministic":
            shares = self._divide_deterministic(ints[mask], n_daughters)
        else:
            shares = self._divide_stochastic(ints[mask], n_daughters)

        daughters: list[NDArray[np.float64]] = []
        for d in range(n_daughters):
            out = ints.astype(np.float64).copy()  # shared species copied as-is
            out[mask] = shares[d].astype(np.float64)
            daughters.append(out)
        return daughters

    def _divide_stochastic(self, pool: NDArray[np.int64], k: int) -> list[NDArray[np.int64]]:
        """Multinomial split: each molecule to a uniformly-random daughter."""
        if self._rng is None:
            raise ValueError(f"method={self._method!r} requires an rng (numpy.random.Generator)")
        remaining = pool.copy()
        shares: list[NDArray[np.int64]] = []
        for d in range(k - 1):
            # Binomial split of what's left across the remaining daughters keeps
            # the joint distribution exactly multinomial(pool, uniform).
            take = self._rng.binomial(remaining, 1.0 / (k - d)).astype(np.int64)
            shares.append(take)
            remaining = remaining - take
        shares.append(remaining)
        return shares

    @staticmethod
    def _divide_deterministic(pool: NDArray[np.int64], k: int) -> list[NDArray[np.int64]]:
        """Even split: floor share to each, leftover to the first daughters."""
        base = pool // k
        rem = pool - base * k  # 0..k-1 leftover molecules per species
        shares = [base.copy() for _ in range(k)]
        # Largest-remainder: give the i-th leftover molecule to daughter i.
        for d in range(k):
            shares[d] = shares[d] + (rem > d).astype(np.int64)
        return shares

    def __repr__(self) -> str:
        return f"Divider(method={self._method!r})"


# ─── Framework volume-growth → compartment-volume coupling ───────────────────


def get_compartment_volume(model: Model | ReactionKernel, name: str) -> float:
    """Read a compartment volume (a model parameter) by name.

    There is no ``Compartment`` object in bngsim — a compartment volume is a
    plain parameter, so this is :meth:`Model.get_param`, named for the coupling
    use it serves: feeding the *current* volume into
    :meth:`UnitConverter.to_concentrations` / :meth:`from_concentrations` as the
    live-volume override.
    """
    return _resolve_model(model).get_param(name)


def set_compartment_volume(model: Model | ReactionKernel, name: str, volume: float) -> None:
    """Couple a framework's volume growth into bngsim's compartment volume.

    Sets the compartment-volume *parameter*. On the **ODE subset** this flows
    through bngsim's variable-volume machinery natively (the integrator dilutes
    by the live compartment symbol; GH #74/#85), so growing or halving (at
    division) a compartment is just this call. On the **SSA subset** the baked
    per-reaction ``ssa_volume_factor`` does *not* track the new value — the
    exchange layer compensates by reading concentrations against the live volume
    (the :class:`UnitConverter` override), but the stochastic *propensities*
    using the live volume are Stage 2.
    """
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"volume must be finite and > 0, got {volume}")
    _resolve_model(model).set_param(name, float(volume))


# ─── ODE-subset-as-model helper ──────────────────────────────────────────────

# Reaction kinds make_subset_model can faithfully reconstruct through ModelBuilder.
_REBUILDABLE_RXN_TYPES = frozenset({"elementary", "functional", "mm"})


def make_subset_model(
    model: Model | ReactionKernel,
    *,
    keep_reactions: Sequence[int] | None = None,
    fixed_species: Sequence[str] | None = None,
    compute_conservation_laws: bool = False,
) -> Model:
    """Build an operator-split subset as its own model (GH #102 Stage 1, #6).

    bngsim runs a whole network with one method — there is no native
    reaction-subset integration. For a *static* operator split the continuous
    (ODE) subset is supplied as its **own model**: the same species namespace,
    only the subset's reactions, and the *other* operator's species marked
    ``fixed=True`` so the integrator holds them at the boundary values the
    orchestrator writes each step via ``set_state``. The ``fixed`` mechanism
    already clamps a species in both the ODE (zeroed derivative) and SSA (skipped
    fire) backends — this helper just constructs the partitioned model; no engine
    change is involved.

    The full species set is kept (so the two subsets share one addressing space
    for :class:`CouplingMap`); only the reaction set is subset and the boundary
    species fixed. A species touched by no kept reaction is already inert, so
    ``fixed_species`` is needed only for boundary species whose value a kept rate
    law *reads* but should not *evolve*.

    Parameters
    ----------
    model : Model or ReactionKernel
        The full network to partition.
    keep_reactions : sequence of int, optional
        0-based indices of the reactions this subset integrates. ``None`` keeps
        all reactions (a pure re-fix with no reaction split).
    fixed_species : sequence of str, optional
        Species names to additionally mark ``fixed`` (the other operator's
        species). Species already fixed in the source stay fixed.
    compute_conservation_laws : bool, optional
        Forwarded to the builder (default ``False`` — the dense O(n³) detector is
        unused by stepping and costly at scale; see GH #102 MVP).

    Returns
    -------
    Model
        A freshly built subset model at the source's **initial** concentrations
        (taken from a reset clone, so it is immune to a prior simulation having
        left the source mid-trajectory). Inject a custom starting state with
        ``set_state`` on the subset afterwards.

    Raises
    ------
    NotImplementedError
        If the source uses features this reconstruction cannot reproduce
        faithfully (events, table functions, discontinuity triggers,
        ``amount_valued`` species, or ``per_species_volume_scaling`` reactions).
        The supported class — mass action, functional, and Michaelis–Menten rate
        laws, parameters, observables, functions — covers the issue's ~100K
        first-order target.
    """
    from bngsim._bngsim_core import ModelBuilder

    m = _resolve_model(model)
    core = m._core
    cgd = core.codegen_data()

    # Refuse features we cannot reconstruct through ModelBuilder rather than
    # silently dropping them (codegen_data carries no events/triggers).
    for attr, label in (
        ("n_events", "events"),
        ("n_discontinuity_triggers", "discontinuity triggers (GH #72)"),
        ("n_table_functions", "table functions (tfun)"),
    ):
        if getattr(core, attr, 0):
            raise NotImplementedError(
                f"make_subset_model cannot reconstruct a model with {label}; "
                "build the operator-split subsets directly with ModelBuilder"
            )
    for s in cgd["species"]:
        if s.get("amount_valued", False):
            raise NotImplementedError(
                "make_subset_model cannot reconstruct amount_valued species "
                f"({s['name']!r}, GH #75); build the subset with ModelBuilder"
            )

    params = cgd["parameters"]
    species = cgd["species"]
    functions = cgd["functions"]
    reactions = cgd["reactions"]
    observables = cgd["observables"]

    fixed_set = set(fixed_species or ())
    known = {s["name"] for s in species}
    unknown = fixed_set - known
    if unknown:
        raise KeyError(f"fixed_species not in the model: {sorted(unknown)}")

    kept_idx = list(range(len(reactions)) if keep_reactions is None else keep_reactions)
    for ri in kept_idx:
        if not (0 <= ri < len(reactions)):
            raise IndexError(f"reaction index {ri} out of range [0, {len(reactions)})")

    # Initial concentrations from a reset clone, so a prior run on `model` (which
    # writes its final state back) cannot leak into the subset's seed.
    probe = m.clone()
    probe.reset()
    init = probe.get_state()

    b = ModelBuilder()
    b.set_compute_conservation_laws(compute_conservation_laws)

    for p in params:
        b.add_parameter(
            p["name"], float(p["value"]), p.get("expression", ""), not p.get("is_const", True)
        )
    for i, s in enumerate(species):
        b.add_species(
            s["name"],
            float(init[i]),
            bool(s.get("fixed", False)) or s["name"] in fixed_set,
            float(s.get("volume_factor", 1.0)),
        )
    for f in functions:
        b.add_function(f["name"], f["expression"])

    param_names = [p["name"] for p in params]
    for ri in kept_idx:
        r = reactions[ri]
        rtype = r["type"]
        if rtype not in _REBUILDABLE_RXN_TYPES:
            raise NotImplementedError(
                f"make_subset_model cannot reconstruct a {rtype!r} reaction "
                "(only elementary / functional / mm); build the subset with ModelBuilder"
            )
        if r.get("per_species_volume_scaling", False):
            raise NotImplementedError(
                "make_subset_model cannot reconstruct a per_species_volume_scaling "
                "reaction (mixed-compartment SBML); build the subset with ModelBuilder"
            )
        if rtype == "elementary":
            # The rate law is the rate-constant parameter's name.
            rate_law = param_names[r["rate_param_indices"][0]]
        elif rtype == "mm":
            # Michaelis–Menten: ModelBuilder wants "kcat,Km"; the enzyme/substrate
            # roles are re-derived from the (order-preserved) reactant list.
            rp = r["rate_param_indices"]
            if len(rp) < 2:
                raise NotImplementedError(
                    "make_subset_model cannot reconstruct a Michaelis–Menten "
                    "reaction missing its [kcat, Km] parameters"
                )
            rate_law = f"{param_names[rp[0]]},{param_names[rp[1]]}"
        else:  # functional
            rate_law = r["function_name"]
        b.add_reaction(
            list(r["reactants"]),
            list(r["products"]),
            rtype,
            rate_law,
            float(r.get("stat_factor", 1.0)),
            bool(r.get("apply_species_factor", True)),
            float(r.get("ssa_volume_factor", 1.0)),
            False,
            # GH #81: preserve the rate-rule-ODE flag so a subset containing a
            # rate-rule reaction `[] → [X]` still integrates X deterministically
            # under SSA rather than firing it as a stochastic channel.
            bool(r.get("is_rate_rule_ode", False)),
            # GH #81: preserve the SSA live-volume correction (variable-volume
            # compartment). The species index is forwarded verbatim, consistent
            # with reactant/product indices — make_subset_model keeps the full
            # species indexing.
            int(r.get("ssa_live_volume_idx0", -1)),
            float(r.get("ssa_live_volume_exp", 0.0)),
            # GH #81: preserve the ODE-only flag (the #86 dilution reaction is
            # excluded from SSA entirely).
            bool(r.get("ode_only", False)),
        )
    for o in observables:
        b.add_observable(o["name"], [(int(i), float(f)) for i, f in o["entries"]])

    return Model(_core=b.build())
