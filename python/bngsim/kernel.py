"""bngsim.kernel — a framework-agnostic reaction kernel (GH #102).

:class:`ReactionKernel` is a thin, hardened facade over a :class:`bngsim.Model`
plus a :class:`bngsim.Simulator` that an *external orchestrator* — first and
foremost a hand-rolled hybrid SSA/ODE splitting loop, and secondarily a
composition framework such as Vivarium — can drive per step with low-overhead,
conserved state exchange:

    set the live state  →  advance by a coupling step ``dt``  →  read it back

The kernel owns no integration of its own: ``advance(dt)`` delegates to the
underlying ``Simulator.run_until`` so the same object works method-agnostically
over the stateful backends (``ode`` / ``ssa`` / ``psa``). The network-free XML
backends (``nfsim`` / ``rulemonkey``) have no stateful per-step API, so
``advance`` raises a clear error for them — the kernel still wraps them for
state/name introspection, but stepping is unsupported.

Two ideas keep the per-step cost dominated by the (already-fast) solve rather
than by Python overhead, which is the whole point at the ~100K-reaction scale
the issue targets:

* **Bulk state exchange.** :meth:`get_state` / :meth:`set_state` move the entire
  live concentration vector as one ``O(n_species)`` numpy round-trip
  (one Python call each), not per-name accessors.
* **A warm simulator.** Repeated ``advance`` calls reuse the same ``Simulator``;
  on the ODE path the C++ layer keeps persistent CVODE memory across calls
  (GH #102 warm path) so no per-step re-codegen or linear-solver rebuild is
  paid. The kernel never calls ``snapshot``/``restore`` on the hot path
  (those rebuild the simulator).

Example
-------
>>> import bngsim
>>> model = bngsim.Model.from_net("model.net")
>>> kernel = bngsim.ReactionKernel(model, method="ode")
>>> dt = 1.0
>>> for _ in range(100):
...     state = kernel.get_state()         # pull coupling species out
...     # ... external orchestrator updates `state` for the shared subset ...
...     kernel.set_state(state)            # inject the exchanged state
...     state = kernel.advance(dt)         # integrate the ODE subset by dt
>>> kernel.time
100.0

Advancing a model step-wise through the kernel reproduces a single standalone
``Simulator.run`` over the same horizon (to integrator tolerance); see
``python/tests/test_kernel.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bngsim._model import Model
from bngsim._simulator import Simulator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bngsim._result import Result

__all__ = ["ReactionKernel"]

# Backends with no stateful per-step API (run_until is not supported).
_STATELESS_BACKENDS = ("nfsim", "rulemonkey")


class ReactionKernel:
    """Drive a bngsim model per step as a pluggable reaction kernel (GH #102).

    Parameters
    ----------
    model : Model
        The model to drive. Its live species concentrations are the kernel's
        state vector (ordered like :attr:`state_names`).
    method : str, optional
        Simulation method passed to :class:`bngsim.Simulator`. Default
        ``"ode"``. Stepping (:meth:`advance`) requires a stateful backend
        (``ode`` / ``ssa`` / ``psa``); ``nfsim`` / ``rulemonkey`` may be
        wrapped for introspection but cannot be advanced.
    **simulator_kwargs
        Forwarded verbatim to :class:`bngsim.Simulator` (e.g. ``codegen``,
        ``jacobian``, ``poplevel``, ``xml_path``, ``sensitivity_params``).

    See Also
    --------
    ReactionKernel.from_simulator : wrap an already-configured Simulator.
    bngsim.Model.get_state : the bulk state primitive the kernel exchanges.

    Notes
    -----
    Not thread-safe — it owns mutable model + simulator state. For parallel
    workers, build one kernel per :meth:`bngsim.Model.clone`.
    """

    __slots__ = ("_sim", "_last_result", "_initial_observables")

    def __init__(self, model: Model, *, method: str = "ode", **simulator_kwargs) -> None:
        if not isinstance(model, Model):
            raise TypeError(f"model must be a bngsim.Model, got {type(model).__name__}")
        self._sim = Simulator(model, method=method, **simulator_kwargs)
        self._last_result: Result | None = None
        # Lazily filled t=0 observables for observables() before the first
        # advance (see _observables_at_current_state).
        self._initial_observables: NDArray[np.float64] | None = None

    @classmethod
    def from_simulator(cls, simulator: Simulator) -> ReactionKernel:
        """Wrap an already-constructed :class:`bngsim.Simulator`.

        Use this when the caller has configured solver options the kernel
        does not surface directly (custom tolerances via ``set_tolerances``,
        a prepared codegen ``.so``, sensitivity parameters, …). The kernel
        adopts the simulator as-is, including its current interactive time.

        Parameters
        ----------
        simulator : Simulator
            The simulator to drive.

        Returns
        -------
        ReactionKernel
        """
        if not isinstance(simulator, Simulator):
            raise TypeError(
                f"simulator must be a bngsim.Simulator, got {type(simulator).__name__}"
            )
        self = cls.__new__(cls)
        self._sim = simulator
        self._last_result = None
        self._initial_observables = None
        return self

    # ─── State exchange ─────────────────────────────────────────────────────

    def get_state(self) -> NDArray[np.float64]:
        """Bulk-copy the live species-concentration vector (GH #102).

        Returns a fresh ``float64`` array of length :attr:`n_species`, ordered
        like :attr:`state_names`. After an :meth:`advance` this is the
        post-step state. One ``O(n_species)`` Python call — the ``get`` half
        of the per-step kernel exchange.
        """
        return self._sim.get_state()

    def set_state(self, state: NDArray[np.float64]) -> None:
        """Bulk-assign the live species-concentration vector (GH #102).

        Parameters
        ----------
        state : ndarray
            1-D array of length :attr:`n_species`, ordered like
            :attr:`state_names`. Copied into the model's live concentrations;
            the next :meth:`advance` reads them as its initial condition. The
            ``set`` half of the per-step kernel exchange.
        """
        self._sim.set_state(state)
        # The injected state invalidates any cached t=0 observables.
        self._initial_observables = None

    # ─── Stepping ───────────────────────────────────────────────────────────

    def advance(
        self,
        dt: float,
        *,
        n_points: int = 2,
        seed: int | None = None,
        **run_kwargs,
    ) -> NDArray[np.float64]:
        """Advance the simulation by ``dt`` and return the post-step state.

        Integrates (or steps, for stochastic backends) from the current
        :attr:`time` to ``time + dt``, then returns :meth:`get_state`. The
        model is left holding the post-step state, so the typical per-step
        loop is ``set_state(...) → advance(dt) → get_state()`` (or just use
        the returned array).

        Parameters
        ----------
        dt : float
            Coupling step. Must be > 0.
        n_points : int, optional
            Output points recorded over the step. Default 2 (endpoints only),
            which is all an orchestrator needs and keeps recording cheap; the
            full sub-step trajectory is available via :attr:`last_result`.
        seed : int, optional
            Random seed for stochastic backends (``ssa`` / ``psa``). Ignored
            by ``ode``. ``None`` draws a fresh seed each step.
        **run_kwargs
            Forwarded to ``Simulator.run_until`` (e.g. ``rtol``, ``atol``,
            ``max_steps``).

        Returns
        -------
        ndarray
            The post-step state vector (a copy; ordered like
            :attr:`state_names`).

        Raises
        ------
        ValueError
            If ``dt <= 0``.
        NotImplementedError
            If the backend is network-free (``nfsim`` / ``rulemonkey``),
            which has no stateful per-step API.
        """
        if self._sim.method in _STATELESS_BACKENDS:
            raise NotImplementedError(
                f"ReactionKernel.advance is not supported for the "
                f"network-free '{self._sim.requested_method}' backend "
                "(no stateful per-step API). Wrap a stateful method "
                "('ode', 'ssa', or 'psa') to step, or use Simulator.run "
                "directly for independent network-free trajectories."
            )
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")

        target = self._sim.current_time + dt
        self._last_result = self._sim.run_until(target, n_points=n_points, seed=seed, **run_kwargs)
        return self._sim.get_state()

    def reset(self) -> None:
        """Reset to the model's initial concentrations and ``time = 0``.

        Restores species to their initial values (:meth:`bngsim.Model.reset`)
        and rewinds the interactive clock, so the kernel can be re-driven from
        scratch. Clears any cached step result.
        """
        self._sim.model.reset()
        self._sim._current_time = 0.0
        self._last_result = None
        self._initial_observables = None

    # ─── Observables ────────────────────────────────────────────────────────

    def observables(self) -> NDArray[np.float64]:
        """Observable values at the current simulation state.

        Returns a ``float64`` array of length :attr:`n_observables`, ordered
        like :attr:`observable_names`. After an :meth:`advance` these are the
        post-step observables (read straight from the step result, no
        recomputation). Before the first advance — or after a :meth:`set_state`
        with no advance since — they are computed once, side-effect-free, from
        the current state via a throwaway model clone.
        """
        if self._last_result is not None:
            return np.asarray(self._last_result.observables[-1], dtype=np.float64)
        return self._observables_at_current_state()

    def _observables_at_current_state(self) -> NDArray[np.float64]:
        """Compute current-state observables without mutating the live model.

        ``Simulator.run`` records the initial (t=0) observables in row 0 of its
        Result *before* integrating, so a 2-point run on an independent
        :meth:`bngsim.Model.clone` of the current state yields the observables
        at exactly this state without touching the kernel's own model. Cached
        until the next :meth:`set_state` / :meth:`advance` / :meth:`reset`.
        """
        if self._initial_observables is not None:
            return self._initial_observables
        if self.n_observables == 0:
            self._initial_observables = np.empty(0, dtype=np.float64)
            return self._initial_observables
        probe_model = self._sim.model.clone()
        probe = Simulator(probe_model, method="ode")
        # Any positive span works: row 0 is the pre-integration initial state.
        row0 = probe.run(t_span=(0.0, 1.0), n_points=2).observables[0]
        self._initial_observables = np.asarray(row0, dtype=np.float64)
        return self._initial_observables

    # ─── Introspection ──────────────────────────────────────────────────────

    @property
    def state_names(self) -> list[str]:
        """Species names, in the order of the :meth:`get_state` vector."""
        return self._sim.model.species_names

    @property
    def species_names(self) -> list[str]:
        """Alias for :attr:`state_names`."""
        return self._sim.model.species_names

    @property
    def observable_names(self) -> list[str]:
        """Observable names, in the order of :meth:`observables`."""
        return self._sim.model.observable_names

    @property
    def n_species(self) -> int:
        """Number of species (length of the state vector)."""
        return self._sim.model.n_species

    @property
    def n_observables(self) -> int:
        """Number of observables."""
        return self._sim.model.n_observables

    @property
    def time(self) -> float:
        """Current interactive simulation time."""
        return self._sim.current_time

    @property
    def method(self) -> str:
        """Internal backend dispatch key (``'ode'`` / ``'ssa'`` / …)."""
        return self._sim.method

    @property
    def model(self) -> Model:
        """The wrapped model (its live concentrations are the kernel state)."""
        return self._sim.model

    @property
    def simulator(self) -> Simulator:
        """The underlying :class:`bngsim.Simulator` driving the steps."""
        return self._sim

    @property
    def last_result(self) -> Result | None:
        """The :class:`bngsim.Result` from the most recent :meth:`advance`.

        ``None`` before the first advance. Carries the full sub-step
        trajectory of the last step (per ``n_points``), beyond the endpoint
        state :meth:`get_state` returns.
        """
        return self._last_result

    def __repr__(self) -> str:
        return (
            f"ReactionKernel(method='{self._sim.method}', "
            f"n_species={self.n_species}, time={self.time:.6g})"
        )
