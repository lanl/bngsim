"""bngsim.vivarium — an optional Vivarium process wrapping the reaction kernel.

A thin shell that exposes a :class:`bngsim.ReactionKernel` as a `vivarium-core
<https://vivarium-core.readthedocs.io/>`_ ``Process``, so a bngsim reaction
network can be dropped into a Vivarium composite and operator-split against other
processes over a shared species store. It is deliberately minimal — the direct
:class:`bngsim.ReactionKernel` API is the primary, framework-agnostic interface
(routing state through a Vivarium store/topology is itself overhead at scale),
and vivarium-core's process interface is migrating to process-bigraph. The shell
is just the three hooks Vivarium needs:

* :meth:`BngsimProcess.ports_schema` — a ``species`` port (one variable per
  species, ``accumulate`` updater) and an ``observables`` port (one per
  observable, ``set`` updater), defaulted from the model's initial state.
* :meth:`BngsimProcess.next_update` — pull the shared species state in from the
  store, ``advance`` the kernel by the step, and return per-species **deltas**
  (so they compose additively with other processes under operator splitting)
  plus the recomputed observable values.
* :meth:`BngsimProcess.calculate_timestep` — the configured coupling step.

This module requires ``vivarium-core`` (an optional extra); importing it without
that dependency raises a clear :class:`ModuleNotFoundError`. The base
``bngsim`` package never imports it — check :data:`bngsim.HAS_VIVARIUM` first, or
just ``from bngsim.vivarium import BngsimProcess`` and handle the import error.

Example
-------
>>> import bngsim
>>> from bngsim.vivarium import BngsimProcess
>>> from vivarium.core.engine import Engine
>>> model = bngsim.Model.from_net("model.net")
>>> proc = BngsimProcess({"model": model, "time_step": 1.0})
>>> engine = Engine(
...     processes={"bngsim": proc},
...     topology={"bngsim": {"species": ("species",), "observables": ("observables",)}},
...     initial_state=proc.initial_state(),
... )
>>> engine.update(100.0)

Notes
-----
The kernel's interactive clock accumulates each ``next_update`` step, so
time-dependent rate laws stay synced to the process's local time. The kernel
re-reads the store's species values each step (``set_state``), making the store
the source of truth — the correct semantics for an operator-split partition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim._model import Model
from bngsim.kernel import ReactionKernel

try:
    from vivarium.core.process import Process
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import guard
    raise ModuleNotFoundError(
        "bngsim.vivarium requires the optional 'vivarium-core' dependency. "
        "Install it with:  pip install 'bngsim[vivarium]'  (or  pip install vivarium-core)."
    ) from exc

if TYPE_CHECKING:
    from bngsim._simulator import Simulator

__all__ = ["BngsimProcess"]


class BngsimProcess(Process):
    """A Vivarium ``Process`` that advances a bngsim reaction network per step.

    Parameters (passed as the vivarium ``parameters`` dict)
    -------------------------------------------------------
    time_step : float
        Coupling step returned by :meth:`calculate_timestep`. Default ``1.0``.
    kernel : ReactionKernel, optional
        A pre-built kernel to drive. Takes precedence over ``model``; use this
        to reuse a kernel built from an already-configured ``Simulator``.
    model : Model, optional
        A model to wrap in a fresh kernel. Required if ``kernel`` is not given.
    method : str, optional
        Method for the fresh kernel when building from ``model``. Default
        ``"ode"``.
    simulator_kwargs : dict, optional
        Extra keyword args forwarded to the ``Simulator`` when building from
        ``model`` (e.g. ``{"codegen": True, "jacobian": "analytical"}``).

    Examples
    --------
    >>> proc = BngsimProcess({"model": model, "time_step": 0.5})
    >>> proc.calculate_timestep(None)
    0.5
    """

    defaults: dict[str, Any] = {
        "time_step": 1.0,
        "kernel": None,
        "model": None,
        "method": "ode",
        "simulator_kwargs": {},
    }

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        super().__init__(parameters)
        self._kernel = self._resolve_kernel(self.parameters)
        # Snapshot the initial store defaults from the kernel's starting state
        # (its live concentrations + observables before any advance).
        self._species_names = self._kernel.state_names
        self._observable_names = self._kernel.observable_names
        self._initial_species = self._kernel.get_state()
        self._initial_observables = self._kernel.observables()

    @staticmethod
    def _resolve_kernel(parameters: dict[str, Any]) -> ReactionKernel:
        kernel = parameters.get("kernel")
        if kernel is not None:
            if not isinstance(kernel, ReactionKernel):
                raise TypeError(
                    f"parameters['kernel'] must be a bngsim.ReactionKernel, "
                    f"got {type(kernel).__name__}"
                )
            return kernel
        model = parameters.get("model")
        if model is None:
            raise ValueError(
                "BngsimProcess requires either parameters['kernel'] (a ReactionKernel) "
                "or parameters['model'] (a bngsim.Model)."
            )
        if not isinstance(model, Model):
            raise TypeError(
                f"parameters['model'] must be a bngsim.Model, got {type(model).__name__}"
            )
        return ReactionKernel(
            model,
            method=parameters.get("method", "ode"),
            **parameters.get("simulator_kwargs", {}),
        )

    # ─── Vivarium hooks ──────────────────────────────────────────────────────

    def ports_schema(self) -> dict[str, Any]:
        """Declare the ``species`` and ``observables`` ports.

        ``species`` variables use the ``accumulate`` updater because
        :meth:`next_update` returns deltas — so a bngsim subset composes
        additively with other processes writing the same species under operator
        splitting. ``observables`` are derived from the state, so they use the
        ``set`` updater (recomputed each step).
        """
        species = {
            name: {
                "_default": float(self._initial_species[i]),
                "_updater": "accumulate",
                "_emit": True,
            }
            for i, name in enumerate(self._species_names)
        }
        observables = {
            name: {
                "_default": float(self._initial_observables[i]),
                "_updater": "set",
                "_emit": True,
            }
            for i, name in enumerate(self._observable_names)
        }
        return {"species": species, "observables": observables}

    def initial_state(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Initial store values: the kernel's starting species + observables."""
        return {
            "species": {
                name: float(self._initial_species[i]) for i, name in enumerate(self._species_names)
            },
            "observables": {
                name: float(self._initial_observables[i])
                for i, name in enumerate(self._observable_names)
            },
        }

    def calculate_timestep(self, states: dict[str, Any] | None) -> float:
        """Return the configured coupling step."""
        return float(self.parameters["time_step"])

    def next_update(self, timestep: float, states: dict[str, Any]) -> dict[str, Any]:
        """Advance the kernel by ``timestep`` from the store's species state.

        Pulls the shared species values out of the ``species`` store into the
        kernel (so the store is the source of truth), advances the kernel, and
        returns per-species deltas (``accumulate``) plus recomputed observable
        values (``set``).
        """
        species_store = states["species"]
        current = np.array([species_store[name] for name in self._species_names], dtype=np.float64)
        self._kernel.set_state(current)
        new = self._kernel.advance(timestep)

        species_update = {
            name: float(new[i] - current[i]) for i, name in enumerate(self._species_names)
        }
        update: dict[str, Any] = {"species": species_update}

        if self._observable_names:
            obs = self._kernel.observables()
            update["observables"] = {
                name: float(obs[i]) for i, name in enumerate(self._observable_names)
            }
        return update

    # ─── Introspection ───────────────────────────────────────────────────────

    @property
    def kernel(self) -> ReactionKernel:
        """The wrapped :class:`bngsim.ReactionKernel`."""
        return self._kernel

    @property
    def simulator(self) -> Simulator:
        """The underlying :class:`bngsim.Simulator`."""
        return self._kernel.simulator
