"""bngsim.RuleMonkeySession — Stateful session API for RuleMonkey simulation."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from bngsim._exceptions import ParameterError, SimulationError, SimulationTimeout
from bngsim._result import Result
from bngsim._rounding import round_half_up
from bngsim._sample_times import normalize_sample_times

logger = logging.getLogger("bngsim")


class RuleMonkeySession:
    """Stateful RuleMonkey simulation session with per-step control.

    RuleMonkey reads BNG XML and implements the ``nf_exact`` network-free
    backend. Use this class for multi-action workflows that need to initialize,
    simulate, add molecules, and simulate again.
    """

    __slots__ = (
        "_core",
        "_xml_path",
        "_initialized",
        "_destroyed",
        "_seed",
    )

    def __init__(
        self,
        xml_path: str | Path,
        *,
        molecule_limit: int | None = None,
        block_same_complex_binding: bool = True,
    ) -> None:
        from bngsim._bngsim_core import HAS_RULEMONKEY

        if not HAS_RULEMONKEY:
            raise RuntimeError(
                "RuleMonkey support is not present in this bngsim install. "
                "The vendored RuleMonkey backend at third_party/rulemonkey/ "
                "is built by default; this install was either configured "
                "with -DBNGSIM_BUILD_RULEMONKEY=OFF or installed from a "
                "wheel that excludes RuleMonkey."
            )

        from bngsim._bngsim_core import RuleMonkeySimulator

        self._xml_path = str(xml_path)
        self._initialized = False
        self._destroyed = False

        self._core = RuleMonkeySimulator(self._xml_path)
        if molecule_limit is not None:
            self._core.set_molecule_limit(int(molecule_limit))
        self._core.set_block_same_complex_binding(bool(block_same_complex_binding))

        self._seed: int | None = None

        logger.debug("RuleMonkeySession created: %s", self._xml_path)

    def __enter__(self) -> RuleMonkeySession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()

    def set_molecule_limit(self, limit: int) -> None:
        """Set the global molecule limit before initialization."""
        self._require_alive()
        self._core.set_molecule_limit(int(limit))

    def set_param(self, name: str, value: float) -> None:
        """Set a parameter value before initialization."""
        self._require_alive()
        try:
            self._core.set_param(name, float(value))
        except (RuntimeError, ValueError) as e:
            raise ParameterError(f"Failed to set RuleMonkey parameter '{name}': {e}") from e

    def clear_param_overrides(self) -> None:
        """Clear all parameter overrides set via ``set_param``."""
        self._require_alive()
        self._core.clear_param_overrides()

    def initialize(self, seed: int | None = None) -> None:
        """Initialize the RuleMonkey session.

        Parameters
        ----------
        seed : int, optional
            RNG seed. ``None`` (default) draws a fresh seed; pass an
            integer for reproducibility. The actual seed used is
            exposed via the :attr:`seed` property and stamped onto every
            ``Result`` returned by :meth:`simulate`.
        """
        from bngsim._seed import _resolve_seed

        self._require_alive()
        used_seed = _resolve_seed(seed)
        try:
            self._core.initialize(used_seed)
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey initialization failed: {e}") from e
        self._initialized = True
        self._seed = used_seed
        logger.info("RuleMonkeySession initialized (seed=%d)", used_seed)

    def destroy(self) -> None:
        """Destroy the session and release C++ resources."""
        if self._destroyed:
            return
        with contextlib.suppress(Exception):
            self._core.destroy_session()
        self._destroyed = True
        self._initialized = False
        logger.debug("RuleMonkeySession destroyed")

    def simulate(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        n_points: int | None = None,
        *,
        timeout: float | None = None,
        sample_times: list[float] | None = None,
    ) -> Result:
        """Run a simulation segment and return a Result.

        Parameters
        ----------
        t_start, t_end, n_points
            Segment endpoints and number of output samples. Required unless
            ``sample_times`` is given.
        sample_times : list[float], optional
            Explicit output instants (GH #184). When provided, overrides
            ``t_start`` / ``t_end`` / ``n_points``: the returned
            :attr:`Result.time` array equals these times exactly (sorted
            ascending, treated as absolute), instead of a uniform grid. The
            live session advances from its current time by the span of the
            list, so this works mid-protocol (after a prior segment or
            ``setConcentration``), continuing the same trajectory. Must
            contain at least two finite points.
        timeout : float, optional
            Wall-clock budget in seconds. ``None`` or ``0`` disables the
            budget. Positive values arm a check polled by upstream every
            ~1024 SSA events; on overrun this method raises
            :class:`bngsim.SimulationTimeout` and the session should be
            destroyed (or the context manager exited) before reuse — the
            live RuleMonkey session sits at the last completed event and
            subsequent state is undefined for bngsim purposes.
        """
        self._require_initialized()

        timeout_seconds: float = 0.0
        if timeout is not None:
            timeout_seconds = float(timeout)
            if timeout_seconds < 0.0:
                raise ValueError(f"timeout must be non-negative or None, got {timeout!r}")

        # Resolve the output schedule. Explicit sample_times (GH #184) override
        # t_start/t_end/n_points; the C++ core ignores those in that branch but
        # still takes them as positional args, so pass the resolved endpoints.
        explicit_times: list[float] = []
        if sample_times is not None:
            explicit_times = normalize_sample_times(sample_times)
            t_start = explicit_times[0]
            t_end = explicit_times[-1]
            n_points = len(explicit_times)
        elif t_start is None or t_end is None or n_points is None:
            raise ValueError(
                "t_start, t_end and n_points are required unless sample_times is given."
            )

        try:
            core_result = self._core.simulate(
                float(t_start), float(t_end), int(n_points), timeout_seconds, explicit_times
            )
        except SimulationTimeout:
            raise
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey simulation failed: {e}") from e
        result = Result(core_result)
        result._seed = self._seed
        return result

    def step_to(self, time: float, *, timeout: float | None = None) -> None:
        """Advance the simulation to *time* without recording output.

        Parameters
        ----------
        time : float
            Target time to advance to.
        timeout : float, optional
            Wall-clock budget; see :meth:`simulate` for semantics.
        """
        self._require_initialized()

        timeout_seconds: float = 0.0
        if timeout is not None:
            timeout_seconds = float(timeout)
            if timeout_seconds < 0.0:
                raise ValueError(f"timeout must be non-negative or None, got {timeout!r}")

        try:
            self._core.step_to(float(time), timeout_seconds)
        except SimulationTimeout:
            raise
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey step_to failed: {e}") from e

    def get_molecule_count(self, mol_type: str) -> int:
        """Get the current count of a molecule type."""
        self._require_initialized()
        return self._core.get_molecule_count(mol_type)

    def add_molecules(self, mol_type: str, count: int) -> None:
        """Add molecules of a given type in their default unbound state.

        A fractional *count* is rounded to the nearest integer (round-half-up,
        GH #51), matching bngsim SSA and the cold-start seed.
        """
        self._require_initialized()
        count = round_half_up(count)
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        self._core.add_molecules(mol_type, count)

    def get_parameter(self, name: str) -> float:
        """Return a parameter value from the loaded XML plus overrides."""
        self._require_alive()
        return self._core.get_parameter(name)

    def get_observable_names(self) -> list[str]:
        """Return observable names from the loaded XML."""
        self._require_alive()
        return self._core.get_observable_names()

    def get_observable_values(self) -> list[float]:
        """Return current observable values from the live session."""
        self._require_initialized()
        return self._core.get_observable_values()

    # ── Exact-species queries & mutations ────────────────────────

    def get_species_count(self, pattern: str) -> int:
        """Get the current count of an exact BNGL species pattern.

        Peer of :meth:`bngsim.NfsimSession.get_species_count`. *pattern* is
        parsed and canonicalized by RuleMonkey's runtime species-pattern
        parser, so it need not be byte-identical to a label RuleMonkey emits.
        Scope is exact, fully-specified, connected species: every molecule
        lists every component, every stateful component carries a concrete
        ``~state``, and bonds are numeric labels. Wildcards (``!+``, ``!?``)
        or omitted components raise.

        Parameters
        ----------
        pattern : str
            Exact BNGL species pattern.

        Returns
        -------
        int
            Current exact species count.
        """
        self._require_initialized()
        try:
            return self._core.get_species_count(pattern)
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey get_species_count failed: {e}") from e

    def add_species(self, pattern: str, count: int) -> None:
        """Add exact BNGL species instances to the live RuleMonkey session.

        Parameters
        ----------
        pattern : str
            Exact, fully-specified, connected BNGL species pattern.
        count : int
            Number of instances to add. Must be > 0. A fractional value is
            rounded to the nearest integer (round-half-up, GH #51).
        """
        self._require_initialized()
        count = round_half_up(count)
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        try:
            self._core.add_species(pattern, count)
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey add_species failed: {e}") from e

    def remove_species(self, pattern: str, count: int) -> None:
        """Remove exact BNGL species instances from the live RuleMonkey session.

        Parameters
        ----------
        pattern : str
            Exact, fully-specified, connected BNGL species pattern.
        count : int
            Number of instances to remove. Must be > 0. A fractional value is
            rounded to the nearest integer (round-half-up, GH #51). Raises if
            fewer than *count* copies are live.
        """
        self._require_initialized()
        count = round_half_up(count)
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        try:
            self._core.remove_species(pattern, count)
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey remove_species failed: {e}") from e

    def set_species_count(self, pattern: str, count: int) -> None:
        """Set the live count for an exact BNGL species pattern.

        Drives the live count of *pattern* to exactly *count*, adding or
        removing the difference. A fractional *count* is rounded to the nearest
        integer (round-half-up, GH #51). Mirrors
        :meth:`bngsim.NfsimSession.set_species_count`; PyBioNetGen's
        ``_apply_nfsim_concentration_changes`` probes this method via
        ``getattr`` to propagate ``setConcentration`` to ``rm`` runs.
        """
        self._require_initialized()
        count = round_half_up(count)
        if count < 0:
            raise ValueError(f"count must be nonnegative, got {count}")
        try:
            self._core.set_species_count(pattern, count)
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey set_species_count failed: {e}") from e

    def evaluate(self, expr: str, overrides: dict[str, float] | None = None) -> float:
        """Evaluate a BNG expression against the live session's current state.

        Resolvable symbols: every declared parameter, the clock ``t`` /
        ``time()``, every observable, and every global function — all
        settled against the current pool, exactly as a rate law would see
        them between events. *overrides* supplies extra ``name → value``
        bindings layered on top for a single evaluation; an override name
        shadows a model symbol on a clash.

        Note
        ----
        Unlike :meth:`bngsim.NfsimSession.evaluate` (which only requires a
        live simulator), the RuleMonkey engine resolves expressions against
        the active session's pool, so this method requires an **initialized**
        session — call :meth:`initialize` first.

        Parameters
        ----------
        expr : str
            Expression string.
        overrides : dict[str, float], optional
            Extra ``name → value`` bindings for this evaluation only.

        Returns
        -------
        float
            Numeric value of the expression.

        Raises
        ------
        ParameterError
            If the expression fails to compile or reference an unknown symbol.
        """
        self._require_initialized()
        try:
            return self._core.evaluate_expression(expr, overrides or {})
        except (RuntimeError, ValueError) as e:
            raise ParameterError(f"Failed to evaluate expression '{expr}': {e}") from e

    # ── State persistence ────────────────────────────────────────

    def save_species(self, path: str | Path) -> None:
        """Write the live session's molecular species to a BNG ``.species`` file.

        Exact peer of :meth:`bngsim.NfsimSession.save_species`. Enumerates
        the live complex pool, deduplicates by graph isomorphism, and writes
        a BNG-format ``.species`` file (``#`` comment header followed by one
        ``<pattern>  <count>`` line per species) readable by BNG2.pl's
        ``readNFspecies``. This is the artifact PyBioNetGen's ``simulate_nf``
        hook reads to thread ``get_final_state`` continuation across
        ``saveConcentrations`` / ``resetConcentrations`` segments — binding
        it makes multi-segment ``method=>"rm"`` runs reproduce the subprocess
        trajectory with no PyBioNetGen change.

        Parameters
        ----------
        path : str or Path
            Output file path. Overwritten if it exists.

        Raises
        ------
        SimulationError
            If the session is not initialized or the file cannot be written.
        """
        self._require_initialized()
        try:
            self._core.save_species(str(path))
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey save_species failed: {e}") from e

    def save_state(self, path: str | Path) -> None:
        """Save the active session state to a binary snapshot file.

        Serializes the full molecular pool plus the RNG state, keyed by a
        fingerprint of the molecule-type schema. A later :meth:`load_state`
        (on a session built from the same XML) resumes the trajectory
        exactly. This is RuleMonkey's in-process state-continuation path;
        NFsim has no equivalent (here RuleMonkey is ahead).

        For threading state into a BNG2.pl-driven host like PyBioNetGen,
        prefer :meth:`save_species` — that goes through BNG's text
        ``.species`` + ``readNFspecies`` mechanism, which is what the
        backend hook actually calls.

        Caveats
        -------
        - The schema fingerprint covers **molecule-type schema only**
          (molecule types, components, allowed states) — not ReactionRule or
          Observable patterns. Loading into a simulator built from a
          different XML that keeps the molecule schema but changes rules /
          observables succeeds silently and continues with the *new* rules.
        - The RNG state is serialized via the C++ stdlib's ``mt19937_64``
          stream operator, which is **not** specified to be byte-identical
          across stdlib implementations. Save/load is reliable only between
          binaries built against the same toolchain.

        Parameters
        ----------
        path : str or Path
            Output snapshot path. Overwritten if it exists.

        Raises
        ------
        SimulationError
            If the session is not initialized or the file cannot be written.
        """
        self._require_initialized()
        try:
            self._core.save_state(str(path))
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey save_state failed: {e}") from e

    def load_state(self, path: str | Path) -> None:
        """Load session state from a snapshot written by :meth:`save_state`.

        Creates a new live session from the snapshot, replacing any existing
        one. The snapshot's schema fingerprint must match this session's XML
        (see :meth:`save_state` caveats). On success the session is
        initialized and :attr:`current_time` reflects the snapshot's logical
        time — feed it to ``simulate(t_start, ...)`` to resume sampling.

        The RNG seed is **not** recoverable from a snapshot, so
        :attr:`seed` reads back ``None`` after :meth:`load_state` even though
        the underlying RNG stream is restored.

        Parameters
        ----------
        path : str or Path
            Snapshot path written by :meth:`save_state`.

        Raises
        ------
        SimulationError
            If the snapshot cannot be read or its schema fingerprint does not
            match this session's model.
        """
        self._require_alive()
        try:
            self._core.load_state(str(path))
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey load_state failed: {e}") from e
        self._initialized = True
        self._seed = None
        logger.debug("RuleMonkeySession state loaded from %s", path)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def seed(self) -> int | None:
        """The integer RNG seed used by ``initialize()``, or ``None`` if
        the session has not been initialized yet."""
        return self._seed

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def current_time(self) -> float:
        self._require_initialized()
        return self._core.current_time()

    def _require_alive(self) -> None:
        if self._destroyed:
            raise SimulationError(
                "RuleMonkeySession has been destroyed. Create a new session to continue."
            )

    def _require_initialized(self) -> None:
        self._require_alive()
        if not self._initialized:
            raise SimulationError(
                "RuleMonkeySession is not initialized. Call initialize(seed) first."
            )

    def __del__(self) -> None:
        if not self._destroyed:
            self.destroy()

    def __repr__(self) -> str:
        state = (
            "destroyed" if self._destroyed else "initialized" if self._initialized else "created"
        )
        return f"RuleMonkeySession(xml_path={self._xml_path!r}, state={state})"
