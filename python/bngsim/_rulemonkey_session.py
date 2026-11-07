"""bngsim.RuleMonkeySession — Stateful session API for RuleMonkey simulation."""

from __future__ import annotations

import contextlib
import logging
import math
import warnings
from pathlib import Path
from typing import Any, NoReturn

from bngsim._exceptions import (
    ParameterError,
    RuleMonkeyWarning,
    SimulationError,
    SimulationTimeout,
)
from bngsim._extension_handle import raise_cannot_pickle, raise_cannot_shallow_copy
from bngsim._result import Result
from bngsim._rounding import round_half_up
from bngsim._sample_times import normalize_sample_times

logger = logging.getLogger("bngsim")


def warn_rulemonkey_diagnostics(core: Any, stacklevel: int = 1) -> None:
    """Emit a :class:`bngsim.RuleMonkeyWarning` for each warning-level
    diagnostic RuleMonkey reported loading *core*'s XML (issue #1022).

    *stacklevel* counts from the caller of this function, as for
    :func:`warnings.warn`: 1 points at that caller, 2 at its caller.
    """
    for message in core.unsupported_warnings():
        warnings.warn(f"RuleMonkey: {message}", RuleMonkeyWarning, stacklevel=stacklevel + 1)


def _on_session_clock(t: float, session_t: float) -> bool:
    """Whether a segment starting at *t* starts on the session clock: the
    tolerance upstream's session ``simulate()`` allows (``on_session_clock`` in
    ``src/rulemonkey_simulator.cpp``)."""
    return abs(t - session_t) <= 1e-9 * max(1.0, abs(session_t))


class RuleMonkeySession:
    """Stateful RuleMonkey simulation session with per-step control.

    RuleMonkey reads BNG XML and implements the ``nf_exact`` network-free
    backend. Use this class for multi-action workflows that need to initialize,
    simulate, add molecules, and simulate again.

    The session clock is the model's clock: a rate law that reads ``time()``,
    or a table function indexed by time, reads absolute model time, as
    ``method="ode"`` and ``method="ssa"`` do (issue #1022). See
    :meth:`initialize` for where it starts and :meth:`simulate` for a segment
    that starts elsewhere.

    Events are placed on time elapsed since the session's start wherever the
    model allows, which keeps their waiting times from rounding away at a large
    start (issue #1063): RuleMonkey reads a copy of the XML whose clock is
    ``time()`` plus the start, as ``Simulator.run`` does, and every time this
    class takes or reports is model time. A model with a table function
    indexed by time runs on model time itself (see
    ``bngsim/_rulemonkey_clock.py``).
    """

    __slots__ = (
        "_core",
        "_xml_path",
        "_initialized",
        "_destroyed",
        "_seed",
        "_clock_readers",
        "_start_unset",
        "_offset",
        "_t0",
        "_eval_symbols",
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

        from bngsim._network_free_checks import (
            refuse_rulemonkey_compartments,
            refuse_rulemonkey_match_once,
            rulemonkey_clock,
        )

        refuse_rulemonkey_compartments(self._xml_path)
        refuse_rulemonkey_match_once(self._xml_path)
        # The functions that read the clock, for which a segment's labels must
        # be the session clock (see simulate). Where the model reads it through
        # its functions alone, the core reads it as time() + __bngsim_t0, set
        # to the session's start, and runs on elapsed time (issue #1063).
        self._clock_readers, tables = rulemonkey_clock(self._xml_path)
        self._offset = bool(self._clock_readers) and not tables
        self._t0 = 0.0
        self._eval_symbols: set[str] | None = None  # read on the first evaluate()
        if self._offset:
            from bngsim._rulemonkey_clock import offset_clock_xml

            with offset_clock_xml(self._xml_path) as rm_xml:
                self._core = RuleMonkeySimulator(rm_xml)
        else:
            self._core = RuleMonkeySimulator(self._xml_path)
        warn_rulemonkey_diagnostics(self._core, stacklevel=2)
        if molecule_limit is not None:
            self._core.set_molecule_limit(int(molecule_limit))
        self._core.set_block_same_complex_binding(bool(block_same_complex_binding))

        self._core.set_reads_clock(bool(self._clock_readers) and not self._offset)

        self._seed: int | None = None
        # Whether the session's start is still the first segment's to set (see
        # initialize).
        self._start_unset = False

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

    def initialize(self, seed: int | None = None, t_start: float | None = None) -> None:
        """Initialize the RuleMonkey session.

        Parameters
        ----------
        seed : int, optional
            RNG seed. ``None`` (default) draws a fresh seed; pass an
            integer for reproducibility. The actual seed used is
            exposed via the :attr:`seed` property and stamped onto every
            ``Result`` returned by :meth:`simulate`.
        t_start : float, optional
            For a model that reads the clock (``time()``, or a table function
            indexed by time), where the seed state sits on the model's clock:
            :attr:`current_time` reads it, and a rate law that reads ``time()``
            starts from its value there. ``None`` (default) starts the clock at
            0, and lets the first :meth:`simulate` move the seed state to its
            own start if nothing has moved the session first (:meth:`step_to`,
            or a change to its molecules), so that ``initialize(seed)`` then
            ``simulate(5, 7, ...)`` runs the model from its initial state at
            t = 5, as ``Simulator.run(t_span=(5, 7))`` does (issue #1022). A
            model that reads no clock cannot tell, and its session always
            starts at 0, which keeps event times small; a segment from
            ``t_start`` is labelled with its own times.
        """
        from bngsim._seed import _resolve_seed

        self._require_alive()
        used_seed = _resolve_seed(seed)
        t0 = 0.0 if t_start is None else float(t_start)
        if not math.isfinite(t0):
            raise ValueError(f"t_start must be finite, got {t_start!r}")
        try:
            self._start_core(used_seed, t0)
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey initialization failed: {e}") from e
        self._initialized = True
        self._seed = used_seed
        self._start_unset = t_start is None
        logger.info("RuleMonkeySession initialized (seed=%d, t_start=%g)", used_seed, t0)

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

        A segment that starts at :attr:`current_time` runs on the session
        clock, the model's. One that starts elsewhere continues from
        :attr:`current_time` by the same span and is labelled with the times
        asked for, which a model that does not read the clock cannot tell
        apart. For a model that reads the clock (``time()``, or a table
        function indexed by time) the labels would not be the times it read,
        so such a segment raises :class:`bngsim.SimulationError`, unless
        :meth:`initialize` left the start unset and nothing has moved the
        session since: the seed state then moves to the segment's start.

        The result's ``solver_stats["n_steps"]`` counts the events since
        :meth:`initialize`, not this segment's alone, as an
        :class:`bngsim.NfsimSession`'s does.

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

        moved = self._place_segment(float(t_start))
        try:
            try:
                core_result = self._core.simulate(
                    float(t_start), float(t_end), int(n_points), timeout_seconds, explicit_times
                )
            except BaseException:
                if moved:
                    # A segment that fails leaves the start where initialize() left it.
                    self._restore_unset_start()
                raise
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

        self._start_unset = False
        try:
            self._core.step_to(float(time) - self._t0, timeout_seconds)
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
        self._start_unset = False
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
        self._start_unset = False
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
        self._start_unset = False
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
        self._start_unset = False
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
        if self._offset:
            from bngsim._network_free_checks import rulemonkey_clock_ref
            from bngsim._rulemonkey_clock import OFFSET_PARAM, evaluate_symbols

            # The engine's clock is elapsed time: read the model's where the
            # expression reads the clock, as the model's own functions do.
            if self._eval_symbols is None:
                self._eval_symbols = evaluate_symbols(self._xml_path)
            clock = rulemonkey_clock_ref(self._eval_symbols | set(overrides or ()))
            expr = clock.sub(f"(time()+{OFFSET_PARAM})", expr)
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
            if self._offset:
                from bngsim._rulemonkey_clock import state_to_model_time

                # The snapshot holds the model's time, as one on model time does.
                state_to_model_time(path, self._t0)
        except (RuntimeError, OSError, ValueError) as e:
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
            if self._offset:
                from bngsim._rulemonkey_clock import OFFSET_PARAM, state_at_elapsed_zero

                # The session restarts its elapsed clock at the snapshot's time.
                with state_at_elapsed_zero(path) as (at_zero, t0):
                    self._core.destroy_session()  # no set_param while a session lives
                    self._core.set_param(OFFSET_PARAM, t0)
                    self._core.load_state(at_zero)
                self._t0 = t0
            else:
                self._core.load_state(str(path))
        except RuntimeError as e:
            raise SimulationError(f"RuleMonkey load_state failed: {e}") from e
        self._initialized = True
        self._seed = None
        self._start_unset = False
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
        return self._t0 + self._core.current_time()

    def _start_core(self, seed: int, t_start: float) -> None:
        """Start the core's session with the seed state at *t_start* on the
        model's clock: on elapsed time from 0 with ``__bngsim_t0`` at *t_start*
        where the model is offset, at *t_start* itself where it reads the clock
        otherwise, and at 0 where it reads none."""
        if self._offset:
            from bngsim._rulemonkey_clock import OFFSET_PARAM

            self._core.destroy_session()  # no set_param while a session lives
            self._core.set_param(OFFSET_PARAM, t_start)
            self._core.initialize(seed, 0.0)
            self._t0 = t_start
        else:
            self._core.initialize(seed, t_start if self._clock_readers else 0.0)

    def _place_segment(self, start: float) -> bool:
        """Check a segment starting at *start* against the session clock, for a
        model that reads it, and move an unset start there (see simulate).
        Whether it moved the start."""
        start_unset, self._start_unset = self._start_unset, False
        if not self._clock_readers:
            return False
        now = self.current_time
        if _on_session_clock(start, now):
            return False
        # Only initialize() leaves the start unset (at 0), and it sets the seed.
        if start_unset and self._seed is not None:
            try:
                self._start_core(self._seed, start)
            except RuntimeError as e:
                # A failed initialize() leaves no session: put back the one there was.
                self._restore_unset_start()
                raise SimulationError(f"RuleMonkey initialization failed: {e}") from e
            return True
        raise SimulationError(
            f"RuleMonkeySession cannot run a segment from t = {start:g}: the session "
            f"clock is at {now:g}, and the model reads the clock in "
            + ", ".join(f"'{n}'" for n in self._clock_readers)
            + ", so the rows would not be labelled with the times it read. Start the "
            "segment at current_time, or pass t_start to initialize() (issue #1022)."
        )

    def _restore_unset_start(self) -> None:
        """Put the session back as initialize() left it, with its start unset."""
        assert self._seed is not None
        self._start_core(self._seed, 0.0)
        self._start_unset = True

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
        # A constructor that raised before it set _destroyed (an unknown
        # keyword, which never enters __init__, or a build without this
        # engine) left nothing to release; reading the unset slot raised
        # AttributeError here, printed beside the real error.
        if not getattr(self, "_destroyed", True):
            self.destroy()

    def __reduce__(self) -> NoReturn:
        raise_cannot_pickle(
            "RuleMonkeySession",
            "Build a RuleMonkeySession(xml_path) in each worker, initialize() it there, and send "
            "results back as arrays or as files written with Result.save(path).",
        )

    def __copy__(self) -> NoReturn:
        # The default copy shared this session's handle into the RuleMonkey engine:
        # a simulate() through the copy advanced the original's system, and
        # dropping the copy destroyed the original's session, whose next
        # simulate() then failed with "No active session". As Model and
        # Simulator refuse (issue #643, bngsim._extension_handle).
        raise_cannot_shallow_copy(
            "RuleMonkeySession",
            "this session's RuleMonkey system, and destroys it when the copy goes",
            "Build a second RuleMonkeySession(xml_path) and initialize() it for an independent "
            "session.",
        )

    def __repr__(self) -> str:
        # getattr: a traceback that shows locals can repr a session whose
        # constructor raised before these were set.
        if getattr(self, "_destroyed", True):
            state = "destroyed"
        else:
            state = "initialized" if getattr(self, "_initialized", False) else "created"
        return f"RuleMonkeySession(xml_path={getattr(self, '_xml_path', None)!r}, state={state})"
