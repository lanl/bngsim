"""Paths that should give the same answer, run and compared (issue #1257).

A recurring kind of bngsim bug is two code paths that should agree and do not:
a plain run and a sensitivity run recording an event's row on different sides
of it (#1182), the same ``.net`` file under ``LC_NUMERIC=C`` and ``de_DE``
(#1235), KLU and the dense solver on an equality the state sits on (#1254).
Each was found by a one-off probe. This runs those probes for any model.

One model, one plain run (the *base*), and each check against it:

=====================  ========================================  =====================
check                  path                                      reference
=====================  ========================================  =====================
``sens_vs_plain``      the state of a sensitivity run            the base
``ajac_vs_fdjac``      ``jacobian="fd"``                         the base (``"auto"``)
``klu_vs_dense``       ``force_sparse_linear_solver=True``       the base (dense)
``cc_vs_interp``       ``codegen=True``                          ``codegen=False``
``cc_vs_mir``          ``BNGSIM_CODEGEN_JIT=mir``                ``codegen=True`` (cc)
``locale``             the run under a comma-decimal locale      the base, under ``C``
``grid``               the run at 8x the output points, read     the base
                       at the base's own sample times
``steady_vs_run``      ``steady_state()``                        a run to equilibrium
                                                                 from the same state
``newton_vs_run``      ``steady_state(method="newton")``         the same run
=====================  ========================================  =====================

and three rules for bugs whose answers are right, which no comparison of
answers sees:

* ``sens_runs``: the sensitivity run is refused while the plain run completes
  (#1229 refused a bend for a jump that did not exist). A refusal is
  UNSUPPORTED, not a finding: many are deliberate, by name. The nightly's
  baseline is what makes one a finding, when a model that ran is refused.
* ``sparsity_covers_jacobian``: every entry central differences of the
  right-hand side find nonzero has a slot in ``jacobian_sparsity`` (#1239: KLU
  factored a pattern that missed entries, and still agreed within tolerance).
* ``backend_as_asked``: ``codegen=False`` runs the interpreter, and
  ``codegen=True`` compiled code (#1213: ``codegen=False`` was ignored on MIR
  builds; same answers either way).

Each returns a :class:`Verdict`: ``pass``, ``diff`` (a finding, with its size),
``unsupported`` (the path declined by type: not a wrong answer), ``skip`` (the
path does not apply here, said why), ``exception`` (a path raised where the base
ran: a finding), ``no_base`` (the model does not load or its plain run does
not complete, so there is nothing to compare against: the parity suites own
that) or ``timeout`` (not finished within the model's cap, or a compile past
bngsim's own limit: that row only).

The oracle has to be careful
----------------------------
Every false positive three rounds of triage found came from the checking, not
from bngsim. So a disagreement between two paths counts only past how far the
reference moves by itself at a tenfold tighter tolerance (:func:`own_drift`):
a model that amplifies every difference, a fast oscillator, makes two correct
paths disagree by that much, and so does one path against a tighter self
(``LV``: 3e-6 apart at rtol 1e-12, 76% at 1e-8). A base that moves by more
than :data:`UNJUDGEABLE_DRIFT` that way is no reference at all, and its paths
are not judged (SKIP, said why). And a cell is judged on the
trajectory's own scale, past an absolute floor of the run's ``atol`` scaled as
``rtol`` is (:data:`PATH_RTOL_FACTOR`), so a state of 1e6 is not held to an
absolute 1e-3. A trajectory wholly under that floor (a BioModel in molar units,
1e-6 at its atol of 1e-9) shows no difference inside it, so a path that stays
inside is SKIP, said why, not PASS; one that clears it is still judged.

Sensitivities against central differences are ``fd_sens``'s, not this module's:
one oracle with one set of rules (``parity_checks/fd_sens``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from _transforms import TRANSFORMS, Inapplicable, transform

#: How many times the run's rtol two correct paths may differ by, before the
#: reference's own drift is counted: their errors are each about rtol, scaled by
#: the problem's conditioning, which a tolerance does not bound.
PATH_RTOL_FACTOR = 1e3
#: And how far past the reference's own drift at a tenfold tighter tolerance.
DRIFT_FACTOR = 10.0
#: A base that moves by more than this at a tenfold tighter tolerance is not
#: converged at its own: no path is judged against it. MODEL2310250001 drives a
#: species through a pole of its uptake law; its plain run moves 2.6% at a
#: tighter tolerance, libRoadRunner's -1.4 to -4.6 across three, and its
#: sensitivity run 359 times the trajectory's peak.
UNJUDGEABLE_DRIFT = 1e-3
#: The parameters a sensitivity run takes, at most.
MAX_SENS_PARAMS = 3
#: The grid check's output points, as a multiple of the base's intervals.
GRID_FACTOR = 8
#: How long the steady-state checks' run is taken: ``steady_state()``'s own
#: default ``max_time``, so the run has the time the solve has.
STEADY_HORIZON = 1e6
#: A run has settled when its last quarter of the horizon moves it by no more
#: than this many times its rtol, on its own scale.
SETTLED_RTOL_FACTOR = PATH_RTOL_FACTOR
#: A root is unstable where its Jacobian has an eigenvalue whose real part is
#: past this share of the spectrum's largest magnitude: a conservation law's
#: zero is rounding, nowhere near it.
UNSTABLE_SHARE = 1e-6
#: The ``steady_state()`` methods judged, each against the same run.
STEADY_METHODS = {"steady_vs_run": "integration", "newton_vs_run": "newton"}
#: Comma-decimal locales, in the order tried.
COMMA_LOCALES = ("de_DE.UTF-8", "de_DE.utf8", "de_DE", "fr_FR.UTF-8", "fr_FR.utf8", "fr_FR")

#: The ``time.monotonic()`` every run of the model being checked must finish
#: by, which :func:`check_all` sets: None outside it, or with no cap.
_DEADLINE: float | None = None

CHECKS = (
    "sens_runs",
    "sens_vs_plain",
    "ajac_vs_fdjac",
    "klu_vs_dense",
    "cc_vs_interp",
    "cc_vs_mir",
    "locale",
    "grid",
    "steady_vs_run",
    "newton_vs_run",
    "sparsity_covers_jacobian",
    "jacobian_values",
    "backend_as_asked",
    # The SSA arm (#1310), on SSA jobs only.
    "ssa_cc_vs_interp",
    "ssa_backend_as_asked",
    # The NF arm (#1312), on NF jobs only.
    "nf_reject_vs_exact",
    # Metamorphic (#1305): the model file transformed, the answer known.
    "reorder",
    "rename",
    "split",
    "rescale_time",
    # State leaks (#1307): a restarted run, a reused Model or Simulator, a batch.
    "restart",
    "reuse",
    "batch",
    # #1309: the base's answer under a jittered grid and tolerance.
    "stable",
)


@dataclass
class Verdict:
    status: str  # pass | diff | unsupported | skip | exception | no_base | timeout
    value: float | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)
    # The two trajectories a comparison judged, kept for attribution (#1309)
    # and never written to a report.
    ref_x: np.ndarray | None = field(default=None, repr=False, compare=False)
    path_x: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class Job:
    """One model and how to run it: its window, grid and tolerances, and any
    Simulator options every run of it takes (``sim``)."""

    model_id: str
    path: Path
    t_start: float = 0.0
    t_end: float = 10.0
    n_points: int = 21
    rtol: float = 1e-8
    atol: float = 1e-10
    max_steps: int = 100_000
    sim: dict = field(default_factory=dict)
    # "ode", "ssa" for the stochastic arm (#1310) or "nf" for the network-free
    # one (#1312, ``path`` a BNG-XML), whose replicates run at seeds seed,
    # seed + 1, ... seed + n_rep - 1.
    method: str = "ode"
    n_rep: int = 0
    seed: int = 1

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "t_start": self.t_start,
            "t_end": self.t_end,
            "n_points": self.n_points,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
            "sim": self.sim,
            "method": self.method,
            "n_rep": self.n_rep,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        return cls(**{**d, "path": Path(d["path"])})


def load(path: Path):
    import bngsim

    path = Path(path)
    if path.suffix == ".net":
        return bngsim.Model.from_net(str(path))
    return bngsim.Model.from_sbml(str(path))


def sens_params(model) -> list[str]:
    """The first few of the model's knobs, less those a sensitivity is refused for
    by construction (compartment sizes, frozen parameters): fd_sens's choice."""
    excluded = set(getattr(model, "compartment_size_params", None) or ())
    excluded |= set(getattr(model, "frozen_params", None) or ())
    return [p for p in model.primary_param_names if p not in excluded][:MAX_SENS_PARAMS]


def _left() -> float | None:
    """The seconds to :data:`_DEADLINE`, a run's ``timeout``: None for none. Past
    it no run starts (``SimulationTimeout``)."""
    if _DEADLINE is None:
        return None
    left = _DEADLINE - time.monotonic()
    if left <= 0:
        import bngsim

        raise bngsim.SimulationTimeout("the model's cap is spent")
    return left


def _trigger_reads(model) -> set[str] | None:
    """The species an event trigger reads, through the functions it calls by
    name: None where one reads an observable that is not a species of its
    name (a ``.net`` group), whose species are not to be had here."""
    core = model._core
    todo = list(core.event_trigger_sources())
    if not todo:
        return set()
    bodies = dict(zip(core.function_names, core.function_expressions, strict=True))
    species, observables = set(model.species_names), set(core.observable_names)
    seen: set[str] = set()
    reads: set[str] = set()
    while todo:
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", todo.pop()):
            if name in seen:
                continue
            seen.add(name)
            if name in bodies:
                todo.append(bodies[name])
            elif name in species:
                reads.add(name)
            elif name in observables:
                return None
    return reads


def held_at_start(model, t: float) -> np.ndarray:
    """The species the model's equations hold still at its start: a derivative
    of exactly 0 there, on a species something in the equations can move (a
    row of the Jacobian's pattern with an entry, and not fixed), and that no
    event trigger reads. A value held exactly is what rounding alone can
    break, as BIOMD0000000908's S = 0 is broken. The rest are not a rounding's
    to move:

    * a species already moving, which the nudge only shifts in time;
    * a species nothing moves, an input: BIOMD0000000265's switches, which
      only its events write;
    * a species a trigger reads, which the nudge can carry across the
      trigger's threshold at the start, where no rounding reaches: in
      BIOMD0000000285, ``PIdeath > 0`` and ``p38death > 0`` both true at
      t = 0 fire neither (initialValue true, as a Level 2 trigger is), and
      the cell they kill at once lives. A model whose triggers read
      observables that are not plain species is not nudged at all.
    """
    x0 = model.get_state()
    held = (np.asarray(model.rhs(x0, t)) == 0.0) & ~np.asarray(model._core.species_is_fixed, bool)
    pattern = model._core.jacobian_sparsity
    moved = np.zeros(len(x0), dtype=bool)
    moved[np.asarray(pattern["row_indices"], dtype=int)] = True
    reads = _trigger_reads(model)
    if reads is None:
        return np.zeros(len(x0), dtype=bool)
    read = np.array([name in reads for name in model.species_names], dtype=bool)
    return held & moved & ~read


def run(job: Job, nudge: float = 0.0, **kw):
    """A run of a fresh model (a run leaves its state on the Model it was given):
    the Result and the Simulator. It has until :data:`_DEADLINE`. ``nudge``
    moves the start of each species :func:`held_at_start` up by that share of
    what a step may err by in it, ``rtol·|x| + atol`` (:data:`START_NUDGE`)."""
    import bngsim

    model = load(job.path)
    if nudge:
        x0 = model.get_state()
        held = held_at_start(model, job.t_start)
        model.set_state(x0 + held * nudge * (job.rtol * np.abs(x0) + job.atol))
    sim = bngsim.Simulator(model, method="ode", **{**job.sim, **kw})
    result = sim.run(
        t_span=(job.t_start, job.t_end),
        n_points=job.n_points,
        rtol=job.rtol,
        atol=job.atol,
        max_steps=job.max_steps,
        timeout=_left(),
    )
    return result, sim


def own_drift(job: Job, base: np.ndarray) -> float:
    """How far the base moves at a tenfold tighter tolerance, on its own scale."""
    tight = Job(
        **{**job.__dict__, "rtol": job.rtol / 10.0, "atol": job.atol / 10.0, "sim": job.sim}
    )
    x, _ = run(tight)
    return gap(np.asarray(x.species, dtype=float), base, job.atol)


def gap(x: np.ndarray, ref: np.ndarray, atol: float) -> float:
    """``max |x - ref|`` over the reference's peak, past an absolute floor of
    ``atol``: 0 where every cell is within atol. A shape or finiteness mismatch
    is ``inf``."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    if x.shape != ref.shape or not np.array_equal(np.isfinite(x), np.isfinite(ref)):
        return float("inf")
    finite = np.isfinite(ref)
    if not finite.any():
        return 0.0
    over = np.abs(x - ref)[finite] - atol
    peak = float(np.abs(ref[finite]).max())
    return float(max(over.max(), 0.0) / max(peak, atol))


class Base:
    """The base run, and the bound two paths may differ by, computed once."""

    def __init__(self, job: Job):
        result, sim = run(job)
        self.job = job
        self.x = np.asarray(result.species, dtype=float)
        # The columns of x, by name: a Result's species, which leave out a
        # parameter an event promotes into the state (Model.species_names has it).
        self.species_names = list(result.species_names)
        # The linear solver the base ran on (0 dense, 1 KLU, 2 BLAS dense), so
        # attribution can run it on the other one (#1309).
        self.linear_solver = int(result.solver_stats.get("linear_solver", 0))
        self.backend = sim.codegen_backend
        # A tenfold tighter run that does not complete leaves the base no
        # reference, not the model unrun: the checks that compare no answers
        # (sparsity, backend, a sensitivity run's refusal) still apply.
        self.drift_error = ""
        try:
            self.drift = own_drift(job, self.x)
        except Exception as e:  # noqa: BLE001 - recorded on every comparison's note
            self.drift = float("inf")
            self.drift_error = f"{type(e).__name__}: {_first_line(e)}"
        self.limit = max(PATH_RTOL_FACTOR * job.rtol, DRIFT_FACTOR * self.drift)

    def against(self, x) -> Base:
        """This base's bound and drift, with ``x`` as the reference: another path
        of the same model (the dense solver, the interpreter, cc)."""
        ref = Base.__new__(Base)
        ref.__dict__.update(self.__dict__, x=np.asarray(x, dtype=float))
        return ref

    def compare(self, x, label: str = "") -> Verdict:
        floor = PATH_RTOL_FACTOR * self.job.atol
        g = gap(x, self.x, floor)
        note = f"{g:.2e} of the trajectory's peak (bound {self.limit:.1e}, own drift {self.drift:.1e})"
        if self.drift_error:
            return Verdict(
                "skip",
                g,
                f"{label}no reference: the base at a tenfold tighter tolerance did not "
                f"complete ({self.drift_error}); {note}",
            )
        if not self.drift <= UNJUDGEABLE_DRIFT:
            return Verdict(
                "skip",
                g,
                f"{label}no reference: the base itself moves by {self.drift:.1e} at a tenfold tighter tolerance; {note}",
            )
        finite = np.abs(self.x[np.isfinite(self.x)])
        peak = float(finite.max()) if finite.size else 0.0
        if g == 0.0 and not peak > floor:
            return Verdict(
                "skip",
                g,
                f"{label}nothing could count: the trajectory's peak {peak:.1e} is under the "
                f"absolute floor {floor:.1e} ({PATH_RTOL_FACTOR:g}·atol); {note}",
            )
        status = "pass" if g <= self.limit else "diff"
        return Verdict(
            status, g, f"{label}{note}", ref_x=self.x, path_x=np.asarray(x, dtype=float)
        )


def _first_line(e: BaseException) -> str:
    return str(e).splitlines()[0][:300] if str(e) else ""


def _declined(e: Exception) -> bool:
    """A refusal by type, or a sensitivity run's refusal by name at run time: a
    SimulationError whose message says forward sensitivity is what it declines
    (an event not smooth in its parameter, a trigger that starts on its
    threshold, a delay; #1229's class). A crash says no such thing."""
    import bngsim

    if isinstance(e, bngsim.SensitivityUnsupportedError):
        return True
    text = str(e).lower()
    return isinstance(e, bngsim.SimulationError) and (
        "forward sensitivity" in text or "forward-sensitivity run" in text
    )


#: The words bngsim's compile timeout is reported in (``_codegen``), where
#: ``codegen=True``'s refusal carries it in its text and not as its cause.
_COMPILE_TIMED_OUT = "Codegen compilation timed out"


def _timed_out(e: BaseException) -> bool:
    """A run stopped by the model's cap, or a compile by bngsim's own limit
    (BIOMD0000000385's 4.2 MB of C past 600 s): a timeout, whatever wraps it,
    and never a crash."""
    import bngsim

    seen, depth = e, 0
    while seen is not None and depth < 16:
        if isinstance(seen, bngsim.SimulationTimeout | subprocess.TimeoutExpired):
            return True
        seen, depth = seen.__cause__ or seen.__context__, depth + 1
    return _COMPILE_TIMED_OUT in str(e)


def _verdict_of(e: Exception) -> Verdict:
    if _timed_out(e):
        # A cap reached in this check: its own row, not every row.
        if _COMPILE_TIMED_OUT in str(e):
            why = f"the compile timed out: {type(e).__name__}: {_first_line(e)}"
        else:
            why = f"not finished within the model's cap: {_first_line(e)}"
        return Verdict("timeout", None, why)
    status = "unsupported" if _declined(e) or isinstance(e, _Declined) else "exception"
    return Verdict(status, None, f"{type(e).__name__}: {_first_line(e)}")


def _guard(fn) -> Verdict:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - a path's failure is that check's verdict
        return _verdict_of(e)


def check_sens(base: Base) -> dict[str, Verdict]:
    """``sens_runs`` and ``sens_vs_plain``: a sensitivity run's state is the plain
    run's. Its sensitivity RHS (analytic or CVODES' difference quotient) is
    recorded, the #1214 class: a slow fallback with right answers."""
    params = sens_params(load(base.job.path))
    if not params:
        skip = Verdict("skip", note="no parameter to differentiate")
        return {"sens_runs": skip, "sens_vs_plain": skip}
    try:
        result, sim = run(base.job, sensitivity_params=params)
    except Exception as e:  # noqa: BLE001 - that it did not complete is the verdict
        if _timed_out(e):
            v = _verdict_of(e)
            return {"sens_runs": v, "sens_vs_plain": v}
        # Where the plain run completes and the sensitivity run does not: a
        # refusal by name, or a derivative that does not exist at the start
        # (a power of a species at 0), or a crash. The rule records it; the
        # baseline makes a new one a finding. Telling them apart by the text
        # of the message would be guesswork.
        skip = Verdict("skip", note="the sensitivity run did not complete")
        why = f"{type(e).__name__}: {_first_line(e)}"
        return {"sens_runs": Verdict("unsupported", None, why), "sens_vs_plain": skip}
    rhs = "analytic" if sim.has_analytic_sens_rhs else "difference_quotient"
    extra = {"sens_rhs": rhs, "params": params}
    if not sim.has_analytic_sens_rhs:
        extra["sens_rhs_decline_reason"] = str(sim.sens_rhs_decline_reason)[:300]
    v = base.compare(result.species)
    v.extra = extra
    return {"sens_runs": Verdict("pass", note=rhs, extra=extra), "sens_vs_plain": v}


def check_ajac(base: Base) -> Verdict:
    if base.job.sim.get("jacobian") == "fd":
        return Verdict("skip", note='the model runs on jacobian="fd"')
    return _guard(lambda: base.compare(run(base.job, jacobian="fd")[0].species))


def _sparse_declined(e: Exception) -> bool:
    # bngsim's own refusal of the option, by name: a model with no sparse
    # Jacobian to hand KLU under the asked strategy (a SimulationError, not a
    # type of its own).
    return "force_sparse_linear_solver" in str(e) and "cannot supply a sparse Jacobian" in str(e)


def check_klu(base: Base) -> Verdict:
    """KLU against the dense solver on the same Jacobian, so the one thing that
    differs is the linear solver and the Jacobian it is handed (KLU's FD
    Jacobian is colored over ``jacobian_sparsity``; the dense one differences
    every column). The FD Jacobian first, where #1254 was found; the analytic
    one where bngsim declines KLU with the FD one.

    The reference is forced dense, not left to the automatic rule: that rule
    sends a model of 50 species or more with a sparse Jacobian to KLU, and the
    base with it, so an unforced reference there is KLU against KLU."""
    import bngsim

    if not bngsim._bngsim_core.HAS_KLU:
        return Verdict("skip", note="a build without KLU")

    def go():
        for jac in ("fd", "auto"):
            try:
                x, _ = run(base.job, force_sparse_linear_solver=True, jacobian=jac)
            except bngsim.SimulationError as e:
                if _sparse_declined(e):
                    continue
                raise
            dense, _ = run(base.job, force_dense_linear_solver=True, jacobian=jac)
            return base.against(dense.species).compare(
                x.species, f'jacobian="{jac}", KLU against dense: '
            )
        return Verdict("unsupported", None, "bngsim declines KLU on either Jacobian, by name")

    return _guard(go)


#: How a machine without a C compiler shows through bngsim's build: Windows'
#: compiler lookup, a ``cc`` that is not there (FileNotFoundError), and macOS's
#: ``cc`` shim without the Command Line Tools.
_NO_COMPILER = ("No C compiler found", "C compiler not found", "invalid active developer path")


def _compiled(job: Job):
    """A ``codegen=True`` run: the Result and the Simulator. Which backend ran is
    the caller's to judge: ``codegen=True`` builds compiled code or raises, so
    one that ran the interpreter is ``backend_as_asked``'s finding, not a
    machine without a compiler."""
    try:
        return run(job, codegen=True)
    except RuntimeError as e:
        if "codegen declined this model" in str(e):
            raise _Declined(_first_line(e)) from e
        if any(s in str(e) for s in _NO_COMPILER):  # a machine without cc: no cc path here
            raise _NoCompiler("codegen=True found no C compiler") from e
        raise


def _paths(base: Base, key: str, fn):
    """``fn()`` once per base: ``cc_vs_interp`` and ``backend_as_asked`` ask for
    the same two runs. A path that raised raises again."""
    memo = base.__dict__.setdefault("_paths", {})
    if key not in memo:
        try:
            memo[key] = (fn(), None)
        except Exception as e:  # noqa: BLE001 - re-raised to each caller
            memo[key] = (None, e)
    value, err = memo[key]
    if err is not None:
        raise err
    return value


def _interpreted(base: Base):
    return _paths(base, "codegen=False", lambda: run(base.job, codegen=False))


def _compiled_of(base: Base):
    return _paths(base, "codegen=True", lambda: _compiled(base.job))


class _NoCompiler(RuntimeError):
    pass


class _Declined(RuntimeError):
    """Codegen's refusal of a model, by name ("codegen declined this model:
    ..."), as against a build that failed."""


def check_cc_vs_interp(base: Base) -> Verdict:
    def go():
        interp, _ = _interpreted(base)
        try:
            compiled, sim = _compiled_of(base)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e))
        if sim.codegen_backend == "exprtk":
            # backend_as_asked's finding; here, the interpreter against itself
            return Verdict("skip", note="codegen=True ran exprtk: no compiled path to compare")
        return base.against(interp.species).compare(
            compiled.species, f"{sim.codegen_backend} against exprtk: "
        )

    return _guard(go)


@contextlib.contextmanager
def _env(name: str, value: str | None):
    """``name`` set to ``value`` (unset for None) for the block."""
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def check_cc_vs_mir(base: Base) -> Verdict:
    import bngsim

    if not bngsim.HAS_MIR:
        return Verdict("skip", note="a build without MIR")

    def go():
        # cc with the selector cleared: a run under BNGSIM_CODEGEN_JIT=mir (the
        # MIR leg sets it for its whole suite) would compare MIR with itself.
        try:
            with _env("BNGSIM_CODEGEN_JIT", None):
                cc, cc_sim = _compiled(base.job)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e))
        if cc_sim.codegen_backend != "cc":
            return Verdict(
                "diff",
                None,
                f"codegen=True without BNGSIM_CODEGEN_JIT ran {cc_sim.codegen_backend}",
            )
        with _env("BNGSIM_CODEGEN_JIT", "mir"):
            mir, sim = run(base.job, codegen=True)
        if sim.codegen_backend != "mir":
            return Verdict("diff", None, f"BNGSIM_CODEGEN_JIT=mir ran {sim.codegen_backend}")
        return base.against(cc.species).compare(mir.species, "mir against cc: ")

    return _guard(go)


def check_grid(base: Base) -> Verdict:
    """The answer must not depend on where it is read: a run asked for 8x the
    output points, read at the base's own sample times, is the base. #1244 was a
    pulse train stepped over at one grid and caught at another: each output
    point is a stop the integrator lands on, and a schedule's stops moved with
    them."""

    def go():
        fine = Job(**{**base.job.__dict__, "n_points": (base.job.n_points - 1) * GRID_FACTOR + 1})
        x, _ = run(fine)
        return base.compare(
            np.asarray(x.species, dtype=float)[::GRID_FACTOR], f"{GRID_FACTOR}x the points: "
        )

    return _guard(go)


def _steady_run(job: Job, state=None, rtol_scale: float = 1.0) -> np.ndarray:
    """A run to :data:`STEADY_HORIZON` from the model's initial state, or from
    ``state``: its samples at 0, 1/4, ..., all of the horizon."""
    import bngsim

    model = load(job.path)
    if state is not None:
        model.set_state(np.asarray(state, dtype=float))
    sim = bngsim.Simulator(model, method="ode", **job.sim)
    r = sim.run(
        t_span=(0.0, STEADY_HORIZON),
        n_points=5,
        rtol=job.rtol * rtol_scale,
        atol=job.atol * rtol_scale,
        max_steps=job.max_steps,
        timeout=_left(),
    )
    return np.asarray(r.species, dtype=float)


def _judged(model) -> np.ndarray:
    """The species a steady state is judged on: all but the write-only
    accumulators, which settle nowhere (all of them, where nothing else is)."""
    sink = np.asarray(model.is_pure_sink(), dtype=bool)
    return ~sink if not sink.all() else np.ones_like(sink)


class _Unsettled(RuntimeError):
    """The run has no state to compare against: it has not settled."""


def _settled_run(base: Base):
    """The run to equilibrium, once per base: its end, the species judged (those
    not write-only accumulators, which settle nowhere), and the bound two
    answers may differ by there, past how far the end moves at a tenfold
    tighter tolerance and over the horizon's last quarter."""
    if not hasattr(base, "_steady"):
        try:
            job = base.job
            keep = _judged(load(job.path))
            try:
                y = _steady_run(job)
                tight = _steady_run(job, rtol_scale=0.1)[-1]
            except Exception as e:  # noqa: BLE001 - the parity suites own a run that fails
                if _timed_out(e):
                    raise  # the model's cap, this check's TIMEOUT: not a missing reference
                raise _Unsettled(
                    f"no reference: the run to t={STEADY_HORIZON:g}, or at a tenfold tighter "
                    f"tolerance, did not complete ({type(e).__name__}: {_first_line(e)})"
                ) from e
            floor = PATH_RTOL_FACTOR * job.atol
            settle = gap(y[-1][keep], y[-2][keep], floor)
            if not settle <= SETTLED_RTOL_FACTOR * job.rtol:
                raise _Unsettled(
                    f"the run has not settled by t={STEADY_HORIZON:g}: it moves "
                    f"{settle:.1e} of its peak over the last quarter"
                )
            drift = gap(tight[keep], y[-1][keep], floor)
            limit = max(PATH_RTOL_FACTOR * job.rtol, DRIFT_FACTOR * max(settle, drift))
            base._steady = (y[-1], keep, limit, settle, drift, y[0])
        except Exception as e:  # noqa: BLE001 - every steady check reads it
            base._steady = e
    if isinstance(base._steady, Exception):
        raise base._steady
    return base._steady


def _refused(e: Exception) -> bool:
    """steady_state() refusing a model by name: one with an event, one that reads
    the time, or one whose conservation law its right-hand side breaks (#710,
    #758). Each says it is not supported for this model."""
    import bngsim

    return isinstance(e, bngsim.SimulationError) and "is not supported for this model" in str(e)


def _largest(x: np.ndarray, ref: np.ndarray, names, keep) -> str:
    i = int(np.argmax(np.where(keep, np.abs(x - ref), -1.0)))
    return f"{names[i]} at {x[i]:.6g} against {ref[i]:.6g}"


def check_steady(base: Base, method: str) -> Verdict:
    """``steady_state(method=...)`` is where a long run from the same initial
    state comes to rest (#1311). Judged only where both say they have arrived:
    the solve's ``converged``, and the run's last quarter of
    :data:`STEADY_HORIZON` moving it by no more than
    :data:`SETTLED_RTOL_FACTOR` times its rtol. Where the two differ, a run from
    the solve's state says which it is: one a run leaves is a ``converged`` that
    is not (DIFF), and so is one off the initial state's conserved totals or an
    unstable root; one the system rests at too is another stable state, the
    model's multistability or a tail slower than the horizon (SKIP, both named,
    for the baseline to hold)."""

    def go():
        import bngsim

        job = base.job
        model = load(job.path)
        keep = _judged(model)
        sim = bngsim.Simulator(model, method="ode", **job.sim)
        try:
            ss = sim.steady_state(
                method=method,
                rtol=job.rtol,
                atol=job.atol,
                max_time=STEADY_HORIZON,
                mask=None if keep.all() else keep,
            )
        except Exception as e:
            if _refused(e):
                return Verdict("unsupported", note=f"{type(e).__name__}: {_first_line(e)}")
            raise
        if not ss.converged:
            return Verdict(
                "skip",
                note=f"steady_state(method={method!r}) did not converge (residual {ss.residual:.1e})",
            )
        try:
            end, _, limit, settle, drift, start = _settled_run(base)
        except _Unsettled as e:
            return Verdict("skip", note=str(e))
        x = np.asarray(ss.concentrations, dtype=float)
        floor = PATH_RTOL_FACTOR * job.atol
        g = gap(x[keep], end[keep], floor)
        note = (
            f"{g:.2e} of the run's peak (bound {limit:.1e}; the run moves {settle:.1e} "
            f"over its last quarter, {drift:.1e} at a tenfold tighter tolerance)"
        )
        if g <= limit:
            return Verdict("pass", g, note)
        where = _largest(x, end, ss.species_names, keep)
        stays = gap(_steady_run(job, state=x)[-1][keep], x[keep], floor)
        if not stays <= limit:
            return Verdict(
                "diff",
                g,
                f"steady_state() converged where a run does not rest: a run from its state "
                f"moves {stays:.1e}; {where} at the run's end; {note}",
            )
        # Another state of the same system keeps the initial state's conserved
        # totals, which the run keeps: one that does not is another system's.
        model = load(job.path)
        laws = np.asarray(model.conservation_laws["coefficients"], dtype=float).reshape(-1, x.size)
        moved = np.abs(laws @ x - laws @ end) / np.maximum(np.abs(laws) @ np.abs(end), floor)
        if moved.size and not moved.max() <= limit:
            k = int(np.argmax(moved))
            return Verdict(
                "diff",
                g,
                f"steady_state() converged off the initial state's conservation law {k} "
                f"(total {laws[k] @ x:.6g} against the run's {laws[k] @ end:.6g}); "
                f"{where} at the run's end; {note}",
            )
        # A run from a root does not leave it, stable or not, if nothing pushes it
        # off: zero for an autocatalyst. Its spectrum says which.
        try:
            jac = np.asarray(model.jacobian(x), dtype=float)
        except bngsim.SimulationError as e:
            # Model.jacobian() refuses a difference quotient whose step decides
            # an entry (#1002): no spectrum, so stable or not is not judged, and
            # its refusal is no finding against steady_state().
            return Verdict(
                "skip",
                g,
                f"steady_state() rests where a run from it stays, apart from the run's "
                f"end, and its stability is not judged: {_first_line(e)}; {where} at the "
                f"run's end; {note}",
            )
        lam = np.linalg.eigvals(jac)
        top = float(lam.real.max()) if lam.size else 0.0
        if top > UNSTABLE_SHARE * float(np.abs(lam).max()):
            if gap(x[keep], start[keep], floor) <= limit:
                # The initial state is that root: the solve stays, as the
                # trajectory does in exact arithmetic, and the run leaves it on
                # rounding alone (BIOMD0000000518, at an eigenvalue of 5.6e-4).
                return Verdict(
                    "skip",
                    g,
                    f"the initial state is an unstable root (an eigenvalue of {top:.2e}): "
                    f"the solve stays on it, the run leaves it on rounding; {where} at "
                    f"the run's end; {note}",
                )
            return Verdict(
                "diff",
                g,
                f"steady_state() converged on an unstable root (an eigenvalue of {top:.2e}), "
                f"where the run comes to rest elsewhere; {where} at the run's end; {note}",
            )
        return Verdict(
            "skip",
            g,
            f"two stable states over the horizon: steady_state()'s, where a run from it "
            f"rests (moves {stays:.1e}), and the run's; the model's multistability, or a "
            f"tail slower than t={STEADY_HORIZON:g}; {where} at the run's end; {note}",
        )

    return _guard(go)


def check_backend(base: Base) -> Verdict:
    """codegen=False runs the interpreter; codegen=True compiled code."""

    def go():
        _, interp = _interpreted(base)
        asked = {"codegen=False": interp.codegen_backend}
        if interp.codegen_backend != "exprtk":
            return Verdict("diff", None, f"codegen=False ran {interp.codegen_backend}", asked)
        try:
            _, sim = _compiled_of(base)
        except _NoCompiler as e:
            return Verdict("skip", note=str(e), extra=asked)
        asked["codegen=True"] = sim.codegen_backend
        if sim.codegen_backend == "exprtk":
            return Verdict("diff", None, "codegen=True ran exprtk", asked)
        return Verdict("pass", note=", ".join(f"{k}: {v}" for k, v in asked.items()), extra=asked)

    return _guard(go)


def sparsity_misses(model, states, t=0.0, rel=1e-6, floor=1e-8) -> list[tuple[int, int]]:
    """The (row, column) entries central differences of the right-hand side find
    nonzero at any of ``states``, which ``jacobian_sparsity`` has no slot for.

    An entry is nonzero when, at both steps ``h`` and ``h/2`` (``rel`` of the
    state's own size, or of 1e-3 of the largest state where it is smaller), its
    contribution ``|J_ij|·scale_j`` clears ``floor`` of the largest one, and the
    two steps agree to 10%: an entry the differences cannot resolve is not
    asked of the pattern.
    """
    pat = model._core.jacobian_sparsity
    n = int(pat["n"])
    if n == 0:
        return []
    cols, rows = np.asarray(pat["col_ptrs"]), np.asarray(pat["row_indices"])
    slots = np.zeros((n, n), dtype=bool)
    for j in range(n):
        slots[rows[cols[j] : cols[j + 1]], j] = True
    missed: set[tuple[int, int]] = set()
    for y in states:
        y = np.asarray(y, dtype=float)
        big = float(np.abs(y).max()) if y.size else 0.0
        # On the state's own scale: a floor of 1e-3 absolute stepped a nanomolar
        # state by its whole size, past what a nonlinear law's two steps agree on
        scale = np.maximum(np.abs(y), (big if big > 0.0 else 1.0) * 1e-3)
        jac = []
        for h_rel in (rel, rel / 2.0):
            cols_ = []
            for j in range(n):
                h = h_rel * scale[j]
                up, dn = y.copy(), y.copy()
                up[j] += h
                dn[j] -= h
                cols_.append(
                    (np.asarray(model.rhs(up, t)) - np.asarray(model.rhs(dn, t))) / (2 * h)
                )
            jac.append(np.stack(cols_, axis=1) * scale[np.newaxis, :])
        c1, c2 = np.abs(jac[0]), np.abs(jac[1])
        top = max(float(c1.max()), float(c2.max()))
        if top == 0.0 or not np.isfinite(top):
            continue
        sure = (c1 > floor * top) & (c2 > floor * top) & (np.abs(c1 - c2) <= 0.1 * c2)
        missed |= {(int(i), int(j)) for i, j in zip(*np.nonzero(sure & ~slots), strict=True)}
    return sorted(missed)


def _states(job: Job) -> list[tuple[float, np.ndarray]]:
    """The engine's whole state, at the start, the middle and the end of the
    window: ``Model.get_state()`` after a run to each, which holds what the
    right-hand side reads, a parameter an event promotes included (a Result's
    species do not)."""
    import bngsim

    out = []
    for t in (job.t_start, 0.5 * (job.t_start + job.t_end), job.t_end):
        model = load(job.path)
        if t > job.t_start:
            try:
                bngsim.Simulator(model, method="ode", **job.sim).run(
                    t_span=(job.t_start, t),
                    n_points=2,
                    rtol=job.rtol,
                    atol=job.atol,
                    timeout=_left(),
                )
            except bngsim.SimulationError:
                # A state this one-interval run does not reach, where the base's
                # grid does (MODEL1112050001: CV_CONV_FAILURE before t = 50), is
                # not asked of the pattern, as a non-finite one is not: whether a
                # run completes is the comparisons' question, not the pattern's.
                continue
        y = np.asarray(model.get_state(), dtype=float)
        if np.all(np.isfinite(y)):
            out.append((t, y))
    return out


def check_sparsity(base: Base) -> Verdict:
    def go():
        model = load(base.job.path)
        states = _states(base.job)
        misses = sorted({m for t, y in states for m in sparsity_misses(model, [y], t=t)})
        if misses:
            names = list(model.species_names)
            shown = ", ".join(f"d{names[i]}/d{names[j]}" for i, j in misses[:5])
            return Verdict(
                "diff",
                float(len(misses)),
                f"{len(misses)} entries the differences find nonzero have no slot: {shown}",
                {"missed": misses[:50]},
            )
        return Verdict("pass", 0.0, f"pattern covers every entry at {len(states)} states")

    return _guard(go)


# jacobian_values (issue #1308). Each entry is differenced over a ladder of
# steps, JAC_STEPS of its column's scale, and each step's difference carries
# its own error bound: the larger of its gaps to the neighbouring steps', which
# moves with truncation at the coarse end and with rounding at the fine end,
# the steps being a decade apart. A step is resolvable where
# that bound is under JAC_AGREE of the entry, and has a say where it is under
# JAC_HEARD. The entry is off only where the analytic one contradicts the
# difference at every resolvable step, by more than ten times that step's bound
# plus JAC_TOL of its column's peak and JAC_FLOOR of the whole matrix's, and no
# step with a say bears it out (within twice its bound).
#
# A say for a step a little noisier than a resolvable one, because rounding
# can make a fine step look resolved and be wrong: in sqrt(1 + u) - 1 with
# u ≈ 1e-6 (BIOMD0000000393's quadratic roots, #533), a step under
# ulp(1)·N/4 leaves 1 + u as it was, so the term's whole slope rounds away and
# the fine steps agree, from 1e-8 to 1e-10, on 0.661 where the entry is
# -1.96e-4. The coarse steps, 2-3% apart from one another, read -1.96e-4. No
# bound on rounding from |f| could see that loss, which happens inside the law.
#
# Every step, and not the best one: a feature narrower than the coarser steps
# (a saturation k·s/(Km + s) with a Km far under the species' scale) holds two
# plateaus, the secant across it at coarse steps and the slope at the point at
# fine ones, and the analytic entry is the second. BIOMD0000000498's dS26/dS26
# is -13.74 to 1e-6 and -13.11 at 1e-8, the analytic -13.11; BIOMD0000000480's
# s1 sits at -1e-14 under a saturation with Km 1e-3, its analytic entry 2,000
# off the coarse secant. A wrong entry is wrong at every scale.
#
# Rounding in a large f is what the gaps bound: blbr's f is 1e11, and at a
# step of 1e-6 its differences came out 2.0345 at two steps a factor of two
# apart, alike, where the entry is 2; a decade either side they do not. The
# floor is what a near-zero column costs: mm_tqssa_stiff's carries
# 2e-19 against entries of order 1, which on its own peak reads 100% off.
JAC_STEPS = tuple(10.0**-k for k in range(2, 11))
JAC_AGREE = 1e-2
JAC_HEARD = 1e-1
JAC_SMOOTH = 1e-3
JAC_TOL = 1e-4
JAC_FLOOR = 1e-8


def jacobian_misfits(model, states, t=0.0, steps=JAC_STEPS) -> list[dict] | None:
    """The entries of the analytic Jacobian that central differences of the
    right-hand side contradict, at any of ``states``; ``None`` when
    ``Model.jacobian`` does not give the analytic one there (nothing of the
    closed form to judge).

    Each column is scaled by the state's own size, as :func:`sparsity_misses`
    scales it, and judged on its own peak (the larger of the analytic one and
    the difference's), so an entry in units of one species is never held to
    another's. An entry the differences cannot resolve at any step, or one on a
    kink, is not judged (see the constants above). A non-finite analytic entry
    is off wherever some step resolves the entry.
    """
    # Asked before Model.jacobian(), which refuses a difference quotient whose
    # step decides an entry (#1002) where it returned one: a model without the
    # closed form has nothing of it to judge, whichever it would do.
    prepare = getattr(model, "prepare_analytical_jacobian", None)
    if prepare is not None and not prepare():
        return None
    out: dict[tuple[int, int], dict] = {}
    for y in states:
        y = np.asarray(y, dtype=float)
        n = y.size
        if n == 0:
            continue
        J = model.jacobian(y, t)
        if getattr(J, "source", None) != "analytical":
            return None
        big = float(np.abs(y).max())
        scale = np.maximum(np.abs(y), max(big, 1.0) * 1e-3)
        f0 = np.asarray(model.rhs(y, t))
        cen, fwd, bwd = [], [], []
        for h_rel in steps:
            c_, f_, b_ = [], [], []
            for j in range(n):
                h = h_rel * scale[j]
                up, dn = y.copy(), y.copy()
                up[j] += h
                dn[j] -= h
                f_up, f_dn = np.asarray(model.rhs(up, t)), np.asarray(model.rhs(dn, t))
                c_.append((f_up - f_dn) / (2 * h))
                f_.append((f_up - f0) / h)
                b_.append((f0 - f_dn) / h)
            sc = scale[np.newaxis, :]
            cen.append(np.stack(c_, axis=1) * sc)
            fwd.append(np.stack(f_, axis=1) * sc)
            bwd.append(np.stack(b_, axis=1) * sc)
        cen, fwd, bwd = map(np.stack, (cen, fwd, bwd))  # (n_steps, n, n)
        finite = np.isfinite(cen)
        k_steps = len(steps)
        # The truncation bound: the larger gap to a neighbouring step.
        trunc = np.full_like(cen, np.inf)
        for k in range(k_steps):
            gaps = [np.abs(cen[k] - cen[m]) for m in (k - 1, k + 1) if 0 <= m < k_steps]
            trunc[k] = np.max(np.stack(gaps), axis=0)
        err = np.where(finite, trunc, np.inf)
        a = np.asarray(J, dtype=float) * scale[np.newaxis, :]
        mag = np.where(finite, np.abs(cen), 0.0)
        peak = np.maximum(
            mag.max(axis=(0, 1)), np.where(np.isfinite(a), np.abs(a), 0.0).max(axis=0)
        )
        top = float(peak.max()) if peak.size else 0.0
        size = np.maximum(mag, JAC_FLOOR * peak[np.newaxis, np.newaxis, :])
        resolvable = finite & (err <= JAC_AGREE * size)
        any_res = resolvable.any(axis=0)
        # A kink in the state (max(0, x) at x = 0, an if() on a species on its
        # threshold) is the #368 trap: the central differences converge, at
        # every step, to the mean of the two one-sided slopes, which no analytic
        # entry is. The one-sided differences tell it at the finest resolvable
        # step: they differ there by the slope's jump, and by O(h) where the law
        # is smooth. MODEL2105110001's NEU sits at -4e-216 on a kink whose sides
        # are 0.3% apart, and its analytic entry is the side it is on.
        finest = np.where(any_res, k_steps - 1 - np.argmax(resolvable[::-1], axis=0), 0)
        at = finest[np.newaxis]
        fwd_f, bwd_f, size_f, err_f = (
            np.take_along_axis(x, at, axis=0)[0] for x in (fwd, bwd, size, err)
        )
        kink = np.abs(fwd_f - bwd_f) > JAC_SMOOTH * size_f + 4.0 * err_f
        slack = JAC_TOL * peak[np.newaxis, :] + JAC_FLOOR * top
        contradicts = np.abs(a[np.newaxis] - cen) > 10.0 * err + slack[np.newaxis]
        everywhere = np.all(~resolvable | contradicts, axis=0)
        # A noisier step's say is a veto, and only where it bears the analytic
        # entry out, within twice its own bound: one that merely fails to
        # contradict it at ten times that bound would excuse any error.
        heard = finite & (err <= JAC_HEARD * size)
        vetoed = np.any(
            heard & (np.abs(a[np.newaxis] - cen) <= 2.0 * err + slack[np.newaxis]), axis=0
        )
        off = any_res & (~np.isfinite(a) | (~kink & everywhere & ~vetoed))
        best = np.argmin(np.where(resolvable, err, np.inf), axis=0)
        d_best = np.take_along_axis(cen, best[np.newaxis], axis=0)[0]
        for i, j in zip(*np.nonzero(off), strict=True):
            finite_a = np.isfinite(a[i, j]) and peak[j] > 0
            e = float(abs(a[i, j] - d_best[i, j]) / peak[j]) if finite_a else float("inf")
            key = (int(i), int(j))
            if key not in out or e > out[key]["off"]:
                out[key] = {
                    "entry": key,
                    "t": float(t),
                    "analytic": float(J[i, j]),
                    "difference": float(d_best[i, j] / scale[j]),
                    "off": e,
                }
    return sorted(out.values(), key=lambda m: -m["off"])


def check_jacobian_values(base: Base) -> Verdict:
    def go():
        model = load(base.job.path)
        states = _states(base.job)
        worst: dict[tuple[int, int], dict] = {}
        for t, y in states:
            found = jacobian_misfits(model, [y], t=t)
            if found is None:
                return Verdict(
                    "skip", note="no analytic Jacobian: Model.jacobian differences the RHS"
                )
            for m in found:
                if m["entry"] not in worst or m["off"] > worst[m["entry"]]["off"]:
                    worst[m["entry"]] = m
        if worst:
            names = list(model.species_names)
            ranked = sorted(worst.values(), key=lambda m: -m["off"])
            shown = ", ".join(
                f"d{names[m['entry'][0]]}/d{names[m['entry'][1]]} {m['off']:.1e}"
                for m in ranked[:5]
            )
            return Verdict(
                "diff",
                ranked[0]["off"],
                f"{len(ranked)} analytic entries off the differences (of their column's peak): {shown}",
                {"off": [{**m, "entry": list(m["entry"])} for m in ranked[:50]]},
            )
        return Verdict("pass", 0.0, f"every judged entry agrees at {len(states)} states")

    return _guard(go)


def check_transform(base: Base, kind: str) -> Verdict:
    """The model file transformed (``_transforms.py``, #1305) and run again: the
    answer must transform with it. Every other check here starts from the one
    loaded model, so a bug in the loader, the model build or the parser makes
    them agree on the wrong answer; this one reads the file a second way. Its
    trajectories are taken by name and judged as every path is, with
    ``Base.compare``: a transform changes the order of floating-point sums, so
    equality is not what is asked."""

    def go():
        with tempfile.TemporaryDirectory(prefix=f"differential_{kind}_") as d:
            try:
                tr = transform(kind, base.job.path, Path(d))
            except Inapplicable as e:
                return Verdict("skip", note=f"{kind} does not apply: {e}")
            c = tr.time_scale
            job = Job(
                **{
                    **base.job.__dict__,
                    "path": tr.path,
                    "t_start": base.job.t_start / c,
                    "t_end": base.job.t_end / c,
                }
            )
            result, _ = run(job)
        want = base.species_names
        # An SBML id bngsim cannot spell as it is (null, In, time, NaN: a word
        # its expressions reserve) is reported as _ant_<id>, and the map back
        # from a fresh name gives the id itself.
        spelled = {w[len("_ant_") :]: w for w in want if w.startswith("_ant_")}
        got = [tr.names.get(n, n) for n in result.species_names]
        got = [g if g in want else spelled.get(g, g) for g in got]
        if sorted(got) != sorted(want):
            odd = sorted(set(got) ^ set(want))[:5]
            return Verdict("diff", None, f"{kind}: the species are not the base's: {odd}")
        at = {n: i for i, n in enumerate(got)}
        x = np.asarray(result.species, dtype=float)[:, [at[n] for n in want]]
        return base.compare(x, f"{kind} ({tr.note}): ")

    return _guard(go)


# State-leak checks (issue #1307). Every other check runs a fresh model, by
# design; callers reuse one, and "the solver leaves scratch state in the
# user's Model, or drops state it should carry" is a family of its own (#786;
# #748, #705, #693). Each run here is compared with the base as every path is.

BATCH_FACTORS = (1.0, 1.1, 0.9, 1.05)
BATCH_THREADS = 4
# Attribution and branch stability (issue #1309). A DIFF says two paths
# disagree, not which is wrong, and a small own drift does not make an answer
# uncontested: on #1304's BIOMD0000000338 the KLU base moved 1e-7 at a tenfold
# tighter tolerance while dense and libRoadRunner sat 3.5% away, and dense once
# landed on KLU's branch.

ATTRIBUTION_TIGHTEN = 100.0
JITTER_RTOL = (0.5, 2.0)
#: The share of what a step may err by, ``rtol·|x| + atol``, that the stable
#: check moves each species' start up by: a start the solver cannot tell from
#: the base's. An answer that moves past its bound from it holds an unstable
#: solution by its rounding alone, which no jittered grid or tolerance shows
#: where the rounding keeps it exactly. BIOMD0000000908's S = 0 is one: its
#: clearance law is -n·S on a reaction that consumes S, so S' = +0.277·S. The
#: file's species order keeps S at 0; reversed, the rounding leaves 3.6e-19 at
#: t = 10, which grows to 1.09e+134 of the peak by t = 300.
START_NUDGE = 1e-3


def _arm_runs(base: Base) -> dict[str, np.ndarray]:
    """The third arms, each a trajectory in the base's columns: the base's own
    route at ``ATTRIBUTION_TIGHTEN`` x tighter tolerances, the base on the
    other linear solver, and libRoadRunner for SBML where it is installed. An
    arm that does not run, does not apply, or does not finish in what is left
    of the model's cap is left out."""
    import bngsim

    job, out = base.job, {}

    def take(name, fn):
        # An arm that cannot run (libRoadRunner gives up, KLU is declined) says
        # nothing, and is left out of the attribution.
        with contextlib.suppress(Exception):
            out[name] = fn()

    f = ATTRIBUTION_TIGHTEN
    tight = Job(**{**job.__dict__, "rtol": job.rtol / f, "atol": job.atol / f})
    take("tighter", lambda: _species(run(tight)[0], base))
    if base.linear_solver == 1:
        take(
            "dense",
            lambda: _species(run(job, force_dense_linear_solver=True)[0], base),
        )
    elif bngsim._bngsim_core.HAS_KLU:
        take(
            "klu",
            lambda: _species(run(job, force_sparse_linear_solver=True)[0], base),
        )
    # libRoadRunner takes no wall-clock budget: past the deadline it is not asked.
    if Path(job.path).suffix != ".net" and (_DEADLINE is None or time.monotonic() < _DEADLINE):
        take("roadrunner", lambda: _roadrunner(base))
    return out


def _roadrunner(base: Base) -> np.ndarray:
    import roadrunner

    job = base.job
    r = roadrunner.RoadRunner(str(job.path))
    r.integrator.relative_tolerance = job.rtol
    r.integrator.absolute_tolerance = job.atol
    # A species' concentration, as bngsim reports it, is [id] in libRoadRunner
    # (the bare id is its amount); a rate rule's parameter is its id. As
    # rr_parity selects them.
    species = set(r.model.getFloatingSpeciesIds()) | set(r.model.getBoundarySpeciesIds())
    sel = [f"[{n}]" if n in species else n for n in base.species_names]
    out = r.simulate(job.t_start, job.t_end, job.n_points, sel)
    return np.asarray(out, dtype=float)


def _species(result, base: Base) -> np.ndarray:
    """A Result's species in the base's column order."""
    names = list(result.species_names)
    x = np.asarray(result.species, dtype=float)
    return x[:, [names.index(n) for n in base.species_names]]


def _worst(verdicts: list[tuple[str, Verdict]]) -> Verdict:
    """The verdict of several comparisons: the first DIFF if any is, else the
    first that is not a PASS, else a PASS. A SKIP of one (point 0 against a
    base too unconverged to judge) must not hide another's DIFF (a threaded
    point against the same point run alone, which needs no base)."""
    bad = [v for _label, v in verdicts if v.status != "pass"]
    if bad:
        return next((v for v in bad if v.status == "diff"), bad[0])
    return Verdict(
        "pass", max(v.value or 0.0 for _, v in verdicts), "; ".join(v.note for _, v in verdicts)
    )


def _solver(job: Job) -> dict:
    return {"rtol": job.rtol, "atol": job.atol, "max_steps": job.max_steps}


def _timeout() -> dict:
    """What is left of the model's cap, for a ``run`` or ``run_batch`` (``run_until``
    takes none: a kill at the cap still leaves the model's other rows)."""
    return {"timeout": _left()}


def check_restart(base: Base) -> Verdict:
    """A run to the middle sample, continued to the end, is the base. Leg by
    leg with ``run_until`` on one Simulator, which carries the events' trigger
    state and pending delays across the stop (#693); and, for a model without
    events, a fresh Model given the state saved at the middle, which cannot
    know which triggers were already true and so is not asked to."""
    import bngsim

    def go():
        job = base.job
        n = job.n_points
        m = (n - 1) // 2
        t_mid = job.t_start + (job.t_end - job.t_start) * m / (n - 1)
        out = []

        sim = bngsim.Simulator(load(job.path), method="ode", **job.sim)
        if job.t_start != 0.0:
            sim.set_time(job.t_start)
        first = sim.run_until(t_mid, n_points=m + 1, **_solver(job))
        second = sim.run_until(job.t_end, n_points=n - m, **_solver(job))
        legs = np.vstack([_species(first, base), _species(second, base)[1:]])
        out.append(("run_until", base.compare(legs, "run_until to the middle and on: ")))

        model = load(job.path)
        if model._core.n_events == 0:
            sim = bngsim.Simulator(model, method="ode", **job.sim)
            head = sim.run(
                t_span=(job.t_start, t_mid), n_points=m + 1, **_solver(job), **_timeout()
            )
            state = np.asarray(sim.get_state(), dtype=float)
            fresh = bngsim.Simulator(load(job.path), method="ode", **job.sim)
            fresh.set_state(state, time=t_mid)
            tail = fresh.run(
                t_span=(t_mid, job.t_end), n_points=n - m, **_solver(job), **_timeout()
            )
            joined = np.vstack([_species(head, base), _species(tail, base)[1:]])
            out.append(("fresh", base.compare(joined, "a fresh model from the middle's state: ")))
        return _worst(out)

    return _guard(go)


def check_reuse(base: Base) -> Verdict:
    """A second run is the base: a second ``run()`` on the same Simulator, and
    a first one on a second Simulator over the same Model, each after the
    first run is undone with ``snapshot()``/``restore()`` (species, parameters,
    and the events' trigger state, #693). ``Model.reset()`` restores species
    only, so an event that writes a parameter would leave it written; restore
    is the documented way back."""
    import bngsim

    def go():
        job = base.job
        span = {"t_span": (job.t_start, job.t_end), "n_points": job.n_points, **_solver(job)}
        out = []

        sim = bngsim.Simulator(load(job.path), method="ode", **job.sim)
        snap = sim.snapshot()
        sim.run(**span, **_timeout())
        sim.restore(snap)
        again = sim.run(**span, **_timeout())
        out.append(
            ("run", base.compare(_species(again, base), "a second run() on one Simulator: "))
        )

        model = load(job.path)
        first = bngsim.Simulator(model, method="ode", **job.sim)
        snap = first.snapshot()
        first.run(**span, **_timeout())
        first.restore(snap)
        second = bngsim.Simulator(model, method="ode", **job.sim).run(**span, **_timeout())
        out.append(
            ("model", base.compare(_species(second, base), "a second Simulator on one Model: "))
        )
        return _worst(out)

    return _guard(go)


def check_batch(base: Base) -> Verdict:
    """``run_batch`` on ``BATCH_THREADS`` threads is the same points run one
    at a time, and its unchanged point is the base. The points scale the
    first parameter a sensitivity run would take (``sens_params``); a model
    with none runs the same point four times, which still asks whether the
    threads share anything."""
    import bngsim

    def go():
        job = base.job
        model = load(job.path)
        knobs = sens_params(model)
        if knobs:
            p, v = knobs[0], float(model.get_param(knobs[0]))
            points = [{p: v * f} for f in BATCH_FACTORS]
        else:
            points = [{} for _ in BATCH_FACTORS]
        kw = {
            "t_span": (job.t_start, job.t_end),
            "n_points": job.n_points,
            "params": points,
            **_solver(job),
        }
        # One at a time first: a point that does not integrate alone (a model
        # at the edge of its domain, moved 10% off its own parameter) is not
        # the batch's to judge. Threads failing where it runs is.
        try:
            serial = bngsim.Simulator(load(job.path), method="ode", **job.sim).run_batch(
                **kw, **_timeout()
            )
        except bngsim.SimulationError as e:
            first = str(e).splitlines()[0][:200]
            return Verdict(
                "skip", note=f"a point of the sweep does not run one at a time: {first}"
            )
        sim = bngsim.Simulator(model, method="ode", **job.sim)
        threaded = sim.run_batch(num_processors=BATCH_THREADS, **kw, **_timeout())
        out = [("point 0", base.compare(_species(threaded[0], base), "the batch's own point: "))]
        for i, (a, b) in enumerate(zip(threaded, serial, strict=True)):
            g = gap(_species(a, base), _species(b, base), PATH_RTOL_FACTOR * job.atol)
            note = (
                f"point {i} on {BATCH_THREADS} threads against one at a time: {g:.2e} of its peak"
            )
            out.append((f"point {i}", Verdict("pass" if g <= base.limit else "diff", g, note)))
        return _worst(out)

    return _guard(go)


def contested(rows: dict[str, dict]) -> bool:
    """Whether a model's rows mark it contested: its ``stable`` row reads DIFF
    (the answer jumps under a jittered grid, tolerance or start), or a DIFF's third
    arms split between its two sides (#1304's BIOMD0000000339: KLU at a
    hundredfold tighter tolerance sides with KLU, the dense solver with
    dense). Its DIFFs are real but unattributed, its PASSes
    luck. ``rows`` maps a check to ``{"status", "extra"}``, as a report holds
    them."""
    for check, row in rows.items():
        extra = row.get("extra") or {}
        if check == "stable" and row.get("status") == "diff":
            return True
        if (extra.get("attribution") or {}).get("verdict") == "split":
            return True
    return False


def attribute(base: Base, v: Verdict, arms: dict[str, np.ndarray] | None = None) -> dict:
    """Which side of a DIFF each third arm agrees with: the comparison's
    ``reference`` (the base, or the path it was judged against), its
    ``path``, ``both`` (the two are within the bound of the arm), or
    ``neither``. Its ``verdict`` is the one side every arm that took one
    took, ``split`` where they differ (the model is contested), or
    ``unattributed``."""
    arms = _arm_runs(base) if arms is None else arms
    atol = PATH_RTOL_FACTOR * base.job.atol
    sides = {}
    for name, x in arms.items():
        to_ref, to_path = gap(x, v.ref_x, atol), gap(x, v.path_x, atol)
        near_ref, near_path = to_ref <= base.limit, to_path <= base.limit
        side = {(True, False): "reference", (False, True): "path", (True, True): "both"}.get(
            (near_ref, near_path), "neither"
        )
        sides[name] = {"side": side, "to_reference": to_ref, "to_path": to_path}
    taken = {a["side"] for a in sides.values() if a["side"] in ("reference", "path")}
    verdict = taken.pop() if len(taken) == 1 else ("split" if taken else "unattributed")
    return {"verdict": verdict, "arms": sides}


def check_stable(base: Base) -> Verdict:
    """The base's answer under a jittered grid (one point fewer, one more,
    twice as many) and tolerance (rtol x0.5, x2), and from a start moved by
    :data:`START_NUDGE` of what a step may err by: a jump past the bound marks
    the model contested, its DIFFs real but unattributed and its PASSes luck.
    So does a run from the nudged start that fails where the base ran. The
    nudged start's jump counts only where its first row after the start is
    still within the bound: the perturbation grew.
    A grid of another length is read at its last point, which every grid
    shares; twice as many at every other point, which are the base's. Each
    rerun has what is left of the model's cap: one it stops makes the check
    TIMEOUT, its own row only."""
    import bngsim

    def go():
        # Where nothing could jump past the bound, SKIP, said why, as
        # Base.compare says it: a base whose tighter run failed (base.limit is
        # inf) or one wholly under the absolute floor. A base that moves past
        # UNJUDGEABLE_DRIFT is judged: a jump past ten times that is contested.
        unjudged = base.compare(base.x, "stable: ")
        if unjudged.status == "skip" and (base.drift_error or base.drift <= UNJUDGEABLE_DRIFT):
            return unjudged
        job, jumps = base.job, {}
        atol = PATH_RTOL_FACTOR * job.atol
        n = job.n_points
        for label, m in (
            ("n_points-1", n - 1),
            ("n_points+1", n + 1),
            ("2x the points", 2 * n - 1),
        ):
            if m < 2:
                continue
            x = _species(run(Job(**{**job.__dict__, "n_points": m}))[0], base)
            g = gap(x[::2], base.x, atol) if m == 2 * n - 1 else gap(x[-1:], base.x[-1:], atol)
            if g > base.limit:
                jumps[label] = g
        for f in JITTER_RTOL:
            jittered = Job(**{**job.__dict__, "rtol": job.rtol * f, "atol": job.atol * f})
            x = _species(run(jittered)[0], base)
            g = gap(x, base.x, atol)
            if g > max(base.limit, f * base.limit):
                jumps[f"rtol x{f:g}"] = g
        nudged = f"a start nudged by {START_NUDGE:g} of a step's allowance"
        try:
            x = _species(run(job, nudge=START_NUDGE)[0], base)
        except Exception as e:
            if _timed_out(e) or not isinstance(e, bngsim.SimulationError):
                raise
            jumps[f"{nudged}, from which the run fails ({_first_line(e)[:160]})"] = float("inf")
        else:
            # Only a departure the nudge grew into counts. One already past the
            # bound at the first row after the start is the model's own laws
            # telling the two starts apart, `if(Bobs==0, kb, 0)` on a B its law
            # holds at exactly 0 (#1228), which no rounding breaks.
            g = gap(x, base.x, atol)
            if g > base.limit and gap(x[1:2], base.x[1:2], atol) <= base.limit:
                jumps[nudged] = g
        if jumps:
            shown = ", ".join(f"{k} {g:.1e}" if np.isfinite(g) else k for k, g in jumps.items())
            return Verdict(
                "diff",
                max(jumps.values()),
                f"contested: the answer jumps past its bound {base.limit:.1e} under {shown}",
                {"contested": True, "jumps": jumps},
            )
        return Verdict("pass", 0.0, "the answer holds under a jittered grid, tolerance and start")

    return _guard(go)


def comma_locale() -> str | None:
    """A comma-decimal LC_NUMERIC this machine has, or None. Asked of a child
    process, so this one's locale is never set."""
    probe = (
        "import locale, sys\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        locale.setlocale(locale.LC_NUMERIC, name)\n"
        "    except locale.Error:\n"
        "        continue\n"
        "    if locale.localeconv()['decimal_point'] == ',':\n"
        "        print(name); break\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe, *COMMA_LOCALES], capture_output=True, text=True, timeout=60
    )
    return out.stdout.strip() or None


_LOCALE_PROBE = """
import json, locale, logging, sys, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
locale.setlocale(locale.LC_NUMERIC, sys.argv[1])
sys.path.insert(0, sys.argv[3])
import numpy as np
from _differential import Job, run
r, _ = run(Job.from_dict(json.loads(sys.argv[2])))
print(json.dumps(np.asarray(r.species, dtype=float).tolist()))
"""


def check_locale(base: Base, name: str | None, timeout: float = 600.0) -> Verdict:
    """The run in a child process under the comma-decimal locale ``name``."""
    if name is None:
        return Verdict("skip", note="no comma-decimal locale on this machine")

    def go():
        here = str(Path(__file__).resolve().parent)
        # The model's cap, where less of it is left than this run's own limit.
        limit = timeout if _DEADLINE is None else min(timeout, _left())
        try:
            out = subprocess.run(
                [sys.executable, "-c", _LOCALE_PROBE, name, json.dumps(base.job.as_dict()), here],
                capture_output=True,
                text=True,
                timeout=limit,
            )
        except subprocess.TimeoutExpired as e:
            if limit < timeout:  # the model's cap: TIMEOUT, this row only
                raise
            # Its own limit, where the run under C finished: a crash, as it read
            # before the cap. A run the locale stalls is this check's finding.
            return Verdict("exception", None, f"{type(e).__name__}: {_first_line(e)}")
        if out.returncode != 0:
            last = (out.stderr.strip().splitlines() or ["?"])[-1][:300]
            return Verdict("diff", None, f"runs under C, fails under {name}: {last}")
        return base.compare(json.loads(out.stdout.strip().splitlines()[-1]), f"{name}: ")

    return _guard(go)


# The SSA arm (issue #1310): checks that apply to an SSA job, and only to one.
SSA_CHECKS = ("ssa_cc_vs_interp", "ssa_backend_as_asked")


# The NF arm (issue #1312): checks that apply to an NF job, and only to one.
NF_CHECKS = ("nf_reject_vs_exact",)


def checks_for(job: Job, checks=CHECKS) -> tuple[str, ...]:
    """The checks of ``checks`` that apply to ``job``'s method."""
    if job.method == "ssa":
        return tuple(c for c in checks if c in SSA_CHECKS)
    if job.method == "nf":
        return tuple(c for c in checks if c in NF_CHECKS)
    return tuple(c for c in checks if c not in SSA_CHECKS + NF_CHECKS)


def _ssa_runs(job: Job, compiled: bool, n: int | None = None) -> tuple[list[np.ndarray], set[str]]:
    """``n`` (``job.n_rep``) SSA replicates at seeds ``job.seed`` on: the species
    trajectories and the propensity backends they report. ``compiled`` is the
    default Simulator (``codegen=True`` is refused for SSA: the default builds
    cc, or mir, where a compiler is), the other ``codegen=False``."""
    import bngsim

    out, backends = [], set()
    for i in range(job.n_rep if n is None else n):
        kw = {} if compiled else {"codegen": False}
        sim = bngsim.Simulator(load(job.path), method="ssa", **kw)
        r = sim.run(t_span=(job.t_start, job.t_end), n_points=job.n_points, seed=job.seed + i)
        out.append(np.asarray(r.species, dtype=float))
        backends.add(r.ssa_diagnostics.get("propensity_backend", "unknown"))
    return out, backends


def _all_elementary(job: Job) -> bool:
    """Every reaction mass action: the compiled and interpreted propensities
    then compute the same doubles, and a seed's draws must agree one for one."""
    reactions = load(job.path)._core.codegen_data()["reactions"]
    return all(r["type"] == "elementary" for r in reactions)


# Compiled propensities serve exact SSA on mass-action models without events,
# up to a size (GH #190); bngsim runs every other model interpreted whatever is
# asked, and a machine without a compiler runs them all so. In the rr_ssa
# corpus 23 models are compiled, 8 mass-action models are not, and 235 with
# functional laws are not.
_NOT_COMPILED = (
    "the default ran interpreted as well: compiled propensities serve exact SSA on "
    "mass-action models without events, up to a size, on a machine with a compiler"
)


def check_ssa(job: Job, checks=SSA_CHECKS) -> dict[str, Verdict]:
    """Compiled propensities against interpreted ones, seed for seed (#1310).

    Every seed's trajectory equal is a PASS. Where they are not, a model of
    mass-action laws alone is a DIFF (both compute the same doubles there, so
    the event sequence must match draw for draw); any other, where rounding in
    a functional law can change a selection, is judged in distribution, by the
    ensemble verdict the SSA parity suites use (``differ.ensemble_verdict`` at
    ``ensemble_k_for(n_rep)``). ``ssa_backend_as_asked`` asks that
    ``codegen=False`` ran the interpreter and the default compiled code, as
    ``backend_as_asked`` does for ODE."""
    # _core, the SSA suites' statistics, sits beside this directory, which a
    # caller that put only this one on the path does not reach.
    parity = str(Path(__file__).resolve().parents[1])
    if parity not in sys.path:
        sys.path.insert(0, parity)
    from _core import differ, oracles

    out: dict[str, Verdict] = {}
    if job.n_rep < 1:
        # No seed to compare: no ensemble is a PASS of nothing, and no backend
        # is a "codegen=False ran []" that is no finding either.
        v = Verdict("skip", note=f"no replicates asked (n_rep {job.n_rep})")
        return {c: v for c in checks}
    try:
        # The Simulator builds a propensity library only for a model it finds
        # eligible, before any run: without one there is nothing compiled to tell
        # apart, and no replicate is run (the corpus's slowest take one past the
        # nightly's cap). #1213's class, codegen=False ignored, needs a default
        # that compiles, so ssa_backend_as_asked has nothing to ask either.
        import bngsim

        if not getattr(bngsim.Simulator(load(job.path), method="ssa"), "_ssa_propensity_lib", ""):
            v = Verdict("skip", note=_NOT_COMPILED)
            return {c: v for c in checks}
        # One replicate says whether the default compiles: most of the corpus
        # does not, and there one each way is all either check reads, not an
        # ensemble of each (the nightly's job has a cap).
        comp, comp_backends = _ssa_runs(job, compiled=True, n=1)
        if comp_backends == {"interpreted"}:
            interp, interp_backends = _ssa_runs(job, compiled=False, n=1)
        else:
            interp, interp_backends = _ssa_runs(job, compiled=False)
            comp, comp_backends = _ssa_runs(job, compiled=True)
    except Exception as e:  # noqa: BLE001 - no base, as for ODE
        v = Verdict("no_base", None, f"the SSA runs: {type(e).__name__}: {_first_line(e)}")
        return {c: v for c in checks}
    no_compiler = comp_backends == {"interpreted"}
    if "ssa_backend_as_asked" in checks:
        asked = {"codegen=False": sorted(interp_backends), "default": sorted(comp_backends)}
        if interp_backends != {"interpreted"}:
            out["ssa_backend_as_asked"] = Verdict(
                "diff", None, f"codegen=False ran {sorted(interp_backends)}", asked
            )
        elif no_compiler:
            out["ssa_backend_as_asked"] = Verdict("skip", note=_NOT_COMPILED, extra=asked)
        else:
            out["ssa_backend_as_asked"] = Verdict(
                "pass",
                note=f"codegen=False interpreted; default {sorted(comp_backends)}",
                extra=asked,
            )
    if "ssa_cc_vs_interp" in checks:
        out["ssa_cc_vs_interp"] = _guard(
            lambda: _ssa_verdict(job, interp, comp, no_compiler, differ, oracles)
        )
    return out


def _ssa_verdict(job, interp, comp, no_compiler, differ, oracles) -> Verdict:
    if no_compiler:
        return Verdict("skip", note=_NOT_COMPILED)
    differing = [
        i for i, (a, b) in enumerate(zip(interp, comp, strict=True)) if not np.array_equal(a, b)
    ]
    n = len(interp)
    if not differing:
        return Verdict("pass", 0.0, f"all {n} seeds identical, draw for draw")
    if _all_elementary(job):
        i = differing[0]
        return Verdict(
            "diff",
            float(len(differing)),
            f"mass-action laws alone, and {len(differing)} of {n} seeds differ "
            f"(first at seed {job.seed + i})",
            {"seeds_differing": [job.seed + j for j in differing]},
        )
    mean_a, sem_a = oracles.ensemble_stats(np.stack(interp))
    mean_b, sem_b = oracles.ensemble_stats(np.stack(comp))
    k = differ.ensemble_k_for(n)
    v = differ.ensemble_verdict(mean_a, sem_a, mean_b, sem_b, k=k)
    note = (
        f"{len(differing)} of {n} seeds differ draw for draw (functional laws); in "
        f"distribution {v['frac_pass']:.3f} of cells within {k:.2f} SE, max z {v['max_z']:.2f}"
    )
    extra = {"seeds_differing": len(differing), "frac_pass": v["frac_pass"], "max_z": v["max_z"]}
    return Verdict("pass" if v["passed"] else "diff", v["max_z"], note, extra)


#: The seconds an NF check may spend escalating a DIFF to more replicates, the
#: way the stochastic suites do (30, then 100 each): a rung that would take
#: longer is not walked, and the row says so.
NF_ESCALATION_BUDGET = 300.0


def _nf_runs(
    job: Job, engine: str, n: int | None = None, seed: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """``n`` (``job.n_rep``) replicates of the BNG-XML at ``job.path`` on one
    network-free engine, ``nf_reject`` (NFsim) or ``nf_exact`` (RuleMonkey), at
    seeds ``seed`` (``job.seed``) on: the observables, ``(n, n_time, n_obs)``,
    and their names. ``job.sim`` is the sessions' options
    (``block_same_complex_binding``, ``molecule_limit``), the same for both."""
    import bngsim

    cls = bngsim.NfsimSession if engine == "nf_reject" else bngsim.RuleMonkeySession
    sess = cls(str(job.path), **job.sim)
    out, names = [], []
    first = job.seed if seed is None else seed
    try:
        for i in range(job.n_rep if n is None else n):
            sess.initialize(seed=first + i)
            r = sess.simulate(job.t_start, job.t_end, job.n_points)
            out.append(np.asarray(r.observables, dtype=float))
            names = list(r.observable_names)
    finally:
        sess.destroy()
    return np.stack(out), names


class _OverBudget(RuntimeError):
    """A rung of escalation that would not fit in :data:`NF_ESCALATION_BUDGET`."""


def check_nf(job: Job, checks=NF_CHECKS) -> dict[str, Verdict]:
    """NFsim against RuleMonkey, in distribution (#1312).

    The two engines sample the same process by different algorithms, so their
    draws never match and their ensembles must: each observable's mean at each
    sample, against the ensemble verdict the SSA parity suites use
    (``differ.ensemble_verdict`` at ``ensemble_k_for(n_rep)``). A DIFF is
    escalated as bng_parity's NF track escalates one (``_run_escalation``: 30,
    then 100 replicates each, stopping where agreement stalls), within
    :data:`NF_ESCALATION_BUDGET`, so finite-sample scatter clears and a real
    divergence stays. A known difference (the context symmetry factor of #281,
    #298, #300, #301) is a DIFF the baseline holds, so a new one alerts."""
    parity = str(Path(__file__).resolve().parents[1])
    if parity not in sys.path:
        sys.path.insert(0, parity)
    import bngsim
    from _core import differ, oracles

    # The stochastic suites' escalation of a DIFF, which bng_parity's NF track
    # walks (GH #69, #185): bng_stoch_run's, so the two read a DIFF alike.
    bng = str(Path(parity) / "bng_parity")
    if bng not in sys.path:
        sys.path.insert(0, bng)
    import bng_stoch_run as stoch

    if "nf_reject_vs_exact" not in checks:
        return {}
    if job.n_rep < 1:
        # No replicates: nothing to compare, and no NFsim run to be the base.
        return {
            "nf_reject_vs_exact": Verdict("skip", note=f"no replicates asked (n_rep {job.n_rep})")
        }
    start = time.perf_counter()
    try:
        reject, names_r = _nf_runs(job, "nf_reject")
    except Exception as e:  # noqa: BLE001 - no base, as for ODE
        v = Verdict("no_base", None, f"the NFsim runs: {type(e).__name__}: {_first_line(e)}")
        return {"nf_reject_vs_exact": v}

    def go():
        try:
            exact, names_e = _nf_runs(job, "nf_exact")
            # one replicate of each engine, for what a rung of escalation costs
            per_rep = (time.perf_counter() - start) / job.n_rep
        except bngsim.ModelError as e:
            # RuleMonkey refuses by name what it does not model (compartments).
            raise _Declined(f"{type(e).__name__}: {_first_line(e)}") from e
        except RuntimeError as e:
            if "RuleMonkey support is not present" in str(e):
                return Verdict("skip", note=_first_line(e))
            if "RuleMonkey cannot honor this BNG XML" in str(e):
                # a refusal by name: a rule whose rate the two engines define
                # apart (TotalRate over a repeated component, ft_total_rate)
                raise _Declined(f"{type(e).__name__}: {_first_line(e)}") from e
            raise
        if not names_r:
            return Verdict("skip", note="the model has no observables to compare")
        if names_e != names_r:
            return Verdict(
                "diff", None, f"the observables differ: NFsim {names_r}, RuleMonkey {names_e}"
            )

        def judge(a, b):
            mean_a, sem_a = oracles.ensemble_stats(a[1])
            mean_b, sem_b = oracles.ensemble_stats(b[1])
            n = a[1].shape[0]
            k = differ.ensemble_k_for(n)
            v = differ.ensemble_verdict(mean_a, sem_a, mean_b, sem_b, k=k)
            # The observable whose means are furthest apart, on its own peak.
            peak = np.maximum(np.nanmax(np.abs(np.stack([mean_a, mean_b])), axis=(0, 1)), 1.0)
            worst = names_r[int(np.argmax(np.nanmax(np.abs(mean_a - mean_b), axis=0) / peak))]
            judged[:] = [v, worst]
            note = (
                f"{n} replicates each: {v['frac_pass']:.3f} of cells within {k:.2f} SE, "
                f"max z {v['max_z']:.2f}; furthest apart: {worst}"
            )
            status = "pass" if v["passed"] else "diff"
            return status, v["frac_pass"], note, "frac_pass", differ.ENSEMBLE_PASS_FRAC

        def escalate(delta, seed, rung):
            cost = per_rep * delta
            if cost > NF_ESCALATION_BUDGET - (time.perf_counter() - t0):
                raise _OverBudget(f"~{cost:.0f}s, past the {NF_ESCALATION_BUDGET:g}s budget")
            more = [_nf_runs(job, e, delta, seed)[0] for e in ("nf_reject", "nf_exact")]
            return [(None, x, names_r) for x in more]

        judged: list = []
        pair = ((None, reject, names_r), (None, exact, names_e))
        status, frac, note, *_ = judge(*pair)
        t0 = time.perf_counter()
        status, frac, note, _, _, rungs, stalled = stoch._run_escalation(
            *pair,
            status,
            frac,
            note,
            "frac_pass",
            differ.ENSEMBLE_PASS_FRAC,
            n_rep_base=job.n_rep,
            seed_base=job.seed,
            compare=judge,
            escalate=escalate,
        )
        v, worst = judged
        extra = {
            "frac_pass": v["frac_pass"],
            "max_z": v["max_z"],
            "worst_observable": worst,
            "rungs": [list(r) for r in rungs],
            "stalled": stalled,
        }
        return Verdict(status, v["max_z"], note, extra)

    return {"nf_reject_vs_exact": _guard(go)}


def check_all(
    job: Job,
    locale_name: str | None,
    checks=CHECKS,
    deadline: float | None = None,
    on_verdict=None,
) -> dict[str, Verdict]:
    """Every check of ``checks`` on one model. A base that does not run is no
    reference: each check is then an exception on the base. Every run has
    until ``deadline`` (a ``time.monotonic()``): a check that has not finished
    by then is TIMEOUT, its own row only, where the runner's cap would take
    every row of the model (MODEL1006230049, 0053 and 0077 take 716-751 s of
    the nightly's 900 s, so any check added to every model tipped them over).
    ``on_verdict(check, verdict)`` hears each as it is reached, so a caller
    killed at the cap still has those."""
    global _DEADLINE
    _DEADLINE = deadline
    try:
        # Only the checks the job's method has: an SSA job the SSA arm's, an NF
        # job the NF arm's.
        checks = checks_for(job, checks)
        return _check_all(job, locale_name, checks, on_verdict or (lambda c, v: None))
    finally:
        _DEADLINE = None


class _Heard(dict):
    """The verdicts so far, each told to ``on_verdict`` as it is set."""

    def __init__(self, on_verdict):
        super().__init__()
        self.on_verdict = on_verdict

    def __setitem__(self, check, verdict):
        super().__setitem__(check, verdict)
        self.on_verdict(check, verdict)

    def update(self, other):
        for check, verdict in other.items():
            self[check] = verdict


def _check_all(job: Job, locale_name: str | None, checks, on_verdict) -> dict[str, Verdict]:
    if job.method in ("ssa", "nf"):
        heard = _Heard(on_verdict)
        heard.update((check_ssa if job.method == "ssa" else check_nf)(job, checks))
        return dict(heard)
    try:
        base = Base(job)
    except Exception as e:  # noqa: BLE001
        # No base, no reference: not a finding of this suite's (the parity suites
        # own a model that does not load or run), and not a pass either.
        v = Verdict("no_base", None, f"the base run: {type(e).__name__}: {str(e)[:300]}")
        return {c: _verdict_of(e) if _timed_out(e) else v for c in checks}
    out: dict[str, Verdict] = _Heard(on_verdict)

    def spent() -> Verdict | None:
        # No check starts past the deadline: one that builds and compiles
        # Simulators without running them (backend_as_asked) has no run to bound.
        try:
            _left()
        except Exception as e:  # noqa: BLE001 - SimulationTimeout, the cap spent
            return _verdict_of(e)
        return None

    if {"sens_runs", "sens_vs_plain"} & set(checks):
        v = spent()
        out.update({"sens_runs": v, "sens_vs_plain": v} if v else check_sens(base))
    one = {
        "ajac_vs_fdjac": check_ajac,
        "klu_vs_dense": check_klu,
        "cc_vs_interp": check_cc_vs_interp,
        "cc_vs_mir": check_cc_vs_mir,
        "grid": check_grid,
        "sparsity_covers_jacobian": check_sparsity,
        "backend_as_asked": check_backend,
    }
    for name, fn in one.items():
        if name in checks:
            out[name] = spent() or fn(base)
    if "locale" in checks:
        out["locale"] = spent() or check_locale(base, locale_name)
    # After every check main runs (#1308), in what is left of the model's cap.
    if "jacobian_values" in checks:
        out["jacobian_values"] = spent() or check_jacobian_values(base)
    # After every check main runs (#1305), in what is left of the model's cap:
    # four more runs ahead of locale could tip a slow model's locale row, which
    # the baseline holds, to TIMEOUT.
    for kind in TRANSFORMS:
        if kind in checks:
            out[kind] = spent() or check_transform(base, kind)
    # After every check main runs (#1307), in what is left of the model's cap.
    for name, fn in (("restart", check_restart), ("reuse", check_reuse), ("batch", check_batch)):
        if name in checks:
            out[name] = spent() or fn(base)
    # After every check main runs (#1311), in what is left of the model's cap:
    # steady_state() takes no timeout, and a kill still leaves the other rows.
    for name, method in STEADY_METHODS.items():
        if name in checks:
            out[name] = spent() or check_steady(base, method)
    # Every DIFF with two trajectories says which side the third arms take
    # (#1309), the arms run once per model however many DIFFs it has, each in
    # what is left of the model's cap; one that does not finish is left out.
    diffs = [v for v in out.values() if v.status == "diff" and v.path_x is not None]
    if diffs:
        try:
            arms = _arm_runs(base)
            for v in diffs:
                v.extra["attribution"] = attribute(base, v, arms)
        except Exception as e:  # noqa: BLE001 - attribution is advice, never a verdict
            for v in diffs:
                v.extra["attribution"] = {"verdict": "unattributed", "error": str(e)[:300]}
    # Last: five reruns of the base, the costliest of what #1309 adds, in what
    # is left once the DIFFs are attributed (a stable row carries no trajectory).
    if "stable" in checks:
        out["stable"] = spent() or check_stable(base)
    return {c: out[c] for c in checks if c in out}
