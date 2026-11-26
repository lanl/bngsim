""":meth:`bngsim.Result.check`'s ``"sensitivity_second_engine"`` check: a run's
forward sensitivities against AMICI's, from the model file (issue #1923, the
third tier of #1893).

``"sensitivity"`` and ``"sensitivity_fd"`` (:mod:`bngsim._check_sensitivity`)
both read the model as bngsim loaded it, so a defect the loader and the
sensitivity code share passes both: the #144 initial assignment ``u(0) = b·v0``
whose ``b`` column agreed with its own difference quotient to 6e-8 while it was
5x off. AMICI reads the SBML itself, derives the sensitivity equations
symbolically and compiles them, so it shares nothing with bngsim but the file.

**Where it runs.** Under ``check("all")``, on a run that computed forward
sensitivities, and only where ``import amici`` works: AMICI is no dependency or
extra of bngsim (it compiles C++ per model, and needs SWIG, a C++ toolchain, a
BLAS, and Python 3.12 or later from 1.1 on). Elsewhere the check is a ``skip``
that says why. The run must be one AMICI can make again from the file alone,
as for ``"second_engine"`` (:func:`bngsim._check.same_as_file`): an SBML or
Antimony model, unchanged since its load and read from a file whose bytes are
still the ones bngsim read, run from its initial state at the file's own
parameters, over a plain ``t_span``. ``budget=`` bounds bngsim's reruns, not
AMICI's build (about 15 s cold for a small model, minutes for a large one) or
its runs; a build is cached under ``~/.cache/bngsim/amici`` (or
``$BNGSIM_AMICI_CACHE_DIR``), keyed by the SBML text and the AMICI install,
about 4 MB a model, and nothing prunes it (``bngsim-cache`` manages the codegen
cache only): remove the directory to clear it. A model whose generated C++
does not compile is kept there as a ``.failed`` file, so it is not built again
at every check; remove that file to retry.

**What is carried over from ``parity_checks/amici_parity``**, which the wheel
cannot import, and only what the check needs:

- *Parameters.* bngsim names a reaction's local parameter
  ``_lp_<reaction>_<id>``; AMICI names it ``<reaction>_<id>``, unless the file
  has a global parameter of that id, which keeps the name (neither column is
  then judged). AMICI's free parameter set is authoritative: a parameter an
  initial assignment or a rule sets is an expression to AMICI, with no column,
  and its column is not judged (bngsim's is then a partial derivative; a
  parameter that reads it gets the whole chain in both engines). Scaling is
  pinned to none, so AMICI's ``dx/dp`` is compared, not its ``dx/d ln p``.
- *Ids.* AMICI renames an id that collides with a symbol of its own (``k``,
  ``p``, ``x``) to ``amici_<id>``. The prefix is undone only where the file has
  no element of that id itself, and an id that two of AMICI's would map to is
  matched to neither (the parity suite's no-guessing rule).
- *States.* AMICI keeps a ``hasOnlySubstanceUnits`` species as an amount, in
  its states and its sensitivities alike, where bngsim reports a
  concentration. Such a species is divided by its compartment's size where the
  file gives that size as a constant (declared, set by no rule and no initial
  assignment), and not compared otherwise; AMICI's absolute tolerances are
  scaled by that size too, or an amount in a small compartment would be held
  as many times looser than bngsim holds its concentration (the parity
  suite's MODEL1509020000, whose amounts sit far under its atol, is that
  case). A species AMICI holds no state for (a constant species, one a rule
  sets) is not compared either; each is named.
- *Initial amounts.* AMICI gives sensitivities to parameters only, so a
  ``sensitivity_ic=`` column is not judged here.
- *The dispositions* in ``AMICI_KNOWN_ISSUES.md`` that touch sensitivities,
  but for the amounts above, are about events or switches: AMICI computes no
  sensitivity jump at a switch time a parameter sets (Class 4's
  MODEL2105110001), misses an equality trigger between its output times and
  fires a strict time trigger at its own threshold (Classes 5 and 7, which the
  parity suite answers by running AMICI through each event's instant and
  leaving the sample on a strict onset unjudged), and reads a trigger or a
  piecewise condition that sits exactly at its threshold at the start
  differently from bngsim (the 2026-11 triage). A model with an event, or
  whose math switches anywhere (a piecewise, a comparison, a logical operator,
  ``floor``, ``ceiling``, ``min``, ``max``, a remainder, a quotient or a delay), is
  therefore not compared: the check skips it and says so, rather than read
  AMICI's known misreadings of it as bngsim's. So is a model whose function
  definition names an argument twice, which defines no one model
  (MODEL2003040001: AMICI binds the first, bngsim the last). The event and
  switch cases are left to #1960, which would carry over the parity suite's
  event instants and onset rows.

**The rule** is the ``"sensitivity"`` tier's, series by series
(:class:`bngsim._check_sensitivity._Columns`): one species and one parameter,
over time, on the larger of its peak in bngsim and in AMICI, past the absolute
tolerance the solver held it to, against a path's allowance
(:func:`bngsim._check.path_margins`, column by column) with the series' own
move at the tenfold tighter tolerance (or half its move at twice the
tolerance) as its drift. A series that neither engine resolves (within
:data:`bngsim._check.TRACE_ATOLS` of its tolerance of zero in both) is held
to its column's whole scale too, so the two engines' noise there is no
finding. One the run gives as 0 where AMICI does not is resolved, on AMICI's
peak. A series converges where its move at the tighter tolerance is at most
:data:`bngsim._check.TOLERANCE_MOVE` of that peak, read on its own scale (the
tolerance tier reads a series it does not resolve, or takes to be held by a
floor, on its column's); one that does not is judged with no drift of its own,
which would forgive its own error, as the quotient check judges it.

AMICI runs twice, at the run's tolerance and at a tenth of it (its build is
cached, so the second costs a run). The tighter run is the reference, and the
move between the two is every series' and species' drift where it is the
larger: it is how far a correct engine strays at the run's tolerance, which
bngsim's own move can understate (BIOMD0000000810 at rtol 1e-7: bngsim 1.2e-4
and AMICI 7e-5 of a series' peak from the solution, 2e-4 apart, and in
agreement once converged). It is not capped: on that model's gamma_C column
both engines err by percent at the run's tolerance, which a cap would read as a
misreading. So the series AMICI moves past :data:`bngsim._check.TOLERANCE_MOVE`
of their peak on, whose allowance is that wide, are named
(``amici_not_converged``). A series AMICI moves past
:data:`bngsim._check.PATH_CEILING` of its peak on has no reference at the run's
tolerance (ten times that would forgive a tenth of the series): it is not
judged, and is named (``no_reference``); where no series is left to judge, the
check is a ``skip``.

A diff at the run's tolerance is judged again a decade tighter: bngsim's run
at a tenth of the tolerance (the tolerance check's) against AMICI at a
hundredth, with bngsim's run at a hundredth as its drift. Either engine's one
move can understate its error (on BIOMD0000000810 to t = 50 at rtol 1e-8,
AMICI's two runs agree with each other and are 1e-3 of some series' peak from
the solution, which both engines reach at rtol 1e-10), and a model read two
ways differs as much at any tolerance. So a diff that holds there is the
reading's (``confirmed_tighter``), judged with every series keeping its own
move a decade tighter as drift, converged or not, so that tolerance error a
series still has there is not read as a misreading; one that does not hold is
the tolerance's:
the verdict is still ``diff``, with ``diff_not_converged``, which ``next``
reads as the tolerance's. Where the tighter runs cannot be made, it says so.
The states AMICI integrates are compared
too, species by species, on every path's rule
(:func:`bngsim._check.path_margins`, as
:meth:`bngsim._check._Context.compare_x` uses it): sensitivities about two
different trajectories say nothing, and a disagreement there is reported as one
about the trajectory (``state_diff``). Each AMICI run takes its ``rtol`` (the
run's, or a tenth of it) and, scaled alike, the smallest of the compared
species' atols for the states, and for the sensitivities the smallest of the
judged series' absolute tolerances, each on AMICI's scale (an amount's times
its compartment's size), and never looser than the run's smallest (AMICI's are
scalars, over the states it does not compare too), so it is held at least as
tight as bngsim on every species and series. A run with no species converged at
its own tolerance is ``no_base`` before AMICI builds anything.

The report names AMICI's version and, from the install's PEP 610 record, its
commit, and says when that is not the commit the parity suite pins
(:data:`PINNED_COMMIT`, ``parity_checks/amici_parity/AMICI_PIN.json``; PyPI's
1.2.0 has the event regression that made the pin a commit on AMICI's main
branch, which PyPI metadata cannot carry). No model this check compares has an
event, but the version is still part of what was compared against.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import sysconfig
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim import _check, _check_sensitivity
from bngsim._check import (
    MAX_NAMED,
    PATH_CEILING,
    TOLERANCE_MOVE,
    _first_line,
    _num,
    _Out,
    _verdict,
    differences,
    moves,
    path_margins,
    spread,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from bngsim._check import _Context
    from bngsim._check_sensitivity import _Columns

NAME = "sensitivity_second_engine"

#: The AMICI commit ``parity_checks/amici_parity/AMICI_PIN.json`` pins (a test
#: holds the two equal). A report against any other install says so.
PINNED_COMMIT = "7fb8c571ac9e4d5aafb85d5da08982ea7c5a19dc"

#: AMICI's step limit for a run: a budget over the whole run there (CVODES'
#: ``mxstep`` as AMICI counts it), where bngsim's ``max_steps`` is a batch it
#: resumes from. Ten times the parity suite's sensitivity budget
#: (``_amici_sens.DEFAULT_SENS_MAX_STEPS``), as the reference run is at a tenth
#: of the run's tolerance: a harmonic oscillator over 500 periods at rtol 1e-9
#: runs out of 100,000 at t = 983 of 1000, and takes 0.2 s with this many. A
#: model AMICI cannot integrate pays the whole budget before it gives up.
MAX_STEPS = 1_000_000

#: bngsim's prefix for a reaction's local parameter.
LOCAL_PREFIX = "_lp_"
#: AMICI's prefix for an id that collides with one of its own symbols.
AMICI_PREFIX = "amici_"

#: What a cached build is keyed by beside the SBML text and the AMICI install:
#: sensitivity code on, conservation laws off (every species a state), no
#: observation model.
_BUILD_FLAGS = "sens=1;cl=0;obs=0;bngsim-check=1"


def _default_cache_dir() -> Path:
    """Where AMICI's builds are kept: ``$BNGSIM_AMICI_CACHE_DIR``, read as the
    codegen cache's is, or ``~/.cache/bngsim/amici``."""
    from bngsim._codegen import _expand_cache_dir, _home_dir

    env = os.environ.get("BNGSIM_AMICI_CACHE_DIR", "").strip()
    if not env:
        return _home_dir() / ".cache" / "bngsim" / "amici"
    return _expand_cache_dir(env)


CACHE_DIR = _default_cache_dir()


# --- reading the file ------------------------------------------------------


def _sbml_text(path: str, fmt: str, sha256: str | None) -> tuple[str | None, str]:
    """The model file as SBML text, and why not where it cannot be had (the
    file is gone, its bytes are not the ones bngsim read, Antimony cannot
    convert it). The bytes are read once, and checked against the sha256 the
    record holds, so AMICI reads what bngsim read."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        return None, f"the model file could not be read again: {_first_line(e)}"
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        return None, _check.FILE_CHANGED
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return None, f"the model file is not UTF-8 text: {_first_line(e)}"
    if fmt == "sbml":
        return text, ""
    try:
        import antimony
    except ImportError:
        return None, "Antimony is not installed, so the model cannot be read again as SBML"
    if antimony.loadAntimonyString(text) < 0:
        return None, f"Antimony could not read the model again: {antimony.getLastError()}"
    return str(antimony.getSBMLString(antimony.getMainModuleName())), ""


def _switch_types() -> dict[int, str]:
    """libSBML's math node types that switch, by what to call each."""
    import libsbml

    named = {
        "AST_FUNCTION_PIECEWISE": "a piecewise",
        "AST_RELATIONAL_EQ": "a comparison",
        "AST_RELATIONAL_NEQ": "a comparison",
        "AST_RELATIONAL_GT": "a comparison",
        "AST_RELATIONAL_GEQ": "a comparison",
        "AST_RELATIONAL_LT": "a comparison",
        "AST_RELATIONAL_LEQ": "a comparison",
        "AST_LOGICAL_AND": "a logical operator",
        "AST_LOGICAL_OR": "a logical operator",
        "AST_LOGICAL_NOT": "a logical operator",
        "AST_LOGICAL_XOR": "a logical operator",
        "AST_LOGICAL_IMPLIES": "a logical operator",
        "AST_FUNCTION_FLOOR": "floor",
        "AST_FUNCTION_CEILING": "ceiling",
        "AST_FUNCTION_MIN": "min",
        "AST_FUNCTION_MAX": "max",
        "AST_FUNCTION_REM": "a remainder",
        "AST_FUNCTION_QUOTIENT": "a quotient",
        "AST_FUNCTION_DELAY": "a delay",
    }
    out: dict[int, str] = {}
    for attr, what in named.items():
        value = getattr(libsbml, attr, None)
        if value is not None:
            out[int(value)] = what
    return out


def _switch_in(node: Any, types: dict[int, str]) -> str | None:
    """What in the math *node* switches, or None."""
    if node is None:
        return None
    stack = [node]
    while stack:
        n = stack.pop()
        what = types.get(int(n.getType()))
        if what is not None:
            return what
        stack.extend(n.getChild(i) for i in range(n.getNumChildren()))
    return None


def _maths(model: Any) -> Iterator[tuple[str, Any]]:
    """Each piece of math that sets the model's dynamics or its start, with
    where it is."""
    for fd in model.getListOfFunctionDefinitions():
        yield f"function {fd.getId()}", fd.getBody()
    for ia in model.getListOfInitialAssignments():
        yield f"the initial assignment to {ia.getSymbol()}", ia.getMath()
    for rule in model.getListOfRules():
        target = rule.getVariable() if rule.isSetVariable() else ""
        yield f"the rule for {target}" if target else "an algebraic rule", rule.getMath()
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            yield f"reaction {r.getId()}'s rate law", r.getKineticLaw().getMath()
        for ref in list(r.getListOfReactants()) + list(r.getListOfProducts()):
            if hasattr(ref, "isSetStoichiometryMath") and ref.isSetStoichiometryMath():
                yield f"reaction {r.getId()}'s stoichiometry", ref.getStoichiometryMath().getMath()


def not_comparable(model: Any) -> str | None:
    """Why AMICI's answer for the libSBML *model* is no judge of bngsim's, or
    None: an event, a switch anywhere in the math, or a function definition
    that names an argument twice (see the module docstring)."""
    if model is None:
        return "libSBML read no model from the file"
    n = model.getNumEvents()
    if n:
        return (
            f"the model has {n} event(s), whose instants AMICI reads differently from bngsim on "
            "some kinds of trigger: a model with events is not compared"
        )
    for fd in model.getListOfFunctionDefinitions():
        args = [fd.getArgument(i).getName() for i in range(fd.getNumArguments())]
        if len(set(args)) != len(args):
            return (
                f"function {fd.getId()} names an argument twice, so the file defines no one "
                "model (AMICI binds the first, bngsim the last)"
            )
    types = _switch_types()
    for where, math_node in _maths(model):
        what = _switch_in(math_node, types)
        if what is not None:
            return (
                f"{where} has {what}: AMICI computes no sensitivity jump at a switch a parameter "
                "moves, and reads one at its threshold at the start differently from bngsim, so "
                "a model whose math switches is not compared"
            )
    return None


def _element_ids(doc: Any) -> set[str]:
    """Every id in the document, and each local parameter as
    ``<reaction>_<id>``, the name AMICI gives it."""
    ids = {e.getId() for e in doc.getListOfAllElements() if e.isSetId()}
    model = doc.getModel()
    if model is not None:
        for r in model.getListOfReactions():
            if not r.isSetKineticLaw():
                continue
            law = r.getKineticLaw()
            local = list(law.getListOfLocalParameters()) + list(law.getListOfParameters())
            ids.update(f"{r.getId()}_{p.getId()}" for p in local)
    return ids


def parameter_ids(model: Any) -> tuple[set[str], set[str]]:
    """The file's global parameter ids, and its local parameters as
    ``<reaction>_<id>``: where a name is in both, AMICI gives the global the
    name and the local another, so neither is matched by name."""
    globals_ = {p.getId() for p in model.getListOfParameters()}
    locals_: set[str] = set()
    for r in model.getListOfReactions():
        if r.isSetKineticLaw():
            law = r.getKineticLaw()
            local = list(law.getListOfLocalParameters()) + list(law.getListOfParameters())
            locals_.update(f"{r.getId()}_{q.getId()}" for q in local)
    return globals_, locals_


def amici_ids(raw: list[str], file_ids: set[str]) -> list[str | None]:
    """AMICI's ids as the file's: its ``amici_`` prefix undone where the file
    has no element of the prefixed id itself, and None for an id that another
    of AMICI's maps to as well (never guessed)."""
    out = [
        r[len(AMICI_PREFIX) :] if r.startswith(AMICI_PREFIX) and r not in file_ids else r
        for r in raw
    ]
    counts: dict[str, int] = {}
    for s in out:
        counts[s] = counts.get(s, 0) + 1
    return [s if counts[s] == 1 else None for s in out]


def bngsim_ids(names: list[str]) -> list[str | None]:
    """bngsim's parameter names as the file's: a local parameter's ``_lp_``
    prefix stripped, None for one another name maps to as well."""
    out = [n[len(LOCAL_PREFIX) :] if n.startswith(LOCAL_PREFIX) else n for n in names]
    counts: dict[str, int] = {}
    for s in out:
        counts[s] = counts.get(s, 0) + 1
    return [s if counts[s] == 1 else None for s in out]


def compartment_sizes(model: Any) -> dict[str, float | None]:
    """Each compartment's size where the file gives it as a constant: declared,
    and set by no rule and no initial assignment; None otherwise."""
    ruled = {r.getVariable() for r in model.getListOfRules() if r.isSetVariable()}
    algebraic = any(r.isAlgebraic() for r in model.getListOfRules())
    out: dict[str, float | None] = {}
    for c in model.getListOfCompartments():
        cid = c.getId()
        fixed = (
            cid not in ruled
            and model.getInitialAssignmentBySymbol(cid) is None
            and c.isSetSize()
            and (c.getConstant() or not algebraic)
        )
        out[cid] = float(c.getSize()) if fixed else None
    return out


def amount_species(model: Any) -> dict[str, str]:
    """``{species id: compartment id}`` for each species AMICI keeps as an
    amount (``hasOnlySubstanceUnits``)."""
    return {
        s.getId(): s.getCompartment()
        for s in model.getListOfSpecies()
        if s.getHasOnlySubstanceUnits()
    }


# --- AMICI -------------------------------------------------------------------


def _import_amici() -> Any:
    """AMICI's package, with the module its solver settings live in imported;
    raises where AMICI cannot be imported."""
    import amici
    import amici.sim.sundials  # noqa: F401 - the enums and the solver

    return amici


def amici_version(amici: Any) -> tuple[str, str | None]:
    """AMICI's version and, where the install records one (PEP 610's
    ``direct_url.json``, as an install from a git commit has), its commit."""
    version = str(getattr(amici, "__version__", "unknown"))
    commit = None
    try:
        import importlib.metadata as md

        raw = md.distribution("amici").read_text("direct_url.json")
        if raw:
            info = json.loads(raw)
            commit = (info.get("vcs_info") or {}).get("commit_id") or None
    except Exception:  # noqa: BLE001 - no record is no commit
        commit = None
    return version, commit


@contextmanager
def _tools_on_path() -> Iterator[None]:
    """This environment's scripts directory first on ``PATH`` while AMICI
    builds: it finds SWIG and CMake there by a ``PATH`` search, and an
    environment's own copies are there whether or not it was activated."""
    bindir = sysconfig.get_path("scripts")
    old = os.environ.get("PATH")
    parts = [p for p in (old or "").split(os.pathsep) if p and p != bindir]
    os.environ["PATH"] = os.pathsep.join([bindir, *parts])
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old


class Engine:
    """A built AMICI model, and what the check reads of it."""

    def __init__(
        self,
        amici: Any,
        model: Any,
        version: str,
        commit: str | None,
        build_seconds: float,
        cached: bool,
    ) -> None:
        self.amici = amici
        self.model = model
        self.version = version
        self.commit = commit
        self.build_seconds = build_seconds
        self.cached = cached
        self.free_ids = [str(i) for i in model.get_free_parameter_ids()]
        self.free_values = [float(v) for v in model.get_unscaled_parameters()]
        self.state_ids = [str(i) for i in model.get_state_ids()]

    def solver(self, *, rtol: float, atol: float, atol_sens: float, method: str) -> Any:
        """AMICI's solver for the run: forward sensitivities, by the run's
        corrector method, at the given tolerances and :data:`MAX_STEPS`."""
        import amici.sim.sundials as ss

        solver = self.model.create_solver()
        solver.set_relative_tolerance(float(rtol))
        solver.set_absolute_tolerance(float(atol))
        solver.set_relative_tolerance_fsa(float(rtol))
        solver.set_absolute_tolerance_fsa(float(atol_sens))
        solver.set_max_steps(MAX_STEPS)
        solver.set_sensitivity_order(ss.SensitivityOrder_first)
        solver.set_sensitivity_method(ss.SensitivityMethod_forward)
        solver.set_internal_sensitivity_method(
            ss.InternalSensitivityMethod_simultaneous
            if method == "simultaneous"
            else ss.InternalSensitivityMethod_staggered
        )
        return solver

    def simulate(
        self,
        ids: list[str],
        t0: float,
        times: NDArray[np.float64],
        *,
        rtol: float,
        atol: float,
        atol_sens: float,
        method: str,
    ) -> tuple[str, NDArray[np.float64], NDArray[np.float64]]:
        """``(failure, x, sx)``: the states ``(n_times, n_states)`` and their
        sensitivities to *ids* ``(n_times, n_states, len(ids))``, bngsim's
        layout, or what went wrong."""
        import amici.sim.sundials as ss

        m = self.model
        m.set_parameter_scale([ss.ParameterScaling_none] * len(self.free_ids))
        m.set_parameter_list([self.free_ids.index(p) for p in ids])
        solver = self.solver(rtol=rtol, atol=atol, atol_sens=atol_sens, method=method)
        m.set_t0(float(t0))
        m.set_timepoints([float(t) for t in times])
        errors: list[str] = []
        with _amici_log_to(errors), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rdata = m.simulate(solver=solver)
        status = int(rdata.status)
        if status != 0:
            try:
                text = str(ss.simulation_status_to_str(status))
            except Exception:  # noqa: BLE001
                text = str(status)
            why = f"status {status} ({text})"
            if errors:
                why += f"; AMICI's first error: {errors[0]}"
            return why, np.empty((0, 0)), np.empty((0, 0, 0))
        x = np.asarray(rdata.x, dtype=float)
        sx = np.transpose(np.asarray(rdata.sx, dtype=float), (0, 2, 1))
        return "", x, sx


@contextmanager
def _amici_log_to(errors: list[str]) -> Iterator[None]:
    """AMICI's log records kept from stderr while it builds or runs: AMICI's
    own console handler is set aside, and each error's first and last lines
    collected into *errors*, for the verdict's note."""
    import logging

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.ERROR:
                # The first line says what kind of error; libSBML's names the
                # element in its last.
                lines = [ln.strip() for ln in record.getMessage().splitlines() if ln.strip()]
                if not lines:
                    errors.append(record.levelname)
                    return
                first = lines[0] if len(lines[0]) <= 150 else lines[0][:150]
                if len(lines) == 1:
                    text = first if first == lines[0] else first + "..."
                else:
                    text = f"{first} ... {lines[-1][:250]}"
                errors.append(text)

    log = logging.getLogger("amici")
    held, propagate = list(log.handlers), log.propagate
    log.handlers = [_Keep()]
    log.propagate = False
    try:
        yield
    finally:
        log.handlers, log.propagate = held, propagate


#: How old a staging directory of a process that is gone must be before it is
#: removed: a cold build of a large model takes minutes, and a cache shared
#: across hosts or containers sees another's live build as a pid that is gone.
_STALE_STAGING_SECONDS = 6 * 3600


@contextmanager
def _output_to(log: Path) -> Iterator[None]:
    """File descriptors 1 and 2 into *log* while AMICI builds: its CMake and
    compiler run as subprocesses that write there, past Python's streams and
    its logging. Left as they are where they cannot be duplicated."""
    for stream in (sys.stdout, sys.stderr):
        with suppress(Exception):  # a stream that cannot flush has nothing to
            stream.flush()
    try:
        saved = (os.dup(1), os.dup(2))
    except OSError:
        yield
        return
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            with suppress(Exception):
                stream.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for f in (fd, *saved):
            os.close(f)


#: A compiler's diagnostic in a generated source file: ``w.cpp:22:11: error:``.
_COMPILER_ERROR = re.compile(r"\S+\.(?:cpp|cc|c|h|hpp):\d+(?::\d+)?:\s*(?:fatal )?error:")


def _build_errors(log: Path) -> tuple[str | None, str | None]:
    """From *log*: the first compiler error in a generated source (the
    model's own failure), and else the first error of the build tools
    (CMake's, SWIG's: the environment's), each as one line."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, None
    tools = None
    for k, line in enumerate(lines):
        if _COMPILER_ERROR.search(line):
            # A source file's path is the staging directory's, gone by now.
            return re.sub(r"^\S*/(?=[^/\s]+:\d+)", "", line.strip())[:300], None
        if tools is None and (line.startswith("CMake Error") or "error:" in line.lower()):
            # CMake puts the reason on the lines after its "CMake Error at".
            after = [ln.strip() for ln in lines[k + 1 : k + 6] if ln.strip()]
            tools = " ".join([line.strip(), *after[:1]])[:300]
    return None, tools


def _remove_dead_staging(root: Path) -> None:
    """Remove the staging directories of builds whose process is gone (one
    killed mid-build leaves its own behind) and that no build has touched for
    :data:`_STALE_STAGING_SECONDS`; any other is left alone. Not on Windows,
    where ``os.kill(pid, 0)`` is no test of a process."""
    if os.name == "nt":
        return
    now = time.time()
    for d in root.glob(".building_*"):
        pid = d.name.removesuffix(".log").rsplit("_", 1)[-1]
        if not pid.isdigit():
            continue
        try:
            if now - d.stat().st_mtime < _STALE_STAGING_SECONDS:
                continue
            os.kill(int(pid), 0)
        except ProcessLookupError:
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            else:
                d.unlink(missing_ok=True)
        except OSError:  # gone already, another user's process, no permission
            continue


def build_engine(amici: Any, sbml: str) -> Engine | str:
    """AMICI's model of *sbml*, built once and cached (:data:`CACHE_DIR`), or
    why it could not be built or loaded.

    A build is made in a directory of this process's own and renamed into
    place, so a cached directory is always a whole build, and two processes
    building the same model at once both end with one (the parity suite's
    ``_amici_sens._build_sens_model``). While it builds, file descriptors 1
    and 2 of the whole process go to a log (AMICI's CMake and compiler write
    there), so output another thread writes then is lost. A build whose
    generated C++ does not compile is the model's: it is kept as a
    ``<build>.failed`` file, which the next check reads instead of building
    again (remove it to retry). A failure of the build tools is the
    environment's, and is not kept."""
    version, commit = amici_version(amici)
    key = hashlib.sha256(
        "\0".join(
            [
                sbml,
                _BUILD_FLAGS,
                version,
                commit or "",
                f"{sys.version_info[0]}.{sys.version_info[1]}",
                platform.machine(),
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    name = f"bngsim_amici_{key}"
    root = Path(CACHE_DIR)
    where = root / name
    t0 = time.perf_counter()
    cached = where.is_dir()
    failed = root / f"{name}.failed"
    if not cached and failed.is_file():
        try:
            reason = failed.read_text(encoding="utf-8").strip()
        except OSError:
            reason = "AMICI could not build it"
        return f"{reason} (an earlier try; remove {failed} to build it again)"
    if not cached:
        staging = root / f".building_{name}_{os.getpid()}"
        log = root / f".building_{name}_{os.getpid()}.log"
        errors: list[str] = []
        try:
            root.mkdir(parents=True, exist_ok=True)
            _remove_dead_staging(root)
            shutil.rmtree(staging, ignore_errors=True)
            with (
                _tools_on_path(),
                _amici_log_to(errors),
                _output_to(log),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                importer = amici.SbmlImporter(sbml, from_file=False)
                importer.sbml2amici(
                    name,
                    str(staging),
                    verbose=False,
                    generate_sensitivity_code=True,
                    compute_conservation_laws=False,
                    observation_model=[],
                )
            try:
                staging.rename(where)
            except OSError:
                # Another process built the same key first: its build is the
                # same model, so it is the one used.
                if not where.is_dir():
                    raise
        except Exception as e:  # noqa: BLE001 - AMICI's failure is not bngsim's
            why = f"AMICI could not build it: {type(e).__name__}: {_first_line(e)[:200]}"
            compiler, tools = _build_errors(log)
            if compiler:
                why = f"AMICI could not build it: its generated C++ did not compile ({compiler})"
                # The model's own failure, which no change here mends: kept,
                # so the next check does not pay for the build again.
                with suppress(OSError):
                    failed.write_text(why + "\n", encoding="utf-8")
            elif tools:
                why = (
                    "AMICI could not build it: its build tools failed (a compiler, CMake or "
                    f"SWIG this environment lacks or cannot run: {tools})"
                )
            elif errors:
                # The messages AMICI points "above" to are its log, kept here.
                why = why.replace(" (see error messages above)", "")
                why += f" (AMICI's first error: {errors[0]})"
            return why
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            with suppress(OSError):
                log.unlink()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = amici.import_model_module(name, str(where))
            model = module.get_model()
        return Engine(amici, model, version, commit, time.perf_counter() - t0, cached)
    except Exception as e:  # noqa: BLE001
        return (
            f"AMICI could not load its build of it ({where}; remove it to build it again): "
            f"{type(e).__name__}: {_first_line(e)}"
        )


# --- the check ---------------------------------------------------------------


def _times(x: float) -> str:
    """A margin as the note gives it: one not finite (the largest double,
    where a value is not finite) as ``inf``."""
    return "inf" if not math.isfinite(x) or x >= 1e300 else f"{x:.2g}"


def _engine_label(engine: Engine) -> str:
    label = f"AMICI {engine.version}"
    if engine.commit:
        label += f" ({engine.commit[:9]})"
    return label


def check(ctx: _Context) -> dict[str, Any]:
    """The ``"sensitivity_second_engine"`` check: see the module docstring."""
    skip = _check_sensitivity._skip(ctx, NAME)
    if skip:
        return skip
    model = ctx.rec.get("model") or {}
    fmt = model.get("format")
    if fmt not in ("sbml", "antimony"):
        return _verdict(
            NAME,
            "unsupported",
            note=f"AMICI reads SBML (and Antimony as SBML); this model is {fmt!r}",
        )
    if model.get("compartment_sizes"):
        return _verdict(
            NAME,
            "skip",
            note="the model was loaded with compartment_sizes= the file does not give",
        )
    skip = _check.same_as_file(ctx, NAME, "AMICI")
    if skip:
        return skip
    sbml, why = _sbml_text(str(model["path"]), str(fmt), model.get("sha256"))
    if sbml is None:
        return _verdict(NAME, "skip", note=why)
    import libsbml

    doc = libsbml.readSBMLFromString(sbml)
    why_not = not_comparable(doc.getModel())
    if why_not:
        return _verdict(NAME, "skip", note=why_not)
    cols = _check_sensitivity._context(ctx)
    not_judged: dict[str, str] = {}
    n_par = len(cols.params)
    for ic in cols.ics:
        not_judged[ic] = "an initial amount: AMICI gives sensitivities to parameters only"
    by_file = bngsim_ids(cols.params)
    globals_, locals_ = parameter_ids(doc.getModel())
    wanted = []
    for j in range(n_par):
        name, sid = cols.params[j], by_file[j]
        if sid is None:
            not_judged[name] = "another of the run's parameters has its id in the file"
        elif name.startswith(LOCAL_PREFIX) and sid in globals_:
            not_judged[name] = (
                f"the file has a global parameter {sid} as well, the name AMICI gives that one"
            )
        elif not name.startswith(LOCAL_PREFIX) and sid in locals_:
            not_judged[name] = "a reaction's local parameter has its name in AMICI's naming too"
        else:
            wanted.append(j)
    if not wanted:
        first = next(iter(not_judged.items()))
        return _verdict(NAME, "skip", note=f"no column AMICI can give ({first[0]}: {first[1]})")
    try:
        amici = _import_amici()
    except Exception as e:  # noqa: BLE001 - an install that cannot import is no install
        return _verdict(
            NAME,
            "skip",
            note="AMICI is not installed, or does not import here "
            f"({type(e).__name__}: {_first_line(e)}): it is no dependency or extra of bngsim "
            "(pip install amici; it compiles C++ per model and needs SWIG, a C++ compiler and "
            "a BLAS, and from AMICI 1.1 on Python 3.12 or later), so the sensitivities were "
            "not checked against a second engine",
        )
    err = ctx._ensure_drift()
    if err:
        return _verdict(
            NAME, "no_base", note=f"the run at a tenfold tighter tolerance raised ({err})"
        )
    assert ctx.moved is not None
    if not (ctx.relevant & (ctx.moved <= TOLERANCE_MOVE)).any():
        # compare_x would say no_base after the build: say it before.
        return _verdict(
            NAME,
            "no_base",
            note=f"no column is converged at the run's tolerance (each moves past "
            f"{TOLERANCE_MOVE:g} of its peak at a tenfold tighter one): nothing is judged",
        )
    # The run's own error, read before AMICI's build: the reruns are what
    # budget= bounds, and a budget spent there leaves no build to wait for.
    _check_sensitivity._tier1(ctx, cols)
    ctx._ensure_spread()
    ctx._left()
    engine = build_engine(amici, sbml)
    if isinstance(engine, str):
        return _verdict(NAME, "skip", note=engine)
    return compare(ctx, cols, engine, doc, wanted, not_judged)


def compare(
    ctx: _Context,
    cols: _Columns,
    engine: Engine,
    doc: Any,
    wanted: list[int],
    not_judged: dict[str, str],
) -> dict[str, Any]:
    """AMICI's run of the model against the run: the columns *wanted* that
    AMICI has, the species it integrates."""
    label = _engine_label(engine)
    changed: dict[str, Any] = {"engine": label}
    sbml_model = doc.getModel()
    file_ids = _element_ids(doc)
    by_file = bngsim_ids(cols.params)
    am_by_file = {
        s: k for k, s in enumerate(amici_ids(engine.free_ids, file_ids)) if s is not None
    }
    use: list[int] = []
    am_ids: list[str] = []
    values = ctx.run.get("parameters") or {}
    for j in wanted:
        k = am_by_file.get(str(by_file[j]))
        p = cols.params[j]
        if k is None:
            not_judged[p] = (
                "AMICI has no free parameter for it: an initial assignment or a rule sets it, "
                "or AMICI holds it fixed"
            )
            continue
        mine = values.get(p)
        theirs = engine.free_values[k]
        if mine is None:
            not_judged[p] = "the run records no value for it"
            continue
        if not math.isclose(float(mine), theirs, rel_tol=1e-9, abs_tol=0.0):
            not_judged[p] = f"AMICI reads it as {theirs:g}, the run as {float(mine):g}"
            continue
        use.append(j)
        am_ids.append(engine.free_ids[k])
    if not use:
        first = next(iter(not_judged.items()))
        return _verdict(
            NAME,
            "skip",
            note=f"no column AMICI can give ({first[0]}: {first[1]})",
            changed=changed,
        )

    # The species AMICI integrates, as concentrations.
    n_sp, n_cols = cols.atol.shape
    state_of = {s: k for k, s in enumerate(amici_ids(engine.state_ids, file_ids)) if s is not None}
    amounts = amount_species(sbml_model)
    sizes = compartment_sizes(sbml_model)
    rows: dict[int, tuple[int, float]] = {}
    species_not_compared: dict[str, str] = {}
    for i, s in enumerate(cols.species):
        k = state_of.get(s)
        if k is None:
            species_not_compared[s] = "no state of AMICI's has its name"
            continue
        size = 1.0
        if s in amounts:
            v = sizes.get(amounts[s])
            if v is None or not math.isfinite(v) or not v > 0:
                species_not_compared[s] = (
                    "AMICI holds it as an amount, in a compartment whose size the file does not "
                    "give as a constant"
                )
                continue
            size = v
        rows[i] = (k, size)
    if not rows:
        first = next(iter(species_not_compared.items()))
        return _verdict(
            NAME,
            "skip",
            note=f"no species AMICI integrates is bngsim's ({first[0]}: {first[1]})",
            changed=changed,
        )

    # AMICI held at least as tight as bngsim on every species and series it
    # gives, on its own scale: an amount is its concentration times the size,
    # and so is the absolute tolerance that resolves it as finely.
    # AMICI's atols are scalars, over states it compares and states it does
    # not alike: never looser than the run's smallest, either.
    rtol = float(ctx.rtol or 0.0)
    order = sorted(rows)
    size_of = np.asarray([rows[i][1] for i in order], dtype=float)

    def smallest(*arrays: NDArray[np.float64]) -> float | None:
        vals = np.concatenate([np.ravel(a) for a in arrays])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        return float(vals.min()) if vals.size else None

    run_atols = np.asarray(ctx.atols, dtype=float)
    atol = smallest(run_atols[order] * size_of, run_atols)
    if atol is None:
        atol = float(ctx.atol or 0.0)
    floors = cols.atol[:, use]
    atol_sens = smallest(floors[order] * size_of[:, None], floors)
    if atol_sens is None:
        atol_sens = atol
    times = np.asarray(ctx.result._time, dtype=float)
    t0 = float(ctx.run.get("t_start", times[0] if times.size else 0.0))
    method = str(ctx.run.get("sensitivity_method") or "staggered")
    n_t = times.size
    x = np.asarray(ctx.result._species, dtype=float)
    compared = np.zeros((n_sp, n_cols), dtype=bool)
    for i in rows:
        for j in use:
            compared[i, j] = not cols.refused[i, j]

    def amici_at(by: float) -> tuple[str, _Answer | None]:
        """AMICI at the run's tolerance over *by*, on bngsim's species (with
        bngsim's own values where AMICI gives none, so nothing else differs),
        or why it could not be had."""
        at = "" if by == 1.0 else f" at 1/{by:g} of the run's tolerance"
        try:
            failure, x_am, sx_am = engine.simulate(
                am_ids,
                t0,
                times,
                rtol=rtol / by,
                atol=atol / by,
                atol_sens=atol_sens / by,
                method=method,
            )
        except Exception as e:  # noqa: BLE001 - AMICI's failure is not bngsim's
            return f"AMICI could not run it{at}: {type(e).__name__}: {_first_line(e)}", None
        if failure:
            return f"AMICI could not run it{at}: {failure}", None
        if x_am.shape[0] != n_t or sx_am.shape[:2] != (n_t, len(engine.state_ids)):
            return (
                f"AMICI returned {x_am.shape} states and {sx_am.shape} sensitivities for "
                f"{n_t} times{at}",
                None,
            )
        x_f = x.copy()
        S_f = np.array(cols.S, dtype=float, copy=True)
        for i, (k, size) in rows.items():
            x_f[:, i] = x_am[:, k] / size
            for c, j in enumerate(use):
                if compared[i, j]:
                    S_f[:, i, j] = sx_am[:, k, c] / size
        seen = np.concatenate([x_f[:, sorted(rows)].ravel(), S_f[:, compared].ravel()])
        if not np.isfinite(seen).all():
            return f"AMICI returned values that are not finite{at}", None
        return "", (x_f, S_f)

    # AMICI at the run's tolerance and at a tenth of it: the tighter run is
    # the reference, and the move between the two is how far a correct engine
    # strays at the run's tolerance, which bngsim's own drift can understate.
    why, at_tol = amici_at(1.0)
    if at_tol is None:
        return _verdict(NAME, "skip", note=why, changed=changed)
    why, reference = amici_at(10.0)
    if reference is None:
        return _verdict(NAME, "skip", note=why, changed=changed)
    return _judge(
        ctx,
        cols,
        engine,
        reference,
        at_tol,
        amici_at,
        compared,
        changed,
        not_judged,
        species_not_compared,
        atol,
        atol_sens,
    )


#: ``(states, sensitivities)`` of one AMICI run on bngsim's species.
_Answer = tuple["NDArray[np.float64]", "NDArray[np.float64]"]


def _series_gap(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each series' largest |a - b| over time, ``inf`` where a value of either
    is not finite (a value the run gives as not finite is a finding, whatever
    AMICI gives; a row the run refuses by design is not judged)."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    g = np.where(fa & fb, np.abs(a - b), 0.0).max(axis=0)
    return np.where(fa.all(axis=0) & fb.all(axis=0), g, np.inf)


class _Level:
    """bngsim's run at one tolerance against AMICI's a decade tighter, series
    by series and species by species: see the module docstring.

    *S*, *x*: bngsim's sensitivities and states; *S_next*, *x_next*: bngsim's
    a decade tighter, whose move is the run's drift; *S_loose*, *x_loose*:
    bngsim's at twice the tolerance, or None; *ref* and *am*: AMICI a decade
    tighter (the reference) and at this tolerance; *floors*, *atols*, *rtol*:
    the tolerances at this level; *judged*: the series compared;
    *withhold*: give a series not converged no drift of its own (False to
    ask whether a gap is more than the series' own move can explain)."""

    def __init__(
        self,
        *,
        S: NDArray[np.float64],
        S_next: NDArray[np.float64],
        S_loose: NDArray[np.float64] | None,
        x: NDArray[np.float64],
        x_next: NDArray[np.float64],
        x_loose: NDArray[np.float64] | None,
        ref: _Answer,
        am: _Answer,
        floors: NDArray[np.float64],
        atols: NDArray[np.float64],
        rtol: float,
        judged: NDArray[np.bool_],
        withhold: bool = True,
    ) -> None:
        n_sp, n_cols = floors.shape
        x_ref, S_ref = ref
        x_am, S_am = am

        def flat(a: NDArray[np.float64]) -> NDArray[np.float64]:
            return a.reshape(-1)

        def shaped(a: NDArray[Any]) -> NDArray[Any]:
            return np.asarray(a).reshape(n_sp, n_cols)

        fin_b, fin_a = np.isfinite(S), np.isfinite(S_ref)
        peak = np.maximum(
            np.where(fin_b, np.abs(S), 0.0).max(axis=0),
            np.where(fin_a, np.abs(S_ref), 0.0).max(axis=0),
        )
        self.peak = peak
        self.d = _series_gap(S, S_ref)
        # Resolved on the larger peak: a series the run gives as 0 where AMICI
        # does not is judged on AMICI's scale, not taken for noise.
        self.resolved = shaped(_check.resolved(flat(peak), flat(floors), rtol))
        # Converged on the series' own scale (the larger peak): its move a
        # decade tighter past its floor, over that peak. The tolerance tier
        # reads a series it does not resolve, and one it takes to be held by a
        # floor, on its column's largest peak; here such a series would keep
        # its whole error as drift.
        d_next = _series_gap(S_next, S)
        move = shaped(moves(flat(d_next), flat(peak), flat(floors)))
        self.converged = ~self.resolved | (move <= TOLERANCE_MOVE)
        # The series' own drift, or half its move at twice the tolerance;
        # none for a series not converged, which would forgive its own error.
        drift = np.where(np.isfinite(d_next), d_next, 0.0)
        if S_loose is not None:
            drift = shaped(spread(flat(drift), flat(_series_gap(S_loose, S)), flat(peak)))
        if withhold:
            drift = np.where(self.converged, drift, 0.0)
        # AMICI's own move between its two runs, which measures how far a
        # correct engine strays at this tolerance. It is not capped: where both
        # engines err by percent (BIOMD0000000810's gamma_C column at rtol
        # 1e-7) a cap would read that error as a misreading. But where AMICI
        # moves past PATH_CEILING of a series' peak it is no reference: ten
        # times that would forgive a tenth of the series and more.
        am_move = _series_gap(S_am, S_ref)
        self.no_reference = judged & self.resolved & (am_move > PATH_CEILING * peak)
        self.wide = judged & self.resolved & ~self.no_reference & (am_move > TOLERANCE_MOVE * peak)
        drift = np.maximum(drift, np.where(np.isfinite(am_move), am_move, 0.0))
        self.use = judged & ~self.no_reference
        own = self.use & self.resolved
        self.ratio = np.zeros((n_sp, n_cols))
        self.rel = np.zeros((n_sp, n_cols))
        for j in range(n_cols):
            if not self.use[:, j].any():
                continue
            r, q = path_margins(
                self.d[:, j],
                peak[:, j],
                floors[:, j],
                rtol,
                drift[:, j],
                self.use[:, j],
                own=own[:, j],
            )
            self.ratio[:, j], self.rel[:, j] = r, q
        self.margin = float(self.ratio.max()) if self.ratio.size else 0.0
        self.value = float(self.rel.max()) if self.rel.size else 0.0

        # The states, on every path's rule (compare_x's), with AMICI's own
        # move as drift too.
        d_x, peak_x = differences(x_ref, x)
        d_next_x, _ = differences(x_next, x)
        relevant = _check.resolved(peak_x, atols, rtol)
        moved = np.where(relevant, moves(d_next_x, peak_x, atols), 0.0)
        self.x_conv = relevant & (moved <= TOLERANCE_MOVE)
        d_loose_x = None if x_loose is None else differences(x_loose, x)[0]
        drift_x = spread(d_next_x, d_loose_x, peak_x)
        move_x, _ = differences(x_am, x_ref)
        drift_x = np.maximum(drift_x, np.where(np.isfinite(move_x), move_x, 0.0))
        self.ratio_x, _ = path_margins(
            d_x, peak_x, atols, rtol, drift_x, np.ones(peak_x.shape, dtype=bool), own=self.x_conv
        )
        self.state_margin = float(self.ratio_x.max()) if self.ratio_x.size else 0.0
        self.failing = self.use & (self.ratio > 1.0)
        self.failing_x = self.ratio_x > 1.0
        self.diff = self.margin > 1.0 or self.state_margin > 1.0


def _judge(
    ctx: _Context,
    cols: _Columns,
    engine: Engine,
    reference: _Answer,
    at_tol: _Answer,
    amici_at: Any,
    judged: NDArray[np.bool_],
    changed: dict[str, Any],
    not_judged: dict[str, str],
    species_not_compared: dict[str, str],
    atol: float,
    atol_sens: float,
) -> dict[str, Any]:
    """The verdict: see the module docstring. *reference* is AMICI's run at a
    tenth of the run's tolerance and *at_tol* its run at the run's;
    ``amici_at(by)`` runs AMICI at the run's tolerance over *by*."""
    n_sp, n_cols = cols.atol.shape
    rtol = float(ctx.rtol or 0.0)
    assert ctx.tight is not None

    def block_of(result: Any) -> NDArray[np.float64]:
        try:
            return _check_sensitivity._block(result)
        except ValueError:
            return np.full(cols.S.shape, np.nan)

    S = np.asarray(cols.S, dtype=float)
    S_tight = block_of(ctx.tight)
    lv = _Level(
        S=S,
        S_next=S_tight,
        S_loose=None if ctx.loose is None else block_of(ctx.loose),
        x=ctx.x,
        x_next=ctx._view(ctx.tight),
        x_loose=None if ctx.loose is None else ctx._view(ctx.loose),
        ref=reference,
        am=at_tol,
        floors=cols.atol,
        atols=ctx.atols,
        rtol=rtol,
        judged=judged,
    )
    status = "diff" if lv.diff else "pass"
    n_judged = int(lv.use.any(axis=0).sum())
    if not lv.use.any() and not lv.diff:
        status = "skip"

    # A disagreement at the run's tolerance is the reading's only if it holds
    # a decade tighter as well: bngsim's run at a tenth of the tolerance
    # against AMICI's at a hundredth, judged alike. A model read two ways
    # differs as much at any tolerance; tolerance error, in either engine
    # (each one's own move can understate it: BIOMD0000000810 at rtol 1e-8),
    # does not.
    confirmed: bool | None = None
    unconfirmed = ""
    if lv.diff:
        why, deeper = amici_at(100.0)
        tight100 = ctx.tight100
        if tight100 is None and deeper is not None:
            try:
                tight100 = ctx.tight100 = ctx.rerun(**ctx._tighter(100.0))
            except _Out:
                why = "the budget ran out before bngsim's run at 1/100 of the tolerance"
            except Exception as e:  # noqa: BLE001 - the diff stands, unconfirmed
                why = (
                    f"bngsim's run at 1/100 of the tolerance raised {type(e).__name__}: "
                    f"{_first_line(e)}"
                )
        if deeper is None or tight100 is None:
            unconfirmed = why
        else:
            lv1 = _Level(
                S=S_tight,
                S_next=block_of(tight100),
                S_loose=None,
                x=ctx._view(ctx.tight),
                x_next=ctx._view(tight100),
                x_loose=None,
                ref=deeper,
                am=reference,
                floors=cols.atol / 10.0,
                atols=np.asarray(ctx.atols, dtype=float) / 10.0,
                rtol=rtol / 10.0,
                judged=lv.use,
                withhold=False,
            )
            # Held where the gap there is more than ten times what the series
            # itself still moves at that tolerance: a series bngsim does not
            # converge even a decade tighter keeps that move as its drift, so
            # its tolerance error is not read as a misreading, and a column
            # twice the truth still is one.
            confirmed = bool(
                (lv.failing & lv1.failing).any() or (lv.failing_x & lv1.failing_x).any()
            )
    tolerance_only = lv.diff and confirmed is False

    margin, value, state_margin = lv.margin, lv.value, lv.state_margin
    state_diff = state_margin > 1.0
    state_worst = ctx._worst(lv.ratio_x)
    use = lv.use
    trace = use & ~lv.resolved
    n_out = int((use & ~lv.converged).sum())
    n_nonfinite = int((use & np.isinf(lv.d)).sum())

    if n_nonfinite:
        head = f"{n_nonfinite} series not finite in the run where AMICI's are"
    else:
        head = f"{value:.2e} of a series' scale at most, {_times(margin)} of the allowance"
    note = (
        f"bngsim's forward sensitivities against {_engine_label(engine)}'s at a tenth of the "
        f"run's tolerance, from the model file: {head}, over {int(use.sum())} series of "
        f"{n_judged} of {n_cols} column(s); the states "
        f"{'disagree' if state_diff else 'agree'}, {_times(state_margin)} of their allowance"
    )
    if state_diff and state_worst:
        note += f" (most in {state_worst})"
    if lv.diff and confirmed:
        note += (
            "; it holds between bngsim's run at a tenth of the tolerance and AMICI's at a "
            "hundredth: the reading's, not the tolerance's"
        )
    elif tolerance_only:
        note += (
            "; bngsim's run at a tenth of the tolerance agrees with AMICI's at a hundredth: the "
            "tolerance's, not the reading's"
        )
    elif lv.diff:
        note += f"; it could not be compared a decade tighter ({unconfirmed})"
    not_converged = _check_sensitivity._named(cols, use & ~lv.converged)
    if n_out:
        note += (
            f"; {n_out} series not converged at the run's tolerance, judged with no drift of "
            f"their own ({', '.join(not_converged[:3])}{', ...' if n_out > 3 else ''})"
        )
    wide = _check_sensitivity._named(cols, lv.wide)
    if wide:
        note += (
            f"; AMICI itself moves past {TOLERANCE_MOVE:g} of their peak between its two runs "
            f"on {len(wide)} series, whose allowance takes ten times that move "
            f"({', '.join(wide[:3])}{', ...' if len(wide) > 3 else ''})"
        )
    no_reference = _check_sensitivity._named(cols, lv.no_reference)
    if no_reference:
        note += (
            f"; AMICI moves past {PATH_CEILING:g} of their peak between its two runs on "
            f"{len(no_reference)} series, no reference at the run's tolerance, not judged "
            f"({', '.join(no_reference[:3])}{', ...' if len(no_reference) > 3 else ''})"
        )
    if trace.any():
        note += (
            f"; {int(trace.sum())} series within {_check.TRACE_ATOLS:g} times their sensitivity "
            "atol of zero, judged on their column's whole scale too"
        )
    if not_judged:
        first = next(iter(not_judged.items()))
        note += f"; {len(not_judged)} column(s) not judged ({first[0]}: {first[1]})"
    if species_not_compared:
        first = next(iter(species_not_compared.items()))
        note += f"; {len(species_not_compared)} species not compared ({first[0]}: {first[1]})"
    pinned: bool | None = None if engine.commit is None else engine.commit == PINNED_COMMIT
    if pinned is None:
        note += (
            f"; {_engine_label(engine)} records no commit, so whether it is the one bngsim's "
            f"AMICI parity suite pins ({PINNED_COMMIT[:9]}) cannot be told"
        )
    elif not pinned:
        note += (
            f"; {_engine_label(engine)} is not the commit bngsim's AMICI parity suite pins "
            f"({PINNED_COMMIT[:9]})"
        )
    n_refused = int((cols.refused & judged.any(axis=0)[None, :]).sum())
    if n_refused:
        note += f"; {n_refused} series not finite in the run by design, not judged"
    if engine.cached:
        note += (
            "; AMICI's build of the model was cached; budget= bounds neither it nor AMICI's run"
        )
    else:
        note += (
            f"; AMICI built the model in {engine.build_seconds:.0f} s; budget= bounds neither "
            "that nor AMICI's run"
        )

    score = np.where(use, lv.ratio, 0.0).reshape(-1)
    entry = _check_sensitivity._columns_entry(cols, np.where(use, lv.ratio, -1.0).reshape(-1))
    gone = set(not_judged)
    entry["columns"] = [c for c in entry["columns"] if c["name"] not in gone]
    extra: dict[str, Any] = {
        "margin": _num(margin),
        **entry,
        "n_judged": n_judged,
        "not_judged": dict(list(not_judged.items())[:MAX_NAMED]),
        "state_margin": _num(state_margin),
        "state_diff": state_diff,
        "species_not_compared": dict(list(species_not_compared.items())[:MAX_NAMED]),
        "n_not_converged": n_out,
        "not_converged": not_converged[:MAX_NAMED],
        "n_amici_not_converged": len(wide),
        "amici_not_converged": wide[:MAX_NAMED],
        "n_no_reference": len(no_reference),
        "no_reference": no_reference[:MAX_NAMED],
        "confirmed_tighter": confirmed,
        "diff_not_converged": tolerance_only,
        "n_not_resolved": int(trace.sum()),
        "not_resolved": _check_sensitivity._named(cols, trace)[:MAX_NAMED],
        "n_refused": n_refused,
        "engine": {
            "name": "AMICI",
            "version": engine.version,
            "commit": engine.commit,
            "pinned": pinned,
            "cached": engine.cached,
            "build_seconds": _num(round(engine.build_seconds, 3)),
            "atol": _num(atol),
            "atol_sensitivities": _num(atol_sens),
        },
    }
    worst = _check_sensitivity._worst(cols, score)
    if state_diff and not margin > 1.0:
        worst = state_worst
    if status == "skip":
        note = (
            "AMICI moves past "
            f"{PATH_CEILING:g} of the peak of every series it gives between its two runs: it is "
            "no reference at the run's tolerance (rerun at a tighter one); " + note
        )
    return _verdict(
        NAME,
        status,
        value if status != "skip" else None,
        note,
        worst=worst,
        changed=changed,
        extra=extra,
    )
