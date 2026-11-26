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
its run; a build is cached under ``~/.cache/bngsim/amici`` (or
``$BNGSIM_AMICI_CACHE_DIR``), keyed by the SBML text and the AMICI install.

**What is carried over from ``parity_checks/amici_parity``**, which the wheel
cannot import, and only what the check needs:

- *Parameters.* bngsim names a reaction's local parameter
  ``_lp_<reaction>_<id>``; AMICI names it ``<reaction>_<id>``. AMICI's free
  parameter set is authoritative: a parameter an initial assignment or a rule
  sets is an expression to AMICI, with no column, and its column is not judged
  (bngsim's is then a partial derivative; a parameter that reads it gets the
  whole chain in both engines). Scaling is pinned to none, so AMICI's
  ``dx/dp`` is compared, not its ``dx/d ln p``.
- *Ids.* AMICI renames an id that collides with a symbol of its own (``k``,
  ``p``, ``x``) to ``amici_<id>``. The prefix is undone only where the file has
  no element of that id itself, and an id that two of AMICI's would map to is
  matched to neither (the parity suite's no-guessing rule).
- *States.* AMICI keeps a ``hasOnlySubstanceUnits`` species as an amount, in
  its states and its sensitivities alike, where bngsim reports a
  concentration. Such a species is divided by its compartment's size where the
  file gives that size as a constant (declared, set by no rule and no initial
  assignment), and not compared otherwise. A species AMICI holds no state for
  (a constant species, one a rule sets) is not compared either; each is named.
- *Initial amounts.* AMICI gives sensitivities to parameters only, so a
  ``sensitivity_ic=`` column is not judged here.
- *The dispositions* in ``AMICI_KNOWN_ISSUES.md`` that touch sensitivities are
  all about events or switches: AMICI computes no sensitivity jump at a switch
  time a parameter sets (Class 4's MODEL2105110001), misses an equality
  trigger between its output times and fires a strict time trigger at its own
  threshold (Classes 5 and 7, which the parity suite answers by running AMICI
  through each event's instant and leaving the sample on a strict onset
  unjudged), and reads a trigger or a piecewise condition that sits exactly at
  its threshold at the start differently from bngsim (the 2026-11 triage). A
  model with an event, or whose math switches anywhere (a piecewise, a
  comparison, a logical operator, ``floor``, ``ceiling``, ``min``, ``max``, a
  remainder or a delay), is therefore not compared: the check skips it and says
  so, rather than read AMICI's known misreadings of it as bngsim's. So is a
  model whose function definition names an argument twice, which defines no
  one model (MODEL2003040001: AMICI binds the first, bngsim the last). The
  event and switch cases are left to a follow-up, which would carry over the
  parity suite's event instants and onset rows.

**The rule** is the ``"sensitivity"`` tier's, series by series
(:class:`bngsim._check_sensitivity._Columns`): one species and one parameter,
over time, on the larger of its peak in bngsim and in AMICI, past the absolute
tolerance the solver held it to, against a path's allowance
(:func:`bngsim._check.path_margins`, column by column) with the series' own
move at the tenfold tighter tolerance (or half its move at twice the
tolerance) as its drift. A series the run does not resolve (within
:data:`bngsim._check.TRACE_ATOLS` of its tolerance of zero) is held to its
column's whole scale too. A series not converged at the run's tolerance is
judged on its own scale with no drift, which would forgive its own error, as
the quotient check judges it (a diff in such series alone is
``diff_not_converged``, which ``next`` reads as the tolerance's). The states AMICI
integrates are compared too, species by species, as every path is
(:meth:`bngsim._check._Context.compare_x`): sensitivities about two different
trajectories say nothing, and a disagreement there is reported as one about
the trajectory (``state_diff``). AMICI runs at the run's ``rtol``, at the
smallest of its atols for the states, and for the sensitivities at the
smallest of the judged series' absolute tolerances, so it is held at least as
tight as bngsim on every series.

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
import shutil
import sys
import sysconfig
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bngsim import _check, _check_sensitivity
from bngsim._check import (
    MAX_NAMED,
    TOLERANCE_MOVE,
    _first_line,
    _num,
    _verdict,
    differences,
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

#: AMICI's step limit for the run: CVODES' hard ceiling there, where bngsim's
#: ``max_steps`` is a batch it resumes from. The parity suite's sensitivity
#: budget, measured on its corpus (``_amici_sens.DEFAULT_SENS_MAX_STEPS``).
MAX_STEPS = 100_000

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
    convert it)."""
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        return None, f"the model file could not be read again: {_first_line(e)}"
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        return None, "the model file changed after the run's model was loaded from it"
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rdata = m.simulate(solver=solver)
        status = int(rdata.status)
        if status != 0:
            try:
                text = str(ss.simulation_status_to_str(status))
            except Exception:  # noqa: BLE001
                text = str(status)
            return f"status {status} ({text})", np.empty((0, 0)), np.empty((0, 0, 0))
        x = np.asarray(rdata.x, dtype=float)
        sx = np.transpose(np.asarray(rdata.sx, dtype=float), (0, 2, 1))
        return "", x, sx


def build_engine(amici: Any, sbml: str) -> Engine | str:
    """AMICI's model of *sbml*, built once and cached (:data:`CACHE_DIR`), or
    why it could not be built or loaded.

    A build is made in a directory of this process's own and renamed into
    place, so a cached directory is always a whole build, and two processes
    building the same model at once both end with one (the parity suite's
    ``_amici_sens._build_sens_model``)."""
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
    if not cached:
        staging = root / f".building_{name}_{os.getpid()}"
        try:
            root.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(staging, ignore_errors=True)
            with _tools_on_path(), warnings.catch_warnings():
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
            return f"AMICI could not build it: {type(e).__name__}: {_first_line(e)}"
        finally:
            shutil.rmtree(staging, ignore_errors=True)
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
    wanted = [j for j in range(n_par) if by_file[j] is not None]
    for j in range(n_par):
        if by_file[j] is None:
            not_judged[cols.params[j]] = "another of the run's parameters has its id in the file"
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
            "(pip install amici; it compiles C++ per model and needs SWIG, a C++ compiler, a "
            "BLAS and Python 3.12 or later), so the sensitivities were not checked against a "
            "second engine",
        )
    err = ctx._ensure_drift()
    if err:
        return _verdict(
            NAME, "no_base", note=f"the run at a tenfold tighter tolerance raised ({err})"
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
            species_not_compared[s] = "AMICI integrates no state for it"
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

    # AMICI held at least as tight as bngsim on every series judged.
    rtol = float(ctx.rtol or 0.0)
    atols = np.asarray(ctx.atols, dtype=float)
    positive = atols[atols > 0]
    atol = float(positive.min()) if positive.size else float(ctx.atol or 0.0)
    floors = cols.atol[np.ix_(sorted(rows), use)]
    floors = floors[np.isfinite(floors) & (floors > 0)]
    atol_sens = float(floors.min()) if floors.size else atol
    times = np.asarray(ctx.result._time, dtype=float)
    failure, x_am, sx_am = engine.simulate(
        am_ids,
        float(ctx.run.get("t_start", times[0] if times.size else 0.0)),
        times,
        rtol=rtol,
        atol=atol,
        atol_sens=atol_sens,
        method=str(ctx.run.get("sensitivity_method") or "staggered"),
    )
    if failure:
        return _verdict(NAME, "skip", note=f"AMICI could not run it: {failure}", changed=changed)
    n_t = times.size
    if x_am.shape[0] != n_t or sx_am.shape[:2] != (n_t, len(engine.state_ids)):
        return _verdict(
            NAME,
            "skip",
            note=f"AMICI returned {x_am.shape} states and {sx_am.shape} sensitivities for "
            f"{n_t} times",
            changed=changed,
        )

    # bngsim's own values where AMICI gives none, so nothing else differs.
    x = np.asarray(ctx.result._species, dtype=float)
    x_cmp = x.copy()
    S_cmp = np.array(cols.S, dtype=float, copy=True)
    compared = np.zeros((n_sp, n_cols), dtype=bool)
    for i, (k, size) in rows.items():
        x_cmp[:, i] = x_am[:, k] / size
        for c, j in enumerate(use):
            S_cmp[:, i, j] = sx_am[:, k, c] / size
            compared[i, j] = True
    live = compared & ~cols.refused
    seen = np.concatenate([x_cmp[:, sorted(rows)].ravel(), S_cmp[:, live].ravel()])
    if not np.isfinite(seen).all():
        return _verdict(
            NAME, "skip", note="AMICI returned values that are not finite", changed=changed
        )
    return _judge(
        ctx,
        cols,
        engine,
        x_cmp,
        S_cmp,
        live,
        changed,
        not_judged,
        species_not_compared,
        use,
        atol_sens,
    )


def _judge(
    ctx: _Context,
    cols: _Columns,
    engine: Engine,
    x_cmp: NDArray[np.float64],
    S_cmp: NDArray[np.float64],
    judged: NDArray[np.bool_],
    changed: dict[str, Any],
    not_judged: dict[str, str],
    species_not_compared: dict[str, str],
    use: list[int],
    atol_sens: float,
) -> dict[str, Any]:
    """The verdict: see the module docstring."""
    n_sp, n_cols = cols.atol.shape
    move_flat, keep_flat, d10_flat, _ = _check_sensitivity._tier1(ctx, cols)
    move = np.asarray(move_flat, dtype=float).reshape(n_sp, n_cols)
    keep = np.asarray(keep_flat, dtype=bool).reshape(n_sp, n_cols)
    d10 = np.asarray(d10_flat, dtype=float).reshape(n_sp, n_cols)
    converged = move <= TOLERANCE_MOVE
    S = np.asarray(cols.S, dtype=float)
    fin_b, fin_a = np.isfinite(S), np.isfinite(S_cmp)
    peak = np.maximum(
        np.where(fin_b, np.abs(S), 0.0).max(axis=0),
        np.where(fin_a, np.abs(S_cmp), 0.0).max(axis=0),
    )
    gap = np.where(fin_b & fin_a, np.abs(S - S_cmp), 0.0).max(axis=0)
    # A value the run gives as not finite is a finding, whatever AMICI gives
    # (a row the run refuses by design is not judged).
    d = np.where(fin_b.all(axis=0) & fin_a.all(axis=0), gap, np.inf)
    # The series' own drift; none for a series not converged at the run's
    # tolerance, which would forgive its own error.
    drift = np.where(np.isfinite(d10), d10, 0.0)
    if ctx.loose is not None:
        try:
            d_loose, _ = differences(cols.flat(_check_sensitivity._block(ctx.loose)), cols.flat(S))
        except ValueError:
            pass
        else:
            drift = spread(drift.reshape(-1), d_loose, peak.reshape(-1)).reshape(n_sp, n_cols)
    drift = np.where(converged & np.isfinite(drift), drift, 0.0)
    own = judged & keep
    rtol = float(ctx.rtol or 0.0)

    # The states, as every path's are judged.
    state = ctx.compare_x(NAME, x_cmp, changed)
    if state["status"] == "no_base":
        return state
    ratio = np.zeros((n_sp, n_cols))
    rel = np.zeros((n_sp, n_cols))
    for j in range(n_cols):
        if not judged[:, j].any():
            continue
        r, q = path_margins(
            d[:, j], peak[:, j], cols.atol[:, j], rtol, drift[:, j], judged[:, j], own=own[:, j]
        )
        ratio[:, j], rel[:, j] = r, q
    margin = float(ratio.max()) if ratio.size else 0.0
    value = float(rel.max()) if rel.size else 0.0
    state_margin = float(state.get("margin") or 0.0)
    state_diff = state["status"] == "diff"
    failing = judged & (ratio > 1.0)
    unconverged_only = bool(failing.any() and not (failing & converged).any() and not state_diff)
    status = "diff" if margin > 1.0 or state_diff else "pass"
    n_judged = int(judged.any(axis=0).sum())
    trace = judged & ~keep
    n_out = int((judged & ~converged).sum())

    note = (
        f"bngsim's forward sensitivities against {_engine_label(engine)}'s, from the model "
        f"file: {value:.2e} of a series' scale at most, {margin:.2g} of the allowance, over "
        f"{int(judged.sum())} series of {n_judged} of {n_cols} column(s); the states "
        f"{'disagree' if state_diff else 'agree'}, {state_margin:.2g} of their allowance"
    )
    if state_diff and state.get("worst_column"):
        note += f" (most in {state['worst_column']})"
    if n_out:
        note += (
            f"; {n_out} series not converged at the run's tolerance, judged on the tolerance alone"
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
    pinned = engine.commit == PINNED_COMMIT
    if not pinned:
        note += (
            f"; {_engine_label(engine)} is not the commit bngsim's AMICI parity suite pins "
            f"({PINNED_COMMIT[:9]})"
        )
    if engine.cached:
        note += "; AMICI's build of the model was cached"
    else:
        note += (
            f"; AMICI built the model in {engine.build_seconds:.0f} s, which budget= does not "
            "bound"
        )
    n_refused = int((cols.refused & judged.any(axis=0)[None, :]).sum())

    score = np.where(judged, ratio, 0.0).reshape(-1)
    entry = _check_sensitivity._columns_entry(cols, np.where(judged, ratio, -1.0).reshape(-1))
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
        "diff_not_converged": unconverged_only,
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
            "atol_sensitivities": _num(atol_sens),
        },
    }
    worst = _check_sensitivity._worst(cols, score)
    if state_diff and not margin > 1.0:
        worst = state.get("worst_column")
    return _verdict(NAME, status, value, note, worst=worst, changed=changed, extra=extra)
