"""The calls a caller brings from another tool, each mapped to bngsim's (issue #1700).

A caller who has not read bngsim's documentation writes the call it knows from
libRoadRunner, Tellurium, SciPy or BioNetGen, runs it, and corrects from the
error. Python's own ``AttributeError`` names nothing to use instead, so the
next attempt is another guess. This table is what bngsim answers with instead:
the module's ``__getattr__`` reads it for ``bngsim.<name>``; each name under
:class:`~bngsim.Model`, :class:`~bngsim.Simulator` and :class:`~bngsim.Result`
is a descriptor on that class (:func:`install`), for the name read on an
instance or on the class, or written; and :meth:`Simulator.run` reads the
``"Simulator.run"`` entries for a keyword it does not take. Each error names
the call to make.

None of these is an alias. A second spelling of a call is a second thing to
keep right, and some would be wrong answers rather than errors: BioNetGen's
``n_steps=10`` gives 11 output points where ``n_points=10`` gives 10, so a run
that accepted ``n_steps`` as ``n_points`` would return one row short with
nothing raised.

:data:`FOREIGN_NAMES` is data so that other text can be generated from it: the
agent guide's translation table (issue #1699, :mod:`bngsim._agent_guide`) is,
and a test pins the table and what each entry points at. When bngsim retires a name of its own,
the name goes in here at removal, with ``origin`` naming the release, so a
caller who learned the old spelling from an older release, or from text
written about one, gets the new one in the error. None has been yet: no public
method of the three classes and no name of ``bngsim.__all__`` has been removed
since the first public release.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import numbers
import os
import re
from collections.abc import Callable, Container, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn


@dataclass(frozen=True)
class ForeignName:
    """One name bngsim does not have, and the call to make instead.

    Attributes
    ----------
    owner : str
        Where the name was looked up: ``"bngsim"`` (the module), ``"Model"``,
        ``"Simulator"`` or ``"Result"`` (an attribute of one, or of the class),
        or ``"Simulator.run"`` (a keyword argument of :meth:`Simulator.run`, and
        of each other run call that takes the keyword the row names), or
        ``"Simulator.<method>"`` (one of that method's own).
        ``"pattern"`` is a habit no error of bngsim's names the call for
        (:data:`PATTERNS`): listed for the agent guide only.
        ``"translation"`` is another tool's call on that tool's own objects,
        which no bngsim object could meet: listed for the agent guide only.
    name : str
        The name as the caller wrote it; for a ``"pattern"`` or a
        ``"translation"``, the code.
    use : str
        bngsim's call, as code, written over ``model``, ``sim`` and ``result``;
        empty where no one call replaces it and :attr:`note` says what does.
    note : str
        What the caller needs beyond the call, in a sentence or two: where the
        translation is not one-to-one, the difference.
    origin : str
        Where the habit comes from: a tool (``"libRoadRunner"``,
        ``"Tellurium"``, ``"AMICI"``, ``"COPASI (basico)"``, ``"BNGL actions"``,
        ``"PySB"``, ``"SciPy"``, ``"pandas"``),
        ``"bngsim"`` for a bngsim name asked of the wrong object,
        ``"guess"`` for a name no tool has that callers try anyway, or
        ``"bngsim X.Y"`` for a name of bngsim's own that release removed.
    """

    owner: str
    name: str
    use: str
    note: str = ""
    origin: str = "guess"

    @property
    def hint(self) -> str:
        """The sentence the error carries after Python's own."""
        use = f"Use {self.use}." if self.use else ""
        return " ".join(part for part in (use, self.note) if part)


#: The Simulator calls whose wrong keywords and positional arguments are named.
#: Each takes ``*_extra`` and ``**_foreign`` typed ``NoReturn`` and checks them
#: first in its body (issue #1700).
RUN_CALLS = (
    "run",
    "run_batch",
    "run_replicates",
    "parameter_scan",
    "bifurcate",
    "compute_all_sensitivities",
    "run_until",
)

# BioNetGen's n_steps and RoadRunner's simulate(start, end, points) both read
# naturally as n_points, and one of the two is off by one.
_N_POINTS = (
    "n_points counts the output points, the first included: BioNetGen's n_steps=10 and "
    "RoadRunner's simulate(0, 10, 11) are both n_points=11."
)
_RUN = "sim.run(t_span=(t0, t1), n_points=n)"
_BUILT = "The method is chosen when the Simulator is built; Simulator.METHODS lists them."
_ON_MODEL = "Parameters and species are the Model's, and sim.model is the one this Simulator runs."
# What describe() says of a model's compartments, reactions and units (issue
# #1843). limit=None: describe() cuts each list at 50 entries by default.
_COMPARTMENTS = 'describe(limit=None).get("compartments", [])'
_COMPARTMENTS_NOTE = (
    'Each entry is a dict, its name under "name". An SBML or Antimony model has '
    "compartments; a .net or BNGL file has none, and the list is empty."
)
_REACTIONS = 'describe(limit=None)["reactions"]'
_REACTIONS_NOTE = (
    "Each entry is a dict; for an SBML or Antimony model, the file's reactions as written, "
    'each with its "id". The channels BNGsim runs can be more or fewer: n_reactions on '
    "the model counts those."
)
_N_REACTIONS_NOTE = (
    "The model's reactions as its file has them, as RoadRunner counts them. n_reactions on "
    "the model counts the channels BNGsim runs, which can be more or fewer."
)
_UNITS_NOTE = (
    "BNGsim converts no unit: species_values says what the numbers are, and each entry of "
    'model.describe()["species"] whether it is an amount or a concentration. A run\'s '
    'result.record["columns"] names the SBML unit ids the file declares.'
)

# What a run's columns are, and how it went (issue #1703).
_RECORD_NOTE = (
    "Each column's quantity (concentration, amount or count), compartment and size, and the "
    "SBML unit ids the file declares; BNGsim converts no unit."
)


def _entries(owner: str, origin: str, *rows: tuple[str, str] | tuple[str, str, str]) -> list:
    return [
        ForeignName(owner, row[0], row[1], row[2] if len(row) > 2 else "", origin) for row in rows
    ]


FOREIGN_NAMES: tuple[ForeignName, ...] = tuple(
    # ── the module ───────────────────────────────────────────────────────
    _entries(
        "bngsim",
        "guess",
        (
            "load",
            "bngsim.Model.load(path)",
            "It reads .net, .bngl, .xml, .sbml and .ant files by suffix.",
        ),
        (
            "run",
            f"bngsim.Simulator(model).{_RUN.removeprefix('sim.')}",
            "bngsim.run_bngl(path) runs the actions a .bngl file holds.",
        ),
        (
            "seed",
            "sim.run(seed=n)",
            "Each run takes its own seed; result.seed reports the one used.",
        ),
        (
            "set_seed",
            "sim.run(seed=n)",
            "Each run takes its own seed; result.seed reports the one used.",
        ),
    )
    + _entries(
        "bngsim",
        "BNGL actions",
        (
            "simulate",
            f"bngsim.Simulator(bngsim.Model.load(path)).{_RUN.removeprefix('sim.')}",
            "A Model holds the network; a Simulator runs it. " + _N_POINTS,
        ),
    )
    + _entries(
        "bngsim",
        "Tellurium",
        ("loada", "bngsim.Model.from_antimony_string(text)"),
        (
            "loadAntimonyModel",
            "bngsim.Model.from_antimony_string(text)",
            "bngsim.Model.from_antimony(path) reads a file.",
        ),
        (
            "loadSBMLModel",
            "bngsim.Model.from_sbml(path)",
            "bngsim.Model.from_sbml_string(text) reads SBML text.",
        ),
    )
    + _entries(
        "bngsim",
        "libRoadRunner",
        (
            "RoadRunner",
            "bngsim.Simulator(bngsim.Model.load(path))",
            "A Model holds the network and its parameters; a Simulator runs it.",
        ),
    )
    + _entries(
        "bngsim",
        "bngsim",
        *(
            (name, f"bngsim.Model.{name}({arg})", "The loaders are classmethods of Model.")
            for name, arg in (
                ("from_net", "path"),
                ("from_bngl", "path"),
                ("from_sbml", "path"),
                ("from_sbml_string", "text"),
                ("from_antimony", "path"),
                ("from_antimony_string", "text"),
            )
        ),
    )
    # ── Model ────────────────────────────────────────────────────────────
    + _entries(
        "Model",
        "guess",
        ("from_file", "bngsim.Model.load(path)", "It reads a model file by its suffix."),
        (
            "from_string",
            "bngsim.Model.from_sbml_string(text) or bngsim.Model.from_antimony_string(text)",
        ),
        ("from_sbml_file", "bngsim.Model.from_sbml(path)"),
        ("from_antimony_file", "bngsim.Model.from_antimony(path)"),
        ("from_net_file", "bngsim.Model.from_net(path)"),
        ("from_bngl_file", "bngsim.Model.from_bngl(path)"),
        (
            "simulate",
            f"bngsim.Simulator(model).{_RUN.removeprefix('sim.')}",
            "A Model holds the network; a Simulator runs it. " + _N_POINTS,
        ),
        (
            "run",
            f"bngsim.Simulator(model).{_RUN.removeprefix('sim.')}",
            "A Model holds the network; a Simulator runs it. " + _N_POINTS,
        ),
        ("set_parameter", "model.set_param(name, value)"),
        ("get_parameter", "model.get_param(name)"),
        ("set_parameters", "model.set_params({name: value, ...})"),
        ("n_params", "model.n_parameters"),
        (
            "parameters",
            "model.param_names",
            "model.get_param(name) and model.set_param(name, value) read and set one.",
        ),
        (
            "species",
            "model.species_names",
            "model.get_concentration(name) and model.set_concentration(name, value) read and set "
            "one; model.get_state() returns them all.",
        ),
        (
            "observables",
            "model.observable_names",
            "A run computes their values: result.observables[name].",
        ),
        # What a model is, in one call (issue #1843): describe() holds it all.
        *(
            (name, "model.describe()", "print(model.describe()) is the readable form.")
            for name in ("info", "to_dict")
        ),
        *(
            (name, f"model.{_COMPARTMENTS}", _COMPARTMENTS_NOTE)
            for name in ("compartments", "compartment_names")
        ),
        *(
            (name, f"model.{_REACTIONS}", _REACTIONS_NOTE)
            for name in ("reactions", "reaction_names")
        ),
        *(
            (name, 'model.describe()["species_values"]', _UNITS_NOTE)
            for name in ("units", "species_units")
        ),
    )
    + _entries(
        "Model",
        "bngsim",
        (
            "summary",
            "print(model.describe())",
            "result.summary() describes a run; model.describe() is the model's, as a dict "
            "that print shows as a report.",
        ),
        (
            "from_bngl_string",
            "bngsim.Model.from_bngl(path)",
            "BNGsim reads BNGL only from a file, through BNG2.pl: write the text to a .bngl "
            "file first. SBML and Antimony text have from_sbml_string and from_antimony_string.",
        ),
    )
    + _entries(
        "Model",
        "BNGL actions",
        ("setParameter", "model.set_param(name, value)"),
        ("setConcentration", "model.set_concentration(name, value)"),
        ("saveConcentrations", "model.save_concentrations()"),
        (
            "resetConcentrations",
            "model.reset()",
            "model.restore_concentrations(label) restores a state saved with "
            "model.save_concentrations(label).",
        ),
    )
    + _entries(
        "Model",
        "libRoadRunner",
        ("getFloatingSpeciesIds", "model.species_names"),
        (
            "getGlobalParameterIds",
            "model.param_names",
            "model.primary_param_names leaves out the ones the model derives from others.",
        ),
        ("getValue", "model.get_param(name)", "model.get_concentration(name) reads a species."),
        (
            "setValue",
            "model.set_param(name, value)",
            "model.set_concentration(name, value) sets a species.",
        ),
        (
            "resetAll",
            "model.reset()",
            "It resets the species only; parameters keep the values set since the model was "
            "loaded. Set them back with model.set_params({...}).",
        ),
        ("getInfo", "model.describe()", "print(model.describe()) is the readable form."),
        ("getCompartmentIds", f"model.{_COMPARTMENTS}", _COMPARTMENTS_NOTE),
        ("getReactionIds", f"model.{_REACTIONS}", _REACTIONS_NOTE),
        ("getNumReactions", 'model.describe()["counts"]["reactions"]', _N_REACTIONS_NOTE),
    )
    # ── Simulator ────────────────────────────────────────────────────────
    + _entries(
        "Simulator",
        "libRoadRunner",
        ("simulate", _RUN, "BNGsim has no simulate(). " + _N_POINTS),
        (
            "reset",
            "sim.model.reset()",
            "Resetting the species is the Model's: it returns them to their initial conditions, "
            "or to the state an unlabeled model.save_concentrations() saved.",
        ),
        (
            "resetAll",
            "sim.model.reset()",
            "It resets the species only; parameters keep the values set since the model was "
            "loaded. Set them back with sim.model.set_params({...}).",
        ),
        ("steadyState", "sim.steady_state()"),
        ("getInfo", "sim.model.describe()", "print(sim.model.describe()) is the readable form."),
        ("getCompartmentIds", f"sim.model.{_COMPARTMENTS}", _COMPARTMENTS_NOTE),
        ("getReactionIds", f"sim.model.{_REACTIONS}", _REACTIONS_NOTE),
        ("getNumReactions", 'sim.model.describe()["counts"]["reactions"]', _N_REACTIONS_NOTE),
        (
            "getSteadyStateValues",
            "sim.steady_state().concentrations",
            "steady_state() returns a SteadyStateResult; ss[name] reads one species.",
        ),
        (
            "gillespie",
            f'bngsim.Simulator(model, method="ssa").{_RUN.removeprefix("sim.")}',
            _BUILT,
        ),
        ("setIntegrator", 'bngsim.Simulator(model, method="ssa")', _BUILT),
        (
            "integrator",
            "sim.method",
            "Choose the method with bngsim.Simulator(model, method=...), and the tolerances "
            "with sim.set_tolerances(rtol=..., atol=...).",
        ),
        (
            "setSeed",
            "sim.run(seed=n)",
            "Each run takes its own seed; result.seed reports the one used.",
        ),
        ("getFloatingSpeciesIds", "sim.model.species_names"),
        ("getGlobalParameterIds", "sim.model.param_names"),
        (
            "getValue",
            "sim.model.get_param(name)",
            "sim.model.get_concentration(name) reads a species.",
        ),
        (
            "setValue",
            "sim.model.set_param(name, value)",
            "sim.model.set_concentration(name, value) sets a species.",
        ),
        (
            "timeCourseSelections",
            "result.observables[name]",
            "A run records every species, observable and expression; select from its Result: "
            "result.species[name], result.observables[name], result.expressions[name].",
        ),
    )
    + _entries(
        "Simulator",
        "guess",
        ("integrate", _RUN, _N_POINTS),
        ("solve", _RUN, _N_POINTS),
        (
            "seed",
            "sim.run(seed=n)",
            "Each run takes its own seed; result.seed reports the one used.",
        ),
        (
            "set_seed",
            "sim.run(seed=n)",
            "Each run takes its own seed; result.seed reports the one used.",
        ),
        (
            "rtol",
            "sim.set_tolerances(rtol=..., atol=...)",
            "Or pass rtol= and atol= to one run().",
        ),
        (
            "atol",
            "sim.set_tolerances(rtol=..., atol=...)",
            "Or pass rtol= and atol= to one run().",
        ),
        ("result", "result = sim.run(...)", "run() returns the Result; the Simulator keeps none."),
        ("set_parameter", "sim.model.set_param(name, value)", _ON_MODEL),
    )
    + _entries(
        "Simulator",
        "bngsim",
        *(
            (name, f"sim.model.{call}", _ON_MODEL)
            for name, call in (
                ("set_param", "set_param(name, value)"),
                ("get_param", "get_param(name)"),
                ("set_params", "set_params({name: value, ...})"),
                ("set_concentration", "set_concentration(name, value)"),
                ("get_concentration", "get_concentration(name)"),
                ("species_names", "species_names"),
                ("param_names", "param_names"),
                ("observable_names", "observable_names"),
            )
        ),
    )
    # ── Result ───────────────────────────────────────────────────────────
    + _entries(
        "Result",
        "pandas",
        *(
            (
                name,
                "result.dataframe",
                "It is a property, read without parentheses: time and one column per "
                "observable. It needs pandas.",
            )
            for name in ("to_dataframe", "to_pandas", "as_dataframe", "to_df", "df")
        ),
        (
            "to_numpy",
            "result.species",
            "result.time, result.species, result.observables and result.expressions are numpy "
            "arrays.",
        ),
    )
    + _entries(
        "Result",
        "SciPy",
        ("t", "result.time"),
        (
            "y",
            "result.species",
            "Its shape is (n_times, n_species), the transpose of SciPy's sol.y: result.species.T "
            "is SciPy's layout.",
        ),
        *(
            (
                name,
                "",
                "run() raises bngsim.SimulationError when a run fails, so a Result it returns is "
                "a run that finished. A stop condition that ends a run early raises "
                "bngsim.StopConditionMet, which carries the cut Result.",
            )
            for name in ("success", "status", "message")
        ),
    )
    + _entries(
        "Result",
        "Tellurium",
        (
            "plot",
            'result.dataframe.plot(x="time")',
            "BNGsim draws no plots; this needs pandas and matplotlib.",
        ),
    )
    + _entries(
        "Result",
        "libRoadRunner",
        (
            "colnames",
            "result.species_names",
            "result.observable_names names the observable columns, and "
            "result.as_roadrunner().colnames gives RoadRunner's.",
        ),
    )
    + _entries(
        "Result",
        "guess",
        ("times", "result.time"),
        # What a run is and how it went (issue #1703): its record says.
        *((name, 'result.record["columns"]', _RECORD_NOTE) for name in ("units", "species_units")),
        ("time_units", 'result.record["time"]["units"]'),
        ("method", 'result.record["run"]["method"]', "result.record says how the run went."),
        ("metadata", "result.record", "json.dumps(result.record) writes it."),
        (
            "warnings",
            "result.notices",
            "Every warning of the run, whatever the warning filters showed.",
        ),
    )
    # ── Simulator.run keywords ───────────────────────────────────────────
    + _entries(
        "Simulator.run",
        "BNGL actions",
        ("t_end", "t_span=(t_start, t_end)", "The interval is one (t_start, t_end) tuple."),
        ("t_start", "t_span=(t_start, t_end)", "The interval is one (t_start, t_end) tuple."),
        (
            "n_steps",
            "n_points=n_steps + 1",
            "BioNetGen's n_steps counts the intervals between output points; n_points counts the "
            "points, the first included, so n_steps=10 is n_points=11.",
        ),
    )
    + _entries(
        "Simulator.run",
        "libRoadRunner",
        ("end", "t_span=(t_start, end)", "The interval is one (t_start, t_end) tuple."),
        ("points", "n_points=points", "Both count the output points, the first included."),
        (
            "steps",
            "n_points=steps + 1",
            "RoadRunner's steps counts the intervals between output points; n_points counts the "
            "points, the first included.",
        ),
        ("integrator", "bngsim.Simulator(model, method=...)", _BUILT),
        (
            "selections",
            "result.observables[name]",
            "run() records every species, observable and expression; select from its Result: "
            "result.species[name], result.observables[name], result.expressions[name].",
        ),
    )
    + _entries(
        "Simulator.run",
        "SciPy",
        (
            "t_eval",
            "sample_times=t_eval",
            "SciPy integrates from t_span[0]; a BNGsim run starts at the first sample time, so "
            "put the start time first.",
        ),
        ("method", "bngsim.Simulator(model, method=...)", _BUILT),
        (
            "y0",
            "model.set_concentration(name, value)",
            "A run starts from the model's state: set it before run(), all of it at once with "
            "model.set_state(values).",
        ),
    )
    + _entries(
        "Simulator.run",
        "guess",
        ("times", "sample_times=times"),
        ("tspan", "t_span=(t_start, t_end)"),
        ("tend", "t_span=(t_start, t_end)", "The interval is one (t_start, t_end) tuple."),
        (
            "duration",
            "t_span=(t_start, t_start + duration)",
            "The interval is one (t_start, t_end) tuple.",
        ),
    )
    # ── Keywords of the other run calls ──────────────────────────────────
    #
    # The Simulator.run rows above serve run_batch, run_replicates,
    # parameter_scan and run_until too, wherever the keyword a row names is one
    # the method takes (keyword_entry): t_end= is t_span= on all but run_until.
    # These are each method's own.
    + _entries(
        "Simulator.parameter_scan",
        "BNGL actions",
        (
            "suffix",
            "",
            "BNGsim writes no files from a scan: parameter_scan() returns one Result per "
            "point, and squeeze=True stacks them into one.",
        ),
    )
    + _entries(
        "Simulator.parameter_scan",
        "guess",
        ("scan_values", "par_scan_vals=scan_values", "BNGsim keeps BioNetGen's name."),
        ("values", "par_scan_vals=values", "BNGsim keeps BioNetGen's name."),
    )
    + _entries(
        "Simulator.run_batch",
        "SciPy",
        (
            "y0",
            "model.set_concentration(name, value)",
            "run_batch() starts every point from the initial conditions (Model.reset): set the "
            "state, then model.save_concentrations() to make it the one reset returns to, or "
            "set an initial-condition parameter in each params row.",
        ),
    )
    + _entries(
        "Simulator.run_batch",
        "bngsim",
        (
            "start",
            "",
            "run_batch() starts every point from the initial conditions (Model.reset); "
            "model.save_concentrations() first makes the current state the one it resets to.",
        ),
    )
    + _entries(
        "Simulator.run_replicates",
        "bngsim",
        (
            "start",
            "",
            "Every replicate starts from the model's state at the call; model.reset() first "
            "starts them from the initial conditions.",
        ),
    )
    + _entries(
        "Simulator.parameter_scan",
        "bngsim",
        (
            "start",
            "reset_to=label",
            "Each point starts from the model's state at the call, or from a state saved with "
            "model.save_concentrations(label); model.reset() first starts them from the initial "
            "conditions.",
        ),
    )
    + _entries(
        "Simulator.run_until",
        "bngsim",
        (
            "max_step",
            "",
            "run_until() takes no max_step: max_steps= is a count of steps, and "
            "sim.run(..., max_step=...) bounds the size of one.",
        ),
        ("timeout", "", "run_until() takes no timeout; sim.run(..., timeout=...) does."),
    )
    + _entries(
        "Simulator.run_until",
        "guess",
        (
            "duration",
            "t=sim.current_time + duration",
            "run_until() runs on to the one time t, its first argument.",
        ),
        *(
            (name, f"t={name}", "run_until() runs on to the one time t, its first argument.")
            for name in ("t_end", "end", "tend")
        ),
        (
            "t_start",
            "sim.set_time(t_start)",
            "run_until() runs on from the simulator's clock, sim.current_time, to t.",
        ),
    )
    + [
        ForeignName(
            f"Simulator.{method}",
            name,
            "",
            f"{method}() writes a uniform grid of n_points output times; "
            "run() takes explicit ones, sample_times=.",
            "bngsim" if name == "sample_times" else ("SciPy" if name == "t_eval" else "guess"),
        )
        for method in RUN_CALLS[1:]
        for name in ("sample_times", "times", "t_eval")
    ]
    # ── Patterns: for the agent guide only ───────────────────────────────
    #
    # Habits whose error cannot name bngsim's call without slowing calls that
    # succeed: the name is the model's own (a parameter, a species), so no
    # descriptor can be put there for it, and a __getattr__, __setattr__ or
    # __getitem__ that looked would cost every lookup or write (see "Attribute
    # lookups" below). Python's own error is what the caller gets; the guide
    # generated from this table (#1699) is where these are said up front.
    + _entries(
        "pattern",
        "libRoadRunner",
        (
            "model.k1 = value",
            'model.set_param("k1", value)',
            "libRoadRunner sets a parameter as an attribute; on a BNGsim Model the write "
            "fails with Python's own AttributeError. A species: "
            "model.set_concentration(name, value).",
        ),
        (
            "model.k1",
            'model.get_param("k1")',
            "A species: model.get_concentration(name).",
        ),
        (
            'model["k1"] = value',
            'model.set_param("k1", value)',
            "A species: model.set_concentration(name, value).",
        ),
        (
            'model["S1"]',
            'model.get_concentration("S1")',
            "A parameter: model.get_param(name).",
        ),
        (
            "sim.k1 = value",
            'sim.model.set_param("k1", value)',
            "libRoadRunner's one object is both the model and its simulator; in BNGsim, "
            "parameters and species are the Model's, and sim.model is it.",
        ),
    )
    + _entries(
        "pattern",
        "pandas",
        (
            "result.A",
            'result.observables["A"]',
            'A species: result.species["A"]; an expression: result.expressions["A"].',
        ),
    )
    + _entries(
        "pattern",
        "guess",
        (
            "from bngsim import simulate",
            "import bngsim",
            "Then bngsim.Simulator(bngsim.Model.load(path)).run(t_span=(t0, t1), n_points=n). "
            "Python reports a name a module lacks, imported with from ... import, as a bare "
            "ImportError, without the hint the same name read as an attribute gets.",
        ),
    )
    # ── Translations: for the agent guide only ───────────────────────────
    #
    # Another tool's call, written as that tool spells it, on that tool's
    # objects. No bngsim object has the name, so no error of bngsim's can
    # meet it; the guide's table is where it is translated.
    + _entries(
        "translation",
        "AMICI",
        (
            "amici.sim.sundials.run_simulation(model, solver)",
            "sim.run(sample_times=times)",
            "AMICI before 1.0: amici.runAmiciSimulation. There is no import or compile step: "
            "bngsim.Model.load(path) reads the SBML, and a Simulator runs it. AMICI integrates "
            "from t0; a BNGsim run starts at the first sample time, so put t0 first.",
        ),
        (
            "model.set_timepoints(times)",
            "sim.run(sample_times=times)",
            "AMICI before 1.0: setTimepoints. The run starts at the first sample time, so put "
            "the start time first.",
        ),
        (
            "model.set_free_parameter_by_name(name, value)",
            "model.set_param(name, value)",
            "AMICI before 1.0: setParameterByName.",
        ),
        (
            "solver.set_sensitivity_order(SensitivityOrder.first)",
            "bngsim.Simulator(model, sensitivity_params=[...])",
            "AMICI before 1.0: setSensitivityOrder. result.sensitivities is "
            "(n_times, n_species, n_params); AMICI's rdata.sx is (n_times, n_params, n_species).",
        ),
        ("rdata.x", "result.species"),
    )
    + _entries(
        "translation",
        "COPASI (basico)",
        ("basico.load_model(path)", "bngsim.Model.load(path)"),
        (
            "basico.run_time_course(duration=d, intervals=n)",
            "sim.run(t_span=(0, d), n_points=n + 1)",
            "intervals counts the steps between output points. basico starts each time course "
            "from the initial values; a BNGsim run continues from the last one unless "
            'start="initial".',
        ),
        (
            'basico.run_time_course(method="stochastic")',
            'bngsim.Simulator(model, method="ssa")',
            _BUILT,
        ),
        ("basico.set_parameters(name, initial_value=v)", "model.set_param(name, v)"),
        ("basico.set_species(name, initial_concentration=c)", "model.set_concentration(name, c)"),
        ("basico.run_steadystate()", "sim.steady_state()"),
    )
    + _entries(
        "translation",
        "BNGL actions",
        (
            'simulate({method=>"ode", t_end=>T, n_steps=>N})',
            'bngsim.Simulator(model, method="ode").run(t_span=(0, T), n_points=N + 1)',
            "bngsim.run_bngl(path) runs the actions a .bngl file holds, as written.",
        ),
        (
            "simulate({..., continue=>1})",
            "sim.run(t_span=(t_end, t_next), n_points=n)",
            "Every run goes on from the model's state; a t_span that starts where the last one "
            "ended continues its clock too.",
        ),
        (
            'parameter_scan({parameter=>"k", par_min=>a, par_max=>b, n_scan_pts=>n, '
            "log_scale=>1, t_end=>T, n_steps=>N})",
            'sim.parameter_scan("k", par_min=a, par_max=b, n_scan_pts=n, log_scale=True, '
            "t_span=(0, T), n_points=N + 1)",
            "It returns one Result per point and writes no files. Without t_span and n_points "
            "it runs (0, 100) at 101 points.",
        ),
        (
            "generate_network({overwrite=>1})",
            "bngsim.Model.from_bngl(path)",
            "It runs BNG2.pl's generate_network and loads the network; a repeated load of an "
            "unchanged file reads a cache.",
        ),
    )
    + _entries(
        "translation",
        "PySB",
        (
            "ScipyOdeSimulator(model, tspan=t).run()",
            "bngsim.Simulator(model).run(sample_times=t)",
            "Export the PySB model first: pysb.export.export(model, 'bngl') returns BNGL text, "
            "which bngsim.Model.from_bngl(path) reads from a file.",
        ),
        (
            "ScipyOdeSimulator(model).run(param_values={...})",
            "model.set_params({name: value, ...})",
            "Set the parameters on the Model, then run. PySB's values hold for that run only; "
            "the Model keeps these.",
        ),
        (
            'BngSimulator(model).run(method="ssa", n_runs=n)',
            'bngsim.Simulator(model, method="ssa").run_replicates(n, t_span=(t0, t1), n_points=m)',
        ),
        ('res.observables["obs"]', 'result.observables["obs"]', "A numpy column."),
    )
)

#: The table's guide-only rows (``owner == "pattern"``).
PATTERNS: tuple[ForeignName, ...] = tuple(e for e in FOREIGN_NAMES if e.owner == "pattern")

#: Names the table must not hold. A descriptor is a real class attribute to
#: ``inspect.getattr_static``, which a ``typing.runtime_checkable`` protocol
#: reads on 3.12 and later, so ``keys`` on Result would make it pass for a
#: mapping, ``shape`` or ``values`` for an array or a frame, and ``copy`` for
#: anything copyable. The names left in the table are bngsim-specific enough
#: that a protocol naming one (``run``, ``plot``) is rare, and the exposure is
#: the price of a hint that costs a successful lookup nothing.
PROTOCOL_NAMES = frozenset(
    {
        "keys", "values", "items", "shape", "dtype", "ndim", "copy", "columns", "index",
        "read", "write", "close", "send", "tolist",
    }
)  # fmt: skip

#: The owners an attribute lookup reports under, and the keyword-argument owner.
ATTRIBUTE_OWNERS = ("bngsim", "Model", "Simulator", "Result")

_BY_OWNER: dict[str, dict[str, ForeignName]] = {}
for _entry in FOREIGN_NAMES:
    _names = _BY_OWNER.setdefault(_entry.owner, {})
    if _entry.name in _names:  # pragma: no cover - caught by the table's own test
        raise RuntimeError(f"FOREIGN_NAMES lists {_entry.owner}.{_entry.name} twice")
    _names[_entry.name] = _entry
del _entry, _names


def lookup(owner: str, name: str) -> ForeignName | None:
    """The entry for ``owner.name``, or ``None``."""
    by_name = _BY_OWNER.get(owner)
    return None if by_name is None else by_name.get(name)


# ─── Attribute lookups ──────────────────────────────────────────────────────
#
# Each name of the table is a data descriptor on its class, which raises the
# AttributeError that names bngsim's call, when the name is read on an instance
# or on the class, or written. A __getattr__ would reach every name, but it is
# not free where the lookup succeeds: CPython does not specialize a property
# read on a class that has one, so each costs some 40 ns more (25 to 43 ns
# measured on 3.10 to 3.13), and on 3.10 and 3.11 a slot read and a method call
# do too; a metaclass __getattr__ costs a class-attribute read 45 to 60 ns, and
# a __setattr__ costs every write some 185 ns, 103 of them per Model.clone().
# A descriptor costs nothing except on its own name. An instance's dir() leaves
# it out, so tab completion and the "Did you mean" Python adds to a typo offer
# no name the class does not have; the API reference skips it too (docs/conf.py).
# dir() of the class itself lists it, with its hint as its docstring, as help()
# does: hiding it there takes a metaclass, which a subclass mixing in another
# (abc.ABC) would then conflict with. Read on the class it raises, as
# getattr(type, "__abstractmethods__") does for a name dir(type) lists, so
# code that walks dir(cls) catches AttributeError, as inspect.getmembers,
# pydoc, Sphinx and unittest.mock do.


class _ForeignAttribute:
    """The descriptor for one :class:`ForeignName` on its class."""

    # No __slots__: an instance's __doc__ is what help() and IPython's ``obj.name?``
    # show of it, and a slot of that name conflicts with the class's docstring.

    def __init__(self, entry: ForeignName) -> None:
        self.entry = entry
        self.__doc__ = f"Not a BNGsim attribute. {entry.hint}"

    def _error(self, what: str) -> AttributeError:
        # Without ``obj``: Python's traceback adds a "Did you mean" of its own to
        # an AttributeError that carries one, and it would sit beside the call
        # this names.
        name = self.entry.name
        return AttributeError(f"{what} has no attribute {name!r}. {self.entry.hint}", name=name)

    def __get__(self, obj: object, cls: type | None = None) -> NoReturn:
        if obj is None:
            assert cls is not None
            raise self._error(f"type object {cls.__name__!r}")
        raise self._error(f"{type(obj).__name__!r} object")

    def __set__(self, obj: object, value: object) -> NoReturn:
        raise self._error(f"{type(obj).__name__!r} object")

    def __delete__(self, obj: object) -> NoReturn:
        raise self._error(f"{type(obj).__name__!r} object")


#: The names :func:`install` put on each class, which ``dir()`` leaves out.
_INSTALLED: dict[type, frozenset[str]] = {}


def install(cls: type, owner: str) -> None:
    """Put *owner*'s table names on *cls*, except one it already has.

    A name the class has is a real attribute, never shadowed. The table should
    not list one, and ``test_foreign_names.py`` fails where it does; ``import
    bngsim`` does not.
    """
    names = []
    for name, entry in _BY_OWNER.get(owner, {}).items():
        if any(name in vars(klass) for klass in cls.__mro__):
            continue
        setattr(cls, name, _ForeignAttribute(entry))
        names.append(name)
    _INSTALLED[cls] = frozenset(names)


def _hidden(cls: type) -> frozenset[str]:
    out: frozenset[str] = frozenset()
    for klass in cls.__mro__:
        out |= _INSTALLED.get(klass, frozenset())
    return out


def public_dir(obj: object) -> list[str]:
    """``dir(obj)`` without the names :func:`install` put on its class."""
    hidden = _hidden(type(obj))
    return [n for n in object.__dir__(obj) if n not in hidden]


def module_attribute_error(module: str, name: str) -> AttributeError:
    """The error ``bngsim.<name>`` raises for a name the module lacks."""
    prefix = f"module {module!r} has no attribute {name!r}"
    entry = None if name[:1] == "_" else lookup("bngsim", name)
    if entry is None:
        return AttributeError(prefix, name=name)
    return AttributeError(f"{prefix}. {entry.hint}", name=name)


# ─── Names that belong to the model ─────────────────────────────────────────


def _bare(name: str) -> str:
    """*name* compared loosely: case folded, a fixed species' ``$`` and a BNGL
    molecule's empty ``()`` dropped, so ``a`` finds ``A()`` and ``$A()``."""
    bare = name.casefold().lstrip("$")
    return bare[:-2] if bare.endswith("()") else bare


#: Past this many candidates a near-match search is the exact-but-loose pass
#: alone: difflib over a network of 10**5 species takes seconds, and this runs
#: on the way to an error.
_FUZZY_LIMIT = 20_000


def near_matches(name: str, candidates: Iterable[str], limit: int = 3) -> list[str]:
    """Up to *limit* of *candidates* close to *name*, closest first.

    A loose match (:func:`_bare`) comes first, then difflib's. Names starting
    with ``_`` are bngsim's own synthesized slots and are offered only for a
    name that starts with one too.
    """
    pool = [c for c in candidates if c != name and (name[:1] == "_" or c[:1] != "_")]
    loose = [c for c in pool if _bare(c) == _bare(name)]
    out = loose[:limit]
    if len(out) < limit and len(pool) <= _FUZZY_LIMIT:
        for c in difflib.get_close_matches(name, pool, n=limit, cutoff=0.6):
            if c not in out:
                out.append(c)
    return out[:limit]


def quote_list(names: Iterable[str]) -> str:
    """``'a', 'b' or 'c'``."""
    quoted = [repr(n) for n in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def value_text(value: object) -> str:
    """*value* as code for a suggested call: itself when it is a short number
    (an integral float as an int, since every count it fills is one), else the
    word ``value``."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = value.item() if hasattr(value, "item") else value
        if isinstance(number, float) and number.is_integer():
            number = int(number)
        text = repr(number)
        if len(text) <= 24:
            return text
    return "value"


def _count_text(value: object) -> str | None:
    """*value* as code for a count (n_points), or ``None`` when it is none."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return None
    number = float(value)
    return str(int(number)) if number.is_integer() else None


def model_source(value: object, wrap: Callable[[str], str] = str) -> tuple[str, str] | None:
    """For an argument where a :class:`~bngsim.Model` belongs: ``(what it is,
    the loader call that reads it)`` when it is a path or model text, else
    ``None``. *wrap* puts each loader call inside the call to suggest."""
    if isinstance(value, os.PathLike):
        return "a path", wrap("bngsim.Model.load(path)")
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return None
    head = value.lstrip()[:200].lower()
    if head.startswith("<") and ("<sbml" in value[:2000].lower() or head.startswith("<?xml")):
        return "SBML text", wrap("bngsim.Model.from_sbml_string(text)")
    low = value[:5000].lower()
    for kind, suffix, marks in _FILE_ONLY_TEXT:
        if any(mark in low for mark in marks):
            return f"{kind} text", (
                f"{wrap('bngsim.Model.load(path)')} on a {suffix} file holding it, since "
                f"BNGsim reads {kind} only from a file"
            )
    if "\n" in value.strip():
        return "model text", (
            f"{wrap('bngsim.Model.from_antimony_string(text)')} for Antimony, or "
            f"{wrap('bngsim.Model.from_sbml_string(text)')} for SBML"
        )
    suffix = os.path.splitext(value.strip())[1].lower()
    looks_like_a_path = suffix in _MODEL_SUFFIXES or os.sep in value or os.path.exists(value)
    return ("a path" if looks_like_a_path else "a string"), wrap("bngsim.Model.load(path)")


_MODEL_SUFFIXES = (".net", ".bngl", ".xml", ".sbml", ".ant")

#: Formats bngsim reads from a file only, by the block headers that mark them:
#: BNGL (rules, handed to BNG2.pl) and BioNetGen's .net (a generated network).
_FILE_ONLY_TEXT = (
    ("BNGL", ".bngl", ("begin model", "begin molecule types", "begin reaction rules")),
    (".net", ".net", ("begin reactions", "begin groups")),
)


# ─── Keyword arguments ──────────────────────────────────────────────────────


def _is_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _short(value: object) -> str:
    """*value* as code when its ``repr`` is short and reads back as the value
    (a numpy array's ``array([...])`` does not), else ``...``."""
    text = repr(value)
    if len(text) > 40:
        return "..."
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return "..."
    return text


def signature_without_catch_alls(fn: Callable[..., Any]) -> inspect.Signature:
    """*fn*'s signature without its ``*args`` and ``**kwargs``.

    :meth:`Simulator.run` takes both only to name the keyword a wrong call
    meant; as ``__signature__`` this is what :func:`inspect.signature`, help()
    and Sphinx show.
    """
    sig = inspect.signature(fn)
    kept = [
        p
        for p in sig.parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    return sig.replace(parameters=kept)


def _is_default(value: object, default: object) -> bool:
    if value is default:
        return True
    try:
        return type(value) is type(default) and bool(value == default)
    except Exception:
        return False


def keyword_entry(method: str, name: str, takes: Container[str]) -> ForeignName | None:
    """The entry for keyword *name* of ``Simulator.<method>``, which takes the
    parameters *takes*: the method's own, or else the ``Simulator.run`` one,
    where the keyword its call names (``t_span=``, ``n_points=``) is one the
    method takes."""
    own = lookup(f"Simulator.{method}", name)
    if own is not None or method == "run":
        return own
    shared = lookup("Simulator.run", name)
    if shared is None:
        return None
    keyword = re.match(r"([a-z_]+)=", shared.use)
    return None if keyword and keyword.group(1) not in takes else shared


def call_error(
    method: str,
    sig: inspect.Signature,
    values: Mapping[str, object],
    extra: tuple,
    unknown: Mapping[str, object],
) -> TypeError:
    """The ``TypeError`` for a call of ``Simulator.<method>`` that passed
    positional arguments *extra* past the ones it takes, or the keywords
    *unknown*.

    *sig* is the method's own signature without its catch-alls, and *values*
    its parameters as the call bound them (its ``locals()`` on entry). Each
    keyword the table has for it (:func:`keyword_entry`) gets its hint, one
    close to a real parameter a "did you mean"; any other is named as Python
    names it. Where :func:`rewrite_call` can write the corrected call, the
    error adds it.
    """
    what = f"Simulator.{method}()"
    params = [p for p in sig.parameters.values() if p.name != "self"]
    positional = [p.name for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    order = [p.name for p in params]
    known = frozenset(order)
    given = {
        p.name: values[p.name]
        for p in params
        if p.name in values and not _is_default(values[p.name], p.default)
    }
    if extra:
        call_args: tuple = (*(values[n] for n in positional), *extra)
        call_kwargs = {k: v for k, v in given.items() if k not in positional}
    else:
        call_args, call_kwargs = (), given
    try:
        corrected = rewrite_call(method, positional, call_args, {**call_kwargs, **unknown}, order)
    except Exception:
        corrected = None
    tail = f" For this call: {corrected}." if corrected else ""
    if extra:
        plural = "" if len(positional) == 1 else "s"
        return TypeError(
            f"{what} takes {len(positional)} positional argument{plural} "
            f"({', '.join(positional)}) but {len(call_args)} were given.{tail}"
        )
    hints = []
    for name in unknown:
        entry = keyword_entry(method, name, known)
        if entry is not None:
            hints.append((name, entry.hint))
            continue
        close = near_matches(name, known, limit=2)
        if close:
            hints.append((name, f"Did you mean {quote_list(close)}?"))
    names = list(unknown)
    if len(names) == 1:
        head = f"{what} got an unexpected keyword argument {names[0]!r}"
        body = f". {hints[0][1]}" if hints else ""
    else:
        head = f"{what} got unexpected keyword arguments {', '.join(map(repr, names))}"
        body = (". " + " ".join(f"{name}: {hint}" for name, hint in hints)) if hints else ""
    return TypeError(f"{head}{body}{tail}")


#: Run keywords that translate to one of the run calls' own one-to-one.
_RUN_START = ("t_start",)
_RUN_END = ("t_end", "end", "tend")
_RUN_STEPS = ("n_steps", "steps")
_RUN_POINTS = ("points",)
_RUN_TIMES = ("times", "t_eval")
_SCAN_VALUES = ("values", "scan_values")


def rewrite_call(
    method: str, positional: list[str], args: tuple, kwargs: dict, order: list[str]
) -> str | None:
    """The call ``Simulator.<method>(*args, **kwargs)`` means, as code, where
    every foreign spelling in it translates one-to-one: RoadRunner's
    ``simulate(0, 10, 11)`` and ``simulate(start=0, end=10, points=11)``,
    BioNetGen's ``t_end=10, n_steps=10``, SciPy's ``t_eval=[...]``.

    *positional* and *order* are the method's positional parameters and all of
    them, in order. ``None`` when something in the call does not translate (a
    count that is not a whole number, a time grid the method takes no
    ``sample_times=`` for), or when two of its arguments say the same thing
    (``t_span=`` and ``t_end=``).
    """
    takes = set(order)
    spans = "t_span" in takes  # all but run_until, which runs on to one time t
    out: dict[str, str] = {}
    start = end = None

    def put(key: str, text: str | None) -> bool:
        if text is None or key in out or key not in takes:
            return False
        out[key] = text
        return True

    if args:
        # RoadRunner's simulate(start, end, points[, selections]) where t_span and
        # n_points stand; selections are dropped, since a Result holds every column.
        if not spans or "t_span" not in positional:
            return None
        lead = positional.index("t_span")
        rest = args[lead:]
        if len(rest) not in (3, 4) or not all(_is_real(a) for a in rest[:3]):
            return None
        if len(rest) == 4:
            fourth = rest[3]
            if "params" in takes and _rows_of_dicts(fourth):
                put("params", _given("params", fourth))  # run_batch's parameter sets
            elif not (_names_list(fourth) and method == "run"):
                return None  # selections, which a Result needs none of, only on run()
        for name, value in zip(positional[:lead], args[:lead], strict=True):
            put(name, _given(name, value))
        start, end = rest[0], rest[1]
        if not put("n_points", _count_text(rest[2])):
            return None
    for name, value in kwargs.items():
        ok = True
        if name in _RUN_START or (name == "start" and _is_real(value)):
            # "start" is RoadRunner's start time; run(start=...) of bngsim's own
            # names a state, and is a string. Two starts (or two ends) are two
            # answers to one question, which a rewrite does not pick between.
            if start is not None:
                return None
            start = value
        elif name in _RUN_END:
            if end is not None:
                return None
            end = value
        elif name in _RUN_STEPS:
            ok = put("n_points", _count_text(value + 1) if _is_real(value) else None)
        elif name in _RUN_POINTS:
            ok = put("n_points", _count_text(value))
        elif name in _RUN_TIMES:
            ok = put("sample_times", _times_text(name, value))
        elif name == "tspan":
            ok = put("t_span", _short(tuple(value)) if isinstance(value, Iterable) else None)
        elif name in _SCAN_VALUES:
            ok = put("par_scan_vals", _given(name, value))
        elif name == "selections":
            continue
        elif name in takes:
            ok = put(name, _given(name, value))
        else:
            return None
        if not ok:
            return None
    if start is not None or end is not None:
        if not spans or "t_span" in out:
            return None
        if any(v is not None and not _is_real(v) for v in (start, end)):
            return None
        span = (
            f"({value_text(start) if start is not None else 0}, "
            f"{value_text(end) if end is not None else 't_end'})"
        )
        out["t_span"] = span
    if not out:
        return None
    rank = {name: i for i, name in enumerate(order)}
    ordered = sorted(out.items(), key=lambda item: rank.get(item[0], len(rank)))
    return f"sim.{method}(" + ", ".join(f"{k}={v}" for k, v in ordered) + ")"


def _rows_of_dicts(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(v, dict) for v in value)
    )


def _names_list(value: object) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)


def _given(name: str, value: object) -> str:
    """A value the call gave, as code for the corrected call: itself when its
    ``repr`` is short, else the keyword's own name (``params=params``), which
    reads as "what you passed" where ``...`` would be passed as Ellipsis."""
    text = _short(value)
    return name if text == "..." else text


def _times_text(name: str, value: object) -> str:
    """``sample_times=`` for times given as *name*: the values when they are a
    short list of numbers, else the caller's own keyword as the value."""
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        items = list(value)
        if all(_is_real(v) for v in items):
            text = _short([value_text(v) if value_text(v) != "value" else v for v in items])
            if text != "...":
                return text.replace("'", "")
    return name
