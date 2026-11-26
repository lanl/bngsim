"""The ``bngsim`` command, also run as ``python -m bngsim`` (issue #1708).

Each subcommand is a thin layer over a public Python call, so the command adds
no behavior of its own:

==================  ==========================================================
command             Python call behind it
==================  ==========================================================
``doctor``          :func:`bngsim.capabilities`, with the Python, the platform
                    and the C compiler codegen would use
``guide``           :func:`bngsim.agent_guide`
``skill``           ``bngsim.agent_guide(skill=True)``
``examples``        :mod:`bngsim.examples`
``inspect MODEL``   ``bngsim.Model.load(MODEL).describe()``
``run MODEL``       ``Simulator(Model.load(MODEL), ...).run(...)``: the
                    run's record and :meth:`~bngsim.Result.digest`, the arrays
                    to ``--out``
``check MODEL``     the run, then :meth:`~bngsim.Result.check`; with
                    ``--sensitivity-params`` or ``--sensitivity-ic``, of the
                    sensitivities too
``run-until``       ``Simulator.run_until(T)`` with ``add_stop_condition``: a
                    condition met is an answer, the result it cut
``run-batch``       ``Simulator.run_batch(params=ROWS)``, digested as a stack
``parameter-scan``  ``Simulator.parameter_scan(NAME, ...)``, digested as a
                    stack
``steady-state``    ``Simulator.steady_state()``; ``steady-state-batch`` and
                    ``steady-states`` the batch and the multistart search, in
                    the forms of :mod:`bngsim._steady_form`
``run-bngl FILE``   a ``.bngl`` file's own actions, as :func:`bngsim.run_bngl`
                    runs them: a digest of each run
``convert TOOL``    the ``bngsim-sbml2net``, ``bngsim-net2sbml``,
                    ``bngsim-sbml2bngl``, ``bngsim-validate-conversion`` and
                    ``bngsim-omex`` commands, unchanged
``cache``           the ``bngsim-cache`` command, unchanged
==================  ==========================================================

The rules for output, which are what let a program use the command:

* With ``--json``, a command prints exactly one JSON object on standard output
  and nothing else goes there. Whatever else writes to standard output while
  the command works, Python or C, is sent to standard error instead. A number
  JSON cannot hold (NaN, an infinity) is written as the string ``"nan"``,
  ``"inf"`` or ``"-inf"``.
* An error prints nothing on standard output. It goes to standard error: with
  ``--json`` as one JSON object, :meth:`bngsim.BngsimError.to_dict` (``type``,
  ``code``, ``message``, ``details``, ``remedies``, ``docs_url``, ``notices``),
  and otherwise
  as a line of text.
* The exit code says what happened, so a caller need not read any text. See
  :data:`EXIT_CODES`.

``convert`` and ``cache`` hand their arguments to the existing commands, which
keep their own output and exit codes.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import math
import os
import shlex
import shutil
import stat
import sys
from collections.abc import Callable, Iterator
from typing import Any, NoReturn

from bngsim._simulator import _NETWORK_FREE_METHODS

# ─── Exit codes ──────────────────────────────────────────────────────────────

EXIT_OK = 0
#: The command was well formed and failed: a model that cannot be read, a
#: missing file or package, or an error with no more specific exit code.
EXIT_ERROR = 1
#: The command line is wrong: an unknown command or option, a missing or
#: invalid argument.
EXIT_USAGE = 2
#: BNGsim refuses the model: it holds a construct BNGsim cannot simulate
#: faithfully. Not a crash: the model, not the call, has to change.
EXIT_REFUSED = 3
#: A run started and failed (the solver gave up).
EXIT_RUN_FAILED = 4
#: A run took longer than its wall-clock limit.
EXIT_TIMEOUT = 5
#: ``bngsim check``: the run finished and its check is not ok (an answer moved,
#: a changed run raised or ran out of time, or the run is not converged).
EXIT_CHECK_FAILED = 6

#: Each exit code and what it means, as the documentation lists them.
EXIT_CODES: dict[int, str] = {
    EXIT_OK: "success",
    EXIT_ERROR: "error: the command was well formed and failed",
    EXIT_USAGE: "usage error: the command line is wrong",
    EXIT_REFUSED: "refusal: the model cannot be simulated faithfully",
    EXIT_RUN_FAILED: "failed run: the solver gave up",
    EXIT_TIMEOUT: "timeout: a run took longer than its limit",
    EXIT_CHECK_FAILED: "check not ok: the run's answer moved, or could not be checked",
}

#: The error codes that are refusals of the model. A SensitivityUnsupportedError
#: or SsaValidationError is a refusal whatever its code.
_REFUSAL_CODES = frozenset(
    {
        "model.unsupported_construct",
        "model.unset_symbol",
        "ssa.unsupported_construct",
        "sens.unsupported",
        "sens.refused_during_run",
        "sens.refused_for_parameters",
    }
)


def exit_code_for(error: BaseException) -> int:
    """The exit code for an error a command raised."""
    from bngsim._exceptions import (
        SensitivityUnsupportedError,
        SimulationError,
        SimulationTimeout,
        SsaValidationError,
        UsageError,
    )

    code = getattr(error, "code", None)
    if isinstance(error, SimulationTimeout) or code == "run.timeout":
        return EXIT_TIMEOUT
    if getattr(error, "_cli_refusal", False):  # see _run_bngl
        return EXIT_REFUSED
    if isinstance(error, SensitivityUnsupportedError | SsaValidationError):
        return EXIT_REFUSED
    if code in _REFUSAL_CODES:
        return EXIT_REFUSED
    if isinstance(error, _CommandLineError | UsageError) or (
        isinstance(code, str) and code.startswith("usage.")
    ):
        return EXIT_USAGE
    if isinstance(error, SimulationError):
        return EXIT_RUN_FAILED
    return EXIT_ERROR


def error_dict(error: BaseException) -> dict[str, Any]:
    """An error as the JSON object a command writes: a BngsimError's
    :meth:`~bngsim.BngsimError.to_dict`, and the same keys for any other."""
    from bngsim._exceptions import UsageError

    if isinstance(error, _CommandLineError):
        # As the UsageError a Python call raises for an argument it cannot take.
        details: dict[str, Any] = {"usage": error.usage} if error.usage else {}
        # argparse names the option a bad value was given to: say which.
        import re

        named = re.match(r"argument (--[\w-]+)", str(error))
        if error.argument is not None:
            details["argument"] = error.argument
        elif named:
            details["argument"] = named.group(1)
        error = UsageError(str(error), details=details or None)
    to_dict = getattr(error, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {
        "type": type(error).__name__,
        "code": "error",
        "message": str(error),
        "details": {},
        "remedies": [],
        "docs_url": None,
        "notices": [],
    }


# ─── The command line ────────────────────────────────────────────────────────


class _CommandLineError(Exception):
    """A command line the command cannot take. Raised in place of argparse's
    exit, so it goes out in the same form as every other error."""

    def __init__(self, message: str, usage: str = "", argument: str | None = None) -> None:
        super().__init__(message)
        self.usage = usage
        self.argument = argument


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _CommandLineError(message, self.format_usage().rstrip())


#: ``bngsim convert TOOL``: each tool and the existing command it runs.
_CONVERT_TOOLS = {
    "sbml2net": ("main", "SBML to a .net network (bngsim-sbml2net)"),
    "net2sbml": ("net2sbml_main", "a .net network to SBML (bngsim-net2sbml)"),
    "sbml2bngl": ("sbml2bngl_main", "SBML to compartmental BNGL (bngsim-sbml2bngl)"),
    "validate": ("validate_main", "check a conversion (bngsim-validate-conversion)"),
    "omex": ("omex_main", "pack or unpack a COMBINE archive (bngsim-omex)"),
}

_EPILOG = (
    "exit codes: "
    + "; ".join(f"{code} {meaning}" for code, meaning in EXIT_CODES.items())
    + ".\nWith --json, standard output carries exactly one JSON object, and an "
    "error is a JSON object on standard error."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="bngsim",
        description="Inspect and run a model, and this install of BNGsim. "
        "Also runs as `python -m bngsim`.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    # A --json after the command is the same flag. SUPPRESS, so a subcommand's
    # default cannot overwrite one given before the command.
    common = _Parser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="print one JSON object"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)

    sub.add_parser(
        "doctor",
        parents=[common],
        help="what this install can do, what is missing, and the fix for each",
        description="What this install can do, what is missing, and the fix for "
        "each: bngsim.capabilities(), with the Python, the platform and the C "
        "compiler codegen would use. Run it first in a new environment, and "
        "paste it into a bug report.",
    )
    guide = sub.add_parser(
        "guide",
        parents=[common],
        help="print the agent guide of this version",
        description="Print the BNGsim agent guide of this version (bngsim.agent_guide()).",
    )
    skill = sub.add_parser(
        "skill",
        parents=[common],
        help="print the agent guide as a SKILL.md file",
        description="Print the agent guide as a SKILL.md file, for agent harnesses that "
        "load skills. For Claude Code, in every project: "
        "bngsim skill --out ~/.claude/skills/bngsim/SKILL.md "
        "(or .claude/skills/bngsim/SKILL.md for one project). Run it again after an "
        "upgrade: the guide is the installed version's.",
    )
    for p, what in ((guide, "the guide"), (skill, "the SKILL.md file")):
        p.add_argument(
            "--out",
            metavar="FILE",
            help=f"write {what} to FILE, as UTF-8, instead of printing it: makes FILE's "
            "folder and replaces any such file. A leading ~ is your home folder, in any shell",
        )
    examples = sub.add_parser(
        "examples",
        parents=[common],
        help="list the example models that ship with BNGsim",
        description="List the example models that ship with BNGsim "
        "(bngsim.examples), with their formats and files.",
    )
    examples.add_argument("name", nargs="?", help="one example, by name")
    inspect = sub.add_parser(
        "inspect",
        parents=[common],
        help="what a model is and which methods it admits",
        description="Load a model and describe it: what it holds, and which "
        "methods it admits and why not (bngsim.Model.load(MODEL).describe()). "
        "Runs nothing. A .bngl file is run through BNG2.pl to generate its network.",
    )
    inspect.add_argument("model", metavar="MODEL", help="a model file (.xml, .sbml, .net, ...)")
    inspect.add_argument(
        "--limit",
        type=_limit,
        default=50,
        help="the longest any list in the description may be (default 50), or 'all'",
    )
    # Listed for --help only: main() hands these to the existing commands
    # before this parser sees them.
    sub.add_parser(
        "convert",
        help="convert a model between formats (bngsim convert TOOL --help)",
        description="tools: "
        + "; ".join(f"{tool}: {what}" for tool, (_, what) in _CONVERT_TOOLS.items()),
    )
    runner = _Parser(add_help=False)
    _run_arguments(runner)
    run = sub.add_parser(
        "run",
        parents=[common, runner],
        help="run a model; print its record and a digest of its values",
        description="Run a model once from its initial state (Simulator.run()) and "
        "print the run's record and a digest of its values (Result.record, "
        "Result.digest()). Arrays never go to the terminal: --out writes them. "
        "Every run has a wall-clock limit (--timeout).",
    )
    del run
    check = sub.add_parser(
        "check",
        parents=[common, runner],
        help="run a model, then check the run (Result.check())",
        description="Run a model as `bngsim run` does, then make the run again with "
        "one setting changed and say whether the answer moved (Result.check()). "
        "Exits 6 when the check is not ok.",
    )
    check.add_argument("--checks", default=None, help=_checks_help())
    check.add_argument(
        "--budget",
        type=_seconds,
        default=_DEFAULT_TIMEOUT,
        help=f"seconds for all the checks' reruns (default {_DEFAULT_TIMEOUT:g})",
    )
    _add_run_calls(sub, common)
    sub.add_parser("cache", help="the codegen artifact cache (bngsim cache --help)")
    _SUBPARSERS.clear()
    _SUBPARSERS.update(sub.choices)
    return parser


def _checks_help() -> str:
    """``--checks``' help, made from :mod:`bngsim._check`'s names, so a check
    added there is listed here without an edit (issue #1925)."""
    from bngsim import _check

    names = ", ".join((*_check.DETERMINISTIC, *_check.STOCHASTIC, *_check.SENSITIVITY))
    return (
        f"'all', or check names separated by commas ({names}); default: the cheap ones "
        f"for the method ({', '.join(_check.DEFAULT_DETERMINISTIC)} for an ODE run, "
        f"{', '.join(_check.DEFAULT_STOCHASTIC)} for a stochastic one), and "
        f"{', '.join(_check.DEFAULT_SENSITIVITY)} too for a run with sensitivities"
    )


#: Each command's parser, by name, as the last parser built holds them: what
#: names a command's usage in a refusal.
_SUBPARSERS: dict[str, argparse.ArgumentParser] = {}


#: The wall-clock limit of a run, and of a check's reruns, where the command
#: line gives none: a command that hangs costs an agent its turn.
_DEFAULT_TIMEOUT = 600.0


def _run_arguments(p: argparse.ArgumentParser) -> None:
    """The options ``run`` and ``check`` share: what to run and how. An option
    the run would not use is refused, not dropped (see :func:`_validate`)."""
    _model_arguments(p)
    _engine_arguments(p)
    _window_arguments(p, sample_times=True)
    _set_argument(p)
    _solver_arguments(p)
    _sensitivity_arguments(p)
    p.add_argument(
        "--steady-state", action="store_true", help="stop the run once it reaches a steady state"
    )
    p.add_argument(
        "--replicates",
        type=_positive_int,
        default=None,
        help="N stochastic runs from N seeds, digested as a stack (run only; N = 1 "
        "is one run); each may take --timeout/N, so all N together keep --timeout",
    )
    _timeout_argument(
        p,
        f"seconds the run may take (default {_DEFAULT_TIMEOUT:g}); exit 5 past it. "
        "Loading the model is not counted.",
    )
    _digest_arguments(p)
    _out_arguments(p)


def _model_arguments(p: argparse.ArgumentParser, *, network: bool = True) -> None:
    p.add_argument("model", metavar="MODEL", help="a model file (.xml, .sbml, .net, .ant, .bngl)")
    if network:
        p.add_argument(
            "--network",
            action="store_true",
            help="for a .bngl file: the network BNG2.pl generates, not the file's "
            "own actions (required for .bngl, refused for any other file)",
        )


def _engine_arguments(p: argparse.ArgumentParser, *, xml: bool = True) -> None:
    p.add_argument(
        "--method",
        default="ode",
        help="ode (default), ssa, psa" + (", nf, rm" if xml else ""),
    )
    if xml:
        p.add_argument("--xml", default=None, help="the BioNetGen XML nf and rm read")
    p.add_argument(
        "--poplevel", type=_positive, default=None, help="psa's population level (psa only)"
    )


def _window_arguments(
    p: argparse.ArgumentParser, *, sample_times: bool = False, each: str = ""
) -> None:
    p.add_argument("--t-start", type=float, default=None, help=f"the start time{each} (default 0)")
    p.add_argument("--t-end", type=float, default=None, help=f"the end time{each}")
    p.add_argument(
        "--n-points",
        type=int,
        default=None,
        help=f"output points{each}, the first included (default 101)",
    )
    if sample_times:
        p.add_argument(
            "--sample-times",
            type=_floats,
            default=None,
            help="increasing output times, separated by commas, in place of --t-start, "
            "--t-end and --n-points",
        )


def _set_argument(p: argparse.ArgumentParser, what: str = "the run") -> None:
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=f"before {what}, set a parameter (Model.set_param), or a species' starting "
        "concentration (Model.set_concentration: its amount over its compartment's "
        "size); repeatable",
    )


def _solver_arguments(p: argparse.ArgumentParser, *, seed: bool = True) -> None:
    if seed:
        p.add_argument("--seed", type=int, default=None, help="the seed of a stochastic run")
    p.add_argument("--rtol", type=_positive, default=None, help="relative tolerance (ode only)")
    p.add_argument(
        "--atol",
        type=_atol,
        default=None,
        help="absolute tolerance (ode only): a number, 'auto' or 'tracking'",
    )


def _sensitivity_arguments(p: argparse.ArgumentParser) -> None:
    """Forward sensitivities (issue #1925): Simulator's keywords of the same
    names. No option of the digest's ``max_parameters``: one named ``--max-...``
    would make ``--max``, which abbreviates ``--max-columns``, ambiguous."""
    p.add_argument(
        "--sensitivity-params",
        type=_names,
        action="append",
        default=None,
        metavar="NAME[,NAME...]",
        help="compute forward sensitivities to these parameters (Simulator's "
        "sensitivity_params; ode only); repeatable",
    )
    p.add_argument(
        "--sensitivity-ic",
        type=_names,
        action="append",
        default=None,
        metavar="SPECIES[,SPECIES...]",
        help="compute forward sensitivities to these species' initial amounts "
        "(Simulator's sensitivity_ic; ode only); repeatable. A comma inside "
        "parentheses, as in R(l,d), is part of the name.",
    )


def _timeout_argument(p: argparse.ArgumentParser, help: str) -> None:
    p.add_argument("--timeout", type=_seconds, default=_DEFAULT_TIMEOUT, help=help)


def _digest_arguments(p: argparse.ArgumentParser, *, species: bool = True) -> None:
    if species:
        p.add_argument("--species", action="store_true", help="digest the species too")
    p.add_argument(
        "--max-columns",
        type=_count,
        default=20,
        help="columns of each kind the digest lists (0 or more)",
    )


def _out_arguments(p: argparse.ArgumentParser, *, text: bool = True, help: str = "") -> None:
    p.add_argument(
        "--out",
        default=None,
        help=help
        or "write the arrays, replacing any such file: .npz (NumPy alone), .h5 or "
        ".hdf5 (needs h5py), .csv or .tsv (one run)",
    )
    if text:
        p.add_argument(
            "--out-kind",
            choices=("observables", "species"),
            default=None,
            help="which columns a .csv or .tsv holds (default observables)",
        )


#: The commands for the run calls beyond run() (issue #1890), in --help's order.
#: Each is named for its Python call, with ``-`` for ``_``, and takes only the
#: options that call can use: argparse refuses any other.
_RUN_CALLS = (
    "run-until",
    "run-batch",
    "parameter-scan",
    "steady-state",
    "steady-state-batch",
    "steady-states",
    "run-bngl",
)

_STACK_OUT = (
    "write the arrays, replacing any such file: .npz (NumPy alone), .h5 or .hdf5 "
    "(needs h5py); one file holds every row"
)
_SS_OUT = "write the arrays to a .npz file (NumPy alone), replacing any such file"


def _add_run_calls(sub: Any, common: argparse.ArgumentParser) -> None:
    until = sub.add_parser(
        "run-until",
        parents=[common],
        help="run a model to a time, cut where a condition holds (Simulator.run_until)",
        description="Run a model from its initial state to --t-end "
        "(Simulator.add_stop_condition, then Simulator.run_until(T)), and print the "
        "run's record and a digest of its values. The run goes to --t-end, and is then "
        "cut at the first output point where a --stop-when condition holds: a "
        "condition met is an answer, not an error, exit 0, with `stop_condition` "
        "naming it. The whole span counts against --timeout.",
    )
    _model_arguments(until)
    _engine_arguments(until, xml=False)
    until.add_argument("--t-end", type=float, default=None, help="the time to run to")
    until.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="output points, the first included (default: one per unit of time, at least 2)",
    )
    until.add_argument(
        "--stop-when",
        action="append",
        default=[],
        metavar="EXPR",
        help="cut the run at the first output point where EXPR holds: a Python expression "
        "of the observables, the species (A() written A__) and t, such as 'A_tot < 10'; "
        "repeatable, checked in order",
    )
    _set_argument(until)
    _solver_arguments(until)
    _timeout_argument(
        until,
        f"seconds the run may take (default {_DEFAULT_TIMEOUT:g}); exit 5 past it. "
        "Loading the model is not counted.",
    )
    _digest_arguments(until)
    _out_arguments(until)

    batch = sub.add_parser(
        "run-batch",
        parents=[common],
        help="run a model once per parameter set; digest the runs as a stack",
        description="Run a model once per parameter set, each from the initial state "
        "(Simulator.run_batch), and print a record and a digest of the runs as a stack: "
        "statistics across the runs at the last time. Rows run in order.",
    )
    _model_arguments(batch)
    _engine_arguments(batch, xml=False)
    _window_arguments(batch, each=" of each run")
    _params_argument(batch)
    _set_argument(batch, "the runs")
    _solver_arguments(batch)
    _timeout_argument(
        batch,
        f"seconds all the runs may take (default {_DEFAULT_TIMEOUT:g}), each its share, "
        "--timeout over the number of rows; exit 5 past it. Loading the model is not counted.",
    )
    _digest_arguments(batch)
    _out_arguments(batch, text=False, help=_STACK_OUT)

    scan = sub.add_parser(
        "parameter-scan",
        parents=[common],
        help="run a model once per value of a parameter; digest the runs as a stack",
        description="Run a model once per value of one parameter (Simulator."
        "parameter_scan), and print a record and a digest of the runs as a stack, "
        "with the values scanned. Give the values (--par-scan-vals) or a range "
        "(--par-min, --par-max, --n-scan-pts, --log-scale).",
    )
    _model_arguments(scan)
    _engine_arguments(scan, xml=False)
    scan.add_argument("--parameter", default=None, help="the parameter to scan (required)")
    scan.add_argument(
        "--par-scan-vals",
        type=_floats,
        default=None,
        help="the values, separated by commas, in place of a range",
    )
    scan.add_argument("--par-min", type=float, default=None, help="the range's first value")
    scan.add_argument("--par-max", type=float, default=None, help="the range's last value")
    scan.add_argument("--n-scan-pts", type=int, default=None, help="the range's number of values")
    scan.add_argument(
        "--log-scale", action="store_true", help="space the range's values geometrically"
    )
    scan.add_argument(
        "--reset-conc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start each value from the state at the scan's start (default), or with "
        "--no-reset-conc from where the last one ended",
    )
    _window_arguments(scan, each=" of each run")
    _set_argument(scan, "the scan")
    _solver_arguments(scan)
    _timeout_argument(
        scan,
        f"seconds all the runs may take (default {_DEFAULT_TIMEOUT:g}), each its share, "
        "--timeout over the number of values; exit 5 past it. Loading the model is not "
        "counted.",
    )
    _digest_arguments(scan)
    _out_arguments(scan, text=False, help=_STACK_OUT)

    ss = sub.add_parser(
        "steady-state",
        parents=[common],
        help="find a model's steady state (Simulator.steady_state)",
        description="Find the steady state a model's ODEs reach from its initial state "
        "(Simulator.steady_state()), and print the solver's verdict and the values "
        "there. Exit 0 whether or not it converged: `converged` says.",
    )
    _steady_arguments(ss)
    _timeout_argument(ss, _deadline_help("the solve"))
    _digest_arguments(ss)
    _out_arguments(ss, text=False, help=_SS_OUT)

    ssb = sub.add_parser(
        "steady-state-batch",
        parents=[common],
        help="find a steady state per parameter set (Simulator.steady_state_batch)",
        description="Find a steady state per parameter set (Simulator."
        "steady_state_batch(params)), and print how many converged and statistics "
        "of the values across those that did.",
    )
    _steady_arguments(ssb, params=True)
    ssb.add_argument(
        "--n-workers", type=_positive_int, default=None, help="threads to solve rows in"
    )
    _timeout_argument(ssb, _deadline_help("all the solves"))
    _digest_arguments(ssb)
    _out_arguments(ssb, text=False, help=_SS_OUT)

    sss = sub.add_parser(
        "steady-states",
        parents=[common],
        help="search a model for its distinct steady states (Simulator.steady_states)",
        description="Search a model for its distinct steady states from many starts "
        "(Simulator.steady_states()), and print each root's stability, how many "
        "starts reached it, and the species that tell the roots apart.",
    )
    _model_arguments(sss)
    _set_argument(sss, "the search")
    sss.add_argument("--n-starts", type=_positive_int, default=None, help="starts (default 200)")
    sss.add_argument(
        "--seed", type=_count, default=None, help="the seed of the starts, 0 or more (default 0)"
    )
    sss.add_argument(
        "--upper",
        type=_positive,
        default=None,
        help="the bound starts draw a species in no conservation law below (default: ten "
        "times the largest concentration the model's state and its steady state hold)",
    )
    sss.add_argument(
        "--decades",
        type=_positive,
        default=None,
        help="how many decades below its bound a start draws a species from (default 4)",
    )
    sss.add_argument(
        "--dedupe-rtol",
        type=_positive,
        default=None,
        help="how close two roots are to be one (default 1e-6)",
    )
    sss.add_argument("--tol", type=_positive, default=None, help="each solve's tolerance")
    sss.add_argument(
        "--n-workers", type=_positive_int, default=None, help="threads to solve starts in"
    )
    _timeout_argument(sss, _deadline_help("the search"))
    _digest_arguments(sss, species=False)
    _out_arguments(sss, text=False, help=_SS_OUT)

    bngl = sub.add_parser(
        "run-bngl",
        parents=[common],
        help="run a .bngl file's own actions (bngsim.run_bngl); digest each run",
        description="Run a .bngl file's own actions as BNG2.pl runs them "
        "(bngsim.run_bngl), and print a digest of each run they make. BNG2.pl "
        "generates the network first, under its own limit.",
    )
    bngl.add_argument("model", metavar="MODEL", help="a .bngl file")
    _solver_arguments(bngl)
    bngl.add_argument(
        "--allow-lossy",
        action="store_true",
        help="run an action BNGsim holds only approximately, with a warning, "
        "instead of refusing the file",
    )
    _timeout_argument(
        bngl,
        f"seconds the actions may take (default {_DEFAULT_TIMEOUT:g}); exit 5 past it. "
        "Generating the network is not counted.",
    )
    _digest_arguments(bngl)
    bngl.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="a directory, made if it is missing, for each run's arrays: N.npz, or "
        "N-SUFFIX.npz for a run with a suffix (N counts the runs from 0)",
    )


def _deadline_help(what: str) -> str:
    return (
        f"seconds {what} may take (default {_DEFAULT_TIMEOUT:g}); exit 5 past it, with the "
        "process stopped there. Loading the model is not counted."
    )


def _params_argument(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--params",
        default=None,
        metavar="ROWS",
        help="the parameter sets (required): JSON, a list of {name: value} objects, or "
        "a .json, .csv or .tsv file of them (a header of names, a row per set)",
    )


def _steady_arguments(p: argparse.ArgumentParser, *, params: bool = False) -> None:
    """The options ``steady_state`` and ``steady_state_batch`` share."""
    _model_arguments(p)
    if params:
        _params_argument(p)
    _set_argument(p, "the solve")
    p.add_argument(
        "--tol", type=_positive, default=None, help="the residual tolerance (default 1e-9)"
    )
    p.add_argument(
        "--max-time", type=_positive, default=None, help="the longest march (default 1e6)"
    )
    p.add_argument(
        "--method",
        dest="ss_method",
        choices=("integration", "newton"),
        default=None,
        help="the solver: integration (default) or newton",
    )
    p.add_argument(
        "--criterion",
        choices=("scaled", "residual"),
        default=None,
        help="what converged asks: scaled (default) or residual",
    )
    p.add_argument("--rtol", type=_positive, default=None, help="the march's relative tolerance")
    p.add_argument(
        "--atol",
        type=_atol,
        default=None,
        help="the march's absolute tolerance: a number, 'auto' or 'tracking'",
    )
    p.add_argument(
        "--max-steps",
        type=_positive_int,
        default=None,
        help="the most CVODE steps per output point of the march",
    )


def _floats(text: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected numbers separated by commas, not {text!r}"
        ) from None
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def _names(text: str) -> list[str]:
    """Names separated by commas. A comma inside parentheses is part of a
    name: a BioNetGen species such as ``R(l,d)`` is one name."""
    names: list[str] = []
    depth, start = 0, 0
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                break
        elif c == "," and depth == 0:
            names.append(text[start:i].strip())
            start = i + 1
    if depth != 0:
        raise argparse.ArgumentTypeError(f"unbalanced parentheses in {text!r}")
    names.append(text[start:].strip())
    if not all(names):
        raise argparse.ArgumentTypeError(
            f"expected names separated by commas, with none empty, not {text!r}"
        )
    return names


def _positive(text: str, what: str = "a number") -> float:
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if not value > 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected {what} > 0, not {text!r}")
    return value


def _count(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a whole number >= 0, not {text!r}")
    return value


def _seconds(text: str) -> float:
    return _positive(text, "a number of seconds")


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number >= 1, not {text!r}")
    return value


def _atol(text: str) -> float | str:
    if text in ("auto", "tracking"):
        return text
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if not value > 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            f"expected a number > 0, 'auto' or 'tracking', not {text!r}"
        )
    return value


def _limit(text: str) -> int | None:
    if text == "all":
        return None
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a whole number or 'all', not {text!r}")
    return value


# ─── The commands ────────────────────────────────────────────────────────────
#
# Each returns (the JSON payload, its text form).


def _doctor(args: argparse.Namespace) -> tuple[Any, str]:
    import platform

    import bngsim
    from bngsim import _codegen_toolchain

    report: dict[str, Any] = dict(bngsim.capabilities())
    report["compiler"] = _compiler_report(_codegen_toolchain)
    compiler, problem = report["compiler"]["command"], report["compiler"]["problem"]
    report["python"] = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }
    report["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    report["package_path"] = os.path.dirname(os.path.abspath(bngsim.__file__))

    build = report["build"]
    lines = [
        f"bngsim {report['version']} (build {build['commit'] or 'unknown'}"
        + (", STALE: the compiled extension is older than its source" if build["stale"] else "")
        + ")",
        f"  at {report['package_path']}",
        f"Python {report['python']['version']} ({report['python']['executable']})",
        f"{report['platform']['system']} {report['platform']['release']} "
        f"{report['platform']['machine']}",
        "C compiler for codegen: "
        + (" ".join(compiler) if compiler else "none")
        + {True: ", builds a test library", False: "", None: ""}[report["compiler"]["builds"]]
        + (f"\n  problem: {problem}" if problem else ""),
        "",
        "Available: " + ", ".join(k for k, v in report["features"].items() if v),
    ]
    if report["missing"]:
        lines.append("Missing, and the fix for each:")
        for name, why in report["missing"].items():
            lines.append(f"  {name}: {why}")
    else:
        lines.append("Missing: nothing")
    return report, "\n".join(lines) + "\n"


def _compiler_report(toolchain: Any) -> dict[str, Any]:
    """The C compiler codegen would use, and whether it builds: a trivial C file
    built with the command every codegen library is built with. ``builds`` is
    ``None`` where that proves nothing (no compiler, or the probe timed out)."""
    compiler, problem = toolchain.peek()
    builds: bool | None = None
    if compiler is not None and problem is None:
        ok, detail = toolchain._probe(compiler)
        builds = ok
        if ok is False:
            problem = toolchain._broken_message(compiler, detail)
        elif ok is None:
            problem = f"the compiler could not be tried: {detail}"
    return {"command": compiler, "builds": builds, "problem": problem}


def _guide(args: argparse.Namespace) -> tuple[Any, str]:
    import bngsim

    skill = args.command == "skill"
    text = bngsim.agent_guide(skill=skill)
    payload: dict[str, Any] = {
        "version": bngsim.__version__,
        "format": "skill" if skill else "markdown",
        "text": text,
        "out": None,
    }
    if args.out is None:
        return payload, text
    path = _write_text(args.out, text, "SKILL.md" if skill else "agent_guide.md")
    # The text is in the file: the payload says where, and does not carry it too.
    payload.update(text=None, out=path)
    return payload, f"{'skill' if skill else 'guide'} written to {path}\n"


def _write_text(out: str, text: str, name: str) -> str:
    """``text`` written to the file ``out``, as the UTF-8 bytes the command
    prints, its folder made first; the file's absolute path. ``~`` is expanded
    here, not by the shell: PowerShell passes it to the program as it is. Not
    through a shell redirect: PowerShell before 7.4 reads a program's output in
    the console's code page, and Windows PowerShell 5.1 writes it again as
    UTF-16.

    A path that names no file this can make is a usage error naming ``--out``:
    a folder (or a path ending in a separator, which ``abspath`` would drop), an
    empty string, ``-``, a ``~`` that names no home folder, and a file the
    system will not let it make, or one that is read-only. A failure while
    writing (a full disk) is not: that raises, and leaves the file as it was."""
    seps = "".join(s for s in (os.sep, os.altsep) if s)
    example = os.path.join(out, name)

    def refuse(why: str, *, suggest: bool = True) -> NoReturn:
        hint = f"; give the file's path, such as {example}" if suggest else ""
        _refuse(f"--out {out or repr(out)}: {why}{hint}", "--out", value=out)

    if out == "":
        refuse("an empty path names no file", suggest=False)
    if out == "-":
        refuse(
            "this writes a file named '-'; leave out --out to print to standard output",
            suggest=False,
        )
    if out.endswith(tuple(seps)):
        refuse("a path ending in a separator is a folder")
    expanded = os.path.expanduser(out)
    if expanded.startswith("~"):
        head = expanded
        for sep in seps:
            head = head.split(sep)[0]
        refuse(
            f"{head} is not a home folder on this machine; for a file whose name "
            f"starts with ~, write {os.path.join(os.curdir, out)}",
            suggest=False,
        )
    path = os.path.abspath(expanded)
    if os.path.isdir(path):
        refuse("that is a folder")
    # Through a link, to the file it names: the link stays a link.
    real = os.path.realpath(path)
    if os.path.exists(real):
        if not os.path.isfile(real):
            refuse(f"cannot write {path}: it is not a regular file", suggest=False)
        # By its mode too: root may write a file whose mode says read-only.
        if not os.access(real, os.W_OK) or not os.stat(real).st_mode & stat.S_IWUSR:
            refuse(f"cannot write {path}: it is read-only", suggest=False)
    folder = os.path.dirname(real)
    tmp = os.path.join(folder, f".{os.path.basename(real)}.{os.getpid()}-{os.urandom(4).hex()}")

    def cannot(error: OSError) -> NoReturn:
        # The path the system names, when not the file's: a parent that is a
        # file or a link to nothing, or a folder it may not write (the folder,
        # never the temporary name).
        where = folder if error.filename == tmp else error.filename
        if where and os.path.abspath(where) not in (path, real):
            if os.path.islink(where) and not os.path.exists(where):
                reason = f"{where} is a link to nothing"
            elif os.path.isfile(where):
                reason = f"{where} is a file"
            else:
                reason = f"{where}: {error.strerror}"
        else:
            reason = error.strerror or str(error)
        refuse(f"cannot write {path}: {reason}", suggest=False)

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = open(tmp, "xb")  # noqa: SIM115  (closed below, after the write)
    except OSError as error:
        cannot(error)
    # Written beside the file and moved over it, so a write that fails part way
    # (a full disk) leaves the file as it was, or no file, never half of one.
    try:
        with f:
            f.write(text.encode("utf-8"))
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    try:
        if os.path.exists(real):
            shutil.copymode(real, tmp)
        # On Windows this fails while another program holds the file open.
        os.replace(tmp, real)
    except OSError as error:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        refuse(f"cannot write {path}: {error.strerror or error}", suggest=False)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    return path


def _example_entry(name: str) -> dict[str, Any]:
    from bngsim import examples

    formats = examples.formats(name)
    return {
        "name": name,
        "description": examples.describe(name),
        "formats": formats,
        "paths": {f: str(examples.path(name, f)) for f in formats},
    }


def _examples(args: argparse.Namespace) -> tuple[Any, str]:
    from bngsim import examples
    from bngsim._exceptions import UsageError

    if args.name is not None:
        if args.name not in examples.names():
            raise UsageError(
                f"no example named {args.name!r}; the examples are {', '.join(examples.names())}",
                details={"argument": "name", "value": args.name, "expected": examples.names()},
            )
        entry = _example_entry(args.name)
        lines = [entry["name"], "", entry["description"], "", "Files:"]
        lines += [f"  {f:9} {p}" for f, p in entry["paths"].items()]
        return entry, "\n".join(lines) + "\n"
    entries = [_example_entry(name) for name in examples.names()]
    lines = []
    for e in entries:
        first = e["description"].splitlines()[0] if e["description"] else ""
        lines.append(f"{e['name']:12} {first}")
        lines.append(f"{'':12} formats: {', '.join(e['formats'])}")
    lines.append("")
    lines.append("`bngsim examples NAME` gives one example's files.")
    return {"examples": entries}, "\n".join(lines) + "\n"


def _inspect(args: argparse.Namespace) -> tuple[Any, str]:
    from bngsim import Model

    description = Model.load(args.model).describe(limit=args.limit)
    return dict(description), str(description) + "\n"


#: The methods each method-bound option is for, by the names --method takes:
#: every network-free name Simulator takes, nf_reject and nf_exact included.
_NETWORK_FREE = _NETWORK_FREE_METHODS
_STOCHASTIC_METHODS = frozenset({"ssa", "psa"}) | _NETWORK_FREE
_OUT_SUFFIXES = (".npz", ".h5", ".hdf5", ".csv", ".tsv")


def _refuse(message: str, argument: str, **details: Any) -> NoReturn:
    from bngsim._exceptions import UsageError

    raise UsageError(message, details={"argument": argument, **details})


def _parse_sets(items: list[str]) -> list[tuple[str, float]]:
    sets = []
    for item in items:
        name, sep, value = item.partition("=")
        name = name.strip()
        try:
            number = float(value)
        except ValueError:
            number = math.nan
        if not sep or not name or not math.isfinite(number):
            _refuse(
                f"--set takes NAME=VALUE with a finite number, not {item!r}", "--set", value=item
            )
        sets.append((name, number))
    return sets


def _validate(args: argparse.Namespace) -> None:
    """Refuse, before anything is loaded or run, a command line whose options
    the call would not use or could not honour. A command takes only the
    options its call can use, so this asks only of those it has."""
    has = vars(args).__contains__
    command = args.command
    method = str(args.method).lower() if has("method") else None
    if has("t_end"):
        sample_times = getattr(args, "sample_times", None)
        if sample_times is None and args.t_end is None:
            _refuse(
                "give --t-end (with --n-points) or --sample-times"
                if has("sample_times")
                else "give --t-end",
                "--t-end",
            )
        if sample_times is not None:
            for flag, value in (
                ("--t-end", args.t_end),
                ("--t-start", args.t_start),
                ("--n-points", args.n_points),
            ):
                if value is not None:
                    _refuse(f"--sample-times sets the times; it takes no {flag}", flag)
            times = sample_times
            if any(b <= a for a, b in zip(times, times[1:], strict=False)):
                _refuse("--sample-times must increase", "--sample-times", value=times)
    if getattr(args, "n_points", None) is not None and args.n_points < 2:
        _refuse("--n-points takes 2 or more", "--n-points")
    sensitive = has("sensitivity_params") and _validate_sensitivities(args, method)
    if method is not None:
        if method not in _STOCHASTIC_METHODS and args.seed is not None:
            _refuse(f"--seed is a stochastic run's; a {method} run takes none", "--seed")
        if method in _STOCHASTIC_METHODS:
            for flag, value in (("--rtol", args.rtol), ("--atol", args.atol)):
                if value is not None:
                    _refuse(f"{flag} is an ODE run's; a {method} run takes none", flag)
        if getattr(args, "xml", None) is not None and method not in _NETWORK_FREE:
            _refuse(f"--xml is for the network-free methods nf and rm, not {method}", "--xml")
        if args.poplevel is not None and method != "psa":
            _refuse(f"--poplevel is psa's, not {method}'s", "--poplevel")
        if method == "psa" and args.poplevel is None:
            _refuse("--method psa needs --poplevel N", "--poplevel")
        if method in _NETWORK_FREE and not has("xml"):
            # run_until, run_batch and parameter_scan run on the model's own
            # network (ode, ssa, psa); a network-free engine runs from its XML.
            # One unavailable here (nf_fixed) gets Simulator's own refusal, not
            # a pointer to a `bngsim run` that refuses it too.
            from bngsim._simulator import normalize_method

            normalize_method(method)
            call = command.replace("-", "_")
            _refuse(
                f"{call} runs ode, ssa or psa, not the network-free {method}; "
                f"`bngsim run --method {method} --xml FILE` makes a network-free run",
                "--method",
                value=args.method,
            )
    is_bngl = str(args.model).lower().endswith(".bngl")
    if has("network"):
        if is_bngl and not args.network:
            from bngsim._exceptions import UsageError

            raise UsageError(
                "a .bngl file can mean its generated network or the file's own "
                f"actions; `bngsim {command}` reads the network: pass --network. The "
                "file's own actions are `bngsim run-bngl FILE`, bngsim.run_bngl(path) in "
                "Python.",
                details={"argument": "--network"},
                remedies=[{"kind": "pass_argument", "argument": "--network"}],
            )
        if args.network and not is_bngl:
            _refuse("--network is for a .bngl file", "--network")
    if command == "run-bngl":
        if not is_bngl:
            _refuse(
                f"`bngsim run-bngl` runs a .bngl file's actions, not {args.model!r}; "
                "`bngsim run` runs any model",
                "model",
                value=args.model,
            )
        if isinstance(args.atol, str):
            _refuse("run_bngl takes a number for --atol, not 'auto' or 'tracking'", "--atol")
    if command == "check" and args.replicates is not None:
        _refuse(
            "`bngsim check` checks one run; --replicates is for `bngsim run` (the "
            "replicates check reruns a stochastic run at other seeds itself)",
            "--replicates",
        )
    if getattr(args, "replicates", None) is not None:
        if method not in _STOCHASTIC_METHODS:
            _refuse(f"--replicates is for a stochastic method, not {method}", "--replicates")
        for flag, value in (
            ("--sample-times", args.sample_times),
            ("--steady-state", args.steady_state or None),
        ):
            if value is not None:
                _refuse(f"--replicates runs run_replicates(), which takes no {flag}", flag)
    if command == "parameter-scan":
        if not args.parameter:
            _refuse("give --parameter NAME, the parameter to scan", "--parameter")
        ranged = (
            ("--par-min", args.par_min),
            ("--par-max", args.par_max),
            ("--n-scan-pts", args.n_scan_pts),
            ("--log-scale", args.log_scale or None),
        )
        if args.par_scan_vals is not None:
            for flag, value in ranged:
                if value is not None:
                    _refuse(f"--par-scan-vals gives the values; it takes no {flag}", flag)
        else:
            for flag, value in ranged[:3]:
                if value is None:
                    _refuse(
                        "give the values (--par-scan-vals) or a range: --par-min, "
                        f"--par-max and --n-scan-pts; {flag} is missing",
                        flag,
                    )
    if has("params"):
        _parse_params(args.params)
    if has("stop_when"):
        for expression in args.stop_when:
            if not expression.strip():
                _refuse("--stop-when takes an expression", "--stop-when", value=expression)
    _validate_out(args)
    if sensitive and args.out is not None and args.out.lower().endswith((".csv", ".tsv")):
        _refuse(
            "a .csv or .tsv holds one kind of column and no sensitivities; write a run "
            "with sensitivities to .npz or .h5",
            "--out",
            value=args.out,
        )
    if has("set"):
        _parse_sets(args.set)


def _validate_sensitivities(args: argparse.Namespace, method: str | None) -> bool:
    """Flatten ``--sensitivity-params`` and ``--sensitivity-ic`` into the lists
    Simulator takes, and refuse what a run could not use. Whether the run
    computes sensitivities."""
    for flag in ("--sensitivity-params", "--sensitivity-ic"):
        attr = flag[2:].replace("-", "_")
        given = getattr(args, attr)
        if given is None:
            continue
        names = [n for item in given for n in (item if isinstance(item, list) else [item])]
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            _refuse(f"{flag} names {', '.join(twice)} more than once", flag, value=names)
        if method in _STOCHASTIC_METHODS:
            _refuse(f"{flag} is an ODE run's; a {method} run takes none", flag)
        setattr(args, attr, names)
    return bool(args.sensitivity_params or args.sensitivity_ic)


def _validate_out(args: argparse.Namespace) -> None:
    out = args.out
    command = args.command
    if command == "run-bngl":
        if out is not None:
            if os.path.exists(out) and not os.path.isdir(out):
                _refuse(
                    f"--out {out!r} is a file; run-bngl writes a directory", "--out", value=out
                )
            parent = os.path.dirname(os.path.abspath(out))
            if not os.path.isdir(parent):
                _refuse(
                    f"--out {out!r}: the directory {parent!r} does not exist", "--out", value=out
                )
        return
    suffixes: tuple[str, ...] = _OUT_SUFFIXES
    if command in ("run-batch", "parameter-scan"):
        suffixes = _STACK_SUFFIXES
    elif command in ("steady-state", "steady-state-batch", "steady-states"):
        suffixes = (".npz",)
    text_out = out is not None and out.lower().endswith((".csv", ".tsv"))
    if getattr(args, "out_kind", None) is not None and not text_out:
        _refuse("--out-kind is for a .csv or .tsv --out", "--out-kind")
    if out is None:
        return
    if not out.lower().endswith(suffixes):
        if text_out and suffixes == _STACK_SUFFIXES:
            _refuse(
                "a .csv or .tsv holds one run; write a stack of runs to .npz or .h5",
                "--out",
                value=out,
            )
        _refuse(
            f"--out takes a file ending {', '.join(suffixes)}, not {out!r}",
            "--out",
            value=out,
        )
    if os.path.isdir(out):
        _refuse(f"--out {out!r} is a directory", "--out", value=out)
    parent = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(parent):
        _refuse(f"--out {out!r}: the directory {parent!r} does not exist", "--out", value=out)
    if text_out and getattr(args, "replicates", None) is not None:
        _refuse(
            "a .csv or .tsv holds one run; write replicates to .npz or .h5", "--out", value=out
        )
    if out.lower().endswith((".h5", ".hdf5")):
        try:
            import h5py  # noqa: F401
        except ImportError:
            from bngsim._error_codes import install
            from bngsim._exceptions import DependencyError

            raise DependencyError(
                "h5py is required for --out .h5. Install with: pip install bngsim[hdf5], "
                "or write .npz",
                remedies=[install("h5py", extra="hdf5")],
            ) from None


_STACK_SUFFIXES = (".npz", ".h5", ".hdf5")


def _once(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object, refused where it names a key twice (``json`` keeps the
    last value without a word)."""
    keys = [k for k, _ in pairs]
    twice = sorted({k for k in keys if keys.count(k) > 1})
    if twice:
        raise ValueError(f"{', '.join(twice)} is given more than once in one row")
    return dict(pairs)


def _parse_params(text: str | None) -> list[dict[str, float]]:
    """``--params``: JSON text, or a .json, .csv or .tsv file, of parameter sets."""
    if text is None:
        _refuse("give --params, the parameter sets", "--params")
    source = text.strip()
    rows: Any
    if source.startswith(("[", "{")):
        try:
            rows = json.loads(source, object_pairs_hook=_once)
        except ValueError as error:
            _refuse(f"--params is not JSON: {error}", "--params")
    else:
        lower = source.lower()
        if not lower.endswith((".json", ".csv", ".tsv")):
            _refuse(
                "--params takes JSON (a list of {name: value} objects) or a .json, .csv "
                f"or .tsv file, not {text!r}",
                "--params",
                value=text,
            )
        if not os.path.isfile(source):
            _refuse(f"--params: no file {source!r}", "--params", value=text)
        # utf-8-sig: a spreadsheet's CSV may start with a byte-order mark.
        with open(source, encoding="utf-8-sig", newline="") as f:
            if lower.endswith(".json"):
                try:
                    rows = json.load(f, object_pairs_hook=_once)
                except ValueError as error:
                    _refuse(f"--params {source!r} is not JSON: {error}", "--params")
            else:
                import csv

                reader = csv.reader(f, delimiter="\t" if lower.endswith(".tsv") else ",")
                table = [row for row in reader if any(cell.strip() for cell in row)]
                if len(table) < 2:
                    _refuse(
                        f"--params {source!r} needs a header of names and a row per set",
                        "--params",
                    )
                header = [name.strip() for name in table[0]]
                twice = sorted({n for n in header if header.count(n) > 1})
                if twice:
                    _refuse(
                        f"--params {source!r} names {', '.join(twice)} more than once",
                        "--params",
                    )
                rows = []
                for n, cells in enumerate(table[1:], start=2):
                    if len(cells) != len(header):
                        _refuse(
                            f"--params {source!r}, line {n}: {len(cells)} values for "
                            f"{len(header)} names",
                            "--params",
                        )
                    rows.append(dict(zip(header, cells, strict=True)))
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        _refuse("--params takes a non-empty list of {name: value} objects", "--params")
    out: list[dict[str, float]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not row:
            _refuse(f"--params row {i} is not a {{name: value}} object", "--params")
        clean: dict[str, float] = {}
        for name, value in row.items():
            try:
                # true and false are JSON's own, not numbers: refused.
                number = math.nan if isinstance(value, bool) else float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not isinstance(name, str) or not name.strip() or not math.isfinite(number):
                _refuse(
                    f"--params row {i}: {name!r} takes a finite number, not {value!r}",
                    "--params",
                )
            clean[name.strip()] = number
        out.append(clean)
    return out


def _load_for_run(
    args: argparse.Namespace,
    *,
    save: bool = True,
    rows: list[dict[str, float]] | None = None,
    scanned: str | None = None,
) -> tuple[Any, dict[str, float]]:
    """The model a command runs, with the ``--set`` values, and those values. A
    species set this way becomes the model's initial state
    (``save_concentrations``), so the run starts from its initial state; with
    ``save=False`` it is the model's state alone, where a call that starts from
    that state (``parameter_scan``) starts."""
    from bngsim import Model
    from bngsim._exceptions import BngsimError, UsageError

    model = Model.load(args.model)
    # Before any --set: what a batch's rows would start from without one.
    probe = model.clone() if rows is not None or scanned is not None else None
    done: dict[str, float] = {}
    species = False
    for name, number in _parse_sets(args.set):
        try:
            if name in model.species_names:
                model.set_concentration(name, number)
                species = True
            else:
                model.set_param(name, number)
        except (BngsimError, ValueError) as error:
            raise UsageError(
                f"--set {name}={number:g}: {error}",
                details={
                    "argument": "--set",
                    "value": f"{name}={number:g}",
                    **dict(getattr(error, "details", None) or {}),
                },
                remedies=list(getattr(error, "remedies", None) or []),
            ) from error
        done[name] = number
    if species and probe is not None:
        # A scan's values each write the parameter, and a species --set holds
        # where it was set: a value that seeds it would not move it. Probed at
        # a value its own is not, so a scan of values equal to it is asked too:
        # --set holds the species away from them all the same.
        if rows is None and scanned is not None and scanned in probe.param_names:
            own = float(probe.get_param(scanned))
            rows = [{scanned: 2.0 * own + 1.0 if math.isfinite(own) else 1.0}]
        if rows:
            _refuse_lost_doses(probe, rows, args, done, held_only=scanned is not None)
    if species and save:
        model.save_concentrations()
    return model, done


def _refuse_lost_doses(
    probe: Any,
    rows: list[dict[str, float]],
    args: argparse.Namespace,
    done: dict[str, float],
    *,
    held_only: bool = False,
) -> None:
    """Refuse a species ``--set`` beside batch rows whose parameters seed a
    species' starting value. The rows start from the model's initial state,
    which the species ``--set`` has to become (``save_concentrations``), and
    that holds every species at the value saved: a row's dose would be dropped.
    Not saving drops the ``--set`` instead. Neither is the batch asked for.

    ``held_only``: a scan, which starts from the state the ``--set`` gives
    without saving it, so only a species the ``--set`` itself holds loses its
    dose."""
    import numpy as np

    from bngsim._exceptions import BngsimError, UsageError

    for name, number in done.items():
        if name not in probe.species_names:
            probe.set_param(name, number)
    start = np.asarray(probe.get_state(), dtype=float)
    names = list(probe.species_names)
    for i, row in enumerate(rows):
        row_model = probe.clone()
        try:
            row_model.set_params(row)
        except (BngsimError, ValueError):
            continue  # a name the call refuses, it refuses itself
        moved = np.asarray(row_model.get_state(), dtype=float) != start
        set_species = [n for n in done if n in names]
        if held_only:
            moved &= np.isin(np.asarray(names, dtype=object), set_species)
        if moved.any():
            dosed = [names[j] for j in np.flatnonzero(moved)]
            call = args.command.replace("-", "_")
            if held_only:
                raise UsageError(
                    f"--set {', '.join(dosed)} with --parameter {args.parameter}: "
                    f"{args.parameter} sets the start of {', '.join(dosed)}, which --set "
                    "holds where it was set, so every value would start it there and the "
                    "scan would not dose it. Leave it out of --set.",
                    details={"argument": "--set", "species": dosed},
                )
            raise UsageError(
                f"--set {', '.join(set_species)} with --params: {call} starts every row "
                "from the model's initial state, so a species --set has to become that "
                f"state, and row {i}'s {', '.join(sorted(row))} could then not move "
                f"{', '.join(dosed)}, whose start it sets. Leave the species out of --set, "
                "or run that row with `bngsim run`.",
                details={"argument": "--set", "row": i, "species": dosed},
            )


def _do_run(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """The run ``run`` and ``check`` make: the Result, and what the payload
    adds to its record (the values set, a stack's seeds)."""
    from bngsim import Result, Simulator

    model, done = _load_for_run(args)
    extra: dict[str, Any] = {"set": done}
    sim_kwargs: dict[str, Any] = {"method": args.method}
    if args.xml is not None:
        sim_kwargs["xml_path"] = args.xml
    if args.poplevel is not None:
        sim_kwargs["poplevel"] = args.poplevel
    for name in ("sensitivity_params", "sensitivity_ic"):
        if getattr(args, name):
            sim_kwargs[name] = list(getattr(args, name))
    with _sensitivity_names_from(args):
        sim = Simulator(model, **sim_kwargs)
    kw: dict[str, Any] = {"timeout": args.timeout}
    if args.sample_times is not None:
        kw["sample_times"] = args.sample_times
    else:
        t_start = 0.0 if args.t_start is None else args.t_start
        kw["t_span"] = (t_start, args.t_end)
        kw["n_points"] = 101 if args.n_points is None else args.n_points
    for name in ("seed", "rtol", "atol"):
        if getattr(args, name) is not None:
            kw[name] = getattr(args, name)
    if args.steady_state:
        kw["steady_state"] = True
    if args.replicates is not None:
        # run_replicates bounds each replicate by its timeout: share the
        # command's limit among them, so all N together keep it.
        kw["timeout"] = args.timeout / args.replicates
        rows = sim.run_replicates(args.replicates, **kw)
        assert isinstance(rows, list)
        extra["seeds"] = [r.seed for r in rows]
        return Result.squeeze(rows), extra
    return sim.run(**kw), extra


@contextlib.contextmanager
def _sensitivity_names_from(args: argparse.Namespace) -> Iterator[None]:
    """A name in ``--sensitivity-params`` or ``--sensitivity-ic`` the model does
    not have, as the usage error ``--set`` makes of one: Simulator's own
    refusal, with its close matches, naming the option."""
    from bngsim._error_codes import pass_argument
    from bngsim._exceptions import BngsimError, UsageError

    try:
        yield
    except BngsimError as error:
        details = dict(error.details or {})
        name = details.get("name")
        for flag, attr, code in (
            ("--sensitivity-params", "sensitivity_params", "param.not_found"),
            ("--sensitivity-ic", "sensitivity_ic", "species.not_found"),
        ):
            if error.code != code or name not in (getattr(args, attr) or ()):
                continue
            message = str(error).removeprefix(f"{attr}: ")
            message = message.replace(f"the names {attr} takes", f"the names {flag} takes")
            remedies = list(error.remedies or [])
            for remedy in remedies:
                keyword = remedy.get("argument")
                if remedy.get("kind") != "pass_argument" or keyword not in (
                    "sensitivity_params",
                    "sensitivity_ic",
                ):
                    continue
                # Simulator names its other keyword; the command, its option.
                (target,) = remedy["value"]
                other = "--" + keyword.replace("_", "-")
                message = message.replace(
                    f"{keyword}=[{target!r}]", f"{other} {shlex.quote(target)}"
                )
                remedies = [pass_argument(other, target)]
                break
            raise UsageError(
                f"{flag} {name}: {message}",
                details={**details, "argument": flag, "value": name},
                remedies=remedies,
            ) from error
        raise


def _write_out(result: Any, args: argparse.Namespace) -> str | None:
    if args.out is None:
        return None
    lower = args.out.lower()
    if lower.endswith((".csv", ".tsv")):
        result.to_csv(
            args.out,
            kind=args.out_kind or "observables",
            sep="\t" if lower.endswith(".tsv") else ",",
        )
    else:
        result.save(args.out)
    return os.path.abspath(args.out)


def _run_payload(
    result: Any, args: argparse.Namespace, extra: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    from bngsim import _record

    sensitive = bool(getattr(args, "sensitivity_params", None)) or bool(
        getattr(args, "sensitivity_ic", None)
    )
    digest = result.digest(
        species=args.species, max_columns=args.max_columns, sensitivities=sensitive
    )
    out = _write_out(result, args)
    payload = {
        "model": os.path.abspath(args.model),
        "record": _record.record(result, start_state=False),
        "digest": dict(digest),
        "out": out,
        **extra,
    }
    lines = [f"{os.path.basename(args.model)}: {digest}"]
    if extra.get("set"):
        lines.append("set: " + ", ".join(f"{k}={v:g}" for k, v in extra["set"].items()))
    if extra.get("seeds"):
        lines.append("seeds: " + ", ".join(str(s) for s in extra["seeds"]))
    if out:
        lines.append(f"arrays written to {out}")
    return payload, lines


def _run(args: argparse.Namespace) -> tuple[Any, str]:
    _validate(args)
    result, extra = _do_run(args)
    payload, lines = _run_payload(result, args, extra)
    return payload, "\n".join(lines) + "\n"


class _CheckNotOk(Exception):
    """``bngsim check`` finished and its check is not ok: the payload is
    written, and the exit code says so."""

    def __init__(self, payload: Any, text: str) -> None:
        super().__init__("the check is not ok")
        self.payload = payload
        self.text = text


def _check(args: argparse.Namespace) -> tuple[Any, str]:
    checks: Any = args.checks
    if checks is not None and checks != "all":
        checks = [c.strip() for c in checks.split(",") if c.strip()]
        if not checks:
            _refuse("--checks takes 'all' or names separated by commas", "--checks")
    _validate(args)
    result, extra = _do_run(args)
    payload, lines = _run_payload(result, args, extra)
    report = result.check(checks, budget=args.budget)
    payload["check"] = dict(report)
    lines.append(str(report))
    text = "\n".join(lines) + "\n"
    if not report["ok"]:
        raise _CheckNotOk(payload, text)
    return payload, text


# ─── The run calls beyond run() (issue #1890) ───────────────────────────────


def _simulator(args: argparse.Namespace, model: Any) -> Any:
    from bngsim import Simulator

    kw: dict[str, Any] = {"method": args.method}
    if getattr(args, "xml", None) is not None:
        kw["xml_path"] = args.xml
    if args.poplevel is not None:
        kw["poplevel"] = args.poplevel
    return Simulator(model, **kw)


def _given(args: argparse.Namespace, *names: str) -> dict[str, Any]:
    """The keywords of the call the command line gives a value, by name: an
    option left out leaves the call's own default."""
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def _span(args: argparse.Namespace) -> tuple[tuple[float, float], int]:
    t_start = 0.0 if args.t_start is None else args.t_start
    return (t_start, args.t_end), (101 if args.n_points is None else args.n_points)


def _stochastic(args: argparse.Namespace) -> bool:
    return str(args.method).lower() in _STOCHASTIC_METHODS


def _run_until(args: argparse.Namespace) -> tuple[Any, str]:
    from bngsim._exceptions import StopConditionMet, UsageError

    _validate(args)
    model, done = _load_for_run(args)
    sim = _simulator(args, model)
    for expression in args.stop_when:
        try:
            sim.add_stop_condition(expression)
        except UsageError as error:
            raise UsageError(
                f"--stop-when {expression!r}: {error}",
                details={
                    **dict(getattr(error, "details", None) or {}),
                    "argument": "--stop-when",
                    "value": expression,
                },
                remedies=list(getattr(error, "remedies", None) or []),
            ) from error
    stopped = None
    with _deadline(args.timeout, "run_until"):
        try:
            result = sim.run_until(args.t_end, **_given(args, "n_points", "seed", "rtol", "atol"))
        except StopConditionMet as met:
            # A condition met is the answer asked for, with the run it cut.
            result, stopped = met.result, met.condition
    payload, lines = _run_payload(result, args, {"set": done, "stop_condition": stopped})
    if stopped is not None:
        lines.insert(1, f"stopped: {stopped} held at t={result.time[-1]:g}")
    return payload, "\n".join(lines) + "\n"


@contextlib.contextmanager
def _names_from(argument: str) -> Iterator[None]:
    """A name the model does not have, given in *argument*, as the usage error
    ``--set`` makes of one: the call's own refusal, with its close matches."""
    from bngsim._exceptions import BngsimError, UsageError

    try:
        yield
    except BngsimError as error:
        if error.code != "param.not_found":
            raise
        raise UsageError(
            f"{argument}: {error}",
            details={**dict(error.details or {}), "argument": argument},
            remedies=list(error.remedies or []),
        ) from error


def _stack(rows: list[Any], args: argparse.Namespace, extra: dict[str, Any]) -> tuple[Any, str]:
    """A batch's or a scan's rows, as one stack, as ``run --replicates`` prints them."""
    from bngsim import Result

    if _stochastic(args):
        extra["seeds"] = [r.seed for r in rows]
    payload, lines = _run_payload(Result.squeeze(rows), args, extra)
    if "parameter" in extra:
        values = ", ".join(_g(v) for v in extra["values"])
        lines.insert(1, f"{extra['parameter']} scanned at {values}")
    return payload, "\n".join(lines) + "\n"


def _run_batch(args: argparse.Namespace) -> tuple[Any, str]:
    _validate(args)
    rows = _parse_params(args.params)
    model, done = _load_for_run(args, rows=rows)
    sim = _simulator(args, model)
    t_span, n_points = _span(args)
    # run_batch bounds each row by its timeout: share the command's limit
    # among them, so all the rows together keep it, as --replicates does.
    with _names_from("--params"):
        results = sim.run_batch(
            t_span,
            n_points,
            params=rows,
            timeout=args.timeout / len(rows),
            **_given(args, "seed", "rtol", "atol"),
        )
    return _stack(list(results), args, {"set": done})


def _parameter_scan(args: argparse.Namespace) -> tuple[Any, str]:
    _validate(args)
    # Each value starts from the state at the scan's start, which a species
    # --set gives; saved as the initial state, a value of a parameter that
    # seeds another species would not dose it.
    model, done = _load_for_run(args, save=False, scanned=args.parameter)
    sim = _simulator(args, model)
    t_span, n_points = _span(args)
    if args.par_scan_vals is not None:
        n_values = len(args.par_scan_vals)
    else:
        n_values = max(int(args.n_scan_pts or 1), 1)
    kw = _given(args, "par_min", "par_max", "n_scan_pts", "seed", "rtol", "atol")
    if args.log_scale:
        kw["log_scale"] = True
    with _names_from("--parameter"):
        rows = sim.parameter_scan(
            args.parameter,
            args.par_scan_vals,
            t_span=t_span,
            n_points=n_points,
            reset_conc=args.reset_conc,
            timeout=args.timeout / n_values,
            **kw,
        )
    values = [r.custom_attrs.get("scan_value") for r in rows]
    return _stack(list(rows), args, {"set": done, "parameter": args.parameter, "values": values})


def _steady_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kw = _given(args, "tol", "max_time", "criterion", "rtol", "atol", "max_steps")
    if args.ss_method is not None:
        kw["method"] = args.ss_method
    return kw


def _steady_payload(
    args: argparse.Namespace,
    model: Any,
    call: str,
    kwargs: dict[str, Any],
    form: dict[str, Any],
    arrays: dict[str, Any],
    done: dict[str, float],
) -> dict[str, Any]:
    from bngsim import _record

    info, values = _record.parameter_values(model)
    out = None
    if args.out is not None:
        import numpy as np

        with open(args.out, "wb") as f:
            np.savez(f, **arrays)
        out = os.path.abspath(args.out)
    return {
        "model": os.path.abspath(args.model),
        "call": {"name": call, "kwargs": kwargs},
        "parameters": {n: float(v) for n, v in zip(info.param_names, values, strict=True)},
        "result": form,
        "out": out,
        "set": done,
    }


def _steady_text(args: argparse.Namespace, form: dict[str, Any], head: str) -> list[str]:
    lines = [f"{os.path.basename(args.model)}: {head}"]
    for kind in form.get("kinds", ["species"]):
        block = form[kind]
        line = f"{kind}: {len(block['columns'])} of {block['n_columns']}"
        if block["n_left_out"]:
            line += f" ({block['n_left_out']} left out)"
        lines.append(line)
        for col in block["columns"]:
            if "value" in col:
                lines.append(f"  {col['name']} {_g(col['value'])}")
            elif "values" in col:
                lines.append(f"  {col['name']} " + ", ".join(_g(v) for v in col["values"]))
            else:
                lines.append(
                    f"  {col['name']} over {col['n_finite']} rows: mean {_g(col['mean'])}, "
                    f"min {_g(col['min'])}, median {_g(col['median'])}, max {_g(col['max'])}"
                )
    return lines


def _g(x: Any) -> str:
    return "?" if x is None else f"{x:.6g}"


def _steady_state(args: argparse.Namespace) -> tuple[Any, str]:
    from bngsim import Simulator, _steady_form

    _validate(args)
    model, done = _load_for_run(args)
    kw = _steady_kwargs(args)
    sim = Simulator(model)
    with _deadline(args.timeout, "steady_state"):
        ss = sim.steady_state(**kw)
    form = _steady_form.steady_state(ss, species=args.species, max_columns=args.max_columns)
    payload = _steady_payload(
        args, model, "steady_state", kw, form, _steady_form.steady_state_arrays(ss), done
    )
    verdict = "converged" if form["converged"] else "NOT CONVERGED"
    head = (
        f"steady state {verdict} by {form['method_used']} ({form['criterion']}), "
        f"residual {_g(form['residual'])}"
        + (f", root {form['root_stability']}" if form["root_stability"] else "")
    )
    lines = _steady_text(args, form, head)
    if payload["out"]:
        lines.append(f"arrays written to {payload['out']}")
    return payload, "\n".join(lines) + "\n"


def _steady_state_batch(args: argparse.Namespace) -> tuple[Any, str]:
    from bngsim import Simulator, _steady_form

    _validate(args)
    rows = _parse_params(args.params)
    model, done = _load_for_run(args, rows=rows)
    kw = _steady_kwargs(args)
    if args.n_workers is not None:
        kw["n_workers"] = args.n_workers
    sim = Simulator(model)
    with _deadline(args.timeout, "steady_state_batch"), _names_from("--params"):
        found = sim.steady_state_batch(rows, **kw)
    found = list(found)
    form = _steady_form.steady_state_batch(
        found, species=args.species, max_columns=args.max_columns
    )
    payload = _steady_payload(
        args,
        model,
        "steady_state_batch",
        kw,
        form,
        _steady_form.steady_state_batch_arrays(found, rows),
        done,
    )
    head = f"{form['n_converged']} of {form['n_rows']} rows converged"
    if form["n_unconverged"]:
        shown = ", ".join(str(i) for i in form["unconverged_rows"])
        more = form["n_unconverged"] - len(form["unconverged_rows"])
        head += f"; not converged: rows {shown}" + (f" and {more} more" if more > 0 else "")
    lines = _steady_text(args, form, head)
    if payload["out"]:
        lines.append(f"arrays written to {payload['out']}")
    return payload, "\n".join(lines) + "\n"


def _steady_states(args: argparse.Namespace) -> tuple[Any, str]:
    from bngsim import Simulator, _steady_form

    _validate(args)
    model, done = _load_for_run(args)
    kw = _given(args, "n_starts", "seed", "upper", "decades", "dedupe_rtol", "tol", "n_workers")
    sim = Simulator(model)
    with _deadline(args.timeout, "steady_states"):
        found = sim.steady_states(**kw)
    form = _steady_form.steady_states(found, max_columns=args.max_columns)
    payload = _steady_payload(
        args, model, "steady_states", kw, form, _steady_form.steady_states_arrays(found), done
    )
    kinds = ", ".join(f"{n} {s}" for s, n in form["stability"].items()) or "none"
    head = (
        f"{form['n_roots']} distinct steady states from {form['n_starts']} starts "
        f"({form['n_failed']} failed): {kinds}"
    )
    lines = _steady_text(args, form, head)
    for k, root in enumerate(form["roots"]):
        lines.insert(
            1 + k,
            f"root {k}: {root['stability']}, reached by {root['multiplicity']}, "
            f"residual {_g(root['residual'])}",
        )
    if payload["out"]:
        lines.append(f"arrays written to {payload['out']}")
    return payload, "\n".join(lines) + "\n"


def _run_bngl(args: argparse.Namespace) -> tuple[Any, str]:
    import warnings

    from bngsim import Model, Result, run_protocol
    from bngsim._exceptions import ConversionError, ConversionWarning

    _validate(args)
    with warnings.catch_warnings():
        # run_protocol says what it cannot run, once; the parse would say it
        # too (as bngsim.run_bngl has it).
        warnings.simplefilter("ignore", ConversionWarning)
        model, protocol = Model.from_bngl(args.model, protocol=True)
    from bngsim._protocol_run import _check_runnable, _method_of
    from bngsim.convert._protocol import Experiment

    try:
        # What run_protocol refuses before anything runs: an action BNGsim
        # cannot run as BNG2.pl would. A refusal of the file (exit 3), where an
        # error the actions meet while they run (a species the model lacks)
        # is an error of the file's (exit 1).
        _check_runnable(protocol, allow_lossy=args.allow_lossy)
    except ConversionError as error:
        error._cli_refusal = True  # type: ignore[attr-defined]
        raise
    steps = protocol.steps[protocol.generation_step or 0 :]
    experiments = [step for step in steps if isinstance(step, Experiment)]
    methods = {_method_of(step) for step in experiments}
    # A stochastic run that names its own seed takes it, not --seed.
    seeded = [s for s in experiments if _method_of(s) in ("ssa", "psa") and "seed" not in s.extra]
    if args.seed is not None and not seeded:
        _refuse(
            "--seed seeds the file's stochastic runs that name no seed of their own, and "
            f"it has none: its runs are {', '.join(sorted(methods)) or 'none'}",
            "--seed",
        )
    for flag, value in (("--rtol", args.rtol), ("--atol", args.atol)):
        if value is not None and "ode" not in methods:
            _refuse(
                f"{flag} is the file's ODE runs', and it has none: "
                f"its runs are {', '.join(sorted(methods)) or 'none'}",
                flag,
            )
    kw = _given(args, "seed", "rtol", "atol")
    with _deadline(args.timeout, "run_protocol"):
        ran = run_protocol(model, protocol, allow_lossy=args.allow_lossy, **kw)
    out_dir = None
    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)
        out_dir = os.path.abspath(args.out)
    digest_kw = {"species": args.species, "max_columns": args.max_columns}
    runs, lines = [], []
    for i, run in enumerate(ran):
        entry: dict[str, Any] = {
            "index": i,
            "kind": run.kind,
            "label": run.label,
            "method": run.method,
        }
        name = f"{i}" + (f"-{_file_word(run.label)}" if run.label else "")
        parts: list[tuple[str, Any, tuple[float, ...]]] = []
        if run.result is not None:
            parts.append(("", run.result, ()))
        if run.scan_results:
            parts.append(("", Result.squeeze(list(run.scan_results)), run.scan_values))
        if run.backward_results:
            parts.append(
                ("backward", Result.squeeze(list(run.backward_results)), run.backward_values)
            )
        for side, result, values in parts:
            block: dict[str, Any] = {"digest": dict(result.digest(**digest_kw))}
            if values:
                block["values"] = [float(v) for v in values]
            if out_dir is not None:
                path = os.path.join(out_dir, name + (f"-{side}" if side else "") + ".npz")
                result.save(path)
                block["out"] = path
            if side:
                entry[side] = block
            else:
                entry.update(block)
            lines.append(
                f"run {i}"
                + (f" ({run.label})" if run.label else "")
                + f", {run.kind}{' ' + side if side else ''}: {result.digest(**digest_kw)}"
            )
        runs.append(entry)
    payload = {"model": os.path.abspath(args.model), "runs": runs, "out": out_dir}
    if not runs:
        lines.append(f"{os.path.basename(args.model)}: the file's actions make no run")
    if out_dir:
        lines.append(f"arrays written to {out_dir}")
    return payload, "\n".join(lines) + "\n"


def _file_word(text: str) -> str:
    import re

    return re.sub(r"[^\w.-]", "_", text)


#: Every command, in the order --help lists them.
_ALL_COMMANDS = (
    "doctor",
    "guide",
    "skill",
    "examples",
    "inspect",
    "run",
    "check",
    *_RUN_CALLS,
    "convert",
    "cache",
)

_COMMANDS: dict[str, Callable[[argparse.Namespace], tuple[Any, str]]] = {
    "doctor": _doctor,
    "guide": _guide,
    "skill": _guide,
    "examples": _examples,
    "inspect": _inspect,
    "run": _run,
    "check": _check,
    "run-until": _run_until,
    "run-batch": _run_batch,
    "parameter-scan": _parameter_scan,
    "steady-state": _steady_state,
    "steady-state-batch": _steady_state_batch,
    "steady-states": _steady_states,
    "run-bngl": _run_bngl,
}


# ─── Output ──────────────────────────────────────────────────────────────────


def _json_safe(value: Any) -> Any:
    """``value`` as JSON can hold it: a float JSON has no number for as a
    string, NumPy scalars and arrays as Python values, a set as a list, and
    anything else JSON has no form for (a ``Path``) as its ``str()``."""
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set | frozenset):
        return [_json_safe(v) for v in sorted(value, key=str)]
    tolist = getattr(value, "tolist", None)  # a NumPy scalar or array
    if callable(tolist) and type(value).__module__ == "numpy":
        return _json_safe(tolist())
    if value is None or isinstance(value, str | int):
        return value
    return str(value)  # a Path, or anything else JSON has no form for


def _dumps(payload: Any) -> str:
    return json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[Callable[[str], None]]:
    """Send everything written to standard output, by Python or by C, to
    standard error, and yield the function that writes to the real standard
    output. That is how a command keeps its own output alone there."""
    sys.stdout.flush()
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        fd = None  # a replaced sys.stdout, as under pytest's capsys
    saved = None
    if fd is not None:
        try:
            sys.stderr.flush()
            saved = os.dup(fd)
            os.dup2(sys.stderr.fileno(), fd)
        except (AttributeError, OSError, ValueError):
            if saved is not None:
                os.close(saved)
            saved = None
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    pending: list[str] = []
    try:
        yield pending.append
    finally:
        # Everything written so far goes out while fd 1 is still standard
        # error: what Python holds for sys.stdout and for the stream it
        # replaced (a handler that kept the original), and what C's stdio holds.
        sys.stdout.flush()
        sys.stdout = real_stdout
        with contextlib.suppress(OSError, ValueError):
            real_stdout.flush()
        _flush_c_stdio()
        if saved is not None:
            os.dup2(saved, fd)  # type: ignore[arg-type]
            os.close(saved)
    _write_stdout(real_stdout, "".join(pending))


#: The most of a failed command's captured output its error JSON keeps: the end.
_MAX_LOG = 20_000

#: While ``--json`` holds standard error: the descriptor of the real standard
#: error, and of the file holding what is written there. A deadline that
#: expires reads both (see :func:`_deadline`).
_HELD: list[tuple[int, int]] = []

#: Whether the command running writes JSON: what a deadline that expires
#: writes its error as.
_AS_JSON: list[bool] = [False]


@contextlib.contextmanager
def _deadline(seconds: float, call: str) -> Iterator[None]:
    """Bound a call that takes no ``timeout`` of its own by *seconds* of wall
    clock: the steady-state calls, ``run_until`` and ``run_protocol``.

    Compiled code cannot be stopped from another thread, so past the limit a
    watchdog thread reports a :class:`~bngsim.SimulationTimeout` as the
    command reports any error (exit 5, with what standard error held), and ends
    the process there, the call still running. The call itself runs where it
    would without a limit, on the main thread."""
    import threading
    import time

    lock = threading.Lock()
    done = [False]
    start = time.monotonic()

    def expire() -> None:
        with lock:
            if done[0]:
                return
            _expire(call, seconds, time.monotonic() - start)

    # A limit past what a timer can wait (about 292 years) is no limit.
    watchdog = threading.Timer(min(seconds, threading.TIMEOUT_MAX), expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        yield
    finally:
        with lock:  # once the call returns, the watchdog can no longer fire
            done[0] = True
        watchdog.cancel()


def _expire(call: str, seconds: float, elapsed: float) -> None:
    """Report that *call* ran out of time, and end the process. Called on the
    watchdog's thread, with the call still running on the main thread."""
    from bngsim._exceptions import SimulationTimeout

    error = SimulationTimeout(
        f"{call} took longer than {seconds:g} s, the command's wall-clock limit "
        "(--timeout), and was stopped there",
        timeout=seconds,
        elapsed=elapsed,
    )
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    _flush_c_stdio()
    fd, log = 2, ""
    if _HELD:
        fd, held = _HELD[-1]
        log = _read_all(held)
    try:
        _write_fd(fd, _error_text(error, _AS_JSON[0], log))
    finally:
        os._exit(EXIT_TIMEOUT)


def _read_all(fd: int) -> str:
    chunks = []
    with contextlib.suppress(OSError):
        os.lseek(fd, 0, os.SEEK_SET)
        while chunk := os.read(fd, 1 << 16):
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", "replace")


def _write_fd(fd: int, text: str) -> None:
    data = text.encode("utf-8")
    with contextlib.suppress(OSError):
        while data:
            data = data[os.write(fd, data) :]


@contextlib.contextmanager
def _stderr_captured(enabled: bool) -> Iterator[list[str]]:
    """With ``--json``, hold everything written to standard error while the
    command works (Python's warnings and log, and what compiled code such as
    SUNDIALS writes itself), so a failure's error is the only thing there: the
    held text goes into it (``log``), or out to standard error on success.
    Yields the list the text is put in when the block ends."""
    box: list[str] = []
    if not enabled:
        yield box
        return
    import tempfile

    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield box  # a replaced sys.stderr, as under pytest's capsys
        return
    sys.stderr.flush()
    with tempfile.TemporaryFile() as held:
        saved = os.dup(fd)
        os.dup2(held.fileno(), fd)
        dying = False
        _HELD.append((saved, held.fileno()))
        try:
            yield box
        except BaseException as stop:
            # An interrupt or an exit, not an error the command reports: what
            # was held goes out to standard error before it propagates.
            dying = not isinstance(stop, Exception)
            raise
        finally:
            with contextlib.suppress(OSError, ValueError):
                sys.stderr.flush()
            _flush_c_stdio()
            _HELD.pop()
            os.dup2(saved, fd)
            os.close(saved)
            held.seek(0)
            box.append(held.read().decode("utf-8", "replace"))
            if dying:
                _write(sys.stderr, box[-1])


class _ReaderGone(Exception):
    """Standard output's reader went away before the command's output was
    written (``bngsim guide | head``)."""


def _write_stdout(stream: Any, text: str) -> None:
    """The command's own output, to standard output. Only a broken pipe here
    is the reader going away; one inside a command is an error of the command."""
    try:
        _write(stream, text)
    except BrokenPipeError as error:
        raise _ReaderGone from error
    except OSError as error:
        # Windows reports a pipe whose reader has closed as EINVAL.
        if sys.platform == "win32" and error.errno == errno.EINVAL:
            raise _ReaderGone from error
        raise


def _flush_c_stdio() -> None:
    """Flush every C stdio stream, so what C code has buffered for standard
    output is written now, not at exit after fd 1 is the real one again."""
    import ctypes

    try:
        if sys.platform == "win32":
            libc = ctypes.CDLL("ucrtbase")
        else:
            libc = ctypes.CDLL(None)
        libc.fflush(None)
    except (OSError, AttributeError):
        pass


def _write(stream: Any, text: str) -> None:
    if not text:
        return
    # As UTF-8 bytes whatever the console's encoding, so a redirect is UTF-8 on
    # Windows too and no character stops the print half way.
    out = getattr(stream, "buffer", None)
    if out is not None:
        out.write(text.encode("utf-8"))
        out.flush()
    else:
        stream.write(text)
        stream.flush()


def _report_error(error: BaseException, as_json: bool, log: str = "") -> int:
    _write(sys.stderr, _error_text(error, as_json, log))
    return exit_code_for(error)


def _error_text(error: BaseException, as_json: bool, log: str = "") -> str:
    """What a command writes to standard error for *error*: with ``as_json``
    one JSON object, holding *log* (what the command wrote there before it
    failed); otherwise *log*, then a line of text."""
    if as_json:
        try:
            d = error_dict(error)
        except Exception as failure:  # the report must not fail on the error it reports
            d = error_dict(RuntimeError(f"{error} (and to_dict() failed: {failure})"))
        if log:
            # What the command wrote to standard error (and output) before it
            # failed, SUNDIALS' own lines among it: inside the one JSON object.
            d["log"] = log[-_MAX_LOG:]
        return _dumps(d)
    parts = [log]
    if isinstance(error, _CommandLineError):
        usage = f"{error.usage}\n" if error.usage else ""
        parts.append(f"{usage}bngsim: error: {error}\n")
        return "".join(parts)
    from bngsim._exceptions import BngsimError

    if not isinstance(error, BngsimError):
        # Not an error BNGsim raises on purpose, so probably a bug in it: the
        # traceback is what a report needs.
        import traceback

        parts.append("".join(traceback.format_exception(error)))
    d = error_dict(error)
    lines = [f"bngsim: error [{d['code']}]: {d['message']}"]
    for remedy in d.get("remedies") or []:
        lines.append(f"  remedy: {json.dumps(_json_safe(remedy))}")
    parts.append("\n".join(lines) + "\n")
    return "".join(parts)


_CONVERT_USAGE = "usage: bngsim convert TOOL [ARGS ...]"


def _passthrough(command: str, rest: list[str]) -> int:
    if command == "cache":
        from bngsim.cache import main as cache_main

        return _exit_status(cache_main, rest, "bngsim cache")
    if not rest or rest[0] not in _CONVERT_TOOLS:
        tools = ", ".join(_CONVERT_TOOLS)
        if rest and rest[0] in ("-h", "--help"):
            _write_stdout(sys.stdout, f"{_CONVERT_USAGE}\n\ntools: {tools}\n")
            return EXIT_OK
        raise _CommandLineError(f"TOOL is one of {tools}", _CONVERT_USAGE)
    from bngsim.convert import _cli as convert_cli

    entry = getattr(convert_cli, _CONVERT_TOOLS[rest[0]][0])
    return _exit_status(entry, rest[1:], f"bngsim convert {rest[0]}")


def _exit_status(entry: Callable[..., int], argv: list[str], prog: str) -> int:
    """Run an existing command, its help and usage naming it as ``prog``."""
    try:
        return int(entry(argv, prog=prog) or 0)
    except SystemExit as stop:  # argparse's --help and usage errors
        code = stop.code
        return code if isinstance(code, int) else (0 if code is None else EXIT_ERROR)


def _command_word(argv: list[str]) -> int | None:
    """The index of the command in ``argv``: the first word not an option."""
    for i, word in enumerate(argv):
        if not word.startswith("-"):
            return i
    return None


def main(argv: list[str] | None = None) -> int:
    """``bngsim`` and ``python -m bngsim``. Returns the exit code, except where
    a call with no time limit of its own (``run-until``, the steady-state
    commands, ``run-bngl``) runs past ``--timeout``: the process then ends
    there with exit 5 (``os._exit``, so no ``atexit`` handler runs), whoever
    called this, since the call cannot be stopped in its thread."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args_list
    if sys.stdout is None or sys.stderr is None:
        # Started with standard output or error closed (`bngsim doctor >&-`):
        # there is nowhere to write the answer. Say so where that is possible.
        if sys.stderr is not None:
            _write(sys.stderr, "bngsim: error: standard output is closed\n")
        return EXIT_ERROR
    try:
        i = _command_word(args_list)
        if (
            i is not None
            and args_list[i] in ("convert", "cache")
            and not {"-h", "--help"} & set(args_list[:i])
        ):
            if i > 0:
                raise _CommandLineError(
                    f"`bngsim {args_list[i]}` takes no options before `{args_list[i]}`; "
                    "give them after it",
                    _CONVERT_USAGE
                    if args_list[i] == "convert"
                    else "usage: bngsim cache [ARGS ...]",
                )
            return _passthrough(args_list[i], args_list[i + 1 :])
        if i is not None and "_" in args_list[i]:
            spelled = args_list[i].replace("_", "-")
            if spelled in _ALL_COMMANDS:
                # The Python call's name: the command is named for it, with - for _.
                raise _CommandLineError(
                    f"there is no command {args_list[i]!r}; the command for "
                    f"{args_list[i]}() is `bngsim {spelled}`",
                    "usage: bngsim [--version] [--json] COMMAND ...",
                )
        parser = _build_parser()
        try:
            args, extra = parser.parse_known_args(args_list)
        except SystemExit as stop:  # --help
            return stop.code if isinstance(stop.code, int) else EXIT_OK
        if extra:
            # An option this command's call would not use: refused, and named.
            command = getattr(args, "command", None)
            subparser = _SUBPARSERS.get(command) if command else None
            what = f"`bngsim {command}`" if command else "bngsim"
            named = [word for word in extra if word.startswith("-")] or extra
            raise _CommandLineError(
                f"{what} takes no {', '.join(named)}"
                + (f"; `bngsim {command} --help` lists what it takes" if command else ""),
                (subparser or parser).format_usage().rstrip(),
                extra[0] if extra[0].startswith("-") else None,
            )
        as_json = bool(getattr(args, "json", False))
        _AS_JSON[0] = as_json
        if args.version:
            import bngsim

            if as_json:
                _write_stdout(sys.stdout, _dumps({"version": bngsim.__version__}))
            else:
                _write_stdout(sys.stdout, f"bngsim {bngsim.__version__}\n")
            return EXIT_OK
        if args.command is None:
            raise _CommandLineError(
                f"a command is required: one of {', '.join(_ALL_COMMANDS)}",
                parser.format_usage().rstrip(),
            )
        failed: BaseException | None = None
        code = EXIT_OK
        with _stderr_captured(as_json) as log:
            try:
                with _stdout_to_stderr() as emit:
                    try:
                        payload, text = _COMMANDS[args.command](args)
                    except _CheckNotOk as not_ok:
                        # A finished check that is not ok is an answer, not an
                        # error: the report goes to standard output, and the
                        # exit code says so.
                        payload, text, code = not_ok.payload, not_ok.text, EXIT_CHECK_FAILED
                    emit(_dumps(payload) if as_json else text)
            except _ReaderGone:
                raise
            except Exception as error:
                failed = error
        captured = "".join(log)
        if failed is not None:
            return _report_error(failed, as_json, captured)
        if captured:
            _write(sys.stderr, captured)
        return code
    except _ReaderGone:
        # The reader went away (`bngsim guide | head`). Point standard output
        # at nothing, so the interpreter's own flush at exit does not fail too.
        with contextlib.suppress(OSError, ValueError):
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        return EXIT_OK
    except Exception as error:
        return _report_error(error, as_json)
