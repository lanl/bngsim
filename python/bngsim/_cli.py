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
``check MODEL``     the run, then :meth:`~bngsim.Result.check`
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
import sys
from collections.abc import Callable, Iterator
from typing import Any

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
        if named:
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

    def __init__(self, message: str, usage: str = "") -> None:
        super().__init__(message)
        self.usage = usage


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
    sub.add_parser(
        "guide",
        parents=[common],
        help="print the agent guide of this version",
        description="Print the BNGsim agent guide of this version (bngsim.agent_guide()).",
    )
    sub.add_parser(
        "skill",
        parents=[common],
        help="print the agent guide as a SKILL.md file",
        description="Print the agent guide as a SKILL.md file, for agent "
        "harnesses that load skills: bngsim skill > ~/.claude/skills/bngsim/SKILL.md",
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
    check.add_argument(
        "--checks",
        default=None,
        help="'all', or check names separated by commas (tolerance, grid, jacobian, "
        "backend, second_engine, replicates); default: the cheap ones for the method",
    )
    check.add_argument(
        "--budget",
        type=_seconds,
        default=_DEFAULT_TIMEOUT,
        help=f"seconds for all the checks' reruns (default {_DEFAULT_TIMEOUT:g})",
    )
    sub.add_parser("cache", help="the codegen artifact cache (bngsim cache --help)")
    return parser


#: The wall-clock limit of a run, and of a check's reruns, where the command
#: line gives none: a command that hangs costs an agent its turn.
_DEFAULT_TIMEOUT = 600.0


def _run_arguments(p: argparse.ArgumentParser) -> None:
    """The options ``run`` and ``check`` share: what to run and how. An option
    the run would not use is refused, not dropped (see :func:`_validate`)."""
    p.add_argument("model", metavar="MODEL", help="a model file (.xml, .sbml, .net, .ant, .bngl)")
    p.add_argument(
        "--network",
        action="store_true",
        help="for a .bngl file: one run of the network BNG2.pl generates, not the "
        "file's own actions (required for .bngl, refused for any other file)",
    )
    p.add_argument("--method", default="ode", help="ode (default), ssa, psa, nf, rm")
    p.add_argument("--xml", default=None, help="the BioNetGen XML nf and rm read")
    p.add_argument(
        "--poplevel", type=_positive, default=None, help="psa's population level (psa only)"
    )
    p.add_argument("--t-start", type=float, default=None, help="the start time (default 0)")
    p.add_argument("--t-end", type=float, default=None, help="the end time")
    p.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="output points, the first included (default 101)",
    )
    p.add_argument(
        "--sample-times",
        type=_floats,
        default=None,
        help="increasing output times, separated by commas, in place of --t-start, "
        "--t-end and --n-points",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="before the run, set a parameter (Model.set_param), or a species' starting "
        "concentration (Model.set_concentration: its amount over its compartment's "
        "size); repeatable",
    )
    p.add_argument("--seed", type=int, default=None, help="the seed of a stochastic run")
    p.add_argument("--rtol", type=_positive, default=None, help="relative tolerance (ode only)")
    p.add_argument(
        "--atol",
        type=_atol,
        default=None,
        help="absolute tolerance (ode only): a number, 'auto' or 'tracking'",
    )
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
    p.add_argument(
        "--timeout",
        type=_seconds,
        default=_DEFAULT_TIMEOUT,
        help=f"seconds the run may take (default {_DEFAULT_TIMEOUT:g}); exit 5 past it. "
        "Loading the model is not counted.",
    )
    p.add_argument("--species", action="store_true", help="digest the species too")
    p.add_argument(
        "--max-columns", type=int, default=20, help="columns of each kind the digest lists"
    )
    p.add_argument(
        "--out",
        default=None,
        help="write the arrays, replacing any such file: .npz (NumPy alone), .h5 or "
        ".hdf5 (needs h5py), .csv or .tsv (one run)",
    )
    p.add_argument(
        "--out-kind",
        choices=("observables", "species"),
        default=None,
        help="which columns a .csv or .tsv holds (default observables)",
    )


def _floats(text: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected numbers separated by commas, not {text!r}"
        ) from None
    if not values:
        raise argparse.ArgumentTypeError("expected at least one time")
    return values


def _positive(text: str, what: str = "a number") -> float:
    try:
        value = float(text)
    except ValueError:
        value = -1.0
    if not value > 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected {what} > 0, not {text!r}")
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
    payload = {"version": bngsim.__version__, "format": "skill" if skill else "markdown"}
    payload["text"] = text
    return payload, text


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


#: The methods each method-bound option is for, by the names --method takes.
_STOCHASTIC_METHODS = frozenset({"ssa", "psa", "nf", "nfsim", "rm", "rulemonkey"})
_NETWORK_FREE = frozenset({"nf", "nfsim", "rm", "rulemonkey"})
_OUT_SUFFIXES = (".npz", ".h5", ".hdf5", ".csv", ".tsv")


def _refuse(message: str, argument: str, **details: Any) -> None:
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
    the run would not use or could not honour."""
    method = str(args.method).lower()
    if args.sample_times is None and args.t_end is None:
        _refuse("give --t-end (with --n-points) or --sample-times", "--t-end")
    if args.sample_times is not None:
        for flag, value in (
            ("--t-end", args.t_end),
            ("--t-start", args.t_start),
            ("--n-points", args.n_points),
        ):
            if value is not None:
                _refuse(f"--sample-times sets the times; it takes no {flag}", flag)
        times = args.sample_times
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            _refuse("--sample-times must increase", "--sample-times", value=times)
    if args.n_points is not None and args.n_points < 2:
        _refuse("--n-points takes 2 or more", "--n-points")
    if method not in _STOCHASTIC_METHODS and args.seed is not None:
        _refuse(f"--seed is a stochastic run's; a {method} run takes none", "--seed")
    if method in _STOCHASTIC_METHODS:
        for flag, value in (("--rtol", args.rtol), ("--atol", args.atol)):
            if value is not None:
                _refuse(f"{flag} is an ODE run's; a {method} run takes none", flag)
    if args.xml is not None and method not in _NETWORK_FREE:
        _refuse(f"--xml is for the network-free methods nf and rm, not {method}", "--xml")
    if args.poplevel is not None and method != "psa":
        _refuse(f"--poplevel is psa's, not {method}'s", "--poplevel")
    if method == "psa" and args.poplevel is None:
        _refuse("--method psa needs --poplevel N", "--poplevel")
    is_bngl = str(args.model).lower().endswith(".bngl")
    if is_bngl and not args.network:
        from bngsim._exceptions import UsageError

        raise UsageError(
            "a .bngl file can mean one run of its generated network or the file's own "
            "actions; this command makes one run of the network: pass --network. The "
            "file's own actions are bngsim.run_bngl(path) in Python.",
            details={"argument": "--network"},
            remedies=[{"kind": "pass_argument", "argument": "--network"}],
        )
    if args.network and not is_bngl:
        _refuse("--network is for a .bngl file", "--network")
    if args.command == "check" and args.replicates is not None:
        _refuse(
            "`bngsim check` checks one run; --replicates is for `bngsim run` (the "
            "replicates check reruns a stochastic run at other seeds itself)",
            "--replicates",
        )
    if args.replicates is not None:
        if method not in _STOCHASTIC_METHODS:
            _refuse(f"--replicates is for a stochastic method, not {method}", "--replicates")
        for flag, value in (
            ("--sample-times", args.sample_times),
            ("--steady-state", args.steady_state or None),
        ):
            if value is not None:
                _refuse(f"--replicates runs run_replicates(), which takes no {flag}", flag)
    _validate_out(args)
    _parse_sets(args.set)


def _validate_out(args: argparse.Namespace) -> None:
    out = args.out
    text_out = out is not None and out.lower().endswith((".csv", ".tsv"))
    if args.out_kind is not None and not text_out:
        _refuse("--out-kind is for a .csv or .tsv --out", "--out-kind")
    if out is None:
        return
    if not out.lower().endswith(_OUT_SUFFIXES):
        _refuse(
            f"--out takes a file ending {', '.join(_OUT_SUFFIXES)}, not {out!r}",
            "--out",
            value=out,
        )
    if os.path.isdir(out):
        _refuse(f"--out {out!r} is a directory", "--out", value=out)
    parent = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(parent):
        _refuse(f"--out {out!r}: the directory {parent!r} does not exist", "--out", value=out)
    if text_out and args.replicates is not None:
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


def _load_for_run(args: argparse.Namespace) -> tuple[Any, dict[str, float]]:
    """The model ``run`` and ``check`` run, with the ``--set`` values, and
    those values. A species set this way becomes the model's initial state
    (``save_concentrations``), so the run starts from its initial state."""
    from bngsim import Model
    from bngsim._exceptions import BngsimError, UsageError

    model = Model.load(args.model)
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
    if species:
        model.save_concentrations()
    return model, done


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

    digest = result.digest(species=args.species, max_columns=args.max_columns)
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


#: Every command, in the order --help lists them.
_ALL_COMMANDS = (
    "doctor",
    "guide",
    "skill",
    "examples",
    "inspect",
    "run",
    "check",
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
    exit_code = exit_code_for(error)
    if as_json:
        try:
            d = error_dict(error)
        except Exception as failure:  # the report must not fail on the error it reports
            d = error_dict(RuntimeError(f"{error} (and to_dict() failed: {failure})"))
        if log:
            # What the command wrote to standard error (and output) before it
            # failed, SUNDIALS' own lines among it: inside the one JSON object.
            d["log"] = log[-_MAX_LOG:]
        _write(sys.stderr, _dumps(d))
        return exit_code
    if log:
        _write(sys.stderr, log)
    if isinstance(error, _CommandLineError):
        usage = f"{error.usage}\n" if error.usage else ""
        _write(sys.stderr, f"{usage}bngsim: error: {error}\n")
        return exit_code
    from bngsim._exceptions import BngsimError

    if not isinstance(error, BngsimError):
        # Not an error BNGsim raises on purpose, so probably a bug in it: the
        # traceback is what a report needs.
        import traceback

        _write(sys.stderr, "".join(traceback.format_exception(error)))
    d = error_dict(error)
    lines = [f"bngsim: error [{d['code']}]: {d['message']}"]
    for remedy in d.get("remedies") or []:
        lines.append(f"  remedy: {json.dumps(_json_safe(remedy))}")
    _write(sys.stderr, "\n".join(lines) + "\n")
    return exit_code


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
    """``bngsim`` and ``python -m bngsim``. Returns the exit code."""
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
        parser = _build_parser()
        try:
            args = parser.parse_args(args_list)
        except SystemExit as stop:  # --help
            return stop.code if isinstance(stop.code, int) else EXIT_OK
        as_json = bool(getattr(args, "json", False))
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
