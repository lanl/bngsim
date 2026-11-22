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

#: Each exit code and what it means, as the documentation lists them.
EXIT_CODES: dict[int, str] = {
    EXIT_OK: "success",
    EXIT_ERROR: "error: the command was well formed and failed",
    EXIT_USAGE: "usage error: the command line is wrong",
    EXIT_REFUSED: "refusal: the model cannot be simulated faithfully",
    EXIT_RUN_FAILED: "failed run: the solver gave up",
    EXIT_TIMEOUT: "timeout: a run took longer than its limit",
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
        error = UsageError(str(error), details={"usage": error.usage} if error.usage else None)
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
        description="Inspect a model and this install of BNGsim. Also runs as `python -m bngsim`.",
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
    sub.add_parser("cache", help="the codegen artifact cache (bngsim cache --help)")
    return parser


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


#: Every command, in the order --help lists them.
_ALL_COMMANDS = ("doctor", "guide", "skill", "examples", "inspect", "convert", "cache")

_COMMANDS: dict[str, Callable[[argparse.Namespace], tuple[Any, str]]] = {
    "doctor": _doctor,
    "guide": _guide,
    "skill": _guide,
    "examples": _examples,
    "inspect": _inspect,
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


def _report_error(error: BaseException, as_json: bool) -> int:
    exit_code = exit_code_for(error)
    if as_json:
        try:
            text = _dumps(error_dict(error))
        except Exception as failure:  # the report must not fail on the error it reports
            text = _dumps(error_dict(RuntimeError(f"{error} (and to_dict() failed: {failure})")))
        _write(sys.stderr, text)
        return exit_code
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
        with _stdout_to_stderr() as emit:
            payload, text = _COMMANDS[args.command](args)
            emit(_dumps(payload) if as_json else text)
        return EXIT_OK
    except _ReaderGone:
        # The reader went away (`bngsim guide | head`). Point standard output
        # at nothing, so the interpreter's own flush at exit does not fail too.
        with contextlib.suppress(OSError, ValueError):
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        return EXIT_OK
    except Exception as error:
        return _report_error(error, as_json)
