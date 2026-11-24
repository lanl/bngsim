"""The codes BNGsim's errors carry, and the remedies they offer (issue #1702).

Every :class:`bngsim.BngsimError` has a ``code``: a stable string a program can
branch on where it would otherwise match the message, which changes between
releases. A code names the decision a caller makes, not the sentence: one for a
parameter name the model does not have, one for a missing optional dependency,
one for each way the ODE solver gives up. An error with no code of its own
carries its class's generic code (``model.error``, ``run.failed``, ...).

This module is the registry. ``scripts/error_codes_page.py`` writes the
reference page from it, and a test checks that every code an error carries is
here and that the page is current.

A remedy is a dict with a ``kind`` and its arguments, something a program can
act on without reading English. The kinds are in :data:`REMEDY_KINDS`. A
refusal with no fix BNGsim can name offers none: an empty list is the honest
answer there.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorCode:
    """One code: what it means and what to do about it."""

    code: str
    #: The class that raises it, or the most general one that can; several,
    #: joined by " or ", where the code is raised as more than one (a missing
    #: name is a KeyError where a lookup took one, a ValueError where an
    #: argument did).
    error: str
    #: What happened, in a sentence.
    meaning: str
    #: What to do about it, in a sentence or two.
    action: str


_CODES = (
    # ─── Generic: an error with no code of its own carries its class's ───────
    ErrorCode(
        "error",
        "BngsimError",
        "An error with no more specific code.",
        "Read the message.",
    ),
    ErrorCode(
        "model.error",
        "ModelError",
        "A model could not be loaded or used, and the error has no more specific code.",
        "Read the message: it names the file, the construct or the call.",
    ),
    ErrorCode(
        "run.failed",
        "SimulationError",
        "A run failed, and the error has no more specific code.",
        "Read the message: it names what failed and where.",
    ),
    ErrorCode(
        "param.error",
        "ParameterError",
        "A parameter could not be read or set, and the error has no more specific code.",
        "Read the message.",
    ),
    ErrorCode(
        "conversion.error",
        "ConversionError",
        "A conversion between formats cannot be done without changing the model.",
        "Read the message: it names what the target format cannot hold.",
    ),
    ErrorCode(
        "coupling.conservation",
        "ConservationError",
        "A conserved total drifted past its tolerance across a coupling boundary.",
        "Shorten the exchange interval or tighten the tolerances.",
    ),
    ErrorCode(
        "usage.invalid_argument",
        "UsageError",
        "An argument to a call is not one it can take.",
        "Read the message: it names the argument and what it takes.",
    ),
    # ─── Loading a model ─────────────────────────────────────────────────────
    ErrorCode(
        "file.not_found",
        "MissingFileError",
        "The file named does not exist.",
        "Check the path. ``details['path']`` is the one BNGsim looked for.",
    ),
    ErrorCode(
        "model.unknown_format",
        "ModelError",
        "``Model.load`` cannot tell a file's format from its suffix.",
        "Rename the file to a suffix in ``details['expected']``, or call the loader for "
        "its format, such as ``Model.from_sbml``.",
    ),
    ErrorCode(
        "model.unsupported_construct",
        "ModelError",
        "The model holds a construct BNGsim cannot simulate correctly, such as an "
        "algebraic rule or a delay.",
        "The model cannot be simulated as written. ``details['constructs']`` lists each "
        "construct and where it is. Change the model, or report the refusal.",
    ),
    ErrorCode(
        "model.unset_symbol",
        "UnderSpecifiedModelError",
        "The model reads a symbol it never gives a value.",
        "Give the symbol a value in the model, or report the refusal.",
    ),
    # ─── Names ───────────────────────────────────────────────────────────────
    ErrorCode(
        "model.invalid",
        "InvalidModelError",
        "A model, or a part of one given to a builder, is not well formed: a name "
        "defined twice, a reaction with no rate, a reference to something not defined.",
        "Fix the model as the message says.",
    ),
    ErrorCode(
        "model.unreadable_file",
        "ModelError",
        "A model file is not what its format says, or its reader stops on a fatal error.",
        "Check the file is the format its name says; the message quotes the reader.",
    ),
    ErrorCode(
        "param.not_found",
        "ParameterError or UsageError",
        "The model has no parameter by that name.",
        "Use a name from ``Model.param_names``. ``details['close_matches']`` lists "
        "near ones, and ``details['is_a']`` says what the name is when it names a "
        "species, an observable or a function instead.",
    ),
    ErrorCode(
        "species.not_found",
        "ModelError or NotFoundError or UsageError",
        "The model has no species by that name.",
        "Use a name from ``Model.species_names``. ``details['close_matches']`` lists near ones.",
    ),
    # ─── Choosing and setting up a run ───────────────────────────────────────
    ErrorCode(
        "output.not_found",
        "NotFoundError or UsageError",
        "A name that is none of the result's species, observables or functions.",
        'Use a name `details["close_matches"]` gives, or one of the result\'s '
        "`species_names` or `observable_names`.",
    ),
    ErrorCode(
        "name.not_found",
        "NotFoundError or UsageError",
        "A name or key a lookup does not find, of a kind with no code of its own.",
        "Read the message: it lists the names there are.",
    ),
    ErrorCode(
        "name.ambiguous",
        "NotFoundError",
        "A name matches more than one entry, so a lookup by it cannot choose.",
        "Index by position, as the message says.",
    ),
    ErrorCode(
        "usage.missing_argument",
        "UsageError",
        "A call needs an argument it was not given.",
        "Pass the argument ``details['argument']`` names.",
    ),
    ErrorCode(
        "usage.conflicting_arguments",
        "UsageError",
        "Two arguments a call takes one or the other of, or that cannot go together, "
        "were both given.",
        "Read the message for the pair, and drop one.",
    ),
    ErrorCode(
        "usage.wrong_type",
        "ArgumentTypeError",
        "An argument is of a type the call does not take.",
        "Read the message: it names the type the call takes.",
    ),
    ErrorCode(
        "usage.unknown_method",
        "UsageError",
        "A `method=` that BNGsim does not know.",
        "Pass one the message lists.",
    ),
    ErrorCode(
        "usage.method_unsupported",
        "UsageError",
        "A call or option the simulator's method does not offer, such as "
        "sensitivities on a stochastic method.",
        "Build the Simulator with the method the remedy names, or leave the option out.",
    ),
    ErrorCode(
        "usage.not_supported",
        "NotSupportedError or UsageError",
        "A call or option this object does not offer, as written.",
        "Read the message for what does.",
    ),
    ErrorCode(
        "usage.span_out_of_reach",
        "UsageError",
        "An ODE run CVODE cannot step through: its length is past the largest double, "
        "or CVODE's step, or its clock (t + h), passed it before the run ended. Whether "
        "a run gets there can turn on the model and its values.",
        "Measure time in a larger unit, so that the run's times are smaller.",
    ),
    ErrorCode(
        "env.feature_not_built",
        "BngsimError or UsageError",
        "A method needs a part of BNGsim this install was built without, such as NFsim "
        "or RuleMonkey.",
        "Install a build that has it; `bngsim.capabilities()` lists what this one has.",
    ),
    ErrorCode(
        "ssa.unsupported_construct",
        "SsaValidationError",
        "The model holds a construct a stochastic method cannot simulate.",
        "Run it with ``method='ode'``. ``details['issues']`` lists each construct.",
    ),
    ErrorCode(
        "codegen.declined",
        "CodegenDeclinedError",
        "``codegen=True`` on a model codegen declines.",
        "Pass ``codegen=None`` to run it on the interpreter.",
    ),
    ErrorCode(
        "codegen.build_failed",
        "CodegenBuildError",
        "Building this model's compiled right-hand side failed or ran out of time; "
        "the toolchain builds other code.",
        "Pass `codegen=False` to run on the interpreted right-hand side, or give the "
        "build more time or cores as the message says.",
    ),
    ErrorCode(
        "solver.sparse_jacobian_unavailable",
        "SparseJacobianUnavailableError",
        "``force_sparse_linear_solver=True`` on a model with no sparse Jacobian.",
        "Leave out ``force_sparse_linear_solver`` to use the dense solver.",
    ),
    # ─── Sensitivities ───────────────────────────────────────────────────────
    ErrorCode(
        "sens.unsupported",
        "SensitivityUnsupportedError",
        "BNGsim cannot compute a requested sensitivity correctly for this model.",
        "Drop the sensitivity, change the model, or use a derivative-free method.",
    ),
    ErrorCode(
        "sens.not_computed",
        "UsageError",
        "A call reads sensitivities, and the result has none: its run was not asked for them.",
        "Run again with `sensitivity_params=` on the Simulator, or with "
        "`compute_all_sensitivities()`.",
    ),
    ErrorCode(
        "sens.refused_during_run",
        "SensitivityRefusedDuringRunError",
        "The solver reached, partway through a run, a construct whose derivative "
        "BNGsim cannot compute.",
        "Read the message: it names the construct and what to change.",
    ),
    ErrorCode(
        "sens.refused_for_parameters",
        "SensitivityRefusedForParametersError",
        "The solver refused a sensitivity partway through a run, for the parameters requested.",
        "Drop the parameters the message names; a smaller set may run.",
    ),
    # ─── The environment ─────────────────────────────────────────────────────
    ErrorCode(
        "env.missing_dependency",
        "DependencyError",
        "A call needs an optional package that is not installed.",
        "Install it. The remedy gives the ``pip install`` command.",
    ),
    ErrorCode(
        "env.broken_dependency",
        "DependencyError",
        "A package a call needs is installed but cannot be imported.",
        "Reinstall it for this Python. The message gives the import error.",
    ),
    ErrorCode(
        "env.no_compiler",
        "CodegenToolchainError",
        "A call needs a C compiler, and there is none.",
        "Install a C compiler, or set ``CC`` to one.",
    ),
    ErrorCode(
        "env.broken_toolchain",
        "CodegenToolchainError",
        "The C compiler is there, but it cannot build a library for any model.",
        "Fix the toolchain. The message names the compiler, the linker and the SDK.",
    ),
    # ─── Runs that fail ──────────────────────────────────────────────────────
    ErrorCode(
        "solver.too_much_work",
        "SimulationError",
        "The ODE solver took ``max_steps`` steps without reaching the next output time.",
        "Read the message. Steps that still advance are retried, so this one did not.",
    ),
    ErrorCode(
        "solver.too_much_accuracy",
        "SimulationError",
        "The tolerances asked for more accuracy than double precision holds.",
        "Loosen ``rtol`` or ``atol``.",
    ),
    ErrorCode(
        "solver.stalled",
        "SimulationError",
        "The ODE solver's step size collapsed, so a whole batch of steps advanced "
        "nothing, typically at a jump in a rate law.",
        "Loosen ``rtol`` or ``atol``, or move the jump onto an event. A larger "
        "``max_steps`` does not help.",
    ),
    ErrorCode(
        "solver.error_test_failure",
        "SimulationError",
        "The ODE solver failed its error test repeatedly at one time, often at a jump "
        "or a singularity.",
        "Read the message: it names the time and, where it can, the species and the "
        "rate law. Check the model there.",
    ),
    ErrorCode(
        "solver.convergence_failure",
        "SimulationError",
        "The ODE solver's Newton iteration failed to converge repeatedly at one time.",
        "Read the message: it names the time and, where it can, the species and the "
        "rate law. Check the model there.",
    ),
    ErrorCode(
        "solver.failed",
        "SimulationError",
        "The ODE solver failed in another way. ``details['flag']`` is SUNDIALS' name for it.",
        "Read the message.",
    ),
    ErrorCode(
        "run.timeout",
        "SimulationTimeout",
        "A run took longer than its wall-clock budget.",
        "Pass a larger ``timeout``. ``details`` has the budget and the time taken.",
    ),
    ErrorCode(
        "run.stop_condition",
        "StopConditionMet",
        "A stop condition was met. The error carries the result up to that point.",
        "Read ``e.result``. This is how a stop condition reports, not a failure.",
    ),
    # ─── Making a run again (Result.rerun) ───────────────────────────────────
    ErrorCode(
        "rerun.not_recorded",
        "UsageError",
        "The result's record lacks a part a rerun needs (``details['missing']``): it was "
        "made by a call that records no run, or loaded from a file an earlier bngsim wrote.",
        "Run the model again with ``Simulator.run``.",
    ),
    ErrorCode(
        "rerun.unsupported",
        "UsageError",
        "The record cannot promise the same run (``details['reason']``): a stack of runs, a "
        "network-free run, stop conditions, carried or declared sensitivities, sensitivities "
        "seeded otherwise on the rebuilt model, a model changed after its load, or one built "
        "in Python with no source.",
        "Rerun one row of a stack; run the others again with ``Simulator.run``.",
    ),
    ErrorCode(
        "rerun.unknown_setting",
        "UsageError",
        "``Result.rerun()`` was given a setting it does not change.",
        "Use one of ``details['accepted']``.",
    ),
    ErrorCode(
        "rerun.model_changed",
        "ModelError",
        "The model file's sha256 differs from the record's, or the model built from the "
        "recorded source is not the one the run used: its table data, parameters or species "
        "differ (``details``).",
        "Load the file as it is now and run it with ``Simulator.run``, or put back the "
        "file the run used.",
    ),
    ErrorCode(
        "rerun.model_missing",
        "ModelError",
        "The model file the run used cannot be read.",
        "Put the file back at ``details['path']``.",
    ),
)

#: The generic codes: an error raised from one with a code outside these carries
#: that code, since it says more than the wrapper's own.
GENERIC = frozenset(
    {
        "error",
        "model.error",
        "run.failed",
        "param.error",
        "conversion.error",
        "usage.invalid_argument",
    }
)

#: Every code, by name.
CODES: dict[str, ErrorCode] = {c.code: c for c in _CODES}

#: What each remedy ``kind`` means, and the keys it carries besides ``kind``.
REMEDY_KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "install_package": (
        "Install a package. ``command`` is the command to run.",
        ("package", "command"),
    ),
    "install_compiler": (
        "Install a C compiler, or set ``CC`` to one.",
        (),
    ),
    "pass_argument": (
        "Call again with this argument. ``value`` is present where one value is the fix.",
        ("argument",),
    ),
    "use_name": (
        "Use this name instead.",
        ("name",),
    ),
    "use_call": (
        "Use this call instead, as for a species named where a parameter was asked for.",
        ("call", "name"),
    ),
    "use_method": (
        "Run with this method instead.",
        ("method",),
    ),
    "loosen_tolerance": (
        "Loosen ``rtol`` or ``atol``.",
        (),
    ),
}

#: Where the reference page lives.
DOCS_BASE = "https://bngsim.readthedocs.io/en/latest/reference/error-codes.html"


def docs_url(code: str) -> str:
    """The reference page's anchor for *code*."""
    return f"{DOCS_BASE}#{anchor(code)}"


def anchor(code: str) -> str:
    """The HTML id of *code* on the reference page."""
    return "code-" + code.replace(".", "-").replace("_", "-")


def install(package: str, extra: str | None = None, command: str | None = None) -> dict:
    """The remedy that installs *package*, through BNGsim's *extra* when it has one."""
    if command is None:
        command = f"pip install 'bngsim[{extra}]'" if extra else f"pip install {package}"
    remedy = {"kind": "install_package", "package": package, "command": command}
    if extra:
        remedy["extra"] = extra
    return remedy


def pass_argument(argument: str, *value: object) -> dict:
    """The remedy that passes *argument*, with its value when one is the fix."""
    remedy: dict = {"kind": "pass_argument", "argument": argument}
    if value:
        remedy["value"] = value[0]
    return remedy


def use_method(method: str) -> dict:
    """The remedy that builds the Simulator with *method* instead."""
    return {"kind": "use_method", "method": method}


def use_name(name: str) -> dict:
    """The remedy that uses *name*, a close match, in place of the one given."""
    return {"kind": "use_name", "name": name}


def use_call(call: str, name: str) -> dict:
    """The remedy that passes *name* to *call*, the method for what it names."""
    return {"kind": "use_call", "call": call, "name": name}


#: SUNDIALS' names for CVODE's hard failures, by the code each gets.
SOLVER_FLAGS = {
    "CV_TOO_MUCH_WORK": "solver.too_much_work",
    "CV_TOO_MUCH_ACC": "solver.too_much_accuracy",
    "CV_ERR_FAILURE": "solver.error_test_failure",
    "CV_CONV_FAILURE": "solver.convergence_failure",
}


#: The reference page, generated from this module.
PAGE = "docs/reference/error-codes.md"


def _classes(error: str) -> str:
    """*error*'s classes as the page names them: "`A`", "`A` or `B`", "`A`, `B`
    or `C`"."""
    names = [f"`{name}`" for name in error.split(" or ")]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def reference_page() -> str:
    """The text of the error-code reference page (``scripts/error_codes_page.py``
    writes it; a test checks it is current)."""
    lines = [
        "---",
        'summary: "The code every BNGsim error and run notice carries, what each means and '
        'what to do about it, and the remedies a program can act on."',
        "# Generated from python/bngsim/_error_codes.py by scripts/error_codes_page.py;",
        "# edit that module and run the script.",
        "---",
        "",
        "# Error and notice codes",
        "",
        "Every BNGsim error carries a `code`, a string that stays the same across "
        "releases where the message may change. A program should branch on the code, "
        "not on the message.",
        "",
        "```python",
        "import bngsim",
        "",
        'model = bngsim.examples.load("binding")  # parameters kf, kr, A0 and B0',
        "",
        "try:",
        '    model.set_param("kff", 1.0)',
        "except bngsim.BngsimError as e:",
        "    e.code       # 'param.not_found'",
        "    e.details    # {'name': 'kff', 'close_matches': ['kf'], 'is_a': None}",
        "    e.remedies   # [{'kind': 'use_name', 'name': 'kf'}]",
        "    e.to_dict()  # type, code, message, details, remedies, docs_url, notices",
        "```",
        "",
        "- `details` says what the error is about: the name, the path, the constructs.",
        "- `remedies` lists fixes a program can act on, each a dict with a `kind`. "
        "It is empty where BNGsim can name no fix, as for a model it cannot simulate "
        "as written.",
        "- `docs_url` points at the code's entry below.",
        "- `notices` lists what the failed run had noted before it raised, such as a "
        "start from the last run's end state, each as on a Result; see "
        "{ref}`Notice codes <notice-codes>`.",
        "",
        "An error with no code of its own carries its class's generic code, such as "
        "`model.error` or `run.failed`. An error of a generic class raised from a BNGsim "
        "error with a specific code, as a loader wraps its parser's, carries that "
        "error's code, details and remedies.",
        "",
        "An error of use is a BNGsim class that is also the built-in it replaces, so an "
        "`except` written for the built-in still catches it: `UsageError` is a "
        "`ValueError`, `DependencyError` an `ImportError`, and `MissingFileError` a "
        "`FileNotFoundError`.",
        "",
        "## Codes",
        "",
    ]
    for c in _CODES:
        lines += [
            f"({anchor(c.code)})=",
            f"### `{c.code}`",
            "",
            f"Raised as {_classes(c.error)}. {c.meaning}",
            "",
            c.action,
            "",
        ]
    lines += [
        "(notice-codes)=",
        "## Notice codes",
        "",
        "A run's warnings and fallbacks are kept on `result.notices`, and on the error "
        "of a run that fails, each a dict with a `category`, a `code` and a `message`. "
        "A notice's code is stable across releases as an error's is, and no code is "
        "both. An error's code names the decision a caller makes; a notice's names the "
        "part of the run that noted it, so a compiler a run could not use is the notice "
        "`codegen.toolchain`, and the error `env.no_compiler` or `env.broken_toolchain` "
        "where a call cannot go on without one.",
        "",
        "| code | meaning | what to do |",
        "|---|---|---|",
    ]
    from bngsim import _notices

    def cell(text: str) -> str:
        return text.replace("|", "\\|")

    for code, meaning in _notices.CODES.items():
        lines.append(f"| `{code}` | {cell(meaning)} | {cell(_notices.ACTIONS[code])} |")
    lines += [
        "",
        "## Remedies",
        "",
        "| kind | keys | meaning |",
        "|---|---|---|",
    ]
    for kind, (meaning, keys) in REMEDY_KINDS.items():
        keys_text = ", ".join(f"`{k}`" for k in keys) or "none"
        lines.append(f"| `{kind}` | {keys_text} | {meaning} |")
    return "\n".join(lines) + "\n"
