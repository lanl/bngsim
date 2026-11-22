"""The ``bngsim`` command (issue #1708, part 1): doctor, guide, skill, examples,
inspect, and convert and cache handed to the existing commands.

The command is a thin layer over public calls, so each test compares what it
prints with what the call returns. The rules that make it usable by a program
are tested as rules: one JSON object on standard output and nothing else there,
an error as a JSON object on standard error, and an exit code per outcome.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import _cli

_ALGEBRAIC = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
<model id="m"><listOfCompartments>
<compartment id="c" size="1" constant="true"/></listOfCompartments>
<listOfSpecies>
<species id="A" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
<species id="B" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
 boundaryCondition="false" constant="false"/>
</listOfSpecies><listOfRules><algebraicRule>
<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><minus/><ci>A</ci><ci>B</ci></apply>
</math></algebraicRule></listOfRules></model></sbml>"""


def _bngsim(*args: str, script: str | None = None) -> subprocess.CompletedProcess:
    """``python -m bngsim ARGS``, or ``script`` then ``main(ARGS)``."""
    if script is None:
        cmd = [sys.executable, "-m", "bngsim", *args]
    else:
        code = script + f"\nfrom bngsim._cli import main\nraise SystemExit(main({list(args)!r}))\n"
        cmd = [sys.executable, "-c", code]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)


def _json_out(run: subprocess.CompletedProcess) -> dict:
    """The one JSON object a successful ``--json`` command prints, and nothing more."""
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)  # raises on anything before or after the object


def _json_error(run: subprocess.CompletedProcess, exit_code: int, code: str) -> dict:
    """The JSON error a failed ``--json`` command prints, with nothing on stdout."""
    assert run.returncode == exit_code, (run.returncode, run.stderr)
    assert run.stdout == ""
    error = json.loads(run.stderr)
    assert set(error) == {
        "type",
        "code",
        "message",
        "details",
        "remedies",
        "docs_url",
        "notices",
    }
    assert error["code"] == code
    return error


def _plain(value):
    """``value`` as JSON reads it back."""
    return json.loads(json.dumps(value))


# ─── The commands ────────────────────────────────────────────────────────────


def test_doctor_is_capabilities_with_the_python_platform_and_compiler():
    report = _json_out(_bngsim("doctor", "--json"))
    caps = bngsim.capabilities()
    for key in caps:
        assert report[key] == _plain(caps[key]), key
    assert report["python"]["executable"] == sys.executable
    assert Path(report["package_path"]) == Path(bngsim.__file__).resolve().parent
    assert set(report["compiler"]) == {"command", "builds", "problem"}
    # A compiler that builds has no problem; one that does not says why.
    assert (report["compiler"]["builds"] is True) == (report["compiler"]["problem"] is None)
    assert set(report["platform"]) == {"system", "release", "machine"}


def test_doctor_text_names_every_missing_feature_with_its_fix():
    run = _bngsim("doctor")
    assert run.returncode == 0, run.stderr
    assert f"bngsim {bngsim.__version__}" in run.stdout
    for name, fix in bngsim.capabilities()["missing"].items():
        assert f"{name}: {fix}" in run.stdout


@pytest.mark.parametrize("command", ["guide", "skill"])
def test_guide_and_skill_as_json_carry_the_guide(command):
    payload = _json_out(_bngsim(command, "--json"))
    assert payload["text"] == bngsim.agent_guide(skill=command == "skill")
    assert payload["version"] == bngsim.__version__


def test_examples_lists_every_example_with_its_files():
    listed = _json_out(_bngsim("examples", "--json"))["examples"]
    assert [e["name"] for e in listed] == bngsim.examples.names()
    for entry in listed:
        assert entry["formats"] == bngsim.examples.formats(entry["name"])
        for fmt, path in entry["paths"].items():
            assert Path(path) == bngsim.examples.path(entry["name"], fmt)
    one = _json_out(_bngsim("examples", "decay", "--json"))
    assert one == next(e for e in listed if e["name"] == "decay")


def test_an_unknown_example_is_a_usage_error():
    error = _json_error(_bngsim("examples", "nope", "--json"), 2, "usage.invalid_argument")
    assert error["details"]["expected"] == bngsim.examples.names()


@pytest.mark.parametrize("limit", [None, 1])
def test_inspect_is_the_models_description(limit):
    path = str(bngsim.examples.path("binding", "sbml"))
    args = ["inspect", path, "--json"]
    if limit is not None:
        args += ["--limit", str(limit)]
    else:
        args += ["--limit", "all"]
    described = _json_out(_bngsim(*args))
    assert described == _plain(bngsim.Model.load(path).describe(limit=limit))


def test_inspect_text_is_the_descriptions_text():
    path = str(bngsim.examples.path("decay", "net"))
    run = _bngsim("inspect", path)
    assert run.returncode == 0, run.stderr
    assert run.stdout == str(bngsim.Model.load(path).describe()) + "\n"


# ─── Errors and exit codes ───────────────────────────────────────────────────


def test_a_refused_model_exits_3_with_the_refusal(tmp_path):
    model = tmp_path / "algebraic.xml"
    model.write_text(_ALGEBRAIC, encoding="utf-8")
    error = _json_error(_bngsim("inspect", str(model), "--json"), 3, "model.unsupported_construct")
    assert error["details"]["constructs"] == [{"construct": "AlgebraicRule", "where": "0 = A - B"}]
    # Without --json the same refusal, as text, and the same exit code.
    run = _bngsim("inspect", str(model))
    assert run.returncode == 3 and run.stdout == ""
    assert "[model.unsupported_construct]" in run.stderr


def test_a_missing_model_file_exits_1(tmp_path):
    missing = tmp_path / "no_such_model.xml"
    error = _json_error(_bngsim("inspect", str(missing), "--json"), 1, "file.not_found")
    assert error["details"]["path"] == str(missing)


@pytest.mark.parametrize(
    "args",
    [
        [],  # no command
        ["frobnicate"],  # no such command
        ["inspect"],  # MODEL missing
        ["inspect", "m.xml", "--limit", "-1"],
        ["doctor", "--no-such-option"],
    ],
)
def test_a_wrong_command_line_exits_2_with_a_json_usage_error(args):
    error = _json_error(_bngsim("--json", *args), 2, "usage.invalid_argument")
    assert error["type"] == "UsageError"
    assert error["details"]["usage"].startswith("usage: bngsim")


def test_json_is_taken_before_or_after_the_command():
    before = _json_out(_bngsim("--json", "examples", "decay"))
    after = _json_out(_bngsim("examples", "decay", "--json"))
    assert before == after


def test_version():
    run = _bngsim("--version")
    assert (run.returncode, run.stdout) == (0, f"bngsim {bngsim.__version__}\n")
    assert _json_out(_bngsim("--version", "--json")) == {"version": bngsim.__version__}


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (bngsim.SimulationTimeout("slow", timeout=1.0, elapsed=2.0), _cli.EXIT_TIMEOUT),
        (bngsim.SimulationError("solver", code="run.timeout"), _cli.EXIT_TIMEOUT),
        (bngsim.ModelError("no", code="model.unsupported_construct"), _cli.EXIT_REFUSED),
        (bngsim.ModelError("no", code="model.unset_symbol"), _cli.EXIT_REFUSED),
        (bngsim.SsaValidationError([]), _cli.EXIT_REFUSED),
        (bngsim.SensitivityUnsupportedError("no"), _cli.EXIT_REFUSED),
        # A refusal partway through a run is a refusal, though it is a SimulationError.
        (bngsim.SensitivityRefusedDuringRunError("no"), _cli.EXIT_REFUSED),
        (bngsim.SimulationError("CVODE failed with flag=-4"), _cli.EXIT_RUN_FAILED),
        (bngsim.UsageError("bad"), _cli.EXIT_USAGE),
        (bngsim.ModelError("bad", code="usage.missing_argument"), _cli.EXIT_USAGE),
        (bngsim.ModelError("unreadable"), _cli.EXIT_ERROR),
        (bngsim.DependencyError("pip install x"), _cli.EXIT_ERROR),
        (ValueError("a bug"), _cli.EXIT_ERROR),
    ],
)
def test_each_outcome_has_its_own_exit_code(error, exit_code):
    assert _cli.exit_code_for(error) == exit_code


def test_exit_codes_are_distinct_and_documented():
    codes = [
        _cli.EXIT_OK,
        _cli.EXIT_ERROR,
        _cli.EXIT_USAGE,
        _cli.EXIT_REFUSED,
        _cli.EXIT_RUN_FAILED,
        _cli.EXIT_TIMEOUT,
        _cli.EXIT_CHECK_FAILED,
    ]
    assert len(set(codes)) == len(codes)
    assert sorted(_cli.EXIT_CODES) == sorted(codes)
    page = (Path(__file__).parents[2] / "docs" / "reference" / "command-line.md").read_text(
        encoding="utf-8"
    )
    for code in codes:
        # Each row starts with the meaning's first words, as --help gives them.
        assert f"| {code} | {_cli.EXIT_CODES[code].split(':')[0]}" in page, code


def test_an_error_that_is_not_bngsims_has_the_same_keys():
    d = _cli.error_dict(OSError("disk full"))
    assert d == {
        "type": "OSError",
        "code": "error",
        "message": "disk full",
        "details": {},
        "remedies": [],
        "docs_url": None,
        "notices": [],
    }


# ─── Standard output carries the one object ──────────────────────────────────

_NOISY = textwrap.dedent(
    """
    import ctypes, os, sys
    from bngsim import _cli

    def noisy(args):
        print("python noise")
        sys.stdout.flush()
        os.write(1, b"fd noise\\n")
        sys.__stdout__.write("held noise\\n")  # buffered, not flushed
        ctypes.CDLL(None).puts(b"c noise")  # buffered by C's stdio, not flushed
        return {"answer": 42}, "text\\n"

    _cli._COMMANDS["doctor"] = noisy
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific: ctypes.CDLL(None)")
def test_what_else_writes_to_stdout_goes_to_stderr():
    run = _bngsim("doctor", "--json", script=_NOISY)
    assert _json_out(run) == {"answer": 42}
    for noise in ("python noise", "fd noise", "held noise", "c noise"):
        assert noise in run.stderr


def test_a_reader_that_goes_away_is_not_an_error():
    proc = subprocess.Popen(
        [sys.executable, "-m", "bngsim", "guide"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    proc.stdout.readline()
    proc.stdout.close()  # as `bngsim guide | head -1` does
    stderr = proc.stderr.read()
    assert proc.wait(timeout=120) == 0
    assert stderr == b""


def test_a_broken_pipe_inside_a_command_is_an_error():
    script = textwrap.dedent(
        """
        from bngsim import _cli

        def fails(args):
            raise BrokenPipeError("a subprocess's pipe")

        _cli._COMMANDS["doctor"] = fails
        """
    )
    run = _bngsim("doctor", "--json", script=script)
    error = _json_error(run, 1, "error")
    assert error["type"] == "BrokenPipeError"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific: a shell closes fd 1")
def test_a_closed_stdout_is_an_error():
    run = subprocess.run(
        ["sh", "-c", f'"{sys.executable}" -m bngsim --version >&-'],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1
    assert run.stderr == "bngsim: error: standard output is closed\n"


def test_an_error_with_details_json_has_no_form_for_is_still_json():
    error = bngsim.ModelError("x", details={"path": Path("/a"), "names": {"b", "a"}})
    assert json.loads(_cli._dumps(_cli.error_dict(error)))["details"] == {
        "path": str(Path("/a")),
        "names": ["a", "b"],
    }


def test_numbers_json_cannot_hold_are_strings():
    text = _cli._dumps({"a": float("nan"), "b": np.float64("inf"), "c": [-np.inf, np.int64(3)]})
    assert json.loads(text) == {"a": "nan", "b": "inf", "c": ["-inf", 3]}


# ─── convert and cache: the existing commands, unchanged ─────────────────────


def test_cache_is_the_bngsim_cache_command(tmp_path):
    ours = _bngsim("cache", "-C", str(tmp_path), "info", "--json")
    theirs = subprocess.run(
        [sys.executable, "-m", "bngsim.cache", "-C", str(tmp_path), "info", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert (ours.returncode, ours.stdout) == (theirs.returncode, theirs.stdout)


def test_convert_runs_the_converter(tmp_path):
    source = bngsim.examples.path("binding", "sbml")
    ours, theirs = tmp_path / "ours.net", tmp_path / "theirs.net"
    run = _bngsim("convert", "sbml2net", str(source), "-o", str(ours))
    assert run.returncode == 0, run.stderr
    from bngsim.convert._cli import main as sbml2net

    assert sbml2net([str(source), "-o", str(theirs)]) == 0
    assert ours.read_text(encoding="utf-8") == theirs.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (["convert", "sbml2net"], ["bngsim.convert"]),
        (
            ["convert", "net2sbml"],
            ["-c", "from bngsim.convert._cli import net2sbml_main as m; m()"],
        ),
        (
            ["convert", "sbml2bngl"],
            ["-c", "from bngsim.convert._cli import sbml2bngl_main as m; m()"],
        ),
        (
            ["convert", "validate"],
            ["-c", "from bngsim.convert._cli import validate_main as m; m()"],
        ),
        (["convert", "omex"], ["-c", "from bngsim.convert._cli import omex_main as m; m()"]),
        (["cache"], ["bngsim.cache"]),
    ],
)
def test_convert_and_cache_help_names_the_bngsim_command(ours, theirs):
    run = _bngsim(*ours, "--help")
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith(f"usage: bngsim {' '.join(ours)} ")
    # The same help as the old command's, but for its name.
    cmd = [sys.executable, *(["-m"] if theirs[0] != "-c" else []), *theirs, "--help"]
    old = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    old_name = old.stdout.split()[1]
    assert old_name.startswith("bngsim-")
    # argparse wraps usage lines at the name's width, so compare the words.
    assert run.stdout.split() == old.stdout.replace(old_name, f"bngsim {' '.join(ours)}").split()


@pytest.mark.parametrize(
    ("args", "usage"),
    [
        (["convert"], "usage: bngsim convert"),
        (["convert", "nope"], "usage: bngsim convert"),
        (["--quiet", "cache", "info"], "usage: bngsim cache"),
    ],
)
def test_convert_and_cache_usage_errors_exit_2(args, usage):
    run = _bngsim(*args)
    assert run.returncode == 2 and run.stdout == ""
    assert run.stderr.startswith(usage)
    error = _json_error(_bngsim(*args, "--json"), 2, "usage.invalid_argument")
    assert error["details"]["usage"].startswith(usage)


def test_the_console_script_is_this_main():
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["bngsim"] == "bngsim._cli:main"
