"""bngsim reads and writes text as UTF-8, whatever the locale (issue #1665).

The package opened text, and decoded the output of the programs it runs,
without naming an encoding, at 33 places, so Python used the locale's. A
``.bngl``, a SED-ML document or an exported file that is valid UTF-8 failed to
load where the locale is ASCII
(``UnicodeDecodeError``), and was read with other characters where it is
Latin-1 or Windows' cp1252. Of the 895 BNGL files in the parity models, 83 hold
a non-ASCII character, often an em dash in a comment.

Each now names ``encoding="utf-8"``. The guard below keeps it so: it reads the
package's source and fails on any text-mode ``open`` (the builtin, ``io.open``,
``os.fdopen``, and ``gzip``/``bz2``/``lzma`` in a text mode), ``read_text``,
``write_text``, temporary file or ``subprocess`` call decoded as text that
names no encoding. It cannot tell ``path.open()`` from the binary ``open`` of
a zip or tar member, so it does not look at ``.open`` on other objects; the
package has none. Ruff's PLW1514 says
the same, but only in preview mode, which also changes how stable rules read
this package (F822 on ``HAS_BNGL``), so the check is a test.

The behaviour is pinned end to end too: a UTF-8 ``.bngl`` and a SED-ML
document, each with an em dash, round-trip in a child process whose locale's
encoding is ASCII.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "bngsim"

pytestmark = pytest.mark.skipif(
    not PACKAGE.is_dir(), reason="no bngsim source root: the guard reads the package's source"
)

_TEMPFILES = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}
_SUBPROCESS = {"run", "Popen", "check_output", "check_call", "call"}
_COMPRESSED = {"gzip", "bz2", "lzma"}


def _mode(call: ast.Call, position: int) -> str | None:
    """The call's mode argument, if it is a string literal; None if not given."""
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value.value if isinstance(kw.value, ast.Constant) else "?"
    if len(call.args) > position:
        arg = call.args[position]
        return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else "?"
    return None


def _names_an_encoding(call: ast.Call, position: int | None) -> bool:
    """An ``encoding=`` that is not ``None``, or one given by position."""
    for kw in call.keywords:
        if kw.arg == "encoding":
            return not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
    return position is not None and len(call.args) > position


def _imported(tree: ast.AST) -> dict[str, tuple[str, str]]:
    """Local name -> (module, name) for ``from subprocess/tempfile/io import ...``."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("subprocess", "tempfile", "io"):
            for alias in node.names:
                names[alias.asname or alias.name] = (node.module, alias.name)
    return names


def unnamed_text_io(source: str) -> list[int]:
    """Lines of *source* with a text-mode open that names no encoding."""
    tree = ast.parse(source)
    imported = _imported(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            module, name = imported.get(func.id, ("builtins", func.id))
        elif isinstance(func, ast.Attribute):
            module = func.value.id if isinstance(func.value, ast.Name) else "?"
            name = func.attr
        else:
            continue
        encoding_at = None  # where an encoding may be given by position
        if name == "open" and module in ("builtins", "io") or name == "fdopen" and module == "os":
            mode = _mode(node, 1)
            text, encoding_at = mode is None or "b" not in mode, 3
        elif name == "open" and module in _COMPRESSED:
            mode = _mode(node, 1)
            text = mode is not None and "t" in mode  # the default mode is binary
        elif name in ("read_text", "write_text") and isinstance(func, ast.Attribute):
            text = True
        elif name in _TEMPFILES and module in ("tempfile", "?"):
            mode = _mode(node, 0)
            text = mode is not None and "b" not in mode  # the default mode is binary
        elif name in _SUBPROCESS and module == "subprocess":
            # Output decoded as text: BNG2.pl echoes a model's lines in its errors.
            # A value that is not a literal may be true, so it counts.
            text = any(
                kw.arg in ("text", "universal_newlines")
                and not (isinstance(kw.value, ast.Constant) and not kw.value.value)
                for kw in node.keywords
            )
        else:
            continue
        if text and not _names_an_encoding(node, encoding_at):
            out.append(node.lineno)
    return out


def test_every_text_open_in_the_package_names_an_encoding():
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for line in unnamed_text_io(path.read_text(encoding="utf-8")):
            found.append(f"{path.relative_to(ROOT)}:{line}")
    # Main had 33 of these: 19 that ruff's PLW1514 finds, 7 it cannot see (a
    # read_text on a variable whose type it does not infer), and 7 subprocess
    # calls that decode their output as text.
    assert found == [], "text opened without encoding=: " + ", ".join(found)


@pytest.mark.parametrize(
    "snippet",
    [
        "open(p)",
        "open(p, 'w')",
        "open(p, mode='r')",
        "Path(p).read_text()",
        "Path(p).write_text(t)",
        "x.read_text(errors='replace')",
        "tempfile.NamedTemporaryFile('w', suffix='.net')",
        "tempfile.NamedTemporaryFile(mode='w+')",
        "subprocess.run(cmd, capture_output=True, text=True)",
        "subprocess.check_output(cmd, universal_newlines=True)",
        "subprocess.run(cmd, text=flag)",  # not a literal: it may be true
        "from subprocess import run\nrun(cmd, text=True)",
        "from tempfile import NamedTemporaryFile as T\nT('w')",
        "open(p, encoding=None)",
        "io.open(p)",
        "os.fdopen(fd, 'w')",
        "gzip.open(p, 'rt')",
    ],
)
def test_the_guard_finds_a_text_open_without_an_encoding(snippet):
    assert unnamed_text_io(snippet) == [snippet.count("\n") + 1]


@pytest.mark.parametrize(
    "snippet",
    [
        "open(p, encoding='utf-8')",
        "open(p, 'rb')",
        "open(p, mode='wb')",
        "Path(p).read_text(encoding='utf-8')",
        "Path(p).read_bytes()",
        "tempfile.NamedTemporaryFile(suffix='.so')",  # binary by default
        "tempfile.NamedTemporaryFile('w', encoding='utf-8')",
        "os.open(p, os.O_RDONLY)",
        "zipfile.ZipFile(p).open('a')",
        "subprocess.run(cmd, capture_output=True)",  # bytes
        "subprocess.run(cmd, text=True, encoding='utf-8', errors='replace')",
        "subprocess.run(cmd, text=False)",
        "open(p, 'r', -1, 'utf-8')",  # the encoding by position
        "os.fdopen(fd, 'wb')",
        "gzip.open(p)",  # binary by default
        "gzip.open(p, 'rt', encoding='utf-8')",
        "tarfile.open(p)",
    ],
)
def test_the_guard_leaves_binary_and_named_opens_alone(snippet):
    assert unnamed_text_io(snippet) == []


# --- end to end, under an ASCII locale ---------------------------------------------

BNGL = """# first-order decay — a comment with an em dash
begin model
begin parameters
  k 1
end parameters
begin molecule types
  A()
end molecule types
begin seed species
  A() 10
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() -> 0 k
end reaction rules
end model
"""

CHILD = textwrap.dedent(
    """
    import locale, math, sys, warnings
    warnings.simplefilter("ignore")
    import bngsim
    from bngsim.convert import parse_bngl_protocol, read_sedml_protocol, write_sedml_protocol
    enc = locale.getpreferredencoding(False).lower().replace("-", "")  # 3.10 has no getencoding
    assert enc in ("usascii", "ascii", "ansi_x3.41968"), enc
    path = sys.argv[1]
    model = bngsim.Model.from_bngl(path)
    r = bngsim.Simulator(model, method="ode").run(t_span=(0, 1), n_points=2)
    assert abs(r.observables["Atot"][-1] - 10 * math.exp(-1)) < 1e-5
    # A protocol whose label holds the em dash, written to SED-ML and read back,
    # read from a path given as text and as a Path (two branches of the reader).
    import pathlib
    spec = parse_bngl_protocol(path.replace(".bngl", "_actions.bngl"))
    by_path = parse_bngl_protocol(pathlib.Path(path.replace(".bngl", "_actions.bngl")))
    assert by_path.experiments[0].label == spec.experiments[0].label
    out = path.replace(".bngl", ".sedml")
    write_sedml_protocol(spec, out)
    back = read_sedml_protocol(out)
    label = back.experiments[0].label
    assert label == spec.experiments[0].label == "decay\\u2014a", label
    print("ok")
    """
)


def _ascii_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "LANG", "PYTHONUTF8"))}
    env.update({"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"})
    return env


CHILD_BNG2_ERROR = textwrap.dedent(
    """
    import sys, warnings
    warnings.simplefilter("ignore")
    import bngsim
    try:
        bngsim.Model.from_bngl(sys.argv[1])
    except UnicodeDecodeError as e:
        print("decode error:", e); sys.exit(1)
    except bngsim.ModelError as e:
        # ascii(): the message holds the em dash, and this locale cannot print it.
        print("refused:", ascii(str(e))); sys.exit(0)
    print("loaded"); sys.exit(2)
    """
)


@pytest.mark.skipif(
    not (os.environ.get("BNGPATH") and shutil.which("perl")), reason="needs BNG2.pl on BNGPATH"
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-specific: Windows has no ASCII C locale, it reads its ANSI code page",
)
def test_bng2pls_error_with_a_non_ascii_line_is_reported_not_a_decode_error(tmp_path):
    # BNG2.pl's error names the line it could not parse, em dash and all; read
    # in the locale's ASCII that output raised UnicodeDecodeError instead.
    bad = tmp_path / "bad.bngl"
    bad.write_text(BNGL.replace("  k 1\n", "  k 1 \u2014 not a value\n"), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-c", CHILD_BNG2_ERROR, str(bad)],
        env=_ascii_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    # BNG2.pl's own refusal, with the line it refused: not some other failure.
    assert "ABORT" in done.stdout and "\\u2014 not a value" in done.stdout, done.stdout


@pytest.mark.skipif(
    not (os.environ.get("BNGPATH") and shutil.which("perl")), reason="needs BNG2.pl on BNGPATH"
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX-specific: Windows has no ASCII C locale, it reads its ANSI code page",
)
def test_a_utf8_bngl_and_its_sedml_round_trip_under_an_ascii_locale(tmp_path):
    bngl = tmp_path / "decay.bngl"
    bngl.write_text(BNGL, encoding="utf-8")
    actions = tmp_path / "decay_actions.bngl"
    actions.write_text(
        "begin model\nend model\ngenerate_network()\n"
        'simulate({method=>"ode",t_end=>1,n_steps=>2,suffix=>"decay—a"})\n',
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(bngl)],
        env=_ascii_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().endswith("ok")


@pytest.mark.skipif(
    not (os.environ.get("BNGPATH") and shutil.which("perl")), reason="needs BNG2.pl on BNGPATH"
)
def test_a_bngl_with_a_byte_that_is_not_utf8_reaches_bng2pl_as_it_is(tmp_path):
    # A file an older editor saved as Latin-1: 0xB5 (the micro sign) in two
    # comments is no UTF-8. BNG2.pl reads bytes and skips comments, so the model
    # loads; read strictly as UTF-8 it was a bare UnicodeDecodeError before
    # BNG2.pl ran. The comment on the parameter's line is kept in the copy BNG2.pl
    # is given, and in the cache key, so both must carry the byte too.
    import bngsim

    legacy = tmp_path / "legacy.bngl"
    text = BNGL.encode("utf-8").replace(b"\xe2\x80\x94", b"\xb5M")
    legacy.write_bytes(text.replace(b"  k 1\n", b"  k 1 # per \xb5M\n"))
    assert legacy.read_bytes().count(b"\xb5") == 2
    model = bngsim.Model.from_bngl(str(legacy))
    assert model.n_species == 1


def test_compiler_output_is_read_as_utf8_off_windows():
    from bngsim import _codegen_toolchain

    expected = "mbcs" if os.name == "nt" else "utf-8"
    assert expected == _codegen_toolchain.TOOL_OUTPUT_ENCODING
