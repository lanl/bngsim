"""What BNGsim says at run time cites no issue or pull request.

An error, a warning, a notice, a log line, the command's output and
``capabilities()["missing"]`` are read by a user or an agent who has no use for
an issue number: the text has to say what happened and what to do. Issue
numbers belong in the changelog and in code comments, as the user docs already
have it (``test_docs_api_examples.test_user_docs_carry_no_issue_references``).

What is left alone, and how this test tells it apart:

* comments, which no parser here sees;
* docstrings, Python's (a bare string statement, which also covers an
  attribute's docstring) and pybind11's (``src/_bngsim_core.cpp`` is all
  bindings and their docstrings, and is not scanned);
* the text of generated code: a match inside a C comment (``/* ... */``) of the
  C that codegen emits, tracked across a file's strings in source order, or on
  a ``#`` comment line of a generated file (BNGL, a manifest).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

# The docs test's pattern: "#1846", "GH-1846", "issue 111", "PR 12", ".../issues/1846",
# "-issue-111". A "# " comment marker and an HTML "&#123;" do not match.
_ISSUE_REFERENCE = re.compile(
    r"(?<!&)#\d+\b"
    r"|\bGH-\d+\b"
    r"|\b(?:issues?|PRs?|pull requests?)\s+#?\d+\b"
    r"|/(?:issues|pull)/\d+"
    r"|\bissue-\d+\b",
    re.IGNORECASE,
)


def _strings(tree: ast.AST) -> list[ast.Constant]:
    """Every string constant that is not a docstring, in source order."""
    bare = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in bare
    ]
    return sorted(found, key=lambda n: (n.lineno, n.col_offset))


def _python_references(path: Path) -> list[str]:
    out = []
    in_c_comment = False
    for node in _strings(ast.parse(path.read_text(encoding="utf-8"))):
        text = node.value
        # Where each C comment opens and closes in this string, carried on from
        # the strings before it.
        state, pos, spans = in_c_comment, 0, []
        while True:
            mark = text.find("*/" if state else "/*", pos)
            spans.append((pos, len(text) if mark == -1 else mark, state))
            if mark == -1:
                break
            pos, state = mark + 2, not state
        in_c_comment = state
        for match in _ISSUE_REFERENCE.finditer(text):
            at = match.start()
            if any(lo <= at <= hi and inside for lo, hi, inside in spans):
                continue
            line = text[text.rfind("\n", 0, at) + 1 : at]
            if line.lstrip().startswith("#"):
                continue  # a comment line of a generated file
            out.append(f"{path.relative_to(_ROOT)}:{node.lineno}: {match.group(0)!r}")
    return out


# C++ read left to right, one token at a time, so a "//" in a string, a '"' char
# and a quoted word inside a comment are each what they are. Only the string
# literals are kept; a raw string R"(...)" is not used in these files.
_CPP_TOKEN = re.compile(
    r"(?P<comment>//[^\n]*|/\*.*?\*/)"
    r"|(?P<directive>^[ \t]*#[^\n]*)"
    r"|(?P<char>'(?:[^'\\\n]|\\.)+')"
    r'|(?P<string>"(?:[^"\\\n]|\\.)*")',
    re.DOTALL | re.MULTILINE,
)


def _cpp_references(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    out = []
    for token in _CPP_TOKEN.finditer(source):
        if token.lastgroup != "string":
            continue
        for match in _ISSUE_REFERENCE.finditer(token.group(0)):
            line = source.count("\n", 0, token.start()) + 1
            out.append(f"{path.relative_to(_ROOT)}:{line}: {match.group(0)!r}")
    return out


def test_python_messages_carry_no_issue_references() -> None:
    files = sorted((_ROOT / "python" / "bngsim").rglob("*.py"))
    assert len(files) > 50
    found = [hit for path in files for hit in _python_references(path)]
    assert not found, "issue references in run-time text:\n" + "\n".join(found)


def test_cpp_messages_carry_no_issue_references() -> None:
    files = sorted((_ROOT / "src").glob("*.cpp")) + sorted((_ROOT / "include").rglob("*.hpp"))
    files = [p for p in files if p.name != "_bngsim_core.cpp"]  # bindings and their docstrings
    assert len(files) > 10
    found = [hit for path in files for hit in _cpp_references(path)]
    assert not found, "issue references in run-time text:\n" + "\n".join(found)


def test_the_scan_sees_what_it_is_meant_to(tmp_path: Path) -> None:
    """A message is caught; a comment, a docstring, a C comment of generated
    code and a generated '#' line are not."""
    py = tmp_path / "m.py"
    py.write_text(
        '"""Module docstring, issue #1."""\n'
        "X = 1\n"
        '"""Attribute docstring, GH #2."""\n'
        "# a comment, issue #3\n"
        "C = ('/* generated (issue #4)'\n"
        "     '   still in the comment (GH #5) */')\n"
        "B = '# generated BNGL comment (GH #6)\\n'\n"
        "def f(): raise ValueError('refused (issue #7)')\n",
        encoding="utf-8",
    )
    global _ROOT
    saved, _ROOT = _ROOT, tmp_path
    try:
        assert [h.split(": ", 1)[1] for h in _python_references(py)] == ["'issue #7'"]
        cpp = tmp_path / "m.cpp"
        cpp.write_text(
            "#include <x> // (GH #1)\n"
            "/* a comment, issue #2 */\n"
            'auto url = "http://x//y";  // issue #3\n'
            "char q = '\"'; // GH #5\n"
            'throw std::runtime_error("refused (issue #4)");\n',
            encoding="utf-8",
        )
        assert [h.split(": ", 1)[1] for h in _cpp_references(cpp)] == ["'issue #4'"]
    finally:
        _ROOT = saved
