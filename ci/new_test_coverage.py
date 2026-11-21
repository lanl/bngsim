#!/usr/bin/env python3
"""Warn when a pull request adds a test file that no check on it runs (#1764).

    python3 ci/new_test_coverage.py --reports DIR --base SHA [--head SHA]

Several workflows run a hand-picked list of test files, and only when certain
paths change: mir.yml, windows-klu.yml, windows-nfsim.yml, windows-tail.yml.
The python-tests legs build without some of what those have (MIR, KLU on
Windows, MSVC), so the tests for it skip there. A test file a pull request adds
is on none of those lists unless someone edits the workflow, so a file that
only does real work under one of them skips in python-tests, runs nowhere else,
and every check is green. Review caught one such MIR check on #1359.

This reads the JUnit reports every python-tests leg wrote, and, for each test
file the change adds, asks two things:

* Did every test in it skip on every leg? A file not collected at all counts:
  nothing in it ran. A file with any test that ran is left alone; partly
  skipped files are out of scope.
* Does another workflow that runs on this pull request run it? A workflow runs
  on it when its ``on: pull_request`` trigger fires for the files the change
  touches (its ``paths:`` or ``paths-ignore:`` filter, read in order, ``!``
  entries included). It runs the file when one of its pytest commands names
  it, by a glob, a directory or as written, or runs the whole of python/tests,
  and no ``--deselect``/``--ignore`` argument leaves it out. forced-jit.yml and
  sanitizers.yml run the whole suite nightly, and on a pull request only when
  their own files change.

What a text read cannot settle never counts as running the file: a pytest
argument that is a shell variable or array or a matrix value, a ``-k``/``-m``
expression, a trigger filter this does not read (``branches:``, or ``types:``
without ``synchronize``). The warning names such a workflow as one it could not
read. Dispatch inputs (``inputs.pytest_args``, and an env var set from one) are
empty on a pull request, so they are read as nothing.

A file that skipped everywhere and that no running workflow runs gets a
``::warning::``. So does one that only a list runs, where that workflow runs on
this pull request for its other changes and its paths filter does not match the
file: the next change to the file alone would run it nowhere. A warning, not a
failure: a new file may skip everywhere for a reason (a quarantine, an optional
dependency no leg installs). Nothing here fails the job: a missing or
unreadable report is itself a warning, and so is a ``git diff`` that fails.

Stdlib only, like the rest of ci/, plus ``git`` for the changed files.
"""

from __future__ import annotations

import argparse
import fnmatch
import posixpath
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_file_counts import Counts, count_reports  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: The workflow whose legs' reports are read; its own pytest commands run the
#: whole suite, so they name nothing.
SUITE_WORKFLOW = "python-tests.yml"

#: A shard's artifact directory: ``python-tests-log-<leg>-<shard>``.
SHARD_DIR = re.compile(r"^python-tests-log-(?P<leg>.+)-(?P<shard>\d+)$")

#: pytest options whose argument is a path or node id it does not run.
_EXCLUDING = frozenset({"--deselect", "--ignore", "--ignore-glob"})
#: pytest options that narrow a run by an expression this does not evaluate.
_NARROWING = frozenset({"-k", "-m"})
#: Other pytest options that take a separate argument, which is not a test path.
_TAKES_ARG = frozenset(
    {"-p", "-c", "-o", "-n", "-W", "--rootdir", "--confcutdir", "--basetemp", "--junitxml"}
    | {"--dist", "--tb", "--durations", "--maxfail", "--max-worker-restart", "--cov"}
    | {"--cov-report", "--override-ini", "--config-file", "--log-file", "--import-mode"}
)
#: Shell words that end a command: what follows is another command.
_COMMAND_ENDS = frozenset({"|", "||", "&&", ";", "&", "|&"})

#: pytest's default ``python_files`` (pyproject.toml sets none; a test pins it).
_TEST_FILE = re.compile(r"^python/tests/(?:.+/)?(?:test_[^/]*|[^/]*_test)\.py$")

#: Characters of GitHub's filter patterns this does not read as GitHub does:
#: ``?`` is zero or one of the character before it there (one character to
#: bash), ``+`` is one or more of it, and ``[...]`` is a class. A filter that
#: uses one is not evaluated: a ``?`` read as bash reads it can say a workflow
#: runs on a change it does not run on.
_FILTER_UNREAD = re.compile(r"[?+\[\]]")

#: A ``${{ ... }}`` expression, swapped for one word before the shell split.
_EXPR = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
#: A word in quotes that holds a glob character, which the shell leaves as is:
#: a whole word, or an option's argument after ``=`` (``--ignore='x*.py'``).
#: A shell expansion (``"${files[@]}"``) is no glob: it is left to be read as one.
_QUOTED_GLOB = re.compile(r"""(?<![^\s=])(['"])([^'"\s$]*[*?[][^'"\s$]*)\1(?!\S)""")
#: An ``--ignore`` path in ``Workflow.excluded``. pytest leaves it out only where
#: it finds it walking down a directory it was given, as with ``--ignore-glob``.
_IGNORE = "ignore:"
#: A ``--deselect`` argument in ``Workflow.excluded``, as written: pytest leaves
#: out each test whose node id starts with it, wherever the test came from. An
#: entry with no mark is a ``--deselect`` glob the shell expands first.
_PREFIX = "prefix:"
#: The mark a quoted glob carries through the shell split, so that an option's
#: argument keeps it and a test path does not. Its text follows in hex, so that
#: no character of it (``#``, ``;``, ``|``) is read as shell syntax.
_QUOTED = "@@quoted:"
#: An ``--ignore-glob`` pattern in ``Workflow.excluded``: pytest matches it with
#: fnmatch, whose ``*`` and ``?`` cross directories and whose ``[...]`` is a class.
_FNMATCH = "fnmatch:"
#: A dispatch input: empty on a pull request.
_INPUT = re.compile(r"^(?:github\.event\.)?inputs\.[\w-]+$")


def glob_re(pattern: str) -> re.Pattern[str]:
    """A path glob: ``*`` and ``?`` stay within one directory, ``**`` crosses
    directories, as GitHub reads ``*`` and ``**`` in a paths filter. A pytest
    argument reaches it with ``**`` already read as ``*``, as bash without
    globstar reads it. GitHub's other filter characters are not read here (see
    :data:`_FILTER_UNREAD`)."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def matches(path: str, pattern: str) -> bool:
    return bool(glob_re(pattern).match(path))


# --- the workflows ----------------------------------------------------------


@dataclass
class Workflow:
    """What one workflow other than python-tests runs, as far as its text says."""

    name: str
    selectors: list[str] = field(default_factory=list)  # globs over file paths
    node_ids: list[str] = field(default_factory=list)  # files a node id names
    whole: bool = False  # a command runs the whole of python/tests
    excluded: list[str] = field(default_factory=list)  # --deselect/--ignore files
    unreadable: list[str] = field(default_factory=list)  # why its commands are unknown
    #: Whether it runs on this pull request: True, False, or None (cannot tell).
    fires: bool | None = None
    why_not: str = ""  # why it does not (or may not) run on this pull request
    #: The workflow file's text, when it was read from one.
    text: str | None = field(default=None, repr=False)

    def names(self, path: str) -> bool:
        return path in self.node_ids or any(matches(path, g) for g in self.selectors)

    def runs(self, path: str) -> bool:
        """It runs on the pull request, its commands read in full, and they run ``path``."""
        if self.fires is not True or self.unreadable:
            return False
        if any(_deselects(path, g) for g in self.excluded):
            return False
        roots = ["python/tests"] if self.whole else []
        roots += [g[:-3] for g in self.selectors if g.endswith("/**") and matches(path, g)]
        if not roots:
            # pytest collects a path it is given even where --ignore or
            # --ignore-glob matches it.
            return path in self.node_ids or any(matches(path, g) for g in self.selectors)
        # They leave out what pytest finds walking down a directory it was
        # given, below that directory. A path given as well that lies under
        # that directory is dropped as a duplicate of it (pytest 9), so they
        # apply to it like the rest; of two directories, the outer is walked.
        outer = min(roots, key=len)
        return not any(_ignores(path, outer, g) for g in self.excluded)

    def fires_for(self, path: str) -> bool:
        """Its trigger fires for a change to ``path`` alone (true where no
        workflow text was read, so there is no trigger to ask). Asked only of a
        workflow that runs on this pull request, whose trigger was read."""
        return self.text is None or fires_on(self.text, [path])[0] is True


def _deselects(path: str, entry: str) -> bool:
    """``entry``, a ``--deselect`` entry of ``Workflow.excluded``, leaves ``path`` out."""
    if entry.startswith(_PREFIX):
        return path.startswith(entry[len(_PREFIX) :])
    if entry.startswith((_FNMATCH, _IGNORE)):
        return False
    # A glob the shell expands: a file it names, or a directory, whose tests'
    # node ids all start with it.
    parts = path.split("/")
    return any(matches("/".join(parts[:k]), entry) for k in range(1, len(parts) + 1))


def _ignores(path: str, root: str, entry: str) -> bool:
    """``entry``, an ``--ignore`` or ``--ignore-glob`` entry of ``Workflow.excluded``,
    leaves ``path`` out of a walk down the directory ``root``. pytest asks it of
    each directory on the way down as well, so one that matches a directory
    leaves out everything under it; ``root`` itself, a path it was given, is
    not asked."""
    parts = path.split("/")
    below = len(root.split("/"))
    tops = ["/".join(parts[:k]) for k in range(below + 1, len(parts) + 1)]
    if entry.startswith(_FNMATCH):
        glob = entry[len(_FNMATCH) :]
        return any(fnmatch.fnmatchcase(top, glob) for top in tops)
    if entry.startswith(_IGNORE):
        ignored = entry[len(_IGNORE) :]
        if ignored.endswith("/**"):
            return ignored[:-3] in tops
        return any(matches(top, ignored) for top in tops)
    return False


def _strip_comments(text: str) -> str:
    """Whole-line ``#`` comments dropped: these files explain themselves in
    comments that name the very files being looked for."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def pytest_commands(text: str) -> list[str]:
    """Each pytest command in a workflow, continuation lines joined."""
    lines = _strip_comments(text).splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        # ``python -m pytest``, or ``pytest`` as the line's first word: not a
        # step name or an input's description that mentions pytest.
        if not re.search(r"-m\s+pytest\b|^\s*(?:-\s*run:\s*)?pytest\s", line):
            continue
        block = [line]
        while block[-1].rstrip().endswith("\\") and i + len(block) < len(lines):
            block.append(lines[i + len(block)])
        out.append(" ".join(ln.rstrip().rstrip("\\") for ln in block))
    return out


def dispatch_env(text: str) -> set[str]:
    """Env vars the workflow sets from a dispatch input: empty on a pull request."""
    pat = r"^\s*(\w+):\s*\$\{\{\s*(?:github\.event\.)?inputs\.[\w-]+\s*\}\}\s*$"
    return set(re.findall(pat, text, re.M))


def _selector(path: str) -> str:
    """A path argument as a glob over file paths: a file or glob as written, a
    directory as everything under it."""
    path = path.rstrip("/")
    if path.endswith(".py") or any(c in path for c in "*?["):
        return path
    return f"{path}/**"


def read_command(command: str, wf: Workflow, empty: Iterable[str] = ()) -> None:
    """Add one pytest command's selectors, exclusions and unknowns to ``wf``.

    ``empty`` names the shell variables that are empty on a pull request.
    ``python/tests`` is the whole suite; a node id selects its file.
    """
    empty = set(empty)

    def word(m: re.Match[str]) -> str:
        return " @@input@@ " if _INPUT.match(m[1]) else f"@@expr:{m[1].replace(' ', '')}@@"

    # A glob in quotes reaches pytest as written, and pytest does not expand
    # it: as a test path it names no file (pytest stops on the path it cannot
    # find). As an option's argument (``--ignore-glob 'x*.py'``) it is the
    # pattern pytest itself matches, so it is marked here, not dropped. After
    # an ``=`` the mark stays joined to its option.
    def mark(m: re.Match[str]) -> str:
        joined = m.start() > 0 and m.string[m.start() - 1] == "="
        return f"{'' if joined else ' '}{_QUOTED}{m[2].encode().hex()}@@ "

    command = _QUOTED_GLOB.sub(mark, command)

    def literal(tok: str) -> str:
        """A word as pytest receives it: a marked word's own text."""
        if not tok.startswith(_QUOTED):
            return tok
        try:
            return bytes.fromhex(tok[len(_QUOTED) : -2]).decode()
        except ValueError:  # the mark's own text, written in a command
            return tok

    lex = shlex.shlex(_EXPR.sub(word, command), posix=True, punctuation_chars=";&|<>")
    lex.whitespace_split = True
    try:
        tokens = list(lex)
    except ValueError as e:  # an unclosed quote
        wf.unreadable.append(f"it does not split as shell words ({e})")
        return
    start = next((i for i, t in enumerate(tokens) if t == "pytest" or t.endswith("/pytest")), None)
    if start is None:
        return
    it = iter(tokens[start + 1 :])
    for tok in it:
        if tok in _COMMAND_ENDS:
            break
        whole_quoted = tok.startswith(_QUOTED) and literal(tok).startswith("-")
        if whole_quoted:
            tok = literal(tok)  # a whole option in quotes (``"--ignore-glob=x*.py"``)
        if tok in {">", ">>", ">&", "<"}:
            next(it, None)
            continue
        if tok == "@@input@@" or tok.lstrip("$").strip("{}") in empty:
            continue
        opt, eq, val = tok.partition("=")
        if opt in _EXCLUDING:
            raw = val if eq else next(it, "")
            arg = literal(raw)
            if "$" in arg or "@@expr" in arg:
                wf.unreadable.append(f"`{opt} {arg}` is not read")
            elif opt == "--ignore-glob":
                # Whatever it starts with: ``*/tests/test_h*.py`` matches too.
                # pytest makes it absolute first, which drops ``./``, ``//``
                # and a trailing ``/``.
                wf.excluded.append(_FNMATCH + posixpath.normpath(arg))
            elif raw.startswith(_QUOTED) or whole_quoted or (eq and re.search(r"[*?[]", arg)):
                # --ignore and --deselect take a path or a node id as written:
                # a glob the shell leaves alone (in quotes, or after ``=``)
                # names none, and leaves nothing out.
                pass
            elif opt == "--ignore":
                # Made absolute by pytest as well (``./python/tests/x.py``).
                path = posixpath.normpath(arg)
                if path.startswith("python/tests"):
                    wf.excluded.append(_IGNORE + _selector(path))
            elif "::" in arg:
                pass  # a node id deselects one test, not the file
            elif re.search(r"[*?[]", arg):
                # After a space the shell expands it, and the first match is
                # the argument: each match is read as left out.
                if arg.startswith("python/tests"):
                    wf.excluded.append(_selector(arg))
            else:
                # A prefix of the node ids as written, whatever it starts with.
                wf.excluded.append(_PREFIX + arg)
            continue
        if opt in _NARROWING:
            if not eq:
                next(it, None)
            wf.unreadable.append(f"it narrows the run with `{opt}`, which is not evaluated")
            continue
        if opt in _TAKES_ARG:
            if not eq:
                next(it, None)
            continue
        if tok.startswith("-"):
            continue
        if tok.startswith(_QUOTED):
            # A quoted node id whose test id holds the glob character
            # (``'f.py::test[a-1]'``) selects its file; a quoted glob path
            # names none.
            file, sep, _ = literal(tok).partition("::")
            if sep and file.startswith("python/tests") and not re.search(r"[*?[]", file):
                wf.node_ids.append(file.rstrip("/"))
            continue
        if "$" in tok or "@@expr" in tok:
            shown = tok.replace("@@expr:", "${{ ").replace("@@", " }}")
            wf.unreadable.append(f"it passes `{shown}`, which is not read")
            continue
        if not tok.startswith("python/tests"):
            continue
        path = tok.split("::", 1)[0].rstrip("/")
        if path == "python/tests":
            wf.whole = True
        elif "::" in tok:
            wf.node_ids.append(path)
        else:
            # The shell's glob, not GitHub's: without globstar, bash reads ``**``
            # as ``*``, which stays within one directory.
            wf.selectors.append(_selector(re.sub(r"\*\*+", "*", path)))


# --- does a workflow run on this pull request -------------------------------


def _block(lines: list[str], i: int) -> list[str]:
    """The lines nested under ``lines[i]``."""
    indent = len(lines[i]) - len(lines[i].lstrip())
    out = []
    for nxt in lines[i + 1 :]:
        if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
            break
        out.append(nxt)
    return out


def _items(lines: list[str], i: int, every: list[str]) -> list[str]:
    """The list under key ``lines[i]``: block items, a flow list, or an alias
    to a list anchored anywhere in ``every``."""
    rest = lines[i].split(":", 1)[1].strip()
    alias = re.fullmatch(r"\*([\w-]+)", rest)
    if alias:
        for j, ln in enumerate(every):
            if re.match(rf"^\s*[\w-]+:\s*&{re.escape(alias[1])}\s*$", ln):
                return _items(every, j, every)
        return []
    rest = re.sub(r"^&[\w-]+\s*", "", rest)
    if rest.startswith("["):
        return [s.strip().strip("\"'") for s in rest.strip("[]").split(",") if s.strip()]
    out = []
    for ln in _block(lines, i):
        m = re.match(r"""^\s*-\s*["']?([^"'#]+?)["']?\s*(?:#.*)?$""", ln)
        if m:
            out.append(m[1])
    return out


def pull_request_trigger(text: str) -> dict[str, list[str]] | None:
    """``None`` when the workflow has no ``pull_request`` trigger, else its
    filters by key (``paths``, ``paths-ignore``, ``branches``, ``types`` ...),
    empty when it fires on every pull request."""
    lines = _strip_comments(text).splitlines()
    on = next((i for i, ln in enumerate(lines) if re.match(r"^[\"']?on[\"']?:", ln)), None)
    if on is None:
        return None
    inline = lines[on].split(":", 1)[1].strip()
    if inline:
        return {} if re.search(r"\bpull_request\b(?!_)", inline) else None
    block = _block(lines, on)
    for j, ln in enumerate(block):
        if not re.match(r"^\s+pull_request:", ln):
            continue
        depth = len(ln) - len(ln.lstrip())
        sub = _block(block, j)
        filters = {}
        for k, s in enumerate(sub):
            key = re.match(r"^\s*([\w-]+):", s)
            if key and len(s) - len(s.lstrip()) > depth:
                filters[key[1]] = _items(sub, k, lines)
        return filters
    return None


def fires_on(text: str, changed: list[str]) -> tuple[bool | None, str]:
    """Whether a workflow's ``pull_request`` trigger fires for ``changed``, and
    if not (or not knowably), why."""
    trig = pull_request_trigger(text)
    if trig is None:
        return False, "it has no pull_request trigger"
    if "branches" in trig or "branches-ignore" in trig:
        return None, "its pull_request trigger has a branches filter, which is not read"
    if "types" in trig and "synchronize" not in trig["types"]:
        return None, "its pull_request types leave out synchronize"
    for key in ("paths", "paths-ignore"):
        odd = next((p for p in trig.get(key, []) if _FILTER_UNREAD.search(p)), None)
        if odd is not None:
            return None, f"its {key} filter has a pattern that is not read (`{odd}`)"
    if "paths" in trig:
        for f in changed:
            on = False
            for p in trig["paths"]:  # in order: a later ``!`` entry takes a file back out
                negated = p.startswith("!")
                if matches(f, p[1:] if negated else p):
                    on = not negated
            if on:
                return True, ""
        return False, "its paths filter matches no file the pull request changes"
    if "paths-ignore" in trig:
        if any(not any(matches(f, p) for p in trig["paths-ignore"]) for f in changed):
            return True, ""
        return False, "its paths-ignore filter covers every file the pull request changes"
    return True, ""


def read_workflows(changed: list[str], root: Path = WORKFLOWS) -> dict[str, Workflow]:
    """Every workflow but python-tests that has a pytest command over python/tests."""
    out: dict[str, Workflow] = {}
    for path in sorted(root.glob("*.yml")):
        if path.name == SUITE_WORKFLOW:
            continue
        # A byte that is not UTF-8 is replaced, not raised on: the job warns only.
        text = path.read_text(encoding="utf-8", errors="replace")
        wf = Workflow(path.name, text=text)
        for command in pytest_commands(text):
            read_command(command, wf, dispatch_env(text))
        if not (wf.selectors or wf.node_ids or wf.whole or wf.unreadable):
            continue
        wf.fires, wf.why_not = fires_on(text, changed)
        out[wf.name] = wf
    return out


# --- the legs' reports ------------------------------------------------------


def read_reports(
    reports: Path, root: Path | None = None
) -> tuple[dict[str, Counts], list[tuple[str, str]]]:
    """Each python-tests leg's per-file counts, its shards' reports summed, and
    ``(shard, reason)`` for each report that could not be read. A missing
    directory is no reports, not an error."""
    legs: dict[str, Counts] = {}
    unread: list[tuple[str, str]] = []
    if not reports.is_dir():
        return legs, unread
    for d in sorted(p for p in reports.iterdir() if p.is_dir()):
        m = SHARD_DIR.match(d.name)
        if not m:
            continue
        counts = legs.setdefault(m["leg"], {})
        for xml in sorted(d.rglob("*.xml")):
            try:
                one = count_reports([xml], root)
            # LookupError: an encoding the declaration names that Python lacks;
            # ValueError: bytes that are not of the declared encoding.
            except (ET.ParseError, OSError, LookupError, ValueError) as e:
                unread.append((f"{m['leg']} {m['shard']}", f"{xml.name}: {e}"))
                continue
            for f, c in one.items():
                entry = counts.setdefault(f, {"collected": 0, "skipped": 0})
                entry["collected"] += c["collected"]
                entry["skipped"] += c["skipped"]
    return legs, unread


def expected_shards(text: str) -> set[str]:
    """``"<leg label> <shard>"`` for every job of python-tests.yml's suite matrix."""
    body = _strip_comments(text)
    job = re.search(r"^  python-tests:\n(.*?)(?=^  [\w-]+:\s*$|\Z)", body, re.M | re.S)
    if not job:
        return set()
    shards = re.search(r"^\s+shard:\s*\[([^\]]*)\]", job[1], re.M)
    labels = re.findall(r"^\s+label:\s*[\"']?([\w.-]+)", job[1], re.M)
    if not shards:
        return set()
    return {f"{lab} {s.strip()}" for lab in labels for s in shards[1].split(",") if s.strip()}


def missing_shards(reports: Path, expected: set[str], unread: Iterable[str] = ()) -> list[str]:
    """The expected shards with no report read: none downloaded, or unreadable."""
    present = set()
    if reports.is_dir():
        for d in reports.iterdir():
            m = SHARD_DIR.match(d.name)
            if m and d.is_dir() and any(d.rglob("*.xml")):
                present.add(f"{m['leg']} {m['shard']}")
    return sorted((expected - present) | set(unread))


def ran_nothing(path: str, legs: dict[str, Counts]) -> bool:
    """Every test in ``path`` skipped on every leg, or the file was not collected."""
    for counts in legs.values():
        c = counts.get(path, {"collected": 0, "skipped": 0})
        if c["collected"] > c["skipped"]:
            return False
    return True


# --- the check --------------------------------------------------------------


def changed_files(base: str, head: str, cwd: Path = REPO_ROOT) -> list[tuple[str, str]]:
    """``(status, path)`` for each file ``head`` changes over ``base``. A rename
    is a deletion and an addition (``--no-renames``), so a renamed file is
    checked as new: whatever ran it under its old name will not under the new
    one. ``-z``, so a path with a space or a non-ASCII name comes back whole."""
    out = subprocess.run(
        ["git", "diff", "--no-renames", "--name-status", "-z", base, head],
        check=True,
        capture_output=True,
        cwd=cwd,
    ).stdout.decode("utf-8", "surrogateescape")
    fields = out.split("\0")
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields) - 1, 2)]


def added_tests(changed: Iterable[tuple[str, str]]) -> list[str]:
    """The python/tests test files among ``changed`` that the change adds."""
    return sorted(f for s, f in changed if s == "A" and _TEST_FILE.match(f))


def added_test_files(base: str, head: str, cwd: Path = REPO_ROOT) -> list[str]:
    """The python/tests test files ``head`` adds over ``base``."""
    return added_tests(changed_files(base, head, cwd))


def _what_ran(path: str, legs: dict[str, Counts]) -> str:
    per = {leg: legs[leg].get(path, {}).get("collected", 0) for leg in sorted(legs)}
    names = ", ".join(per)
    if len(set(per.values())) == 1:
        n = next(iter(per.values()))
        if n == 0:
            return f"no python-tests leg ({names}) collects it"
        tests = "its one test skips" if n == 1 else f"all {n} of its tests skip"
        return f"{tests} on every python-tests leg ({names})"
    each = "; ".join(
        f"{leg}: {f'{n} skipped' if n else 'not collected'}" for leg, n in per.items()
    )
    return f"nothing in it runs on any python-tests leg ({each})"


def _note(w: Workflow, path: str) -> str:
    """Why workflow ``w`` does not count as running ``path``, or ``""``."""
    if w.fires is None:
        return f"Whether {w.name} runs on this pull request is not known: {w.why_not}."
    if w.fires is False:
        if w.whole or w.names(path):
            what = "runs the whole suite" if w.whole else "names it"
            return f"{w.name} {what}, but not on this pull request: {w.why_not}."
        return ""
    if w.unreadable:
        return (
            f"{w.name} runs on this pull request, but its pytest list could not be read: "
            f"{w.unreadable[0]}."
        )
    if w.whole or w.names(path):
        return (
            f"{w.name} runs on this pull request, but leaves it out with --deselect or --ignore."
        )
    return ""


def _trigger_gap(w: Workflow, path: str) -> tuple[str, str] | None:
    """Why a change to ``path`` alone does not fire ``w``, and what would make
    it, when that is not simply a paths filter that matches nothing of it."""
    trig = pull_request_trigger(w.text or "") or {}
    if "paths" in trig:
        last = None  # the last entry that matches it decides, in order
        for p in trig["paths"]:
            negated = p.startswith("!")
            if matches(path, p[1:] if negated else p):
                last = p
        if last is not None and last.startswith("!"):
            return (
                f"its paths filter takes it back out with `{last}`",
                "Take it out of that `!` entry",
            )
        return None
    hit = next((p for p in trig.get("paths-ignore", []) if matches(path, p)), None)
    if hit is not None:
        return (
            f"its paths-ignore filter leaves it out (`{hit}`)",
            "Take it out of that paths-ignore filter",
        )
    return None


def _late_message(
    path: str, legs: dict[str, Counts], late: list[Workflow], missing: list[str]
) -> str:
    one = len(late) == 1
    unread = f" (no report was read from {', '.join(missing)})" if missing else ""
    head = (
        f"{path} is new, and {_what_ran(path, legs)}{unread}. {', '.join(w.name for w in late)} "
        f"{'runs' if one else 'run'} it on this pull request only for the pull request's "
        "other changes"
    )
    gaps = [(w.name, _trigger_gap(w, path)) for w in late]
    if all(gap is None for _, gap in gaps):
        unmatched = (
            "its paths filter does not match" if one else "no paths filter of those matches"
        )
        return (
            f"{head}: {unmatched} it, so a later change to it alone will run it nowhere. "
            f"Add it to {'that paths filter' if one else 'those paths filters'}."
        )
    each = " ".join(
        f"{name}: {gap[0]}. {gap[1]}."
        if gap
        else f"{name}: its paths filter does not match it. Add it to that paths filter."
        for name, gap in gaps
    )
    return f"{head}, so a later change to it alone will run it nowhere. {each}"


def warnings_for(
    added: Iterable[str],
    legs: dict[str, Counts],
    workflows: dict[str, Workflow],
    missing: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """``(file, message)`` for each added file that no check on the change runs."""
    listed = sorted(w.name for w in workflows.values() if w.selectors or w.node_ids)
    missing = list(missing)
    out = []
    for path in added:
        if not ran_nothing(path, legs):
            continue
        running = [w for w in workflows.values() if w.runs(path)]
        if running:
            # Run here only because this pull request changes other files its
            # list workflow's paths filter matches: the next change to this file
            # alone runs it nowhere. A whole-suite workflow also runs nightly.
            late = [w for w in running if not w.whole and not w.fires_for(path)]
            if len(late) == len(running):
                out.append((path, _late_message(path, legs, late, missing)))
            continue
        first = f"{path} is new, and {_what_ran(path, legs)}"
        if missing:
            first += f" (no report was read from {', '.join(missing)})"
        parts = [first + ".", "No workflow that runs on this pull request runs it."]
        parts += [n for n in (_note(w, path) for w in workflows.values()) if n]
        if listed:
            parts.append(
                "If it needs what one of these workflows builds, add it to that workflow's "
                f"pytest list and paths filter: {', '.join(listed)}."
            )
        out.append((path, " ".join(parts)))
    return out


def _data(text: str) -> str:
    """A workflow command's message, escaped as GitHub reads it: a newline
    would end the command."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _prop(text: str) -> str:
    """A workflow command's property value (``file=``), escaped as GitHub reads it."""
    return _data(text).replace(":", "%3A").replace(",", "%2C")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", type=Path, required=True, help="the legs' shard artifacts")
    ap.add_argument("--base", required=True, help="the commit the change is measured from")
    ap.add_argument("--head", default="HEAD")
    a = ap.parse_args(argv)

    try:
        changed = changed_files(a.base, a.head)
    except (subprocess.CalledProcessError, OSError) as e:
        why = e.stderr.decode("utf-8", "replace").strip() if getattr(e, "stderr", None) else str(e)
        why = (why.splitlines() or [str(e)])[0]  # git's first line says what failed
        msg = f"the files this change touches could not be read ({why}); none checked"
        print(f"::warning::{_data(msg)}")
        return 0
    added = added_tests(changed)
    legs, unread = read_reports(a.reports)
    for shard, why in unread:
        print(
            f"::warning::{_data(f'the python-tests report of {shard} could not be read: {why}')}"
        )
    if not legs:
        print(f"::warning::{_data(f'no python-tests reports under {a.reports}; nothing checked')}")
        return 0
    suite = WORKFLOWS / SUITE_WORKFLOW
    text = suite.read_text(encoding="utf-8", errors="replace") if suite.is_file() else ""
    expected = expected_shards(text) if text else set()
    missing = missing_shards(a.reports, expected, (s for s, _ in unread))
    if missing:
        print(
            f"::warning::no python-tests report was read from {', '.join(missing)}; "
            "a new file in those shards may be warned about wrongly"
        )
    workflows = read_workflows([f for _, f in changed])
    found = warnings_for(added, legs, workflows, missing)
    unrun = sum(1 for f, _ in found if not any(w.runs(f) for w in workflows.values()))
    late = len(found) - unrun
    also = f"; {late} that a check runs only for this change's other files" if late else ""
    print(f"{len(added)} test file(s) added; {unrun} that no check on this change runs{also}")
    for path, msg in found:
        print(f"::warning file={_prop(path)}::{_data(msg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
