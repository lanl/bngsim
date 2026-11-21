"""The agent guide runs as written, fits its budget and ships in the package (#1699).

``bngsim/agent_guide.md`` is what an agent reads to learn bngsim, through
:func:`bngsim.agent_guide`, ``python -m bngsim guide`` or the documentation.
An agent copies its blocks, so a block that no longer runs, or a value that no
longer comes out, teaches the wrong call with the authority of the docs:

* every Python block runs, in page order, in one namespace, in an empty
  directory, as the guide says it does, with warnings raised as errors;
* a ``print`` line whose comment states a value prints it;
* the translation table is the one :data:`bngsim._foreign_names.FOREIGN_NAMES`
  generates, so it says what bngsim's wrong-name errors say;
* the guide stays under :data:`BUDGET_CHARS` characters, so it can be loaded
  whole into an agent's context (about 8,000 tokens).
"""

from __future__ import annotations

import ast
import contextlib
import importlib.metadata
import inspect
import io
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim import _agent_guide
from bngsim._foreign_names import _ForeignAttribute

#: The guide's size budget, in characters: about 8,000 tokens, small enough to
#: be loaded whole beside a task. Raise it only with something taken out.
BUDGET_CHARS = 32_000

_FENCE = re.compile(r"^```(\w*)\s*$(.*?)^```\s*$", re.MULTILINE | re.DOTALL)

# `print(x)  # value` and `print(x, y)  # v1 v2`: the comment up to a colon is
# what the line prints.
_STATED = re.compile(r"^print\((?P<expr>.+)\)\s+#\s*(?P<value>[^:]+?)\s*(?::.*)?$")


def _blocks() -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _FENCE.finditer(bngsim.agent_guide())]


def test_the_guide_ships_in_the_package_and_is_markdown():
    text = bngsim.agent_guide()
    assert text.startswith("# bngsim agent guide\n")
    assert "bngsim.agent_guide()" in text
    assert "agent_guide" in bngsim.__all__
    assert "bngsim.agent_guide()" in bngsim.__doc__


def test_the_guide_fits_its_budget():
    text = bngsim.agent_guide()
    assert len(text) <= BUDGET_CHARS, (
        f"the agent guide is {len(text):,} characters, over its {BUDGET_CHARS:,} budget"
    )


def test_every_block_is_python():
    """Every fenced block is one the test runs: no block is left out unrun."""
    languages = {language for language, _ in _blocks()}
    assert languages == {"python"}


def test_the_translation_table_is_generated_from_foreign_names():
    text = bngsim.agent_guide()
    assert _agent_guide._with_table(text, _agent_guide.translation_table()) == text, (
        "the agent guide's translation table differs from FOREIGN_NAMES': "
        "run `python scripts/agent_guide_table.py`"
    )


#: A name the guide's prose reads on a bngsim object, in backticks: ``model.x``,
#: ``sim.x``, ``result.x``, ``ss.x``, ``bngsim.x`` and ``bngsim.Model.x``.
_PROSE_NAME = re.compile(
    r"\b(bngsim\.Model|bngsim\.Simulator|bngsim\.Result|Model|Simulator|Result|bngsim|"
    r"model|sim|ssa|result|ss)\.([A-Za-z_]\w*)"
)


def _prose() -> str:
    """The guide without its code blocks and its generated table, which other
    tests check."""
    text = _FENCE.sub("", bngsim.agent_guide())
    start = text.index(_agent_guide.TABLE_START)
    return text[:start] + text[text.index(_agent_guide.TABLE_END) :]


def test_every_name_the_prose_reads_exists():
    """The calls table, the rules and "What needs what" are prose no block runs:
    each bngsim name they spell must be a real one, not a renamed or a foreign
    one."""
    from bngsim._foreign_names import _ForeignAttribute, lookup

    on = {
        "bngsim": bngsim,
        "bngsim.Model": bngsim.Model, "Model": bngsim.Model, "model": bngsim.Model,
        "bngsim.Simulator": bngsim.Simulator, "Simulator": bngsim.Simulator,
        "sim": bngsim.Simulator, "ssa": bngsim.Simulator,
        "bngsim.Result": bngsim.Result, "Result": bngsim.Result, "result": bngsim.Result,
        "ss": bngsim.SteadyStateResult,
    }  # fmt: skip
    code = " ".join(re.findall(r"`([^`]+)`", _prose()))
    names = _PROSE_NAME.findall(code)
    assert len(names) > 30
    for obj, attr in names:
        target = on[obj]
        owner = target.__name__.rpartition(".")[2]
        if lookup(owner, attr) is not None:
            continue  # a wrong name the prose names as one, from the table
        assert hasattr(target, attr) or attr in dir(target), f"{obj}.{attr}"
        assert not isinstance(vars(target).get(attr), _ForeignAttribute), f"{obj}.{attr}"
    # A keyword the prose passes to a Simulator or a run call is one it takes.
    for call, keyword in re.findall(
        r"\b(Simulator|run|run_replicates|parameter_scan)\([^)]*?\b(\w+)=", _prose()
    ):
        target = bngsim.Simulator if call == "Simulator" else getattr(bngsim.Simulator, call)
        assert keyword in inspect.signature(target).parameters, f"{call}({keyword}=)"


def test_the_capabilities_and_extras_the_guide_names_exist():
    section = bngsim.agent_guide().split("## What needs what")[1].split("\n## ")[0]
    features = bngsim.capabilities()["features"]
    rows = [line.split("|") for line in section.splitlines() if line.startswith("| ")][1:]
    named = [cap for row in rows for cap in re.findall(r"`(\w+)`", row[3])]
    assert named and all(cap in features for cap in named), named
    extras = set(importlib.metadata.metadata("bngsim").get_all("Provides-Extra") or ())
    for extra in re.findall(r"bngsim\[(\w+)\]", section):
        assert extra in extras, extra
    for method in re.findall(r'method="(\w+)"', _prose()):
        assert method in bngsim.Simulator.METHODS, method


def test_the_table_lists_each_tool_the_guide_names():
    table = _agent_guide.translation_table()
    for tool in _agent_guide.TOOLS:
        assert f"**{tool}**" in table


def test_the_skill_is_the_guide_with_a_header():
    skill = bngsim.agent_guide(skill=True)
    assert skill.startswith("---\nname: bngsim\ndescription: >-\n")
    assert skill.endswith(bngsim.agent_guide())
    front = skill.split("---\n")[1]
    assert len(front.splitlines()) <= 10


def _stated(*args: object, stated: str, line: str) -> None:
    """What ``print(*args)`` would show, checked against the value its comment states."""
    shown = " ".join(str(arg) for arg in args)
    assert shown == stated, f"{line!r} prints {shown!r}"


def _checked(code: str) -> str:
    """*code* with each top-level ``print(...)  # value`` line turned into a check
    of that call's own output, so no other print can shift what it is compared
    with."""
    lines = []
    for line in code.splitlines():
        stated = _STATED.match(line)
        if stated:
            line = (
                f"_stated({stated.group('expr')}, stated={stated.group('value')!r}, line={line!r})"
            )
        lines.append(line)
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def guide_run(tmp_path_factory) -> tuple[dict[str, object], int]:
    """The guide's blocks run in page order, in one namespace, in an empty
    directory, with warnings raised as errors: the namespace they leave, and
    how many stated values they checked."""
    namespace: dict[str, object] = {"__name__": "__guide__", "_stated": _stated}
    n_checked = 0
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("guide"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _, code in _blocks():
                checked = _checked(code)
                n_checked += checked.count("_stated(")
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(checked, "agent_guide.md", "exec"), namespace)
    finally:
        os.chdir(cwd)
    return namespace, n_checked


def test_every_block_runs_and_prints_what_it_says(guide_run):
    _, n_checked = guide_run
    assert n_checked >= 5, "the guide's stated values were not found"


def test_python_m_bngsim_prints_the_guide_and_the_skill():
    for command, expected in (("guide", bngsim.agent_guide()), ("skill", None)):
        out = subprocess.run(
            [sys.executable, "-m", "bngsim", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        if expected is not None:
            assert out == expected
        else:
            assert out == bngsim.agent_guide(skill=True)


def test_python_m_bngsim_with_no_command_says_what_it_takes():
    run = subprocess.run(
        [sys.executable, "-m", "bngsim"], capture_output=True, text=True, encoding="utf-8"
    )
    assert run.returncode == 2
    assert "guide" in run.stderr and "skill" in run.stderr


# ─── Coverage: what the guide leaves out, it leaves out on purpose ─────────────
#
# The tests above stop the guide saying something wrong. They do not stop it
# falling silent: a new public call, keyword, simulation method, capability or
# extra is public the day it merges, and nothing would ask whether an agent
# reading the guide should hear of it. These ask, owner by owner:
#
# * ``bngsim:<name>`` for each name in ``bngsim.__all__``;
# * ``<Class>:<member>`` for each public member a class the guide uses (names,
#   or reaches a member of) defines
#   in bngsim (not one inherited from NumPy or the standard library, and not a
#   wrong name bngsim answers with an error, which the table covers);
# * ``<call>(<keyword>)`` for each keyword of a call the guide uses, where the
#   call is ``Simulator`` (the constructor), ``Simulator.run`` or
#   ``bngsim.capabilities``;
# * ``method:``, ``feature:`` and ``extra:`` for each ``Simulator.METHODS``
#   value (used when quoted), ``capabilities()`` feature (in backticks) and
#   install extra (as ``bngsim[name]``).
#
# A class or call the guide does not use is one line, which stands for its
# members and keywords. Use is read from the guide's code, its blocks and its
# backticked spans, with each variable resolved through the namespace the
# blocks leave: ``ss.converged`` uses ``SteadyStateResult:converged`` because
# ``ss`` holds one, and ``sim.run(atol=...)`` uses ``Simulator.run(atol)``,
# not ``Simulator.bifurcate(atol)``. The table's first column, other tools'
# names, is not use. A keyword with no default is used by any call.
#
# Every unused key is listed in LEFT_OUT, and the two must match exactly, so a
# new name fails until the guide uses it or a line lists it, a decision a
# reviewer sees. The list is not a to-do list: most of the surface is for
# help(), not for a first read of a page held to a budget.

#: The names the guide leaves out on purpose, one key per line.
LEFT_OUT = Path(__file__).with_name("agent_guide_left_out.txt")

_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _guide_code() -> tuple[list[ast.Module], str]:
    """The guide's code, parsed: each block, and each backticked span outside
    them that is Python; and the guide's text. Both leave out the table's first
    column, which is other tools' names, not bngsim's."""
    blocks = [code for _, code in _blocks()]
    prose = _FENCE.sub("", bngsim.agent_guide())
    start, end = prose.index(_agent_guide.TABLE_START), prose.index(_agent_guide.TABLE_END)
    table, prose = prose[start:end], prose[:start] + prose[end:]
    cells = [re.split(r"(?<!\\)\|", row)[2] for row in table.splitlines() if row.startswith("| `")]
    text = "\n".join([*blocks, prose, *cells])
    trees = [ast.parse(code) for code in blocks]
    for span in re.findall(r"`([^`\n]+)`", "\n".join([prose, *cells])):
        with contextlib.suppress(SyntaxError):
            trees.append(ast.parse(span.replace("\\|", "|")))
    return trees, text


def _public_classes() -> dict[str, type]:
    return {
        name: obj
        for name in bngsim.__all__
        if inspect.isclass(obj := getattr(bngsim, name)) and not issubclass(obj, BaseException)
    }


class _Uses(ast.NodeVisitor):
    """The keys the guide's code uses, each variable resolved through *namespace*."""

    def __init__(self, namespace: dict[str, object]) -> None:
        self.namespace = namespace
        self.by_name = _public_classes()
        self.classes = set(self.by_name.values())
        self.keys: set[str] = set()

    def _owner(self, node: ast.AST) -> object | None:
        """The module or class an expression is, or the class of what it holds
        or returns: ``model``, ``Model``, ``bngsim.Model``, ``Simulator(m)``,
        ``sim.steady_state()``."""
        if isinstance(node, ast.Name):
            if node.id == "bngsim":
                return bngsim
            if (cls := self.by_name.get(node.id)) is not None:
                return cls
            held = self.namespace.get(node.id)
            return type(held) if type(held) in self.classes else None
        if isinstance(node, ast.Attribute) and self._owner(node.value) is bngsim:
            obj = getattr(bngsim, node.attr, None)
            return obj if obj in self.classes else None
        if isinstance(node, ast.Call) and (call := self._call(node.func)) is not None:
            if call in self.by_name:
                return self.by_name[call]  # a constructor
            owner, _, method = call.rpartition(".")
            target = getattr(self.by_name.get(owner, bngsim), method, None)
            try:
                returns = inspect.signature(target).return_annotation  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            if isinstance(returns, str):
                returns = self.by_name.get(returns.strip("'\"").rpartition(".")[2])
            return returns if returns in self.classes else None
        return None

    def _call(self, node: ast.AST) -> str | None:
        """The call key of a callee: ``Simulator``, ``Simulator.run``, ``bngsim.f``."""
        if isinstance(node, ast.Name) and node.id in self.by_name:
            return node.id
        if isinstance(node, ast.Attribute) and (owner := self._owner(node.value)) is not None:
            if owner is bngsim:
                return (
                    node.attr
                    if inspect.isclass(getattr(bngsim, node.attr, None))
                    else (f"bngsim.{node.attr}")
                )
            return f"{owner.__name__}.{node.attr}"
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        owner = self._owner(node.value)
        if owner is bngsim:
            self.keys.add(f"bngsim:{node.attr}")
        elif owner is not None:
            self.keys.add(f"{owner.__name__}:{node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call = self._call(node.func)
        if call is not None:
            self.keys.add(f"{call}()")
            self.keys.update(f"{call}({kw.arg})" for kw in node.keywords if kw.arg)
        self.generic_visit(node)


def _keywords(target: object) -> list[tuple[str, bool]]:
    """A callable's keywords, each with whether a call must pass it."""
    try:
        params = inspect.signature(target).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    return [
        (name, p.default is p.empty)
        for name, p in params.items()
        if name != "self" and not name.startswith("_") and p.kind not in _VARIADIC
    ]


def _coverage(namespace: dict[str, object]) -> tuple[dict[str, bool], set[str]]:
    """Each key of the public surface, and whether the guide uses it."""
    trees, text = _guide_code()
    uses = _Uses(namespace)
    for tree in trees:
        uses.visit(tree)
    used = uses.keys
    words = "\n".join(ast.unparse(tree) for tree in trees)
    for name in bngsim.__all__:  # a name imported or passed bare is used too
        if re.search(rf"\b{re.escape(name)}\b", words):
            used.add(f"bngsim:{name}")

    surface: dict[str, bool] = {
        f"bngsim:{name}": f"bngsim:{name}" in used for name in bngsim.__all__
    }
    calls: dict[str, object] = {}
    for cname, cls in _public_classes().items():
        # A class is used when the guide names it or reaches a member of one.
        if not (surface[f"bngsim:{cname}"] or any(k.startswith(f"{cname}:") for k in used)):
            continue
        calls[cname] = cls
        for owner in cls.__mro__:
            if not owner.__module__.startswith("bngsim"):
                continue
            for name, value in vars(owner).items():
                if name.startswith("_") or isinstance(value, _ForeignAttribute):
                    continue
                key = f"{cname}:{name}"
                surface[key] = key in used or f"{cname}.{name}()" in used
                if surface[key] and callable(getattr(cls, name, None)):
                    calls[f"{cname}.{name}"] = getattr(cls, name)
    for name in bngsim.__all__:
        obj = getattr(bngsim, name)
        if surface[f"bngsim:{name}"] and inspect.isroutine(obj):
            calls[f"bngsim.{name}"] = obj
    for call, target in calls.items():
        called = f"{call}()" in used
        for name, required in _keywords(target):
            key = f"{call}({name})"
            surface[key] = key in used or (required and called)
    # A feature is used in "What needs what", where the guide says what each
    # needs: elsewhere a backticked ``nfsim`` may be the method, not the feature.
    needs = bngsim.agent_guide().split("## What needs what")[1].split("\n## ")[0]
    for name in bngsim.Simulator.METHODS:
        surface[f"method:{name}"] = bool(re.search(rf"[\"']{re.escape(name)}[\"']", text))
    for name in bngsim.capabilities()["features"]:
        surface[f"feature:{name}"] = bool(re.search(rf"`{re.escape(name)}`", needs))
    for name in importlib.metadata.metadata("bngsim").get_all("Provides-Extra") or ():
        surface[f"extra:{name}"] = bool(re.search(rf"bngsim\[[\w,]*\b{re.escape(name)}\b", text))
    return surface, used


def _left_out_lines() -> list[str]:
    lines = LEFT_OUT.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_every_public_name_is_in_the_guide_or_left_out_on_purpose(guide_run):
    namespace, _ = guide_run
    surface, _ = _coverage(namespace)
    unused = {key for key, is_used in surface.items() if not is_used}
    listed = set(_left_out_lines())
    new, stale = sorted(unused - listed), sorted(listed - unused)
    assert not new and not stale, (
        "the agent guide's coverage changed. Use each new public name in "
        "python/bngsim/agent_guide.md if an agent meeting bngsim should know it, "
        f"or add its key to python/tests/{LEFT_OUT.name} exactly as printed here, "
        "in Python's sorted() order. A stale line is a name that is gone or that "
        "the guide now uses: drop it. (Extras are read from the installed "
        "metadata: reinstall after editing pyproject.toml.)"
        f"\nnew, in neither: {new}\nstale lines: {stale}"
    )


def test_the_left_out_list_is_sorted_and_has_no_duplicates():
    lines = _left_out_lines()
    assert lines == sorted(set(lines)), "keep the list in Python's sorted() order, once each"
