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

import contextlib
import importlib.metadata
import inspect
import io
import re
import subprocess
import sys
import warnings

import bngsim
from bngsim import _agent_guide

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


def test_every_block_runs_and_prints_what_it_says(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__guide__", "_stated": _stated}
    n_checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _, code in _blocks():
            checked = _checked(code)
            n_checked += checked.count("_stated(")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(checked, "agent_guide.md", "exec"), namespace)
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
