"""The documented Python examples must name API that exists (lanl/bngsim#343).

The README quickstart — the first bngsim code a new user ever runs — had drifted
four ways from the library: ``Model.from_net_file`` (never existed),
``sim.run(t_end=, n_steps=)`` (the parameters are ``t_span``/``n_points``),
``result.times`` (it is ``result.time``), ``result["A"]`` (``Result`` is not
subscriptable) and ``result.to_dataframe()`` (``Result.dataframe`` is a
property). Nothing in the suite read the prose, so five wrong lines sat on the
project's front page through a release.

Two guards, cheap and complementary:

* :func:`test_front_door_example_runs` EXECUTES the first Python block of each
  front-door page as a reader would paste it: in an empty directory, with the
  optional extras' modules blocked, so the block must write its own model and
  must not need pandas or any other extra (issue #1691). It then checks the
  block's answer against the closed form of the model the three pages share.
* :func:`test_documented_api_names_exist` STATICALLY resolves every ``bngsim.…``
  attribute chain, method name and keyword argument in every Python block under
  ``docs/`` — pages whose examples need corpora, XML side files or optional
  engines and so cannot be executed here. It cannot prove an example produces the
  right answer; it does prove every name it spells is real.

The static pass is deliberately conservative: it only reasons about variables it
watched being bound to a ``Model``, ``Simulator`` or ``Result``, so an example
built out of anything else is passed over rather than guessed at. Its job is to
have no false alarms, because a docs check that cries wolf gets deleted.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import sys
from pathlib import Path

import bngsim
import pytest

_ROOT = Path(__file__).resolve().parents[2]

# ```python … ``` and its ```py alias. Indented fences (inside a list item) are
# not matched: docs/ has none, and an indented block's dedent is a guess.
_FENCE = re.compile(r"^```(?:python|py)\s*$(.*?)^```\s*$", re.MULTILINE | re.DOTALL)

# CHANGELOG.md is excluded on purpose — release notes quote the API as it was,
# and rewriting history to satisfy a linter would be the wrong repair.
_SCANNED = ("README.md", "SUPPORT_MATRIX.md")

# Front-door pages whose FIRST Python block is a self-contained load-and-run:
# the ones a newcomer meets before any fixture or corpus. Named rather than
# sniffed, so a later page that happens to look self-contained is not silently
# conscripted into the executed set.
_FRONT_DOOR = ("README.md", "docs/index.md", "docs/quickstart.md")

# The top-level modules of pyproject's optional extras (and the test extra). A
# bare `pip install bngsim` brings numpy, python-libsbml, sympy and mpmath only,
# so a front-door block may import none of these (issue #1691).
_EXTRA_MODULES = (
    "pandas",
    "antimony",
    "h5py",
    "xarray",
    "bionetgen",
    "jax",
    "jaxlib",
    "diffrax",
    "vivarium",
    "roadrunner",
    "bioservices",
    "scipy",
    "jsonschema",
    "lxml",
)

# Calls that hand back a Result, whatever the receiver: Simulator.run,
# NfsimSession.simulate, ReactionKernel.run_until.
_RESULT_FACTORIES = frozenset({"run", "run_until", "simulate"})


def _markdown_files() -> list[Path]:
    """Every prose file with Python examples, README first."""
    files = [_ROOT / name for name in _SCANNED]
    files += sorted((_ROOT / "docs").rglob("*.md"))
    return [f for f in files if f.exists()]


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    """``(1-based line of the opening fence, source)`` for each Python block."""
    text = path.read_text(encoding="utf-8")
    return [(text[: m.start()].count("\n") + 1, m.group(1)) for m in _FENCE.finditer(text)]


def _docs_present() -> None:
    if not (_ROOT / "docs").is_dir():
        pytest.skip("docs/ lives in the source root and is not present in an installed wheel")


def _attribute_chain(node: ast.AST) -> list[str] | None:
    """``bngsim.Model.from_net`` → ``["bngsim", "Model", "from_net"]``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def _rejects_keyword(func: object, keyword: str) -> bool:
    """Would calling ``func(**{keyword: …})`` be a TypeError?

    ``False`` whenever that cannot be decided — a ``**kwargs`` tail, or a
    C-extension callable with no introspectable signature.
    """
    try:
        sig = inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return False
    return keyword not in params


class _ExampleChecker(ast.NodeVisitor):
    """Resolve the bngsim names one documented block spells.

    Two passes over the block: :meth:`_bind` learns which local variables hold a
    ``Model``, a ``Simulator`` or a ``Result``, then the visitor checks every use
    of those variables and every ``bngsim.…`` chain. Binding first means an
    example that uses a variable above its assignment (rare, but legal inside a
    function) is still understood.
    """

    def __init__(self) -> None:
        self.problems: list[str] = []
        self._kinds: dict[str, type] = {}

    def check(self, tree: ast.Module) -> list[str]:
        for node in ast.walk(tree):
            self._bind(node)
        self.visit(tree)
        # ``model.evaluator.define_function(…)`` reaches the same missing name
        # from the outer chain and again from the inner one; report it once.
        return list(dict.fromkeys(self.problems))

    # ─── binding ──────────────────────────────────────────────────────────

    def _bind(self, node: ast.AST) -> None:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            return
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not targets:
            return
        chain = _attribute_chain(node.value.func) or []
        kind: type | None = None
        if chain[:2] == ["bngsim", "Model"] or chain[-1:] == ["clone"]:
            kind = bngsim.Model
        elif chain[:2] == ["bngsim", "Simulator"]:
            kind = bngsim.Simulator
        elif len(chain) >= 2 and chain[-1] in _RESULT_FACTORIES:
            kind = bngsim.Result
        if kind is None:
            return
        for name in targets:
            self._kinds[name] = kind

    # ─── checks ───────────────────────────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _attribute_chain(node)
        if chain and chain[0] == "bngsim":
            obj: object = bngsim
            for part in chain[1:]:
                if not hasattr(obj, part):
                    self.problems.append(f"{'.'.join(chain)} — no attribute {part!r}")
                    break
                obj = getattr(obj, part)
        elif chain and (kind := self._kinds.get(chain[0])) is not None:
            # Only the first hop is ours; past it the object could be anything.
            if not hasattr(kind, chain[1]):
                self.problems.append(f"{kind.__name__} has no attribute {chain[1]!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _attribute_chain(node.func)
        method: object | None = None
        if chain and chain[0] == "bngsim" and len(chain) == 3:
            owner = getattr(bngsim, chain[1], None)
            method = getattr(owner, chain[2], None)
        elif chain and len(chain) == 2 and (kind := self._kinds.get(chain[0])) is not None:
            method = getattr(kind, chain[1], None)
        if method is not None:
            for kw in node.keywords:
                if kw.arg is not None and _rejects_keyword(method, kw.arg):
                    self.problems.append(f"{'.'.join(chain)}() rejects keyword {kw.arg!r}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            kind = self._kinds.get(node.value.id)
            if kind is not None and not hasattr(kind, "__getitem__"):
                self.problems.append(f"{kind.__name__} is not subscriptable")
        self.generic_visit(node)


def test_documented_python_blocks_parse() -> None:
    """Every documented Python block is syntactically valid Python."""
    _docs_present()
    failures: list[str] = []
    blocks = 0
    for path in _markdown_files():
        for line, code in _python_blocks(path):
            blocks += 1
            try:
                ast.parse(code)
            except SyntaxError as exc:
                rel = path.relative_to(_ROOT)
                failures.append(f"{rel}:{line}: {exc}")
    assert not failures, "unparseable documented examples:\n  " + "\n  ".join(failures)
    assert blocks > 20, f"only {blocks} documented Python blocks found — did the fence regex rot?"


def test_documented_api_names_exist() -> None:
    """No documented example names a bngsim attribute or keyword that is gone."""
    _docs_present()
    failures: list[str] = []
    for path in _markdown_files():
        for line, code in _python_blocks(path):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue  # reported by test_documented_python_blocks_parse
            for problem in _ExampleChecker().check(tree):
                failures.append(f"{path.relative_to(_ROOT)}:{line}: {problem}")
    assert not failures, "documented examples name API that does not exist:\n  " + "\n  ".join(
        failures
    )


@pytest.mark.parametrize("page", _FRONT_DOOR)
def test_front_door_example_runs(
    page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first Python block of each front-door page executes as written."""
    _docs_present()
    path = _ROOT / page
    if not path.exists():
        pytest.skip(f"{page} is not present in this checkout")
    blocks = _python_blocks(path)
    assert blocks, f"{page} has no Python example to run"
    code = blocks[0][1]
    assert "model.net" in code, f"{page}'s first Python block is not the load-and-run example"

    # As pasted after `pip install bngsim` (issue #1691): an empty directory, so
    # the block has to write the model it loads, and none of the extras, which
    # the base install does not bring (None in sys.modules makes an import raise).
    monkeypatch.chdir(tmp_path)
    for name in _EXTRA_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    namespace: dict[str, object] = {"__name__": "__docs_example__"}
    exec(compile(code, f"{page} (first python block)", "exec"), namespace)  # noqa: S102

    # Oracle: the three pages share one model, A -> B at k = 0.5 from A(0) =
    # 100, so A(t) = 100 e^(-0.5 t) at whatever t_end the block chose. Pinned,
    # not read from the model, so a page whose k drifts from the 0.6738 it
    # prints fails too. rel 1e-4: the run uses the solver's default tolerances,
    # far looser than the closed form; a wrong model or k is off by far more.
    result = namespace["result"]
    t_end = float(result.time[-1])  # type: ignore[attr-defined]
    a_end = float(result.observables["A"][-1])  # type: ignore[attr-defined]
    assert a_end == pytest.approx(100.0 * math.exp(-0.5 * t_end), rel=1e-4)
    # Each page prints 0.6738 = 100 e^(-5), its value at t = 10: a block that
    # runs to another time while keeping that comment is wrong too.
    assert t_end == pytest.approx(10.0), f"{page}'s example prints A(10) but runs to t = {t_end:g}"
