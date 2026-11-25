"""The documented Python examples must name API that exists (lanl/bngsim#343).

The README quickstart — the first bngsim code a new user ever runs — had drifted
four ways from the library: ``Model.from_net_file`` (never existed),
``sim.run(t_end=, n_steps=)`` (the parameters are ``t_span``/``n_points``),
``result.times`` (it is ``result.time``), ``result["A"]`` (``Result`` is not
subscriptable) and ``result.to_dataframe()`` (``Result.dataframe`` is a
property). Nothing in the suite read the prose, so five wrong lines sat on the
project's front page through a release.

Three guards, cheap and complementary:

* :func:`test_front_door_example_runs` EXECUTES the first Python block of each
  front-door page as a reader would paste it: in an empty directory, with the
  optional extras' modules blocked, so the block must write its own model and
  must not need pandas or any other extra (issue #1691). It then checks the
  block's answer against the closed form of the model the three pages share,
  and that each line whose comment states a value has it: what a print line
  prints, or the shape of an array a bare expression shows.
  :func:`test_quickstart_sweep_block_runs_on_the_first_blocks_model` runs the
  quickstart's sweep block after its first block, as the page reads.
* :func:`test_each_page_runs_top_to_bottom` EXECUTES every Python block of every
  page, in page order and in one namespace, in an empty directory, as a reader
  would paste the page from the top (issue #1693). Each page builds the model
  its blocks use, so the blocks are examples a reader can run and not fragments.
  A line whose comment states a value must have it, and a block may not warn
  unless its prose says it does. A block that needs what a test run may not
  have (BNG2.pl, the network, an optional extra) is passed over by name in
  :data:`_NOT_RUN`, with the reason, and reported as a skip; where the need is
  met, as jax's is on the Python legs, it runs.
* :func:`test_documented_api_names_exist` STATICALLY resolves every ``bngsim.…``
  attribute chain, method name and keyword argument in every Python block under
  ``docs/``, the blocks :data:`_NOT_RUN` passes over among them. It cannot prove
  an example produces the right answer; it does prove every name it spells is
  real.

The static pass is deliberately conservative: it only reasons about variables it
watched being bound to a ``Model``, ``Simulator`` or ``Result``, so an example
built out of anything else is passed over rather than guessed at. Its job is to
have no false alarms, because a docs check that cries wolf gets deleted.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import gc
import importlib
import importlib.util
import inspect
import io
import logging
import math
import re
import sys
import tokenize
import traceback
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import bngsim
import numpy as np
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

# A front-door line whose comment starts with a value: a print line, which must
# print it (`print(result.time.shape)  # (101,)`, `print(...)  # 0.6738, which
# is ...`), or a bare expression, as the README writes them, whose comment gives
# the shape of the array it shows (`result.species  # (101, 2) array`). A comment
# that starts with words ("# trajectory of ...") states no value.
_STATED_VALUE = re.compile(
    r"^(?:print\((?P<printed>.+?)\)|(?P<shown>[A-Za-z_][\w.]*(?:\[[^\]]*\])*))"
    r"\s+#\s*(?P<value>\([^)]*\)|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)

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

    # What the block says it shows: each line whose comment states a value must
    # have it, so a comment and the code under it cannot drift apart
    # (n_points=11 under "# (101,)", or "# 0.9999" over A(10)). The expression
    # is evaluated again on the finished namespace; these lines only read the
    # result.
    stated = [m for m in map(_STATED_VALUE.match, code.splitlines()) if m]
    assert stated, f"{page}'s first block has no line whose comment states its value"
    for m in stated:
        try:
            claim = ast.literal_eval(m["value"])
        except (ValueError, SyntaxError):
            pytest.fail(f"{page}: the comment {m['value']!r} starts like a value but is not one")
        if m["printed"] is not None:
            shown, got = f"print({m['printed']})", eval(m["printed"], namespace)  # noqa: S307
        else:
            shown, got = m["shown"], eval(m["shown"], namespace)  # noqa: S307
        if isinstance(claim, tuple) and hasattr(got, "shape"):
            got = tuple(got.shape)  # an array, printed or shown: its comment gives the shape
        assert got == claim, f"{page}: {shown} gives {got!r}, but its comment says {m['value']}"


def test_quickstart_sweep_block_runs_on_the_first_blocks_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quickstart's "Parameter sweeps" block runs on the model its first block made.

    The page reads top to bottom, and the sweep block uses that block's ``model``
    and ``sim``, so it may set only the parameter that model has, ``k``. It set
    ``kf``, ``kr`` and ``kcat``, so pasted next it raised ParameterError.
    """
    _docs_present()
    path = _ROOT / "docs" / "quickstart.md"
    if not path.exists():
        pytest.skip("docs/quickstart.md is not present in this checkout")
    heading = path.read_text(encoding="utf-8").splitlines().index("## Parameter sweeps") + 1
    blocks = _python_blocks(path)
    sweep = next(code for start, code in blocks if start > heading)

    monkeypatch.chdir(tmp_path)
    for name in _EXTRA_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    namespace: dict[str, object] = {"__name__": "__docs_example__"}
    exec(compile(blocks[0][1], "docs/quickstart.md (first python block)", "exec"), namespace)  # noqa: S102
    exec(compile(sweep, "docs/quickstart.md (Parameter sweeps)", "exec"), namespace)  # noqa: S102

    # Oracle: A -> B is first order, so on result2's grid (101 points to t = 100,
    # dt = 1) A falls by e^(-k) per step at the k = 1.0 the block sets, wherever
    # the run starts. rel 1e-4 for the solver's default tolerances, as above. The
    # block ends by setting k = 0.25.
    a = namespace["result2"].observables["A"]  # type: ignore[attr-defined]
    assert float(a[1]) == pytest.approx(float(a[0]) * math.exp(-1.0), rel=1e-4)
    assert namespace["model"].get_param("k") == 0.25  # type: ignore[attr-defined]


# ─── Every page runs, top to bottom (issue #1693) ─────────────────────────────


class _Need(NamedTuple):
    """What a block needs that a test run may not have, and whether it has it."""

    reason: str
    met: Callable[[], bool]
    #: What a block with this need calls, so an entry cannot drift onto a block
    #: that needs nothing: a block's name is its place on the page.
    calls: re.Pattern[str]
    #: The skip reason where the need is not met, in words the skip audit knows.
    absent: str
    #: Met by the dev extra, which python-tests.yml installs, so every Python leg
    #: of a pull request runs the block. Only the other needs count against the
    #: one-block-in-ten limit.
    dev: bool


def _importable(module: str) -> Callable[[], bool]:
    return lambda: importlib.util.find_spec(module) is not None


# The dev extra brings jax, pandas and h5py; the nightly sanitizer job installs
# only the test extra, so these blocks pass over there.
_JAX = _Need(
    "needs jax, the jax extra",
    _importable("jax"),
    re.compile(r"\bjax\b"),
    "JAX not installed",
    True,
)
_PANDAS = _Need(
    "needs pandas, the pandas extra",
    _importable("pandas"),
    re.compile(r"\.dataframe\b"),
    "could not import 'pandas'",
    True,
)
_HDF5 = _Need(
    "needs h5py, the hdf5 extra",
    _importable("h5py"),
    re.compile(r"\.save\(\s*\"[^\"]*\.h5\""),
    "could not import 'h5py'",
    True,
)
#: A BNGL load runs BNG2.pl to turn the rules into a network. No CI leg installs
#: it; a checkout whose $BNGPATH finds one runs these blocks. (The probe may
#: import PyBioNetGen, so it runs once.)
_BNG2 = _Need(
    "needs BNG2.pl, which turns BNGL rules into a network",
    functools.cache(lambda: bool(bngsim.HAS_BNGL)),
    re.compile(r"\b(?:from_bngl|run_bngl)\(|Model\.load\([^)]*\.bngl|validate=\"bng2\""),
    "no BNG2.pl found",
    False,
)
#: Never run: a test must not reach the internet.
_NETWORK = _Need(
    "needs the network: it downloads a BioModels entry",
    lambda: False,
    re.compile(r"\bfrom_biomodels\("),
    "a BioModels download, which no test makes",
    False,
)

#: The blocks the page runs pass over, by page and by name, each with its reason.
#: A block's name is the heading it sits under, numbered "(1)", "(2)" in page
#: order where one heading holds more than one Python block. Keep this short: a
#: block that needs a file belongs on a page that writes the file, or uses one
#: of the models bngsim.examples ships.
_NOT_RUN: dict[tuple[str, str], _Need] = {
    ("docs/quickstart.md", "Other model formats (2)"): _BNG2,
    ("docs/user-guide/interchange.md", "Compartmental BNGL (SBML to cBNGL) (2)"): _BNG2,
    ("docs/user-guide/loading-models.md", "Load a model (2)"): _BNG2,
    ("docs/user-guide/loading-models.md", "BNGL models (1)"): _BNG2,
    ("docs/user-guide/loading-models.md", "The actions block"): _BNG2,
    ("docs/user-guide/loading-models.md", "Run the file's own experiment"): _BNG2,
    ("docs/user-guide/loading-models.md", "The network cache"): _BNG2,
    ("docs/user-guide/loading-models.md", "Models from BioModels"): _NETWORK,
    ("docs/user-guide/pybnf.md", "Checkpoint and restart"): _HDF5,
    ("docs/user-guide/results.md", "Read a result by name"): _PANDAS,
    ("docs/user-guide/results.md", "Save and load results (HDF5)"): _HDF5,
    ("docs/user-guide/results.md", "Export to text files"): _PANDAS,
    ("docs/user-guide/sensitivities.md", "Differentiable solving with JAX"): _JAX,
}

#: Blocks whose prose says they warn, with the warning they give: the page run
#: fails on any other warning a block raises, other than a deprecation from a
#: library a block does not call itself.
_WARNS_ON_PURPOSE: dict[tuple[str, str], re.Pattern[str]] = {
    # The run warns naming the assignment-rule species it has no row for.
    ("docs/user-guide/sensitivities.md", "Species set by an `<assignmentRule>`"): re.compile(
        r"Assignment-rule species \['stimulus'\] have NaN"
    ),
}

# A fenced block of any language, so a "# comment" inside a ```bash block is not
# read as a heading, and a Markdown heading outside one.
_ANY_FENCE = re.compile(r"^(`{3,})[^\n]*$.*?^\1\s*$", re.MULTILINE | re.DOTALL)
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


class _Block(NamedTuple):
    line: int  # of the opening fence, 1-based
    name: str  # the heading it sits under, numbered if that heading has several
    offset: int  # newlines before the code, so line numbers in a traceback are the page's
    code: str


def _named_blocks(path: Path) -> list[_Block]:
    """The page's Python blocks, in page order, each named for its heading."""
    text = path.read_text(encoding="utf-8")
    front = re.match(r"---\n.*?\n---\n", text, re.DOTALL)  # YAML front matter
    body_start = front.end() if front else 0
    fenced = [(m.start(), m.end()) for m in _ANY_FENCE.finditer(text)]
    headings = [
        (m.start(), m.group(1))
        for m in _HEADING.finditer(text)
        if m.start() >= body_start and not any(a <= m.start() < b for a, b in fenced)
    ]
    found = []
    for m in _FENCE.finditer(text):
        above = [title for at, title in headings if at < m.start()]
        found.append((m, above[-1] if above else "the top of the page"))
    per_heading: dict[str, int] = {}
    for _, title in found:
        per_heading[title] = per_heading.get(title, 0) + 1
    blocks, seen = [], dict.fromkeys(per_heading, 0)
    for m, title in found:
        seen[title] += 1
        name = title if per_heading[title] == 1 else f"{title} ({seen[title]})"
        blocks.append(
            _Block(
                line=text.count("\n", 0, m.start()) + 1,
                name=name,
                offset=text.count("\n", 0, m.start(1)),
                code=m.group(1),
            )
        )
    return blocks


# ─── What a line's comment says it shows ──────────────────────────────────────
#
# A statement line (an expression, or an assignment to one name, whose value
# is then the name's) whose comment starts with a Python literal states its value:
# "# (3, 2): (n_species, n_params)" states (3, 2), "# 0.6738, which is ..."
# 0.6738, "# True when ..." True, "# 'param.not_found'" that string. A comment of
# words ("# number of laws", "# e.g. [...]") states none. A print line states
# what it prints, as text; a bare expression states its value, as a REPL shows
# it, and a tuple stated for an array is its shape. A print line whose comment
# starts or ends with "..." quotes part of a message: each part between the
# ellipses is in what it prints.


# A literal followed by one of these words states a case, not this run's value:
# "# False if any check failed", "# "stable" or "unstable" for each root".
_CONDITIONAL = frozenset({"if", "when", "where", "unless", "while", "or", "and", "for"})


def _has_ellipsis(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(node))


def _stated_value(comment: str) -> str | None:
    """The longest Python literal *comment* starts with, as written, or None.

    None for a literal with "..." in it (``{"commit": ..., "stale": ...}``
    shows the keys, not the values) and for one that states a case.
    """
    cuts = {len(comment)} | {i for i, c in enumerate(comment) if c in " :,;"}
    for end in sorted(cuts, reverse=True):
        head = comment[:end].strip().rstrip(",;:").rstrip()
        try:
            ast.literal_eval(head)
            tree = ast.parse(head, mode="eval")
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if _has_ellipsis(tree):
            return None
        rest = comment[len(head) :].split()
        if rest and rest[0].lower() in _CONDITIONAL:
            return None
        return head
    return None


def _is_excerpt(comment: str) -> bool:
    return comment.startswith("... ") or comment.endswith(" ...")


def _with_stated_checks(code: str) -> tuple[str, set[int]]:
    """*code* with each statement line whose comment states a value turned into a
    check of that value, on the same line, at the same indent, and the numbers
    of the lines it turned."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, set()  # test_documented_python_blocks_parse reports it
    comments = {
        tok.start[0]: tok.string[1:].strip()
        for tok in tokenize.generate_tokens(io.StringIO(code).readline)
        if tok.type == tokenize.COMMENT
    }
    lines = code.splitlines()
    turned: set[int] = set()
    for node in ast.walk(tree):
        assigned = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        )
        if not (isinstance(node, ast.Expr) or assigned) or node.lineno != node.end_lineno:
            continue
        n, comment = node.lineno, comments.get(node.lineno)
        line = lines[n - 1]
        # The statement is the line, before its comment: not after a `;` or an `if x:`.
        rest = line[node.end_col_offset :].lstrip()
        if comment is None or line[: node.col_offset].strip() or not rest.startswith("#"):
            continue
        value, indent = node.value, line[: node.col_offset]
        if assigned:  # `x = f(...)  # value`: the value is x's
            stated = _stated_value(comment)
            if stated is not None:
                name = node.targets[0].id  # type: ignore[attr-defined]
                statement = ast.get_source_segment(code, node)
                lines[n - 1] = f"{indent}{statement}; __stated__('value', {stated!r}, {n}, {name})"
                turned.add(n)
            continue
        printed = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "print"
            and not value.keywords
        )
        stated = _stated_value(comment)
        if printed and _is_excerpt(comment):
            kind, claim = "excerpt", comment
        elif stated is not None:
            kind, claim = ("printed" if printed else "value"), stated
        else:
            continue
        args = value.args if printed else [value]  # type: ignore[union-attr]
        sources = ", ".join(ast.get_source_segment(code, arg) or "" for arg in args)
        lines[n - 1] = f"{indent}__stated__({kind!r}, {claim!r}, {n}, {sources})"
        turned.add(n)
    return "\n".join(lines) + "\n", turned


def _says(kind: str, claim: str, args: tuple[object, ...]) -> bool:
    """Whether what the line shows is what its comment says (see above)."""
    if kind != "value":
        shown = " ".join(str(arg) for arg in args)
        if kind == "printed":
            return shown == claim
        return all(part.strip() in shown for part in claim.split("...") if part.strip())
    (got,), expected = args, ast.literal_eval(claim)
    if isinstance(expected, tuple) and hasattr(got, "shape"):
        return tuple(got.shape) == expected
    if hasattr(got, "shape"):
        try:
            return bool(np.array_equal(np.asarray(got), np.asarray(expected)))
        except (ValueError, TypeError):
            return False
    try:
        return bool(got == expected)
    except (ValueError, TypeError):
        return False


# A call that empties or prunes the compiled-code cache.
_CACHE_SWEEP = re.compile(r"\b(?:prune|clear|clean)_codegen_cache\(")

# Deprecations from the libraries the examples use (numpy, pandas, xarray, jax)
# come and go with their versions. One raised at a page's own line is the
# example's; one raised inside a library is that library's. A ResourceWarning
# fires when the collector frees an object, which may be another test's: only
# one raised at the page's own line is the page's.
_ECOSYSTEM_WARNINGS = (
    DeprecationWarning,
    PendingDeprecationWarning,
    FutureWarning,
    ImportWarning,
    ResourceWarning,
)


def _documented_pages() -> list[str]:
    """Every page with a Python block, as a path from the repository root."""
    if not (_ROOT / "docs").is_dir():
        return []
    return [
        path.relative_to(_ROOT).as_posix() for path in _markdown_files() if _python_blocks(path)
    ]


# An empty parameter set would skip in pytest's own words, which the skip audit
# does not know: with no docs/, skip in the words _docs_present uses.
_NO_DOCS = pytest.param(
    "",
    marks=pytest.mark.skip(
        reason="docs/ lives in the source root and is not present in an installed wheel"
    ),
)


@pytest.fixture
def bngsim_logger_as_found() -> Iterator[None]:
    """Put the ``bngsim`` logger back as the test found it. A page may turn
    bngsim's logging on, as the solver guide does, and the tests that run after
    it in this process should not log for it."""
    logger = logging.getLogger("bngsim")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)  # setLevel, not the attribute: it also drops the level cache
    logger.propagate = propagate


@pytest.mark.parametrize("page", _documented_pages() or [_NO_DOCS])
@pytest.mark.usefixtures("bngsim_logger_as_found")
def test_each_page_runs_top_to_bottom(
    page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every Python block of the page runs, in page order, in one namespace.

    As a reader pastes the page from its top into one session: in an empty
    directory, so a block may read only a file an earlier block wrote or a model
    bngsim.examples ships, and with each block seeing the names the blocks above
    it bound. A block may not warn unless its prose says it does, and a line
    whose comment states a value must have it. A failure names the page, the
    block and the page line.
    """
    _docs_present()
    path = _ROOT / page
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if _CACHE_SWEEP.search(text):
        # A page that prunes or clears the compiled-code cache, as the
        # code-generation guide does, gets one of its own, not the suite's.
        codegen = importlib.import_module("bngsim._codegen")
        monkeypatch.setattr(codegen, "CACHE_DIR", tmp_path / "cache")
    wrong: list[str] = []

    fired: set[int] = set()

    def stated(kind: str, claim: str, n: int, *args: object) -> None:
        fired.add(n)
        if not _says(kind, claim, args):
            at = block.offset + n  # the code's first line is its fence's
            shown = " ".join(map(str, args)) if kind != "value" else repr(args[0])
            wrong.append(f"line {at}: {lines[at - 1].strip()}\n    shows {shown[:300]!r}")

    namespace: dict[str, object] = {"__name__": "__docs_page__", "__stated__": stated}
    ran = 0
    for block in _named_blocks(path):
        need = _NOT_RUN.get((page, block.name))
        if need is not None and not need.met():
            continue  # reported by test_a_block_the_page_runs_pass_over_is_reported
        failure = None
        checked, stated_here = _with_stated_checks(block.code)
        fired.clear()
        gc.collect()  # so the garbage of the tests before this one is freed outside it
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    source = "\n" * block.offset + checked
                    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
            except Exception as exc:  # noqa: BLE001 - any exception is the finding
                frames = traceback.extract_tb(exc.__traceback__)
                on_page = [f.lineno for f in frames if f.filename == str(path) and f.lineno]
                at = on_page[-1] if on_page else block.line
                failure = (
                    f"{page}: the block at line {block.line}, under {block.name!r}, raised at "
                    f"line {at}:\n    {lines[at - 1].strip()}\n"
                    + "".join(traceback.format_exception_only(type(exc), exc))
                )
        if failure is not None:
            pytest.fail(failure, pytrace=False)
        # A stated line in an `except` that nothing raised into, or in a branch
        # not taken, checked nothing: the example no longer shows what it says.
        for n in sorted(stated_here - fired):
            at = block.offset + n
            wrong.append(f"line {at}: {lines[at - 1].strip()}\n    never ran")
        warned = [
            w
            for w in caught
            if w.filename == str(path) or not issubclass(w.category, _ECOSYSTEM_WARNINGS)
        ]
        expected = _WARNS_ON_PURPOSE.get((page, block.name))
        if expected is not None:
            said = [w for w in warned if expected.search(str(w.message))]
            assert said, f"{page}: the block under {block.name!r} no longer warns, as it says"
            warned = [w for w in warned if w not in said]
        if warned:
            w = warned[0]
            pytest.fail(
                f"{page}: the block at line {block.line}, under {block.name!r}, warned "
                f"({w.filename}:{w.lineno}):\n{w.category.__name__}: {str(w.message)[:400]}",
                pytrace=False,
            )
        ran += 1
    assert ran, f"{page}: no block ran"
    assert not wrong, f"{page}: a line shows what its comment does not say:\n" + "\n".join(wrong)


@pytest.mark.parametrize(
    "page, name", sorted(_NOT_RUN), ids=[f"{page}: {name}" for page, name in sorted(_NOT_RUN)]
)
def test_a_block_the_page_runs_pass_over_is_reported(page: str, name: str) -> None:
    """A block the page run passes over here shows as a skip that says why, so
    the summary counts it; where its need is met, the page run ran it."""
    need = _NOT_RUN[(page, name)]
    if not need.met():
        pytest.skip(
            f"{page}, {name!r} is passed over by the page run: {need.reason}; {need.absent}"
        )


def test_the_blocks_the_page_runs_pass_over_are_few_and_named() -> None:
    """Each :data:`_NOT_RUN` and :data:`_WARNS_ON_PURPOSE` entry names a block a
    page has; a passed-over block makes the call its need names; and no more
    than one block in ten is passed over on a pull request's Python legs."""
    _docs_present()
    blocks = {
        (page, b.name): b.code for page in _documented_pages() for b in _named_blocks(_ROOT / page)
    }
    names = set(blocks)
    assert len(names) > 90, f"only {len(names)} documented Python blocks: did the fence regex rot?"
    stale = sorted((set(_NOT_RUN) | set(_WARNS_ON_PURPOSE)) - names)
    assert not stale, f"entries name blocks no page has (renamed or moved?): {stale}"
    for key, need in _NOT_RUN.items():
        assert need.reason.startswith("needs "), key
        assert need.calls.search(blocks[key]), (
            f"{key} is passed over because it {need.reason}, but it makes no such call: "
            "has a block been added or moved under its heading?"
        )
    off_ci = sum(not need.dev for need in _NOT_RUN.values())
    assert off_ci * 10 <= len(names), (
        f"{off_ci} of {len(names)} blocks never run on CI: give their pages a model instead"
    )


# An issue or pull request reference in prose, a link or an anchor: "#1846",
# "lanl/bngsim#1846", "GH-1846", "issue 111", "PR 12", ".../issues/1846", ".../pull/12",
# and a heading anchor built from one, "-issue-111". A Markdown heading's "# "
# and a code comment's "# 1" have a space after the "#", so they do not match,
# and neither does an HTML character reference, "&#123;".
_ISSUE_REFERENCE = re.compile(
    r"(?<!&)#\d+\b"
    r"|\bGH-\d+\b"
    r"|\b(?:issues?|PRs?|pull requests?)\s+#?\d+\b"
    r"|/(?:issues|pull)/\d+"
    r"|\bissue-\d+\b",
    re.IGNORECASE,
)


def test_user_docs_carry_no_issue_references() -> None:
    """The user guides, the quickstart and the agent guide describe what BNGsim
    does, for its users. Issue numbers belong in the changelog and in code
    comments."""
    _docs_present()
    pages = sorted((_ROOT / "docs" / "user-guide").glob("*.md"))
    pages += [_ROOT / "docs" / "quickstart.md", _ROOT / "python" / "bngsim" / "agent_guide.md"]
    assert len(pages) > 10
    found = []
    for page in pages:
        text = page.read_text(encoding="utf-8")  # whole, as "issue" can end a line
        for match in _ISSUE_REFERENCE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found.append(f"{page.relative_to(_ROOT)}:{line}: {match.group(0)!r}")
    assert not found, "issue references in the user docs:\n" + "\n".join(found)
