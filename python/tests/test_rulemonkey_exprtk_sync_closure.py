"""The RuleMonkey drift guard covers every bngsim file ``expression.cpp`` includes.

RuleMonkey's standalone build compiles its own copy of bngsim's expression
layer (``third_party/bngsim_expr`` plus ``third_party/exprtk``). Inside a bngsim
build it links the host ``bngsim::expression`` target instead, and that copy is
never compiled. ``scripts/vendor_rulemonkey.py`` refuses to vendor while the copy
differs from this tree, but only for the files ``EXPRTK_SYNC_FILES`` lists.

That list was written by hand, and it fell behind. GH #1330 made
``expression.hpp`` include ``bngsim/dual.hpp`` and ``expression.cpp`` include
``exprtk_dual_pre.hpp`` and ``exprtk_dual.hpp``; GH #1235 made it include
``bngsim/c_locale_numbers.hpp``. None of the four was listed. The guard still
fired, because ``expression.cpp`` itself had moved, but once RuleMonkey's copy
carried the four headers, a later edit to any one of them would have passed the
guard while RuleMonkey's copy kept the old text. ``expr_compat.hpp`` was added to
the list after GH #49 for the same reason.

So this test follows ``#include "..."`` and ``#include <...>`` lines from
``src/expression.cpp``, resolving each as the compiler does for the
``bngsim_expression`` target, and asks for the list to be exactly the bngsim
files they reach. It reads every include line, so one under ``#if 0`` or in a
comment counts (and fails loudly), while a computed ``#include MACRO`` is not
seen. It also checks that each listed RuleMonkey path mirrors its file's place
here, the layout RuleMonkey's ``rm_bngsim_expr`` include paths rely on; it
checks the table, not RuleMonkey's tree.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_RULEMONKEY = REPO_ROOT / "scripts" / "vendor_rulemonkey.py"

pytestmark = pytest.mark.skipif(
    not VENDOR_RULEMONKEY.exists(),
    reason="scripts/ is not in this checkout (installed package)",
)

#: Where bngsim_expression's include path reaches (CMakeLists.txt): the public
#: headers and the vendored ExprTk header. A quoted include is looked up beside
#: the including file first, as the compiler does.
INCLUDE_DIRS = (REPO_ROOT / "include", REPO_ROOT / "third_party" / "exprtk")

#: Where each bngsim root lands in RuleMonkey's tree (RuleMonkey's
#: scripts/vendor_exprtk.py and the rm_bngsim_expr target in its CMakeLists.txt).
RULEMONKEY_LAYOUT = {
    REPO_ROOT / "include": "third_party/bngsim_expr/include",
    REPO_ROOT / "src": "third_party/bngsim_expr/src",
    REPO_ROOT / "third_party" / "exprtk": "third_party/exprtk",
}

_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"\n]+)[>"]', re.MULTILINE)


def _load_vendor_rulemonkey():
    name = "_vendor_rulemonkey_for_sync_closure"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, VENDOR_RULEMONKEY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve(includer: Path, delimiter: str, target: str) -> Path | None:
    """The bngsim file ``#include`` reaches from ``includer``, or None (a system header)."""
    candidates = [includer.parent / target] if delimiter == '"' else []
    candidates += [root / target for root in INCLUDE_DIRS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _include_closure(start: Path) -> set[Path]:
    seen = {start.resolve()}
    pending = [start.resolve()]
    while pending:
        current = pending.pop()
        text = current.read_text(encoding="utf-8", errors="replace")
        for delimiter, target in _INCLUDE.findall(text):
            found = _resolve(current, delimiter, target.strip())
            if found is not None and found not in seen:
                seen.add(found)
                pending.append(found)
    return seen


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_the_walk_reaches_the_files_every_build_has_needed():
    """Not vacuous: the walk finds what expression.cpp has included since GH #49."""
    closure = {_rel(p) for p in _include_closure(REPO_ROOT / "src" / "expression.cpp")}
    assert {
        "src/expression.cpp",
        "include/bngsim/expression.hpp",
        "include/bngsim/expr_compat.hpp",
        "third_party/exprtk/exprtk.hpp",
    } <= closure


def test_sync_files_are_exactly_what_expression_cpp_includes():
    vendor = _load_vendor_rulemonkey()
    listed = {_rel(p.resolve()) for p in vendor.EXPRTK_SYNC_FILES.values()}
    reached = {_rel(p) for p in _include_closure(REPO_ROOT / "src" / "expression.cpp")}

    unlisted = sorted(reached - listed)
    assert not unlisted, (
        "src/expression.cpp includes bngsim files that EXPRTK_SYNC_FILES in "
        "scripts/vendor_rulemonkey.py does not list, so the RuleMonkey drift guard "
        f"cannot see them: {unlisted}. Add each one there, and to the FILES table of "
        "RuleMonkey's scripts/vendor_exprtk.py upstream."
    )
    unreached = sorted(listed - reached)
    assert not unreached, (
        f"EXPRTK_SYNC_FILES lists files src/expression.cpp no longer includes: {unreached}"
    )


def test_each_sync_file_sits_where_its_includer_finds_it_in_rulemonkey():
    vendor = _load_vendor_rulemonkey()
    misplaced = []
    for rulemonkey_rel, bngsim_path in vendor.EXPRTK_SYNC_FILES.items():
        bngsim_path = bngsim_path.resolve()
        for root, rulemonkey_root in RULEMONKEY_LAYOUT.items():
            if bngsim_path.is_relative_to(root.resolve()):
                expected = (
                    f"{rulemonkey_root}/{bngsim_path.relative_to(root.resolve()).as_posix()}"
                )
                break
        else:
            expected = None
        if rulemonkey_rel != expected:
            misplaced.append((rulemonkey_rel, _rel(bngsim_path), expected))
    assert not misplaced, (
        "EXPRTK_SYNC_FILES pairs a bngsim file with a RuleMonkey path that is not its "
        f"mirror (RuleMonkey path, bngsim path, expected RuleMonkey path): {misplaced}"
    )
