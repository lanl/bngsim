"""``--summary`` previews a vendor refresh; it must never perform one.

`scripts/vendor_rulemonkey.py --summary` printed the impact summary and then
fell through to ``ensure_clean_destination`` / ``copy_tree`` /
``write_metadata``: it *performed* the refresh as a side effect of being asked
what the refresh would do. ``scripts/RULEMONKEY_VENDORING.md`` documents that
exact invocation twice -- once under "1. Preview Before Writing" and once under
"Preview the impact without writing:" -- so following the doc on a clean tree
silently rewrote twelve files under ``third_party/rulemonkey/``.

The four siblings all stop after printing. ``vendor_nfsim.py`` and
``vendor_sundials.py`` say ``return 0`` at the precise spot where
``vendor_rulemonkey.py`` said ``print()``; ``vendor_mir.py`` and
``vendor_exprtk.py`` return the summary call directly. Five scripts in one
family, one never updated -- the same shape as `test_vendor_script_git_cwd.py`,
and the reason the property is pinned here for the whole family rather than for
the one script that happened to drift.

Why this is a behaviour test and not a grep for ``return 0``: the defect was
*self-consistent with its own help text*. ``--summary`` advertised "...impact
before refreshing", which describes the buggy code accurately; only the doc and
the four siblings disagreed. Help text that agrees with a bug is not evidence,
so the load-bearing tests here drive ``main()`` and watch for writes. (There is
a secondary lint on the help strings at the end, clearly marked as such.)

Two ways a test like this passes without proving anything, both guarded:

* **Vacuous pass.** A script that dies before reaching its summary path writes
  nothing and would "pass". Every case asserts the summary function was
  actually reached, so a stub drifting out of sync with a script fails loudly
  instead of silently proving nothing.
* **Passing by never writing at all.** A fix that broke the refresh outright
  would also satisfy "summary does not write". `test_a_plain_refresh_still_writes`
  is the positive control: it proves this harness can see the writes it reports
  as absent.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
VENDOR_RULEMONKEY = SCRIPTS_DIR / "vendor_rulemonkey.py"

pytestmark = pytest.mark.skipif(
    not VENDOR_RULEMONKEY.exists(),
    reason="scripts/ is not in this checkout (installed package)",
)

#: Any 40-hex string. The stubs hand the same one to every seam that reports a
#: commit, so a script's own "resolved ref matches canonical main" consistency
#: check is satisfied without a real upstream checkout.
COMMIT = "0" * 40

#: Wordings the family uses to promise a no-write preview. Three spellings for
#: one property is why the lint below matches on meaning rather than on a
#: single phrase.
NO_WRITE_PHRASES = ("write nothing", "no-write", "without writing")


def _load(path: Path, name: str) -> types.ModuleType:
    """Import a scripts/ module by path -- scripts/ is not a package."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def _load_vendor_script(script: str) -> types.ModuleType:
    return _load(SCRIPTS_DIR / script, f"_vendor_summary_{Path(script).stem}")


def _const(value: object):
    """A stub that ignores its arguments and returns ``value``.

    Deliberately signature-agnostic: these seams differ in arity across the
    five scripts, and this file pins *whether a write happens*, not the call
    signatures of each script's internals.
    """

    def stub(*args: object, **kwargs: object) -> object:
        return value

    return stub


class Recorder:
    """Records that a function was called, without perturbing control flow."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def watch(self, monkeypatch: pytest.MonkeyPatch, mod: types.ModuleType, name: str) -> None:
        # Record and no-op rather than raise: an exploding stub would change
        # the control flow under test, and a script that writes twice should
        # get to report both writes.
        def stub(*args: object, **kwargs: object) -> object:
            self.calls.append(name)
            return None

        monkeypatch.setattr(mod, name, stub)


def _guard_vendored_tree(monkeypatch: pytest.MonkeyPatch, mod: types.ModuleType) -> list[str]:
    """Watch the filesystem primitives for writes into ``mod.VENDOR_DIR``.

    Belt-and-braces behind the per-script ``writers`` list: those name the
    helpers that exist today, and a script that grew a bare
    ``(VENDOR_DIR / "VENDOR.json").write_text(...)`` inside ``main()`` would
    slip past them.

    Scoped to the vendored directory on purpose. A blanket "no shutil.rmtree"
    guard flags ``tempfile.TemporaryDirectory`` cleaning up its own export
    directory, which is not a write to anything that matters -- so calls
    outside ``VENDOR_DIR`` are delegated to the real implementation and the
    scripts keep behaving normally.
    """
    seen: list[str] = []
    vendor_dir = Path(mod.VENDOR_DIR).resolve()

    def guard(target: object, name: str) -> None:
        real = getattr(target, name)

        def stub(*args: object, **kwargs: object) -> object:
            for arg in args:
                if not isinstance(arg, (str, Path)):
                    continue
                try:
                    candidate = Path(arg).resolve()
                except (OSError, ValueError):  # pragma: no cover - defensive
                    continue
                if candidate == vendor_dir or vendor_dir in candidate.parents:
                    seen.append(f"{name} -> {candidate}")
                    return None
            return real(*args, **kwargs)

        monkeypatch.setattr(target, name, stub)

    guard(Path, "write_text")
    guard(Path, "write_bytes")
    guard(Path, "unlink")
    guard(Path, "mkdir")
    guard(shutil, "copytree")
    guard(shutil, "copy2")
    guard(shutil, "rmtree")
    guard(shutil, "move")
    return seen


#: Per-script recipe for reaching the ``--summary`` branch of ``main()``.
#:
#: ``stubs``   seams that would otherwise need a real upstream checkout;
#:             ``"{repo_path}"`` is replaced with a temporary directory
#: ``summary`` the function that prints the preview -- reaching it is the
#:             non-vacuity proof for the case
#: ``writers`` every module-level function that mutates the vendored tree
SUMMARY_CASES: dict[str, dict[str, object]] = {
    "vendor_rulemonkey.py": {
        "argv": ["--rulemonkey-repo", "{repo}", "--summary"],
        "stubs": {
            "validate_export_contract": None,
            "verify_source_checkout": {
                "canonical_main_commit": COMMIT,
                "canonical_main_ref": "origin/main",
                "canonical_remote": "origin",
            },
            "resolve_ref": ("origin/main", COMMIT),
            "export_rulemonkey_tree": "{repo_path}",
            "compare_trees": [],
        },
        "summary": "print_summary",
        "writers": ("ensure_clean_destination", "copy_tree", "write_metadata"),
    },
    "vendor_nfsim.py": {
        "argv": ["--nfsim-repo", "{repo}", "--summary"],
        "stubs": {
            "resolve_ref": ("origin/master", COMMIT),
            "export_nfsim_tree": "{repo_path}",
            "compare_trees": [],
        },
        "summary": "print_summary",
        "writers": ("ensure_clean_destination", "copy_tree", "write_metadata"),
    },
    "vendor_mir.py": {
        "argv": ["--mir-repo", "{repo}", "--summary"],
        "stubs": {
            "verify_source_checkout": {
                "canonical_main_commit": COMMIT,
                "canonical_main_ref": "origin/master",
            },
            "resolve_ref": ("origin/master", COMMIT),
        },
        # mir folds "print the preview" and "return the exit code" into one call.
        "summary": "do_summary",
        "writers": ("do_write",),
    },
    "vendor_exprtk.py": {
        "argv": ["--exprtk-repo", "{repo}", "--summary"],
        "stubs": {
            "verify_source_checkout": {
                "canonical_main_commit": COMMIT,
                "canonical_main_ref": "origin/master",
            },
            "resolve_ref": ("origin/master", COMMIT),
            "header_bytes_from_commit": b"// exprtk stub\n",
            "extract_header_metadata": {},
            "read_vendor_metadata": {},
        },
        "summary": "print_summary",
        "writers": ("write_refresh",),
    },
    "vendor_sundials.py": {
        # sundials vendors a release archive, so it has no --*-repo argument.
        "argv": ["--summary"],
        "stubs": {
            "read_vendor_metadata": {},
            "candidate_archive_path": "{repo_path}",
            "inspect_archive": {},
            "metadata_defaults": {},
            "build_metadata": {},
            "metadata_mismatches": [],
        },
        "summary": "print_summary",
        "writers": ("write_metadata",),
    },
}


class Run:
    """What one ``main()`` invocation did."""

    def __init__(self, rc: int, summarized: Recorder, wrote: Recorder, fs: list[str]) -> None:
        self.rc = rc
        self.summarized = summarized.calls
        self.wrote = wrote.calls
        self.fs = fs


def _run(
    script: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_argv: tuple[str, ...] = (),
    include_summary_flag: bool = True,
    stub_overrides: dict[str, object] | None = None,
) -> Run:
    """Drive one vendor script's ``main()`` and report what it called."""
    case = SUMMARY_CASES[script]
    mod = _load_vendor_script(script)

    repo = tmp_path / "vendor-candidate"
    repo.mkdir(exist_ok=True)

    stubs = dict(case["stubs"])  # type: ignore[arg-type]
    stubs.update(stub_overrides or {})
    for name, value in stubs.items():
        monkeypatch.setattr(mod, name, _const(repo if value == "{repo_path}" else value))

    summarized = Recorder()
    if case["summary"] == "do_summary":
        # mir returns do_summary(...) directly, so the stub owes it an exit code.
        def do_summary(*args: object, **kwargs: object) -> int:
            summarized.calls.append("do_summary")
            return 0

        monkeypatch.setattr(mod, "do_summary", do_summary)
    else:
        summarized.watch(monkeypatch, mod, str(case["summary"]))

    wrote = Recorder()
    for name in tuple(case["writers"]):  # type: ignore[arg-type]
        wrote.watch(monkeypatch, mod, name)

    fs = _guard_vendored_tree(monkeypatch, mod)

    argv = [arg.format(repo=str(repo)) for arg in list(case["argv"])]  # type: ignore[arg-type]
    if not include_summary_flag:
        argv = [arg for arg in argv if arg != "--summary"]
    monkeypatch.setattr(sys, "argv", [script, *argv, *extra_argv])

    return Run(mod.main(), summarized, wrote, fs)


class TestSummaryDoesNotWrite:
    """The family-wide property, so this cannot drift back in one script."""

    @pytest.mark.parametrize("script", sorted(SUMMARY_CASES))
    def test_summary_reaches_the_preview_and_writes_nothing(
        self, script: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        run = _run(script, monkeypatch, tmp_path)

        # Non-vacuity first: a script that never got as far as its preview
        # trivially "writes nothing", and would hide a real regression.
        assert run.summarized, (
            f"{script} --summary never reached {SUMMARY_CASES[script]['summary']}(), so this "
            "case proves nothing about writing -- the stub table has drifted from the script."
        )
        assert not run.wrote, (
            f"{script} --summary called {', '.join(sorted(set(run.wrote)))}. --summary is a "
            "preview: it must report what a refresh would do, not do it."
        )
        assert not run.fs, f"{script} --summary wrote into the vendored tree: {run.fs}"
        assert run.rc == 0, f"{script} --summary returned {run.rc}, expected a clean 0"

    def test_every_vendor_script_is_covered(self) -> None:
        """A sixth script must not be able to join the family unnoticed.

        Silently skipping an unknown script is how a family-wide invariant
        quietly becomes a single-script one again.
        """
        on_disk = {path.name for path in SCRIPTS_DIR.glob("vendor_*.py")}
        assert on_disk, "no scripts/vendor_*.py found"
        assert on_disk == set(SUMMARY_CASES), (
            "scripts/vendor_*.py and this file's SUMMARY_CASES disagree: "
            f"unlisted={sorted(on_disk - set(SUMMARY_CASES))}, "
            f"stale={sorted(set(SUMMARY_CASES) - on_disk)}. Every vendor script owes "
            "this file a demonstration that --summary writes nothing."
        )


class TestRuleMonkeySummaryRegression:
    """The script that actually drifted, pinned in more detail."""

    def test_a_plain_refresh_still_writes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Positive control: without ``--summary`` the refresh must still happen.

        Without this, a fix that simply stopped writing under every flag would
        pass the whole file. This is what makes the absence of writes above a
        real observation rather than a blind harness.
        """
        run = _run("vendor_rulemonkey.py", monkeypatch, tmp_path, include_summary_flag=False)
        assert run.rc == 0
        assert "copy_tree" in run.wrote, (
            "a plain refresh no longer copies the exported tree -- the no-write "
            "assertions in this file would then be vacuous"
        )
        assert "write_metadata" in run.wrote, "a plain refresh no longer writes VENDOR.json"

    def test_summary_with_check_still_checks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``--summary --check`` keeps its combined meaning.

        The fix returns from the summary branch only when ``--check`` is
        absent, matching vendor_nfsim.py. Combined, the summary still prints
        and the check still decides the exit code -- the early return must not
        swallow the check.
        """
        run = _run(
            "vendor_rulemonkey.py",
            monkeypatch,
            tmp_path,
            extra_argv=("--check",),
            stub_overrides={"compare_trees": ["differs: src/foo.cpp"]},
        )
        assert run.summarized, "--summary --check did not print the summary"
        assert run.rc == 1, (
            "--summary --check reported a clean tree despite a difference: the early "
            "return from the summary branch swallowed the check"
        )
        assert not run.wrote, f"--summary --check wrote via {sorted(set(run.wrote))}"
        assert not run.fs


def test_summary_help_never_promises_a_refresh() -> None:
    """Secondary lint: no ``--summary`` help may describe writing afterwards.

    Not the property -- the behaviour tests above are. This exists because the
    drifted script's help read "...impact before refreshing", which described
    the bug correctly and would have led a reader to think the write was the
    intent rather than the defect.
    """
    for script in sorted(SUMMARY_CASES):
        source = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
        marker = '"--summary",'
        assert marker in source, f"{script} has no --summary flag"
        help_text = source.split(marker, 1)[1].split(")", 1)[0]
        assert any(phrase in help_text for phrase in NO_WRITE_PHRASES), (
            f"{script}'s --summary help does not promise a no-write preview "
            f"(looked for {NO_WRITE_PHRASES}): {help_text.strip()}"
        )
        assert "before refreshing" not in help_text, (
            f"{script}'s --summary help still says it previews 'before refreshing', which "
            "describes a summary that then writes."
        )
