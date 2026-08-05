"""The BNGsim half of a run's provenance must identify an ARTIFACT (GH #163).

A parity run records which engines produced it so a golden can be reproduced.
For PyBioNetGen that record has always been a resolved git commit. For BNGsim it
was a bare ``__version__`` — sufficient only while bngsim could not be installed
from PyPI. It can now, so the version string no longer identifies anything: it
bumps only at release, so a PyPI install, a ``ship_wheel.py`` wheel, and every
commit between two releases all report the same one.

The replacement discriminators, both asserted here:
  * ``bngsim_build_commit`` — the commit the loaded ``_bngsim_core`` was compiled
    from, baked in by CMake. Identifies the SOURCE.
  * ``bngsim_install`` — PEP 610 origin. Identifies the ARTIFACT, which the build
    commit alone cannot: the release protocol builds the published wheel from the
    release commit, so a locally built wheel of that commit reports the identical
    build commit (measured: PyPI 0.12.2 and a ship_wheel.py wheel both report
    ``1737003f0c81``).

These tests run wherever bngsim is importable — they do not need the parity env
(no bionetgen), which is deliberate: the bngsim half of the record is read from
bngsim directly, so it must survive a broken/absent bridge.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

_BNG_PARITY = Path(__file__).resolve().parent.parent / "bng_parity"
if str(_BNG_PARITY) not in sys.path:
    sys.path.insert(0, str(_BNG_PARITY))

import bngsim_backend as bb  # noqa: E402

_REPO = _BNG_PARITY.parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _BNG_PARITY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bngsim_or_skip():
    try:
        import bngsim
    except Exception as exc:  # pragma: no cover - env without bngsim
        pytest.skip(f"bngsim not importable: {exc}")
    return bngsim


# ─── the discriminators themselves ────────────────────────────────────────


def test_build_commit_is_a_commit_not_a_version():
    """The whole point: a value a version string cannot supply.

    CMake writes ``rev-parse --short=12``, plus a ``+dirty`` suffix when the tree
    it built from had uncommitted changes. Asserting the SHAPE is what separates
    a real fix from re-recording ``__version__`` under a new key.
    """
    bngsim = _bngsim_or_skip()
    commit = bb.bngsim_build_commit()
    if commit is None:
        pytest.skip("extension built with no reachable git (CMake wrote 'unknown')")
    assert re.fullmatch(r"[0-9a-f]{12}(\+dirty)?", commit), (
        f"bngsim_build_commit {commit!r} is not a 12-char short sha (+optional +dirty)"
    )
    assert commit.split("+")[0] != bngsim.__version__


def test_build_commit_reads_bngsim_own_accessor():
    """One defining site for where the commit lives — bngsim's, not a copy here.

    The harness must never re-derive it (``getattr(_bngsim_core, '__build_commit__')``
    open-coded), or bngsim moving it leaves this reporting a stale/absent value
    while every other reader is fine.
    """
    _bngsim_or_skip()
    from bngsim import _bngsim_core

    baked = getattr(_bngsim_core, "__build_commit__", None)
    expected = None if (not baked or baked == "unknown") else baked
    assert bb.bngsim_build_commit() == expected


def test_editable_install_build_commit_exists_in_this_repo():
    """An editable/local-dir bngsim was built HERE, so its commit must resolve.

    Catches a build-commit string that is well-shaped but fabricated. Skipped for
    a wheel or index install, whose commit legitimately comes from another
    checkout (CI's, for a PyPI wheel).
    """
    _bngsim_or_skip()
    if bb.bngsim_install() not in ("editable", "local-dir"):
        pytest.skip("bngsim is not installed from this source tree")
    commit = bb.bngsim_build_commit()
    if commit is None:
        pytest.skip("extension built with no reachable git")
    r = subprocess.run(
        ["git", "-C", str(_REPO), "cat-file", "-e", f"{commit.split('+')[0]}^{{commit}}"],
        capture_output=True,
    )
    assert r.returncode == 0, f"build commit {commit!r} is not a commit in {_REPO}"


@pytest.mark.parametrize(
    "raw,expected",
    [
        # No direct_url.json at all — pip writes one only for a DIRECT reference,
        # so this is the PyPI/TestPyPI/mirror case the old record could not name.
        (None, "index"),
        ("", "index"),
        # A wheel handed to `pip install` (what bootstrap_parity_env.py does).
        (
            '{"url": "file:///tmp/bngsim_wheel_x/bngsim-0.12.2-cp312-cp312-macosx_11_0_arm64.whl",'
            ' "archive_info": {}}',
            "wheel:bngsim-0.12.2-cp312-cp312-macosx_11_0_arm64.whl",
        ),
        # An editable install (a dev checkout) vs a plain local directory.
        ('{"url": "file:///repo", "dir_info": {"editable": true}}', "editable"),
        ('{"url": "file:///repo", "dir_info": {}}', "local-dir"),
        # A VCS install records the resolved commit; report it, truncated like
        # bionetgen_commit() so the two are comparable at a glance.
        (
            '{"url": "https://github.com/lanl/bngsim", "vcs_info":'
            ' {"vcs": "git", "commit_id": "1737003f0c81aabbccddeeff0011223344556677"}}',
            "vcs:1737003f0c81",
        ),
        # A direct sdist URL — not a wheel, but still not an index install.
        (
            '{"url": "https://files.example/bngsim-0.12.2.tar.gz", "archive_info": {}}',
            "archive:bngsim-0.12.2.tar.gz",
        ),
        # Unparseable metadata must read as "unknown", never as a confident label.
        ("{not json", None),
    ],
)
def test_install_origin_classification(raw, expected):
    assert bb._classify_direct_url(raw) == expected


def test_install_origin_carries_no_absolute_path():
    """This string lands in the committed golden ``_meta`` — no home directories.

    ``parity_golden`` scrubs exactly one field (``bionetgen_path``); a label that
    smuggled a path in would be committed verbatim.
    """
    for raw in (
        '{"url": "file:///Users/someone/wheelhouse/bngsim-0.12.2-cp312-cp312-linux.whl",'
        ' "archive_info": {}}',
        '{"url": "file:///Users/someone/Code/bngsim", "dir_info": {"editable": true}}',
    ):
        label = bb._classify_direct_url(raw)
        assert "/" not in label, f"install label {label!r} leaks a filesystem path"


# ─── how the discriminators reach a run's record ──────────────────────────


def test_backend_status_records_the_bngsim_artifact():
    """Present and populated wherever bngsim is, INDEPENDENT of ``available``.

    ``available`` is the bridge's verdict; the bngsim half of the record is read
    from bngsim directly and collected before bionetgen is touched. An env with a
    broken or absent bridge still records which bngsim is installed in it —
    otherwise the one field that identifies the engine blanks itself exactly when
    something is wrong and you most need it.
    """
    bngsim = _bngsim_or_skip()
    s = bb.backend_status()
    for key in (
        "bngsim_version",
        "bngsim_build_commit",
        "bngsim_install",
        "bngsim_bridge_version",
    ):
        assert key in s
    assert s["bngsim_version"] == bngsim.__version__
    assert s["bngsim_install"] is not None
    assert s["bngsim_build_commit"] == bb.bngsim_build_commit()


def test_operator_self_check_prints_the_artifact_fields():
    """``python bngsim_backend.py`` is what ``--check-only`` runs to vet an env.

    It is the ONE place an operator sees this, so a field added to the record but
    left out of the print is invisible where it matters most.
    """
    _bngsim_or_skip()
    r = subprocess.run(
        [sys.executable, str(_BNG_PARITY / "bngsim_backend.py")],
        capture_output=True,
        text=True,
    )
    for key in ("bngsim_build_commit", "bngsim_install"):
        assert key in r.stdout, f"{key} missing from the operator self-check output"


def test_golden_meta_hoists_only_keys_backend_status_produces():
    """Drift tripwire: ``parity_golden`` reads the backend dict by name.

    The golden hoists a few fields to the top of ``_meta`` beside the rest of the
    reproduction record. Each is a second site naming a key defined in
    ``backend_status()`` — rename one there and the golden silently records null
    forever, which reads exactly like "this run had no such provenance".
    """
    src = (_BNG_PARITY / "parity_golden.py").read_text()
    hoisted = set(re.findall(r"\(backend or \{\}\)\.get\(\"([a-z_]+)\"\)", src))
    assert "bngsim_build_commit" in hoisted, (
        "golden _meta no longer hoists bngsim_build_commit beside bionetgen_commit"
    )
    produced = set(bb.backend_status())
    assert hoisted <= produced, (
        f"golden _meta hoists keys backend_status() never sets: {hoisted - produced}"
    )


# ─── the bootstrap's bngsim source ────────────────────────────────────────


def test_bootstrap_offers_the_pypi_source():
    """The ABORT that used to say "bngsim is not on PyPI" now names three sources.

    It is operator-facing and it used to foreclose a valid option. Asserting the
    parser ACCEPTS ``--bngsim-pypi`` pins the capability, not the wording.
    """
    boot = _load("bootstrap_parity_env")
    args = boot.build_parser().parse_args(["--venv", "x", "--bngsim-pypi", "0.12.2"])
    assert args.bngsim_pypi == "0.12.2"


def test_bootstrap_bngsim_sources_are_mutually_exclusive():
    """Two sources named would otherwise mean "whichever the resolver checks first"."""
    boot = _load("bootstrap_parity_env")
    with pytest.raises(SystemExit):
        boot.build_parser().parse_args(
            ["--venv", "x", "--bngsim-pypi", "0.12.2", "--build-bngsim"]
        )


@pytest.mark.parametrize(
    "value,expected",
    [
        # A bare version is PINNED, not left to resolve: reproducing a golden
        # wants that release, not whatever is newest at install time.
        ("0.12.2", "bngsim==0.12.2"),
        (" 0.12.2 ", "bngsim==0.12.2"),
        ("==0.12.2", "bngsim==0.12.2"),
        (">=0.12,<0.13", "bngsim>=0.12,<0.13"),
    ],
)
def test_pypi_requirement_pins_a_bare_version(value, expected):
    boot = _load("bootstrap_parity_env")
    assert boot._pypi_requirement(value) == expected


def test_pypi_requirement_refuses_a_guess():
    """`--bngsim-pypi bngsim` or a typo must abort, not silently install something."""
    boot = _load("bootstrap_parity_env")
    with pytest.raises(SystemExit):
        boot._pypi_requirement("latest")
