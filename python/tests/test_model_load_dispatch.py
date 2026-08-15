"""Tests for ``Model.load()`` suffix dispatch and the Antimony optional-dependency
error message.

``Model.load`` is a thin router over the format-specific factories, so these
tests pin the *routing* (which factory a suffix selects, what an unroutable
suffix says) rather than re-testing each loader — the per-format behaviour is
covered by test_antimony.py / test_bngsim.py.

The ``antimony`` import guard is tested by hiding the module from importlib, so
the assertion runs identically whether or not the optional dependency is
installed in the environment.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import bngsim
import pytest
from bngsim import Model
from bngsim._exceptions import ModelError

SSYS = Path(__file__).resolve().parents[2] / "benchmarks" / "models" / "antimony" / "ssys"


# ─── Suffix dispatch ──────────────────────────────────────────────────────────


def test_load_dispatches_net(simple_decay_net: Path):
    """.net routes to from_net."""
    model = Model.load(simple_decay_net)
    assert model.n_species == Model.from_net(simple_decay_net).n_species


def test_load_dispatches_sbml(data_dir: Path):
    """.xml routes to from_sbml."""
    sbml = data_dir / "BIOMD0000000003.xml"
    model = Model.load(sbml)
    assert model.n_species == Model.from_sbml(sbml).n_species


def test_load_accepts_sbml_suffix(data_dir: Path, tmp_path: Path):
    """.sbml is accepted as an alias for .xml."""
    src = data_dir / "BIOMD0000000003.xml"
    dst = tmp_path / "model.sbml"
    dst.write_bytes(src.read_bytes())
    assert Model.load(dst).n_species == Model.from_sbml(src).n_species


def test_load_dispatch_is_case_insensitive(data_dir: Path, tmp_path: Path):
    """An upper-cased suffix routes the same as a lower-cased one."""
    dst = tmp_path / "MODEL.XML"
    dst.write_bytes((data_dir / "BIOMD0000000003.xml").read_bytes())
    assert Model.load(dst).n_species > 0


def test_load_dispatches_antimony():
    """.ant routes to from_antimony."""
    pytest.importorskip("antimony")
    if not SSYS.is_dir():
        pytest.skip(f"Antimony benchmark directory not found: {SSYS}")
    ant = next(iter(sorted(SSYS.glob("*.ant"))), None)
    if ant is None:
        pytest.skip("no .ant fixture available")
    assert Model.load(ant).n_species == Model.from_antimony(ant).n_species


def test_load_forwards_defer_jacobian(simple_decay_net: Path):
    """defer_jacobian reaches the underlying factory."""
    # from_net's eager hatch derives at load; the observable effect is only that
    # the call is accepted and the model is usable either way.
    assert Model.load(simple_decay_net, defer_jacobian=False).n_species > 0
    assert Model.load(simple_decay_net, defer_jacobian=None).n_species > 0


# ─── Unroutable suffixes ──────────────────────────────────────────────────────


def test_load_unknown_suffix_lists_supported(tmp_path: Path):
    """An unroutable suffix names every suffix that would have worked."""
    p = tmp_path / "model.json"
    p.write_text("{}")
    with pytest.raises(ModelError) as exc:
        Model.load(p)
    msg = str(exc.value)
    for suffix in (".ant", ".bngl", ".net", ".sbml", ".xml"):
        assert suffix in msg


def test_load_dispatches_bngl(monkeypatch, tmp_path: Path):
    """.bngl routes to from_bngl, forwarding defer_jacobian (GH #162).

    Routing only — network generation itself needs BNG2.pl and is covered by
    test_bngl_loading.py, which skips without it. Stubbing the factory is what
    lets the *dispatch* assertion run in every environment, which is the point
    of this file.
    """
    seen = {}

    def _fake(cls, path, **kw):
        seen.update(path=Path(path), **kw)
        return "sentinel"

    monkeypatch.setattr(Model, "from_bngl", classmethod(_fake))
    p = tmp_path / "model.bngl"
    p.write_text("begin model\nend model\n")

    assert Model.load(p, defer_jacobian=False) == "sentinel"
    assert seen == {"path": p, "defer_jacobian": False}


def test_load_bngl_rejects_compartment_sizes(tmp_path: Path):
    """compartment_sizes= is refused for .bngl, as it already is for .net.

    BNG2.pl bakes each volume into the generated rate constants, so the override
    would have to happen in the source before generation — accepting a dict that
    does nothing is issue #164's expensive failure. Refused before BNG2.pl is
    even looked for, so this asserts in any environment.
    """
    p = tmp_path / "model.bngl"
    p.write_text("begin model\nend model\n")
    with pytest.raises(ModelError, match="compartment_sizes"):
        Model.load(p, compartment_sizes={"cell": 2.0})


def test_load_suffixless_path_is_a_model_error(tmp_path: Path):
    """A path with no suffix at all reports the missing suffix, not a KeyError."""
    p = tmp_path / "model"
    p.write_text("")
    with pytest.raises(ModelError, match="missing"):
        Model.load(p)


def test_load_missing_file_raises_filenotfound(tmp_path: Path):
    """A routable suffix on a nonexistent file still raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Model.load(tmp_path / "nope.net")


# ─── Antimony optional-dependency message ─────────────────────────────────────


@pytest.fixture
def antimony_hidden(monkeypatch: pytest.MonkeyPatch):
    """Make ``import antimony`` fail, however it is already installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "antimony" or name.startswith("antimony."):
            raise ImportError("No module named 'antimony'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "antimony", raising=False)


def test_missing_antimony_names_the_extra(antimony_hidden, tmp_path: Path):
    """The guard names the install command, not just the missing module.

    A bare ``import antimony`` surfaced as ModuleNotFoundError, which never told
    the caller that .ant support ships as an extra.
    """
    p = tmp_path / "model.ant"
    p.write_text("model m()\n  S = 1\nend\n")
    with pytest.raises(ImportError, match=r"bngsim\[antimony\]"):
        Model.from_antimony(p)


def test_missing_antimony_string_loader_names_the_extra(antimony_hidden):
    """The string entry point carries the same message as the file one."""
    with pytest.raises(ImportError, match=r"bngsim\[antimony\]"):
        Model.from_antimony_string("model m()\n  S = 1\nend\n")


def test_missing_antimony_via_load_names_the_extra(antimony_hidden, tmp_path: Path):
    """Model.load('.ant') surfaces the guard rather than swallowing it."""
    p = tmp_path / "model.ant"
    p.write_text("model m()\n  S = 1\nend\n")
    with pytest.raises(ImportError, match=r"bngsim\[antimony\]"):
        Model.load(p)


def test_has_antimony_flag_matches_importability():
    """The advertised capability flag agrees with the actual import."""
    try:
        import antimony  # noqa: F401

        installed = True
    except ImportError:
        installed = False
    assert bngsim.HAS_ANTIMONY is installed
