"""Issue #1494: a BNGL model that reads another file is not served a stale network.

``Model.from_bngl`` caches the network BNG2.pl generates under a key built from the
flattened ``.bngl`` text and the BNG2.pl file. A model that reads a parameter set
with ``readFile({file=>..., blocks=>["parameters"]})`` before ``generate_network``
depends on that file too, which the key never saw: after the file went from k = 1 to
k = 2, the load still had k = 1 and A(1) = 36.79 where 100*exp(-2) = 13.53. The key
also missed the Perl2 library BNG2.pl actually loads, which ``$BNGPATH`` can point
at another install, and which a BioNetGen checkout edits in place. And BNG2.pl ran
from a scratch directory, so a relative ``readFile`` path was refused.

A model that keeps a read action is now generated afresh on every load, the key
includes the library, and BNG2.pl runs from the model's directory.

The tests marked as needing BNG2.pl run it for real; the rest stand in a fake for
it that records how it was called.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import bngsim
import pytest
from bngsim import _bngl_loader
from bngsim._bngpath import resolve_bng

_BNG = resolve_bng()
needs_bng2 = pytest.mark.skipif(not _BNG.ok, reason=f"BNG2.pl unavailable: {_BNG.why_not()}")

MODEL = """\
begin model
begin molecule types
  A()
end molecule types
begin seed species
  A() 100
end seed species
begin observables
  Molecules Atot A()
end observables
begin reaction rules
  A() -> 0 k
end reaction rules
end model
"""


def _params(k: float) -> str:
    return f"begin parameters\n  k {k}\nend parameters\n"


def _driver(params_ref: str, *, after_generate: bool = False) -> str:
    read = f'readFile({{file=>"{params_ref}", blocks=>["parameters"]}})\n'
    generate = "generate_network({overwrite=>1})\n"
    # BNG2.pl reads the model block where it stands, so a parameter set the
    # rules use is read above it.
    return MODEL + generate + read if after_generate else read + MODEL + generate


@pytest.fixture
def cache(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(_bngl_loader, "CACHE_DIR", cache)
    return cache


# ─── A fake BNG2.pl ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_bng2(tmp_path, monkeypatch):
    """Stand in for BNG2.pl: write a decay network whose k is the one in the file
    the scratch model's readFile names (resolved from the directory BNG2.pl runs
    in), else 1, and record each call's working directory."""
    bng2 = tmp_path / "bng" / "BNG2.pl"
    bng2.parent.mkdir()
    bng2.write_text("# not run\n")
    calls: list[Path | None] = []

    def run(bng2_path, scratch, workdir, *, timeout, source_name, cwd=None):
        calls.append(cwd)
        m = re.search(r'readFile\(\{file=>"([^"]+)"', scratch.read_text())
        k = 1.0
        if m:
            params = Path(m.group(1))
            params = params if params.is_absolute() else Path(cwd or workdir) / params
            found = re.search(r"\bk\s+(\S+)", params.read_text()) if params.is_file() else None
            k = float(found.group(1)) if found else k
        net = workdir / f"{scratch.stem}.net"
        net.write_text(
            f"begin parameters\n 1 k {k}\nend parameters\n"
            "begin species\n 1 A() 100\nend species\n"
            "begin reactions\n 1 1 0 k\nend reactions\n"
            "begin groups\n 1 Atot 1\nend groups\n"
        )
        return net

    monkeypatch.setattr(_bngl_loader, "resolve_bng2", lambda explicit=None: bng2)
    monkeypatch.setattr(_bngl_loader, "_run_bng2", run)
    return calls


def test_an_edited_parameter_file_is_read_again(tmp_path, cache, fake_bng2):
    """The second load was a cache hit and kept k = 1."""
    params = tmp_path / "params.bngl"
    model = tmp_path / "m.bngl"
    model.write_text(_driver(str(params)))
    params.write_text(_params(1))
    assert bngsim.Model.from_bngl(model).get_param("k") == 1.0
    params.write_text(_params(2))
    assert bngsim.Model.from_bngl(model).get_param("k") == 2.0
    assert len(fake_bng2) == 2
    assert not cache.exists() or not any(cache.iterdir())


def test_a_relative_read_path_resolves_against_the_models_directory(tmp_path, cache, fake_bng2):
    """BNG2.pl ran from a scratch directory, where a relative path names nothing."""
    (tmp_path / "params.bngl").write_text(_params(3))
    model = tmp_path / "m.bngl"
    model.write_text(_driver("params.bngl"))
    assert bngsim.Model.from_bngl(model).get_param("k") == 3.0
    assert fake_bng2 == [tmp_path]


def test_a_model_that_reads_nothing_is_still_served_from_the_cache(tmp_path, cache, fake_bng2):
    """Control: a second load of an unchanged model is a cache hit."""
    model = tmp_path / "m.bngl"
    model.write_text(MODEL + "generate_network({overwrite=>1})\n")
    first = bngsim.Model.from_bngl(model)
    second = bngsim.Model.from_bngl(model)
    assert first.get_param("k") == second.get_param("k") == 1.0
    assert len(fake_bng2) == 1


def test_a_read_after_generate_network_does_not_turn_the_cache_off(tmp_path, cache, fake_bng2):
    """Control: a readFile after generate_network belongs to the experiment, which
    the loader drops, so the network does not depend on the file."""
    params = tmp_path / "params.bngl"
    params.write_text(_params(5))
    model = tmp_path / "m.bngl"
    model.write_text(_driver(str(params), after_generate=True))
    bngsim.Model.from_bngl(model)
    bngsim.Model.from_bngl(model)
    assert len(fake_bng2) == 1


@pytest.mark.parametrize(
    "action",
    [
        'readFile({file=>"other.bngl", skip_actions=>1})',
        'readModel({file=>"other.bngl"})',
        'readNetwork({file=>"other.net"})',
        'readSBML("other.xml")',
        'readNFspecies("seed.species")',
    ],
)
def test_every_read_action_turns_the_cache_off(tmp_path, cache, fake_bng2, action):
    model = tmp_path / "m.bngl"
    model.write_text(MODEL + action + "\ngenerate_network({overwrite=>1})\n")
    bngsim.Model.from_bngl(model)
    bngsim.Model.from_bngl(model)
    assert len(fake_bng2) == 2


# ─── The key follows the library BNG2.pl loads ───────────────────────────────


def _install(root: Path, version: str) -> Path:
    (root / "Perl2" / "XML").mkdir(parents=True)
    (root / "Perl2" / "BNGModel.pm").write_text("package BNGModel;\n1;\n")
    (root / "Perl2" / "XML" / "Reader.pm").write_text("1;\n")
    (root / "VERSION").write_text(version + "\n")
    (root / "BNG2.pl").write_text("# driver\n")
    return root / "BNG2.pl"


@pytest.fixture
def no_library_env(monkeypatch):
    monkeypatch.delenv("BNGPATH", raising=False)
    monkeypatch.delenv("BioNetGenRoot", raising=False)


@pytest.mark.parametrize("var", ["BNGPATH", "BioNetGenRoot"])
def test_the_key_follows_a_library_another_install_supplies(
    tmp_path, no_library_env, monkeypatch, var
):
    """BNG2.pl loads Perl2 from $BNGPATH (else $BioNetGenRoot) whatever BNG2.pl
    was resolved from; the key named only BNG2.pl."""
    bng2 = _install(tmp_path / "a", "2.9.3")
    _install(tmp_path / "b", "2.9.9")
    own = _bngl_loader._cache_key("model", bng2)
    monkeypatch.setenv(var, str(tmp_path / "b"))
    assert _bngl_loader._cache_key("model", bng2) != own
    monkeypatch.setenv(var, str(tmp_path / "a"))
    assert _bngl_loader._cache_key("model", bng2) == own


def test_bngpath_wins_over_bionetgenroot(tmp_path, no_library_env, monkeypatch):
    bng2 = _install(tmp_path / "a", "2.9.3")
    monkeypatch.setenv("BNGPATH", str(tmp_path / "b"))
    monkeypatch.setenv("BioNetGenRoot", str(tmp_path / "c"))
    assert _bngl_loader._bng2_library_root(bng2) == tmp_path / "b"


def test_an_empty_bngpath_is_the_root_directory_as_perl_reads_it(
    tmp_path, no_library_env, monkeypatch
):
    """Perl's ``exists`` takes a set but empty $BNGPATH, and catdir("", "Perl2") is
    /Perl2, so BNG2.pl does not fall back to its own directory."""
    bng2 = _install(tmp_path / "a", "2.9.3")
    monkeypatch.setenv("BNGPATH", "")
    assert _bngl_loader._bng2_library_root(bng2) == Path(os.sep)


@pytest.mark.parametrize("edit", ["Perl2/BNGModel.pm", "Perl2/XML/Reader.pm", "VERSION"])
def test_an_in_place_edit_of_the_library_changes_the_key(tmp_path, no_library_env, edit):
    """A BioNetGen checkout edits its Perl2 modules in place; BNG2.pl itself is
    unchanged, and so was the key."""
    bng2 = _install(tmp_path / "a", "2.9.3")
    before = _bngl_loader._cache_key("model", bng2)
    target = tmp_path / "a" / edit
    target.write_text(target.read_text() + "# edited\n")
    assert _bngl_loader._cache_key("model", bng2) != before


def test_a_file_replaced_with_its_time_kept_changes_the_key(tmp_path, no_library_env):
    """An upgrade by tar or rsync -t keeps each file's time; the size still moves."""
    bng2 = _install(tmp_path / "a", "2.9.3")
    target = tmp_path / "a" / "Perl2" / "BNGModel.pm"
    st = target.stat()
    before = _bngl_loader._cache_key("model", bng2)
    target.write_text(target.read_text() + "sub added { 1 }\n")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert _bngl_loader._cache_key("model", bng2) != before


def test_an_edit_that_keeps_the_size_changes_the_key(tmp_path, no_library_env):
    """The modification time is part of the key, as it is for BNG2.pl itself."""
    bng2 = _install(tmp_path / "a", "2.9.3")
    target = tmp_path / "a" / "Perl2" / "BNGModel.pm"
    before = _bngl_loader._cache_key("model", bng2)
    target.write_text(target.read_text().replace("1;", "2;"))
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert _bngl_loader._cache_key("model", bng2) != before


def test_a_link_to_bng2_pl_keys_on_the_library_beside_its_target(tmp_path, no_library_env):
    """BNG2.pl takes its library from its own directory with links resolved."""
    bng2 = _install(tmp_path / "a", "2.9.3")
    link_dir = tmp_path / "bin"
    link_dir.mkdir()
    (link_dir / "BNG2.pl").symlink_to(bng2)
    assert _bngl_loader._bng2_library_root(link_dir / "BNG2.pl") == bng2.resolve().parent


# ─── With BNG2.pl itself ─────────────────────────────────────────────────────


def _atot_at_1(model_path: Path) -> tuple[float, float]:
    m = bngsim.Model.from_bngl(model_path)
    r = bngsim.Simulator(m, method="ode").run(t_span=(0, 1), n_points=2, rtol=1e-10, atol=1e-10)
    return m.get_param("k"), float(r.observables["Atot"][-1])


@needs_bng2
def test_bng2_an_edited_parameter_file_is_read_again(tmp_path, cache):
    """The issue's case: A(1) stayed 36.79 (k = 1) after the file said k = 2."""
    params = tmp_path / "params.bngl"
    model = tmp_path / "m.bngl"
    model.write_text(_driver(str(params)))
    for k in (1, 2, 3):
        params.write_text(_params(k))
        loaded_k, atot = _atot_at_1(model)
        assert loaded_k == k
        assert atot == pytest.approx(100 * math.exp(-k), rel=1e-7)


@needs_bng2
def test_bng2_a_relative_parameter_file_is_read_from_the_models_directory(tmp_path, cache):
    """Refused before: BNG2.pl ran in a scratch directory and could not find it."""
    (tmp_path / "params.bngl").write_text(_params(2))
    model = tmp_path / "m.bngl"
    model.write_text(_driver("params.bngl"))
    loaded_k, atot = _atot_at_1(model)
    assert loaded_k == 2
    assert atot == pytest.approx(100 * math.exp(-2), rel=1e-7)


@needs_bng2
def test_bng2_an_edited_model_read_by_a_driver_is_read_again(tmp_path, cache):
    """A driver that reads the whole model with readFile(skip_actions=>1)."""
    inner = tmp_path / "inner.bngl"
    driver = tmp_path / "driver.bngl"
    driver.write_text(
        f'readFile({{file=>"{inner}", skip_actions=>1}})\ngenerate_network({{overwrite=>1}})\n'
    )
    for k in (1, 2):
        inner.write_text(
            MODEL.replace("begin molecule types", _params(k) + "begin molecule types")
        )
        loaded_k, atot = _atot_at_1(driver)
        assert loaded_k == k
        assert atot == pytest.approx(100 * math.exp(-k), rel=1e-7)


@needs_bng2
def test_bng2_a_model_that_reads_nothing_is_still_cached(tmp_path, cache):
    """Control: a second load of an unchanged model is served from the cache."""
    model = tmp_path / "m.bngl"
    model.write_text(MODEL.replace("begin molecule types", _params(1) + "begin molecule types"))
    first = _bngl_loader.bngl_to_net(model)
    assert first.parent == cache
    mtime = first.stat().st_mtime_ns
    assert _bngl_loader.bngl_to_net(model) == first
    assert first.stat().st_mtime_ns == mtime


@needs_bng2
def test_bng2_runs_with_the_library_bngpath_names(tmp_path, cache, monkeypatch):
    """Control: the key reads the library from $BNGPATH, as BNG2.pl does, and a
    load with it set still generates and caches."""
    monkeypatch.setenv("BNGPATH", str(Path(_BNG.bng2_pl).resolve().parent))
    model = tmp_path / "m.bngl"
    model.write_text(MODEL.replace("begin molecule types", _params(1) + "begin molecule types"))
    assert _bngl_loader._bng2_library_root(Path(_BNG.bng2_pl)) == Path(os.environ["BNGPATH"])
    assert _bngl_loader.bngl_to_net(model).parent == cache
