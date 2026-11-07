"""Issue #1486 — the codegen cache key names the architecture.

The cache names each compiled RHS ``rhs_<key>_<hash>``, and the key was
``_CODEGEN_VERSION`` and a digest of the emitters, nothing about the machine. Two
builds of one bngsim on two architectures that share a cache directory (a native
arm64 Python and an x86_64 one under Rosetta on one Mac, or the x86-64 and
aarch64 nodes of a cluster on one home directory) therefore named the same
model's library alike, and whichever ran second loaded the other's: "dlopen
failed ... incompatible architecture (have 'arm64', need 'x86_64')".

The key's last field is now the architecture of the running interpreter, so the
two name their libraries apart. ``bngsim.cache`` reads an artifact that differs
from this install's key in that field alone as the other architecture's build of
the same source, not as an orphan (test_cache_cli.py), so an orphan sweep from
one does not take the other's cache.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import bngsim._codegen as cg
import numpy as np
import pytest

NET = """begin parameters
    1 k 0.5
end parameters
begin species
    1 A() 1.0
    2 B() 0
end species
begin reactions
    1 1 2 k
end reactions
begin groups
    1 A_tot 1
end groups
"""


@pytest.mark.parametrize(
    "machine, tag",
    [
        ("arm64", "arm64"),
        ("x86_64", "x86-64"),
        ("aarch64", "aarch64"),
        ("AMD64", "amd64"),
        ("ARM64", "arm64"),
        ("i686", "i686"),
        ("ppc64le", "ppc64le"),
        ("", "unknown"),
    ],
)
def test_the_tag_is_a_key_field_safe_spelling_of_the_machine(machine, tag):
    assert cg._codegen_arch_tag(machine) == tag
    assert cg._artifact_key_field(f"1+0+{tag}") == f"1+0+{tag}", "no escaping needed"


def test_the_key_ends_with_this_processs_architecture():
    import platform

    assert cg._codegen_arch_tag(platform.machine()) == cg._CODEGEN_ARCH
    assert cg._codegen_cache_key_for(cg._CODEGEN_ARCH) == cg._CODEGEN_CACHE_KEY
    assert cg._CODEGEN_CACHE_KEY.rsplit("+", 1)[1] == cg._CODEGEN_ARCH
    assert cg._artifact_stem("0123456789abcdef").startswith(f"rhs_{cg._CODEGEN_CACHE_KEY}_")


def _other_arch() -> str:
    return "x86-64" if cg._CODEGEN_ARCH != "x86-64" else "arm64"


def test_two_architectures_compute_two_keys():
    """One source, two machines: the same version and digest, two keys."""
    mine, theirs = cg._CODEGEN_CACHE_KEY, cg._codegen_cache_key_for(_other_arch())
    assert mine != theirs
    assert mine.rsplit("+", 1)[0] == theirs.rsplit("+", 1)[0]


def _run(path) -> float:
    with contextlib.redirect_stderr(io.StringIO()):
        sim = bngsim.Simulator(bngsim.Model.from_net(str(path)), method="ode", codegen=True)
        result = sim.run(t_span=(0.0, 1.0), n_points=3)
    return float(np.asarray(result.species)[-1, 0])


def _libraries(cache) -> set:
    suffix = cg._shared_lib_suffix()
    return {p for p in cache.iterdir() if p.name.endswith(suffix)}


def test_another_architectures_library_is_never_loaded(tmp_path, monkeypatch):
    """The issue's reproducer, on one machine: a run on the other architecture
    leaves its library in a shared cache, and a run here must build its own
    rather than dlopen that one. The other architecture's library is planted by
    running under its key and then overwriting what was built with bytes no
    loader takes, standing in for a binary of another architecture."""
    cache = tmp_path / "codegen"
    monkeypatch.setattr(cg, "CACHE_DIR", cache)
    net = tmp_path / "m.net"
    net.write_text(NET)

    with monkeypatch.context() as m:
        m.setattr(cg, "_CODEGEN_CACHE_KEY", cg._codegen_cache_key_for(_other_arch()))
        _run(net)
    (planted,) = _libraries(cache)
    assert f"+{_other_arch()}_" in planted.name
    planted.write_bytes(b"built for another architecture; loading this must never happen\n")

    a1 = _run(net)

    assert a1 == pytest.approx(np.exp(-0.5), rel=1e-6)
    (built,) = _libraries(cache) - {planted}
    assert f"+{cg._CODEGEN_ARCH}_" in built.name
    assert planted.read_bytes().startswith(b"built for another architecture")
