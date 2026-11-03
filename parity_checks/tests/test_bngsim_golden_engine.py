"""GH #175 regression: the bng_parity golden/sweep must run GENUINE bngsim.

The bug: the sweep drove the bngsim side through ``bionetgen.run(simulator='bngsim')``,
which — with a STOCK BNG2.pl (no backend-hook patch) — routes a BNGL ``simulate``
through ``run_network``/NFsim (BNG2.pl), NOT bngsim. So ``golden.json`` was BNG2.pl
output mislabelled bngsim, and the engine guards (which checked a *predicted route*)
never caught it.

The fix: the sweep drives ``_bng_common.run_bngsim_job`` — BNG2.pl generates the
network/XML only, then bngsim simulates IN-PROCESS via the bridge's direct route.

These tests pin the fix:

  * Unit (cheap, always run): the track classifier maps each method to the right
    engine — ode/ssa/psa run on bngsim, network-free ``nf`` is rerouted to
    RuleMonkey (``rm``), and an unsupported ``pla`` is refused (no silent fallback).
  * Engine (needs bngsim + BNG2.pl + perl): the DECISIVE proof — on a copy of the
    BNG install with ``run_network`` deleted, ``run_bngsim_job`` still produces
    numeric output (it never needs run_network), while the old
    ``bionetgen.run(simulator='bngsim')`` BNGL route crashes (proving it was BNG2.pl
    all along). Each test removes the run_network of the install its route runs,
    from a copy: the installs themselves are never touched.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import _core
import pytest

_BNG_PARITY = Path(__file__).resolve().parent.parent / "bng_parity"
if str(_BNG_PARITY) not in sys.path:
    sys.path.insert(0, str(_BNG_PARITY))


def _load_bng_common():
    spec = importlib.util.spec_from_file_location("_bng_common", _BNG_PARITY / "_bng_common.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_bng_common()
MODELS = _BNG_PARITY / "models"


# --------------------------------------------------------------------------- #
# Unit: the track classifier (no engine needed — pure text -> engine decision).
# --------------------------------------------------------------------------- #
_ODE_BNGL = 'simulate({method=>"ode",t_end=>10,n_steps=>10})'
_SSA_BNGL = 'simulate({method=>"ssa",t_end=>10,n_steps=>10,seed=>1})'
_NF_BNGL = 'simulate({method=>"nf",t_end=>10,n_steps=>10,seed=>1})'
_RM_BNGL = 'simulate({method=>"rm",t_end=>10,n_steps=>10,seed=>1})'
_PSA_BNGL = 'simulate({method=>"ssa",poplevel=>100,t_end=>10,n_steps=>10,seed=>1})'
_PLA_BNGL = 'simulate({method=>"pla",t_end=>10,n_steps=>10})'


@pytest.mark.parametrize(
    "text, stochastic, expected",
    [
        (_ODE_BNGL, False, "ode"),
        (_SSA_BNGL, True, "ssa"),
        (_NF_BNGL, True, "nf"),  # method-faithful: nf -> NFsim
        (_RM_BNGL, True, "rm"),  # method-faithful: rm -> RuleMonkey
        (_PSA_BNGL, True, "psa"),
        (_PLA_BNGL, True, None),  # bngsim cannot run pla -> refused (no fallback)
    ],
)
def test_classify_bngsim_track(text, stochastic, expected):
    assert C.classify_bngsim_track(text, stochastic=stochastic) == expected


def test_network_free_routing_is_method_faithful():
    """nf runs on NFsim and rm runs on RuleMonkey — never one substituted for the
    other (they are swappable engines, but the model gets the one it asked for)."""
    assert C.classify_bngsim_track(_NF_BNGL, stochastic=True) == "nf"
    assert C.classify_bngsim_track(_RM_BNGL, stochastic=True) == "rm"


def test_pla_is_refused_never_silently_substituted():
    """A pla-only model has no bngsim track; the driver must REFUSE, not fall back."""
    assert C.classify_bngsim_track(_PLA_BNGL, stochastic=True) is None


def test_is_stochastic_text_and_seed_parsing():
    assert C._is_stochastic_text(_ODE_BNGL) is False
    assert C._is_stochastic_text(_SSA_BNGL) is True
    assert C._parse_seed(_SSA_BNGL) == 1
    assert C._parse_seed(_ODE_BNGL) is None


# --------------------------------------------------------------------------- #
# Engine: the genuine-bngsim proof. Needs bngsim + a usable BNG2.pl + perl.
# --------------------------------------------------------------------------- #
def _bngsim_available() -> bool:
    try:
        import bionetgen  # noqa: F401

        return bool(getattr(bionetgen, "BNGSIM_AVAILABLE", False))
    except Exception:
        return False


_BNG = _core.resolve_bng()


def _bng_paths():
    """(bngpath, bng2_pl, run_network), or None when any piece is unavailable.

    This suite needs the legacy binary too — the decisive GH #175 lever deletes
    ``run_network`` from a copy of the install — so a resolution without it does
    not count.
    """
    if not _BNG.ok:
        return None
    run_network = _BNG.root / "bin" / "run_network"
    if not run_network.exists():
        return None
    return str(_BNG.root), str(_BNG.bng2_pl), run_network


engine = pytest.mark.skipif(
    not _bngsim_available() or _bng_paths() is None,
    # The resolver's trail names every place BNG2.pl was looked for; append the
    # two extra preconditions this suite has beyond a bare BNG2.pl.
    reason=(
        _BNG.why_not()
        if not _BNG.ok
        else "needs an importable, version-compatible bngsim + run_network under <BNGPATH>/bin/"
    ),
)


@pytest.fixture
def bng_without_run_network(tmp_path, monkeypatch):
    """``make(root) -> copy``: a copy of the BNG install at ``root`` with
    ``bin/run_network`` deleted, and ``$BNGPATH`` pointed at the copy.

    Decisive GH #175 lever: with run_network gone, only an engine that does NOT
    depend on it can still produce output. A copy, because renaming the binary in
    place left the user's install broken whenever the test was killed before it
    could restore it. ``$BNGPATH`` too, because BNG2.pl takes its root from
    ``$BNGPATH`` before its own location: a copy run under the original's
    ``$BNGPATH`` finds the original's run_network.
    """

    def make(root) -> Path:
        root = Path(root)
        if root.name.lower() == "bng2.pl":
            root = root.parent
        if not (root / "bin" / "run_network").is_file():
            pytest.skip(f"{root} has no bin/run_network to remove")
        copy = tmp_path / "bng_without_run_network"
        shutil.copytree(root, copy, symlinks=True)
        (copy / "bin" / "run_network").unlink()
        monkeypatch.setenv("BNGPATH", str(copy))
        return copy

    return make


@engine
def test_run_bngsim_job_is_genuine_bngsim_without_run_network(bng_without_run_network, tmp_path):
    """run_bngsim_job produces numeric output with run_network GONE => genuine bngsim."""
    bng = bng_without_run_network(_BNG.root)
    model = MODELS / "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl"
    assert model.exists()
    out = tmp_path / "out"
    info = C.run_bngsim_job(model, out, str(bng / "BNG2.pl"), timeout=180)
    assert info["engine"] == "bngsim"
    assert info["track"] == "ode"
    gdat = list(out.glob("*.gdat"))
    assert gdat, "bngsim must have written a .gdat trajectory"
    # non-trivial trajectory (more than just the t=0 row)
    time, vals, _names = C.read_dat(gdat[0])
    assert len(time) > 1 and vals.shape[1] >= 1


@engine
def test_legacy_bngl_route_is_bng2pl_not_bngsim(bng_without_run_network, tmp_path, monkeypatch):
    """Contrast: bionetgen.run(<bngl>, simulator='bngsim') CRASHES with run_network
    gone — proving that path is BNG2.pl/run_network, not bngsim (the GH #175 bug).

    The run_network removed is the one that route runs. ``bionetgen.run`` takes
    BNG2.pl from PyBioNetGen's own config (``get_conf()["bngpath"]``, its bundled
    copy) and ignores ``$BNGPATH``. Removing ``$BNGPATH``'s run_network instead left
    the route its own, and the test failed with DID NOT RAISE on any machine whose
    ``$BNGPATH`` names another install, which the resolver has preferred since
    1f216d4d. So the route is pointed at a copy of its own install without one.
    """
    import bionetgen
    import bionetgen.modelapi.runner as runner
    from bionetgen.main import get_conf

    conf = get_conf()
    bng = bng_without_run_network(conf["bngpath"])
    patched = {key: conf[key] for key in conf}
    patched["bngpath"] = str(bng)
    monkeypatch.setattr(runner, "get_conf", lambda: patched)

    model = MODELS / "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl"
    # Broad on purpose: the point is that the route CRASHES with run_network gone;
    # the exact error (missing-binary / subprocess / bionetgen) is not what we pin.
    with pytest.raises(Exception) as raised:  # noqa: B017
        bionetgen.run(str(model), out=str(tmp_path / "out"), simulator="bngsim")
    # ... but it must be run_network's absence that stopped it, not a broken copy.
    err = raised.value
    said = " ".join(
        str(part) for part in (err, getattr(err, "stdout", ""), getattr(err, "stderr", ""))
    )
    assert "run_network" in said, said[-2000:]
