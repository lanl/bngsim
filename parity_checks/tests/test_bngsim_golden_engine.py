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
  * Engine (needs bngsim + BNG2.pl + perl): the DECISIVE proof — with the bundled
    ``run_network`` moved aside, ``run_bngsim_job`` still produces numeric output
    (it never needs run_network), while the old ``bionetgen.run(simulator='bngsim')``
    BNGL route crashes (proving it was BNG2.pl all along). The move-aside is wrapped
    in a fixture that ALWAYS restores run_network, even on a hard assertion failure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

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


def _bng_paths():
    """(bngpath, bng2_pl, run_network) the active bionetgen actually uses, or None."""
    try:
        from bionetgen.main import get_conf

        bngpath = get_conf().get("bngpath")
    except Exception:
        bngpath = os.environ.get("BNGPATH") or os.environ.get("BNG2_PL")
    if not bngpath:
        return None
    p = Path(bngpath)
    bng2 = p / "BNG2.pl" if p.is_dir() else p
    run_network = (p if p.is_dir() else p.parent) / "bin" / "run_network"
    if not (bng2.exists() and run_network.exists()):
        return None
    import shutil

    if shutil.which("perl") is None:
        return None
    return str(bngpath), str(bng2), run_network


engine = pytest.mark.skipif(
    not _bngsim_available() or _bng_paths() is None,
    reason="needs an importable, version-compatible bngsim + a bundled BNG2.pl/run_network + perl",
)


@pytest.fixture
def run_network_moved_aside():
    """Move the bundled run_network aside, yield, then ALWAYS restore it.

    Decisive GH #175 lever: with run_network gone, only an engine that does NOT
    depend on it can still produce output. The finally clause restores the binary
    even if the test body raises, so the user's BNG install is left untouched.
    """
    paths = _bng_paths()
    assert paths is not None
    _, _, run_network = paths
    disabled = run_network.with_suffix(run_network.suffix + ".disabled_by_test")
    run_network.rename(disabled)
    try:
        # Yield the paths captured BEFORE the move — _bng_paths() would now return
        # None (it checks run_network.exists(), which is false while moved aside).
        yield paths
    finally:
        if disabled.exists():
            disabled.rename(run_network)


@engine
def test_run_bngsim_job_is_genuine_bngsim_without_run_network(run_network_moved_aside):
    """run_bngsim_job produces numeric output with run_network GONE => genuine bngsim."""
    _, bng2, _ = run_network_moved_aside
    model = MODELS / "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl"
    assert model.exists()
    out = Path(tempfile.mkdtemp(prefix="genuine_bngsim_")) / "out"
    info = C.run_bngsim_job(model, out, bng2, timeout=180)
    assert info["engine"] == "bngsim"
    assert info["track"] == "ode"
    gdat = list(out.glob("*.gdat"))
    assert gdat, "bngsim must have written a .gdat trajectory"
    # non-trivial trajectory (more than just the t=0 row)
    time, vals, _names = C.read_dat(gdat[0])
    assert len(time) > 1 and vals.shape[1] >= 1


@engine
def test_legacy_bngl_route_is_bng2pl_not_bngsim(run_network_moved_aside):
    """Contrast: bionetgen.run(<bngl>, simulator='bngsim') CRASHES with run_network
    gone — proving that path is BNG2.pl/run_network, not bngsim (the GH #175 bug)."""
    import bionetgen

    model = MODELS / "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl"
    out = tempfile.mkdtemp(prefix="bridge_bngl_route_")
    # Broad on purpose: the point is that the route CRASHES with run_network gone;
    # the exact error (missing-binary / subprocess / bionetgen) is not what we pin.
    with pytest.raises(Exception):  # noqa: B017
        bionetgen.run(str(model), out=out, simulator="bngsim")
