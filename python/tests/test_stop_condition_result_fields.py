"""GH #559 — a stop-condition Result must keep everything the run produced.

``Simulator._truncate_result`` rebuilt the partial Result from eight fields and
dropped the rest, so stopping early cost the caller the sensitivity blocks, the
seed, the SSA reaction statistics and the SSA/PSA diagnostics. Two of those
losses were worse than a missing value:

* the truncated result is built with ``core=None``, and ``Result.__init__``
  fills the SSA diagnostics with inert defaults in that branch, so a run on the
  ``cc`` propensity backend came back claiming ``"interpreted"`` — a wrong
  answer where there should have been none;
* a sensitivity run that stopped early raised "no parameter sensitivities were
  computed for this result. Enable them via Simulator(..., sensitivity_params=
  [...])" from the empty block, telling the caller to enable the thing they had
  enabled.

The last test here is the one that keeps this fixed: it walks every field a
Result holds rather than the handful this bug happened to cover, so a field
added later cannot go missing from the truncated copy unnoticed.
"""

from __future__ import annotations

import contextlib
import io

import bngsim
import numpy as np
import pytest
from bngsim._result import Result

NET = """begin parameters
    1 k       0.3  # Constant
end parameters
begin species
    1 A() 100
end species
begin reactions
    1 1 0 k #_R1
end reactions
begin groups
    1 A_tot                 1
end groups
"""

STOP = "A_tot < 50"


@pytest.fixture
def net(tmp_path):
    p = tmp_path / "decay.net"
    p.write_text(NET)
    return str(p)


def _model(net):
    # Every run starts from a fresh model: a Simulator leaves the model
    # advanced, and a second run from that state would trip the stop condition
    # at t=0 and truncate to a single point.
    with contextlib.redirect_stderr(io.StringIO()):
        return bngsim.Model.from_net(net)


def _stopped(net, **kw):
    """Run until the stop condition fires; return (partial, full) results."""
    sim = bngsim.Simulator(_model(net), **kw)
    sim.add_stop_condition(STOP, label="half")
    run = {"t_span": (0.0, 10.0), "n_points": 11}
    if kw.get("method") == "ssa":
        run["seed"] = 7
    with contextlib.redirect_stderr(io.StringIO()):
        full = bngsim.Simulator(_model(net), **kw).run(**run)
        with pytest.raises(bngsim.StopConditionMet) as exc:
            sim.run(**run)
    partial = exc.value.result
    # The cut is a real one: some of the run survives, and some is gone.
    assert 0 < partial.time.shape[0] < full.time.shape[0]
    return partial, full


# ── The three fields the issue names ─────────────────────────────────────────


def test_the_seed_survives(net):
    partial, _ = _stopped(net, method="ssa")
    assert partial.seed == 7


def test_the_propensity_backend_is_not_fabricated(net):
    """The inert default says "unknown"; the run's real backend must win."""
    partial, full = _stopped(net, method="ssa")
    assert (
        partial.ssa_diagnostics["propensity_backend"]
        == (full.ssa_diagnostics["propensity_backend"])
    )
    assert partial.ssa_diagnostics == full.ssa_diagnostics


def test_the_sensitivity_block_survives_and_is_sliced(net):
    partial, full = _stopped(net, method="ode", sensitivity_params=["k"])
    n = partial.time.shape[0]
    assert partial.sensitivity_params == ["k"]
    assert partial.sensitivities.shape == (n, *full.sensitivities.shape[1:])
    assert np.array_equal(partial.sensitivities, full.sensitivities[:n])


def test_output_sensitivities_no_longer_refuses_what_was_enabled(net):
    """The refusal named the one thing the caller had asked for."""
    partial, full = _stopped(net, method="ode", sensitivity_params=["k"])
    n = partial.time.shape[0]
    assert np.array_equal(
        partial.output_sensitivities("A_tot"), full.output_sensitivities("A_tot")[:n]
    )


# ── The other per-time blocks ────────────────────────────────────────────────


def test_reaction_statistics_survive_and_are_sliced(net):
    partial, full = _stopped(net, method="ssa", reaction_stats=True)
    n = partial.time.shape[0]
    assert partial.reaction_labels == full.reaction_labels
    assert np.array_equal(partial.reaction_firing_counts, full.reaction_firing_counts[:n])
    assert np.array_equal(
        partial.reaction_propensity_integrals, full.reaction_propensity_integrals[:n]
    )


@pytest.mark.parametrize("block", ["time", "species", "observables"])
def test_the_value_blocks_are_the_full_run_s_prefix(net, block):
    partial, full = _stopped(net, method="ode")
    n = partial.time.shape[0]
    assert np.array_equal(
        np.asarray(getattr(partial, block)), np.asarray(getattr(full, block))[:n]
    )


# ── The guard: no field may be dropped ───────────────────────────────────────

# `_core` is deliberately absent (a truncated result owns no C++ handle, which
# is why the fields above have to be forwarded by hand in the first place).
_NOT_FORWARDED = {"_core"}


def _same(got, want):
    """Equality that tolerates the NaN inside the PSA diagnostics dict."""
    if isinstance(want, np.ndarray):
        return isinstance(got, np.ndarray) and np.array_equal(got, want, equal_nan=True)
    if isinstance(want, float) and np.isnan(want):
        return isinstance(got, float) and np.isnan(got)
    if isinstance(want, dict):
        return (
            isinstance(got, dict)
            and set(got) == set(want)
            and all(_same(got[k], want[k]) for k in want)
        )
    if isinstance(want, (list, tuple)):
        return (
            type(got) is type(want)
            and len(got) == len(want)
            and all(_same(g, w) for g, w in zip(got, want, strict=False))
        )
    return got == want


@pytest.mark.parametrize("kw", [{"method": "ode", "sensitivity_params": ["k"]}, {"method": "ssa"}])
def test_every_field_is_carried_over(net, kw):
    """Walk Result's own slots, so a field added later cannot be forgotten."""
    partial, full = _stopped(net, **kw)
    n = partial.time.shape[0]
    missing = []
    for name in Result.__slots__:
        if name in _NOT_FORWARDED:
            continue
        want = getattr(full, name)
        got = getattr(partial, name)
        if isinstance(want, np.ndarray) and want.size and want.shape[0] == full.time.shape[0]:
            want = want[:n]
        if not _same(got, want):
            missing.append(name)
    assert not missing, f"truncated result lost or changed: {missing}"
