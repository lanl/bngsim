"""A seed outside 0 to 2**64 - 1 is refused by name (issue #1515).

The bindings take a seed as ``uint64_t``. A negative seed, or one past
2**64 - 1, reached them and raised pybind11's ``TypeError: run():
incompatible function arguments`` on every method and session, though the
docs promised any integer. It now raises a ``ValueError`` that names the
range, before anything runs.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import pytest

RUN = {"t_span": (0.0, 1.0), "n_points": 2}
RANGE = r"seed must be an integer from 0 to 2\*\*64 - 1, got "


@pytest.fixture
def model(simple_decay_net: Path) -> bngsim.Model:
    return bngsim.Model.from_net(str(simple_decay_net))


@pytest.fixture
def dummy_model(simple_decay_net: Path) -> bngsim.Model:
    """The Simulator's model on the XML-backed network-free path."""
    return bngsim.Model.from_net(str(simple_decay_net))


def _simulator(model, method):
    kw = {"poplevel": 10} if method == "psa" else {}
    return bngsim.Simulator(model, method=method, **kw)


@pytest.mark.parametrize("method", ["ode", "ssa", "psa"])
@pytest.mark.parametrize("seed", [-1, 2**64])
def test_run_refuses_a_seed_out_of_range(model, method, seed):
    """``ode`` reads the seed too: it orders simultaneous equal-priority
    events (GH #242)."""
    with pytest.raises(ValueError, match=RANGE + str(seed)):
        _simulator(model, method).run(seed=seed, **RUN)


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_replicates_refuse_their_base_seed_out_of_range(model, seed):
    with pytest.raises(ValueError, match=RANGE + str(seed)):
        _simulator(model, "ssa").run_replicates(2, seed=seed, **RUN)


@pytest.mark.parametrize("method", ["ssa", "psa"])
@pytest.mark.parametrize("seed", [0, 2**32, 2**63, 2**64 - 1])
def test_a_seed_in_range_runs_and_is_reported(model, method, seed):
    """Control: SSA and PSA seed ``std::mt19937_64`` with all 64 bits."""
    result = _simulator(model, method).run(seed=seed, **RUN)
    assert result.seed == seed


def _has(flag: str) -> bool:
    try:
        from bngsim import _bngsim_core

        return bool(getattr(_bngsim_core, flag))
    except (ImportError, AttributeError):
        return False


@pytest.mark.skipif(not _has("HAS_NFSIM"), reason="NFsim not compiled in")
@pytest.mark.parametrize("seed", [-1, 2**64])
def test_an_nfsim_session_refuses_a_seed_out_of_range(nfsim_xml, seed):
    with (
        bngsim.NfsimSession(str(nfsim_xml)) as session,
        pytest.raises(ValueError, match=RANGE + str(seed)),
    ):
        session.initialize(seed=seed)


@pytest.mark.skipif(not _has("HAS_RULEMONKEY"), reason="RuleMonkey not compiled in")
@pytest.mark.parametrize("seed", [-1, 2**64])
def test_a_rulemonkey_session_refuses_a_seed_out_of_range(nfsim_xml, seed):
    with (
        bngsim.RuleMonkeySession(str(nfsim_xml)) as session,
        pytest.raises(ValueError, match=RANGE + str(seed)),
    ):
        session.initialize(seed=seed)


@pytest.mark.skipif(not _has("HAS_RULEMONKEY"), reason="RuleMonkey not compiled in")
def test_rulemonkey_keeps_all_64_bits(dummy_model, nfsim_xml):
    """Control: RuleMonkey's generator reads all 64 bits, so 2**32 + 12345 is
    a seed of its own there."""
    sim = bngsim.Simulator(dummy_model, method="rm", xml_path=str(nfsim_xml))
    result = sim.run(seed=2**32 + 12345, t_span=(0.0, 1.0), n_points=2)
    assert result.seed == 2**32 + 12345
