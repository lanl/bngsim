"""NFsim refuses a seed its generators cannot hold (issue #1515).

NFsim seeds both its generators, the System's ``NfsimRNG`` and the
process-wide ``MTRand``, as reference MT19937's ``init_genrand`` does: from
the seed's low 32 bits. ``seed=2**32 + 12345`` therefore ran the
``seed=12345`` trajectory bit for bit, while ``Result.seed`` and
``session.seed`` reported 4294979641, so two runs that read as independent
were one. Such a seed is now refused by name, as numpy's legacy MT19937
refuses it. Seeds up to 2**32 - 1 run as before.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest


def _has_nfsim() -> bool:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        return bool(HAS_NFSIM)
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_nfsim(), reason="NFsim not compiled in")

RUN = {"t_span": (0.0, 10.0), "n_points": 6}


@pytest.fixture
def dummy_model(simple_decay_net: Path) -> bngsim.Model:
    """The Simulator's model on the XML-backed network-free path."""
    return bngsim.Model.from_net(str(simple_decay_net))


def _run(dummy_model, nfsim_xml, seed):
    sim = bngsim.Simulator(dummy_model, method="nf", xml_path=str(nfsim_xml))
    return sim.run(seed=seed, **RUN)


@pytest.mark.parametrize(
    ("seed", "low_word"),
    [
        (2**32, 0),
        (2**32 + 12345, 12345),
        (2**63, 0),
        (2**64 - 1, 2**32 - 1),
    ],
)
def test_a_seed_past_32_bits_is_refused(dummy_model, nfsim_xml, seed, low_word):
    """Each ran as its low word, and reported the value passed."""
    with pytest.raises(ValueError, match=f"seed {seed} would run as seed {low_word}"):
        _run(dummy_model, nfsim_xml, seed)


def test_the_refusal_names_the_range(dummy_model, nfsim_xml):
    with pytest.raises(ValueError, match=r"from 0 to 4294967295 \(2\*\*32 - 1\)"):
        _run(dummy_model, nfsim_xml, 2**32 + 12345)


def test_the_largest_32_bit_seed_runs(dummy_model, nfsim_xml):
    """Control: 2**32 - 1 is a seed NFsim holds, and it is its own run, not
    seed 0's."""
    top = _run(dummy_model, nfsim_xml, 2**32 - 1)
    zero = _run(dummy_model, nfsim_xml, 0)
    assert top.seed == 2**32 - 1
    assert not np.array_equal(np.asarray(top.observables), np.asarray(zero.observables))


def test_a_32_bit_seed_still_reproduces(dummy_model, nfsim_xml):
    """Control: the same seed on two fresh simulators gives one trajectory."""
    a = _run(dummy_model, nfsim_xml, 12345)
    b = _run(dummy_model, nfsim_xml, 12345)
    assert a.seed == b.seed == 12345
    np.testing.assert_array_equal(np.asarray(a.observables), np.asarray(b.observables))


def test_a_session_refuses_it(nfsim_xml):
    """``session.seed`` reported 4294979641 for the seed-12345 run."""
    with bngsim.NfsimSession(str(nfsim_xml)) as session:
        with pytest.raises(ValueError, match="would run as seed 12345"):
            session.initialize(seed=2**32 + 12345)
        assert session.seed is None


def test_a_refused_reinitialize_leaves_the_live_session(nfsim_xml):
    """The seed is checked before the session is torn down. On main a second
    ``initialize(seed=2**32)`` replaced the session with seed 0's and
    reported 4294967296. Refused halfway through, the session goes on from
    where it was, as a seed-7 session that was never refused does; one put
    back at t = 0 would run its first half again. The two run one after the
    other, since every ``initialize`` reseeds NFsim's process-wide generator."""
    with bngsim.NfsimSession(str(nfsim_xml)) as session:
        session.initialize(seed=7)
        session.simulate(0.0, 5.0, n_points=6)
        with pytest.raises(ValueError, match="would run as seed 0"):
            session.initialize(seed=2**32)
        assert session.seed == 7
        kept = session.simulate(5.0, 10.0, n_points=6)
    with bngsim.NfsimSession(str(nfsim_xml)) as fresh:
        fresh.initialize(seed=7)
        fresh.simulate(0.0, 5.0, n_points=6)
        expected = fresh.simulate(5.0, 10.0, n_points=6)
    assert kept.seed == 7
    np.testing.assert_array_equal(np.asarray(kept.observables), np.asarray(expected.observables))
