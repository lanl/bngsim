"""Named saved concentration states (issue #11).

BNG2.pl supports *named* saved states — ``saveConcentrations("t=0")`` …
``saveConcentrations("start_competition")`` … ``resetConcentrations("name")`` —
so a multi-phase protocol (pre-equilibrate → intervene → restore an *earlier*
state) round-trips faithfully. bngsim's single unnamed slot silently collapsed
all names, so a block that saved two states and restored the first restored the
wrong one.

These tests cover the ``Model.save_concentrations(label=...)`` /
``Model.restore_concentrations(label=...)`` multi-slot store and the
``Simulator`` delegators, and that the default (unlabeled) slot still routes
through the historical ``reset()`` path untouched.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest


class TestModelNamedStates:
    def test_named_states_coexist_and_restore_independently(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        m.set_concentration("A()", 50.0)
        m.save_concentrations("halfA")
        m.set_concentration("A()", 10.0)
        m.save_concentrations("tenA")
        # Scribble the live state; neither named snapshot should move.
        m.set_concentration("A()", 999.0)

        assert m.saved_concentration_labels == ["halfA", "tenA"]
        m.restore_concentrations("halfA")
        assert m.get_concentration("A()") == pytest.approx(50.0)
        m.restore_concentrations("tenA")
        assert m.get_concentration("A()") == pytest.approx(10.0)
        # Restoring the first again proves it was not overwritten by the second.
        m.restore_concentrations("halfA")
        assert m.get_concentration("A()") == pytest.approx(50.0)

    def test_default_slot_independent_of_named_slots(self, simple_decay_net: Path):
        """Unlabeled save/reset is unchanged by named snapshots (back-compat)."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        seed_A = m.get_concentration("A()")  # 100 from the .net
        m.set_concentration("A()", 42.0)
        m.save_concentrations("checkpoint")  # named save does NOT touch default
        m.set_concentration("A()", 7.0)

        # reset() / unlabeled restore still returns to the .net seed, not 42.
        m.reset()
        assert m.get_concentration("A()") == pytest.approx(seed_A)
        m.restore_concentrations()  # same as reset()
        assert m.get_concentration("A()") == pytest.approx(seed_A)
        # The named checkpoint is still there and distinct.
        m.restore_concentrations("checkpoint")
        assert m.get_concentration("A()") == pytest.approx(42.0)

    def test_unlabeled_save_still_rebases_reset(self, simple_decay_net: Path):
        """The historical single-slot behavior: save_concentrations() with no
        label makes reset() return to the saved state."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        m.set_concentration("A()", 33.0)
        m.save_concentrations()  # default slot
        m.set_concentration("A()", 1.0)
        m.reset()
        assert m.get_concentration("A()") == pytest.approx(33.0)

    def test_has_saved_concentrations(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        assert m.has_saved_concentrations() is False
        assert m.has_saved_concentrations("x") is False
        m.save_concentrations("x")
        assert m.has_saved_concentrations() is True
        assert m.has_saved_concentrations("x") is True
        assert m.has_saved_concentrations("y") is False

    def test_restore_unknown_label_raises(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        m.save_concentrations("known")
        with pytest.raises(bngsim.ModelError, match="No saved concentration state named"):
            m.restore_concentrations("missing")

    def test_named_snapshot_is_a_copy(self, simple_decay_net: Path):
        """A stored snapshot is decoupled from later live-state edits and from
        the array returned by get_state()."""
        m = bngsim.Model.from_net(str(simple_decay_net))
        m.set_concentration("A()", 20.0)
        m.save_concentrations("snap")
        # Mutating the live state after the save must not alter the snapshot.
        m.set_state(np.zeros(m.n_species))
        m.restore_concentrations("snap")
        assert m.get_concentration("A()") == pytest.approx(20.0)

    def test_clone_carries_named_states_decoupled(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        m.set_concentration("A()", 60.0)
        m.save_concentrations("s")
        c = m.clone()
        assert c.saved_concentration_labels == ["s"]
        c.restore_concentrations("s")
        assert c.get_concentration("A()") == pytest.approx(60.0)

        # Independent stores: a new save on the clone does not leak to the parent.
        c.set_concentration("A()", 5.0)
        c.save_concentrations("clone_only")
        assert "clone_only" not in m.saved_concentration_labels
        assert "clone_only" in c.saved_concentration_labels


class TestSimulatorSaveRestoreDelegators:
    def test_simulator_save_restore_named(self, simple_decay_net: Path):
        m = bngsim.Model.from_net(str(simple_decay_net))
        sim = bngsim.Simulator(m, method="ode")
        # Advance the model off its seed, then snapshot the post-run state.
        sim.run_until(10.0)
        advanced = m.get_concentration("A()")
        assert advanced < 100.0  # decayed
        sim.save_concentrations("mid")

        # Move further, then restore the named snapshot through the Simulator.
        sim.run_until(20.0)
        assert m.get_concentration("A()") < advanced
        sim.restore_concentrations("mid")
        assert m.get_concentration("A()") == pytest.approx(advanced)

        # A run after the restore continues from the restored state (the backend
        # was rebuilt to seed from it).
        r = sim.run(t_span=(0, 10), n_points=11)
        a0 = r.species[0, r.species_names.index("A()")]
        assert a0 == pytest.approx(advanced)
