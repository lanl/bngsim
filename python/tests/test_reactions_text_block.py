"""Regression tests for issue #13: `.net` files with a `reactions_text` block.

BNG2.pl emits a ``begin reactions_text ... end reactions_text`` block for some
models (when the model sets the corresponding print option). The block is
purely informational — it restates the numeric ``reactions`` block in
human-readable pattern form (``1 A(b) -> B(a) k1``).

The loader's block dispatch matched ``begin reactions`` as a *substring* of
``begin reactions_text``, so the pattern lines were fed to the numeric reaction
parser, where ``std::stoi("A(b)")`` threw ``stoi: no conversion``. The fix
recognizes and skips the ``reactions_text`` block, as the loader already does
for other optional blocks (``molecule types``, ``observables``, ...).

Two published/tutorial models in the parity corpus failed to load solely on
this collision (ComplexDegradation N=6, BaruaBCR_2012 N=1122); this test pins
the fix.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest
from bngsim import Model, Simulator

# Same network as the reactions_text_block.net fixture, but with no
# reactions_text block at all. Loading both must produce identical models.
_EQUIVALENT_NO_BLOCK = (
    textwrap.dedent(
        """
    begin parameters
       1 k1 1.0
    end parameters
    begin species
       1 A() 100
       2 B() 0
    end species
    begin reactions
       1 1 2 k1
    end reactions
    begin groups
       1 GA 1
       2 GB 2
    end groups
    """
    ).strip()
    + "\n"
)


class TestReactionsTextBlock:
    def test_loads_without_crash(self, reactions_text_block_net: Path) -> None:
        """The .net loads — this is the primary regression for issue #13.

        Before the fix this raised ModelError ("... stoi: no conversion").
        """
        model = Model.from_net(reactions_text_block_net)
        # The informational block is skipped: only the numeric reactions block
        # defines the network (A -> B), so there is exactly one reaction.
        assert model.n_species == 2
        assert model.n_reactions == 1

    def test_reactions_text_is_ignored_not_parsed(
        self, reactions_text_block_net: Path, tmp_path: Path
    ) -> None:
        """The network is identical with and without the reactions_text block.

        The block is redundant with the numeric ``reactions`` block, so an
        otherwise-identical .net that omits it must produce an equivalent model
        and an identical ODE trajectory.
        """
        without = tmp_path / "no_reactions_text.net"
        without.write_text(_EQUIVALENT_NO_BLOCK)

        m_with = Model.from_net(reactions_text_block_net)
        m_without = Model.from_net(without)
        assert m_with.n_species == m_without.n_species
        assert m_with.n_reactions == m_without.n_reactions

        r_with = Simulator(m_with, method="ode").run(t_span=(0.0, 5.0), n_points=11)
        r_without = Simulator(m_without, method="ode").run(t_span=(0.0, 5.0), n_points=11)
        assert np.allclose(r_with.species, r_without.species)

    def test_integrates_to_analytic_decay(self, reactions_text_block_net: Path) -> None:
        """A -> B with k1=1 integrates to A(t) = A0*exp(-k1*t)."""
        model = Model.from_net(reactions_text_block_net)
        result = Simulator(model, method="ode").run(t_span=(0.0, 5.0), n_points=11)
        # A is species index 0, B(0)=100.
        A_end = result.species[-1, 0]
        expected = 100.0 * math.exp(-1.0 * 5.0)
        assert A_end == pytest.approx(expected, rel=1e-5)
