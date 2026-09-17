"""Regression tests for issue #571: an unresolvable species initial-concentration
token must fail loudly, not seed a silent 0.0.

``parse_species()`` reads the IC column as a number or a parameter name. When the
token is neither — a typo, a parameter declared after the species block, or an
arithmetic IC such as ``2*A0`` (``std::stod`` stops at ``2`` and the whole token
is then looked up as a parameter name and misses) — the loader used to default
the IC to 0.0. The species started at zero, every downstream observable and flux
was wrong, and the run still reported success. The loader now raises, the way an
unknown rate-law parameter (``test_model_validation.py::test_unknown_elementary_param``)
and an out-of-range observable index already do.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from bngsim import Model
from bngsim._bngsim_core import NetworkModel
from bngsim._exceptions import ModelError


def _write_net(tmp_path: Path, ic: str) -> Path:
    """A two-species network whose second species carries initial condition ``ic``."""
    p = tmp_path / "model.net"
    p.write_text(
        textwrap.dedent(
            f"""
            begin parameters
               1 k1 0.5
               2 A0 100
            end parameters
            begin species
               1 A() A0
               2 B() {ic}
            end species
            begin reactions
               1 1 2 k1
            end reactions
            begin groups
               1 Atot 1
               2 Btot 2
            end groups
            """
        ).strip()
        + "\n"
    )
    return p


class TestUnresolvedSpeciesIC:
    def test_typo_parameter_name_is_refused(self, tmp_path: Path) -> None:
        """A misspelled or undeclared IC parameter name raises at load time."""
        net = _write_net(tmp_path, "B0_typo")
        with pytest.raises(ModelError, match="neither a number nor a declared parameter"):
            Model.from_net(net)

    def test_arithmetic_ic_is_evaluated(self, tmp_path: Path) -> None:
        """Raw arithmetic in the IC column is evaluated, as ``run_network`` does.

        ``run_network`` reports 200 for ``2*A0`` with ``A0 = 100``. The loader
        lifts the expression into a synthetic ``_InitialConc<N>`` parameter —
        exactly what BNG2.pl's ``generate_network`` writes for a BNGL
        seed-species expression — so it rides the parameters block's existing
        ExprTk path (issue #600).
        """
        net = _write_net(tmp_path, "2*A0")
        model = Model.from_net(net)
        assert list(model._core.get_initial_state()) == [100.0, 200.0]
        assert "_InitialConc1" in list(model.param_names)

    def test_lifted_expression_naming_an_unknown_symbol_still_fails(self, tmp_path: Path) -> None:
        """Evaluating expressions must not reopen the silent-zero hole of #571.

        A typo *inside* an expression cannot be caught by the bare-name check, so
        it reaches the parameter compile — which refuses it since issue #602
        rather than leaving a seed behind.
        """
        net = _write_net(tmp_path, "2*B0_typo")
        with pytest.raises(ModelError, match="failed to compile parameter"):
            Model.from_net(net)

    def test_expression_seed_species_via_initialconc_param_loads(self, tmp_path: Path) -> None:
        """An expression-valued seed species is supported and resolves to its value.

        BNGL allows expression seed abundances (``A() 2*A0``), but BNG2.pl does not
        write that arithmetic into the ``.net`` species column. It lifts each
        expression into a synthetic ``_InitialConc<N>`` ConstantExpression
        parameter and writes that parameter's *name* — a single token — into the
        species block, so ``parse_species`` sees a declared reference and resolves
        it through the expression evaluator. The fix rejects only a token that
        resolves to no parameter, never this shape. (Values verified against
        BNG2.pl 2.9.3 output for the same model.)
        """
        p = tmp_path / "model.net"
        p.write_text(
            textwrap.dedent(
                """
                begin parameters
                   1 A0            100
                   2 f             2
                   3 _InitialConc1 2*A0
                   4 _InitialConc2 A0/f
                   5 k1            0.5
                end parameters
                begin species
                   1 A() _InitialConc1
                   2 B() _InitialConc2
                end species
                begin reactions
                   1 1 2 k1
                end reactions
                begin groups
                   1 Atot 1
                   2 Btot 2
                end groups
                """
            ).strip()
            + "\n"
        )
        model = Model.from_net(p)
        assert list(model._core.get_initial_state()) == [200.0, 50.0]

    def test_core_loader_raises_valueerror(self, tmp_path: Path) -> None:
        """The raw core loader raises ValueError — the same failure shape the issue
        contrasts against for an out-of-range observable index. ``Model.from_net``
        wraps it as ``ModelError``."""
        net = _write_net(tmp_path, "B0_typo")
        with pytest.raises(ValueError, match="Failed to load .net file:.*neither a number"):
            NetworkModel.from_net(str(net))

    def test_declared_parameter_ic_still_loads(self, tmp_path: Path) -> None:
        """A declared parameter name as the IC still resolves — the fix rejects
        only the miss, not the supported single-parameter-reference shape."""
        net = _write_net(tmp_path, "A0")
        model = Model.from_net(net)
        assert list(model._core.get_initial_state()) == [100.0, 100.0]

    def test_numeric_ic_still_loads(self, tmp_path: Path) -> None:
        """A plain numeric IC is unaffected."""
        net = _write_net(tmp_path, "42")
        model = Model.from_net(net)
        assert list(model._core.get_initial_state()) == [100.0, 42.0]
