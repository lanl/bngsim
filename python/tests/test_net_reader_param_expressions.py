"""Tests for .net parameter-expression evaluation in bngsim._net_reader (issue #554).

The reader used to evaluate each parameter expression with Python's ``eval``.
BNGL spells exponentiation ``^``, which Python reads as bitwise XOR, so ``10^2``
came back as 8 and ``1e-4^3`` raised ``TypeError`` into a bare ``except`` that
substituted 0.0 — and that number then seeded any species initial condition
written as a parameter name. ``parse_net_file`` now evaluates through the
engine's own expression evaluator, so it reports what ``Model.from_net`` puts in
the same slot.
"""

from __future__ import annotations

import math
import sys
import textwrap
import warnings
from pathlib import Path

import bngsim
import pytest
from bngsim._net_reader import build_model_from_parsed, parse_net_file


def _write_net(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "model.net"
    p.write_text(textwrap.dedent(body).strip() + "\n")
    return p


def _params_net(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    body = "\n".join(f"  {i + 1} {name} {expr}" for i, (name, expr) in enumerate(rows))
    return _write_net(tmp_path, "begin parameters\n" + body + "\nend parameters\n")


def _values(tmp_path: Path, rows: list[tuple[str, str]]) -> dict[str, float]:
    parsed = parse_net_file(_params_net(tmp_path, rows))
    return {name: value for name, value, _expr, _is_expr in parsed["parameters"]}


class TestParameterExpressionEvaluation:
    def test_caret_is_exponentiation_not_xor(self, tmp_path: Path) -> None:
        """BNGL '^' raises to a power; Python's eval read it as bitwise XOR (10^2 -> 8)."""
        assert _values(tmp_path, [("A0", "10^2")]) == {"A0": 100.0}

    def test_caret_on_float_operands(self, tmp_path: Path) -> None:
        """Float '^' was a TypeError the bare except turned into 0.0."""
        vals = _values(tmp_path, [("rad_cell", "1e-4"), ("vol", "rad_cell^3")])
        assert vals["vol"] == pytest.approx(1e-12, rel=1e-12)

    def test_bngl_conditional_and_logical_and(self, tmp_path: Path) -> None:
        """`if(c,t,f)` and `&&` are BNGL, not Python — both used to land on 0.0."""
        vals = _values(
            tmp_path,
            [("sel", "1"), ("R0", "100"), ("Rtot", "if((sel>=1)&&(sel<10), R0, 0.5*R0)")],
        )
        assert vals["Rtot"] == 100.0

    def test_parameter_named_lambda(self, tmp_path: Path) -> None:
        """A parameter named `lambda` is a Python SyntaxError but a fine BNGL name."""
        vals = _values(tmp_path, [("lambda", "3"), ("d", "4"), ("x_0", "lambda/d")])
        assert vals["x_0"] == 0.75

    def test_literal_is_not_flagged_as_expression(self, tmp_path: Path) -> None:
        """A plain number stays a constant; anything left over is an expression."""
        parsed = parse_net_file(_params_net(tmp_path, [("k", "0.1"), ("A0", "10^2")]))
        assert parsed["parameters"] == [
            ("k", 0.1, "0.1", False),
            ("A0", 100.0, "10^2", True),
        ]

    def test_forward_reference_resolves(self, tmp_path: Path) -> None:
        """Every parameter is registered before any expression is compiled."""
        assert _values(tmp_path, [("a", "b*2"), ("b", "3.0")])["a"] == 6.0

    def test_empty_parameters_block(self, tmp_path: Path) -> None:
        parsed = parse_net_file(_write_net(tmp_path, "begin parameters\nend parameters\n"))
        assert parsed["parameters"] == []


class TestSpeciesInitialConditions:
    def test_ic_named_by_expression_parameter(self, tmp_path: Path) -> None:
        """The headline symptom: the model reported A0 == 100 while integrating from 8."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 A0 10^2
              2 k  0.1
            end parameters
            begin species
              1 A() A0
              2 B() 0
            end species
            begin reactions
              1 1 2 k
            end reactions
            begin groups
              1 Atot 1
              2 Btot 2
            end groups
            """,
        )
        parsed = parse_net_file(net)
        assert parsed["species"] == [("A()", 100.0, False), ("B()", 0.0, False)]
        assert parsed["species_ic_params"] == [(0, "A0")]

        model = build_model_from_parsed(parsed)
        assert model.get_param("A0") == 100.0
        assert list(model.get_state()) == [100.0, 0.0]

    def test_ic_param_ref_is_recorded_on_the_built_model(self, tmp_path: Path) -> None:
        """The reference — not just the number — reaches the model, seeding sensitivities."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 A0 10^2
              2 k  0.1
            end parameters
            begin species
              1 A() A0
              2 B() 0
            end species
            begin reactions
              1 1 2 k
            end reactions
            begin groups
              1 Atot 1
            end groups
            """,
        )
        core = build_model_from_parsed(parse_net_file(net))._core
        assert core.species_ic_param_refs == [(0, 0)]

    def test_unknown_ic_parameter_is_refused(self, tmp_path: Path) -> None:
        """An IC token naming no declared parameter fails loudly (issue #571).

        Defaulting it to 0.0 seeded a silently wrong trajectory that still ran to
        completion. ``net_file_loader.cpp`` now raises, and this reader — kept in
        step with it (issue #554) — must raise the same way rather than agree on
        the wrong number.
        """
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 k 0.1
            end parameters
            begin species
              1 A() nosuch
              2 B() 0
            end species
            begin reactions
              1 1 2 k
            end reactions
            begin groups
              1 Atot 1
            end groups
            """,
        )
        with pytest.raises(ValueError, match="neither a number nor a declared parameter"):
            parse_net_file(net)

    def test_arithmetic_ic_is_refused(self, tmp_path: Path) -> None:
        """An arithmetic IC (``2*A0``) is the same unresolvable-token path as a typo.

        ``float()`` rejects it and no parameter is named ``2*A0``, so it is refused
        rather than silently truncated to its numeric prefix or defaulted to 0.0
        (issue #571). A BNGL expression seed species does not reach a ``.net`` as
        raw arithmetic: BNG2.pl lifts it into a synthetic ``_InitialConc<N>``
        parameter and writes that name here, which resolves. Raw ``2*A0`` is a
        hand-authored token no producer emits, so rejecting it costs no supported
        input.
        """
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 k  0.1
              2 A0 100
            end parameters
            begin species
              1 A() 2*A0
              2 B() 0
            end species
            begin reactions
              1 1 2 k
            end reactions
            begin groups
              1 Atot 1
            end groups
            """,
        )
        with pytest.raises(ValueError, match="neither a number nor a declared parameter"):
            parse_net_file(net)


class TestAgreementWithFromNet:
    """The two documented .net loaders must put the same numbers in the same slots."""

    @pytest.mark.parametrize(
        "rel",
        [
            "benchmarks/models/net/ode/energy_example1.net",
            "benchmarks/models/net/curated/prion_aggregation.net",
            "benchmarks/models/net/ode/Repressilator.net",
            "benchmarks/models/net/ode/wofsy_goldstein.net",
            "tests/data/ic_derived_compound.net",
        ],
    )
    def test_parameters_and_initial_state_match(self, rel: str) -> None:
        path = Path(__file__).resolve().parents[2] / rel
        if not path.exists():
            pytest.skip(f"{rel} not present in this checkout")
        reference = bngsim.Model.from_net(str(path))
        parsed = parse_net_file(path)
        for name, value, _expr, _is_expr in parsed["parameters"]:
            expected = reference.get_param(name)
            assert value == expected or (math.isnan(value) and math.isnan(expected)), (
                f"{rel}: parameter {name}"
            )
        assert list(build_model_from_parsed(parsed).get_state()) == list(reference.get_state())


class TestUnevaluableExpressionWarning:
    def test_unknown_symbol_warns(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match=r"could not compile.*bad = 'nosuch\*2'"):
            vals = _values(tmp_path, [("k", "0.1"), ("bad", "nosuch*2")])
        assert vals["bad"] == 0.0

    def test_syntax_error_warns(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match=r"could not compile"):
            _values(tmp_path, [("k", "0.1"), ("bad", "foo+")])

    def test_expression_that_is_legitimately_zero_is_quiet(self, tmp_path: Path) -> None:
        """0.0 is only suspicious when the evaluator never produced it."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _values(tmp_path, [("z", "1-1"), ("k", "0.1")])["z"] == 0.0

    def test_only_the_root_cause_is_named(self, tmp_path: Path) -> None:
        """A parameter reading a zeroed one compiled fine; naming it too is noise."""
        with pytest.warns(UserWarning) as record:
            _values(tmp_path, [("bad", "nosuch*2"), ("dep", "bad")])
        message = str(record[0].message)
        assert "bad = 'nosuch*2'" in message
        assert "dep = " not in message

    def test_corpus_parses_without_warnings(self) -> None:
        """No .net file shipped in this tree trips the warning."""
        root = Path(__file__).resolve().parents[2]
        files = sorted(
            set((root / "tests/data").rglob("*.net"))
            | set((root / "benchmarks/models/net").rglob("*.net"))
        )
        if not files:
            pytest.skip(".net model corpus not present in this checkout")
        offenders = []
        for path in files:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                parse_net_file(path)
            offenders += [
                (path.name, str(w.message))
                for w in caught
                if "could not compile" in str(w.message)
            ]
        assert offenders == []


class TestWithoutTheCompiledExtension:
    """`parse_net_file` is documented as usable with no C++ extension present."""

    @pytest.fixture
    def no_core(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # None in sys.modules makes `from bngsim._bngsim_core import ...` raise
        # ImportError, which is what an install without the extension looks like.
        monkeypatch.setitem(sys.modules, "bngsim._bngsim_core", None)

    def test_plain_arithmetic_still_evaluates(self, no_core: None, tmp_path: Path) -> None:
        vals = _values(tmp_path, [("NA", "6.022e23"), ("conc", "1e-9"), ("n", "conc*NA")])
        assert vals["n"] == pytest.approx(6.022e14, rel=1e-12)

    def test_caret_is_exponentiation_on_the_fallback_too(
        self, no_core: None, tmp_path: Path
    ) -> None:
        assert _values(tmp_path, [("A0", "10^2")])["A0"] == 100.0

    def test_bngl_syntax_is_refused_not_guessed(self, no_core: None, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"cannot evaluate .net parameter Rtot"):
            _values(tmp_path, [("R0", "100"), ("Rtot", "if(R0>1, R0, 0.5*R0)")])

    def test_python_keyword_name_is_refused_not_guessed(
        self, no_core: None, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match=r"cannot evaluate .net parameter x_0"):
            _values(tmp_path, [("lambda", "3"), ("d", "4"), ("x_0", "lambda/d")])
