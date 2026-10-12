"""Independent review of issue #803 step 5: ``parse_net_file`` is the C++ loader's reading.

Every test here checks the dict door (``parse_net_file`` and
``build_model_from_parsed``) against something that is neither door: a closed
form, a hand integral, Python arithmetic, or a mass-action RHS written from the
dict alone. Where ``Model.from_net`` is the reference, the claim under test is
the documented one ("the number ``Model.from_net`` puts in its slot").

Tests that fail without a marker are defects in the change under review. The
``xfail(strict=True)`` ones are defects that predate it (in the C++ loader or in
``ModelBuilder``); each retires itself when its defect is fixed.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import textwrap
import warnings
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim import build_model_from_parsed, parse_net_file
from bngsim._exceptions import ModelError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _net_corpus  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_ODE_NETS = _REPO / "benchmarks" / "models" / "net" / "ode"


def _net(tmp_path: Path, body: str, name: str = "m.net") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def _ode(model: bngsim.Model, t_end: float, n_points: int) -> np.ndarray:
    result = bngsim.Simulator(model, method="ode").run(
        t_span=(0.0, t_end), n_points=n_points, rtol=1e-10, atol=1e-12
    )
    return np.asarray(result.species)


def _both_doors(path: Path) -> tuple[bngsim.Model, bngsim.Model]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return bngsim.Model.from_net(str(path)), build_model_from_parsed(parse_net_file(path))


# ─── A stat-factor prefix, read and built, against closed forms ──────────────

_HOMODIMER = """
begin parameters
    1 kf 0.02
    2 A0 50
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1,1 2 0.5*kf
end reactions
begin groups
    1 Atot 1
end groups
"""


class TestStatFactor:
    def test_homodimer_matches_its_closed_form_from_both_doors(self, tmp_path: Path) -> None:
        """A + A -> B at `0.5*kf`: dA/dt = -2 * 0.5*kf * A^2, so A = A0 / (1 + kf*A0*t)."""
        path = _net(tmp_path, _HOMODIMER)
        parsed = parse_net_file(path)
        (rxn,) = parsed["reactions"]
        assert (rxn["type"], rxn["rate_law"], rxn["stat_factor"]) == ("elementary", "kf", 0.5)
        assert parsed["species_ic_params"] == [(0, "A0")]

        t = np.linspace(0.0, 4.0, 9)
        expected = 50.0 / (1.0 + 0.02 * 50.0 * t)
        for model in _both_doors(path):
            y = _ode(model, 4.0, 9)
            np.testing.assert_allclose(y[:, 0], expected, rtol=1e-7)
            np.testing.assert_allclose(y[:, 1], (50.0 - expected) / 2.0, rtol=1e-7, atol=1e-9)

    def test_an_edited_rate_constant_and_seed_parameter_build_the_edited_model(
        self, tmp_path: Path
    ) -> None:
        """The documented parse, modify, build: kf -> 0.04 and A0 -> 20, so
        A = 20 / (1 + 0.8 t). The IC names A0, so A0 is what has to be edited."""
        parsed = parse_net_file(_net(tmp_path, _HOMODIMER))
        parsed["parameters"] = [
            (name, {"kf": 0.04, "A0": 20.0}.get(name, value), expr, is_expr)
            for name, value, expr, is_expr in parsed["parameters"]
        ]
        y = _ode(build_model_from_parsed(parsed), 4.0, 9)
        t = np.linspace(0.0, 4.0, 9)
        np.testing.assert_allclose(y[:, 0], 20.0 / (1.0 + 0.8 * t), rtol=1e-7)

    def test_a_stat_factor_on_a_function_rate(self, tmp_path: Path) -> None:
        """`3*f` with f() = 2k, k = 0.5: dA/dt = -3 * 1 * A, so A = 10 exp(-3t)."""
        path = _net(
            tmp_path,
            """
            begin parameters
                1 k 0.5
            end parameters
            begin species
                1 A() 10
            end species
            begin functions
                1 f() k*2
            end functions
            begin reactions
                1 1 0 3*f
            end reactions
            begin groups
                1 Atot 1
            end groups
            """,
        )
        (rxn,) = parse_net_file(path)["reactions"]
        assert (rxn["type"], rxn["rate_law"], rxn["stat_factor"]) == ("functional", "f", 3.0)
        t = np.linspace(0.0, 1.0, 6)
        for model in _both_doors(path):
            np.testing.assert_allclose(
                _ode(model, 1.0, 6)[:, 0], 10.0 * np.exp(-3.0 * t), rtol=1e-7
            )


def _mass_action_rhs_from_the_dict(parsed: dict):
    """A mass-action RHS written from the dict alone: stat_factor * k * reactants,
    and a clamped (`$`) species held still. No bngsim code runs in it."""
    pv = {name: value for name, value, _, _ in parsed["parameters"]}
    fixed = np.array([f for _, _, f in parsed["species"]], dtype=bool)

    def rhs(y: np.ndarray) -> np.ndarray:
        dy = np.zeros(len(y))
        for rxn in parsed["reactions"]:
            assert rxn["type"] == "elementary", rxn
            rate = rxn["stat_factor"] * pv[rxn["rate_law"]]
            for i in rxn["reactants"]:
                rate *= y[i]
            for i in rxn["reactants"]:
                dy[i] -= rate
            for i in rxn["products"]:
                dy[i] += rate
        dy[fixed] = 0.0
        return dy

    return rhs


@pytest.mark.parametrize(
    "where, name",
    [
        ("data", "homodimer_ssa.net"),  # 0.5
        ("data", "ssa_aaa.net"),  # 1/6
        ("data", "fixed_species.net"),  # a clamped reactant
        ("data", "nanomolar_dimer_sink.net"),  # 0.5, clamped
        ("ode", "blbr.net"),  # 2, 4, 6
        ("ode", "RAFi_ground.net"),  # 0.5, 2
        ("ode", "catalysis.net"),  # a folded unit factor 1.66e-12, clamped
    ],
)
def test_the_dict_alone_gives_the_loaders_rhs(data_dir: Path, where: str, name: str) -> None:
    """The dict's stat factors, rate names and clamps, fed to an RHS written by
    hand, give the RHS Model.from_net integrates."""
    path = (data_dir if where == "data" else _ODE_NETS) / name
    if not path.is_file():
        pytest.skip(f"{path} not present")
    parsed = parse_net_file(path)
    y0 = np.array([ic for _, ic, _ in parsed["species"]])
    y = 1.1 * y0 + 0.5
    want = np.asarray(bngsim.Model.from_net(str(path)).rhs(y, 0.0))
    got = _mass_action_rhs_from_the_dict(parsed)(y)
    assert np.max(np.abs(got - want)) <= 1e-12 * max(float(np.max(np.abs(want))), 1e-300)


# ─── Evaluated values, against Python arithmetic ─────────────────────────────


def test_values_are_the_evaluated_numbers(tmp_path: Path) -> None:
    path = _net(
        tmp_path,
        """
        begin parameters
            1 A0 100
            2 k 2^3
            3 kk if(A0>50, A0/4, A0)
            4 lambda 3
            5 x0 lambda/4
            6 Atot 2*A0+kk
            7 c exp(1)*ln(10)
            8 m min(3,2)+max(1,4)
        end parameters
        begin species
            1 A() Atot
            2 B() 3*k+1
            3 C() 1.5e2
            4 $D() k
        end species
        begin reactions
            1 1 2 k
        end reactions
        """,
    )
    parsed = parse_net_file(path)
    values = {name: value for name, value, _, _ in parsed["parameters"]}
    oracle = {
        "A0": 100.0,
        "k": 8.0,
        "kk": 25.0,
        "lambda": 3.0,
        "x0": 0.75,
        "Atot": 225.0,
        "c": math.e * math.log(10.0),
        "m": 6.0,
        "_InitialConc1": 25.0,
    }
    assert values == pytest.approx(oracle, rel=1e-15)
    assert parsed["parameters"][-1][2:] == ("3*k+1", True)
    assert parsed["species"] == [
        ("A()", 225.0, False),
        ("B()", 25.0, False),
        ("C()", 150.0, False),
        ("D()", 8.0, True),
    ]
    assert parsed["species_ic_params"] == [(0, "Atot"), (1, "_InitialConc1"), (3, "k")]
    assert list(bngsim.Model.from_net(str(path))._core.get_initial_state()) == [
        225.0,
        25.0,
        150.0,
        8.0,
    ]


def test_an_edited_literal_reaches_the_parameters_derived_from_it(tmp_path: Path) -> None:
    """k -> 8 in the dict: build() re-evaluates kd = k/4 = 2, so A = 100 exp(-2t)."""
    path = _net(
        tmp_path,
        """
        begin parameters
            1 k 2.0
            2 kd k/4
            3 A0 100
        end parameters
        begin species
            1 A() A0
            2 B() 0
        end species
        begin reactions
            1 1 2 kd
        end reactions
        """,
    )
    parsed = parse_net_file(path)
    parsed["parameters"][0] = ("k", 8.0, "2.0", False)
    model = build_model_from_parsed(parsed)
    assert model._core.get_param("kd") == 2.0
    t = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(_ode(model, 1.0, 5)[:, 0], 100.0 * np.exp(-2.0 * t), rtol=1e-7)


# Every tracked .net file, then every untracked one on this machine as `local:`
# cases. See _net_corpus.py for why both, and why they are kept apart.
@pytest.mark.parametrize("net", _net_corpus.sweep_params())
def test_the_dicts_numbers_are_the_ones_from_net_uses(net: Path) -> None:
    """The round-trip test compares the *built* model, which re-evaluates every
    expression; this compares the numbers the dict itself reports."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = parse_net_file(net)
        core = bngsim.Model.from_net(str(net))._core
    np.testing.assert_array_equal(
        [v for _, v, _, _ in parsed["parameters"]],
        [core.get_param(n) for n, _, _, _ in parsed["parameters"]],
    )
    np.testing.assert_array_equal(
        [ic for _, ic, _ in parsed["species"]], np.asarray(core.get_initial_state())
    )


# ─── Table functions, against a hand integral ────────────────────────────────


def test_an_inline_step_table_matches_its_hand_integral(tmp_path: Path) -> None:
    """dB/dt = drive(t), a step table 0|10|20|30 on [0,1)|[1,2)|[2,3)|[3,inf).

    No committed .net carries an inline tfun and codegen_data() carries no table
    data or method, so the round-trip sweep cannot see this branch drift.
    """
    path = _net(
        tmp_path,
        """
        begin parameters
          1 k0 0.0
        end parameters
        begin functions
          1 drive() tfun([0,1,2,3],[0,10,20,30],time, method=>"step")
        end functions
        begin species
          1 A() 1
          2 B() 0
        end species
        begin reactions
          1 1 1,2 drive
        end reactions
        begin groups
          1 B_tot 2
        end groups
        """,
    )
    expected = [0.0, 0.0, 0.0, 5.0, 10.0, 20.0, 30.0, 45.0]  # t = 0, 0.5, ..., 3.5
    for model in _both_doors(path):
        np.testing.assert_allclose(_ode(model, 3.5, 8)[:, 1], expected, rtol=1e-7, atol=1e-6)


# ─── Defects in the change under review (these fail) ─────────────────────────


class TestFilesItCannotRead:
    def test_a_missing_file_is_a_file_not_found_error(self, tmp_path: Path) -> None:
        """Before step 5 parse_net_file raised FileNotFoundError (Path.read_text),
        as Model.from_net still does; now it raises ValueError."""
        with pytest.raises(FileNotFoundError):
            bngsim.Model.from_net(tmp_path / "absent.net")
        with pytest.raises(FileNotFoundError):
            parse_net_file(tmp_path / "absent.net")

    def test_a_directory_is_refused(self, tmp_path: Path) -> None:
        """Before step 5 this was IsADirectoryError; now std::ifstream opens the
        directory, reads nothing, and parse_net_file returns an empty model."""
        with pytest.raises((OSError, ValueError)):
            parse_net_file(tmp_path)


_REFUSED_AT_BUILD = {
    "unknown_rate_constant": """
        begin parameters
            1 k 1
        end parameters
        begin species
            1 A() 10
        end species
        begin reactions
            1 1 0 kx
        end reactions
        """,
    "observable_index_out_of_range": """
        begin parameters
            1 k 1
        end parameters
        begin species
            1 A() 10
        end species
        begin reactions
            1 1 0 k
        end reactions
        begin groups
            1 Atot 5
        end groups
        """,
    "duplicate_parameter": """
        begin parameters
            1 k 1.0
            2 k 2.0
        end parameters
        begin species
            1 A() 10
        end species
        begin reactions
            1 1 0 k
        end reactions
        """,
}


@pytest.mark.parametrize("case", sorted(_REFUSED_AT_BUILD))
def test_parse_net_file_refuses_what_from_net_refuses(tmp_path: Path, case: str) -> None:
    """docs/reference/api.md: parse_net_file "raises ValueError where Model.from_net
    refuses the file". It only refuses what the loader's *parse* refuses; these
    three from_net refuses in build(), and parse_net_file returns a dict for the
    first two and raises RuntimeError for the third."""
    path = _net(tmp_path, _REFUSED_AT_BUILD[case])
    with pytest.raises(ModelError):
        bngsim.Model.from_net(str(path))
    with pytest.raises(ValueError):
        parse_net_file(path)


_SPECIES_FIRST = """
begin species
    1 A() 2*A0
    2 B() 0
end species
begin parameters
    1 k 2.0
    2 A0 50
end parameters
begin reactions
    1 1 2 k
end reactions
"""


def test_an_expression_ic_ahead_of_the_parameters_block(tmp_path: Path) -> None:
    """A(0) = 2*A0 = 100. The old Python reader returned 100; the loader's phase 1
    drops the lifted `_InitialConc1` when the parameters block replaces its list,
    so parse_net_file now dies on a bare KeyError."""
    parsed = parse_net_file(_net(tmp_path, _SPECIES_FIRST))
    assert parsed["species"][0][1] == 100.0


def test_an_sbml_to_net_assignment_rule_species_reports_like_from_net(tmp_path: Path) -> None:
    """sbml_to_net writes an AssignmentRule species as a clamped species sharing
    its name with a function (#515). Model.from_net rebuilds the report map from
    that; build_model_from_parsed does not, so its X column starts at the frozen
    0 instead of the rule's c*A = 20."""
    path = _net(
        tmp_path,
        """
        begin parameters
            1 k 1
            2 c 2
        end parameters
        begin species
            1 A 10
            2 $X 0
        end species
        begin functions
            1 X() c*Aobs
        end functions
        begin reactions
            1 1 0 k
        end reactions
        begin groups
            1 Aobs 1
        end groups
        """,
    )
    t = np.linspace(0.0, 1.0, 3)
    rule = 2.0 * 10.0 * np.exp(-t)
    loaded, built = _both_doors(path)
    np.testing.assert_allclose(_ode(loaded, 1.0, 3)[:, 1], rule, rtol=1e-7)
    np.testing.assert_allclose(_ode(built, 1.0, 3)[:, 1], rule, rtol=1e-7)


# ─── Defects that predate the change (strict xfail, self-retiring) ───────────


def test_from_net_seeds_an_expression_ic_ahead_of_the_parameters_block(
    tmp_path: Path,
) -> None:
    core = bngsim.Model.from_net(str(_net(tmp_path, _SPECIES_FIRST)))._core
    assert list(core.get_initial_state()) == [100.0, 0.0]


def test_from_net_with_two_parameters_blocks(tmp_path: Path) -> None:
    """A second parameters block used to replace the first while the name -> index
    map kept the first block's names, so `A() A0` was seeded with k's value (2.0).
    BNG2.pl writes each block once, and both doors now refuse a repeat (#803)."""
    path = _net(
        tmp_path,
        """
        begin parameters
            1 A0 100
        end parameters
        begin parameters
            1 k 2
        end parameters
        begin species
            1 A() A0
        end species
        """,
    )
    with pytest.raises(ModelError, match="more than one parameters block"):
        bngsim.Model.from_net(str(path))
    with pytest.raises(ValueError, match="more than one parameters block"):
        parse_net_file(path)


@pytest.mark.parametrize("expression", ["Atot*k", "f*2"])
def test_a_parameter_that_reads_the_state_is_refused_by_every_door(
    tmp_path: Path, expression: str
) -> None:
    """k2 reads Atot or f() = Atot*k. It used to compile, evaluate to 0.0 at load
    and build silently, where BNG2.pl refuses the model (#844). Refused by
    from_net, by parse_net_file, and by build_model_from_parsed on a dict whose
    parameter was edited to read the state."""
    body = """
        begin parameters
            1 k 2.0
            2 k2 {expression}
        end parameters
        begin species
            1 A() 100
        end species
        begin functions
            1 f() Atot*k
        end functions
        begin reactions
            1 1 0 k
        end reactions
        begin groups
            1 Atot 1
        end groups
        """
    path = _net(tmp_path, body.format(expression=expression))
    with pytest.raises(ModelError, match="reads the"):
        bngsim.Model.from_net(str(path))
    with pytest.raises(ValueError, match="reads the"):
        parse_net_file(path)
    ok = _net(tmp_path, body.format(expression="k*3"))
    parsed = parse_net_file(ok)
    parsed["parameters"] = [
        (n, v, expression, True) if n == "k2" else (n, v, e, x)
        for n, v, e, x in parsed["parameters"]
    ]
    with pytest.raises(RuntimeError, match="reads the"):
        build_model_from_parsed(parsed)


def test_from_net_refuses_a_directory(tmp_path: Path) -> None:
    try:
        bngsim.Model.from_net(tmp_path)
    except (OSError, ValueError, RuntimeError):
        return
    raise AssertionError("Model.from_net loaded a directory as an empty model")


def test_an_edited_functional_entry_naming_a_parameter_fires(tmp_path: Path) -> None:
    """A dict edit that points a function-rate reaction at a parameter, k = 2:
    dA/dt = -k*A = -200 at A = 100."""
    path = _net(
        tmp_path,
        """
        begin parameters
            1 k 2.0
        end parameters
        begin species
            1 A() 100
            2 B() 0
        end species
        begin functions
            1 f() k*3
        end functions
        begin reactions
            1 1 2 f
        end reactions
        begin groups
            1 Atot 1
        end groups
        """,
    )
    parsed = parse_net_file(path)
    assert parsed["reactions"][0]["type"] == "functional"
    parsed["reactions"][0]["rate_law"] = "k"
    rhs = build_model_from_parsed(parsed).rhs(np.array([100.0, 0.0]), 0.0)
    np.testing.assert_allclose(rhs, [-200.0, 200.0])


def test_a_relative_path_survives_a_change_of_directory(
    tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    for name in ("tfun_time_indexed.net", "cumNcases.tfun"):
        shutil.copy(data_dir / name, sub / name)
    monkeypatch.chdir(tmp_path)
    parsed = parse_net_file("sub/tfun_time_indexed.net")
    monkeypatch.chdir(sub)
    try:
        model = build_model_from_parsed(parsed)
    except RuntimeError as exc:
        raise AssertionError(f"the dict no longer builds after a chdir: {exc}") from exc
    assert model._core.table_function_names == ["cumNcases"]
