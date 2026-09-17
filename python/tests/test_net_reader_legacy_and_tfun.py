"""The .net constructs `build_model_from_parsed` used to choke on (issue #597).

`Model.from_net` runs two post-parse steps that used to live only in
`net_file_loader.cpp`: it reads the legacy `Sat` / `Hill` / `MM` rate-law tokens
and the operands that follow them, and it reads each `functions` line's
`tfun(...)` call to register the table before the expression that calls it is
compiled. `bngsim._net_reader` — the
reader `bngsim.__all__` exports and `docs/user-guide/loading-models.md` documents
— had neither, so ten `.net` files in `tests/data` loaded under one documented
loader and raised `ModelBuilder: failed to compile function ... ERR239 -
Undefined symbol: 'Hill'` under the other.

What this file pins:

* the eight that now load build the *same model*, not merely a model —
  identical `codegen_data()` to `Model.from_net`'s. Loading was never the bar
  (issue #554 was two loaders that both loaded and disagreed by a factor);
* the two `Sat`/`Hill` files refuse by name. The rewrite is ~150 lines of C++
  that also allocates non-colliding names and emits deprecation warnings;
  reimplementing it here would be the second dialect #554 was about, so this
  reader says what it does not read and points at the loader that does;
* the reader no longer *drops* a legacy token's operands. `Sat k3 K4` used to
  parse to a bare `rate_law="Sat"` with `k3 K4` discarded — a lossy parse under
  a function documented as the universal interchange format.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import bngsim
import pytest
from bngsim._net_reader import build_model_from_parsed, parse_net_file

#: The six `.net` files whose `functions` block calls `tfun(...)`, and the two
#: `MM` ones. Every one of these raised from inside `ModelBuilder` before #597.
TFUN_NETS = [
    "tfun_time_indexed.net",
    "tfun_uppercase_time.net",
    "tfun_param_indexed.net",
    "tfun_paren_param.net",
    "tfun_step_time_indexed.net",
    "wrap_single.net",
]
MM_NETS = ["mm_tqssa.net", "mm_tqssa_stiff.net"]


def _codegen_json(model) -> str:
    return json.dumps(model._core.codegen_data(), sort_keys=True, default=str)


class TestBuildsWhatFromNetBuilds:
    @pytest.mark.parametrize("name", TFUN_NETS + MM_NETS)
    def test_same_model_as_from_net(self, data_dir: Path, name: str) -> None:
        """Both documented loaders must produce the same model, not just a model."""
        path = data_dir / name
        via_loader = bngsim.Model.from_net(str(path))
        via_reader = build_model_from_parsed(parse_net_file(path))
        assert _codegen_json(via_reader) == _codegen_json(via_loader)

    @pytest.mark.parametrize("name", TFUN_NETS)
    def test_same_table_functions_as_from_net(self, data_dir: Path, name: str) -> None:
        """Including the table names, which the whole-body/embedded split decides."""
        path = data_dir / name
        via_loader = bngsim.Model.from_net(str(path))
        via_reader = build_model_from_parsed(parse_net_file(path))
        assert via_reader._core.table_function_names == via_loader._core.table_function_names
        assert via_reader._core.table_function_names  # and it registered something

    def test_wrap_single_keeps_the_arithmetic_around_the_call(self, data_dir: Path) -> None:
        """A tfun inside arithmetic becomes a synthetic table, not the whole body.

        `f_complex() (tfun('drive.tfun',time)+5)/k_scale` — the table is named
        `f_complex__tfun0` and the `+5)/k_scale` survives in the expression. Had
        the reader treated the whole body as the table, the wrapper arithmetic
        would have been silently dropped and every rate scaled wrong.
        """
        model = build_model_from_parsed(parse_net_file(data_dir / "wrap_single.net"))
        assert model._core.table_function_names == ["f_complex__tfun0"]
        funcs = model._core.codegen_data()["functions"]
        expr = next(f["expression"] for f in funcs if f["name"] == "f_complex")
        assert expr == "(tfun_f_complex__tfun0()+5)/k_scale"

    def test_relative_tfun_path_resolves_against_the_net_file(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`tfun('cumNcases.tfun')` is beside the .net, not beside the process.

        BNG writes the table next to the network it belongs to, so the loader
        passes the .net's own directory to `ModelBuilder.set_net_file_dir`. With
        the working directory somewhere else entirely, a reader that skipped that
        step could only find the file by accident.
        """
        monkeypatch.chdir(tmp_path)
        model = build_model_from_parsed(parse_net_file(data_dir / "tfun_time_indexed.net"))
        assert model._core.table_function_names == ["cumNcases"]

    def test_parse_reports_the_net_files_own_directory(self, data_dir: Path) -> None:
        parsed = parse_net_file(data_dir / "tfun_time_indexed.net")
        assert Path(parsed["net_file_dir"]) == data_dir

    def test_inline_tfun_data_registers(self, tmp_path: Path) -> None:
        """`tfun([xs],[ys],index)` — data in the model rather than in a file.

        No `.net` in `tests/data` carries one, so this is the only exercise of
        the inline branch through the reader.
        """
        net = tmp_path / "inline.net"
        net.write_text(
            textwrap.dedent(
                """
                begin parameters
                  1 k0 0.0
                end parameters
                begin functions
                  1 drive() tfun([0,1,2,3],[0,10,20,30],time)
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
                """
            ).strip()
            + "\n"
        )
        via_loader = bngsim.Model.from_net(str(net))
        via_reader = build_model_from_parsed(parse_net_file(net))
        assert via_reader._core.table_function_names == ["drive"]
        assert _codegen_json(via_reader) == _codegen_json(via_loader)

    def test_a_function_naming_no_table_is_untouched(self, tmp_path: Path) -> None:
        """The tfun step must be inert on the overwhelming majority of models."""
        net = tmp_path / "plain.net"
        net.write_text(
            textwrap.dedent(
                """
                begin parameters
                  1 k1 0.1
                end parameters
                begin species
                  1 A() 100
                  2 B() 0
                end species
                begin functions
                  1 myRate() k1*2
                end functions
                begin reactions
                  1 1 2 myRate
                end reactions
                begin groups
                end groups
                """
            ).strip()
            + "\n"
        )
        model = build_model_from_parsed(parse_net_file(net))
        assert model._core.table_function_names == []
        funcs = model._core.codegen_data()["functions"]
        assert [(f["name"], f["expression"]) for f in funcs] == [("myRate", "k1*2")]


class TestMichaelisMenten:
    """`MM kcat Km` needed no rewrite at all — only parsing.

    `ModelBuilder` has the Michaelis-Menten rate law and takes its two
    parameters as one `"kcat,Km"` string, which is exactly what
    `net_file_loader.cpp` hands it. The reader was reading only the rate column,
    so `kcat Km` went in the bin and the bare `MM` was wrapped as an expression
    for ExprTk to reject.
    """

    def test_parses_to_the_builders_own_mm_form(self, data_dir: Path) -> None:
        parsed = parse_net_file(data_dir / "mm_tqssa.net")
        (rxn,) = parsed["reactions"]
        assert rxn["type"] == "mm"
        assert rxn["rate_law"] == "kcat,Km"
        assert rxn["legacy_constants"] == ["kcat", "Km"]

    def test_short_mm_line_falls_through_to_elementary(self, tmp_path: Path) -> None:
        """`MM` with no operands is a rate-law *name*, as in the C++ loader.

        `parse_reactions` gates the MM branch on `tokens.size() >= 6` and lets a
        shorter line reach the elementary branch, where `MM` is just an
        (unresolvable) parameter name. Mirrored here so the two loaders refuse
        the same line for the same reason rather than one inventing an MM rate
        law out of a truncated one.
        """
        net = tmp_path / "short_mm.net"
        net.write_text(
            textwrap.dedent(
                """
                begin parameters
                  1 kcat 1.0
                end parameters
                begin species
                  1 E() 10
                  2 S() 100
                end species
                begin reactions
                  1 1,2 1,2 MM kcat
                end reactions
                begin groups
                end groups
                """
            ).strip()
            + "\n"
        )
        parsed = parse_net_file(net)
        (rxn,) = parsed["reactions"]
        assert rxn["type"] == "elementary"
        assert rxn["rate_law"] == "MM"
        assert rxn["legacy_constants"] == []


class TestLegacySatHillRefused:
    """Refused by name, with the rewrite to write by hand.

    The `ModelBuilder: failed to compile function '__net_reader_func_0': Hill —
    ... ERR239 - Undefined symbol: 'Hill'` this replaces named neither the file,
    the reaction, nor the loader that does read it.
    """

    @pytest.mark.parametrize(
        ("name", "token", "formula"),
        [
            ("hill_rewrite.net", "Hill", "Vmax*S^(h-1)/(Kh^h+S^h)"),
            ("sat_rewrite.net", "Sat", "k/(K+S)"),
        ],
    )
    def test_message_names_the_reaction_the_token_and_the_way_out(
        self, data_dir: Path, name: str, token: str, formula: str
    ) -> None:
        parsed = parse_net_file(data_dir / name)
        with pytest.raises(ValueError) as excinfo:
            build_model_from_parsed(parsed)
        message = str(excinfo.value)
        assert "reaction 1" in message
        assert token in message
        assert "Model.from_net" in message
        assert formula in message

    @pytest.mark.parametrize("name", ["hill_rewrite.net", "sat_rewrite.net"])
    def test_from_net_still_loads_the_same_file(self, data_dir: Path, name: str) -> None:
        """The message's advice has to be true: the loader it names must work."""
        with pytest.warns(UserWarning, match="auto-rewritten by bngsim loader"):
            model = bngsim.Model.from_net(str(data_dir / name))
        assert model._core.n_species == 2

    @pytest.mark.parametrize(
        ("name", "token", "operands"),
        [
            ("hill_rewrite.net", "Hill", ["k", "K", "n"]),
            ("sat_rewrite.net", "Sat", ["k3", "K4"]),
        ],
    )
    def test_parse_carries_the_operands_rather_than_dropping_them(
        self, data_dir: Path, name: str, token: str, operands: list[str]
    ) -> None:
        """A refusal at build time must not cost the parse its fidelity.

        `parse_net_file` is documented as the universal interchange format, so it
        keeps reporting what the line says — and it used to report a bare
        `rate_law="Sat"` with the rate constants thrown away entirely.
        """
        parsed = parse_net_file(data_dir / name)
        (rxn,) = parsed["reactions"]
        assert rxn["type"] == "legacy"
        assert rxn["rate_law"] == token
        assert rxn["legacy_constants"] == operands
