"""Tests for build_model_from_parsed reaction rate handling."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from bngsim._net_reader import build_model_from_parsed, parse_net_file


def _write_net(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "model.net"
    p.write_text(textwrap.dedent(body).strip() + "\n")
    return p


class TestBuildModelFromParsedRates:
    def test_elementary_expression_rate_becomes_functional(self, tmp_path: Path) -> None:
        """Non-parameter expression in the rate column → functional + synthetic function."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 0.5*kp2
            end reactions
            begin groups
            end groups
            """,
        )
        model = build_model_from_parsed(parse_net_file(net))
        cd = model._core.codegen_data()
        assert len(cd["functions"]) == 1
        assert cd["functions"][0]["name"] == "__net_reader_func_0"
        assert cd["functions"][0]["expression"] == "0.5*kp2"
        rx = cd["reactions"][0]
        assert rx["type"] == "functional"
        assert rx["function_name"] == "__net_reader_func_0"

    def test_elementary_reuses_declared_function_no_duplicate(self, tmp_path: Path) -> None:
        """When rate_law names an existing .net function, that function is used once."""
        net = _write_net(
            tmp_path,
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
            """,
        )
        model = build_model_from_parsed(parse_net_file(net))
        cd = model._core.codegen_data()
        names = [f["name"] for f in cd["functions"]]
        assert names == ["myRate"]
        assert len(names) == 1
        rx = cd["reactions"][0]
        assert rx["type"] == "functional"
        assert rx["function_name"] == "myRate"

    def test_elementary_parameter_rate_unchanged(self, tmp_path: Path) -> None:
        """Declared parameter name in the rate column stays elementary."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 k1 0.1
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 k1
            end reactions
            begin groups
            end groups
            """,
        )
        model = build_model_from_parsed(parse_net_file(net))
        cd = model._core.codegen_data()
        assert cd["functions"] == []
        rx = cd["reactions"][0]
        assert rx["type"] == "elementary"
        assert rx["function_name"] == "k1"

    def test_synthetic_name_avoids_collision_with_user_function(self, tmp_path: Path) -> None:
        """Reserved-style name in ``begin functions`` must not collide with synthetic."""
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin functions
              1 __net_reader_func_0() 1.0
            end functions
            begin reactions
              1 1 2 0.5*kp2
            end reactions
            begin groups
            end groups
            """,
        )
        model = build_model_from_parsed(parse_net_file(net))
        cd = model._core.codegen_data()
        names = sorted(f["name"] for f in cd["functions"])
        assert "__net_reader_func_0" in names
        assert "__net_reader_func_1" in names
        syn = next(f for f in cd["functions"] if f["name"] == "__net_reader_func_1")
        assert syn["expression"] == "0.5*kp2"
        rx = cd["reactions"][0]
        assert rx["type"] == "functional"
        assert rx["function_name"] == "__net_reader_func_1"

    def test_forced_elementary_matching_function_name_stays_single_function(
        self, tmp_path: Path
    ) -> None:
        """If parsed data still says elementary but rate matches a function, reuse it."""
        net = _write_net(
            tmp_path,
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
            """,
        )
        parsed = parse_net_file(net)
        parsed["reactions"][0]["type"] = "elementary"
        model = build_model_from_parsed(parsed)
        cd = model._core.codegen_data()
        assert [f["name"] for f in cd["functions"]] == ["myRate"]
        assert cd["reactions"][0]["type"] == "functional"
        assert cd["reactions"][0]["function_name"] == "myRate"

    def test_empty_rate_law_raises(self, tmp_path: Path) -> None:
        parsed = parse_net_file(
            _write_net(
                tmp_path,
                """
                begin parameters
                  1 k1 0.1
                end parameters
                begin species
                  1 A() 100
                  2 B() 0
                end species
                begin reactions
                  1 1 2 k1
                end reactions
                begin groups
                end groups
                """,
            )
        )
        parsed["reactions"][0]["rate_law"] = "   "
        parsed["reactions"][0]["type"] = "elementary"
        with pytest.raises(ValueError, match="empty or whitespace-only rate_law"):
            build_model_from_parsed(parsed)

    def test_malformed_trailing_operator_raises(self, tmp_path: Path) -> None:
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 0.5*
            end reactions
            begin groups
            end groups
            """,
        )
        with pytest.raises(ValueError, match="ends with an operator"):
            build_model_from_parsed(parse_net_file(net))

    def test_malformed_unbalanced_paren_raises(self, tmp_path: Path) -> None:
        net = _write_net(
            tmp_path,
            """
            begin parameters
              1 kp2 2.0
            end parameters
            begin species
              1 A() 100
              2 B() 0
            end species
            begin reactions
              1 1 2 (0.5*kp2
            end reactions
            begin groups
            end groups
            """,
        )
        with pytest.raises(ValueError, match="unmatched '\\('"):
            build_model_from_parsed(parse_net_file(net))
