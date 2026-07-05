"""Regression tests for issue #41 — `.net` loader must recognize the `$`
clamp marker when the species carries a `@compartment::` prefix (e.g.
`@CP::$Sink()`) that BNG2.pl emits for cBNGL models.

Covers all three loader paths:
- `bngsim._codegen._strip_fixed_marker` (the shared helper)
- `bngsim._codegen._parse_species_line` (used by the Python codegen)
- `bngsim._net_reader.parse_net_file` (the pure-Python ModelBuilder path)
- C++ `NetworkModel.from_net` reached via `bngsim.Model.from_net`
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._codegen import _parse_species_line, _strip_fixed_marker
from bngsim._net_reader import parse_net_file


def _write_compartmental_clamp_net(tmp_path: Path) -> Path:
    p = tmp_path / "sink_compart.net"
    p.write_text(
        textwrap.dedent(
            """
            begin parameters
                1 k         1.0  # Constant
            end parameters
            begin species
                1 @CP::X() 100
                2 @CP::$Sink() 0
            end species
            begin reactions
                1 1 2 k #_R1
            end reactions
            begin groups
                1 X_obs                1
                2 Sink_obs             2
            end groups
            """
        ).strip()
        + "\n"
    )
    return p


class TestStripFixedMarker:
    """Unit tests for the shared `$`-stripping helper."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("$Sink()", ("Sink()", True)),
            ("Sink()", ("Sink()", False)),
            ("@CP::$Sink()", ("@CP::Sink()", True)),
            ("@CP::Sink()", ("@CP::Sink()", False)),
            ("@my_comp::$X(a~b)", ("@my_comp::X(a~b)", True)),
            ("@C::$ATP()", ("@C::ATP()", True)),
            # Non-clamp `@` and stray-prefix forms must not flip the flag.
            ("@CP::", ("@CP::", False)),
            ("@", ("@", False)),
        ],
    )
    def test_strip(self, raw: str, expected: tuple[str, bool]) -> None:
        assert _strip_fixed_marker(raw) == expected

    def test_parse_species_line_compartmental_clamp(self) -> None:
        idx, name, conc, is_fixed = _parse_species_line("2 @CP::$Sink() 0")
        assert (idx, name, conc, is_fixed) == (2, "@CP::Sink()", "0", True)


class TestNetReaderCompartmentalClamp:
    """The pure-Python `_net_reader.parse_net_file` path."""

    def test_clamp_marker_after_compartment_prefix(self, tmp_path: Path) -> None:
        parsed = parse_net_file(_write_compartmental_clamp_net(tmp_path))
        species = parsed["species"]
        # Order must match the .net; `$` is stripped from the stored name.
        assert species[0] == ("@CP::X()", 100.0, False)
        assert species[1] == ("@CP::Sink()", 0.0, True)


class TestCxxLoaderCompartmentalClamp:
    """The C++ `NetworkModel.from_net` path reached by `Model.from_net`."""

    def test_clamped_compartmental_species_stays_at_ic(self, tmp_path: Path) -> None:
        net = _write_compartmental_clamp_net(tmp_path)
        model = bngsim.Model.from_net(net)

        # The `$` marker must be stripped from the stored species name; the
        # `@compartment::` prefix is preserved.
        assert model.species_names == ["@CP::X()", "@CP::Sink()"]

        result = bngsim.Simulator(model, method="ode").run(t_span=(0.0, 10.0), n_points=11)
        sink_idx = model.species_names.index("@CP::Sink()")
        x_idx = model.species_names.index("@CP::X()")
        species = np.asarray(result.species)

        # Clamped Sink stays at its IC (0); free X drains to ~0 under X -> Sink.
        np.testing.assert_allclose(species[:, sink_idx], 0.0, atol=1e-10)
        assert species[0, x_idx] == pytest.approx(100.0, abs=1e-10)
        assert species[-1, x_idx] < 1.0
