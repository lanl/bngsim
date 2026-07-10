"""Tests for the output-selector resolver on Result (GH #195).

Covers ``Result.resolve_outputs`` (typed/alias/bare selector resolution +
structured metadata) and ``Result.outputs`` (value accessor). Pure-Python
name resolution — no sensitivity computation is involved.

Two test vehicles are used:

* a real ``func_composition.net`` simulation (species ``A()/B()/C()``,
  observables ``A_tot/B_tot/C_tot``, user functions ``fRate0/fRate_cond``
  plus filtered ``_rateLawN`` columns), for the end-to-end happy path; and
* synthetic raw-array :class:`Result` objects (the same constructor path
  ``Result.squeeze`` / HDF5 load use) for the controlled edge cases —
  name collisions, alias normalization, error messages — that no single
  shipped fixture exercises cleanly.
"""

from __future__ import annotations

from pathlib import Path

import bngsim
import numpy as np
import pytest
from bngsim._result import Result

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def func_composition_result(data_dir: Path) -> Result:
    """A real ODE result with species, observables and user functions.

    species: ``A()``, ``B()``, ``C()``;
    observables: ``A_tot``, ``B_tot``, ``C_tot``;
    expressions (filtered): ``fRate0``, ``fRate_cond`` (``_rateLawN`` hidden).
    """
    model = bngsim.Model.from_net(str(data_dir / "func_composition.net"))
    return bngsim.Simulator(model, method="ode").run(t_span=(0, 10), n_points=11)


def _raw_result(
    *,
    species_names: list[str],
    observable_names: list[str],
    expression_names: list[str],
    n_times: int = 5,
) -> Result:
    """Build a Result from raw arrays with caller-chosen column names.

    Each column is filled with a distinct constant so that a resolved
    selector's value array can be checked back to the column it named.
    Layout: species columns start at 10.0, observables at 20.0,
    expressions at 30.0 (each +1 per column).
    """
    t = np.linspace(0.0, 1.0, n_times)

    def _block(base: float, names: list[str]) -> np.ndarray:
        if not names:
            return np.empty((n_times, 0))
        cols = [np.full(n_times, base + i) for i in range(len(names))]
        return np.stack(cols, axis=-1)

    return Result(
        _time=t,
        _species=_block(10.0, species_names),
        _observables=_block(20.0, observable_names),
        _expressions=_block(30.0, expression_names),
        _species_names=species_names,
        _observable_names=observable_names,
        _expression_names=expression_names,
    )


# ── Typed selectors + aliases ───────────────────────────────────────────────


class TestTypedSelectors:
    """Canonical typed selectors and their accepted aliases."""

    def test_species_selector(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("species:A()")
        assert meta == {
            "selector": "species:A()",
            "kind": "species",
            "name": "A()",
            "index": 0,
            "column_label": "A()",
        }

    def test_observable_selector(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("observable:B_tot")
        assert meta["kind"] == "observable"
        assert meta["name"] == "B_tot"
        assert meta["index"] == 1
        assert meta["selector"] == "observable:B_tot"
        assert meta["column_label"] == "B_tot"

    def test_expression_selector(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("expression:fRate0")
        assert meta["kind"] == "expression"
        assert meta["name"] == "fRate0"
        assert meta["index"] == 0

    def test_state_alias_maps_to_species(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("state:C()")
        assert meta["kind"] == "species"
        assert meta["selector"] == "species:C()"
        assert meta["index"] == 2

    def test_function_alias_maps_to_expression(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("function:fRate_cond")
        assert meta["kind"] == "expression"
        assert meta["selector"] == "expression:fRate_cond"

    def test_prefix_is_case_insensitive(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("Observable:A_tot")
        assert meta["selector"] == "observable:A_tot"

    def test_surrounding_whitespace_tolerated(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("  observable : A_tot ")
        assert meta["selector"] == "observable:A_tot"

    def test_single_string_equivalent_to_one_element_list(self, func_composition_result: Result):
        as_str = func_composition_result.resolve_outputs("observable:A_tot")
        as_list = func_composition_result.resolve_outputs(["observable:A_tot"])
        assert as_str == as_list

    def test_list_preserves_input_order(self, func_composition_result: Result):
        metas = func_composition_result.resolve_outputs(
            ["expression:fRate0", "species:A()", "observable:C_tot"]
        )
        assert [m["selector"] for m in metas] == [
            "expression:fRate0",
            "species:A()",
            "observable:C_tot",
        ]


# ── foo() normalization ─────────────────────────────────────────────────────


class TestFunctionCallNormalization:
    """The ``foo()`` header convention is stripped for expressions only."""

    def test_expression_paren_suffix_stripped(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("expression:fRate0()")
        assert meta["name"] == "fRate0"
        assert meta["selector"] == "expression:fRate0"
        assert "()" not in meta["column_label"]

    def test_function_alias_with_parens(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("function:fRate_cond()")
        assert meta["selector"] == "expression:fRate_cond"

    def test_expression_with_and_without_parens_agree(self, func_composition_result: Result):
        bare = func_composition_result.resolve_outputs("expression:fRate0")
        paren = func_composition_result.resolve_outputs("expression:fRate0()")
        assert bare == paren

    def test_species_parens_are_not_stripped(self, func_composition_result: Result):
        """Species names legitimately end in ``()`` — must match verbatim."""
        (meta,) = func_composition_result.resolve_outputs("species:A()")
        assert meta["name"] == "A()"
        # And the bare-stripped form must NOT resolve as a species.
        with pytest.raises(ValueError):
            func_composition_result.resolve_outputs("species:A")


# ── Bare-name resolution ────────────────────────────────────────────────────


class TestBareNames:
    """A bare name resolves only if unique across all three kinds."""

    def test_bare_observable_unique(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("A_tot")
        assert meta["selector"] == "observable:A_tot"

    def test_bare_species_unique(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("B()")
        assert meta["selector"] == "species:B()"

    def test_bare_function_unique(self, func_composition_result: Result):
        (meta,) = func_composition_result.resolve_outputs("fRate0")
        assert meta["selector"] == "expression:fRate0"

    def test_bare_function_with_parens(self, func_composition_result: Result):
        """A bare ``foo()`` retries as the expression ``foo``."""
        (meta,) = func_composition_result.resolve_outputs("fRate0()")
        assert meta["selector"] == "expression:fRate0"

    def test_bare_function_with_and_without_parens_agree(self, func_composition_result: Result):
        assert func_composition_result.resolve_outputs(
            "fRate_cond"
        ) == func_composition_result.resolve_outputs("fRate_cond()")

    def test_bare_ambiguous_species_observable(self):
        """Same bare name as a species and an observable → ambiguity error."""
        r = _raw_result(
            species_names=["shared"],
            observable_names=["shared"],
            expression_names=[],
        )
        with pytest.raises(ValueError) as exc:
            r.resolve_outputs("shared")
        msg = str(exc.value)
        assert "species:shared" in msg
        assert "observable:shared" in msg
        assert "ambiguous" in msg.lower()

    def test_bare_ambiguous_across_three_kinds(self):
        r = _raw_result(
            species_names=["X"],
            observable_names=["X"],
            expression_names=["X"],
        )
        with pytest.raises(ValueError) as exc:
            r.resolve_outputs("X")
        msg = str(exc.value)
        for typed in ("species:X", "observable:X", "expression:X"):
            assert typed in msg

    def test_typed_selector_resolves_ambiguous_name(self):
        """A typed selector disambiguates a name that is bare-ambiguous."""
        r = _raw_result(
            species_names=["shared"],
            observable_names=["shared"],
            expression_names=[],
        )
        assert r.resolve_outputs("species:shared")[0]["kind"] == "species"
        assert r.resolve_outputs("observable:shared")[0]["kind"] == "observable"


# ── Error handling ──────────────────────────────────────────────────────────


class TestSelectorErrors:
    """Unknown / unresolved / malformed selectors raise clear errors."""

    def test_unknown_kind_prefix(self, func_composition_result: Result):
        with pytest.raises(ValueError) as exc:
            func_composition_result.resolve_outputs("reaction:A")
        msg = str(exc.value)
        assert "reaction" in msg
        # Lists the valid prefixes and aliases.
        assert "species:" in msg
        assert "observable:" in msg
        assert "expression:" in msg
        assert "state:" in msg and "function:" in msg

    def test_unresolved_typed_lists_candidates(self, func_composition_result: Result):
        with pytest.raises(ValueError) as exc:
            func_composition_result.resolve_outputs("observable:nope")
        msg = str(exc.value)
        assert "nope" in msg
        # Candidate observable names are listed.
        assert "A_tot" in msg and "B_tot" in msg

    def test_unresolved_bare_lists_all_candidates(self, func_composition_result: Result):
        with pytest.raises(ValueError) as exc:
            func_composition_result.resolve_outputs("does_not_exist")
        msg = str(exc.value)
        # All three candidate lists appear.
        assert "A()" in msg  # species
        assert "A_tot" in msg  # observable
        assert "fRate0" in msg  # expression

    def test_empty_selector(self, func_composition_result: Result):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            func_composition_result.resolve_outputs("")

    def test_whitespace_only_selector(self, func_composition_result: Result):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            func_composition_result.resolve_outputs("   ")

    def test_non_string_selector(self, func_composition_result: Result):
        with pytest.raises(TypeError):
            func_composition_result.resolve_outputs([123])

    def test_error_in_list_identifies_bad_one(self, func_composition_result: Result):
        """One bad selector in a list raises (the good ones don't mask it)."""
        with pytest.raises(ValueError):
            func_composition_result.resolve_outputs(["observable:A_tot", "observable:bad"])


# ── _rateLawN filtering consistency ─────────────────────────────────────────


class TestRateLawFiltering:
    """Auto-generated ``_rateLawN`` functions stay unresolvable (issue #58)."""

    def test_rate_law_not_in_expression_names(self, func_composition_result: Result):
        # Sanity: the filtered view hides them but raw_* still has them.
        assert "_rateLaw1" not in func_composition_result.expression_names
        assert "_rateLaw1" in func_composition_result.raw_expression_names

    def test_typed_rate_law_selector_unresolved(self, func_composition_result: Result):
        with pytest.raises(ValueError):
            func_composition_result.resolve_outputs("expression:_rateLaw1")

    def test_bare_rate_law_unresolved(self, func_composition_result: Result):
        with pytest.raises(ValueError):
            func_composition_result.resolve_outputs("_rateLaw1")

    def test_paren_rate_law_unresolved(self, func_composition_result: Result):
        with pytest.raises(ValueError):
            func_composition_result.resolve_outputs("expression:_rateLaw1()")


# ── outputs() value accessor ────────────────────────────────────────────────


class TestOutputsAccessor:
    """``Result.outputs`` stacks the named columns into one array."""

    def test_shape_and_order(self, func_composition_result: Result):
        vals = func_composition_result.outputs(["observable:A_tot", "expression:fRate0"])
        assert vals.shape == (func_composition_result.n_times, 2)

    def test_values_match_named_accessors(self, func_composition_result: Result):
        r = func_composition_result
        vals = r.outputs(["observable:A_tot", "species:B()", "expression:fRate0"])
        np.testing.assert_array_equal(vals[:, 0], r.observables["A_tot"])
        np.testing.assert_array_equal(vals[:, 1], r.species[:, r.species_names.index("B()")])
        np.testing.assert_array_equal(vals[:, 2], r.expressions["fRate0"])

    def test_single_string_argument(self, func_composition_result: Result):
        vals = func_composition_result.outputs("observable:A_tot")
        assert vals.shape == (func_composition_result.n_times, 1)
        np.testing.assert_array_equal(vals[:, 0], func_composition_result.observables["A_tot"])

    def test_empty_selector_list(self, func_composition_result: Result):
        vals = func_composition_result.outputs([])
        assert vals.shape == (func_composition_result.n_times, 0)

    def test_collided_columns_pick_correct_kind(self):
        """Typed selectors over a name collision pull the right columns."""
        r = _raw_result(
            species_names=["shared"],
            observable_names=["shared"],
            expression_names=[],
        )
        vals = r.outputs(["species:shared", "observable:shared"])
        # species block starts at 10.0, observable block at 20.0.
        assert vals[0, 0] == 10.0
        assert vals[0, 1] == 20.0

    def test_batch_result_3d(self, func_composition_result: Result):
        """A stacked batch result yields (n_sims, n_times, n_outputs)."""
        batch = Result.squeeze([func_composition_result, func_composition_result])
        vals = batch.outputs(["observable:A_tot", "expression:fRate0"])
        assert vals.shape == (2, batch.n_times, 2)
        np.testing.assert_array_equal(vals[0], vals[1])
